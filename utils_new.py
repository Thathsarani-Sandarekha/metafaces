import gc
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
import numpy as np
from skimage import filters
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

from diffusers import DDIMScheduler
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging, USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers, replace_example_docstring, is_torch_xla_available
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline, EXAMPLE_DOC_STRING, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.models.attention_processor import Attention

from diffusers.models.embeddings import (
    GaussianFourierProjection, 
    TimestepEmbedding, 
    Timesteps, 
    ImageProjection, 
    TextImageProjection, 
    ImageHintTimeEmbedding, 
    TextTimeEmbedding, 
    ImageTimeEmbedding, 
    TextImageTimeEmbedding)
from diffusers.models.unets.unet_2d_blocks import get_down_block, get_up_block, UNetMidBlock2DCrossAttn, UNetMidBlock2DSimpleCrossAttn, UNetMidBlock2D
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttnAddedKVProcessor,
    AttnProcessor,
    AttentionProcessor
)

from grounding_sam_utils import GroundingSAM

# from evaluation import *

LATENT_RESOLUTIONS = [32, 64]
logger = logging.get_logger(__name__)

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

if is_torch_xla_available():

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"

repo_id = "ShilongLiu/GroundingDINO"
dino_checkpoint_filename = "groundingdino_swinb_cogcoor.pth"
dino_config_filename = "GroundingDINO_SwinB.cfg.py"

sam_checkpoint_path = '/teamspace/studios/this_studio/metafaces_UI/Grounded_Segment_Anything/sam_vit_h_4b8939.pth'

def get_dynamic_threshold(tensor):
    return filters.threshold_otsu(tensor.cpu().numpy())

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    """
    Function to apply Gaussian smoothing on each 2D slice of a 3D tensor.
    """

    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                      np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = torch.Tensor(kernel / kernel.sum()).to(input_tensor.dtype).to(input_tensor.device)
    
    # Add batch and channel dimensions to the kernel
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Iterate over each 2D slice and apply convolution
    smoothed_slices = []
    for i in range(input_tensor.size(0)):
        slice_tensor = input_tensor[i, :, :]
        slice_tensor = F.conv2d(slice_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)[0, 0]
        smoothed_slices.append(slice_tensor)
    
    # Stack the smoothed slices to get the final tensor
    smoothed_tensor = torch.stack(smoothed_slices, dim=0)

    return smoothed_tensor

def attn_map_to_binary(attention_map, scaler=1.):
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask

def cos_dist(a, b):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    res = a_norm @ b_norm.T

    return 1 - res

def gen_nn_map(src_features, src_mask,  tgt_features, tgt_mask, device, batch_size=100, tgt_size=768):
    resized_src_features = F.interpolate(src_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_src_features = resized_src_features.permute(1,2,0).view(tgt_size**2, -1)
    resized_tgt_features = F.interpolate(tgt_features.unsqueeze(0), size=tgt_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_tgt_features = resized_tgt_features.permute(1,2,0).view(tgt_size**2, -1)

    nearest_neighbor_indices = torch.zeros(tgt_size**2, dtype=torch.long, device=device)
    nearest_neighbor_distances = torch.zeros(tgt_size**2, dtype=src_features.dtype, device=device)

    if not batch_size:
        batch_size = tgt_size**2

    for i in range(0, tgt_size**2, batch_size):
        distances = cos_dist(resized_src_features, resized_tgt_features[i:i+batch_size])
        distances[~src_mask] = 2.
        min_distances, min_indices = torch.min(distances, dim=0)
        nearest_neighbor_indices[i:i+batch_size] = min_indices
        nearest_neighbor_distances[i:i+batch_size] = min_distances

    return nearest_neighbor_indices, nearest_neighbor_distances

def cyclic_nn_map(features, masks, latent_resolutions, device):
    bsz = features.shape[0]
    nn_map_dict = {}
    nn_distances_dict = {}

    for tgt_size in latent_resolutions:
        nn_map = torch.empty(bsz, bsz, tgt_size**2, dtype=torch.long, device=device)
        nn_distances = torch.full((bsz, bsz, tgt_size**2), float('inf'), dtype=features.dtype, device=device)

        for i in range(bsz):
            for j in range(bsz):
                if i != j:
                    nearest_neighbor_indices, nearest_neighbor_distances = gen_nn_map(features[j], masks[tgt_size][j], features[i], masks[tgt_size][i], device, batch_size=None, tgt_size=tgt_size)
                    nn_map[i,j] = nearest_neighbor_indices
                    nn_distances[i,j] = nearest_neighbor_distances

        nn_map_dict[tgt_size] = nn_map
        nn_distances_dict[tgt_size] = nn_distances
    
    return nn_map_dict, nn_distances_dict

def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True,
                downscale_rate=None) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)

    if downscale_rate:
        pil_img = pil_img.resize((int(pil_img.size[0] // downscale_rate), int(pil_img.size[1] // downscale_rate)))

    if display_image:
        display(pil_img)
    return pil_img

class DIFTLatentStore:
    def __init__(self, steps: List[int], up_ft_indices: List[int]):
        self.steps = steps
        self.up_ft_indices = up_ft_indices
        self.dift_features = {}

    def __call__(self, features: torch.Tensor, t: int, layer_index: int):
        if t in self.steps and layer_index in self.up_ft_indices:
            self.dift_features[f'{int(t)}_{layer_index}'] = features

class FeatureInjector:
    def __init__(self, nn_map, nn_distances, attn_masks, memory_module, inject_range_alpha=[(10,20,0.8)], swap_strategy='min', dist_thr='dynamic', inject_unet_parts=['up']):
        self.nn_map = nn_map
        self.nn_distances = nn_distances
        self.attn_masks = attn_masks
        self.memory_module = memory_module

        self.inject_range_alpha = inject_range_alpha if isinstance(inject_range_alpha, list) else [inject_range_alpha]
        self.swap_strategy = swap_strategy # 'min / 'mean' / 'first'
        self.dist_thr = dist_thr
        self.inject_unet_parts = inject_unet_parts
        self.inject_res = [64]

    def inject_outputs(self, output, curr_iter, output_res, extended_mapping, place_in_unet, anchors_cache=None):
        curr_unet_part = place_in_unet.split('_')[0]

        # Inject only in the specified unet parts (up, mid, down)
        if (curr_unet_part not in self.inject_unet_parts) or output_res not in self.inject_res:
            return output

        # print(f"nn_map type: {type(self.nn_map)}")
        # print(f"nn_map value: {self.nn_map}")
        # print(f"Available nn_map keys: {list(self.nn_map.keys())}")
        # print(f"Requested output_res: {output_res}")
        
        bsz = output.shape[0]
        nn_map = self.nn_map[output_res]
        nn_distances = self.nn_distances[output_res]
        attn_masks = self.attn_masks[output_res]
        vector_dim = output_res**2

        alpha = next((alpha for min_range, max_range, alpha in self.inject_range_alpha if min_range <= curr_iter <= max_range), None)
        if alpha:
            old_output = output#.clone()
            for i in range(bsz):
                other_outputs = []

                # for name in CHARACTER_DESCRIPTIONS:
                #     stored_features = memory_module.retrieve_features(name)

                if self.swap_strategy == 'min':
                    curr_mapping = extended_mapping[i]

                    # If the current image is not mapped to any other image, skip
                    if not torch.any(torch.cat([curr_mapping[:i], curr_mapping[i+1:]])):
                        continue

                    min_dists = nn_distances[i][curr_mapping].argmin(dim=0)
                    curr_nn_map = nn_map[i][curr_mapping][min_dists, torch.arange(vector_dim)]

                    curr_nn_distances = nn_distances[i][curr_mapping][min_dists, torch.arange(vector_dim)]
                    dist_thr = get_dynamic_threshold(curr_nn_distances) if self.dist_thr == 'dynamic' else self.dist_thr
                    dist_mask = curr_nn_distances < dist_thr
                    final_mask_tgt = attn_masks[i] & dist_mask

                    other_outputs = old_output[curr_mapping][min_dists, curr_nn_map][final_mask_tgt]

                    output[i][final_mask_tgt] = alpha * other_outputs + (1 - alpha)*old_output[i][final_mask_tgt]

            if anchors_cache and anchors_cache.is_cache_mode():
                if place_in_unet not in anchors_cache.h_out_cache:
                    anchors_cache.h_out_cache[place_in_unet] = {}

                anchors_cache.h_out_cache[place_in_unet][curr_iter] = output

        return output

    def inject_anchors(self, output, curr_iter, output_res, extended_mapping, place_in_unet, anchors_cache):
        curr_unet_part = place_in_unet.split('_')[0]

        # Inject only in the specified unet parts (up, mid, down)
        if (curr_unet_part not in self.inject_unet_parts) or output_res not in self.inject_res:
            return output

        bsz = output.shape[0]
        nn_map = self.nn_map[output_res]
        nn_distances = self.nn_distances[output_res]
        attn_masks = self.attn_masks[output_res]
        vector_dim = output_res**2

        alpha = next((alpha for min_range, max_range, alpha in self.inject_range_alpha if min_range <= curr_iter <= max_range), None)
        if alpha:

            anchor_outputs = anchors_cache.h_out_cache[place_in_unet][curr_iter]

            old_output = output#.clone()
            for i in range(bsz):
                other_outputs = []

                if self.swap_strategy == 'min':
                    min_dists = nn_distances[i].argmin(dim=0)
                    curr_nn_map = nn_map[i][min_dists, torch.arange(vector_dim)]

                    curr_nn_distances = nn_distances[i][min_dists, torch.arange(vector_dim)]
                    dist_thr = get_dynamic_threshold(curr_nn_distances) if self.dist_thr == 'dynamic' else self.dist_thr
                    dist_mask = curr_nn_distances < dist_thr
                    final_mask_tgt = attn_masks[i] & dist_mask

                    other_outputs = anchor_outputs[min_dists, curr_nn_map][final_mask_tgt]

                    output[i][final_mask_tgt] = alpha * other_outputs + (1 - alpha)*old_output[i][final_mask_tgt]

        return output
    
class QueryStore:
    def __init__(self, mode='store', t_range=[0, 1000], strength_start=1, strength_end=1):
        """
        Initialize an empty ActivationsStore
        """
        self.query_store = defaultdict(list)
        self.mode = mode
        self.t_range = t_range
        self.strengthes = np.linspace(strength_start, strength_end, (t_range[1] - t_range[0])+1)

    def set_mode(self, mode): # mode can be 'cache' or 'inject'
        self.mode = mode

    def cache_query(self, query, place_in_unet: str):
        self.query_store[place_in_unet] = query

    def inject_query(self, query, place_in_unet, t):
        if t >= self.t_range[0] and t <= self.t_range[1]:
            relative_t = t - self.t_range[0]
            strength = self.strengthes[relative_t]
            new_query = strength * self.query_store[place_in_unet] + (1 - strength) * query
        else:
            new_query = query

        return new_query
    
class AnchorCache:
    def __init__(self):
        self.input_h_cache = {} # place_in_unet, iter, h_in
        self.h_out_cache = {} # place_in_unet, iter, h_out
        self.anchors_last_mask = None
        self.dift_cache = None

        self.mode = 'cache' # mode can be 'cache' or 'inject'

    def is_inject_mode(self):
        return self.mode == 'inject'

    def is_cache_mode(self):
        return self.mode == 'cache'

class AttentionStore:
    def __init__(self, attention_store_kwargs):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.attn_res = attention_store_kwargs.get('attn_res', (32,32))
        self.token_indices = attention_store_kwargs['token_indices']
        bsz = self.token_indices.size(1)
        self.mask_background_query = attention_store_kwargs.get('mask_background_query', False)
        self.original_attn_masks = attention_store_kwargs.get('original_attn_masks', None)
        self.extended_mapping = attention_store_kwargs.get('extended_mapping', torch.ones(bsz, bsz).bool())
        self.mask_dropout = attention_store_kwargs.get('mask_dropout', 0.0)
        torch.manual_seed(0) # For dropout mask reproducibility

        self.curr_iter = 0
        self.ALL_RES = [32, 64]
        self.step_store = defaultdict(list)
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}
        self.last_mask_dropout = {res: None for res in self.ALL_RES}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads: int):
        if is_cross and attn.shape[1] == np.prod(self.attn_res):
            guidance_attention = attn[attn.size(0)//2:]
            batched_guidance_attention = guidance_attention.reshape([guidance_attention.shape[0]//attn_heads, attn_heads, *guidance_attention.shape[1:]])
            batched_guidance_attention = batched_guidance_attention.mean(dim=1)
            self.step_store[place_in_unet].append(batched_guidance_attention)

    def aggregate_last_steps_attention(self) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        attention_maps = torch.cat([torch.stack(x[-20:]) for x in self.step_store.values()]).mean(dim=0)
        bsz, wh, _ = attention_maps.shape

        # Create attention maps for each concept token, for each batch item
        agg_attn_maps = []
        for i in range(bsz):
            curr_prompt_indices = []

            for concept_token_indices in self.token_indices:
                if concept_token_indices[i] != -1:
                    curr_prompt_indices.append(attention_maps[i, :, concept_token_indices[i]].view(*self.attn_res))

            agg_attn_maps.append(torch.stack(curr_prompt_indices))

        # Upsample the attention maps to the target resolution
        # and create the attention masks, unifying masks across the different concepts
        for tgt_size in self.ALL_RES:
            pixels = tgt_size ** 2
            tgt_agg_attn_maps = [F.interpolate(x.unsqueeze(1), size=tgt_size, mode='bilinear').squeeze(1) for x in agg_attn_maps]

            attn_masks = []
            for batch_item_map in tgt_agg_attn_maps:
                concept_attn_masks = []

                for concept_maps in batch_item_map:
                    concept_attn_masks.append(torch.from_numpy(attn_map_to_binary(concept_maps, 1.)).to(attention_maps.device).bool().view(-1))

                concept_attn_masks = torch.stack(concept_attn_masks, dim=0).max(dim=0).values
                attn_masks.append(concept_attn_masks)

            attn_masks = torch.stack(attn_masks)
            self.last_mask[tgt_size] = attn_masks.clone()

            # Add mask dropout
            if self.curr_iter < 1000:
                rand_mask = (torch.rand_like(attn_masks.float()) < self.mask_dropout)
                attn_masks[rand_mask] = False

            self.last_mask_dropout[tgt_size] = attn_masks.clone()

    def get_extended_attn_mask_instance(self, width, i):
        attn_mask = self.last_mask_dropout[width]
        if attn_mask is None:
            return None
        
        n_patches = width**2
        

        output_attn_mask = torch.zeros((attn_mask.shape[0] * attn_mask.shape[1],), device=attn_mask.device, dtype=torch.bool)
        for j in range(attn_mask.shape[0]):
            if i==j:
                output_attn_mask[j*n_patches:(j+1)*n_patches] = 1
            else:
                if self.extended_mapping[i,j]:
                    if not self.mask_background_query:
                        output_attn_mask[j*n_patches:(j+1)*n_patches] = attn_mask[j].unsqueeze(0) #.expand(n_patches, -1)
                    else:
                        raise NotImplementedError('mask_background_query is not supported anymore')

        return output_attn_mask
    
class MetafacesAttnStoreProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, record_attention=True, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        # if attention_probs.requires_grad:
        if record_attention:
            self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class MetafacesExtendedAttnXFormersAttnProcessor:

    def __init__(self, place_in_unet, attnstore, extended_attn_kwargs, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        self.t_range = extended_attn_kwargs.get('t_range', [])
        self.extend_kv_unet_parts = extended_attn_kwargs.get('extend_kv_unet_parts', ['down', 'mid', 'up'])

        self.place_in_unet = place_in_unet
        self.curr_unet_part = self.place_in_unet.split('_')[0]
        self.attnstore = attnstore

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        perform_extend_attn: bool = False,
        query_store: Optional[QueryStore] = None,
        feature_injector: Optional[FeatureInjector] = None,
        anchors_cache: Optional[AnchorCache] = None,
        # character_names=None,
        **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, wh, channel = hidden_states.shape
            height = width = int(wh ** 0.5)

        is_cross = encoder_hidden_states is not None
        perform_extend_attn = perform_extend_attn and (not is_cross) and \
                              any([self.attnstore.curr_iter >= x[0] and self.attnstore.curr_iter <= x[1] for x in self.t_range]) and \
                              self.curr_unet_part in self.extend_kv_unet_parts

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if (self.curr_unet_part in self.extend_kv_unet_parts) and query_store and query_store.mode == 'cache':
            query_store.cache_query(query, self.place_in_unet)
        elif perform_extend_attn and query_store and query_store.mode == 'inject':
            query = query_store.inject_query(query, self.place_in_unet, self.attnstore.curr_iter)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()

        if perform_extend_attn:
            # Anchor Caching
            if anchors_cache and anchors_cache.is_cache_mode():
                if self.place_in_unet not in anchors_cache.input_h_cache:
                    anchors_cache.input_h_cache[self.place_in_unet] = {}

                # Hidden states inside the mask, for uncond (index 0) and cond (index 1) prompts
                subjects_hidden_states = torch.stack([x[self.attnstore.last_mask_dropout[width]] for x in hidden_states.chunk(2)])
                anchors_cache.input_h_cache[self.place_in_unet][self.attnstore.curr_iter] = subjects_hidden_states

            if anchors_cache and anchors_cache.is_inject_mode():
                # We make extended key and value by concatenating the original key and value with the query.
                anchors_hidden_states = anchors_cache.input_h_cache[self.place_in_unet][self.attnstore.curr_iter]

                anchors_keys = attn.to_k(anchors_hidden_states, *args)
                anchors_values = attn.to_v(anchors_hidden_states, *args)

                extended_key = torch.cat([torch.cat([key.chunk(2, dim=0)[x], anchors_keys[x].unsqueeze(0)], dim=1) for x in range(2)])
                extended_value = torch.cat([torch.cat([value.chunk(2, dim=0)[x], anchors_values[x].unsqueeze(0)], dim=1) for x in range(2)])

                extended_key = attn.head_to_batch_dim(extended_key).contiguous()
                extended_value = attn.head_to_batch_dim(extended_value).contiguous()

                # attn_masks needs to be of shape [batch_size, query_tokens, key_tokens]
                hidden_states = xformers.ops.memory_efficient_attention(
                    query, extended_key, extended_value,  op=self.attention_op, scale=attn.scale
                )
            else:
                # Pre-allocate the output tensor
                ex_out = torch.empty_like(query)

                for i in range(batch_size):
                    start_idx = i * attn.heads
                    end_idx = start_idx + attn.heads

                    attention_mask = self.attnstore.get_extended_attn_mask_instance(width, i%(batch_size//2))

                    curr_q = query[start_idx:end_idx]

                    if i < batch_size//2:
                        curr_k = key[:batch_size//2]
                        curr_v = value[:batch_size//2]
                    else:
                        curr_k = key[batch_size//2:]
                        curr_v = value[batch_size//2:]

                    curr_k = curr_k.flatten(0,1)[attention_mask].unsqueeze(0)
                    curr_v = curr_v.flatten(0,1)[attention_mask].unsqueeze(0)

                    curr_k = attn.head_to_batch_dim(curr_k).contiguous()
                    curr_v = attn.head_to_batch_dim(curr_v).contiguous()

                    hidden_states = xformers.ops.memory_efficient_attention(
                        curr_q, curr_k, curr_v, 
                        op=self.attention_op, scale=attn.scale
                    )

                    ex_out[start_idx:end_idx] = hidden_states

                hidden_states = ex_out
        else:
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()

            # attn_masks needs to be of shape [batch_size, query_tokens, key_tokens]
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, op=self.attention_op, scale=attn.scale
            )

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if (feature_injector is not None):
            output_res = int(hidden_states.shape[1] ** 0.5)

            if anchors_cache and anchors_cache.is_inject_mode():
                hidden_states[batch_size//2:] = feature_injector.inject_anchors(hidden_states[batch_size//2:], self.attnstore.curr_iter, output_res, self.attnstore.extended_mapping, self.place_in_unet, anchors_cache)
            else:
                hidden_states[batch_size//2:] = feature_injector.inject_outputs(hidden_states[batch_size//2:], self.attnstore.curr_iter, output_res, self.attnstore.extended_mapping, self.place_in_unet, anchors_cache)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
       
def register_extended_self_attn(unet, attnstore, extended_attn_kwargs):
    attn_procs = {}
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attn = (i % 2 == 0)

        if name.startswith("mid_block"):
            place_in_unet = f"mid_{i}"
        elif name.startswith("up_blocks"):
            place_in_unet = f"up_{i}"
        elif name.startswith("down_blocks"):
            place_in_unet = f"down_{i}"
        else:
            continue

        if is_self_attn:
            attn_procs[name] = MetafacesExtendedAttnXFormersAttnProcessor(place_in_unet, attnstore, extended_attn_kwargs)
        else:
            attn_procs[name] = MetafacesAttnStoreProcessor(attnstore, place_in_unet)

    unet.set_attn_processor(attn_procs)

@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None

class MetafacesSDXLUNet2DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
    ):
        super().__init__()

        self.latent_store = DIFTLatentStore(steps=[261], up_ft_indices=[0])
        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
        
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                attention_type=attention_type,
            )
        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                attention_head_dim=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                skip_time_act=resnet_skip_time_act,
                only_cross_attention=mid_block_only_cross_attention,
                cross_attention_norm=cross_attention_norm,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, tuple) or isinstance(cross_attention_dim, list):
                positive_len = cross_attention_dim[0]

            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = PositionNet(
                positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type
            )

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
  
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors
    
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def enable_freeu(self, s1, s2, b1, b2):
        
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
    
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
    
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    scale=lora_scale,
                )
            
            self.latent_store(sample.detach(), t=timestep, layer_index=i)

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


class MetafacesExtendAttnSDXLPipeline(
    StableDiffusionXLPipeline
):


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        
        attention_store_kwargs: Optional[Dict] = None,
        extended_attn_kwargs: Optional[Dict] = None,
        share_queries: bool = False,
        query_store_kwargs: Optional[Dict] = {},
        feature_injector: Optional[FeatureInjector] = None,
        anchors_cache: Optional[AnchorCache] = None,

        instance_latents: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if share_queries:
            query_store = QueryStore(**query_store_kwargs)
        else:
            query_store = None

        self.attention_store = AttentionStore(attention_store_kwargs)
        register_extended_self_attn(self.unet, self.attention_store, extended_attn_kwargs)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        if instance_latents is not None:
            n_instances = instance_latents.shape[0]
            instance_noise = latents[:n_instances].clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.attention_store.curr_iter = i

                if instance_latents is not None:
                    noised_instances = self.scheduler.add_noise(instance_latents, instance_noise, t.repeat(n_instances).long())
                    latents[:n_instances] = noised_instances

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                if share_queries and (i >= query_store.t_range[0] and i <= query_store.t_range[1]):
                    query_store.set_mode('cache')
                    noise_pred_vanilla = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs={'query_store': query_store, 
                                                'perform_extend_attn': False, 
                                                'record_attention': False},
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    query_store.set_mode('inject')

                noise_pred = self.unet(
                       latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs={'query_store': query_store, 
                                                'perform_extend_attn': True, 
                                                'record_attention': True, 
                                                'feature_injector': feature_injector,
                                                'anchors_cache': anchors_cache},
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    # xm.mark_step()
                    pass
                
                # Update attention store mask
                self.attention_store.aggregate_last_steps_attention()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

def load_pipeline(gpu_id=0):
    float_type = torch.float16
    sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    unet = MetafacesSDXLUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=float_type)
    scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

    story_pipeline = MetafacesExtendAttnSDXLPipeline.from_pretrained(
        sd_id, unet=unet, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler
    ).to(device)
    story_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    
    return story_pipeline

def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        concept_token = [concept_token]

    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']

    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    for i, token_id in enumerate(concept_token_id):
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc

    return token_indices

def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping

def create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type):
    # if seed is int
    if isinstance(seed, int):
        g = torch.Generator('cuda').manual_seed(seed)
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=float_type)
    elif isinstance(seed, list):
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = torch.empty(shape, device=device, dtype=float_type)
        for i, seed_i in enumerate(seed):
            g = torch.Generator('cuda').manual_seed(seed_i)
            curr_latent = randn_tensor(shape, generator=g, device=device, dtype=float_type)
            latents[i] = curr_latent[i]

    if same_latent:
        latents = latents[:1].repeat(batch_size, 1, 1, 1)

    return latents, g

class MemoryModule:
    def __init__(self):
        # Dictionary to store features for each character
        self.memory = {}

    def store_features(self, character_name, features):
        """
        Store extracted features for a given character.
        Args:
            character_name (str): The name of the character.
            features (torch.Tensor): Extracted feature tensor from the UNet.
        """
        self.memory[character_name] = features.detach().clone()

    def retrieve_features(self, character_name):
        """
        Retrieve stored features for a given character.
        Args:
            character_name (str): The name of the character.
        Returns:
            torch.Tensor: Stored feature tensor.
        """
        if character_name in self.memory:
            return self.memory[character_name].clone()
        else:
            return None

# Load models
dino_model = GroundingSAM.load_grounding_dino(repo_id, dino_checkpoint_filename, dino_config_filename, device=device)

sam_model = GroundingSAM.load_sam(sam_checkpoint_path, device=device)

# Create an instance of GroundingSAM
grounding_sam = GroundingSAM(dino_model=dino_model, sam_model=sam_model, device=device)


def generate_mask_and_extract_tensor(image_tensor, grounding_sam, prompt, device):
    """
    Generate and display the mask for the given image.

    Args:
        image_path (str): Path to the input image.
        grounding_sam (GroundingSAM): Initialized instance of GroundingSAM.
        prompt (str): Prompt describing the object to detect in the image.
        device (str): Device to run the model on ("cuda" or "cpu").
    """

    image_source = (image_tensor * 255).byte().permute(1, 2, 0).cpu().numpy()

    annotated_frame, mask, _ = grounding_sam.detect_and_segment(
        image=image_tensor,
        image_source=image_source,
        prompt=prompt,
        device=device,
    )

    mask = torch.tensor(mask).to(device).bool()

    extracted_tensor = image_tensor * mask.unsqueeze(0)

    y_indices, x_indices = torch.where(mask)
    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()

        cropped_tensor = image_tensor[:, y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        cropped_annotated_frame = annotated_frame[y_min:y_max, x_min:x_max, :]

        cropped_image = (cropped_tensor * 255).byte().permute(1, 2, 0).cpu().numpy()

        cropped_mask = cropped_mask.cpu().numpy()

        print(f"Cropping character region at ({x_min}, {y_min}) to ({x_max}, {y_max})")

    else:
        print(f"No bounding box found for '{prompt}'. Mask might be empty.")
        cropped_tensor = extracted_tensor
    
    return cropped_tensor, cropped_mask, cropped_image, cropped_annotated_frame

def generate_prompts(CONSTANT, character_descriptions, scene_prompts):
    prompts = []
    for scene_prompt in scene_prompts:
        prompt = f"{CONSTANT} {scene_prompt}"
        for name, desc in character_descriptions.items():
            prompt = prompt.replace(name, desc)
        prompts.append(prompt)
    return prompts

import math

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def get_position(row, col, rows, cols):
    if rows == 1:
        if col == 0:
            return "on the left side"
        elif col == cols - 1:
            return "on the right side"
        else:
            return "in the middle"
    elif cols == 1:
        if row == 0:
            return "on the top"
        elif row == rows - 1:
            return "on the bottom"
        else:
            return "in the middle"
    else:
        position = ""
        if row == 0:
            position += "on the top row"
        elif row == rows - 1:
            position += "on the bottom row"
        else:
            position += "in the middle row"
        
        if col == 0:
            position += " on the left side"
        elif col == cols - 1:
            position += " on the right side"
        else:
            position += " in the middle"
        
        return position

def generate_dynamic_anchor_prompt(CHARACTER_DESCRIPTIONS, CONSTANT):
    num_characters = len(CHARACTER_DESCRIPTIONS)
    
    if num_characters == 1:
        return f"{CONSTANT} {next(iter(CHARACTER_DESCRIPTIONS.values()))}"

    
    grid_size = math.ceil(math.sqrt(num_characters))  
    rows = math.ceil(num_characters / grid_size)
    cols = grid_size
    
    prompt = f"{CONSTANT} a split-image composition: "
    
    character_list = list(CHARACTER_DESCRIPTIONS.items())

    for i, (name, desc) in enumerate(character_list):
        row = i // cols
        col = i % cols
        position = get_position(row, col, rows, cols)
        prompt += f"{position}, {desc}."

    if num_characters > 1:
        prompt

    return prompt.strip()


memory_module = MemoryModule()

def run_batch_generation(story_pipeline, CONSTANT, CHARACTER_DESCRIPTIONS, SCENE_PROMPTS, concept_token, memory_module,
                        seed=40, n_steps=50, mask_dropout=0.5,
                        same_latent=False, share_queries=True,
                        perform_sdsa=True, perform_injection=True,
                        downscale_rate=4, n_achors=2):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs= {'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}


    anchor_prompt = generate_dynamic_anchor_prompt(CHARACTER_DESCRIPTIONS, CONSTANT)
    print(anchor_prompt)
    modified_scene_prompts = [anchor_prompt] + SCENE_PROMPTS  

    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}


    prompts = generate_prompts(CONSTANT, CHARACTER_DESCRIPTIONS, modified_scene_prompts)

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)
    anchor_mappings = create_anchor_mapping(batch_size, anchor_indices=list(range(n_achors)))

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout,
        'extended_mapping': anchor_mappings
    }

    latents, g = create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type)

    out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                        attention_store_kwargs=default_attention_store_kwargs,
                        extended_attn_kwargs=extended_attn_kwargs,
                        share_queries=share_queries,
                        query_store_kwargs=query_store_kwargs,
                        num_inference_steps=n_steps)
    last_masks = story_pipeline.attention_store.last_mask

    dift_features = unet.latent_store.dift_features['261_0'][batch_size:]
    dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)

    nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)

    torch.cuda.empty_cache()
    gc.collect()
    
    if perform_injection:
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, memory_module=memory_module, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

    
        print("Generating final scene with consistent character features...")
        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            num_inference_steps=n_steps)

  
    scene_images = [np.array(x) for x in out.images]


    anchor_image = scene_images[0]  

    print("Using the first scene image as the anchor image...")

    print("Extracting individual characters from anchor image using GroundingSAM...")

    extracted_character_images = {}
    extracted_character_tensors = {}

    masks_list = []
    cropped_images_list = []
    cropped_annotated_frames_list = []

    for name, desc in CHARACTER_DESCRIPTIONS.items():
        anchor_tensor = torch.from_numpy(anchor_image).permute(2, 0, 1).float() / 255.0
        anchor_tensor = anchor_tensor.to(device)

        extracted_tensor, mask, cropped_image, cropped_annotated_frame = generate_mask_and_extract_tensor(
            image_tensor=anchor_tensor,
            grounding_sam=grounding_sam,
            prompt=name, 
            device=device
        )

        print(f"Extracted tensor for character {name}...")

        print(f"Storing extracted features for {name}...")
        memory_module.store_features(name, extracted_tensor)

        masks_list.append(mask)
        cropped_images_list.append(cropped_image)
        cropped_annotated_frames_list.append(cropped_annotated_frame)

    remaining_scene_images = scene_images[1:]

    print("Displaying scene images...")

    return scene_images, masks_list, cropped_images_list, cropped_annotated_frames_list, anchor_prompt, prompts