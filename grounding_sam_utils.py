import torch
from PIL import Image
from Grounded_Segment_Anything.GroundingDINO.groundingdino.models import build_model as build_dino
from Grounded_Segment_Anything.segment_anything.build.lib.segment_anything import build_sam, SamPredictor
from huggingface_hub import hf_hub_download
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.slconfig import SLConfig
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.utils import clean_state_dict
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, Model
from torchvision.ops import box_convert
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util import box_ops
import supervision as sv
import numpy as np
import cv2

class GroundingSAM:
    def __init__(self, dino_model, sam_model, device="cpu"):
        """
        Initialize the GroundingSAM class with Grounding DINO and SAM models.
        """
        self.dino = dino_model
        self.sam = sam_model
        self.device = device

    @staticmethod
    def load_grounding_dino(repo_id, filename, ckpt_config_filename, device="cpu"):
        """
        Load the Grounding DINO model.
        """
        # Download config and model checkpoint
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        model = build_dino(args)

        # Load model weights
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        return model

    @staticmethod
    def load_sam(checkpoint_path, device="cpu"):
        """
        Load the SAM model.
        """
        sam_model = build_sam(checkpoint=checkpoint_path)
        return SamPredictor(sam_model.to(device))

    def annotate_boxes(self, image_source, boxes, logits, phrases):
        """
        Annotate the image with bounding boxes and labels.
        """
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])  # Scale boxes to image size
        xyxy_boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # Assign placeholder class IDs (indices of the detected boxes)
        class_ids = np.arange(len(phrases))
        
        # Create detections with class IDs
        detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)

        # Create labels
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

        # Annotate the image
        annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame  # Convert back to RGB


    def detect_bounding_boxes(self, image, image_source, prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Detect bounding boxes using Grounding DINO.
        """

        # Run Grounding DINO detection
        boxes, logits, phrases = predict(
            model=self.dino,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        annotated_frame = self.annotate_boxes(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        return annotated_frame, boxes

    def segment_masks(self, image_source, boxes, device):
        """
        Generate fine-grained segmentation masks using SAM.
        """
        self.sam.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_xyxy.to(device), image_source.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.cpu()

    def draw_mask(self, mask, annotated_frame, random_color=True):
        """
        Overlay segmentation mask on the image.
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(annotated_frame).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image.numpy() * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    def detect_and_segment(self, image, image_source, prompt, device):
        """
        Combined detection and segmentation workflow.
        """
        # Detect bounding boxes
        annotated_frame, boxes = self.detect_bounding_boxes(image, image_source, prompt)
        
        # Generate segmentation masks
        segmented_frame_masks = self.segment_masks(image_source, boxes, device)
        mask = segmented_frame_masks[0][0].cpu().numpy()
        inverted_mask = ((1 - mask) * 255).astype(np.uint8)

        annotated_frame_with_mask = self.draw_mask(segmented_frame_masks[0][0], annotated_frame)
        
        return annotated_frame_with_mask, mask, inverted_mask
