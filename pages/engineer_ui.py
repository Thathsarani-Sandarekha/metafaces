import os
import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

from utils_new import *
from evaluation import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# CONSTANTS & SETUP
# -----------------------------
st.set_page_config(page_title="ML Engineer Dashboard", layout="wide")
st.title("🧠 ML Engineer Dashboard")

CONSTANT = "A hyper-realistic digital painting of"
gpu_id = 0
story_pipeline = load_pipeline(gpu_id=gpu_id)

# -----------------------------
# IMAGE SAVING FUNCTION
# -----------------------------
def save_image_locally(image_array, filename_prefix, idx, output_dir="generated"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename_prefix}_{idx}.png")
    img = Image.fromarray(image_array)
    img.save(filepath)
    return filepath

# -----------------------------
# CHARACTER DESCRIPTION SECTION
# -----------------------------
st.header("🧍‍♂️ Character Descriptions")
col1, col2 = st.columns(2)

with col1:
    char1_name = st.text_input("Character 1 Name", value="")
    char1_desc = st.text_area("Description for Character 1", value="")

with col2:
    char2_name = st.text_input("Character 2 Name", value="")
    char2_desc = st.text_area("Description for Character 2", value="")

CHARACTER_DESCRIPTIONS = {
    char1_name: char1_desc,
    char2_name: char2_desc
}

# -----------------------------
# SCENE PROMPTS SECTION
# -----------------------------
st.header("🎬 Scene Prompts")
scene_prompts = []
for i in range(3):
    scene_prompt = st.text_area(f"Scene Prompt {i + 1}", value=f"A scene description {i + 1}")
    scene_prompts.append(scene_prompt)
# -----------------------------
# STATE INITIALIZATION
# -----------------------------
if "scene_images" not in st.session_state:
    st.session_state.scene_images = []
    st.session_state.scene_image_paths = []
    st.session_state.anchor_image_paths = []
    st.session_state.masks_list = []
    st.session_state.cropped_images_list = []
    st.session_state.cropped_annotated_frames_list = []
    st.session_state.scene_prompts = []
    st.session_state.char_desc = {}

# -----------------------------
# GENERATE BUTTON
# -----------------------------
if st.button("🚀 Generate Images"):
    with st.spinner("Generating images..."):
        try:
            scene_images, masks_list, cropped_images_list, cropped_annotated_frames_list, anchor_prompt, prompts = run_batch_generation(
                story_pipeline=story_pipeline,
                CONSTANT=CONSTANT,
                CHARACTER_DESCRIPTIONS=CHARACTER_DESCRIPTIONS,
                SCENE_PROMPTS=scene_prompts,
                concept_token=list(CHARACTER_DESCRIPTIONS.keys()),
                memory_module=memory_module
            )

            # Save results to session_state
            st.session_state.scene_images = scene_images
            st.session_state.masks_list = masks_list
            st.session_state.cropped_images_list = cropped_images_list
            st.session_state.cropped_annotated_frames_list = cropped_annotated_frames_list
            st.session_state.scene_prompts = scene_prompts
            st.session_state.char_desc = CHARACTER_DESCRIPTIONS

            # Save generated images
            st.session_state.scene_image_paths = [
                save_image_locally(img, "scene", idx) for idx, img in enumerate(scene_images[1:], start=1)
            ]
            st.session_state.anchor_image_paths = [
                save_image_locally(img, "anchor", idx) for idx, img in enumerate(cropped_images_list)
            ]

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# -----------------------------
# DISPLAY GENERATED RESULTS
# -----------------------------
if st.session_state.scene_images:
    st.subheader("📍 Anchor Images:")
    cols_anchor = st.columns(len(st.session_state.cropped_images_list))
    for idx, anchor_img in enumerate(st.session_state.cropped_images_list):
        with cols_anchor[idx]:
            anchor_pil = Image.fromarray(anchor_img)
            st.image(anchor_pil, caption=f"Anchor {idx + 1}")

    st.subheader("🎬 Scene Images:")
    cols_scene = st.columns(len(st.session_state.scene_images[1:]))
    for idx, img in enumerate(st.session_state.scene_images[1:], start=1):
        with cols_scene[idx - 1]:
            img_pil = Image.fromarray(img)
            st.image(img_pil, caption=f"Scene {idx}")

    st.subheader("🖼️ Extracted Masks:")
    cols_mask = st.columns(len(st.session_state.masks_list))
    for idx, mask in enumerate(st.session_state.masks_list):
        with cols_mask[idx]:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            st.image(mask_pil, caption=f"Mask {idx + 1}")

    st.subheader("📌 Annotated Frames:")
    cols_frame = st.columns(len(st.session_state.cropped_annotated_frames_list))
    for idx, frame in enumerate(st.session_state.cropped_annotated_frames_list):
        with cols_frame[idx]:
            frame_pil = Image.fromarray(frame)
            st.image(frame_pil, caption=f"Frame {idx + 1}")

# # -----------------------------
# # EVALUATION BUTTON
# # -----------------------------
# if st.session_state.scene_image_paths and st.button("📊 Evaluate Results"):
#     with st.spinner("Evaluating image consistency and quality..."):
#         try:
#             # Evaluate CLIP similarity for each scene
#             st.header("📈 Evaluation Metrics")

#             # for i, (img_path, prompt) in enumerate(zip(st.session_state.scene_image_paths, st.session_state.scene_prompts)):
#             for i, img_path in enumerate(st.session_state.scene_image_paths):
#                 logits, probs = evaluate_clip_similarity(img_path, st.session_state.scene_prompts)
#                 st.markdown(f"📌 **Scene {i+1}**")
#                 st.markdown(f"🔎 **CLIP Probabilities:** `{probs}`")
                
#                 for j, prob in enumerate(probs.squeeze().tolist()):
#                     st.markdown(f"🔎 **Prompt {j+1}:** `{st.session_state.scene_prompts[j]}` → **CLIP Probability:** `{prob:.4f}`")


#             # DreamSim: Evaluate each anchor against corresponding segmented region
#             for i, (anchor_path, scene_path) in enumerate(zip(st.session_state.anchor_image_paths, st.session_state.scene_image_paths)):
#                 anchor_img = anchor_path
#                 scene_img = np.array(Image.open(scene_path))
#                 character_prompt = list(st.session_state.char_desc.values())[i % 2]  

#                 score = evaluate_dreamsim_similarity(
#                     anchor_image=anchor_img,
#                     scene_image=scene_img,
#                     prompt=character_prompt,
#                     device=torch.device("cuda"),
#                     visualize=True
#                 )
#                 st.markdown(f"🎭 **DreamSim Similarity [Anchor {i+1} vs Scene {i+1}]:** `{score:.4f}`")

#         except Exception as e:
#             st.error(f"❌ Evaluation failed: {e}")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🧪 *Advanced Interface for Engineers: Evaluate consistency and semantic alignment*")
