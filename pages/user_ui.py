import streamlit as st

from utils_new import *

# -----------------------------
# CONSTANT PART
# -----------------------------
CONSTANT = "A hyper-realistic digital painting of"

gpu_id = 0
story_pipeline = load_pipeline(gpu_id=gpu_id)

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("Story-Based Multi-Character Image Generation")

## Input for CHARACTER_DESCRIPTIONS (Fixed to two characters)
st.header("Character Descriptions")
col1, col2 = st.columns(2)

with col1:
    char1_name = st.text_input("Character 1 Name", value="")
    char1_desc = st.text_area("Description for Character 1", value="")

with col2:
    char2_name = st.text_input("Character 2 Name", value="")
    char2_desc = st.text_area("Description for Character 2", value="")

# Create character descriptions dictionary
CHARACTER_DESCRIPTIONS = {
    char1_name: char1_desc,
    char2_name: char2_desc
}

# Input for SCENE_PROMPTS (Fixed to 3 prompts)
st.header("Scene Prompts")
scene_prompts = []
for i in range(3):
    scene_prompt = st.text_area(f"Scene Prompt {i + 1}", value=f"A scene description {i + 1}")
    scene_prompts.append(scene_prompt)


# ✅ Add a "Generate" button at the bottom
if st.button("🚀 Generate"):
    with st.spinner("Generating images..."):
        try:
            # Run generation function
            scene_images, masks_list, cropped_images_list, cropped_annotated_frames_list, anchor_prompt, prompts = run_batch_generation(
                story_pipeline=story_pipeline,
                CONSTANT=CONSTANT,
                CHARACTER_DESCRIPTIONS=CHARACTER_DESCRIPTIONS,
                SCENE_PROMPTS=scene_prompts,
                concept_token=list(CHARACTER_DESCRIPTIONS.keys()),
                memory_module=memory_module
            )

            # Display Anchor Images (First Row)
            st.subheader("🧷 Anchor Images:")
            cols_anchor = st.columns(len(cropped_images_list))
            for idx, anchor_img in enumerate(cropped_images_list):
                with cols_anchor[idx]:
                    anchor_img_pil = Image.fromarray(anchor_img)
                    st.image(anchor_img_pil, caption=f"Anchor {idx + 1}")

            # Display Scene Images (Second Row)
            st.subheader("🎬 Generated Scene Images:")
            cols_scene = st.columns(len(scene_images[1:]))  # Skip index 0 if reserved
            for idx, img in enumerate(scene_images[1:], start=1):
                with cols_scene[idx - 1]:
                    img_pil = Image.fromarray(img)
                    st.image(img_pil, caption=f"Scene {idx}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("💡 *Consistent Character Generation using Diffusion Models*")