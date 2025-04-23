# 🧠 MetaFaces: Dynamic Memory Networks for Consistent Character Representation in Multi-Image Visual Storytelling

📄 **Project Page:** [https://thathsarani-sandarekha.github.io/metafaces/](https://thathsarani-sandarekha.github.io/metafaces_web_page/)

---

## 📚 About the Project

MetaFaces is a novel framework that addresses the challenge of consistent multi-character representation in image-based visual storytelling. Built on top of Stable Diffusion XL, our system introduces:

- 🧠 Dynamic memory modules to retain character-specific features  
- 🧲 Anchor-based injection mechanisms for preserving identity  
- 🎯 Attention-guided segmentation using Grounding DINO + SAM to align characters across scenes  

This project was developed as part of our undergraduate research to advance controllable and consistent character generation in diffusion models.

---

## ⚙️ Setup Instructions

To get started:

1. Clone this repository
2. Set up the required Python environment and dependencies - use requirements.txt
3. Make sure Grounding SAM is configured correctly (see below)
### 🔁 Place Grounding SAM

Our system integrates Grounding SAM for character segmentation and mask guidance. You must clone the official Grounding SAM repository and **place the folder** inside this repo with the customized version provided here.

🔗 **Grounding SAM Repository:** [[https://github.com/IDEA-Research/GroundingSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)]

4. ✅ **Run the main app using**:
```bash
python app.py
