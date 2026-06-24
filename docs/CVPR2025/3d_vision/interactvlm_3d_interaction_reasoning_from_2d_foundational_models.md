---
title: >-
  [Paper Note] InteractVLM: 3D Interaction Reasoning from 2D Foundational Models
description: >-
  [CVPR 2025][3D Vision][Human-Object Interaction Reconstruction] InteractVLM leverages the extensive visual knowledge of large-scale vision-language models (VLMs) to transfer the reasoning capabilities of 2D foundational models to 3D space via a "Render-Localize-Lift" framework. It realizes 3D contact point estimation for humans and objects from a single in-the-wild image, applying it to joint human-object interaction reconstruction and achieving a 20.6% F1 score improvement i…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human-Object Interaction Reconstruction"
  - "3D Contact Estimation"
  - "Vision-Language Models"
  - "Multi-View Localization"
  - "Semantic Contact"
date: 2026-05-08
content_hash: ef607e70ed0394e9
---

# InteractVLM: 3D Interaction Reasoning from 2D Foundational Models

**Conference**: CVPR 2025  
**arXiv**: [2504.05303](https://arxiv.org/abs/2504.05303)  
**Code**: [https://interactvlm.is.tue.mpg.de](https://interactvlm.is.tue.mpg.de)  
**Area**: 3D Vision  
**Keywords**: Human-Object Interaction Reconstruction, 3D Contact Estimation, Vision-Language Models, Multi-View Localization, Semantic Contact

## TL;DR
InteractVLM leverages the extensive visual knowledge of large-scale vision-language models (VLMs) to transfer the reasoning capabilities of 2D foundational models to 3D space via a "Render-Localize-Lift" framework. It realizes 3D contact point estimation for humans and objects from a single in-the-wild image, applying it to joint human-object interaction reconstruction and achieving a 20.6% F1 score improvement in contact estimation tasks.

## Background & Motivation

1. **Background**: 3D human-object interaction (HOI) reconstruction is crucial for applications like robotics and mixed reality. Existing methods estimate either 3D humans or 3D objects, but rarely combine both. Knowing the contact between humans and objects can significantly improve joint reconstruction.

2. **Limitations of Prior Work**:
    - Existing contact estimation methods (such as DECO) rely on expensive motion capture systems or manually annotated 3D contact data, which limits scalability.
    - Existing methods treat contact as simple binary classification, failing to consider the semantic relationships in multi-object interactions.
    - In-the-wild images lack paired ground-truth 3D contact annotations.

3. **Key Challenge**: 3D contact estimation requires 3D spatial understanding, but large-scale annotated 3D contact data is scarce. Meanwhile, VLMs with extensive visual knowledge can only reason in 2D space.

4. **Goal**:
    - How to leverage VLM knowledge to compensate for the lack of 3D contact annotations.
    - How to translate the 2D reasoning capability of VLMs into 3D contact localization capability.
    - Propose a new task of "semantic human contact": given an object label, predict the body contact points associated with that object.

5. **Key Insight**: Although VLMs only reason in 2D, they contain rich commonsense knowledge about human-object interactions, which can be unlocked through fine-tuning with a small amount of 3D data.

6. **Core Idea**: Down-project 3D problems to 2D via multi-view rendering to let the VLM guide contact localization, and then back-project and lift them to 3D space.

## Method

The core mechanism of InteractVLM is: first, let the VLM "understand" the interaction image and generate contact reasoning tokens, then precisely label contact regions on the 3D geometric surface using an innovative multi-view localization (MV-Loc) module. The entire system elegantly combines the 2D semantic understanding of VLMs with 3D geometric awareness.

### Overall Architecture

The input is an in-the-wild RGB image, and the output consists of 3D contact points on the human body and object surfaces. The system contains two major components:
1. **VLM Inference Module**: Receives the image and text prompts, generates text output containing `<HCON>` and `<OCON>` contact tokens, and produces guiding embeddings.
2. **MV-Loc Multi-View Localization Module**: Translates VLM 2D reasoning into 3D contact prediction through a three-step "Render-Localize-Lift" (RLL) framework.

### Key Designs

1. **VLM Interaction Inference Module ($\Psi$):**
    - **Function**: Understand the human-object interaction scene from the RGB image and generate contact reasoning information.
    - **Mechanism**: Add two special tokens `<HCON>` (human contact) and `<OCON>` (object contact) to the vocabulary of LLaVA, and fine-tune the VLM via LoRA to learn to generate text containing these tokens. Extract the embeddings corresponding to these tokens in the final layer of the VLM, and project them through a projection layer $\Gamma$ to obtain feature embeddings $E^H$ and $E^O$ as semantic guidance signals for subsequent localization. During training, cross-entropy loss $\mathcal{L}_{token}$ for token prediction is utilized.
    - **Design Motivation**: VLMs trained on internet-scale data possess extensive commonsense knowledge about human-object interactions; fine-tuning with a small amount of 3D contact data can activate this knowledge for contact reasoning.

2. **Render-Localize-Lift (RLL) Framework:**
    - **Function**: Translate the 3D contact localization problem into 2D segmentation, and then map it back to 3D.
    - **Mechanism**: A three-step process: (1) **Render**: Render the SMPL+H human mesh (in star-shaped canonical pose) and object mesh (retrieved from Objaverse via OpenShape) into 2D images from $J$ fixed viewpoints, using normal shading to enhance cross-view correspondence; (2) **Localize**: Feed the rendered images into the encoder and decoder of SAM, predicting 2D contact masks under the guidance of VLM embeddings; (3) **Lift**: Lift the 2D contact to 3D contact points using pre-computed 2D-3D pixel-to-vertex mappings.
    - **Design Motivation**: Directly localizing contact in 3D space is challenging for existing foundational models, while downprojecting to 2D allows reusing powerful 2D segmentation models like SAM.

3. **FeatLift Feature Lifting Network ($\Phi$):**
    - **Function**: Convert the 2D feature embeddings generated by the VLM into 3D-aware features to ensure multi-view consistency.
    - **Mechanism**: Design a lifting network that takes 2D embeddings $E^{H,O}$ and camera parameters $K$ as input, and outputs 3D-aware embeddings $E^{H,O}_{3D} = \Phi(E^{H,O}, K)$. The network includes a spatial understanding network (two-layer 128-dimensional FC+ReLU) and view-specific 256-dimensional transforms. By encoding camera parameters into the embeddings, contact predictions across different views are kept consistent.
    - **Design Motivation**: Simply concatenating camera parameters to multi-view rendering is insufficient to guarantee 3D consistency; the features must be explicitly made to "perceive" 3D spatial relationships.

### Loss & Training

The total loss consists of several components:
- **Token Prediction Loss** $\mathcal{L}_{token}$: Cross-entropy loss to supervise the VLM in generating correct contact tokens.
- **2D Mask Loss**: focal-weighted BCE + Dice loss to supervise 2D contact masks.
- **3D Human Contact Loss** $\mathcal{L}^H_C$: focal loss + L1 sparse regularization, encouraging precise localization while avoiding false positives.
- **3D Object Contact Loss** $\mathcal{L}^O_C$: Dice loss + MSE loss.

Training employs LoRA (rank 8) to fine-tune the VLM, with the image encoder frozen and the decoder trained separately. DeepSpeed + bfloat16 mixed precision is used, training for 30 epochs on 4 A100 GPUs.

## Key Experimental Results

### Main Results

**Binary Human Contact Estimation (DAMON Dataset)**

| Method | F1 (%) | Precision (%) | Recall (%) | Geodesic (cm) |
|------|--------|---------------|------------|---------------|
| POSA^PIXIE | 31.0 | 42.0 | 34.0 | 33.00 |
| BSTRO | 46.0 | 51.0 | 53.0 | 38.06 |
| DECO | 55.0 | 65.0 | 57.0 | 21.32 |
| **InteractVLM** | **75.6** | **75.2** | **76.0** | **2.89** |

F1 improved by 20.6%, and the geodesic distance dropped significantly from 21.32cm to 2.89cm.

**Object Affordance Prediction (PIAD Dataset)** also achieved SOTA performance.

### Ablation Study

| Configuration | Description |
|------|------|
| Different Training Data Volumes | Surpasses DECO trained on full data using only 40% of DAMON data |
| Semantic Contact vs Binary Contact | Semantic contact distinguishes multi-object interactions, which conventional methods cannot do |
| Auxiliary VQA Data Usage | VQA data generated by GPT-4o helps training |
| Comparison with LEMON | Although LEMON uses paired data, InteractVLM achieves comparable performance with unpaired data |

### Key Findings
- The commonsense knowledge of VLMs is the key factor for significant performance gains; even when trained on the same data as DECO, the VLM's knowledge alone brings a 20% F1 improvement.
- The geodesic distance drops from 21cm to 2.89cm, indicating a qualitative leap in contact localization accuracy.
- The method can scale to 80 human contact categories and 32 object affordance categories, far exceeding the prior limit of 21 categories.
- Extreme data efficiency: 40% of training data exceeds fully supervised DECO.

## Highlights & Insights
- **Render-Localize-Lift Framework**: Elegantly translates 3D problems into 2D problems. The core cleverness lies in utilizing known 3D geometry to eliminate the depth ambiguity of 2D-to-3D back-projection. This paradigm can be transferred to any task requiring fine-grained labeling on 3D surfaces.
- **Semantic Contact Task**: Upgrades from "whether contact exists" to "contact with which object". The definition of this problem itself is valuable and translatable to scenarios such as robotic grasp planning.
- **VLM Knowledge Distillation**: The idea of fine-tuning VLMs with a small amount of 3D data to acquire 3D understanding capabilities can be extended to other 3D tasks, avoiding large-scale 3D annotations.

## Limitations & Future Work
- 3D object shapes are retrieved from a database via OpenShape, which might fail for novel objects not present in the database.
- The human body uses a canonical star-shaped pose; self-occlusion issues can occur in rendering under extreme poses.
- Validation is conducted only on limited datasets (DAMON, PIAD), and larger-scale evaluations in real-world scenarios are still missing.
- Optimization-based 3D HOI reconstruction relies on contact quality, so incorrect contacts can propagate to the reconstruction results.

## Related Work & Insights
- **vs DECO**: DECO directly regresses vertex contact probability from images, whereas InteractVLM achieves 3D localization via VLM-guided multi-view rendering, utilizing stronger priors to get a 20.6% F1 improvement.
- **vs LEMON**: LEMON requires paired human-object geometric data for training and covers 21 categories; InteractVLM uses unpaired data to cover 80+ categories while achieving comparable performance.
- **vs PARIS3D**: Both use VLM+SAM, but PARIS3D performs 3D segmentation on objects, whereas InteractVLM handles human-object interaction scenes and introduces FeatLift to ensure multi-view consistency.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Novel approach of transferring VLM knowledge to 3D contact estimation, with an elegantly designed RLL framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Verified on multiple datasets and tasks with thorough ablation, but lacks larger-scale in-the-wild evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear, with rich diagrams and detailed method descriptions.
- **Value**: ⭐⭐⭐⭐ The RLL framework has general applicability, and the definition of the semantic contact task is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning](reconstructing_close_human_interaction_with_appearance_and_proxemics_reasoning.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[CVPR 2025\] 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)
- [\[CVPR 2025\] Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)

</div>

<!-- RELATED:END -->
