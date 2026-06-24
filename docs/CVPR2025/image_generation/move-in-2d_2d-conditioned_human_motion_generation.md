---
title: >-
  [Paper Note] Move-in-2D: 2D-Conditioned Human Motion Generation
description: >-
  [CVPR 2025][Image Generation][motion generation] Defines a new task of human motion generation conditioned on 2D scene images and text, constructs the 300K-scale HiC-Motion dataset, and generates motion sequences that naturally project onto the scene using an in-context conditioning diffusion Transformer, enabling downstream human video generation.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "motion generation"
  - "2D scene conditioning"
  - "diffusion transformer"
  - "in-context learning"
  - "HiC-Motion"
date: 2026-05-08
content_hash: c5b63da4c97f189d
---

# Move-in-2D: 2D-Conditioned Human Motion Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.13185](https://arxiv.org/abs/2412.13185)  
**Code**: [Project Page](https://hhsinping.github.io/Move-in-2D)  
**Area**: Image Generation  
**Keywords**: motion generation, 2D scene conditioning, diffusion transformer, in-context learning, HiC-Motion

## TL;DR

Defines a new task of human motion generation conditioned on 2D scene images and text, constructs the 300K-scale HiC-Motion dataset, and generates motion sequences that naturally project onto the scene using an in-context conditioning diffusion Transformer, enabling downstream human video generation.

## Background & Motivation

**Background**: Significant progress has been made in human video generation, where the most effective methods rely on predefined human motion sequences (such as OpenPose, DensePose) as control signals. Existing methods typically extract motion from other videos, which limits action categories and global scene adaptation.

**Limitations of Prior Work**:
- Text-only conditioned motion generation (MDM, MLD) cannot guarantee spatial compatibility with the target scene.
- 3D scene-conditioned methods (HUMANISE, SceneDiff) rely on 3D meshes/point clouds, which are expensive to acquire and mostly limited to simple indoor actions.
- There is a lack of large-scale datasets containing the triad of motion sequences, text descriptions, and scene images.

**Key Challenge**: Scene-aware motion generation is required, but 3D scene acquisition is highly expensive; 2D images are ubiquitous but have not previously been used as a conditioning modality for motion generation.

**Goal**: Propose a motion generation paradigm conditioned on a single 2D scene image, enabling the generated motion to be naturally compatible with the scene when projected onto the 2D plane.

**Key Insight**: Substitute 3D scenes with 2D images as the conditioning modality to drastically expand the scope of applicable scenes (indoors, outdoors, wild), and construct a corresponding dataset to train a diffusion model.

**Core Idea**: 2D scene images provide semantic and spatial layout clues, enabling the generation of scene-compatible human motion without 3D reconstruction.

## Method

### Overall Architecture

1. Input: Background scene image $s$ + text prompt $p$
2. Motion Representation: A 256-frame sequence, where each frame consists of 6D rotations $\theta_b$ of 23 SMPL joints, global orientation $\theta_g$, and camera translation $\pi \in \mathbb{R}^3$.
3. Model: Conditional generation based on a Diffusion Transformer (DiT), supporting classifier-free guidance (CFG).

### Key Designs

#### 1. HiC-Motion Dataset Construction

Filters 300k videos containing single-person motion from 30 million open-domain web videos:
- Filters single-person videos using Keypoint R-CNN and OpenPose detections.
- Retains videos with motion frame counts > 256.
- Extracts pseudo-ground-truth motion in SMPL format using 4D-Humans.
- Obtains background images by removing the human body using Mask R-CNN and a basic inpainting model.
- Covers 1000+ categories (daily activities, sports, etc.), vastly exceeding prior works.

#### 2. Multi-Conditional Transformer

Three condition injection mechanisms:
- **In-context conditioning**: Text tokens (encoded by CLIP-B) and scene tokens (encoded by DINO-B as 240 patch tokens) are concatenated with the motion sequence as extra tokens.
- **AdaLN**: Diffusion timesteps are injected via Adaptive Layer Normalization (AdaLN) to enhance temporal smoothness.
- **Cross-attention**: An alternative mechanism, but experiments demonstrate that in-context conditioning is superior.

Final Architecture: 8 Transformer blocks, 512 hidden dimension, 4 attention heads, and a 1000-step cosine noise schedule.

#### 3. Two-Stage Training Strategy

- **Stage 1**: Train on the full 300K videos for 600K iterations to learn scene semantics and diverse motion generation.
- **Stage 2**: Fine-tune on a hybrid dataset (60% large motion + 40% static background) for 600K iterations to decouple the influence of camera motion.

### Loss & Training

MSE reconstruction loss $\mathcal{L}_{mse} = \mathbb{E}_{x_0, t} \| x_0 - \mathcal{M}(x_t | t, c) \|^2$, paired with Classifier-Free Guidance (CFG, applying joint guidance on both text and scene conditions).

## Key Experimental Results

### Main Results: Quantitative Evaluation

| Method | FID↓ | Accuracy↑ | Diversity↑ | Multimodality↑ |
|------|------|-----------|------------|----------------|
| MDM | 164.6 | 0.325 | 24.8 | 18.9 |
| MLD | 85.9 | 0.322 | 25.1 | 19.5 |
| SceneDiff (3D) | 543.8 | 0.203 | 4.2 | 3.9 |
| HUMANISE (3D) | 159.9 | 0.225 | 23.3 | 20.0 |
| MDM+ (HiC trained) | 46.0 | 0.620 | 23.0 | 17.6 |
| **Ours** | **44.6** | **0.661** | **26.0** | **20.1** |

### VLM Automated Evaluation (GPT-4o Scoring, 0-5 Scales)

| Method | Scene-Align↑ | Text-Align↑ | Quality↑ | Total↑ |
|------|-------------|-------------|----------|--------|
| MDM | 2.25 | 1.35 | 1.50 | 5.10 |
| MLD | 2.85 | 1.95 | 1.90 | 6.70 |
| **Ours** | **3.55** | **2.70** | **2.85** | **9.10** |

### Ablation Study: Condition Injection Methods

| Timestep | Text | Scene | FID↓ | Accuracy↑ |
|----------|------|-------|------|-----------|
| AdaLN | In-Context | **In-Context** | **44.6** | **0.661** |
| AdaLN | In-Context | Cross-Attn | 47.7 | 0.567 |
| In-Context | In-Context | In-Context | 62.9 | 0.554 |

### Key Findings

1. MDM+ trained on the HiC-Motion dataset reduces the FID by 72% compared to the original MDM, validating the critical importance of large-scale data.
2. In-context conditioning outperforms cross-attention for both scene and text conditions, as the shared token space facilitates cross-modal interactions.
3. Ours vs. Ours-scene: Adding text conditioning improves Accuracy by 37% but decreases Multimodality (as text constraints limit diversity).

## Highlights & Insights

- **Novel Task Definition**: Motion generation conditioned on 2D scenes fills the gap between text-only and 3D scene-conditioned generations, offering high practical utility.
- **Dataset Scale Breakthrough**: 300K motion sequences, vastly exceeding HumanML3D (14.6K) and Motion-X (81K), covering diverse indoor and outdoor scenes.
- **Cross-Domain Application of In-Context Learning**: Translates the LLM's in-context paradigm to motion diffusion models, enabling text and image tokens to interact within a shared space.
- **Two-Step Video Generation Pipeline**: Generates motion control signals first, then renders the final video with Champ/Gen-3, yielding quality significantly superior to direct SVD generation.
- **2D Projection Design Choice**: The model additionally predicts camera translation parameters $\pi$, allowing motion to be naturally mapped onto the image plane via perspective projection.

## Limitations & Future Work

1. Lack of camera motion control—displacement in the generated motion can be conflated with camera translation effects.
2. The two-step video generation pipeline is not jointly optimized with the motion generation model.
3. Motion extracted from internet videos consists of pseudo-ground-truth (using 4D-Humans), which introduces systematic noise.
4. Scene compatibility relies on implicit learning, without explicit guarantees of physical plausibility (e.g., foot penetration into the ground).

## Related Work & Insights

- **MDM/MLD**: Dominant methods for text-conditioned motion generation, but they lack scene awareness.
- **HUMANISE**: Competes as a pioneer in 3D scene + text-conditioned generation, but is restricted to the 643 indoor scenes of ScanNet.
- **HiC Dataset Series**: The paradigm of constructing massive human-centric video datasets inspired the creation of HiC-Motion.
- **Insights**: The 2D-conditioned paradigm can be extended to other scene-interaction tasks (e.g., robotic manipulation planning, AR content generation), lowering the barriers imposed by 3D reconstruction.

## Rating

⭐⭐⭐⭐ — The definition of the new task is highly valuable, the dataset engineering is solid, and the experiments are thorough (including VLM evaluations and downstream video generation applications). However, the core methodology (diffusion Transformer + in-context learning) is relatively standard, indicating moderate technical novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Lifting Motion to the 3D World via 2D Diffusion](lifting_motion_to_the_3d_world_via_2d_diffusion.md)
- [\[ICCV 2025\] DreamDance: Animating Human Images by Enriching 3D Geometry Cues from 2D Poses](../../ICCV2025/image_generation/dreamdance_animating_human_images_by_enriching_3d_geometry_cues_from_2d_poses.md)
- [\[CVPR 2025\] MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[CVPR 2025\] Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction](nonisotropic_gaussian_diffusion_for_realistic_3d_human_motion_prediction.md)

</div>

<!-- RELATED:END -->
