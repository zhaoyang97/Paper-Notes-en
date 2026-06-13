---
title: >-
  [Paper Note] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation
description: >-
  [ICCV 2025][3D Vision][egocentric motion] This paper proposes UniEgoMotion, the first unified egocentric motion model that achieves 3D human motion reconstruction, forecasting…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "egocentric motion"
  - "diffusion model"
  - "motion reconstruction"
  - "motion forecasting"
  - "motion generation"
  - "head-centric representation"
date: 2026-05-08
content_hash: a32e84a719d5d62d
---

# UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation

**Conference**: ICCV 2025
**arXiv**: [2508.01126](https://arxiv.org/abs/2508.01126)  
**Code**: [UniEgoMotion](https://chaitanya100100.github.io/UniEgoMotion/)  
**Area**: 3D Vision
**Keywords**: egocentric motion, diffusion model, motion reconstruction, motion forecasting, motion generation, head-centric representation

## TL;DR

This paper proposes UniEgoMotion, the first unified egocentric motion model that achieves 3D human motion reconstruction, forecasting, and generation from an egocentric perspective within a single model, via a conditional motion diffusion framework and a head-centric motion representation. The large-scale EE4D-Motion dataset is also released.

## Background & Motivation

**Egocentric human motion understanding** (from the first-person perspective of head-mounted devices) is critical for AR/VR, assistive technologies, and healthcare, yet faces three core challenges:

**Inherent difficulty of the egocentric viewpoint**: Head-mounted forward-facing cameras capture only a minimal portion of the user's body, requiring models to infer full-body motion from dynamic first-person views with frequent occlusions and motion blur. Motion forecasting and generation in egocentric settings remain largely unexplored.

**Neglect of scene context**: Most existing methods rely on explicit 3D scene representations (point clouds, voxels, SDF, etc.), which are unavailable in real-world egocentric applications. Prior work typically discards semantic information from egocentric images and relies solely on device trajectories, leading to degraded performance in activities with limited head movement (e.g., cooking, playing instruments).

**Task-isolated model design**: Motion reconstruction, forecasting, and generation are modeled as separate tasks with independent pipelines, lacking a unified framework to share motion priors and scene understanding capabilities.

**Core insight**: Egocentric images themselves contain rich scene semantic information sufficient to infer plausible 3D motion without explicit 3D scene reconstruction. A conditional diffusion model with flexible conditioning enables all three egocentric motion tasks within a unified framework.

## Method

### Overall Architecture

UniEgoMotion is a Transformer-based conditional motion diffusion model. Its inputs consist of egocentric video frames $\boldsymbol{I}_{1:N}$, 6-DoF device trajectories $\boldsymbol{T}_{1:N}$, and noisy motion $\boldsymbol{X}^t_{1:N}$. The core idea is to support three tasks through different condition combinations:

- **Reconstruction**: $\boldsymbol{C} = \{\boldsymbol{T}_{1:N}, \boldsymbol{I}_{1:N}\}$ (full conditioning)
- **Generation**: $\boldsymbol{C} = \{I_1\}$ (single image; remaining inputs replaced by learnable masks)
- **Forecasting**: $\boldsymbol{C} = \{\boldsymbol{T}_{1:n}, \boldsymbol{I}_{1:n}\}$ (partial observation; future motion extrapolated via diffusion inpainting)

During training, condition inputs are randomly masked to enable unified training across all three tasks; at inference, learnable mask tokens are filled in to support flexible inference.

### Diffusion Modeling

A standard conditional diffusion framework is adopted, with the objective of directly predicting clean motion (x-prediction):

$$\mathcal{L} = \mathbb{E}_{t, \boldsymbol{X}^t \sim q_t(\cdot|\boldsymbol{X})} \left[ \|\boldsymbol{X} - \mathcal{M}(\boldsymbol{X}^t, t, \boldsymbol{C})\|_2^2 \right]$$

For the forecasting task, a diffusion repainting strategy is employed: the observed motion $\boldsymbol{X}_{1:n}$ is first reconstructed, and known frames are overwritten at each denoising step to ensure consistency.

### Key Designs

- **Motion input**: Per-frame SMPL-X parameters are projected into latent vectors $f_X(X_i)$ via a linear layer.
- **Trajectory conditioning**: Linear projection $f_T(T_i)$ is added to the motion embeddings.
- **Image conditioning**: A ViT encoder $f_I(I_i)$ initialized with pretrained DINOv2 injects scene context via cross-attention.
- **Transformer decoder**: Multi-layer decoder blocks process motion tokens and cross-attend to image features.

A key design choice: DINOv2's fine-grained features substantially outperform CLIP and EgoVideo encoders, indicating that precise scene context is more important than egocentric action semantics.

### Head-Centric Motion Representation

The conventional pelvis-centric SMPL-X parameterization $(R^r_i, t^r_i, \theta_i, \beta_i)$ suffers from two problems: (1) misalignment with the egocentric device position; and (2) local joint angles requiring the model to learn complex forward kinematics.

UniEgoMotion proposes a head-centric representation:
1. Global SE(3) transforms for all joints are computed via forward kinematics.
2. A canonical reference frame $_cM_i$ is obtained by removing pitch, roll, and height (i.e., projecting the head trajectory onto the ground plane).
3. Motion is represented as $(_cM_i, \; _cM_i \odot M^h_i, \; _cM_i \odot \boldsymbol{M}^j_i)$.
4. Trajectories are expressed as inter-frame residuals, achieving translation invariance.

This representation eliminates kinematic-chain dependencies among joints and significantly reduces foot-ground penetration and floating artifacts.

### EE4D-Motion Dataset

Derived from the large-scale EgoExo4D dataset via a multi-view optimization and sequence-level smoothed SMPL-X fitting pipeline, EE4D-Motion provides over 110 hours of synchronized egocentric video–3D motion pairs covering real-world activities. The training set contains 143K samples and the evaluation set contains 4,400 samples.

## Key Experimental Results

### Main Results: Egocentric Motion Reconstruction

| Method | MPJPE↓ | MPJPE-PA↓ | MPJPE-H↓ | Foot Slide↓ | Semantic Sim.↑ | FID↓ |
|--------|--------|-----------|----------|-------------|----------------|------|
| AvatarPoser | 0.116 | 0.068 | 0.240 | 7.85 | 0.872 | 0.082 |
| EgoEgo | 0.130 | 0.075 | 0.272 | 3.90 | 0.858 | 0.068 |
| EgoAllo | 0.163 | 0.071 | 0.273 | 4.10 | 0.885 | 0.043 |
| **UniEgoMotion** | **0.100** | **0.053** | **0.180** | **3.62** | **0.918** | **0.027** |

UniEgoMotion achieves state-of-the-art performance across all metrics, reducing MPJPE by 13.8% relative to AvatarPoser and improving semantic similarity by 3.3%.

### Ablation Study: Motion Forecasting and Generation

| Method | Forecasting MPJPE (2–4s)↓ | Foot Slide↓ | FID↓ | Generation MPJPE (0–2s)↓ | Foot Slide↓ | FID↓ |
|--------|--------------------------|-------------|------|--------------------------|-------------|------|
| LSTM | 0.238 | 7.23 | 0.058 | 0.216 | 6.83 | 0.090 |
| Two-stage | 0.253 | 3.55 | 0.038 | 0.222 | 4.35 | 0.037 |
| **UniEgoMotion** | **0.206** | **2.60** | 0.047 | 0.226 | **2.89** | 0.043 |
| Pelvis-centric | 0.245 | 3.56 | 0.042 | 0.232 | 3.94 | 0.039 |
| Global Repre. | 0.293 | 3.16 | 0.046 | 0.228 | 3.65 | 0.035 |
| CLIP encoder | 0.214 | 3.75 | 0.043 | 0.238 | 3.57 | 0.037 |

**Key Findings**:
- Deterministic LSTM prediction of mean motion results in severe foot sliding (7.23); UniEgoMotion achieves a Foot Slide of only 2.60.
- Head-centric vs. pelvis-centric representation: Foot Slide decreases from 3.56 to 2.60, eliminating floating and penetration artifacts.
- DINOv2's fine-grained features outperform both CLIP and EgoVideo, demonstrating that scene context understanding is more critical than egocentric action features.
- The Transformer decoder architecture outperforms encoder variants and 1D-UNet, validating the advantage of cross-attention for flexible conditioning.

## Highlights & Insights

1. **Elegant unified framework design**: Through conditional masking and diffusion repainting, a single model seamlessly supports reconstruction, forecasting, and generation while sharing motion priors.
2. **First formulation of egocentric motion forecasting and generation tasks**: Scene-aware motion modeling is extended to a more practical egocentric setting without requiring explicit 3D scene input.
3. **Innovation in head-centric representation**: Natural alignment with egocentric devices, elimination of kinematic-chain dependencies, and substantial reduction of physically implausible artifacts.
4. **Importance of image encoder choice**: DINOv2's fine-grained features outperform text-aligned CLIP and egocentric-specialized EgoVideo, revealing that fine-grained scene context understanding matters more than high-level semantics.

## Limitations & Future Work

- The EE4D-Motion dataset is based on pseudo ground truth (fitted via multi-view optimization), so motion accuracy is bounded by SMPL-X fitting quality.
- Long-horizon evaluation (>2s) for forecasting and generation is difficult, as multiple plausible motions may all deviate from the ground truth.
- Text-conditioned motion generation is not explored, limiting interactivity.
- Motion diversity and scene reasoning in the single-image generation setting still have room for improvement.

## Related Work & Insights

- **Scene-aware motion generation**: Methods generating motion from 3D scenes (point clouds/voxels/SDF); only one prior work generates motion from wide-angle scene RGB images.
- **Egocentric motion reconstruction**: EgoEgo, EgoAllo, AvatarPoser, etc.; most discard image semantics and rely solely on device trajectories.
- **Motion forecasting**: Traditional methods (MLP/RNN/GCN/Transformer) and diffusion-based approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to unify three egocentric motion tasks; head-centric representation is elegantly designed.
- **Technical Quality**: ⭐⭐⭐⭐ — Method is clean and effective; ablations are thorough; pseudo-GT data is a limitation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across all three tasks with multiple metrics and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear; analogies are vivid and accessible.
- **Overall Score**: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span](../../NeurIPS2025/3d_vision/gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)
- [\[ICCV 2025\] AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)

</div>

<!-- RELATED:END -->
