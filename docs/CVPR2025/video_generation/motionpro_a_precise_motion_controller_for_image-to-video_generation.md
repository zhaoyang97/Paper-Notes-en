---
title: >-
  [Paper Note] MotionPro: A Precise Motion Controller for Image-to-Video Generation
description: >-
  [CVPR 2025][Video Generation][Image-to-Video Generation] MotionPro is proposed to achieve fine-grained, controllable image-to-video generation that distinguishes between object and camera motions, utilizing dual signals of region-wise trajectories and motion masks.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Image-to-Video Generation"
  - "Motion Control"
  - "Region-wise Trajectory"
  - "Motion Mask"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 5080f00a38622fb9
---

# MotionPro: A Precise Motion Controller for Image-to-Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2505.20287](https://arxiv.org/abs/2505.20287)  
**Code**: [https://github.com/zhw-zhang/MotionPro-page](https://github.com/zhw-zhang/MotionPro-page)  
**Area**: Video Generation  
**Keywords**: Image-to-Video Generation, Motion Control, Region-wise Trajectory, Motion Mask, Diffusion Models

## TL;DR

MotionPro is proposed to achieve fine-grained, controllable image-to-video generation that distinguishes between object and camera motions, utilizing dual signals of region-wise trajectories and motion masks.

## Background & Motivation

Existing controllable I2V methods (such as DragNUWA and DragAnything) mainly rely on large-kernel Gaussian filtering to diffuse sparse trajectories as condition signals, which suffer from two key limitations:
1. **Inaccurate fine-grained motion**: Gaussian filtering (with kernel sizes up to $99 \times 99$) diffuses trajectory signals into surrounding areas, leading to the loss of fine-grained motion details and generating unnatural movements (e.g., head turning).
2. **Motion category ambiguity**: Relying solely on trajectory conditions fails to distinguish "object motion" from "camera motion". For instance, a downward trajectory on a planet could represent either a downward camera pan or an upward movement of the planet, making a single trajectory signal highly ambiguous.

Furthermore, while MOFA-Video introduces a motion region mask, it is restricted to post-processing optical flow masking rather than being injected into the network as a generative condition, resulting in local distortions in the synthesized videos.

## Method

### Overall Architecture

MotionPro is built upon the pre-trained SVD (Stable Video Diffusion). During training, **region-wise trajectories** and **motion masks** are extracted from input videos as dual control signals. High-fidelity motion encoders encode these signals into multi-scale features, which are then injected into the multi-scale feature layers of the 3D-UNet via Adaptive Feature Modulation. Concurrently, LoRA fine-tuning is applied to all attention modules to enhance motion-trajectory alignment.

### Key Designs

1. **Region-wise Trajectory**: The DOT optical flow tracking model is utilized to estimate the optical flow maps $f^i$ and visibility masks $M^i$ of the video. The global visibility mask is computed as $M_g = \prod_{i=1}^{L} M^i$. The masked optical flow maps are then divided into $k \times k$ (default $k=8$) local regions, and trajectories within these regions are sparsely selected via a random region selection mask $M_{sel}$ (with the mask ratio randomly sampled from $[r_{min}, 1.0]$). **Core Advantage**: Compared to Gaussian filtering, this directly preserves the original trajectory information within local regions, maintaining precise motion details.

2. **Motion Mask**: The temporal average of optical flow magnitudes across all frames is computed as $f_{avg} = \frac{1}{L} \sum_{i=1}^{L} \|f^i\|_2$. Positions where $f_{avg} > 1$ are marked as motion regions to obtain $M_{mot}$, which is repeated $L$ times to form the motion mask sequence $\mathbf{M}_{mot} \in \{0,1\}^{L \times H \times W \times 1}$. **Function**: Globally identifies motion regions, clarifies the target motion category (object/camera motion), and eliminates ambiguity in trajectory signals.

3. **Adaptive Feature Modulation**: The trajectory and mask are concatenated and encoded by a lightweight Motion Encoder into multi-scale features $l_s$. At each scale, spatial-temporal convolutional layers predict scale $\gamma_s$ and bias $\beta_s$ to modulate the video latent features: $h_s' = GN(h_s) \cdot \gamma_s + \beta_s + h_s$. To guarantee training stability, zero initialization is adopted so that $\gamma_s = 0, \beta_s = 0$ at start.

### Loss & Training

- Denoising score matching (DSM) loss using the EDM training protocol: $\mathcal{L} = \mathbb{E}[\lambda_\sigma \|\hat{\mathbf{z}}_0 - \mathbf{z}_0\|_2^2]$
- LoRA rank is set to 32, with only the Motion Encoder and LoRA layers being trained
- Learning rate of $1 \times 10^{-5}$ using the AdamW optimizer
- Trained on 6 A800 GPUs with a batch size of 48
- Video resolution of $320 \times 512$, 16 frames at 8 fps

## Key Experimental Results

### Main Results — Fine-Grained Motion Control

| Dataset | Metric | MotionPro | MOFA-Video | DragNUWA | Gain |
|--------|------|-----------|------------|----------|------|
| WebVid-10M | FVD ↓ | **59.88** | 87.70 | 96.65 | -27.82 |
| WebVid-10M | FID ↓ | **10.40** | 12.18 | 13.19 | -1.78 |
| MC-Bench | MD-Img ↓ | **10.56** | 13.94 | - | -3.38 |
| MC-Bench | MD-Vid ↓ | **8.34** | 10.50 | - | -2.16 |

### Main Results — Object-Level Motion Control (MC-Bench)

| Method | MD-Img ↓ | MD-Vid ↓ | Frame Consis. ↑ |
|------|----------|----------|-----------------|
| MOFA-Video | 15.56 | 12.04 | 0.9951 |
| DragAnything | 12.30 | 11.37 | 0.9917 |
| **MotionPro** | **10.48** | **8.59** | 0.9943 |

### Ablation Study

| Configuration | MD-Vid (fine) ↓ | MD-Vid (obj) ↓ | Description |
|------|----------------|----------------|------|
| $k=1$ | Higher | Higher | Region size is too small, insufficient trajectory signals |
| $k=4$ | Medium | Medium | Performance gradually improves |
| $k=8$ | **Best** | **Best** | Optimal balance between precision and coverage |
| $k=16$ | Slightly worse | Slightly worse | Excessive region size harms fine-grained control |
| MotionPro$_C$ (Concatenation Injection) | Higher | Higher | Feature concatenation requires strict spatial-temporal alignment |
| MotionPro$_+$ (Additive Injection) | Medium | Medium | Additive fusion yields sub-optimal performance |
| **MotionPro** (Modulation Injection) | **Best** | **Best** | Indirect modulation does not require strict alignment |

### Key Findings

- The average optical flow magnitude of videos generated by MotionPro is 8.95, which is substantially higher than the 4.95 of MOFA-Video, indicating richer and more dynamic motion variations.
- The motion mask effectively resolves motion category ambiguity typically observed in DragNUWA (e.g., misinterpreting object motion as camera motion).
- The MC-Bench benchmark includes 1.1K user-annotated image-trajectory pairs, aligning closely with real-world scenarios.

## Highlights & Insights

- **Region-wise Trajectory replacing Gaussian Filtering**: A simple yet effective design change that directly employs original optical flow within local regions instead of Gaussian dilation.
- **Conditional Utilization of Motion Mask**: Injecting the mask into the generation process as a conditioning signal rather than using it merely for post-processing fundamentally eliminates motion category ambiguity.
- **MC-Bench Benchmark**: Fills the gap in controllable I2V evaluation caused by the lack of user-annotated benchmarks.
- Feature modulation (FiLM-style) is more suitable for fusing motion control signals compared to concatenation or addition.

## Limitations & Future Work

- Built upon the SVD architecture, which limits the resolution to $320 \times 512$, with no adaptation yet to stronger DiT-based foundation models.
- Training requires pre-computing optical flows using DOT, which increases data preprocessing overhead.
- Only supports a single reference image paired with trajectories/masks as input, and does not support text-driven motion control.
- Scenarios involving independent multiple-object motion control remain unexplored.

## Related Work & Insights

- DragNUWA/DragAnything: Representative works of Gaussian trajectory-based methods and the primary baselines for comparison.
- MOFA-Video: A two-stage framework (sparse to dense trajectory) whose performance is limited as the mask is only utilized in post-processing.
- Motion-I2V: Shares a similar sparse-to-dense trajectory estimation pipeline.
- The region-wise design scheme presented in this work can be generalized to other tasks such as video editing and 3D scene motion control.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-signal design utilizing region-wise trajectories and motion masks is simple yet highly effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Proposes the new MC-Bench benchmark and provides detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear problem motivation.
- **Value**: ⭐⭐⭐⭐ Provides practical and valuable insights for controllable video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] MotiF: Making Text Count in Image Animation with Motion Focal Loss](motif_making_text_count_in_image_animation_with_motion_focal_loss.md)
- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](towards_precise_scaling_laws_for_video_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
