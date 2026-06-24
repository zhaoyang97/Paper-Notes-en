---
title: >-
  [Paper Note] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models
description: >-
  [ECCV 2024][Image Generation] WildVidFit proposes a video-free training virtual try-on framework. By utilizing an image-based conditional diffusion model and a diffusion guidance module (VideoMAE + DINO-V2), it achieves temporally consistent garment try-on effects in complex in-the-wild videos.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: ed58cd7d0208326d
---

# WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2407.10625](https://arxiv.org/abs/2407.10625)  
**Area**: Image Generation

## TL;DR

WildVidFit proposes a video-free training virtual try-on framework. By utilizing an image-based conditional diffusion model and a diffusion guidance module (VideoMAE + DINO-V2), it achieves temporally consistent garment try-on effects in complex in-the-wild videos.

## Background & Motivation

Video virtual try-on aims to generate realistic sequences where the target garment naturally fits the pose and body shape of the source person in the video. Existing methods face two core challenges:

**Limitations of Explicit Warping**: Traditional methods rely on optical flow estimation for garment warping and blending, which is highly prone to pixel misalignment when encountering complex human motions and body occlusions, significantly degrading the performance.

**Video Data and Computational Overhead**: Video-level models require large amounts of high-quality annotated videos and massive computational resources to train additional temporal modules, and their generalization ability is also limited to specific datasets.

**Complexity of In-the-Wild Scenes**: Dance videos on platforms like TikTok contain intense body movements, frequent occlusions, and complex backgrounds, which are challenging for existing methods.

The core idea of WildVidFit is to decompose video try-on into two sub-tasks: (1) constructing a fine-grained image try-on model capable of handling complex movements and occlusions, and (2) extending it to the video domain via diffusion guidance to maintain inter-frame consistency, without requiring any video-level training.

## Method

### Overall Architecture

WildVidFit consists of two core modules:

1. **Single-Stage Image Try-On Network**: A conditional image generation network built on Stable Diffusion, which takes person representation (cloth-agnostic RGB + pose map) and garment representation (garment image + edge map) as conditional inputs.
2. **Diffusion Guidance Module**: It utilizes pre-trained VideoMAE and DINO-V2 models to introduce temporal consistency constraints during the diffusion sampling process, without any fine-tuning.

### Key Designs

**Input Preprocessing**:
- Person representation: A cloth-agnostic RGB image A and a pose map P are generated via segmentation and pose estimation, preserving person identity while removing the original clothing.
- Garment representation: A Sobel operator is applied to the garment image G to extract the edge map $E_g$, and the feature vector $F_g \in \mathbb{R}^{257 \times 2048}$ is extracted via DINO-V2.

**Single-Stage Implicit Warping**:
- The main UNet inherits the weights of Stable Diffusion, with a newly added condition branch to extract multi-scale personal features.
- The garment feature $F_g$ is used instead of text for cross-attention to achieve implicit warping (inspired by TryOnDiffusion), avoiding the limitations of explicit optical flow estimation.
- Conditional features are only injected through convolutions into the UNet decoder, preserving pre-trained priors.

**Temporal Consistency via Diffusion Guidance**:
- **VideoMAE Guidance** ($\mathcal{L}_{MAE}$): The generated frame sequence is randomly masked and input into a pre-trained VideoMAE for reconstruction. It is hypothesized that smoother videos are easier to reconstruct, yielding lower reconstruction loss.
- **DINO-V2 Guidance** ($\mathcal{L}_{SIM}$): The consistency of adjacent frames in the DINO-V2 feature space is constrained through spherical distance.
- Both losses collectively constitute the temporal loss to guide the sampling process via gradients.

**Long Video Generation Strategy**:
- The long video is divided into overlapping short segments, with adjacent segments shifted by stride $s$ (typically $L/2$ or $L/4$).
- The overlapping frames are averaged in each denoising step to achieve smooth transitions between segments.

### Loss & Training

The standard diffusion objective is used in the training phase:

$$\mathcal{L} = \mathbb{E}_{x,c,\epsilon,t}[w_t \|\hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x\|_2^2]$$

Temporal guidance during inference:

$$\hat{\epsilon}_t = \epsilon_\theta(z_t; t, c) - w_1 \nabla_{z_t} \mathcal{L}_{MAE}(z_t) - w_2 \nabla_{z_t} \mathcal{L}_{SIM}(z_t)$$

where $w_1 = 2000$, $w_2 = 1000$, and the mask ratio is 0.7.

## Key Experimental Results

### Main Results

**Table 1: Image try-on comparison on the VITON-HD dataset**

| Method | SSIM↑ | LPIPS↓ | FID↓ | KID↓ | User↑ |
|------|-------|--------|------|------|-------|
| CP-VTON | 0.785 | 0.2871 | 48.86 | 4.42 | 3.86% |
| HR-VTON | 0.878 | 0.0987 | 11.80 | 0.37 | 6.62% |
| LaDI-VTON | 0.871 | 0.0941 | 13.01 | 0.66 | 16.02% |
| DCI-VTON | 0.882 | 0.0786 | 11.91 | 0.51 | 12.18% |
| **WildVidFit** | **0.883** | **0.0773** | **8.67** | **0.10** | **61.32%** |

**Table 2: Video try-on comparison on the VVT and TikTok datasets**

| Method | Dataset | VFID↓ | User↑ |
|------|--------|-------|-------|
| HR-VTON | VVT | 4.852 | 9.46% |
| LaDI-VTON | VVT | 4.442 | 4.24% |
| ClothFormer | VVT | 4.192 | 46.44% |
| **WildVidFit** | **VVT** | **4.202** | **39.86%** |
| HR-VTON | TikTok | 25.43 | 0.00% |
| LaDI-VTON | TikTok | 14.24 | 26.90% |
| **Ours** | **TikTok** | **9.87** | **73.10%** |

### Ablation Study

**Table 3: Ablation of edge maps and CFG (VITON-HD)**

| Edge maps | Guidance scale | FID↓ | KID↓ |
|-----------|---------------|------|------|
| ✗ | 2 | 8.93 | 0.12 |
| ✓ | 1 | 9.47 | 0.17 |
| ✓ | 2 | **8.67** | **0.10** |
| ✓ | 3 | 8.68 | 0.10 |

**Table 4: Ablation of temporal modules (TikTok)**

| Method | VFID↓ |
|------|-------|
| Image-based| 13.45 |
| + Fully cross-frame attention | 12.14 |
| + Guidance with $\mathcal{L}_{MAE}$ | 10.64 |
| + Guidance with $\mathcal{L}_{MAE}$ and $\mathcal{L}_{SIM}$ | **9.87** |

### Key Findings

1. WildVidFit achieves an FID of only 8.67 on VITON-HD, significantly outperforming DCI-VTON (11.91), with a user preference of 61.32%.
2. On TikTok in-the-wild videos, WildVidFit outperforms LaDI-VTON (14.24) with a VFID of 9.87, achieving a user preference of 73.10%.
3. On VVT, the image-only method combined with diffusion guidance matches the performance of the specialized video method ClothFormer.
4. The contribution of each temporal module is significant: cross-frame attention reduces VFID by 1.31, MAE guidance reduces it further by 1.50, and SIM guidance reduces it by another 0.77.

## Highlights & Insights

- **Video-free Training Video Generation Paradigm**: The temporal priors of pre-trained video/image models are elegantly injected into the sampling process via the diffusion guidance mechanism, completely avoiding the prohibitive cost of video-level training.
- **Implicit Warping Superior to Explicit Warping**: Cross-attention is used instead of optical flow estimation for garment fitting. This does not rely on strict pixel-level alignment, inherently offering better capabilities for handling occlusions.
- **Joint Training on Multiple Datasets to Enhance Generalization**: Joint training on the VITON-HD, DressCode, and TikTok datasets enables cross-dataset garment transfer.

## Limitations & Future Work

1. Diffusion guidance requires decoding latent representations to compute losses at each sampling step, resulting in a slower inference speed.
2. Due to memory constraints, temporal losses are only computed in the garment region, which may have some impact on overall consistency.
3. The fixed segment length of VideoMAE limits the flexibility of processing ultra-long videos.
4. For extreme occlusions or very large-amplitude movements, performance may still degrade.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

</div>

<!-- RELATED:END -->
