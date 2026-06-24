---
title: >-
  [Paper Note] SuperGaussian: Repurposing Video Models for 3D Super Resolution
description: >-
  [ECCV 2024][3D Vision][3D Super Resolution] SuperGaussian is proposed to achieve 3D super-resolution by repurposing pre-trained video upscaling models. It requires no category-specific training, can handle various 3D input formats (Gaussians, NeRF, meshes, etc.), and outputs high-quality Gaussian Splatting models.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Super Resolution"
  - "Video Upscaling"
  - "Gaussian Splatting"
  - "Category-Agnostic"
  - "3D Generation"
date: 2026-05-08
content_hash: 12c265da0902bbe9
---

# SuperGaussian: Repurposing Video Models for 3D Super Resolution

**Conference**: ECCV 2024  
**arXiv**: [2406.00609](https://arxiv.org/abs/2406.00609)  
**Code**: [Project Page](https://supergaussian.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D Super Resolution, Video Upscaling, Gaussian Splatting, Category-Agnostic, 3D Generation

## TL;DR

SuperGaussian is proposed to achieve 3D super-resolution by repurposing pre-trained video upscaling models. It requires no category-specific training, can handle various 3D input formats (Gaussians, NeRF, meshes, etc.), and outputs high-quality Gaussian Splatting models.

## Background & Motivation

### Background

The detail quality of current 3D generative models lags far behind that of image and video generative models. The main reasons: (1) the resolution of 3D representations (voxel grids, tri-planes) is limited; (2) high-quality 3D training data is scarce (at most million-scale vs. billion-scale for images). The key observation is that any 3D representation can be rendered as a video, allowing the repurposing of mature video upscaling models to enhance 3D quality while leveraging video temporal consistency to guarantee 3D consistency.

### Proposed Approach

**Goal**: ### Overall Architecture

Two-step pipeline: (1) Render a low-resolution video from a low-resolution 3D input along a smooth trajectory, and perform $4\times$ upscaling using a pre-trained video upscaler; (2) Optimize 3D Gaussian Splatting on the upscaled video to obtain a high-fidelity 3D output.


## Method

### Overall Architecture

Two-step pipeline: (1) Render low-resolution video from a low-resolution 3D input along a smooth trajectory, and perform $4\times$ upscaling using a pre-trained video upscaler; (2) Optimize 3D Gaussian Splatting on the upscaled video to obtain a high-fidelity 3D output.

### Key Designs

**Video Upscaling Prior**: VideoGigaGAN is employed as the video upscaler. Compared to frame-by-frame image upscaling, the temporal consistency of the video model significantly reduces blur issues after 3D reconstruction. The video upscaler is fine-tuned on the MVImgNet dataset to handle degradations specific to low-resolution Gaussian rendering.

**Domain-Adaptive Fine-Tuning**: Low/high-resolution video pairs are generated from MVImgNet by downsampling images to $64 \times 64$, fitting a low-resolution Gaussian model, and rendering it as input; the original video is resized to $256 \times 256$ as the target. Joint fine-tuning is performed using Charbonnier regression loss + LPIPS perceptual loss + GAN loss.

**3D Optimization**: The standard Gaussian Splatting optimization pipeline is used, completing within 2K steps. Known camera parameters are directly provided without requiring SfM estimation. $L_1$ + SSIM losses are used.

### Loss & Training

Fine-tuning stage: Charbonnier loss (weight 10) + LPIPS loss (weight 15) + GAN loss (weight 0.05) + $R_1$ regularization

## Key Experimental Results

### MVImgNet Dataset Comparison

Low-resolution Gaussian Splatting upscaling ($64 \rightarrow 256\text{px}$):


### Main Results

| Method | LPIPS↓ | NIQE↓ | FID↓ | IS↑ |
|------|--------|-------|------|-----|
| Instruct-G2G | 0.1867 | 8.33% | 32.56% | 10.52% |
| Super-NeRF | 0.2204 | 8.84% | 37.54% | 10.40% |
| Pre-hoc image | 0.1524 | 7.65% | 27.04% | 11.27% |
| **SuperGaussian** | **0.1290** | **6.80%** | **24.32%** | **11.69%** |

### Blender Synthetic Dataset Comparison

$4\times$ upscaling ($200 \rightarrow 800\text{px}$), using TensoRF as the 3D representation:


### Ablation Study

| Method | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|--------|-------|------|
| FastSR-NeRF | 0.075 | 30.47% | 0.944 |
| NeRF-SR | 0.076 | 28.46% | 0.921 |
| **SuperGaussian** | **0.067** | 28.44% | 0.923 |

### Key Findings

- Video prior vs. image prior: 3D reconstruction is sharper after video upscaling, whereas image upscaling leads to blur due to inter-frame inconsistency.
- Fine-tuning shows significant efficacy on severely degraded inputs (e.g., 4K Gaussians or 1K-step NeRF), and can even recover readable Chinese characters.
- The closer the upscaling trajectory is to the target object, the better the performance.
- The entire pipeline takes about 141 seconds to complete, making it the most efficient among all baselines.

## Highlights & Insights

1. **Simple and general** modular design: $3\text{D} \rightarrow \text{video} \rightarrow \text{upscaling} \rightarrow 3\text{D}$, where each component can be independently replaced and upgraded.
2. Leverages the temporal consistency of video models to compensate for the lack of 3D consistency, which is simpler and more effective than image models combined with various consistency enhancement strategies.
3. Category-agnostic and input-format-agnostic, allowing integration directly into existing 3D workflows.

## Limitations & Future Work

- Relies on the generalization capability of pre-trained video models.
- Cannot recover missing/occluded regions in the input.
- The inference speed of the video upscaler is limited.

## Related Work & Insights

Applying 2D generative priors to 3D is a highly popular direction (e.g., DreamFusion using image diffusion). This work is the first to systematically demonstrate that video priors outperform image priors for 3D super-resolution. The framework can be continuously upgraded alongside advancements in video models (such as Sora).

## Rating

- Novelty: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](../../AAAI2026/3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[CVPR 2025\] Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](../../CVPR2025/3d_vision/kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)

</div>

<!-- RELATED:END -->
