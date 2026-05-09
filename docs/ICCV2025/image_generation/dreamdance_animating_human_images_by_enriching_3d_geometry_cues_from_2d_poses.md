---
title: >-
  [Paper Note] DreamDance: Animating Human Images by Enriching 3D Geometry Cues from 2D Poses
description: >-
  [ICCV 2025][Image Generation][Human image animation] DreamDance proposes a human image animation framework that takes only 2D skeleton pose sequences as input. It first generates mutually aligned depth maps and normal maps from 2D poses via a Mutually Aligned Geometry Diffusion Model to enrich 3D geometric guidance, then integrates multi-level guidance signals through an SVD-based Cross-Domain Controlled Video Diffusion Model to synthesize high-quality human animations. The method achieves state-of-the-art performance on the TikTok dataset (FVD 153.07 vs. Champ 170.20).
tags:
  - ICCV 2025
  - Image Generation
  - Human image animation
  - depth-normal map generation
  - geometry attention
  - SVD ControlNet
  - cross-domain controller
date: 2026-05-08
content_hash: 83679905ca70cef4
---

# DreamDance: Animating Human Images by Enriching 3D Geometry Cues from 2D Poses

**Conference**: ICCV 2025
**arXiv**: [2412.00397](https://arxiv.org/abs/2412.00397)
**Code**: [Project Page](https://pang-yatian.github.io/Dreamdance-webpage/)
**Area**: Human Image Animation / Video Generation
**Keywords**: Human image animation, depth-normal map generation, geometry attention, SVD ControlNet, cross-domain controller

## TL;DR
DreamDance proposes a human image animation framework that takes only 2D skeleton pose sequences as input. It first generates mutually aligned depth maps and normal maps from 2D poses via a Mutually Aligned Geometry Diffusion Model to enrich 3D geometric guidance, then integrates multi-level guidance signals through an SVD-based Cross-Domain Controlled Video Diffusion Model to synthesize high-quality human animations. The method achieves state-of-the-art performance on the TikTok dataset (FVD 153.07 vs. Champ 170.20).

## Background & Motivation

### Core Problem
Human image animation aims to generate dynamic and realistic videos from static human images and motion control signals, with broad applications in film production, social media, and online retail.

### Limitations of Prior Work

**Reliance on 2D pose guidance only**: Methods such as AnimateAnyone and MagicAnimate use skeleton poses as control signals, but 2D poses lack 3D information, leading to intra-frame inconsistencies (e.g., clothing deformation) and inter-frame incoherence (e.g., flickering artifacts).

**Drawbacks of SMPL-based 3D approaches**: Champ introduces SMPL parametric body models to render normal and depth maps as additional guidance, but suffers from:
   - Cumbersome SMPL motion generation, typically requiring prediction from existing videos with limited editability
   - Independent SMPL and pose models that may produce misaligned control signals
   - SMPL rendering that focuses only on body geometry while ignoring visual details such as clothing and hair

**Insufficient temporal modeling**: Early methods (DisCo, DreamPose) generate frames independently and lack temporal consistency; even with temporal attention (AnimateAnyone), the use of image diffusion models provides weak temporal priors.

### Key Insight

Human images naturally exhibit multi-level correlations: **coarse skeleton poses → fine-grained geometry cues (depth/normal maps) → explicit appearance details**. Capturing this coarse-to-fine hierarchy to enrich guidance signals enables high-quality animation without relying on SMPL.

## Method

### Overall Architecture

DreamDance consists of two stages:

**Stage 1**: Mutually Aligned Geometry Diffusion Model — jointly generates mutually aligned depth maps, normal maps (and low-resolution RGB) from a reference image and target poses to enrich geometric guidance.

**Stage 2**: Cross-Domain Controlled Video Diffusion Model — integrates multi-level guidance signals (pose, depth, normal) on top of SVD to generate high-resolution human animations.

Formally:
$$x_{1:T}, n_{1:T}, d_{1:T} = G_1^{low\_res}(X, p_{1:T})$$
$$Y_{1:T} = G_2^{high\_res}(X, p_{1:T}, n_{1:T}, d_{1:T})$$

### Key Design 1: Mutually Aligned Geometry Diffusion Model

**Unified diffusion process**: RGB $\mathbf{x}$, normal map $\mathbf{n}$, and depth map $\mathbf{d}$ are jointly generated. The noisy latents of the three modalities are concatenated along the batch dimension into a unified latent $\mathbf{z_t} = \text{concat}(\mathbf{x_t}, \mathbf{n_t}, \mathbf{d_t})$, and fed into a shared UNet for noise prediction:

$$\ell = \mathbb{E}_{t,\mathbf{z,i,p},\epsilon}\left[\|\epsilon - \epsilon_\theta(\mathbf{z_t};\mathbf{i},\mathbf{p})\|_2^2\right]$$

**Domain Embedding**: Since the original diffusion model carries a strong RGB-domain prior while depth/normal maps follow different distributions, a one-hot domain vector is encoded via positional encoding and added to the UNet's time embedding to accelerate training convergence.

**Reference image control**: A Reference UNet extracts fine-grained features $\mathbf{x_{ref}}$ from the reference image, injected by concatenating them to the K/V of the spatial attention:

$$\mathbf{k} = W_k \cdot \text{concat}(\mathbf{x}, \mathbf{x_{ref}}), \quad \mathbf{v} = W_v \cdot \text{concat}(\mathbf{x}, \mathbf{x_{ref}})$$

**Geometry Attention**: Ensures mutual consistency among the generated RGB, normal, and depth maps. Each modality supplies Q from itself, while K and V are derived from the concatenation of all three modalities:

$$\mathbf{k_i} = W_k \cdot \text{cat}(\mathbf{x_i}, \mathbf{x_n}, \mathbf{x_d}), \quad \mathbf{v_i} = W_v \cdot \text{cat}(\mathbf{x_i}, \mathbf{x_n}, \mathbf{x_d})$$

**Three-stage training strategy**:
1. Disable geometry attention and temporal attention; train each modality independently.
2. Activate geometry attention while freezing all other modules to focus on cross-modal alignment.
3. Activate temporal attention while freezing all other modules to ensure temporal consistency.

**Multi-domain CFG**: Different modalities require different CFG guidance scales (especially normal maps). Separate guidance coefficients $s_x, s_n, s_d$ are applied to RGB, normal, and depth, respectively.

### Key Design 2: Cross-Domain Controlled Video Diffusion Model

**Cross-domain Controller**: Integrates pose, depth, and normal guidance signals through three steps:
1. Each modality is embedded into a feature space via a domain-specific lightweight convolutional layer.
2. Features interact and fuse via a geometry attention mechanism analogous to Stage 1.
3. The fused guidance features are added to the noisy latent of the SVD ControlNet.

$$\mathbf{f_i} = \text{GeoAttn}(F_p(\mathbf{p_i}), F_d(\mathbf{d_i}), F_n(\mathbf{n_i}))$$

**SVD ControlNet**: All pretrained SVD parameters are frozen; a trainable copy is maintained and connected via zero-initialized convolutional layers to ensure training stability.

**Robust Conditioning**: Since depth/normal maps generated in Stage 1 may contain artifacts, a dropout strategy is adopted — **control signals are randomly replaced with zero-valued images** — encouraging the model to exploit information from other modalities and temporal frames, effectively mitigating error accumulation.

## Experiments

### Main Results

Quantitative comparison on the TikTok dataset:

| Method | Original Guidance | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID-VID ↓ | FVD ↓ |
|--------|------------------|------|--------|--------|---------|-----------|-------|
| MRAA | 2D | 3.21E-4 | 29.39 | 0.672 | 0.296 | 54.47 | 284.82 |
| MagicAnimate | 2D | 3.13E-4 | 29.16 | 0.714 | 0.239 | 21.75 | 179.07 |
| AnimateAnyone | 2D | - | 29.56 | 0.718 | 0.285 | - | 171.90 |
| Champ | **2D+3D** | 3.02E-4 | 29.84 | 0.773 | 0.235 | 26.14 | 170.20 |
| **DreamDance** | **2D** | **2.89E-4** | **29.90** | **0.798** | **0.233** | **19.86** | **153.07** |

Key findings:
- DreamDance surpasses Champ, which relies on 2D+3D (SMPL) guidance, using only 2D pose input.
- FVD improves from 170.20 to 153.07 (~10% gain), demonstrating that enriched geometric guidance effectively enhances temporal consistency.
- SSIM improves from 0.773 to 0.798, indicating a significant gain in intra-frame structural consistency.

### Ablation Study

Effect of different conditioning combinations:

| Guidance Condition | L1 ↓ | SSIM ↑ | FID-VID ↓ | FVD ↓ |
|-------------------|------|--------|-----------|-------|
| w/o Pose | 3.38E-4 | 0.743 | 22.38 | 175.37 |
| w/o Depth | 3.95E-4 | 0.701 | 24.56 | 193.23 |
| w/o Normal | 3.67E-4 | 0.723 | 23.28 | 183.84 |
| **Pose + Depth + Normal** | **2.89E-4** | **0.798** | **19.86** | **153.07** |

Effectiveness of Geometry Attention:

| Controller | L1 ↓ | SSIM ↑ | FVD ↓ |
|-----------|------|--------|-------|
| w/o GeoAttn | 3.36E-4 | 0.767 | 165.27 |
| **w/ GeoAttn** | **2.89E-4** | **0.798** | **153.07** |

Key findings:
- Depth maps contribute most to generation quality (FVD rises from 153 to 193 when removed).
- Normal maps also contribute substantially (FVD rises from 153 to 183 when removed).
- Geometry Attention reduces FVD from 165 to 153, effectively ensuring multi-modal alignment.

### Efficiency Analysis

| Method | Geometry Guidance Acquisition Time |
|--------|-----------------------------------|
| Champ (3D prediction + smoothing + rendering) | 0.98 s/frame |
| DreamDance (Stage 1) | 1.13 s/frame |

Runtime is comparable, yet DreamDance consolidates all steps into a single diffusion model, resulting in a simpler and more accessible pipeline.

## Highlights & Insights

1. **SMPL-free design**: A generative model replaces the rigid 3D parametric model for geometry guidance acquisition, avoiding SMPL's drawbacks (poor editability, signal misalignment, neglect of clothing details).
2. **Unified diffusion for geometry**: RGB, depth, and normal maps are jointly generated within a single diffusion process, ensuring semantic alignment by construction.
3. **Three-stage progressive training**: Independent training → cross-modal alignment → temporal consistency, avoiding interference among multiple objectives.
4. **Robust conditioning strategy**: Dropping control signals via dropout mitigates error accumulation in the two-stage pipeline in a simple yet effective manner.
5. **TikTok-Dance5K dataset**: A dataset comprising 5K high-quality dance videos with complete annotations is constructed and will be released publicly.

## Limitations & Future Work

1. Although the robust conditioning strategy mitigates error accumulation in the two-stage pipeline, it cannot be entirely eliminated.
2. Stage 1 trains at low resolution, potentially losing high-frequency geometric details.
3. The ability to handle complex hand gestures and facial expressions remains to be validated (poor SMPL hand reconstruction is a known limitation of Champ, but whether DreamDance genuinely resolves this requires further verification).
4. Inference requires two-stage forward passes, which may result in slower overall inference speed compared to single-stage approaches.

## Related Work & Insights

- **Human image animation**: DisCo/DreamPose (per-frame diffusion) → AnimateAnyone/MagicAnimate (temporal attention) → Champ (SMPL guidance) → Ours (diffusion-based geometry guidance).
- **Geometry generation**: HyperHuman (joint appearance and geometry generation), Depth/Normal Estimation (Metric3D).
- **Video diffusion**: SVD (strong temporal prior), AnimateDiff (motion modules); this work adopts SVD as the video foundation model.

## Rating

- Novelty: ⭐⭐⭐⭐ — Replacing SMPL with a diffusion model for geometry guidance is a novel but incremental contribution.
- Technical Depth: ⭐⭐⭐⭐ — Geometry attention, multi-domain CFG, and robust conditioning constitute a complete and well-designed system.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validation on two datasets with comprehensive ablations and rich qualitative results.
- Value: ⭐⭐⭐⭐⭐ — Eliminating the SMPL dependency substantially lowers the barrier to use, and the dataset will be publicly released.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Generative Modeling of Shape-Dependent Self-Contact Human Poses](generative_modeling_of_shape-dependent_self-contact_human_poses.md)
- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](../../ICLR2026/image_generation/direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[ICCV 2025\] CompleteMe: Reference-based Human Image Completion](completeme_reference-based_human_image_completion.md)

<!-- RELATED:END -->
