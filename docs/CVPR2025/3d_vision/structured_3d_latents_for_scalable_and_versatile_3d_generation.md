---
title: >-
  [Paper Note] Structured 3D Latents for Scalable and Versatile 3D Generation
description: >-
  [CVPR 2025][3D Vision][3D Generation] Proposes Structured LATents (SLat/TRELLIS), a unified 3D latent representation that fuses sparse 3D grids with DINOv2 multi-view features. It supports decoding into various formats such as radiance fields, 3D Gaussians, and meshes. A rectified flow Transformer with up to 2B parameters is trained on 500K 3D assets, generating high-quality 3D assets in approximately 10 seconds and supporting flexible local editing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Generation"
  - "Structured Latent Space"
  - "Sparse Voxels"
  - "Multi-Format Decoding"
  - "Rectified Flow"
date: 2026-05-08
content_hash: 80409c558887327b
---

# Structured 3D Latents for Scalable and Versatile 3D Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.01506](https://arxiv.org/abs/2412.01506)  
**Code**: [GitHub (TRELLIS)](https://github.com/Microsoft/TRELLIS)  
**Area**: 3D Vision/3D Generation  
**Keywords**: 3D Generation, Structured Latent Space, Sparse Voxels, Multi-Format Decoding, Rectified Flow

## TL;DR

Proposes Structured LATents (SLat/TRELLIS), a unified 3D latent representation that fuses sparse 3D grids with DINOv2 multi-view features. It supports decoding into various formats such as radiance fields, 3D Gaussians, and meshes. A rectified flow Transformer with up to 2B parameters is trained on 500K 3D assets, generating high-quality 3D assets in approximately 10 seconds and supporting flexible local editing.

## Background & Motivation

- **Fragmentation of 3D Representations**: Meshes, point clouds, radiance fields, and 3D Gaussians each have their own pros and cons—radiance fields and Gaussians render excellent appearance but make geometry extraction difficult, while meshes have precise geometry but weak appearance modeling. A unified generation paradigm has been lacking.
- **Limitations of Prior Latent Spaces**: Dimensional representations like triplanes are difficult to decode into multiple formats; point/voxel-based approaches fail to capture both fine geometry and appearance simultaneously; and they require expensive 3D fitting preprocessing.
- **Demand for Fitting-free Approaches**: Previous methods require pre-fitting 3D data to specific representations (e.g., fitting 3DGS first, then encoding), which is time-consuming and lossy.
- **Scalability Bottlenecks**: Most existing 3D generation models have only hundreds of millions of parameters and are trained on tens of thousands of samples, lagging far behind the scale of 2D generative models.
- **Editing Flexibility**: Most 3D generation methods do not support post-generation editing (local modifications or flexible format switching).

## Method

### Overall Architecture

**Encoding**: Render dense multi-view images of the 3D asset → Extract feature maps with DINOv2 → Project and aggregate onto sparse active voxels → Encode via Sparse VAE into structured latents $\boldsymbol{z} = \{(\boldsymbol{z}_i, \boldsymbol{p}_i)\}_{i=1}^L$.  
**Decoding**: Different decoders decode SLat into 3D Gaussians, radiance fields (Strivec CP decomposition), or meshes (FlexiCubes), respectively.  
**Generation**: A two-stage pipeline consisting of (1) a rectified flow Transformer to generate the sparse structure $\{\boldsymbol{p}_i\}$, and (2) a rectified flow Transformer to generate local latent variables $\{\boldsymbol{z}_i\}$.

### Key Designs

**Design 1: Sparse Latent Space via DINOv2 Visual Feature Aggregation**
- **Function**: Encodes complete geometry and appearance information without requiring 3D fitting.
- **Mechanism**: Defines active voxels (surface intersections, $L \ll N^3$, average 20K) on a $64^3$ 3D grid. It renders 150 multi-view images of the 3D asset and uses pre-trained DINOv2 to extract feature maps. Each voxel is projected onto multi-view feature maps to retrieve and average the corresponding features, yielding $\boldsymbol{f}_i$. Active voxels provide the coarse structure, while DINOv2 features provide fine geometry and appearance.
- **Design Motivation**: DINOv2 has demonstrated strong 3D perception capabilities; leveraging its features directly avoids training a specialized 3D encoder. The combination of sparse voxels and rich features decouples structure from details.

**Design 2: Feature Sharing Across Multi-Format Decoders**
- **Function**: Supports decoding the same latent space into Gaussians, radiance fields, or meshes.
- **Mechanism**: Initializes by end-to-end training the encoder and decoder with a 3DGS decoder (high fidelity + efficient). After freezing the encoder, decoders for other formats are trained individually. All decoders share the same Transformer backbone structure, differing only in the output layers: Gaussians output displacement, color, scale, opacity, and rotation ($K$ per voxel); radiance fields output CP decomposition vectors; and meshes output FlexiCubes parameters and SDF values.
- **Design Motivation**: Demonstrates that the latent space learned using Gaussians as a proxy task can transfer to other representations, verifying that SLat is indeed representation-agnostic.

**Design 3: Two-Stage Sparse Rectified Flow Transformer Generation**
- **Function**: Efficiently generates sparse structures and fine latent variables.
- **Mechanism**: Stage 1 structure generator $\mathcal{G}_S$: Compresses the binary active grid into a low-resolution feature grid using a 3D convolutional VAE, then denoises it using a Transformer (dense grid + CFM loss). Stage 2 latent generator $\mathcal{G}_L$: Packs voxels using sparse convolution ($2^3$ regions) and restores them via upsampling after Transformer denoising. Both stages inject timesteps via adaLN and text/image conditions via cross-attention. It is scaled up to 2B parameters.
- **Design Motivation**: Decoupling structures and latents simplifies the tasks of each stage. Sparse convolution leverages the inherent complexity/sparsity of 3D data ($L \ll N^3$) to drastically reduce computation.

### Loss & Training

VAE Training: 3DGS decoding uses $\mathcal{L}_1$, D-SSIM, LPIPS rendering loss, and KL regularization. Radiance fields employ a similar rendering loss. Meshes utilize depth and normal map rendering losses. The generative model works with the CFM objective $\mathcal{L}_{CFM} = \mathbb{E}\|\boldsymbol{v}_\theta(\boldsymbol{x}, t) - (\boldsymbol{\epsilon} - \boldsymbol{x}_0)\|^2$.

## Key Experimental Results

### Reconstruction Fidelity Comparison

| Method | PSNR ↑ | LPIPS ↓ | CD ↓ | F-score ↑ |
|------|--------|---------|------|----------|
| LN3Diff | 26.44 | 0.076 | 0.0299 | 0.9649 |
| 3DTopia-XL | 25.34 | 0.074 | 0.0128 | 0.9939 |
| CLAY | - | - | 0.0124 | 0.9976 |
| **Ours (SLat)** | **32.74** | **0.025** | **0.0083** | **0.9999** |

### Generation Performance ~10s/object

- Comprehensively outperforms methods such as Shap-E, LN3Diff, CLAY, and 3DTopia-XL on the Toys4k benchmark.
- Supports text- and image-conditioned generation.
- Supports tuning-free local editing (deletion, addition, replacement).

### Key Findings

1. SLat achieves a reconstruction PSNR of 32.74, greatly surpassing other latent space methods (such as LN3Diff at 26.44), proving that DINOv2 feature aggregation is highly effective.
2. F-score of 0.9999—nearly lossless geometric reconstruction.
3. The encoder trained on Gaussians can be directly transferred to radiance field and mesh decoders, validating that SLat is representation-agnostic.
4. Scaling up to 500K data and 2B parameters brings significant quality improvements.

## Highlights & Insights

- **Fitting-Free Training**: Completely avoids time-consuming 3D pre-fitting, encoding directly from rendered images.
- **Representation-Agnostic Unified Latent Space**: Achieves high-quality decoding from a single latent space into three formats (Gaussians, radiance fields, and meshes) for the first time.
- **Flexible Editing**: Supports detail variations (modifying appearance while keeping structure) and regional editing (local regeneration) without requiring additional training.
- **DINOv2's 3D Perception Capabilities**: Fully validated, lowering the dependency of 3D generation on dedicated 3D encoders.

## Limitations & Future Work

- Error accumulation in two-stage generation: poor structure generation directly degrades latent variable generation.
- Current training data and conditioning mainly cover object-level tasks; scene-level generation remains to be explored.
- Active voxel resolution of $64^3$ may limit the expression of ultra-fine geometric details.

## Related Work & Insights

- The sparse voxel + visual feature architecture of SLat can scale to scene-level 3D generation.
- The potential of DINOv2 as a general 3D encoder warrants further exploration.
- The success of rectified flow in 3D generation demonstrates its applicability beyond 2D domains.

## Rating

⭐⭐⭐⭐⭐ — A milestone work that, for the first time, realizes high-quality, multi-format, and editable unified 3D generation. Elegant design (sparse voxels + DINOv2), well-scaled (500K data / 2B parameters), and fully open-source (code, models, and data are public), casting a profound impact on the 3D generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Native and Compact Structured Latents for 3D Generation](../../CVPR2026/3d_vision/native_and_compact_structured_latents_for_3d_generation.md)
- [\[CVPR 2025\] SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE](sar3d_autoregressive_3d_object_generation_and_understanding_via_multi-scale_3d_v.md)
- [\[CVPR 2025\] MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation](mar-3d_progressive_masked_auto-regressor_for_high-resolution_3d_generation.md)
- [\[CVPR 2025\] Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)
- [\[ICCV 2025\] From One to More: Contextual Part Latents for 3D Generation](../../ICCV2025/3d_vision/from_one_to_more_contextual_part_latents_for_3d_generation.md)

</div>

<!-- RELATED:END -->
