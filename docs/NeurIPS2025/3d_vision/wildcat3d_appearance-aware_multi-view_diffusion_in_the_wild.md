---
title: >-
  [Paper Note] WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild
description: >-
  [NeurIPS 2025][3D Vision][Novel View Synthesis] This paper proposes WildCAT3D, which extends the multi-view diffusion model CAT3D to learn scene-level novel view synthesis from in-the-wild internet data (e.g., tourist photographs) by explicitly modeling global appearance conditions, while simultaneously supporting appearance-controlled generation.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Novel View Synthesis
  - Multi-View Diffusion
  - Appearance Modeling
  - Scene-Level NVS
  - In-the-Wild Data
date: 2026-05-08
content_hash: 4df92d9a6449b325
---

# WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild

**Conference**: NeurIPS 2025
**arXiv**: [2506.13030](https://arxiv.org/abs/2506.13030)
**Code**: [Project Page](https://wildcat3d.github.io)
**Area**: 3D Vision
**Keywords**: Novel View Synthesis, Multi-View Diffusion, Appearance Modeling, Scene-Level NVS, In-the-Wild Data

## TL;DR

This paper proposes WildCAT3D, which extends the multi-view diffusion model CAT3D to learn scene-level novel view synthesis from in-the-wild internet data (e.g., tourist photographs) by explicitly modeling global appearance conditions, while simultaneously supporting appearance-controlled generation.

## Background & Motivation

Novel view synthesis (NVS) has achieved remarkable progress on object-level scenes in recent years, particularly with the multi-view diffusion paradigm exemplified by CAT3D. However, **scene-level NVS** still faces significant challenges:

**Root Cause**:

**Scarcity of multi-view training data**: Clean multi-view data primarily originates from synthetic renderings or isolated objects in crowdsourced videos, with limited data diversity and licensing constraints.

**Abundance but inconsistency of in-the-wild data**: A large volume of scene images (e.g., tourist photos) exists on the internet, but different photographs of the same scene vary substantially in lighting, weather, occlusion, and other factors, making them incompatible with existing multi-view diffusion architectures.

**Core Idea**: Inconsistent data can be leveraged through explicit disentanglement of "content" and "appearance" — when denoising target views, the model is allowed to "peek" at coarse-grained appearance information (e.g., weather, aspect ratio) for each view without leaking fine-grained details. During inference, the appearance embedding from source views is replicated to all target views to ensure generation consistency.

## Method

### Overall Architecture

WildCAT3D extends the CAT3D framework with two core modifications: (1) an appearance encoding branch, and (2) a warp conditioning mechanism. The overall input shape is $v \times (2k + d + 7) \times n \times n$, where $k+7$ denotes the original CAT3D channels (latent + camera embedding + binary mask), $d$ is the appearance embedding channel dimension, and $k$ is the warp latent channel dimension. The model learns the distribution $p(\mathbf{I}^u | \mathbf{I}^o, \mathbf{c}^a, A_\phi(\mathbf{I}^a), \mathbf{w}^o)$.

### Key Designs

1. **Generalizable Appearance Encoder**: A lightweight convolutional network that compresses image latents into a low-dimensional vector ($d=8$ dimensions), serving as an information bottleneck — encoding coarse-grained global appearance attributes such as weather, lighting, and aspect ratio, without leaking fine-grained image content. During training, this embedding is replicated across $n \times n$ spatial positions and concatenated to the CAT3D input channels, allowing the model to observe appearance conditions for all views (including noised unobserved views) during training. The encoder is jointly trained with the denoising objective and generalizes to novel scenes at inference time.

2. **Appearance-Aware CFG**: Directly applying standard CFG to the appearance condition leads to oversaturation artifacts, as the appearance embedding correlates with image brightness and color balance. The paper introduces a customized CFG scheme: in the "unconditional" setting, the appearance condition is retained while other observed view conditions are dropped, i.e., $p^{(\text{uncond})}(\mathbf{I}^u | \mathbf{c}^u, A_\phi(\mathbf{I}^a))$. The model thus observes appearance embeddings for all views in both conditional and unconditional settings.

3. **Appearance-Conditioned Inference**: At inference time, the appearance embedding of the first observed view $\mathbf{a}_0 = A_\phi(\mathbf{I}_0)$ is selected and replicated into the appearance channels of all unobserved views. This mechanism also enables appearance transfer — injecting the appearance embedding from an external image, or performing text-driven appearance control by retrieving appearance-matched images via CLIP.

4. **Warp Conditioning**: This mechanism addresses the inherent scale ambiguity in single-view NVS. Depth is estimated for the source view using DepthAnything, then aligned to a COLMAP point cloud via RANSAC to obtain metric scale. Source-view pixels are back-projected into a 3D point cloud and rendered into each target camera view. The VAE latent of the warped image is injected as an additional conditioning channel, providing a spatial cue for correct scene placement.

### Loss & Training

- A base CAT3D model is first trained on CO3D and Re10K datasets.
- The full WildCAT3D model is then fine-tuned on MegaScenes and CO3D.
- The same denoising loss as the original LDM is used.
- Default configuration uses $v=8$ view slots: 1 observed + 7 unobserved randomly sampled scene views.
- At inference, $v$ can be increased to 16 for video generation.

## Key Experimental Results

### Main Results (Single-View NVS Benchmarks)

| Method | DTU PSNR↑ | DTU FID↓ | Mip-NeRF360 PSNR↑ | Mip-NeRF360 FID↓ |
|--------|-----------|----------|--------------------|------------------|
| ZeroNVS (released) | 5.799 | 160.0 | 6.999 | 137.0 |
| MS NVS | 8.795 | 85.96 | 14.06 | 64.41 |
| **WildCAT3D** | **10.77** | **57.32** | **14.77** | **42.17** |

| Method | Re10K PSNR↑ | Re10K FID↓ | MegaScenes PSNR↑ | MegaScenes FID↓ |
|--------|-------------|-----------|-------------------|-----------------|
| MS NVS | 17.22 | 60.01 | 13.40 | 11.58 |
| **WildCAT3D** | **21.58** | **24.70** | **13.92** | **9.871** |

### Ablation Study

| Configuration | DTU PSNR↑ | DTU FID↓ | MipNeRF PSNR↑ | MipNeRF FID↓ | Notes |
|---------------|-----------|----------|---------------|-------------|-------|
| WildCAT3D | **10.77** | **57.32** | **14.77** | **42.17** | Full model |
| -warp | 9.795 | - | - | - | Without warp conditioning |
| -warp-app | - | - | - | - | Without warp and appearance → inconsistent outputs |

### Key Findings

- WildCAT3D surpasses prior state-of-the-art across all datasets while using **fewer** training data sources.
- The advantage is particularly pronounced on out-of-distribution datasets (DTU object-level and Mip-NeRF 360 scene-level).
- Removing warp conditioning significantly degrades view alignment; removing appearance modeling leads to inconsistent output images.
- Appearance embeddings naturally cluster into semantically meaningful groups (e.g., night scenes, blue sky, indoor environments), demonstrating that the encoder learns useful representations.
- The framework enables novel applications including appearance interpolation and text-driven appearance editing.

## Highlights & Insights

- The strategy of **converting "data inconsistency" into an advantage** is highly elegant: by explicitly modeling appearance variation, inconsistent in-the-wild data becomes a rich training resource.
- The appearance encoder is designed as an information bottleneck (only 8 dimensions), encoding global appearance while preventing content leakage — a refined engineering choice.
- The customized CFG strategy avoids oversaturation artifacts, reflecting a deep understanding of diffusion model behavior.
- The warp conditioning mechanism resolves scale ambiguity by injecting coarse geometric constraints without enforcing hard constraints, allowing the model to correct depth estimation errors.

## Limitations & Future Work

- Training relies on COLMAP SfM point clouds for depth alignment, making the approach inapplicable when SfM fails.
- Appearance modeling operates at a coarse granularity (global vector), and locally varying appearance differences (e.g., partial shadows) may not be precisely controlled.
- Only single-view input for scene-level NVS is demonstrated; the performance under multi-view input settings is not thoroughly evaluated.
- Generation quality remains bounded by the capacity of the underlying LDM.

## Related Work & Insights

- **CAT3D**: The baseline method directly extended in this work; the current state-of-the-art multi-view diffusion model.
- **MegaScenes**: A large-scale in-the-wild scene dataset providing both training data and an NVS baseline.
- **NeRF-W / Ha-NeRF**: Prior methods for modeling appearance variation in in-the-wild data, but requiring per-scene optimization.
- Insight: Information bottleneck-style appearance encoding can be generalized to other multi-view generation tasks to handle data inconsistency.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First work to incorporate in-the-wild tourist photographs into multi-view diffusion model training; the appearance disentanglement design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-dataset evaluation, ablation studies, and application demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; method description is systematic.
- **Value**: ⭐⭐⭐⭐⭐ Unlocks abundant in-the-wild data for scene-level 3D generation; highly pioneering contribution.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](../../ICCV2025/3d_vision/materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[NeurIPS 2025\] Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion](scaffold_diffusion_sparse_multi-category_voxel_structure_generation_with_discret.md)
- [\[ICCV 2025\] Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](../../ICCV2025/3d_vision/bootstrap3d_improving_multiview_diffusion_model_with_synthet.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](../../ICCV2025/3d_vision/spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[NeurIPS 2025\] GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion](geocomplete_geometry-aware_diffusion_for_reference-driven_image_completion.md)

<!-- RELATED:END -->
