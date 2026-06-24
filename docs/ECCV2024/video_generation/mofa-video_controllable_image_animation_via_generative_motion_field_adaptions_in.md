---
title: >-
  [Paper Note] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model
description: >-
  [ECCV 2024][Video Generation] This paper proposes MOFA-Video, which equips a frozen image-to-video diffusion model (SVD) with controllable motion capabilities by designing multiple domain-specific motion field adapters (MOFA-Adapters). It supports various control signals and their combinations, such as hand-drawn trajectories and facial landmarks, to achieve open-domain controllable image animation.
tags:
  - "ECCV 2024"
  - "Video Generation"
date: 2026-05-08
content_hash: 27517df6d473067a
---

# MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2405.20222](https://arxiv.org/abs/2405.20222)  
**Area**: Image Generation

## TL;DR

This paper proposes MOFA-Video, which equips a frozen image-to-video diffusion model (SVD) with controllable motion capabilities by designing multiple domain-specific motion field adapters (MOFA-Adapters). It supports various control signals and their combinations, such as hand-drawn trajectories and facial landmarks, to achieve open-domain controllable image animation.

## Background & Motivation

- **In-domain animation methods** (e.g., SadTalker) can finely control specific categories (faces, fluids) but are restricted to specific domains and cannot generalize to the open domain.
- **Diffusion-based I2V models** (e.g., SVD, AnimateDiff) can handle open-domain image animation, but the generated content may deviate from the input image and only support text or simple idle animations, exhibiting weak control capabilities.
- **Limitations of existing control methods**: DragNUWA models trajectories via adaptive normalization but suffers from poor spatial correspondence; MotionCtrl relies on T2V models and lacks a world coordinate system.
- **Core Problem**: How to build a unified framework to achieve fine-grained controllable animation from multiple motion domains on open-domain images?

## Method

### Overall Architecture

MOFA-Video appends the MOFA-Adapter as a motion control module to a frozen Stable Video Diffusion (SVD) model, similar to the concept of ControlNet. The key is to unify control signals from different domains into a sparse motion vector representation, which is then used to generate videos through a unified adapter structure.

### Key Designs

**1. MOFA-Adapter Structure**:
- **Sparse-to-Dense (S2D) Motion Generation Network**: Receives the first frame image and sparse motion prompts to generate a dense optical flow field, utilizing a CMP network structure.
- **Reference Image Encoder**: A multi-scale convolutional feature encoder that extracts multi-scale features of the first frame for warping.
- **Fusion Encoder**: A trainable copy of the SVD encoder, fusing the warped features with the features of the SVD decoder.

**2. Domain-Aware Motion Control**:
- **Open-Domain Trajectory**: Trained by sampling sparse motion vectors from video optical flow, accepting hand-drawn trajectories during inference.
- **Facial Landmarks**: Converts facial landmark displacements into sparse motion vectors, simplifying the framework with a unified representation.
- **Multi-Adapter Composition**: MOFA-Adapters from different domains can be combined in a zero-shot manner, utilizing a mask-aware strategy to fuse control signals from different regions.

**3. Long Video Generation**: A periodic sampling strategy is proposed. Within each diffusion step, frames are grouped (14 frames per group, with a 7-frame overlap), and the predicted noise of overlapping frames is averaged to achieve temporal consistency for longer videos.

### Loss & Training

The SVD parameters are frozen, and only the MOFA-Adapter parameters $\theta_{\mathcal{M}}$ are optimized:

$$\mathcal{L} = \| \mathcal{S}(\mathcal{V}_t, t, \mathcal{M}(\mathcal{V}_t, t, I, F^s; \theta_{\mathcal{M}})) - \mathcal{V} \|^2$$

where $\mathcal{S}$ is the frozen SVD and $\mathcal{V}$ is the video latent representation.

## Key Experimental Results

### Main Results

**Trajectory Control Comparison (vs. DragNUWA)**:

| Method | Frame Consistency↑ | LPIPS↓ | FID↓ | FVD↓ | Control Accuracy (User)↑ | Visual Quality (User)↑ |
|------|-----------|--------|------|------|----------------|----------------|
| DragNUWA | 0.9302 | 0.2705 | 19.66 | 91.38 | 2.76 | 3.18 |
| **MOFA-Video** | **0.9390** | **0.2274** | **16.82** | **86.76** | **3.58** | **3.42** |

**Portrait Animation Comparison (vs. SadTalker, StyleHEAT)**:

| Method | CPBD↑ | ID↑ | Fidelity (User)↑ | Naturalness (User)↑ | Visual Quality (User)↑ |
|------|-------|-----|--------------|--------------|----------------|
| SadTalker | 0.3218 | 0.9188 | 4.15 | 3.12 | 3.97 |
| StyleHEAT | 0.2577 | 0.7993 | 3.26 | 3.65 | 3.70 |
| **MOFA-Video** | **0.4075** | **0.9293** | **4.80** | **3.97** | **4.52** |

### Ablation Study

**Network Architecture Ablation (Trajectory Control)**:

| Variant | LPIPS↓ | FID↓ | FVD↓ |
|------|--------|------|------|
| w/o warping (pure sparse conditions) | 0.2619 | 18.80 | 184.27 |
| w/o S2D (sparse warping) | 0.2376 | 16.87 | 81.80 |
| w/o tuning (directly using the reconstruction model) | 0.2163 | 16.97 | 102.17 |
| **Full Model** | **0.2274** | **16.82** | **86.76** |

### Key Findings

- Sparse condition models cannot precisely control the trajectory of the target object due to the lack of spatial warping operations, which results in spatial misalignment.
- Sparse warping models can control trajectories but produce severe visual artifacts due to the lack of dense optical flow guidance.
- MOFA-Adapters for different domains must be trained separately; directly using the open-domain model for facial animation leads to unnatural expressions.
- The periodic sampling strategy significantly outperforms the naive frame grouping method, effectively resolving error accumulation and temporal inconsistency in long videos.

## Highlights & Insights

- Unifying multi-domain motion control into a sparse motion vector problem is an elegant and scalable design.
- The explicit sparse-to-dense optical flow generation combined with the feature warping strategy achieves a good balance between control accuracy and generation quality.
- The zero-shot composition capability of multiple MOFA-Adapters makes it possible to simultaneously control both facial expressions and background motion.
- Compared to the implicit trajectory modeling of DragNUWA, the explicit optical flow method better confines the motion regions.

## Limitations & Future Work

- Cannot control or generate new content that deviates significantly from the input image (constrained by the short-video training data of SVD).
- Visual artifacts such as blurriness or structure loss may occur under large motion guidance.
- Video length is restricted by SVD's 14-frame window, requiring additional periodic sampling strategies for long videos.

## Rating

- **Novelty**: 7/10 — The adapter concept derives from ControlNet; the core innovation lies in the unified modeling of motion fields and multi-domain composition.
- **Technical Depth**: 8/10 — Explicit motion modeling with S2D + warping is solidly designed, and the multi-adapter composition scheme is reasonable.
- **Experimental Thoroughness**: 8/10 — Comparative and ablation experiments are comprehensive, though quantitative evaluations on long videos are lacking.
- **Impact**: 7/10 — Provides a practical, unified framework for controllable video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [\[ECCV 2024\] Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)
- [\[ECCV 2024\] MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)
- [\[ECCV 2024\] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [\[ECCV 2024\] Evaluating Text-to-Visual Generation with Image-to-Text Generation](evaluating_text-to-visual_generation_with_image-to-text_generation.md)

</div>

<!-- RELATED:END -->
