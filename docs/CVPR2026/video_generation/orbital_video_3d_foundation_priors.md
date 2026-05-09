---
title: >-
  [Paper Note] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors
description: >-
  [CVPR 2026][Video Generation][orbital video generation] This paper proposes leveraging the latent features of a 3D foundation generative model (Hunyuan3D) as shape priors, injecting them into a base video diffusion model via a multi-scale 3D adapter, to generate geometrically realistic and view-consistent orbital videos from a single image.
tags:
  - CVPR 2026
  - Video Generation
  - orbital video generation
  - 3D priors
  - video diffusion
  - multi-view consistency
  - geometric realism
date: 2026-05-08
content_hash: d8dfbbdb92c51df9
---

# Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors

**Conference**: CVPR 2026
**arXiv**: [2604.12309](https://arxiv.org/abs/2604.12309)
**Code**: N/A
**Area**: 3D Vision / Video Generation
**Keywords**: orbital video generation, 3D priors, video diffusion, multi-view consistency, geometric realism

## TL;DR

This paper proposes leveraging the latent features of a 3D foundation generative model (Hunyuan3D) as shape priors, injecting them into a base video diffusion model via a multi-scale 3D adapter, to generate geometrically realistic and view-consistent orbital videos from a single image.

## Background & Motivation

**Background**: Orbital video generation—synthesizing videos from an object image and a camera trajectory—has attracted considerable attention. Existing methods primarily rely on pixel-level attention to ensure multi-view consistency.

**Limitations of Prior Work**: Pixel-level attention fails to establish effective pixel correspondences under large viewpoint changes (e.g., front-to-back views), leading to distortions and unnatural structures. Some methods introduce 2D foundation models (e.g., monocular depth maps) as geometric conditions, but such 2.5D priors cannot model complete object geometry and remain insufficient for unobserved or occluded regions.

**Key Challenge**: Video diffusion models lack 3D world knowledge; pixel-level attention or 2.5D priors alone cannot guarantee geometric realism under large viewpoint variations.

**Goal**: To exploit the ability of 3D foundation models to encode complete object geometry, providing effective 3D shape constraints for video generation.

**Key Insight**: The latent features of a 3D foundation model can serve as effective 3D shape priors, simultaneously offering auxiliary constraints and enhancing view consistency.

**Core Idea**: Extract two-scale latent features from the 3D foundation model—a global shape vector and view-dependent latent images—and inject them into the video diffusion model via a multi-scale adapter.

## Method

### Overall Architecture

Built upon an SVD-based video diffusion model, the input image is simultaneously fed into the 3D foundation model (Hunyuan3D) to obtain shape priors. Features at two scales are injected into each Transformer block via cross-attention through the multi-scale 3D adapter, guiding video generation. The 3D feature extraction incurs only approximately 2 seconds of additional overhead during inference.

### Key Designs

1. **Dual-Scale 3D Foundation Priors**:

    - **Function**: Provide global and local information about complete object geometry.
    - **Mechanism**: (i) Global latent vector $\hat{\bm{p}}_0 \in \mathbb{R}^{L \times D}$: obtained by denoising from DINOv2 feature conditions via a rectified flow model, encoding overall structural guidance. (ii) Local latent images $\hat{\mathbf{L}} \in \mathbb{R}^{M \times H_l \times W_l \times D'}$: volumetric features are queried from the global vector on a regular 3D grid and projected onto $M=8$ canonical viewpoints.
    - **Design Motivation**: The global vector provides holistic structural constraints, while the local latent images supply view-dependent fine-grained geometric details. Using compact latent features avoids the computational cost of explicit mesh extraction.

2. **Multi-Scale 3D Adapter**:

    - **Function**: Efficiently inject 3D priors into the base video model.
    - **Mechanism**: For the input feature $\mathbf{f}_i^{(0)}$ of each Transformer block, cross-attention with the global vector first produces $\mathbf{f}_i^{(1)}$, followed by cross-attention with the latent images to produce $\mathbf{f}_i^{(2)}$. The global vector is replicated $N$ times to share a unified shape reference across frames.
    - **Design Motivation**: The adapter operates as a plug-and-play module, preserving the capabilities inherited from the base video model's general pre-training and supporting flexible model replacement.

3. **Hunyuan3D as the Shape Prior Source**:

    - **Function**: Provide high-quality 3D shape reconstruction.
    - **Mechanism**: Hunyuan3D is selected because (i) it models complete object geometry directly in a 3D latent space without relying on intermediate novel-view synthesis steps; and (ii) it decouples shape and appearance through explicit geometric supervision, yielding semantically rich latent representations.
    - **Design Motivation**: Unlike prior 3D methods that depend on novel-view synthesis, the latent features of a native 3D generative architecture are more suitable as shape conditions.

### Loss & Training

Standard denoising objective: $\mathcal{L} = \mathbb{E}[w(t) \| \mathcal{V}_\sigma(\bm{z}_t) - \bm{\epsilon} \|_2^2]$. The 3D foundation model is frozen; only the adapter (0.3B parameters) is trained. Training is conducted on Objaverse-XL synthetic rendered data for 80K iterations.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP-S↑ | MEt3R↓ |
|--------|-------|-------|--------|---------|--------|
| SV3D | 20.48 | 0.91 | 0.12 | 92.84 | 0.07 |
| Hi3D | 19.32 | 0.90 | 0.14 | 90.61 | 0.09 |
| Hunyuan3D (rendering) | 20.25 | 0.91 | 0.11 | 93.44 | - |
| Wonder3D | 19.53 | 0.89 | 0.15 | 89.03 | - |
| **Ours (21 frames)** | **22.78** | **0.92** | **0.09** | **94.19** | **0.05** |

### Ablation Study

| Configuration | PSNR↑ | CLIP-S↑ | MEt3R↓ |
|---------------|-------|---------|--------|
| No prior (baseline) | 20.06 | 91.26 | 0.08 |
| + Global vector | 21.86 | 93.12 | 0.06 |
| + Global + Local (full) | **22.78** | **94.19** | **0.05** |

### Key Findings

- The global vector substantially improves multi-view consistency (MEt3R reduced from 0.08 to 0.06) and geometric realism (CLIP-S gains nearly 2 points).
- Local volumetric features further boost overall performance, particularly visual fidelity (PSNR gain of approximately 1 point).
- The overhead of 3D feature extraction is minimal (global vector: 1.8s; volumetric features: 0.34s; projection: 0.11s).

## Highlights & Insights

- Using latent features of the 3D foundation model rather than explicit meshes as conditions is a key innovation: it avoids costly mesh extraction while retaining complete shape information.
- The adapter acts as a soft constraint: the video model retains its stochasticity and the ability to balance image and shape conditions, preventing over-constrained generation.

## Limitations & Future Work

- Training is conducted exclusively on synthetic data; a domain gap may exist for real-world scenes.
- The object orientation inferred by the 3D foundation model may not be fully aligned with the target.
- Evaluation is limited to object-level videos; extension to scene-level content has not been explored.
- The framework is extensible to longer videos and more complex camera trajectories.

## Related Work & Insights

- **vs. SV3D/Hi3D**: These methods lack 3D priors and produce unrealistic structures under large viewpoint changes; the proposed approach addresses this via a 3D foundation model.
- **vs. Iterative Refinement Methods**: Methods such as Hi3D require coarse 3D reconstruction followed by refinement, which is time-consuming and quality-dependent on the initial result; the proposed prior requires only a single training-agnostic inference pass.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using latent features from a 3D foundation model as priors for video generation is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark, multi-baseline comparisons with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and well-organized.
- Value: ⭐⭐⭐⭐ Significant contribution to orbital video generation and novel-view synthesis.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[ICCV 2025\] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors](../../ICCV2025/video_generation/normalcrafter_learning_temporally_consistent_normals_from_video_diffusion_priors.md)
- [\[CVPR 2026\] Gloria: Consistent Character Video Generation via Content Anchors](gloria_consistent_character_video_generation_via_content_anchors.md)
- [\[ICLR 2026\] BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](../../ICLR2026/video_generation/bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)
- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)

<!-- RELATED:END -->
