---
title: >-
  [Paper Note] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image
description: >-
  [ICLR 2026][3D Vision][single-image 3D scene generation] This paper proposes One2Scene, which decomposes the ill-posed problem of generating an explorable 3D scene from a single image into three sub-tasks: (1) panorama generation to expand visual coverage, (2) a feed-forward 3DGS network that constructs an explicit 3D geometric scaffold from sparse anchor views, and (3) scaffold-guided novel view synthesis via Dual-LoRA that fuses high-quality anchor views with geometric priors. The method achieves geometrically consistent and photorealistic scene generation under large viewpoint changes, significantly outperforming state-of-the-art methods.
tags:
  - ICLR 2026
  - 3D Vision
  - single-image 3D scene generation
  - panoramic depth estimation
  - 3D Gaussian Splatting
  - geometric scaffold
  - novel view synthesis
date: 2026-05-08
content_hash: 311778a479dd3641
---

# One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image

**Conference**: ICLR 2026
**arXiv**: [2602.19766](https://arxiv.org/abs/2602.19766)
**Code**: [Project Page](https://one2scene5406.github.io/)
**Area**: 3D Vision / Scene Generation
**Keywords**: single-image 3D scene generation, panoramic depth estimation, 3D Gaussian Splatting, geometric scaffold, novel view synthesis

## TL;DR
This paper proposes One2Scene, which decomposes the ill-posed problem of generating an explorable 3D scene from a single image into three sub-tasks: (1) panorama generation to expand visual coverage, (2) a feed-forward 3DGS network that constructs an explicit 3D geometric scaffold from sparse anchor views, and (3) scaffold-guided novel view synthesis via Dual-LoRA that fuses high-quality anchor views with geometric priors. The method achieves geometrically consistent and photorealistic scene generation under large viewpoint changes, significantly outperforming state-of-the-art methods.

## Background & Motivation

**Background**: Generating explorable 3D scenes from a single image is a core challenge in 3D vision. Reconstruction-based methods (NeRF/3DGS) require dense multi-view images, while sparse-view methods cannot extrapolate beyond the observed region. Generative approaches include video diffusion models (ReconX/ViewCrafter), panoramic pipelines (DreamScene360/DreamCube), and navigation-with-inpainting methods (WonderJourney/Pano2Room).

**Limitations of Prior Work**: (1) Video diffusion methods lack a persistent 3D representation, leading to geometric collapse due to accumulated errors over long sequences; (2) panoramic methods observe the scene from a single point and lack explicit 3D information, causing severe distortion under large viewpoint changes; (3) iterative navigation methods suffer from accumulated errors that cause global semantic drift and stretched geometry.

**Key Challenge**: The extreme scarcity of information in a single image stands in fundamental conflict with the requirement for a globally consistent 3D scene. Existing methods either lack global coverage (single-viewpoint methods), geometric constraints (generative methods), or suffer from error accumulation (iterative methods).

**Goal**: (a) How to obtain global visual coverage from a single image? (b) How to establish explicit 3D geometric constraints? (c) How to maintain geometric consistency and visual quality under large viewpoint changes?

**Key Insight**: The problem is decomposed into three more tractable sub-problems — first expanding 2D coverage via panorama generation, then establishing a 3D scaffold via multi-view stereo matching, and finally constraining novel view synthesis with scaffold priors. A key insight is reformulating monocular panoramic depth estimation as a multi-view stereo matching problem, thereby leveraging strong geometric priors learned from large-scale multi-view datasets.

**Core Idea**: By providing an explicit 3D geometric scaffold as a stable global geometric and appearance prior for single-image scene generation, the method fundamentally avoids error accumulation and scale ambiguity.

## Method

### Overall Architecture
**Input**: A single image. **Output**: A 3D scene explorable from arbitrary viewpoints (high-quality novel view images).

Three stages:
- **Stage 1 (Panorama Generation)**: Single image → cubemap panorama → 6 anchor views
- **Stage 2 (3D Scaffold Construction)**: 6 sparse anchor views → feed-forward 3DGS network → explicit 3D geometric scaffold (0.5 seconds)
- **Stage 3 (Scaffold-Guided Synthesis)**: Coarse scaffold-rendered views + high-quality anchor views → diffusion model → photorealistic novel views

### Key Designs

1. **Panoramic Anchor View Generation**:

    - **Function**: Expands a single image into a 360° panorama, then projects it into 6 cubemap anchor views.
    - **Mechanism**: Hunyuan-Pano-DiT is used to generate the panorama, which is then projected into 6 perspective cubemap views (FoV = 95°, with 2.5° overlap between adjacent views).
    - **Design Motivation**: The panorama provides global semantic coverage, while cubemap projection enables the use of multi-view stereo matching priors trained on perspective images — more robust than directly processing equirectangular panoramas (which suffer from projection distortion).

2. **Feed-Forward 3DGS Geometric Scaffold (Bidirectional Fusion)**:

    - **Function**: Predicts 3D Gaussian parameters feed-forwardly from 6 sparse anchor views to construct an explicit 3D scaffold.
    - **Mechanism**: Built upon the VGGT backbone. Panoramic depth estimation is reformulated as multi-view stereo matching using the 6 cubemap views as "multi-view" inputs. The key innovation is the Bidirectional Fusion module: 6 view features $F_i$ → Cube-to-Equirectangular (C2E) projection into a unified equirectangular space → convolutional fusion → E2C transformation back to cubemap space → residual connection: $F_i' = F_i + E2C(H_c(C2E(\{F_i\})))$. Gaussian centers are computed via depth unprojection: $\mu = K^{-1}ud + \Delta$
    - **Design Motivation**: The 6 cubemap views have minimal overlap (only 2.5°), and existing multi-view models (e.g., VGGT) degrade significantly under such sparse overlap. The Bidirectional Fusion module enforces cross-view consistency via an intermediate equirectangular representation, while residual connections preserve view-specific details.

3. **Scaffold-Guided Novel View Synthesis (Dual-LoRA)**:

    - **Function**: Leverages scaffold priors to generate photorealistic images from arbitrary viewpoints.
    - **Mechanism**: Built upon the SEVA architecture. Scaffold-rendered views carry rich geometric information but contain artifacts and holes, while anchor views are high quality but lack geometric information — two heterogeneous conditioning signals. The Dual-LoRA strategy uses two independent LoRA modules to process anchor views and scaffold-rendered views respectively, then fuses both into the noisy latent via 3D attention. A memory condition (selecting the most recently generated frames from a memory bank) further ensures temporal consistency over long sequences.
    - **Design Motivation**: Naïve channel-wise concatenation cannot effectively differentiate and exploit the two heterogeneous conditioning signals. Dual-LoRA enables the model to separately learn to extract useful information from high-quality appearance and coarse geometry.

### Loss & Training
- **Stage 2 (3DGS)**: Composite loss = MSE rendering loss + LPIPS perceptual loss + SILog depth loss. Trained for 80K iterations on Structured3D / Deep360 / Matterport3D / Stanford2D3D.
- **Stage 3 (Synthesis)**: Based on SEVA; Adam optimizer, lr = 1.25e-5, batch = 16, 40K iterations. Training data are obtained from DL3DV and RealEstate10K via sparse reconstruction with MVSplat, deliberately simulating artifacts from sparse inputs.

## Key Experimental Results

### Main Results: Explorable 3D Scene Generation (WorldScore Benchmark Variant)

| Method | NIQE↓ | Q-Align↑ | CLIP-I↑ | CamMC↓ | RotErr↓ |
|---|---|---|---|---|---|
| DreamScene360 | 8.40 | 1.91 | 74.24 | - | - |
| WonderJourney | 4.97 | 3.02 | 77.92 | - | - |
| SEVA | 4.53 | 3.20 | 87.82 | 0.558 | 0.165 |
| VMem | 6.86 | 2.95 | 75.80 | 0.998 | 0.569 |
| **One2Scene** | **4.43** | **4.13** | **89.95** | **0.389** | **0.107** |

### Ablation Study: Impact of Scaffold Quality on Final Generation

| Configuration | NIQE↓ | Q-Align↑ | CLIP-I↑ | CamMC↓ |
|---|---|---|---|---|
| Replace with AnySplat | 4.96 | 3.61 | 81.96 | 0.616 |
| **Ours (Full)** | **4.43** | **4.13** | **89.95** | **0.389** |

### Key Findings
- **Scaffold quality is decisive**: Replacing the proposed scaffold with AnySplat drops CLIP-I from 89.95 to 81.96 and raises CamMC from 0.389 to 0.616, confirming that a high-quality scaffold is central to the approach.
- **Leading depth estimation**: On Matterport3D (finetuned), AbsRel = 0.0391 vs. Prev. SOTA 0.0850 (>50% gain); on Stanford2D3D (zero-shot), AbsRel = 0.0675, surpassing all prior methods.
- **Efficiency**: Reconstructing the scaffold from 6 sparse views takes only 0.5 seconds (H20), 5.6× faster than AnySplat (20 views, 2.8 seconds).
- **Resolving scale ambiguity**: SEVA suffers from severe scale ambiguity (camera passing through walls) due to the absence of 3D constraints; the scaffold in One2Scene provides stable global scale anchoring.

## Highlights & Insights
- **Reformulating panoramic depth estimation as multi-view stereo matching** is a particularly elegant insight: projecting the panorama into cubemap faces unlocks models trained on large-scale multi-view datasets, bypassing the scarcity of panoramic depth data. This idea is transferable to any panoramic scene understanding task.
- **The Bidirectional Fusion module (C2E–E2C)**: performing global fusion in equirectangular space and projecting back to perspective space elegantly resolves cross-view consistency under extremely sparse overlap — a general-purpose solution for panoramic feature aggregation.
- **Dual-LoRA for heterogeneous conditioning**: given two conditions of different quality and nature (high-quality appearance vs. coarse geometry with artifacts), encoding them with separate LoRA modules before fusion substantially outperforms direct concatenation. This strategy is transferable to any generative task requiring the fusion of heterogeneous conditioning signals.
- **Systems thinking in three-stage decomposition**: an intractable problem is split into three tractable sub-problems, with the output of each stage providing progressively stronger constraints for the next.

## Limitations & Future Work
- Subtle inconsistencies between generated views may still remain (post-reconstruction optimization could further mitigate this).
- The quality of the panorama generation model directly affects all downstream stages — failures at this stage cannot be recovered.
- Training data construction relies on MVSplat's sparse reconstruction quality to simulate artifacts, which may not cover all real-world scenarios.
- The current method handles only static scenes; supporting dynamic scenes is a direction for future work.

## Related Work & Insights
- **vs. SEVA**: SEVA performs camera-controlled novel view synthesis directly from a single image but lacks a persistent 3D representation, leading to scale ambiguity and geometric inconsistency. One2Scene addresses this via an explicit scaffold providing global constraints.
- **vs. VMem**: VMem uses CUT3R for online reconstruction to maintain consistency, but low-quality generated frames in turn degrade reconstruction — a vicious cycle. One2Scene's pre-built scaffold avoids this problem entirely.
- **vs. Pano2Room**: Pano2Room builds scenes through iterative navigation and inpainting with strong indoor-scene priors, limiting generalization. One2Scene is feed-forward and imposes no scene-type restrictions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The three-stage decomposition and multi-view stereo reformulation are innovative, though individual components build on existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dimensional evaluation is comprehensive, ablations are sufficient, and depth estimation benchmark results are strong.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem decomposition is clear, and the motivation chain is logically coherent.
- **Value**: ⭐⭐⭐⭐ Represents a significant advance for single-image 3D scene generation; the three-stage paradigm may become a standard pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [\[ICLR 2026\] Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](../../CVPR2026/3d_vision/pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)
- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](learning_unified_representation_of_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
