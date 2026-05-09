---
title: >-
  [Paper Note] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image
description: >-
  [ICLR 2026][3D Vision][3D Scene Generation] One2Scene proposes a three-stage framework that decomposes single-image explorable 3D scene generation into: panorama generation → feed-forward 3D Gaussian splatting for geometric scaffold construction → scaffold-guided novel view synthesis. By reformulating panoramic depth estimation as a multi-view stereo matching problem, the method achieves geometrically consistent and freely explorable 3D scene generation.
tags:
  - ICLR 2026
  - 3D Vision
  - 3D Scene Generation
  - Gaussian Splatting
  - Novel View Synthesis
  - Panorama
  - Feed-forward Reconstruction
date: 2026-05-08
content_hash: 6d106ebae7d7a19e
---

# One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image

**Conference**: ICLR 2026
**arXiv**: [2602.19766](https://arxiv.org/abs/2602.19766)
**Code**: [https://one2scene5406.github.io/](https://one2scene5406.github.io/)
**Area**: 3D Vision / Scene Generation
**Keywords**: 3D Scene Generation, Gaussian Splatting, Novel View Synthesis, Panorama, Feed-forward Reconstruction

## TL;DR

One2Scene proposes a three-stage framework that decomposes single-image explorable 3D scene generation into: panorama generation → feed-forward 3D Gaussian splatting for geometric scaffold construction → scaffold-guided novel view synthesis. By reformulating panoramic depth estimation as a multi-view stereo matching problem, the method achieves geometrically consistent and freely explorable 3D scene generation.

## Background & Motivation

### Limitations of Prior Work

**State of the Field**: 1. **Single-image 3D scene generation is a severely ill-posed problem**: Generating freely explorable 3D scenes from a single image lacks 3D geometric information, causing existing methods to produce severe geometric distortions and artifacts under large viewpoint changes.
2. **Reconstruction methods require dense inputs**: NeRF and 3DGS typically require hundreds of input images; sparse-view reconstruction methods struggle to extrapolate to unobserved regions.
3. **Video diffusion methods are geometrically inconsistent**: 3D scene methods based on video generation (ReconX, ViewCrafter, etc.) suffer from accumulating geometric errors in long sequences and closed-loop scenes, leading to collapse.
4. **Panorama-based methods have limited exploration range**: DreamScene360, DreamCube, and similar methods convert panoramas to 3D scenes but support only limited viewpoint exploration, with rendering quality degrading sharply at distant viewpoints.
5. **Iterative navigation methods suffer from error accumulation**: Methods such as WonderJourney that perform incremental navigation and inpainting lead to global semantic drift and stretching artifacts.
6. **Scale ambiguity**: Single-image input causes methods such as SEVA to suffer from scale ambiguity, resulting in distorted object sizes and physically implausible phenomena such as cameras passing through walls.

## Method

### Overall Architecture: Three-Stage Decomposition

The ill-posed problem of generating a scene from a single image is decomposed into three tractable sub-tasks:

**Stage 1: Panoramic Anchor View Generation**

- Hunyuan-Pano-DiT is used to expand a single input image into a 360° panorama.
- The panorama provides global coverage, alleviating the information deficiency.
- Single-image-to-panorama is a better-posed task compared to directly generating arbitrary novel views.

**Stage 2: Feed-Forward 3D Gaussian Geometric Scaffold**

The core innovation — reformulating panoramic depth estimation as multi-view stereo matching:

1. **Anchor view projection**: The 360° panorama is projected into 6 perspective cubemap views (FoV expanded to 95°, with 2.5° overlap between adjacent views), leveraging geometric priors from large-scale multi-view datasets.
2. **Bidirectional fusion module**: A cross-view consistency mechanism is integrated into the DPT head of pretrained VGGT:
    - Cube-to-Equirectangular (C2E): projects the 6 view feature maps into a unified equirectangular latent space.
    - After convolutional fusion, an Equirectangular-to-Cube (E2C) transform maps features back to each view.
    - Residual connections preserve view-specific details: $\mathbf{F}'_i = \mathbf{F}_i + \text{E2C}(\mathbf{F}_e)$
3. **Gaussian parameter prediction**: For each pixel, the model predicts Gaussian centers (depth unprojection + offset), opacity, covariance, and color.

Training uses MSE + LPIPS rendering loss and SILog depth loss, with training conducted on four datasets: Structured3D, Deep360, Matterport3D, and Stanford2D3D. **Reconstruction takes only 0.5 seconds on an H20 GPU.**

**Stage 3: Scaffold-Guided Novel View Synthesis**

- The 3D scaffold is used to render a coarse image $\mathbf{I}^{\text{render}}$ at the target viewpoint (containing artifacts and occlusion holes, yet retaining substantial structural information).
- **Dual-LoRA training strategy**: Built on the SEVA architecture, two independent LoRA modules handle high-quality anchor views and coarse rendered views respectively; features are fused via 3D attention, significantly outperforming naive concatenation.
- **Memory conditioning mechanism**: During inference, previously generated frames with camera poses nearest to the current target are retrieved from a memory bank as additional conditions, ensuring spatiotemporal consistency.

Training data is generated on DL3DV and RealEstate10K using MVSplat for sparse reconstruction, deliberately simulating artifacts from sparse inputs.

## Key Experimental Results

### Explorable 3D Scene Generation (Table 1)

Evaluated on an adapted WorldScore benchmark (40 scenes covering indoor/outdoor and real/stylized settings):

### Main Results

| Method | NIQE↓ | Q-Align↑ | CLIP-I↑ | TransErr↓ | RotErr↓ | CamMC↓ |
|---|---|---|---|---|---|---|
| DreamScene360 | 8.40 | 1.91 | 74.24 | - | - | - |
| WonderJourney | 4.97 | 3.02 | 77.92 | - | - | - |
| SEVA | 4.53 | 3.20 | 87.82 | 0.460 | 0.165 | 0.558 |
| VMem | 6.86 | 2.95 | 75.80 | 0.573 | 0.569 | 0.998 |
| **One2Scene** | **4.43** | **4.13** | **89.95** | **0.326** | **0.107** | **0.389** |

One2Scene achieves comprehensive superiority in visual quality (NIQE/Q-Align), semantic consistency (CLIP-I), and geometric consistency (CamMC reduced by 30% vs. SEVA and 61% vs. VMem).

### Panoramic Depth Estimation (Table 3, Matterport3D / Stanford2D3D)

### Ablation Study

| Method | MP3D AbsRel↓ | MP3D δ₁↑ | S2D3D AbsRel↓ | S2D3D δ₁↑ |
|---|---|---|---|---|
| HRDFuse | 0.0967 | 91.62 | 0.0935 | 91.40 |
| Depth Anywhere | 0.0850 | 91.70 | 0.1180 | 91.00 |
| Ours (Zero-shot) | 0.1070 | 88.97 | **0.0675** | **95.20** |
| Ours (Finetune) | **0.0391** | **98.09** | 0.0444 | 96.95 |

In the zero-shot setting, the proposed method already surpasses all baselines on Stanford2D3D; after fine-tuning, AbsRel improves by more than 50%.

### Reconstruction Efficiency

With 6 sparse input views, reconstruction takes 0.5 seconds on an H20 GPU — 5.6× faster than AnySplat (2.8 seconds). Replacing the scaffold network with AnySplat leads to a significant drop in scene generation quality (Q-Align drops from 4.13 to 3.61).

## Highlights & Insights

- **Elegant problem decomposition**: Deconstructing single-image-to-explorable-scene into three tractable sub-tasks, each with a clearly defined responsibility.
- **Novel reformulation of panoramic depth as multi-view stereo matching**: Leverages geometric priors learned from large-scale multi-view data, circumventing the bottleneck of scarce panoramic depth datasets.
- **Efficient feed-forward reconstruction**: A complete 3D scaffold is built in 0.5 seconds, balancing efficiency and accuracy.
- **Dual-LoRA effectively fuses heterogeneous conditions**: Separately handling high-quality anchors and coarse rendered views yields substantially better results than naive concatenation.
- **Generalization across scene types**: Consistently strong performance on indoor, outdoor, real, and stylized scenes.

## Limitations & Future Work

- Generated views may exhibit subtle inconsistencies, with residual local artifacts at large viewpoint changes.
- Generation quality depends on Hunyuan-Pano-DiT; notable artifacts in the panorama propagate to subsequent stages.
- The 3D scaffold renderer produces holes in heavily occluded regions; although the synthesis network can inpaint these, available information remains limited.
- The three-stage sequential pipeline leaves room for end-to-end optimization to further improve performance.
- The coverage of scene types in training data remains limited; the authors plan to construct a larger-scale dataset.

## Related Work & Insights

- **Sparse-view reconstruction**: Feed-forward Gaussian splatting models such as MVSplat, VGGT, and NoPosplat, which have limited extrapolation capability under extremely sparse views.
- **Video diffusion-based 3D generation**: ReconX, ViewCrafter, and VMem leverage geometric priors from DUSt3R/CUT3R, but accumulate errors over long sequences.
- **Panorama-based methods**: DreamScene360 and Pano2Room construct 3D from panoramas, but have limited exploration ranges or rely on strong indoor priors.
- **Pose-conditioned novel view synthesis**: SEVA and CAT3D use camera poses to guide generation but lack a persistent geometric representation.
- **Iterative navigation**: WonderJourney and Höllein et al. perform incremental exploration with inpainting, suffering from severe global semantic drift.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The three-stage decomposition, reformulation of panoramic depth as multi-view stereo matching, and Dual-LoRA design are all meaningful contributions.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Evaluation across three dimensions, multi-benchmark comparisons, ablations, depth estimation analysis, and efficiency profiling — highly comprehensive.
- ⭐⭐⭐⭐ **Value**: 0.5-second reconstruction, support for diverse indoor and outdoor scene types, with strong application potential.
- ⭐⭐⭐⭐ **Writing Quality**: Clear structure, rich illustrations, and logically sound problem decomposition.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer](color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)
- [\[ICLR 2026\] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [\[ICLR 2026\] RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering](radiogs_radiometric_gaussian_surfels.md)
- [\[ICLR 2026\] Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting](stylos_multi-view_3d_stylization_with_single-forward_gaussian_splatting.md)
- [\[ICLR 2026\] Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)

<!-- RELATED:END -->
