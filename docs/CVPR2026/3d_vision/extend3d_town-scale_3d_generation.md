---
title: >-
  [Paper Note] Extend3D: Town-Scale 3D Generation
description: >-
  [CVPR 2026][3D Vision][3D scene generation] This paper proposes Extend3D, a training-free 3D scene generation pipeline that extends the voxel latent space of a pretrained object-level 3D generative model (Trellis) and introduces overlapping patch joint denoising, under-noising SDEdit initialization, and 3D-aware optimization to generate town-scale large-scale 3D scenes from a single image, surpassing existing methods in both human preference evaluations and quantitative metrics.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D scene generation
  - large-scale scenes
  - training-free
  - extended latent space
  - voxel generation
date: 2026-05-08
content_hash: f49960d9e1c26d57
---

# Extend3D: Town-Scale 3D Generation

**Conference**: CVPR 2026
**arXiv**: [2603.29387](https://arxiv.org/abs/2603.29387)
**Code**: None (project page available)
**Area**: 3D Vision
**Keywords**: 3D scene generation, large-scale scenes, training-free, extended latent space, voxel generation

## TL;DR

This paper proposes Extend3D, a training-free 3D scene generation pipeline that extends the voxel latent space of a pretrained object-level 3D generative model (Trellis) and introduces overlapping patch joint denoising, under-noising SDEdit initialization, and 3D-aware optimization to generate town-scale large-scale 3D scenes from a single image, surpassing existing methods in both human preference evaluations and quantitative metrics.

## Background & Motivation

1. **State of the Field**: 3D generative models (e.g., Trellis, Hunyuan3D) can already produce high-quality 3D objects, but are trained on object-level data and operate within fixed-size latent spaces for representing 3D content.
2. **Limitations of Prior Work**:
   - Fixed latent space size limits output detail; the larger the scene, the blurrier the result (analogous to low-resolution images);
   - 3D scene datasets are scarce, restricting data-driven scene generation methods to a limited set of categories;
   - Outpainting-based methods (e.g., SynCity, 3DTown) generate scenes block by block, resulting in inter-block inconsistencies and visible seams.
3. **Root Cause**: The latent space of object-level models is insufficient to represent the fine-grained details of large-scale scenes, yet the lack of scene-level training data makes directly training a scene-level model infeasible.
4. **Paper Goals**: How to leverage pretrained object-level 3D generative models to achieve high-fidelity large-scale 3D scene generation?
5. **Starting Point**: Drawing inspiration from MultiDiffusion for high-resolution 2D image generation, the authors extend the 3D latent space along the x/y directions and employ overlapping patch joint generation, while incorporating structural priors and optimization to address 3D-specific issues (e.g., ground plane disappearance, incorrect object rotation).
6. **Core Idea**: Extend the latent space of an object-level 3D model horizontally, and achieve town-scale 3D scene generation through overlapping patch joint denoising, point cloud prior initialization, and 3D-aware loss optimization.

## Method

### Overall Architecture

Extend3D follows a two-stage pipeline consistent with Trellis: sparse structure generation followed by structured latent variable (SLat) generation. Both stages operate on the extended latent space. Given a single scene image, a monocular depth estimator (MoGe-2) first extracts a point cloud as a structural prior, after which SDEdit initialization and optimized denoising are applied in the extended latent space to produce a large-scale 3D scene.

### Key Designs

1. **Overlapping Patch-wise Flow**:
   - **Function**: Enables multiple patches within the extended latent space to be generated simultaneously with mutual influence.
   - **Mechanism**: The extended latent space $\mathbf{Z}_t \in \mathbb{R}^{aN \times bN \times N}$ (where $a, b$ are expansion factors) is partitioned into overlapping patches via a sliding window. Each patch independently computes a vector field, which is then aggregated by averaging over overlapping regions. Image conditions are cropped and aligned accordingly. The key formula is: $\bm{v}(\mathbf{Z}_t, \mathcal{I}, t) = \sum_{i,j} \phi_{i,j}^{-1}(\bm{v}_{i,j}) \oslash \sum_{i,j} \mathbf{1}_{\mathbb{W}_{i,j}}$
   - **Design Motivation**: Unlike SynCity and similar methods that generate patches sequentially, overlapping patches allow adjacent regions to mutually correct each other. The small stride of the sliding window captures local structural variation, while central objects can leverage the strengths of the object-level model. Ablations show that $d=2$ causes local structural distortions, which are resolved at $d=4$.

2. **Under-noising SDEdit Initialization**:
   - **Function**: Initializes scene structure from a monocular depth point cloud and inpaints occluded regions.
   - **Mechanism**: The point cloud is voxelized and encoded into a latent variable $\mathbf{Z}_0^{(g)}$. Rather than using standard SDEdit (where $t_{\text{noise}} = t_{\text{start}}$), the method sets $t_{\text{start}} > t_{\text{noise}}$, i.e., the degree of denoising exceeds that of noising. This causes the model to treat missing or occluded regions as additional noise and complete them. The scene is progressively refined by iteratively applying $O_n = \text{SDEdit}(O_{n-1})$.
   - **Design Motivation**: Standard SDEdit faces an inherent trade-off: a small $t_{\text{start}}$ fails to fill in gaps, while a large $t_{\text{start}}$ destroys existing structure. Under-noising breaks this trade-off, analogous to using high-frequency noise to enhance detail in super-resolution tasks.

3. **3D-Aware Optimization (Optimize with Prior)**:
   - **Function**: Optimizes the vector field at each denoising step to prevent the object-level model's denoising trajectory from drifting toward object-centric dynamics.
   - **Mechanism**: Separate optimization losses are designed for each stage. For the sparse structure stage: $\mathcal{L}_{\text{SS}} = -\frac{1}{|\mathbb{P}|}\sum_{\bm{p}\in\mathbb{P}} \log \sigma((\mathcal{D}(\mathbf{Z}_t^{\text{SS}} - t\cdot\hat{\bm{v}}_t))_{\bm{p}})$, which constrains voxels at point cloud positions from vanishing. For the SLat stage: $\mathcal{L}_{\text{SLat}} = \text{LPIPS}(\hat{\mathcal{I}}, \mathcal{I}) - \text{SSIM}(\hat{\mathcal{I}}, \mathcal{I})$, which renders the 3D output to the input viewpoint via differentiable rendering and compares it against the original image.
   - **Design Motivation**: During denoising, the object-level model tends to bias sub-scene outputs toward object-like structures (e.g., ground planes disappear, objects rotate arbitrarily). Optimization ensures that the denoising trajectory remains consistent with scene-level dynamics while eliminating inter-patch seams.

### Loss & Training

No training is required; all components are applied at inference time. Both optimization losses use the Adam optimizer to optimize the vector field $\hat{\bm{v}}_t$ at each denoising step. Dilated sampling is employed during the sparse structure stage to ensure global consistency.

## Key Experimental Results

### Main Results (Quantitative, 100 input images)

| Method | LPIPS↓ | SSIM↑ | PSNR↑ | CD↓ | F-score↑ |
|--------|--------|-------|-------|-----|----------|
| Trellis | 0.650 | 0.239 | 10.0 | 0.0315 | 0.442 |
| Hunyuan3D | 0.683 | 0.255 | 10.4 | 0.0192 | 0.567 |
| EvoScene | 0.482 | 0.310 | 13.2 | 0.0188 | 0.498 |
| **Ours w/o SLat optim** | **0.400** | **0.333** | **13.8** | **0.0078** | **0.708** |
| **Ours (full)** | **0.240** | **0.611** | **20.4** | **0.0086** | **0.694** |

### Ablation Study ($a=b=2$)

| Configuration | LPIPS↓ | SSIM↑ | PSNR↑ | CD↓ | F-score↑ |
|---------------|--------|-------|-------|-----|----------|
| Patch-wise flow only | 0.606 | 0.209 | 9.63 | 0.0348 | 0.261 |
| + Initialization | 0.425 | 0.312 | 13.0 | 0.0083 | 0.693 |
| + SS optimization | 0.400 | 0.333 | 13.8 | 0.0078 | 0.708 |
| + SLat optimization (full) | 0.240 | 0.611 | 20.4 | 0.0086 | 0.694 |

### Key Findings

- In human preference evaluations, Extend3D consistently outperforms all baselines across four dimensions—geometry, fidelity, appearance, and completeness (vs. Trellis: 50–67%; vs. Hunyuan3D: 73–76%; vs. EvoScene: 87%).
- Initialization is essential: without it ($t_{\text{start}}=1$), the structure collapses entirely.
- Under-noising naturally inpaints occluded regions without disrupting existing structure, outperforming standard SDEdit.
- Larger division factors $d$ yield better results ($d=8$ is optimal), at the cost of increased computational overhead.
- SLat optimization substantially improves texture quality (LPIPS: 0.400 → 0.240; PSNR: 13.8 → 20.4).

## Highlights & Insights

- **The Under-noising Concept**: The observation that setting $t_{\text{start}} > t_{\text{noise}}$ causes the model to treat structural incompleteness as noise and complete it is simple yet profound. This insight generalizes to other generative tasks requiring joint editing and completion.
- **Scene-level Generation Without Scene-level Data**: The method entirely reuses knowledge from an object-level model; by simply extending the latent space and incorporating prior guidance, it generates town-scale scenes without relying on scarce 3D scene datasets.
- **Overlapping Patch Joint Denoising vs. Sequential Outpainting**: Simultaneously generating all patches allows adjacent regions to mutually correct each other, yielding greater consistency than the sequential approach of SynCity.

## Limitations & Future Work

- The expansion factors $a, b$ and division factor $d$ require manual tuning, and computational cost scales accordingly.
- The method depends on the quality of the monocular depth estimator; inaccurate MoGe-2 estimates propagate errors downstream.
- Physical plausibility of the generated scenes (e.g., gravity, occlusion relationships) is not explicitly modeled.
- Performance on highly elongated scenes (e.g., long corridors) remains unverified; current demonstrations focus primarily on square or rectangular layouts.
- SLat optimization slightly increases CD (0.0078 → 0.0086), potentially introducing minor geometric bias in certain structures.

## Related Work & Insights

- **vs. SynCity**: Sequential outpainting leads to inter-block inconsistencies and visible seams; Extend3D avoids this by generating all patches simultaneously.
- **vs. 3DTown / EvoScene**: These methods also use point cloud initialization but rely on RePaint for per-patch completion and cannot address the systematic biases of object-level models (e.g., ground plane disappearance).
- **vs. MultiDiffusion**: The core idea of extending latent space for high-resolution generation is adapted from 2D to 3D, but 3D-specific challenges—such as object-centricity and spatial alignment—require additional priors and optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Under-noising and 3D-aware optimization are valuable contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation covering human preference, quantitative metrics, and ablations, though the dataset size is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Methodology is clearly presented with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Achieving town-scale 3D scene generation without scene-level training data carries high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](text-image_conditioned_3d_generation.md)
- [\[CVPR 2026\] S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds](s2am3d_scale-controllable_part_segmentation_of_3d_point_cloud.md)
- [\[CVPR 2026\] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)
- [\[CVPR 2026\] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)

<!-- RELATED:END -->
