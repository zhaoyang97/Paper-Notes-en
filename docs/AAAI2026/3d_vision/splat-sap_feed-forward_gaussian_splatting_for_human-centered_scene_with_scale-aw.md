---
title: >-
  [Paper Note] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction
description: >-
  [AAAI 2026][3D Vision][feed-forward Gaussian splatting] This paper proposes Splat-SAP, a feed-forward method that reconstructs scale-aware point maps from wide-baseline stereo camera inputs and renders free-viewpoint vid…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "feed-forward Gaussian splatting"
  - "human-centered scene"
  - "scale-awareness"
  - "point map reconstruction"
  - "free-viewpoint rendering"
date: 2026-05-08
content_hash: 0a377141176be20c
---

# Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2511.22704](https://arxiv.org/abs/2511.22704)  
**Code**: [Project Page](https://yaourtb.github.io/Splat-SAP)  
**Area**: 3D Vision
**Keywords**: feed-forward Gaussian splatting, human-centered scene, scale-awareness, point map reconstruction, free-viewpoint rendering

## TL;DR

This paper proposes Splat-SAP, a feed-forward method that reconstructs scale-aware point maps from wide-baseline stereo camera inputs and renders free-viewpoint video of human-centered scenes via a Gaussian Plane, requiring neither per-scene optimization nor 3D geometric supervision.

## Background & Motivation

Feed-forward free-viewpoint video synthesis is critical for applications such as telepresence and sports broadcasting. Existing feed-forward Gaussian splatting methods suffer from the following challenges:

**Challenge 1: Geometric failure under large-baseline inputs**
- Methods such as MVSplat and MVSGaussian rely on multi-view stereo matching to establish geometric priors.
- These methods require substantial overlap between input views.
- When the two input cameras are widely separated (large baseline), reliable geometric priors cannot be obtained.

**Challenge 2: Scale ambiguity in DUSt3R-based methods**
- DUSt3R/MASt3R introduce point map representations capable of predicting reasonable geometry under large baselines.
- However, they normalize point maps into a scale-invariant canonical space.
- During per-frame inference, inconsistent scale normalization causes severe temporal jitter in reconstructions.
- Depth variations induced by human motion produce large discontinuities in canonical space.

**Challenge 3: Difficulty in acquiring 3D supervision data**
- Training scale-aware geometric foundation models typically requires large quantities of 3D data.
- Acquiring 3D geometric data is time-consuming and cumbersome.

The core contribution of Splat-SAP is to learn a scale-aware point map transformation in a self-supervised manner, mapping canonical-space point maps to metric space without any 3D geometric supervision.

## Method

### Overall Architecture

A coarse-to-fine two-stage pipeline:
- **Stage 1 (2D coarse stage)**: Starting from MASt3R-initialized point maps, learns an affine transformation (scaling + translation) to map them from canonical space to metric space.
- **Stage 2 (3D fine stage)**: Projects the transformed point maps to the target viewpoint, performs stereo refinement via a 3D cost volume, constructs a Gaussian Plane, and renders high-quality novel views.

### Key Designs

#### 1. **Scale-Aware Geometry Reconstruction**: Self-supervised affine transformation learning

**Point map initialization**: MASt3R is used to predict point maps $X^i$ (in canonical space) from low-resolution (512×288) stereo inputs.

**Scale factor learning**:
- Camera intrinsic focal length $f$ and baseline distance $d$ are embedded via positional encoding.
- Global information is extracted from ViT features via self-attention and cross-attention.
- A 3D scaling factor $S$ is predicted by an MLP to handle distortions in the original point maps.

$$S = MLP(f_s, f_c, e), \quad e = PE(f, d)$$

**Per-pixel translation learning**:
- Scaling alone cannot eliminate per-pixel offsets between the two point maps.
- Inspired by view consistency checks in MVS, features from one view are projected into the other to obtain correspondences.
- A GRU iteratively estimates per-pixel translation:

$$T^i = GRU(F^i, F^{j \rightarrow i}, SX^i)$$

The final metric-space point positions are: $X_t^i = SX^i + T^i$

**Design motivation**: Scaling (via intrinsic embedding) combined with translation (via extrinsic projection) constitutes an affine transformation from canonical space to metric space.

#### 2. **Gaussian Plane Rendering**: Efficient and complete rendering

**3D refinement**:
- The transformed point set is projected to the target viewpoint via α-blending to obtain an initial depth map $\mathcal{D}^k$.
- Multiple depth candidates are sampled along the camera ray near the initial depth.
- Source-view features are warped to the target view to construct a 3D cost volume.
- Refined depth is obtained via 3D convolution and depth probability regression: $\bar{d} = \Sigma_n w_n d_n$.

**Gaussian Plane construction**:
- Gaussian primitives are anchored on the target-view image plane rather than using source-view point maps as Gaussian positions, substantially reducing redundancy in overlapping regions.
- Color initialization: weighted colors warped from source views:
$$C^k = \Sigma_i w_c^i C^{i \rightarrow k}$$
- Remaining attributes (rotation, scale, opacity) are predicted by convolutional heads from aggregated features.
- Residual color learning: $\mathcal{P}_c = \alpha C + (1-\alpha) \Delta C$

Final rendering is performed at 1024×576 resolution and splatted to 1280×720 output.

#### 3. **Self-supervised Training Strategy**: No 3D geometric supervision required

**Stage 1 loss**:
$$\mathcal{L}_{stage1} = \mathcal{L}_{render} + \gamma \mathcal{L}_{CD}$$

where $\mathcal{L}_{CD}$ is a Chamfer distance regularization between the two 6D point sets (XYZ+RGB), encouraging both point maps to converge to a consistent geometry. MASt3R weights are frozen during training.

**Stage 2 loss**:
$$\mathcal{L}_{stage2} = \lambda_1 \mathcal{L}_{render}(\hat{I}_f, I_f^{gt}) + \lambda_2 \mathcal{L}_{render}(\hat{I}_h, I_h^{gt})$$

Both stages require no 3D geometric supervision and are trained entirely with rendering losses.

### Loss & Training

- Rendering loss: $\mathcal{L}_{render} = 0.8 \mathcal{L}_1 + 0.2 \mathcal{L}_{ssim}$
- Stage 1: 100k iterations for the affine learning module (using all training data)
- Stage 2: 60k iterations per camera type for the rendering module
- Trainable on a single RTX 3090 (24 GB)

## Key Experimental Results

### Main Results (Rendering Quality)

| Method | Camera PSNR↑ | Camera SSIM↑ | GoPro PSNR↑ | GoPro SSIM↑ | Mobile PSNR↑ | Mobile SSIM↑ |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| NoPoSplat | 25.035 | 0.866 | 26.128 | 0.889 | 21.594 | 0.591 |
| 4D-GS | 27.814 | 0.906 | 27.244 | 0.907 | 25.655 | 0.825 |
| MVSplat | 27.899 | 0.902 | 29.942 | 0.934 | 26.545 | 0.805 |
| MVSGaussian | 29.326 | 0.957 | 27.413 | 0.926 | 19.927 | 0.683 |
| ENeRF | 28.272 | 0.943 | 29.906 | 0.943 | 20.579 | 0.640 |
| **Splat-SAP** | **32.220** | **0.957** | **31.640** | **0.955** | **25.721** | **0.827** |

Splat-SAP achieves substantial PSNR improvements on Camera and GoPro datasets (+2.9 and +1.7 dB, respectively).

### Geometry Reconstruction Quality

| Method | Pred→GT CD↓ | GT→Pred CD↓ | Notes |
|------|------------|------------|------|
| DUSt3R | 0.305 | 0.160 | Significant foreground–background misalignment |
| VGGT | 0.288 | 0.129 | Difficulty in two-view alignment |
| Pow3R | 0.281 | 0.134 | Insufficient even with camera calibration |
| MASt3R | 0.212 | 0.069 | Baseline geometry |
| Prompt-DA | 0.205 | 0.063 | Adds uncertainty estimation |
| Ours w/o Translation | 0.191 | 0.046 | Scaling only |
| **Ours Full** | **0.172** | **0.027** | Scaling + translation |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Notes |
|------|-------|-------|--------|------|
| Stage 1 rendering | 24.844 | 0.794 | 0.296 | Auxiliary layer rendering at coarse stage only |
| Stage 2 initial color | 27.308 | 0.856 | 0.169 | Warped color after geometric refinement |
| **Stage 2 full splatting** | **28.703** | **0.889** | **0.169** | Complete pipeline |

### Key Findings

1. Per-pixel translation learning is critical for eliminating point map alignment errors (Pred→GT CD reduced from 0.191 to 0.172).
2. The 3D refinement module corrects holes and artifacts from Stage 1.
3. Residual color learning and the splatting mechanism further improve rendering quality.
4. The method remains competitive on Mobile data (alternating zoom scenarios).
5. Fully self-supervised training without 3D ground truth still outperforms methods that rely on 3D supervision, such as DUSt3R.

## Highlights & Insights

1. **Self-supervised scale recovery**: Camera intrinsic embedding and extrinsic projection are elegantly exploited to learn the canonical-to-metric affine transformation without any 3D supervision.
2. **Gaussian Plane design**: Anchoring Gaussians on the target-view image plane avoids redundancy from dual source-view point maps.
3. **Coarse-to-fine geometric strategy**: 2D affine coarse alignment followed by 3D cost-volume refinement progressively improves geometric accuracy.
4. **Chamfer distance regularization**: CD computed in 6D space (position + color) simultaneously constrains geometric and appearance consistency.
5. **Practical multi-camera support**: A single affine module is shared across camera types; only one rendering module per camera type needs to be trained.

## Limitations & Future Work

1. **Foreground–background boundary floaters**: MASt3R may produce floaters at human silhouette boundaries, which the refinement module cannot correct since these regions are observed by only one view.
2. Only stereo input is supported; scenarios with more than two views are not explored.
3. The method has a strong dependency on the pretrained MASt3R model.
4. The smaller performance gap relative to MVSplat on Mobile data indicates room for improvement in zoom-variant scenarios.
5. Camera calibration information is required, limiting applicability in uncalibrated settings.

## Related Work & Insights

- **DUSt3R/MASt3R**: Pioneering works on point map representations; Splat-SAP builds upon them to resolve the scale ambiguity.
- **GPS-Gaussian/GPS-Gaussian+**: Precursor feed-forward stereo Gaussian methods, but require dense view overlap.
- **NoPoSplat/Splat3R**: Leverage point maps for static scene rendering but lack stereo constraints.
- **ENeRF**: A feed-forward method combining cost volumes with NeRF; Splat-SAP adopts its depth probability regression strategy.
- **Insight**: The combination of point maps and stereo matching appears to be a promising paradigm for sparse-view human rendering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Self-supervised scale recovery and the Gaussian Plane design are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple camera types with dual evaluation of rendering and geometry.
- **Writing Quality**: ⭐⭐⭐⭐ — The two-stage structure is clearly presented, though some details require consulting the supplementary material.
- **Value**: ⭐⭐⭐⭐⭐ — Direct practical value for real-time applications such as telepresence and sports broadcasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](../../ICLR2026/3d_vision/splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)
- [\[CVPR 2026\] AeroGS: Scale-Aware Gaussian Splatting for Pose-Free Dynamic UAV Scene Reconstruction](../../CVPR2026/3d_vision/aerogs_scale-aware_gaussian_splatting_for_pose-free_dynamic_uav_scene_reconstruc.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[CVPR 2026\] AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend](../../CVPR2026/3d_vision/amb3r_accurate_feed-forward_metric-scale_3d_reconstruction_with_backend.md)
- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](../../CVPR2026/3d_vision/vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)

</div>

<!-- RELATED:END -->
