---
title: >-
  [Paper Note] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction
description: >-
  [AAAI 2026][3D Vision][Feed-Forward Gaussian Splatting] Proposes Splat-SAP, a feed-forward method that reconstructs scale-aware point maps from highly sparse binocular camera inputs, enabling high-quality free-viewpoint rendering of human-centered scenes via Gaussian Planes without any 3D supervision.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Feed-Forward Gaussian Splatting"
  - "Human-Centered Scene"
  - "Scale-Aware"
  - "Point Map Reconstruction"
  - "Free-Viewpoint Rendering"
date: 2026-05-08
content_hash: 04c2e3d99cf8823e
---

# Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction

**Conference**: AAAI 2026  
**arXiv**: [2511.22704](https://arxiv.org/abs/2511.22704)  
**Code**: [Project Page](https://yaourtb.github.io/Splat-SAP)  
**Area**: 3D Vision  
**Keywords**: Feed-Forward Gaussian Splatting, Human-Centered Scene, Scale-Aware, Point Map Reconstruction, Free-Viewpoint Rendering

## TL;DR

Proposes Splat-SAP, a feed-forward method that reconstructs scale-aware point maps from highly sparse binocular camera inputs, enabling high-quality free-viewpoint rendering of human-centered scenes via Gaussian Planes without any 3D supervision.

## Background & Motivation

Feed-forward free-viewpoint video synthesis is crucial for applications such as telecommunication and sports broadcasting. Existing feed-forward Gaussian splatting methods confront the following limitations:

**Challenge 1: Geometric Failure under Highly Sparse Inputs**
- Methods like MVSplat and MVSGaussian establish geometric priors based on multi-view stereo matching.
- These methods require large overlapping areas between input viewpoints.
- When the two input cameras are far apart (highly sparse), they fail to provide reasonable geometric priors.

**Challenge 2: Scale-Invariance Issue in DUSt3R-based Series**
- DUSt3R and MASt3R introduce point map representations, capable of predicting reasonable geometry even under highly sparse configurations.
- However, they normalize the point maps to a scale-invariant canonical space.
- During sequential frame inference, scale normalization across different frames causes severe jitter in reconstruction results.
- Depth variations caused by human movement trigger massive jumps in the canonical space.

**Challenge 3: Difficulty in Acquiring 3D Supervision Data**
- Training scale-aware geometric foundation models typically requires massive 3D data.
- Collecting 3D geometric data is time-consuming and tedious.

The core contribution of Splat-SAP is learning scale-aware point map transformations in a self-supervised manner, mapping point maps from canonical space to physical space without any 3D geometric supervision.

## Method

### Overall Architecture

A two-stage coarse-to-fine pipeline:
- **Stage 1 (2D Coarse Stage)**: Starting from point maps initialized by MASt3R, an affine transformation (scaling + translation) is learned to transform them from canonical space to physical space.
- **Stage 2 (3D Fine Stage)**: Projects the transformed point maps onto the target viewpoint, performs stereo refinement via a 3D cost volume, and constructs a Gaussian Plane for high-quality rendering.

### Key Designs

#### 1. **Scale-Aware Geometry Reconstruction**: Self-Supervised Affine Transformation Learning

**Point Map Initialization**: MASt3R is used to predict the point maps $X^i$ (in canonical space) of two views from low-resolution (512×288) binocular inputs.

**Scale Factor Learning**:
- Embeds camera intrinsic focal length $f$ and binocular distance $d$ using positional encoding.
- Combines global information from self-attention and cross-attention of ViT features.
- Predicts a 3-dimensional scale factor $S$ through an MLP (handling distortions in the raw point maps).

$$S = MLP(f_s, f_c, e), \quad e = PE(f, d)$$

**Pixel-wise Translation Learning**:
- Scaling alone cannot eliminate pixel-wise offsets between the two point maps.
- Inspired by MVS view-consistency checks, features from one view are warped to another view to obtain correspondences.
- Iteratively computes pixel-wise translation using a GRU:

$$T^i = GRU(F^i, F^{j \rightarrow i}, SX^i)$$

Finally, the point positions in physical space: $X_t^i = SX^i + T^i$

**Design Motivation**: Scaling (via intrinsic embeddings) + translation (via extrinsic projections) together formulate exactly the affine transformation from canonical space to physical space.

#### 2. **Gaussian Plane Rendering**: Efficient and Complete Rendering

**3D Refinement**:
- Projects the transformed point sets onto the target viewpoint via $\alpha$-blending to obtain an initial depth map $\mathcal{D}^k$.
- Samples multiple depth candidates along camera rays near the initial depth.
- Warps source view features to the target view to construct a 3D cost volume.
- Regresses the refined depth $\bar{d} = \Sigma_n w_n d_n$ via 3D convolution and depth probability volume.

**Gaussian Plane Construction**:
- Anchors Gaussian primitives on the target viewpoint plane instead of using point maps from two source views as Gaussian positions.
- This significantly reduces Gaussian redundancy in overlapping areas.
- Color Initialization: Obtains weighted colors from source views via warping.
$$C^k = \Sigma_i w_c^i C^{i \rightarrow k}$$
- Other attributes (rotation, scaling, opacity) are predicted from aggregated features via convolutional heads.
- Color Residual Learning: $\mathcal{P}_c = \alpha C + (1-\alpha) \Delta C$

Ultimately rendered at 1024×576 resolution, splatting outputs a high-resolution 1280×720 image.

#### 3. **Self-Supervised Training Strategy**: Without 3D Geometric Supervision

**Stage 1 Loss**:
$$\mathcal{L}_{stage1} = \mathcal{L}_{render} + \gamma \mathcal{L}_{CD}$$

Where $\mathcal{L}_{CD}$ is the Chamfer Distance regularization between two 6D point sets (XYZ + RGB), encouraging both point maps to converge to a consistent geometry. During training, the weights of MASt3R are frozen.

**Stage 2 Loss**:
$$\mathcal{L}_{stage2} = \lambda_1 \mathcal{L}_{render}(\hat{I}_f, I_f^{gt}) + \lambda_2 \mathcal{L}_{render}(\hat{I}_h, I_h^{gt})$$

Both stages do not require 3D geometric supervision and are trained entirely using rendering losses.

### Loss & Training

- Rendering Loss: $\mathcal{L}_{render} = 0.8 \mathcal{L}_1 + 0.2 \mathcal{L}_{ssim}$
- Stage 1: Trains the affine learning module with 100k iterations (using all training data).
- Stage 2: Trains the rendering module with 60k iterations for each camera type.
- Can be trained on a single RTX 3090 (24GB).

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

PSNR significantly leads on Camera and GoPro datasets (+2.9 and +1.7 dB).

### Geometry Reconstruction Quality

| Method | Pred→GT CD↓ | GT→Pred CD↓ | Description |
|------|------------|------------|------|
| DUSt3R | 0.305 | 0.160 | Massive foreground-background misalignment |
| VGGT | 0.288 | 0.129 | Difficulty in two-view alignment |
| Pow3R | 0.281 | 0.134 | Insufficient even with camera calibration |
| MASt3R | 0.212 | 0.069 | Baseline geometry |
| Prompt-DA | 0.205 | 0.063 | Add uncertainty |
| Ours w/o Translation | 0.191 | 0.046 | Scaling only |
| **Ours Full** | **0.172** | **0.027** | Scaling + translation |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|-------|-------|--------|------|
| Stage 1 Rendering | 24.844 | 0.794 | 0.296 | Auxiliary rendering in the coarse stage only |
| Stage 2 Initial Color | 27.308 | 0.856 | 0.169 | Warped color after geometric refinement |
| **Stage 2 Final Splatting** | **28.703** | **0.889** | **0.169** | Full pipeline |

### Key Findings

1. Pixel-wise translation learning is crucial for eliminating point map alignment errors (Pred→GT CD drops from 0.191 to 0.172).
2. The 3D refinement module effectively rectifies holes and artifacts from Stage 1.
3. Color residual learning and splatting mechanisms further improve rendering quality.
4. The method remains competitive on the Mobile dataset (alternating zoom scenes).
5. Entirely self-supervised training requires no 3D ground truth, yet outperforms DUSt3R which requires 3D supervision.

## Highlights & Insights

1. **Self-Supervised Scale Recovery**: Ingeniously utilizes camera intrinsic embedding and extrinsic projections to learn the affine transformation from canonical to physical space without requiring 3D supervision.
2. **Gaussian Plane Design**: Anchors Gaussians on the target viewpoint plane, eliminating redundancy from the source-view point maps.
3. **Coarse-to-Fine Geometry Strategy**: 2D affine coarse alignment followed by 3D cost volume refinement, progressively enhancing geometric accuracy.
4. **Chamfer Distance Regularization**: Computes Chamfer Distance in a 6-dimensional space (position + color) to constrain both geometry and appearance consistency.
5. **Practical Multi-Camera Support**: A single generalized affine module is used, while only one rendering module needs to be trained per camera type.

## Limitations & Future Work

1. **Foreground-Background Boundary Floaters**: MASt3R may predict floaters at human boundaries; since these regions are observed by only one view, the refinement module cannot correct them.
2. Currently only supports binocular inputs, leaving scenarios with more than two views unexplored.
3. Heavy reliance on the pre-trained MASt3R model.
4. The narrow performance gap compared to MVSplat on the Mobile dataset indicates room for improvement in camera-zooming scenarios.
5. Requires camera calibration information, limiting applications in certain uncalibrated setups.

## Related Work & Insights

- **DUSt3R/MASt3R**: Pioneering works in point map representation, upon which Splat-SAP resolves the scale ambiguity.
- **GPS-Gaussian/GPS-Gaussian+**: Precursor works for binocular Gaussians, but requiring dense overlaps.
- **NoPoSplat/Splat3R**: Utilizes point maps for static scene rendering but lacks stereo constraints.
- **ENeRF**: A feed-forward method using cost volumes + NeRF, from which Splat-SAP draws inspiration for depth probability regression.
- **Insight**: Combining point maps with stereo matching might be the optimal paradigm for sparse-view human-centered rendering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel self-supervised scale recovery and Gaussian Plane design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple camera types, with dual evaluations on both rendering and geometry.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear two-stage structure, although some details require referring to the supplementary material.
- **Value**: ⭐⭐⭐⭐⭐ — Direct value for real-time applications such as telecommunication and sports broadcasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniSH: Unifying Scene and Human Reconstruction in a Feed-Forward Pass](../../CVPR2026/3d_vision/unish_unifying_scene_and_human_reconstruction_in_a_feed-forward_pass.md)
- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](../../ICLR2026/3d_vision/splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)
- [\[CVPR 2026\] Z-Order Transformer for Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/z-order_transformer_for_feed-forward_gaussian_splatting.md)
- [\[CVPR 2026\] AeroGS: Scale-Aware Gaussian Splatting for Pose-Free Dynamic UAV Scene Reconstruction](../../CVPR2026/3d_vision/aerogs_scale-aware_gaussian_splatting_for_pose-free_dynamic_uav_scene_reconstruc.md)
- [\[ICLR 2026\] Signal Structure-Aware Gaussian Splatting for Large-Scale Scene Reconstruction](../../ICLR2026/3d_vision/signal_structure-aware_gaussian_splatting_for_large-scale_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
