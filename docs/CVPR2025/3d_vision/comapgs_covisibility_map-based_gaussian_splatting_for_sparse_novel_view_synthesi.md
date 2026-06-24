---
title: >-
  [Paper Note] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis
description: >-
  [CVPR 2025][3D Vision][Sparse View Synthesis] This paper proposes CoMapGS, which utilizes pixel-level covisibility maps to guide initial point cloud enhancement and adaptively weighted supervision in sparse-view 3DGS, representing the first attempt to explicitly focus on and recover high-uncertainty single-view regions.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sparse View Synthesis"
  - "3D Gaussian Splatting"
  - "Covisibility Map"
  - "Point Cloud Enhancement"
  - "Uncertainty-aware"
date: 2026-05-08
content_hash: 6d5c4863c56e2342
---

# CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2503.20998](https://arxiv.org/abs/2503.20998)  
**Code**: [youngkyoonjang.github.io/projects/comapgs](https://youngkyoonjang.github.io/projects/comapgs)  
**Area**: 3D Vision  
**Keywords**: Sparse View Synthesis, 3D Gaussian Splatting, Covisibility Map, Point Cloud Enhancement, Uncertainty-aware

## TL;DR

This paper proposes CoMapGS, which utilizes pixel-level covisibility maps to guide initial point cloud enhancement and adaptively weighted supervision in sparse-view 3DGS, representing the first attempt to explicitly focus on and recover high-uncertainty single-view regions.

## Background & Motivation

Sparse novel view synthesis faces three core challenges:

1. **Imbalanced Regional Supervision**: Highly covisible regions are over-optimized due to multi-view availability, whereas single-view regions (visible under only one training view) are neglected due to the lack of multi-view constraints.
2. **Sparse Point Cloud Initialization**: COLMAP keypoint matching under a small number of training images yields extremely sparse point clouds, lacking geometric details.
3. **High-Uncertainty Regions**: Single-view regions lack multi-view geometric constraints; existing methods either ignore or penalize these regions instead of leveraging their information.

Existing methods (e.g., FSGS, CoR-GS) primarily focus on optimizing highly covisible regions, lacking effective recovery strategies for single-view regions. The core innovation of this paper is: **utilizing covisibility maps to quantify the uncertainty level of each pixel, and performing differentiated point cloud enhancement and supervisory weighting accordingly**.

## Method

### Overall Architecture

Based on sparse-view 3DGS methods like CoR-GS, CoMapGS introduces three key steps: (1) generating pixel-level covisibility maps using MASt3R dense correspondence predictions; (2) enhancing the initial point cloud in both low- and high-uncertainty regions through dense correspondence triangulation and monocular depth estimation alignment; (3) training a proximity MLP classifier and combining it with the covisibility map for adaptively weighted proximity loss supervision.

### Key Designs

1. **Covisibility Map Generation and Initial Point Cloud Enhancement**:
    - Function: Quantify the multi-view covisibility frequency of each pixel and enhance the sparse point cloud regionally based on this.
    - Mechanism: Predict dense correspondences for each training image pair using MASt3R, and accumulate the match counts $M_i(x,y)$ for each pixel to obtain the covisibility map. Low-uncertainty regions ($M_i \geq 1$) supplement the COLMAP sparse point cloud $P_C$ by triangulating dense correspondence points $P_T$, keeping new points whose distance to $P_C$ exceeds a threshold $\epsilon$. High-uncertainty regions (single-view regions where $M_i = 0$) generate $P_d^{high}$ via monocular depth estimation back-projection, and an anisotropic scaling transformation $f_{scale}$ is learned to align them to the coordinate system of the triangulated point cloud.
    - Design Motivation: COLMAP keypoint matching yields extremely sparse points with few images, whereas dense correspondences from MASt3R can significantly increase point cloud density; although monocular depth has arbitrary scale, it can be aligned by learning from known regions.

2. **Covisibility Map-Weighted Proximity Loss**:
    - Function: Adaptively adjust supervision strength based on regional uncertainty levels.
    - Mechanism: Train a 3-layer MLP classifier $f_p$ to distinguish between the enhanced point cloud $P_{final}$ (positive samples) and randomly shifted points (negative samples), outputting a proximity score $s \in [0,1]$. For Gaussians inside the camera frustum, the weight $w_{in} = 1/(M_i(\pi(g,\mathbf{H}_i)) + 1)$ is inversely proportional to the covisibility frequency—maximizing single-view region weight (=1) while reducing it for highly covisible regions. For Gaussians outside the camera frustum, a linear decay weight $w_{out}$ based on the scene's average covisibility score $S$ is enabled when $S > 0.7$.
    - Design Motivation: Highly covisible regions are already sufficiently supervised by standard reconstruction losses, so the proximity loss should focus more on under-constrained single-view regions; Gaussians outside the frustum in highly covisible scenes should also be moderately constrained.

3. **Regional Strategy for Enhanced Point Cloud**:
    - Function: Divide the enhanced point clouds into regions of different confidence levels based on covisibility for separate processing.
    - Mechanism: The final point cloud is $P_{final} = P_u^{low} \cup P_s^{high}$, where $P_u^{low}$ comes from triangulation (high confidence) and $P_s^{high}$ comes from aligned monocular depth back-projection (low confidence). The classifier and weighted supervision naturally distinguish the reliability of points from different sources.
    - Design Motivation: Triangulated points are more reliable due to multi-view verification, while depth back-projected points, although less accurate, are crucial for filling empty regions.

### Loss & Training

A proximity loss term is added to the total loss:

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1(I, I^*) + \lambda\mathcal{L}_{D\text{-}SSIM}(I, I^*) + \mathcal{L}_p$$

where the proximity loss $\mathcal{L}_p = \frac{1}{|G|}\sum_{g \in G}(\chi(g)w_{in} + (1-\chi(g))w_{out}) \cdot (1-s)$, and $\chi(g)$ indicates whether the Gaussian is inside the frustum. This method can be seamlessly integrated into existing methods such as FSGS and CoR-GS.

## Key Experimental Results

### Main Results

| Dataset/View | Metric | CoR-GS | CoR-GS + CoMapGS | Gain |
|------------|------|--------|-------------------|------|
| LLFF 3-view | PSNR/SSIM/LPIPS | 20.47/0.717/0.199 | 21.11/0.747/0.182 | +0.64/+0.030/-0.017 |
| LLFF 6-view | PSNR/SSIM/LPIPS | 24.78/0.844/0.116 | 25.20/0.854/0.108 | +0.42/+0.010/-0.008 |
| LLFF 9-view | PSNR/SSIM/LPIPS | 26.48/0.881/0.086 | 26.73/0.886/0.082 | +0.25/+0.005/-0.004 |
| Mip-NeRF 360 12-view | PSNR/SSIM/LPIPS | 19.16/0.574/0.414 | 19.68/0.591/0.394 | +0.52/+0.017/-0.020 |
| Mip-NeRF 360 24-view | PSNR/SSIM/LPIPS | 23.32/0.729/0.271 | 23.46/0.734/0.264 | +0.14/+0.005/-0.007 |

### Ablation Study (LLFF 6-view)

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|-------|-------|--------|------|
| CoR-GS baseline | 24.777 | 0.844 | 0.116 | Baseline |
| + Proximity loss only | 24.787 | 0.845 | 0.116 | Limited gain without point cloud enhancement |
| + Low-uncertainty point cloud enhancement △ | 24.90 | 0.849 | 0.112 | Point cloud in dense covisibility regions |
| + △ + Proximity loss | 25.153 | 0.854 | 0.109 | Significant synergistic effect |
| + Full point cloud enhancement | 25.076 | 0.852 | 0.109 | Adding single-view region points |
| **Full CoMapGS** | **25.204** | **0.854** | **0.108** | All components |

### Key Findings

- **Synergistic effect between initial point cloud enhancement and weighted supervision**: Using proximity loss alone yields a marginal improvement of 0.01 PSNR, but when combined with enhanced point clouds, the improvement exceeds 0.25 PSNR.
- Greater improvements are observed with fewer views (0.64 PSNR gain for 3-view vs 0.25 for 9-view), demonstrating that the method is particularly effective for extremely sparse scenes.
- Significant improvements are achieved even by only enhancing point clouds in low-uncertainty regions (△) (+0.12 PSNR, -0.004 LPIPS), indicating that point cloud density is a key bottleneck.
- LPIPS improvement on Mip-NeRF 360 is particularly prominent (-0.020) because outdoor scenes contain more high-uncertainty regions.

## Highlights & Insights

- **First to explicitly focus on single-view regions**: While previous methods ignore or penalize high-uncertainty regions, this paper does the opposite, applying stronger constraints to these regions through proximity loss.
- The concept of the covisibility map is simple yet powerful, compressing complex multi-view geometric relationships into a single integer count per pixel.
- **Plug-and-play** design: CoMapGS can be directly integrated into FSGS and CoR-GS without altering original training pipelines.

## Limitations & Future Work

- Reliance on MASt3R for dense correspondence prediction increases preprocessing computational costs.
- The proximity MLP classifier is trained offline and does not participate in the online optimization of 3DGS, potentially failing to exploit dynamic geometric variations during training.
- Monocular depth alignment employs simple linear regression (anisotropic scaling), which may not sufficiently handle complex non-linear depth scale variations.
- Slightly lower PSNR than ReconFusion (a diffusion-model-based method) on 3-view, but superior in SSIM/LPIPS.

## Related Work & Insights

- Unlike the covisibility concept proposed in DyCheck, this work extends covisibility from an evaluation tool to a training signal.
- The point cloud enhancement strategy (dense correspondence triangulation + depth alignment) can be used independently, benefiting all 3DGS-based methods.
- The proximity classifier workflow is similar to occupancy prediction in SDF fields but is more lightweight, making it worth exploring in other scene reconstruction tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of covisibility-map-guided adaptive supervision is novel, marking the first focus on recovering single-view regions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple settings on LLFF and Mip-NeRF 360 with comprehensive ablation studies and comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Rigorous notation definitions, systematic methodology description, and clear figures.
- Value: ⭐⭐⭐⭐ A practical plug-and-play module that provides sustained contributions to sparse-view synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](../../ECCV2024/3d_vision/coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[CVPR 2025\] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis](splatflow_multi-view_rectified_flow_model_for_3d_gaussian_splatting_synthesis.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->
