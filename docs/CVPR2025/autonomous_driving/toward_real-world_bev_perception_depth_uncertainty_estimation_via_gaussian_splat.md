---
title: >-
  [Paper Note] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting
description: >-
  [CVPR 2025][Autonomous Driving][BEV Perception] GaussianLSS introduces depth uncertainty modeling into the classic Lift-Splat-Shoot (LSS) framework. By calculating the variance of the depth distribution and converting it into a 3D Gaussian representation, which is then efficiently rasterized using Gaussian Splatting to generate uncertainty-aware BEV features, the method achieves state-of-the-art (SOTA) performance among unprojection methods on nuScenes…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "BEV Perception"
  - "Depth Uncertainty"
  - "Gaussian Splatting"
  - "Lift-Splat-Shoot"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: c951dd1b5f708aee
---

# Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2504.01957](https://arxiv.org/abs/2504.01957)  
**Code**: [https://hcis-lab.github.io/GaussianLSS/](https://hcis-lab.github.io/GaussianLSS/)  
**Area**: Autonomous Driving  
**Keywords**: BEV Perception, Depth Uncertainty, Gaussian Splatting, Lift-Splat-Shoot, Semantic Segmentation

## TL;DR

GaussianLSS introduces depth uncertainty modeling into the classic Lift-Splat-Shoot (LSS) framework. By calculating the variance of the depth distribution and converting it into a 3D Gaussian representation, which is then efficiently rasterized using Gaussian Splatting to generate uncertainty-aware BEV features, the method achieves state-of-the-art (SOTA) performance among unprojection methods on nuScenes, while being 2.5$\times$ faster and saving 70% of GPU memory compared to projection methods.

## Background & Motivation

**Background**: BEV perception is a core task in autonomous driving, providing a unified spatial representation for 3D detection, semantic segmentation, motion prediction, and planning. Existing methods are divided into two main paradigms: (1) 2D unprojection methods (e.g., LSS, FIERY) — which estimate depth to \"lift\" 2D features to 3D space; (2) 3D projection methods (e.g., BEVFormer, SimpleBEV, PointBEV) — which project predefined 3D queries onto the image plane to sample features without requiring explicit depth estimation.

**Limitations of Prior Work**: (1) 3D projection methods achieve the highest accuracy but are computationally expensive due to high 3D grid sampling costs, making real-time deployment difficult; (2) Although traditional LSS is efficient, it heavily relies on accurate depth estimation — which is inherently an ill-posed problem, and depth errors directly propagate to the BEV representation; (3) Existing LSS variants utilize softmax probability distributions for \"soft\" depth assignment but lack explicit modeling of depth uncertainty — softmax can generate vastly different probabilities across adjacent depth bins, leading to unstable BEV features.

**Key Challenge**: Unprojection methods are highly efficient but their accuracy is limited by the quality of depth estimation; projection methods are highly accurate but too slow. The challenge is to find a solution that is both efficient and robust to depth errors.

**Goal**: (1) Introduce depth uncertainty modeling into the LSS framework to reduce dependency on precise depth estimation; (2) Leverage Gaussian Splatting to achieve efficient BEV feature aggregation.

**Key Insight**: The authors observe that the variance of the depth distribution itself encodes information about depth estimation uncertainty — a large variance indicates high uncertainty, suggesting that features should \"spread\" over a wider spatial range to cover potential object locations. This aligns perfectly with the spatial \"expansion\" property of Gaussian distributions.

**Core Idea**: Calculate the mean and variance of the depth distribution for each pixel, convert them into 3D Gaussian distributions (where Mean = 3D position, Covariance = spatial uncertainty range), and then render them onto the BEV plane via Gaussian Splatting to achieve uncertainty-aware BEV feature aggregation.

## Method

### Overall Architecture

Input multi-view images → Backbone extracts features → CNN predicts splat feature $F_i$, opacity $\alpha_i$, and depth distribution $P_i$ → Depth uncertainty transformation (Mean $\mu$, Variance $\sigma^2$ $\rightarrow$ 3D Gaussians) → Multi-scale Gaussian Splatting rendering to the BEV plane → Fuse multi-scale BEV features → Segmentation head outputs predictions.

### Key Designs

1. **Depth Uncertainty Modeling**:

    - **Function**: Explicitly extract uncertainty information from the depth probability distribution.
    - **Mechanism**: Given the discrete depth distribution $P$ in LSS, compute the depth mean $\mu = \sum_{i} P_i(p) d_i$ and depth variance $\sigma^2 = \sum_{i} P_i(p)(d_i - \mu)^2$, and then define a tolerance range $\hat{\mathbf{D}} = [\mu - k\sigma, \mu + k\sigma]$. This range transforms a "point estimation" into an "interval estimation with uncertainty" — when the model is uncertain about the depth (large $\sigma$), features will spread over a wider depth range. $k$ is the error tolerance coefficient, empirically set to 0.5.
    - **Design Motivation**: In traditional LSS, the softmax depth distribution appears probabilistic, but it is actually only used for a weighted sum, failing to utilize key information regarding the \"spreadness\" of the distribution. Variance directly measures the confidence of the depth estimation.

2. **3D Uncertainty Transformation and Gaussian Representation**:

    - **Function**: Convert 1D depth uncertainty into 3D spatial Gaussian distributions.
    - **Mechanism**: Utilize camera intrinsic matrix $I$ and extrinsic matrix $E$ to back-project the pixel-depth points $(u,v,d_i)$ corresponding to each depth bin into the 3D space as $p_i^{3d} = E^{-1}(d_i \cdot I^{-1}[u,v,1]^T)$, and then calculate the 3D mean $\mu_{3d} = \sum_i P_i(p) p_i^{3d}$ and covariance matrix $\Sigma = \sum_i P_i(p)(p_i^{3d} - \mu_{3d})(p_i^{3d} - \mu_{3d})^T$. This yields 3D Gaussians $\mathcal{N}(\mu_{3d}, \Sigma)$, naturally representing spatial positions and uncertainty.
    - **Design Motivation**: Depth uncertainty is 1D in the camera coordinate system, but when mapped to world coordinates, it stretches along the ray direction as a 3D ellipsoid. A Gaussian distribution is the most natural mathematical tool to describe this spatial uncertainty.

3. **Multi-Scale BEV Feature Rendering**:

    - **Function**: Efficiently render BEV features using Gaussian Splatting, and alleviate the depth mean inconsistency issue through multi-scale processing.
    - **Mechanism**: Project 3D Gaussians (including mean, covariance, features, and opacity) onto the BEV plane, and render using alpha-blending: $\mathbf{F}_{BEV}(\mathbf{x}) = \sum_i F_i \alpha_i \exp(-\frac{1}{2}(\mathbf{x}-\mu_i)^\top\Sigma_i^{-1}(\mathbf{x}-\mu_i))$. To address BEV feature distortion caused by depth mean jumps between adjacent pixels, render BEV features at multiple resolutions (50×50, 100×100, 200×200) and then upsample and fuse them.
    - **Design Motivation**: The rasterization of Gaussian Splatting is highly efficient (based on tile-based rendering) and naturally supports spatial expansion (via covariance matrices), perfectly fitting uncertainty-aware feature aggregation. Multi-scale rendering borrows concepts from Feature Pyramid Networks (FPN).

### Loss & Training

Three loss functions are employed: focal loss for segmentation ($\lambda_1=1$), L1 loss for centerness ($\lambda_2=2$), and L2 loss for offset ($\lambda_3=0.1$). Optimizer: AdamW, learning rate: $3 \times 10^{-4}$ with cosine annealing, total batch size: 8, GPU: 2$\times$ RTX 4090, training epochs: 50. Backbone is EfficientNet-B4.

## Key Experimental Results

### Main Results

nuScenes Vehicle BEV semantic segmentation (IoU, 224$\times$480 resolution, without visibility filtering):

| Method | Type | Backbone | IoU↑ |
|------|------|----------|------|
| BEVFormer | 3D projection | RN-50 | 35.8 |
| SimpleBEV | 3D projection | RN-50 | 36.9 |
| PointBEV | 3D projection | EN-b4 | **38.7** |
| FIERY static | 2D unprojection | EN-b4 | 35.8 |
| CVT | 2D unprojection | EN-b4 | 31.4 |
| GaussianLSS | 2D unprojection | EN-b4 | **38.3** |

Efficiency comparison:

| Method | FPS↑ | GPU Mem (GiB)↓ | IoU |
|------|------|------------|-----|
| PointBEV | 32.0 | 1.26 | 38.7 |
| CVT | 107.6 | 0.35 | 31.4 |
| GaussianLSS | **80.2** | **0.33** | 38.3 |

### Ablation Study

Effect of the error tolerance coefficient $k$:

| k value | Vehicle IoU | Description |
|-----|-------------|------|
| 0.25 | ~37.0 | Too small, insufficient uncertainty coverage |
| 0.50 | **38.3** | Optimal |
| 1.00 | ~38.0 | Still within reasonable range |
| 2.00 | ~35.0 | Too large, features over-diffused |
| Direct extent prediction | 37.0 | Without uncertainty, 1.3% drop |

Effectiveness of opacity learning:

| Epoch | Retained Gaussian Ratio (α>0.01) | Vehicle IoU |
|-------|---------------------|-------------|
| Initial | ~100% | Low |
| After convergence | ~20% | Optimal |

### Key Findings

- GaussianLSS achieves SOTA (38.3 IoU) among unprojection methods, only 0.4% lower than the strongest projection method PointBEV, but is 2.5$\times$ faster and saves 74% of GPU memory.
- Direct prediction of a fixed extent performs 1.3% worse than learning uncertainty, proving that uncertainty modeling is superior to deterministic position prediction.
- Performance is stable within the range of $k \in [0.5, 1.25]$, but an excessively large $k$ causes feature over-diffusion, leading to performance degradation.
- GaussianLSS outperforms PointBEV on distant objects ($>30$m) — demonstrating that uncertainty modeling is especially important in long-range scenarios with high depth ambiguity.
- After training convergence, 80% of the Gaussian points have an opacity below 0.01, indicating that the model automatically learns to focus only on semantically relevant regions.

## Highlights & Insights

- **Uncertainty $\approx$ Target Extent**: Depth variance not only reflects the estimation uncertainty but also implicitly encodes the spatial range of objects (larger objects have more \"dispersed\" depth distributions). This is an elegant dual interpretation.
- **A New Use Case for Gaussian Splatting**: Transferring 3DGS from rendering tasks to feature aggregation in BEV perception is a highly creative application. The efficient rasterization of GS is naturally suited for scenarios requiring spatial expansion.
- **Adaptive Pruning of Opacity**: The model automatically learns to filter out 80% of redundant points using opacity, achieving adaptive sparsification without requiring post-processing.

## Limitations & Future Work

- Validated only on nuScenes; not tested on other datasets such as Waymo or Argoverse.
- Currently only handles single-frame perception and does not utilize temporal information — incorporating temporal context would allow uncertainty to propagate and update across time.
- Object shape prediction (IoU shape quality) is slightly inferior to projection methods.
- Multi-scale rendering introduces additional computational overhead, which might not be ideal for scenarios with extreme real-time requirements.
- Future work can extend this approach to more BEV tasks like 3D detection and map segmentation.

## Related Work & Insights

- **vs. LSS (Philion & Fidler)**: LSS pioneered the lift-splat paradigm but only performed softmax depth weighting, lacking uncertainty awareness. GaussianLSS introduces variance modeling and GS rendering to the same paradigm, representing a fundamental upgrade to LSS.
- **vs. PointBEV**: PointBEV achieves high accuracy using a coarse-to-fine 3D grid strategy (a projection method) but is relatively slow. GaussianLSS achieves comparable accuracy and much faster speed using unprojection + GS.
- **vs. BEVFormer**: BEVFormer uses 3D queries for cross-attention, which is computationally expensive. GaussianLSS replaces the attention mechanism with efficient GS rasterization.

## Rating

- Novelty: ⭐⭐⭐⭐ Elegantly and clearly integrates depth uncertainty and GS rendering into BEV perception.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple tasks on nuScenes with comprehensive ablations, efficiency analysis, and long-range analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly described method with intuitive illustrations.
- Value: ⭐⭐⭐⭐ Friendly to practical deployment (fast + low memory usage), representing a substantial advancement for unprojection methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](../../ICCV2025/autonomous_driving/6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[CVPR 2025\] PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting](pansplat_4k_panorama_synthesis_with_feed-forward_gaussian_splatting.md)
- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)

</div>

<!-- RELATED:END -->
