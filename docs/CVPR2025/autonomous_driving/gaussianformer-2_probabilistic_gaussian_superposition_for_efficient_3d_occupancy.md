---
title: >-
  [Paper Note] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction
description: >-
  [CVPR 2025][Autonomous Driving][3D Occupancy Prediction] This paper proposes GaussianFormer-2, reinterpreting 3D semantic Gaussians from a probabilistic perspective: each Gaussian represents the occupancy probability distribution of its neighborhood. By aggregating geometric predictions via probability multiplication and normalizing semantic predictions using a Gaussian Mixture Model (GMM), it completely eliminates the issues of Gaussians describing empty regions and redundan…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "3D Occupancy Prediction"
  - "Gaussian Representation"
  - "Probabilistic Modeling"
  - "Sparse Scene Representation"
  - "Gaussian Mixture Models"
date: 2026-05-08
content_hash: 4dc4e1d096130988
---

# GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction

**Conference**: CVPR 2025  
**arXiv**: [2412.04384](https://arxiv.org/abs/2412.04384)  
**Code**: [huang-yh/GaussianFormer](https://github.com/huang-yh/GaussianFormer)  
**Area**: Autonomous Driving  
**Keywords**: 3D Occupancy Prediction, Gaussian Representation, Probabilistic Modeling, Sparse Scene Representation, Gaussian Mixture Models

## TL;DR

This paper proposes GaussianFormer-2, reinterpreting 3D semantic Gaussians from a probabilistic perspective: each Gaussian represents the occupancy probability distribution of its neighborhood. By aggregating geometric predictions via probability multiplication and normalizing semantic predictions using a Gaussian Mixture Model (GMM), it completely eliminates the issues of Gaussians describing empty regions and redundant overlaps, achieving SOTA performance with only 8.9% of the Gaussians.

## Background & Motivation

### Background
3D semantic occupancy prediction is a crucial task in vision-based autonomous driving, providing detailed descriptions of the geometry and semantics of the surrounding environment. Mainstream methods include: (1) dense voxel-based methods (computationally heavy); (2) plane-based (BEV/TPV) methods (lossy compression of information); (3) sparse 3D semantic Gaussian-based methods (GaussianFormer, object-centric but still redundant).

### Limitations of Prior Work
1. **Gaussians still describe empty regions**: In GaussianFormer, Gaussians model both occupied and unoccupied regions uniformly through a semantic attribute $\mathbf{c}$. Consequently, in outdoor scenes (where most space is empty), a large number of Gaussians are classified as "empty," resulting in extremely low Gaussian utilization.
2. **Aggregation process leads to redundant overlap**: GaussianFormer aggregates the contributions of each Gaussian through simple addition (Eq.1), leading to unbounded semantic logits. To fit this unbounded output, the model learns to assign more Gaussians to the same region, exacerbating overlap.
3. **Imprecise initialization**: The original learnable initialization strategy randomly initializes Gaussian positions at the start of training. However, the local receptive fields of Gaussians limit their ability to "move" to the correct positions during subsequent refinement.

### Key Challenge
How to make 3D semantic Gaussians truly "object-centric"—only describing occupied regions, wasting no computation on empty regions, while avoiding redundant overlaps among Gaussians?

### Key Insight
Reinterpret Gaussians from a probabilistic perspective: each Gaussian is understood as an occupancy **probability distribution** of its neighborhood, where the probability is 100% at the center and exponentially decays with distance. This naturally restricts Gaussians to describing only non-empty regions.

### Core Idea
Propose a probabilistic Gaussian superposition model: (1) geometry prediction—aggregating the occupancy probability of each Gaussian via the probability multiplication theorem, avoiding the unbounded issue of additive aggregation; (2) semantic prediction—using a Gaussian Mixture Model (GMM) to generate normalized semantic predictions, preventing unnecessary overlap. Meanwhile, design a distribution-based initialization module that learns pixel-wise occupancy distributions instead of surface depth, yielding more precise Gaussian initialization.

## Method

### Overall Architecture
GaussianFormer-2 maintains the overall architecture of GaussianFormer: extracting features via a 2D image backbone $\to$ initializing Gaussians $\to$ multi-block attention refinement (self-attention, image cross-attention, attribute refinement) $\to$ aggregating outputs for occupancy prediction. The core changes lie in the geometric/semantic aggregation methods and the Gaussian initialization strategy.

### Key Designs

#### 1. Probabilistic Gaussian Superposition—Geometry Prediction

- **Function**: Limit each Gaussian to describe only occupied regions and avoid redundant overlaps via probability multiplication.
- **Mechanism**:
    - Set the occupancy probability at each Gaussian center to 100%, exponentially decaying with distance according to the covariance matrix: $\alpha(\mathbf{x}; \mathbf{G}) = \exp(-\frac{1}{2}(\mathbf{x}-\mathbf{m})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{m}))$
    - Assuming the occupancy probability of each Gaussian is independent, aggregate the total occupancy probability using the probability multiplication theorem: $\alpha(\mathbf{x}) = 1 - \prod_{i=1}^{P}(1 - \alpha(\mathbf{x}; \mathbf{G}_i))$
- **Design Motivation**:
    - A 100% probability at the Gaussian center means each Gaussian **must** describe occupied regions, preventing them from being "wasted" on empty space.
    - Probability multiplication ensures $\alpha(\mathbf{x}) \geq \alpha(\mathbf{x}; \mathbf{G}_i)$: as long as a point is close to any Gaussian, it is predicted as occupied, eliminating the need for multiple overlapping Gaussians to inflate the probability.
    - Compared to additive aggregation (GaussianFormer), probability multiplication is naturally bounded (outputs in $[0,1]$), eliminating gradient signals that drive overlap.

#### 2. Gaussian Mixture Model—Semantic Prediction

- **Function**: Generate normalized semantic class probabilities to avoid Gaussian overlap caused by unbounded logits.
- **Mechanism**:
    - Remove the "empty" class from semantic attributes (already handled by geometric prediction).
    - Treat all Gaussians as a GMM: use opacity $a_i$ as the prior probability ($l^1$-normalized), Gaussian distributions as conditional probabilities, and softmax-normalized $\tilde{\mathbf{c}}_i$ as semantics.
    - Aggregate via Bayesian posterior: $\mathbf{e}(\mathbf{x}) = \sum_{i=1}^P p(\mathbf{G}_i|\mathbf{x}) \tilde{\mathbf{c}}_i$
    - Final occupancy prediction: $\hat{\mathbf{o}}(\mathbf{x}) = [1-\alpha(\mathbf{x}); \alpha(\mathbf{x}) \cdot \mathbf{e}(\mathbf{x})]$
- **Design Motivation**: A GMM naturally produces a normalized output, where the semantics of each point are dominated by the nearest Gaussian rather than an indiscriminate superposition of all Gaussians. This encourages each Gaussian to focus on independent spatial regions, reducing redundancy.

#### 3. Distribution-Based Initialization Module

- **Function**: Provide more precise sample-level initialization for probabilistic Gaussians, initializing Gaussians directly close to occupied areas.
- **Mechanism**:
    - Uniformly sample $R$ reference points along the ray for each pixel, querying ground truth occupancy annotations to obtain binary labels $\mathbf{l} = \{l_i\}_{i=1}^R$.
    - Predict pixel-wise occupancy distributions $\hat{\mathbf{l}}$ using image backbone $B$ and distribution predictor $M$.
    - Train with BCE loss: $loss_{init} = BCE(\hat{\mathbf{l}}, \mathbf{l}) = BCE(M(B(\mathcal{I})), \mathbf{l})$.
- **Design Motivation**:
    - Unlike depth prediction (which only captures visible surfaces), distribution prediction learns the complete occupancy distribution (including occluded areas).
    - No extra LiDAR supervision is needed; it relies solely on occupancy annotations.
    - Probabilistic Gaussians require initialization near non-empty regions (due to the 100% center probability), which the learnable initialization strategy cannot satisfy.

## Key Experimental Results

### Main Results — nuScenes (Surrounding 3D Semantic Occupancy Prediction)

| Method | IoU | mIoU | Num of Gaussians/Voxels |
|------|-----|------|-----------|
| MonoScene | 23.96 | 7.31 | 200×200×16 |
| TPVFormer | 30.86 | 17.10 | - |
| SurroundOcc | 31.49 | 20.30 | 200×200×16 |
| GaussianFormer | 29.83 | 19.10 | - |
| **Ours (Ch.128)** | 30.56 | 20.02 | - |
| **Ours (Ch.192)** | **31.74** | **20.82** | - |

GaussianFormer-2 (Ch.192) achieves an mIoU of **20.82**, outperforming SurroundOcc and GaussianFormer.

### KITTI-360 (Monocular Occupancy Prediction)

Compared to GaussianFormer, GaussianFormer-2 achieves significant improvements on SSCBench-KITTI-360 (reported as outperforming GaussianFormer by a clear margin).

### Efficiency Comparison

GaussianFormer-2 requires only **8.9%** of the number of Gaussians in GaussianFormer to achieve SOTA performance, vastly improving Gaussian utilization.

### Key Findings

1. **Probability Multiplication vs. Additive Aggregation**: Probability multiplication eliminates approximately 90% of redundant Gaussians (Gaussians describing empty regions) in GaussianFormer, significantly increasing the ratio of effective Gaussians.
2. **Crucial GMM Semantic Normalization**: After removing the "empty" category and normalizing semantics with softmax, the Gaussian overlap rate decreases significantly.
3. **Distribution Initialization vs. Depth Initialization**: Distribution initialization can perceive occluded occupied regions (e.g., behind obstacles), whereas depth initialization can only perceive visible surfaces.
4. Performance improves across various categories such as barrier, bicycle, bus, and car, with particularly notable improvements in barrier (+1.87) and bicycle (+2.18).

## Highlights & Insights

1. **Elegant and profound reinterpretation from a probabilistic perspective**: Reinterpreting 3D semantic Gaussians from "distribution values" to "occupancy probabilities" brings a fundamental improvement to the entire aggregation framework—transitioning from addition to probability multiplication, unbounded to bounded, and encouraging overlap to suppressing overlap.
2. **Unification of theory and practice**: Both the probability multiplication theorem and Gaussian Mixture Models have solid mathematical foundations, serving as solutions derived from first principles rather than ad-hoc engineering tricks.
3. **Clear root cause of efficiency gains**: The reduction of the Gaussian count to 8.9% is not achieved through pruning or compression, but because the utilization of each Gaussian has improved from ~10% to nearly 100%.
4. **Generalized concept of distribution initialization**: Learning pixel-wise 3D occupancy distributions (instead of surface depth) is a valuable idea that can be generalized to other 3D perception tasks.

## Limitations & Future Work

1. The independence assumption of probability multiplication might not hold completely in dense scenes (occupancy probabilities of adjacent objects could be correlated).
2. Using $l^1$ normalization for the opacity prior in GMMs might not be the optimal choice; more complex prior modeling could further improve performance.
3. The distribution initialization module introduces extra training overhead (requiring per-pixel occupancy distribution prediction).
4. Validation is mainly conducted on nuScenes and KITTI-360; validation on larger-scale datasets (such as Waymo) is missing.

## Related Work & Insights

- **Relationship with GaussianFormer (v1)**: GaussianFormer-2 is an upgrade to the core modeling paradigm of v1. While v1 proposed the concept of 3D semantic Gaussians, its efficiency was limited by additive aggregation and empty-region modeling. v2 thoroughly resolves these issues through probabilistic reinterpretation.
- **Analogy with Alpha Compositing in 3DGS Rendering**: The formulation of the probability multiplication theorem shares similarities with the alpha blending equation in 3DGS rendering, though it is applied here to occupancy prediction instead of rendering.
- **Research Directions in Sparse Representation**: The evolution from voxels $\to$ BEV/TPV $\to$ semantic Gaussians $\to$ probabilistic Gaussians demonstrates the development trajectory of 3D representation from dense to sparse, and ultimately to truly object-centric representations.
- **Insights for Occupancy Forecasting**: Probabilistic Gaussians are naturally suited for modeling temporal changes—the movement of Gaussians corresponds to object motion, and changes in probability correspond to appearance/disappearance.

## Rating

⭐⭐⭐⭐⭐ (5/5)

Mathematically elegant, highly efficient in engineering, and comprehensively evaluated. Reinterpreting Gaussian representation from a probabilistic perspective is a profound and impactful contribution that addresses the fundamental limitations of GaussianFormer v1. Achieving SOTA results with only 8.9% of the Gaussians demonstrates the true potential of sparse representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](../../ECCV2024/autonomous_driving/gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [\[CVPR 2025\] Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting](spatiotemporal_decoupling_for_efficient_vision-based_occupancy_forecasting.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](../../ICCV2025/autonomous_driving/gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)

</div>

<!-- RELATED:END -->
