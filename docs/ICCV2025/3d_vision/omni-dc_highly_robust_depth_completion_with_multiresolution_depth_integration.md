---
title: >-
  [Paper Note] Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration
description: >-
  [ICCV 2025][depth completion] This paper presents OMNI-DC, a highly robust depth completion model that achieves zero-shot generalization across diverse datasets and sparse depth patterns via a multiresolution Discrete De…
tags:
  - "ICCV 2025"
  - "depth completion"
  - "multiresolution depth integration"
  - "zero-shot generalization"
  - "Laplacian loss"
  - "scale normalization"
date: 2026-05-08
content_hash: 0e2f8062e331217e
---

# Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration

**Conference**: ICCV 2025
**arXiv**: [2411.19278](https://arxiv.org/abs/2411.19278)  
**Code**: [GitHub](https://github.com/princeton-vl/OMNI-DC)  
**Area**: Depth Completion / 3D Vision
**Keywords**: depth completion, multiresolution depth integration, zero-shot generalization, Laplacian loss, scale normalization

## TL;DR

This paper presents OMNI-DC, a highly robust depth completion model that achieves zero-shot generalization across diverse datasets and sparse depth patterns via a multiresolution Discrete Depth Integration module (Multi-res DDI), a Laplacian loss, and scale normalization.

## Background & Motivation

Depth completion (DC) aims to predict dense depth maps from RGB images and sparse depth inputs, with broad applications in autonomous driving, 3D reconstruction, and novel view synthesis. Existing DC methods perform well within a single domain (e.g., NYUv2 or KITTI) but frequently fail catastrophically in **cross-dataset and cross-sensor** settings, forcing practitioners to train domain-specific models for each new scenario.

This paper targets the most challenging setting: **zero-shot generalization across sparsity levels and sensor types with a single model**. Three core challenges are identified:

**Extremely sparse inputs**: DDI-based methods suffer from linearly accumulating errors when depth points are scarce. When the distance $n$ between known depth points is large, the variance of integrated depth grows as $n \cdot \sigma^2$, causing predictions to collapse far from known points.

**Training convergence difficulty**: High ambiguity in sparse depth regions dominates the L1 loss, preventing the model from simultaneously capturing global structure and local detail.

**Cross-domain scale discrepancy**: Depth scales differ drastically between indoor (<1 m) and urban (>100 m) scenes, leading to loss imbalance and capacity issues when mixed during training.

## Method

### Overall Architecture

OMNI-DC pipeline: RGB image and normalized sparse depth are fed into a neural network → multiresolution depth gradient maps are predicted → integrated into a dense depth map via Multi-res DDI → upsampling + SPN refinement.

### Key Designs

1. **Multiresolution Depth Integration (Multi-res DDI)**: The core contribution of this paper. The original DDI formulates depth completion as a linear least-squares problem, solving for dense depth via gradient constraints and sparse depth constraints. Its limitation is linear noise accumulation over long integration distances. Multi-res DDI addresses this by having the network predict $R$ depth gradient maps $\{\hat{\mathbf{G}}^r\}_{r=1,...,R}$ at different resolutions, each differing by a factor of 2. Average-pooled downsampled versions of the target depth map are used to impose constraints at multiple scales simultaneously:

$$\mathcal{E}_G^R = \sum_{r=1}^R \sum_{i,j} (\mathbf{G}_{i,j}^{r,x} - \hat{\mathbf{G}}_{i,j}^{r,x})^2 + (\mathbf{G}_{i,j}^{r,y} - \hat{\mathbf{G}}_{i,j}^{r,y})^2$$

This reduces the number of integration steps between distant pixels from $n$ to $n/2^{R-1}$, significantly mitigating error accumulation. Since the number of low-resolution constraints decreases exponentially, the additional computational overhead is negligible.

2. **Laplacian Loss**: Conventional L1/L2 losses are dominated by ambiguous regions, causing the model to overemphasize global structure at the expense of local detail. A probabilistic Laplacian loss is introduced, where the model predicts a depth mean $\hat{\mathbf{D}}$ and a per-pixel scale parameter $b$:

$$L_{Lap} = \log(2b) + |\mathbf{D}^{gt} - \hat{\mathbf{D}}| / b$$

This allows the model to adaptively predict large $b$ (high uncertainty) in highly ambiguous regions, redirecting optimization capacity toward informative regions. The final loss is $L = L_1 + 0.5 \cdot L_{Lap} + 2.0 \cdot L_{gm}$.

3. **Scale Normalization**: All depths are transformed to log space (converting multiplicative scale factors to additive ones), and the sparse depth input is normalized by its median. Crucially, **normalization is applied only to network inputs, while the DDI uses sparse depth at its original scale**, ensuring scale equivariance between the output depth and the input sparse depth.

### Loss & Training

- Large-scale training on 5 synthetic datasets (573K images) covering indoor, outdoor, and urban scenes
- Synthetic sparse depth patterns: SfM (SIFT keypoint sampling) and LiDAR (4–128 line simulation)
- Two noise types simulated: outlier noise (COLMAP mismatches) and boundary noise (LiDAR-camera viewpoint discrepancy)
- CompletionFormer used as backbone; 3-resolution DDI; DySPN refinement
- Training on 10×48GB GPUs for approximately 6 days

## Key Experimental Results

### Main Results

Zero-shot evaluation on 7 real-world datasets; aggregated results over 4 datasets under synthetic sparse depth patterns:

| Method | 0.7% RMSE | 0.03% RMSE | 10% Noise RMSE | SIFT RMSE | LiDAR-8 RMSE |
|--------|:---:|:---:|:---:|:---:|:---:|
| OGNI-DC | 0.187 | 0.557 | 0.333 | 0.524 | 0.415 |
| G2-MonoDepth | 0.168 | 0.434 | 0.214 | 0.391 | 0.306 |
| Marigold | 0.367 | 0.384 | 0.406 | 0.453 | 0.378 |
| **OMNI-DC** | **0.135** | **0.289** | **0.147** | **0.211** | **0.231** |

On ETH3D-SfM outdoor split: OMNI-DC RMSE=1.069 vs. Marigold 1.883 (**43% reduction**). On KITTI 8-line LiDAR: zero-shot MAE=0.597, **outperforming all methods trained on KITTI**.

### Ablation Study

| Ablation | ETH3D-SfM RMSE | KITTI-64 RMSE |
|----------|:---:|:---:|
| DDI Res=1 (original) | 0.595 | 1.210 |
| DDI Res=1,2 | 0.489 | 1.218 |
| **DDI Res=1,2,3** | **0.459** | **1.188** |
| L1 only | 0.666 | 1.234 |
| L1 + $L_{Lap}$ | 0.598 | 1.224 |
| **L1 + $L_{Lap}$ + $L_{gm}$** | **0.490** | **1.173** |
| Linear depth | 0.886 | 1.289 |
| Log depth | 0.627 | 1.293 |
| **Log + Normalize** | **0.490** | **1.173** |
| Random pattern | 0.714 | 1.490 |
| **Rand + Synthetic + Noise** | **0.490** | **1.173** |

Each component yields substantial gains; Multi-res DDI contributes most on ETH3D (sparse SfM point regime).

### Key Findings

- Training exclusively on synthetic data without any real-world data yields large margins over competitors on real-world benchmarks.
- The model has only 85M parameters—an order of magnitude smaller than Depth Pro (907M)—and runs 93× faster than Marigold.
- In novel view synthesis applications (3DGS + depth loss), the model substantially improves rendering quality (PSNR gain of 4.76).

## Highlights & Insights

- **Elegant design of Multi-res DDI**: A straightforward multiscale extension resolves the fundamental problem of long-range integration error accumulation at negligible computational cost.
- **Core insight of the Laplacian loss**: Teaching the model to "acknowledge uncertainty" avoids futile optimization in ambiguous regions.
- **Elegant guarantee of scale equivariance**: Normalizing only the network input while retaining the original scale within DDI theoretically ensures output scale consistency with the input sparse depth.
- **Unexpected success of synthetic-only training**: Challenges the common assumption that real data is indispensable.

## Limitations & Future Work

- Synthetic training data may not cover the full diversity of real-world scenes.
- A performance gap versus domain-specifically trained models remains on densely sampled indoor inputs (NYU, 500 points).
- The SPN refinement module follows a conventional design; more advanced upsampling strategies remain unexplored.
- Temporal consistency in video sequences has not been investigated.

## Related Work & Insights

- Proposes key improvements over the DDI framework introduced in OGNI-DC, serving as its direct follow-up.
- The Laplacian loss draws on probabilistic prediction (e.g., Bayesian depth estimation) but is applied to depth completion for the first time.
- Scale normalization is conceptually related to scale-invariant losses in monocular depth estimation but differs fundamentally in design.
- Provides a plug-and-play depth prior for downstream tasks such as 3DGS and NeRF.

## Rating

- **Novelty**: 7/10 — Each component is well-motivated; Multi-res DDI is the central contribution.
- **Technical Quality**: 9/10 — Theoretical analysis supported by large-scale experiments and thorough ablations.
- **Practicality**: 9/10 — Zero-shot generalization, compact model, fast inference, open-source.
- **Writing Quality**: 8/10 — Problem analysis is thorough and method motivation is clearly articulated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] RePoseD: Efficient Relative Pose Estimation with Known Depth Information](reposed_efficient_relative_pose_estimation_with_known_depth_information.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[ICCV 2025\] FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation](fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)

</div>

<!-- RELATED:END -->
