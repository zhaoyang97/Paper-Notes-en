---
title: >-
  [Paper Note] Occupancy Learning with Spatiotemporal Memory
description: >-
  [ICCV 2025][Autonomous Driving][3D occupancy prediction] This paper proposes ST-Occ, a scene-level spatiotemporal occupancy representation learning framework. Through a Unified Temporal Modeling paradigm…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D occupancy prediction"
  - "temporal fusion"
  - "spatiotemporal memory"
  - "uncertainty awareness"
date: 2026-05-08
content_hash: 7101dd5569e2ef4f
---

# Occupancy Learning with Spatiotemporal Memory

**Conference**: ICCV 2025
**arXiv**: [2508.04705](https://arxiv.org/abs/2508.04705)  
**Code**: [https://github.com/matthew-leng/ST-Occ](https://github.com/matthew-leng/ST-Occ)  
**Area**: Autonomous Driving
**Keywords**: 3D occupancy prediction, temporal fusion, spatiotemporal memory, autonomous driving, uncertainty awareness

## TL;DR

This paper proposes ST-Occ, a scene-level spatiotemporal occupancy representation learning framework. Through a Unified Temporal Modeling paradigm, it employs a spatiotemporal memory bank defined in scene coordinates along with an uncertainty- and dynamics-aware memory attention mechanism. ST-Occ outperforms the prior state of the art by 3 mIoU on the Occ3D benchmark while reducing temporal inconsistency by 29%.

## Background & Motivation

3D occupancy representation has emerged as an important fine-grained perception paradigm for modeling surrounding environments in autonomous driving. However, efficiently aggregating 3D occupancy information across multiple frames remains challenging along three dimensions:

**Efficiency**: Occupancy representations are dense voxel features; the additional height dimension makes storing and processing multi-frame historical features highly resource-intensive.

**Uncertainty**: Occlusion and illumination variation cause voxel-level uncertainty to accumulate across frames, degrading prediction robustness.

**Dynamics**: Dynamic objects in the scene induce voxel displacement; without accurate modeling, historical features become misaligned.

Existing methods primarily adopt recurrent or stacking-based per-frame queue temporal fusion, extending BEV-based approaches to 3D occupancy, but incur large computational and memory overhead with low efficiency in exploiting temporal information. This paper proposes constructing a unified spatiotemporal memory in the **scene coordinate frame** (rather than the ego-vehicle frame) to overcome these bottlenecks.

## Method

### Overall Architecture

The core idea of ST-Occ is **Unified Temporal Modeling**: a single unified memory $\mathbf{M}$ in scene coordinates replaces the conventional per-frame queue. Given current-frame multi-view images $I_t$, an occupancy encoder extracts an ego-coordinate occupancy representation $\mathbf{V}_t$. A memory attention module then fuses $\mathbf{V}_t$ with historical information $\mathbf{H}_t$ from the spatiotemporal memory to obtain $\tilde{\mathbf{V}}_t$, after which the memory is updated.

### Key Designs

1. **Spatiotemporal Memory**: A global representation $\mathbf{M} \in \mathbb{R}^{H_G \times W_G \times Z_G \times C_G}$ is maintained in scene coordinates. When fusing $k$ temporal frames, conventional methods must store $k$ complete representations, whereas Unified Temporal Modeling requires only one, yielding substantially improved memory efficiency. The memory stores three types of temporal attributes $\mu = \{\mathbf{c}, \delta, \mathbf{f}\}$:

    - **Historical class activation $\mathbf{c}$**: A softmax class prediction vector updated incrementally with exponential decay $\alpha=0.5$.
    - **Mean log-variance $\delta$**: Used for uncertainty estimation.
    - **Occupancy flow $\mathbf{f}$**: A 2D motion vector in bird's-eye view for motion compensation of dynamic instances.

2. **Memory Attention**: The current occupancy representation $\mathbf{V}_t$ is conditioned on historical information from the spatiotemporal memory. The core formulation is:
    $$(1-u) \cdot \text{DA}(V_{t_p}, p+f, V_t) + u \cdot \text{DA}(V_{t_p}, p+f, \chi[\mathbf{M}_t, T_t])$$
   where $u$ is an uncertainty weight obtained by encoding temporal attributes via an MLP, and $f$ is the occupancy flow for dynamic compensation. The inputs to uncertainty $u$ include the historical class activation $\mathbf{c}$, log-variance $\delta$, and the cosine similarity $\varepsilon$ between current and historical features.

3. **Temporal Consistency Metric (mSTCV)**: The paper introduces the mean Spatiotemporal Classification Variability, which measures the rate of classification change across frames for voxels corresponding to the same real-world location, quantifying the stability of temporal predictions. It is computed as the proportion of non-empty voxels whose classification changes per frame, averaged over all frames.

### Loss & Training

The total loss consists of three components:
$$\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{nll} + \mathcal{L}_{of}$$

- $\mathcal{L}_{occ}$: Occupancy prediction loss, comprising Focal loss, Lovász softmax loss, affinity loss, and depth loss.
- $\mathcal{L}_{nll}$: Gaussian negative log-likelihood loss for log-variance prediction.
- $\mathcal{L}_{of}$: L1 loss for occupancy flow prediction.

Training details: learning rate $2 \times 10^{-4}$, 26 training epochs, with temporal modeling disabled during the first 3 epochs for training stability. Ground-truth labels for occupancy flow are computed on-the-fly from temporal displacements of instance bounding boxes in nuScenes annotations.

## Key Experimental Results

### Main Results (Tables)

3D occupancy prediction results on the Occ3D benchmark (ResNet50 backbone):

| Method | mIoU | barrier | car | driv. surf. | sidewalk | terrain | manmade | vegetation |
|------|------|---------|-----|-------------|----------|---------|---------|------------|
| FB-OCC† (w/o temporal) | 37.39 | 44.83 | 47.97 | 78.83 | 49.06 | 52.22 | 39.07 | 34.61 |
| FB-OCC (w/ temporal) | 39.11 | 44.74 | 49.10 | 80.07 | 51.18 | 55.13 | 42.19 | 37.53 |
| **ST-Occ (ours)** | **42.13** | **49.62** | **52.55** | **84.26** | **56.09** | **59.85** | **45.27** | **40.11** |
| ViewFormer† | 37.80 | 44.89 | 48.90 | 81.93 | 53.72 | 55.50 | 42.18 | 36.29 |
| ViewFormer | 41.44 | 50.16 | 53.36 | 84.67 | 57.43 | 59.64 | 47.57 | 40.38 |
| ViewFormer† + ST-Occ | **42.30** | **50.61** | **53.24** | **85.28** | **58.39** | **60.39** | **48.02** | **41.42** |

Temporal consistency evaluation (mSTCV ↓):

| Method | mSTCV (%) | mSTCV† (%) |
|------|-----------|------------|
| FB-OCC | 12.18 | 8.57 |
| ST-Occ | **8.68** | **6.48** |

### Ablation Study (Tables)

Contribution of individual design components (Occ3D):

| Configuration | mIoU |
|------|------|
| No Temporal (FB-OCC baseline) | 37.39 |
| Mem. Attn. | 41.17 |
| Mem. Attn. + Dynamics | 41.73 |
| Mem. Attn. + Uncertainty | 41.85 |
| **ST-Occ (full)** | **42.13** |

Temporal fusion efficiency comparison (same fusion operation and number of frames):

| Temporal Modeling | Train Mem. (GB) ↓ | Fusion Time (ms) ↓ | Infer. Mem. (GB) ↓ | FPS ↑ |
|-------------|----------------|----------------|----------------|-------|
| Recurrent | 12.89 | 705 | 10.08 | 5.95 |
| Stacked | 19.02 | 84 | 11.29 | 5.42 |
| **Unified (ours)** | **10.90** | **24** | **5.57** | **8.65** |

Sub-component ablation (contribution of temporal attributes):

| c | ε | δ | f | mIoU |
|---|---|---|---|------|
| - | - | - | - | 41.17 |
| ✓ | - | - | - | 41.45 |
| ✓ | ✓ | - | - | 41.73 |
| ✓ | ✓ | ✓ | - | 41.85 |
| - | - | - | ✓ | 41.73 |
| ✓ | ✓ | ✓ | ✓ | **42.13** |

### Key Findings

- ST-Occ outperforms FB-OCC by 3 mIoU, with 2.8× greater efficiency in exploiting temporal information.
- The unified temporal modeling achieves a fusion time of only 24 ms (vs. 705 ms for recurrent) and an inference memory footprint of only 5.57 GB (vs. 11.29 GB for stacking).
- Dynamics-awareness primarily benefits dynamic categories (e.g., car +1.2 IoU), while uncertainty-awareness primarily benefits static categories.
- Performance continues to improve and temporal consistency strengthens as the number of fused frames increases, with computational cost growing far more slowly than conventional methods.

## Highlights & Insights

- **Scene-coordinate unified memory** is the core innovation: replacing per-frame queues with a single global representation fundamentally resolves the storage and computation bottleneck of multi-frame 3D occupancy features.
- The decoupled design of uncertainty and dynamics awareness is elegant: the MLP-encoded $u$ automatically balances the contribution of the current frame and historical frames, while occupancy flow $f$ provides dynamic compensation.
- mSTCV is a valuable new metric that fills a gap in temporal consistency evaluation for occupancy prediction.
- The method exhibits strong plug-and-play applicability, as it can replace the temporal modules of both FB-OCC and ViewFormer.

## Limitations & Future Work

- Dynamic modeling relies on nuScenes annotations to compute ground-truth occupancy flow labels; future work could derive dynamic information directly from temporal features.
- The approach is extensible to sparse-query-based perception methods.
- The size of the scene memory is constrained by GPU memory; scalability to extremely large-scale scenes warrants further investigation.

## Related Work & Insights

- The method builds upon BEVFormer's temporal self-attention but breaks away from the per-frame alignment paradigm.
- PasCo introduced uncertainty awareness for occupancy prediction; this paper integrates that concept into temporal modeling.
- The scene-level representation is conceptually analogous to global map construction in SLAM, but realized in a differentiable, learning-based manner.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified temporal modeling paradigm is a concise and effective contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablations with well-designed efficiency and temporal consistency evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with coherent mathematical derivations.
- **Value**: ⭐⭐⭐⭐ Provides a new efficient paradigm for temporal modeling in occupancy prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)

</div>

<!-- RELATED:END -->
