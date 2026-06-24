---
title: >-
  [Paper Note] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images
description: >-
  [ICCV 2025][3D Vision][3D Scene Graph] This paper proposes FROSS, a method that lifts 2D scene graphs directly into 3D space and represents objects as Gaussian distributions, achieving faster-than-real-time (144 FPS) online 3D semantic scene graph generation without requiring precise point cloud reconstruction.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Scene Graph"
  - "Real-time"
  - "Gaussian Distribution"
  - "RGB-D"
  - "Scene Understanding"
date: 2026-05-08
content_hash: 2e2727bc6d65fad8
---

# FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images

**Conference**: ICCV 2025
**arXiv**: [2507.19993](https://arxiv.org/abs/2507.19993)  
**Code**: [Available](https://github.com/Howardkhh/FROSS)  
**Area**: 3D Vision / Scene Understanding
**Keywords**: 3D Scene Graph, Real-time, Gaussian Distribution, RGB-D, Scene Understanding

## TL;DR

This paper proposes FROSS, a method that lifts 2D scene graphs directly into 3D space and represents objects as Gaussian distributions, achieving faster-than-real-time (144 FPS) online 3D semantic scene graph generation without requiring precise point cloud reconstruction.

## Background & Motivation

### Limitations of Prior Work

**Background**: 3D semantic scene graphs (SSGs) represent objects as nodes and inter-object relationships as edges, serving as a critical data structure for high-level scene understanding in robotics, AR, and related domains. Existing methods face two major challenges:

**High computational cost**: Mainstream methods rely on precise point cloud reconstruction and segmentation (e.g., SLAM), requiring substantial computational resources.

**Non-incremental processing**: Offline methods require complete scene data (point clouds or full image sequences), making them unsuitable for open-world incremental exploration.

**Key Challenge**: 3D SSGs are fundamentally intended for high-level semantic understanding, where precise object poses and geometry are **not strictly necessary**. For instance, robot planning requires only relative spatial relationships, and in 3D scene synthesis, SSGs serve merely as a structural scaffold. This observation motivates an entirely new paradigm — **bypassing point cloud reconstruction and lifting directly from 2D scene graphs to 3D**.

## Method

### Overall Architecture

FROSS consists of four core modules:
1. **RT-DETR Object Detection**: Detects objects from RGB-D images.
2. **EGTR Relationship Extraction**: Extracts inter-object relationships using RT-DETR's self-attention features to construct a 2D scene graph.
3. **2D→3D Lifting**: Back-projects 2D Gaussian distributions into 3D space to build a local 3D SSG.
4. **Global SSG Fusion**: Integrates local SSGs into a global SSG via a Gaussian merging algorithm.

### Key Designs

**2D Gaussian Representation**: Each detection bounding box is modeled as a 2D uniform distribution, with the mean at the box center and covariance matrix defined as:

$$\Sigma_i^{2D} = \frac{1}{12} \begin{bmatrix} W_i^2 & 0 \\ 0 & H_i^2 \end{bmatrix}$$

**3D Back-Projection**: Camera intrinsics $K$, rotation $R$, and translation $t$ are used to map 2D Gaussians into 3D space. The depth-dimension variance is approximated by the mean of the other dimensional variances, addressing the missing depth variance in 2D→3D projection.

**Hellinger Distance-Based Merging**: For objects of the same category, the Hellinger distance between Gaussian distributions is computed; nodes with distance below threshold $\delta_d = 0.85$ are merged via weighted fusion:

$$\mu_k = \frac{w_i\mu_i + w_j\mu_j}{w_i + w_j}$$

Weights reflect detection frequency, granting higher weight to objects detected from multiple viewpoints and spatial positions, thereby mitigating viewpoint bias. Relationship predictions are determined by majority voting.

### Loss & Training

- The 2D scene graph model (EGTR + RT-DETRv2-M) is trained on the 3DSSG or Visual Genome dataset.
- Object confidence threshold is set to 0.7; the top 10 relationships per 2D scene graph are retained.
- Main experiments use ground-truth trajectories; ablation studies validate robustness under ORB-SLAM3 estimated trajectories.

## Key Experimental Results

### Main Results (Table)

| Method | Rel. Recall | Obj. Recall | Pred. Recall | mRecall Obj. | Latency (ms) |
|--------|------------|------------|-------------|-------------|-------------|
| SGFN | 22.0 | 51.6 | 27.5 | 37.7 | 161 |
| Wu | 23.3 | 53.8 | 28.4 | 43.8 | 191 |
| Kim | 9.1 | 59.0 | 7.1 | 51.0 | 310 |
| **FROSS** | **27.9** | **62.4** | **33.0** | **63.8** | **7** |

FROSS improves relationship recall by 19.7% and object recall by 16.0% over the second-best baseline, with a latency of only 7 ms — 23× faster than the fastest baseline.

### Ablation Study (Table)

| Setting | Rel. Recall | Obj. Recall | Pred. Recall |
|---------|------------|------------|-------------|
| FROSS (Predicted 2D SG) | 27.9 | 62.4 | 33.0 |
| FROSS (GT 2D SG) | 55.8 | 88.6 | 56.0 |
| FROSS (SLAM Trajectory) | 22.7 | 25.8 | 27.2 |
| FROSS (GT Trajectory) | 22.3 | 26.1 | 27.8 |

### Key Findings

1. **2D SG quality is the bottleneck**: Performance doubles when using GT 2D SGs, indicating that current results represent only a lower bound.
2. **Robust to trajectory estimation errors**: Performance under SLAM trajectories is comparable to that with GT trajectories.
3. **Merging threshold governs a trade-off**: Lower thresholds preserve more objects (higher Obj. Recall); higher thresholds promote relationship aggregation (higher Rel. Recall).
4. **Runtime analysis**: The system achieves 144.09 FPS, with the merging algorithm contributing only 0.12 ms.

## Highlights & Insights

1. **Paradigm shift**: By bypassing the conventional point cloud reconstruction pipeline and building 3D SSGs directly from 2D scene graphs, the method substantially simplifies the overall workflow.
2. **Elegance of Gaussian representation**: Representing object locations and spatial extents as Gaussian distributions is both lightweight and well-suited for supporting merging operations.
3. **Depth variance compensation**: Approximating the missing depth variance with the mean of other dimensional variances during 2D→3D back-projection is a simple yet effective solution.
4. **ReplicaSSG dataset**: An extension of the Replica dataset with relationship annotations using Visual Genome category definitions, enabling zero-shot transfer evaluation.

## Limitations & Future Work

1. **2D SG quality caps performance**: Current results are heavily dependent on the accuracy of the 2D SG model.
2. **Limited semantic relationship vocabulary**: Experiments use only 7 predicate categories.
3. **Simplistic depth variance assumption**: Assuming depth variance equals the mean of other dimensional variances may be inaccurate for elongated objects.
4. **Dynamic scenes not considered**: The framework assumes a static environment.

## Related Work & Insights

- **SceneGraphFusion**: Achieves real-time SSG via multi-threaded point cloud reconstruction and segmentation, but incurs high system latency.
- **EGTR**: An end-to-end relationship extractor that leverages self-attention features from object detectors.
- **RT-DETR**: A real-time DETR-based detector that underpins the throughput of the entire pipeline.
- **Insight**: High-level semantic tasks do not necessarily require precise geometric information; appropriate approximations can yield order-of-magnitude efficiency gains.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Practical Value | 4.5 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](../../CVPR2025/3d_vision/slam3r_real-time_dense_scene_reconstruction_from_monocular_rgb_videos.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [\[ICCV 2025\] 3D Test-time Adaptation via Graph Spectral Driven Point Shift](3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)
- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)

</div>

<!-- RELATED:END -->
