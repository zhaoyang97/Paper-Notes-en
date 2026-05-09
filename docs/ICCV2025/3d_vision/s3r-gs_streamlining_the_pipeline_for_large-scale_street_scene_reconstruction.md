---
title: >-
  [Paper Note] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction
description: >-
  [3D Vision] S3R-GS identifies three major computational redundancies in conventional street scene reconstruction pipelines—unnecessary local-to-global coordinate transformations, excessive 3D-to-2D projections, and inefficient rendering of distant content—and proposes instance-specific projection, temporal visibility filtering, and adaptive level-of-detail (LOD) strategies to reduce reconstruction time to 20%–50% of competing methods while maintaining state-of-the-art rendering quality.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: 468d6d6fdbd15f84
---

# S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.08217](https://arxiv.org/abs/2503.08217)
- **Area**: 3D Vision
- **Keywords**: 3D Gaussian Splatting, large-scale street scene reconstruction, autonomous driving, dynamic scenes, computational efficiency

## TL;DR
S3R-GS identifies three major computational redundancies in conventional street scene reconstruction pipelines—unnecessary local-to-global coordinate transformations, excessive 3D-to-2D projections, and inefficient rendering of distant content—and proposes instance-specific projection, temporal visibility filtering, and adaptive level-of-detail (LOD) strategies to reduce reconstruction time to 20%–50% of competing methods while maintaining state-of-the-art rendering quality.

## Background & Motivation

Large-scale street scene reconstruction is critical for applications such as autonomous driving, yet existing 3DGS methods face a fundamental challenge when applied at scale: **per-frame reconstruction cost escalates rapidly as scene size grows**.

Through systematic analysis of conventional pipelines, the authors identify three sources of computational redundancy:

**Unnecessary local-to-global transformations**: At each rendering frame, Gaussians belonging to dynamic objects must be transformed from their local coordinate systems to the global coordinate system, incurring substantial redundant matrix multiplications.

**Excessive 3D-to-2D projections**: All Gaussians are projected onto the image plane, yet the vast majority lie outside the current view frustum, making these projection computations entirely wasteful.

**Inefficient rendering of distant content**: All visible Gaussians undergo $\alpha$-blending uniformly, whereas small, distant Gaussians contribute negligibly to image quality while consuming considerable computation.

Furthermore, existing methods rely on **3D ground-truth bounding boxes** to separate dynamic and static elements, which are costly to annotate and limit practical applicability.

## Method

### Overall Architecture

S3R-GS consists of two stages—scene modeling and scene reconstruction—with the primary contributions targeting the reconstruction pipeline.

### Key Design 1: Instance-Specific Projection

Conventional methods first transform dynamic object Gaussians from local to global coordinates and then project them collectively onto the camera plane. S3R-GS bypasses the global transformation by precomputing instance-specific camera parameters for each object:

$$W_{t,i} = W_t \cdot W_{t,i2g}$$

During rendering, the corresponding camera is selected based on the Gaussian's instance ID, replacing two sequential transformations with a single matrix multiplication.

### Key Design 2: Temporal Visibility Filtering (Temporal Separation)

Each Gaussian is assigned a temporal visibility interval $v = (v_s, v_e)$ and a lifetime $l = (l_s, l_e)$. At rendering time $t$, only Gaussians satisfying $t_s \leq t \leq t_e$ are selected for projection, substantially reducing invalid 3D-to-2D projections.

Visibility intervals are dynamically updated using the actual visibility mask $M_t$ after rendering:

$$l_{s,i} = \min(l_{s,i}, t),\quad l_{e,i} = \max(l_{e,i}, t)$$

$$t_s = l_s - 0.1,\quad t_e = l_e + 0.1$$

### Key Design 3: Adaptive Level-of-Detail (Adaptive LOD)

For distant Gaussians whose projected 2D scale falls below a threshold $r$, the method applies:

1. **Probabilistic culling**: Gaussians are stochastically discarded based on depth, with higher discard probability at greater distances:

$$p = p_{max} + (p_{max} - 10^{-2}) \cdot \min(0, \frac{d-D}{D})$$

2. **Noise-based displacement**: Retained distant Gaussians receive depth-proportional positional noise to approximate the average color of the surrounding region.
3. **Distance-aware neural field**: Depth is incorporated as an input during color queries, enabling the network to implicitly learn color variations across different LOD levels.

### 2D Box-Based Scene Decomposition

As an alternative to 3D bounding boxes, the method leverages 2D bounding boxes together with SAM to obtain object masks, and projects LiDAR point clouds to derive coarse 3D trajectories $TXYZ \in \mathbb{R}^{T \times 3}$. A NeuralODE is introduced to learn continuous motion trajectories:

$$\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, c)$$

where $c$ is an instance embedding and $\mathbf{z}(t) = [\Delta XYZ_t + XYZ_t, R_t]$, enabling smooth pose estimation.

### BEV Semantic Initialization Enhancement

To address the limited LiDAR coverage of tall structures, additional initialization points are distributed along the $z$-axis within a BEV grid, improving scene completeness.

## Experiments

### Main Results: Argoverse 2 Large-Scale Street Scenes

| Method | Avg. PSNR↑ | Avg. SSIM↑ | Avg. LPIPS↓ | Reconstruction Time↓ |
|--------|-----------|-----------|-----------|---------------------|
| SUDS | 20.84 | 0.662 | 0.601 | - |
| ML-NSG | 21.15 | 0.680 | 0.555 | 49.10h |
| 4DGF | 24.97 | 0.772 | 0.447 | 54.39h |
| **S3R-GS** | **25.68** | **0.780** | **0.435** | **26.71h** |

S3R-GS surpasses all competing methods in reconstruction quality while reducing reconstruction time to less than half that of 4DGF.

### Ablation Study: Component Contributions (KITTI)

| Component | PSNR | Training Time |
|-----------|------|--------------|
| Baseline (4DGF) | Reference | Reference |
| + Instance-specific projection | ≈ | Significant ↓ |
| + Temporal visibility filtering | ≈ | Further ↓ |
| + Adaptive LOD | Slight ↑ | Further ↓ |
| + 2D decomposition | Slight ↓ | ≈ |

Instance-specific projection and temporal visibility filtering are the primary contributors to acceleration, while adaptive LOD further reduces cost without sacrificing quality.

### Key Findings
- On long sequences (full KITTI), the speedup is even more pronounced, with S3R-GS requiring approximately 20% of the time needed by competing methods.
- Although 2D box-based decomposition yields slightly lower accuracy than 3D box-based methods, it substantially improves practical applicability.
- BEV semantic initialization effectively improves reconstruction quality for tall structures.

## Highlights & Insights

1. **Systematic analysis**: Rather than proposing new modules, the work rigorously examines computational redundancy at each stage of the pipeline.
2. **Near-linear scalability**: After optimization, per-frame reconstruction cost scales far more gracefully with scene size.
3. **Practicality-oriented**: Replacing 3D bounding boxes with 2D boxes enables deployment in in-the-wild scenarios.
4. **Plug-and-play compatibility**: Each optimization strategy is independently applicable and can be integrated into other street scene 3DGS methods.

## Limitations & Future Work
- The 2D decomposition combined with NeuralODE may lack robustness under extremely high-speed motion or frequent occlusion.
- Probabilistic culling in adaptive LOD may introduce rendering instability in distant regions.
- LiDAR point clouds are still required for initialization, precluding fully sensor-agnostic operation.

## Related Work & Insights
- DrivingGaussian / StreetGaussian / HUGS / 4DGF: 3DGS-based street scene reconstruction methods
- EmerNeRF / NSG: NeRF-based street scene reconstruction methods
- NeuralODE: continuous-time modeling framework

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Systematic pipeline optimization rather than module stacking
- **Practicality**: ⭐⭐⭐⭐⭐ — Significant speedup with quality improvement; 2D boxes lower annotation barrier
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive validation across three datasets with ablations
- **Writing Quality**: ⭐⭐⭐⭐ — Problem analysis is thorough; proposed solutions are concise and effective

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](../../ICLR2026/3d_vision/urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)

</div>

<!-- RELATED:END -->
