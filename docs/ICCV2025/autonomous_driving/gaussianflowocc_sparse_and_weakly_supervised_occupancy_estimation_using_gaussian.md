---
title: >-
  [Paper Note] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow
description: >-
  [ICCV 2025][Autonomous Driving][Occupancy Grid Estimation] This paper proposes GaussianFlowOcc, which replaces dense voxel grids with sparse 3D Gaussian distributions for occupancy estimation. A Gaussian Transformer is i…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Occupancy Grid Estimation"
  - "3D Gaussian Splatting"
  - "Weak Supervision"
  - "Temporal Flow"
  - "Sparse Representation"
date: 2026-05-08
content_hash: a1715431061a771d
---

# GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow

**Conference**: ICCV 2025
**arXiv**: [2502.17288](https://arxiv.org/abs/2502.17288)
**Code**: [GitHub](https://github.com/boschresearch/GaussianFlowOcc)
**Area**: Autonomous Driving
**Keywords**: Occupancy Grid Estimation, 3D Gaussian Splatting, Weak Supervision, Temporal Flow, Sparse Representation

## TL;DR

This paper proposes GaussianFlowOcc, which replaces dense voxel grids with sparse 3D Gaussian distributions for occupancy estimation. A Gaussian Transformer is introduced for efficient scene modeling, and a Temporal Module estimates per-Gaussian 3D temporal flow to handle dynamic objects. The method substantially outperforms existing approaches on nuScenes under weak supervision (51%+ mIoU improvement) while achieving 50× faster inference.

## Background & Motivation

3D semantic occupancy estimation is a core task in autonomous driving, providing dense voxel-level understanding of the surrounding environment. However, existing methods face three key limitations:

**Reliance on dense 3D annotations**: Most occupancy estimation models require costly 3D voxel ground truth labels, typically obtained via LiDAR accumulation and manual correction, making them prohibitively expensive and difficult to scale.

**Inefficiency of dense voxel representations**: Traditional dense 3D voxel grids waste substantial computation on "empty" voxels — the majority of 3D space in real driving scenes is unoccupied. 3D convolution operations further exacerbate the computational burden.

**Neglect of scene dynamics in weakly supervised methods**: Existing self/weakly supervised methods (e.g., SelfOcc, OccNeRF) train via temporal rendering consistency but do not address temporal inconsistencies caused by dynamic object motion — moving objects occupy different positions across adjacent frames, leading to erroneous supervision signals during direct rendering.

The core insight of GaussianFlowOcc is that replacing dense voxels with sparse Gaussians not only reduces computation but also naturally supports Gaussian Splatting for efficient 2D rendering-based training. Furthermore, explicitly modeling scene dynamics by learning per-Gaussian 3D flow resolves the temporal inconsistency problem inherent to weak supervision.

## Method

### Overall Architecture

The model takes multi-view camera images as input. After feature extraction via an image encoder, a Gaussian Transformer iteratively refines an initial set of Gaussians into a final 3D scene representation. Gaussian Heads predict per-Gaussian attributes (opacity, scale, rotation, semantics), while a Temporal Module estimates 3D flow to adjacent frames. Depth and semantic maps are rendered via Gaussian Splatting and supervised with 2D pseudo-labels (semantics from GroundedSAM, depth from Metric3D).

The scene is represented as a set of $N$ Gaussians $\mathcal{G} = \{G_1, \ldots, G_N\}$, where each $G_i = (\mu, \sigma, s, r, c)$ encodes position, opacity, scale, rotation, and semantic logits.

### Key Designs

1. **Gaussian Transformer (Efficient 3D Scene Modeling)**: The core architectural contribution. It consists of $B$ iterative blocks, each executing the following steps in sequence:

    - **Positional Encoding**: The Gaussian positions $\mathcal{G}_\mu^{b-1}$ from the previous block are encoded via an MLP and added to the features.
    - **GICA (Gaussian-Image Cross-Attention)**: Deformable cross-attention projects Gaussian positions onto image feature maps to sample local information.
    - **ISA (Induced Self-Attention)**: Inspired by Set Transformer, $M$ learnable induction points $P \in \mathbb{R}^{M \times D}$ ($M \ll N$) reduce quadratic complexity $\mathcal{O}(N^2)$ to $\mathcal{O}(MN)$:
    $H = \text{MHA}(P, \mathcal{G}_f, \mathcal{G}_f), \quad \text{ISA}(\mathcal{G}_f) = \text{MHA}(\mathcal{G}_f, H, H)$
    This enables processing $N=10000$ Gaussians, whereas standard attention requires over 50GB of GPU memory at $N=5000$.
    - **ITA (Induced Temporal Attention)**: Similar to ISA but for temporal information propagation; induction points first aggregate Gaussian features from the previous frame, then the current frame interacts with them.
    - **Gaussian Rectification**: An MLP estimates positional residuals $\Delta\mathcal{G}_\mu^b$ to update Gaussian positions.

2. **Temporal Module (3D Temporal Flow Estimation)**: The key component for addressing dynamic objects under weak supervision. It estimates a 3D displacement for each Gaussian at each time step:

$$\vec{v}(t) = \text{MLP}_v(\mathcal{G}_f \oplus \Psi(t))$$

where $\Psi \in \mathbb{R}^{2T \times D}$ are learnable time tokens for each time step. The estimated offsets are added to Gaussian positions before Gaussian Splatting renders them into the corresponding temporal frames. This module **requires no additional losses or ground truth motion data** — motion is implicitly learned through existing rendering losses, since only accurate motion estimates yield correct renderings of dynamic objects in temporal frames.

3. **Temporal Gaussian Splatting (Enhanced Temporal Supervision)**: During training, camera parameters and labels from $T$ adjacent frames are loaded. The current Gaussian estimates (with Temporal Module displacement corrections) are rendered into these temporal viewpoints, and additional 2D rendering losses are computed. This effectively increases viewpoint overlap, compensating for the limited frustum overlap among multi-camera rigs in autonomous driving.

### Loss & Training

- Rendering losses: depth MSE loss $\mathcal{L}_{depth}$ + semantic binary cross-entropy loss $\mathcal{L}_{seg}$
- 2D pseudo-labels generated by GroundedSAM (semantic segmentation) and Metric3D (depth estimation)
- Temporal Gaussian Splatting horizon $T=6$
- ResNet-50 backbone, image resolution $256 \times 704$
- $N=10000$ Gaussians, 3 Transformer blocks, $M=500$ induction points
- 18 epochs of training on 4× A100 GPUs
- Voxelization is a post-processing step used only for benchmark evaluation

## Key Experimental Results

### Main Results

**Occ3D-nuScenes Weakly Supervised Occupancy Estimation**

| Method | Backbone | mIoU↑ | IoU↑ | RayIoU↑ | FPS↑ |
|--------|----------|-------|------|---------|------|
| SelfOcc | R50 | 10.54 | 45.01 | - | 1.15 |
| OccNeRF | R101 | 10.81 | 22.81 | - | 1.27 |
| GaussianOcc | R101 | 11.26 | - | 11.85 | 5.57 |
| GaussTR | 2×ViT | 13.26 | 45.19 | - | 0.20 |
| **GaussianFlowOcc** | **R50** | **17.08** | **46.91** | **16.47** | **10.2** |

Relative improvements: mIoU is 29% higher than GaussTR and 51%+ higher than voxel-based methods; inference speed is 50× faster than GaussTR.

### Ablation Study

**Effect of the Temporal Module**

| Configuration | mIoU | RayIoU | Notes |
|---------------|------|--------|-------|
| w/o Temporal Module | 14.18 | 14.46 | Temporal inconsistency unaddressed |
| **w/ Temporal Module** | **17.08** | **16.47** | **+20% mIoU improvement** |

**Attention Mechanism Ablation**

| Self-Attention | Temporal Attention | mIoU |
|---------------|-------------------|------|
| ✗ | ✗ | 13.81 |
| ✓ | ✗ | 14.60 |
| ✗ | ✓ | 14.47 |
| ✓ | ✓ | **17.08** |

**Gaussian Attribute Ablation**

| Opacity | Scale | Rotation | mIoU | Notes |
|---------|-------|----------|------|-------|
| ✗ | ✗ | ✗ | 9.48 | Position only |
| ✓ | ✗ | ✗ | 13.12 | + Opacity |
| ✓ | ✓ | ✗ | 14.98 | + Scale |
| ✓ | ✓ | ✓ | **17.08** | Full attributes |

### Key Findings

- The Temporal Module yields a 20% mIoU improvement, confirming that dynamic object motion compensation is the critical bottleneck in weakly supervised occupancy estimation.
- ISA enables the use of 10,000 Gaussians, whereas standard attention causes memory overflow and training divergence beyond 5,000.
- A lightweight ResNet-50 backbone surpasses GaussTR, which uses a 2×ViT-L backbone.
- The 3D Gaussian representation demonstrates clear advantages in modeling thin or flat objects (traffic signs, poles, pedestrians), unconstrained by voxel resolution.
- A temporal horizon of $T=6$ is optimal; training diverges for $T>8$.
- Opacity is the single most important Gaussian attribute (+3.64 mIoU).

## Highlights & Insights

- **Paradigm shift from dense to sparse representation**: The transition from dense voxels to sparse Gaussians is not merely an efficiency gain — it fundamentally changes the representational basis for scene understanding, with continuous Gaussian parameters naturally suited to express fine-grained geometry.
- **Elegant self-supervised dynamic modeling**: The Temporal Module implicitly learns motion through rendering consistency without requiring ground truth flow annotations, providing a principled solution to temporal inconsistency under weak supervision.
- **Practical utility of Induced Attention**: Adapting the Set Transformer concept to 3D scene modeling enables large-scale Gaussian processing at $\mathcal{O}(MN)$ versus $\mathcal{O}(N^2)$ complexity.
- **Practical inference speed**: At 10.2 FPS — 50× faster than GaussTR (0.2 FPS) — the method reaches a level suitable for real-world deployment.

## Limitations & Future Work

- The quality of 2D pseudo-labels directly constrains the performance ceiling; errors from foundation models (GroundedSAM, Metric3D) propagate into training.
- Training instability for temporal horizons exceeding 8 limits the exploitation of long-range temporal information.
- Increasing the number of Gaussians beyond $N=10000$ yields no further improvement, suggesting that adaptive density control may be necessary.
- The current design uses only single-frame input; multi-frame feature fusion may yield further gains.
- A performance gap relative to fully supervised methods remains; semi-supervised approaches combining limited 3D annotations are worth exploring.

## Related Work & Insights

- GaussianFlowOcc shares the sparse Gaussian representation with GaussianFormer and GaussianWorld, but is the first to introduce temporal flow modeling and weakly supervised training within this framework.
- The Induced Attention mechanism, derived from Set Transformer, is demonstrated to be viable for 3D visual tasks.
- The Temporal Gaussian Splatting paradigm is generalizable to other temporal dense prediction tasks.
- This work provides a principled approach for weakly supervised 3D scene understanding methods to handle dynamic environments.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A complete framework integrating sparse Gaussian occupancy estimation, temporal flow, and weak supervision, with multiple first-of-its-kind contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation on nuScenes with thorough ablations (attention mechanisms, attributes, temporal horizon, parameter counts), though limited to a single dataset.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The paper is clearly structured, with each contribution backed by corresponding experiments and high-quality figures.
- **Value**: ⭐⭐⭐⭐⭐ — Inference efficiency and weak supervision compatibility confer high practical value, with significant implications for the architectural design of autonomous driving perception systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] Self-Supervised Sparse Sensor Fusion for Long Range Perception](self-supervised_sparse_sensor_fusion_for_long_range_perception.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
