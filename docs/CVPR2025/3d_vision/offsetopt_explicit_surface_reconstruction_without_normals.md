---
title: >-
  [Paper Note] OffsetOPT: Explicit Surface Reconstruction without Normals
description: >-
  [CVPR 2025][3D Vision][Surface reconstruction] Proposes OffsetOPT, a normal-free explicit surface reconstruction method. By training a triangle prediction network on uniformly distributed point clouds and generalizing it to arbitrary point clouds via point-wise offset optimization, it achieves state-of-the-art performance in both overall quality and sharp detail preservation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Surface reconstruction"
  - "point cloud processing"
  - "explicit surfaces"
  - "triangle mesh"
  - "offset optimization"
date: 2026-05-08
content_hash: 90135c0065825ddb
---

# OffsetOPT: Explicit Surface Reconstruction without Normals

**Conference**: CVPR 2025  
**arXiv**: [2503.15763](https://arxiv.org/abs/2503.15763)  
**Code**: [GitHub](https://github.com/EnyaHermite/OffsetOPT)  
**Area**: 3D Vision  
**Keywords**: Surface reconstruction, point cloud processing, explicit surfaces, triangle mesh, offset optimization

## TL;DR

Proposes OffsetOPT, a normal-free explicit surface reconstruction method. By training a triangle prediction network on uniformly distributed point clouds and generalizing it to arbitrary point clouds via point-wise offset optimization, it achieves state-of-the-art performance in both overall quality and sharp detail preservation.

## Background & Motivation

- Reconstructing surfaces from 3D point clouds is a core task in computer vision and computer graphics.
- Mainstream methods based on implicit neural representations (such as occupancy fields or SDFs) require high-quality oriented normals and rely on Marching Cubes to extract explicit meshes.
- Marching Cubes is incompatible with Unsigned Distance Fields (UDFs), which limits the reconstruction of open surfaces.
- Implicit methods tend to over-smooth sharp surface details.
- Although existing neural computing (explicit) methods possess strong generalizability and do not depend on normals, they heavily rely on ideal Poisson disk sampling distributions.
- Edge manifoldness (each edge sharing at most two triangles) in explicit methods requires explicit processing, increasing complexity.
- An explicit reconstruction method that requires neither normals nor specific point cloud distributions is highly desirable.

## Method

### Overall Architecture

OffsetOPT consists of two stages: (1) Supervised training of a Transformer-based triangle prediction network on uniformly distributed synthetic meshes (ABC dataset) to learn predicting adjacent triangular faces from local KNN neighborhood geometry. (2) Freezing network parameters and optimizing point-wise 3D offsets for new, unseen point clouds to align the point distribution closer to the uniform distribution preferred by the network, thereby improving triangle prediction accuracy. Finally, explicit triangle meshes are directly output without requiring implicit representation conversion.

### Key Designs

**1. Triangle Prediction Network**
- **Function**: Predicts adjacent triangular faces based on the local KNN neighborhood geometry of each point.
- **Mechanism**: Normalizes the $K$ nearest neighbors of each point $\mathbf{p}$ (by dividing by the nearest neighbor distance and scaling by $\eta_0=0.01$), adds positional encoding, and feeds them into a 5-layer Transformer. The model outputs a $K \times K$ symmetric probability matrix $\bar{\mathbf{O}}$, where each element $(i,j)$ represents the probability of the triangle $(\mathbf{p}, \mathbf{q}_i, \mathbf{q}_j)$. Triangles in the same row share the edge $(\mathbf{p}, \mathbf{q}_i)$, and edge manifoldness is ensured by selecting the top-2 in each row.
- **Design Motivation**: The $K \times K$ matrix representation naturally encodes the edge-sharing relationship, embedding edge manifoldness control directly within the prediction structure without requiring explicit post-processing. Normalization ensures prediction robustness across different resolutions.

**2. Point Offset Optimization**
- **Function**: Generalizes the trained network to point clouds with arbitrary distributions while automatically promoting edge manifoldness.
- **Mechanism**: Freezes the network and optimizes a 3D offset $\Delta\mathbf{p}$ for each point. This is initialized as $\Delta\mathbf{p}^0 = 0.25 \times (\mathbf{p} - \mathbf{q}_1)$ (pulling points away from their nearest neighbor). The optimization objective is to minimize the average BCE loss of the triangle predictions (using pseudo-labels). The step size is controlled by normalizing the gradient direction with $d_0(\mathbf{p})$, and a collision constraint of $d_{t+1} > d_0/2$ is set to prevent collisions between points.
- **Design Motivation**: The core intuition is to "adjust point positions to match the uniform distribution preferred by the network." Since training is conducted on uniform point clouds only, direct application to non-uniform point clouds performs poorly. Offset optimization elegantly converts the distribution adaptation problem into a gradient optimization task. Empirically, it is observed to significantly boost the ratio of manifold edges from 75% to 99%.

**3. Controlled Gradient Update Mechanism**
- **Function**: Prevents points from drifting too far or colliding during the offset optimization process.
- **Mechanism**: Normalizes the raw gradient and multiplies it by the nearest neighbor distance $d_0(\mathbf{p})$ to determine the step size: $\tilde{\nabla}_t(\Delta\mathbf{p}) = d_0(\mathbf{p}) \cdot \nabla_t / \|\nabla_t\|$. After each update, the new distance is checked, and the update is accepted only if $d_{t+1}(\mathbf{p}) > 0.5 \times d_0(\mathbf{p})$.
- **Design Motivation**: Uncontrolled gradient updates may cause points to drift away from the surface or collide with each other. Distance-adaptive step sizes and collision detection ensure optimization stability without destroying the local structure of the point cloud.

### Loss & Training

The training phase utilizes standard BCE loss (supervised):

$$\mathcal{L}_{\text{train}} = \text{BCE}(\bar{\mathbf{O}}, \mathbf{O}^*)$$

The offset optimization phase utilizes unsupervised BCE loss (pseudo-labels):

$$\mathcal{L}_{\text{opt}} = \frac{1}{N \times K \times K} \sum_n \sum_i \sum_j \text{BCE}(O_{nij})$$

## Key Experimental Results

### Main Results: ABC Test Set

| Method | CD1↓ | F1↑ | NC↑ | ECD1↓ (Sharp) | EF1↑ (Sharp) |
|------|------|-----|-----|-------------|------------|
| SPSR (+Normals) | 0.400 | 0.901 | 0.972 | 26.160 | 0.108 |
| DSE | 0.285 | 0.949 | 0.985 | 0.538 | 0.929 |
| CircNet | 0.284 | 0.950 | 0.985 | 0.708 | 0.924 |
| NKSR (+Normals) | 0.370 | 0.918 | 0.978 | 27.499 | 0.097 |
| **OffsetOPT** | **0.283** | **0.951** | **0.988** | **0.402** | **0.941** |

*OffsetOPT outperforms all methods without requiring normals, achieving a significant lead in sharp detail preservation (EF1).*

### FAUST Human Mesh

| Method | CD1↓ | F1↑ | ECD1↓ | EF1↑ |
|------|------|-----|-------|------|
| DSE | 0.218 | 0.995 | 0.883 | 0.801 |
| NKSR (+Normals) | 0.302 | 0.972 | 2.737 | 0.501 |
| **OffsetOPT** | **0.217** | **0.996** | **0.561** | **0.896** |

*On human models, OffsetOPT achieves the best performance across both overall and sharp metrics.*

### Key Findings

- Implicit methods (SPSR, NKSR) perform poorly in sharp feature preservation (EF1 < 0.52), whereas explicit methods generally excel.
- Offset optimization significantly enhances two key capabilities: (1) reconstructing surfaces from arbitrary point clouds, and (2) automatically ensuring 99%+ edge manifoldness.
- Trained only on simple synthetic ABC shapes, the model generalizes well to diverse scenarios such as FAUST humans, ScanNet indoor scenes, and CARLA autonomous driving.
- OffsetOPT (without normals) even outperforms NKSR (using ground-truth normals).

## Highlights & Insights

1. **Clever Design of Decoupled Training and Optimization**: Training on ideal distributions and adapting to arbitrary distributions via offset optimization elegantly converts the generalization problem into an optimization task.
2. **Implicit Edge Manifoldness**: Edge manifoldness constraint is naturally achieved via the $K \times K$ matrix structure and top-2 selection, completely avoiding explicit handling.
3. **Outstanding Advantage in Sharp Features**: Explicit reconstruction naturally bypasses the over-smoothing issue common in implicit methods.

## Limitations & Future Work

- Primarily targeting dense point clouds; handling noise and sparse point clouds is currently out of scope.
- A small fraction of non-manifold edges (<1%) may still occur, requiring simple post-processing.
- Offset optimization requires multiple iterations (with a decaying learning rate), which increases inference time.
- Future work could explore improvements in robustness against noisy point clouds.

## Related Work & Insights

- Compared to implicit methods requiring normals such as NKSR, OffsetOPT demonstrates the unique advantages of the explicit reconstruction paradigm.
- The concept of offset optimization can be extended to other geometric deep learning tasks requiring distribution adaptation.
- The $K \times K$ matrix-based triangle representation offers an elegant, structured solution for controlling edge manifoldness.

## Rating

⭐⭐⭐⭐ — The proposed methodology is novel and elegant. The core innovation of offset optimization addresses a key bottleneck in explicit reconstruction methods. It outperforms normal-reliant SOTA methods without any normal information, showing significant superiority over implicit methods in preserving sharp features. It demonstrates strong generalization, although the limitation of being applicable mainly to dense point clouds should be noted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GauSTAR: Gaussian Surface Tracking and Reconstruction](gaustar_gaussian_surface_tracking_and_reconstruction.md)
- [\[CVPR 2025\] ProbeSDF: Light Field Probes for Neural Surface Reconstruction](probesdf_light_field_probes_for_neural_surface_reconstruction.md)
- [\[CVPR 2025\] Parametric Point Cloud Completion for Polygonal Surface Reconstruction](parametric_point_cloud_completion_for_polygonal_surface_reconstruction.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
