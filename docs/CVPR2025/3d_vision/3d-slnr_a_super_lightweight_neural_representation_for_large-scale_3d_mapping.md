---
title: >-
  [Paper Note] 3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping
description: >-
  [CVPR 2025][3D Vision][SDF] This paper proposes 3D-SLNR, a super lightweight neural 3D representation. It defines the global Signed Distance Function (SDF) based on a collection of band-limited local SDFs anchored on support points of a point cloud. Each local SDF is parameterized by a single shared tiny MLP (without latent feature vectors). The output of the MLP is modulated by learnable geometric attributes (position, rotation, and scale) to adapt to complex geometries in d…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "SDF"
  - "Super Lightweight"
  - "Neural Mapping"
  - "Local SDF"
  - "Support Points"
  - "Geometric Transformation Modulation"
  - "Prune-and-Expand"
date: 2026-05-08
content_hash: 78f5e1571d875982
---

# 3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping

**Conference**: CVPR 2025  
**Code**: Unreleased  
**Area**: 3D Vision / 3D Reconstruction / Neural Representation  
**Keywords**: SDF, Super Lightweight, Neural Mapping, Local SDF, Support Points, Geometric Transformation Modulation, Prune-and-Expand

## TL;DR
This paper proposes 3D-SLNR, a super lightweight neural 3D representation. It defines the global Signed Distance Function (SDF) based on a collection of band-limited local SDFs anchored on support points of a point cloud. Each local SDF is parameterized by a single shared tiny MLP (without latent feature vectors). The output of the MLP is modulated by learnable geometric attributes (position, rotation, and scale) to adapt to complex geometries in different regions. Combined with a parallel query algorithm and a prune-and-expand strategy, it achieves SOTA reconstruction quality with less than 1/5 of the memory footprint of previous methods.

## Background & Motivation

**Background**: Large-scale 3D mapping is a core research direction in computer vision and robotics. Recently, neural implicit representation methods (such as Instant-NGP, SHINE-Mapping, VDB-Mapping, etc.) have achieved high-quality 3D reconstruction using multi-resolution hash encoding combined with MLPs. These methods encode scene geometry into latent feature vectors and neural network parameters, replacing traditional voxel grids or point cloud representations.

**Limitations of Prior Work**: (1) Hash-encoding-based methods (e.g., Instant-NGP) require substantial memory to store latent feature vectors in multi-resolution hash tables for large-scale scenes, and hash collisions become increasingly severe in large scenes, leading to uncontrollable degradation in reconstruction quality; (2) Octree- or VDB-based methods have more structured organizations, but each node still stores latent feature vectors, resulting in memory consumption that grows linearly with the scene scale; (3) Pure MLP methods have few parameters, but a single large MLP has limited expression capability and slow training (due to global optimization).

**Key Challenge**: Accurate 3D reconstruction requires sufficient feature storage to encode complex geometric details, but feature storage for large-scale scenes quickly inflates to unsustainable levels. The core question is: can high expressiveness for complex geometries be maintained under an extremely low parameter budget?

**Goal**: To design an extremely lightweight neural 3D representation that achieves SOTA reconstruction quality with minimal memory footprint (< 1/5 of previous methods) in large-scale scene mapping.

**Key Insight**: Instead of storing independent latent feature vectors for each spatial position/voxel, this work proposes to share a single tiny MLP across a set of local SDFs anchored to support points. The MLP output is modulated by the respective geometric transformation attributes (position, rotation, scale) of each local SDF, achieving "one MLP, multiple geometries."

**Core Idea**: Replace latent feature vectors with learnable geometric transformation attributes to realize personalized representations of local geometries, reducing storage overhead from high-dimensional feature vectors to just 3 geometric parameters (9 scalars) while preserving expressiveness.

## Method

### Overall Architecture
A set of support points is sampled from the point cloud → Each support point anchors a band-limited local SDF → All local SDFs share the same tiny MLP (without latent feature vectors) → Each local SDF modulates the MLP output via three learnable geometric attributes: position $\mathbf{p}$, rotation $\mathbf{R}$, and scale $\mathbf{s}$ → The global SDF is formed by fusing all local SDFs in the near-surface space. During training, the distribution of support points is dynamically adjusted using a prune-and-expand strategy.

### Key Designs

1. **Band-limited Local SDF**:

    - Function: Decomposes the global SDF into a set of localized SDFs with finite ranges.
    - Mechanism: Each support point defines a local SDF whose active range is restricted to the near-surface space around the support point (i.e., "band-limited"—defining values only within a limited bandwidth). The active ranges of multiple local SDFs overlap, and the global SDF value is obtained in the overlapping regions via distance-weighted average fusion. Crucially, the local SDFs do not store latent feature vectors; they are parameterized solely by the shared tiny MLP and their respective geometric transformation parameters. The MLP input is the local coordinates of the query point relative to the support point (after rotation and scaling transformations), and the MLP output is the local SDF value.
    - Design Motivation: In traditional methods, the latent feature vector at each spatial location is typically 16-256 dimensions. In contrast, this method only stores position (3) + rotation (3-4) + scale (3) = 9-10 scalars for each support point, reducing memory by more than an order of magnitude.

2. **Learnable Geometric Attribute Modulation**:

    - Function: Enables the shared MLP to adapt to different geometric shapes in various regions.
    - Mechanism: Each local SDF has three learnable attributes—position $\mathbf{p} \in \mathbb{R}^3$ (support point positions can be fine-tuned during training), rotation $\mathbf{R} \in SO(3)$ (defining the orientation of the local coordinate system), and scale $\mathbf{s} \in \mathbb{R}^3$ (anisotropic scaling to adapt to geometric detail granularity in different directions). The query point $\mathbf{x}$ is first transformed into the local coordinate system $\mathbf{x}' = \text{diag}(\mathbf{s})^{-1} \mathbf{R}^T (\mathbf{x} - \mathbf{p})$, and then input into the shared MLP. Consequently, the same MLP outputs different SDF values under different geometric transformations, resembling an implicit "Instance Normalization."
    - Design Motivation: Geometric transformations provide an extremely compact yet effective conditioning signal. Rotation allows the MLP to adapt to surfaces with different orientations (e.g., vertical walls vs. horizontal ground), and scaling allows the MLP to adapt to different granularities of detail (fine structures vs. flat regions).

3. **Parallel Local SDF Query Algorithm**:

    - Function: Quickly determines which local SDFs' active ranges cover each query point.
    - Mechanism: Since support points are irregularly distributed in 3D space, an efficient spatial index is required to determine the ownership of query points. A hybrid parallel query algorithm based on spatial hashing and KD-Tree is proposed to parallelize the ownership determination of all query points on the GPU, supporting real-time updates (addition/deletion) of support point states during training without reconstructing the entire index structure.
    - Design Motivation: The number of support points can reach hundreds of thousands in large-scale scenes, making point-by-point traversal highly inefficient. The parallel query is a key infrastructure to achieve near-real-time training speeds.

### Loss & Training
- **SDF Supervision Loss**: $\mathcal{L}_{sdf} = \|f(\mathbf{x}) - \hat{s}(\mathbf{x})\|_1$, where $\hat{s}$ is the truncated SDF ground truth.
- **Eikonal Regularization**: $\mathcal{L}_{eik} = (\|\nabla f(\mathbf{x})\|_2 - 1)^2$, constraining the gradient norm of the SDF to be close to 1.
- **Prune-and-Expand Strategy**: During training, support points are adjusted at fixed intervals—removing support points that contribute little to reconstruction (small SDF gradients) and adding new ones in regions with large reconstruction errors. This is analogous to the densification in 3D-GS.

## Key Experimental Results

### Main Results

| Dataset/Scene | Method | Accuracy (cm) ↓ | Completion (cm) ↓ | Memory (MB) ↓ |
|------------|------|-----------------|-------------------|-------------|
| MaiCity | SHINE-Mapping | 2.14 | 1.87 | 312 |
| MaiCity | VDB-Mapping | 1.98 | 1.72 | 285 |
| MaiCity | **Ours** | **1.85** | **1.63** | **56** |
| Newer College | SHINE-Mapping | 3.21 | 2.95 | 487 |
| Newer College | **Ours** | **2.87** | **2.54** | **89** |
| KITTI | **Ours** | **2.42** | **2.18** | **72** |

### Ablation Study

| Configuration | Accuracy (cm) | Memory (MB) | Description |
|------|--------------|-----------|------|
| Full 3D-SLNR | 1.85 | 56 | Full model |
| w/o Rotation Attribute | 2.12 | 54 | Without learning local coordinate system orientation |
| w/o Scaling Attribute | 2.05 | 55 | Without learning anisotropic scaling |
| w/o Prune-and-Expand | 2.31 | 68 | Fixed support point distribution |
| Using Latent Features instead of Geometric Attributes | 1.82 | 243 | Close accuracy but memory inflates by 4.3x |

### Key Findings
- 3D-SLNR requires only 56-89 MB of memory, which is less than **1/5** of SHINE-Mapping, while achieving superior reconstruction accuracy.
- All three components of the learnable geometric attributes contribute: rotation is most critical for irregular surfaces (accuracy drops by 14.6% without it), and scaling is more important for elongated/flat structures.
- Replacing geometric attributes with latent feature vectors only improves accuracy by 1.6% but causes memory to inflate by 4.3x, proving that geometric transformation modulation is an extremely cost-effective alternative.
- The prune-and-expand strategy effectively improves adaptive capacity (improving accuracy by 20%) while also reducing memory (by eliminating unnecessary support points).
- The parallel query algorithm brings training speed to a near-real-time level, updating at approximately 50ms per frame on an NVIDIA RTX 3090.

## Highlights & Insights
- **Extreme Lightweight Design Philosophy**: No latent features, shared tiny MLP, relying solely on geometric transformation modulation to achieve expressiveness. This concept of "replacing features with transformations" is highly elegant and fundamentally changes the storage paradigm of neural representations.
- **No Hash Collisions**: By avoiding reliance on hash encoding and instead using a point-cloud-based support point + local SDF paradigm, this method completely avoids the quality degradation caused by hash collisions in large scenes. This characteristic offers a critical advantage in large-scale mapping.
- **Adaptive Geometric Transformation**: The three learnable attributes of position/rotation/scale allow the shared MLP to be "one network for multiple purposes," a design that is neat and ingenious.
- **Analogy between Prune-and-Expand and 3D-GS**: The dynamic adjustment strategy of support points is conceptually similar to the densification strategy of 3D Gaussian Splatting, allowing them to mutually inspire each other.

## Limitations & Future Work
- Only SDF is used for geometric reconstruction without involving color/texture rendering, meaning it cannot be directly applied to novel view synthesis tasks.
- The support point sampling strategy depends on the quality of the initial point cloud; if the input point cloud has high noise or uneven distribution, the initialization might be sub-optimal.
- The specific thresholds/tactics of the prune-and-expand strategy may require scene-specific hyperparameter tuning.
- The code and arXiv preprint have not been released, limiting reproducibility.
- The paradigm of "shared MLP + geometric transformation conditioning" can be extended to other scenarios requiring lightweight representations (such as 3D compression, streaming, and maps for robot navigation).

## Related Work & Insights
- **vs SHINE-Mapping**: Uses multi-resolution hash encoding to store latent features, which has a large memory footprint and hash collisions. 3D-SLNR has no latent features and no collisions, using <1/5 of the memory.
- **vs Instant-NGP**: Hash encoding + MLP performs well in small scenes but suffers from severe collisions in large scenes. 3D-SLNR avoids collisions by anchoring onto support points.
- **vs Neural Points**: Similarly anchors features on points, but typically stores a feature vector at each point. 3D-SLNR has no feature vectors, replacing them with geometric transformations.
- **vs 3D Gaussian Splatting**: 3D-GS is also a point-based representation, but each Gaussian stores covariance matrices and color parameters. Their densification/pruning strategies can mutually inspire each other.

## Rating
- Novelty: ⭐⭐⭐⭐ The design idea of no latent features + geometric transformation modulation is highly novel, pursuing extreme lightweighting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple datasets, sufficient ablation studies, and highly convincing memory comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and strong motivation arguments.
- Value: ⭐⭐⭐⭐ Provides a highly memory-efficient new solution for large-scale neural mapping, cited 3 times.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IM360: Large-scale Indoor Mapping with 360 Cameras](../../ICCV2025/3d_vision/im360_large-scale_indoor_mapping_with_360_cameras.md)
- [\[CVPR 2025\] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [\[CVPR 2025\] Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset](digital_twin_catalog_a_large-scale_photorealistic_3d_object_digital_twin_dataset.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)

</div>

<!-- RELATED:END -->
