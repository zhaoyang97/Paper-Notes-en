---
title: >-
  [Paper Note] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis
description: >-
  [AAAI 2026][3D Vision][Point Cloud Analysis] This paper analyzes the limitations of conventional graph construction methods (ball query), specifically sparse connectivity at boundary points and noisy connectivity at junction regions, and proposes a graph smoothing module (symmetric adjacency optimization + von Neumann kernel) and a local geometry learning module (adaptive shape features + cylindrical coordinate transformation), achieving competitive performance on classification and segmentation tasks.
tags:
  - AAAI 2026
  - 3D Vision
  - Point Cloud Analysis
  - Graph Smoothing
  - Local Geometry Learning
  - von Neumann Kernel
  - Cylindrical Coordinate System
date: 2026-05-08
content_hash: 16438dc0e24b2d8d
---

# Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis

**Conference**: AAAI 2026
**arXiv**: [2601.11102](https://arxiv.org/abs/2601.11102)
**Code**: [https://github.com/shangboyuan/GSPoint](https://github.com/shangboyuan/GSPoint)
**Area**: 3D Vision / Point Cloud Analysis
**Keywords**: Point Cloud Analysis, Graph Smoothing, Local Geometry Learning, von Neumann Kernel, Cylindrical Coordinate System

## TL;DR

This paper analyzes the limitations of conventional graph construction methods (ball query), specifically sparse connectivity at boundary points and noisy connectivity at junction regions, and proposes a graph smoothing module (symmetric adjacency optimization + von Neumann kernel) and a local geometry learning module (adaptive shape features + cylindrical coordinate transformation), achieving competitive performance on classification and segmentation tasks.

## Background & Motivation

### State of the Field

Point cloud analysis methods are broadly categorized into four classes:
- **Voxel-based methods** (VoxNet, etc.): Convert point clouds into voxel grids and apply 3D convolutions, at the cost of geometric precision.
- **MLP-based methods** (PointNet++, etc.): Operate directly on raw point clouds but are sensitive to noise.
- **Transformer-based methods** (Point Transformer, etc.): Capture global context via self-attention, but incur high computational cost.
- **Graph-based methods** (DGCNN, etc.): Explicitly model inter-point relationships and effectively organize unstructured point clouds.

Graph-based methods are further divided into fixed-graph methods (e.g., Dynamic Graph CNN with adaptive edge weights) and graph learning methods (dynamic graph structures capturing multi-scale relationships).

### Limitations of Prior Work

The authors identify two fundamental problems in conventional graph construction methods, particularly ball query:

#### 1. Sparse Connectivity at Boundary Points

Boundary points (located in high-curvature regions or structural edges) have sparse spatial distributions in their neighborhoods, yielding far fewer neighbors within search radius $r$ than interior points. This leads to:
- Out-degree $d_i^{(out)} \leq d_i^{(in)} \leq k$
- Difficulty propagating boundary geometric features to neighboring regions
- Reduced model capacity to extract discriminative geometric features

#### 2. Noisy Connectivity at Junction Regions

Junction points (at the interface of different instances) reside in high local point density areas where distinct instances are close in Euclidean distance. Since ball query relies solely on Euclidean distance, points from different instances are included in the same neighborhood, producing cross-instance noisy connections. For example, points on a fuselage may incorrectly include wing points in their neighborhood.

### Root Cause

The fixed graph structure of ball query leads to imbalanced degree distributions: boundary points have excessively low out-degree (sparse connectivity), while junction points have excessively high out-degree (noisy connectivity), limiting the extraction of discriminative features.

### Starting Point

The paper approaches the problem from the perspective of **graph structure optimization**: (1) balancing degree distributions via symmetrization and normalization; (2) leveraging multi-hop relationships (von Neumann kernel) to enhance boundary point connectivity and suppress noisy connections; (3) extracting richer local geometric information on the optimized graph structure.

## Method

### Overall Architecture

GSPoint adopts a hierarchical downsampling architecture, where each block consists of two core modules:
1. **Graph Smoothing Module**: Symmetric adjacency optimization + finite-step graph smoothing (von Neumann kernel) → Top-K selection for neighborhood refinement.
2. **Local Geometry Learning Module**: Adaptive shape features (covariance matrix eigenvalues) + cylindrical coordinate transformation (distributional features) → enhanced feature extraction.

### Key Designs

#### 1. Graph Smoothing Module

**Function**: Optimizes the graph structure constructed by ball query to balance the degree distribution at boundary and junction points.

**Step 1 – Symmetric Adjacency Optimization**:

$$\mathbf{A}_{sym} = \left\lfloor \frac{\mathbf{A} + \mathbf{A}^\top}{2} \right\rfloor$$

This eliminates directional connectivity, ensuring $d_u^{(in)} = d_u^{(out)}$. Symmetric normalization is then applied:

$$\tilde{\mathbf{A}} = \mathbf{D}^{-1/2} \mathbf{A}_{sym} \mathbf{D}^{-1/2}$$

The normalized weight $\tilde{a}_{ij} = 1/\sqrt{d_i d_j}$ assigns **higher weights to low-degree (boundary) points and suppresses high-degree (junction) points**.

**Step 2 – Multi-hop Relationship Modeling**:

Using $\tilde{\mathbf{A}}^T$ directly has two drawbacks: (1) numerical instability at high orders; (2) only paths of exactly length $T$ are considered, ignoring shorter paths. The von Neumann kernel is introduced:

$$K_{NEU} = (I - \alpha \tilde{\mathbf{A}})^{-1} = \lim_{T \to \infty} \sum_{t=0}^{T} (\alpha \tilde{\mathbf{A}})^t$$

A finite-step approximation is used in practice:

$$\mathbf{S}_T = \sum_{t=0}^{T} (\alpha \tilde{\mathbf{A}})^t, \quad \alpha \in (0,1)$$

**Core Property**: Due to degree distribution characteristics, the propagation weight $(\tilde{\mathbf{A}}^T)_{vj}$ from a low-degree boundary point $\mathbf{p}_v$ to any point $\mathbf{p}_j$ within a $T$-hop neighborhood is higher than the corresponding weight $(\tilde{\mathbf{A}}^T)_{uj}$ from a high-degree junction point $\mathbf{p}_u$. This **naturally enhances boundary point connectivity while suppressing junction noise**.

The refined neighborhood $\mathcal{N}'(i)$ is obtained by applying row-wise Top-K selection on $\mathbf{S}_T$.

**Design Motivation**:
- Symmetrization eliminates directional inconsistency in ball query
- Normalization amplifies the influence of low-degree (boundary) points
- The von Neumann kernel accounts for all path lengths from 1 to $T$, providing greater stability than $\tilde{\mathbf{A}}^T$ alone
- $\alpha$ controls the trade-off between local consistency and global connectivity

#### 2. Local Geometry Learning Module

**Function**: Extracts richer geometric features over the refined neighborhood.

**2a. Adaptive Shape Features**:

Eigenvalue decomposition is applied to the neighborhood covariance matrix of each point $\mathbf{p}_i$:

$$\mathbf{C}_i = \mathbf{V}_i \boldsymbol{\Lambda}_i \mathbf{V}_i^\top$$

The eigenvalues $\lambda^{(1)} \geq \lambda^{(2)} \geq \lambda^{(3)}$ encode rich geometric information (planarity, sphericity, linearity, etc.). Since fixed descriptors lack flexibility, a **learnable MLP** $\phi(\boldsymbol{\Lambda})$ maps the eigenvalues to adaptive shape features.

**2b. Cylindrical Coordinate Distribution Features**:

Neighborhood points are transformed from Cartesian to cylindrical coordinates:
1. Displacement vectors $\Delta\mathbf{p}_j = \mathbf{p}_j - \mathbf{p}_i$ are projected onto the three principal axes.
2. Converted to cylindrical coordinates $(h', \omega', \cos\theta)$: $h'$ quantifies axial anisotropy, $\omega'$ describes radial distance distribution.
3. Height and radial distances are normalized.

**Design Motivation**:
- Classical geometric descriptors (planarity, sphericity, etc.) are hand-crafted and lack scale flexibility for complex structures.
- A learnable network can adaptively extract the most task-relevant information from eigenvalues.
- Cylindrical coordinates capture **anisotropy and distance distribution** in the neighborhood more effectively than Cartesian coordinates.
- The two feature types are complementary: shape features describe the local geometry "type," while distributional features describe the spatial layout pattern of the neighborhood.

#### Enhanced Feature Extraction Function

$$\mathbf{x}_i^{(l+1)} = \mathcal{A}\left(\sigma\left(\psi'([\mathbf{x}_j^{(l)} \| (\mathbf{p}_i - \mathbf{p}_j) \| \mathbf{p}'_j^{(l)}])_{j \in \mathcal{N}'(i)}\right)\right) \| \phi(\boldsymbol{\Lambda}_i)$$

Compared to the standard 3D relative coordinates $(\mathbf{p}_i - \mathbf{p}_j)$, the formulation adds 3D cylindrical coordinates $\mathbf{p}'$, uses a mapping function $\psi': \mathbb{R}^{\eta+6} \to \mathbb{R}^{\eta'}$, and concatenates shape features $\phi(\boldsymbol{\Lambda})$.

### Loss & Training

- Classification: Cross-entropy loss
- ModelNet40: 1024 points without normals, random translation augmentation, 500 epochs
- ScanObjectNN: Random scaling and rotation augmentation, 250 epochs
- ShapeNetPart: 2048-point sampling, random scaling and jittering, 300 epochs
- S3DIS: Voxel downsampling at 0.04m, random scaling/rotation/jittering, 100 epochs

## Key Experimental Results

### Main Results

#### Classification

| Method | ModelNet40 OA (%) | ScanObjectNN OA (%) |
|--------|-------------------|---------------------|
| PointNet++ | 91.9 | 73.7 |
| PointMLP | 94.1 | 85.4 |
| PointNeXt | 93.2 | 87.7 |
| PointGPT-S | 94.0 | 86.9 |
| **GSPoint** | **94.5** | **88.1** |

#### Segmentation

| Method | ShapeNetPart Ins.mIoU (%) | S3DIS mIoU (%) |
|--------|---------------------------|----------------|
| PointNeXt | 87.0 | 70.5 |
| GSLCN | 87.1 | 68.1 |
| PointWavelet | 86.8 | 71.3 |
| **GSPoint** | **87.2** | **71.5** |

### Ablation Study

| Config | ModelNet40 OA | ScanObjectNN OA | ShapeNetPart | S3DIS |
|--------|---------------|-----------------|--------------|-------|
| A: Baseline | 92.6 | 86.9 | 86.5 | 68.2 |
| D: SA+GS | 93.9 | 87.3 | 87.0 | 70.2 |
| G: Λ+p' | 93.6 | 87.1 | 86.9 | 69.8 |
| H: SA+GS+Λ | 94.0 | 87.5 | 87.1 | 70.9 |
| **J: Full** | **94.5** | **88.1** | **87.2** | **71.5** |

#### Graph Smoothing as a Plug-and-Play Module

| Base Method | + GSPoint Graph Smoothing (OA Gain) |
|-------------|--------------------------------------|
| PointNet++ | +1.1 (ModelNet40), **+9.2** (ScanObjectNN) |
| PointMLP | +0.3, +0.4 |
| PointNeXt | +0.6, +0.2 |

### Key Findings

1. **SA+GS synergy outperforms individual components**: Symmetric adjacency optimization and graph smoothing must be used together to achieve full effectiveness.
2. **Largest gains on S3DIS** (+3.3% mIoU): Boundary/junction issues are more pronounced in large-scale indoor scenes.
3. **Highly effective as a plug-and-play module**: Notably, applying graph smoothing to PointNet++ on ScanObjectNN yields a +9.2% improvement, indicating that the original method is severely constrained by graph structure quality.
4. **Robustness to hyperparameters**: Performance is stable within $\alpha \in [0.4, 0.6]$ and $T \in [3, 4]$.
5. **Visualization confirms correctness**: After graph smoothing, refined neighborhoods are more accurate — fuselage points no longer include wing points.

## Highlights & Insights

1. **Thorough problem analysis**: Starting from degree imbalance in ball query, the authors rigorously derive the causes of sparse and noisy connectivity through mathematical analysis — this analysis itself constitutes a contribution.
2. **Elegant introduction of the von Neumann kernel**: A classical tool from graph spectral theory is applied to the problem of point cloud neighborhood construction, with a solid theoretical foundation.
3. **Meaningful cylindrical coordinate design**: Establishing a canonical coordinate system along principal component directions captures anisotropic structures more effectively than Euclidean coordinates.
4. **Generality as a plug-and-play component**: The graph smoothing module is backbone-agnostic and broadly applicable to a variety of graph-based methods.

## Limitations & Future Work

1. **Computational overhead of the von Neumann kernel**: Computing $\mathbf{S}_T$ and performing Top-K selection introduces additional cost; the paper does not provide a detailed efficiency analysis.
2. **Limited gains on ModelNet40** (+1.9%): The synthetic dataset is relatively simple, where boundary/junction issues are less prominent.
3. **No comparison with large-scale outdoor scene methods** (e.g., KITTI, nuScenes): Point cloud density and noise patterns may differ significantly in real-world applications.
4. **Differentiability of eigenvalue decomposition**: The handling of gradients in the presence of repeated eigenvalues is not discussed.
5. Self-supervised pre-training could be explored to improve generalization to unseen categories.

## Related Work & Insights

- **DGCNN** (Wang et al., 2019): Dynamic graph convolution; this paper analyzes the limitations of its Euclidean-distance-based graph construction.
- **PointNeXt** (Qian et al., 2022): Improves PointNet++ training strategies; the proposed graph smoothing further enhances its performance.
- **Graph Spectral Theory**: The von Neumann kernel originates from the graph spectral learning literature; this paper introduces it into point cloud graph construction.
- **Insight**: Many point cloud methods focus on convolution or attention design while overlooking **the quality of the graph structure itself** — optimizing graph structure may serve as a general and broadly applicable improvement strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ — In-depth problem analysis and theoretically grounded introduction of the von Neumann kernel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four benchmarks, detailed ablations, and plug-and-play validation provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from motivation analysis to methodological derivation is exceptionally clear.
- Value: ⭐⭐⭐⭐ — The generality and plug-and-play nature of the graph smoothing module offer practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](../../CVPR2026/3d_vision/adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[CVPR 2026\] ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis](../../CVPR2026/3d_vision/learning_coordinate-based_convolutional_kernels_for_continuous_se3_equivariant_a.md)
- [\[AAAI 2026\] DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression](deepraht_learning_predictive_raht_for_point_cloud_attribute_compression.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](../../ICCV2025/3d_vision/upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
