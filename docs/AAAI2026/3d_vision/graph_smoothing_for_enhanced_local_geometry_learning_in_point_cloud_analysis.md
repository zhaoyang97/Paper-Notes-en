---
title: >-
  [Paper Note] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis
description: >-
  [AAAI 2026][3D Vision][point cloud analysis] Analyzes the issues of conventional graph construction methods (such as ball query) generating sparse connections at boundary points and noisy connections at intersection areas. Proposes a graph smoothing module (symmetric adjacency optimization + von Neumann kernel) and a local geometry learning module (adaptive shape features + cylindrical coordinate transformation), achieving competitive performance on classification and segment…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "point cloud analysis"
  - "graph smoothing"
  - "local geometry learning"
  - "von Neumann kernel"
  - "cylindrical coordinate system"
date: 2026-05-08
content_hash: 1718ab7b4301d4b9
---

# Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis

**Conference**: AAAI 2026  
**arXiv**: [2601.11102](https://arxiv.org/abs/2601.11102)  
**Code**: [https://github.com/shangboyuan/GSPoint](https://github.com/shangboyuan/GSPoint)  
**Area**: 3D Vision / Point Cloud Analysis  
**Keywords**: point cloud analysis, graph smoothing, local geometry learning, von Neumann kernel, cylindrical coordinate system

## TL;DR

Analyzes the issues of conventional graph construction methods (such as ball query) generating sparse connections at boundary points and noisy connections at intersection areas. Proposes a graph smoothing module (symmetric adjacency optimization + von Neumann kernel) and a local geometry learning module (adaptive shape features + cylindrical coordinate transformation), achieving competitive performance on classification and segmentation tasks.

## Background & Motivation

### Background

Point cloud analysis methods are primarily divided into four categories:
- **Voxel-based methods** (e.g., VoxNet): Convert point clouds to voxel grids to apply 3D convolutions, sacrificing geometric precision.
- **MLP-based methods** (e.g., PointNet++): Directly operate on raw point clouds but are sensitive to noise.
- **Transformer-based methods** (e.g., Point Transformer): Capture global context via self-attention but suffer from high computational overhead.
- **Graph-based methods** (e.g., DGCNN): Explicitly model relations between points and effectively organize unstructured point clouds.

Graph-based methods are further divided into fixed graph methods (e.g., adaptive edge weights in Dynamic Graph CNN) and graph learning methods (dynamic graph structures capturing multi-scale relationships).

### Limitations of Prior Work

The authors identify two fundamental issues in conventional graph construction methods (especially ball query):

#### 1. Sparse Connections at Boundary Points

The spatial distribution of points around boundary points (high-curvature regions or structural edges) is sparse, with far fewer neighbors within the search radius $r$ compared to interior points. This leads to:
- Out-degree $d_i^{(out)} \leq d_i^{(in)} \leq k$
- The difficulty of propagating boundary geometric features to neighboring regions
- Degradation in the model's ability to extract discriminative geometric features

#### 2. Noisy Connections in Intersection Areas

Intersection points (where different instances meet) exhibit high local point density, and different instances are spatially close in Euclidean distance. Since ball query relies solely on Euclidean distance, it incorporates points from different instances into the same neighborhood, creating cross-instance noisy connections. For example, points on the fuselage might erroneously include points on the wings in their neighborhood.

### Key Challenge

The fixed graph structure of ball query leads to an imbalanced degree distribution: the out-degree of boundary points is too low (sparse connections), while the out-degree of intersection points is too high (noisy connections), which limits the extraction of discriminative features.

### Key Insight

Approaching from the perspective of **graph structure optimization**: (1) balancing the degree distribution through symmetrization and normalization; (2) using multi-hop relationships (von Neumann kernel) to enhance boundary point connections and suppress noisy connections; (3) extracting richer local geometric information over the optimized graph structure.

## Method

### Overall Architecture

GSPoint adopts a hierarchical downsampling architecture, where each block contains two core modules:
1. **Graph Smoothing Module**: Symmetric adjacency optimization + finite-step graph smoothing (von Neumann kernel) $\to$ Top-K selection to optimize neighborhood.
2. **Local Geometry Learning Module**: Adaptive shape features (eigenvalues of covariance matrix) + cylindrical coordinate transformation (distribution features) $\to$ Enhanced feature extraction.

### Key Designs

#### 1. Graph Smoothing Module

**Function**: Optimize the graph structure constructed by ball query to balance the degree distribution of boundary points and intersection points.

**Step 1 - Symmetric Adjacency Optimization**:

$$\mathbf{A}_{sym} = \left\lfloor \frac{\mathbf{A} + \mathbf{A}^\top}{2} \right\rfloor$$

Eliminate directional connections to achieve $d_u^{(in)} = d_u^{(out)}$. Then perform symmetric normalization:

$$\tilde{\mathbf{A}} = \mathbf{D}^{-1/2} \mathbf{A}_{sym} \mathbf{D}^{-1/2}$$

The normalized weight is $\tilde{a}_{ij} = 1/\sqrt{d_i d_j}$, where **low-degree points (boundaries) obtain higher weights, and high-degree points (intersections) are suppressed**.

**Step 2 - Multi-hop Relationship Modeling**:

Directly using $\tilde{\mathbf{A}}^T$ poses two issues: (1) high-order numerical instability; (2) only considering paths of exact length T while ignoring shorter paths. The von Neumann kernel is introduced:

$$K_{NEU} = (I - \alpha \tilde{\mathbf{A}})^{-1} = \lim_{T \to \infty} \sum_{t=0}^{T} (\alpha \tilde{\mathbf{A}})^t$$

In practice, a finite-step approximation is used:

$$\mathbf{S}_T = \sum_{t=0}^{T} (\alpha \tilde{\mathbf{A}})^t, \quad \alpha \in (0,1)$$

**Core Property**: Due to the degree distribution characteristics, the propagation weight $(\tilde{\mathbf{A}}^T)_{vj}$ of a low-degree boundary point $\mathbf{p}_v$ to any point $\mathbf{p}_j$ in the T-hop neighborhood is higher than $(\tilde{\mathbf{A}}^T)_{uj}$ of a high-degree point $\mathbf{p}_u$. This **naturally enhances boundary point connections and suppresses noisy connections at intersection points**.

Finally, the optimized neighborhood $\mathcal{N}'(i)$ is obtained by performing Top-K selection on each row of $\mathbf{S}_T$.

**Design Motivation**:
- Symmetrization eliminates directional inconsistency in ball query.
- Normalization allows low-degree points (boundaries) to obtain stronger influence.
- The von Neumann kernel comprehensively considers paths of all lengths (from 1 to T), being more stable than $\tilde{\mathbf{A}}^T$ alone.
- $\alpha$ controls the trade-off between local consistency and global connectivity.

#### 2. Local Geometry Learning Module

**Function**: Extract richer geometric features over the optimized neighborhood.

**2a. Adaptive Shape Features**:

Perform eigenvalue decomposition on the neighborhood covariance matrix of each point $\mathbf{p}_i$:

$$\mathbf{C}_i = \mathbf{V}_i \boldsymbol{\Lambda}_i \mathbf{V}_i^\top$$

The eigenvalues $\lambda^{(1)} \geq \lambda^{(2)} \geq \lambda^{(3)}$ contain rich geometric information (planarity, sphericity, linearity, etc.). Since static descriptors are not flexible enough, a **learnable MLP** $\phi(\boldsymbol{\Lambda})$ is used to map the eigenvalues into adaptive shape features.

**2b. Cylindrical Coordinate Distribution Features**:

Transform neighborhood points from Cartesian coordinates to a cylindrical coordinate system:
1. Project the displacement vector $\Delta\mathbf{p}_j = \mathbf{p}_j - \mathbf{p}_i$ onto three principal axes.
2. Convert to cylindrical coordinates $(h', \omega', \cos\theta)$: $h'$ quantifies axial anisotropy, and $\omega'$ describes the radial distance distribution.
3. Normalize the height and radial distance.

**Design Motivation**:
- Classical geometric descriptors (planarity, sphericity, etc.) are hand-crafted, lacking scale flexibility in complex structures.
- The learnable network can adaptively extract descriptors that are most useful for downstream tasks from the eigenvalues.
- Cylindrical coordinates capture the **anisotropy and distance distribution** of the neighborhood better than Cartesian coordinates.
- The two features are complementary: shape features depict the local geometric "type", while distribution features describe the spatial distribution pattern of the neighborhood.

#### Enhanced Feature Extraction Function

$$\mathbf{x}_i^{(l+1)} = \mathcal{A}\left(\sigma\left(\psi'([\mathbf{x}_j^{(l)} \| (\mathbf{p}_i - \mathbf{p}_j) \| \mathbf{p}'_j^{(l)}])_{j \in \mathcal{N}'(i)}\right)\right) \| \phi(\boldsymbol{\Lambda}_i)$$

Compared to standard 3D relative coordinates $(\mathbf{p}_i - \mathbf{p}_j)$, 3D cylindrical coordinates $\mathbf{p}'$ are added, with mapping function $\psi': \mathbb{R}^{\eta+6} \to \mathbb{R}^{\eta'}$, and the shape feature $\phi(\boldsymbol{\Lambda})$ is concatenated.

### Loss & Training

- Classification: Cross-entropy loss
- ModelNet40: 1024 points without normals, random translation augmentation, 500 epochs
- ScanObjectNN: Random scaling and rotation augmentation, 250 epochs
- ShapeNetPart: 2048 points sampled, random scaling and jitter, 300 epochs
- S3DIS: Voxel downsampling of 0.04m, random scaling/rotation/jitter, 100 epochs

## Key Experimental Results

### Main Results

#### Classification Tasks

| Method | ModelNet40 OA (%) | ScanObjectNN OA (%) |
|------|-----------------|-------------------|
| PointNet++ | 91.9 | 73.7 |
| PointMLP | 94.1 | 85.4 |
| PointNeXt | 93.2 | 87.7 |
| PointGPT-S | 94.0 | 86.9 |
| **GSPoint** | **94.5** | **88.1** |

#### Segmentation Tasks

| Method | ShapeNetPart Ins.mIoU (%) | S3DIS mIoU (%) |
|------|------------------------|--------------|
| PointNeXt | 87.0 | 70.5 |
| GSLCN | 87.1 | 68.1 |
| PointWavelet | 86.8 | 71.3 |
| **GSPoint** | **87.2** | **71.5** |

### Ablation Study

| Configuration | ModelNet40 OA | ScanObjectNN OA | ShapeNetPart | S3DIS |
|------|-------------|----------------|-------------|-------|
| A: Baseline | 92.6 | 86.9 | 86.5 | 68.2 |
| D: SA+GS | 93.9 | 87.3 | 87.0 | 70.2 |
| G: Λ+p' | 93.6 | 87.1 | 86.9 | 69.8 |
| H: SA+GS+Λ | 94.0 | 87.5 | 87.1 | 70.9 |
| **J: Full** | **94.5** | **88.1** | **87.2** | **71.5** |

#### Graph Smoothing as a Plug-and-Play Module

| Base Method | + GSPoint Graph Smoothing (OA Gain) |
|---------|----------------------|
| PointNet++ | +1.1 (ModelNet40), **+9.2** (ScanObjectNN) |
| PointMLP | +0.3, +0.4 |
| PointNeXt | +0.6, +0.2 |

### Key Findings

1. **Synergistic effect of SA+GS outperforms individual usage**: Symmetric adjacency optimization and graph smoothing must cooperate to maximize their effectiveness.
2. **Largest improvement on S3DIS** (+3.3% mIoU): In large-scale indoor scenes, boundary/intersection issues are more prominent.
3. **Extremely effective as a plug-and-play module**: Particularly on PointNet++, ScanObjectNN improves by 9.2%, indicating that the raw method is severely limited by graph structure.
4. **Insensitive to hyperparameters**: Performance remains stable within the range of $\alpha \in [0.4, 0.6]$ and $T \in [3,4]$.
5. **Visual validation**: The neighborhood after graph smoothing is indeed more accurate—fuselage points no longer incorporate wing points.

## Highlights & Insights

1. **Thorough problem analysis**: Starting from the degree imbalance of ball query, the causes of sparse and noisy connections are rigorously analyzed using mathematical derivation. This analysis alone is highly valuable.
2. **Ingenious introduction of the von Neumann kernel**: Leverages classical tools from graph spectral theory to solve point cloud neighborhood construction, establishing a solid theoretical foundation.
3. **Meaningful design of cylindrical coordinate transformation**: Establishes a canonical coordinate system using principal component directions, capturing anisotropic structures better than directly using Euclidean coordinates.
4. **Plug-and-play generality**: The graph smoothing module is independent of specific backbones and can be widely applied to various graph-based methods.

## Limitations & Future Work

1. **von Neumann kernel requires calculating $\mathbf{S}_T$ and Top-K selection**: This introduces additional computational overhead, which is not thoroughly analyzed of its efficiency in the paper.
2. **Relatively limited improvement on ModelNet40** (+1.9%): Synthetic datasets are relatively simple, making boundary/intersection issues less prominent.
3. **No comparison with methods on large-scale outdoor scenes (e.g., KITTI, nuScenes)**: Point cloud density and noise patterns may differ in practical applications.
4. **Differentiability of eigenvalue decomposition**: The gradient propagation issue under identical eigenvalues is not discussed.
5. Future work can explore self-supervised pre-training to enhance generalization capabilities on unseen categories.

## Related Work & Insights

- **DGCNN** (Wang et al., 2019): Dynamic graph convolution; this work analyzes the shortcomings of its Euclidean distance-based graph construction.
- **PointNeXt** (Qian et al., 2022): Improves the training strategies of PointNet++; the proposed graph smoothing can further enhance its performance.
- **Graph Spectral Theory**: The von Neumann kernel originates from graph spectral learning, which this work introduces into point cloud graph construction.
- **Insight**: Many point cloud methods focus on the design of convolution/attention but ignore the **quality of the graph structure itself**—optimizing the graph structure could be a universal way to gain performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — Deep problem analysis; the introduction of the von Neumann kernel is theoretically novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Full coverage across four benchmarks, detailed ablations, and plug-and-play validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Highly clear logical chain from motivation analysis to methodology derivation.
- Value: ⭐⭐⭐⭐ — The generality and plug-and-play characteristics of the graph smoothing module carry practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 4D Local Modeling Toward Dynamic Global Perception for Ambiguity-free Rotation-Invariant Point Cloud Analysis](../../CVPR2026/3d_vision/4d_local_modeling_toward_dynamic_global_perception_for_ambiguity-free_rotation-i.md)
- [\[ICLR 2026\] Spiking Discrepancy Transformer for Point Cloud Analysis](../../ICLR2026/3d_vision/spiking_discrepancy_transformer_for_point_cloud_analysis.md)
- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](../../CVPR2026/3d_vision/adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[CVPR 2026\] ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis](../../CVPR2026/3d_vision/learning_coordinate-based_convolutional_kernels_for_continuous_se3_equivariant_a.md)
- [\[ICLR 2026\] 3DSMT: A Hybrid Spiking Mamba-Transformer for Point Cloud Analysis](../../ICLR2026/3d_vision/3dsmt_a_hybrid_spiking_mamba-transformer_for_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
