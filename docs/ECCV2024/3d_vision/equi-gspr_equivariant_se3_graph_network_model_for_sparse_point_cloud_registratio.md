---
title: >-
  [Paper Note] Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration
description: >-
  [ECCV 2024][3D Vision][Point cloud registration] This paper proposes Equi-GSPR, a sparse point cloud registration method based on SE(3) equivariant graph neural networks. By utilizing equivariant message passing, low-rank feature transformation (LRFT), and implicit feature space similarity matching, it achieves SOTA registration performance on indoor and outdoor datasets with low model complexity.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Point cloud registration"
  - "SE(3) equivariance"
  - "Graph neural networks"
  - "Feature descriptors"
  - "Similarity matching"
date: 2026-05-08
content_hash: cac79a7166c98611
---

# Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration

**Conference**: ECCV 2024  
**arXiv**: [2410.05729](https://arxiv.org/abs/2410.05729)  
**Code**: [https://github.com/alexandor91/se3-equi-graph-registration](https://github.com/alexandor91/se3-equi-graph-registration)  
**Area**: 3D Vision  
**Keywords**: Point cloud registration, SE(3) equivariance, Graph neural networks, Feature descriptors, Similarity matching

## TL;DR

This paper proposes Equi-GSPR, a sparse point cloud registration method based on SE(3) equivariant graph neural networks. By utilizing equivariant message passing, low-rank feature transformation (LRFT), and implicit feature space similarity matching, it achieves SOTA registration performance on indoor and outdoor datasets with low model complexity.

## Background & Motivation

**Background**: Point cloud registration is a fundamental task in 3D alignment and reconstruction. In recent years, deep learning-based methods (DGR, PointDSC, Predator) have made significant progress, but they mostly rely on precise point-to-point correspondence supervision or complex outlier rejection.

**Limitations of Prior Work**:
   - Correspondences established from raw point clouds often suffer from an extremely high outlier-to-inlier ratio, leading to severe registration errors.
   - Geometric feature descriptors ignore the global topological connectivity of the data and the SE(3) rotation equivariance.
   - Methods introducing rotational equivariance, such as RoReg, suffer from extreme computational burdens (e.g., 30 minutes per registration), yielding poor real-time performance.
   - Feature matching-based methods require explicit point-to-point search processes, making them vulnerable to a large number of outliers.

**Key Challenge**: How to fully exploit the inherent symmetry (rotational equivariance) of point cloud data to improve registration accuracy while maintaining low computational complexity?

**Goal**: Design an efficient point cloud registration model utilizing SE(3) equivariance without explicit correspondence supervision.

**Key Insight**: Capture topological and geometric features of point clouds using graph convolutions, learn equivariant feature representations through SE(3) message passing, and perform similarity matching in an implicit feature space instead of explicit point searching.

**Core Idea**: Learn rotation-equivariant features via SE(3) equivariant graph networks, and combine low-rank feature transformation compression with implicit similarity matching to achieve robust sparse point cloud registration without explicit correspondence supervision.

## Method

### Overall Architecture

Input: $N$ sparsely sampled points (1024) for both source and target frames $\rightarrow$ Feature descriptor extraction (PointNet++ style MLP) $\rightarrow$ SE(3) equivariant graph convolutional layers to learn equivariant features $\rightarrow$ Concatenate node features and coordinate embeddings $\rightarrow$ LRFT to compress feature dimension $\rightarrow$ Calculate dot-product similarity matrix $\rightarrow$ Similarity weighted aggregation + decoding to predict relative transformation (translation + quaternion rotation).

### Key Designs

1. **Feature Descriptor Module (Feature Descriptor)**:

    - **Function**: Extract initial geometric features from the neighborhood of sparsely sampled points.
    - **Mechanism**: Use a PointNet++ style shallow MLP to compute point features through neighborhood feature aggregation:
    $\vec{h}_i^{l_1+1} = \frac{1}{n}\sum_{k \in \mathcal{N}(i)} f_h(\vec{h}_k^{l_1}, \vec{x}_k - \vec{x}_i)$
   Supports being replaced by pretrained descriptors (such as FCGF).
    - **Design Motivation**: The modular design allows the model to flexibly use self-trained or pretrained descriptors, and end-to-end trained descriptors are better suited for the subsequent equivariant graph layers.

2. **SE(3) Equivariant Graph Network Layer (Equivariant Graph Network)**:

    - **Function**: Enhance the receptive field and SE(3) equivariance of feature descriptors through equivariant message passing.
    - **Mechanism**: Based on the equivariant graph representation by Satorras et al., three quantities are updated in each layer:
        - **Message update**: $\vec{m}_{ik} = \phi_m(\vec{h}_i^{l_2}, \vec{h}_k^{l_2}, \|\vec{x}_k^{l_2} - \vec{x}_i^{l_2}\|^{1/2})$
        - **Coordinate embedding update**: $\vec{x}_i^{l_2+1} = \vec{x}_i^{l_2} + C\sum_{k \in \mathcal{N}(i)} \exp(\vec{x}_k^{l_2} - \vec{x}_i^{l_2})\phi_x(\text{proj}_{\vec{\mathcal{F}}_{ik}}\vec{m}_{ik})$
        - **Latent feature update**: $\vec{h}_i^{l_2+1} = \phi_h(\vec{h}_i^{l_2}, \sum_{k \in \mathcal{N}(i)} \text{proj}_{\vec{\mathcal{F}}_{ik}}\vec{m}_{ik})$
    - **Local Equivariant Reference Frame**: Following ClofNet's approach, a local equivariant frame $\vec{\mathcal{F}}_{ik} = (\vec{a}_{ik}, \vec{b}_{ik}, \vec{c}_{ik})$ is constructed using pairwise coordinate embeddings. Projecting messages onto this frame preserves SO(3) invariance:
    $\hat{\vec{m}}_{ik} = x_{ik}^{\vec{a}}\vec{a}_{ik} + x_{ik}^{\vec{b}}\vec{b}_{ik} + x_{ik}^{\vec{c}}\vec{c}_{ik}$
    - **Design Motivation**: Equivariant features allow the model to learn more efficiently from data symmetries. Neighborhood search is restricted to a local range (ball query radius of 0.3m), reducing the graph adjacency matrix complexity from $O(n^2)$ to $O(n)$. Four equivariant graph layers are used.

3. **Low-Rank Feature Transformation (LRFT)**:

    - **Function**: Compress the number of features to improve the reliability and computational efficiency of similarity matching.
    - **Mechanism**: Inspired by LoRA, two stacked linear layers are used to construct a low-rank constrained mapping:
    $\hat{\vec{H}}_{src}, \hat{\vec{H}}_{tar} = (\vec{A}\vec{B})^T(\vec{H}_{src}, \vec{H}_{tar})$
   Where $\vec{A} \in \mathbb{R}^{N \times r}$, $\vec{B} \in \mathbb{R}^{r \times N'}$, and $r \ll \min(N, N')$. The configuration is 1024/(32+3)/128.
    - **Design Motivation**: The low-rank constraint captures the essential feature correlation between descriptors (matrix low-rank theorem), making matching more reliable on more compact features while reducing subsequent similarity computation overhead.

4. **Similarity Computation & Rank Verification (Similarity + Rank Verification)**:

    - **Function**: Calculate the feature similarity matrix between the source and target frames, and remove outlier correspondences using rank regularization and sub-matrix verification.
    - **Mechanism**: Compute dot-product similarity after normalization: $\vec{S}_{ij} = \langle\hat{\vec{h}}_i \cdot \hat{\vec{h}}_j\rangle$. Rank regularization ensures the rank of the similarity matrix is close to $r$:
    $\mathcal{L}_{Reg} = |(\text{Trace}(\hat{\vec{S}}^T\hat{\vec{S}}))^{1/2} - r|$
    - For each matching element $\hat{\vec{S}}_{ij}$, the determinant of a 7x7 sub-matrix (5x5 for boundaries) centered around it is checked to verify local consistency. Invalid matching rows are set to zero.
    - **Design Motivation**: Matching in an implicit feature space eliminates the need for explicit point correspondence search, bypassing interference from a large number of outliers. The rank constraint automatically identifies and suppresses outlier correspondences.

### Loss & Training

- **Total Loss**: $\mathcal{L}_{total} = \mathcal{L}_{rot} + \mathcal{L}_{trans} + \beta\mathcal{L}_{Reg}$, $\beta = 0.05$
- **Rotation Loss**: $\mathcal{L}_{rot} = \arccos\frac{\text{Trace}(\hat{\vec{R}}^T\vec{R}^*) - 1}{2}$ (measured in radians)
- **Translation Loss**: $\mathcal{L}_{trans} = \|\hat{\vec{t}} - \vec{t}^*\|^2$
- **Training Details**: Single RTX 3090, voxel downsampling to 1024 points (5cm for 3DMatch, 30cm for KITTI), 16 nearest neighbors used for graph construction, node feature dimension 32, coordinate embedding dimension 3.

## Key Experimental Results

### Indoor 3DMatch Benchmark

| Method | RE(°)↓ | TE(cm)↓ | RR(%)↑ | F1(%)↑ | Time(s)↓ |
|------|--------|---------|--------|--------|----------|
| RANSAC-100k+refine | 2.17 | 6.76 | 92.30 | 81.43 | 5.51 |
| DGR | 2.40 | 7.48 | 91.30 | 89.76 | 1.36 |
| SpinNet | 1.93 | 6.24 | 93.74 | 92.07 | 2.84 |
| PointDSC | 2.06 | 6.55 | 93.28 | 89.35 | 0.09 |
| RoReg | 1.84 | 6.28 | 93.70 | 91.60 | 2226 |
| **Equi-GSPR (Ours)** | **1.67** | **5.68** | **94.60** | **94.35** | **0.12** |

### Outdoor KITTI Benchmark

| Method | RE(°)↓ | TE(cm)↓ | RR(%)↑ | F1(%)↑ | Time(s)↓ |
|------|--------|---------|--------|--------|----------|
| RANSAC-100k+refine | 1.28 | 18.42 | 77.20 | 74.07 | 15.65 |
| DGR | 1.45 | 14.60 | 76.62 | 73.84 | 0.86 |
| SpinNet | 1.08 | 10.75 | 82.83 | 80.91 | 3.46 |
| PointDSC | 1.63 | 12.31 | 74.41 | 70.08 | 0.31 |
| **Equi-GSPR (Ours)** | **0.92** | **8.74** | **83.83** | **85.09** | **0.14** |

### Comparison with Different Sampling Points (3DMatch RR%)

| Points | 4096 | 2048 | 1024 | 512 | 256 | Average |
|--------|------|------|------|-----|-----|------|
| FCGF-Reg | 91.7 | 90.3 | 89.5 | 85.7 | 80.5 | 87.5 |
| SpinNet | 93.8 | 93.6 | 93.7 | 89.5 | 85.7 | 91.3 |
| **Ours** | **95.3** | **94.8** | **94.6** | **91.3** | **88.5** | **92.9** |

### Ablation Study

| Configuration | RE(°)↓ | RR(%)↑ | Description |
|------|--------|--------|------|
| FCGF Descriptor + Ours | 1.62 | 93.87 | Pretrained descriptors can also work |
| FPHF Descriptor + Ours | 1.83 | 83.62 | Handcrafted descriptors drop in performance |
| w/o descriptor layer | 10.26 | 61.39 | Descriptor layer is crucial |
| w/o equivariant graph layers | 9.64 | 62.45 | Equivariance is the core |
| Replace with regular GCNN | 8.32 | 68.52 | Equivariant graph layers significantly outperform regular GCN |
| w/o LRFT | 2.76 | 83.09 | Low-rank compression improves matching reliability |
| Ball Query graph construction | **1.67** | **94.60** | Slightly better than KNN |
| w/o rank regularization | 6.41 | 76.45 | Rank constraint helps immensely |
| w/o sub-matrix verification | 2.58 | 87.76 | Outlier filtering is effective |
| SpinNet + Equi-GCNN | 2.93 | 82.16 | Directly replacing descriptors degrades performance |

### Key Findings

- Equivariant graph layers are the core source of performance (removing them drops RR from 94.6% to 62.5%).
- There exists an optimal point for the rank configuration of LRFT (around $r=35$), beyond which the performance drops.
- The model maintains a consistent advantage under different numbers of sampling points, still reaching 88.5% RR at 256 points.
- The inference speed is 0.12 seconds, which is nearly 20,000 times faster than RoReg (2226 seconds), showing potential for visual odometry.

## Highlights & Insights

- **Efficiency from Equivariance**: t-SNE visualization clearly demonstrates that the features generated by the equivariant graph CNN remain equivariant to rotational inputs, whereas regular GCNNs and SpinNet (SO(2)) do not possess this property.
- **Implicit Matching Instead of Explicit Searching**: Performing similarity matching in the feature space instead of exhaustive point-to-point correspondence search elegantly avoids the outlier problem.
- **Clever Use of Rank Constraint**: Drawing inspiration from LoRA, the low-rank constraint not only compresses computational cost but also automatically rejects unreliable matches through rank regularization and sub-matrix verification.
- **Modular Design**: The descriptor layer can be replaced with pretrained models, enhancing flexibility.

## Limitations & Future Work

- The input point sequence is sorted by ray length, which may not be robust to certain scene layouts.
- Dynamic scenes or cases with severe occlusions are not addressed.
- The window size for sub-matrix rank verification (5x5, 7x7) is fixed and may require scene adaptation.
- The assumption of normalizing the source frame to a canonical frame may be limited in large-scale scenes.
- Graph attention mechanisms for achieving input order invariance have not been explored.

## Related Work & Insights

- **vs PointDSC**: PointDSC explicitly rejects outlier correspondences via spatial consistency, whereas Equi-GSPR automatically achieves this through rank constraints in the implicit feature space.
- **vs RoReg**: RoReg also utilizes rotation features but suffers from extremely slow inference (2226 seconds), whereas Equi-GSPR takes only 0.12 seconds.
- **vs SpinNet**: SpinNet only has SO(2) equivariance, whereas Equi-GSPR achieves full SE(3) equivariance.
- **vs DGR**: DGR treats correspondence prediction as a classification problem requiring precise supervision, whereas Equi-GSPR does not require explicit correspondence supervision.
- **Inspiration**: The combination of equivariance and low-rank constraints can be generalized to other 3D tasks (SLAM, scene flow estimation, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination design of SE(3) equivariant graph network, LRFT, and implicit matching is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Both indoor and outdoor datasets, multiple sampling points, comprehensive ablations, and efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations and clear illustrations.
- Value: ⭐⭐⭐⭐ The 0.12-second inference speed gives it application potential for real-time registration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MHopReg: Efficient Hierarchical Multi-Hop Graph Search for Point Cloud Registration](../../CVPR2026/3d_vision/mhopreg_efficient_hierarchical_multi-hop_graph_search_for_point_cloud_registrati.md)
- [\[CVPR 2026\] ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis](../../CVPR2026/3d_vision/learning_coordinate-based_convolutional_kernels_for_continuous_se3_equivariant_a.md)
- [\[ECCV 2024\] Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion](explicitly_guided_information_interaction_network_for_cross-modal_point_cloud_co.md)
- [\[ECCV 2024\] SegPoint: Segment Any Point Cloud via Large Language Model](segpoint_segment_any_point_cloud_via_large_language_model.md)
- [\[ECCV 2024\] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)

</div>

<!-- RELATED:END -->
