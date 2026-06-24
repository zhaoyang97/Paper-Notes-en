---
title: >-
  [Paper Note] RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation
description: >-
  [ECCV 2024][Autonomous Driving][LiDAR Semantic Segmentation] This paper proposes RAPiD (Range-Aware Pointwise Distance Distribution) features, a local geometric representation for LiDAR point clouds that is invariant to rigid transformations and adaptive to changes in point density. Combined with a dual-stage nested autoencoder and channel attention-based fusion, it achieves state-of-the-art segmentation performance on SemanticKITTI (76.1 mIoU) and nuScenes (83.6 mIoU).
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "LiDAR Semantic Segmentation"
  - "Isometry Invariance"
  - "Distance Distribution Representation"
  - "Autoencoder Embedding"
  - "Attention Fusion"
date: 2026-05-08
content_hash: abc200896c4a6811
---

# RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.10159](https://arxiv.org/abs/2407.10159)  
**Code**: [https://github.com/l1997i/rapid_seg](https://github.com/l1997i/rapid_seg)  
**Area**: Autonomous Driving  
**Keywords**: LiDAR Semantic Segmentation, Isometry Invariance, Distance Distribution Representation, Autoencoder Embedding, Attention Fusion

## TL;DR

This paper proposes RAPiD (Range-Aware Pointwise Distance Distribution) features, a local geometric representation for LiDAR point clouds that is invariant to rigid transformations and adaptive to changes in point density. Combined with a dual-stage nested autoencoder and channel attention-based fusion, it achieves state-of-the-art segmentation performance on SemanticKITTI (76.1 mIoU) and nuScenes (83.6 mIoU).

## Background & Motivation

3D LiDAR semantic segmentation is a fundamental task in autonomous driving scene understanding. Existing methods primarily rely on coordinates and intensity as input features, which suffer from two core limitations: (1) lack of invariance to rigid transformations (rotation and translation); and (2) poor performance in sparse or occluded point cloud regions. While data augmentation can partially alleviate transformation issues, it cannot guarantee coverage of all possible transformations in the wild.

The key challenge lies in the need for a representation that concurrently satisfies three requirements: capturing local geometric structures, maintaining rigid transformation invariance, and adapting to noisy LiDAR environments, which existing methods struggle to meet simultaneously.

This paper is inspired by the Pointwise Distance Distribution (PDD) in crystallography, which is an isometry invariant representing local structures by computing the distance between a point and its neighbors. However, directly applying PDD to large-scale LiDAR point clouds introduces problems such as extremely high dimensionality, heavy computation, and neglect of locality. The core idea of this work is to adapt PDD into RAPiD features tailored for LiDAR scenes by leveraging the ring-like scanning structures of LiDAR and semantic categories for localization, and incorporating a 4D distance (3D geometry + reflectivity) to enhance semantic discriminability.

## Method

### Overall Architecture

The input point cloud generates three types of features: coordinate features $F_C$, reflectivity features $F_I$, and RAPiD features $F_R$. $F_C \oplus F_I$ are processed by a VSA voxel encoder to obtain voxel representations, while $F_R$ is compressed into low-dimensional voxel embeddings via a RAPiD Autoencoder. The two embeddings are fused through channel attention and then fed into the backbone (Minkowski-UNet34) for segmentation prediction.

### Key Designs

1. **RAPiD Feature**: Given a point $\bm{p}_j$ and its k local neighbors, a 4D distance matrix is computed: $\bm{\rho}_{j,l} = \|[\bm{p}_j - \bm{p}_{j,l}, g(r_j) - g(r_{j,l})]\|_2$, where $g(\cdot)$ maps the reflectivity to the same numerical range as the Euclidean distance. Sorting by row and column yields a $u \times k$ RAPiD matrix. Design Motivation: (a) distances are invariant under rigid transformations, ensuring the isometric invariance of features; (b) the 4D distance incorporates differences in reflectivity, allowing different surface materials to produce distinct distance distributions to enhance inter-class discriminability; (c) different $k$ values ($k_{close}, k_{mid}, k_{far}$) are applied based on the distance range to adapt to the varying point densities of LiDAR data.

2. **Intra-Ring RAPiD (R-RAPiD) and Intra-Class RAPiD (C-RAPiD)**: R-RAPiD constrains the RoI to the same laser beam ring, leveraging the inherent isotropic radiation structure of LiDAR to reduce computational overhead. The point cloud is segmented into B rings via beam IDs, and RAPiD is computed independently within each ring. C-RAPiD constrains the RoI within the same semantic class, utilizing semantic labels to enhance intra-class feature consistency. Ground-truth labels are used during training, while pseudo-labels generated by a pre-trained R-RAPiD-Seg are used during testing. The two designs are complementary: R-RAPiD is label-independent and highly generalizable, while C-RAPiD enhances intra-class embedding fidelity.

3. **Dual-stage Nested RAPiD Autoencoder**: The outer VSA AE compresses high-dimensional pointwise features into voxel representations $H^v \in \mathbb{R}^{c \times l \times d}$ using scatter-sum aggregation and cross-attention interaction. The inner AE further reduces the dimensionality along the voxel dimension ($d \to d'$) using convolutional layers and ConvFFN to facilitate information exchange between voxels. A key innovation is the class-aware contrastive loss $\mathcal{L}_{contr}$, which maximizes the distance between embeddings of different classes and minimizes the distance between embeddings of the same class to address the non-uniqueness issue of AE embeddings.

4. **Channel Attention Fusion (FuAtten)**: After concatenating the coordinate embeddings $E_C$, reflectivity embeddings $E_I$, and RAPiD embeddings $E_R$, a squeeze-excitation mechanism generates channel-level weights $\bm{a}_z = \sigma(\mathbf{W}_2 \delta(\mathbf{W}_1 \mathbf{z}))$ to adaptively weigh the features of each channel. This avoids the dimension explosion and training bias issues caused by simple concatenation.

### Loss & Training

The total loss of AE is $\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{contr}$, where $\mathcal{L}_{recon}$ is the MSE reconstruction loss.

A two-stage training strategy is adopted: in the first stage, the RAPiD AE is trained independently; in the second stage, the AE parameters are frozen and integrated into the joint network for segmentation training. Two architectural variants are introduced: R-RAPiD-Seg (a lightweight version using only R-RAPiD) and C-RAPiD-Seg (a high-performance version utilizing both R-RAPiD and C-RAPiD).

Training is performed on 4×A100 GPUs with a learning rate of 1e-3, optimized via SGD with a cosine schedule and a 2-epoch warmup, for a total of 100 epochs. The inference time is approximately 105ms/frame.

## Key Experimental Results

### Main Results

| Dataset | Metric | RAPiD-Seg | Prev. SOTA | Gain |
|--------|------|-----------|----------|------|
| SemanticKITTI test | mIoU | **76.1** | UniSeg 75.2 (Multi-modal) | +0.9 |
| nuScenes test | mIoU | **83.6** | UniSeg 83.5 (Multi-modal) | +0.1 |

SemanticKITTI test breakdown (vs. other LiDAR-only methods):

| Method | mIoU | truck | o.veh | park | o.gro |
|------|------|-------|-------|------|-------|
| PCSeg | 72.9 | 58.6 | 68.6 | 71.5 | 36.9 |
| RangeFormer | 73.3 | 59.9 | 66.2 | 73.0 | 42.4 |
| **RAPiD-Seg** | **76.1** | **72.5** | **80.7** | **78.2** | **46.0** |

nuScenes test (vs. multi-modal methods):

| Method | Modality | mIoU | truck | trail | const |
|------|------|------|-------|-------|-------|
| UniSeg | L+C | 83.5 | 76.7 | 86.3 | 80.5 |
| LidarMultiNet | L+C | 81.4 | 74.8 | 86.9 | 71.5 |
| **RAPiD-Seg** | **L** | **83.6** | **79.0** | **88.5** | **84.6** |

### Ablation Study

SemanticKITTI val gradual component addition:

| Configuration | mIoU | Δ | Description |
|------|------|---|------|
| Baseline | 70.04 | - | Without RAPiD |
| + Geometric RAPiD | 71.21 | +1.17 | 3D distance is effective |
| + Reflectivity | 71.93 | +1.89 | Integrating reflectivity brings significant gains |
| + RAPiD Embedding (AE) | 72.15 | +2.11 | AE compression is effective |
| + Attention Fusion | 72.32 | +2.28 | Attention outperforms concatenation |
| + All (R+C-RAPiD) | **73.02** | **+2.98** | Synergy of all components is optimal |

RAPiD vs. PDD comparison:

| Method | mIoU | truck | o.veh | park |
|------|------|-------|-------|------|
| Baseline | 70.0 | 59.8 | 70.3 | 69.2 |
| PDD (original) | 66.2 | 40.3 | 65.8 | 67.5 |
| **RAPiD+R** | **73.0** | **70.4** | **78.5** | **75.8** |

### Key Findings

- Directly using PDD features drops the performance by 3.8 mIoU compared to the baseline, indicating that the original PDD is not suitable for LiDAR scenarios. The range-aware design of RAPiD and the integration of reflectivity are crucial.
- The single-modal (LiDAR-only) method outperforms multi-modal methods (with camera), showing that local geometric features extracted by RAPiD are more effective than RGB information.
- Rigid object categories (such as truck, o.veh, and park) see the most significant improvements, aligning with the design objective of RAPiD's invariance to rigid transformations.
- R-RAPiD-Seg already achieves a 2.3 mIoU improvement, and C-RAPiD-Seg further increases this to 3.0 mIoU, demonstrating the effectiveness of the intra-class feature constraints.

## Highlights & Insights

- **Cross-domain transfer from crystallography to autonomous driving**: Adapting PDD to RAPiD serves as an elegant scientific paradigm, though it requires non-trivial adaptations (range-awareness, 4D distance with reflectivity, and LiDAR ring structures).
- **Elegant 4D distance design**: Integrating reflectivity in the same scale as geometric distance is simple, elegant, and highly effective.
- **Necessity of the dual-stage nested AE**: The outer stage handles the irregular point-to-voxel conversion while the inner stage processes high-to-low dimensional compression, displaying a clear division of labor.
- **Class-aware contrastive loss**: Resolves the non-uniqueness problem of AE embeddings, regularizing intra-class features to cluster together and pushing inter-class features apart.

## Limitations & Future Work

- C-RAPiD depends on semantic labels or pseudo-labels, and the quality of the pseudo-labels directly impacts overall performance.
- RAPiD features must be pre-calculated, increasing the complexity of the data preprocessing pipeline.
- In extremely sparse, far-range regions, the number of neighbor points might be insufficient to compute a meaningful RAPiD.
- The two-stage training scheme of the AE increases the overall training complexity.
- The method has only been validated for single-frame segmentation, without exploring multi-frame temporal fusion.

## Related Work & Insights

- **PDD (Widdowson & Kurlin)**: Isometry invariant in crystallography, which serves as the theoretical foundation for RAPiD.
- **Cylinder3D**: Cylindrical partitioning for LiDAR point cloud scenarios, a common spatial partitioning backbone.
- **SPVCNN**: A point-voxel hybrid method that provides a backbone reference for RAPiD-Seg.
- **Insight**: Mathematical invariants from other domains can find unexpected applications in new fields, but customized adaptations are crucial.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Successfully introduces PDD from crystallography to LiDAR segmentation; the 4D distance and range-aware designs are novel and unique.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Achieves SOTA on two mainstream datasets with highly detailed ablation studies and a convincing comparison of PDD vs. RAPiD.
- Writing Quality: ⭐⭐⭐⭐ — Highly rigorous mathematical formulations, but high information density makes it somewhat challenging to read.
- Value: ⭐⭐⭐⭐ — Establishes a new paradigm for point cloud feature design, which will definitely inspire future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)
- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)
- [\[CVPR 2026\] Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation](../../CVPR2026/autonomous_driving/learning_to_identify_out-of-distribution_objects_for_3d_lidar_anomaly_segmentati.md)
- [\[ICLR 2026\] Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation](../../ICLR2026/autonomous_driving/adaptive_augmentation-aware_latent_learning_for_robust_lidar_semantic_segmentati.md)

</div>

<!-- RELATED:END -->
