---
title: >-
  [Paper Note] MoST: Efficient Monarch Sparse Tuning for 3D Representation Learning
description: >-
  [CVPR 2025][3D Vision][PEFT] Proposes MoST, the first reparameterization-based 3D PEFT method, which designs a Point Monarch structured matrix (incorporating KNN-based local feature smoothing into the Monarch matrix) to outperform full fine-tuning on multiple 3D benchmarks while tuning only 3.6% of the parameters.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "PEFT"
  - "point cloud"
  - "Monarch matrix"
  - "reparameterization"
  - "3D representation learning"
  - "K-Rectify"
date: 2026-05-08
content_hash: 8e798a433d37526d
---

# MoST: Efficient Monarch Sparse Tuning for 3D Representation Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.18368](https://arxiv.org/abs/2503.18368)  
**Code**: To be confirmed  
**Area**: 3D Vision  
**Keywords**: PEFT, point cloud, Monarch matrix, reparameterization, 3D representation learning, K-Rectify

## TL;DR

Proposes MoST, the first reparameterization-based 3D PEFT method, which designs a Point Monarch structured matrix (incorporating KNN-based local feature smoothing into the Monarch matrix) to outperform full fine-tuning on multiple 3D benchmarks while tuning only 3.6% of the parameters.

## Background & Motivation

**Background**: The pre-train-then-fine-tune paradigm in 3D point clouds (such as Point-MAE, ReCon, PointGPT, etc.) requires full fine-tuning of the entire model, which incurs high computational and memory overheads. While Parameter-Efficient Fine-Tuning (PEFT) is mature in NLP and 2D vision, it remains in an exploratory stage in the 3D point cloud domain.

**Limitations of Prior Work**:
1. **Adapter/Prompt methods** (e.g., IDPT, DAPT, PPT, PointGST): Introduce additional inference overhead and are exclusively designed for Transformers, failing to generalize to other architectures like Mamba or U-Net.
2. **Reparameterization methods like LoRA**: Introduce no inference overhead, but their low-rank assumption captures global information while ignoring local geometric features, resulting in poor performance in 3D PEFT.
3. Although the Monarch matrix has stronger expressiveness than LoRA, it also fails to capture the spatial local structure of point clouds.

**Key Insight**: Experiments reveal that the L2 distance of features within the KNN neighborhood is highly correlated with classification accuracy—the lower the distance (the smoother local features are), the better the performance. LoRA and Monarch exhibit high local distances, whereas Point Monarch achieves the lowest.

## Method

### Overall Architecture

During training, MoST reparameterizes the dense weight update matrix as a sparse Point Monarch structured matrix, and folds it back into the original weights during inference, thereby introducing **zero inference overhead**. It is applicable to any backbone containing dense layers.

### Key Designs

#### Module 1: Point Monarch Structured Matrix

Based on the standard Monarch matrix $M = PLP^\top R$, two K-Rectify operations are incorporated:
$$\text{Point Monarch} = K \cdot PLP^\top R \cdot K$$

Where $L, R$ are block-diagonal matrices with $b$ blocks (each of size $d/b \times d/b$), with the parameter count $2d^2/b \ll d^2$. $P$ is a permutation matrix representing unit stride permutation (channel shuffle). **K** is a linear transformation for KNN-based local token relation, capturing spatial local features of the point cloud.

#### Module 2: K-Rectify (Parameter-Free Local Feature Smoothing)

K-Rectify implements parameter-free local information exchange in three steps:
1. **KNN grouping**: Identifies the K-nearest neighbors for each patch center in 3D coordinate space.
2. **IDW interpolation**: Calculates the new center feature by applying Inverse Distance Weighting (IDW) to neighbor features.
3. **Residual correction**: $Kx = x + \lambda x_{new}$, where $\lambda$ is a hyperparameter.

Matrix form: $K = I + \lambda(A \odot D)$, where $A$ is the KNN adjacency matrix and $D$ is the normalized inverse distance matrix. K itself is sparse and contains no learnable parameters.

#### Module 3: Multi-Layer Feature Fusion Strategy

A parameter-free backbone-to-head alignment strategy: features output from various levels of the backbone are fused to avoid "knowledge bottlenecks", enhancing the transfer of pre-trained knowledge to the downstream task head.

### Loss & Training

Standard downstream task losses are used (cross-entropy for classification, cross-entropy + Dice loss for segmentation) without introducing extra regularization. The core lies in the structural constraint of the weight update matrix during training (the sparsity of Point Monarch), rather than innovations in the loss function.

## Key Experimental Results

### Main Results

**Comparison of PEFT Across Multiple Backbones** (ScanObjectNN PB_50_RS / ModelNet40 acc.%):

| Method | 3D? | No Inference Overhead? | Point-MAE | ReCon | Mamba3D | PointGPT |
|------|-----|-----------|-----------|-------|---------|----------|
| Full FT | - | - | 85.18/93.8 | 90.01/92.5 | 92.05/94.7 | 93.4/94.1 |
| LoRA | ✗ | ✓ | 82.76/92.50 | 85.70/92.87 | 87.16/92.42 | 91.92/92.95 |
| DAPT | ✓ | ✗ | 88.27/92.99 | 89.31/93.27 | 88.55/92.87 | 93.02/94.2 |
| PointGST | ✓ | ✗ | 89.3/93.5 | 89.49/93.6 | 89.97/93.72 | 94.83/94.8 |
| **MoST (b=8)** | **✓** | **✓** | **92.92/94.77** | **93.55/95.06** | **93.30/95.18** | **97.50/96.23** |

Key highlights: MoST outperforms Full FT by 7.74% on Point-MAE, and reaches 97.50% on PointGPT!

### Ablation Study

**Effect of Block Size b** (Point-MAE, PB_50_RS/MN40):

| b | Params (M) | PB_50_RS | ModelNet40 |
|---|----------|----------|------------|
| 32 | 0.8 | 91.95 | 94.04 |
| 16 | 1.3 | 92.71 | 94.49 |
| 8 | 2.3 | 92.92 | 94.77 |

**Contribution of Each Component** (K-Rectify Ablation):
- Standard Monarch (without K-Rectify) achieves performance between LoRA and MoST.
- Adding either the prepended K or postpended K brings improvement, whereas applying both (the complete MoST) yields the best performance.
- Ranking of local feature distances: MoST < Full FT < Monarch < LoRA, which perfectly aligns with the performance ranking.

### Key Findings

1. **Reparameterization > Adapter/Prompt**: MoST outperforms adapter-like methods across all backbones with zero inference overhead.
2. **Local Feature Smoothing is Crucial for 3D PEFT**: The L2 distance of KNN neighborhood features is highly correlated with overall performance.
3. **Exceptional Generalizability**: Applicable to all three types of backbones: Transformer, Mamba, and hierarchical architectures.
4. **Combinable with Other Matrix Decompositions**: Further compressing parameters is possible by applying Low-Rank or Kronecker decompositions to $L/R$.
5. **Outperforming Full Fine-Tuning**: Outperforms Full FT on Point-MAE (+7.74%), I2P-MAE (+3.16%), and almost all classification tasks.

## Highlights & Insights

1. **First 3D Reparameterization PEFT**: Fills an important gap, as prior work in the 3D domain only explored adapter or prompt-based methods.
2. **Simple and Intuitive Intuition**: Local feature smoothing for point clouds $\rightarrow$ better representations $\rightarrow$ higher performance. This hypothesis is thoroughly verified by experiments.
3. **Elegant Design of K-Rectify**: Parameter-free, utilizes 3D coordinate geometric information, compatible with batch matmul, and maintains the block-diagonal structure of Monarch.
4. **High Practical Value**: Reaches an impressive 97.5% accuracy on ScanObjectNN while taking only 3.6% of the parameters.

## Limitations & Future Work

1. K-Rectify requires explicit 3D coordinates (point cloud xyz) and cannot be directly applied to modalities without spatial coordinates.
2. KNN search incurs computational overhead on large-scale point clouds (although the paper states the linear transformation bottleneck is larger).
3. The performance gain on scene-level segmentation (S3DIS) is relatively minor compared to object-level tasks.
4. Hyperparameters $\lambda$ and KNN-$K$ for K-Rectify require task-specific tuning.

## Related Work & Insights

- **Monarch Matrix** (Dao et al.): An efficient structured matrix, which MoST extends to irregular 3D point clouds.
- **LoRA**: A representative of low-rank reparameterized PEFT, but it neglects local features, thus underperforming MoST in 3D PEFT.
- **IDPT/DAPT/PointGST**: 3D adapter/prompt methods, which introduce inference overhead and limit backbone types.
- **Insights**: Similar "geometry-aware sparse matrix" designs can be extended to PEFT for graph neural networks and mesh processing.

## Rating

⭐⭐⭐⭐ — Clear research motivation (local features $\rightarrow$ structured matrix), simple yet effective design (Point Monarch = Monarch + K-Rectify), and solid experiments (5 backbones across multiple tasks). The 97.5% accuracy on ScanObjectNN is highly competitive. Great academic novelty and simplicity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](../../AAAI2026/3d_vision/point-sra_self-representation_alignment_for_3d_representation_learning.md)
- [\[CVPR 2025\] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](../../NeurIPS2025/3d_vision/on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](../../ICCV2025/3d_vision/spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)

</div>

<!-- RELATED:END -->
