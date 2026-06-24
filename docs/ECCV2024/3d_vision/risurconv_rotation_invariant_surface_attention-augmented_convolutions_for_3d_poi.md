---
title: >-
  [Paper Note] RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation
description: >-
  [ECCV 2024][3D Vision][Point Cloud Classification] RISurConv is proposed to construct local triangular surfaces and extract highly representative Rotation Invariant Surface Properties (RISP). Combined with attention-augmented convolutions, it achieves the first rotation-invariant point cloud analysis network to surpass non-rotation-invariant methods in accuracy.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Point Cloud Classification"
  - "Rotation Invariance"
  - "Self-Attention"
  - "Surface Attributes"
  - "3D Segmentation"
date: 2026-05-08
content_hash: aea2d8aa0574a14c
---

# RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2408.06110](https://arxiv.org/abs/2408.06110)  
**Code**: [Yes](https://github.com/cszyzhang/RISurConv)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Classification, Rotation Invariance, Self-Attention, Surface Attributes, 3D Segmentation

## TL;DR

RISurConv is proposed to construct local triangular surfaces and extract highly representative Rotation Invariant Surface Properties (RISP). Combined with attention-augmented convolutions, it achieves the first rotation-invariant point cloud analysis network to surpass non-rotation-invariant methods in accuracy.

## Background & Motivation

### Key Challenge

**Background**: Deep learning on 3D point clouds mostly focuses on translation and point permutation invariance, while rotation invariance is less studied. Existing rotation-invariant methods (such as RIConv, RIConv++) ensure rotation invariance through hand-crafted features, but their performance is far below that of non-rotation-invariant methods (such as PointTransformer v2). The primary reason is that global information is lost during the generation of rotation-invariant features, and the LRF/LRA (Local Reference Frame/Axis) is unstable. **Goal**: The goal of this paper is to narrow or even eliminate the accuracy gap between rotation-invariant and non-rotation-invariant methods.

## Method

### Overall Architecture

1. Construct a $K$-nearest neighbor (KNN) local point set for each reference point.
2. Build two triangular surfaces for each neighbor and extract 14-dimensional Rotation Invariant Surface Properties (RISP).
3. Embed RISP using an MLP, and then refine features through two self-attention (SA) layers.
4. Utilize five RISurConv layers + a Transformer Encoder + fully connected layers to output classification/segmentation results.

### Key Designs

**Rotation Invariant Surface Properties (RISP)**: For each neighboring point $x_i$, two adjacent neighbors $x_{i-1}$ and $x_{i+1}$ are selected to construct two triangular surfaces. The extracted 14-dimensional features include: distance $L_0$, 5 Euclidean space angles (triangular interior angles and dihedral angles), and 8 tangent space angles (angles between normal vectors and edges). RISP mathematically describes the double triangles and their relations thoroughly, ensuring geometric completeness.

**RISurConv Operator**: Consists of two self-attention modules—SA1 refines features among $K$ points in the neighborhood, and SA2 refines global features among $N$ representative points. The two modules work synergistically to enhance feature representation.

### Loss & Training

Cross-entropy loss is used for classification, and standard segmentation loss is used for segmentation.

## Key Experimental Results

### Main Results

**ModelNet40 Classification Accuracy (Overall Accuracy %)**:

| Method | Rotation Invariant | z/z | SO3/SO3 | z/SO3 | Std. |
|------|----------|-----|---------|-------|------|
| PointNet++ | ✗ | 89.3 | 85.0 | 28.6 | 33.8 |
| Pt Transformer v2 | ✗ | 94.2 | 88.3 | 51.8 | 23.0 |
| RIConv++ | ✓ | 91.3 | 91.3 | 91.3 | 0.0 |
| **RISurConv** | **✓** | **96.0** | **96.0** | **96.0** | **0.0** |

**ScanObjectNN Real-World Classification (PB_T50_RS)**:

| Method | z/z | SO3/SO3 | z/SO3 |
|------|-----|---------|-------|
| RIConv++ | 80.3 | 80.3 | 80.3 |
| **RISurConv** | **93.1** | **93.1** | **93.1** |

**ShapeNet Part Segmentation (mIoU %)**:

| Method | SO3/SO3 | z/SO3 |
|------|---------|-------|
| RIConv++ (xyz+nor) | 80.5 | 80.5 |
| **RISurConv (xyz+nor)** | **81.5** | **81.5** |

### Ablation Study

| Ablation Item | Accuracy |
|--------|------|
| Full Model (A) | 96.0 |
| Remove $L_0$ (B) | 95.5 |
| Tangent Space Angles Only (C) | 90.9 |
| $L_0 + \phi$ Only (D) | 88.2 |
| Remove SA1+SA2+TE (E) | 92.8 |
| Remove Transformer Encoder (D) | 94.3 |

### Key Findings

- This represents the first rotation-invariant method to surpass all non-rotation-invariant methods on ModelNet40 (96.0% vs. 94.2% of PT v2).
- Outperforms RIConv++ by 12.8 percentage points on ScanObjectNN.
- Angle features are more critical than distance features, with tangent space and Euclidean space angles serving as complementary properties.
- Self-attention modules facilitate a more uniform feature distribution.

## Highlights & Insights

1. **Outperforming Non-Rotation-Invariant Methods for the First Time**: It is demonstrated that rotation-invariant features do not have to come at the expense of accuracy.
2. **Local Triangular Surface Construction**: Captures local geometric structures more effectively compared to point-wise operations.
3. **Completeness of RISP**: The 14-dimensional features completely describe the double-triangle structure; adding more features does not further improve performance.
4. Self-attention enables weight redistribution, enhancing feature efficacy.

## Limitations & Future Work

- Relatively large parameter size (14M vs. 0.4M for RIConv++), leading to a certain decrease in inference speed.
- The quality of normal vector estimation affects classification accuracy (w/o normal is 0.4% lower than w/ normal).
- There is still room for improvement regarding fine-grained classification.

## Related Work & Insights

- RIConv only considers local features, leading to diminished accuracy.
- GCAConv employs LRF, but LRF is inherently unstable.
- **Insight**: Transitioning from point-wise operations to surface-wise representation is a promising path for enhancing 3D feature representation.

## Rating

- Novelty: ★★★★★ The local triangular surface and RISP feature designs are ingenious, breaking the performance ceiling of rotation invariance for the first time.
- Practicality: ★★★★☆ Rotation invariance is highly important for robotics and autonomous driving scenarios.
- Experimental Quality: ★★★★★ Comprehensive validation on multiple datasets with detailed ablation studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms](../../AAAI2026/3d_vision/enhancing_rotation-invariant_3d_learning_with_global_pose_awareness_and_attentio.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation](dual-level_adaptive_self-labeling_for_novel_class_discovery_in_point_cloud_segme.md)
- [\[AAAI 2026\] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning](../../AAAI2026/3d_vision/hierarchical_direction_perception_via_atomic_dot-product_operators_for_rotation-.md)
- [\[CVPR 2026\] 4D Local Modeling Toward Dynamic Global Perception for Ambiguity-free Rotation-Invariant Point Cloud Analysis](../../CVPR2026/3d_vision/4d_local_modeling_toward_dynamic_global_perception_for_ambiguity-free_rotation-i.md)

</div>

<!-- RELATED:END -->
