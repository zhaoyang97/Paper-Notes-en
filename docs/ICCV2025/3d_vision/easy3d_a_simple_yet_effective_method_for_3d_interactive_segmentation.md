---
title: >-
  [Paper Note] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation
description: >-
  [3D Vision] This paper proposes Easy3D, a simple yet effective method for 3D interactive instance segmentation. By combining a sparse voxel encoder, a lightweight Transformer decoder, and an implicit click fusion strategy, Easy3D consistently outperforms state-of-the-art methods on both in-domain and out-of-domain datasets. It is also the first work to successfully apply learned negative embeddings to implicit click fusion.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: f885872fa57c380d
---

# Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation

## Paper Info
- **Conference**: ICCV 2025
- **arXiv**: [2504.11024](https://arxiv.org/abs/2504.11024)
- **Code**: [simonelli-andrea.github.io/easy3d](https://simonelli-andrea.github.io/easy3d)
- **Area**: 3D Vision / 3D Interactive Segmentation
- **Keywords**: 3D interactive instance segmentation, voxel encoder, implicit click fusion, negative embedding, cross-domain generalization
- **Authors**: Andrea Simonelli, Norman Müller, Peter Kontschieder (Meta Reality Labs Zürich)

## TL;DR

This paper proposes Easy3D, a simple yet effective method for 3D interactive instance segmentation. By combining a sparse voxel encoder, a lightweight Transformer decoder, and an implicit click fusion strategy, Easy3D consistently outperforms state-of-the-art methods on both in-domain and out-of-domain datasets. It is also the first work to successfully apply learned negative embeddings to implicit click fusion.

## Background & Motivation

With the increasing prevalence of 3D digital environments acquired via NeRF, 3DGS, LiDAR scanning, and other modalities, the demand for 3D interactive segmentation continues to grow. Users define target objects through 3D clicks, and the system generates segmentation masks that can be iteratively refined via positive/negative clicks.

Existing methods suffer from two key limitations:

**AGILE3D**: employs a voxel encoder with explicit click fusion (max operation over per-click masks), resulting in poor generalization to out-of-domain data.

**Point-SAM**: employs a ViT encoder with implicit click fusion, but has a large parameter count, low efficiency, and limited generality due to its point-group representation.

Core observation: the domain-independent nature of voxel representations and the strong generalization of implicit click fusion are complementary and can be combined to address both limitations.

## Method

### Overall Architecture

Input: a 3D scene point cloud $S_P \in \mathbb{R}^{N_P \times 6}$ (each point with 3D coordinates + color) and a set of user clicks $C = \{c_1, ..., c_{N_C}\} \in \mathbb{R}^{N_C \times 4}$ (each click with 3D coordinates + positive/negative label).

Pipeline:
1. **Scene preprocessing**: voxelize the point cloud into $S_V$ at resolution $V_S = 5$ cm
2. **Scene encoding**: a sparse U-Net encodes the voxelized scene → scene embeddings $S_E$
3. **Click encoding**: positional encoding + label encoding → click embeddings $C_E$
4. **Decoding**: a Two-Way Transformer performs bidirectional attention to update $S_E$ and $C_E$
5. **Click fusion**: implicit fusion yields the segmentation mask $M_V$
6. **Post-processing**: map back to the original point cloud to obtain $M_P$

### Voxelization vs. Point-Group

Advantages of voxelization:
- Reduces dimensionality while preserving an explicit, universal metric resolution
- Enables the use of efficient sparse convolution libraries (SpConv)
- Domain-independent representation, more robust to variations in object type and point density
- Point-SAM's point-group representation relies on dataset-specific geometry and density distributions

### Implicit vs. Explicit Click Fusion (Core Comparison)

**Explicit fusion** (AGILE3D):
- Each click independently predicts a mask → masks are merged via max operation
- The decoder is unaware of positive/negative labels (only distinguished during the max operation)
- Individual click masks exhibit low confidence on out-of-domain data, leading to poor fusion

**Implicit fusion** (Easy3D/SAM):
- Introduces a learned Output Embedding that participates in attention alongside all click embeddings within the decoder
- The decoder is aware of positive/negative labels and aggregates information from all clicks via attention
- Final mask = dot product between the updated Output Embedding and scene embeddings

### Learned Negative Embedding — First Application in Implicit Fusion

A second learned output embedding (negative output embedding) is introduced, analogous to the position-free negative click in AGILE3D:
- Automatically learns which scene regions are typically background during training
- Final mask = regions where the positive output embedding mask exceeds the negative output embedding mask
- Effectively suppresses background even with a single user click

### Loss & Training

- User interactions are simulated automatically: the first click is placed at the object center, and subsequent clicks are placed at the center of the largest error region
- Iterates for $N_C = 10$ clicks
- Loss: DICE + Cross Entropy, equal weights, accumulated over all rounds and back-propagated once
- Trained for 1k epochs using PyTorch + SpConv, lr=1e-4 with polynomial decay

## Key Experimental Results

### Main Results: Cross-Dataset Interactive Segmentation (Table 1, all models trained on ScanNet40 only)

| Test Set | Method | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 |
|----------|--------|-------|-------|-------|-------|--------|
| ScanNet40 | AGILE3D | 63.0 | 70.6 | 75.1 | 79.7 | 83.5 |
| ScanNet40 | **Easy3D** | **68.2** | **74.6** | **77.3** | 79.6 | 81.7 |
| S3DIS | AGILE3D | 58.5 | 70.7 | 77.4 | 83.6 | 88.3 |
| S3DIS | Point-SAM | 38.8 | — | 67.1 | 72.2 | 80.6 |
| S3DIS | **Easy3D** | **65.7** | **76.0** | **80.8** | **84.9** | 87.8 |
| KITTI-360 | AGILE3D | 34.8 | 40.7 | 42.7 | 44.4 | 49.6 |
| KITTI-360 | Point-SAM | 44.0 | — | 67.1 | 72.2 | 80.8 |
| KITTI-360 | **Easy3D** | **46.3** | **58.7** | **66.7** | **76.2** | **83.6** |

On the most challenging out-of-domain dataset KITTI-360, Easy3D achieves IoU@10 of 83.6, surpassing AGILE3D by **+34**.

### Ablation Study: Click Fusion Strategy + Negative Embedding (Table 3)

| Test Set | Fusion | Neg. Emb. | IoU@1 | IoU@3 | IoU@10 |
|----------|--------|-----------|-------|-------|--------|
| ScanNet40 | Explicit | ✗ | 59.6 | 73.2 | 82.6 |
| ScanNet40 | Explicit | ✓ | 62.7 | 75.2 | 83.6 |
| ScanNet40 | Implicit | ✗ | 66.4 | 76.3 | 81.2 |
| ScanNet40 | **Implicit** | **✓** | **68.2** | **77.3** | **81.7** |
| KITTI-360 | Explicit | ✗ | 31.0 | 40.0 | 46.3 |
| KITTI-360 | Explicit | ✓ | 34.5 | 42.6 | 48.2 |
| KITTI-360 | Implicit | ✗ | 44.9 | 65.7 | 83.2 |
| KITTI-360 | **Implicit** | **✓** | **46.3** | **66.7** | **83.6** |

### Comparison with Non-Interactive Methods (Table 2, trained on ScanNet20 only)

| Setting | Method | mAP | AP50 | AP25 |
|---------|--------|-----|------|------|
| ScanNet20→ScanNet20 | Mask3D | 51.5 | 77.0 | 90.2 |
| ScanNet20→ScanNet20 | AGILE3D | 53.5 | 75.6 | 91.3 |
| ScanNet20→ScanNet20 | **Easy3D** | **56.1** | **79.5** | **93.1** |
| ScanNet20→ScanNet40 | Mask3D | 5.3 | 13.1 | 24.7 |
| ScanNet20→ScanNet40 | AGILE3D | 24.8 | 45.7 | 72.4 |
| ScanNet20→ScanNet40 | **Easy3D** | **39.2** | **64.6** | **85.5** |

The advantage on unseen categories is particularly pronounced (mAP 39.2 vs. 24.8).

### Key Findings

1. **Implicit fusion substantially outperforms explicit fusion in out-of-domain generalization**: on KITTI-360, IoU@10 differs by +73% (83.6 vs. 48.2)
2. **Explicit fusion holds a marginal advantage on in-domain data with many clicks**: attributed to high per-click mask confidence within the known domain
3. **Negative embedding consistently improves all settings**: regardless of fusion type or domain
4. **Voxels outperform point-groups**: domain-independent voxel representations yield more stable cross-domain performance
5. **Applicable to Gaussian Splatting scenes**: Easy3D demonstrates clear advantages on GS-ScanNet40

## Highlights & Insights

- **The name "Easy" is well-deserved**: existing components (voxel encoder + implicit fusion + negative embedding) are combined simply yet to outstanding effect
- **First systematic analysis of implicit vs. explicit fusion**: clearly reveals the respective strengths and weaknesses of each strategy
- **VR application demonstration**: real-time 3D interactive segmentation and object manipulation on a consumer-grade headset, demonstrating strong practical utility
- No pretraining required; training from scratch for 1k epochs suffices

## Limitations & Future Work

- Implicit fusion is marginally inferior to explicit fusion in in-domain settings with many clicks (gap is small)
- The voxel resolution is fixed at 5 cm, which may limit performance on extremely fine-grained or large-scale scenes
- Currently supports single-object segmentation only
- Further analysis on ScanNet++ is lacking

## Related Work & Insights

- **Natural extension of SAM to 3D**: SAM's implicit fusion paradigm is successfully transferred to 3D
- **Continued value of sparse voxel convolution**: in the era of Transformers, voxels combined with sparse convolution remain an efficient backbone for 3D processing
- **Support for 3DGS scene understanding**: Easy3D can be directly integrated into Gaussian Splatting rendering pipelines

## Rating ⭐⭐⭐⭐

The method is simple and effective, with sound engineering design. Ablation experiments clearly demonstrate the contribution of each component. Cross-domain generalization is particularly impressive, especially on KITTI-360 and Gaussian Splatting scenes. The VR application demonstration further underscores its practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)

</div>

<!-- RELATED:END -->
