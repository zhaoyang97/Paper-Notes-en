---
title: >-
  [Paper Note] Zero-Shot Multi-Object Scene Completion
description: >-
  [ECCV 2024][3D Vision] OctMAE is proposed, a hybrid architecture fusing Octree U-Net and latent 3D MAE to achieve high-quality, near-real-time multi-object scene shape completion from a single RGB-D image. Efficiency and generalization are significantly enhanced via an occlusion-masking strategy and 3D Rotary Position Embedding (RoPE).
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 12688fc53908d389
---

# Zero-Shot Multi-Object Scene Completion

**Conference**: ECCV 2024  
**arXiv**: [2403.14628](https://arxiv.org/abs/2403.14628)  
**Code**: [Project Page](https://sh8.io/#/oct_mae)  
**Area**: 3D Vision

## TL;DR

OctMAE is proposed, a hybrid architecture fusing Octree U-Net and latent 3D MAE to achieve high-quality, near-real-time multi-object scene shape completion from a single RGB-D image. Efficiency and generalization are significantly enhanced via an occlusion-masking strategy and 3D Rotary Position Embedding (RoPE).

## Background & Motivation

- Existing single-object shape completion methods perform poorly in complex, multi-object real-world scenes.
- Object-centric methods require class-specific priors, limiting them to a few categories.
- VoxFormer extends MAE to 3D but uses dense voxels (memory-constrained to low resolutions).
- Lack of large-scale, multi-class 3D scene completion datasets.
- **Core Problem**: How to achieve zero-shot multi-object scene completion across a wide range of object categories.

## Method

### Overall Architecture

1. Features are extracted from a 2D image using a pre-trained ResNeXt-50, which are then back-projected to 3D using the depth map to obtain features and coordinates.
2. The 3D point features are converted into an octree representation (LoD-9, $512^3$ resolution).
3. An Octree U-Net encodes the features into an LoD-5 latent space.
4. A latent 3D MAE processes the encoded features and occlusion mask tokens.
5. An Octree U-Net decoder restores the features to LoD-9, predicting occupancy hierarchically and SDF/normals at the final layer.

### Key Designs

**OctMAE Architecture (Core Innovation)**:
- Octree U-Net handles efficient local feature encoding/decoding (from LoD-9 to LoD-5).
- 3D MAE performs global reasoning in the LoD-5 latent space (reducing the token count to hundreds or thousands).
- It combines the local perception of CNNs with the global understanding of Transformers.

**Occlusion-Masking Strategy**:
- Instead of placing mask tokens at all empty voxels (dense masking $\to$ memory explosion), mask tokens are only placed at occluded voxels.
- Occluded voxels are identified by determining which voxels are behind objects via depth testing.
- This drastically reduces the number of mask tokens, enabling the use of full attention instead of deformable attention.

**3D RoPE (Rotary Position Embedding)**:
- The 3D coordinates are encoded separately into rotation matrices $R(p^x)$, $R(p^y)$, and $R(p^z)$.
- These matrices form a block-diagonal matrix applied to the query/key of each attention layer.
- This is more efficient than learnable relative position embedding (which requires $N' \times N'$ calculations) and more generalizable than absolute position embedding.

**Large-Scale Dataset Construction**:
- Over 12k 3D models across 601 classes are selected from Objaverse, supplemented by GSO data.
- BlenderProc is used to perform physical placement and realistic lighting to render 1M images.
- This covers hand-sized objects (4–40 cm), filling the gap in zero-shot multi-object scene completion datasets.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{nrm} + \mathcal{L}_{SDF} + \sum_{h \in \{5,6,7,8,9\}} \mathcal{L}_{occ}^h$$

Normal L2 loss + SDF L2 loss + binary cross-entropy occupancy loss for each LoD. Empty voxels are pruned hierarchically to avoid unnecessary computation.

## Key Experimental Results

### Main Results

| Method | 3D Representation | Synthetic CD↓ | YCB-V CD↓ | HB CD↓ | HOPE CD↓ |
|------|--------|---------------|-----------|--------|----------|
| VoxFormer | Dense | 44.54 | 30.32 | 34.84 | 47.75 |
| ConvONet | Dense | 23.68 | 32.87 | 26.71 | 20.95 |
| MCC | Implicit | 43.37 | 35.85 | 19.59 | 17.53 |
| AICNet | Dense | 15.64 | 12.26 | 11.87 | 11.40 |
| Minkowski | Sparse | 11.47 | 8.04 | 8.81 | 8.56 |
| OCNN | Sparse | 9.05 | 7.10 | 7.02 | 8.05 |
| **OctMAE** | **Sparse** | **6.48** | **6.40** | **6.14** | **6.97** |

### Ablation Study

Position encoding comparison (synthetic dataset):

| Type | CD↓ | F1↑ | NC↑ |
|------|-----|-----|-----|
| No PE | 11.32 | 0.778 | 0.808 |
| CPE | 9.91 | 0.785 | 0.811 |
| APE | 8.61 | 0.782 | 0.825 |
| RPE | 7.81 | 0.804 | 0.830 |
| **RoPE** | **6.48** | **0.839** | **0.848** |

3D attention mechanism comparison (HOPE dataset):

| Method | Occlusion Masking | CD↓ | F1↑ |
|------|----------|-----|-----|
| 3D Deformable | ✗ | 12.14 | 0.703 |
| Neighbor Attn | ✗ | 9.26 | 0.727 |
| Octree Attn | ✗ | 7.99 | 0.752 |
| Octree Attn | ✓ | 7.54 | 0.772 |
| **Full Attn** | **✓** | **6.97** | **0.803** |

### Key Findings

- OctMAE achieves state-of-the-art performance across all four datasets and can generalize zero-shot to real scenes even when trained only on synthetic data.
- 3D RoPE makes a significant contribution to performance, reducing Chamfer Distance (CD) from 11.32 to 6.48.
- The occlusion-masking strategy makes full attention feasible, which significantly outperforms deformable attention.
- Sparse representations (Octree/Minkowski) comprehensively outperform dense/implicit representations.
- Latent-space 3D MAE is the key to generalization; adding MAE to the same U-Net architecture (Minkowski/OCNN) yields substantial improvements.

## Highlights & Insights

- Performing MAE in the latent space is an effective strategy to scale up 3D Transformers, as the token count remains controllable at LoD-5.
- Occlusion masking is an elegant design tailored for scene completion tasks: tokens are placed only where generation is most needed.
- The application of 3D RoPE in 3D vision is highly noteworthy, offering greater efficiency and better generalization than traditional position encodings.
- The creation of the large-scale Objaverse-derived dataset is an independent and valuable contribution to the community.

## Limitations & Future Work

- Requires foreground segmentation masks as input.
- Scene completion remains challenging under extreme occlusion (where only a tiny fraction of the object is visible).
- The octree representation may lose details on extremely thin structures like cables.
- Achieves near-real-time performance but is not yet strictly real-time.

## Rating

- Novelty: ⭐⭐⭐⭐ — Hybrid Octree + MAE architecture and occlusion-masking strategy.
- Effectiveness: ⭐⭐⭐⭐⭐ — Comprehensive state-of-the-art performance and zero-shot generalization.
- Practicality: ⭐⭐⭐⭐ — Addresses practical needs of robotic scene understanding.
- Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)

</div>

<!-- RELATED:END -->
