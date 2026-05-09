---
title: >-
  [Paper Note] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion
description: >-
  [CVPR 2026][Segmentation][Panoramic semantic segmentation] This paper proposes REL, a three-channel depth representation based on cylindrical coordinates (Rectified Depth + EGVIA + LOA), and a Spherical Multi-Modal Fusion module (SMMF) for panoramic semantic segmentation. The approach achieves 63.06% average mIoU on Stanford2D3D (a 2.35% gain over the HHA baseline) and reduces performance variance under 3D perturbations by approximately 70%.
tags:
  - CVPR 2026
  - Segmentation
  - Panoramic semantic segmentation
  - depth representation
  - multimodal fusion
  - cylindrical coordinates
  - RGB-D
date: 2026-05-08
content_hash: 34e6d466de504ba9
---

# REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion

**Conference**: CVPR 2026
**arXiv**: [2601.16788](https://arxiv.org/abs/2601.16788)
**Code**: None
**Area**: Semantic Segmentation
**Keywords**: Panoramic semantic segmentation, depth representation, multimodal fusion, cylindrical coordinates, RGB-D

## TL;DR

This paper proposes REL, a three-channel depth representation based on cylindrical coordinates (Rectified Depth + EGVIA + LOA), and a Spherical Multi-Modal Fusion module (SMMF) for panoramic semantic segmentation. The approach achieves 63.06% average mIoU on Stanford2D3D (a 2.35% gain over the HHA baseline) and reduces performance variance under 3D perturbations by approximately 70%.

## Background & Motivation

**Background**: Panoramic semantic segmentation (PASS) aims at full scene understanding from a 360°×180° ultra-wide field of view and is widely applied in autonomous driving, AR/VR, and related domains. Dominant approaches adopt Equirectangular Projection (ERP) to convert spherical data into 2D images for processing.

**Limitations of Prior Work**: The HHA representation widely used in RGB-D methods has two critical shortcomings: (a) the second degree of freedom of the surface normal direction (lateral azimuth) is absent, leading to incomplete geometric information; (b) HHA computation relies on camera pose and intrinsics (e.g., focal length), making it inconvenient for processing pure image data.

**Key Challenge**: ERP-projected panoramic images exhibit large distortion and local feature variation across regions, yet existing multimodal fusion strategies apply the same fusion scheme globally, lacking region-adaptive capability. Moreover, cylindrical unrolling disrupts the continuity of scene structure.

**Goal**: Design a more complete panoramic depth representation and achieve region-adaptive multimodal fusion to improve the accuracy and robustness of panoramic semantic segmentation.

**Key Insight**: Exploit the cylindrical geometry inherent to ERP projection to design a three-channel cylindrical-coordinate-based representation (REL), and achieve spherically-aware region-level adaptive fusion by sampling overlapping regions on the lateral surface of the cylinder.

**Core Idea**: Use cylindrical coordinates $\rho\theta z$ to fully encode 3D position and surface normal direction (REL), and employ Spherical Dynamic Mixture-of-Experts Fusion (SMMF) to apply different fusion strategies for different regions.

## Method

### Overall Architecture

REL-SF4PASS consists of two core components: (1) the REL depth representation module, which converts raw depth maps into a three-channel representation; and (2) the SMMF fusion module, which inserts fusion units at multiple stages of the network to perform region-adaptive RGB-REL feature fusion. The overall architecture adopts a dual-branch design that processes RGB and REL inputs separately.

### Key Designs

1. **REL Representation (3 channels)**:

   - **Rectified Depth (ReD)**: Planar distance $\rho = d \cdot \cos(\phi)$, eliminating the influence of the height direction.
   - **EGVIA**: Observes a strong linear correlation between normalized image height $H$ and the normal-gravity inclination angle $A$ in panoramic images; fuses both via $\lambda \cdot A + (1-\lambda) \cdot H$ for horizontal surfaces, and uses $A$ alone for vertical surfaces.
   - **LOA**: The angle between the surface normal and the tangent direction $\hat{T}$, supplying the second degree of freedom of the normal that HHA lacks.
   - REL requires only the raw depth map and no camera intrinsics or extrinsics.

2. **SMMF (Spherical Dynamic Multi-Modal Fusion)**:

   - Samples $m \times n$ overlapping regions on the lateral surface of the cylinder; horizontally, regions are allowed to wrap across the left and right boundaries of the panoramic image.
   - Vertically, regions expand symmetrically upward and downward from the equator ($\phi = 0°$), ensuring equatorial symmetry.
   - Each region has an independent Gate Network that determines the fusion strategy (MoE architecture with $B$ expert fusion operations).
   - A two-stage soft-to-hard training scheme is adopted: soft-probability training of all experts first, followed by hard one-hot expert selection.

3. **Fusion Early-Stopping Mechanism**: If a fusion unit at a given stage selects "no fusion" (RGB only), all subsequent fusion units are forced to adopt the same no-fusion decision, avoiding redundant computation.

### Loss & Training

Standard cross-entropy loss for semantic segmentation. Two-stage training: in the soft training stage, expert selection probabilities are non-zero for all experts; in the hard training stage, probabilities are enforced to be one-hot.

## Key Experimental Results

### Main Results (Stanford2D3D Panoramic Dataset)

| Method | Modality | 3-fold Avg. mIoU | Fold1 mIoU |
|---|---|---|---|
| Trans4PASS+ (2024) | RGB | 53.7% | 53.6% |
| SGTA4PASS (2023) | RGB | 55.3% | 56.4% |
| Twin (2025) | RGB | 55.85% | - |
| SFSS (2024) | RGB-HHA | 60.60% | - |
| CMX* (reproduced) | RGB-HHA | 60.71% | 63.98% |
| **REL-SF4PASS (Ours)** | **RGB-REL** | **63.06%** | **67.37%** |

### Robustness under 3D Perturbations (SGA Evaluation, 16 Rotation Configurations)

| Representation | mIoU Range | Performance Variance |
|---|---|---|
| HHA + SMMF | 59.13~65.85% | High variance |
| REL + SMMF (Ours) | 63.39~67.37% | **~70% variance reduction** |

### Key Findings

- REL outperforms HHA under all 16 3D perturbation configurations, validating the rotational robustness of the cylindrical coordinate representation.
- Region-level fusion via SMMF is more effective than uniform fusion; regions near the equator are semantically richer and benefit from finer-grained fusion.
- REL requires no camera parameters, making it easier to generalize across different acquisition devices.
- RGB-REL achieves 63.06% average mIoU, surpassing all known methods including RGB-HHA.

## Highlights & Insights

- **Natural fit of cylindrical coordinates**: Since panoramic images are produced by cylindrical projection, representing depth information in cylindrical coordinates is geometrically self-consistent.
- **Discovery of the plane-normal relationship**: The statistical finding that $H$ and $A$ are highly correlated in panoramic images is the key insight motivating the design of EGVIA.
- **Camera-parameter-free**: REL can be computed from the depth map alone, greatly simplifying the data processing pipeline.
- **70% variance reduction**: The substantial improvement in robustness to 3D perturbations carries practical engineering value.

## Limitations & Future Work

- Validation is limited to the single Stanford2D3D dataset, with no evaluation on outdoor scenes.
- The number of regions $m \times n$ in SMMF requires manual specification; adaptive determination may yield better results.
- The method has not been combined with recent large-scale pretrained backbones such as DINOv2.
- Whether the information gain of the LOA channel remains stable in near-vertical surface scenarios warrants further analysis.

## Related Work & Insights

- **HHA representation** is a classic design for RGB-D segmentation (Gupta et al.); REL serves as a direct and improved replacement.
- **CMX**'s cross-modal feature rectification and fusion module serves as the design inspiration for SMMF.
- **DynMM**'s sample-adaptive fusion is extended by SMMF to the region level, achieving finer granularity.

## Rating

- Novelty: ⭐⭐⭐⭐ (REL representation is rigorously designed; the cylindrical coordinate formulation is geometrically natural)
- Experimental Thoroughness: ⭐⭐⭐ (single dataset, but robustness analysis is thorough)
- Writing Quality: ⭐⭐⭐⭐ (mathematical derivations are clear; physical interpretations are well explained)
- Value: ⭐⭐⭐⭐ (practical contribution to panoramic segmentation; directly substitutable for HHA)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2026\] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)
- [\[CVPR 2026\] GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](masked_representation_modeling_for_domain-adaptive_segmentation.md)

<!-- RELATED:END -->
