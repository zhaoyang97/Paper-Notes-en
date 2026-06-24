---
title: >-
  [Paper Note] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting
description: >-
  [ICCV 2025 (Highlight)][3D Vision][Semantic occupancy annotation] This paper proposes AutoOcc, a fully automatic vision-centric pipeline for open-ended semantic occupancy annotation. By leveraging vision-language model (VLM)-guided differentiable Gaussian splatting (VL-GS), AutoOcc generates 3D semantic occupancy without any human labels, achieving IoU 83.01 / mIoU 20.92 on Occ3D-nuScenes with camera-only input, substantially outperforming existing automatic annotation method…
tags:
  - "ICCV 2025 (Highlight)"
  - "3D Vision"
  - "Semantic occupancy annotation"
  - "vision-language models"
  - "3D Gaussian splatting"
  - "open-vocabulary"
  - "automatic annotation"
date: 2026-05-08
content_hash: 3ee9ec35c5a7201e
---

# AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting

**Conference**: ICCV 2025 (Highlight)  
**arXiv**: [2502.04981](https://arxiv.org/abs/2502.04981)  
**Code**: Not publicly available  
**Area**: 3D Scene Understanding / Autonomous Driving / Semantic Occupancy Annotation  
**Keywords**: Semantic occupancy annotation, vision-language models, 3D Gaussian splatting, open-vocabulary, automatic annotation  

## TL;DR
This paper proposes AutoOcc, a fully automatic vision-centric pipeline for open-ended semantic occupancy annotation. By leveraging vision-language model (VLM)-guided differentiable Gaussian splatting (VL-GS), AutoOcc generates 3D semantic occupancy without any human labels, achieving IoU 83.01 / mIoU 20.92 on Occ3D-nuScenes with camera-only input, substantially outperforming existing automatic annotation methods.

## Background & Motivation
3D semantic occupancy prediction is critical for autonomous driving and embodied intelligence, yet high-quality annotation is prohibitively expensive (nuScenes annotation required 4,000+ human hours). Existing automatic and semi-automatic annotation approaches follow three paradigms:
1. **Human-assisted annotation**: Time-consuming and costly.
2. **Point cloud voxelization** (SurroundOcc, OpenOcc): Relies on LiDAR point clouds and manually defined 3D priors, requiring multi-stage post-processing.
3. **2D-to-3D projection** (OVIR-3D, SAMPro3D): Projects 2D segmentation results into 3D, but struggles to ensure 3D consistency.

Common limitations across these paradigms include: (a) heavy dependence on LiDAR point clouds, which are inherently sparse and incomplete; (b) requirement for manual pre-annotation or post-processing; (c) restriction to closed-set or open-set categories, with no capability to handle objects beyond predefined classes (e.g., collapsed bollards, plastic debris on roads). Meanwhile, self-supervised methods (GaussianOcc, LangOcc, VEON), while annotation-free, underperform in geometric accuracy, spatiotemporal consistency, and cross-dataset generalization.

## Core Problem
How to build a **fully automated, annotation-free pipeline that supports open-ended semantic categories** for 3D occupancy annotation, while ensuring high geometric accuracy, spatiotemporal consistency, robust handling of dynamic objects, and generalization to unseen datasets?

## Method

### Overall Architecture
AutoOcc is an end-to-end vision-centric automatic annotation pipeline with the following stages:
1. **Input**: Multi-view image sequences (LiDAR optional).
2. **Vision-Language Guidance**: A VLM generates semantic attention maps and constructs a dynamically updated semantic query list.
3. **VL-GS Reconstruction**: Semantic-aware scalable Gaussian primitives serve as an intermediate representation; a self-estimated optical flow module handles dynamic objects.
4. **Output**: 3D semantic occupancy annotations are directly generated via cumulative GS-Voxel Splatting.

### Key Designs
1. **Semantic Attention Map**: A VLM (e.g., InternVL) is applied to multi-view images to enumerate all possible object categories. Attention maps are extracted and aggregated from the Transformer decoder, and a **dynamically updated semantic query list** is constructed. A semantic merging strategy consolidates semantically similar sub-vocabulary items (e.g., "tree" and "shrub") into unified categories (e.g., "vegetation"), fundamentally enabling open-ended semantic annotation. The attention maps additionally serve: (a) as prompts for SAM to generate instance-level segmentation masks; and (b) to guide UniDepth for semantic-level depth estimation, decoupling foreground and background while excluding sky regions.

2. **Vision-Language Guided Gaussian Splatting (VL-GS)**:

    - **Semantic-aware Scalable Gaussians**: The core innovation — different semantic objects occupy different spatial "weights." Each Gaussian primitive is assigned a semantic attribute and an adaptive scale factor. Large background regions (e.g., buildings) are represented by sparse, large-scale Gaussians, while fine-grained objects (e.g., cyclists) are modeled by dense, small-scale Gaussians. Scale factors are adaptively adjusted based on Gaussian value gradients, accounting for anisotropic shapes and spatial overlap when computing occupancy depth from Gaussian ellipsoids to voxels. Gaussians belonging to the same semantic category share similar scale factor ranges.
    - **Self-estimated Flow Module**: 3D flow vectors are estimated by minimizing Chamfer Distance between semantically corresponding adjacent Gaussian primitives across temporal frames. A dynamic indicator function $\mathbb{1}(D)$ determines whether an object is in motion; dynamic Gaussian primitives are assigned motion attributes accordingly, effectively resolving ghosting artifacts and spatiotemporal inconsistencies caused by dynamic objects.
    - **LiDAR Geometric Constraint (Optional)**: When LiDAR data is available, anchor centers are computed from multi-frame point cloud aggregation. A geometry-aware loss $L_{geo}$ is applied to align the distribution of Gaussian ellipsoids with the geometric priors of the corresponding semantic regions.

3. **Cumulative GS-Voxel Splatting**: VL-GS is cumulatively splatted onto a voxel grid of arbitrary resolution. The semantic label of each voxel is determined by a weighted combination of occupancy depth, opacity, and semantic probability of the contributing Gaussian primitives: $\digamma(o) = \sum_i d_i G(x_i) \alpha_i \text{softmax}(\gamma_i)$. The entire process is a forward pass, making it both efficient and accurate.

### Loss & Training
- AdamW optimizer with an initial learning rate of 0.005; positional parameters decay at a rate of 0.98 every 250 steps.
- Progressive image resolution scaling: starting at 225×400 and doubling every 300 steps to the original resolution.
- Geometric constraint loss $L_{geo}$ (when LiDAR is used): aligns Gaussian ellipsoid centers with LiDAR anchor points of the corresponding semantic regions.
- Semantic rendering is achieved via $\alpha$-blending; semantic class probabilities are computed using softmax.

## Key Experimental Results

### Semantic Occupancy Annotation on Occ3D-nuScenes

| Method | Input | IoU | mIoU | Type |
|--------|-------|-----|------|------|
| GaussianOcc | C | 51.22 | 12.59 | Self-supervised |
| LangOcc | C | 46.55 | 12.04 | Self-supervised |
| VEON | C | 57.92 | 14.51 | Self-supervised |
| SurroundOcc* | L | 68.87 | 18.59 | Point cloud voxelization |
| OpenOcc* | C&L | 70.59 | 17.76 | Point cloud voxelization |
| OVIR-3D | C&L | 54.30 | 18.47 | 2D-to-3D projection |
| VLM-LiDAR | C&L | 73.28 | 16.32 | 2D-to-3D projection |
| **AutoOcc-V** | **C** | **83.01** | **20.92** | **Ours (camera-only)** |
| **AutoOcc-M** | **C&L** | **88.62** | **25.84** | **Ours (multimodal)** |

AutoOcc-V (camera-only) outperforms the best self-supervised method VEON by +25.09 IoU (+43.3%) and +6.41 mIoU. AutoOcc-M surpasses the LiDAR-based OpenOcc* by +18.03 IoU and +8.08 mIoU.

### Zero-Shot Cross-Dataset Generalization on SemanticKITTI

| Method | Input | IoU | mIoU | mIoU-base |
|--------|-------|-----|------|-----------|
| GaussianOcc | C | 22.42 | 4.18 | 6.84 |
| OVO | C | 20.94 | 5.83 | 8.61 |
| SurroundOcc | L | 27.83 | 6.39 | 10.45 |
| VLM-LiDAR | C&L | 28.12 | 5.32 | 8.69 |
| **AutoOcc-V** | **C** | **35.64** | **9.36** | **12.02** |
| **AutoOcc-M** | **C&L** | **41.23** | **12.76** | **17.03** |

Under zero-shot cross-dataset evaluation, AutoOcc-M effectively annotates novel classes unseen during nuScenes training, whereas self-supervised methods largely fail (near-zero IoU on multiple categories).

### Annotation Efficiency Comparison

| Method | Annotation Time | Memory | No. of Primitives | Open-ended | Label-free |
|--------|-----------------|--------|-------------------|------------|------------|
| Auto+Human | 4,000+ human hours | — | 1.2M | ✗ | ✗ |
| SurroundOcc | 1,000+ GPU hours | 73G | 3.0M | ✗ | ✗ |
| GaussianOcc | 60 GPU hours | 32G | 0.8M | ✗ | ✓ |
| **AutoOcc** | **30 GPU hours** | **5.0G** | **0.3M** | **✓** | **✓** |

### Ablation Study
- Without Self-estimated Flow Module (w/o SFM): mIoU drops from 20.92 to 16.84 (−4.08), indicating that dynamic object handling is critical.
- Without Semantic-aware Scalable Gaussians (w/o SSG): IoU drops from 83.01 to 80.27 (−2.74), validating the importance of multi-scale modeling.
- Without LiDAR geometric constraint (w/o $L_{geo}$): IoU 83.01→81.49, mIoU 20.92→20.36, confirming the value of geometric priors.
- Among the three components, the self-estimated flow module has the largest impact on mIoU, while scalable Gaussians contribute most to IoU.

## Highlights & Insights
- **First fully automatic vision-centric pipeline for open-ended semantic occupancy annotation**, requiring no human labels or predefined categories.
- **Semantic-aware scalable Gaussian design** is elegant: primitives of the same semantic category share scale factor ranges, enabling adaptive representation — sparse large Gaussians for buildings, dense small Gaussians for pedestrians — with only 0.3M primitives for high-quality scene representation.
- **Effective dynamic object handling**: The self-estimated flow module combined with dynamic Gaussian primitives resolves ghosting artifacts common in reconstruction-based methods and enables reasoning about occluded regions.
- **Exceptional efficiency**: Full dataset annotation in 30 GPU hours with only 5G memory, representing 1/33 the time and 1/14 the memory of SurroundOcc.
- **Strong generalization**: Maintains a clear advantage in zero-shot evaluation on SemanticKITTI, and even surpasses human annotation performance in adverse weather conditions (rain, nighttime).

## Limitations & Future Work
- The quality of VLM-generated semantic attention maps is bounded by the VLM's capabilities, potentially missing rare or fine-grained objects.
- In camera-only mode, reliance on UniDepth for depth estimation may introduce geometric errors in complex scenes.
- Although open-ended categories are supported, the semantic merging strategy relies on heuristic rules (gradient thresholds), which may fail under extreme long-tail distributions.
- The paper does not address applicability to online or real-time annotation scenarios; the current pipeline is offline.
- Categories such as construction vehicles remain challenging across all methods (AutoOcc-M achieves only 4.32 IoU), indicating room for improvement in modeling rare, small categories.

## Related Work & Insights
- **vs. GaussianOcc**: GaussianOcc also uses Gaussian splatting as an intermediate representation for self-supervised occupancy estimation, but employs vanilla Gaussians without semantic-aware multi-scale modeling or dynamic object handling, achieving only IoU 51.22 compared to AutoOcc-V's 83.01. AutoOcc achieves a qualitative leap through VLM guidance and semantic-aware scalable Gaussians.
- **vs. VEON**: VEON (ECCV 2024) is an open-vocabulary occupancy estimation method based on self-supervision and visual foundation model features. AutoOcc leads by 25+ IoU points; the core difference lies in AutoOcc's use of differentiable reconstruction as its central engine to guarantee 3D consistency, whereas VEON's 2D feature lifting cannot effectively resolve multi-view semantic conflicts.
- **vs. OpenOcc/SurroundOcc**: These traditional annotation pipelines rely on LiDAR and manual priors, with complex and time-consuming multi-stage processing. AutoOcc comprehensively outperforms them even with camera-only input, at one to two orders of magnitude faster annotation speed.
- AutoOcc's paradigm of "using VLMs to automatically generate annotations for training downstream models" offers a new data flywheel framework, enabling large-scale occupancy annotation data generation at minimal cost.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First work to combine VLM guidance with semantic-aware scalable Gaussians for fully automatic open-ended occupancy annotation, integrating multiple innovations systematically; however, the combination of VLMs and Gaussian splatting has precedents in other tasks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers two datasets (nuScenes and KITTI), three categories of baseline comparisons (point cloud voxelization / 2D-to-3D projection / self-supervised), zero-shot generalization, efficiency analysis, ablation studies, and qualitative analysis under adverse weather — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and Table 1's method comparison is intuitive; however, the method section contains numerous formulas and some symbol definitions could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a highly practical problem (occupancy annotation cost) with an order-of-magnitude efficiency improvement (30 GPU hours vs. 4,000+ human hours); open-ended capability is critical for autonomous driving safety — well-deserving of Highlight recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[NeurIPS 2025\] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](../../NeurIPS2025/3d_vision/planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Online Language Splatting](online_language_splatting.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)

</div>

<!-- RELATED:END -->
