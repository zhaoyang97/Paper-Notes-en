---
title: >-
  [Paper Note] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] This paper proposes a novel "Segment-then-Splat" paradigm that assigns Gaussians to distinct object sets prior to 3D reconstruction…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Open-Vocabulary Segmentation"
  - "Dynamic Scenes"
  - "CLIP Embeddings"
  - "Object Tracking"
date: 2026-05-08
content_hash: 0d2b75eecd0b259c
---

# Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2503.22204](https://arxiv.org/abs/2503.22204)  
**Code**: [GitHub](https://vulab-ai.github.io/Segment-then-Splat/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Open-Vocabulary Segmentation, Dynamic Scenes, CLIP Embeddings, Object Tracking

## TL;DR

This paper proposes a novel "Segment-then-Splat" paradigm that assigns Gaussians to distinct object sets prior to 3D reconstruction, thereby eliminating geometric and semantic ambiguity and enabling unified 3D open-vocabulary segmentation for both static and dynamic scenes.

## Background & Motivation

3D open-vocabulary querying is critical for applications in robotics, autonomous driving, and augmented reality. Virtually all existing methods follow a "Splat-then-Segment" paradigm, which suffers from three fundamental problems:

**Limitations of 2D pixel-level segmentation**: Most methods (e.g., LangSplat, LEGaussians) learn a language field alongside 3DGS, rendering language embeddings into 2D feature maps for querying. This is essentially 2D segmentation, yielding inconsistent results across viewpoints and failing to extract genuine 3D object information.

**Geometric-semantic ambiguity**: Under the Splat-then-Segment paradigm, individual Gaussians may encode geometric and semantic information from multiple objects, leading to blurred object boundaries.

**Inapplicability to dynamic scenes**: In dynamic settings, the same Gaussian may represent different objects at different timesteps, causing Gaussian–object misalignment that existing methods cannot directly handle.

The core insight of this paper is that if Gaussians are assigned to individual objects before reconstruction, each Gaussian belongs to exactly one object, eliminating the aforementioned ambiguities and naturally extending to dynamic scenes.

## Method

### Overall Architecture

The Segment-then-Splat pipeline consists of four stages: (1) a robust object tracking module extracts multi-view masks; (2) COLMAP-initialized Gaussians are assigned to distinct object sets according to these masks; (3) reconstruction is optimized under per-object set constraints; (4) CLIP embeddings are associated with each object set to enable open-vocabulary querying.

### Key Designs

1. **Robust Object Tracking Module**: SAM is applied on the first frame using a grid-point prompt to generate initial segmentations, followed by SAM2 to track objects throughout the sequence. Three post-processing strategies are designed to address practical challenges:

    - **New Object Detection**: The ratio of segmented regions is compared every $\Delta t$ frames to detect newly appearing objects.
    - **Multiple-Tracking Resolution**: Overlapping masks are filtered via IoU thresholds to ensure each pixel belongs to exactly one object at a given granularity.
    - **Lost-Tracking Recovery**: A "geometry–appearance distance" between Gaussian sets is used to merge instances representing the same object:
    $d(\mathbf{G}_i, \mathbf{G}_j) = \lambda_d |\overline{\mathbf{M}_i} - \overline{\mathbf{M}_j}|_2 + (1 - \lambda_d) |\overline{\mathbf{C}_i} - \overline{\mathbf{C}_j}|_2$

2. **Object-Specific Gaussian Initialization**: The visibility of each Gaussian center across all viewpoints is analyzed to determine its corresponding object mask region, assigning a three-level (large/medium/small) object ID. For objects not covered by COLMAP, randomly initialized compensation Gaussians are added; background Gaussians are generated to fill unsegmented regions.

3. **Multi-Granularity Constrained Optimization**: In addition to the standard rendering loss, an object-level loss is introduced:
    $\mathcal{L}_{obj} = \mathcal{L}_1(M_i^p \otimes I_i, \hat{I_i^p})$
   To avoid the computational overhead of supervising all objects per iteration, only $m$ objects are randomly sampled at each step. Optimization proceeds from fine to coarse granularity (small → medium → large), since small objects are subsets of larger ones; reversing the order disrupts the internal structure of large objects.

4. **Partial Mask Filtering**: 3D segmentation can reconstruct occluded regions of objects, but 2D mask supervision does not cover occluded parts, introducing supervision bias. Toward the end of training, each reconstructed object is rendered into 2D images and IoU with the masks is computed; low-IoU masks are discarded so that final optimization is guided only by consistent masks.

### Loss & Training

The overall loss is: $\mathcal{L} = \mathcal{L}_{render} + \mathcal{L}_{obj}$

where the rendering loss is: $\mathcal{L}_{render} = (1-\lambda_r)\mathcal{L}_1(\hat{I_i}, I_i) + \lambda_r \mathcal{L}_{DSSIM}(\hat{I_i}, I_i)$

The object loss is introduced progressively across stages: stage 1 optimizes only fine granularity, stage 2 adds medium granularity, and stage 3 optimizes all three levels simultaneously. After training, CLIP embeddings are computed for each object set (averaged across viewpoints after partial mask filtering) to enable open-vocabulary querying.

## Key Experimental Results

### Main Results

**Static scene segmentation (mIoU↑ / Training time min↓):**

| Method | LERF_OVS mIoU | LERF_OVS Time | 3DOVS mIoU | 3DOVS Time |
|--------|--------------|---------------|-----------|-----------|
| LangSplat (2D) | 46.37 | 62 | 82.49 | 68.9 |
| G-Grouping (2D) | 29.59 | 77 | 76.24 | 56.1 |
| OpenGaussian (3D) | 42.43 | 69.75 | 31.00 | 59.4 |
| **Ours (3D)** | **52.10** | **50.75** | **88.53** | **9.4** |

**Dynamic scene segmentation:**

| Method | HyperNeRF mIoU | HyperNeRF Time | Neu3D mIoU | Neu3D Time |
|--------|---------------|----------------|-----------|-----------|
| DGD (3D) | 7.83 | 1564.5 | 1.65 | 1733 |
| **Ours (3D)** | **69.48** | **218** | **44.00** | **161.3** |

### Ablation Study

| Configuration | ramen mIoU | waldo_kitchen mIoU | Note |
|--------------|-----------|-------------------|------|
| Supervise 1 object per iteration | 51.09 | 33.97 | Baseline |
| Supervise 3 objects per iteration | 54.38 | 40.71 | Better balance |
| Supervise 9 objects per iteration | 56.48 | 41.59 | Diminishing returns |
| Without partial mask filtering | 42.19 | 31.94 | Significant drop |
| **With partial mask filtering** | **54.38** | **40.71** | Substantial gain in complex scenes |

### Key Findings

- 3D segmentation "sees" occluded regions, causing incomplete GT mask coverage that slightly lowers mIoU; this actually indicates more complete 3D segmentation rather than a deficiency.
- Optimization in dynamic scenes is nearly 10× faster than DGD, as no dynamic language field needs to be learned.
- Each component of the robust tracking module (new object detection, multiple-tracking resolution, lost-tracking recovery) contributes substantively to overall performance.

## Highlights & Insights

- **Paradigm innovation**: The paper inverts the long-standing Splat-then-Segment paradigm into Segment-then-Splat, yielding a conceptually clean and highly effective approach.
- **Unified framework**: Static and dynamic scenes are handled within the same framework without any scene-specific modifications.
- **Efficiency**: No auxiliary language field is required; a single reconstruction pass suffices, substantially reducing training time.
- **Multi-granularity**: Large/medium/small three-level granularity queries are supported, with an optimization ordering strategy that ensures structural integrity at each level.

## Limitations & Future Work

- The method relies on SAM2 for initial segmentation and tracking, which may fail in highly dense or visually similar complex scenes.
- Text queries involving relational descriptions across multiple objects (e.g., "the sheep on the chair in front of the table") cannot be handled, as semantic encoding is performed independently per object.
- Kalman filtering could be introduced to improve the temporal consistency of SAM2 tracking.

## Related Work & Insights

This paper contrasts with OpenGaussian (contrastive learning + K-means clustering) and GaussianCut (graph-cut-based segmentation). The core insight is that rather than attempting to disentangle mixed Gaussian semantics post-reconstruction, it is preferable to maintain Gaussian–object correspondence from the outset. This principle generalizes to other 3D representation learning tasks requiring semantic decomposition.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Paradigm-level innovation; placing segmentation before reconstruction is a novel and elegant idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both static and dynamic datasets with thorough ablations, though large-scale scene experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive illustrations, and well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ Unifies 3D open-vocabulary segmentation for static and dynamic scenes with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](../../CVPR2026/3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](../../CVPR2026/3d_vision/extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[NeurIPS 2025\] Online Segment Any 3D Thing as Instance Tracking](online_segment_any_3d_thing_as_instance_tracking.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
