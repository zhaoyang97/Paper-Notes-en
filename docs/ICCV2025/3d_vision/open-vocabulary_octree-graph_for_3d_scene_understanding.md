---
title: >-
  [Paper Note] Open-Vocabulary Octree-Graph for 3D Scene Understanding
description: >-
  [ICCV 2025][3D Vision][Open-vocabulary] This paper proposes Octree-Graph, a novel scene representation combining adaptive octrees with a graph structure. Through Chronological Group-based Segment Merging (CGSM) and Insta…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Open-vocabulary"
  - "3D scene understanding"
  - "octree"
  - "scene graph"
  - "semantic segmentation"
date: 2026-05-08
content_hash: 92d915d1dca7ab12
---

# Open-Vocabulary Octree-Graph for 3D Scene Understanding

**Conference**: ICCV 2025
**arXiv**: [2411.16253](https://arxiv.org/abs/2411.16253)
**Code**: [GitHub](https://github.com/wangzg1/Octree-Graph)
**Area**: 3D Vision
**Keywords**: Open-vocabulary, 3D scene understanding, octree, scene graph, semantic segmentation

## TL;DR

This paper proposes Octree-Graph, a novel scene representation combining adaptive octrees with a graph structure. Through Chronological Group-based Segment Merging (CGSM) and Instance Feature Aggregation (IFA), it obtains accurate semantic objects and enables efficient open-vocabulary 3D scene understanding.

## Background & Motivation

Open-vocabulary 3D scene understanding is critical for embodied agents. Existing methods leverage pretrained VLMs for object segmentation and project results onto point clouds to construct 3D maps, but suffer from two core issues:

**Inefficient spatial representation**: Mainstream methods build 3D maps on point clouds, which are unordered discrete coordinates requiring large storage, and lack explicit occupancy information and spatial connectivity, hindering downstream tasks such as path planning and text-based retrieval.

**Inaccurate semantic segmentation**: Existing methods overlook the inherent imprecision of foundation models in segmentation and feature extraction, leading to degraded 3D object segmentation and semantic quality.

## Method

### Overall Architecture

Given an RGB-D sequence and camera poses, the pipeline consists of four steps:
1. Extract 2D segmentation proposals and semantic features using VLMs
2. Merge segments into instances via CGSM
3. Aggregate features for each instance via IFA
4. Construct the Octree-Graph for downstream applications

### Chronological Group-based Segment Merging (CGSM)

Existing segment merging strategies are either frame-level, which is susceptible to noise, or global-level, which introduces redundant computation. The key designs of CGSM are:

- **Chronological grouping**: Frames are divided into groups in temporal order with interval $I$, preserving spatiotemporal details of adjacent frames while avoiding global interference.
- **Semantics-guided under-segmentation filtering**: For under-segmented segments $\mathcal{S}_m$ that may contain distinct objects, the variance of semantic features among internal segments is computed; those exceeding threshold $\tau_u$ are filtered out.
- **Dynamic threshold decay**: The overall similarity is $\phi = \phi_{\text{geo}}^{\text{iou}} + \phi_{\text{geo}}^{\text{ior}} + \phi_{\text{sem}}^{v} + \phi_{\text{sem}}^{c}$, and threshold $\theta_i$ decays linearly to merge partially observed segments with low spatial overlap.

### Instance Feature Aggregation (IFA)

Semantic features are fused for each instance, jointly considering representativeness and discriminability:

$$a_{i,j}^{v} = \cos(\mathbf{f}_{i,j}^{v}, \bar{\mathbf{f}}_{i}^{v}) - \sum_{\mathcal{O}_k \in \mathcal{N}_i} \cos(\mathbf{f}_{i,j}^{v}, \bar{\mathbf{f}}_{k}^{v})$$

where weights are normalized via softmax. The intuition is that features closer to their own cluster center and farther from neighboring instances receive higher aggregation weights.

### Adaptive Octree

Conventional octrees use cubic voxels, requiring very deep trees to approximate objects with large aspect ratios (e.g., walls). The adaptive octree adjusts the size of each node according to object shape:

$$\mathbf{d}_l = (\mathbf{b}_{\max} - \mathbf{b}_{\min}) / 2^l$$

where $\mathbf{b}_{\max}$ and $\mathbf{b}_{\min}$ are the corner coordinates of the object bounding box, and $l$ is the octree depth.

### Octree-Graph Construction

- **Node** $\mathbf{N}_i$: contains semantics $n_i^s$, center $n_i^c$, and adaptive octree $n_i^o$
- **Edge** $\mathbf{E}_{i,j}$: contains semantic relation $e_{i,j}^s$, spatial distance $e_{i,j}^d$, and 3D direction vector $\mathbf{e}_{i,j}^v$

## Key Experimental Results

### Main Results — 3D Semantic Segmentation

| Method | Replica mIoU↑ | ScanNet mIoU↑ | ScanNet mAcc↑ |
|--------|--------------|---------------|---------------|
| ConceptFusion | 0.10 | 0.08 | 0.15 |
| ConceptGraph | 0.18 | 0.16 | 0.28 |
| HOV-SG | 0.231 | 0.222 | 0.431 |
| **Ours** | **0.320** | **0.393** | **0.601** |

The proposed method significantly outperforms all baselines on both Replica and ScanNet, with gains of +17.1% mIoU and +17.0% mAcc over HOV-SG on ScanNet.

### 3D Instance Segmentation (ScanNet200)

| Method | AP↑ | AP50↑ | AP25↑ |
|--------|-----|-------|-------|
| Mask-Clustering | 12.0 | 23.3 | 30.1 |
| **Ours (z.s.)** | **14.3** | **25.8** | **33.6** |

### Ablation Study — Merging Strategy

| Merging Strategy | mIoU↑ | mAcc↑ |
|-----------------|-------|-------|
| Frame-level | 0.323 | 0.519 |
| Global-level | 0.286 | 0.476 |
| **CGSM (I=200)** | **0.356** | **0.574** |

### Path Planning

| Method | SR(1.0m) | SR(0.5m) | SR(0.25m) |
|--------|----------|----------|-----------|
| HOV-SG | 55.25 | 46.75 | 32.16 |
| **Ours** | **97.88** | **96.88** | **96.38** |

The proposed method substantially outperforms HOV-SG in path planning success rate, as Octree-Graph supports navigation to arbitrary free-space regions.

## Highlights & Insights

1. The adaptive octree elegantly addresses the inefficiency of conventional octrees in representing objects with large aspect ratios.
2. CGSM achieves a balance between exploiting local temporal details and avoiding global redundancy through its grouping strategy.
3. IFA performs weighted aggregation by accounting for inter-instance discriminability, yielding greater robustness than simple averaging.
4. Storage is drastically reduced: all adaptive octrees together occupy only 42 KB, compared to 6.8 MB for point clouds.

## Limitations & Future Work

- The method depends on the segmentation quality of 2D foundation models, and challenges remain for small objects and occluded scenes.
- The octree depth $L_{\max}=4$ may be insufficient to represent highly fine-grained geometric details.
- Path planning may still incorrectly mark certain free regions as occupied due to octree discretization.

## Related Work & Insights

- ConceptGraph, HOV-SG: 3D scene graph methods
- OpenScene, ConceptFusion: point/mesh-level 3D maps
- OVIR-3D, MaskClustering: instance-level 3D maps
- PlenOctrees, OctreeOcc: octree structures for rendering/semantics

## Rating

- Novelty: ⭐⭐⭐⭐ (the hybrid representation of adaptive octrees and graph structure is highly novel)
- Technical Depth: ⭐⭐⭐⭐ (CGSM and IFA are elegantly designed)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 tasks across 4 datasets)
- Practical Value: ⭐⭐⭐⭐⭐ (highly valuable for embodied agents)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](../../NeurIPS2025/3d_vision/openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)
- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)

</div>

<!-- RELATED:END -->
