---
title: >-
  [Paper Note] SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs
description: >-
  [ECCV 2024][3D Vision][Coarse Localization] SceneGraphLoc is proposed to perform coarse localization of query images within a reference map composed of multimodal 3D scene graphs. Without relying on large-scale image databases, it achieves localization accuracy comparable to state-of-the-art image-level methods while reducing storage requirements by three orders of magnitude.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Coarse Localization"
  - "3D Scene Graph"
  - "Cross-Modal"
  - "Contrastive Learning"
  - "Scene Retrieval"
date: 2026-05-08
content_hash: 9bc6bb25c6ce20df
---

# SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs

**Conference**: ECCV 2024  
**arXiv**: [2404.00469](https://arxiv.org/abs/2404.00469)  
**Code**: Yes (scenegraphloc.github.io)  
**Area**: 3D Vision  
**Keywords**: Coarse Localization, 3D Scene Graph, Cross-Modal, Contrastive Learning, Scene Retrieval

## TL;DR

SceneGraphLoc is proposed to perform coarse localization of query images within a reference map composed of multimodal 3D scene graphs. Without relying on large-scale image databases, it achieves localization accuracy comparable to state-of-the-art image-level methods while reducing storage requirements by three orders of magnitude.

## Background & Motivation

Coarse visual localization (place recognition) is a fundamental task in computer vision and robotics, which is typically modeled as an image retrieval problem: matching query images against a large-scale database of geo-localized images. However, current state-of-the-art methods (such as AnyLoc) rely heavily on massive image databases, resulting in substantial storage overhead and slow query speed. Although cross-modal methods attempt to bridge different data modalities, they are usually limited to matching between only two modalities (e.g., image-to-point-cloud), which restricts their applicability.

This paper introduces a novel problem setup: localizing query images within a multimodal reference map composed of 3D Scene Graphs (3DSGs). 3DSGs integrate multiple modalities such as point clouds, images, semantic categories, object attributes, and relations, serving as a lightweight and efficient scene representation. The core advantage of this setup is that once the scene graph is constructed, each node only needs to store a fixed-size embedding vector, completely eliminating the need to preserve the raw image database.

## Method

### Overall Architecture

SceneGraphLoc consists of two parallel branches for embedding generation:

1. **Scene Graph Node Embedding**: Generates a fixed-dimensional embedding $e_v \in \mathbb{R}^D$ for each node (object instance) in the scene graph, fusing five modalities: point clouds, images, structure, attributes, and relationships.
2. **Query Image Embedding**: Segments the query image into regular patches and generates an embedding $e_q \in \mathbb{R}^D$ for each patch, representing the visible objects in that patch.

The training goal is to use contrastive learning to pull the embeddings of positive pairs (patches and nodes corresponding to the same object) close to 0 in distance, while pushing negative pairs apart. During inference, scene retrieval is accomplished through nearest-neighbor matching and similarity scoring.

### Key Designs

**Multimodal Node Embedding**: Each scene graph node merges five modalities:
- **Point Cloud Embedding** $e_v^{\mathcal{P}}$: Extracts geometric features from object-level point clouds using PointNet.
- **Image Embedding** $e_v^{\mathcal{I}}$: Selects the top-$K_{view}=10$ images with the highest visibility for each object, extracts multi-level features using multi-scale bounding box cropping and DINOv2, and then aggregates multi-view information via a Transformer encoder.
- **Structural Embedding** $e_v^{\mathcal{S}}$: Encodes relative spatial relations between objects using a Graph Attention Network (GAT).
- **Attribute Embedding** $e_v^{\mathcal{A}}$ and **Relationship Embedding** $e_v^{\mathcal{R}}$: Encoded using Bag-of-Words features combined with Feed-Forward Networks (FFN), respectively.

The five modalities are concatenated using softmax attention weighting and then mapped to a unified dimension $D$ via a two-layer MLP:

$$e_v = \text{MLP}\left(\bigoplus_{k \in \mathcal{K}} \frac{\exp(w_k)}{\sum_j \exp(w_j)} e_v^k\right)$$

**Query Image Embedding**: DINOv2 is utilized as the backbone to extract patch-level features, which are then mapped to dimension $D$ through a 4-layer CNN residual block and a 3-layer MLP. Compared to directly using panoptic segmentation (which is prone to over- or under-segmentation), the patch-based strategy is more robust.

**Scene Graph-Image Similarity**: For each candidate scene graph $\mathcal{G}_i$, the average similarity between all patches and their nearest-neighbor nodes is computed:

$$s(\mathcal{G}_i, I) = \frac{1}{|\mathcal{Q}_I|} \sum_{q \in \mathcal{Q}_I} [1 - \delta(e_q, \text{NN}(q))]$$

### Loss & Training

A bidirectional N-pair contrastive loss is adopted, comprising static loss and temporal loss:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{static}} + (1 - \alpha) \cdot \mathcal{L}_{\text{temp}}$$

- **Static Loss**: The query image and its concurrent scene graph form a positive pair.
- **Temporal Loss**: Leverages scans of the same room at different times in 3RScan (where objects may move or lighting may change) to enhance robustness against temporal variations.

Negative samples originate from patches within the same image that observe different objects (image-side negatives) and nodes from different scenes (scene-graph-side negatives). Bidirectional computation renders the embedding space more discriminative.

## Key Experimental Results

### Main Results

Recall@K results on the 3RScan dataset (50-scene selection, temporal scenes):

| Method | Map Modality | $R^t$@1 | $R^t$@3 | $R^t$@5 | Storage (MB) |
|------|---------|---------|---------|---------|---------|
| LidarCLIP | Point Cloud | 14.1 | 10.3 | 15.6 | 1000.4 |
| LIP-Loc | Point Cloud | 12.3 | 18.6 | 15.2 | 1001.0 |
| OpenMask3D | Point Cloud + Image | 21.1 | 38.1 | 48.0 | 1020.1 |
| **SceneGraphLoc (w/o Image)** | Point Cloud + Others | **28.2** | **46.2** | **56.4** | **1005.4** |
| **SceneGraphLoc (w/ Image)** | Full Modalities | **69.3** | **78.6** | **84.4** | **1005.4** |
| CVNet | Image | 66.5 | 77.0 | 81.7 | 1239.1 |
| AnyLoc | Image | 80.6 | 87.4 | 90.0 | 5720.3 |

### Ablation Study

Modality ablation on the 3RScan validation set (10-scene selection):

| Point Cloud | Image | Attribute | Structure | Relation | R@1 (DINOv2) | $R^t$@1 (DINOv2) |
|------|------|------|------|------|-------------|-----------------|
| ✓ | | | | | 45.2 | 43.9 |
| ✓ | | ✓ | | | 56.3 | 54.8 |
| ✓ | | ✓ | ✓ | | 58.4 | 56.5 |
| ✓ | | ✓ | ✓ | ✓ | 63.7 | 62.7 |
| ✓ | ✓ | | | | - | 80.2 |
| ✓ | ✓ | ✓ | ✓ | ✓ | - | 88.5 |

### Key Findings

1. **Significant Lead in Cross-Modal Setup**: Even without using the image modality, SceneGraphLoc significantly outperforms other cross-modal methods (LidarCLIP, LIP-Loc, OpenMask3D).
2. **Accuracy Close to Image-Based Methods with 1000x Less Storage**: The image-enabled version of SceneGraphLoc achieves an $R^t$@1 of 69.3% on 3RScan (vs 80.6% for AnyLoc), but its storage requirements are reduced by up to three orders of magnitude.
3. **Contribution from Every Modality**: From point-cloud-only to full modalities, R@1 improves from 45.2% to 63.7% (without image) and 88.5% (with image).
4. **DINOv2 >> GCVit**: Extracting image features using DINOv2 yields significantly better performance than GCVit.
5. **Inference Speed Advantage**: Retrieving from 50 scenes takes only about 1 ms for SceneGraphLoc, whereas AnyLoc requires 1826 ms.

## Highlights & Insights

- **Novel Problem Formulation**: For the first time, the task of localizing query images within multimodal 3D scene graphs is proposed, representing a highly promising lightweight localization paradigm.
- **Knowledge-Distillation-Style Design**: The mapping phase distills multimodal information into fixed-size embeddings. During inference, there is no need to access the raw modal data, achieving dual advantages in both storage and speed.
- **Temporal Robustness**: Utilizing scans of the same scene at different times as positive samples makes the model robust to environmental changes.

## Limitations & Future Work

- When the query image contains few visible object categories (e.g., predominantly walls), localization tends to fail due to the lack of sufficient discriminative information.
- When using predicted scene graphs from SceneGraphFusion on ScanNet, performance degrades due to imprecise instance segmentation and the absence of attribute annotations.
- A performance gap still exists compared to pure image-based methods (e.g., AnyLoc), especially when the number of scenes is large.
- Introducing additional modalities, such as textual descriptions or floor plans, could be considered to further improve performance.

## Related Work & Insights

- Scene graphs are widely applied in fields such as embodied AI, SLAM, and task planning; this work demonstrates their immense potential in localization tasks.
- Contrastive learning combined with multimodal fusion is an effective paradigm for cross-modal matching, which can be generalized to other cross-modal retrieval tasks.
- The patch-level features of DINOv2 exhibit superior performance in object-level matching tasks and are worthy of further exploration in more visual localization scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel problem setup, utilizing scene graphs for coarse visual localization for the first time
- **Technical Quality**: ⭐⭐⭐⭐ — Rational multimodal fusion design, with a comprehensive contrastive learning framework
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets + multiple baselines + detailed ablation studies
- **Value**: ⭐⭐⭐⭐⭐ — 1000x storage reduction, highly valuable for robotics/AR deployment
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CrossOver: 3D Scene Cross-Modal Alignment](../../CVPR2025/3d_vision/crossover_3d_scene_cross-modal_alignment.md)
- [\[ECCV 2024\] The NeRFect Match: Exploring NeRF Features for Visual Localization](the_nerfect_match_exploring_nerf_features_for_visual_localization.md)
- [\[ECCV 2024\] Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion](explicitly_guided_information_interaction_network_for_cross-modal_point_cloud_co.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](sceneverse_scaling_3d_vision-language_learning_for_grounded_scene_understanding.md)

</div>

<!-- RELATED:END -->
