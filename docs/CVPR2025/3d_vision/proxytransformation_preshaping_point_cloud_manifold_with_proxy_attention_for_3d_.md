---
title: >-
  [Paper Note] ProxyTransformation: Preshaping Point Cloud Manifold with Proxy Attention for 3D Visual Grounding
description: >-
  [CVPR 2025][3D Vision][3D Visual Grounding] This paper proposes ProxyTransformation, which enhances the point cloud manifold structure efficiently before training via deformable point clustering and proxy attention mechanisms. It utilizes textual information to guide sub-manifold translation and image information to guide intra-sub-manifold transformations, achieving a significant improvement of 7.49% on the ego-centric 3D visual grounding task.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Visual Grounding"
  - "Point Cloud Enhancement"
  - "Deformable Clustering"
  - "Proxy Attention"
  - "Manifold Transformation"
date: 2026-05-08
content_hash: 82bfe1a5e3902152
---

# ProxyTransformation: Preshaping Point Cloud Manifold with Proxy Attention for 3D Visual Grounding

**Conference**: CVPR 2025  
**arXiv**: [2502.19247](https://arxiv.org/abs/2502.19247)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Visual Grounding, Point Cloud Enhancement, Deformable Clustering, Proxy Attention, Manifold Transformation

## TL;DR

This paper proposes ProxyTransformation, which enhances the point cloud manifold structure efficiently before training via deformable point clustering and proxy attention mechanisms. It utilizes textual information to guide sub-manifold translation and image information to guide intra-sub-manifold transformations, achieving a significant improvement of 7.49% on the ego-centric 3D visual grounding task.

## Background & Motivation

Ego-centric 3D visual grounding (3DVG) is a core perception capability for embodied AI, requiring the localization of 3D targets in multi-view RGB-D observations based on language descriptions. However, it faces several key challenges:

1. **Poor Point Cloud Quality**: Point clouds reconstructed from depth sensors contain significant noise (such as depth errors from non-Lambertian surfaces) and are sparsely sampled (only ~2%) due to computational limits, destroying the manifold structure in target regions.
2. **Background Redundancy**: Most of the sampled points lie in the background area, leading to insufficient density of foreground target points.
3. **Existing Enhancement Methods are Inapplicable**: Traditional point cloud denoising/completion methods require time-consuming preprocessing, making them unsuitable for real-time scenarios; moreover, they only operate on the single point cloud modality and fail to leverage available multi-modal information in the task.
4. **Infeasible Global Transformation**: The manifold structures of different local regions in-scene level point clouds vary significantly, making them impossible to address with a single global transformation.

**Core Problem**: How can one leverage multi-modal information to enhance point cloud sub-manifolds in real-time without increasing offline preprocessing overhead?

## Method

### Overall Architecture

The ProxyTransformation module is integrated into the EmbodiedScan baseline. First, key sub-manifold regions are localized via deformable point clustering. Then, Proxy Attention is leveraged to jointly utilize textual and image information to learn transformation matrices and translation vectors for each sub-manifold. Finally, the transformed point cloud is fed into the 3D backbone.

### Key Designs

**Design 1: Deformable Point Clustering**

- **Function**: Adaptively selects the most critical point cloud sub-regions for transformation.
- **Mechanism**: Initializes 3D uniform grids as reference points and performs ball query centered at these reference points to obtain initial clusters. Then, a 3D offset network (a lightweight CNN) is used to predict the offset for each reference point, shifting the cluster centers toward critical regions. Finally, it clustered again using these offset reference points to obtain better sub-manifold regions.
- **Design Motivation**: Uniform grids provide stable spatial priors, offsetting the loss of geometric information caused by sparse sampling; deformable offsets allow cluster centers to adaptively move to foreground/target regions, increasing diversity.

$$\hat{q}_t = q_t + \Gamma_{offset}(\mathcal{N}_t)$$

**Design 2: Proxy Attention**

- **Function**: Achieves cross-modal feature interaction with linear complexity.
- **Mechanism**: Introduces proxy tokens $P$ (which can be text or image features) to decompose the standard $O(N^2)$ self-attention into two steps: first, key-value features are compressed via proxy tokens ($\text{Attn}(P,K,V)$), and then the original query broadcasts information from the compressed representation ($\text{Attn}(Q,P,V^P)$), reducing the complexity to $O(Nnd)$ where $n \ll N$.
- **Design Motivation**: The number of tokens $N$ in scene-level point clouds is massive, rendering standard self-attention computationally infeasible. Proxy attention uses a small number of proxy tokens ($n \ll N$) as an information bottleneck to enable linear-complexity global interactions.

$$O^P = \sigma(QP^T) \sigma(PK^T) V$$

**Design 3: Dual-path Decomposition of Text-guided Translation and Image-guided Transformation**

- **Function**: Leverages complementary information from different modalities to optimize inter-sub-manifold and intra-sub-manifold geometric structures, respectively.
- **Mechanism**: Textual features containing global spatial relation information (such as "the chair next to the table") are used as proxy tokens to guide the learning of inter-sub-manifold translation vectors $T \in \mathbb{R}^{n \times 3}$. Image features containing local fine-grained semantics (textures, poses) are used as proxy tokens to guide the learning of intra-sub-manifold linear transformation matrices $M \in \mathbb{R}^{n \times 3 \times 3}$.
- **Design Motivation**: Any 3D spatial transformation can be decomposed into a linear transformation and a translation. Text is adept at expressing spatial relations, making it suitable for guiding translation, whereas images excels at capturing local details, making them suitable for guiding internal scale/rotation adjustments.

$$\hat{\mathcal{P}} = \mathcal{M} \odot \mathcal{P} \oplus \mathcal{T}$$

### Loss & Training

The model uses the same 9-DoF bounding box regression loss as the baseline, and the ProxyTransformation module is trained end-to-end to implicitly learn the optimal transformations.

## Key Experimental Results

### Main Results on the EmbodiedScan Validation Set

| Method | Training Set | Easy AP25 | Hard AP25 | Overall AP25 | Overall AP50 |
|------|--------|-----------|-----------|-------------|-------------|
| EmbodiedScan | Full | 39.82 | 31.02 | 39.10 | 18.48 |
| EmbodiedScan | Mini | 33.87 | 30.49 | 33.59 | 14.40 |
| DenseG | Mini | 40.17 | 34.38 | 39.70 | 18.31 |
| **ProxyTransformation** | **Mini** | **41.66** | **34.38** | **41.08** | **19.00** |

### Ablation Study

| Component | Easy AP25 | Hard AP25 | Overall AP25 |
|------|-----------|-----------|-------------|
| Baseline| 37.05 | 30.60 | 36.53 |
| + Grid Prior | 40.39 | 32.60 | 39.76 |
| + Offsets | 40.59 | 32.18 | 39.91 |
| + Proxy Transformation | **41.66** | **34.38** | **41.08** |

### Attention Efficiency Comparison

| Attention Type | FLOPs | Params | Overall AP25 |
|-----------|-------|--------|-------------|
| Self-Attention | 8.36G | 2.52M | 40.14 |
| **Proxy Attention** | **5.29G** | **1.82M** | **41.08** |

### Key Findings

1. Using only the Mini training set (~20% data), ProxyTransformation outperforms the baseline trained on the full training set, demonstrating the effectiveness of manifold enhancement.
2. The Grid Prior alone contributes a +3.23 gain in Overall AP25, indicating that the uniform grid prior plays a crucial role in compensating for sparse point clouds.
3. Proxy Attention outperforms Self-Attention while requiring only approximately 63% of the FLOPs, verifying the efficiency-accuracy advantage of the proxy mechanism.
4. The computational overhead of the attention blocks is reduced by 40.6%, which is highly significant for real-time applications.

## Highlights & Insights

1. **Precise Work Division of Multi-modal Information**: The design of text-to-global translation and image-to-local transformation aligns perfectly with the characteristics of the two modalities.
2. **New Paradigm for Point Cloud Enhancement**: Enhancing point clouds through coordinate transformations before feature learning, rather than processing in the feature space, is more direct and computationally efficient.
3. **Generality of Proxy Attention**: As a general linear-complexity attention mechanism, it allows the selection of different proxy tokens based on task requirements.

## Limitations & Future Work

1. Transformation parameters (e.g., number of clusters, points per cluster) need to be manually tuned and might require adjustment for different scenes.
2. Sub-manifold transformations may cause overlaps or self-intersections, which currently lacks geometric constraint constraints.
3. Evaluation was only performed on the EmbodiedScan benchmark, leaving broader 3D understanding tasks untested.
4. Extending ProxyTransformation to dynamic scenarios (such as video-level 3DVG) could be explored.

## Related Work & Insights

- **EmbodiedScan**: A benchmark framework for ego-centric 3DVG, upon which this work is constructed.
- **Deformable DETR / DCNv3**: The concept of deformable offsets is extended from 2D to 3D point cloud clustering.
- **Linear Attention**: Proxy attention shares similar complexity advantages with linear attention but achieves better expressiveness through the selection of proxy tokens.
- Insight: Geometrically transforming and enhancing raw data before feature learning is an under-explored, yet effective, preprocessing strategy.

## Rating

⭐⭐⭐⭐ — The idea of multi-modality guided point cloud enhancement is novel and intuitive, with an elegant and efficient design for proxy attention. Generating results that outperform the full-data baseline with only 20% of the training data is highly convincing. The main drawbacks are that evaluation is confined to a single benchmark and the code is not open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[CVPR 2026\] PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding](../../CVPR2026/3d_vision/pv-ground_text-guided_point-voxel_interaction_for_3d_visual_grounding.md)
- [\[CVPR 2026\] EG-3DVG: Expression and Geometry Aware Grounding Decoder for 3D Visual Grounding](../../CVPR2026/3d_vision/eg-3dvg_expression_and_geometry_aware_grounding_decoder_for_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
