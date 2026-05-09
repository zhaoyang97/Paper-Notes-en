---
title: >-
  [Paper Note] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes CLIP-GS, the first multimodal representation learning framework based on 3D Gaussian Splatting (3DGS). It serializes 3DGS into tokens via a GS Tokenizer and aligns multimodal representations using an Image Voting Loss, achieving comprehensive improvements over point-cloud-based methods on cross-modal retrieval, zero-shot, and few-shot 3D classification tasks.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - multimodal representation learning
  - CLIP
  - contrastive learning
  - zero-shot classification
date: 2026-05-08
content_hash: 3f9618827a48bb6a
---

# CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2412.19142](https://arxiv.org/abs/2412.19142)
**Code**: None
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, multimodal representation learning, CLIP, contrastive learning, zero-shot classification

## TL;DR

This paper proposes CLIP-GS, the first multimodal representation learning framework based on 3D Gaussian Splatting (3DGS). It serializes 3DGS into tokens via a GS Tokenizer and aligns multimodal representations using an Image Voting Loss, achieving comprehensive improvements over point-cloud-based methods on cross-modal retrieval, zero-shot, and few-shot 3D classification tasks.

## Background & Motivation

3D representation learning is a fundamental topic in 3D vision. Existing 3D multimodal models (e.g., ULIP, OpenShape, Uni3D) primarily operate on **point cloud** inputs. However, point clouds, as sparse spatial representations, fail to capture texture information and have limited reconstruction capacity. In contrast, 3D Gaussian Splatting (3DGS), as an emerging 3D representation technique, models objects via explicit Gaussian primitives (encoding position, rotation, scale, color, and opacity attributes), offering superior spatial precision and geometry capture capability.

The core problem is: **how to design a 3D encoder that processes 3DGS inputs and aligns it with CLIP's vision-language representation space?**

Key challenges include:
- The attribute dimensionality of 3DGS (14-dim: 3 position + 3 color + 1 opacity + 3 scale + 4 rotation) far exceeds that of point clouds (3-dim position + optional color), making direct application of existing networks ineffective.
- Rendered images from different viewpoints exhibit large feature variance, and single-image contrastive learning may lead to suboptimal optimization.
- The generation and storage cost of large-scale 3DGS data is high.

## Method

### Overall Architecture

CLIP-GS comprises three core components:
1. **Scalable triplet generation**: ~240K 3D models are curated from Objaverse; each is paired with a 3DGS representation, 36 rendered images, and 1 text description.
2. **GS feature extraction**: FPS & kNN construct Gaussian patches → GS Tokenizer produces serialized tokens → Transformer layers output 3DGS embeddings.
3. **Multimodal alignment**: The image and text encoders of EVA-CLIP are frozen; only the 3DGS encoder is trained, aligning all three modalities via contrastive losses.

### Key Designs

1. **GS Tokenizer**: Each Gaussian primitive in 3DGS is a 14-dimensional vector (at SH degree=0). FPS & kNN first organize the 3DGS into $g$ local patches, each containing $n$ neighboring Gaussian points. The tokenizer normalizes opacity and scale attributes via Sigmoid, linearizes rotation quaternions into $3\times3$ rotation matrices, and applies a **multi-order sorting strategy** (xyz-order, Hilbert curve, Z-order) to reorganize patches. The GS Refinement Block processes inputs through two branches: position and color attributes are fed into a point cloud encoder for feature extraction, while all Gaussian attributes are processed via $1\times3$ Conv + BN + ReLU; the outputs are then fused to produce GS tokens.

2. **Image Voting Loss**: This addresses the large feature variance across rendered views. For each 3DGS, $K=5$ images from different viewpoints are sampled. The pretrained EVA-CLIP computes a semantic consistency score $S_i$ between each image and its text description, which serves as the weight for that image in the contrastive loss. The score is defined as the cosine similarity between the text embedding and the image embedding. Consequently, viewpoints with higher semantic consistency with the text receive larger weights, guiding gradient optimization:
$$\mathcal{L}_{\text{img}} = -\frac{1}{2N}\sum_{i=1}^N S_i \cdot (\text{Contra}(E_i^G, E^I) + \text{Contra}(E_i^I, E^G))$$

3. **Efficient 3DGS generation**: The spherical harmonics (SH) degree is set to 0, retaining only a single RGB color per Gaussian primitive, substantially reducing storage requirements. Point clouds are used to initialize the position and color attributes of 3DGS, requiring only 5,000 optimization iterations. Most 3DGS representations contain 10K–20K Gaussian points.

### Loss & Training

The total loss consists of two terms:
- **3D-Text contrastive loss** $\mathcal{L}_{\text{text}}$: one-to-one correspondence between 3DGS and text, standard contrastive learning.
- **3D-Image voting loss** $\mathcal{L}_{\text{img}}$: one-to-many relationship, weighted by the voting mechanism.

$$\mathcal{L} = \mathcal{L}_{\text{text}} + \mathcal{L}_{\text{img}}$$

The image and text encoders of EVA-CLIP are completely frozen; only the 3DGS encoder $F^G$ is trained. The Transformer layers are initialized with pretrained weights from a point cloud model.

## Key Experimental Results

### Main Results (Multimodal Retrieval, Objaverse-GS)

| Method | 3D Rep. | Text→3D R@1 | 3D→Text R@1 | Image→3D R@1 | 3D→Image R@1 |
|---|---|---|---|---|---|
| ULIP 2 | Point Cloud | 4.5 | 5.3 | 5.6 | 25.0 |
| OpenShape-PointBERT | Point Cloud | 24.4 | 22.6 | 61.6 | 53.8 |
| Uni3D | Point Cloud | 27.8 | 23.1 | 65.1 | 49.3 |
| **CLIP-GS** | **3DGS** | **36.8** | **30.0** | **75.6** | **56.9** |

CLIP-GS achieves gains of +9.0 on Text→3D and +10.5 on Image→3D.

### Ablation Study (Design Choices, Objaverse-GS Zero-Shot)

| Configuration | 3D Rep. | Top1 | Top3 | Top5 |
|---|---|---|---|---|
| Uni3D (baseline) | P&C | 33.6 | 52.3 | 60.1 |
| + Fine-tune | P&C | 46.9 | 68.5 | 75.9 |
| + Fine-tune | 3DGS (all attr.) | 44.8 | 66.3 | 74.1 |
| + GS Tokenizer | 3DGS | 47.9 | 69.9 | 76.8 |
| + Image Voting Loss | 3DGS | **48.5** | **70.3** | **77.5** |

Directly using all 3DGS attributes degrades performance (44.8 vs. 46.9). The GS Tokenizer effectively incorporates additional attributes (+3.1 Top1), and the Image Voting Loss yields further improvement (+0.6 Top1).

### Key Findings

1. **3DGS consistently outperforms point clouds**: CLIP-GS surpasses the best point-cloud-based methods across all tasks, including retrieval, zero-shot classification, and few-shot classification.
2. **Few-shot classification**: On ModelNet-GS 10-shot/10-way, CLIP-GS achieves 95.4% (±0.2), outperforming PointRWKV's 94.8% (±2.8) with notably smaller standard deviation.
3. **Zero-shot classification**: Trained on only ~240K samples—far fewer than the million-scale datasets used by point cloud methods—CLIP-GS already demonstrates strong zero-shot capability.
4. **Effective transfer from point cloud pretraining**: Initializing the Transformer with point-cloud-pretrained weights outperforms initialization from 2D image-pretrained weights.

## Highlights & Insights

- **Correct research direction**: Extending multimodal representation learning from point clouds to 3DGS is a natural and valuable direction; the richer texture and geometry expressiveness of 3DGS demonstrably translates into improved downstream performance.
- **Elegant GS Tokenizer design**: The dual-branch processing of position/color and Gaussian-specific attributes simultaneously reuses pretrained point cloud encoder knowledge and effectively exploits the additional information in 3DGS.
- **Image Voting Loss addresses a practical challenge**: Semantic inconsistency across rendered views is an inherent difficulty in 3D-2D alignment, and the proposed weighting mechanism offers a principled solution.
- **High data efficiency**: A model trained on only ~240K samples outperforms point cloud models trained on million-scale datasets.

## Limitations & Future Work

- Validation is limited to object-level 3DGS (Objaverse, ModelNet); extension to scene-level 3DGS remains unexplored.
- SH degree is fixed at 0, discarding view-dependent color variation information.
- The training data scale of ~240K remains relatively small; performance is expected to improve as 3DGS datasets grow.
- Larger-scale 3DGS encoders beyond the ~88M Base model have not been explored.
- No demonstrations of downstream application scenarios (e.g., scene understanding for robot navigation or autonomous driving) are provided.

## Related Work & Insights

- The paper inherits Uni3D's training paradigm (freezing CLIP encoders, training the 3D encoder) while replacing point cloud inputs with 3DGS.
- The multi-order sorting strategy (Hilbert curve, Z-order) in the GS Tokenizer draws on the use of space-filling curves for serializing 3D data.
- The effective transfer of point cloud pretrained weights suggests sufficient structural commonality between 3DGS and point clouds in terms of spatial organization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First work to introduce 3DGS into multimodal alignment pretraining; the GS Tokenizer and Image Voting Loss are well-motivated designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of retrieval, zero-shot, few-shot, and ablation experiments with fair comparisons against multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed method presentation, and intuitive figures.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for multimodal learning with 3DGS; experimental results are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Online Language Splatting](online_language_splatting.md)
- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[NeurIPS 2025\] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](../../NeurIPS2025/3d_vision/va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)
- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)
- [\[ICCV 2025\] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)

</div>

<!-- RELATED:END -->
