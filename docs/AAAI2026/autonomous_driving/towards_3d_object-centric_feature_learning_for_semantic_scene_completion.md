---
title: >-
  [Paper Note] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion
description: >-
  [AAAI 2026][Autonomous Driving][Semantic Scene Completion] This paper proposes Ocean, a framework that leverages instance masks extracted by MobileSAM to guide 3D object-centric feature learning. Through Semantic Group A…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Semantic Scene Completion"
  - "Object-Centric Learning"
  - "MobileSAM"
  - "Linear Attention"
  - "BEV Representation"
date: 2026-05-08
content_hash: cc63b27de436ae84
---

# Towards 3D Object-Centric Feature Learning for Semantic Scene Completion

**Conference**: AAAI 2026
**arXiv**: [2511.13031](https://arxiv.org/abs/2511.13031)
**Code**: None
**Area**: Autonomous Driving / 3D Semantic Scene Completion
**Keywords**: Semantic Scene Completion, Object-Centric Learning, MobileSAM, Linear Attention, BEV Representation

## TL;DR

This paper proposes Ocean, a framework that leverages instance masks extracted by MobileSAM to guide 3D object-centric feature learning. Through Semantic Group Attention (SGA3D) and Global Similarity-Guided Attention (GSGA), Ocean achieves instance-level feature aggregation in 3D space, and refines scene representations via an Instance-aware Local Diffusion (ILD) module, attaining state-of-the-art performance on SemanticKITTI and SSCBench-KITTI360.

## Background & Motivation

### State of the Field

Visual 3D Semantic Scene Completion (SSC) aims to partition 3D space into voxels and predict the semantic label of each voxel, producing dense 3D environment representations for downstream planning tasks in autonomous driving. Compared to LiDAR-based approaches, monocular vision solutions offer lower cost and easier deployment.

### Limitations of Prior Work

Most existing methods follow an **ego-centric paradigm**, projecting 2D features into 3D space and then aggregating and diffusing features across the entire scene. This global paradigm introduces two core problems:

**Semantic ambiguity**: Features from different objects are blended, causing semantic confusion. For instance, empty spaces between multiple roadside vehicles are incorrectly assigned features similar to those of the vehicles.

**Geometric ambiguity**: Features of empty and occupied voxels are mixed, leading to inaccurate geometry predictions, manifested as spurious "trailing" predictions behind vehicles.

### Root Cause

Existing object-centric methods (e.g., Symphonies using instance queries, GaussianFormer using Gaussian primitives) lack **explicit object-level correspondences**, which limits performance gains. Leveraging vision foundation models such as MobileSAM faces two additional challenges:

1. Mask priors are confined to the 2D image plane and do not support comprehensive 3D feature interaction.
2. Prior masks may contain errors and omissions, leading to performance degradation.

## Method

### Overall Architecture

The Ocean pipeline proceeds as follows:
1. An image encoder extracts multi-scale visual features.
2. The LSS method projects 2D features into a 3D voxel space.
3. 3D voxel query proposals are selected based on depth predictions.
4. **SGDA blocks**: MobileSAM masks guide object-level feature aggregation.
5. **ILD module**: Instance features refine the scene BEV representation.
6. A 3D prediction head outputs semantic occupancy predictions.

### Key Designs

#### 1. SemGroup Dual Attention (SGDA) Block

SGDA comprises two complementary modules:

**3D Semantic Group Attention (SGA3D)**:

**Function**: Leverages instance masks from MobileSAM to perform grouped feature interaction between voxel queries and image pixels belonging to the same instance in 3D space.

**Mechanism**:
- 3D query proposals are projected onto the image plane, and instance IDs are assigned via nearest-neighbor sampling.
- Queries and pixels sharing the same instance ID are grouped into the same cluster.
- **Scattered linear attention** is applied within each cluster for efficient feature aggregation.

$$\tilde{Q} = \text{Concat}\left[\frac{\varphi(Q^j) A^j \sum_{i=1}^{m^j} \varphi(K_i^j)^T V_i^j}{\varphi(Q^j) \sum_{i=1}^{m^j} \varphi(K_i^j)^T}\right]_{j=1}^{M}$$

where $A^j$ is the depth similarity matrix and $M$ is the number of instances.

**3D Depth Extension**: A depth similarity matrix $A^j$ is constructed using the similarity between the predicted depth probability distribution and the projected depth of each query, extending 2D object-centric learning into 3D space. Specifically, depth bins serve as a bridge: pixel depth is represented as a softmax probability distribution while query depth corresponds to a specific bin, enabling computation of an $m \times n$ depth similarity matrix.

**Design Motivation**: Linear attention reduces computational complexity from $O(n^2)$ to $O(n)$, making it suitable for handling large numbers of instances and pixels. Multi-scale feature aggregation combines high-level semantic information with low-level textural details.

**Global Similarity-Guided Attention (GSGA)**:

**Function**: Employs deformable attention guided by MobileSAM intermediate features to supplement global feature aggregation, compensating for SGA3D's focus on local instance interactions.

**Mechanism**: For each query proposal, the similarity between the query and MobileSAM intermediate features at deformable offset positions is computed as instance-aware weights to filter and emphasize features from the same object:

$$\hat{Q} = \sum_k (G_k W \mathcal{F}(p_q + \Delta p_k)) \odot A_k$$

**Design Motivation**: Addresses errors and omissions in MobileSAM masks. Deformable attention allows the model to flexibly attend to regions beyond rigid segmentation boundaries, reducing the impact of imperfect masks.

#### 2. Instance-aware Local Diffusion (ILD) Module

**Function**: Enhances the BEV representation using instance-level features to compensate for information loss caused by projection limitations.

**Dynamic Instance Decoder (DID)**:
- Groups multi-scale features by instance mask and sums them to obtain instance-level aggregated representations.
- Uses a lightweight deconvolution block to generate $x \times y \times (C_2+1)$-dimensional BEV features from instance features.
- Applies Gumbel-Softmax to dynamically select which instances participate in the final BEV reconstruction.

$$\hat{w}_l = \frac{\mathcal{Z}_l w_l}{\sum_{l=1}^L \mathcal{Z}_l w_l + \epsilon}, \quad \hat{\mathcal{P}} = \sum_{l=1}^L \mathcal{P}_l \odot \hat{w}_l$$

**Local Attention Refinement**: Window attention (Swin Transformer style) is applied, using aggregated features as Query and instance-aware features as Key/Value to refine BEV features within a local scope.

### Loss & Training

The total loss function is:

$$\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_r \mathcal{L}_{recon} + \mathcal{L}_{ce} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem}$$

- $\mathcal{L}_{ce}$: Cross-entropy loss supervising voxel semantic predictions.
- $\mathcal{L}_{scal}^{sem}$, $\mathcal{L}_{scal}^{geo}$: Scale-aware semantic and geometric losses.
- $\mathcal{L}_d$: Depth distribution supervision ($\lambda_d=0.001$).
- $\mathcal{L}_{recon}$: BEV reconstruction loss constraining the quality of generated BEV features ($\lambda_r=0.1$).

Training setup: 4× GeForce RTX 3090, AdamW optimizer, initial learning rate 3e-4, weight decay 0.01, EfficientNetB7 as the 2D backbone.

## Key Experimental Results

### Main Results

**SemanticKITTI Validation Set**:

| Method | IoU↑ | mIoU↑ | Note |
|--------|------|-------|------|
| MonoScene | 34.16 | 11.08 | First monocular SSC |
| VoxFormer | 42.95 | 12.20 | Depth-guided sparse-to-dense |
| OccFormer | 34.53 | 12.32 | Local-global decomposition |
| Symphonies | 42.19 | 15.04 | Instance-query object-centric |
| LOMA | 43.01 | 15.10 | Vision-language fusion |
| CGFormer | 44.41 | 16.63 | Context-aware geometry |
| HTCL | 44.23 | 17.09 | Temporal information |
| **Ocean (Ours)** | **45.62** | **17.40** | **Object-centric + MobileSAM** |

**SSCBench-KITTI360 Test Set**:

| Method | IoU↑ | mIoU↑ |
|--------|------|-------|
| CGFormer | 48.07 | 20.05 |
| SGFormer | 46.35 | 18.30 |
| **Ocean (Ours)** | **48.19** | **20.28** |

### Ablation Study

**Contribution of Each Module** (SemanticKITTI Validation Set):

| Config | SGA3D | GSGA | LA | DID | IoU↑ | mIoU↑ |
|--------|-------|------|----|-----|------|-------|
| M0 (Baseline) | | | | | 44.62 | 15.80 |
| M1 | ✓ | | | | 45.77 | 16.49 |
| M2 | ✓ | ✓ | | | 46.01 | 16.80 |
| M3 | ✓ | ✓ | ✓ | | 45.39 | 17.10 |
| M4 (Full) | ✓ | ✓ | ✓ | ✓ | 46.40 | 17.39 |

**SGA3D Module Ablation**:

| Method | IoU↑ | mIoU↑ | Params | FLOPs |
|--------|------|-------|--------|-------|
| **SGA3D (Ours)** | **46.40** | **17.39** | **0.280M** | **1.333G** |
| SGA3D w/o multi-scale | 46.44 | 16.64 | 0.280M | 1.298G |
| SGA3D w/o 3D extension | 45.84 | 17.30 | 0.280M | 1.333G |
| DFA2D | 46.14 | 16.39 | 0.202M | 2.231G |
| DFA3D | 46.15 | 16.97 | 0.330M | 2.807G |

**Dynamic Instance Decoder Fusion Strategies**:

| Strategy | IoU↑ | mIoU↑ |
|----------|------|-------|
| Direct sum | 45.79 | 16.45 |
| Softmax weighting | 45.89 | 16.88 |
| Sigmoid weighting | 46.30 | 16.92 |
| **Dynamic selection (Ours)** | **46.40** | **17.39** |

### Key Findings

1. Introducing instance priors via SGA3D improves IoU by 1.15 and mIoU by 0.69, validating the effectiveness of object-centric feature learning.
2. GSGA complements the local limitations of SGA3D, yielding a further 0.31 gain in mIoU.
3. SGA3D achieves 0.42 higher mIoU than DFA3D with fewer parameters and approximately half the FLOPs.
4. Dynamic instance selection outperforms simple summation by 0.94 in mIoU.
5. Ocean performs particularly well on categories with consistent surface information, such as roads, sidewalks, and traffic signs.

## Highlights & Insights

1. **Depth bins as an elegant bridge**: Associating pixel depth probability distributions with query projected depths via depth bins elegantly extends 2D mask priors into 3D space.
2. **Complementary dual-attention design**: SGA3D handles fine-grained intra-instance interactions locally, while GSGA compensates for mask errors and omissions — the two modules are mutually complementary.
3. **Gumbel-Softmax dynamic selection**: Instance importance is learned dynamically, avoiding manually designed weights or fixed fusion strategies.
4. **Scattered linear attention**: Reduces computational complexity to linear scale, making large-scale instance processing feasible.
5. **Thorough experimental design**: Detailed ablation studies are provided for each module, with fair comparisons against alternative designs such as DFA3D.

## Limitations & Future Work

1. **Difficulty with distant objects and heavy occlusion**: When visual information is insufficient, the model struggles to extract discriminative features (failure cases are shown in Figure 7 of the paper).
2. **MobileSAM inference overhead**: Although lightweight, the additional segmentation model still increases inference time.
3. **Monocular input only**: Multi-view or temporal information is not exploited (HTCL achieves 17.09 mIoU using temporal cues).
4. **Limited improvement on background categories**: Categories such as terrain show less gain compared to foreground objects.
5. **Replacing MobileSAM with a lighter segmentation head** is a promising direction to reduce inference cost.

## Related Work & Insights

- **Comparison with Symphonies**: Both are object-centric methods, but Ocean employs explicit MobileSAM priors rather than implicit instance queries, achieving a 2.36 improvement in mIoU.
- **Relationship with GaussianFormer**: Both aim to model objects with sparse representations, but Ocean places greater emphasis on explicit instance-level interaction.
- **Insight**: Priors from vision foundation models (SAM series) can effectively guide 3D perception, but dedicated designs are required to handle their imperfections.

## Rating

- Novelty: ⭐⭐⭐⭐ — Combining the object-centric paradigm with MobileSAM priors represents a meaningful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Module ablations are highly detailed with fair comparisons against alternatives.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with rich illustrations.
- Value: ⭐⭐⭐⭐ — Introduces a new paradigm for 3D SSC with strong implications for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Unleashing Semantic and Geometric Priors for 3D Scene Completion](unleashing_semantic_and_geometric_priors_for_3d_scene_completion.md)
- [\[AAAI 2026\] HD2-SSC: High-Dimension High-Density Semantic Scene Completion for Autonomous Driving](hd2-ssc_high-dimension_high-density_semantic_scene_completion_for_autonomous_dri.md)
- [\[CVPR 2026\] OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective](../../CVPR2026/autonomous_driving/occufly_a_3d_vision_benchmark_for_semantic_scene_completion_from_the_aerial_pers.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](../../CVPR2026/autonomous_driving/sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)
- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)

</div>

<!-- RELATED:END -->
