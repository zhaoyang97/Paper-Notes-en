---
title: >-
  [Paper Note] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation
description: >-
  [CVPR 2025][3D Vision][Point Cloud Instance Segmentation] Relation3D enhances the modeling of internal relations within scene features and relations between queries in Transformer-based 3D instance segmentation via three components: Adaptive Superpoint Aggregation Module (ASAM), Contrastive Learning-guided Superpoint Refinement (CLSR), and Relation-aware Self-Attention (RSA), achieving SOTA results on ScanNetV2/ScanNet++/ScanNet200/S3DIS.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Cloud Instance Segmentation"
  - "Relation Modeling"
  - "Contrastive Learning"
  - "Adaptive Superpoint Aggregation"
  - "Transformer"
date: 2026-05-08
content_hash: 4f92016604f43079
---

# Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2506.17891](https://arxiv.org/abs/2506.17891)  
**Code**: [GitHub](https://github.com/)  
**Area**: 3D Vision / Point Cloud Instance Segmentation  
**Keywords**: Point Cloud Instance Segmentation, Relation Modeling, Contrastive Learning, Adaptive Superpoint Aggregation, Transformer

## TL;DR

Relation3D enhances the modeling of internal relations within scene features and relations between queries in Transformer-based 3D instance segmentation via three components: Adaptive Superpoint Aggregation Module (ASAM), Contrastive Learning-guided Superpoint Refinement (CLSR), and Relation-aware Self-Attention (RSA), achieving SOTA results on ScanNetV2/ScanNet++/ScanNet200/S3DIS.

## Background & Motivation

**Background**: 3D point cloud instance segmentation aims to predict the binary foreground mask and semantic label for each object instance in a scene. Current mainstream methods are based on the Transformer encoder-decoder framework, utilizing instance queries to interact with scene features via mask attention to generate instance masks. Representative works include SPFormer, Mask3D, QueryFormer, and Maft.

**Limitations of Prior Work**: Existing Transformer-based methods primarily model the "external relations" between scene features and query features via mask attention, while neglecting two types of "internal relations": (1) relations among scene features (superpoints)—insufficient feature consistency of superpoints within the same instance and inadequate discrimination between different instances; (2) relations among query features—traditional self-attention only implicitly computes similarity, lacking explicit modeling of spatial and geometric relations.

**Key Challenge**: Superpoint features are aggregated through pooling, but point features within the same superpoint vary significantly (variance of 1.86). Pooling introduces inappropriate features and blurs discriminative ones. Meanwhile, position embeddings are often inaccurate (the learnable position encoding of SPFormer lacks concrete spatial meaning, and the position encodings of Mask3D/Maft deviate from actual mask positions), leading to insufficient spatial relation modeling in self-attention.

**Goal**: (1) How to effectively model the relations among scene features? (2) How to better model the relations among queries?

**Key Insight**: Approached from the perspective of feature relation modeling—for scene features, replacing pooling with adaptive weight aggregation and guiding the update direction using contrastive learning; for query features, embedding explicit spatial and geometric relations as biases into self-attention.

**Core Idea**: Enhancing the relation modeling capability of the Transformer decoder at both scene feature and query feature levels through three complementary relation modeling modules (ASAM + CLSR + RSA).

## Method

### Overall Architecture

The input point cloud (including coordinates, colors, normals) goes through a Sparse UNet to extract point-level features $F \in \mathbb{R}^{N \times C}$, which are then aggregated into superpoint-level features $F_{\text{super}} \in \mathbb{R}^{M \times C}$ via ASAM. $K$ instance queries $Q \in \mathbb{R}^{K \times C}$ are initialized and fed into the Transformer decoder for iterative updating. The decoder includes RSA (enhancing relations among queries), mask attention (external relations between queries and the scene), and CLSR executed every $r$ layers (guiding superpoint feature updates via contrastive learning).

### Key Designs

1. **Adaptive Superpoint Aggregation Module (ASAM)**:

    - **Function**: Adaptively aggregates point-level features into superpoint-level features, highlighting discriminative points and suppressing inappropriate features.
    - **Mechanism**: Max-pooling and mean-pooling are performed on point-level features $F$ respectively to obtain $F_{\max}$ and $F_{\text{mean}}$. The differences between these pooled features and the original point features are calculated, and two independent MLPs predict the weight for each point as $\mathcal{W}_{\max} = \text{MLP}_1(F_{\max} - F)$. After softmax normalization within each superpoint, weighted aggregation is performed. Finally, the results of the two paths are concatenated and down-projected via an MLP. The entire process can be parallelized using point-wise MLPs and torch-scatter.
    - **Design Motivation**: Direct pooling introduces noise when point features within a superpoint vary significantly. Adaptively allocating weights based on the differences from pooling statistics allows meaningful and discriminative point features to obtain higher weights.

2. **Contrastive Learning-guided Superpoint Refinement (CLSR)**:

    - **Function**: Utilizes query features to reverse-update superpoint features in the decoder, and constrains the update direction via contrastive learning.
    - **Mechanism**: Employs a dual-path structure where superpoint features serve as $\mathcal{Q}$, and query features serve as $\mathcal{K}$ and $\mathcal{V}$ to perform cross-attention (opposite to the conventional setup). A superpoint relation matrix $R_{\text{super}}^{\text{GT}}$ is constructed based on instance annotations. The cosine similarity matrix $\mathcal{S}$ of normalized superpoint features is calculated, and constrained using a BCE loss: $L_{\text{cont}} = \text{BCE}(\frac{\mathcal{S}+1}{2}, R_{\text{super}}^{\text{GT}})$. Refinement is performed every $r=3$ layers to control computational overhead.
    - **Design Motivation**: Mask attention only models the unidirectional relation from query to scene. The dual-path design allows bidirectional information flow to accelerate convergence. The contrastive loss explicitly guides superpoint features of the same instance to be closer and those of different instances to be farther apart.

3. **Relation-aware Self-Attention (RSA)**:

    - **Function**: Incorporates explicit spatial and geometric relations among queries into self-attention.
    - **Mechanism**: First, the 3D bounding box (center $x,y,z$ and scale $l,w,h$) corresponding to each query's mask is calculated. Then, the relative spatial relations (coordinate differences / log of scales) and relative geometric relations (log of scale ratios) between each pair of queries are calculated to obtain a 6D relation encoding $\mathfrak{T} \in \mathbb{R}^{K \times K \times 6}$. After dimension expansion via sin-cos position encoding, a linear transformation yields $R_q \in \mathbb{R}^{K \times K \times \mathcal{H}}$, which is added to the attention score as a bias: $\text{RSA}(Q) = \text{Softmax}(\frac{\mathcal{QK}^T}{\sqrt{C}} + R_q)\mathcal{V}$.
    - **Design Motivation**: Traditional position embeddings do not match the actual mask positions, leading to inaccurate spatial relation modeling. Directly calculating explicit relations using the bounding box corresponding to the mask and embedding them into the attention weights effectively combines implicit relation modeling with explicit spatial and geometric relations.

### Loss & Training

The total loss is $L_{all} = \lambda_1 L_{ce} + \lambda_2 L_{bce} + \lambda_3 L_{dice} + \lambda_4 L_{center} + \lambda_5 L_{score} + \lambda_6 L_{cont}$, where the first five terms are inherited from Maft, and the contrastive loss $L_{cont}$ is newly added (weight $\lambda_6=1$). The contrastive loss is computed after ASAM and after each CLSR step. Powered by a single RTX 4090 card, training runs for 512 epochs using the AdamW optimizer with a maximum learning rate of 0.0002. The voxel size is 0.02m, and $K=400$ (500 for ScanNet++/ScanNet200).

## Key Experimental Results

### Main Results

**ScanNetV2 Val / Test Set**:

| Method | val mAP | val AP50 | val AP25 | test mAP | test AP50 | test AP25 |
|------|---------|---------|---------|----------|----------|----------|
| Maft | 58.4 | 75.9 | 84.5 | 57.8 | 77.4 | - |
| SPFormer | 56.3 | 73.9 | 82.9 | 54.9 | 77.0 | 85.1 |
| **Relation3D** | **62.5** | **80.2** | **87.0** | **62.2** | **81.6** | **90.1** |

**ScanNet++ Val / Test Set**: mAP 23.1 -> **28.2** (+5.1), Test set 20.9 -> **24.2** (+3.3)

### Ablation Study

Stage-by-stage effect of contrastive loss $L_{cont}$ (lower is better):

| Stage | Maft baseline | After ASAM | 2nd CLSR | 3rd CLSR |
|------|:---:|:---:|:---:|:---:|
| $L_{cont}$ | 1.057 | 0.7255 | 0.5841 | **0.5739** |

### Key Findings

- ASAM enhances the discriminativeness of superpoint features compared to standard pooling, and CLSR further reduces contrastive loss stage by stage.
- RSA makes relationship modeling in self-attention more effective after incorporating explicit spatial and geometric relations.
- Compared to the baseline Maft, mAP increases by +4.1, AP50 by +4.3, and AP25 by +2.5 on ScanNetV2 val.
- T-SNE visualization clearly demonstrates the clustering of same-instance features and the separation of different-instance features.

## Highlights & Insights

- Precisely targets the limitation of insufficient modeling of two types of internal relations in Transformer-based 3D instance segmentation.
- Contrastive learning acts on superpoint features rather than directly on queries, guiding the enhancement of scene representation quality.
- Inspired by 2D object detection (Relation-DETR), RSA is the first to introduce relation priors into 3D instance segmentation.
- All improvements do not increase inference computational overhead (CLSR overhead during training is controllable, and RSA overhead is minimal).

## Limitations & Future Work

- The method relies on a fixed superpoint pre-segmentation, and the quality of superpoints directly impacts subsequent performance.
- Contrastive learning requires ground-truth instance annotations to construct the relation matrix, making it inapplicable to unsupervised settings.
- The improvement is larger on ScanNet200 with a high number of classes, suggesting that relation modeling is more beneficial in complex scenes.
- The applicability to outdoor point clouds (e.g., autonomous driving) has not been discussed.

## Related Work & Insights

- The relation prior bias strategy of **Relation-DETR** is successfully transferred to 3D scenes.
- **Contrastive learning** shows promising results in the 3D domain as an auxiliary loss to constrain the feature space structure.
- The dual-path (query↔superpoint) concept can be extended to other Transformer tasks that require bidirectional information flow.

## Rating

- **Novelty**: 7/10 — Some individual modules show novelty, but the overall architecture is a combination and transfer of existing techniques.
- **Experimental Thoroughness**: 9/10 — Four datasets, detailed ablation studies, and rich visualizations.
- **Writing Quality**: 8/10 — Well-defined problems and convincing motivational derivations.
- **Value**: 7/10 — Solid experimental results, but the generalizability of the method remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sketchy Bounding-Box Supervision for 3D Instance Segmentation](sketchy_bounding-box_supervision_for_3d_instance_segmentation.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[CVPR 2025\] Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)
- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](../../ICCV2025/3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)

</div>

<!-- RELATED:END -->
