---
title: >-
  [Paper Note] Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding
description: >-
  [CVPR 2025][3D Vision][Open-vocabulary 3D segmentation] This paper proposes MPEC (Masked Point-Entity Contrastive learning), which trains a 3D encoder using a two-level contrastive loss framework: cross-view point-to-entity contrastive learning and entity-to-language alignment. This approach achieves open-vocabulary semantic understanding while preserving entity-level geometric-spatial information, obtaining a state-of-the-art f-mIoU of 66.0% on ScanNet and demonstrating stro…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Open-vocabulary 3D segmentation"
  - "Contrastive learning"
  - "Point cloud"
  - "Entity-aware"
  - "Scene understanding"
date: 2026-05-08
content_hash: e3a7b624f66176fa
---

# Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding

**Conference**: CVPR 2025  
**arXiv**: [2504.19500](https://arxiv.org/abs/2504.19500)  
**Code**: [https://mpec-3d.github.io](https://mpec-3d.github.io)  
**Area**: 3D Vision  
**Keywords**: Open-vocabulary 3D segmentation, Contrastive learning, Point cloud, Entity-aware, Scene understanding

## TL;DR

This paper proposes MPEC (Masked Point-Entity Contrastive learning), which trains a 3D encoder using a two-level contrastive loss framework: cross-view point-to-entity contrastive learning and entity-to-language alignment. This approach achieves open-vocabulary semantic understanding while preserving entity-level geometric-spatial information, obtaining a state-of-the-art f-mIoU of 66.0% on ScanNet and demonstrating strong generalization capabilities on downstream tasks across 8 datasets.

## Background & Motivation

Open-vocabulary 3D scene understanding is crucial for embodied AI, where agents must understand and interact with objects described by arbitrary text. Existing methods face several key challenges:

- **2D-3D Information Gap**: Existing methods distill or fuse semantic features from 2D VLMs into 3D representations, but 2D features lack global spatial relationships and multi-view consistency.
- **The Conflict Between Semantics and Geometry**: Purely semantic alignment (e.g., OpenScene using 2D-VL feature distillation) lacks geometric information in 3D representations, making it difficult to distinguish objects with similar colors but different semantics.
- **Lack of Entity Concept**: The contrastive units in existing 3D self-supervised methods (PointContrast, CSC, GroupContrast) are neighborhoods, regions, or semantic prototypes, which lack explicit object concepts and are difficult to align with language.
- **Scarcity of Large-Scale 3D-Language Data**: There is insufficient paired 3D-text data to directly train 3D-language alignment.

Core Motivation: Utilize 3D entity masks to establish 3D representations that both preserve spatial-geometric information and align with language—enhancing instance differentiation via entity-level cross-view contrastive learning, and achieving open-vocabulary understanding via entity-to-language contrastive learning.

## Method

### Overall Architecture

MPEC consists of two stages of contrastive learning: (1) Point-to-Entity alignment, which performs cross-view contrastive learning based on entity masks on two augmented views of the same scene, pulling together point features of the same entity across different views and pushing apart different entities; (2) Entity-to-Language alignment, which projects the fused point features into the CLIP text embedding space via a VL-Adapter to achieve open-vocabulary understanding. The inputs are a 3D point cloud, entity mask proposals, and text descriptions.

### Key Designs

1. **Cross-View Point-to-Entity Contrastive Learning**:
    - Function: Encode entity-level 3D spatial information into point features, enhancing consistency for points within the same entity and differentiability between different entities.
    - Mechanism: Generate two augmented views $\mathcal{P}_u, \mathcal{P}_v$ from the input point cloud, randomly mask non-overlapping grid regions, and replace the colors of masked points with a learnable token. These are then fed into a shared 3D UNet to extract point-wise features $\mathbf{F}_u, \mathbf{F}_v$. Using entity masks, the average cosine similarity $\mathbf{s}_{i,e_k}^{u \to v}$ from each point to each entity in the other view is computed. For points belonging to an entity, an InfoNCE loss is applied to drive matching with the corresponding entity, and background points are similarly matched with background points. The final bidirectional symmetric contrastive loss is $\mathcal{L}_{p2e}$.
    - Design Motivation: Cross-view contrastive learning ensures that features learn view-invariant entity representations (unlike self-supervised point-level contrast, using entities as units aligns better with the granularity of linguistic descriptions); random masking increases reconstruction difficulty to prevent shortcut learning.

2. **Entity-to-Language Contrastive Alignment**:
    - Function: Map the learned 3D point features into the CLIP text embedding space to achieve open-vocabulary zero-shot segmentation.
    - Mechanism: Fuse the point features from both views by averaging them into $\mathbf{F}_{\mathcal{P}}$, which is then projected to the CLIP dimension $\mathbf{F}_{VL}$ through a two-layer MLP (VL-Adapter). A frozen CLIP text encoder extracts $\mathbf{F}_T$ from text descriptions (including captions and referential texts). For the text-to-entity direction, a cross-entropy loss aligns each text with its corresponding entity. For the entity-to-text direction, a binary cross-entropy loss is used (since one entity may match multiple descriptions). The final bidirectional loss is $\mathcal{L}_{e2l} = \alpha \cdot \ell^{t \to e} + \beta \cdot \ell^{e \to t}$.
    - Design Motivation: Directly applying a symmetric cross-entropy loss is unsuitable due to many-to-many relationships; the BCE loss allows an entity to correctly match multiple text descriptions. The lightweight two-layer MLP design preserves the geometric information learned by the 3D encoder.

3. **Training Data and Entity Mask Sources**:
    - Function: Provide the entity masks and text descriptions required for the training pipeline.
    - Mechanism: Use off-the-shelf 3D instance segmentation models to generate entity mask proposals. Leverage SceneVerse's data pipeline to generate descriptive/referring text for entities based on scene graphs and 2D foundation models. Training data is compiled from ScanNet + 3RScan + HM3D + MultiScan (aggregated by SceneVerse).
    - Design Motivation: Leverage existing 3D segmentation models and VL data generation pipelines directly to avoid expensive manual annotation.

### Loss & Training

- Total loss: $\mathcal{L}_{overall} = \mathcal{L}_{p2e} + \mathcal{L}_{e2l}$
- $\mathcal{L}_{p2e}$: Symmetric InfoNCE loss based on a temperature parameter $\tau$.
- $\mathcal{L}_{e2l}$: Text-to-entity cross-entropy + entity-to-text binary cross-entropy, with weight parameters $\alpha, \beta$ to balance their scales.

## Key Experimental Results

### Main Results (ScanNet Open-Vocabulary 3D Semantic Segmentation)

| Method | Network | f-mIoU↑ | f-mAcc↑ |
|------|------|---------|---------|
| OpenScene-3D | SPUNet32 | 57.8 | 70.3 |
| RegionPLC | SPUNet32 | 59.6 | 77.5 |
| OV3D | SPUNet16 | 64.0 | 76.3 |
| **MPEC** | SPUNet16 | **64.6** | **79.5** |
| **MPEC** | SPUNet32 | **66.0** | **81.3** |

### Zero-Shot Transfer

| Method | SceneVerse-val f-mIoU | Matterport3D f-mIoU | Matterport3D f-mAcc |
|------|----------------------|--------------------|--------------------|
| OpenScene | 41.3 | 49.7* | 64.0* |
| RegionPLC | 39.1 | 28.9 | 43.8 |
| **MPEC** | **45.0** | **47.7** | **69.8** |

*OpenScene was trained in-domain on Matterport3D

### Fine-Grained / Long-Tail Scenarios (ScanNet200)

| Method | f-mIoU | f-mAcc |
|------|--------|--------|
| RegionPLC | 9.1 | 17.3 |
| OV3D | 8.7 | - |
| **MPEC** | **10.8** | **27.4** |

### Data-Efficient Experiments (ScanNet Data Efficient, 1% Scenes)

| Method | mIoU |
|------|------|
| GroupContrast | 30.7 |
| **MPEC** | **40.8** |

### Key Findings

- MPEC significantly outperforms the previous SOTA (OV3D's 64.0%/76.3%) with 66.0% f-mIoU and 81.3% f-mAcc on ScanNet.
- Zero-shot transfer to unseen Matterport3D: Even without in-domain training, MPEC's f-mAcc (69.8%) exceeds that of in-domain trained OpenScene (64.0%) and OV3D (65.7%).
- Under the ScanNet200 long-tail scenarios, the f-mAcc is improved by about 10% (vs. RegionPLC), indicating that entity-level contrastive learning effectively enhances fine-grained differentiation.
- In the 1% data-efficient scenario, the performance jumps from 30.7% to 40.8% mIoU, proving that the learned representations are highly generalizable.
- As a drop-in replacement for the 3D encoder of PQ3D, it yields performance gains across multiple high-level tasks, including visual grounding, question answering, and captioning.

## Highlights & Insights

- **Elegant Design of Entity-Level Contrast**: It scales point cloud self-supervised contrastive learning from "point/neighborhood/prototype" granularity to "entity" granularity. This aligns with the natural granularity of language descriptions and serves as a key bridge for subsequent alignment.
- **Synergy of Cross-View Masking and Contrastive Learning**: Masking increases difficulty to force the encoder to learn more robust representations, while entity masks provide semantic supervision—these two masking mechanisms are combined ingeniously.
- **The Critical Role of 3D Information**: Purely 2D distillation solutions like OpenScene fail in scenarios with visual ambiguity (same color, different objects). MPEC's entity contrast naturally encodes spatial locations, resolving this issue.
- **Potential as a General 3D Encoder**: Consistently brings improvements across 7 types of tasks on 8 datasets, showing potential to become a fundamental backbone for 3D-LLMs.

## Limitations & Future Work

- Entity masks rely on existing 3D instance segmentation models (e.g., Mask3D); the quality of these masks directly affects the contrastive learning performance.
- The CLIP text encoder is frozen and struggles with long or complex spatial descriptions, which limits fine-grained visual grounding capabilities.
- Training data is mainly from indoor scenes (ScanNet series); generalization to outdoor or large-scale scenes remains to be verified.
- Text descriptions are automatically generated by foundation models, which may introduce noise.
- Only the SPUNet backbone has been validated; adapting to Transformer architectures has not yet been explored.

## Related Work & Insights

- **Intrinsic Difference from OpenScene**: OpenScene distills 2D VLM features into 3D space, which lacks 3D geometric information. In contrast, MPEC directly learns geometry-preserving entity representations in the 3D space before aligning them with language.
- **Difference from GroupContrast**: GroupContrast uses semantic prototypes for contrastive learning without explicit object concepts, making it unsuitable for language alignment. MPEC's entity-level contrast naturally bridges 3D and language.
- **Insight**: The key to 3D scene understanding is not just semantic alignment but also preserving spatial-geometric information. Entity-level contrastive learning provides an elegant solution.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of entity-level cross-view contrast and language alignment shows significant innovation, though individual components have prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 8 datasets, zero-shot, fine-tuning, data-efficient, and high-level reasoning scenarios, with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivation and clear pipeline illustration, though the overall paper text is quite dense.
- Value: ⭐⭐⭐⭐⭐ Provides a new pre-training paradigm for 3D scene understanding, with great potential as a general 3D encoder.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[CVPR 2025\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)
- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [\[CVPR 2025\] GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)

</div>

<!-- RELATED:END -->
