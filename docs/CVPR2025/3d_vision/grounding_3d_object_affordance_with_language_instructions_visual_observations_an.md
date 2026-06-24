---
title: >-
  [Paper Note] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions
description: >-
  [CVPR 2025][3D Vision][3D affordance grounding] This work proposes the first multimodal multi-view 3D affordance grounding task and the AGPIL dataset (containing 30,972 point cloud-image-language triplets). It designs LMAffordance3D, a VLM-based framework that fuses 2D/3D spatial features with linguistic semantics to generalize from full-view to partial/rotation-view scenarios.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D affordance grounding"
  - "multi-modal fusion"
  - "point cloud"
  - "VLM"
  - "embodied intelligence"
date: 2026-05-08
content_hash: f5e24383903afa4f
---

# Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions

**Conference**: CVPR 2025  
**arXiv**: [2504.04744](https://arxiv.org/abs/2504.04744)  
**Code**: [Project Page](https://sites.google.com/view/lmaffordance3d)  
**Area**: 3D Vision  
**Keywords**: 3D affordance grounding, multi-modal fusion, point cloud, VLM, embodied intelligence

## TL;DR

This work proposes the first multimodal multi-view 3D affordance grounding task and the AGPIL dataset (containing 30,972 point cloud-image-language triplets). It designs LMAffordance3D, a VLM-based framework that fuses 2D/3D spatial features with linguistic semantics to generalize from full-view to partial/rotation-view scenarios.

## Background & Motivation

**Background**: Affordance grounding aims to identify manipulable regions of objects, serving as a key link between perception and action in embodied intelligence. Prior research has primarily focused on 2D images or single-modality 3D point clouds.

**Limitations of Prior Work**:
- 2D affordance methods are difficult to directly map to 3D space for robotic manipulation tasks.
- 3D methods (such as 3D AffordanceNet) rely solely on geometric information, leading to limited generalization capability and confusion when handling similar objects.
- Existing datasets either utilize only single-modality input, lack guidance from language instructions, or overlook the issue of incomplete point clouds caused by occlusion and rotation in real-world scenarios.

**Key Challenge**: In the real 3D world, object observations are often limited to partial point clouds due to viewpoints, occlusions, and rotations; however, humans learn new object affordances via linguistic instructions, visual demonstrations, and interactions—which existing methods fail to utilize simultaneously.

**Key Insight**: Inspired by cognitive science, this work models 3D affordance grounding as a multimodal task (incorporating language, images, and point clouds) and constructs a comprehensive benchmark covering three viewpoints (full, partial, and rotation) and two settings (seen and unseen).

## Method

### Overall Architecture

LMAffordance3D is an end-to-end single-stage framework consisting of four core components:
1. **Vision Encoder**: Processes images (ResNet18 $\to$ 2D features) and point clouds (PointNet++ $\to$ 3D features), fusing them via an MLP and Self-Attention to obtain multimodal spatial features $F_S$.
2. **VLM Core**: Employs LLaVA-7B as the backbone. A tokenizer encodes language instructions into $F_T$, and an Adapter (a two-layer MLP with an activation layer) maps spatial features $F_S$ into the semantic space $F_{SP}$. These are then concatenated and fed into the VLM.
3. **Decoder**: Based on cross-attention, it uses spatial features as Query, instruction features as Key, and semantic features as Value to decode the affordance features $F_A$.
4. **Segmentation Head**: Employs upsampling, two linear layers, Batch Normalization (BN), and Sigmoid to output point-wise affordance probabilities of shape $(B, 2048, 1)$.

### Key Designs

**1. Multimodal Vision Encoder Design**
- **Function**: Extracts 2D and 3D features using ResNet18 and PointNet++ respectively, then fuses them via an MLP and Self-Attention.
- **Mechanism**: RGB images contain color, scene, and interaction information, while point clouds capture shape, scale, and geometric characteristics. The two modalities are complementary and aligned through a shared semantic space.
- **Design Motivation**: Instead of directly utilizing CLIP (which has large parameters and is difficult to deploy), a lightweight design is adopted to suit robotic deployment scenarios.

**2. Cross-Attention-Based Decoder**
- **Function**: Splits the VLM output into instruction features and semantic features, fusing spatial and semantic information via cross-attention.
- **Mechanism**: Spatial features (Query) query the semantic features (Value), with the attention distribution guided by the instruction features (Key).
- **Design Motivation**: To ensure that different language instructions can guide the model to focus on different affordance regions of the same object.

**3. AGPIL Dataset Construction**
- **Function**: Builds the first multimodal, multi-view 3D affordance dataset, containing 30,972 images, 41,628 point clouds, and 30,972 language instructions.
- **Mechanism**: Point clouds are acquired from 3D AffordanceNet (covering full, partial, and rotation views), images are sourced from AGD20K and PIAD, and language instructions are generated by GPT-4o combined with images and then manually filtered.
- **Design Motivation**: Covers 23 object categories and 17 affordance classes. Each annotation is represented as a $(2048, 17)$ probability matrix, and seen/unseen splits are defined to thoroughly evaluate generalization capabilities.

### Loss & Training

$$Loss = \omega_f L_{focal} + \omega_d L_{dice}$$

- Focal Loss: Handles the imbalance between positive and negative samples.
- Dice Loss: Optimizes the overlapping segmentation regions.

## Key Experimental Results

### Main Results

| Method | Full-view AUC↑ | Full-view SIM↑ | Partial AUC↑ | Rotation AUC↑ |
|---|---|---|---|---|
| 3D AffordanceNet | 0.807 | 0.483 | 0.761 | 0.595 |
| IAG | 0.849 | 0.545 | 0.809 | 0.679 |
| OpenAD | 0.858 | 0.587 | 0.815 | 0.733 |
| PointRefer | 0.877 | 0.595 | 0.821 | 0.756 |
| **Ours** | **0.890** | **0.610** | **0.848** | **0.782** |

The advantages are more prominent under the unseen setting: Full-view AUC achieves 0.774 vs. PointRefer's 0.755, and MAE reaches 0.095 vs. 0.118.

### Ablation Study

- Among the 17 affordance categories, "stab" achieves the highest AUC of 0.997, while "wrapping" is the most challenging with an AUC of only 0.689.
- Performance progressively decreases from Full $\to$ Partial $\to$ Rotation views, highlighting the challenges posed by incomplete point clouds.
- Performance drops by approximately 10-15% from Seen $\to$ Unseen configurations, yet Ours exhibits a larger performance advantage on Unseen data compared to Seen data.

### Key Findings

1. Multimodal fusion (images + point clouds + language) significantly outperforms single-modality methods, improving the AUC by 3-8%.
2. Linguistic instruction guidance enables the model to distinguish between different functional regions of the same object.
3. The largest improvements are observed in rotation-view scenarios (AUC +2.6%), where the semantic understanding of the VLM compensates for geometric uncertainty.
4. Generalization capability on unseen objects is significantly enhanced, demonstrating that multimodal fusion improves knowledge transferability.

## Highlights & Insights

- Introduces the first 3D affordance grounding task formulation that simultaneously leverages language instructions, visual observations, and interactions.
- The AGPIL dataset fills the gap in multimodal, multi-view, and probabilistic annotation resources.
- The paradigm of incorporating VLMs into 3D affordance tasks is noteworthy, as prior knowledge from the VLM substantially boosts generalization on unseen categories.
- Features an end-to-end, single-stage design (without requiring 2D detection bounding boxes), offering superior scalability.

## Limitations & Future Work

- Images and point clouds originate from different scenes (matched via category-level associations), leading to visual-geometric inconsistency.
- LLaVA-7B introduces substantial inference overhead, which is unfavorable for real-world robot deployment.
- The rotation-view setting remains a performance bottleneck; rotation-equivariant networks could be considered for enhancement.
- Only static object affordance is supported, without considering dynamic scenes.
- The language instructions are phrase-level in granularity; more complex instruction understanding remains unexplored.

## Related Work & Insights

- 3D AffordanceNet pioneered 3D affordance datasets, and this work extends it towards multimodality and multi-view configurations.
- AffordanceLLM validated the effectiveness of VLMs in 2D affordance, and this work extends it to 3D for the first time.
- Insight: The visual-semantic alignment capability of VLMs can serve as a cross-modal bridge, inspiring future exploration of more advanced 3D understanding tasks.

## Rating

⭐⭐⭐⭐ — The task definition is novel and the dataset construction is solid. The technical approach is reasonable, though the innovation is moderate (mostly modular combination). The completeness of the multi-view and unseen settings is highly commendable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[CVPR 2025\] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [\[CVPR 2025\] Guiding Human-Object Interactions with Rich Geometry and Relations](guiding_human-object_interactions_with_rich_geometry_and_relations.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)

</div>

<!-- RELATED:END -->
