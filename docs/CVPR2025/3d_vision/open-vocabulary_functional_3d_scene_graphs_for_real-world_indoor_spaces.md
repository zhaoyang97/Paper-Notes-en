---
title: >-
  [Paper Note] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces
description: >-
  [CVPR 2025][3D Vision][3D Scene Graph] Proposes a novel task of functional 3D scene graphs, utilizing VLMs and LLMs to construct 3D scene graphs featuring objects, interactive elements, and their functional relationships from RGB-D images through a progressive detection-description-reasoning pipeline, and establishes the FunGraph3D real-world dataset.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Scene Graph"
  - "Functional Relationship"
  - "Interactive Element"
  - "Foundation Model"
  - "Open-Vocabulary"
date: 2026-05-08
content_hash: 14ef530444bf9d75
---

# Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces

**Conference**: CVPR 2025  
**arXiv**: [2503.19199](https://arxiv.org/abs/2503.19199)  
**Code**: [https://openfungraph.github.io](https://openfungraph.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D Scene Graph, Functional Relationship, Interactive Element, Foundation Model, Open-Vocabulary

## TL;DR

Proposes a novel task of functional 3D scene graphs, utilizing VLMs and LLMs to construct 3D scene graphs featuring objects, interactive elements, and their functional relationships from RGB-D images through a progressive detection-description-reasoning pipeline, and establishes the FunGraph3D real-world dataset.

## Background & Motivation

**Background**: 3D scene graphs organize indoor entities into graph structures. Existing methods such as Open3DSG and ConceptGraph can reconstruct scene graphs, but nodes are restricted solely to objects, and edges only represent spatial relationships (e.g., "TV on the wall").

**Limitations of Prior Work**: Existing works lack small interactive element nodes (such as switches, handles, buttons) and functional relationship edges (e.g., "switch controls light"). Spatial relationships are already implicitly encoded by object positions, offering limited value, whereas functional relationships are crucial for robotic manipulation.

**Key Challenge**: Constructing functional scene graphs requires understanding part-level interactive elements and causal relationships. However, training data is severely scarce, and many functional relationships cannot be inferred purely from static vision.

**Goal**: (1) Formalize the functional 3D scene graph; (2) Enable zero-shot construction utilizing foundation models without training data; (3) Establish a labeled evaluation dataset.

**Key Insight**: Foundation models (VLMs and LLMs) encode rich functional knowledge; VLMs can identify objects and interactive elements, while LLMs possess common-sense functional relationships. This functional knowledge can be extracted in a zero-shot manner through carefully crafted prompting and progressive reasoning.

**Core Idea**: The construction of functional scene graphs is decomposed into three stages: progressive detection (objects $\rightarrow$ interactive elements), multi-view VLM+LLM description, and sequential functional relationship reasoning (local $\rightarrow$ remote).

## Method

### Overall Architecture

OpenFunGraph defines $\mathcal{G} = (\mathcal{O}, \mathcal{I}, \mathcal{R})$, where $\mathcal{O}$ denotes objects, $\mathcal{I}$ represents interactive elements, and $\mathcal{R}$ indicates functional relationship edges. The pipeline consists of: node detection $\rightarrow$ node description $\rightarrow$ functional relationship reasoning.

### Key Designs

1. **Progressive Node Detection**:

    - Function: Detect objects and interactive elements, and fuse them into 3D.
    - Mechanism: Objects are identified using RAM++ to obtain query tags, which are then detected via GroundingDINO. For interactive elements, the system queries GPT-4 for candidate interactive elements associated with each identified object, concatenates the object tag with the element tag (e.g., "door. handle") as the GroundingDINO prompt to improve small-target detection, and then back-projects them via depth maps for multi-view fusion into 3D.
    - Design Motivation: A progressive "object-to-part" strategy with object contextual prompts significantly improves the detection rate of small interactive elements.

2. **Multi-View Collaboration Description**:

    - Function: Generate natural language descriptions for each node.
    - Mechanism: For objects, the top-$N_v$ representative views are selected and described using LLaVA, then summarized by GPT-4. For interactive elements, multi-scale cropping and red-contour highlighting are employed to guide the VLM's attention, followed by multi-view summarized descriptions.
    - Design Motivation: Single views are highly susceptible to occlusion, and small interactive elements require multi-scale zooming and visual highlighting for accurate VLM comprehension.

3. **Sequential Functional Relationship Reasoning**:

    - Function: Infer functional relationships between objects and interactive elements.
    - Mechanism: Local relationships (e.g., cabinet door - handle) are processed first through 3D spatial overlap filtering combined with LLM common-sense judgment. Remote relationships (e.g., light - switch) are processed next, where candidate pairs are generated by LLMs, visually verified by VLMs, and assigned confidence scores by LLMs. This is implemented via step-by-step Chain-of-Thought reasoning.
    - Design Motivation: The reasoning logics for local and remote relationships are fundamentally different; hence, sequential step-by-step reasoning is significantly more reliable than a single-pass inference.

### Loss & Training

The approach is purely based on zero-shot inference using foundation models without training. It utilizes RAM++/GroundingDINO for detection, LLaVA v1.6 for visual understanding, and GPT-4 for common-sense reasoning.

## Key Experimental Results

### Main Results

Node detection Recall@10 (SceneFun3D / FunGraph3D):

| Method | Objects | Interactive Elements | Overall |
|------|------|---------|------|
| Open3DSG* | 70.7/58.1 | 61.8/33.9 | 64.7/43.6 |
| ConceptGraph*+IED | 77.1/66.3 | 59.5/33.4 | 66.0/45.0 |
| **OpenFunGraph** | **87.8/79.1** | **79.5/57.6** | **82.8/65.8** |

### Ablation Study

| Configuration | Impact |
|------|------|
| Without auxiliary object tag prompts | Interactive element detection significantly drops |
| Without multi-scale description | Poor quality of descriptions |
| One-step reasoning (without local/remote progressive steps) | Decreased accuracy in relationship reasoning |

### Key Findings

- Interactive element detection remains the primary bottleneck (79.5% vs. 87.8% for objects), emphasizing that part-level detection is still highly challenging.
- ConceptGraph is virtually incapable of detecting interactive elements (only 8.6% originally), as existing scene graph methods do not consider part-level understanding.
- FunGraph3D is significantly more challenging than SceneFun3D.

## Highlights & Insights

- The problem formulation of **functional 3D scene graphs** is a key contribution, representing a qualitative leap from spatial relations to functional relations.
- The **auxiliary tag prompting strategy for interactive elements** is simple yet effective, and can be transferred to any part detection task.
- The **local-to-remote sequential reasoning** avoids the reasoning confusion of LLMs when confronted with a large number of candidates.

## Limitations & Future Work

- Static observations cannot definitively determine certain remote relationships.
- Relying on a pipeline of multiple large models results in high inference costs.
- Currently limited to indoor environments.
- Evaluation metrics based on embedding cosine similarity may contain biases.

## Related Work & Insights

- **vs. Open3DSG**: Infers spatial relations based on CLIP+GNN, but fails to handle part-level entities and functional relationships.
- **vs. ConceptGraph**: Focuses solely on objects and spatial relations, with very poor capabilities in interactive element detection.
- **vs. SceneFun3D**: Provides annotations for interactive elements but lacks object and relationship annotations.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Functional 3D scene graph is a brand-new task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient evaluation across two datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation and intuitive pipeline illustration.
- Value: ⭐⭐⭐⭐⭐ Pioneering work with high potential impact on robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)

</div>

<!-- RELATED:END -->
