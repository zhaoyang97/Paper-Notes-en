---
title: >-
  [Paper Note] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding
description: >-
  [CVPR 2025][3D Vision][3D Visual Grounding] This paper proposes SeeGround, a training-free zero-shot 3D visual grounding framework. By representing the 3D scene as a hybrid of query-aligned rendered images and spatially-enhanced textual descriptions, it leverages 2D vision-language models to outperform previous zero-shot methods on ScanRefer by 7.7% in accuracy.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Visual Grounding"
  - "Zero-Shot"
  - "Vision-Language Models"
  - "Perspective Adaptation"
  - "Open-Vocabulary"
date: 2026-05-08
content_hash: 609025e32cdeacce
---

# SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding

**Conference**: CVPR 2025  
**arXiv**: [2412.04383](https://arxiv.org/abs/2412.04383)  
**Code**: [https://seeground.github.io](https://seeground.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D Visual Grounding, Zero-Shot, Vision-Language Models, Perspective Adaptation, Open-Vocabulary

## TL;DR

This paper proposes SeeGround, a training-free zero-shot 3D visual grounding framework. By representing the 3D scene as a hybrid of query-aligned rendered images and spatially-enhanced textual descriptions, it leverages 2D vision-language models to outperform previous zero-shot methods on ScanRefer by 7.7% in accuracy.

## Background & Motivation

**Background**: 3D visual grounding (3DVG) aims to localize target objects in 3D scenes based on textual descriptions, serving as a key technology for augmented reality and robotic perception. Supervised methods like ScanRefer and BUTD-DETR perform excellently on annotated datasets but heavily rely on extensive 3D annotations.

**Limitations of Prior Work**: (1) Supervised methods are constrained by predefined classes and annotated datasets, making them scale poorly to open-vocabulary scenarios; (2) Large-scale 3D scene labeling is extremely expensive, hindering scalability; (3) Recent zero-shot methods (e.g., LLM-Grounder and ZSVG3D) only use text to describe 3D scenes, ignoring critical visual cues like color, texture, and state—which are difficult to articulate precisely in language.

**Key Challenge**: 2D VLMs trained on large-scale 2D data possess strong open-vocabulary understanding capabilities but cannot directly process 3D data (point clouds, voxels), while purely textual descriptions of 3D scenes lose rich visual details.

**Goal**: Build a bridge that enables 2D VLMs to "see" and comprehend 3D scenes, achieving zero-shot open-vocabulary 3D visual grounding.

**Key Insight**: Convert 3D scenes into a 2D VLM-compatible hybrid representation—where query-aligned rendered images provide visual cues, and textualized 3D spatial descriptions provide precise positioning information.

**Core Idea**: Dynamically select rendering perspectives based on the query text (rather than using fixed bird's-eye views or multi-views), label objects with visual prompts on the rendered images to establish 2D-3D correspondences, and allow the VLM to complete grounding by simultaneously viewing visual details and reading spatial relationships.

## Method

### Overall Architecture

The input consists of a 3D scene point cloud and a text query. First, an open-vocabulary 3D detector is used to obtain the 3D bounding boxes and semantic labels of all objects in the scene, which are stored in an Object Lookup Table (OLT). Then, a rendering perspective is dynamically selected based on the query to generate a 2D image. Visual prompts are overlayed on the image to label the objects. Finally, the image, spatial descriptions, and query are fed into the VLM to output the target object ID, which is then used to retrieve the final 3D bounding box from the OLT.

### Key Designs

1. **Perspective Adaptation Module**:

    - **Function**: Dynamically select the most suitable rendering perspective based on the query text.
    - **Mechanism**: The VLM first parses the query $\mathsf{Q}$ to identify the anchor object $\boldsymbol{A}$ (e.g., "patterned chair") and candidate targets $\mathcal{O}^{(C)}$. A virtual camera is then initially positioned at the center of the scene, facing the anchor object, and moved backward and upward to ensure proper field-of-view coverage. The rotation and translation matrices are calculated using $\text{look\_at\_view\_transform}$, rendering the query-aligned 2D image $\mathbf{I} = \text{Render}(\mathcal{S}, \mathbf{R}_c, \mathbf{T}_c)$.
    - **Design Motivation**: Fixed viewpoints (like bird's-eye views or standard multi-views) fail to align with the spatial semantics of the query—e.g., "the window on the right" must be understood from the speaker's perspective rather than from above. Dynamic perspectives avoid information redundancy and occlusion issues.

2. **Fusion Alignment Module**:

    - **Function**: Establish precise correspondences between the objects in the 2D image and the 3D spatial descriptions.
    - **Mechanism**: The 3D boxes of objects in the OLT are projected onto the rendered image to detect and filter out occluded objects. Visual prompts (numerical tags) are overlayed on the visible objects, enabling the VLM to explicitly associate the objects seen in the image with their 3D locations in the text description. The labeled image, spatial description text, and query are then fed together into the VLM for inference.
    - **Design Motivation**: When a scene contains multiple similar objects (e.g., several chairs), the VLM struggles to match the textual "chair 3 at coordinates (2.1, 3.4, 0.5)" with the correct chair in the image. Visual prompt tags explicitly establish this mapping, acting as a crucial "bridge."

3. **Hybrid 3D Scene Representation**:

    - **Function**: Create a VLM-compatible input that contains both visual and spatial details.
    - **Mechanism**: Represent the 3D scene as $(\mathbf{I}, \mathcal{T}) = \mathbf{F}(\mathcal{S}, \mathsf{Q}, \mathcal{OLT})$. The textual component $\mathcal{T}$ contains the 3D box (center, size) and semantic label of each object, providing accurate 3D positions. The image component $\mathbf{I}$ offers visual cues such as color, texture, shape, and state. The two modalities are complementary—text compensates for the lack of precise 3D coordinates in the image, while the image fills in the visual details that are hard to describe in words.
    - **Design Motivation**: Text-only methods cannot distinguish between visually similar objects (e.g., "patterned chair" vs. "solid-color chair"), whereas image-only methods lack precise 3D localization. The hybrid representation allows the VLM to leverage the strengths of both modalities.

### Loss & Training

SeeGround is a training-free zero-shot method, requiring no loss function or training process. It uses closed-source VLMs like GPT-4V/Claude or open-source VLMs like LLaVA as the reasoning engine. Object detection only needs to be performed once per scene, and the results are cached in the OLT to be reused across all queries.

## Key Experimental Results

### Main Results

Zero-shot 3DVG performance on the ScanRefer validation set (Acc@0.25):

| Method | Supervision | Unique | Multiple | Overall |
|---|---|---|---|---|
| LLM-Grounder | Zero-Shot | 30.4 | 8.6 | 12.8 |
| ZSVG3D | Zero-Shot | 57.2 | 21.8 | 28.6 |
| **SeeGround** | **Zero-Shot** | **71.7** | **28.3** | **36.3** |
| WS-3DVG | Weakly Supervised | 53.8 | 22.5 | 28.5 |
| BUTD-DETR | Fully Supervised | 84.2 | 46.6 | 52.2 |

Nr3D Dataset:

| Method | Easy | Hard | Dep. | Indep. | Overall |
|---|---|---|---|---|---|
| ZSVG3D | 42.6 | 31.0 | 28.2 | 42.2 | 37.3 |
| **SeeGround** | **54.0** | **36.4** | **40.1** | **47.1** | **44.4** |

### Ablation Study

| Configuration | ScanRefer Acc@0.25 | Description |
|---|---|---|
| Text-Only Description (No Image) | ~28-29 | Missing visual information |
| Fixed Bird's-Eye View | ~30-32 | Occlusion and viewpoint misalignment |
| Multi-View | ~32-34 | Redundant and difficult for the VLM to consolidate |
| **Query-Aligned Perspective + Visual Prompts**| **36.3** | Full method |

### Key Findings

- SeeGround system shifts the SOTA zero-shot performance limit by 7.7% (36.3 vs 28.6) on ScanRefer and by 7.1% on Nr3D compared to the previous state-of-the-art ZSVG3D.
- The training-free framework is even able to outperform the weakly-supervised counterpart WS-3DVG (36.3 vs 28.5).
- Perspective adaptation provides the most significant performance gain for queries that contain directional descriptions (e.g., "left", "right").
- Even when textual descriptions are not completely accurate, visual cues can help robustly resolve references—demonstrating the efficacy of multimodal integration.

## Highlights & Insights

1. **"Cross-Modal Bridge" Design**: Representing 3D scenes in a hybrid image+text format elegantly enables 2D VLMs to comprehend 3D spatial relations.
2. **Query-Driven View Selection**: Instead of using fixed views, the rendering perspective is dynamically adjusted based on the query text. This "task-aware observation strategy" can be transferred to embodied AI scenarios.
3. **Correspondence via Visual Prompts**: Labeling IDs directly on the images to build 2D-3D correspondences simply and effectively resolves ambiguities in multi-object scenes.

## Limitations & Future Work

- The performance depends heavily on the quality of the pretrained 3D detector—missed detections or false positives directly impact final localization.
- The reasoning capacity of the VLMs limits the understanding of complex spatial relations (e.g., nested references).
- The quality of the rendered images is influenced by the density of the point cloud; sparse point clouds might yield blurred images.
- Future work could introduce multi-turn reasoning or self-refinement to improve the handling of complex queries.

## Related Work & Insights

- **vs. LLM-Grounder**: LLM-Grounder only uses LLMs for textual reasoning and lacks visual information; SeeGround introduces visual modalities to significantly improve localization accuracy.
- **vs. ZSVG3D**: ZSVG3D is also zero-shot but primarily relies on 3D descriptions in text format; SeeGround's hybrid representation provides richer information.
- **vs. Agent3D-Zero**: Agent3D-Zero uses multi-view VLMs for 3D question answering; SeeGround focuses on visual grounding, where its query-driven perspective is more precise.
- **Insight**: The key to "making 2D models understand 3D" lies in information representation—there is no need for 3D native inputs; appropriate 2D projections matched with spatial text descriptions are sufficient.

## Rating

- **Novelty**: 7/10 — The hybrid representation and dynamic views are intuitive and effective, though the core remains heavily dependent on VLM capabilities.
- **Experimental Thoroughness**: 8/10 — Comprehensive evaluations on two standard benchmarks (ScanRefer and Nr3D) with thorough ablation studies.
- **Writing Quality**: 8/10 — The framework diagrams are clear, and intuitive examples demonstrate the advantages of the method.
- **Value**: 8/10 — Significantly pushes the zero-shot 3DVG SOTA forward, opening up a viable path for training-free methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [\[CVPR 2025\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)
- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)

</div>

<!-- RELATED:END -->
