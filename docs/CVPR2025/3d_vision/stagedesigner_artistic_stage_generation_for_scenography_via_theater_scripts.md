---
title: >-
  [Paper Note] StageDesigner: Artistic Stage Generation for Scenography via Theater Scripts
description: >-
  [CVPR 2025][3D Vision][Stage design generation] This paper proposes StageDesigner, the first AI-driven framework for artistic stage generation. It leverages LLMs to analyze scripts to extract scene and imagery descriptions, implements foreground entity layouts via a multi-level collision map, and generates background images aligned with the narrative atmosphere using a foreground projection module and a layout-controlled diffusion model.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Stage design generation"
  - "script analysis"
  - "3D scene synthesis"
  - "layout control"
  - "LLM-driven"
date: 2026-05-08
content_hash: 4368fdb1795087b7
---

# StageDesigner: Artistic Stage Generation for Scenography via Theater Scripts

**Conference**: CVPR 2025  
**arXiv**: [2503.02595](https://arxiv.org/abs/2503.02595)  
**Code**: [Project Page](https://deadsmither5.github.io/2025/01/03/StageDesigner/)  
**Area**: 3D Vision/Scene Generation  
**Keywords**: Stage design generation, script analysis, 3D scene synthesis, layout control, LLM-driven

## TL;DR

This paper proposes StageDesigner, the first AI-driven framework for artistic stage generation. It leverages LLMs to analyze scripts to extract scene and imagery descriptions, implements foreground entity layouts via a multi-level collision map, and generates background images aligned with the narrative atmosphere using a foreground projection module and a layout-controlled diffusion model.

## Background & Motivation

Artistic stage design is a complex task that translates textual narratives into immersive visual environments, which is traditionally time-consuming and highly reliant on professional expertise. Although recent 3D indoor scene synthesis and text-to-image technologies have made progress, stage generation faces unique challenges:

- **Spatial Coherence**: Requires sightline management from the audience's perspective to avoid obstructing key elements.
- **Thematic Alignment**: The generated scenes must faithfully reflect the emotional tone and symbolic meaning of the script.
- **Narrative Fidelity**: Scenic elements need to be consistent with script content and spatial relationships.
- Existing indoor scene generation methods (such as LayoutGPT) do not consider the audience's perspective and occlusion issues.

In addition, the lack of a dataset dedicated to stage generation evaluation represents a significant gap in this field.

## Method

### Overall Architecture

StageDesigner consists of three modules: the script analysis module extracts scene and imagery descriptions from theater scripts; the foreground generation module creates and places 3D objects; and the background generation module utilizes foreground projection to calculate occlusion intervals, generating a background image consistent with the narrative atmosphere.

### Key Designs

#### Key Design 1: Script Analysis Module

- **Function**: Decomposes the raw script into two key components: scene descriptions and imagery descriptions.
- **Mechanism**: Utilizes an LLM (GPT-4o) to extract entities, spatial relationships (guiding foreground generation), themes, emotional tones, and atmospheres (guiding background generation) from the script, while filtering out noisy information.
- **Design Motivation**: Raw scripts contain extensive content unrelated to visual generation (e.g., dialogue, psychological descriptions), which can introduce noise if used directly. Decomposing the script allows subsequent modules to focus on relevant information.

#### Key Design 2: Multi-level Collision Map

- **Function**: Ensures that foreground entities are reasonably placed on the stage without overlapping.
- **Mechanism**: Initializes a collision map on an $N \times N$ stage floor to mark occupied locations, and establishes separate collision maps for the front, left, right, and top of each anchor entity. Free areas are searched in the corresponding collision maps based on spatial relationship types (floor adjacent, surface attached, top-placed).
- **Design Motivation**: Direct generation of all entity coordinates by an LLM is prone to boundary violations and overlaps. By having the LLM predict coordinates for only a small number of anchor entities, and letting the collision map automatically place non-anchor entities, spatial conflicts are greatly reduced.

#### Key Design 3: Foreground Projection Module

- **Function**: Calculates the occlusion intervals of foreground entities on the background, ensuring key background elements remain visible from the audience's perspective.
- **Mechanism**: Models the audience's line of sight as parallel rays, tracks the lines of sight from the leftmost and rightmost audience positions along the edges of entities, and computes the projected bounding boxes of each entity on the background. The layout bounding boxes of background elements must avoid these occlusion intervals.
- **Design Motivation**: The core difference between stage design and general indoor scene generation lies in the need to consider a fixed audience perspective to guarantee that key background elements are not obstructed by the foreground.

### Loss & Training

Training-free process. StageDesigner is a training-free system that directly utilizes pre-trained LLMs and a layout-controlled diffusion model (ReCo).

## Key Experimental Results

### Main Results: Foreground Layout Coherence

| Method | Out-of-Bound (m³) ↓ | OIS (m³) ↓ | IWG (m³) ↑ |
|------|-------------------|-----------|-----------|
| LayoutGPT* | 6.46 | 18.2 | **14.5** |
| **StageDesigner** | **0.0468** | **0.756** | 9.03 |

### Diversity and Thematic Alignment

| Method | Class Diversity ↑ | CLIP-sim ↑ |
|------|-----------------|-----------|
| LayoutGPT* | 7.46 | 29.1 |
| **StageDesigner** | **11.7** | **30.3** |

### User Study

| Preference Dimension | StageDesigner Preference Rate |
|---------|------------------|
| Layout Coherence | **70%** |
| Overall Preference | **70%** |

### Key Findings

- Out-of-Bound volume decreased from 6.46m³ to 0.0468m³ (a 99% reduction), indicating that corner coordinate representation is significantly superior to single-point representation.
- Entity overlapping (OIS) decreased from 18.2m³ to 0.756m³ (a 96% reduction), validating the effectiveness of the multi-level collision map.
- An average of 11.7 distinct classes were generated per stage (compared to 7.46), resulting in richer scenes.
- IWG is lower because LayoutGPT generates oversized entities (which yields a large intersection with the ground truth, but simultaneously leads to severe boundary violations and overlaps).

## Highlights & Insights

1. **First AI Stage Generation Framework**: Pioneers a new research direction, extending scene generation to the domain of theatrical arts.
2. **Visual Management via Foreground Projection**: Occlusion calculation from the audience's perspective is a core innovation that distinguishes stage design from general scene generation.
3. **StagePro-V1 Dataset**: Consists of 276 real stage scenes annotated by professional designers, successfully filling a data gap in this research area.

## Limitations & Future Work

- Entity retrieval relies on a subset of Objaverse, resulting in limited availability of 3D assets.
- Background generation uses the ReCo model, which exhibits limited quality in complex scenes.
- Lighting design, a crucial component of stage design, is not yet considered.
- The framework currently only handles proscenium stages, without addressing immersive or circular stages.

## Related Work & Insights

- **LayoutGPT**: Generates CSS layout for furniture using LLMs, which is adapted as a baseline in this work.
- **Holodeck**: Generates multi-room environments with LLMs, providing the Objaverse subset.
- **ReCo**: A layout-controlled diffusion model that unifies text and position tokens.
- The visual management strategy of this framework can be extended to exhibition design, museum exhibition layout, and other related fields.

## Rating

⭐⭐⭐⭐ — First AI stage generation framework, with a unique problem definition and an ingeniously designed foreground projection module. The StagePro-V1 dataset holds long-term value. However, as a training-free system, the generation quality is bounded by the capabilities of the underlying foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing](treemeshgpt_artistic_mesh_generation_with_autoregressive_tree_sequencing.md)
- [\[CVPR 2025\] HandOS: 3D Hand Reconstruction in One Stage](handos_3d_hand_reconstruction_in_one_stage.md)
- [\[CVPR 2025\] MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds](mv-dust3r_single-stage_scene_reconstruction_from_sparse_views_in_2_seconds.md)
- [\[CVPR 2026\] MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer](../../CVPR2026/3d_vision/meshflow_efficient_artistic_mesh_generation_via_meshvae_and_flow-based_diffusion.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](../../ICCV2025/3d_vision/baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)

</div>

<!-- RELATED:END -->
