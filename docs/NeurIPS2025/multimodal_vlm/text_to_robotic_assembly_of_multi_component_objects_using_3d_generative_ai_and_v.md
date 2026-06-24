---
title: >-
  [Paper Note] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM reasoning] This paper proposes an end-to-end pipeline that converts natural language input into 3D mesh models via 3D generative AI, then leverages zero-shot multimodal reasoning of VLMs to automatically decompose the mesh into multi-component 3D models (structural components + panel components), which are subsequently assembled into physical objects by a robotic arm. The system also supports interactive user feedback through dialogue to adj…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "VLM reasoning"
  - "robotic assembly"
  - "3D generative AI"
  - "multi-component objects"
  - "human-robot collaboration"
date: 2026-05-08
content_hash: 82018ca7800f5968
---

# Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.02162](https://arxiv.org/abs/2511.02162)  
**Code**: Unavailable (depends on Autodesk internal platform)  
**Area**: Multimodal VLM
**Keywords**: VLM reasoning, robotic assembly, 3D generative AI, multi-component objects, human-robot collaboration

## TL;DR

This paper proposes an end-to-end pipeline that converts natural language input into 3D mesh models via 3D generative AI, then leverages zero-shot multimodal reasoning of VLMs to automatically decompose the mesh into multi-component 3D models (structural components + panel components), which are subsequently assembled into physical objects by a robotic arm. The system also supports interactive user feedback through dialogue to adjust component assignments.

## Background & Motivation

3D generative AI (e.g., DreamFusion, Get3D, Latte3D) can generate diverse 3D geometries from text, but converting them into physical objects presents two major challenges:

**Lack of component-level representation**: Existing 3D generative models produce monolithic meshes without the component-level decomposition required for robotic assembly—i.e., they do not identify which parts form the structural skeleton and which are functional surfaces (e.g., the seat of a chair, the shade of a lamp).

**Function-aware component assignment**: Decomposing a mesh into predefined components requires simultaneous reasoning over geometry and object function. For example, a stool requires horizontal panels placed on the seat surface to form a flat surface, while a lamp requires panels on the shade frame to diffuse light. Such reasoning is too complex for traditional rule-based methods.

**Diversity of user preferences**: Panel assignments may have multiple valid configurations depending on user preferences, necessitating support for human-in-the-loop interactive adjustment.

Limitations of prior work:
- 3D printing pipelines (Sketch2Prototype, Style2Fab) support only single-material additive manufacturing
- Part-aware generative models (PartGen, StructureNet) focus on geometric reconstruction rather than assembly
- 3D segmentation methods do not account for robot reachability and functional roles

## Method

### Overall Architecture

The system consists of five stages: (1) text input → (2) 3D mesh generation (Autodesk Project Bernini) → (3) mesh voxelization into structural components → (4) VLM reasoning for panel component assignment → (5) assembly by UR20 robotic arm.

### Key Designs

1. **Mesh Discretization**: Two types of assembly components are defined—structural components (load-bearing skeleton, cuboid) and panel components (planar pieces attached to the structure). The AI-generated mesh is voxelized according to structural component dimensions to form the primary load-bearing frame. The key innovation lies in determining panel placement based on object function and geometry.

2. **VLM Function-Aware Part Selection**: Google Gemini 2.5 Pro is prompted with three inputs—object description (to understand function), axonometric view (to understand geometry), and component type (to understand panel function)—and performs zero-shot reasoning to identify parts requiring panels. For example, for "chair" the output is "Parts = seat, backrest". System prompts constrain the VLM to select the minimal set of parts that satisfy functional requirements.

3. **VLM Geometry-Aware Label Mapping**: The first VLM step yields abstract part names; the second step localizes them to specific mesh faces in the 3D model. The approach merges coplanar faces, assigns unique integer labels to each face (excluding inward-facing vertical faces and downward-facing horizontal faces that are unreachable by the robotic arm), and invokes the VLM again on a labeled axonometric rendering to map part names to specific label numbers.

4. **Human-in-the-Loop Conversational Feedback**: Users can modify the VLM's initial assignment via natural language (e.g., "I only want panels on the seat, not the backrest"). The VLM receives user feedback along with the labeled mesh image and outputs an updated label set. No task-specific training is required; the system relies entirely on the VLM's zero-shot reasoning capability.

5. **Robotic Assembly Execution**: The multi-component assembly is exported as a coordinate list $C = \{(x_i, y_i, z_i, r_{x_i}, r_{y_i}, r_{z_i})\}$ and a component type list $T = \{t_0, t_1\}$. Components are sorted bottom-to-top while maintaining connectivity, and a UR20 robotic arm equipped with a Robotiq gripper executes the pick-and-place sequence.

### Loss & Training

- **No training required**: The system relies entirely on zero-shot reasoning by the VLM (Gemini 2.5 Pro).
- System prompts are carefully designed to constrain output format and minimize selection.
- Unreachable faces (inward-facing/downward-facing) are excluded in preprocessing to prevent the VLM from generating infeasible plans.

## Key Experimental Results

### Main Results (User Preference Evaluation)

| Method | Chair (%) | Table (%) | Lamp (%) | Shelf (%) | Bin (%) | **Average (%)** |
|--------|-----------|-----------|----------|-----------|---------|-----------------|
| **VLM (Ours)** | 96.9 | 100.0 | 81.3 | 100.0 | 75.0 | **90.6** |
| Rule-based | 18.8 | 100.0 | 34.4 | 100.0 | 43.8 | 59.4 |
| Random | 0.0 | 0.0 | 0.0 | 6.3 | 6.3 | 2.5 |

32 participants × 5 objects × 3 methods = 480 judgments. Multiple selections were allowed.

### Statistical Significance Tests (McNemar Paired Test)

| Comparison | $\chi^2$ | p-value | Conclusion |
|------------|---------|---------|------------|
| VLM vs. Rule-based | 38.11 | <0.001 | VLM significantly outperforms rule-based |
| VLM vs. Random | 137.11 | <0.001 | VLM significantly outperforms random |
| Rule-based vs. Random | 88.17 | <0.001 | Rule-based outperforms random |

All comparisons remain significant after Bonferroni correction ($p^* < 0.017$).

### Key Findings

- The rule-based method (placing panels on all upward-facing surfaces) matches VLM performance on tables and shelves (both 100%), but falls substantially behind on geometrically complex objects (chair: 18.8%, lamp: 34.4%).
- User feedback reveals multiple valid configurations (e.g., panels only on the seat, panels only on the lamp shade), confirming the necessity of human-in-the-loop interaction.
- The robotic arm placed no panels on unreachable positions in any assembly, confirming that VLM outputs comply with manufacturing constraints.
- No task-specific training is required; the system relies entirely on the VLM's pretrained multimodal knowledge.

## Highlights & Insights

1. **End-to-end text-to-physical-object pipeline**: The first work to chain 3D generative AI, VLM reasoning, and robotic assembly into a complete pipeline.
2. **Zero-shot VLM reasoning for manufacturing decisions**: Demonstrates the potential of VLM geometric and functional reasoning in industrial settings without fine-tuning.
3. **Progressive reasoning decomposition**: The complex component assignment task is decomposed into two sequential VLM steps (first select part names, then map to label numbers), reducing the difficulty of single-pass inference.
4. **Preprocessing to exclude unreachable faces**: Domain knowledge is cleverly exploited to constrain the VLM's output space.

## Limitations & Future Work

- Only two predefined component types are supported (structural + panel); extensions to hinges, handles, and varied materials are not addressed.
- Evaluation is limited to five common object categories with simple prompts; complex or unconventional objects (e.g., sculptures, musical instruments) are not tested.
- The pipeline depends on Autodesk's closed-source platform (Project Bernini, Fusion 360), limiting reproducibility.
- VLM label mapping may fail on highly symmetric or densely labeled meshes.
- Assembly sequencing relies solely on bottom-to-top ordering without considering more complex assembly constraints.

## Related Work & Insights

- CLIPort, SayCan, and related works demonstrate VLM language grounding for robotic manipulation; this paper extends the paradigm to assembly geometry generation.
- Part-aware models such as PartGen and StructureNet focus on geometric modeling, whereas this paper focuses on functional reasoning.
- Insight: The world knowledge encoded in VLMs is sufficient for zero-shot manufacturing decision reasoning; the key lies in appropriate task decomposition and output constraint design.

## Rating

- Novelty: ⭐⭐⭐⭐ The end-to-end pipeline is novel; first use of VLMs for function-aware assembly component assignment.
- Experimental Thoroughness: ⭐⭐⭐ The user study is of moderate scale but limited in object variety; failure case analysis for VLM reasoning is absent.
- Writing Quality: ⭐⭐⭐⭐ The pipeline description is clear and prompt designs are transparent.
- Value: ⭐⭐⭐ Proof-of-concept stage; constrained by closed-source tools and limited component types.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)

</div>

<!-- RELATED:END -->
