---
title: >-
  [Paper Note] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation
description: >-
  [CVPR 2026][LLM Agent][3D scene generation] This paper proposes SceneAssistant—a VLM agentic framework driven purely by visual feedback—that designs 14 functionally complete Action APIs enabling Gemini-3.0-Flash to itera…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "3D scene generation"
  - "open-vocabulary"
  - "VLM agent"
  - "visual feedback"
  - "ReAct"
  - "Action API"
date: 2026-05-08
content_hash: 8263ecdc2ee7d08a
---

# SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation

**Conference**: CVPR 2026
**arXiv**: [2603.12238](https://arxiv.org/abs/2603.12238)
**Code**: [github.com/ROUJINN/SceneAssistant](https://github.com/ROUJINN/SceneAssistant)
**Area**: 3D Vision / LLM Agent
**Keywords**: 3D scene generation, open-vocabulary, VLM agent, visual feedback, ReAct, Action API

## TL;DR

This paper proposes SceneAssistant—a VLM agentic framework driven purely by visual feedback—that designs 14 functionally complete Action APIs enabling Gemini-3.0-Flash to iteratively generate and refine open-vocabulary 3D scenes within a ReAct closed loop, requiring neither predefined spatial relation templates nor external layout solvers. On a human evaluation of 30 scenes, it achieves a Layout score of 7.600 (vs. SceneWeaver 5.800) and a Human Preference rate of 65%.

## Background & Motivation

**Background**: Text-to-3D scene generation methods fall into three categories: (1) data-driven methods (3D-FRONT, ATISS, etc.) constrained to specific indoor categories; (2) procedural methods (Infinigen, ProcTHOR) requiring complex scripts or templates; (3) LLM-based methods (Holodeck, SceneWeaver, LayoutVLM) that leverage LLM reasoning to generate spatial constraints and optimize layouts via solvers.

**Limitations of Prior Work**: LLM-based methods rely on predefined spatial relation primitives (e.g., "on," "face_to," "in front of"), which are domain-specific (typically for indoor scenes). When user descriptions involve complex spatial configurations beyond predefined vocabulary, the optimization process fails or yields suboptimal layouts. Most methods are open-loop—once a layout is generated, it is not corrected based on rendered results.

**Key Observation**: Modern VLMs (pretrained on internet-scale data) already possess **latent spatial perception and planning capabilities**. These capabilities can be elicited and exploited through carefully designed operation interfaces, rather than being replaced by external optimization or predefined templates.

**Key Insight**: Rather than treating 3D scene generation as a constraint satisfaction problem, this work simulates the workflow of a human 3D designer—observe → reason → act → observe → iteratively refine. A comprehensive Action API keeps the VLM operating within its "optimal reasoning range," while a visual feedback closed loop provides self-correction capability.

## Method

### Overall Architecture

The user provides a natural language scene description $d$ → the VLM agent (Gemini-3.0-Flash) iterates under the ReAct paradigm: at each step it receives the current scene rendering + object metadata + history of action sequences → reasons and selects a batch of Action APIs to execute → the Blender engine executes the actions and renders a new image → visual feedback is returned to the VLM → the loop continues until the agent calls Finish or the maximum step count $T_M = 20$ is reached. 3D assets are generated via a Z-Image (text-to-image) + Hunyuan3D (image-to-3D mesh) pipeline.

### Key Designs

1. **Functionally Complete Action API System (14 Atomic Operations)**
   - **Function**: Abstracts low-level Blender operations into semantically intuitive commands, covering the complete operation space across three categories.
   - **Object creation/deletion**: Create (description → 3D asset generation), Duplicate, Delete. After Create, the object is initially placed at the scene center; the agent observes its appearance in the next step before deciding placement. Delete supports removing unsatisfactory results → recreating with a revised text description.
   - **6-DoF manipulation**: Place (absolute XYZ positioning) + Rotate (XYZ rotation) covering full 6 degrees of freedom. Scale controls size. Translate provides incremental displacement for fine-tuning.
   - **Camera control**: ViewScene (panoramic presets), FocusOn (focus on a specific object), RotateCamera / MoveCamera (arbitrary camera states).
   - **Design Motivation**: Requiring the VLM to directly generate Blender Python code introduces syntactic overhead that distracts reasoning attention → abstracting to semantic APIs allows the VLM to focus on high-level spatial planning. Ablation validates this: replacing APIs with JSON output → Layout drops by 0.595, Preference drops by 29 pp (cognitive load dispersal effect).

2. **Pure Visual Feedback Closed Loop**
   - **Function**: Enables the VLM to make decisions based solely on rendered images, emulating how a human observes and adjusts in 3D software.
   - **Core Mechanism**: (a) Each step provides only the current rendering (without accumulating historical images to avoid overload) + action history + current object coordinate data; (b) **Visual augmentation**—object name labels and a coordinate axis HUD are overlaid on renderings, bridging the gap between 2D observation and 3D manipulation; (c) **System message mechanism**—BVH-tree collision detection automatically notifies the agent, and constraint violations (e.g., mixing Create with Manipulate) are rejected with notification.
   - **Design Motivation**: Removing visual feedback (one-shot generation) → Layout drops by 1.345, Preference drops by 38 pp → the largest single-component impact. Removing Visual Prompting → the agent cannot precisely locate objects, producing chaotic layouts. This demonstrates that closed-loop operation and visual augmentation are both indispensable.

3. **Self-Correction and Quality Control**
   - **Function**: Addresses the inherent uncertainty of 3D generative models (which may produce poor-quality or appearance-mismatched assets).
   - **Core Mechanism**: The agent observes the appearance of a newly generated object in the next step → if unsatisfied, it can Delete + modify the text description and recreate. Objects are automatically prevented from sinking below the ground (raised if $Z < 0$). Collision detection results are fed back via system messages.
   - **Design Motivation**: 3D generative models (Hunyuan3D) are stochastic → closed-loop feedback makes the system robust to generation failures without assuming single-pass success.

### Scene Editing Capability

SceneAssistant supports interactive human–agent collaboration: users may inject editing instructions (via system messages) at any node in the agent's execution trajectory, ranging from correcting layouts to adding new elements. Typically, one round of human feedback after the agent completes the initial scene is sufficient to refine details.

### Loss & Training

No training is required. The system is entirely training-free, with agent behavior driven purely by prompt engineering. The system prompt defines operational norms (e.g., +Z is up, build the scene incrementally, verify renderings each step).

## Key Experimental Results

### Main Results: Human Evaluation (10 evaluators, 1–10 scale)

| Scene Type | Method | Layout Correctness↑ | Object Quality↑ | Human Preference↑ |
|---|---|:---:|:---:|:---:|
| Indoor (8 scenes) | Holodeck | 4.475 | 4.763 | 6.25% |
| Indoor (8 scenes) | SceneWeaver | 5.800 | 6.150 | 36.25% |
| Indoor (8 scenes) | **SceneAssistant** | **6.888** | **6.950** | **61.25%** |
| Open-vocab (22 scenes) | NoActionAPI | 7.005 | 6.591 | 35.91% |
| Open-vocab (22 scenes) | NoVisFeedback | 6.255 | 5.673 | 26.82% |
| Open-vocab (22 scenes) | **SceneAssistant** | **7.600** | **7.277** | **65.00%** |

### Ablation Study

| Ablation Variant | Layout↑ | Obj Quality↑ | Pref↑ | Gap vs. Full System |
|---|:---:|:---:|:---:|---|
| **SceneAssistant (Full)** | **7.600** | **7.277** | **65.00%** | — |
| NoActionAPI (JSON output) | 7.005 | 6.591 | 35.91% | Layout −0.595, Pref −29 pp |
| NoVisFeedback (one-shot) | 6.255 | 5.673 | 26.82% | Layout −1.345, Pref −38 pp |
| NoVisualPrompt (no labels/HUD) | — | — | — | Chaotic layout, object localization failure |
| NoCollisionCheck (no collision feedback) | — | — | — | Object interpenetration cannot be self-corrected |

### Key Findings

- **Visual feedback is the most critical component**: its removal causes the largest Layout drop (−1.345); one-shot generation cannot perceive or correct spatial misalignment.
- **The cognitive load reduction effect of Action APIs is significant**: with the same visual feedback, API vs. JSON → 29 pp Preference gap; JSON forces the agent to manage low-level data structures, dispersing reasoning attention.
- Holodeck achieves only 6.25% Preference on indoor scenes → the limitations of predefined spatial relations and the Unity pipeline are evident.
- SceneAssistant performs more strongly on non-indoor scenes (Layout 7.600) → open-vocabulary capability is the core differentiating advantage.
- Collision detection feedback is critical for physical plausibility → pure visual feedback alone is insufficient to implicitly infer interpenetration issues.

## Highlights & Insights

- **The Action API abstraction level is precise**—neither too low-level (Blender code) nor too high-level (predefined spatial relations), landing exactly in the VLM's "optimal reasoning range" with semantically intuitive instructions such as "translate the sofa 0.5 meters to the right."
- **The pure visual feedback closed-loop paradigm**—requires no structured intermediate representations such as scene graphs or hypergraphs, directly leveraging the VLM's visual understanding for greater generality and simplicity.
- **Modular and extensible architecture**—adding new Action APIs (e.g., GenerateFloorTexture) requires no modification to the framework core.
- **Pragmatic human–agent collaboration design**—acknowledges the limits of VLM visual perception and allows one round of human feedback to close the remaining gap.

## Limitations & Future Work

- Evaluation relies solely on human evaluation (30 scenes × 10 evaluators), lacking reproducible automated metrics.
- Performance is bounded by the capabilities of the VLM (Gemini-3.0-Flash) and the 3D generator (Hunyuan3D) → upgrading either model would directly improve results.
- The 20-step maximum may be insufficient for complex scenes, yet increasing it accumulates errors and cost.
- No direct comparison with SceneWeaver on open-vocabulary scenes is performed (SceneWeaver does not support open-vocabulary inputs).
- Token costs of API calls are not analyzed → the economic feasibility of real-world deployment remains to be assessed.

## Related Work & Insights

- **vs. Holodeck**: Predefined spatial relations + physics solver → restricted to indoor domains, Indoor Pref only 6.25%.
- **vs. SceneWeaver**: Reflective agent but still reliant on predefined spatial primitives + hybrid tool interface, 36.25%.
- **vs. SceneCraft / 3D-GPT**: Direct Blender code generation → frequent syntax errors + dispersed reasoning attention.
- **vs. TreeSearchGen**: Global–local tree search supports backtracking but at higher complexity.
- **Insights**: The API abstraction design paradigm for VLM-as-Agent has reference value for any system requiring VLM–tool interaction (not limited to 3D generation). The observation that "VLMs already possess latent spatial capabilities; the key is how to elicit them" warrants deeper investigation.

## Rating

⭐⭐⭐⭐ (4/5)

Overall assessment: The paper presents an elegant pure visual feedback agentic framework with a well-designed Action API abstraction, and open-vocabulary capability constitutes a clear differentiating advantage. The main shortcomings are insufficient evaluation (human evaluation only, limited scene count) and the fact that the method is a combination of VLM capability and engineering design rather than an algorithmic innovation. Nevertheless, as a systems-level contribution, it makes a meaningful advancement to the 3D scene generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_mllm_agent_3d_reasoning_gaussian.md)
- [\[ICLR 2026\] PerfGuard: A Performance-Aware Agent for Visual Content Generation](../../ICLR2026/llm_agent/radiometrically_consistent_gaussian_surfels_for_inverse_rendering.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](../../ACL2026/llm_agent/diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[CVPR 2026\] Gen-n-Val: Agentic Image Data Generation and Validation](gen_n_val_agentic_image_data_generation_and_validation.md)
- [\[ICLR 2026\] PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](../../ICLR2026/llm_agent/physcensis_physics-augmented_llm_agents_for_complex_physical_scene_arrangement.md)

</div>

<!-- RELATED:END -->
