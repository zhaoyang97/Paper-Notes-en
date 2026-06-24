---
title: >-
  [Paper Note] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation
description: >-
  [CVPR 2025][LLM Agent][3D Scene Generation] Proposes SceneAssistant, a closed-loop agentic framework based on visual feedback. By designing a fully functioning suite of Action APIs (13 atomic operations spanning object search, deletion, 6DoF spatial operations, and camera control) for VLMs, this approach enables iterative, open-vocabulary 3D scene generation using the ReAct paradigm. It significantly outperforms Holodeck and SceneWeaver in both indoor (preference rate of 61.2…
tags:
  - "CVPR 2025"
  - "LLM Agent"
  - "3D Scene Generation"
  - "VLM Agent"
  - "Visual Feedback"
  - "Action API"
  - "Open-Vocabulary"
date: 2026-05-08
content_hash: e305fa2f90be51f0
---

# SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation

**Conference**: CVPR 2025  
**arXiv**: [2603.12238](https://arxiv.org/abs/2603.12238)  
**Code**: [https://github.com/ROUJINN/SceneAssistant](https://github.com/ROUJINN/SceneAssistant)  
**Area**: LLM Agent  
**Keywords**: 3D Scene Generation, VLM Agent, Visual Feedback, Action API, Open-Vocabulary

## TL;DR
Proposes SceneAssistant, a closed-loop agentic framework based on visual feedback. By designing a fully functioning suite of Action APIs (13 atomic operations spanning object search, deletion, 6DoF spatial operations, and camera control) for VLMs, this approach enables iterative, open-vocabulary 3D scene generation using the ReAct paradigm. It significantly outperforms Holodeck and SceneWeaver in both indoor (preference rate of 61.25%) and open-domain (preference rate of 65.00%) scenarios.

## Background & Motivation
**Background**: Text-driven 3D scene generation primarily follows two paradigms. Data-driven methods (e.g., Director3D, DreamScene) employ NeRF/3DGS representations but are limited by the diversity of available datasets like 3D-FRONT. Retrieval-based methods (e.g., Holodeck, LayoutVLM, SceneWeaver) retrieve assets from large object repositories like Objaverse, using VLMs for high-level planning and external solvers for layout optimization.

**Limitations of Prior Work**: Layout optimization in retrieval-based methods relies on **predefined spatial relationship primitives** (such as "on", "face_to", "against_wall"), which are typically designed for indoor environments. When user descriptions involve complex or unconventional configurations (e.g., "a small boat near a lighthouse"), fixed spatial vocabularies fail to capture the semantics, leading to under-constrained or erroneous layouts. Furthermore, methods like 3D-GPT and SceneCraft require VLMs to directly write Blender Python scripts, where complex syntax distracts the model from spatial reasoning.

**Key Challenge**: (1) All existing approaches are **open-loop**—rendering directly after layout generation without refining based on the rendered results; (2) forcing VLMs to handle low-level data structures (JSON/Python) imposes an excessive cognitive load, degrading reasoning quality.

**Goal**: Sub-problem 1: How can VLMs precisely control 6DoF layouts without relying on predefined spatial templates? Sub-problem 2: How can self-correction be achieved through iterative feedback?

**Key Insight**: The authors make a crucial observation: modern VLMs (pretrained on internet-scale data) already possess **implicit spatial perception and planning capabilities**. The problem is not a lack of spatial reasoning in VLMs, but rather the absence of suitable interfaces to "unleash" these capabilities. Therefore, the design philosophy is: **stimulate existing capabilities rather than training new ones**.

**Core Idea**: Use a fully-featured suite of atomic operation APIs as the "bridging interface" to allow VLMs to focus on high-level spatial reasoning, while implementing a visual feedback loop for self-correction.

## Method

### Overall Architecture
Given a user's natural language description $d$, the goal is to synthesize a 3D scene $s_d$. The framework adopts the ReAct paradigm to construct a closed-loop: at each timestep $t$, the VLM receives the **rendered image of the current scene** (with visual annotations) + **coordinate, rotation, and size data for all objects in the scene** + **historical action sequence**. After reasoning, it outputs a batch of Action API calls. It runs for at most $T_M = 20$ steps, either terminating when the agent autonomously calls Finish or when it reaches the maximum step limit. 3D assets are generated via a multi-stage pipeline utilizing Z-Image (text-to-image) + Hunyuan3D (image-to-3D mesh) and ultimately rendered in the Blender EEVEE engine.

### Key Designs

1. **Complete Action API Suite (13 atomic operations, Table 1)**:

    - **Function**: Provides three functional sets of operations—object management (Create/Duplicate/Delete), 6DoF spatial manipulation (Place/Translate/Rotate/Scale), and camera control (ViewScene/FocusOn/RotateCamera/MoveCamera), along with GenerateFloorTexture and Finish.
    - **Mechanism**: "Completeness" refers to Place (setting absolute XYZ coordinates) + Rotate (setting XYZ rotation angles) covering the full 6 degrees of freedom of objects. Any spatial configuration can theoretically be achieved through a finite sequence of these operations. RotateCamera + MoveCamera can generate arbitrary camera states. Scale controls the normalization of dimensions to real-world scale, and Translate provides incremental fine-tuning.
    - **Design Motivation**: **"Reducing operational complexity to improve reasoning quality"**. Experiments demonstrate that directly outputting JSON (the NoActionAPI baseline, which is theoretically equivalent in expressiveness) drops the Layout Correctness from 7.600 to 7.005, and the preference rate from 65% to 35.91%. This is because the "cognitive distraction" of generating the entire JSON string prevents the VLM from focusing on precise spatial adjustments. The API abstraction shields the agent from low-level complexity.

2. **Visual Feedback-Driven Closed-Loop Iteration**:

    - **Function**: In each step, the agent receives the rendered image of **only the current frame** (historical images are not accumulated to prevent information overload) + a text list of all historical actions + the current scene's object data.
    - **Mechanism**: Follows the principle of "reliable information, potentially flawed reasoning"—directly providing the current scene state (images + data) without passing prior reasoning, memory, or plans, allowing the agent to plan anew from the current state at each step. It outputs a **batch** of actions per step rather than a single one to improve efficiency.
    - **Backend Mechanism**: (a) **Automatic Ground Rectification**—objects below $z=0$ are automatically raised to the ground (though floating detection relies on the VLM itself); (b) **BVH Tree Collision Detection**—collisions are detected and notified to the agent via system messages; (c) **Create/Manipulate Separation Constraint**—object creation must be batched independently and cannot be mixed with manipulation, ensuring the agent sees the appearance of the generated asset before deciding how to place it. If constraints are violated, the entire batch of actions is rejected, and the agent is notified.
    - **Design Motivation**: Closed-loop visual feedback plays a dual role: automatically assessing the quality of 3D asset generation (deleting low-quality objects and regenerating them) and perceiving and correcting spatial misalignment. Ablations show that removing feedback results in numerous errors in object scale and orientation.

3. **Visual Prompting**:

    - **Function**: Overlays two types of information on the rendered image: (a) unique semantic name tags for each object, and (b) a coordinate axis HUD in the top-right corner showing the direction of the global reference frame.
    - **Mechanism**: Tags enable instance disambiguation and precise localization (the VLM maps names to specific objects), while the coordinate axis HUD provides a **persistent global spatial anchor** to help the VLM map 2D perspective cues into 3D execution parameters.
    - **Design Motivation**: Bridging the gap between screen-space observations and 6DoF manipulations. Ablation studies show that without visual annotations, the agent cannot precisely locate objects, leading to chaotic layouts. This serves as a **necessary bridge** for precise operations.

4. **Human-AI Collaborative Scene Editing (Section 3.3)**:

    - **Function**: Users can inject natural language editing instructions via system messages at any point during the agent's execution trajectory.
    - **Mechanism**: Leverages the strong instruction-following capabilities of the VLM to compensate for its occasionally limited fine-grained visual perception. A typical usage pattern is: after the agent executes 20 steps and the layout is basically formed, a round of human feedback can correct details or perform complex density adjustments (e.g., "add plants to every table").
    - **Design Motivation**: Pure reliance on the agent's visual perception has its limits; human feedback significantly raises the ceiling of capabilities.

### Loss & Training
**Training-Free**—The entire framework is training-free. The VLM backbone is Gemini-1.5-Flash, used directly in a zero-shot manner. The 3D asset pipeline is: Z-Image (text-to-image) $\rightarrow$ background removal $\rightarrow$ Hunyuan3D (image-to-3D mesh). All scenes are rendered in Blender EEVEE.

## Key Experimental Results

### Main Results
30 test scenes (8 indoor + 22 open-domain), with each scene evaluated independently by 10 human evaluators. Evaluation metrics: Spatial Layout Correctness (1-10 scale), Object Quality (1-10 scale), and Human Preference Rate.

| Scene Type | Method | Layout Correctness↑ | Object Quality↑ | Preference Rate↑ |
|-----------|------|:---:|:---:|:---:|
| Indoor (8 scenes) | Holodeck | 4.475 | 4.763 | 6.25% |
| Indoor (8 scenes) | SceneWeaver | 5.800 | 6.150 | 36.25% |
| Indoor (8 scenes) | **SceneAssistant** (Ours) | **6.888** | **6.950** | **61.25%** |
| Open-Domain (22 scenes) | NoActionAPI | 7.005 | 6.591 | 35.91% |
| Open-Domain (22 scenes) | NoVisFeedback | 6.255 | 5.673 | 26.82% |
| Open-Domain (22 scenes) | **SceneAssistant** (Ours) | **7.600** | **7.277** | **65.00%** |

### Ablation Study

| Configuration | Qualitative Effect | Core Reason |
|------|----------|----------|
| Full SceneAssistant | Optimal | Complete framework |
| w/o Visual Prompting | Messy layout, objects out of control | No labels $\rightarrow$ unable to precisely map object names to entities |
| w/o Collision Check | Object penetration | Visual feedback alone is insufficient to detect physical clipping |
| w/o Action API (JSON) | Layout↓, 35.91% preference rate | Generating JSON causes cognitive distraction |
| w/o Visual Feedback | Scaling/orientation errors | Unable to perceive the actual rendering results |

### Key Findings
- **API abstraction is the largest contributor**: While NoActionAPI and the full version share the same degrees of freedom and visual feedback, their preference rates differ by ~30% (35.91% vs 65.00%), demonstrating that the design of the operational interface is vital for VLM reasoning quality.
- **Collision detection feedback cannot be omitted**: The VLM cannot reliably detect object penetration from rendered images alone; explicit system messages are required.
- **Without specialized design, SceneAssistant outperforms Holodeck and SceneWeaver** which are specifically designed for indoor environments, and avoids defaulting to redundant objects (e.g., windows, cabinets).
- **The agent loses control almost entirely when visual annotations are removed**, indicating that the VLM's spatial manipulation capabilities heavily rely on explicit visual-semantic alignment.

## Highlights & Insights
- **The trade-off between "Operational Complexity" and "Reasoning Quality"** is the most central insight. Although JSON and the designed API have the same theoretical expressiveness, the API abstraction boosts VLM performance by ~30%. This suggests that when designing VLM tools, **reducing the operational complexity of the tool is more important than increasing its expressiveness**. This principle generalizes to any scenario requiring VLM manipulation (e.g., code generation, GUI actions, robotic control).
- **The design philosophy of "stimulating existing capabilities"** is highly instructive. Instead of training new capabilities in the VLM, existing implicit spatial reasoning abilities are "unleashed" through tool design. This is more lightweight and generalizable than fine-tuning.
- **The design constraint of separating Create and Manipulate processes** is extremely practical. Due to the uncertainty of 3D asset generation models (e.g., whether a generated book is horizontal or vertical), creating the asset first to observe it before manipulating is a robust general strategy to address generative uncertainty.

## Limitations & Future Work
- **Low iterative efficiency**: Each step requires Blender rendering + VLM reasoning. Running 20 steps means at least 20 renders and 20 VLM calls, which can lead to long end-to-end processing times.
- **Complete dependence on external 3D asset models**: The overall quality of the scene is capped by Hunyuan3D. If certain long-tail object generations fail, the agent may fall into a dead loop of Delete + Create.
- **Spatial reasoning limits are bound to the VLM**: The VLM's ability to estimate exact numerical distances is limited (e.g., "move the chair 0.3 meters to the left"), meaning fine-grained spatial relationships still rely on trial and error.
- **Lack of global optimization**: Incremental construction means that objects placed early constrain the layout freedom of subsequent ones, lacking a global layout optimization stage.

## Related Work & Insights
- **vs Holodeck/SceneWeaver**: They utilize predefined spatial relationships + external solvers for layout, whereas SceneAssistant replaces them with APIs + a visual feedback loop. The advantage is freedom from fixed spatial vocabularies; the disadvantage is low iterative efficiency.
- **vs 3D-GPT/SceneCraft**: They require LLMs to directly write Blender Python code, while SceneAssistant uses an API layer abstraction to avoid syntactic overhead. Although SceneCraft's code is more flexible, its VLM reasoning quality is inferior.
- **vs TreeSearchGen**: Tree search + backtracking vs linear iteration + visual feedback. SceneAssistant is simpler but lacks backtracking capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of a visual feedback loop and a complete API design is novel, though the agentic framework and the ReAct paradigm themselves are existing concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated by human assessors across 30 scenes, compared with multiple baselines, and includes ablation studies, but lacks automated metrics or larger-scale quantitative evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The motivation builds up progressively, the logic spanning method-experiments-ablations is sound, and the supplementary material provides thorough prompts.
- **Value**: ⭐⭐⭐⭐ Offers important insights into VLM tool design philosophy; the conclusion that "reducing operational complexity improves reasoning quality" can be widely cited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PerfGuard: A Performance-Aware Agent for Visual Content Generation](../../ICLR2026/llm_agent/radiometrically_consistent_gaussian_surfels_for_inverse_rendering.md)
- [\[CVPR 2025\] ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Generation](ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)
- [\[CVPR 2025\] Visual Agentic AI for Spatial Reasoning with a Dynamic API](visual_agentic_ai_for_spatial_reasoning_with_a_dynamic_api.md)
- [\[CVPR 2025\] Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback](sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)
- [\[CVPR 2026\] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](../../CVPR2026/llm_agent/realm_mllm_agent_3d_reasoning_gaussian.md)

</div>

<!-- RELATED:END -->
