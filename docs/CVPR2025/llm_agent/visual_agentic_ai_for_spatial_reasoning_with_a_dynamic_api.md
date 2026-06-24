---
title: >-
  [Paper Note] Visual Agentic AI for Spatial Reasoning with a Dynamic API
description: >-
  [CVPR 2025][LLM Agent][3D spatial reasoning] This paper proposes VADAR, an agentic program synthesis approach for 3D spatial reasoning. Multiple LLM agents collaborate to generate Pythonic APIs and dynamically extend new functions to solve common subproblems during the solving process, overcoming the limitations of prior methods like VisProg/ViperGPT that rely on static, human-defined APIs. At the same time, it introduces a new benchmark involving multi-step spatial localizat…
tags:
  - "CVPR 2025"
  - "LLM Agent"
  - "3D spatial reasoning"
  - "program synthesis"
  - "dynamic API"
  - "multi-LLM agent collaboration"
  - "GPT-4o"
date: 2026-05-08
content_hash: 963223b753afee98
---

# Visual Agentic AI for Spatial Reasoning with a Dynamic API

**Conference**: CVPR 2025  
**arXiv**: [2502.06787](https://arxiv.org/abs/2502.06787)  
**Code**: [https://github.com/damianomarsili/VADAR](https://github.com/damianomarsili/VADAR)  
**Area**: LLM Agent  
**Keywords**: 3D spatial reasoning, program synthesis, dynamic API, multi-LLM agent collaboration, GPT-4o

## TL;DR
This paper proposes VADAR, an agentic program synthesis approach for 3D spatial reasoning. Multiple LLM agents collaborate to generate Pythonic APIs and dynamically extend new functions to solve common subproblems during the solving process, overcoming the limitations of prior methods like VisProg/ViperGPT that rely on static, human-defined APIs. At the same time, it introduces a new benchmark involving multi-step spatial localization and reasoning, outperforming existing zero-shot methods on 3D understanding tasks.

## Background & Motivation
**Background**: Visual reasoning—the ability to interpret the visual world—is crucial for embodied agents operating in 3D scenes. VLMs (such as GPT-4V, LLaVA) can answer image-related questions, but show a significant degradation in performance on 3D spatial reasoning involving depth, distance, and spatial relations.

**Limitations of Prior Work**: Prior program synthesis methods like VisProg/ViperGPT rely on static, human-defined APIs (such as `find_object()`, `compute_distance()`), which limits the scope of queries they can handle—operations not present in the API cannot be executed.

**Key Challenge**: 3D spatial reasoning tasks require multi-step localization and reasoning (e.g., "find the table $\rightarrow$ find the lamp above the table $\rightarrow$ find the object to the left of the lamp"), and static APIs cannot cover such diverse query requirements.

**Goal**: To enable program synthesis methods to handle a broader range of 3D spatial reasoning queries through dynamic API generation.

**Key Insight**: To let three specialized LLM agents (Signature design $\rightarrow$ Implementation $\rightarrow$ Solver) collaborate to build and extend the API, mimicking the process of a programmer incrementally building a tool library.

**Core Idea**: Three-agent collaboration to dynamically generate Pythonic APIs (Signature $\rightarrow$ DFS Implementation $\rightarrow$ Program Solving) + a new Omni3D-Bench to validate multi-step 3D spatial reasoning.

## Method

### Overall Architecture
Given an image and a 3D spatial query, three GPT-4o agents collaborate: the Signature Agent proposes new API function signatures (per batch of 10 questions) $\rightarrow$ the API Agent recursively implements functions using DFS $\rightarrow$ the Program Agent calls the complete API to generate the solving program. The foundation visual modules include GroundingDINO+SAM2 (localization), UniDepth (depth estimation), and VLM VQA.

### Key Designs

1. **Three-Agent Collaborative Architecture**:

    - **Signature Agent**: Takes the current API signatures + a batch of 10 questions as input $\rightarrow$ proposes new function signatures and docstrings (e.g., `_get_material(image, bbox)` or `_is_in_front_of(image, bbox1, bbox2)`). Method names must start with an underscore. Handling 10 questions per batch avoids redundant methods.
    - **API Agent**: Recursively implements each new signature using depth-first search (DFS) on the dependency tree—if an unimplemented method is called within the implementation, it recursively implements the dependent method first. It supports up to 5 retries upon failure (feeding error messages back to the LLM) and includes infinite recursion detection.
    - **Program Agent**: Generates a Python program for each query using the complete API (pre-defined + dynamically generated), and stores the final answer in the `final_result` variable.
    - **Design Motivation**: Separation of concerns—optimizing the three stages of signature design, implementation, and solving individually to prevent cognitive overload on a single agent.

2. **Dynamic API Growth Mechanism**:

    - The pre-defined API contains only 5 fundamental modules: `loc()` (GroundingDINO+SAM2 localization), `vqa()` (VLM question answering), `depth()` (UniDepth depth), `same_object()`, and `get_2D_object_size()`.
    - Dynamically generated API functions can compose these fundamental modules; for example, `_is_in_front_of()` is implemented by comparing the `depth()` return values of two bboxes.
    - The API grows incrementally with question batches, allowing subsequent similar questions to reuse existing functions.
    - **Design Motivation**: The static APIs of VisProg/ViperGPT cannot cover diverse 3D reasoning requirements.

3. **Omni3D-Bench New Benchmark**:

    - 500 (image, question, answer) triplets from the Omni3D real-world 3D scene dataset.
    - Four categories of questions: counting, numerical other (distance/size/ratio), yes/no, and multiple-choice.
    - Non-templated queries that require multi-step spatial localization and reasoning.

### Loss & Training
Zero-shot—only in-context learning is used, with GPT-4o powering all three agents, requiring no task-specific training data.

## Key Experimental Results

### Omni3D-Bench (500 questions, real 3D scenes)

| Method | count | numeric | y/n | multi-choice | Total |
|------|:---:|:---:|:---:|:---:|:---:|
| GPT-4o (Direct) | 28.1 | 35.5 | 66.7 | 57.2 | 42.9 |
| Claude 3.5-Sonnet | 22.4 | 20.6 | 62.2 | 50.6 | 32.2 |
| ViperGPT | 20.0 | 15.4 | 56.0 | 42.4 | 33.5 |
| VisProg | 2.9 | 0.9 | 54.7 | 25.9 | 21.1 |
| **VADAR** | **21.7** | **35.5** | 56.0 | **57.6** | **40.4** |
| **VADAR + oracle** | 89.5 | - | 100.0 | 94.1 | **94.4** |

### CLEVR (1155 questions, synthetic 3D scenes)

| Method | Total |
|------|:---:|
| GPT-4o | 58.4 |
| ViperGPT | 26.2 |
| VisProg | 31.2 |
| **VADAR** | **53.6** |
| **VADAR + oracle** | **83.0** |

### Ablation Study (Oracle analysis)

| Configuration | Omni3D-Bench | CLEVR | Description |
|------|:---:|:---:|------|
| VADAR | 40.4% | 53.6% | Actual performance |
| VADAR + oracle | **94.4%** | **83.0%** | When visual modules are perfect |
| VisProg + oracle | 66.0% | 39.9% | Comparison: Static API upper bound |
| ViperGPT + oracle | 54.9% | 40.6% | Comparison: Fixed function set limitation |

### Key Findings
- **Core advantages of dynamic vs. static APIs**: In oracle mode, VADAR scores 94.4% vs. VisProg 66.0% (+28.4%), proving that dynamic API generation is logically far superior to static APIs.
- The current performance bottleneck lies mainly in the accuracy of the visual modules (the 40.4% $\rightarrow$ 94.4% gap), while the program synthesis logic itself is highly robust.
- VADAR is on par with GPT-4o on the numeric (other) metric (35.5%), but lags behind GPT-4o on non-spatial GQA (46.1% vs. 54.9%), indicating that its advantages are concentrated in spatial reasoning.
- Primary sources of failure: GroundingDINO detection misses, severe occlusions, and reasoning chains longer than 5 steps.

## Highlights & Insights
- **The paradigm shift from "using tools" to "creating tools" is highly significant**—agents are not constrained by human-predefined capability boundaries. This approach essentially equips the AI system with "self-expansion" capabilities, as the API library grows stronger with usage.
- **The generalizability of the dynamic API extension concept** can be transferred to any domain requiring LLM program synthesis—robotics control, scientific computing, data analysis, etc.
- **Multi-step 3D reasoning benchmark** fills an evaluation gap—existing benchmarks are mostly single-step and cannot measure the compositionality of agent reasoning.

## Limitations & Future Work
- The quality of dynamically generated APIs depends heavily on the code generation capability of the LLM—weaker LLMs may generate buggy or highly inefficient functions.
- Unlimited growth of the API library requires a management mechanism (deduplication, version control, quality filtering), otherwise it introduces redundancy and conflicts.
- Communication overhead and failure recovery mechanisms for multi-agent coordination require deeper design.
- Current 3D scene understanding relies on existing perception APIs (e.g., GroundedSAM), propagating perception errors downstream.
- The scale of the benchmark is relatively small; large-scale validation remains to be conducted.

## Related Work & Insights
- **vs. VisProg / ViperGPT**: These pioneering methods use static, human-defined APIs for program synthesis, whereas VADAR renders the APIs themselves dynamically generatable.
- **vs. Voyager (MineCraft)**: Voyager also utilizes LLMs to dynamically generate skill libraries in Minecraft, but is geared towards 2D games. VADAR introduces a similar philosophy to 3D spatial reasoning.
- **Profound Inspiration for the Agent Field**: The ability of "tool creation" may be one of the key milestones toward AGI—one of the core strengths of humans is creating and improving tools.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Highly innovative dynamic API generation paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmark + zero-shot comparisons + ablation analysis
- Writing Quality: ⭐⭐⭐⭐ Clear
- Value: ⭐⭐⭐⭐⭐ Highly inspiring for the LLM tool-use paradigm

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning](../../ECCV2024/llm_agent/hydra_a_hyper_agent_for_dynamic_compositional_visual_reasoning.md)
- [\[CVPR 2025\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)
- [\[CVPR 2025\] Feature4X: Bridging Any Monocular Video to 4D Agentic AI with Versatile Gaussian Feature Fields](feature4x_bridging_any_monocular_video_to_4d_agentic_ai_with_versatile_gaussian_.md)
- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](../../ACL2025/llm_agent/agentic_reasoning_tools.md)
- [\[CVPR 2025\] TANGO: Training-free Embodied AI Agents for Open-world Tasks](tango_training-free_embodied_ai_agents_for_open-world_tasks.md)

</div>

<!-- RELATED:END -->
