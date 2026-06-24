---
title: >-
  [Paper Note] TANGO: Training-free Embodied AI Agents for Open-world Tasks
description: >-
  [CVPR 2025][LLM Agent][Embodied AI] This paper proposes TANGO, which orchestrates two minimal navigation foundation primitives (PointGoal Navigation + Memory-based Exploration) through the program generation capability of LLMs. Without any task-specific training and using only few-shot examples, TANGO achieves state-of-the-art (SOTA) results across three distinct embodied AI tasks: Open-Set ObjectGoal Navigation, Multi-Modal Lifelong Navigation, and Open Embodied QA…
tags:
  - "CVPR 2025"
  - "LLM Agent"
  - "Embodied AI"
  - "Navigation"
  - "LLM Program Synthesis"
  - "Zero-shot"
  - "Open-world"
date: 2026-05-08
content_hash: 2fc105c7f166ff0f
---

# TANGO: Training-free Embodied AI Agents for Open-world Tasks

**Conference**: CVPR 2025  
**arXiv**: [2412.10402](https://arxiv.org/abs/2412.10402)  
**Code**: To be confirmed  
**Area**: LLM Agent  
**Keywords**: Embodied AI, Navigation, LLM Program Synthesis, Zero-shot, Open-world

## TL;DR
This paper proposes TANGO, which orchestrates two minimal navigation foundation primitives (PointGoal Navigation + Memory-based Exploration) through the program generation capability of LLMs. Without any task-specific training and using only few-shot examples, TANGO achieves state-of-the-art (SOTA) results across three distinct embodied AI tasks: Open-Set ObjectGoal Navigation, Multi-Modal Lifelong Navigation, and Open Embodied QA, demonstrating the generalizability of the "minimal primitive set + LLM composition" paradigm.

## Background & Motivation
**Background**: LLMs have demonstrated the ability to compose multiple modules for complex tasks in visual reasoning (e.g., VisProg, ViperGPT). However, extending this program composition capability to embodied agents—navigating and interacting within 3D environments—remains a major challenge.

**Limitations of Prior Work**: Existing embodied AI methods are designed and trained for task-specific scenarios—ObjectNav models can only perform ObjectNav, and EQA models can only perform EQA. Encountering a new task type requires re-collecting data, designing rewards, and training policies from scratch.

**Key Challenge**: A unified framework is needed to handle diverse open-world embodied tasks without additional training, yet the diversity of embodied tasks seemingly necessitates task-specific training.

**Goal**: To leverage the program composition capability of LLMs to automatically assemble simple navigation primitives into programs that solve complex embodied tasks.

**Key Insight**: Extending the program composition capability of LLMs from 2D images to 3D embodied environments, teaching the LLM how to orchestrate navigation primitives using only few-shot examples.

**Core Idea**: Combining PointGoal Navigation and memory-based exploration as foundational primitives, which the LLM orchestrates via in-context learning. This is analogous to composing complex navigation paths using basic "move forward" and "turn" commands.

### Loss & Training
The training-free framework does not involve any task-level training. The PointGoal navigation model is pre-trained (via RL in the Habitat environment) but is not fine-tuned for any downstream tasks. The LLM learns to compose the primitives solely through few-shot prompting.

## Method

### Overall Architecture
Given an embodied task instruction, the LLM receives the task description and available navigation/perception primitives to generate a solution program through few-shot prompting. During runtime, it invokes PointGoal Navigation and the memory-based exploration policy.

### Key Designs

1. **Minimal Primitive Design (Two Only)**:

    - **PointGoal Navigation**: Navigates the agent autonomously to a target location given its coordinates. It uses a pre-trained PointGoal navigation model that only accepts $(x, y, z)$ coordinates as input.
    - **Memory-based Exploration**: Systematically explores unknown environments by maintaining a memory map of explored areas, prioritizing unexplored regions to maximize coverage.
    - **Design Motivation**: "If you can navigate to any location (PointGoal) and explore environments systematically (Exploration), most navigation-related tasks can be decomposed into a combination of these two operations"—representing an elegant, minimal design.

2. **LLM Program Composition**:

    - **Function**: The LLM receives task descriptions and API specifications for the available primitives, generating executable Python programs that combine these primitives through few-shot prompting.
    - **Mechanism**: Different tasks require different primitive combinations—ObjectGoal Nav requires "explore $\rightarrow$ detect target $\rightarrow$ navigate to target", Lifelong Nav requires "navigate to multiple locations sequentially", and Embodied QA requires "explore $\rightarrow$ gather information $\rightarrow$ reason and answer".
    - **Design Motivation**: Leveraging the code generation and reasoning capabilities of LLMs to eliminate individual training for each task. A small number of in-context examples is sufficient for the LLM to understand how to use the primitives.

### Loss & Training
Training-free—no task-specific training is required. The framework solely utilizes a pre-trained PointGoal navigation model and a general-purpose LLM.

## Key Experimental Results

### Open-Set ObjectGoal Navigation (HM3D-OVON val-unseen)

| Method | SR (%) | SPL (%) | Training Requirements |
|------|:---:|:---:|------|
| Baseline RL | 18.6 | - | Reinforcement Learning |
| VLFM | 35.2 | - | Task-specific Training |
| DAgRL+OD | 37.1 | 19.9 | Task-specific Training |
| **Ours** | **35.5** | **19.5** | **Zero-shot** |

### Multi-Modal Lifelong Navigation (GOAT-Bench val-unseen)

| Method | SR (%) | SPL (%) |
|------|:---:|:---:|
| Modular GOAT | 24.9 | - |
| SenseAct-NN Skill Chain | 29.5 | - |
| **Ours** | **32.1** | **16.5** |

TANGO outperforms the Prev. SOTA by +2.6% on GOAT-Bench.

### Open Embodied QA (OpenEQA, LLM-match Score 1-5)

| Method | Score |
|------|:---:|
| Blind LLMs | 35.5 ± 1.7 |
| Socratic LLMs + Frame Captions | 38.1 ± 1.8 |
| **Ours** | **37.2 ± 1.8** |
| Human Agent | 85.1 ± 1.1 |

### Failure Analysis

| Failure Type | Proportion |
|---------|:---:|
| Detection Failure (Object missed/false detection) | ~34% |
| LLM Pseudocode Error | 18% |
| Exploration Policy Uncovered | ~10% |
| Prompt-related Issues | 11% |

### Key Findings
- A single framework achieves or nears SOTA performance on three different embodied AI tasks, validating the generalizability of the "minimal primitive set + LLM composition" approach.
- The primary bottleneck lies in the perception layer (detection failure at 34%) rather than the planning layer (LLM errors at 18%), showing that the LLM composition logic is generally reliable.
- The memory mechanism contributes significantly (+2.6%) to GOAT-Bench lifelong navigation as spatial memory can be reused across targets.

## Highlights & Insights
- **The "minimal primitive set" design philosophy** is highly elegant. Rather than designing more primitives, the core is finding a complete and foundational set of navigation primitives and allowing the LLM's composition capabilities to cover complex tasks. This is analogous to the "orthogonality" principle in programming language design.
- **Extending program synthesis from visual reasoning to embodied environments** is natural and effective. While VisProg/ViperGPT verified LLM program synthesis capabilities on 2D images, TANGO successfully extends this paradigm to 3D embodied environments.
- **Practical value of being training-free**: Eliminates the need to collect data and train models for every new environment or task, significantly lowering deployment barriers.

## Limitations & Future Work
- The PointGoal navigation model still requires pre-training—it is not entirely training-free, only task-level training is avoided.
- Lacks capability for precise grasping or manipulation tasks (e.g., opening doors), as current primitives only support navigation-related actions.
- Exploration efficiency is bounded by the memory-based exploration policy, which might be inefficient in large-scale open environments.
- Highly sensitive to the quality of LLM prompts; poor few-shot examples can lead to incorrect primitive combinations.
- Has not been validated on real physical robots (tested only in simulation environments).

## Related Work & Insights
- **vs SayCan**: SayCan requires a large set of pre-trained skill primitives, each of which must be trained individually. TANGO is much more flexible with only two general primitives, illustrating that "less is more."
- **vs ProgPrompt**: Both belong to the LLM program synthesis paradigm. However, ProgPrompt uses a fixed API, whereas TANGO's primitive design is more tailored to the actual requirements of embodied navigation.
- **vs VOYAGER**: Voyager is also an LLM-driven agent, but it targets skill library construction in Minecraft. TANGO focuses on real-world embodied tasks, featuring more fundamental primitive designs.
- **Inspiration for general Embodied AI**: If two simple primitives can support three distinct tasks, introducing further basic primitives (like grasping, pushing/pulling) could potentially support a wider range of embodied tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically applies LLM program composition to multiple embodied tasks; the "minimal primitive set" concept is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Zero-shot SOTA achieved on three different tasks, validating the generalizability of the approach.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise.
- Value: ⭐⭐⭐⭐ High reference value for training-free general-purpose embodied AI frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Natural Language PDDL (NL-PDDL): Open-world Goal-oriented Commonsense Regression Planning in Embodied AI](../../ICLR2026/llm_agent/natural_language_pddl_nl-pddl_for_open-world_goal-oriented_commonsense_regressio.md)
- [\[CVPR 2026\] BAMI: Training-Free Bias Mitigation in GUI Grounding](../../CVPR2026/llm_agent/bami_training-free_bias_mitigation_in_gui_grounding.md)
- [\[NeurIPS 2025\] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](../../NeurIPS2025/llm_agent/contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)
- [\[CVPR 2025\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)
- [\[ACL 2026\] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](../../ACL2026/llm_agent/zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)

</div>

<!-- RELATED:END -->
