---
title: >-
  [Paper Note] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents
description: >-
  [CVPR 2026][LLM Agent][GUI agent] This paper proposes HATS — a hardness-aware trajectory synthesis framework that identifies and handles semantically ambiguous GUI actions via two closed-loop modules: hardness-driven exploration and alignment-guided refinement, significantly improving the cross-environment generalization of GUI agents.
tags:
  - CVPR 2026
  - LLM Agent
  - GUI agent
  - trajectory synthesis
  - semantic ambiguity
  - hardness-aware
  - data augmentation
date: 2026-05-08
content_hash: 8b85fbb311636dc3
---

# HATS: Hardness-Aware Trajectory Synthesis for GUI Agents

**Conference**: CVPR 2026
**arXiv**: [2603.12138](https://arxiv.org/abs/2603.12138)
**Code**: None
**Area**: LLM Agent
**Keywords**: GUI agent, trajectory synthesis, semantic ambiguity, hardness-aware, data augmentation

## TL;DR

This paper proposes HATS — a hardness-aware trajectory synthesis framework that identifies and handles semantically ambiguous GUI actions via two closed-loop modules: hardness-driven exploration and alignment-guided refinement, significantly improving the cross-environment generalization of GUI agents.

## Background & Motivation

**State of the Field**: VLM-based GUI agents have shown great potential for automating digital tasks. The dominant training paradigm relies on high-quality action trajectory data — each trajectory containing a task description, UI screenshots, and an action sequence. Various trajectory synthesis pipelines have been designed to produce training data at scale.

**Limitations of Prior Work**: Existing trajectory synthesis methods tend to produce agents that only handle simple interactions with poor generalization. The authors identify the root cause as these pipelines ignoring **semantically ambiguous actions** — actions whose meaning depends on context, action sequence, or visual environment — for example: (1) context-dependent actions: the same button carries different meanings depending on page state; (2) sequence-dependent actions: the correctness of certain actions depends on the results of prior actions; (3) visually ambiguous actions: UI elements that are difficult to distinguish visually.

**Root Cause**: Semantically ambiguous actions are common in real-world GUI interactions and are critical for training robust agents, yet they are severely underrepresented and poorly handled in existing datasets. This leads to semantic misalignment between task instructions and actual execution, causing agents to frequently fail in complex real-world environments.

**Paper Goals**: Design a framework that (1) actively discovers and collects high-quality trajectories containing semantically ambiguous actions, and (2) automatically detects and repairs instruction-execution alignment issues.

**Starting Point**: Introduce the concept of "hardness" to quantify the degree of semantic ambiguity in actions, using it as the core signal for data collection and quality assurance.

**Core Idea**: Form a closed loop through hardness-driven active exploration and alignment-guided iterative refinement — the exploration module supplies challenging trajectories to the refinement module, whose feedback updates the hardness signal to guide future exploration.

## Method

### Overall Architecture

HATS consists of two complementary modules forming a closed loop: (1) **Hardness-Driven Exploration (HDE)**, which steers data collection toward semantically ambiguous yet informative interaction regions; and (2) **Alignment-Guided Refinement (AGR)**, which iteratively validates and repairs instruction-execution alignment in synthesized trajectories. The two modules form a positive feedback loop: HDE supplies high-hardness trajectory samples to AGR, while AGR's refinement results feed back to update the hardness signal and improve subsequent exploration strategies.

### Key Designs

1. **Semantic Ambiguity Definition & Hardness Quantification**:

    - *Function*: Assign each GUI action a hardness score reflecting its degree of semantic ambiguity.
    - *Mechanism*: Compute hardness by jointly considering three sources of ambiguity: (1) contextual ambiguity — the degree to which the meaning of an action changes across different UI states; (2) sequential ambiguity — the extent to which the correctness of an action depends on prior actions; (3) visual ambiguity — the degree to which UI elements are visually distinguishable. A high hardness score indicates that the action is more likely to cause agent misjudgment.
    - *Design Motivation*: Unlike naively equating "failure rate" with "hardness," this multi-dimensional definition more accurately characterizes the root sources of ambiguity, making the hardness signal both interpretable and actionable.

2. **Hardness-Driven Exploration (HDE)**:

    - *Function*: Guide the trajectory collection process to favor coverage of high-hardness, high-information interaction scenarios.
    - *Mechanism*: Assign higher sampling probability to high-hardness regions within the exploration policy. Specifically, the behavior strategy of the exploration agent is adjusted according to the current hardness distribution, causing it to more frequently visit GUI paths containing semantically ambiguous actions. An experience buffer is maintained to record explored high-hardness interaction patterns and avoid redundant sampling.
    - *Design Motivation*: Random exploration wastes significant resources on simple, repetitive interactions, while the samples that truly improve generalization are the "hard" ones containing ambiguous actions. Actively biasing toward high-hardness regions substantially improves data efficiency.

3. **Alignment-Guided Refinement (AGR)**:

    - *Function*: Iteratively validate and repair instruction-execution alignment issues in synthesized trajectories.
    - *Mechanism*: Perform alignment verification on each synthesized trajectory — checking whether each action step is consistent with the intent of the task instruction. When an inconsistency is detected, a repair operation is executed: adjusting action annotations, modifying instruction descriptions, or resampling the affected trajectory segment. The refinement process is iterative: alignment quality is re-evaluated after each round of corrections until a quality threshold is met. Error patterns discovered during refinement are fed back to HDE to update the hardness signal.
    - *Design Motivation*: Even with active collection of high-hardness trajectories, the inherent nature of ambiguity means synthesized instruction-action pairs may still suffer from semantic misalignment due to missing contextual details, necessitating post-hoc correction.

### Loss & Training

- GUI agents are trained on high-quality trajectories synthesized by HATS.
- Training follows the standard VLM fine-tuning paradigm, including an action prediction loss (predicting the next action given UI screenshots and task description).
- The HDE and AGR modules operate during the data synthesis phase, forming a closed-loop iterative data production process; the final high-quality trajectories are used during training.

## Key Experimental Results

### Main Results

| Method | MiniWob++ | WebArena | AndroidWorld | Avg. Gain |
|--------|-----------|----------|--------------|-----------|
| Random synthesis baseline | Base | Base | Base | — |
| Simple filtering pipeline | +Moderate | +Moderate | +Moderate | Moderate |
| HATS (Ours) | **Best** | **Best** | **Best** | **Significantly outperforms SOTA** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full HATS | Best | Complete model |
| w/o HDE | Significant drop | Random exploration fails to cover ambiguous scenarios |
| w/o AGR | Moderate drop | Uncorrected trajectories contain semantic misalignment |
| w/o closed-loop feedback | Drop | HDE and AGR operating independently are weaker than the closed loop |

### Key Findings

- Semantically ambiguous actions constitute a small fraction of existing datasets yet are critical for generalization — HATS substantially increases their presence through targeted collection.
- The closed-loop design outperforms the two modules operating independently — error patterns discovered by AGR demonstrably improve HDE's exploration strategy.
- Agents trained with HATS consistently outperform SOTA baselines across multiple diverse GUI environments, demonstrating strong cross-environment generalization.
- The hardness signal gradually converges over training iterations, indicating that the framework automatically learns the truly difficult interaction patterns in the environment.

## Highlights & Insights

- **Depth of problem formulation**: Rather than simply "generating more data" or "filtering better," the paper precisely pinpoints "semantically ambiguous actions" as the neglected root cause — the problem analysis is more valuable than the technical solution itself.
- **Elegance of the closed-loop design**: The exploration module produces data, the refinement module validates quality, and quality feedback guides exploration — this self-improving data flywheel mechanism is both simple and effective.
- **Transferable paradigm**: The "identify hard samples → actively collect → verify and repair" data production paradigm is broadly applicable to agent training in domains such as robotic control and code generation.

## Limitations & Future Work

- The specific metrics for hardness quantification (how to compute contextual/sequential/visual ambiguity) rely on heuristic design and may require adaptation for different GUI environments.
- The AGR module itself depends on a capable VLM to judge alignment quality, introducing an additional model dependency.
- The current framework focuses on web and mobile GUI; coverage of complex interactions in desktop applications (e.g., IDEs, design tools) is insufficient.
- Future work could incorporate the hardness signal into the training phase as curriculum learning (easy before hard), rather than using it solely during data synthesis.

## Related Work & Insights

- **vs. WebVoyager/DigiRL et al.**: Existing GUI agent data synthesis primarily relies on demonstration recording or reinforcement learning exploration. HATS's key distinction is explicitly modeling the semantic ambiguity of actions and designing the data production system around it.
- **vs. CogAgent/ShowUI**: These methods focus on model architecture improvements (e.g., high-resolution understanding, UI element grounding), while HATS addresses generalization from a data quality perspective — the two approaches are complementary.
- This work suggests that in agent training, "hardness-awareness" matters more than "data volume" — a carefully curated small set of high-quality hard samples may outperform a large collection of simple ones.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The conceptual definition of "semantic ambiguity" and the closed-loop hardness-driven framework are clear highlights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple GUI benchmarks with ablations, though unavailability of the full HTML paper limits access to details.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is articulated clearly; method description is well-structured.
- **Value**: ⭐⭐⭐⭐ Provides a systematic solution to data quality issues in GUI agent training.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[CVPR 2026\] Gen-n-Val: Agentic Image Data Generation and Validation](gen_n_val_agentic_image_data_generation_and_validation.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)

<!-- RELATED:END -->
