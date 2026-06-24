---
title: >-
  [Paper Note] VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search
description: >-
  [ACL 2025][VLM Reasoning][Multimodal Reasoning] This paper proposes the VisuoThink framework, which dynamically integrates visual aids during reasoning and explores multiple reasoning paths through vision-text interleaved reasoning and predictive rollout tree search. Without fine-tuning, VisuoThink achieves SOTA performance on geometric and spatial reasoning tasks (reaching up to 48.5% Accuracy@1 on Geomverse-109, a 21.8% improvement over the best baseline).
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Multimodal Reasoning"
  - "Tree Search"
  - "Vision-Language Interleaved Thinking"
  - "Test-Time Scaling"
  - "Geometric Reasoning"
date: 2026-05-08
content_hash: 7276246a0fd0a164
---

# VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search

**Conference**: ACL 2025  
**arXiv**: [2504.09130](https://arxiv.org/abs/2504.09130)  
**Code**: [GitHub](https://github.com/ekonwang/VisuoThink)  
**Area**: Multimodal VLM / LLM Reasoning  
**Keywords**: Multimodal Reasoning, Tree Search, Vision-Language Interleaved Thinking, Test-Time Scaling, Geometric Reasoning

## TL;DR

This paper proposes the VisuoThink framework, which dynamically integrates visual aids during reasoning and explores multiple reasoning paths through vision-text interleaved reasoning and predictive rollout tree search. Without fine-tuning, VisuoThink achieves SOTA performance on geometric and spatial reasoning tasks (reaching up to 48.5% Accuracy@1 on Geomverse-109, a 21.8% improvement over the best baseline).

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) have achieved significant progress in various tasks. Following the success of the o1 model series, researchers have begun exploring the integration of "slow thinking" mechanisms into LVLMs. Prior methods extend deep thinking capabilities through phased reasoning, curriculum learning, and tree search data generation.

**Limitations of Prior Work**: Existing expansion methods suffer from two core limitations: (1) they view visual information solely as static input, making the reasoning process rely entirely on textual chains—creating a "visual blind spot" and ignoring the potential of active vision during reasoning; (2) although methods like VisualSketchpad and VoT have attempted to incorporate visual aids, they are limited to single-step assistance or simplified visual prompts (such as emojis), lacking the multi-step vision-language interleaved reasoning process typical of human slow thinking.

**Key Challenge**: When humans solve complex geometric problems, they repeatedly draw auxiliary lines, visualize intermediate steps, and explore different reasoning paths. However, current LVLM reasoning methods lack both multi-step visual assistance and systematic search mechanisms to explore alternative reasoning paths.

**Goal**: (1) To enable multi-step vision-text interleaved reasoning during inference; (2) to realize test-time scaling for reasoning through search strategies.

**Key Insight**: Inspired by the slow thinking mechanism in human cognition, this work combines visual tool execution with tree search algorithms. This approach generates reliable visual aids at each reasoning step while optimizing the selection of reasoning paths through rollout search.

**Core Idea**: To model multimodal reasoning as a tree search process consisting of "vision-text interleaved expansion + predictive rollout + self-voting selection," enabling multimodal slow thinking in LVLMs.

## Method

### Overall Architecture

The VisuoThink framework consists of a loop over three core phases: (1) Vision-text interleaved expansion: generating $k$ candidate reasoning paths, where each path contains a Thought-Action-Observation loop; (2) Predictive rollout: simulating the complete reasoning process down to the final result for each candidate node; (3) Selection: selecting the most promising path via a self-voting mechanism. This ultimately constructs a reasoning tree, yielding the final answer along the optimal path.

### Key Designs

1. **Vision-Text Interleaved Thinking**:

    - **Function**: Dynamically fuse vision and text information at each step of reasoning.
    - **Mechanism**: Adopting a ReAct-like iterative loop of Thought-Action-Observation: (1) Thought phase: the model performs text-based reasoning based on current visual information and plans the required visual aids for the next step; (2) Action phase: it calls external tools (such as Python matplotlib) to generate or modify visual info (such as drawing auxiliary lines, labeling key features, etc.); (3) Observation phase: it processes the visual feedback returned by the tool, integrating it into the next step of reasoning. Unlike VisualSketchpad, the key distinction lies in the step-by-step construction rather than generating all visual aids at once.
    - **Design Motivation**: Step-by-step visual construction naturally integrates with search techniques, allowing search algorithms to evaluate and optimize visual reasoning paths at each step.

2. **Predictive Rollout Search**:

    - **Function**: Evaluate the potential of current candidate paths by simulating future reasoning outcomes.
    - **Mechanism**: At each reasoning step, the model samples $k$ candidate nodes $S_t = \{s_t^1, ..., s_t^k\}$. For each candidate node, a rollout simulation is executed—performing vision-text interleaved reasoning along a single path until a final result $r_t^i$ is reached. Then, a self-voting mechanism $\mathbf{Select}(S_t) = \arg\max_{s_t^i \in S_t} \mathbf{Vote}(A_{t-1}, s_t^i, r_t^i)$ is applied to select the optimal candidate. The LVLM itself acts as the heuristic function, conducting the vote by comprehensively considering the historical context, the candidate nodes, and the simulation results.
    - **Design Motivation**: Visual reasoning often requires multiple steps to reach a conclusion, and single-step evaluation is insufficient to judge the potential of a path. Guided by predictive rollouts, the model can "forsee" the possible outcomes of each path to make more informed decisions.

3. **Two-Stage Framework for Geometry Problem Solving**:

    - **Function**: Design a specialized visual construction and algebraic computation flow for geometry problems.
    - **Mechanism**: Phase I (Visual Construction): The model progressively generates auxiliary lines defined by geometric constraints (such as connecting points, drawing perpendicular/parallel lines), ending with an AUX-END token; Phase II (Algebraic Computation): The geometric relationships are converted into solvable equations and executed using Python code for precise computation, alleviating the hallucination issues of LVLM numerical reasoning.
    - **Design Motivation**: Geometry problems cannot rely solely on visual construction or the model's intrinsic capabilities; precise numerical calculation tools are required to ensure the accuracy of results.

### Loss & Training

VisuoThink is an inference-time framework and **does not require any fine-tuning**. Reasoning is directly performed using existing SOTA closed-source models (GPT-4o, Claude-3.5-sonnet) and open-source models (Qwen2-VL-72B-Instruct). The search hyperparameters include the tree width $k$ and the maximum reasoning steps $\tau$.

## Key Experimental Results

### Main Results

| Dataset | Method | GPT-4o | Claude-3.5 | Qwen2-VL-72B |
|--------|------|--------|------------|--------------|
| Geomverse-109 | CoT | 11.1 | 14.4 | 5.6 |
| Geomverse-109 | VisualSketchpad+Eq | 13.3 | 17.8 | 11.1 |
| Geomverse-109 | VisuoThink (w/o rollout) | 24.4 | 26.7 | 19.0 |
| Geomverse-109 | **VisuoThink** | **28.9** | **27.8** | **25.6** |
| Visual Nav (level-3) | CoT | 18.8 | 37.5 | 6.7 |
| Visual Nav (level-3) | VoT+Executor | 62.5 | 68.8 | 25.0 |
| Visual Nav (level-3) | **VisuoThink** | **93.8** | **93.8** | **81.3** |
| Visual Tiling (level-2) | CoT | 0.8 | 0.8 | 0.0 |
| Visual Tiling (level-2) | **VisuoThink** | **51.2** | **84.0** | **20.2** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| VisuoThink w/o rollout vs w/ rollout | Avg. +4.1% on geometric tasks | Moderate improvement from rollout search in geometric tasks |
| VisuoThink w/o rollout vs w/ rollout | Avg. +34.7% on spatial reasoning | Substantial improvement in spatial reasoning tasks |
| Reasoning steps 10→20→40 | +54.1% → +6.5% (GPT-4o) | More steps are helpful but yield diminishing marginal returns |
| Tree width $k=1\to3\to5\to7$ | Inverted U-shape trend | Excessively large tree width instead reduces performance due to node evaluation noise |

### Key Findings

- VisuoThink significantly outperforms baselines across all evaluated models and tasks, achieving an average improvement of 17.1% and 16.7% over CoT and VisualSketchpad respectively on Geomverse-109.
- The gain from rollout search is much greater in spatial reasoning tasks than in geometric tasks, likely because spatial reasoning provides stronger supervisory signals (e.g., visual states like the agent's final position).
- Increasing the number of reasoning steps from 10 to 20 brings substantial improvement, but from 20 to 40 the marginal return drops sharply, indicating that merely increasing trial-and-error opportunities cannot resolve the most difficult samples.
- There exists an optimal value for tree width (typically 3-5); an excessively large width causes confusion due to inherent errors in the model's node evaluation, which is a counter-intuitive but important finding.

## Highlights & Insights

- Integrates multimodal tree search into LVLM reasoning for the first time, realizing joint exploration of vision and language reasoning paths.
- Achieves substantial performance improvements during inference without fine-tuning, fully exploiting the capability boundaries of existing models.
- The discrepancy in gains between strong and weak supervision in rollout search aligns with findings from DeepSeek-R1, underscoring the importance of outcome-based supervision.
- The inverted U-shaped trend of tree width reveals that "more search is not always better," offering general insights for test-time scaling research.

## Limitations & Future Work

- High computational overhead: Predictive rollout search introduces a significant computational burden, making it unsuitable for real-time applications.
- Reliance on external tool interactions, which may require additional adaptation in certain deployment environments.
- Constrained by the capability ceiling of the base VLM, failing to overcome fundamental limitations of the model.
- Evaluations are limited to geometric and spatial reasoning; performance on broader visual reasoning tasks (e.g., physical world understanding, chart reasoning) remains unverified.
- The self-voting selection mechanism relies on the model's own judgment, which may be unreliable in certain scenarios.

## Related Work & Insights

- VisualSketchpad (Hu et al., 2024) and VoT (Wu et al., 2024) are pioneering works in vision-aided reasoning but are limited to single-step assistance.
- The success of the o1 series and DeepSeek-R1 in textual slow thinking provides the motivation for multimodal expansion.
- The successful application of MCTS in AlphaZero and LLM decoding inspired the design of predictive rollout search.
- This framework could potentially be integrated with training-time methods (such as RLHF) to further enhance the multimodal reasoning capabilities of LVLMs.

## Rating

- **Novelty**: 8/10 — Multimodal tree search is a novel and valuable paradigm.
- **Technical Depth**: 7/10 — The framework is clearly designed, but components are relatively standard (variants of ReAct and MCTS).
- **Experimental Thoroughness**: 8/10 — Multi-model and multi-task evaluation with in-depth ablation analysis.
- **Writing Quality**: 8/10 — Clear structure with good visual presentation.
- **Value**: 7/10 — Computational cost limits direct application, but the core concepts are inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[ICML 2025\] Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger](../../ICML2025/vlm_reasoning/re-ranking_reasoning_context_with_tree_search_makes_large_vision-language_models.md)
- [\[ACL 2025\] Progressive Multimodal Reasoning via Active Retrieval](progressive_multimodal_reasoning_via_active_retrieval.md)
- [\[ICLR 2026\] Empowering Small VLMs to Think with Dynamic Memorization and Exploration](../../ICLR2026/vlm_reasoning/empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)
- [\[ICLR 2026\] Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search](../../ICLR2026/vlm_reasoning/mini-o3_scaling_up_reasoning_patterns_and_interaction_turns_for_visual_search.md)

</div>

<!-- RELATED:END -->
