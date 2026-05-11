---
title: >-
  [Paper Note] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation
description: >-
  [CVPR 2026][Robotics][VLN] ProFocus is a training-free progressive framework that achieves state-of-the-art zero-shot VLN performance on R2R and REVERIE benchmarks through two mechanisms: proactive perception (converting…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "VLN"
  - "proactive perception"
  - "MCTS"
  - "zero-shot navigation"
  - "LLM agent"
date: 2026-05-08
content_hash: 007a00111146e9d1
---

# ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation

**Conference**: CVPR 2026
**arXiv**: [2603.05530](https://arxiv.org/abs/2603.05530)
**Code**: None
**Area**: Robotics / Vision-Language Navigation
**Keywords**: VLN, proactive perception, MCTS, zero-shot navigation, LLM agent

## TL;DR
ProFocus is a training-free progressive framework that achieves state-of-the-art zero-shot VLN performance on R2R and REVERIE benchmarks through two mechanisms: proactive perception (converting panoramic observations into semantic maps and having an LLM generate targeted visual queries) and focused reasoning (BD-MCTS filtering top-k high-value waypoints from large navigation histories).

## Background & Motivation
**Background**: Vision-and-Language Navigation (VLN) requires agents to navigate physical environments following natural language instructions. Foundation model-based VLN methods have shown promise through either post-training adaptation or zero-shot prompting, yet both suffer from two critical shortcomings.

**Limitations of Prior Work**: (1) *Passive visual perception* — VLM-driven methods uniformly process panoramic or multi-view visual inputs, causing redundant information to inflate visual token counts and diffuse attention across irrelevant features, obscuring instruction-relevant fine-grained cues; (2) *Unfocused reasoning* — both paradigms receive large amounts of unprioritized historical context containing past observations and waypoints, whereby long trajectory histories dilute attention and impede precise reasoning.

**Key Challenge**: Navigation requires selective perception (acquiring only task-relevant information) and focused reasoning (attending only to high-value historical waypoints), whereas existing methods process everything in bulk on both fronts.

**Goal**: To proactively acquire task-relevant visual observations to reduce perceptual redundancy, and to focus reasoning on high-value waypoints within extensive navigation histories.

**Key Insight**: Establish a closed-loop perception–reasoning cycle through LLM–VLM collaboration, and employ an MCTS variant to filter key waypoints from the global history.

**Core Idea**: An LLM determines "what needs to be known" based on the semantic map and generates visual queries; a VLM performs fine-grained perception within specified regions; BD-MCTS then focuses decision-making on the top-k high-value waypoints extracted from the full navigation history.

## Method

### Overall Architecture
ProFocus comprises three specialized agents: an orchestration agent $\mathcal{A}_{\mathrm{orch}}^{\boldsymbol{\theta}}$ (LLM, responsible for spatial reasoning and semantic evaluation), a perception agent $\mathcal{A}_{\mathrm{perc}}^{\boldsymbol{\phi}}$ (VLM, responsible for fine-grained perception), and a decision agent $\mathcal{A}_{\mathrm{dec}}^{\boldsymbol{\psi}}$ (LLM, reasoning over top-k candidates). The framework takes panoramic images and language instructions as input and outputs navigation actions.

### Key Designs

1. **Ego-centric Semantic Map**:

    - **Function**: Converts panoramic observations into a structured textual representation encoding object positions, depths, and directional relationships.
    - **Mechanism**: The panorama is divided into $K$ directional views; a VLM detects all objects in parallel, producing $\{(\boldsymbol{b}_i, \textit{obj}_i)\}_{i=1}^{N_t}$. Monocular depth estimation yields depth $d_i$, and the heading angle is computed as $h_i = \pi \cdot (\frac{x_1 + x_2 - F}{F})$. The semantic map is then constructed as $\mathcal{C}_t = \{(h_i, \textit{obj}_i(\boldsymbol{b}_i), d_i)\}_{i=1}^{N_t}$ and formatted as natural language text.
    - **Design Motivation**: Compressing visual information into structured text enables the LLM to perform spatial reasoning (e.g., "the object to the left") while avoiding the processing of large volumes of raw pixels.

2. **Reasoning-Driven Proactive Perception Loop**:

    - **Function**: Proactively acquires instruction-relevant visual information based on task demands, rather than passively processing all inputs.
    - **Mechanism**: The orchestration agent generates a visual query $\boldsymbol{q}$ and a focus region $\boldsymbol{R}_{\text{focus}}^t$ conditioned on the semantic map, trajectory history, and instruction: $(\boldsymbol{q}, \boldsymbol{R}_{\text{focus}}^t) = \mathcal{A}_{\mathrm{orch}}^{\boldsymbol{\theta}}(\mathcal{C}_t, \boldsymbol{\tau}_t, \mathcal{I}, \mathcal{H}_{\text{query}})$. The perception agent then performs fine-grained analysis within the focus region: $\boldsymbol{a}_i^t = \mathcal{A}_{\mathrm{perc}}^{\boldsymbol{\phi}}(\boldsymbol{\mathcal{O}}_t|_{\boldsymbol{R}_{\text{focus}}^t}, \boldsymbol{q})$. The loop iterates until the information is deemed sufficient ($s_t = \text{sufficient}$).
    - **Design Motivation**: Restricting perception to instruction-relevant regions reduces visual token count, enhances fine-grained attribute recognition through targeted queries, and enables adaptive perception.

3. **Branch-Diverse MCTS (BD-MCTS)**:

    - **Function**: Filters top-k high-value waypoint candidates from large navigation histories to guide focused reasoning in the decision agent.
    - **Mechanism**: A search tree $\mathcal{T} = \langle \boldsymbol{V}_{\mathcal{T}}, \boldsymbol{E}_{\mathcal{T}}, Q, N \rangle$ is maintained across three phases — (I) new nodes are initialized using semantic value $V_{\text{sem}}(u)$ in lieu of random rollouts; (II) $Q$-values are dynamically refined via backpropagation along the path: $Q(v) \leftarrow Q(v) + \frac{R_t - Q(v)}{N(v)}$; (III) candidates are ranked by a path-aggregated score with distance penalty $\text{Score}(v) = V_{\text{path}}(v) - \lambda \cdot \frac{d_{\mathcal{G}}(v_t, v)}{\max_{u} d_{\mathcal{G}}(v_t, u)}$, with each parent node contributing at most 2 child nodes to maintain branch diversity.
    - **Design Motivation**: Standard MCTS selects a single optimal action, whereas BD-MCTS selects diverse top-k candidates. The distance penalty ensures physical reachability, and the branch diversity constraint ensures coverage of different exploration directions.

### Loss & Training
ProFocus is a **fully training-free** framework requiring no fine-tuning or training of any model component. It employs off-the-shelf LLMs (Qwen3-Max / DeepSeek-V3) and VLMs (Qwen3-VL-Max / GLM-4.5V).

## Key Experimental Results

### Main Results
Results on R2R validation unseen set:

| Method | NE↓ | OSR↑ | SR↑ | SPL↑ |
|--------|-----|------|-----|------|
| NavGPT (GPT-4) | 6.46 | 42.0 | 34.0 | 29.0 |
| MapGPT (GPT-4V) | 5.63 | 57.6 | 43.7 | 34.8 |
| MSNav (GPT-4o) | 5.24 | 65.0 | 46.0 | 40.0 |
| **ProFocus (Q3+Q3VL)** | **4.92** | **65.0** | **52.5** | **39.8** |
| **ProFocus (DS3+GLM)** | 5.21 | 63.0 | 50.0 | 41.2 |

### Ablation Study

| Configuration | SR↑ | SPL↑ | Notes |
|---------------|-----|------|-------|
| NavGPT† (DS3) | 36.0 | 28.1 | No proactive perception, no focused reasoning |
| MapGPT† (GLM) | 41.4 | 30.8 | Passive VLM-based perception |
| ProFocus (DS3+GLM) | 50.0 | 41.2 | + Proactive Perception + BD-MCTS |
| ProFocus (Q3+Q3VL) | 52.5 | 39.8 | Stronger backbone further improves results |

### Key Findings
- SR improves from 36.0% (NavGPT) to 52.5%, an absolute gain of 16.5 percentage points.
- Proactive perception improves both efficiency and accuracy by reducing visual tokens and enhancing fine-grained attribute recognition.
- The branch diversity constraint in BD-MCTS is critical for avoiding local optima.
- The training-free framework already surpasses certain trained methods under zero-shot settings.

## Highlights & Insights
- The three-agent collaboration exhibits a clear division of labor — orchestration (planning), perception (execution), decision (reasoning) — mirroring the human cognitive process during navigation.
- The performance gains of proactive over passive perception demonstrate that selective, high-quality visual information is superior to exhaustive but noisy input.
- BD-MCTS introduces a human-like prioritized memory access mechanism into VLN: humans do not uniformly recall all past states either.
- The training-free design allows seamless integration of stronger LLM/VLM backbones as they become available.

## Limitations & Future Work
- **API overhead**: Each navigation step requires multiple LLM/VLM calls, resulting in non-trivial latency.
- The semantic map quality depends on the accuracy of object detection and depth estimation.
- Evaluation is conducted with specific LLM/VLM combinations; prompt engineering may need adjustment when switching models.
- Only R2R and REVERIE are evaluated; generalization to more challenging continuous-environment navigation remains unverified.

## Related Work & Insights
- **NavGPT**: Converts panoramic scenes to text descriptions for GPT-4-based decision-making; relies on passive perception.
- **MapGPT**: Employs VLM-based map representations but still processes all views passively.
- **AO-Planner**: Uses SAM-based visual affordance prompting but lacks an active query mechanism.
- The proactive perception + focused reasoning paradigm generalizes to other tasks requiring long-horizon historical reasoning, such as dialogue systems and long-document question answering.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of a proactive perception loop and BD-MCTS is novel in the VLN context; the closed-loop perception–reasoning design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two standard benchmarks (R2R and REVERIE) with multiple model configurations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rigorous formalization.
- **Value**: ⭐⭐⭐⭐ The training-free framework is highly practical and can serve as a general-purpose enhancement module for LLM/VLM-based navigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[AAAI 2026\] Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](../../AAAI2026/robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)

</div>

<!-- RELATED:END -->
