---
title: >-
  [Paper Note] GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation
description: >-
  [NeurIPS 2025][Robotics][GUI Navigation] This paper proposes GUI-Rise, a framework that jointly designs three subtasks—structured reasoning (progress estimation + decision reasoning), action prediction, and history summarization—combined with GRPO reinforcement learning and a history summarization reward, to significantly improve the cross-domain generalization of GUI navigation agents.
tags:
  - NeurIPS 2025
  - Robotics
  - GUI Navigation
  - Structured Reasoning
  - History Summarization
  - GRPO
  - Chain-of-Thought
date: 2026-05-08
content_hash: a9145f36ab07fe99
---

# GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation

**Conference**: NeurIPS 2025
**arXiv**: [2510.27210](https://arxiv.org/abs/2510.27210)
**Code**: [leon022.github.io/GUI-Rise](https://leon022.github.io/GUI-Rise)
**Area**: Robotics
**Keywords**: GUI Navigation, Structured Reasoning, History Summarization, GRPO, Chain-of-Thought

## TL;DR

This paper proposes GUI-Rise, a framework that jointly designs three subtasks—structured reasoning (progress estimation + decision reasoning), action prediction, and history summarization—combined with GRPO reinforcement learning and a history summarization reward, to significantly improve the cross-domain generalization of GUI navigation agents.

## Background & Motivation

- **GUI navigation is an important direction**: MLLM-driven GUI navigation agents can translate natural language instructions into interface operations, but maintaining consistency across multi-step interactions remains challenging.
- **Poor generalization of existing methods**: GPT-4-based prompt engineering approaches are constrained by frozen model capabilities; SFT-based open-source approaches tend to overfit static instruction-action pairs and generalize poorly across domains.
- **Deficient history representations**: Existing systems either encode only action sequences (losing visual state information) or use full screenshot sequences (high computational cost, severe context window limitations), neither of which integrates historical information as efficiently as humans.
- **Insufficient long-horizon reasoning**: Effective GUI decision-making requires reliance on completed actions and interface evolution, yet current agents exhibit clear shortcomings in long-term coherent reasoning.
- **SFT struggles to learn structured reasoning**: Ablation experiments show that training structured CoT with SFT alone leads to performance degradation; reinforcement learning is required to genuinely improve reasoning quality.
- **Lack of supervision for history summarization**: Unsupervised history summarization may yield low-quality summaries that mislead policy learning, necessitating specially designed reward functions to ensure summarization quality.

## Method

### Overall Architecture

GUI-Rise executes a cycle of three subtasks at each interaction step: (1) **Structured Reasoning**—analyzing the current screenshot and history summary to generate a CoT containing progress estimation and decision reasoning; (2) **Action Prediction**—producing executable GUI actions (type + value + coordinates) based on the reasoning output; (3) **History Summarization**—compressing new information into a concise textual summary for use in the next step. The outputs of all three subtasks are serialized via XML tags and generated autoregressively by an MLLM (Qwen-VL series).

### Key Design 1: Structured Reasoning Subtask

- **Function**: Decomposes CoT reasoning into two explicit phases: Progress Estimation and Decision Reasoning.
- **Mechanism**: The agent first evaluates task completion progress based on the current screenshot $\mathbf{o}_t$ and history $\mathbf{h}_{t-1}$, then determines the direction of the next action by combining the user instruction $\mathbf{u}$ and prior decisions.
- **Design Motivation**: Mimics the cognitive strategy humans use when navigating interfaces—first assessing "where am I in the task," then deciding "what to do next"—achieving step-by-step coherence and interpretability.

### Key Design 2: History Summarization Subtask

- **Function**: At each step, compresses the current observation $\mathbf{o}_t$, the previous summary $\mathbf{h}_{t-1}$, and the instruction $\mathbf{u}$ into a concise textual memory $\mathbf{h}_t$.
- **Mechanism**: Replaces raw screenshot sequences or action lists with semantic summaries, tracking task progress continuously in fixed-length text without being constrained by context window size.
- **Design Motivation**: Raw screenshots are computationally expensive and force context truncation; pure action sequences lose visual state information. Semantic summaries combine hierarchical abstraction with task-context grounding, effectively supporting multi-step reasoning.

### Key Design 3: Two-Stage Training Strategy (Cold-start + RL)

- **Function**: The first stage uses GPT-4o-mini-generated pseudo-labels for SFT cold-start; the second stage fine-tunes with GRPO reinforcement learning in a simulated GUI environment.
- **Mechanism**: Cold-start establishes foundational reasoning and summarization capabilities, avoiding overly sparse rewards in the early RL phase; the RL stage develops adaptive reasoning strategies through environment interaction.
- **Design Motivation**: Training structured reasoning directly with SFT yields poor results (ablation Row 3 shows a large performance drop); RL is necessary to elicit reasoning exploration in small models.

### Key Design 4: Triple Reward Function Design

- **Function**: Three complementary reward functions are designed: format reward $\mathcal{R}^f$, action reward $\mathcal{R}^a$, and history summarization reward $\mathcal{R}^h$.
- **Mechanism**: Total reward $r_{t,i} = r^f_{t,i} + \lambda^a \cdot r^a_{t,i} + \lambda^h \cdot r^h_{t,i}$. The history summarization reward evaluates the quality of a summary's support for future actions via $k$ additional rollouts, directly linking summarization value to subsequent action correctness.
- **Design Motivation**: Unsupervised summaries may be low-quality and mislead the policy (ablation Row 5 vs. Row 6); the feedback loop of "summary → future action accuracy" drives the model to actively learn to extract task-relevant historical cues.

## Loss & Training

- **Cold-start stage**: Standard token-level cross-entropy loss $\mathcal{L}_{\text{CE}}$, applied to SFT on pseudo-labels (CoT + action + summary).
- **RL stage**: GRPO algorithm, computing advantage $A_{t,i}$ via group-level normalization; the weighted sum of three rewards is used for policy gradient optimization without a value network.

## Key Experimental Results

### Table 1: Mind2Web Cross-Domain Evaluation (Step SR)

| Method | Backbone | Cross-Task | Cross-Website | Cross-Domain |
|--------|----------|-----------|--------------|-------------|
| ShowUI-2B | Qwen2-VL-2B | 37.2 | 35.1 | 35.2 |
| **GUI-Rise** | Qwen2-VL-2B | **38.8** | **35.4** | **39.7** |
| Qwen2.5-VL-3B | Qwen2.5-VL-3B | 48.3 | 43.5 | 44.1 |
| **GUI-Rise** | Qwen2.5-VL-3B | 46.2 | **44.7** | **47.6** |
| ShowUI-2B (ZS) | Qwen2-VL-2B | 18.6 | 16.8 | 21.4 |
| **GUI-Rise (ZS)** | Qwen2-VL-2B | **24.2** | **21.1** | **29.7** |

**Key Findings**: In the zero-shot setting, GUI-Rise outperforms ShowUI on Cross-Domain by 38.7% (29.7 vs. 21.4).

### Table 2: AITW Mobile Evaluation (Overall Accuracy)

| Method | Backbone | In-Domain | Zero-Shot |
|--------|----------|-----------|-----------|
| ShowUI-2B | Qwen2-VL-2B | 70.0 | 35.9 |
| **GUI-Rise** | Qwen2-VL-2B | **71.1** | **54.1** |
| Qwen2.5-VL-3B | Qwen2.5-VL-3B | 72.5 | 38.9 |
| **GUI-Rise** | Qwen2.5-VL-3B | **73.7** | **56.0** |

**Key Findings**: In the zero-shot setting, GUI-Rise outperforms ShowUI by 50.7% (54.1 vs. 35.9), with a gain of +15.5 points on the complex WebShop task.

### Ablation Study (AITW Overall)

| Configuration | TST | SCoT | HS | HSR | Overall |
|--------------|-----|------|----|-----|---------|
| Baseline | × | × | × | × | 67.2 |
| + RL only | ✓ | × | × | × | 66.0 |
| + SCoT SFT only | × | ✓ | × | × | 42.6 |
| + RL + SCoT | ✓ | ✓ | × | × | 69.8 |
| + History | ✓ | ✓ | ✓ | × | 70.7 |
| + History Reward | ✓ | ✓ | ✓ | ✓ | **71.1** |

## Highlights & Insights

- The **three-subtask joint framework** is elegantly designed; the reasoning (progress estimation + decision analysis) → action → summarization cycle closely mirrors human cognitive patterns.
- The **history summarization reward** is highly innovative: by using rollouts to directly bind summarization quality to future action correctness, it forms a self-improving feedback loop.
- **Outstanding cross-domain generalization**: in zero-shot settings, GUI-Rise achieves relative improvements of 38.7% (Mind2Web) and 50.7% (AITW) over ShowUI, validating the importance of structured reasoning for generalization.
- Ablation experiments clearly demonstrate the contribution of each component, particularly revealing the important finding that "SFT cannot learn structured reasoning effectively and RL is indispensable."

## Limitations & Future Work

- The model is trained only on offline data and cannot learn or adapt to new scenarios from online interaction.
- Pseudo-labels depend on GPT-4o-mini, imposing an upper-bound constraint on cold-start label quality.
- Validation is limited to 2B/3B scale models; scalability to larger models remains unclear.
- History summaries are purely textual, potentially losing fine-grained visual information (e.g., small button states).

## Related Work & Insights

- **GUI Agents**: CogAgent, SeeClick, ShowUI, UI-TARs, and others improve GUI navigation via SFT or large-scale reasoning data, but cross-domain generalization remains limited; GUI-Rise introduces structured CoT + RL to substantially address this.
- **GUI Memory Mechanisms**: Early approaches relied solely on action sequences (SeeClick); later work used screenshot windows (ShowUI, UI-Hawk), but suffered from information loss or high overhead; GUI-Rise's semantic summarization approach is more efficient.
- **RL for LLMs**: GRPO has been validated in code generation and mathematical reasoning; UI-R1 extends it to single-step GUI tasks; GUI-Rise is the first to apply GRPO with a history summarization reward to multi-step GUI navigation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The three-subtask framework and history summarization reward design exhibit high novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers Mind2Web/AITW/MiniWob/AndroidWorld/OSWorld with complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rigorous formal definitions.
- **Value**: ⭐⭐⭐⭐ — Substantial cross-domain generalization improvements with meaningful contributions to the GUI Agent field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices](../../ICCV2025/robotics/guiodyssey_a_comprehensive_dataset_for_cross-app_gui_navigation_on_mobile_device.md)
- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)
- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)

</div>

<!-- RELATED:END -->
