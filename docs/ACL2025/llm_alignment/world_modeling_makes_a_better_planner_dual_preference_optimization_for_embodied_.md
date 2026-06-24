---
title: >-
  [Paper Note] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning
description: >-
  [ACL 2025][LLM Alignment][Embodied Task Planning] A Dual Preference Optimization ($D^2PO$) framework is proposed. By jointly optimizing preference learning for the dual objectives of state prediction (world modeling) and action selection, vision-language models simultaneously learn to "understand world dynamics" and "make better decisions" during embodied task planning. This allows a 7B model to significantly outperform GPT-4o.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Embodied Task Planning"
  - "World Models"
  - "Preference Optimization"
  - "DPO"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 1e1af6031d60579c
---

# World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning

**Conference**: ACL 2025  
**arXiv**: [2503.10480](https://arxiv.org/abs/2503.10480)  
**Code**: None  
**Area**: Embodied AI / LLM Alignment  
**Keywords**: Embodied Task Planning, World Models, Preference Optimization, DPO, Vision-Language Models

## TL;DR

A Dual Preference Optimization ($D^2PO$) framework is proposed. By jointly optimizing preference learning for the dual objectives of state prediction (world modeling) and action selection, vision-language models simultaneously learn to "understand world dynamics" and "make better decisions" during embodied task planning. This allows a 7B model to significantly outperform GPT-4o.

## Background & Motivation

Embodied task planning requires AI systems to execute real-world tasks through physical interactions, establishing strict requirements for both correctness and efficiency. Existing approaches face the following challenges:

1. **Inherent Limitations of LVLMs**: LVLMs operate based solely on static snapshots of the environment, lacking the ability to model the dynamic nature of physical interactions. This leads to dependency constraint violations (e.g., placing an object before picking it up) and inefficient planning (e.g., repeating unnecessary steps).
2. **Limitations of Prior Training Methods**:
    - Prompt-based methods are constrained by the intrinsic capabilities of the models.
    - SFT learns only from successful trajectories while ignoring failure experiences.
    - Existing RL methods require designing reward functions or training reward models.
3. **Underutilization of World Models**: Some methods employ LLMs as world models to guide search during inference, which introduces additional computational overhead and fails to cultivate world-modeling capabilities during training.

Humans possess internal world models, building their understanding and prediction of the external world through continuous interaction with the environment. **Core Idea**: Can models learn world modeling during the training stage itself, thereby eliminating the need for an external world model during inference?

## Method

### Overall Architecture

$D^2PO$ consists of two core modules:
1. **Data Exploration**: Automatically collecting trajectories and preference data in simulated environments via Step-wise Tree Search.
2. **Dual Preference Optimization**: Jointly optimizing two DPO objectives: action selection and state prediction.

Overall process: Step-wise tree search for environment exploration to collect positive and negative samples $\rightarrow$ SFT warm-up $\rightarrow$ $D^2PO$ preference optimization.

### Key Designs

1. **Step-wise Tree Search Data Exploration**: Automatically collecting training data in simulated environments without manual annotation or expert demonstrations. It comprises three components: ① Action sampling and evaluation—sampling $K$ actions at each state and evaluating them using a hybrid scoring mechanism (50% GPT-4o process reward score + 50% environmental executability binary score); ② Iterative tree expansion—a breadth-first strategy selecting high-scoring actions ($\ge$ threshold $\tau$) for expansion; ③ Trajectory validation and backtracking—backtracking to extract trajectories after reaching the target, constructing preference pairs for both action selection and state prediction.

2. **Dual Preference Optimization ($D^2PO$)**: Extends DPO into two simultaneously optimized targets:
    - **Action Selection Optimization**: Given the goal and historical context, optimizing the probability of selecting the correct action-reasoning pair (chosen) over the incorrect pair (rejected).
    - **State Prediction Optimization**: Given the current state and action, optimizing the probability of predicting the correct next-state description (expressed in natural language, including object attributes, spatial relations, agent states, etc.).
   
   **Key Insight**: The state prediction objective is not just an auxiliary task but enhances planning capability by forcing the model to understand the consequences of actions. No explicit world model prediction is required during inference.

3. **VoTa-Bench Vision Benchmark**: Extends the text-only LoTa-Bench by introducing first-person visual observations as inputs and feedback. It utilizes an open-domain generation evaluation (allowing the model to generate non-executable skills) and adds 646 unseen environment samples to test generalization capability. In total, it comprises 549 seen + 646 unseen test samples, covering 108 objects and 120 scenes.

### Loss & Training

Total loss function: 
$$\mathcal{L}_{total} = \mathcal{L}_{action}(\pi_\theta; \pi_{ref}) + \lambda \mathcal{L}_{state}(\pi_\theta; \pi_{ref})$$

where $\lambda = 1$ balances the two objectives. Training strategy:
- SFT phase: Full-parameter fine-tuning for 3 epochs, learning rate 3e-5, batch size 32.
- $D^2PO$ phase: 1 epoch, learning rate 5e-7, batch size 32.
- Training data: 4.5k SFT samples + 15k DPO samples.
- States use image inputs and output textual descriptions.
- Evaluation: Maximum of 25 steps, temperature 0.

## Key Experimental Results

### Main Results

VoTa-Bench (Seen) overall performance (%SR / %PL):

| Model | Method | SR↑ | PL↑ |
|------|------|-----|-----|
| GPT-4o | zero-shot | 14.39 | 10.37 |
| GPT-4o | + ICL | 23.50 | 18.78 |
| Qwen2-VL-72B | zero-shot | 11.66 | 7.10 |
| Qwen2-VL-7B | + SFT | 44.63 | 40.33 |
| Qwen2-VL-7B | + DPO | 53.92 | 49.37 |
| **Qwen2-VL-7B** | **+ $D^2PO$** | **58.11** | **53.33** |
| LLaVA-1.6-7B | + SFT | 41.35 | 37.56 |
| LLaVA-1.6-7B | + DPO | 49.54 | 44.38 |
| **LLaVA-1.6-7B** | **+ $D^2PO$** | **54.83** | **50.23** |
| LLaMA-3.2-11B | + SFT | 42.99 | 35.33 |
| LLaMA-3.2-11B | + DPO | 46.08 | 39.73 |
| **LLaMA-3.2-11B** | **+ $D^2PO$** | **51.18** | **44.84** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| SFT $\rightarrow$ DPO (Action Only) | SR +15.95% (Relative) | Learning from failure experiences is valuable |
| SFT $\rightarrow$ $D^2PO$ (Action + State) | SR +27.29% (Relative) | World modeling provides further improvement |
| $D^2PO$ vs DPO | SR +9.84%, PL +11.35% (Average) | Incremental contribution of the state prediction objective |
| Qwen2-VL-7B $D^2PO$ vs GPT-4o | 58.11% vs 14.39% | 7B model outperforms GPT-4o by 43+ points |

### Key Findings

- **World Modeling Significantly Enhances Planning Capability**: $D^2PO$ improves SR by an average of 9.84% compared to DPO, validating the core hypothesis.
- **Learning from Mistakes**: DPO/$D^2PO$ leverages failure trajectories (whereas SFT only uses successful ones), simulating the human mechanism of "learning from mistakes".
- **Outperforming Process Reward Models**: The 7B $D^2PO$ model significantly outperforms GPT-4o (acting as a process reward model), demonstrating that direct environmental interaction feedback is more effective than LLM scoring.
- **Physics-Aware Efficient Planning**: The improvement in the PL metric indicates that the model has developed physical awareness, resulting in more efficient planning paths.
- **Generalization to Unseen Environments**: The model exhibits strong performance in unseen scenarios as well, proving that world modeling assists in generalization.

## Highlights & Insights

- The idea of enhancing "making better decisions" by "learning to imagine the consequences of actions" is highly intuitive and effective.
- Representing world dynamics in natural language is a smart design choice, directly leveraging the prior knowledge of LLMs.
- The tree-search data collection method is fully automated and requires no human annotation, offering excellent scalability.
- Free of additional world model inference during test time, the world-modeling objective during training is successfully internalized into the planning capability of the model.

## Limitations & Future Work

- Validated only in the AI2-THOR simulation environment; not yet extended to the real physical world.
- State descriptions are in textual form, which might lose fine-grained visual details.
- Data collection relies on GPT-4o as a process reward evaluator.
- The choice of $\lambda = 1$ has not been fully ablated (is there a more optimal balance?).
- Direct comparison with online RL methods such as PPO/GRPO is lacking.

## Related Work & Insights

- ETO (Song et al., 2024) applies DPO to LLM-based embodied planning but only optimizes actions, whereas this work adds the state prediction dimension.
- It shares a conceptual lineage with the Dreamer series (Hafner et al.), but represents state transitions in natural language rather than inside a latent space.
- **Insight**: Auxiliary objectives (such as predicting environmental changes) can effectively enhance the performance of primary tasks (planning); this paradigm can be transferred to other decision-making scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — Jointly optimizing the world model and policy using dual DPO is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation involving three models, multiple task types, and both seen/unseen scenarios.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and well-explained framework.
- Value: ⭐⭐⭐⭐ — The 7B model outperforms GPT-4o, showing practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization](atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)

</div>

<!-- RELATED:END -->
