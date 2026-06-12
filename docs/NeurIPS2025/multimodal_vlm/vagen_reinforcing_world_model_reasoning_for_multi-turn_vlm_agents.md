---
title: >-
  [Paper Note] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM Agent] VAGEN is a framework that structures the reasoning process of VLM agents into StateEstimation and TransitionModeling to build an internal world model…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "VLM Agent"
  - "World Model"
  - "Reinforcement Learning"
  - "POMDP"
  - "Multi-Turn Interaction"
date: 2026-05-08
content_hash: f9b744057f1aa736
---

# VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents

**Conference**: NeurIPS 2025
**arXiv**: [2510.16907](https://arxiv.org/abs/2510.16907)  
**Code**: [http://mll.lab.northwestern.edu/VAGEN](http://mll.lab.northwestern.edu/VAGEN)  
**Area**: Multimodal VLM
**Keywords**: VLM Agent, World Model, Reinforcement Learning, POMDP, Multi-Turn Interaction

## TL;DR

VAGEN is a framework that structures the reasoning process of VLM agents into StateEstimation and TransitionModeling to build an internal world model, and combines WorldModeling Reward with Bi-Level GAE for efficient multi-turn RL training. A 3B model trained under this framework (0.82) surpasses GPT-5 (0.75) and Gemini 2.5 Pro (0.67).

## Background & Motivation

Multi-turn agent tasks require accurate interpretation and tracking of dynamic environments, a challenge that is further amplified when agents perceive the world visually rather than textually. VLM agent tasks are fundamentally **Partially Observable Markov Decision Processes (POMDPs)**: the visual observation $o_t$ received by the agent is only a partial view of the true state $s_t$, requiring the agent to first estimate the true world state before acting effectively.

**Core Problem**: Can VLM agents be trained to construct an internal world model through explicit visual state reasoning?

Limitations of prior work:
- Current VLM agents lack explicit internal world modeling to enhance visual state reasoning in multi-turn tasks.
- Existing RL frameworks (e.g., VLM-R1) are primarily optimized for single-turn settings and cannot capture evolving interaction contexts.
- Standard GAE methods suffer from instability in long-range credit assignment under sparse reward signals in multi-turn scenarios.
- No existing VLM handles multi-turn visual agent tasks well — even the strongest model, GPT-5, achieves only 0.75.

## Method

### Overall Architecture

Multi-turn VLM agent tasks are modeled as a POMDP $(\mathcal{S}, \mathcal{O}, \mathcal{A}, P, R, \Omega, \gamma)$. At each turn $t$, the agent generates an output $a_t = \langle z_t, a_t^e \rangle$ containing reasoning tokens and an executable action. The key innovation lies in structuring the reasoning tokens $z_t$ into two core components of a world model.

### Key Designs

1. **Systematic Comparison of Five Reasoning Strategies**: By controlling format rewards during RL training, five reasoning strategies ranging from implicit to explicit are studied:

    - **NoThink**: Generates only executable actions, $z_t = \emptyset$
    - **FreeThink**: Free-form natural language reasoning
    - **StateEstimation**: Explicitly describes the current state belief $\hat{s}_t$, learning $\hat{s}_t \to s_t$
    - **TransitionModeling**: Explicitly predicts the next state $\hat{s}_{t+1}$, learning $\hat{s}_{t+1} \to s_{t+1}$
    - **WorldModeling**: Combines both StateEstimation and TransitionModeling for a complete world model

   Conclusion: WorldModeling (0.76) > FreeThink (0.67) >> NoThink (0.28), demonstrating the critical importance of explicit visual state reasoning.

2. **Visual State Representation Selection**: Three types of internal belief representations are explored:

    - **Natural Language**: Optimal for general tasks; leverages pretrained semantic knowledge (Sokoban: 0.61, FrozenLake: 0.71)
    - **Structured Format**: Optimal for high-precision manipulation tasks; provides exact coordinates (PrimitiveSkill avg: 0.94)
    - **Symbolic Representation**: Worst-performing; models struggle to map abstract symbols to raw visual input

   Core Insight: **Representation selection is task-dependent** and there is no universal solution.

3. **WorldModeling Reward**: An LLM-as-a-Judge framework is used to evaluate how well the agent's state descriptions and predictions match the ground-truth states:
    $r_t^{reason} = \beta_s \cdot \mathbb{I}_{\text{StateEstimation}}(\hat{s}_t, s_t) + \beta_w \cdot \mathbb{I}_{\text{TransitionModeling}}(\hat{s}_{t+1}, s_{t+1})$
   This provides dense per-turn reward signals to compensate for the sparsity of task rewards.

4. **Bi-Level GAE**: Advantage estimation is computed at two levels to address the credit assignment problem in multi-turn scenarios:

    - **Turn-level**: Computes turn-level TD errors and advantages across turns using $\gamma_{\text{turn}}$:
    $\delta_t^{turn} = r_t + \gamma_{turn} V_\phi(\bar{\tau}_{\leq a_{t+1}}) - V_\phi(\bar{\tau}_{\leq a_t})$
    - **Token-level**: Propagates turn-level advantages to token level within each turn using $\gamma_{\text{token}}$
    - Key linkage: The computed turn-level advantage $A_t^{turn}$ initializes the advantage of the last token in that turn, enabling hierarchical propagation.

### Loss & Training

A PPO objective is used with observation token masking to prevent learning from observation tokens — a critical design, as the absence of masking causes training failure. The combined reward is $r_t = r_t^{reason} + r_t^{format} + R(s_t, a_t)$. Training uses Qwen2.5-VL-3B as the backbone with $\beta_s = \beta_w = 0.5$.

## Key Experimental Results

### Main Results

| Model / Method | Sokoban | FrozenLake | Navigation | PrimitiveSkill | SVG | Overall |
|----------------|---------|------------|------------|---------------|-----|---------|
| Qwen2.5-VL-3B (base) | 0.14 | 0.22 | 0.24 | 0.00 | 0.54 | **0.21** |
| GPT-5 | 0.70 | 0.75 | 0.78 | 0.66 | 0.85 | **0.75** |
| Claude 4.5 Sonnet | 0.31 | 0.67 | 0.67 | 0.53 | 0.88 | **0.64** |
| Gemini 2.5 Pro | 0.58 | 0.63 | 0.63 | 0.50 | 0.86 | **0.67** |
| VAGEN-Base (WorldModeling) | 0.61 | 0.78 | 0.79 | 0.91 | 0.78 | **0.76** |
| **VAGEN-Full** | **0.79** | **0.80** | **0.81** | **0.97** | **0.79** | **0.82** |

### Ablation Study

| Configuration | Sokoban | Navigation | PrimitiveSkill | Notes |
|---------------|---------|------------|---------------|-------|
| NoThink | 0.57/0.09 | 0.00 | 0.00 | Collapses without reasoning |
| FreeThink | 0.57/0.68 | 0.67 | 0.66 | Implicit reasoning is effective but insufficient |
| StateEstimation | 0.56/0.68 | 0.74 | 0.00 | Strong on navigation, poor on manipulation |
| TransitionModeling | 0.41/0.76 | 0.62 | 0.82 | Strong on manipulation, slightly weaker on navigation |
| WorldModeling | 0.61/0.71 | 0.79 | 0.91 | Most balanced |
| VAGEN-Full | **0.79/0.74** | **0.81** | **0.97** | Comprehensive gains from reward and credit assignment |

**Reasoning Strategies vs. RL Baselines**: Vanilla-PPO (0.26) << GRPO w/ Mask (0.54) < Turn-PPO (0.55) << VAGEN-Base (0.76) < VAGEN-Full (0.82)

### Key Findings

- A 3B model trained with VAGEN (0.82) surpasses all proprietary reasoning models (GPT-5: 0.75, o3: 0.73).
- StateEstimation excels at navigation (understanding current observations); TransitionModeling excels at manipulation (predicting future states); WorldModeling achieves the most balanced performance.
- Observation token masking is essential — its absence causes Vanilla-PPO to fail entirely.
- Bi-Level GAE and WorldModeling Reward each contribute inconsistent individual gains, but their combination yields the most stable overall improvement.

## Highlights & Insights

- Modeling VLM agents as POMDPs and explicitly constructing a world model constitutes an elegant theoretical framework.
- The systematic comparison of five reasoning strategies provides deep insights into "what VLM agents should think about."
- The result of a small model outperforming large proprietary models is highly compelling, demonstrating that the right training paradigm matters more than model scale.
- Bi-Level GAE addresses a critical pain point in multi-turn RL (credit assignment under sparse rewards) and holds general applicability.

## Limitations & Future Work

- The reward signal from LLM-as-a-Judge may lack precision, introducing the risk of reward hacking (response convergence and over-optimization are observed in the paper).
- Ground-truth state information must be provided by the environment to compute the WorldModeling Reward, limiting broader applicability.
- The SVG task has no world dynamics, so only Bi-Level GAE applies; the WorldModeling Reward is not applicable.
- Validation in open-world environments has not been conducted — all tasks are structured.

## Related Work & Insights

- **vs. VLM-R1**: VLM-R1 is a single-turn RL method with no advantage on multi-turn agent tasks; VAGEN optimizes over full interaction trajectories.
- **vs. DeepSeek-R1**: VAGEN draws on R1's format reward design but extends it to multi-turn settings and incorporates world model reasoning.
- **vs. GiGPO/AReaL**: Concurrent works focus on long-horizon credit assignment and asynchronous scaling, but do not explore explicit world model reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of POMDP perspective, world model reasoning, and Bi-Level GAE is unique; the systematic study of five strategies is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five tasks, five reasoning strategies, three representations, multiple RL baselines, and comparisons with proprietary models — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with progressively developed research questions and information-rich tables.
- Value: ⭐⭐⭐⭐⭐ The result of a 3B model surpassing GPT-5 is impressive and provides a systematic guide for VLM agent training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[CVPR 2026\] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](../../CVPR2026/multimodal_vlm/rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)
- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](../../CVPR2026/multimodal_vlm/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)

</div>

<!-- RELATED:END -->
