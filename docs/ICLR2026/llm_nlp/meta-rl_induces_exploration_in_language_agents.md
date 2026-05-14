---
title: >-
  [Paper Note] Meta-RL Induces Exploration in Language Agents
description: >-
  [ICLR 2026][LLM/NLP][Meta-RL] This paper proposes LaMer, a framework that introduces Meta-Reinforcement Learning (Meta-RL) into LLM agent training. By optimizing rewards across episodes and enabling context-based policy…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "Meta-RL"
  - "LLM Agent"
  - "Exploration & Exploitation"
  - "Multi-turn Interaction"
  - "Cross-episode Training"
  - "Self-reflection"
date: 2026-05-08
content_hash: e787c13fa0f3df7e
---

# Meta-RL Induces Exploration in Language Agents

**Conference**: ICLR 2026
**arXiv**: [2512.16848](https://arxiv.org/abs/2512.16848)
**Code**: [mlbio-epfl/LaMer](https://github.com/mlbio-epfl/LaMer)
**Area**: LLM/NLP
**Keywords**: Meta-RL, LLM Agent, Exploration & Exploitation, Multi-turn Interaction, Cross-episode Training, Self-reflection

## TL;DR

This paper proposes LaMer, a framework that introduces Meta-Reinforcement Learning (Meta-RL) into LLM agent training. By optimizing rewards across episodes and enabling context-based policy adaptation via self-reflection, LaMer equips language agents with active exploration capabilities, achieving absolute performance gains of 11%, 14%, and 19% on Sokoban, MineSweeper, and Webshop, respectively.

## Background & Motivation

### State of the Field

LLMs have progressively transitioned from dialogue systems toward decision-making agents (e.g., ReAct, Reflexion), capable of interacting with environments through multi-turn text observation–action loops. However, existing RL-trained LLM agents suffer from a fundamental limitation: **the absence of active exploration**. In tasks requiring trial-and-error learning, agents tend to converge prematurely to suboptimal policies, failing to systematically explore and adapt to new environments as humans do.

### Limitations of Prior Work

**Prompting-based methods** (Zero-shot, ReAct, Reflexion): rely on frozen LLMs, exhibit limited exploratory behavior, and have low performance ceilings.

**Standard RL training** (PPO, GRPO, GiGPO): each episode is sampled independently with a fixed policy, precluding test-time adaptation through trial and error.

**Offline distillation methods**: depend on offline data, enabling imitation rather than active exploration; predominantly focus on single-turn reasoning rather than multi-turn agent tasks.

### Core Idea

Multi-turn tasks typically yield sparse success signals only at the end of an episode. By treating **multiple episodes as a single trial**, the exploration–exploitation trade-off naturally becomes a **cross-episode RL problem**—precisely the framework of Meta-RL. Training across multiple distinct but related environments compels the agent to learn generalizable exploration strategies.

## Method

### Overall Architecture

LaMer (LLM Agent with Meta-RL) consists of two core components:

1. **Cross-episode training framework**: encourages the agent to explore in early episodes and exploit accumulated experience in subsequent ones.
2. **Context-based policy adaptation via self-reflection**: enables policy adaptation through textual reflection in context, without gradient updates.

### Cross-episode Training

Each trial consists of $N$ sequentially ordered episodes:

$$\mathcal{T} = (\tau^{(0)}, \tau^{(1)}, \dots, \tau^{(N-1)})$$

where the policy at each episode is conditioned on the history accumulated from preceding episodes. The key contribution is the definition of a **cross-episode discounted return**:

$$G_t^{(n)} = \underbrace{g_t^{(n)}}_{\text{intra-episode}} + \underbrace{\sum_{m=n+1}^{N-1} \gamma_{\text{traj}}^{m-n} g_0^{(m)}}_{\text{cross-episode}}$$

where $g_t^{(n)} = \sum_{l=t}^{T-1} \gamma_{\text{step}}^{l-t} r_l^{(n)}$ denotes the intra-episode return, and $\gamma_{\text{traj}}$ is the cross-episode discount factor.

The Meta-RL optimization objective is:

$$J(\theta) = \mathbb{E}_{\mathcal{T} \sim \pi_\theta} \left[ \sum_{n=0}^{N-1} \gamma_{\text{traj}}^n \sum_{t=0}^{T-1} \gamma_{\text{step}}^t r_t^{(n)} \right]$$

The parameter $\gamma_{\text{traj}}$ controls the exploration–exploitation trade-off: smaller values favor rapid exploitation, while larger values encourage long-horizon exploration.

### Context-based Policy Adaptation (Self-reflection)

At the end of each episode, the agent generates a textual reflection summarizing prior experience:

$$\pi_\theta^{(n)}(\cdot) = \pi_\theta(\cdot | \mathcal{H}^{(n)})$$

where $\mathcal{H}^{(n)}$ is the inter-episode memory containing historical trajectories and reflections. The reflection step itself is trained using rewards obtained from the subsequent episode.

### Key Difference from Standard RL

- **Standard RL**: independently samples a set of episodes per task and computes gradients independently.
- **Meta-RL (LaMer)**: episodes within the same trial are generated sequentially, with each episode conditioned on preceding ones.

### Loss & Training

Gradient estimation:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mathcal{T}} \left[ \sum_{n=0}^{N-1} \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(n)} | s_t^{(n)}, \mathcal{H}^{(n)}) A_t^{(n)} \right]$$

The framework is compatible with mainstream optimizers including PPO, GRPO, and GiGPO; GiGPO is used by default.

## Key Experimental Results

### Main Results

Base model: Qwen3-4B; $N=3$ episodes; group size = 8 (RL baseline uses group size = 24 for fairness).

| Method | Sokoban p@1/p@2/p@3 | MineSweeper p@1/p@2/p@3 | Webshop p@1/p@2/p@3 |
|--------|---------------------|--------------------------|---------------------|
| Zero-shot | 6.8/9.8/12.9 | 4.5/6.6/8.6 | 1.4/2.1/2.3 |
| ReAct | 7.2/9.6/12.5 | 6.3/7.0/10.9 | 3.1/4.5/4.5 |
| Reflexion | 6.4/9.8/12.1 | 5.5/7.2/9.8 | 2.7/3.3/3.5 |
| PPO | 12.5/15.4/16.8 | 29.7/34.2/35.5 | 53.1/54.5/54.9 |
| GiGPO | 41.6/43.6/44.1 | 52.0/54.9/55.1 | 73.4/74.6/75.2 |
| **LaMer** | **42.4/52.0/55.9** | **44.1/66.4/74.4** | **67.8/84.4/89.1** |

LaMer outperforms all baselines on p@3: +11.8% on Sokoban, +19.3% on MineSweeper, and +13.9% on Webshop.

### OOD Generalization (ALFWorld)

| Method | Pick(i.d.) | Look(i.d.) | Clean(i.d.) | Heat(i.d.) | Cool(o.o.d.) | Pick2(o.o.d.) |
|--------|-----------|-----------|------------|-----------|-------------|--------------|
| Prompting | 91.9 | 52.9 | 48.4 | 44.8 | 42.8 | 21.2 |
| RL | 95.5 | 83.0 | 67.9 | 86.6 | 58.1 | 36.0 |
| **Meta-RL** | **97.7** | **100.0** | **90.2** | **89.5** | **81.0** | **50.2** |

On OOD tasks, LaMer surpasses RL by 23% (Cool) and 14% (Pick2).

### Ablation Study

**Memory configuration ablation** (p@3):

| Memory Content | Sokoban | MineSweeper | Webshop |
|----------------|---------|-------------|---------|
| Trajectory only | 34.8 | 69.5 | 89.3 |
| Reflection only | **56.4** | **80.5** | **92.8** |
| Both | 55.9 | 74.4 | 89.1 |

Reflection yields substantial gains; reflection-only even outperforms the default setting, as reflections are more concise and focused.

**Effect of $\gamma_{\text{traj}}$**:
- Optimal $\gamma_{\text{traj}}=0.6$ for Sokoban/Webshop (requires balancing immediate and long-term returns).
- Optimal $\gamma_{\text{traj}}=0.9$ for MineSweeper (requires more strategic exploration).

### Key Findings

1. Meta-RL maintains higher trajectory diversity (measured by entropy of the empirical distribution), achieving a better exploration–exploitation balance.
2. On harder task instances (more boxes/mines), Meta-RL consistently leads RL by 5–10%.
3. Meta-RL exhibits superior test-time scaling: the gain from p@1 to p@3 is far larger than that of RL (Sokoban: 13.5% vs. <5%).

## Highlights & Insights

1. **First application of Meta-RL to LLM agent training**: adapts the cross-task generalization paradigm of classical Meta-RL to multi-episode LLM interaction.
2. **Elegant formalization**: $\gamma_{\text{traj}}$ provides a simple and interpretable knob for controlling exploration–exploitation.
3. **Dual role of self-reflection**: serves simultaneously as an adaptation mechanism and a training signal, with ablations confirming its critical contribution.
4. **New perspective on test-time scaling**: Meta-RL can be viewed as amortizing test-time computation through multi-episode training.
5. **No additional training data required**: uses the same number of trajectories as standard RL, differing only in how those trajectories are organized.

## Limitations & Future Work

1. **Training time is approximately 2× that of RL**: episodes within a trial must be generated sequentially, limiting parallelism.
2. **Only one base model evaluated** (Qwen3-4B): effectiveness on larger models remains to be verified.
3. **Limited environment types**: primarily text-format game and web environments; complex real-world agent tasks remain unexplored.
4. **Context length constraints**: multi-episode histories and reflections rapidly saturate the context window.

## Related Work & Insights

- **Reflexion** (Shinn et al., 2023): employs multi-episode interaction with reflection but uses a frozen LLM without training.
- **GiGPO** (Feng et al., 2025): the strongest single-episode RL baseline; LaMer extends it to the multi-episode setting.
- **Test-time compute scaling**: LaMer provides a training-based approach to improve test-time scaling behavior.
- **Inspiration**: the framework can be combined with stronger reasoning models (e.g., the R1 series) to explore synergies between reasoning and exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First adaptation of Meta-RL to LLM agents with an elegant formalization.
- **Technical Depth**: ⭐⭐⭐⭐ — Cross-episode reward propagation mechanism is well-designed with clear theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four environments + OOD generalization + difficulty generalization + detailed ablations.
- **Value**: ⭐⭐⭐⭐ — General-purpose framework compatible with mainstream RL algorithms.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A solid contribution that opens a new direction for training exploratory LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](../../NeurIPS2025/llm_nlp/system_prompt_optimization_with_meta-learning.md)
- [\[AAAI 2026\] Blue Teaming Function-Calling Agents](../../AAAI2026/llm_nlp/blue_teaming_function-calling_agents.md)
- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [\[AAAI 2026\] Scaling Equitable Reflection Assessment in Education via Large Language Models and Role-Based Feedback Agents](../../AAAI2026/llm_nlp/scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](../../AAAI2026/llm_nlp/scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)

</div>

<!-- RELATED:END -->
