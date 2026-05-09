---
title: >-
  [Paper Note] Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL
description: >-
  [NeurIPS 2025][LLM/NLP][Offline Reinforcement Learning] This paper proposes PNLC, a method that trains a lightweight goal-conditioned value function as a "natural language critic" to guide LLM agents in multi-turn planning and self-refinement at the thought-step level. Without direct fine-tuning or inference-time search, PNLC significantly outperforms existing methods on complex interactive tasks such as web navigation, social reasoning, and persuasion, while achieving 8–10× faster inference.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Offline Reinforcement Learning
  - Goal-Conditioned Value Function
  - LLM Agent Planning
  - Natural Language Critic
  - Multi-Turn Interactive Tasks
date: 2026-05-08
content_hash: 59826b567732f3d4
---

# Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL

**Conference**: NeurIPS 2025
**arXiv**: [2505.18098](https://arxiv.org/abs/2505.18098)
**Code**: [Project Page](https://jxihong.github.io/pnlc_website/)
**Area**: NLP Understanding / LLM Agent Planning
**Keywords**: Offline Reinforcement Learning, Goal-Conditioned Value Function, LLM Agent Planning, Natural Language Critic, Multi-Turn Interactive Tasks

## TL;DR
This paper proposes PNLC, a method that trains a lightweight goal-conditioned value function as a "natural language critic" to guide LLM agents in multi-turn planning and self-refinement at the thought-step level. Without direct fine-tuning or inference-time search, PNLC significantly outperforms existing methods on complex interactive tasks such as web navigation, social reasoning, and persuasion, while achieving 8–10× faster inference.

## Background & Motivation

**State of the Field**: LLMs require long-horizon reasoning and strategic behavior for goal-oriented complex interactive tasks (e.g., negotiation, persuasion, social reasoning games). Existing approaches fall into two categories: (a) multi-turn RL fine-tuning — sample-inefficient and computationally expensive; (b) inference-time search (e.g., MCTS) — requires multiple LLM calls with high latency.

**Limitations of Prior Work**: RL fine-tuning cannot be applied to frontier models exposed only via API (e.g., GPT-4o); MCTS search requires ~46 seconds per sample on WebShop; LLM self-evaluation tends to be overly optimistic, making effective self-refinement difficult.

**Root Cause**: How can LLM agents be endowed with long-horizon planning capabilities for complex interactive tasks without directly fine-tuning the LLM or substantially increasing inference cost?

**Paper Goals**: A lightweight, learnable module is needed that provides value estimates over multiple possible outcomes during LLM inference, enabling effective self-refinement.

**Starting Point**: Rather than training a policy, train a critic. An offline RL approach is used to train a goal-conditioned value function, which is then deployed at inference time as a "natural language critic" supplying rich outcome evaluation to the LLM.

**Core Idea**: Train a lightweight MLP value function at the thought-step level to predict goal-achievement probability. At inference time, a natural language critic generates multiple positive/negative goals with associated probabilities to guide iterative self-refinement of high-level strategies — without any search.

## Method

### Overall Architecture
PNLC consists of two phases: offline training and inference-time planning. In the offline phase, trajectory data is processed via summarization → embedding → training the goal-conditioned value function. At inference time: current state + proposed thought → critic generates goals + values → LLM self-refines.

Formally, the MDP is defined as $M=(\mathcal{S}, \mathcal{A}, P, r, \rho, \gamma)$. Agent actions $a_t$ are decomposed into a thought $a_t^{\text{tht}}$ and an environment action $a_t^{\text{env}}$. The goal-conditioned Q-value function $Q(s, a^{\text{tht}}, g)$ predicts the probability of achieving goal $g$ from state $s$ after taking thought $a_t^{\text{tht}}$.

### Key Designs

1. **Offline Goal-Conditioned Value Function Training**:

    - Function: Learns a goal-conditioned value function from task-relevant trajectory datasets.
    - Mechanism: (a) Trajectory summarization — compresses full interaction histories into concise, decision-relevant descriptions; (b) Embedding — converts text to low-dimensional vectors using GPT-3 embeddings; (c) Goal sampling — randomly samples future states from each trajectory as goals. Training follows the IQL algorithm with loss: $L_Q = \mathbb{E}[(r(s,g) + \gamma\hat{V}(s',g) - Q(s,a^{\text{tht}},g))^2]$
    - Design Motivation: Trajectory summarization reduces decision-space complexity; embeddings allow the value function to use only a 2-layer MLP (<1M parameters); random goal sampling enables multi-dimensional evaluation.

2. **Inference-Time Natural Language Critic**:

    - Function: Generates natural language feedback about possible outcomes for the LLM.
    - Mechanism: (a) The LLM generates 4 hypothetical goals (2 positive + 2 negative); (b) the value function estimates the achievement probability for each goal; (c) results are converted to natural language descriptions (e.g., "70% probability the user will agree, 30% risk of rejection"); (d) the LLM iteratively refines its thought based on the feedback (up to 2 rounds).
    - Design Motivation: Goal-conditioned value functions provide multi-dimensional feedback that is more informative than scalar values; positive/negative goals help the LLM identify risks.

3. **Lightweight MLP Value Function**:

    - Function: Supports fast training and inference via a minimal architecture.
    - Mechanism: Input is the concatenation of state, thought, and goal embeddings; a 2-layer fully connected network (128×128) outputs a scalar probability.
    - Design Motivation: A Transformer of LLM scale cannot serve as a value function for API-only models; a <1M-parameter MLP is sufficiently expressive given that embeddings already encode semantic information.

### Loss & Training
A goal-conditioned variant of IQL (Implicit Q-Learning) is used. The Q-function is trained with MSE loss; the V-function uses expectile regression ($\tau=0.8$). Only 2.5k low-quality trajectories (generated by GPT-3.5) are required to train an effective critic.

## Key Experimental Results

### Main Results

| Method | WebShop Score | Avalon Win Rate | Persuasion Donation | Inference Time |
|--------|--------------|-----------------|---------------------|----------------|
| ReAct | 55.1 | 21.0% | 0.54 | 5s |
| Reflexion | 60.8 | 26.0% | 0.54 | ~15s |
| LATS (n=30) | 74.9 | 38.0% | 0.78 | ~46s |
| Agent Q (n=30) | 77.1 | — | — | ~46s |
| Online ArCHer | 62.3 | 19.0% | 0.36 | — |
| **PNLC (Ours)** | **78.2** | **47.0%** | **0.87** | **5–6s** |

### Ablation Study

| Configuration | WebShop | Avalon | Persuasion |
|---------------|---------|--------|------------|
| PNLC (full) | 78.2 | 47.0% | 0.87 |
| w/o goal conditioning (scalar value only) | 55.4 | 25.0% | 0.53 |
| w/o refinement step | 55.6 | 28.0% | 0.61 |
| ReAct+Replan (LLM self-evaluation) | 59.1 | 22.0% | 0.62 |

### Key Findings
- **PNLC achieves SOTA across all three diverse tasks** with only 5–6s inference time, approximately 8× faster than LATS (n=30).
- **Goal conditioning is critical**: removing it reduces performance to be indistinguishable from ReAct (55.4 vs. 55.1), demonstrating that multi-dimensional goal feedback is essential.
- **Offline learning outperforms LLM intuition**: allowing the LLM to self-assess goal probabilities (ReAct+Replan) yields substantially lower performance than the data-driven critic (59.1 vs. 78.2), confirming that LLMs are overly optimistic about goal reachability in long-horizon tasks.
- **RL fine-tuning performs worst**: Online ArCHer, which fine-tunes a smaller model, underperforms all other methods.

## Highlights & Insights
- **"Train the critic, not the policy" paradigm**: This elegantly circumvents the limitation of non-fine-tunable API models by shifting the learning burden to a lightweight module, offering substantial practical deployment value.
- **Abstraction at the thought level**: Learning the value function at the level of thoughts (high-level strategic intentions) rather than actions (raw text) substantially reduces decision-space complexity.
- **Interpretable feedback via goal-conditioned value functions**: Natural language descriptions of multiple positive/negative goals with associated probabilities are more amenable to LLM comprehension and utilization than scalar values.

## Limitations & Future Work
- **Task-specific value functions**: A separate value function must be trained for each new task; cross-task transfer remains an open problem.
- **Reliance on LLM goal generation and refinement**: The approach may fail in specialized domains beyond the LLM's knowledge.
- **No data sensitivity analysis**: The minimum quantity and quality of trajectories required to train an effective critic remain uncharacterized.
- **Value function calibration**: Whether the probability estimates are reliable has not been analyzed.

## Related Work & Insights
- **vs. RL fine-tuning (ArCHer)**: The proposed method requires no LLM parameter updates, supports API-only models, and incurs orders-of-magnitude lower training cost.
- **vs. inference-time search (LATS/MCTS)**: Inference time scales as a constant rather than exponentially with search depth, yielding an enormous practical advantage in deployment.
- **vs. self-refinement (Reflexion)**: Reflexion requires multiple full trajectory rollouts, whereas PNLC needs only a single refinement step with a lightweight critic.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Pioneering "planning without search" paradigm combining critic learning with LLM planning; goal-conditioned value functions at the thought level are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across three diverse tasks with detailed ablations, though data sensitivity analysis is absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic and intuitive figures.
- Value: ⭐⭐⭐⭐⭐ — Compatible with any API-accessible LLM; represents a breakthrough in inference efficiency with high practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[NeurIPS 2025\] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)
- [\[NeurIPS 2025\] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model](streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)
- [\[NeurIPS 2025\] EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)

<!-- RELATED:END -->
