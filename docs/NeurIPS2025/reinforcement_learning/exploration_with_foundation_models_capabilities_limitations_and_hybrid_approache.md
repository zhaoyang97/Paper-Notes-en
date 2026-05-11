---
title: >-
  [Paper Note] Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches
description: >-
  [NeurIPS 2025][Reinforcement Learning][foundation models] This paper systematically evaluates the zero-shot exploration capabilities of LLMs/VLMs on classic RL exploration tasks (bandits, Gridworld, Atari)…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "foundation models"
  - "exploration"
  - "reinforcement-learning"
  - "VLM"
  - "knowing-doing gap"
date: 2026-05-08
content_hash: e2363cb94ba0a656
---

# Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches

**Conference**: NeurIPS 2025
**arXiv**: [2509.19924](https://arxiv.org/abs/2509.19924)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: foundation models, exploration, reinforcement-learning, VLM, knowing-doing gap

## TL;DR

This paper systematically evaluates the zero-shot exploration capabilities of LLMs/VLMs on classic RL exploration tasks (bandits, Gridworld, Atari), identifies a *knowing-doing gap* in VLMs — where high-level reasoning succeeds but low-level control fails — and proposes a simple VLM-RL hybrid framework that substantially accelerates learning under idealized conditions.

## Background & Motivation

Exploration under sparse rewards remains a fundamental challenge in RL. Foundation models (LLMs/VLMs) possess strong semantic priors and reasoning capabilities; whether these can be leveraged to improve exploration efficiency is an open question.

Limitations of prior work:

**Narrow evaluation scope**: MAB experiments focus only on complex prompt engineering, without studying the effect of simple instruction phrasing.

**Incomplete environment hierarchy**: No systematic, progressive evaluation spanning simple (bandit) to complex (Atari) settings.

**Unclear failure modes**: Why do VLMs fail in visual environments — is it a comprehension problem or an execution problem?

This paper addresses these questions through a **three-level progressive evaluation** (MAB → Gridworld → Atari) and reveals the root causes of failure via qualitative analysis.

## Method

### Overall Architecture

A three-tier evaluation framework:
1. **Multi-Armed Bandit** (isolating the exploration–exploitation tradeoff): compares the effect of implicit vs. explicit prompts on LLM exploration behavior.
2. **Gridworld** (introducing state transitions and memory requirements): tests LLM spatial navigation in deterministic/stochastic environments.
3. **Atari hard-exploration games** (high-dimensional visual input + sparse rewards): evaluates GPT-4o zero-shot gameplay.
4. **Hybrid framework**: a VLM periodically takes over control from a PPO agent to serve as a semantic exploration guide.

### Key Designs

**Prompt design ablation**:
- Implicit (v1): *"Your goal is to maximize the total reward by pulling the arm with the highest probability"* → requires the LLM to infer the need for exploration.
- Explicit (v2): *"Your goal is to maximize the total reward by finding out which arm has the highest probability"* → directly instructs exploration.

**Temporal information in Atari**: A frame skip of $m=6$ steps (rather than consecutive 4 frames) is introduced to increase temporal diversity and help the VLM infer motion direction. A unified minimal prompt is used across all games.

**Hybrid algorithm**:
- The PPO agent relinquishes control to the VLM with probability $\epsilon$ for $T$ steps.
- The VLM acts as a "semantic explorer," steering the agent toward promising state regions.
- PPO resumes standard on-policy learning from the new states.

### Loss & Training

- The hybrid framework uses the standard PPO loss.
- The VLM operates zero-shot without any training.
- Baselines: PPO + RND (Random Network Distillation) as a strong exploration baseline.
- Evaluation metrics: cumulative reward, regret, and learning curves.

## Key Experimental Results

### MAB Experiments

| Model | Implicit prompt (v1) | Explicit prompt (v2) | UCB | Thompson Sampling |
|------|-----------------|-----------------|-----|-------------------|
| GPT-3.5 | High regret | Moderate regret | Low regret | Low regret |
| GPT-4 | Moderate regret | **Near-optimal** | Low regret | Low regret |
| Gemini 1.0 | High regret | Moderate regret | — | — |
| Gemini 1.5 | Moderate regret | Moderate-low | — | — |

Suboptimality gap analysis (GPT-4, explicit prompt):

| $\Delta$ | GPT-4 vs. UCB/TS |
|----------|-------------------|
| 0.6 | Competitive |
| 0.4 | Competitive |
| 0.2 | **Clearly inferior** |

### Atari Zero-Shot Experiments

| Game | GPT-4o | RB 250K | RB 2.5M | RB 25M | Human |
|------|--------|---------|---------|--------|------|
| Freeway | **21** | 8 | 32 | 32 | 29.6 |
| Gravitar | **500** | 64 | 199 | 2405 | 3351 |
| Montezuma | **0** | 0 | 50 | 544 | 4753 |
| Pitfall | -158 | -26 | -7 | -7 | 6464 |
| Private Eye | -1000 | 503 | 125 | 1573 | 69571 |
| Solaris | **600** | 681 | 1137 | 2093 | 12326 |
| Venture | 0 | 8 | 20 | 1513 | 1188 |

### Gridworld Results

| Setting | Action Only | Simple Plan | Focused Plan | PPO/RecPPO |
|------|------------|-------------|--------------|-------------|
| Deterministic | LLM performs well | LLM excels | LLM excels | Slow convergence |
| Stochastic (partial obs.) | **Severe degradation** | Some improvement | Some improvement | Eventually converges |

### Hybrid Framework Results (Freeway)

| Method | Score after 100K steps | Convergence Speed |
|------|-------------|---------|
| Vanilla PPO | ~5 | Slow |
| PPO + RND | ~15 | Moderate |
| **PPO + VLM** | **~25** | **Fast** |

### Key Findings

1. **Explicit prompts substantially improve exploration**: LLMs do not infer the need to explore on their own and require explicit instruction.
2. **Knowing-doing gap**: VLMs correctly identify "move upward" in Freeway and recognize enemies to fire at in Gravitar (+250 points), yet completely fail in games requiring precise temporal control.
3. **Failure mode taxonomy**:
   - **Precise control failure**: Montezuma (correct reasoning of "get the key" but unable to execute the jump).
   - **Self-identification failure**: Venture (unable to identify the pink square as the player character).
   - **Temporal reasoning failure**: Pitfall (understands "jump over the pit" but misjudges timing).
4. **Hybrid framework is effective under idealized conditions**: On Freeway, where the VLM strategy is correct and control is simple, the hybrid approach substantially outperforms PPO+RND.

## Highlights & Insights

- **Precise characterization of the knowing-doing gap**: The failure is not that VLMs misunderstand the game, but that they cannot translate understanding into precise low-level actions — a fundamental bottleneck for current VLMs as autonomous agents.
- **Progressive evaluation design**: The MAB → Gridworld → Atari progression systematically exposes the capability boundaries of foundation models at each level.
- **Honest experimental design**: The hybrid framework is validated on Freeway — a game where the VLM is already known to perform well — and is explicitly presented as an upper-bound analysis rather than a general solution.
- **Practical implication**: Foundation models are better suited as "semantic accelerators" for RL than as end-to-end controllers.

## Limitations & Future Work

- The hybrid framework is validated on only one game (Freeway); generalizability remains unknown.
- The computational cost of VLM inference (per-step GPT-4o calls) and its tradeoff with sample efficiency are not quantified.
- Atari evaluation is limited to GPT-4o; open-source VLMs are not compared.
- The intervention mechanism is non-adaptive — decisions about when to transfer control to the VLM and when to return it to RL should be grounded in uncertainty estimates.
- Alternative integration modes, such as using VLMs as reward shapers or state abstractors, are not explored.

## Related Work & Insights

- **Atari-GPT** (Waytowich et al.): evaluates VLMs on dense-reward Atari; this paper focuses on sparse-reward hard-exploration games.
- **BALROG** (Paglieri et al.): identifies the knowing-doing gap in NetHack; this paper independently validates the same phenomenon in Atari.
- **TextAtari** (Li et al.): removing the visual bottleneck substantially improves LLM reasoning, confirming that low-level control is the primary bottleneck.
- **Intelligent Go-Explore** (Lu et al.): uses GPT-4 to replace handcrafted heuristics for selecting states to revisit — a more successful paradigm than the direct control approach taken here.
- **Motif** (Klissarov et al.): employs LLMs as intrinsic reward functions rather than direct controllers, making better use of semantic understanding capabilities.

## Rating

- **Novelty**: 7/10 — The systematic evaluation is valuable, but the knowing-doing gap concept has prior precedent.
- **Experimental Thoroughness**: 7/10 — The progressive evaluation design is strong; validation of the hybrid framework is insufficient.
- **Value**: 6/10 — The hybrid framework is overly simplistic and serves more as a proof of concept.
- **Writing Quality**: 8/10 — Structure is clear; qualitative analysis is vivid and well-illustrated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[ICLR 2026\] Optimistic Task Inference for Behavior Foundation Models](../../ICLR2026/reinforcement_learning/optimistic_task_inference_behavior_models.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)

</div>

<!-- RELATED:END -->
