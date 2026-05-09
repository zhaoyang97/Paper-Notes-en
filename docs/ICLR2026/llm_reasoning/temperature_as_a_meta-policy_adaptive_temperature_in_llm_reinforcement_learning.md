---
title: >-
  [Paper Note] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning
description: >-
  [ICLR 2026][LLM Reasoning][Temperature Scheduling] This paper proposes TAMPO (Temperature Adaptive Meta Policy Optimization), which reframes the sampling temperature as a learnable meta-policy. Through a bilevel loop — an inner loop for LLM policy optimization and an outer loop for adaptively updating the temperature distribution based on trajectory advantage signals — TAMPO requires no additional rollouts and consistently outperforms fixed-temperature baselines on mathematical reasoning benchmarks.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Temperature Scheduling
  - Meta-Policy
  - GRPO
  - Adaptive Exploration
  - Mathematical Reasoning
date: 2026-05-08
content_hash: d41841c1e560d0e9
---

# Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2602.11779](https://arxiv.org/abs/2602.11779)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Temperature Scheduling, Meta-Policy, GRPO, Adaptive Exploration, Mathematical Reasoning

## TL;DR

This paper proposes TAMPO (Temperature Adaptive Meta Policy Optimization), which reframes the sampling temperature as a learnable meta-policy. Through a bilevel loop — an inner loop for LLM policy optimization and an outer loop for adaptively updating the temperature distribution based on trajectory advantage signals — TAMPO requires no additional rollouts and consistently outperforms fixed-temperature baselines on mathematical reasoning benchmarks.

## Background & Motivation

- Temperature is the core parameter governing the exploration–exploitation trade-off in LLM sampling.
    - High temperature encourages diversity but introduces noise; low temperature improves focus but risks premature convergence.
- Existing RL training methods (e.g., GRPO) treat temperature as a **fixed hyperparameter**, ignoring the dynamic needs across training.
- Although entropy regularization and KL penalties also affect exploration, temperature directly modulates the sampling distribution and is more transparent and controllable.
- **Core argument**: Temperature should be a learnable decision variable rather than a manually tuned hyperparameter.

## Method

### Overall Architecture

TAMPO adopts a hierarchical bilevel loop structure:

- **Inner loop**: Generates rollouts using a selected temperature $T_s$ and updates the LLM policy $\pi_\theta$ via GRPO.
- **Outer loop**: Reuses inner-loop rollouts to update the temperature meta-policy $\pi(T)$ based on trajectory advantage signals.

### Key Designs

Each trajectory implicitly encodes its "preferred temperature" — the temperature under which that trajectory is most likely to be generated:

$$T^* = \arg\max_{T_k \in \mathcal{T}} \ell_{T_k}(\tau_i)$$

where $\ell_T(\tau_i) = \frac{1}{|\tau_i|} \sum_{t=1}^{|\tau_i|} \log \pi_{\theta,T}(o_{i,t} | s_{i,t})$ denotes the average log-likelihood.

### Temperature-Specific Advantage

For each trajectory $\tau_i$ and virtual candidate temperature $T_k$:

1. Compute $\ell_{T_k}(\tau_i)$: the likelihood of the trajectory under temperature $T_k$.
2. Apply sparsemax normalization to obtain $\hat{\ell}_{T_k}(\tau_i)$ (summing to 1 across $K$ candidate temperatures).
3. Temperature-specific advantage: $\mathcal{A}_i^{(T_k)} = \hat{\ell}_{T_k}(\tau_i) \cdot A_i$.

Intuition:
- Trajectories with positive advantage → reinforce their most likely generating temperature.
- Trajectories with negative advantage → penalize their most likely generating temperature.

### Meta-Policy Update

1. Batch aggregation: $\mathcal{A}_\mathcal{B}^{(T_k)} = \frac{1}{|\mathcal{B}|G} \sum_b \sum_i \mathcal{A}_{b,i}^{(T_k)}$
2. EMA smoothing: $\bar{\mathcal{A}}_s^{(T_k)} = (1-\alpha)\bar{\mathcal{A}}_{s-1}^{(T_k)} + \alpha \mathcal{A}_\mathcal{B}^{(T_k)}$
3. Min-max normalization to obtain a probability distribution: $\pi_s(T_k) = \frac{\tilde{\mathcal{A}}_s^{(T_k)}}{\sum_j \tilde{\mathcal{A}}_s^{(T_j)}}$

### Temperature Sampling

Temperature is sampled from the meta-policy via nucleus sampling (top-p), with $p=0.7$ providing the best exploration–exploitation balance.

### Design Characteristics

- **Zero additional rollouts**: Fully reuses trajectory data from the inner loop.
- **Non-differentiable optimization**: Temperature is non-differentiable in LLM RL; TAMPO circumvents this via likelihood signals.
- **Negligible overhead**: The meta-policy maintains only a list of temperature advantages and is discarded at inference time.

## Key Experimental Results

### Main Results: Mathematical Reasoning Benchmarks (DS-Qwen-1.5B)

| Method | Average | AIME24 | MATH-500 | AMC23 | Minerva | OlympiadBench |
|--------|---------|--------|----------|-------|---------|---------------|
| DS-Qwen-1.5B (no RL) | 39.1 | 13.3 | 76.2 | 45.0 | 22.8 | 38.4 |
| GRPO ($T_s$:0.9) | 42.0 | 20.0 | 75.2 | 50.0 | 26.1 | 38.7 |
| GRPO ($T_s$:1.5) | 42.6 | 23.3 | 75.4 | 52.5 | 22.8 | 39.0 |
| GRPO ($T_s$:0.9→1.5) | 42.8 | 16.7 | 76.6 | 55.0 | 24.6 | 41.0 |
| **TAMPO** | **44.5** | **23.3** | **76.8** | **55.0** | **27.9** | **39.6** |

### Ablation Study: EMA Coefficient $\alpha$

| $\alpha$ | Average | AIME24 | MATH-500 | AMC23 | Minerva | OlympiadBench |
|---------|---------|--------|----------|-------|---------|---------------|
| 0.01 | 41.6 | 20.0 | 75.2 | 50.0 | 25.4 | 37.5 |
| **0.05** | **44.5** | **23.3** | **76.8** | **55.0** | **27.9** | **39.6** |
| 0.10 | 43.6 | 23.3 | 75.4 | 57.5 | 23.2 | 38.8 |

### Ablation Study: Meta-Policy Sampling Strategy

| top-p | Average |
|-------|---------|
| 0.9 | 43.0 |
| **0.7** | **44.5** |
| 0.5 | 42.2 |
| 0 (greedy) | 40.9 |

### Cross-Task Generalization (Qwen2.5-3B-Instruct → ECQA)

| Method | Pass@1 | Pass@8 |
|--------|--------|--------|
| No RL | 73.06% | 77.76% |
| GRPO | 75.07% | 78.94% |
| **TAMPO** | **76.12%** | **79.67%** |

### Key Findings

1. TAMPO outperforms the best fixed-temperature baseline by **+1.9%** (Pass@1) and **+1.7%** (Pass@8) on average.
2. The learned temperature dynamics favor higher temperatures (~1.3) after warmup to encourage exploration, gradually decreasing as training proceeds.
3. Greedy sampling ($p=0$) yields the worst results, indicating that exploration of the temperature space itself requires stochasticity.
4. Training time is identical to the baseline (~9h54min on 8×V100).
5. The method generalizes effectively to commonsense reasoning tasks.

## Highlights & Insights

- **Elevating temperature from a hyperparameter to a decision variable**: a novel problem formulation.
- **No additional rollouts required**: Virtual temperature likelihood computation cleverly reuses existing trajectory data.
- **Learned temperature policy aligns with intuition**: A high-to-low exploration–exploitation transition emerges naturally.
- **Fully compatible with existing RL methods**: Can be plugged into GRPO, DAPO, REINFORCE++, and other critic-free approaches.
- **Negligible computational overhead**: Only $K$ temperature advantage estimates need to be maintained.

## Limitations & Future Work

- The candidate temperature set $\mathcal{T}$ still requires manual specification of range and granularity.
- The unimodal property of trajectory likelihood with respect to temperature may not hold in all settings.
- Main experiments are conducted only on 1.5B models; validation on larger models is insufficient.
- The temperature meta-policy is shared across prompts; prompt-level adaptation remains unexplored.

## Related Work & Insights

- Critic-free RL: GRPO, DAPO, REINFORCE++
- Exploration–exploitation: ε-greedy, temperature annealing, UCB, entropy regularization
- Meta-policy: MLSH (hierarchical RL), Meta-SAC (automatic entropy coefficient)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The formulation of temperature as a meta-policy is original.
- **Technical Depth**: ⭐⭐⭐⭐ — Theoretical derivations are clear, though the method itself is concise.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five benchmarks with comprehensive ablations, but limited model scale.
- **Practicality**: ⭐⭐⭐⭐⭐ — Improves RL training with zero additional computational cost.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICLR 2026\] Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval](evoking_user_memory_personalizing_llm_via_recollection-familiarity_adaptive_retr.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)

<!-- RELATED:END -->
