---
title: >-
  [Paper Note] Sample-efficient and Scalable Exploration in Continuous-Time RL
description: >-
  [Reinforcement Learning] This paper proposes COMBRL, an algorithm that achieves scalable and sample-efficient exploration in continuous-time model-based RL by maximizing a weighted sum of extrinsic reward and epistemic uncertainty, with theoretical guarantees of sublinear regret.
tags:
  - Reinforcement Learning
date: 2026-05-08
content_hash: 456fd1c29466dc58
---

# Sample-efficient and Scalable Exploration in Continuous-Time RL

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2510.24482](https://arxiv.org/abs/2510.24482)
- **Code**: [https://go.klem.nz/combrl](https://go.klem.nz/combrl)
- **Area**: Reinforcement Learning
- **Keywords**: continuous-time RL, model-based RL, optimistic exploration, epistemic uncertainty, Gaussian processes, Bayesian neural networks

## TL;DR
This paper proposes COMBRL, an algorithm that achieves scalable and sample-efficient exploration in continuous-time model-based RL by maximizing a weighted sum of extrinsic reward and epistemic uncertainty, with theoretical guarantees of sublinear regret.

## Background & Motivation
- Most RL algorithms assume discrete-time dynamics, yet real-world control systems (robotics, biological processes) are naturally described by ODEs. Discretization may obscure critical temporal behaviors and limit control flexibility.
- Prior continuous-time MBRL methods (e.g., OCORL) achieve optimistic exploration via joint optimization over plausible dynamics, but at prohibitive computational cost: coupling optimization over the plausible dynamics set expands the input dimension from $d_u$ to $d_u + d_x$, making the approach intractable for high-dimensional systems.
- Prior methods rely on extrinsic reward signals and cannot handle unsupervised RL or system identification scenarios.
- **Core Problem**: How to design an exploration mechanism that is simultaneously scalable, sample-efficient, and theoretically grounded within a continuous-time ODE framework?

## Method

### Overall Architecture
COMBRL (Continuous-time Optimistic Model-Based RL) operates in an episodic continuous-time setting, alternating between two steps:
1. **Model Learning**: A probabilistic model (GP or BNN) is trained to learn the ODE $\bm{f}^*$, yielding mean prediction $\bm{\mu}_n(\bm{z})$ and epistemic uncertainty $\bm{\sigma}_n(\bm{z})$.
2. **Policy Selection**: The policy is selected by maximizing a reward-plus-uncertainty objective.

### Key Design: Optimistic Planning Objective
At each episode $n$, policy $\bm{\pi}_n$ is selected by maximizing:

$$\bm{\pi}_n = \arg\max_{\bm{\pi} \in \Pi} \int_0^T \frac{r(\bm{x}'(s), \bm{u}(s)) + \lambda_n \cdot \|\bm{\sigma}_{n-1}(\bm{x}'(s), \bm{u}(s))\|}{1 + \lambda_n} ds$$

where $\lambda_n$ controls the trade-off between extrinsic reward and epistemic uncertainty:
- $\lambda_n = 0$: greedy policy (pure exploitation)
- $0 < \lambda_n < \infty$: balanced exploration and exploitation
- $\lambda_n \to \infty$: purely unsupervised exploration

### $\lambda_n$ Selection Strategies
- **Static**: Fixed hyperparameter tuned via grid search.
- **Annealing Schedule**: $\lambda_n = \lambda_0 \cdot (1 - n/N)$, encouraging more exploration early and exploitation late.
- **Automatic Tuning**: Adaptive selection based on mutual information gain.

### Key Distinction from Prior Methods
COMBRL does not require joint optimization over the set of plausible dynamics. In practice, it uses the mean model $\bm{\mu}_n$ selected from $\mathcal{M}_{n-1} \cap \mathcal{F}$, and realizes optimistic exploration via intrinsic reward. This avoids the reparameterization trick in OCORL that causes input-dimension inflation.

### Measurement Selection Strategy (MSS)
In continuous-time RL, the agent must also decide when to observe and control the system. The MSS $S = (S_n)_{n \geq 1}$ specifies measurement time points within each episode, affecting data collection quality and the regret bound.

### Theoretical Guarantees
**Theorem 1 (Regret Bound)**: Under assumptions of Lipschitz continuity, sub-Gaussian noise, and a well-calibrated model, the cumulative regret of COMBRL satisfies:

$$R_N \leq \mathcal{O}\left(\sqrt{\mathcal{I}_N^3(\bm{f}^*, S) \cdot N}\right)$$

where $\mathcal{I}_N$ denotes model complexity. For RBF kernels with equidistant MSS, $\mathcal{I}_N$ grows as $\text{polylog}(N)$, guaranteeing sublinear regret.

**Theorem 2 (Unsupervised Sample Complexity)**: Under pure intrinsic exploration ($\lambda_n \to \infty$), the maximum epistemic uncertainty decays at rate $\mathcal{O}(\sqrt{\mathcal{I}_N^3 / N})$.

## Key Experimental Results

### Main Results: Learning Performance under GP Dynamics

| Environment | Method | Asymptotic Performance | Compute Ratio |
|------|------|----------|-----------|
| Pendulum | Mean (λ=0) | Suboptimal | 1× |
| Pendulum | PETS | Moderate | ~1× |
| Pendulum | OCORL | Near-optimal | ~3× |
| Pendulum | **COMBRL** | **Near-optimal** | **~1×** |
| MountainCar | Mean (λ=0) | Suboptimal | 1× |
| MountainCar | **COMBRL** | **Optimal** | ~1× |

> COMBRL matches or surpasses OCORL in performance while requiring only approximately one-third of its computational cost.

### Ablation Study: Effect of Intrinsic Reward

| Environment | Mean (λ=0) | PETS | COMBRL (auto λ) | Performance Gain |
|------|-----------|------|-----------------|---------|
| Reacher (easy) | ~Baseline | Moderate | Best | Significant |
| Finger (spin) | ~Baseline | Moderate | Best | Significant |
| Cartpole (balance) | ~Baseline | Near-best | Best | Moderate |
| Hopper (stand) | ~Baseline | Moderate | Best | Significant |

> COMBRL achieves the largest performance gains in sparse-reward or underactuated tasks, with consistent improvements in higher-dimensional domains. Automatic tuning of $\lambda_n$ proves effective.

### Key Findings
1. COMBRL consistently outperforms the greedy baseline and PETS across all tested environments.
2. COMBRL matches OCORL in performance while incurring approximately one-third of the computational overhead.
3. Models learned via unsupervised exploration transfer to unseen downstream tasks.
4. Automatic $\lambda_n$ tuning achieves performance close to the best manually tuned hyperparameter.

## Highlights & Insights
- **Unified Framework**: A single scalar $\lambda_n$ elegantly unifies supervised and unsupervised RL settings.
- **Scalability**: By avoiding optimization over the plausible dynamics set, the method is compatible with neural network models such as BNNs.
- **Theoretical Completeness**: Both supervised regret bounds and unsupervised sample complexity bounds are established.
- **Explicit MSS Dependence**: This work is the first to explicitly characterize the impact of measurement strategy on continuous-time RL performance.

## Limitations & Future Work
- The theoretical analysis relies on RKHS smoothness assumptions and well-calibrated model assumptions, which BNNs may not fully satisfy in practice.
- Experiments are validated only on moderate-dimensional tasks (up to DMC environments); performance on very high-dimensional settings (e.g., pixel-based inputs) remains to be verified.
- Optimal selection of $\lambda_n$ requires further investigation, and the theoretical guarantees of the automatic tuning approach are limited.

## Related Work & Insights
- **Continuous-time MBRL**: OCORL (Treven et al., 2023) provides theoretical guarantees but does not scale; Yildiz et al. (2021) adopt a greedy approach without exploration.
- **Intrinsic Motivation / Unsupervised RL**: Sekar et al. (2020), Pathak et al. (2019), Sukhija et al. (2023) all operate in discrete time.
- **Discrete-time Counterpart**: Sukhija et al. (2025b) study a discrete-time variant; COMBRL addresses the distinct theoretical and experimental requirements of the continuous-time setting.

## Rating
- Novelty: ⭐⭐⭐⭐ — Unifies reward-plus-uncertainty optimistic exploration in the continuous-time setting, handling both supervised and unsupervised scenarios.
- Theoretical Depth: ⭐⭐⭐⭐ — Dual guarantees of sublinear regret and sample complexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-environment comparisons, ablations, and automatic tuning validation.
- Value: ⭐⭐⭐⭐ — Computationally efficient and applicable to continuous-time physical control systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)

</div>

<!-- RELATED:END -->
