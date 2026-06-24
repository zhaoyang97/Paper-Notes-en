---
title: >-
  [Paper Note] Online Optimization for Offline Safe Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Offline Safe Reinforcement Learning] This paper proposes O3SRL, a framework that formalizes offline safe reinforcement learning as a minimax optimization problem. By combining an offline RL oracle with EXP3-based online optimization for adaptive Lagrange multiplier adjustment, O3SRL avoids unstable off-policy evaluation and achieves high reward under strict safety constraints.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Offline Safe Reinforcement Learning"
  - "Minimax Optimization"
  - "Multi-Armed Bandit"
  - "Constrained Policy Optimization"
  - "No-Regret Algorithm"
date: 2026-05-08
content_hash: 5828fcb82b016e27
---

# Online Optimization for Offline Safe Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.22027](https://arxiv.org/abs/2510.22027)  
**Code**: [GitHub](https://github.com/yassineCh/O3SRL)  
**Area**: Reinforcement Learning
**Keywords**: Offline Safe Reinforcement Learning, Minimax Optimization, Multi-Armed Bandit, Constrained Policy Optimization, No-Regret Algorithm

## TL;DR

This paper proposes O3SRL, a framework that formalizes offline safe reinforcement learning as a minimax optimization problem. By combining an offline RL oracle with EXP3-based online optimization for adaptive Lagrange multiplier adjustment, O3SRL avoids unstable off-policy evaluation and achieves high reward under strict safety constraints.

## Background & Motivation

Offline reinforcement learning (Offline RL) learns decision policies from fixed datasets without environment interaction and has seen success in autonomous driving, robotics, and related domains. However, safety-critical applications (e.g., healthcare, power systems) additionally require learned policies to satisfy **cumulative cost constraints**—giving rise to Offline Safe Reinforcement Learning (OSRL).

OSRL inherits dual challenges from both offline RL and safe RL: (1) distributional shift—learned policies may encounter state-action pairs unseen in the offline dataset; and (2) safety constraint satisfaction—cost constraints must hold after deployment.

Limitations of prior work are pronounced:

- **Lagrangian relaxation-based methods** (e.g., BEAR-Lag, CPQ, COptiDICE) require solving coupled optimization problems that are highly unstable in practice, prone to oscillation/divergence, or yielding overly conservative policies (near-zero reward).
- **Safety-focused methods** such as FISOR produce zero-violation policies but at the cost of very low reward.
- Critically, the regime of **strict safety constraints** (small cost threshold $\kappa$) is an important yet severely underexplored setting that existing methods almost universally handle poorly.

This paper's starting point is to reformulate OSRL as a **minimax optimization** problem and employ **no-regret online optimization** to adaptively adjust Lagrange multipliers, providing theoretical guarantees while avoiding unstable off-policy evaluation (OPE).

## Method

### Overall Architecture

O3SRL operates within the Constrained Markov Decision Process (CMDP) framework. Given an offline dataset $\mathcal{D}_{OSRL}$, the objective is:

$$\max_{\pi} \mathbb{E}_{\tau \sim \pi}[R(\tau)] \quad \text{s.t.} \quad \mathbb{E}_{\tau \sim \pi}[C(\tau)] \leq \kappa$$

Lagrangian relaxation converts this into a dual (minimax) problem:

$$\min_{\lambda \geq 0} \max_{D \in \Delta\Pi} L(D, \lambda) = \mathbb{E}_{\pi \sim D}[V^{\pi}_{r - \lambda(c - (1-\gamma)\kappa)}]$$

Under the Slater condition, strong duality guarantees equivalence between the primal and dual problems.

### Key Designs

1. **Alternating Iteration: Offline RL Oracle + No-Regret Updates**: Each iteration performs two steps—(a) absorbing the cost constraint into a modified reward $r'_i = r_i - \lambda_{t-1}(c_i - (1-\gamma)\kappa)$ and invoking the offline RL oracle to optimize this reward, yielding policy distribution $D_t$; (b) updating $\lambda_t$ via a no-regret algorithm based on the current policy distribution. The algorithm returns the averaged policy $\bar{D}$ and averaged multiplier $\bar{\lambda}$. Convergence to a minimax equilibrium is guaranteed (Theorem 1: $\epsilon = \epsilon_{\text{offline-RL}}(n) + R_T(\Lambda)/T$).

2. **Discretization + EXP3 Multi-Armed Bandit**: Two practical challenges arise in the general framework—OPE accumulates errors across iterations and is computationally expensive, and running offline RL to convergence each round is costly. The solution discretizes the continuous search space for $\lambda$ into $K$ values $\{\lambda^{(1)}, \dots, \lambda^{(K)}\}$, treating each as an "arm," and applies the EXP3 bandit algorithm to adaptively adjust arm selection probabilities based on historical performance. A key advantage is that the MAB algorithm requires no OPE. An $\epsilon$-approximate equilibrium is guaranteed (Theorem 2), with error decomposed into three terms: $\epsilon_{\text{offline-RL}}(n) + \sqrt{K/T} + 1/K$.

3. **Practical Approximations**: (a) A **stochastic oracle** replaces the exact oracle—each round performs only $M$ gradient steps rather than running to convergence (warm-started from the previous round); (b) The **last-iterate policy** $\pi_T$ is returned instead of the average distribution, avoiding the need to store all intermediate policies. Experiments show that even $K=2$ (two arms) and a small $M=10$ achieve state-of-the-art performance.

### Loss & Training

The default underlying offline RL algorithm is TD3+BC. The search space is $\Lambda = [0, 5]$ with $K=5$ arms by default. Training runs for $T=100{,}000$ total iterations, with arm probabilities updated every $M=10$ gradient steps.

## Key Experimental Results

### Main Results

Evaluated on 8 tasks from the DSRL Bullet benchmark under strict cost threshold $\kappa=5$:

| Task | Metric | O3SRL | FISOR | CAPS | CDT | CPQ | Notes |
|------|--------|-------|-------|------|-----|-----|-------|
| BallRun | Reward↑ | **0.25** | 0.09 | 0.07 | 0.27 | 0.09 | O3SRL achieves highest safe reward |
| BallRun | Cost↓ | **0.00** | 1.28 | 0.00 | 2.57 | 2.20 | CDT/FISOR violate constraints |
| CarRun | Reward↑ | 0.96 | 0.74 | 0.97 | 0.99 | 0.93 | All methods perform comparably |
| BallCircle | Reward↑ | **0.62** | 0.32 | 0.33 | 0.61 | 0.56 | **Best among safe methods** |
| AntCircle | Reward↑ | **0.48** | 0.24 | 0.33 | 0.45 | 0.00 | CDT violates constraints |

**Core finding**: O3SRL is the only method that satisfies safety constraints on all 8 tasks. Competing methods are either safe but low-reward (FISOR/CAPS), or high-reward but frequently constraint-violating (CDT/CCAC/BEAR-Lag).

### Ablation Study

| Configuration | Key Finding | Notes |
|---------------|-------------|-------|
| K=2 vs K=5 vs K=10 | K=2 is effective but yields lower reward; K=5 offers the best trade-off; K=10 shows diminishing returns | Coarse discretization suffices |
| κ=5 → κ=20 → κ=40 | As constraints relax, the policy automatically shifts toward higher reward | No budget-specific tuning required |
| TD3+BC vs IQL | Both underlying RL algorithms yield comparable performance | **Plug-and-play**; not tied to a specific algorithm |

### Key Findings

- Even the simplest $K=2$ variant maintains safety across all tasks, demonstrating that EXP3 remains effective under coarse discretization.
- O3SRL's advantage is most pronounced under **strict safety constraints** ($\kappa=5$)—precisely where competing methods are weakest.
- The plug-and-play property (interchangeable underlying offline RL algorithms) confers strong generality.

## Highlights & Insights

- The paper elegantly reframes the unstable Lagrangian optimization of OSRL as a **multi-armed bandit** problem, entirely bypassing the accumulated errors of OPE—a compelling instance of reducing a complex problem to a simpler one.
- Theoretical analysis is complete, with convergence guarantees spanning both the general framework (Theorem 1) and the practical approximation (Theorem 2).
- The empirical finding that $K=2$ suffices suggests that the effective degrees of freedom in practice may be lower than expected—the safety-reward trade-off may itself be intrinsically low-dimensional.

## Limitations & Future Work

- The current work is limited to the offline setting; extension to offline-to-online safe RL (offline pre-training followed by online fine-tuning) is a natural next step.
- Discretization of $\lambda$ introduces $O(1/K)$ approximation error, which may be insufficient for extremely stringent safety requirements.
- Overall performance depends on the quality of the underlying offline RL algorithm and is limited when offline data coverage is poor.
- Experiments are conducted solely in the DSRL Bullet simulation environment, without evaluation on real-world safety-critical applications.

## Related Work & Insights

- O3SRL is complementary to CAPS (which switches among a set of pre-trained policies based on the cost budget): O3SRL is trained end-to-end, whereas CAPS combines existing policies.
- The paradigm of "solving offline problems via online optimization" has broader applicability—e.g., offline multi-objective RL and offline fair RL.
- The idea of using multi-armed bandits for adaptive hyperparameter selection merits transfer to other constrained RL settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — The minimax + MAB framework combination is novel, with strong integration of theory and practice
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ — 8 tasks with extensive ablations, but lacks real-world validation
- **Writing Quality**: ⭐⭐⭐⭐☆ — Theoretical derivations are clear; the layered progression from general framework to practical approximation is well-structured
- **Value**: ⭐⭐⭐⭐☆ — Provides a reliable and practical new solution for offline safe RL; plug-and-play property offers practical engineering value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Boundary-to-Region Supervision for Offline Safe Reinforcement Learning](boundary_to_region_supervision_for_offline_safe_rl.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](../../ICML2025/reinforcement_learning/online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[ICML 2025\] Extreme Value Policy Optimization for Safe Reinforcement Learning](../../ICML2025/reinforcement_learning/extreme_value_policy_optimization_for_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Gradient-Variation Online Adaptivity for Accelerated Optimization with Hölder Smoothness](gradient-variation_online_adaptivity_for_accelerated_optimization_with_hölder_sm.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)

</div>

<!-- RELATED:END -->
