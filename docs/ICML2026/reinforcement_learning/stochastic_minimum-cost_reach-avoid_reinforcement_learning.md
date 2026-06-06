---
title: >-
  [Paper Note] Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][reach-avoid] This paper proposes the Reach-Avoid Probability Certificate (RAPC), which uses a max-min-clamped Bellman contraction operator to lower-bound the value function by the reac…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "reach-avoid"
  - "probability certificate"
  - "Bellman contraction"
  - "compensation factor"
  - "gradient correction"
date: 2026-05-08
content_hash: 1def50c993f3c7f5
---

# Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.11975](https://arxiv.org/abs/2605.11975)  
**Code**: None  
**Area**: Reinforcement Learning / Safe RL / Reach-Avoid Control  
**Keywords**: reach-avoid, probability certificate, Bellman contraction, compensation factor, gradient correction

## TL;DR
This paper proposes the Reach-Avoid Probability Certificate (RAPC), which uses a max-min-clamped Bellman contraction operator to lower-bound the value function by the reach-avoid probability. It introduces an adversarial $\gamma^T$-decay "compensation factor" for normalization, and employs symmetric gradient projection to jointly optimize the conflicting objectives of "cost" and "reach-avoid probability." On MuJoCo, it achieves both lower cumulative cost and higher reach success rate than RC-PPO / RESPO / CPPO.

## Background & Motivation

**Background**: Safe RL treats reach-avoid (reaching the goal while avoiding hazards) as a core paradigm. Mainstream approaches include CMDP (Achiam, Sauté, CPPO), reward shaping, HJ reachability, and barrier functions. Many real-world scenarios (AGV routing, autonomous driving) require both "probability $\ge p$ of satisfying reach-avoid" and "minimum expected cost"—i.e., the stochastic minimum-cost reach-avoid problem.

**Limitations of Prior Work**: Three main approaches each have critical drawbacks: (1) CMDP encodes reach implicitly via sparse reward/shaping, losing the semantics of reach-avoid, making weight tuning difficult and often leading to infeasibility; (2) chance-constrained/CVaR methods characterize the tail risk of "cumulative return," but not whether temporal reach-avoid specifications are satisfied, thus missing the point; (3) HJ-based RC-PPO (So 2024) directly targets minimum-cost reach-avoid, but **only supports deterministic environments** and cannot handle stochastic noise.

**Key Challenge**: In existing theory, the "probabilistic reach-avoid constraint" and "expected cumulative cost minimization" objectives are **structurally incompatible**—the former requires computing event probability distributions, the latter requires expected returns. The CMDP framework forcibly merges both into a single expectation, inevitably causing distortion.

**Goal**: (a) Define a certificate (RAPC) that provides a lower bound for "$P(\mathbf{RA})\ge p$" and is learnable via Bellman updates; (b) Design an objective that minimizes cost within the feasible set and maximizes reach probability outside it; (c) Provide convergence proof and demonstrate effectiveness in stochastic environments.

**Key Insight**: Formulate the reach-avoid probability certificate as the fixed point of a Bellman operator—using two shaping functions $g, h$ (Eq. 1) to encode "reaching the goal" and "entering danger" into the boundaries, and then applying max-min clamping to make the operator a $\gamma$-contraction.

**Core Idea**: Use the "max-min-clamped" Bellman operator $B^\pi[V]=\max\{h,\min\{g,\gamma\mathbb E V'\}\}$, whose unique fixed point $V_{g,h}^\pi$ provides the certificate $\mathbb P_\pi(\mathbf{RA}_x)\ge-V_{g,h}^\pi(x)/M$; introduce a compensation factor $\phi_\gamma^\pi(x)=\mathbb E[\gamma^T\mid\mathbf{RA}_x]$ to correct the "long horizon → overly conservative estimate" issue; finally, use symmetric gradient correction to jointly optimize cost and probability.

## Method

### Overall Architecture
RAPCPO (Algorithm 1) is an actor-critic framework. In each main loop: (1) Interact for horizon $H$ steps, collecting $(x_t,a_t,c_t,g(x_t),h(x_t),x_{t+1})$ into the buffer; (2) Train the RAPC critic $Q_{g,h}(x,a;\eta)$ using Eq. 17; (3) Train the cost critic $Q_c(x,a;\kappa)$ via TD; (4) Train the compensation factor $\phi_\gamma(x;\xi)$ using $y_t=\gamma^{T-t}$ from successful reach-avoid trajectories; (5) Compute the critic-induced feasible set $\mathcal X_p^{\pi_{\theta_l}}$ and construct partitioned objectives; (6) Update the actor using symmetric projected gradients; (7) Repeat until convergence.

### Key Designs

1. **Reach-Avoid Probability Certificate (RAPC) + max-min-clamped Bellman operator**:

    - **Function**: Provides a learnable function $V_{g,h}^\pi$ as a strict lower bound for $\mathbb P_\pi(\mathbf{RA}_x)$, replacing the intractable true probability.
    - **Mechanism**: Define $g(x)<0$ on $\mathcal T$ (set to $-M$), $g(x)>0$ on $\mathcal X\setminus\mathcal T$; $h(x)=M$ on $\mathcal F$, $-M$ on $\mathcal X\setminus\mathcal F$. The operator $B^\pi[V](x)=\max\{h(x),\min\{g(x),\gamma\mathbb E_{a\sim\pi,x'}[V(x')]\}\}$ (Eq. 9) is a $\gamma$-contraction (Lemma 4.4) with a unique fixed point $V_{g,h}^\pi$. Along successful reach-avoid trajectories (hit time $T$, terminal state in $\mathcal T$), the operator reduces to linear recursion $V(x_t)=\gamma V(x_{t+1})$, with boundary $V(x_T)=-M$, so $V(x_0)=-\gamma^T M$. Theorem 4.5 gives $\mathbb P_\pi(\mathbf{RA}_x)\ge-V_{g,h}^\pi(x)/M$, i.e., the RAPC.
    - **Design Motivation**: Directly learning $\mathbb P(\mathbf{RA})$ lacks Bellman recursion; using fixed-$\gamma$ Bellman (Xue 2026, Eq. 8) leads to extremely sparse rewards due to "all-zero signals outside the target"; the max-min-clamped operator preserves probabilistic meaning and, since $g(x)>0$ in non-target states provides dense signals, greatly improves training efficiency (Table 2: enhanced Bellman reach rate 0.79 vs fixed-$\gamma$ 0.44 on HalfCheetah).

2. **Compensation factor $\phi_\gamma^\pi(x)$ to counteract $\gamma^T$ decay**:

    - **Function**: Eliminates the bias in $V_{g,h}^\pi$ caused by $\gamma^T$ compression for long hit times, so that $-V/(M\phi)$ approximates the true reach probability; otherwise, the certificate is overly conservative.
    - **Mechanism**: From the approximate decomposition $V_{g,h}^\pi(x)\approx\mathbb E_\pi[-M\gamma^T\mid\mathbf{RA}_x]\,\mathbb P_\pi(\mathbf{RA}_x)=-M\phi_\gamma^\pi(x)\mathbb P_\pi(\mathbf{RA}_x)$ (Eq. 11), derive the normalized estimate $\hat p_\pi(x)=-V_{g,h}^\pi(x)/(M\phi_\gamma^\pi(x))$ (Eq. 13). $\phi$ is fitted by a neural network $\phi_\gamma(x;\xi)$, trained only on successful reach-avoid rollouts with label $y_t=\gamma^{T-t}$ using MSE (Eq. 19); updates are skipped for unsuccessful trajectories.
    - **Design Motivation**: Without $\phi$, "long-horizon tasks with high true probability but small $V$ values" cause the algorithm to mistakenly deem states infeasible, leading to "standing still"; with $\phi$, feasible set determination is more accurate. Experiments (Fig 6) show that removing $\phi$ on HalfCheetah leads to a sharp increase in extra cost and a drop in reach rate.

3. **Certificate-based state partitioning + symmetric gradient correction**:

    - **Function**: Emphasizes cost reduction in "feasible states" and probability increase in "infeasible states," using projection to balance the two when their objectives conflict.
    - **Mechanism**: Use the current critic to construct the surrogate feasible set $\mathcal X_p^{\pi_{\theta_l}}=\{x:V_{g,h}^{\pi_{\theta_l}}(x)\le-pM\phi(x),\,\phi(x)\ge 0\}$ (Eq. 15). Compute three gradients: $g_r^{in},g_r^{out}$ are the reach probability gradients (using $-V_{g,h}/\phi$) for feasible/infeasible states, $g_c^{in}$ is the cost gradient. If $\langle g_r^{in},g_c^{in}\rangle<0$ (conflict), use symmetric projection $\tilde g_r^{in}=g_r^{in}-\frac{\langle g_r^{in},g_c^{in}\rangle}{\|g_c^{in}\|^2}g_c^{in}$ to remove the negative direction (Eq. 21), then combine $g_{mix}=\tilde g_r^{in}+\tilde g_c^{in}$; finally, $g_\theta=g_r^{out}+g_{mix}$ (Eq. 23). This is a PCGrad-style two-objective projection.
    - **Design Motivation**: Directly summing $g_r+g_c$ causes mutual cancellation when objectives conflict, leading to unstable training; symmetric projection allows both objectives to progress "within subspaces that do not harm each other," resulting in more stable convergence and typically exceeding the reach probability threshold $p$ (a nice property noted in the paper).

### Loss & Training

- **RAPC critic loss** (Eq. 17): $\mathcal J_{Q_{g,h}}(\eta)=\frac12\mathbb E[(Q_{g,h}(x,a;\eta)-\hat Q_{g,h}(x,a))^2]$, with the target being the max-min-clamped Bellman backup (Eq. 18).
- **Cost critic loss**: Standard TD.
- **$\phi$ loss**: MSE to $\gamma^{T-t}$, updated only on successful trajectories.
- **Actor**: Uses the composite gradient from Eq. 23, implemented based on PPO; the paper sets $p=0.5$.
- **Convergence**: Under standard step-size and bounded parameter conditions, almost surely converges to a generalized stable point of the surrogate objective in the sense of differential inclusion (Appendix B.2).

## Key Experimental Results

### Main Results

**Deterministic reach-avoid (same iteration budget)** Table 1:

| Method | PointGoal reach | FixedWing reach |
|---|---|---|
| RC-PPO | 62.29% | 73.98% |
| **RAPCPO (ours)** | **78.49%** | **88.67%** |

Fig 2 further shows that RAPCPO achieves lower cumulative cost than RESPO / CPPO / Sauté / PPO$_\beta$ in both environments.

**Stochastic reach-avoid (10% Gaussian action noise, Safety Hopper / HalfCheetah)** Fig 5: RAPCPO achieves **lowest cost + highest reach rate** in both environments; Sauté / CPPO's CVaR constraints are overly conservative, and RC-PPO is unstable under stochasticity.

### Ablation Study

**Bellman form comparison (Table 2, same iteration budget)**: Whether to use the enhanced Bellman formula (Eq. 9) instead of the fixed-$\gamma$ Bellman (Eq. 8).

| Method | Safety HalfCheetah | Safety Hopper | PointGoal | FixedWing |
|---|---|---|---|---|
| Fixed-$\gamma$ Bellman | 0.44 | 0.32 | 0.45 | 0.47 |
| **Enhanced Bellman** | **0.80** | **0.94** | **0.78** | **0.88** |

**Compensation factor $\phi$ ablation (Fig 6)**: Removing $\phi$ significantly increases cumulative cost (severe over-conservatism on FrozenLake) and reduces reach rate, confirming $\phi$ as key to RAPCPO.

**$p$ hyperparameter (Fig 7, Safety Hopper)**: With $p=0$, the reach signal is too weak and the agent fails; $p\in[0.1,0.7]$ is optimal; $p\ge 0.8$ causes cost to explode due to stochastic noise.

### Key Findings
- The max-min-clamped Bellman operator is the key design for "guaranteeing probabilistic semantics while providing dense reward," decoupling reach-avoid and cost so that the two critics can be trained independently and stably.
- The compensation factor $\phi$, though seemingly a numerical correction, is in fact an essential fix for "long-horizon reach-avoid estimation bias"; without it, the algorithm's reach rate drops significantly.
- Symmetric gradient projection is a general trick for "dual-critic + dual-objective" algorithms and can be applied to many Safe RL settings.
- Increasing the reach threshold $p$ is not always better—in stochastic environments, too large $p$ forces the policy to choose conservative long paths, causing cost to explode, which is a valuable engineering insight.

## Highlights & Insights
- **Clear theoretical structure**: Operator contraction → fixed point → probability certificate → compensation factor → feasible set → projected gradient, each step addresses a specific engineering pain point with no redundant design.
- **Dense signal is key**: Allowing $g(x)$ to take nonzero positive values in non-target states, though a small change, transforms Bellman learning from "almost sparse" to "almost dense," which is fundamental for success on stochastic MuJoCo.
- The idea of expressing reach-avoid probability as a Bellman fixed point inspires similar operators for other temporal logic specifications (LTL until, response), opening up significant potential in formal RL.
- Dual-gradient symmetric projection can also be applied to "multi-task RL," "alignment RL," etc., as a lightweight and robust engineering package.

## Limitations & Future Work
- Theorem 4.5 provides only a sufficient condition, not necessary and sufficient; i.e., $V_{g,h}^\pi(x)<0$ may not cover all truly feasible reach-avoid states, possibly missing feasible points.
- $\phi$ is trained only on successful trajectories; when early training success rate is very low, $\phi$ is severely underfit, distorting feasible set determination. The paper does not discuss cold-start stability.
- Experiments are limited to 5 MuJoCo + FrozenLake environments, lacking real high-dimensional visuomotor/autonomous driving benchmarks; simulated noise is simple Gaussian, while real-world deployment faces more complex model uncertainty.
- The forward invariant assumption for $\mathcal X$ is strong; behavior outside the boundary is undefined, whereas real robot physical boundaries are often crossed.
- $p$ is manually tuned, with optimal values varying by task, lacking an adaptive mechanism.

## Related Work & Insights
- **vs RC-PPO (So 2024)**: RC-PPO uses HJ reachability for minimum-cost reach-avoid but only for deterministic cases; RAPCPO extends to stochastic settings with Bellman certificates + compensation factor, and is more stable even in deterministic environments (Table 1: reach rate higher by 15–16 points).
- **vs CMDP (CPPO / Sauté / RESPO)**: CMDP constrains surrogate cumulative cost, misaligned with reach-avoid semantics; RAPCPO directly optimizes reach-avoid probability, and ablation shows CMDP is too conservative under state-wise cost.
- **vs CVaR / chance-constrained**: These characterize the tail risk of return, not event probability, and do not address the "must satisfy specification with probability $\ge p$" goal.
- **vs Barrier / CBF (Ames 2019, Xue 2026)**: Both provide formal guarantees but only for specification satisfaction, not cost optimization; RAPCPO balances performance and safety within the probability certificate framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The trio of max-min-clamped Bellman, compensation factor, and symmetric projection each address a specific pain point, with strong combinatorial originality.
- Experimental Thoroughness: ⭐⭐⭐ MuJoCo + FrozenLake coverage is reasonable, but lacks real robot/autonomous driving benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear storyline and notation, with well-explained operator design motivation.
- Value: ⭐⭐⭐⭐ Provides the first stable and trainable baseline for stochastic reach-avoid, directly reusable by the safe RL community.

## Related Papers

- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](../../ICLR2026/reinforcement_learning/solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](../../NeurIPS2025/reinforcement_learning/robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)
- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](../../AAAI2026/reinforcement_learning/deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](../../AAAI2026/reinforcement_learning/good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)
- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](../../ICLR2026/reinforcement_learning/solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](../../NeurIPS2025/reinforcement_learning/robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)
- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](../../AAAI2026/reinforcement_learning/deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](../../AAAI2026/reinforcement_learning/good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)
- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)

</div>

<!-- RELATED:END -->
