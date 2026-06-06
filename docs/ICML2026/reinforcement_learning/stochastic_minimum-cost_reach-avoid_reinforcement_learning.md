---
title: >-
  [Paper Note] Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][reach-avoid] Ours proposes the Reach-Avoid Probability Certificate (RAPC), which utilizes a max-min-clamped Bellman contraction operator to lower-bound the reach-avoid probability with…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "reach-avoid"
  - "probabilistic certificate"
  - "Bellman contraction"
  - "compensation factor"
  - "gradient correction"
date: 2026-05-08
content_hash: 4cdc18eb62620fd9
---

# Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.11975](https://arxiv.org/abs/2605.11975)  
**Code**: None  
**Area**: Reinforcement Learning / Safe RL / Reach-Avoid Control  
**Keywords**: reach-avoid, probabilistic certificate, Bellman contraction, compensation factor, gradient correction

## TL;DR
Ours proposes the Reach-Avoid Probability Certificate (RAPC), which utilizes a max-min-clamped Bellman contraction operator to lower-bound the reach-avoid probability with a value function. Combined with a "compensation factor" against adversarial $\gamma^T$ decay for normalization and symmetric gradient projection to jointly optimize conflicting targets of "cost" and "reach-avoid probability," the method achieves lower cumulative costs and higher reach success rates than RC-PPO, RESPO, and CPPO on MuJoCo.

## Background & Motivation

**Background**: Safe RL treats reach-avoid (reaching target + avoiding danger) as a core paradigm; mainstream approaches include CMDP (Achiam, Sauté, CPPO), reward shaping, HJ reachability, and barrier functions. Numerous real-world scenarios (AGV paths, autonomous driving) require simultaneously satisfying "probability $\ge p$ reach-avoid" and "minimizing expected cost" — specifically, the stochastic minimum-cost reach-avoid problem.

**Limitations of Prior Work**: Three categories of methods have fundamental limitations: (1) CMDP implicitly encodes reach through sparse rewards/shaping, losing the semantics of reach-avoid, making weight tuning difficult, and easily falling into infeasibility; (2) chance-constrained / CVaR methods characterize tail risk of "cumulative return" rather than "satisfying temporal reach-avoid specifications," which is irrelevant; (3) HJ-based RC-PPO (So 2024) directly addresses minimum-cost reach-avoid but **only supports deterministic environments** and cannot handle stochastic noise.

**Key Challenge**: In existing theory, the "probabilistic reach-avoid constraint" and "minimum expected cumulative cost" objectives are **structurally incompatible** — the former requires calculating probability distributions of events, while the latter requires calculating expectations of returns. Forcing both into the CMDP expectation framework inevitably leads to distortion.

**Goal**: (a) Define a certificate (RAPC) that provides a lower bound for "$P(\mathbf{RA})\ge p$" and can be learned via Bellman equations; (b) design an objective function to minimize cost within the feasible set and maximize reach probability in the infeasible set; (c) provide convergence proofs that function in stochastic environments.

**Key Insight**: Treat the reach-avoid probability certificate as the fixed point of a Bellman operator — by using two shaping functions $g, h$ (Eq. 1) to encode "reaching targets" and "entering danger" into the boundaries, and then using max-min clamping to ensure the operator is a $\gamma$-contraction.

**Core Idea**: Give a certificate of $\mathbb P_\pi(\mathbf{RA}_x)\ge-V_{g,h}^\pi(x)/M$ using the unique fixed point $V_{g,h}^\pi$ of the "max-min-clamped" Bellman operator $B^\pi[V]=\max\{h,\min\{g,\gamma\mathbb E V'\}\}$; introduce a compensation factor $\phi_\gamma^\pi(x)=\mathbb E[\gamma^T\mid\mathbf{RA}_x]$ to correct the "long horizon $\rightarrow$ over-conservative estimation" issue; finally, use symmetric gradient correction to simultaneously optimize cost and probability.

## Method

### Overall Architecture
RAPCPO (Algorithm 1) is an actor-critic framework. Each iteration of the main loop: (1) runs $H$ interaction steps, collecting $(x_t,a_t,c_t,g(x_t),h(x_t),x_{t+1})$ into the buffer; (2) trains the RAPC critic $Q_{g,h}(x,a;\eta)$ using Eq. 17; (3) trains the cost critic $Q_c(x,a;\kappa)$ using TD; (4) trains the compensation factor $\phi_\gamma(x;\xi)$ using $y_t=\gamma^{T-t}$ from successful reach-avoid trajectories; (5) calculates the critic-induced feasible set $\mathcal X_p^{\pi_{\theta_l}}$ and constructs partitioned objectives; (6) updates the actor using symmetric projected gradients; (7) repeats until convergence.

### Key Designs

1. **Reach-Avoid Probability Certificate (RAPC) + max-min-clamped Bellman Operator**:

    - **Function**: Provides a learnable function $V_{g,h}^\pi$ as a strict lower bound for $\mathbb P_\pi(\mathbf{RA}_x)$, replacing the incomputable true probability.
    - **Mechanism**: Defines $g(x)<0$ on $\mathcal T$ (set to $-M$) and $g(x)>0$ on $\mathcal X\setminus\mathcal T$; $h(x)=M$ on $\mathcal F$ and $-M$ on $\mathcal X\setminus\mathcal F$. The operator $B^\pi[V](x)=\max\{h(x),\min\{g(x),\gamma\mathbb E_{a\sim\pi,x'}[V(x')]\}\}$ (Eq. 9) is a $\gamma$-contraction (Lemma 4.4) with a unique fixed point $V_{g,h}^\pi$. Along successful reach-avoid trajectories (hit time $T$, terminal state in $\mathcal T$), the operator degenerates into a linear recurrence $V(x_t)=\gamma V(x_{t+1})$ with boundary condition $V(x_T)=-M$, thus $V(x_0)=-\gamma^T M$. Theorem 4.5 gives $\mathbb P_\pi(\mathbf{RA}_x)\ge-V_{g,h}^\pi(x)/M$, which is the RAPC.
    - **Design Motivation**: Directly learning $\mathbb P(\mathbf{RA})$ lacks Bellman recurrence; using fixed-$\gamma$ Bellman (Xue 2026, Eq. 8) results in extremely sparse rewards because "signals outside the target are all zero"; the max-min-clamped operator preserves the probabilistic meaning while providing dense signals via $g(x)>0$ in non-target states, significantly improving training efficiency (Table 2: enhanced Bellman reach rate 0.79 vs. fixed-$\gamma$ 0.44 on HalfCheetah).

2. **Compensation Factor $\phi_\gamma^\pi(x)$ Against $\gamma^T$ Decay**:

    - **Function**: Eliminates the bias where $V_{g,h}^\pi$ is squashed by $\gamma^T$ due to long hit times, making $-V/(M\phi)$ approximate the true reach probability to prevent overly conservative certificates.
    - **Mechanism**: Derives a normalized estimate $\hat p_\pi(x)=-V_{g,h}^\pi(x)/(M\phi_\gamma^\pi(x))$ (Eq. 13) from the approximate decomposition $V_{g,h}^\pi(x)\approx\mathbb E_\pi[-M\gamma^T\mid\mathbf{RA}_x]\,\mathbb P_\pi(\mathbf{RA}_x)=-M\phi_\gamma^\pi(x)\mathbb P_\pi(\mathbf{RA}_x)$ (Eq. 11). $\phi$ is fitted with a neural network $\phi_\gamma(x;\xi)$; training data comes only from successful reach-avoid rollouts with labels $y_t=\gamma^{T-t}$, trained via MSE (Eq. 19); updates are skipped if the current trajectory fails.
    - **Design Motivation**: Without $\phi$, tasks with long horizons and high true probability but small $V$ values would lead the algorithm to assume infeasibility and exhibit "passive" behavior; with $\phi$, feasible set determination is more accurate. Experiments (Fig 6) show that removing $\phi$ on HalfCheetah causes skyrocketing extra costs and drops in reach rates.

3. **Certificate-based State Partitioning + Symmetric Gradient Correction**:

    - **Function**: Focuses on reducing cost in "feasible states" and increasing probability in "infeasible states," using projection to find a compromise when two objective directions conflict.
    - **Mechanism**: Constructs a surrogate feasible set $\mathcal X_p^{\pi_{\theta_l}}=\{x:V_{g,h}^{\pi_{\theta_l}}(x)\le-pM\phi(x),\,\phi(x)\ge 0\}$ (Eq. 15) using the current critic. Three gradients are calculated: $g_r^{in}, g_r^{out}$ are gradients for the reach probability term (substituted by $-V_{g,h}/\phi$) on feasible/infeasible states, respectively, and $g_c^{in}$ is the cost gradient. If $\langle g_r^{in},g_c^{in}\rangle<0$ (conflict), a symmetric projection $\tilde g_r^{in}=g_r^{in}-\frac{\langle g_r^{in},g_c^{in}\rangle}{\|g_c^{in}\|^2}g_c^{in}$ is used to remove opposing components (Eq. 21), then synthesized as $g_{mix}=\tilde g_r^{in}+\tilde g_c^{in}$; finally $g_\theta=g_r^{out}+g_{mix}$ (Eq. 23). This is a PCGrad-style multi-objective projection.
    - **Design Motivation**: Simple concatenation of $g_r+g_c$ during conflict results in mutual cancellation and unstable training; symmetric projection allows both objectives to advance "within subspaces that do not harm the other," leading to more stable convergence and usually causing reach probability to exceed the threshold $p$.

### Loss & Training
- **RAPC critic loss** (Eq. 17): $\mathcal J_{Q_{g,h}}(\eta)=\frac12\mathbb E[(Q_{g,h}(x,a;\eta)-\hat Q_{g,h}(x,a))^2]$, where the target is the max-min-clamped Bellman backup (Eq. 18).
- **Cost critic loss**: Standard TD.
- **$\phi$ loss**: MSE to $\gamma^{T-t}$, updated only on successful trajectories.
- **Actor**: Uses the composite gradient from Eq. 23, implemented based on PPO; the paper sets $p=0.5$.
- **Convergence**: Under standard step-size and bounded parameter conditions, it converges almost surely to a generalized stationary point of the surrogate objective in the sense of differential inclusion (Appendix B.2).

## Key Experimental Results

### Main Results

**Deterministic reach-avoid (same iteration budget)** Table 1:

| Method | PointGoal reach | FixedWing reach |
|---|---|---|
| RC-PPO | 62.29% | 73.98% |
| **RAPCPO (ours)** | **78.49%** | **88.67%** |

Fig 2 further shows that RAPCPO’s cumulative cost in both environments is lower than RESPO / CPPO / Sauté / PPO$_\beta$.

**Stochastic reach-avoid (10% Gaussian action noise, Safety Hopper / HalfCheetah)** Fig 5: RAPCPO achieves **lowest cost + highest reach rate** simultaneously in both environments. Among baselines, Sauté / CPPO's CVaR constraints are overly conservative, and RC-PPO is unstable under stochasticity.

### Ablation Study

**Bellman Form Comparison (Table 2, same iteration budget)**: Comparing the enhanced Bellman formula (Eq. 9) against the fixed-$\gamma$ Bellman (Eq. 8).

| Method | Safety HalfCheetah | Safety Hopper | PointGoal | FixedWing |
|---|---|---|---|---|
| Fixed-$\gamma$ Bellman | 0.44 | 0.32 | 0.45 | 0.47 |
| **Enhanced Bellman** | **0.80** | **0.94** | **0.78** | **0.88** |

**Compensation Factor $\phi$ Ablation (Fig 6)**: Removing $\phi$ significantly increases cumulative cost (severe over-conservatism on FrozenLake) and drops the reach rate, verifying $\phi$ as a critical component of RAPCPO.

**Hyperparameter $p$ (Fig 7, Safety Hopper)**: At $p=0$, the reach signal is too weak, and the agent fails; $p\in[0.1,0.7]$ is the sweet spot; at $p\ge 0.8$, the cost explodes due to stochastic noise.

### Key Findings
- The max-min-clamped Bellman operator is the key design for "guaranteeing probabilistic semantics while providing dense rewards," decoupling reach-avoid from cost to allow independent and stable critic training.
- The compensation factor $\phi$ appears to be a numerical correction but is essentially a fundamental fix for "long-horizon reach-avoid estimation bias"; without it, the reach rate drops significantly.
- Symmetric gradient projection is a universal trick for "dual critic + dual objective" algorithms, applicable to many Safe RL scenarios.
- High reach thresholds $p$ are not necessarily better — in stochastic environments, excessive $p$ forces policies into conservative long paths, causing cost explosions, which is a useful engineering insight.

## Highlights & Insights
- **Clear Theoretical Structure**: Operator contraction $\rightarrow$ fixed point $\rightarrow$ probabilistic certificate $\rightarrow$ compensation factor $\rightarrow$ feasible set $\rightarrow$ projected gradient; every step addresses a specific engineering pain point without redundancy.
- **Dense Signals are Key**: $g(x)$ can take non-zero positive values in non-target states. This small change transforms Bellman learning from "nearly sparse" to "nearly dense," which is the fundamental reason for success in stochastic MuJoCo.
- The idea of framing reach-avoid probability as a Bellman fixed point inspires the construction of similar operators for other temporal logic specifications (LTL until, response), offering significant potential for extension in formal RL.
- Dual-gradient symmetric projection can be applied to "multi-task RL," "alignment RL," etc., as a lightweight and stable engineering package.

## Limitations & Future Work
- Theorem 4.5 only provides a sufficient condition, not a necessary one; $V_{g,h}^\pi(x)<0$ may not cover all states that truly satisfy reach-avoid, potentially missing feasible points.
- $\phi$ is only trained on successful trajectories; when success rates are extremely low in early training, $\phi$ is severely underfitted, distorting feasible set determination. The paper lacks discussion on cold-start stability.
- Experiments are limited to 5 MuJoCo environments + FrozenLake, lacking high-dimensional visuomotor or autonomous driving benchmarks; simulation noise is simple Gaussian.
- The Forward Invariant assumption for $\mathcal X$ is strong; behaviors outside boundaries are undefined, yet physical boundaries of real robots are often crossed.
- $p$ is manually tuned and lacks an adaptive mechanism for different tasks.

## Related Work & Insights
- **vs. RC-PPO (So 2024)**: RC-PPO uses HJ reachability for minimum-cost reach-avoid but is deterministic only; RAPCPO extends this to stochasticity via Bellman certificates + compensation factors and is more stable in deterministic environments (15-16 points higher reach rate in Table 1).
- **vs. CMDP (CPPO / Sauté / RESPO)**: CMDP uses cumulative cost surrogates, which are misaligned with reach-avoid semantics; RAPCPO directly optimizes reach-avoid probability, and ablations show CMDP is too conservative in state-wise cost settings.
- **vs. CVaR / Chance-constrained**: These characterize return tail risks rather than event probabilities, failing to address the "must satisfy $\ge p$ specification" goal.
- **vs. Barrier / CBF (Ames 2019, Xue 2026)**: These provide formal guarantees but only for specification satisfaction without cost optimization; RAPCPO balances performance and safety.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The trio of max-min-clamped Bellman + compensation factor + symmetric projection effectively addresses specific pain points with strong combinatorial originality.
- Experimental Thoroughness: ⭐⭐⭐ MuJoCo + FrozenLake range is reasonable, but lacks real robot/autonomous driving benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear storyline and notation; operator design motivation is well-explained.
- Value: ⭐⭐⭐⭐ Provides the first stable, trainable baseline for stochastic reach-avoid, directly reusable for the safe RL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning](convergence_of_two-timescale_markovian_stochastic_approximations_with_applicatio.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](../../ICLR2026/reinforcement_learning/solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](../../NeurIPS2025/reinforcement_learning/robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)
- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](../../AAAI2026/reinforcement_learning/deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](../../AAAI2026/reinforcement_learning/good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)

</div>

<!-- RELATED:END -->
