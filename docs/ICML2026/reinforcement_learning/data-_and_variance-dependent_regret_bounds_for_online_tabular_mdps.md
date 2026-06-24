---
title: >-
  [Paper Note] Data- and Variance-dependent Regret Bounds for Online Tabular MDPs
description: >-
  [ICML 2026][Reinforcement Learning][Online MDP] For online episodic tabular MDPs with known transitions, this work designs a unified best-of-both-worlds algorithm based on optimistic follow-the-regularized-leader (OFTRL) with log-barrier. It provides first-order, second-order, and path-length data-dependent regret upper bounds in the adversarial regime, as well as variance-aware gap-independent and gap-dependent polylog bounds in the stochastic regime…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Online MDP"
  - "best-of-both-worlds"
  - "OFTRL"
  - "log-barrier"
  - "data-dependent regret"
  - "variance-dependent regret"
date: 2026-05-08
content_hash: 56a1b650c9afbb70
---

# Data- and Variance-dependent Regret Bounds for Online Tabular MDPs

**Conference**: ICML 2026  
**arXiv**: [2602.01903](https://arxiv.org/abs/2602.01903)  
**Code**: None  
**Area**: Reinforcement Learning / Online Learning / Bandit Theory  
**Keywords**: Online MDP, best-of-both-worlds, OFTRL, log-barrier, data-dependent regret, variance-dependent regret  

## TL;DR
For online episodic tabular MDPs with known transitions, this work designs a unified best-of-both-worlds algorithm based on optimistic follow-the-regularized-leader (OFTRL) with log-barrier. It provides first-order, second-order, and path-length data-dependent regret upper bounds in the adversarial regime, as well as variance-aware gap-independent and gap-dependent polylog bounds in the stochastic regime, complemented by matching lower bounds.

## Background & Motivation

**Background**: Online episodic tabular MDP is a standard abstraction in RL theory—a learner repeatedly interacts with an MDP with $S$ states, $A$ actions, and $H$ layers over $T$ episodes. In each episode, the environment provides a loss function, and the learner observes bandit feedback along the sampled trajectory. Mainstream solvers follow two lines: first, global optimization over the set of all occupancy measures $\Omega(P)$ (minimax optimal but computationally heavy); second, policy optimization at each state (treating each state as a multi-armed bandit, more practical but with an additional $H$ factor in regret). In the adversarial regime, the minimax rate is known to be $\tilde{O}(\sqrt{HSAT})$, while in the stochastic regime, it can reach gap-dependent $O(\log T)$.

**Limitations of Prior Work**: Existing results are fragmented and mutually incompatible. First, best-of-both-worlds algorithms (near-optimal in both regimes) and fine-grained data-dependent bounds (e.g., first-order small-loss $L^\star$) are typically provided by different algorithms, making selection impossible when the environment is unknown. Second, the only data-dependent result in the adversarial regime is the first-order bound; second-order and path-length bounds, mature in bandit literature, remain a gap for MDPs. Third, gap-dependent bounds in the stochastic regime (e.g., Jin et al. 2021) include a $1/\min_{s,a}\Delta(s,a)$ factor and lack variance-aware versions.

**Key Challenge**: Unifying these fine-grained bounds into a single algorithm is difficult because, under bandit feedback in MDPs, loss estimation errors propagate downstream to value estimates along the dynamics. Unlike multi-armed bandits, estimation errors for each state-action pair cannot be controlled independently. It is necessary to design loss and Q-estimators where the bias "aligns" with fine-grained complexity measures to enable self-bounding analysis.

**Goal**: Construct a single algorithm that, under the known-transition setting, simultaneously achieves: (1) first-order, second-order, and path-length data-dependent bounds in the adversarial regime; (2) variance-aware gap-independent and polylog gap-dependent bounds in the stochastic regime; (3) coverage for both global optimization and policy optimization routes; (4) minimax optimality through lower bound proofs.

**Key Insight**: Using OFTRL + log-barrier + adaptive learning rate as the backbone, this work migrates the loss-shifting technique of Jin et al. (2021) from FTRL to the OFTRL framework and switches between two loss predictions (gradient-descent-style vs. empirical-mean-style) to leverage path-length bounds and variance-aware gap-dependent bounds, respectively.

**Core Idea**: Use OFTRL to carry multiple data dependencies—the stability of OFTRL is controlled by the "shifted loss" $\tilde{\ell}_t = \hat{\ell}_t - m_t$. By appropriately choosing $m_t$, the stability term can be made to converge to the required complexity measures.

## Method

### Overall Architecture
Consider a known-transition $H$-layer tabular MDP $M=(\mathcal{S},\mathcal{A},P,H,s_0)$. In each episode $t$, the learner selects a policy $\pi_t$ and interacts along a trajectory. The objective is to minimize regret against all stationary policies:
$\mathrm{Reg}_T = \max_{\pi \in \Pi} \mathbb{E}\bigl[\sum_{t=1}^T V^{\pi_t}(s_0; \ell_t) - V^{\pi}(s_0; \ell_t)\bigr]$.

The work consists of three parts: (i) Section 3 defines new complexity measures; (ii) Sections 4–5 provide global and policy optimization algorithms, respectively, sharing the "OFTRL + log-barrier + adaptive learning rate + loss-shifting" template, differing only in loss/Q-estimators and loss prediction choices; (iii) Section 6 proves four lower bounds $\Omega(\sqrt{SAL^\star})$, $\Omega(\sqrt{SAQ_\infty})$, $\Omega(\sqrt{HV_1})$, and $\Omega(\sqrt{SAV_T})$ through classic hard-instance construction, matching the global optimization upper bounds.

### Key Designs

**1. New Data-dependent Complexity Measures: Translating "loss sequence tractability" into computable quantities**

To make an algorithm automatically adapt to how "easy" the adversarial loss is or how "small" the stochastic loss noise is, these properties must be expressed as computable quantities. In the adversarial regime, this work introduces three measures: first-order small-loss $L^\star = \min_{\pi} \mathbb{E}[\sum_t V^\pi(s_0;\ell_t)]$, second-order $Q_\infty = \min_{\ell^\star} \mathbb{E}[\sum_t \sum_h \|\ell_t(h)-\ell^\star(h)\|_\infty^2]$, and path-length $V_1 = \mathbb{E}[\sum_t \|\ell_{t+1}-\ell_t\|_1]$. In the stochastic regime, occupancy-weighted variance $V = \max_\pi \sum_{s,a} q^\pi(s,a)\sigma^2(s,a)$ and conditional occupancy-weighted variance $V_c(s)$ are introduced. While $Q_\infty$ and $V_1$ are standard in bandit literature, they were previously missing for MDPs. Compared to existing $\mathrm{Var}_{\max}$ and $\mathrm{Var}^c_{\max}$, the measures $V$ and $V_c$ remove redundant $V^{\pi^\star}(s')$ variance terms—precision is possible because transitions are known, resulting in bounds approximately $H^2$ times tighter than existing ones.

**2. Global Optimization: Running OFTRL on occupancy sets with loss-shifting**

The global optimization version (Algorithm 1, Thm 4.1 / 4.2) solves for $q^{\pi_t} = \arg\min_{q\in\Omega(P)}\{\langle q, \sum_{\tau<t}\hat\ell_\tau + m_t\rangle + \psi_t(q)\}$ each episode, where $\psi_t(q) = \sum_{s,a} \tfrac{1}{\eta_t(s,a)}\log(1/q(s,a))$ is a per-coordinate log-barrier, and the learning rate grows adaptively based on stability $\zeta_t$: $1/\eta_{t+1} = 1/\eta_t + \eta_t \zeta_t/\log T$. The loss estimator uses an optimistic IW form $\hat\ell_t(s,a) = m_t(s,a) + I_t(s,a)(\ell_t - m_t)/q^{\pi_t}(s,a)$. The key to fine-grained bounds is the loss-shifting function:

$$g_t(s,a) = Q^{\pi_t}(s,a;\tilde\ell_t) - V^{\pi_t}(s;\tilde\ell_t) - \tilde\ell_t(s,a),$$

which rewrites OFTRL to operate on advantages. Thus, stability is naturally bounded by the second moment of the advantage, yielding polylog gap-dependent bounds through self-bounding analysis. Two loss predictions $m_t$ serve different roles: gradient-descent $m_{t+1}=(1-\xi)m_t+\xi\ell_t$ leverages $V_1$, while empirical-mean $m_t = \sum_\tau I_\tau \ell_\tau / N_{t-1}$ allows stability to converge to $V_c$ for variance-aware gap-dependent bounds.

**3. Policy Optimization + Optimistic Q-estimator: Local updates with similar adaptability**

Global optimization requires solving a convex problem on $\Omega(P)$, which is computationally heavy. A more practical approach is to treat each state as a local bandit solver using per-state closed-form updates: $\pi_t(\cdot|s) = \arg\min_{p\in\Delta(A)} \{\langle p, \sum_{\tau<t}(\hat Q_\tau(s,\cdot) - B_\tau(s,\cdot)) + m_t(s,\cdot)\rangle + \psi_t(p)\}$. The difficulty lies in the estimator: first-order Q-estimators used in FTRL would leave an irreducible bias term when $m_t \neq 0$. This work constructs a "more optimistic" Q-estimator $\hat Q_t$ that applies IW to current losses and injects predictions for future values, such that $\mathbb{E}_t[\hat Q_t - B_t]$ exactly equals the true advantage. With bias precisely canceled, stability analysis can mirror the global optimization route, albeit with an additional $H$ factor in the upper bounds.

### Loss & Training
This is a purely theoretical work; "training" refers to the iterative updates of OFTRL. Shared hyperparameters: $H \le S$ assumption, initial learning rate $1/\eta_1 = 2H$, loss prediction step size $\xi = 1/4$, and log-barrier coefficients growing adaptively with stability $\zeta_t = q^{\pi_t}(s,a)^2 \cdot \min\{(\hat\ell_t-m_t)^2, (\hat\ell_t+g_t-m_t)^2\}$. All regret bounds are "parameter-free" with respect to unknown complexity measures like $L^\star$.

## Key Experimental Results

This is a theoretical work without empirical data; results are presented as theorems. Key comparisons are summarized below (leading terms, omitting log factors; $U = \sum_{s,a\neq\pi^\star(s)} H^2\log(T)/\Delta(s,a)$, $U_{\mathrm{Var}} = \sum_{s,a\neq\pi^\star(s)} HV_c(s)\log(T)/\Delta(s,a)$, $C$ is total adversarial corruption).

### Main Results: Global Optimization Regret Bounds

| Method | Adversarial regime | Stochastic + Corruption regime |
|------|------|------|
| Zimin & Neu (2013) | $\sqrt{HSAT}$ | $\sqrt{HSAT}$ |
| Lee et al. (2020) | $\sqrt{SAL^\star}$ | $\sqrt{SAL^\star}$ |
| Jin et al. (2021) | $\sqrt{HSAT}$ | $U_{\mathrm{Jin}} + \sqrt{U_{\mathrm{Jin}}C}$ (with $1/\min\Delta$) |
| **Ours Thm 4.1** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty, V_1\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **Ours Thm 4.2** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### Ablation Study: Policy Optimization vs. Global Optimization

| Method | Adversarial regime | Stochastic + Corruption regime |
|------|------|------|
| Luo et al. (2021) | $\sqrt{H^3 SAT}$ | $\sqrt{H^3 SAT}$ |
| Dann et al. (2023a) | $\sqrt{H^2 SAL^\star}$ | $U + \sqrt{UC}$ |
| **Ours Thm 5.2** | $\sqrt{H^2 SA \min\{L^\star, HT{-}L^\star, Q_\infty, V_1\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **Ours Thm 5.3** | $\sqrt{H^2 SA \min\{L^\star, HT{-}L^\star, Q_\infty\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### Key Findings
- Global optimization versions are minimax optimal for $L^\star$, $Q_\infty$, and $V_1$. Policy optimization versions differ by a factor of $H$, attributed to the known $H$-gap phenomenon.
- Choice of loss prediction $m_t$ is critical: empirical mean enables $V_c$ but not $V_1$; gradient descent enables $V_1$ but degrades $V_c$ to $V$.
- The gap-dependent polylog bounds (Thm 4.2) are cleaner than Jin et al. (2021), as the $1/\min\Delta$ term is absorbed by variance $V_c$.

## Highlights & Insights
- The combination of OFTRL + log-barrier + adaptive learning rate is a "Swiss Army Knife" for data dependency. Stability $\zeta_t$ naturally aligns with $L^\star, Q_\infty, V_1, V_c$ by simply swapping $m_t$.
- The "more optimistic Q-estimator" is the soul of the policy optimization version. By including a prediction term to cancel bias, stability analysis can be reused.
- Known transitions allow for a more precise variance definition ($V, V_c$), refining results by approximately $H^2$ compared to unknown-transition literature.

## Limitations & Future Work
- **Limitations**: Restricted to known transitions. Unknown transition settings remain an open problem for second-order/path-length/variance-aware adaptability.
- **Limitations**: Policy optimization holds an extra $H$ factor compared to global optimization; closing this gap remains open.
- **Future Work**: Extending OFTRL + log-barrier to linear MDPs or general function approximation.
- **Future Work**: Empirical validation of the theoretical improvements in constants and logarithmic factors for practical regret.

## Related Work & Insights
- **vs. Jin et al. (2021)**: Uses FTRL, limited to first-order and $1/\min\Delta$ dependent bounds. This work adopts OFTRL + specific loss predictions to cover second-order, path-length, and cleaner variance-aware bounds.
- **vs. Dann et al. (2023a)**: First best-of-both-worlds via policy optimization (first-order). This work upgrades the architecture to OFTRL with a more optimistic Q-estimator for a fuller suite of data dependencies.
- **Insight**: Adapting bandit theory to MDPs requires redesigning estimators to align with MDP structures rather than just applying formulas. This methodology is valuable for future exploration of constrained or multi-agent MDPs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)
- [\[ICLR 2026\] Q-Learning with Fine-Grained Gap-Dependent Regret](../../ICLR2026/reinforcement_learning/q-learning_with_fine-grained_gap-dependent_regret.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](../../NeurIPS2025/reinforcement_learning/improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[ICML 2026\] ALSO: Adversarial Online Strategy Optimization for Social Agents](also_adversarial_online_strategy_optimization_for_social_agents.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
