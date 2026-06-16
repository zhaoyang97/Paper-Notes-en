---
title: >-
  [Paper Note] Data- and Variance-dependent Regret Bounds for Online Tabular MDPs
description: >-
  [ICML 2026][Reinforcement Learning][best-of-both-worlds] This paper designs a single best-of-both-worlds algorithm based on Optimistic Follow-the-Regularized-Leader (OFTRL) with a log-barrier for online episodic tabular MDPs with known transitions. It provides three types of data-dependent regret upper bounds (first-order, second-order, and path-length) in the adversarial re
tags:
  - ICML 2026
  - Reinforcement Learning
  - best-of-both-worlds
  - OFTRL
  - log-barrier
date: 2026-05-08
content_hash: 0ff81d235874a693
---
# Data- and Variance-dependent Regret Bounds for Online Tabular MDPs

**Conference**: ICML 2026  
**arXiv**: [2602.01903](https://arxiv.org/abs/2602.01903)  
**Code**: None  
**Area**: Reinforcement Learning / Online Learning / Bandit Theory  
**Keywords**: Online MDP, best-of-both-worlds, OFTRL, log-barrier, data-dependent regret, variance-dependent regret  

## TL;DR
This paper designs a single best-of-both-worlds algorithm based on Optimistic Follow-the-Regularized-Leader (OFTRL) with a log-barrier for online episodic tabular MDPs with known transitions. It provides three types of data-dependent regret upper bounds (first-order, second-order, and path-length) in the adversarial regime, as well as variance-aware gap-independent and gap-dependent polylogarithmic bounds in the stochastic regime, supported by matching lower bounds.

## Background & Motivation

**Background**: Online episodic tabular MDP is a standard abstraction in RL theory—a learner interacts repeatedly with an MDP with $S$ states, $A$ actions, and $H$ layers over $T$ episodes. In each episode, the environment provides a loss function, and the learner observes only bandit feedback along the sampled trajectory. Mainstream solvers follow two lines: first, global optimization over the set of all occupancy measures $\Omega(P)$ (minimax optimal but computationally heavy); second, policy optimization at each state (treating each state as a multi-armed bandit, more practical but incurs an extra $H$ factor in regret). In the adversarial regime, the minimax rate is $\tilde{O}(\sqrt{HSAT})$, while in the stochastic regime, gap-dependent $O(\log T)$ rates are achievable.

**Limitations of Prior Work**: Existing results are fragmented and mutually incompatible. First, best-of-both-worlds algorithms (performing near-optimally in both regimes) and fine-grained data-dependent bounds (e.g., first-order small-loss $L^\star$) are usually provided by different algorithms, making selection difficult when the environment is unknown. Second, the only data-dependent result in the adversarial regime is the first-order bound; second-order and path-length bounds, which are mature in bandit literature, remain a gap in MDPs. Third, gap-dependent bounds in the stochastic regime (e.g., Jin et al. 2021) include an extra $1/\min_{s,a}\Delta(s,a)$ factor and lack variance-aware versions.

**Key Challenge**: Unifying these fine-grained bounds into a single algorithm is difficult because loss estimation errors under bandit feedback propagate downstream to value estimates along the dynamics. Unlike multi-armed bandits, estimation errors for each state-action pair cannot be controlled independently. One must design loss and Q-estimators whose bias "aligns" with fine-grained complexity measures to ensure the self-bounding analysis succeeds.

**Goal**: Construct a single algorithm that simultaneously achieves: (1) three data-dependent bounds (first-order, second-order, and path-length) in the adversarial regime; (2) variance-aware gap-independent and polylogarithmic gap-dependent bounds in the stochastic regime; (3) coverage of both global optimization and policy optimization routes; (4) minimax optimality through matching lower bounds.

**Key Insight**: Using OFTRL + log-barrier + adaptive learning rate as the backbone, this work transfers the loss-shifting techniques from FTRL (Jin et al. 2021) to the OFTRL framework. It switches between two types of loss predictions (gradient-descent-style vs. empirical-mean-style) to trigger path-length bounds and variance-aware gap-dependent bounds, respectively.

**Core Idea**: Use OFTRL to carry multiple data dependencies—the stability of OFTRL is controlled by the "shifted loss" $\tilde{\ell}_t = \hat{\ell}_t - m_t$. By appropriately choosing $m_t$, the stability term can be made to converge to the required complexity measures.

## Method

### Overall Architecture
Consider a known-transition $H$-layer tabular MDP $M=(\mathcal{S},\mathcal{A},P,H,s_0)$. In each episode $t$, the learner selects a policy $\pi_t$ and interacts along a trajectory to minimize the regret relative to all stationary policies:
$\mathrm{Reg}_T = \max_{\pi \in \Pi} \mathbb{E}\bigl[\sum_{t=1}^T V^{\pi_t}(s_0; \ell_t) - V^{\pi}(s_0; \ell_t)\bigr]$.

The work consists of three parts: (i) Section 3 defines new complexity measures; (ii) Sections 4–5 provide global optimization and policy optimization algorithms, respectively, sharing the "OFTRL + log-barrier + adaptive learning rate + loss-shifting" template, differing only in the choice of loss/Q-estimators and loss predictions; (iii) Section 6 proves four lower bounds $\Omega(\sqrt{SAL^\star})$, $\Omega(\sqrt{SAQ_\infty})$, $\Omega(\sqrt{HV_1})$, and $\Omega(\sqrt{SAV_T})$ using hard instance constructions to match the global optimization upper bounds.

### Key Designs

**1. New Data-dependent Complexity Measures: Translating loss sequence "tractability" into computable quantities**

To make an algorithm automatically adapt to "how easy the adversarial loss is" and "how small the stochastic loss noise is," these properties must be written as computable quantities. For the adversarial regime, three measures are introduced: first-order small-loss $L^\star = \min_{\pi} \mathbb{E}[\sum_t V^\pi(s_0;\ell_t)]$ (easier when expert cumulative loss is small), second-order $Q_\infty = \min_{\ell^\star} \mathbb{E}[\sum_t \sum_h \|\ell_t(h)-\ell^\star(h)\|_\infty^2]$ (smaller when losses fluctuate around a baseline), and path-length $V_1 = \mathbb{E}[\sum_t \|\ell_{t+1}-\ell_t\|_1]$ (smaller when the loss sequence changes slowly). For the stochastic regime, occupancy-weighted variance $V = \max_\pi \sum_{s,a} q^\pi(s,a)\sigma^2(s,a)$ and conditional occupancy-weighted variance $V_c(s)$ are introduced. $Q_\infty$ and $V_1$ are standard in bandit literature but were previously missing in MDPs; $V$ and $V_c$ remove the redundant $V^{\pi^\star}(s')$ variance terms compared to existing $\mathrm{Var}_{\max}$ measures, resulting in bounds approximately $H^2$ times tighter.

**2. Global Optimization Algorithm: OFTRL over occupancy sets with loss-shifting**

Algorithm 1 (Theorems 4.1/4.2) solves for $q^{\pi_t} = \arg\min_{q\in\Omega(P)}\{\langle q, \sum_{\tau<t}\hat\ell_\tau + m_t\rangle + \psi_t(q)\}$ the occupancy measure set $\Omega(P)$ each episode, where $\psi_t(q) = \sum_{s,a} \tfrac{1}{\eta_t(s,a)}\log(1/q(s,a))$ is a per-coordinate log-barrier. The learning rate grows adaptively based on the stability term $\zeta_t$ as $1/\eta_{t+1} = 1/\eta_t + \eta_t \zeta_t/\log T$. The loss estimator uses an optimistic Importance Weighting (IW) form: $\hat\ell_t(s,a) = m_t(s,a) + I_t(s,a)(\ell_t - m_t)/q^{\pi_t}(s,a)$. The key to fine-grained bounds is the loss-shifting function:

$$g_t(s,a) = Q^{\pi_t}(s,a;\tilde\ell_t) - V^{\pi_t}(s;\tilde\ell_t) - \tilde\ell_t(s,a),$$

which reformulates OFTRL as rolling on advantages. Consequently, the stability term is naturally bounded by the second moment of the advantage, enabling polylogarithmic gap-dependent bounds via self-bounding analysis. Two types of $m_t$ are used: gradient-descent-style $m_{t+1}=(1-\xi)m_t+\xi\ell_t$ triggers $V_1$, while empirical-mean-style $m_t = \sum_\tau I_\tau \ell_\tau / N_{t-1}$ lets stability converge to $V_c$ for variance-aware gap-dependent bounds.

**3. Policy Optimization Algorithm + Optimistic Q-estimator: Local updates with similar adaptability**

To reduce the computational burden of global optimization, policy optimization (Theorems 5.2/5.3) treats each state as a local bandit solver using per-state closed-form updates: $\pi_t(\cdot|s) = \arg\min_{p\in\Delta(A)} \{\langle p, \sum_{\tau<t}(\hat Q_\tau(s,\cdot) - B_\tau(s,\cdot)) + m_t(s,\cdot)\rangle + \psi_t(p)\}$. A "more optimistic" Q-estimator $\hat Q_t$ is constructed, which applies IW to the current loss and injects future value predictions so that $\mathbb{E}_t[\hat Q_t - B_t]$ exactly equals the true advantage. This cancels the bias, allowing the stability analysis from global optimization to be reused. The trade-off is an extra $H$ factor in all bounds (e.g., $\tilde{O}(\sqrt{H^2 SA \cdot \min\{L^\star,\ldots,V_1\}})$), but it provides layer-by-layer, computationally friendly updates.

## Loss & Training
"Training" refers to the iterative OFTRL updates. Shared hyperparameters: $H \le S$ assumption, initial learning rate $1/\eta_1 = 2H$, loss prediction step $\xi = 1/4$, and log-barrier coefficients growing with the stability term $\zeta_t = q^{\pi_t}(s,a)^2 \cdot \min\{(\hat\ell_t-m_t)^2, (\hat\ell_t+g_t-m_t)^2\}$. All regret bounds are "parameter-free" with respect to unknown complexity measures—the algorithm does not need prior knowledge of $L^\star, Q_\infty, V_1, V, \text{ or } V_c$.

## Key Experimental Results

### Main Results: Comparison of Global Optimization Regret Bounds
| Method | Adversarial Regime | Stochastic + Adversarial Corruption Regime |
|------|------|------|
| Zimin & Neu (2013) | $\sqrt{HSAT}$ | $\sqrt{HSAT}$ |
| Lee et al. (2020) | $\sqrt{SAL^\star}$ | $\sqrt{SAL^\star}$ |
| Jin et al. (2021) | $\sqrt{HSAT}$ | $U_{\mathrm{Jin}} + \sqrt{U_{\mathrm{Jin}}C}$, contains $1/\min\Delta$ |
| **Ours (Thm 4.1)** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty, V_1\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **Ours (Thm 4.2)** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### Ablation Study: Policy Optimization vs. Global Optimization (Similar adaptability, extra $H$)
| Method | Adversarial Regime | Stochastic + Corruption Regime |
|------|------|------|
| Luo et al. (2021) | $\sqrt{H^3 SAT}$ | $\sqrt{H^3 SAT}$ |
| Dann et al. (2023a) | $\sqrt{H^2 SAL^\star}$ | $U + \sqrt{UC}$ |
| **Ours (Thm 5.2)** | $\sqrt{H^2 SA \min\{L^\star, HT{-}L^\star, Q_\infty, V_1\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **Ours (Thm 5.3)** | $\sqrt{H^2 SA \min\{L^\star, HT{-}L^\star, Q_\infty\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### Key Findings
- Lower bounds $\Omega(\sqrt{SAL^\star})$, $\Omega(\sqrt{SA Q_\infty})$, $\Omega(\sqrt{H V_1})$, and $\Omega(\sqrt{SA V_T})$ indicate that the global optimization version is minimax optimal for $L^\star, Q_\infty, \text{ and } V_1$.
- The two types of loss predictions must be used separately: empirical mean $m_t$ triggers $V_c$ but not $V_1$; gradient descent $m_t$ triggers $V_1$ but makes $V_c$ degrade to $V$.
- The gap-dependent polylogarithmic bound (Thm 4.2) is cleaner than Jin et al. (2021), as the $H^3 S \log T / \min \Delta$ term is absorbed by using variance $V_c$ instead of $H^2$, yielding real polylogarithmic improvements for low-variance MDPs.

## Highlights & Insights
- The combination of OFTRL + log-barrier + adaptive learning rate acts as a "Swiss army knife" for data dependency. The quadratic form of the stability term $\zeta_t \propto q^{\pi_t}^2(\hat\ell_t-m_t)^2$ is naturally compatible with $L^\star, Q_\infty, V_1, \text{ and } V_c$—switching $m_t$ changes the adaptability without rewriting the algorithm.
- The "more optimistic Q-estimator" is the soul of the policy optimization version. It precisely cancels the bias introduced by $m_t$ in the OFTRL setting, offering a methodology for extending OFTRL to more complex RL settings like unknown transitions or function approximation.
- Redefining variance as $V$ and $V_c$ (excluding the $\mathrm{Var}_{s'\sim P}[V^{\pi^\star}(s')]$ term) in known-transition settings tightens variance bounds by $H^2$, suggesting that complexity measures should be tightly coupled with the strength of the assumptions.

## Limitations & Future Work
- Restricted to known transitions. Extending this to unknown transitions while maintaining second-order/path-length/variance adaptability remains an open problem.
- Policy optimization bounds incur an extra $H$ factor compared to global optimization. Whether this $H$-gap can be eliminated remains unknown.
- Computational complexity: Global optimization requires solving a convex optimization with log-barrier at each step, which might be costly in practice.
- The results cover only tabular MDPs; combining OFTRL + log-barrier with linear MDPs or general function approximation remains a future direction.

## Related Work & Insights
- **vs. Jin et al. (2021)**: They use FTRL and provide only first-order and $1/\min\Delta$-dependent bounds; this work uses OFTRL + two $m_t$ predictions to include second-order, path-length, and better variance-aware bounds.
- **vs. Dann et al. (2023a)**: This work upgrades their policy optimization architecture to OFTRL and introduces a more optimistic Q-estimator.
- **vs. Wei & Luo (2018) / Ito et al. (2022a)**: They obtained second-order/path-length bounds in multi-armed bandits via OFTRL + log-barrier; this work extends that technical route to tabular MDPs by resolving error propagation through the $g_t$ loss-shifting function.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](../../NeurIPS2025/reinforcement_learning/improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[ICML 2026\] ALSO: Adversarial Online Strategy Optimization for Social Agents](also_adversarial_online_strategy_optimization_for_social_agents.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)

</div>

<!-- RELATED:END -->
