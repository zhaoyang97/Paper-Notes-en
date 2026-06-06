---
title: >-
  [Paper Note] Data- and Variance-dependent Regret Bounds for Online Tabular MDPs
description: >-
  [ICML 2026][Reinforcement Learning][Online MDP] This paper designs a single best-of-both-worlds algorithm based on optimistic follow-the-regularized-leader (OFTRL) with a log-barrier for online episodic tabular MDPs with…
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
content_hash: e6da14c2f58734ba
---

# Data- and Variance-dependent Regret Bounds for Online Tabular MDPs

**Conference**: ICML 2026  
**arXiv**: [2602.01903](https://arxiv.org/abs/2602.01903)  
**Code**: None  
**Area**: Reinforcement Learning / Online Learning / Bandit Theory  
**Keywords**: Online MDP, best-of-both-worlds, OFTRL, log-barrier, data-dependent regret, variance-dependent regret  

## TL;DR
This paper designs a single best-of-both-worlds algorithm based on optimistic follow-the-regularized-leader (OFTRL) with a log-barrier for online episodic tabular MDPs with known transitions. It provides three data-dependent regret bounds (first-order, second-order, and path-length) in adversarial regimes, and variance-aware gap-independent and gap-dependent polylog bounds in stochastic regimes, supported by matching lower bounds.

## Background & Motivation

**Background**: Online episodic tabular MDP is a standard abstraction in RL theory. A learner interacts with an MDP with $S$ states, $A$ actions, and $H$ layers over $T$ episodes. In each episode, the environment provides a loss function, and the learner observes bandit feedback along the sampled trajectory. Mainstream solvers follow two paths: global optimization over the set of occupancy measures $\Omega(P)$ (minimax optimal but computationally heavy), or policy optimization at each state (treating each state as a multi-armed bandit, more practical but with an additional $H$ factor in regret). In adversarial regimes, the minimax rate is $\tilde{O}(\sqrt{HSAT})$, while in stochastic regimes, it can reach gap-dependent $O(\log T)$.

**Limitations of Prior Work**: Existing results are fragmented and mutually incompatible. First, best-of-both-worlds algorithms (near-optimal in both regimes) and fine-grained data-dependent bounds (e.g., first-order small-loss $L^\star$) are usually provided by different algorithms, making selection impossible when the environment is unknown. Second, the only data-dependent result in adversarial regimes is the first-order bound; second-order and path-length bounds, mature in bandit literature, remain a blank in MDPs. Third, gap-dependent bounds in stochastic regimes (e.g., Jin et al. 2021) include an extra $1/\min_{s,a}\Delta(s,a)$ factor and lack variance-aware versions.

**Key Challenge**: The difficulty of unifying these refined bounds into a single algorithm lies in the propagation of loss estimation errors along the dynamics to downstream value estimates under bandit feedback. Unlike multi-armed bandits, estimation errors for each state-action pair cannot be controlled independently. Loss and Q-estimators must be designed to "align" biases with fine-grained complexity measures to enable self-bounding analysis.

**Goal**: Construct a single algorithm that simultaneously achieves: (1) Three data-dependent bounds (first-order, second-order, path-length) in adversarial regimes; (2) Variance-aware gap-independent and polylog gap-dependent bounds in stochastic regimes; (3) Coverage of both global optimization and policy optimization routes; (4) Minimax optimality proven through lower bounds.

**Key Insight**: Using OFTRL + log-barrier + adaptive learning rate as the backbone, this work transfers the loss-shifting technique for FTRL from Jin et al. (2021) to the OFTRL framework. It switches between two types of loss predictions (gradient-descent vs. empirical mean) to leverage path-length bounds and variance-aware gap-dependent bounds, respectively.

**Core Idea**: Use OFTRL to carry multiple data dependencies. The stability of OFTRL is controlled by the "shifted loss" $\tilde{\ell}_t = \hat{\ell}_t - m_t$. By appropriately choosing $m_t$, the stability term can converge to the required complexity measures.

## Method

### Overall Architecture
Consider a fixed $H$-layer tabular MDP $M=(\mathcal{S},\mathcal{A},P,H,s_0)$ with known transitions. In each episode $t$, the learner selects a policy $\pi_t$ and interacts along a trajectory. The goal is to minimize regret against all stationary policies:
$\mathrm{Reg}_T = \max_{\pi \in \Pi} \mathbb{E}\bigl[\sum_{t=1}^T V^{\pi_t}(s_0; \ell_t) - V^{\pi}(s_0; \ell_t)\bigr]$.

The work consists of three parts: (i) Section 3 defines new complexity measures; (ii) Sections 4–5 provide global and policy optimization algorithms, respectively, sharing the "OFTRL + log-barrier + adaptive learning rate + loss-shifting" template, differing only in estimators and loss predictions; (iii) Section 6 proves lower bounds of $\Omega(\sqrt{SAL^\star})$, $\Omega(\sqrt{SAQ_\infty})$, $\Omega(\sqrt{HV_1})$, and $\Omega(\sqrt{SAV_T})$ through classic hard instance construction, matching the global optimization upper bounds.

### Key Designs

1. **New Data-Dependent Complexity Measures**:
    - **Function**: Precisely characterize "how easy the adversarial loss sequence is" and "how small the stochastic loss noise is" using computable quantities, which OFTRL then automatically adapts to.
    - **Mechanism**: In adversarial regimes, three measures are introduced: first-order small-loss $L^\star = \min_{\pi} \mathbb{E}[\sum_t V^\pi(s_0;\ell_t)]$; second-order $Q_\infty = \min_{\ell^\star} \mathbb{E}[\sum_t \sum_h \|\ell_t(h)-\ell^\star(h)\|_\infty^2]$, which decreases when losses fluctuate around a baseline; and path-length $V_1 = \mathbb{E}[\sum_t \|\ell_{t+1}-\ell_t\|_1]$, which decreases when the loss sequence changes slowly. In stochastic regimes, occupancy-weighted variance $V = \max_\pi \sum_{s,a} q^\pi(s,a)\sigma^2(s,a)$ and conditional occupancy-weighted variance $V_c(s)$ are introduced to characterize noise remaining after reaching $s$.
    - **Design Motivation**: $Q_\infty$ and $V_1$ are standard in bandit literature but were previously absent in MDPs. Compared to existing $\mathrm{Var}_{\max}$ or $\mathrm{Var}^c_{\max}$, $V$ and $V_c$ remove redundant $V^{\pi^\star}(s')$ variance terms. Thanks to the known transition setting, these measures are $H^2$ times tighter.

2. **Global Optimization Algorithm (Algorithm 1, Theorem 4.1 / 4.2)**:
    - **Function**: Operates OFTRL directly on the occupancy set $\Omega(P)$ to obtain adversarial bounds of $\tilde{O}(\sqrt{SA \cdot \min\{L^\star, HT-L^\star, Q_\infty, V_1\}})$ and stochastic bounds of $\tilde{O}(\sqrt{SAV_T})$ and $\mathrm{polylog}(T)$.
    - **Mechanism**: Each episode solves $q^{\pi_t} = \arg\min_{q\in\Omega(P)}\{\langle q, \sum_{\tau<t}\hat\ell_\tau + m_t\rangle + \psi_t(q)\}$, where $\psi_t(q) = \sum_{s,a} \tfrac{1}{\eta_t(s,a)}\log(1/q(s,a))$ is a per-coord log-barrier, and the learning rate $1/\eta_{t+1} = 1/\eta_t + \eta_t \zeta_t/\log T$ grows adaptively based on the stability term $\zeta_t$. The loss estimator uses an optimistic IW form $\hat\ell_t(s,a) = m_t(s,a) + I_t(s,a)(\ell_t - m_t)/q^{\pi_t}(s,a)$. The key loss-shifting function $g_t(s,a) = Q^{\pi_t}(s,a;\tilde\ell_t) - V^{\pi_t}(s;\tilde\ell_t) - \tilde\ell_t(s,a)$ equivalently modifies OFTRL to run on advantages. Stability is naturally bounded by the second moment of advantages, leading to polylog gap-dependent bounds via self-bounding. Two $m_t$ types are used: GD-style $m_{t+1}=(1-\xi)m_t+\xi\ell_t$ ($\xi=1/4$) for path-length $V_1$, and empirical mean $m_t = \sum_\tau I_\tau \ell_\tau / N_{t-1}$ for variance-aware gap-dependent bounds.
    - **Design Motivation**: Migrating the FTRL+log-barrier route from Jin et al. (2021) to OFTRL. The additional $m_t$ term provides "optimism"—when predictions are accurate, stability is nearly zero, yielding refined $V_1$ and $V_c$ bounds; when inaccurate, it degrades to the $\sqrt{HSAT}$ worst case.

3. **Policy Optimization Algorithm (Theorem 5.2 / 5.3) + More Optimistic Q-estimator**:
    - **Function**: Uses OFTRL at each state as a local bandit solver, achieving data/variance-dependent bounds $\tilde{O}(\sqrt{H^2 SA \cdot \min\{L^\star,\ldots,V_1\}})$ similar to global optimization but with an extra $H$ factor. It features per-state closed-form updates, making it computationally friendly.
    - **Mechanism**: OFTRL is run on each $s$: $\pi_t(\cdot|s) = \arg\min_{p\in\Delta(A)} \{\langle p, \sum_{\tau<t}(\hat Q_\tau(s,\cdot) - B_\tau(s,\cdot)) + m_t(s,\cdot)\rangle + \psi_t(p)\}$. Here, $\hat Q_t$ is a new "more optimistic" Q-estimator: it applies IW to the current loss and injects future value predictions into the estimate to cancel the bias introduced by OFTRL loss prediction. While Dann et al. (2023a) used first-order Q-estimators in FTRL policy optimization, moving to OFTRL leaves an extra bias term when $m_t \neq 0$. This new construction ensures $\mathbb{E}_t[\hat Q_t - B_t]$ equals the true advantage, allowing the stability analysis to reuse the global optimization route.
    - **Design Motivation**: Policy optimization is more practical, but prior results were limited to first-order bounds. To achieve second-order, path-length, and variance-aware bounds, one must resolve the compatibility between optimistic predictions, per-state local updates, and layer-by-layer value estimation.

### Loss & Training
This is a purely theoretical work; "training" refers to the iterative updates of OFTRL. Shared hyperparameters: assumption $H \le S$, initial learning rate $1/\eta_1 = 2H$, loss prediction step size $\xi = 1/4$, and log-barrier coefficients growing adaptively with the stability term $\zeta_t = q^{\pi_t}(s,a)^2 \cdot \min\{(\hat\ell_t-m_t)^2, (\hat\ell_t+g_t-m_t)^2\}$. All regret upper bounds are "parameter-free" with respect to unknown complexity measures—the algorithm does not require prior knowledge of $L^\star, Q_\infty, V_1, V, V_c$.

## Key Experimental Results

As a theoretical work, there is no experimental data. Results are provided as theorems and tables. Two key comparison tables are summarized below (showing leading terms, omitting logarithmic factors; $U = \sum_{s,a\neq\pi^\star(s)} H^2\log(T)/\Delta(s,a)$, $U_{\mathrm{Var}} = \sum_{s,a\neq\pi^\star(s)} HV_c(s)\log(T)/\Delta(s,a)$, and $C$ is the total adversarial corruption).

### Main Results: Comparison of Global Optimization Regret Bounds
| Method | Adversarial Regime | Stochastic + Adversarial Corruption |
|------|------|------|
| Zimin & Neu (2013) | $\sqrt{HSAT}$ | $\sqrt{HSAT}$ |
| Lee et al. (2020) | $\sqrt{SAL^\star}$ | $\sqrt{SAL^\star}$ |
| Jin et al. (2021) | $\sqrt{HSAT}$ | $U_{\mathrm{Jin}} + \sqrt{U_{\mathrm{Jin}}C}$ (incl. $1/\min\Delta$) |
| **Ours (Thm 4.1)** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty, V_1\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **Ours (Thm 4.2)** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### Ablation Study: Policy Optimization vs. Global Optimization (Similar adaptability, extra $H$)
| Method | Adversarial Regime | Stochastic + Corruption |
|------|------|------|
| Luo et al. (2021) | $\sqrt{H^3 SAT}$ | $\sqrt{H^3 SAT}$ |
| Dann et al. (2023a) | $\sqrt{H^2 SAL^\star}$ | $U + \sqrt{UC}$ |
| **Ours (Thm 5.2)** | $\sqrt{H^2 SA \min\{L^\star, \ldots, V_1\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **Ours (Thm 5.3)** | $\sqrt{H^2 SA \min\{L^\star, \ldots, Q_\infty\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### Key Findings
- Section 6 constructs hard instances to prove lower bounds of $\Omega(\sqrt{SAL^\star})$, $\Omega(\sqrt{SA Q_\infty})$, $\Omega(\sqrt{H V_1})$, and $\Omega(\sqrt{SA V_T})$, meaning the global optimization version is minimax optimal for $L^\star, Q_\infty, V_1$. The variance-aware gap-independent bound is also off only by logarithmic factors.
- Two types of loss predictions must be used separately: empirical mean prediction $m_t$ leverages $V_c$ but cannot give $V_1$; GD-style prediction gives $V_1$ but $V_c$ degrades to the looser $V$.
- The gap-dependent polylog bound (Thm 4.2) is cleaner than Jin et al. (2021)—the latter has an extra $H^3 S \log T / \min_{s,a\neq\pi^\star(s)} \Delta(s,a)$ term, which is absorbed in this work by replacing $H^2$ with variance $V_c$, implying real polylog improvements for MDPs with small variance.

## Highlights & Insights
- The combination of OFTRL + log-barrier + adaptive learning rate is a "Swiss Army Knife" for fitting multiple data dependencies into one algorithm. The quadratic form of the stability term $\zeta_t \propto q^{\pi_t}^2(\hat\ell_t-m_t)^2$ is naturally compatible with four complexity measures; simply changing $m_t$ changes adaptability without rewriting the algorithm. This "skeletal + plugin" idea can be migrated to other bandit-style RL problems.
- The "more optimistic Q-estimator" is the soul of the policy optimization version. The Q-estimator from the FTRL route leaves a non-vanishing bias when moved to OFTRL due to $m_t$. This work adds a prediction term to the estimator to precisely cancel the bias, providing a reference for future extensions of OFTRL to unknown transitions or function approximation.
- Redefining variance as $V, V_c$ by removing the $\mathrm{Var}_{s'\sim P}[V^{\pi^\star}(s')]$ term in known transition settings makes the variance bounds $H^2$ times tighter. This suggests that in refined regret analysis, "assumption strength" and "measure definition" should be coupled.

## Limitations & Future Work
- **Limitations**: Restricted to known transitions. Simultaneous control of loss and transition estimation errors for second-order/path-length/variance-aware adaptability remains an open problem for both global and policy optimization.
- The policy optimization upper bounds have an extra $H$ factor compared to global optimization and do not match the lower bounds; whether this $H$ can be removed is still open.
- Computational complexity: Global optimization requires solving convex optimization with log-barriers on $\Omega(P)$ at each step, which might be costly in practice.
- The results only cover tabular MDPs. Combining OFTRL + log-barrier with linear MDPs or general function approximation is a potential direction.
- As a theoretical paper, it lacks empirical verification to see if theoretically tighter constants translate to smaller actual regret.

## Related Work & Insights
- **vs Jin et al. (2021)**: Both use occupancy + log-barrier, but they use FTRL and only provide first-order small-loss bounds and gap-dependent bounds with $1/\min\Delta$ factors. This work uses OFTRL with two loss predictions to include second-order, path-length, and variance-aware bounds while removing the $1/\min\Delta$ term.
- **vs Dann et al. (2023a)**: They provided the first best-of-both-worlds algorithm under policy optimization (first-order $\sqrt{H^2 SAL^\star}$). This work upgrades the architecture to OFTRL and designs a more optimistic Q-estimator to obtain a full suite of second-order, path-length, and variance-aware bounds.
- **vs Wei & Luo (2018) / Ito et al. (2022a)**: They obtained second-order/path-length bounds using OFTRL + log-barrier in multi-armed bandit settings. This work pushes this technical route to tabular MDPs for the first time, handling loss estimation error propagation caused by MDP dynamics using the new $g_t$ function and optimistic Q-estimator.

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
