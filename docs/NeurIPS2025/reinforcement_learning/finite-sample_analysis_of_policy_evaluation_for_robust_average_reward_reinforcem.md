---
title: >-
  [Paper Note] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Robust Reinforcement Learning] This work provides the first finite-sample complexity analysis for policy evaluation in robust average reward MDPs. By constructing a carefully design…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Robust Reinforcement Learning"
  - "Average Reward MDP"
  - "Policy Evaluation"
  - "Finite-Sample Analysis"
  - "Semi-norm Contraction"
date: 2026-05-08
content_hash: 3860bc305b9ec013
---

# Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2502.16816](https://arxiv.org/abs/2502.16816)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Robust Reinforcement Learning, Average Reward MDP, Policy Evaluation, Finite-Sample Analysis, Semi-norm Contraction

## TL;DR
This work provides the first finite-sample complexity analysis for policy evaluation in robust average reward MDPs. By constructing a carefully designed semi-norm, it proves that the robust Bellman operator is a contraction, and combines this with a truncated Multi-Level Monte Carlo (MLMC) estimator to achieve finite expected sample complexity, ultimately attaining an order-optimal sample complexity of $\tilde{\mathcal{O}}(\epsilon^{-2})$.

## Background & Motivation

**Background**: Robust RL addresses the sim-to-real gap and related challenges by performing worst-case optimization over uncertainty sets on transition probabilities. In the discounted reward setting, the robust Bellman operator naturally enjoys a contraction property under the sup-norm due to the discount factor $\gamma < 1$, and finite-sample analyses are relatively mature.

**Limitations of Prior Work**: The average reward setting is better suited for applications requiring long-term sustained efficiency (e.g., queuing systems, inventory management, network control). However, **even the non-robust average reward Bellman operator fails to be a contraction under any norm**, rendering standard fixed-point iteration analysis inapplicable. As a result, existing work on robust average reward RL provides only asymptotic convergence guarantees (based on ODE analysis), without finite-sample complexity bounds.

**Key Challenge**: The natural contraction mechanism provided by $\gamma < 1$ in the discounted setting is entirely absent in the average reward setting. The min operation introduced by robustness further compounds the analytical difficulty—contraction must be established simultaneously across all transition models in the uncertainty set.

**Goal**: (a) In what sense does the robust average reward Bellman operator admit a contraction? (b) How can one estimate the support functions involving nonlinear worst-case transition effects with finite samples? (c) What is the resulting sample complexity?

**Key Insight**: The paper treats the worst-case transition matrices across all uncertainty sets as a family of linear maps, exploits the fact that their joint spectral radius is strictly less than 1, and constructs an extremal norm—from which a semi-norm enabling one-step contraction is derived.

**Core Idea**: A semi-norm is constructed via extremal norm combined with quotient-space correction to prove contraction of the robust Bellman operator, and truncated MLMC is employed to achieve finite-sample policy evaluation.

## Method

### Overall Architecture
Given a policy $\pi$ and a nominal model $\tilde{\mathsf{P}}$ (the center of the uncertainty set), the goal is to estimate the robust value function $V$ and the robust average reward $g$ using only samples from the nominal model. The approach proceeds in two phases: (1) stochastic approximation iterations to estimate $V_T$; (2) using $V_T$ to estimate $g_T$. The central technical challenge is constructing an estimator for the robust Bellman operator with finite samples and controlled bias/variance.

### Key Designs

1. **Semi-norm Contraction Construction (Core Theoretical Contribution)**:

    - Function: Proves that the robust Bellman operator $\mathbf{T}_g(V)(s) = \sum_a \pi(a|s)[r(s,a) - g + \sigma_{\mathcal{P}_s^a}(V)]$ is a contraction under a certain semi-norm.
    - Mechanism (non-robust case): For a single transition matrix $\mathsf{P}^{\pi}$, ergodicity yields a unique stationary distribution $d^\pi$. Define the fluctuation matrix $Q^\pi = \mathsf{P}^\pi - \mathbf{e}^\top d^\pi$, whose eigenvalues lie strictly inside the unit disk. A norm $\|\cdot\|_Q$ is constructed via the discrete Lyapunov equation such that $\|Q^\pi x\|_Q \leq \alpha \|x\|_Q$ for $\alpha < 1$. The semi-norm is defined as $\|x\|_{\mathsf{P}} = \|Q^\pi x\|_Q + \epsilon \inf_{c \in \mathbb{R}} \|x - c\mathbf{e}\|_Q$, with kernel equal to the space of constant vectors.
    - Mechanism (robust case): For the family of fluctuation matrices $\{Q_\mathsf{P}^\pi\}$ over all $\mathsf{P} \in \mathcal{P}$, the joint spectral radius is strictly less than 1. By the Berger–Wang theorem, an extremal norm $\|\cdot\|_{\text{ext}}$ is constructed such that all $Q_\mathsf{P}^\pi$ share a uniform contraction factor $\alpha$. The resulting semi-norm is $\|x\|_{\mathcal{P}} = \sup_{\mathsf{P} \in \mathcal{P}} \|Q_\mathsf{P}^\pi x\|_{\text{ext}} + \epsilon \inf_{c \in \mathbb{R}} \|x - c\mathbf{e}\|_{\text{ext}}$, guaranteeing $\|\mathbf{T}_g(V_1) - \mathbf{T}_g(V_2)\|_{\mathcal{P}} \leq \gamma \|V_1 - V_2\|_{\mathcal{P}}$ with $\gamma = \alpha + \epsilon < 1$.
    - Design Motivation: This is the only theoretical pathway capable of overcoming the absence of discount-factor contraction in the average reward setting.

2. **Truncated MLMC Estimator (Technical Contribution)**:

    - Function: Constructs finite-sample unbiased/low-bias estimators for the support functions $\sigma_{\mathcal{P}_s^a}(V)$ under TV and Wasserstein uncertainty sets.
    - Mechanism: Standard MLMC samples the level $N$ from a geometric distribution $\text{Geom}(\Psi)$ and requires $2^{N+1}$ samples; when $\Psi < 0.5$, the expected sample count is infinite. The key innovation is setting $\Psi = 0.5$ and truncating $N' = \min\{N, N_{\max}\}$, so that the expected sample count becomes $\mathbb{E}[M] = N_{\max} + 2 = \mathcal{O}(N_{\max})$ (linear rather than exponential growth).
    - Bias decays exponentially at rate $2^{-N_{\max}/2}$; variance grows linearly with $N_{\max}$.
    - Design Motivation: Prior MLMC approaches incur infinite expected sample complexity, yielding only asymptotic convergence guarantees.

3. **Treatment of Three Uncertainty Set Classes**:

    - **Contamination model**: $\sigma_{\mathcal{P}_s^a}(V) = (1-\delta)(\tilde{\mathsf{P}}_s^a)^\top V + \delta \min_s V(s)$, which is linear in the nominal transition and can be estimated unbiasedly with a single sample.
    - **Total Variation (TV)**: The support function involves a dual optimization over the span semi-norm, requiring MLMC. Lipschitz property: $|\sigma_{\mathcal{P}_{TV}}(V) - \sigma_{\mathcal{Q}_{TV}}(V)| \leq (1+1/\delta)\|V\|_{\text{sp}}\|p-q\|_1$.
    - **Wasserstein distance**: The support function involves a two-level inf-sup optimization, also requiring MLMC. The Lipschitz constant is tighter: $|\sigma_{\mathcal{P}_W}(V) - \sigma_{\mathcal{Q}_W}(V)| \leq \|V\|_{\text{sp}}\|p-q\|_1$.

4. **Robust Average Reward TD Learning (Algorithm 2)**:

    - Phase 1 ($T$ iterations): $V_{t+1}(s) \leftarrow V_t(s) + \eta_t(\hat{\mathbf{T}}_{g_0}(V_t)(s) - V_t(s))$, followed by centering $V_{t+1}(s) = V_{t+1}(s) - V_{t+1}(s_0)$.
    - Phase 2 ($T$ iterations): Estimate $g_T$ using $V_T$ via $g_T \leftarrow g_t + \beta_t(\bar{\delta}_t - g_t)$.

### Sample Complexity Results
- Contamination uncertainty set: $\mathcal{O}\left(\frac{SAt_{\text{mix}}^2}{\epsilon^2(1-\gamma)^2}\right)$
- TV and Wasserstein: $\tilde{\mathcal{O}}\left(\frac{SAt_{\text{mix}}^2}{\epsilon^2(1-\gamma)^2}\right)$
- All results are order-optimal in $\epsilon$: $\tilde{\mathcal{O}}(\epsilon^{-2})$.

## Key Experimental Results

This is a purely theoretical work with no numerical experiments. The main results are stated as sample complexity bounds in theorem form.

### Comparison of Main Theoretical Results

| Result | Uncertainty Set | Sample Complexity | vs. Prior Work |
|--------|----------------|-------------------|----------------|
| Theorem 6.1 (Policy Evaluation) | Contamination | $\mathcal{O}(SA t_{\text{mix}}^2 / (\epsilon^2(1-\gamma)^2))$ | First non-asymptotic result |
| Theorem 6.1 (Policy Evaluation) | TV / Wasserstein | $\tilde{\mathcal{O}}(SA t_{\text{mix}}^2 / (\epsilon^2(1-\gamma)^2))$ | First non-asymptotic result |
| Theorem 6.2 (Average Reward Estimation) | All three classes | $\tilde{\mathcal{O}}(SA t_{\text{mix}}^2 / (\epsilon^2(1-\gamma)^2))$ | First non-asymptotic result |

### Comparison of Technical Lemmas

| Lemma | Content | Key Quantity |
|-------|---------|--------------|
| Theorem 5.1 | Expected sample count of truncated MLMC | $\mathbb{E}[M] = N_{\max} + 2$ |
| Theorem 5.2 | Bias decay rate | $\leq 6(1+1/\delta) \cdot 2^{-N_{\max}/2} \|V\|_{\text{sp}}$ (TV) |
| Theorem 5.4 | Variance bound | $\leq 3\|V\|_{\text{sp}}^2 + 144(1+1/\delta)^2 \|V\|_{\text{sp}}^2 N_{\max}$ (TV) |

### Key Findings
- Semi-norm contraction is the breakthrough for finite-sample analysis in average reward robust RL.
- The choice $\Psi = 0.5$ in truncated MLMC is the critical step that eliminates exponential sample growth—any $\Psi < 0.5$ leads to infinite expected sample count.
- Results are order-optimal in $\epsilon$, but tightness with respect to $S$, $A$, and $\gamma$ remains an open problem.

## Highlights & Insights
- **Elegance of the semi-norm construction**: Starting from the Lyapunov norm for a single transition matrix, the paper generalizes to the robust setting via a three-step pipeline—joint spectral radius → extremal norm → quotient-space correction—each step with clear mathematical motivation. This construction technique may be applicable to other stochastic approximation problems lacking natural contraction.
- **Engineering ingenuity of truncated MLMC**: Setting $\Psi = 0.5$ allows the exponential growth of the geometric distribution to be exactly cancelled by the exponential growth in sample count, yielding linear expected complexity—a concise yet profound trick.
- **Unified treatment of three uncertainty set classes**: A Lipschitz lemma unifies the analysis across different divergence measures.

## Limitations & Future Work
- **Purely theoretical with no experiments**: Algorithm performance on practical MDPs is not validated, and constant factors are not compared.
- **Requires an ergodicity assumption** (Assumption 3.1), along with the condition that the uncertainty set radius is small enough to ensure ergodicity under all $\mathsf{P} \in \mathcal{P}$.
- **Dependence on $S$, $A$, $\gamma$ may not be tight**; the authors acknowledge that tightening these dependencies is an open problem.
- **Function approximation is not considered**; all analyses are in the tabular setting.
- **Finite-sample analysis for policy optimization** (as opposed to evaluation) remains unresolved.

## Related Work & Insights
- **vs wang2023model**: Proposes robust RVI TD/Q-learning with only ODE-based asymptotic convergence guarantees, and uses untruncated MLMC leading to infinite sample complexity. The present work achieves the first finite-sample bounds via semi-norm contraction and truncated MLMC.
- **vs zhang2021finite**: Provides finite-sample analysis for non-robust average reward TD; the present work extends this framework to the robust setting, additionally handling bias and nonlinearity introduced by the uncertainty set.
- **vs wang2022policy, zhou2024natural**: Policy evaluation for robust discounted RL, which exploits sup-norm contraction from $\gamma < 1$—a fundamentally different technical approach. The semi-norm method in this paper addresses the essential difficulty of $\gamma = 1$.
- The semi-norm construction technique may offer insights for stability analysis in control theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The semi-norm contraction construction and truncated MLMC are two independent and elegant theoretical innovations.
- Experimental Thoroughness: ⭐⭐ Purely theoretical with no experiments or numerical validation.
- Writing Quality: ⭐⭐⭐⭐ Proof ideas are clear and proof sketches are excellent, though the density of formulas raises the barrier for non-theory readers.
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in finite-sample analysis for robust average reward RL; the semi-norm construction has broad theoretical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)

</div>

<!-- RELATED:END -->
