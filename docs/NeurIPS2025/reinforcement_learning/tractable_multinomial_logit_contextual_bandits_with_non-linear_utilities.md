---
title: >-
  [Paper Note] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities
description: >-
  [NeurIPS 2025][Reinforcement Learning][contextual bandits] This work presents ONL-MNL, the first **computationally tractable** and **statistically optimal** algorithm for the MNL contextual bandit problem under non-linear utility functions (including neural networks), achieving $\widetilde{\mathcal{O}}(\sqrt{T})$ regret without relying on NTK assumptions.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - contextual bandits
  - multinomial logit
  - non-linear utility functions
  - combinatorial assortment optimization
  - UCB
  - neural networks
date: 2026-05-08
content_hash: e9b645d2ccec762e
---

# Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities

**Conference**: NeurIPS 2025
**arXiv**: [2601.06913](https://arxiv.org/abs/2601.06913)
**Authors**: Taehyun Hwang, Dahngoon Kim, Min-hwan Oh (Seoul National University)
**Code**: Not available
**Area**: Reinforcement Learning
**Keywords**: contextual bandits, multinomial logit, non-linear utility functions, combinatorial assortment optimization, UCB, neural networks

## TL;DR

This work presents ONL-MNL, the first **computationally tractable** and **statistically optimal** algorithm for the MNL contextual bandit problem under non-linear utility functions (including neural networks), achieving $\widetilde{\mathcal{O}}(\sqrt{T})$ regret without relying on NTK assumptions.

## Background & Motivation

### State of the Field

The multinomial logit (MNL) contextual bandit is a standard sequential assortment selection model widely used in recommendation systems, online retail, and personalized marketing. At each round, an agent observes item features and selects a subset (assortment) to present to a user, who then chooses an item according to the MNL choice model. The objective is to minimize cumulative regret relative to the optimal assortment policy.

### Limitations of Prior Work

- Nearly all existing MNL contextual bandit work **assumes the utility function is linear in the features**, which fails to capture complex non-linear user behavior.
- Zhang & Luo (2025) were the first to consider general utility function classes, but they face a **fundamental tension between computational tractability and statistical efficiency**:
    - The $\varepsilon$-greedy approach is computationally tractable but achieves only $\widetilde{\mathcal{O}}(T^{2/3})$ regret (stochastic setting) or $\widetilde{\mathcal{O}}(T^{5/6})$ (adversarial setting).
    - Log-barrier and Feel-Good TS methods achieve $\widetilde{\mathcal{O}}(\sqrt{T})$ regret but are **computationally intractable** (non-convex optimization).
- Existing neural bandit methods rely on NTK approximations, requiring unrealistically large over-parameterization.

### Root Cause

The core motivation is to design an MNL contextual bandit algorithm that **simultaneously achieves computational tractability and $\widetilde{\mathcal{O}}(\sqrt{T})$ optimal regret**, supporting non-linear parameterized utility functions including neural networks.

## Method

### Problem Formulation

- **MNL choice model**: The probability that a user selects item $i$ from the displayed assortment $S_t$ is $p(i|X_t,S_t,\mathbf{w}^*) = \frac{\exp(f_{\mathbf{w}^*}(\mathbf{x}_{ti}))}{1+\sum_{j\in S_t}\exp(f_{\mathbf{w}^*}(\mathbf{x}_{tj}))}$
- **Non-linear parameterization**: The utility function $f_\mathbf{w}(\cdot)$ belongs to a general parameterized function class $\mathcal{F}$ (e.g., a neural network), differentiable with respect to parameter $\mathbf{w}$ but not necessarily with respect to feature $\mathbf{x}$.
- **Equivalence set**: $\mathcal{W}^* := \{\mathbf{w} \in \mathcal{W}: f_\mathbf{w}(\mathbf{x}) = f_{\mathbf{w}^*}(\mathbf{x})\ \forall \mathbf{x}\}$, accommodating multiple global optima arising from parameter symmetries in neural networks.
- **Objective**: Minimize cumulative regret $\mathrm{Regret}_T = \sum_{t=1}^T [R_t(S_t^*, \mathbf{w}^*) - R_t(S_t, \mathbf{w}^*)]$.

### Core Technical Challenges

1. The negative log-likelihood of the non-linear MNL model lacks the self-concordance property available in the linear case.
2. Gradients depend on the unknown parameter, precluding direct application of concentration inequalities from linear models.
3. The non-convex optimization landscape admits multiple global optima (e.g., permutation symmetry of neurons in neural networks).

### Algorithm: ONL-MNL (Two-Phase Design)

**Phase I — Uniform Exploration ($t_0$ rounds)**:

- Randomly sample assortments uniformly and collect user choice data.
- Minimize the negative log-likelihood $\mathcal{L}(\mathbf{w})$ to obtain an initial estimate $\hat{\mathbf{w}}_0$.
- Establish a convergence rate via the generalized geometric condition (Assumption 4): $\min_{\tilde{\mathbf{w}}\in\mathcal{W}^*}\|\hat{\mathbf{w}}_0 - \tilde{\mathbf{w}}\|_2^2 = \mathcal{O}(1/t_0)$.

**Phase II — Optimistic Exploration (remaining $T-t_0$ rounds)**:

- Construct a **linearized MNL model** by performing a first-order Taylor expansion of $f_\mathbf{w}$ around the current estimate $\hat{\mathbf{w}}_t$.
- Minimize the regularized negative log-likelihood using $\hat{\mathbf{w}}_0$ as the regularization center rather than the origin.
- Construct **optimistic utility estimates**:

$$z_{ti} = f_{\hat{\mathbf{w}}_t}(\mathbf{x}_{ti}) + \sqrt{\beta_t}\|\nabla f_{\hat{\mathbf{w}}_t}(\mathbf{x}_{ti})\|_{\mathbf{V}_t^{-1}} + \frac{\beta_t C_h}{\lambda}$$

- Select the assortment maximizing the optimistic expected revenue: $S_t = \arg\max_{S\in\mathcal{S}} \widetilde{R}_t(S)$.
- Update the Gram matrix $\mathbf{V}_{t+1} = \mathbf{V}_t + \sum_{i\in S_t}\nabla f_{\hat{\mathbf{w}}_t}(\mathbf{x}_{ti})\nabla f_{\hat{\mathbf{w}}_t}(\mathbf{x}_{ti})^\top$.

### Key Technical Innovations

**Generalized Geometric Condition (Assumption 4)**: The expected squared loss $\ell_\mathrm{sq}(\mathbf{w})$ is required to satisfy a $(\tau,\gamma)$-growth condition or $\mu$-local strong convexity relative to the equivalence set $\mathcal{W}^*$. This is strictly weaker than the assumption of a globally unique optimum required by prior work. It naturally accommodates highly non-convex loss landscapes with arbitrarily many spurious local optima and multiple global optima, making it well-suited to neural networks.

**$\lambda$-Independent Concentration Inequality**: For the large regularization regime $\lambda = \mathcal{O}(\sqrt{T})$, a novel concentration inequality is derived that is independent of $\lambda$ (up to logarithmic factors), breaking the bottleneck of existing Bernstein-type bounds that require $\lambda = \mathcal{O}(d\log T)$.

**Inductive Proof of Confidence Sets**: Mathematical induction is used to recursively establish that the estimation error $\min_{\tilde{\mathbf{w}}\in\mathcal{W}^*}\|\hat{\mathbf{w}}_t-\tilde{\mathbf{w}}\|_2^2 = \widetilde{\mathcal{O}}(1/t_0)$ holds for all rounds in Phase II, guaranteeing $\beta_t = \widetilde{\mathcal{O}}(\mu^{-2}\kappa^{-4}d_w)$.

### Main Theorem

Let $\kappa := \min_{\mathbf{w},X,S,i} p(0|X,S,\mathbf{w})\cdot p(i|X,S,\mathbf{w})$, and set $t_0 = \lceil \kappa^{-3/2}d_w\sqrt{T}\rceil$. Then with high probability:

$$\mathrm{Regret}_T = \widetilde{\mathcal{O}}\left(\kappa^{-3/2}d_w\sqrt{T} + \kappa^{-2}\mu^{-1}d_w\sqrt{T}\right)$$

This achieves $\widetilde{\mathcal{O}}(\sqrt{T})$ in $T$ and is **independent of the total number of items $N$**.

## Key Experimental Results

### Experiment 1: Cumulative Regret under Realizable and Misspecified Settings

Setup: $N=100$ items, $K=5$ maximum assortment size, $d=3$ feature dimension, 30 random seeds, $T=1000$ rounds.

| Algorithm | Utility Type | Realizability | Performance (Gaussian context) | Key Property |
|-----------|-------------|---------------|-------------------------------|--------------|
| **ONL-MNL (Ours)** | Non-linear | Realizable | Lowest regret, fast convergence | Tractable + statistically optimal |
| **ONL-MNL (Ours)** | Non-linear | Misspecified | Still achieves low regret | Strong robustness |
| $\varepsilon$-greedy-MNL | General | Realizable | Regret significantly higher than ONL-MNL | Tractable but suboptimal |
| UCB-MNL | Linear | Misspecified | Regret grows linearly | Fails to capture non-linearity |
| TS-MNL | Linear | Misspecified | Regret grows linearly | Linear assumption breaks down |
| OFU-MNL+ | Linear | Misspecified | Regret grows linearly | Near-optimal but limited to linear |

- Realizable setting: True utility is a 2-layer sigmoid neural network (3 hidden neurons), $t_0=50$.
- Misspecified setting: True utility is $\cos(2\pi(\mathbf{x}^\top\mathbf{w}^*)) - \frac{1}{2}(\mathbf{x}^\top\mathbf{w}^*)$; estimator uses 15 hidden neurons, $t_0=100$.

### Experiment 2: Effect of Number of Items $N$ on Regret

Setup: Realizable environment (10 hidden neurons), $T=500$, comparing $N=100$ and $N=800$.

| Algorithm | $N=100$ (Gaussian) | $N=800$ (Gaussian) | Effect of Increasing $N$ |
|-----------|-------------------|-------------------|--------------------------|
| **ONL-MNL (Ours)** | Low regret | Slight increase (Phase I only) | Phase II slope unchanged; learning efficiency unaffected by $N$ |
| $\varepsilon$-greedy-MNL | Moderate regret | Regret increases significantly | Theoretical regret $\propto N^{1/3}$, consistent with theory |

Consistent conclusions are observed under both Gaussian and Uniform context distributions: the convergence slope of ONL-MNL in Phase II is independent of $N$.

### Theoretical Comparison with Prior Methods

| Algorithm | Utility Type | Computationally Tractable | Regret | Depends on $N$? |
|-----------|-------------|--------------------------|--------|-----------------|
| $\varepsilon$-greedy [Zhang & Luo] | Lipschitz | ✓ | $\widetilde{\mathcal{O}}((d_wNK)^{1/3}T^{2/3})$ | Yes |
| Log-barrier [Zhang & Luo] | Lipschitz | ✗ | $\widetilde{\mathcal{O}}(K^2\sqrt{d_wNT})$ | Yes |
| Feel-Good TS [Zhang & Luo] | Lipschitz | ✗ | $\widetilde{\mathcal{O}}(K^2\sqrt{d_wNT})$ | Yes |
| **ONL-MNL (Ours)** | Smooth | ✓ | $\widetilde{\mathcal{O}}(\kappa^{-2}\mu^{-1}d_w\sqrt{T})$ | **No** |

## Highlights & Insights

- **First tractable and optimal algorithm**: ONL-MNL is the first method to simultaneously achieve computational tractability and $\widetilde{\mathcal{O}}(\sqrt{T})$ regret for the non-linear utility MNL bandit, resolving the fundamental computation–statistics tension in prior work.
- **No NTK assumptions required**: The algorithm directly handles general non-linear parameterizations including neural networks, without requiring over-parameterization or specific architectural constraints.
- **Regret independent of $N$**: Unlike prior methods that exhibit polynomial dependence on the total number of items $N$, ONL-MNL scales to large item pools.
- **Generalized geometric condition**: The proposed assumption strictly weakens existing geometric conditions, accommodating non-convex loss landscapes with multiple global optima, naturally suited to neural networks.
- **Novel concentration inequality**: A new parameter estimation concentration bound independent of the regularization parameter $\lambda$ is derived, breaking the technical bottleneck that arises when $\lambda = \mathcal{O}(\sqrt{T})$.

## Limitations & Future Work

- **Instance-dependent factor $\kappa$**: The regret bound's $\kappa^{-1}$ term can be as large as $\mathcal{O}(K^2)$ in the worst case; eliminating or weakening this dependence remains an open problem.
- **Phase I requires i.i.d. contexts**: Assumption 3 requires contexts in the exploration phase to be independent and identically distributed; although Phase II admits adversarial contexts, this remains a restriction.
- **Difficulty verifying the generalized geometric condition**: Assumption 4 is difficult to verify a priori for general neural network architectures, and the paper does not provide concrete verification methods.
- **Limited experimental scale**: Experiments are conducted only with $N\leq 800$ and $d=3$; high-dimensional features and larger real-world datasets are not considered.
- **Fully adversarial setting not studied**: Although Phase II theoretically permits adversarial contexts, experiments are conducted only under stochastic contexts.
- **Extension to MNL MDPs**: The paper proposes extending the framework to MNL MDPs as future work but provides no concrete methodology.

## Related Work & Insights

- **UCB-MNL [Oh & Iyengar 2021], TS-MNL [Cheung & Simchi-Levi 2017]**: Support only linear utilities; regret does not converge under non-linear settings.
- **OFU-MNL+ [Lee & Oh 2024]**: A near-optimal algorithm for the linear utility case with improved $\kappa$ dependence, but still restricted to linear utilities.
- **Zhang & Luo [2025]**: The first to consider general utilities, but face a computation–statistics tradeoff; their computationally tractable method achieves only $T^{2/3}$ regret.
- **NTK-based neural bandits [Zhou et al., Hwang et al.]**: Require over-parameterization assumptions and are unsuitable for practically sized networks.
- **Non-NTK neural bandits [Xu et al. 2024, Huang et al. 2021]**: Require specific activation functions (quadratic) or context distributions (Gaussian), limiting their applicability.
- **Linear/GLM bandits [Abbasi-Yadkori et al., Filippi et al.]**: Global strong convexity assumptions do not extend to non-linear models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to resolve the tension between computational tractability and statistical optimality in the non-linear MNL bandit.
- Experimental Thoroughness: ⭐⭐⭐ — Core claims are validated, but experiments are small-scale and lack real-world datasets.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear, technical development is coherent, and theoretical derivations are rigorous.
- Value: ⭐⭐⭐⭐ — Fills an important theoretical gap with potential practical impact for applications such as recommendation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)

</div>

<!-- RELATED:END -->
