---
title: >-
  [Paper Note] Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability
description: >-
  [ICML2025][Reinforcement Learning][dynamic regret] By replacing traditional KKT analysis with the concept of mixability, this paper proposes a continuous-space online learning framework based on exponential weights and fixed-share updates, significantly improving the dependence of dynamic regret on dimension $d$ from $O(d^{10/3})$ to $O(d)$ for curved loss functions (such as squared/logistic loss).
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "dynamic regret"
  - "online convex optimization"
  - "mixability"
  - "exp-concavity"
  - "non-stationary online learning"
  - "exponential weights method"
  - "fixed-share"
date: 2026-05-08
content_hash: f205e4602a047937
---

# Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability

**Conference**: ICML2025  
**arXiv**: [2506.10616](https://arxiv.org/abs/2506.10616)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: dynamic regret, online convex optimization, mixability, exp-concavity, non-stationary online learning, exponential weights method, fixed-share

## TL;DR

By replacing traditional KKT analysis with the concept of mixability, this paper proposes a continuous-space online learning framework based on exponential weights and fixed-share updates, significantly improving the dependence of dynamic regret on dimension $d$ from $O(d^{10/3})$ to $O(d)$ for curved loss functions (such as squared/logistic loss).

## Background & Motivation

**Non-stationary online learning** is a highly active research area in recent years. In the standard Online Convex Optimization (OCO) framework, the learner interacts with the environment over $T$ rounds. In each round, the learner submits a prediction $\mathbf{w}_t$, and the environment generates a convex loss function $f_t$. In non-stationary environments, **dynamic regret** is commonly used to measure the performance of algorithms:

$$\text{D-Reg}_T = \sum_{t=1}^T f_t(\mathbf{w}_t) - \sum_{t=1}^T f_t(\mathbf{u}_t)$$

where $\{\mathbf{u}_t\}$ is a time-varying comparator sequence, and $P_T = \sum_{t=2}^T \|\mathbf{u}_t - \mathbf{u}_{t-1}\|_2$ is the path length, which reflects the non-stationarity of the environment.

**Limitations of Prior Work**:

- The dynamic regret of general convex losses has been well-studied, with the optimal bound being $O(\sqrt{T(1+P_T)})$.
- For losses with **stronger curvature** (e.g., squared loss, logistic loss), namely exp-concave losses, Baby & Wang (2021, 2022a) obtained a dynamic regret of $\widetilde{O}(d^{10/3} T^{1/3} P_T^{2/3})$ through KKT condition analysis.
- However, the known lower bound is $\Omega(d^{1/3} T^{1/3} P_T^{2/3})$, showing a massive gap in dimension dependence between $d^{10/3}$ and $d^{1/3}$.
- KKT analysis is technically complex and difficult to generalize to broader scenarios.

## Method

### Core Idea: From Exp-concavity to Mixability

The key innovation of this work is using **mixability** instead of exp-concavity to characterize loss curvature.

**Exp-concavity**: A function $f$ is $\eta$-exp-concave if $e^{-\eta f(\mathbf{w})}$ is concave, i.e., for any distribution $P$:

$$f(\mathbb{E}_P[\mathbf{w}]) \leq -\frac{1}{\eta} \ln \mathbb{E}_{\mathbf{w}\sim P}[e^{-\eta f(\mathbf{w})}]$$

**Mixability (Weaker Condition)**: It only requires the **existence** of some $\mathbf{w}_{\text{mix}}$ (not necessarily the mean) that satisfies the above inequality. Exp-concavity $\Rightarrow$ mixability, but the converse does not hold.

Important examples:

| Loss Function | Mixability Coefficient | Exp-concavity Coefficient |
|---------|----------------|-------------------|
| Squared loss $(z-y)^2$ | $\frac{1}{2B^2}$ (unconstrained space) | $\frac{1}{2(B+D)^2}$ (bounded domain) |
| Logistic loss $\log(1+e^{-yz})$ | $1$ (unconstrained space) | $e^{-D}$ (bounded domain) |

Mixability allows obtaining a larger curvature coefficient $\eta$ on unconstrained spaces, which is the key to improving the dimension dependence.

### Algorithm Framework: Fixed-share for Continuous Space (Algorithm 1)

The algorithm maintains a **distribution** $P_t$ over the parameter space $\mathbb{R}^d$ (rather than a single parameter vector), involving a two-step update:

**Step 1 — Exponential weights update**:

$$\widetilde{P}_{t+1}(\mathbf{u}) \propto P_t(\mathbf{u}) \exp(-\eta f_t(\mathbf{u}))$$

**Step 2 — Fixed-share mixture**:

$$P_{t+1}(\mathbf{u}) = (1-\mu) \widetilde{P}_{t+1}(\mathbf{u}) + \mu \mathcal{N}(\mathbf{w}_0, I_d)$$

where $\mu = 1/T$. In each round, a small fraction of the probability mass is "reset" to the initial Gaussian distribution, allowing the algorithm to adapt to environmental changes.

**Prediction** constructs $z_t$ through the mixability condition, such that $\ell(z_t, y_t) \leq m_t(P_t)$ (mix loss).

### Three-stage Analysis Framework

The dynamic regret is decomposed into three terms:

$$\text{D-Reg}_T \leq \underbrace{\sum \ell(z_t,y_t) - \sum m_t(P_t)}_{\text{(A) mixability gap} \leq 0} + \underbrace{\sum m_t(P_t) - \sum \mathbb{E}_{Q_t}[f_t]}_{\text{(B) mixability regret}} + \underbrace{\sum \mathbb{E}_{Q_t}[f_t] - \sum f_t(\mathbf{u}_t)}_{\text{(C) comparator gap}}$$

- **(A)** is guaranteed to be non-positive by the mixability condition.
- **(B)** is bounded via KL divergence by introducing an analytical Gaussian distribution $Q_t = \mathcal{N}(\mathbf{u}_t, \sigma^2 I_d)$.
- **(C)** is bounded as $\leq \beta d \sigma^2 T / 2$ by loss smoothness + Gaussian expectation.

Selecting $\sigma = \Theta(P_T^{1/3} T^{-1/3})$ to balance (B) and (C) yields the final bound.

### Equivalent Implementation: Follow-the-Leading-History (Algorithm 2)

Fixed-share is equivalent to a **two-layer online ensemble**:

- **Base Learner Layer**: A new base learner $\mathcal{B}_i$ is added in each round, initialized with a Gaussian and updated via exponential weights.
- **Meta-Learner Layer**: Allocates weights using the Hedge method according to the mix loss, with the weight of the new learner set to $\mu$.

### Extension to General OCO (Algorithm 3)

For general exp-concave losses (which are not necessarily mixable on unconstrained spaces), two improvements are introduced:

1. **Surrogate Loss**: $\widetilde{f}_t(\mathbf{w}) = \mathbf{g}_t^\top(\mathbf{w}-\mathbf{w}_t) + \frac{\gamma}{2}(\mathbf{g}_t^\top(\mathbf{w}-\mathbf{w}_t))^2$, eliminating the smoothness assumption.
2. **Projection Step**: Projects the distribution onto a constraint set $\mathscr{M}$ (a family of Gaussian mixtures with means in $\mathcal{W}$ and bounded covariances), ensuring proper learning.

## Key Experimental Results

This paper is a **purely theoretical contribution** and does not contain experimental parts. The main theoretical results are as follows:

| Loss Type | Method | Dynamic Regret Bound | Proper | Complexity |
|---------|------|-----------|--------|-------|
| 1-d squared | Baby & Wang 2021 | $\widetilde{O}(T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| 1-d squared | **Ours Corollary 1** | $\widetilde{O}(T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| Least-squares | Baby & Wang 2022b | $\widetilde{O}(d^{10/3}T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| Least-squares | **Ours Corollary 2** | $\widetilde{O}(d \cdot T^{1/3}P_T^{2/3})$ | ✗ | $O(T)$ |
| Logistic | Baby et al. 2023 | $\widetilde{O}(d^{10/3}T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| Logistic | **Ours Corollary 3** | $\widetilde{O}(d \cdot T^{1/3}P_T^{2/3})$ | ✗ | poly$(T)$ |
| General exp-concave | Baby & Wang 2022a | $\widetilde{O}(d^{10/3}T^{1/3}P_T^{2/3})$ | ✗* | $O(T)$ |
| General exp-concave | **Ours Theorem 3** | $\widetilde{O}(d \cdot T^{1/3}P_T^{2/3})$ | ✓ | — |

**Key Findings**: The dimension dependence is improved from $d^{10/3}$ to $d$, narrowing the gap with the lower bound $\Omega(d^{1/3})$.

## Highlights & Insights

1. **Conceptual Breakthrough**: Mixability is introduced into dynamic regret analysis for the first time, providing a simpler and more general theoretical tool than KKT analysis.
2. **Significant Dimension Improvement**: $d^{10/3} \to d$, which makes a huge difference in high-dimensional problems (e.g., a difference of $\sim 10^4$ times when $d=100$).
3. **Elegant Analysis Framework**: The three-stage decomposition (mixability gap / regret / comparator gap) provides a clear characterization of the source of regret.
4. **No Prior Knowledge Required**: The algorithm does not need to know the path length $P_T$ in advance; it automatically adapts to non-stationarity by maintaining a distribution.
5. **Proper Learning of Algorithm 3**: Achieves proper predictions in general exp-concave scenarios through surrogate loss and a projection step, resolving the challenge that existing methods cannot achieve proper learning on arbitrary convex domains.
6. **Equivalence Theorem (Theorem 2)**: Unveils the deep connection between fixed-share and FLH-type algorithms in continuous spaces.

## Limitations & Future Work

1. **Improper learning**: Algorithm 1 is improper for least-squares and logistic regression (predictions may exceed the linear hypothesis space), which is less desirable than the proper methods of Baby & Wang (2022b) and Baby et al. (2023).
2. **Computational Efficiency**: Logistic loss lacks closed-form updates and requires sampling approximations; the computational overhead of the projection step in general OCO remains unclear.
3. **Remaining Dimension Gap**: The optimal bound is $\widetilde{O}(d \cdot T^{1/3} P_T^{2/3})$ while the lower bound is $\Omega(d^{1/3})$, leaving the gap between $d$ and $d^{1/3}$ not yet fully closed.
4. **Inability to Accelerate to $O(\log T)$**: While KKT methods can accelerate to $O(\log T)$ per round using geometric covering, it remains unclear how to achieve this under the fixed exponential weights framework.
5. **Purely Theoretical Work**: Lacks experimental validation and does not demonstrate performance on practical non-stationary learning tasks.

## Related Work & Insights

- **Baby & Wang (2021, 2022a, b), Baby et al. (2023)**: The line of work using KKT analysis for dynamic regret of exp-concave losses, which serves as the direct baseline that this paper improves upon.
- **Vovk (2001), van Erven et al. (2012)**: Foundational works on the concept of mixability, which this paper generalizes from static regret to dynamic regret.
- **Cesa-Bianchi et al. (2012b)**: Dynamic regret analysis of fixed-share under the finite-expert setting, which this paper generalizes to continuous spaces.
- **Hazan & Seshadhri (2009)**: FLH algorithms; this paper proves their equivalence to fixed-share in continuous spaces.
- **Zhang et al. (2018), Zhao et al. (2024)**: Optimal methods for dynamic regret of convex losses; the framework in this paper can be viewed as the corresponding extension to curved losses.

**Key Insight**: The fact that mixability, despite being a weaker condition than exp-concavity, can yield superior dimension dependence highlights the importance of choosing "the right level of abstraction" to obtain tighter theoretical guarantees in online learning. The paradigm of maintaining distributions (rather than point estimates) combined with fixed-share could potentially be generalized to other non-stationary learning problems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Significant conceptual contribution by analyzing dynamic regret using mixability for the first time)
- Experimental Thoroughness: ⭐⭐ (Purely theoretical, no experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear analysis, complete proof sketches, and comprehensive comparisons)
- Value: ⭐⭐⭐⭐ (Substantially narrows the theoretical gap, though computational bottlenecks and practical utility remain to be verified)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](../../NeurIPS2025/reinforcement_learning/dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[ICML 2025\] On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning](on_the_dynamic_regret_of_following_the_regularized_leader_optimism_with_history_.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Establishing Linear Surrogate Regret Bounds for Convex Smooth Losses via Convolutional Fenchel–Young Losses](../../NeurIPS2025/reinforcement_learning/establishing_linear_surrogate_regret_bounds_for_convex_smooth_losses_via_convolu.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](../../NeurIPS2025/reinforcement_learning/improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)

</div>

<!-- RELATED:END -->
