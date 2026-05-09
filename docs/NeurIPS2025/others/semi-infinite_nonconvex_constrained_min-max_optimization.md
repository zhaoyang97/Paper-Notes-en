---
title: >-
  [Paper Note] Semi-infinite Nonconvex Constrained Min-Max Optimization
description: >-
  [NeurIPS 2025][Semi-infinite programming] For nonconvex min-max optimization problems with infinitely many nonconvex constraints, this paper proposes the iDB-PD (Inexact Dynamic Barrier Primal-Dual) algorithm. Under the Łojasiewicz regularity condition, it establishes the first global non-asymptotic convergence guarantees: stationarity $\mathcal{O}(\epsilon^{-3})$, feasibility $\mathcal{O}(\epsilon^{-6\theta})$, and complementary slackness $\mathcal{O}(\epsilon^{-3\theta/(1-\theta)})$.
tags:
  - NeurIPS 2025
  - Semi-infinite programming
  - nonconvex constraints
  - min-max optimization
  - dynamic barrier method
  - convergence complexity
date: 2026-05-08
content_hash: eb5ed38fe1606264
---

# Semi-infinite Nonconvex Constrained Min-Max Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.12007](https://arxiv.org/abs/2510.12007)
**Code**: Unavailable
**Area**: Others
**Keywords**: Semi-infinite programming, nonconvex constraints, min-max optimization, dynamic barrier method, convergence complexity

## TL;DR

For nonconvex min-max optimization problems with infinitely many nonconvex constraints, this paper proposes the iDB-PD (Inexact Dynamic Barrier Primal-Dual) algorithm. Under the Łojasiewicz regularity condition, it establishes the first global non-asymptotic convergence guarantees: stationarity $\mathcal{O}(\epsilon^{-3})$, feasibility $\mathcal{O}(\epsilon^{-6\theta})$, and complementary slackness $\mathcal{O}(\epsilon^{-3\theta/(1-\theta)})$.

## Background & Motivation

Many problems in modern AI can be formulated as constrained min-max optimization, for example:
- **Robust multi-task learning**: optimizing the worst-case loss on a priority task while constraining the losses of other tasks to remain below a threshold, where the weight distributions in the constraints belong to an uncertainty set, yielding infinitely many constraints.
- **Robust energy-constrained DNN training**: minimizing the worst-case loss under distributional uncertainty while satisfying nonconvex energy and sparsity constraints.

The formal problem statement is:

$$\min_{x \in \mathbb{R}^n} \max_{y \in Y} \phi(x, y), \quad \text{s.t.} \quad \psi(x, w) \leq 0, \quad \forall w \in W$$

where $Y, W$ are compact sets, and both $\phi$ and $\psi$ may be nonconvex in $x$. This setting involves three major challenges:

**Nonconvexity**: The objective and constraints are nonconvex in $x$, precluding global solutions.

**Infinite constraints**: $w \in W$ is a continuous set, yielding infinitely many constraints.

**Nested max structure**: Both the objective and constraints contain inner maximization.

Limitations of prior work:
- Standard min-max algorithms assume finite-dimensional constraints and convexity.
- Semi-infinite programming (SIP) methods typically assume convexity or lack the max structure in the objective.
- Nonconvex constrained optimization requires special regularity conditions to avoid convergence to infeasible stationary points.
- The most closely related work, Yao et al., handles only the convex case.

## Method

### Overall Architecture

The iDB-PD algorithm is a primal-dual method consisting of:
1. **Primal update** ($x$): A QP subproblem is solved to obtain a search direction that simultaneously reduces the objective and improves feasibility.
2. **Dual update** ($y, w$): The inner maximization problems are approximately solved via gradient ascent.

### Key Designs

#### 1. **QP Search Direction**

At each iteration $k$, the search direction $d_k$ is obtained by solving the following QP:

$$d_k = \arg\min_d \| d + \nabla_x \phi(x_k, y_k) \|^2 \quad \text{s.t.} \quad \nabla_x \psi(x_k, w_k)^\top d + \alpha_k \rho(x_k, w_k) \leq 0$$

where $\rho(x, w) = \|\nabla_x \psi(x, w)\|$ is the dynamic barrier function. This QP admits a closed-form solution:

$$d_k = -\nabla_x \phi(x_k, y_k) - \lambda_k \nabla_x \psi(x_k, w_k)$$

$$\lambda_k = \frac{1}{\|\nabla_x \psi(x_k, w_k)\|^2} [-\nabla_x \psi(x_k, w_k)^\top \nabla_x \phi(x_k, y_k) + \alpha_k \rho(x_k, w_k)]_+$$

**Design motivation**: The objective component $-\nabla_x \phi$ drives loss reduction, while the constraint component $-\lambda_k \nabla_x \psi$ drives feasibility improvement. The QP finds an optimal balance between the two.

#### 2. **Indicator Function and Dual Stability**

The update of $\lambda_k$ diverges when $\|\nabla_x \psi\| \to 0$. To address this, an indicator function is introduced:

$$\zeta(x, w) = [\psi(x, w)]_+ \|\nabla_x \psi(x, w)\|$$

When $\zeta(x_k, w_k) = 0$, the current point is either feasible or at a critical point of the constraint, and $\lambda_k$ is set to zero, reducing the search direction to the pure objective gradient $d_k = -\nabla_x \phi(x_k, y_k)$.

This design eliminates the requirement in Hinder & Sidford (2020) that dual iterates be assumed bounded.

#### 3. **Łojasiewicz Regularity Condition**

The Łojasiewicz inequality is imposed on the squared infeasibility residual $[\psi(x, w)]_+^2$:

$$[\psi(x, w)]_+^{2\theta} \leq \mu \|\nabla_x \psi(x, w) [\psi(x, w)]_+\|, \quad \forall x \in \mathbb{R}^n, \forall w \in W$$

where $\theta \in (0, 1)$. When $\theta = 1/2$, this reduces to the classical Polyak-Łojasiewicz condition.

**Design motivation**: This condition ensures that the gradient of the constraint does not decay too rapidly near critical points, guaranteeing that the algorithm can escape from infeasible regions toward feasibility. It is naturally satisfied for DNNs with smooth activation functions such as tanh and softplus.

### Loss & Training

At each iteration, the complete algorithm performs:
1. Determine $\lambda_k$ based on $\zeta(x_k, w_k)$ (set via the QP solution if the constraint is active, otherwise set to zero).
2. Compute the search direction $d_k$ and update $x_{k+1} = x_k + \gamma_k d_k$.
3. Approximately solve $y_{k+1} \approx \arg\max_{y \in Y} \phi(x_{k+1}, y)$ via $N_k$ steps of (accelerated) gradient ascent.
4. Approximately solve $w_{k+1} \approx \arg\max_{w \in W} \psi(x_{k+1}, w)$ via $M_k$ steps of (accelerated) gradient ascent.

Parameter choices: $\gamma_k = \mathcal{O}(T^{-1/3})$, $\alpha_k = T^{1/3} / (k+2)^{1+\omega}$, $N_k = \mathcal{O}(\log(k+1))$, and $M_k$ depends on $\theta$ and the current infeasibility residual.

## Key Experimental Results

### Main Results — Robust Multi-Task Learning

Experiments are conducted on five datasets: Multi-MNIST, CHD49, Multi-Fashion-MNIST, Yeast, and 20NG. One of two classification tasks is designated as the priority task (objective), and the other as the constrained task.

| Method | Stationarity | Feasibility | Complementary Slackness | Note |
|---|---|---|---|---|
| Adaptive Discretization + COOPER | ✗ Diverges | ✗ Infeasible | ✗ Diverges | Classical discretization fails |
| **iDB-PD** | ✓ Converges | ✓ Converges | ✓ Converges | All three metrics converge consistently |

### Comparison with Weighted Aggregation Method (GDMA)

| Method | $\rho$ | Stationarity | Feasibility | Note |
|---|---|---|---|---|
| GDMA | $\rho=1$ | Low | Non-convergent | Weight too small to enforce constraints |
| GDMA | $\rho=10$ | Unstable | Partially convergent | Weight too large causes oscillation |
| **iDB-PD** | Adaptive | **Converges** | **Converges** | No manual weight tuning required |

### Key Findings

1. **Discretization methods fail on this problem**: Adaptive discretization + COOPER cannot achieve feasibility convergence and diverges in stationarity, demonstrating that classical SIP methods are unsuitable for the nonconvex min-max setting.
2. **Difficulty of tuning $\rho$ in weighted aggregation**: GDMA is highly sensitive to $\rho$; small values fail to satisfy constraints while large values cause instability. iDB-PD automatically balances objective and constraints via dual variables.
3. **Under the special case $\theta = 1/2$**, all three complexity bounds unify to $\mathcal{O}(\epsilon^{-3})$, matching the known optimal result for finite-constraint nonconvex optimization.

## Highlights & Insights

- This work provides the first non-asymptotic convergence guarantees for nonconvex semi-infinite min-max optimization, filling a theoretical gap at the intersection of SIP and min-max optimization.
- The introduction of the indicator function $\zeta(x, w)$ elegantly resolves the dual divergence issue, eliminating the need to assume bounded dual iterates.
- The QP subproblem admits a closed-form solution, keeping per-iteration computational cost low and making the method suitable for large-scale problems.
- The convergence guarantees depend on the Łojasiewicz exponent $\theta$, providing a flexible theoretical framework for practical problems (different values of $\theta$ yield different convergence rates).

## Limitations & Future Work

- **Non-smooth activations such as ReLU do not satisfy the Łojasiewicz condition**, requiring smooth alternatives such as softplus—a practical limitation in modern deep learning.
- Experiments are conducted only on small-scale networks (single hidden layer + tanh), and scalability to modern deep networks is not demonstrated.
- The choice of $M_k$ depends on the unknown exponent $\theta$, which must be estimated or conservatively approximated in practice.
- A stochastic optimization (SGD) variant is absent; the current analysis requires exact gradients or near-exact inner solvers.

## Related Work & Insights

- **DBGD (Hinder & Sidford 2020)**: Dynamic barrier gradient descent, an important foundation for this work, but requires bounded dual assumptions and provides no convergence guarantee when $\tau = 1$.
- **COOPER**: A PyTorch constrained optimization library using Lagrangian methods for discretized problems, serving as a baseline in experiments.
- **GDMA**: Gradient descent multi-ascent for nonconvex-strongly-concave min-max problems; a constrained version is used for comparison.
- **Insight**: For constrained optimization, adaptive dual regulation (via indicator functions, dynamic barriers, etc.) is more robust than fixed penalties or weights.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First treatment of the nonconvex semi-infinite min-max setting; significant theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments are limited in scale (single hidden-layer networks) and present only qualitative convergence curves without quantitative accuracy comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and notation is consistent, though the highly theoretical content limits readability.
- **Value**: ⭐⭐⭐⭐ Provides theoretical tools for robust optimization and safety-constrained learning, though practical applicability warrants further validation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[NeurIPS 2025\] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)

<!-- RELATED:END -->
