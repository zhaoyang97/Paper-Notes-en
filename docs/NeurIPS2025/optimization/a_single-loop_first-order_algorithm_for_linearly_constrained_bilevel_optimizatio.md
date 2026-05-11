---
title: >-
  [Paper Note] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization
description: >-
  [NeurIPS 2025][Optimization][bilevel optimization] For bilevel optimization problems with coupled linear constraints in the lower-level problem, this paper proposes SFLCB…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "bilevel optimization"
  - "constrained optimization"
  - "first-order methods"
  - "augmented Lagrangian"
  - "single-loop algorithm"
date: 2026-05-08
content_hash: 6c8bd7a4e962280f
---

# A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.24710](https://arxiv.org/abs/2510.24710)
**Code**: [ShenGroup/SFLCB](https://github.com/ShenGroup/SFLCB)
**Area**: Optimization
**Keywords**: bilevel optimization, constrained optimization, first-order methods, augmented Lagrangian, single-loop algorithm
**Authors**: Wei Shen, Jiawei Zhang, Minhui Huang, Cong Shen (UVA, UW-Madison, Meta)

## TL;DR

For bilevel optimization problems with coupled linear constraints in the lower-level problem, this paper proposes SFLCB, a single-loop first-order algorithm that eliminates Hessian dependence via a penalty-based reformulation combined with augmented Lagrangian, improving the iteration complexity from $O(\epsilon^{-3}\log(\epsilon^{-1}))$ to $O(\epsilon^{-3})$.

## Background & Motivation

- **Background**: Bilevel optimization (BLO) underpins a broad range of ML problems, including hyperparameter optimization, data cleaning, meta-learning, neural architecture search, and reinforcement learning, where the upper-level objective depends on the optimal solution of a lower-level problem.

- **Limitations of Prior Work**: Existing work predominantly addresses unconstrained BLO, whereas practical applications—such as distributed optimization, SVM hyperparameter tuning, adversarial training, and traffic network design—frequently involve lower-level problems with coupled constraints $\mathcal{Y}(x) = \{y \mid Bx + Ay - b \leq 0\}$. The corresponding theory and algorithms remain far from mature. Implicit gradient methods (Tsaknakis et al., Khanduri et al., Xu & Zhu) can handle constrained BLO but require computing lower-level Hessian matrices, incurring prohibitive cost at scale. Among existing first-order approaches, Kwon et al. (2023b) handles only $x$-independent constraints and requires a projection at each step, while Jiang et al. (2024a) supports coupled constraints but relies on double- or triple-loop structures with complexity $O(\epsilon^{-3}\log\epsilon^{-1})$ or $O(\epsilon^{-5}\log\epsilon^{-1})$, which are difficult to implement.

- **Key Challenge**: Prior work (Kwon, Jiang) does not establish non-asymptotic error bounds between the value-function reformulation $\Phi_\delta$ and the original objective $\Phi$ under coupled constraints, leaving the theoretical justification of the reformulation incomplete.

- **Goal**: To develop a single-loop, Hessian-free first-order algorithm for linearly constrained BLO that achieves $O(\epsilon^{-3})$ iteration complexity, thereby resolving an open question in the field.

## Method

### Overall Architecture

SFLCB proceeds in three steps:

1. **Penalty-based reformulation**: The original bilevel problem is reformulated as a single-level minimax problem $\min_x \Phi_\delta(x) = \min_{y \in \mathcal{Y}(x)} \max_{z \in \mathcal{Y}(x)} \Phi_\delta(x,y,z)$, where $\Phi_\delta(x,y,z) = f(x,y) + \frac{1}{\delta}[g(x,y) - g(x,z)]$ and $\delta$ is the penalty parameter.
2. **Slack variables + augmented Lagrangian**: Slack variables $\alpha, \beta$ are introduced to convert inequality constraints into equality constraints, after which an augmented Lagrangian function $K$ is constructed so that it is strongly convex in $y'$ and strongly concave in $z'$.
3. **Single-loop GDA**: Gradient descent-ascent (GDA) is applied to $K$; each iteration requires only first-order gradients with no Hessian computation.

### Key Designs

**Key Design 1: Theoretical guarantees for the reformulation**

- **Theorem 4.1** (value approximation): $0 \leq \Phi(x) - \Phi_\delta(x) \leq \frac{\delta l_{f,0}^2}{2\mu_g}$, and $\|y_\delta^*(x) - y^*(x)\| \leq \frac{2\delta l_{f,0}}{\mu_g}$. Setting $\delta = O(\epsilon)$ suffices to control the approximation error.
- **Theorem 4.9** (gradient approximation): Under LICQ and strict complementarity, $\|\nabla\Phi(x) - \nabla\Phi_\delta(x)\| \leq O(\delta)$. This is the first non-asymptotic gradient error bound established under coupled constraints $\mathcal{Y}(x)$.

**Key Design 2: Strong convexity-concavity via augmented Lagrangian**

The standard Lagrangian $L_\delta$ is merely convex in $y'$ and concave in $z'$, which impedes fast convergence. An augmented Lagrangian is constructed by adding quadratic terms:

$$K(x,y',z',u,v) = L_\delta + \frac{\rho_1}{2}\|h(x,y)-\alpha\|^2 - \frac{\rho_2}{2}\|h(x,z)-\beta\|^2$$

Choosing $\rho_1 \leq \frac{\mu_g - \delta l_{f,1}}{\sigma_{\max}^2(A)}$ and $\rho_2 \leq \frac{\mu_g}{\sigma_{\max}^2(A)}$ guarantees that $K$ is strongly convex in $y'$ and strongly concave in $z'$, while sharing the same optimal point as $L_\delta$.

**Key Design 3: Novel Lyapunov function and error bounds**

The central technical innovation in the convergence analysis is the Lyapunov function:

$$V_t = \frac{1}{4}K(x_t, y_t', z_t', u_t, v_t) + 2q(x_t, v_t) - d(x_t, z_t', u_t, v_t)$$

A descent lemma for $V_t$ is established. Key techniques include bounding the Lagrange multiplier error $\|u_{t+1} - u_\delta^*(x_t)\|$ using the full-rank property of $A$, and, when full rank does not hold, employing a warm-start strategy that fixes $x_0$ and runs the algorithm for $O(\epsilon^{-2})$ steps to obtain a high-quality initial point without affecting the overall complexity order.

### Loss & Training

The final optimization objective is:

$$\min_{x, y' \in \mathcal{P}_y, v} \max_{z' \in \mathcal{P}_y, u} K(x, y', z', u, v)$$

where $\mathcal{P}_y = \{y \in \mathbb{R}^{d_y}, \alpha \in \mathbb{R}_-^{d_h}\}$. Projection onto $\mathcal{P}_y$ reduces to truncating $\alpha, \beta$ to the non-positive orthant and is computationally negligible.

## Key Experimental Results

### Main Results

Complexity comparison (Table 1):

| Method | Lower-level constraints | Iteration complexity | Loop structure |
|--------|------------------------|---------------------|----------------|
| Kwon et al. (2023b) | $y \in \mathcal{Y}$ ($x$-independent) | $O(\epsilon^{-3}\log\epsilon^{-1})$ | Single/double loop |
| BLOCC (Jiang 2024a) | General $h(x,y) \leq 0$ | $O(\epsilon^{-5}\log\epsilon^{-1})$ | Triple loop |
| BLOCC (Jiang 2024a) | $B(x)+A(x)y \leq 0$, $A(x)$ full rank | $O(\epsilon^{-3}\log\epsilon^{-1})$ | Double loop |
| **SFLCB** | $Bx+Ay-b \leq 0$, $A$ full rank | $O(\epsilon^{-3})$ | **Single loop** |
| **SFLCB** | $Ay \leq 0$, initial point satisfies LICQ | $O(\epsilon^{-3})$ | **Single loop** |
| **SFLCB** | $Ay \leq 0$ (general) | $O(\epsilon^{-4})$ | **Single loop** |

### Key Findings

1. **Toy example**: SFLCB converges stably to a local optimum of the upper-level objective across 200 random initializations, confirming algorithmic validity.
2. **SVM hyperparameter optimization** (diabetes dataset): SFLCB converges significantly faster than GAM, LV-HBA, BLOCC, and BiC-GAFFA, reaching lower upper-level objective values within the same number of iterations (Fig. 2).
3. **Traffic network design** (3-node & 9-node): SFLCB substantially outperforms BLOCC on both network scales, achieving higher upper-level utility (Fig. 3).
4. **Sensitivity analysis on $\delta$**: Larger $\delta$ accelerates early convergence but introduces larger asymptotic approximation error; overly small $\delta$ slows convergence. Intermediate values (0.05–0.5) balance speed and accuracy, consistent with theoretical predictions (Fig. 4).

## Highlights & Insights

1. **Strong theoretical contribution**: The paper is the first to establish non-asymptotic error bounds on both the value and gradient of $\Phi$ versus $\Phi_\delta$ under coupled constraints $\mathcal{Y}(x)$ (Theorems 4.1 & 4.9), providing a rigorous foundation for the minimax reformulation.
2. **Single-loop and Hessian-free**: Each iteration requires only first-order gradients and a simple projection (truncation to the non-positive orthant), making the algorithm far easier to implement than BLOCC's triple-loop structure.
3. **Genuine complexity improvement**: The log factor is eliminated, reducing complexity from $O(\epsilon^{-3}\log\epsilon^{-1})$ to $O(\epsilon^{-3})$; an $O(\epsilon^{-4})$ guarantee is also provided for the general (non-full-rank) case.
4. **Lyapunov function design**: The construction of $V_t$ as a mixture of $K$, the dual function $q$, and the descent function $d$, together with the accompanying error bounds, carries independent theoretical value and may be generalized to other constrained optimization settings.
5. **Warm-start strategy**: For the case where $A$ is not full rank, a preliminary run with fixed $x_0$ yields a high-quality initial point at cost $O(\epsilon^{-2})$, without affecting the overall complexity order.

## Limitations & Future Work

1. **Deterministic setting only**: All analyses assume exact gradients; stochastic gradient (SGD) settings are not covered, limiting direct applicability to large-scale deep learning.
2. **Linear constraints only**: The algorithm and theory strictly require $h(x,y) = Bx + Ay - b$ and cannot handle nonlinear coupled constraints.
3. **LICQ and strict complementarity assumptions**: The gradient approximation result (Theorem 4.9) and some convergence results rely on LICQ and strict complementarity; degenerate cases are not addressed.
4. **Gap from optimality**: First-order methods for unconstrained BLO have achieved $O(\epsilon^{-2})$; whether $O(\epsilon^{-3})$ is optimal for the constrained setting remains an open question.
5. **Small-scale experiments**: The SVM experiments use only the diabetes dataset, and the largest traffic network considered has only 9 nodes; validation on large-scale or high-dimensional problems is lacking.

## Related Work & Insights

- **First-order methods for unconstrained BLO**: penalty-based approaches (Kwon et al. 2023a, Shen & Chen 2023, Lu 2024); iterative differentiation (Maclaurin et al. 2015, Grazzi et al. 2020).
- **Implicit gradient methods for constrained BLO**: Tsaknakis et al. 2022 (linear constraints $Ay \leq b$), Khanduri et al. 2023 (perturbation smoothing), Xu & Zhu 2023 (Clarke subdifferential).
- **First-order methods for constrained BLO**: Yao et al. 2024a/b (proximal Lagrangian / doubly regularized gap), Kornowski et al. 2024 (Goldstein stationarity, near-optimal but dimension-dependent).
- **Most closely related work**: BLOCC (Jiang et al. 2024a) uses a triple-loop structure for general coupled constraints; Kwon et al. 2023b adopts single/double loops but handles only $x$-independent constraints.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Solid theoretical contributions (non-asymptotic gradient error bounds under coupled constraints are new results); the Lyapunov function construction reflects genuine technical innovation; however, the overall reformulation framework follows established paradigms.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experiments cover a toy example, SVM hyperparameter optimization, and traffic network design, with sensitivity analysis included, but the scale is limited and comparisons with additional methods such as Kornowski et al. 2024 are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rigorous theoretical exposition; Table 1 provides a clear summary; notation density in some sections reduces readability.
- **Value**: ⭐⭐⭐⭐ — Provides the simplest and most efficient first-order algorithm currently available for constrained BLO, with direct implications for distributed optimization and constrained learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](constrained_network_slice_assignment_via_llms.md)

</div>

<!-- RELATED:END -->
