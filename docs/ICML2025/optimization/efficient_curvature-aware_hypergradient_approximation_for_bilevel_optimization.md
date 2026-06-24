---
title: >-
  [Paper Note] Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization
description: >-
  [ICML2025][Optimization][bilevel optimization] Proposes the NBO framework, which exploits the inherent structure of hypergradients in bilevel optimization (where solving the lower-level problem and the Hessian-inverse-vector product share the same Hessian). By efficiently integrating curvature information via inexact Newton methods to approximate the hypergradient, it improves the gradient computational complexity by a factor of $\kappa \log \kappa$ compared to the SOTA in de…
tags:
  - "ICML2025"
  - "Optimization"
  - "bilevel optimization"
  - "hypergradient"
  - "inexact Newton"
  - "curvature information"
  - "meta-learning"
date: 2026-05-08
content_hash: 298193e22b1957e5
---

# Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization

**Conference**: ICML2025  
**arXiv**: [2505.02101](https://arxiv.org/abs/2505.02101)  
**Code**: To be confirmed  
**Area**: Optimization  
**Keywords**: bilevel optimization, hypergradient, inexact Newton, curvature information, meta-learning

## TL;DR
Proposes the NBO framework, which exploits the inherent structure of hypergradients in bilevel optimization (where solving the lower-level problem and the Hessian-inverse-vector product share the same Hessian). By efficiently integrating curvature information via inexact Newton methods to approximate the hypergradient, it improves the gradient computational complexity by a factor of $\kappa \log \kappa$ compared to the SOTA in deterministic settings.

## Background & Motivation
Bilevel optimization is widely applied in machine learning tasks such as hyperparameter optimization, meta-learning, adversarial training, and neural architecture search. Its core challenge lies in estimating the hypergradient (implicit gradient):

$$\nabla \Phi(x) = \nabla_1 f(x, y^*(x)) - \nabla_{12}^2 g(x, y^*(x)) u^*(x)$$

where $u^*(x) = [\nabla_{22}^2 g(x,y^*(x))]^{-1} \nabla_2 f(x,y^*(x))$. Existing methods typically treat **solving the lower-level problem** and **computing the Hessian-inverse-vector product** as independent tasks and process them separately. However, they share the same Hessian $\nabla_{22}^2 g$, and this inherent structure has not been fully utilized.

SHINE (Ramzi et al., 2022) made the first attempt to exploit this structure by using quasi-Newton matrices to handle both sub-problems simultaneously, but only provided asymptotic convergence analysis, lacking a precise characterization of the convergence rate and computational complexity. This paper aims to **systematically explore and utilize the benefits of the hypergradient structure**.

## Method

### Framework: NBO (Newton-based Bilevel Optimization)

**Core Idea**: Utilize inexact Newton methods to simultaneously solve the lower-level optimal solution $y^*(x)$ and the Hessian-inverse-vector product $u^*(x,y)$, where both share the same Hessian $H_k = \nabla_{22}^2 g(x^k, y^k)$.

**Step 1: Curvature-Aware Hypergradient Approximation**. Given $(x, y)$, simultaneously solve the linear system:

$$[\nabla_{22}^2 g(x,y)](v, u) = (\nabla_2 g(x,y), \nabla_2 f(x,y))$$

to obtain the inexact Newton direction $v$ (approximating the lower-level Newton step) and $u$ (approximating the Hessian-inverse-vector product).

**Step 2: Compute Approximate Hypergradient**:

$$d_x = \nabla_1 f(x, y-v) - \nabla_{12}^2 g(x, y-v) u$$

**Key Formula — Quadratic Decay of Single-Step Newton**: The classical Newton step $y^+ = y - [\nabla_{22}^2 g]^{-1} \nabla_2 g$ can improve the hypergradient approximation error from linear decay to quadratic decay:

$$\|\hat{\nabla}\Phi(x, y^+) - \nabla\Phi(x)\| \leq \frac{C L_{g,2}}{2\mu} \|y^*(x) - y\|^2$$

**Transforming Linear Systems into Quadratic Programming**: The sub-problems (8) and (10) are strongly convex quadratic programming problems, which are solved approximately using GD or CG.

### Algorithmic Instances

| Algorithm | Setting | Sub-problem Solver | Characteristics |
|------|------|-------------|------|
| **NBO-GD** | Deterministic | Gradient Descent (T+1 steps) | Degenerates to the SOBA/Dagréou framework when T=0 |
| **NBO-CG** | Deterministic | Conjugate Gradient | Better Hessian-vector product complexity |
| **NSBO-SGD** | Stochastic | Stochastic Gradient Descent | Suitable for large-scale scenarios |

### Convergence and Complexity (Deterministic Setting)

Under standard assumptions (lower-level strong convexity parameter $\mu$, gradient Lipschitz continuity, etc.), the convergence guarantee of NBO-GD is:

$$\min_{0 \leq k \leq K-1} \|\nabla\Phi(x^k)\|^2 \leq O(\kappa^3 / K)$$

| Method | Gradient Computation | Hessian-Vector Product |
|------|---------|---------------|
| AmIGO-GD | $O(\kappa^4 \log\kappa \cdot \epsilon^{-1})$ | $O(\kappa^4 \log\kappa \cdot \epsilon^{-1})$ |
| AID-BiO | $O(\kappa^4 \log\kappa \cdot \epsilon^{-1})$ | $O(\kappa^{3.5} \log\kappa \cdot \epsilon^{-1})$ |
| **Ours (NBO-GD)** | $O(\kappa^3 \cdot \epsilon^{-1})$ | $O(\kappa^4 \cdot \epsilon^{-1})$ |
| **Ours (NBO-CG)** | $O(\kappa^3 \cdot \epsilon^{-1})$ | $O(\kappa^{3.5} \log\kappa \cdot \epsilon^{-1})$ |

The gradient complexity is improved by a factor of $\kappa \log\kappa$.

### Stochastic Setting (NSBO-SGD)

The total sample complexity is $O(\kappa^9 \epsilon^{-2})$, which improves upon AmIGO's $O(\kappa^9 \log\kappa \cdot \epsilon^{-2})$ by a factor of $\log\kappa$.

### Lyapunov Analysis

Defines the Lyapunov function $V_k = \Phi(x^k) - \Phi^* + b_y \|y^k - y^*(x^k)\|^2 + b_u \|u^k - u^*(x^k)\|^2$. It proves that under the inexact Newton step, $\|y^{k+1} - y^*(x^{k+1})\|$ satisfies a descent condition of quadratic decay + linear perturbation, ensuring by induction that the iterates always lie within the local convergence region of Newton's method.

## Key Experimental Results

### Synthetic Experiments (Hyperparameter Optimization + Logistic Regression)

| Comparison | Result |
|------|------|
| NBO-GD (T=1) vs AmIGO (Q=10) | Comparable descent speed of the objective function |
| NBO-GD (T=1) vs AmIGO (Q=1) | NBO-GD is significantly superior |
| Evaluation by runtime | NBO-GD (T=1) is the fastest |

**Key Findings**: Even with only T=1 step to approximate the Newton direction, NBO-GD matches the performance of AmIGO with Q=10 steps of GD, indicating that curvature information is much more efficient than multi-step GD.

### Hyperparameter Optimization (IJCNN1 + Covtype)

- **IJCNN1**: The convergence speed of NSBO-SGD is significantly superior to all baselines including SOBA, SABA, StoBiO, AmIGO, SHINE, F2SA, and MA-SABA.
- **Covtype**: NSBO-SGD also converges the fastest, achieving the lowest test error.

### Data Cleaning (MNIST + FashionMNIST, 50% Noisy Labels)

- NSBO-SGD and AmIGO achieve the lowest test error.
- NSBO-SGD converges the fastest, demonstrating the efficiency advantage of integrating curvature information.

### Variance Reduction and Momentum Extensions

- NSBO-SAGA outperforms SABA.
- MA-NSBO-SAGA outperforms MA-SABA.
- The NBO framework is compatible with variance reduction techniques (such as SAGA, STORM, PAGE) and momentum techniques.

## Highlights & Insights
1. **Elegant Structural Insight**: The structure where the lower-level problem and the Hessian-inverse-vector product in the hypergradient share the same Hessian is systematically utilized. This is a highly natural yet overlooked observation.
2. **Newton Step Brings Quadratic Decay**: A single Newton step can improve the hypergradient estimation error from $O(\|y-y^*\|)$ to $O(\|y-y^*\|^2)$.
3. **Consistency Between Theory and Experiments**: The gradient complexity in the deterministic setting is improved by $\kappa \log\kappa$ times, which is also validated by the significant acceleration in experiments.
4. **T=1 Already Effective**: The inner loop requires only 1 step of approximating the Newton direction to outperform 10 steps of GD, leading to low practical deployment costs.
5. **Framework Generality**: NBO can be easily integrated with techniques such as CG, variance reduction (SAGA), and momentum (MA).

## Limitations & Future Work
1. **Requires the initial point to be within the local Newton convergence region** (BOX 1/BOX 2 conditions), which may necessitate an additional warm-up step in practice.
2. **Limited Improvement in Stochastic Settings**: NSBO-SGD only achieves a $\log\kappa$ improvement, which is less significant than the improvement in the deterministic setting.
3. **Lower-level Problem Must Be Strongly Convex**: Although extension to the convex case via aggregate functions (BAMM+NBO) is discussed, the theoretical guarantees are limited to the strongly convex case.
4. **Non-smooth/Constrained Scenarios Unconsidered**: Extensions to lower-level problems containing constraints or non-smooth terms have not yet been explored.
5. **High Cost of Hessian-Vector Products in Deep Networks**: For large-scale neural networks, the practical overhead of computing exact Hessian-vector products remains substantial.

## Related Work & Insights
- **SHINE** (Ramzi et al., 2022): Also utilizes the hypergradient structure, but with quasi-Newton methods and lacks non-asymptotic analysis.
- **AmIGO** (Arbel & Mairal, 2022a): A representative framework that solves sub-problems using multi-step GD.
- **SOBA/SABA** (Dagréou et al., 2022): Single-loop methods, to which NBO-GD degenerates when T=0.
- **F2SA** (Kwon et al., 2023): A fully first-order method that avoids Hessians but has higher complexity.
- **Insight**: The idea of sharing the Hessian can be generalized to multi-level optimization or scenarios with multiple coupled sub-problems.

## Rating
- Novelty: ⭐⭐⭐⭐ — Simple and elegant structural insight. Incorporating inexact Newton into bilevel optimization is a natural yet novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive baselines, covering synthetic experiments, hyperparameter optimization, data cleaning, and various extensions.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, rigorous theories, and clear tabular comparisons.
- Value: ⭐⭐⭐⭐ — Provides a more efficient hypergradient estimation method for bilevel optimization, with contributions in both theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](sassha_sharpness-aware_adaptive_second-order_optimization_with_stable_hessian_ap.md)
- [\[ICLR 2026\] Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation](../../ICLR2026/optimization/reducing_contextual_stochastic_bilevel_optimization_via_structured_function_appr.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](../../NeurIPS2025/optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
