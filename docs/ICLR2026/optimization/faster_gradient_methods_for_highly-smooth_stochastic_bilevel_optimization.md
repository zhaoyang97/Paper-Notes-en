---
title: >-
  [Paper Note] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization
description: >-
  [ICLR2026][Optimization][bilevel optimization] By reinterpreting the F2SA method as a forward-difference approximation to the hyper-gradient, this paper proposes the F2SA-p algorithm family based on higher-order finite differences. Under higher-order smoothness conditions, the SFO complexity for stochastic bilevel optimization is improved from $\tilde{\mathcal{O}}(\epsilon^{-6})$ to $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$, and a matching lower bound of $\Omega(\epsilon^{-4})$ is established, showing near-optimality for sufficiently large $p$.
tags:
  - ICLR2026
  - Optimization
  - bilevel optimization
  - stochastic optimization
  - finite difference
  - hyper-gradient estimation
  - complexity lower bound
date: 2026-05-08
content_hash: 6d6b6716edc4a9b8
---

# Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization

**Conference**: ICLR2026
**arXiv**: [2509.02937](https://arxiv.org/abs/2509.02937)
**Code**: [GitHub](https://github.com/TrueNobility303/F2BA)
**Area**: Optimization
**Keywords**: bilevel optimization, stochastic optimization, finite difference, hyper-gradient estimation, complexity lower bound

## TL;DR
By reinterpreting the F2SA method as a forward-difference approximation to the hyper-gradient, this paper proposes the F2SA-p algorithm family based on higher-order finite differences. Under higher-order smoothness conditions, the SFO complexity for stochastic bilevel optimization is improved from $\tilde{\mathcal{O}}(\epsilon^{-6})$ to $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$, and a matching lower bound of $\Omega(\epsilon^{-4})$ is established, showing near-optimality for sufficiently large $p$.

## Background & Motivation
Bilevel optimization appears widely in meta-learning, hyperparameter tuning, adversarial training, and reinforcement learning. The general form is $\min_{\bm{x}} f(\bm{x}, \bm{y}^*(\bm{x}))$, where $\bm{y}^*(\bm{x}) = \arg\min_{\bm{y}} g(\bm{x}, \bm{y})$. Existing methods fall into two categories:

1. **Hessian-vector product (HVP)-based methods**: e.g., BSA, stocBiO — require a stochastic Hessian oracle and rely on stronger assumptions.
2. **Pure first-order methods**: e.g., F2SA — require only a stochastic gradient oracle, rely on weaker assumptions, and scale in practice to LLM training at the 32B scale.

F2SA achieves an SFO complexity upper bound of $\tilde{\mathcal{O}}(\epsilon^{-6})$ under standard SGD assumptions, yet a significant gap remains relative to the $\Omega(\epsilon^{-4})$ lower bound for single-level optimization. The central question is: **can pure first-order methods achieve optimal complexity in bilevel optimization?**

## Core Problem
In the stochastic bilevel optimization setting with a nonconvex upper-level and strongly convex lower-level function, can the higher-order smoothness of the objective be exploited to break through the $\tilde{\mathcal{O}}(\epsilon^{-6})$ complexity barrier of existing pure first-order methods and approach the $\Omega(\epsilon^{-4})$ lower bound?

## Method

### Key Insight: F2SA as Forward Differencing
The authors first reveal a deep connection between F2SA and finite differences. Define the perturbed lower-level problem $g_\nu(\bm{x}, \bm{y}) = \nu f(\bm{x}, \bm{y}) + g(\bm{x}, \bm{y})$ and its optimal value function $\ell_\nu(\bm{x})$. One can show that:

$$\frac{\partial^2}{\partial \nu \partial \bm{x}} \ell_\nu(\bm{x})\big|_{\nu=0} = \nabla \varphi(\bm{x})$$

The gradient of the F2SA penalty function is essentially a **forward-difference** approximation $(\ell_\nu - \ell_0)/\nu$ to $\partial \ell_\nu / \partial \nu|_{\nu=0}$, yielding only a first-order error guarantee of $\mathcal{O}(\nu)$.

### F2SA-p: Higher-Order Finite Differences
Since forward differencing is merely the simplest among many finite-difference schemes, higher-order schemes can naturally reduce the approximation error. By classical results in numerical analysis (Lemma 3.1), a $p$-th order finite difference achieves $\mathcal{O}(\nu^p)$ approximation accuracy:

- **$p=1$ (forward difference)**: $\alpha_0=-1, \alpha_1=1$ — recovers the original F2SA.
- **$p=2$ (central difference)**: $\alpha_{-1}=-1/2, \alpha_1=1/2$ — corresponds to the symmetric penalty formulation F2SA-2.

For general even $p$, the algorithm follows a double-loop structure:

1. **Inner loop**: For each $j = -p/2, \ldots, p/2$, run $K$ steps of SGD to solve the perturbed lower-level problem $g_{j\nu}(\bm{x}, \cdot)$, approximating $\bm{y}_{j\nu}^*(\bm{x})$.
2. **Outer loop**: Construct a hyper-gradient estimate $\Phi_t$ via a linear combination of gradient information at each perturbed point using finite-difference coefficients $\{\alpha_j\}$, then perform a normalized gradient step $\bm{x}_{t+1} = \bm{x}_t - \eta_x \Phi_t / \|\Phi_t\|$.

### Higher-Order Smoothness Assumption
The improvements rely on higher-order smoothness of the objective in the lower-level variable $\bm{y}$ (Assumption 2.5), requiring Lipschitz continuity of higher-order derivatives of $f$ and $g$ with respect to $\bm{y}$. This is strictly weaker than requiring joint higher-order smoothness in $(\bm{x}, \bm{y})$. The softmax function, which is smooth to all orders, is a canonical example; thus many hyperparameter tuning problems involving logistic regression (e.g., data hyper-cleaning, learn-to-regularize) naturally satisfy this condition.

### Complexity Analysis
The Lipschitz continuity of higher-order derivatives of $\ell_\nu(\bm{x})$ with respect to $\nu$ is established via the higher-dimensional Faà di Bruno formula (Lemma 3.2), ensuring an $\mathcal{O}(\nu^p)$ finite-difference error. The main theorem gives:

$$\text{SFO complexity} = \tilde{\mathcal{O}}\left(p \kappa^{9+2/p} \epsilon^{-4-2/p}\right)$$

When $p = \Omega(\log\epsilon^{-1}/\log\log\epsilon^{-1})$, the complexity simplifies to $\tilde{\mathcal{O}}(\kappa^9 \epsilon^{-4})$, matching the optimal complexity of HVP-based methods under stochastic Hessian assumptions.

### Normalized Gradient Descent
Unlike the original F2SA, the outer loop employs a normalized gradient step $\bm{x}_{t+1} = \bm{x}_t - \eta_x \Phi_t / \|\Phi_t\|$. Normalization controls the magnitude of perturbations to $\bm{y}_{j\nu}^*(\bm{x}_t)$ and simplifies the inner-loop analysis. The authors note that the same guarantees should be achievable with standard gradient steps via a more involved analysis.

### Even vs. Odd $p$
For even $p$, the central-difference coefficient satisfies $\alpha_0 = 0$, requiring only $p$ perturbed points; odd $p$ requires $p+1$ points. Consequently, F2SA-2 is strictly superior to F2SA ($p=1$): both solve 2 lower-level problems, but F2SA-2 achieves a higher-order error guarantee. Even without second-order smoothness, F2SA-2 degrades gracefully to first-order error and is no worse than F2SA.

### Lower Bound
A lower bound is established by constructing a fully separable bilevel instance ($f(\bm{x}, \bm{y}) \equiv f_{\bm{U}}(\bm{x})$, $g(\bm{x}, y) = \mu y^2/2$), which reduces the bilevel problem to a single-level one and inherits the $\Omega(\epsilon^{-4})$ lower bound of Arjevani et al. (2023). This construction elegantly satisfies all higher-order smoothness conditions by design, avoiding the smoothness violations present in constructions from prior work (Dagréou et al. 2024, Kwon et al. 2024a).

## Key Experimental Results
Experiments are conducted on a learn-to-regularize logistic regression task on the 20 Newsgroups dataset (18,000 samples, 130,107 features, 20 classes), which naturally satisfies arbitrary-order higher-order smoothness.

- Baselines: F2SA-p ($p \in \{2,3,5,8,10\}$) vs. F2SA vs. stocBiO vs. MRBO vs. VRBO.
- Inner loop $K=10$, outer loop $T=1000$.
- The F2SA-p family exhibits monotonically improving convergence as $p$ increases, validating the theoretical predictions.
- Appendix experiments on a 5-layer ReLU MLP suggest potential applicability to non-smooth, non-convex settings.

## Highlights & Insights
1. **Elegant theoretical insight**: Reinterpreting F2SA as a finite-difference approximation provides a natural and principled direction for algorithmic improvement.
2. **Nearly free improvement**: F2SA-2 solves only 2 lower-level problems (identical to F2SA) yet achieves a second-order error guarantee.
3. **Complete theoretical picture**: Upper bound $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$ paired with lower bound $\Omega(\epsilon^{-4})$ establishes near-optimality under higher-order smoothness.
4. **Tighter known bounds**: The $p=1$ case improves the $\kappa$ dependence; the $p=2$ case corrects prior work's Hessian convergence bounds.
5. **Weaker assumptions**: Only higher-order smoothness in $\bm{y}$ is required, without joint higher-order smoothness or stochastic Hessian assumptions.

## Limitations & Future Work
1. A gap between upper and lower bounds persists for small $p$ (especially $p=1$); the optimal complexity for $p=1$ remains open.
2. The dependence on condition number $\kappa$ retains a gap between the $\Omega(\kappa^9)$ lower and upper bounds.
3. Experiments are limited to convex lower-level problems; extension to nonconvex–nonconvex structured bilevel optimization remains unexplored.
4. The method has not been combined with variance reduction or momentum techniques, leaving room for further acceleration.
5. The higher-order smoothness assumption limits applicability; it is generally not satisfied in deep networks.

## Related Work & Insights

| Method | Smoothness Order | Complexity | Requires HVP |
|--------|-----------------|------------|--------------|
| BSA | 1st order | $\tilde{\mathcal{O}}(\epsilon^{-6})$ SFO + $\tilde{\mathcal{O}}(\epsilon^{-4})$ HVP | Yes |
| stocBiO | 1st order | $\tilde{\mathcal{O}}(\epsilon^{-4})$ | Yes |
| F2SA | 1st order | $\tilde{\mathcal{O}}(\kappa^{12}\epsilon^{-6})$ | No |
| **F2SA-p** | **$p$-th order** | **$\tilde{\mathcal{O}}(p\kappa^{9+2/p}\epsilon^{-4-2/p})$** | **No** |
| Lower bound | $p$-th order | $\Omega(\epsilon^{-4})$ | — |

Compared to Chayti & Jaggi (2024), who establish a finite-difference connection only for meta-learning and symmetric approximations, this work generalizes to general bilevel optimization and arbitrary-order finite differences. Compared to Huang et al. (2025), which requires joint higher-order smoothness, this paper requires only higher-order smoothness in $\bm{y}$.

The finite-difference perspective reframes algorithm design for bilevel optimization and may extend naturally to other problems involving the implicit function theorem (e.g., minimax and compositional optimization). F2SA-2 serves as a near-zero-cost drop-in replacement for F2SA with stronger theoretical guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ — The finite-difference perspective is novel and natural; the algorithm family is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments are relatively simple, limited to convex lower-level logistic regression; large-scale validation is insufficient.
- Writing Quality: ⭐⭐⭐⭐⭐ — The paper is clearly structured with a complete logical chain from insight to algorithm to theory.
- Value: ⭐⭐⭐⭐ — Establishes important new results in bilevel optimization theory, significantly narrowing the gap between upper and lower bounds.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](../../NeurIPS2025/optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)

<!-- RELATED:END -->
