---
title: >-
  [Paper Note] Incremental Gradient Descent with Small Epoch Counts is Surprisingly Slow on Ill-Conditioned Problems
description: >-
  [ICML 2025][Optimization][incremental gradient descent] This paper systematically investigates the convergence behavior of incremental gradient descent (IGD) in the small epoch regime ($K \lesssim \kappa$). It proves that IGD is at least $n$ times slower than SGD with replacement in this regime, and its convergence rate catastrophically deteriorates to an exponential rate when the component functions are non-convex.
tags:
  - "ICML 2025"
  - "Optimization"
  - "incremental gradient descent"
  - "permutation-based SGD"
  - "convergence lower bound"
  - "small epoch regime"
  - "condition number"
date: 2026-05-08
content_hash: 1476c0e0c3c24891
---

# Incremental Gradient Descent with Small Epoch Counts is Surprisingly Slow on Ill-Conditioned Problems

---

**Conference**: ICML 2025  
**arXiv**: [2506.04126](https://arxiv.org/abs/2506.04126)  
**Code**: None  
**Area**: Optimization  
**Keywords**: incremental gradient descent, permutation-based SGD, convergence lower bound, small epoch regime, condition number

## TL;DR

This paper systematically investigates the convergence behavior of incremental gradient descent (IGD) in the small epoch regime ($K \lesssim \kappa$). It proves that IGD is at least $n$ times slower than SGD with replacement in this regime, and its convergence rate catastrophically deteriorates to an exponential rate when the component functions are non-convex.

---

## Background & Motivation

**Background**: Finite-sum optimization $\min_x F(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$ is core to machine learning. Permutation-based SGD (such as Random Reshuffling) is widely used due to its superior practical performance compared to SGD with replacement. Recent theoretical works have proven that RR achieves a convergence rate of $\mathcal{O}(1/nK^2)$ under smooth strongly convex objectives, outperforming the $\mathcal{O}(1/nK)$ rate of SGD with replacement.

**Limitations of Prior Work**: Almost all of the aforementioned convergence analyses are restricted to the "large epoch regime"—where the number of epochs $K$ is much larger than the condition number $\kappa = L/\mu$. However, in practical deep learning training, the condition number of neural network optimization landscapes is extremely large, and due to computational constraints, the number of epochs $K$ is typically much smaller than $\kappa$. Lower bounds by Safran & Shamir (2021) indicate that RR cannot even outperform SGD with replacement in the small epoch regime, but the exact behavior of IGD in this regime remains an open question.

**Key Challenge**: Existing analyses require step sizes $\eta \leq \mathcal{O}(1/nL)$ to control the cumulative error within a single epoch. However, when $K$ is small, the step size must be set to at least $\Omega(1/K)$ to drive the iterates close to the optimal solution. These two requirements contradict each other when $K \ll \kappa$, making existing upper bounds loose.

**Goal**: As a first step toward understanding the convergence behavior of permutation-based SGD in the small epoch regime, this paper comprehensively characterizes the convergence lower and upper bounds of IGD (the simplest deterministic permutation SGD) in this regime.

**Key Insight**: IGD is deterministic (with a fixed permutation $\sigma_k =$ identity), which makes it technically more tractable to analyze than random permutations. Tight lower bounds can be obtained by carefully constructing objective functions.

**Core Idea**: Through three classes of elegant function constructions (with shared Hessian, strongly convex components, and non-convex components), this paper reveals that the convergence behavior of IGD progressively deteriorates from $\Omega(G^2/\mu K)$ to an exponential $\Omega((1+\kappa/2nK)^n)$ in the small epoch regime.

## Method

### Overall Architecture

The paper focuses on the finite-sum problem $F(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$, where $F$ is $\mu$-strongly convex and each $f_i$ is $L$-smooth. It considers IGD, the simplest form of permutation-based SGD (where updates are performed sequentially from 1 to $n$ in a fixed order in each epoch), and analyzes its convergence rate when $K \leq \kappa$. The analysis is categorized into three levels: (1) components sharing the same Hessian; (2) components being individually strongly convex; (3) allowing non-convex components. Meanwhile, tight bounds in the large epoch regime are provided as benchmarks.

### Key Designs

1. **Identical Hessian Lower Bound (Theorem 3.1)**:

    - **Function**: Establishes a fundamental lower bound under the most favorable scenario.
    - **Mechanism**: Constructs a 3-dimensional objective function where all components share the same Hessian ($\nabla^2 f_i = \nabla^2 F$), and proves that for any constant step size, the final iterate of IGD satisfies $F(x_n^K) - F(x^*) \gtrsim G^2/(\mu K)$. This is $n$ times slower than the $\mathcal{O}(G^2/\mu nK)$ upper bound of SGD with replacement.
    - **Design Motivation**: Even in the most "similar" case of component functions (sharing the same curvature), IGD is still slower than SGD with replacement, indicating that deterministic permutation itself is detrimental in the small epoch regime. The corresponding upper bound (Theorem 3.2) matches in the 1-dimensional case, proving that this lower bound is tight.

2. **Strongly Convex Component Lower Bound (Theorem 3.3)**:

    - **Function**: Reveals the severe deterioration of convergence rate when the identical Hessian assumption is removed.
    - **Mechanism**: Constructs a 4-dimensional objective function where the minimizers of individual components form a regular $n$-gon through rotational transformations. By carefully choosing the initialization point, the iterative sequence maintains rotational symmetry, also forming a regular $n$-gon and keeping a constant distance from the global optimum. The proven lower bound is $\Omega(LG^2/\mu^2 \cdot \min\{1, \kappa^2/K^4\})$.
    - **Design Motivation**: Compared to Theorem 3.1, this lower bound is consistently larger throughout the small epoch regime (larger by a factor of $\kappa K$ or $\kappa^3/K^3$) and matches the upper bound in Theorem 3.2 when $K = \Theta(\sqrt{\kappa})$. This highlights the severe impact of component Hessian heterogeneity on the convergence of IGD.

3. **Non-convex Component Lower Bound (Theorem 3.5)**:

    - **Function**: Demonstrates the catastrophic deceleration caused by non-convex components.
    - **Mechanism**: Constructs a 2-dimensional function allowing some component functions to be concave in certain directions. The proven lower bound contains a term of $(1 + L/(2\mu nK))^n$, which is approximately $\exp(\kappa/(2K))$—exponentially large when $K \ll \kappa$.
    - **Design Motivation**: This starkly contrasts with the case of strongly convex components. Interestingly, experiments show that RR performs robustly on the same construction, suggesting that random permutations inherently mitigate the effect of non-convex components. This is the first work in the literature to show that permutation-based SGD can suffer from exponentially slow convergence in the small epoch regime.

### Additional Results

**Existence of Optimal Permutations (Theorem 3.7)**: Using the Herding algorithm, this paper proves the existence of a permutation that allows permutation-based SGD to outperform SGD with replacement even in the small epoch regime, achieving a convergence rate of $\mathcal{O}(H^2 L^2 G_*^2 / (\mu^3 n^2 K^2))$. However, this permutation relies on gradient information at the optimum and is not practical.

**Tight Bounds for Large Epoch Regime (Section 4)**: Establishes matching upper and lower bounds for both convex and non-convex component cases when $K \gtrsim \kappa$. In the non-convex component case, the rate gap in the large epoch regime is only by a factor of $\kappa$, contrasting sharply with the exponential gap in the small epoch regime.

## Key Experimental Results

### Main Convergence Rate Comparison

| Regime | Component Assumption | Lower Bound (IGD) | Upper Bound (Arbitrary Permutation SGD) | SGD with Replacement |
|------|---------|-----------|-------------------|-----------|
| Small epoch, identical Hessian | Strongly convex | $\Omega(G^2/\mu K)$ | $\mathcal{O}(G_*^2/\mu K)$ | $\mathcal{O}(G^2/\mu nK)$ |
| Small epoch | Strongly convex | $\Omega(LG^2/\mu^2 \cdot \min\{1,\kappa^2/K^4\})$ | $\mathcal{O}(L^2G_*^2/\mu^3 K^2)$ | $\mathcal{O}(G^2/\mu nK)$ |
| Small epoch | Including non-convex | $\Omega(G^2/L \cdot (1+L/2\mu nK)^n)$ | — | $\mathcal{O}(G^2/\mu nK)$ |
| Large epoch | Convex | $\Omega(LG^2/\mu^2 K^2)$ | $\mathcal{O}(LG_*^2/\mu^2 K^2)$ | — |
| Large epoch | Including non-convex | $\Omega(L^2G^2/\mu^3 K^2)$ | $\mathcal{O}(L^2G^2/\mu^3 K^2)$ | — |

### Key Rate Comparison

| Comparison | Gap Factor |
|------|---------|
| IGD vs SGD with replacement (small epoch, identical Hessian) | Factor of $n$ |
| IGD (strongly convex components vs identical Hessian) | Factor of $\kappa K$ or $\kappa^3/K^3$ |
| IGD (non-convex vs strongly convex components, small epoch) | Exponential vs Polynomial |
| IGD (non-convex vs convex components, large epoch) | Factor of only $\kappa$ |

### Key Findings

- IGD in the small epoch regime is at least $n$ times slower than SGD with replacement, even under the most favorable assumptions.
- After removing the assumption of identical component Hessians, the convergence of IGD deteriorates further, with higher powers of $\kappa$ appearing in the lower bound.
- When non-convex components are allowed, the deterioration in the small epoch regime is exponential, whereas it is only polynomial in the large epoch regime—showing a qualitative difference in behavior between the small and large epoch regimes.
- An (impractical) optimal permutation exists that can outperform SGD with replacement in the small epoch regime, showing that the choice of permutation is crucial.

## Highlights & Insights

- The core contribution of this paper is the first systematic characterization of the convergence landscape in the small epoch regime, revealing its fundamental differences from the large epoch regime—providing important guidance for understanding practical deep learning training.
- The rotating polygon construction in Theorem 3.3 is highly elegant, perfectly bridging geometric intuition and convergence analysis. The idea of keeping the iteration sequence symmetric within a regular $n$-gon is very impressive.
- The exponential lower bound in Theorem 3.5 is unexpected, showing that non-convex components can lead to catastrophically slow convergence under small epochs.
- At the same time, tight bounds are provided for the large epoch regime, which, when compared with the small epoch results, forms a complete picture—non-convex components only cause a slowdown by a factor of $\kappa$ in the large epoch regime but lead to an exponential slowdown in the small epoch regime.
- Experimental validation (Appendix G) confirms the iterative behavior of the theoretical constructions, lending credibility to the results.
- The paper is clearly written. The overview in Table 1 and Figure 1 provides excellent references, making the complex theoretical results clear at a glance.

## Limitations & Future Work

- The lower bounds are specifically for IGD (fixed permutation) and have not yet been extended to RR or other random permutation strategies—this is the most important open problem.
- The upper bound in Theorem 3.2 is limited to the 1-dimensional case; high-dimensional generalization remains a technical challenge (although it can be directly applied to diagonal Hessian objectives).
- All results assume a constant step size; the behavior under varying step-size schedules (such as cosine schedule) is unknown—in practice, step sizes are almost always varying.
- Non-convex objective functions (where $F$ itself is non-convex) are not covered, as only strongly convex $F$ with potentially non-convex $f_i$ is considered.
- There is still a gap between these results and practical deep learning scenarios (randomized permutations, adaptive step sizes, large-scale non-convex problems).
- The optimal permutation proven in Theorem 3.7 requires gradient information at the optimal solution and is impractical—whether practical near-optimal permutation strategies can be designed is an important direction.
- The quantitative relationship between component function heterogeneity (such as Hessian discrepancy) and the convergence rate has not been fully revealed.

## Related Work & Insights

- The RR small epoch lower bound $\Omega(G^2/\mu nK)$ by Safran & Shamir (2021) is an important baseline for this paper, which improves IGD's lower bound from $G^2/\mu K^2$ to $G^2/\mu K$.
- The gradient balanced permutation strategy GraB (Lu et al., 2022) is optimal in the large epoch regime, but its behavior in the small epoch regime remains unknown—this is an interesting open question.
- The upper bounds by Mishchenko et al. (2020) serve as a direct comparison, and this paper reveals their looseness under multiple settings.
- The results of this paper imply that: in practical scenarios with large condition numbers and few training epochs (such as the early stages of large model training), the choice of permutation strategy might be more important than previously thought.
- Borrowing the Herding algorithm from combinatorial optimization to optimization theory demonstrates the value of cross-disciplinary tools in analyzing permutation strategies.
- Implications for practice: on highly ill-conditioned problems, specialized permutation strategies or learning rate schedules may need to be designed specifically for the small epoch regime.

## Rating

⭐⭐⭐⭐ The theoretical contribution is solid. The three-level progressive lower bound constructions (identical Hessian $\rightarrow$ strongly convex $\rightarrow$ non-convex components) are well-structured, each revealing new mathematical phenomena. In particular, the jump from polynomial to exponential under non-convex components is highly unexpected. The paper structure is clear, and the summary table is crisp. However, as a purely theoretical work, its direct guidance for practical algorithm design is limited, and the main results are restricted to IGD rather than the more widely-used RR, leaving many open questions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](quantum_optimization_via_gradient-based_hamiltonian_descent.md)
- [\[ICLR 2026\] Corner Gradient Descent](../../ICLR2026/optimization/corner_gradient_descent.md)
- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
