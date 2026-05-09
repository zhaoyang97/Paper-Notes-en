---
title: >-
  [Paper Note] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding
description: >-
  [ICLR2026][LLM Evaluation][Quadratic Programming] This paper proposes FlexQP — an "always feasible" convex quadratic programming (QP) solver based on $\ell_1$ elastic relaxation — and combines it with deep unfolding to learn an LSTM feedback policy that accelerates convergence, yielding Deep FlexQP. When embedded as a submodule within an SQP framework, it solves nonlinear trajectory optimization problems 4–16× faster than OSQP, reduces safety violations in predictive safety filters by over 70%, and improves task completion rates by 43%.
tags:
  - ICLR2026
  - LLM Evaluation
  - Quadratic Programming
  - Deep Unfolding
  - ADMM
  - Sequential Quadratic Programming
  - LSTM Policy
  - PAC-Bayes
date: 2026-05-08
content_hash: 45141a169ee43ce2
---

# Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding

**Conference**: ICLR2026
**arXiv**: [2512.01565](https://arxiv.org/abs/2512.01565)
**Code**: To be confirmed
**Area**: LLM Evaluation
**Keywords**: Quadratic Programming, Deep Unfolding, ADMM, Sequential Quadratic Programming, LSTM Policy, PAC-Bayes

## TL;DR
This paper proposes FlexQP — an "always feasible" convex quadratic programming (QP) solver based on $\ell_1$ elastic relaxation — and combines it with deep unfolding to learn an LSTM feedback policy that accelerates convergence, yielding Deep FlexQP. When embedded as a submodule within an SQP framework, it solves nonlinear trajectory optimization problems 4–16× faster than OSQP, reduces safety violations in predictive safety filters by over 70%, and improves task completion rates by 43%.

## Background & Motivation

**Background**: Quadratic programming (QP) is a foundational subproblem in optimal control, combinatorial optimization, and machine learning. Sequential quadratic programming (SQP) handles nonlinear non-convex constrained optimization by iteratively solving QP subproblems.

**Limitations of Prior Work**: Constraint linearization in SQP frequently produces infeasible QP subproblems. Conventional solvers such as OSQP either terminate with an error or require dedicated infeasibility recovery routines (e.g., SNOPT's elastic mode), which do not scale well. Moreover, tuning ADMM hyperparameters ($\rho, \sigma, \alpha$) is notoriously difficult.

**Core Idea**: The paper recasts constrained QP as unconstrained optimization via $\ell_1$ exact relaxation — recovering the original solution when feasible (Theorem 3.1) and automatically identifying the sparsest constraint violation when infeasible. Deep unfolding combined with an LSTM then replaces manual hyperparameter tuning with a dimension-agnostic learned feedback policy.

## Method

### Overall Architecture
Primal QP: $\min_x \frac{1}{2}x^\top P x + q^\top x$, s.t. $Gx \leq h, Ax = b$ → introduce slack variables → $\ell_1$ elastic relaxation → ADMM splitting → two-block iteration (linear system solve + soft thresholding). Deep unfolding treats the $K$-step ADMM iteration as a $K$-layer network and learns per-layer parameters.

### Key Design 1: Exact Relaxation in FlexQP
- Hard constraints are replaced by the $\ell_1$ penalty $\mu_I \|Gx+s-h\|_1 + \mu_E \|Ax-b\|_1$
- **Theorem 3.1**: When $\mu_I \geq \|y_I^*\|_\infty$ and $\mu_E \geq \|y_E^*\|_\infty$, the relaxed solution coincides exactly with the original solution
- **Theorem 3.2**: Convergence is guaranteed under a weak coercivity assumption
- In the infeasible case, $z_I^*, z_E^*$ automatically provide sparse certificates of constraint violation

### Key Design 2: LSTM Feedback Policy
- Independent policies are learned for inequality constraints ($\pi_I$), equality constraints ($\pi_E$), and the relaxation parameter ($\pi_\alpha$)
- Policy inputs consist of ADMM variables together with primal/dual residuals, applied in batch along the constraint dimension → **dimension-agnostic**, generalizing to large-scale problems
- The LSTM captures long-range dependencies in the optimization history to adaptively adjust $\rho, \mu, \alpha$

### Key Design 3: Normalized Training Loss and PAC-Bayes Bound
- The training loss incorporates Lagrange multipliers: $\min_\theta \sum_k \|\xi^k(\theta) - \xi^*\|_2 / \|\xi^*\|_2$, $\xi = (x, y_I, y_E)$, implicitly enforcing $\mu \geq |y^*|$
- A log-scale PAC-Bayes loss (Eq. 14) is proposed: in the regime of very small residuals, it carries orders of magnitude more information than the standard loss (Eq. 13), yielding tighter generalization guarantees

## Key Experimental Results

### Small- and Medium-Scale QP (500 training / 1000 test problems)

| Solver | Convergence Speed (iterations) | Final Residual |
|--------|-------------------------------|----------------|
| OSQP (hand-tuned) | Baseline | Baseline |
| Deep OSQP | Better than OSQP | Better than OSQP |
| Deep OSQP-Improved | Further improvement | Further improvement |
| **Deep FlexQP** | **Fastest among all methods** | **Lowest among all methods** |

### Large-Scale QP (10k variables / 10–20k constraints)

| Problem Class | Advantage of Deep FlexQP |
|---------------|--------------------------|
| Portfolio Optimization (10k var, 10k con) | Fewest iterations; generalizes via fine-tuning a small model |
| SVM (10k var, 20k con) | Fewest CG iterations |

### SQP Nonlinear Optimization

| Metric | Deep FlexQP + SQP vs. OSQP + SQP |
|--------|----------------------------------|
| Trajectory optimization speed | **4–16× faster** (average over 100 problems) |
| Safety filter safety violations | **Reduced by >70%** |
| Safety filter task completion rate | **Improved by 43%** |

### Key Findings
- The FlexQP architecture itself (elastic relaxation + LSTM) is the primary driver of superiority — under the same loss function, Deep OSQP variants benefit far less from fine-tuning than Deep FlexQP
- Training on small-scale problems followed by fine-tuning for 5 epochs on 100 large-scale problems suffices to generalize to 10k+ dimensional instances
- The log-scale PAC-Bayes bound renders generalization guarantees practically meaningful, whereas the standard bound is uninformative in the small-residual regime

## Highlights & Insights
- **Theoretical elegance**: The combination of $\ell_1$ exact relaxation, ADMM, and deep unfolding is organic, with each component backed by rigorous mathematical guarantees
- **High practical value**: The approach resolves the core engineering bottleneck of infeasible subproblems in SQP without requiring additional recovery routines
- The dimension-agnostic LSTM policy design enables a single trained model to generalize to problems of arbitrary scale

## Limitations & Future Work
- Training overhead on large-scale problems remains substantial (approximately 3 hours per epoch); full training would require 300+ days
- Validation is limited to dense QP instances; sparse QP problems (e.g., power network optimization) may require different strategies
- The interpretability of the LSTM policy is limited, making it difficult to understand the learned tuning rules

## Related Work & Insights
- Compared with Deep OSQP by Saravanos et al. (2025), the key advantages of FlexQP lie in native infeasibility handling and vector-level (rather than scalar-level) penalty parameter policies
- This work may inspire the application of deep unfolding to other optimization algorithms such as interior-point methods and Frank-Wolfe

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of elastic relaxation and deep unfolding is novel, though each individual component is not entirely new
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers small- to large-scale QP and nonlinear SQP across finance, ML, and control domains
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rigorous theoretical derivations
- Value: ⭐⭐⭐⭐⭐ Addresses a core engineering pain point in SQP with broad application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](../../AAAI2026/llm_evaluation/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[CVPR 2026\] Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](../../CVPR2026/llm_evaluation/hiercos_making_deep_features_hierarchyaware_via_co.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/llm_evaluation/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)

</div>

<!-- RELATED:END -->
