---
title: >-
  [Paper Note] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers
description: >-
  [ICLR 2026 (Oral)][Optimization][Hard-constrained neural networks] This paper proposes the Πnet architecture, which appends a Douglas-Rachford operator splitting-based orthogonal projection layer to the output of a neura…
tags:
  - "ICLR 2026 (Oral)"
  - "Optimization"
  - "Hard-constrained neural networks"
  - "orthogonal projection"
  - "operator splitting"
  - "implicit function theorem"
  - "Douglas-Rachford"
date: 2026-05-08
content_hash: 048d88aca300f48e
---

# Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers

**Conference**: ICLR 2026 (Oral)  
**arXiv**: [2508.10480](https://arxiv.org/abs/2508.10480)  
**Code**: [github.com/antonioterpin/pinet](https://github.com/antonioterpin/pinet)  
**Area**: Optimization / Constrained Neural Networks  
**Keywords**: Hard-constrained neural networks, orthogonal projection, operator splitting, implicit function theorem, Douglas-Rachford

## TL;DR

This paper proposes the Πnet architecture, which appends a Douglas-Rachford operator splitting-based orthogonal projection layer to the output of a neural network to guarantee strict satisfaction of convex constraints, and employs the implicit function theorem for efficient backpropagation. Πnet substantially outperforms existing methods in training time, solution quality, and hyperparameter robustness.

## Background & Motivation

Many real-world applications require solving parametric constrained optimization problems: given a context (parameter) $x$, solve $\min_y \varphi(y,x)$ s.t. $y \in \mathcal{C}(x)$. Such problems arise frequently in power systems, logistics scheduling, model predictive control, and motion planning.

### Limitations of Prior Work

**Soft constraint methods**: Add penalty terms for constraint violations to the loss function. These fail to guarantee feasibility at inference time, and penalty coefficient tuning is notoriously difficult.

**DC3**: Enforces feasibility via equality completion and inequality correction, but is essentially a soft-constraint approach and is sensitive to hyperparameters.

**Loop unrolling**: Methods such as Dykstra's projection algorithm require backpropagating through all iterations, incurring prohibitive memory and computational costs.

**cvxpylayers/JAXopt**: General-purpose but lack structure-aware optimization for projection problems, resulting in long training times.

### Root Cause

Can one design a **feasible-by-design** neural network architecture whose outputs automatically satisfy given convex constraints for any network weights? The key challenges are: how to efficiently perform the forward projection, and how to backpropagate through it efficiently.

## Method

### Overall Architecture

The overall pipeline of Πnet (as shown in Figure 1):

1. **Backbone network**: An arbitrary standard neural network $f(x;\theta)$ produces a raw output $y_{raw}$.
2. **Projection layer**: $y_{raw}$ is orthogonally projected onto the feasible set $\mathcal{C}(x)$, yielding $y = \Pi_{\mathcal{C}(x)}(y_{raw})$.
3. **Training**: Gradients through the projection layer are computed efficiently via the implicit function theorem, and standard optimizers update the backbone parameters.

### Key Designs

1. **Constraint set decomposition** → Makes projection tractable → **Design Motivation**: Decompose general convex constraints into easily projected subsets.

   The constraint set $\mathcal{C}$ is represented as $\mathcal{C} = \Pi_d(\mathcal{A} \cap \mathcal{K})$, where:
    - $\mathcal{A}$ is an affine subspace (hyperplane) defined by matrix $A$ and offset $b$
    - $\mathcal{K} = \mathcal{K}_1 \times \mathcal{K}_2$ is a Cartesian product of simple sets (e.g., box constraints)
    - The individual projections $\Pi_\mathcal{A}$ and $\Pi_\mathcal{K}$ each admit closed-form solutions

   This decomposition covers a wide range of practical constraints: polyhedra, second-order cones, sparse constraints, simplices, and their intersections.

2. **Forward pass: Douglas-Rachford operator splitting** → Iteratively solves the projection → **Design Motivation**: Exploit problem structure for efficient projection.

   The projection problem is reformulated as composite optimization $\min_z g(z) + h(z)$, where $g = \mathcal{I}_\mathcal{A}$ and $h = \|y - y_{raw}\|^2 + \mathcal{I}_\mathcal{K}$.

   Douglas-Rachford iterations:
    - $z_{k+1} = \Pi_\mathcal{A}(s_k)$ (affine projection, closed-form)
    - $t_{k+1} = \Pi_\mathcal{K}(\cdot)$ (box or cone projection, closed-form)
    - $s_{k+1} = s_k + \omega(t_{k+1} - z_{k+1})$

   Under strict feasibility, the iterations converge to the true projection.

3. **Backward pass: implicit function theorem** → Avoids loop unrolling → **Design Motivation**: Reduce the computational cost of backpropagation.

   The fixed point $s_\infty(y_{raw}) = \Phi(s_\infty(y_{raw}), y_{raw})$ satisfies the implicit function condition.

   Backpropagation requires solving a single linear system $(I - \partial\Phi/\partial s)^\top \xi = v$ via the bicgstab iterative solver, with per-step cost comparable to one forward iteration.

4. **Matrix equilibration (Ruiz equilibration)** → Improves numerical conditioning → **Design Motivation**: Accelerate convergence of the projection layer.

   Diagonal scaling $D_r A D_c$ improves the condition number of $A$, enabling faster convergence of the forward iterations.

5. **Automatic hyperparameter tuning** → Evaluates projection quality on a validation subset → **Design Motivation**: Reduce user burden in hyperparameter selection.

   Only a small number of parameters need tuning ($\sigma$, $\omega$, number of iterations, etc.), and an automated tuning pipeline is provided.

### Loss & Training

- **Self-supervised loss**: Directly optimizes the original objective $\mathcal{L}(y,x) = \varphi(y,x)$.
- Training can be interpreted as projected gradient descent in the raw output space.
- **Key design decision**: The constraint layer is activated during training (not only at inference), because:
    - Some problems diverge without constraints
    - The projection of an unconstrained optimum is generally not the constrained optimum
    - Constraints serve as an inductive bias that improves learning

## Key Experimental Results

### Main Results

Comparison on convex and non-convex benchmark problems (DC3 benchmark):

| Method | Relative Suboptimality (RS) | Constraint Violation (CV) | Single-instance Inference | Batch Inference |
|--------|-----------------------------|--------------------------|--------------------------|-----------------|
| Πnet | ≤5% (most cases) | <10⁻⁵ | 0.0056s | 0.013s |
| DC3 | Worse, especially on large problems | Large on big problems | 0.0019s | 0.002s |
| JAXopt | Comparable to Πnet | Comparable to Πnet | 0.0134s | 0.137s |
| Solver (IPOPT) | Optimal | 0 | 0.034s | 41.7s |

### Training Efficiency

| Method | Epochs | Training Time |
|--------|--------|--------------|
| Πnet | 50 | Seconds |
| DC3 | 1000 | Long |
| JAXopt | 12 | ~14 hours on large problems |

### Ablation Study

| Configuration | RS | CV | Inference Time |
|---------------|----|----|----------------|
| Default (no equilibration, default params) | Moderate | Poor | 0.55s/batch |
| Auto (auto-tuned, no equilibration) | Improved | Improved | 1.89s/batch |
| Πnet (auto-tuned + equilibration) | Best | Best | 0.28s/batch |

### Key Findings

1. **Reliable constraint satisfaction**: Πnet maintains extremely low constraint violation (<10⁻⁵) across all experiments, whereas DC3 exhibits severe violations on large problems.
2. **Rapid training**: Satisfactory performance is achieved within 50 epochs, one to two orders of magnitude faster than DC3 (1000 epochs) and JAXopt.
3. **Hyperparameter robustness**: DC3 is highly sensitive to hyperparameters (diverging on large problems with default settings), while Πnet with automatic tuning requires virtually no manual adjustment.
4. **Multi-vehicle motion planning**: Successfully scales to scenarios with up to 15 vehicles and 750 timesteps (~9000 variables and constraints), demonstrating practical scalability.
5. **Second-order cone constraints**: Successfully extended to second-order cone constraints, with both RS and CV below 10⁻⁶.

## Highlights & Insights

1. **Methodological clarity**: The core idea is concise—projection combined with the implicit function theorem—yet careful engineering (equilibration, automatic tuning) yields outstanding practical performance.
2. **Constraints as inductive bias**: Activating constraints during training is an advantage rather than an obstacle; constraints help the network better learn the distribution of feasible solutions.
3. **Modular design**: The projection layer can be directly appended to any backbone network without architectural modifications.
4. **JAX + GPU**: An efficient, GPU-ready open-source implementation is provided.
5. **Broad generality**: Supports combinations of multiple constraint types (polyhedra + cones + sparsity) through a unified decomposition framework.

## Limitations & Future Work

1. **Convex constraints only**: The current framework requires $\mathcal{C}(x)$ to be convex; non-convex constraints require additional treatment (e.g., sequential convexification).
2. **Decomposition selection**: Different choices of $\mathcal{A}$ and $\mathcal{K}$ affect efficiency, and no fully automatic strategy for optimal decomposition is yet available.
3. **Collision avoidance**: In the multi-vehicle motion planning application, constraints are decoupled across vehicles; coupled non-convex constraints such as collision avoidance are not addressed.
4. **Large-scale problems**: Although a 9000-variable case is demonstrated, scalability to substantially larger problems remains to be validated.
5. **Integration with reinforcement learning**: Only a proof-of-concept for human preference optimization is presented; deeper RL integration warrants further exploration.

## Related Work & Insights

- **DC3 (Donti et al., 2021)**: The primary baseline, using equality completion and inequality correction; essentially a soft-constraint approach.
- **RAYEN (Tordesillas, 2023)**: Restores feasibility via line-segment scaling but requires expensive offline preprocessing.
- **cvxpylayers/JAXopt**: General-purpose differentiable convex optimization layers, lacking structure-aware optimization for projection problems.
- **LinSATNet/GLinSAT**: Limited to specific constraint types (non-negative linear / bounded constraints).

### Insights for Research

1. Exploiting problem structure (projection vs. general optimization) can yield order-of-magnitude efficiency gains.
2. Hard constraints can serve as beneficial inductive biases for neural networks, rather than merely conditions to be satisfied.
3. Engineering details (matrix equilibration, automatic tuning) are critical to practical performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of Douglas-Rachford and the implicit function theorem is not entirely new, but its systematic application and engineering optimization in the context of hard-constrained neural networks is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage from benchmarks to real applications (motion planning), including ablation studies and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, detailed appendix; oral presentation quality.
- Value: ⭐⭐⭐⭐⭐ — Provides a GPU-ready open-source toolkit with broad impact across PDE solving, robotics, and scheduling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](../../CVPR2026/optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](../../ICML2026/optimization/balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[AAAI 2026\] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](../../AAAI2026/optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)

</div>

<!-- RELATED:END -->
