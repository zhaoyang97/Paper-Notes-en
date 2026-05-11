---
title: >-
  [Paper Note] Learning (Approximately) Equivariant Networks via Constrained Optimization
description: >-
  [NeurIPS 2025][equivariance] This paper proposes ACE (Adaptive Constrained Equivariance), a framework that formulates equivariant neural network training as a constrained optimization problem. Via dual methods…
tags:
  - "NeurIPS 2025"
  - "equivariance"
  - "constrained optimization"
  - "homotopy methods"
  - "approximate symmetry"
  - "dual methods"
date: 2026-05-08
content_hash: 2cb40df4e4473032
---

# Learning (Approximately) Equivariant Networks via Constrained Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.13631](https://arxiv.org/abs/2505.13631)
**Code**: None
**Area**: Machine Learning Theory / Equivariant Neural Networks
**Keywords**: equivariance, constrained optimization, homotopy methods, approximate symmetry, dual methods

## TL;DR

This paper proposes ACE (Adaptive Constrained Equivariance), a framework that formulates equivariant neural network training as a constrained optimization problem. Via dual methods, ACE automatically and progressively transitions from a flexible non-equivariant model to an equivariant one, adapting to both fully and partially symmetric data without manual hyperparameter tuning.

## Background & Motivation

Equivariant neural networks encode symmetries architecturally to improve generalization and sample efficiency, yet their training faces three core challenges:

1. **Complex loss landscapes**: Equivariance constraints complicate the loss landscape even when data are fully symmetric, slowing optimization.
2. **Partially symmetric real-world data**: Effects such as noise, measurement bias, and dynamical phase transitions break perfect symmetry, causing strictly equivariant models to underfit.
3. **Manual tuning burden**: Existing relaxation methods (e.g., penalty weights in REMUL, annealing schedules in PennPaper) require extensive domain-specific tuning.

Limitations of prior work:

- **REMUL**: Adds an equivariance penalty to the loss and adaptively adjusts weights $\alpha, \beta$, but provides no guarantee on the degree of equivariance in the final solution.
- **PennPaper**: Manually decreases the perturbation parameter $\gamma$ to zero, but is sensitive to the chosen schedule and introduces additional hyperparameters via a Lie-derivative penalty.

## Method

### Overall Architecture

A homotopy architecture $f_{\theta,\gamma} = f_{\theta,\gamma}^L \circ \cdots \circ f_{\theta,\gamma}^1$ is constructed, where each layer is defined as $f_{\theta,\gamma}^i = f_\theta^{\text{eq},i} + \gamma_i f_\theta^{\text{neq},i}$. When $\gamma = 0$ the model is equivariant; when $|\gamma_i| > 0$ deviations are permitted.

Training is formulated as a constrained optimization problem, and a dual method (gradient descent–ascent) is employed to automatically adjust $\gamma$.

### Key Designs

1. **Equality-constraint scheme (fully symmetric data, Algorithm 1)**:
    - Function: Constrains $\gamma_i = 0$ via dual variables $\lambda_i$ that automatically govern the transition rate.
    - Mechanism: Initialize $\gamma^{(0)} = 1$ (non-equivariant); dual variables $\lambda_i$ grow continuously whenever $\gamma_i > 0$, progressively pushing the model toward equivariance. Key updates: $\gamma_i^{(t+1)} = \gamma_i^{(t)} - \eta_p(\nabla_{\gamma_i} J_0^{(t)} + \lambda_i^{(t)})$, $\lambda_i^{(t+1)} = \lambda_i^{(t)} + \eta_d \gamma_i^{(t)}$.
    - Design Motivation: The dual method is equivalent to adaptive annealing—tightening the equivariance constraint at a rate governed by its actual effect on downstream performance.

2. **Elastic inequality-constraint scheme (partially symmetric data, Algorithm 2)**:
    - Function: Replaces equality constraints with $|\gamma_i| \leq u_i$, where slack variables $u_i$ are also optimized.
    - Mechanism: A term $\frac{\rho}{2}\|u\|^2$ is added to the objective to penalize large slacks. The optimum satisfies $u_i^* = \lambda^*/\rho$, so layers with tighter constraints correspond to larger $\lambda_i$. Projected update: $\lambda_i^{(t+1)} = [\lambda_i^{(t)} + \eta_d(|\gamma_i^{(t)}| - u_i^{(t)})]_+$.
    - Design Motivation: When data are partially symmetric, $\gamma_i$ does not vanish in certain layers; the magnitude of the dual variable automatically identifies which layers require relaxed equivariance.

3. **Theoretical guarantees (Theorem 4.1 & 4.2)**:
    - Function: Provide explicit bounds on the approximation error and degree of equivariance violation after removing $\gamma$.
    - Mechanism: Thm 4.1 — $\|f_{\theta,\gamma}(x) - f_{\theta,0}(x)\| \leq [\sum_{k=0}^{L-1}(1+\bar{\gamma})^k] \bar{\gamma} B M^{L-1} \|x\|$; Thm 4.2 — $\|\rho_Y(g)f_{\theta,\gamma}(x) - f_{\theta,\gamma}(\rho_X(g)x)\| \leq 2\bar{\gamma}(M + C\bar{\gamma})^{L-1}LB^2\|x\|$.
    - Design Motivation: These bounds guarantee that when $\gamma_i$ is sufficiently small, the error incurred by truncating to an equivariant model remains controlled.

### Loss & Training

Lagrangian for the equality-constraint formulation: $\hat{L}(\theta, \gamma, \lambda) = \frac{1}{N}\sum_{n=1}^N \ell_0(f_{\theta,\gamma}(x_n), y_n) + \sum_{i=1}^L \lambda_i \gamma_i$

Crucially, no equivariance penalty is used ($\beta = 0$); the method relies entirely on the constraint and dual updates. Only two learning rates $\eta_p, \eta_d$ and the elastic constant $\rho = 1$ are required.

## Key Experimental Results

### Main Results (Tables)

CMU MoCap motion prediction MSE ($\times 10^{-2}$):

| Model | Run | Walk |
|-------|-----|------|
| EGNN | 50.9±0.9 | 28.7±1.6 |
| EGNO (original) | 33.9±1.7 | 8.1±1.6 |
| EGNO + ACE (equality) | improved | improved |
| EGNO + ACE (elastic inequality) | **best** | **best** |

N-Body physical simulation: SEGNN + ACE outperforms standard SEGNN in both validation MSE and sample efficiency.

### Ablation Study

- The equality-constraint scheme (Alg. 1) improves convergence trajectories on fully symmetric data: flexible early exploration followed by progressive tightening toward equivariance.
- The elastic inequality scheme (Alg. 2) maintains partial equivariance while improving performance on noisy or symmetry-broken data.
- Equivariance error is verified to approach zero progressively during training (Figure 4).
- The theoretical bound (Thm 4.2) is consistent with observed trends in equivariance violation.

### Key Findings

- ACE consistently improves performance across multiple architectures (SEGNN, EGNN, EGNO, p4m-CNN) and tasks (N-Body, motion prediction, image classification).
- On fully symmetric data, the advantage of ACE stems from smoothing the optimization landscape.
- On partially symmetric data, ACE automatically identifies which layers require relaxed equivariance (those with larger $\lambda_i$).
- Sample efficiency gains are substantial: ACE achieves lower error for the same number of samples.
- Robustness to input perturbations is also improved.

## Highlights & Insights

- **No manual tuning required**: The equivariance transition is fully automatic, requiring no schedules, penalty weights, or domain knowledge.
- **Theory–practice consistency**: Theoretical bounds accurately predict observed experimental behavior.
- **High generality**: Applicable to any differentiable architecture $f_{\theta,\gamma}$ where $f_{\theta,0}$ is equivariant.
- The deep connection between dual methods and homotopy/simulated annealing provides an optimization-theoretic perspective on the approach.

## Limitations & Future Work

- $\gamma_i$ does not reach exactly zero in finite iterations, necessitating a final truncation step that incurs the error characterized by Thm 4.1.
- The non-equivariant branch $f_\theta^{\text{neq},i}$ increases parameter count and computational cost.
- Scalability to large models (e.g., large GNNs) has not been verified.
- Convergence guarantees for the dual method in non-convex settings rely on a "sufficiently expressive" parameterization assumption.
- Only discrete groups (e.g., p4m) and continuous groups (e.g., SE(3)) have been tested; applicability to larger groups remains to be explored.

## Related Work & Insights

- **REMUL (EquivMTL)**: A penalty-based approach requiring tuning of $\alpha, \beta$ with no equivariance guarantee; ACE supersedes it via dual methods.
- **PennPaper**: Relies on manual schedules and a Lie-derivative penalty; ACE automates this process via the dual formulation.
- **Residual Pathway Priors**: Adds an invariant branch to increase flexibility, but lacks a progressive tightening mechanism.
- The constrained optimization perspective of ACE generalizes naturally to other structured neural network constraints (e.g., sparsity, low-rank).

## Rating

⭐⭐⭐⭐ — Contributes both theoretically and methodologically, providing a general, tuning-free solution for training equivariant networks with comprehensive experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](on_universality_classes_of_equivariant_networks.md)
- [\[NeurIPS 2025\] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)

</div>

<!-- RELATED:END -->
