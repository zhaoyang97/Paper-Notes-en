---
title: >-
  [Paper Note] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs
description: >-
  [NeurIPS 2025][Scientific Computing][PINNs] By combining perturbation theory with PINNs, this work decomposes nonlinear PDEs into a sequence of linear subproblems. After learning the latent space of the linear operator v…
tags:
  - "NeurIPS 2025"
  - "Scientific Computing"
  - "PINNs"
  - "perturbation theory"
  - "transfer learning"
  - "partial differential equations"
  - "closed-form solution"
date: 2026-05-08
content_hash: 79cc281c661c7bf6
---

# One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs

**Conference**: NeurIPS 2025
**arXiv**: [2511.11137](https://arxiv.org/abs/2511.11137)  
**Code**: None  
**Area**: Scientific Computing
**Keywords**: PINNs, perturbation theory, transfer learning, partial differential equations, closed-form solution

## TL;DR
By combining perturbation theory with PINNs, this work decomposes nonlinear PDEs into a sequence of linear subproblems. After learning the latent space of the linear operator via a Multi-Head PINN, transfer to new PDE instances is achieved through a closed-form solution within 0.2 seconds, attaining errors on the order of $10^{-3}$.

## Background & Motivation

**Background**: PINNs solve PDEs by embedding physical laws into neural networks, but each new instance typically requires retraining from scratch.

**Limitations of Prior Work**: Although strategies such as Multi-Head PINNs enable shared computation across multiple instances, their generalization to nonlinear PDEs remains limited, and cross-instance transfer still requires iterative optimization.

**Goal**: Extend one-shot transfer learning for nonlinear PDEs from ODEs to PDEs.

**Core Idea**: Treat the nonlinear term $\epsilon P(u)$ as a perturbation, expand in powers of $\epsilon$ to obtain a sequence of linear subproblems, and exploit a shared latent-space representation to enable closed-form transfer.

## Method

### Overall Architecture
For a nonlinear PDE $\mathcal{D}u + \epsilon P(u) = f(x,t)$, where $\mathcal{D}$ is a linear operator, $P(u)$ is a polynomial perturbation, and $\epsilon < 1$, the solution is expanded as $u \approx \sum_{i=0}^p \epsilon^i u_i$, yielding $p+1$ linear subproblems $\mathcal{D}u_j = f_j$, where each $f_j$ depends only on $u_0, \ldots, u_{j-1}$. A Multi-Head PINN solves these linear subproblems and learns a latent-space representation of the operator $\mathcal{D}$; the network backbone is then frozen, and closed-form weights for new instances are computed directly.

### Key Designs

1. **Perturbative Expansion**:

    - **Function**: Systematically reduces a nonlinear PDE to a sequence of linear subproblems.
    - **Mechanism**: The solution is expanded in powers of the perturbation parameter $\epsilon$ as $u = \sum_{i=0}^p \epsilon^i u_i$. Substituting into the original equation and collecting terms of equal order yields $p+1$ linear PDEs of the form $\mathcal{D}u_j = f_j$. Initial and boundary conditions are imposed entirely on the zeroth-order problem $\mathcal{D}u_0 = f$; higher-order problems use homogeneous conditions.
    - **Design Motivation**: All linear subproblems share the same linear operator $\mathcal{D}$, enabling reuse of its latent-space representation.
    - **Theoretical Basis**: When $\epsilon < 1$ and the polynomial coefficients satisfy $P_l \leq 1$, higher-order terms decay progressively and the series converges.

2. **Multi-Head PINN + Closed-Form Transfer**:

    - **Function**: Learns a reusable latent space and enables zero-iteration transfer.
    - **Mechanism**: The network outputs $u_k = H(x,t)W_k$, where $H$ is the shared hidden-layer activation and $W_k$ are head-specific weights. For a new instance $(f^*, g^*, B^*)$, $H$ is frozen and the loss with respect to $W^*$ constitutes a quadratic optimization problem. Since every term is convex in $W^*$, setting $\partial\mathcal{L}/\partial W^* = 0$ yields the closed-form solution $W^* = M^{-1}(\cdot)$.
    - **Key Advantage**: The matrix $M$ depends only on $\mathcal{D}$ and the sampling strategy, not on the specific $(f^*, g^*, B^*)$; therefore $M^{-1}$ can be precomputed and reused across all new instances.
    - **Design Motivation**: Eliminates the need to retrain for each new instance, enabling sub-second adaptation.

3. **Iterative Solve and Combination**:

    - **Function**: Constructs the final nonlinear solution.
    - **Mechanism**: $u_0$ is first obtained via MH-PINN or closed-form transfer, then substituted into the expression for $f_1$ to solve for $u_1$, and so forth. The final solution is $u = \sum_{i=0}^p \epsilon^i u_i$.
    - **Design Motivation**: Each step requires solving only one linear problem, keeping the computational cost tractable.

### Loss & Training
The physics-informed losses for individual heads are summed with weighting: $\mathcal{L}_k = w_{pde}(\mathcal{D}u_k - f_k)^2 + w_{IC}(u_k(x,0) - g_k(x))^2 + w_{BC}\sum_\mu (u_k(\mu,t) - B_{\mu,k}(t))^2$, and the total loss is $\mathcal{L} = \frac{1}{K}\sum_{k=0}^K \mathcal{L}_k$.

## Key Experimental Results

### Main Results

| PDE Type | Parameters | Relative Error | Transfer Time | Notes |
|----------|-----------|---------------|--------------|-------|
| KPP-Fisher | $n_1=n_2=1$ | $1.1 \times 10^{-3}$ | 0.149 s | $p=10$, canonical parameters |
| KPP-Fisher | varying $n_1,n_2$ | $1.2 \times 10^{-3}$ | ~0.15 s | Different dynamics |
| KPP-Fisher | larger $\epsilon$ | $1.9 \times 10^{-2}$ | ~0.15 s | Near the boundary of validity |
| Wave equation | standard | comparable accuracy | ~0.15 s | Validates cross-operator capability |

### Ablation Study

| Configuration | Key Finding | Notes |
|--------------|------------|-------|
| $p$ vs. error | Error decreases then plateaus as $p$ increases | Fully converged at $p=10$ |
| $\epsilon$ vs. error | A clear threshold exists | Solution diverges beyond the threshold |
| Classical solver comparison | Comparable accuracy | SciPy solve_ivp: 0.162 s |

### Key Findings
- When $\epsilon$ exceeds a threshold, the error grows sharply; the threshold correlates with solution amplitude—larger amplitudes require smaller $\epsilon$.
- High-degree polynomials (>10) require smaller $\epsilon$ values because perturbation terms grow faster.
- Transfer speed (0.149 s) is comparable to classical numerical solvers (0.162 s), with the advantage lying in reusing precomputed quantities across instances.
- Different values of $n_1$ and $n_2$ affect the propagation speed toward the equilibrium state: increasing $n_2$ accelerates propagation toward the equilibrium value of 1, while increasing $n_1$ decelerates it.

## Highlights & Insights
- Closed-form transfer is the central contribution: once the latent space of the linear operator has been learned, adapting to a new instance requires only a matrix inversion with no gradient updates. This represents an important step toward advancing transfer learning from "few-shot fine-tuning" to "zero-shot computation."
- The method's scope of applicability is well-defined: $\epsilon$ must be sufficiently small, and failure cases are easy to identify (solution divergence), a self-diagnostic property of considerable value in scientific computing.
- The approach extends to transfer between related operators: the latent space remains reusable when coefficients $a_\alpha(t)$ vary slightly.

## Limitations & Future Work
- Restricted to polynomial perturbation terms; nonlinear terms involving derivatives are not yet supported.
- Validated only on 2D PDEs (1D space + time); extension to higher dimensions is a future direction.
- The effective range of $\epsilon$ is problem-dependent.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of perturbation theory and PINN-based transfer is original.
- Experimental Thoroughness: ⭐⭐⭐ Validation scenarios are limited.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear.
- Value: ⭐⭐⭐ Offers specific value within the scientific computing domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICLR 2026\] Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](../../ICLR2026/scientific_computing/supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/scientific_computing/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)

</div>

<!-- RELATED:END -->
