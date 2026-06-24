---
title: >-
  [Paper Note] Hamiltonian Neural PDE Solvers through Functional Approximation
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][Hamiltonian mechanics] Grounded in the Riesz representation theorem, this work approximates infinite-dimensional Hamiltonian functionals via learnable integral kernel functionals (IKF). Functional derivatives are obtained through automatic differentiation, yielding an energy-conserving neural PDE solver (HNS) that demonstrates superior stability and generalization on 1D/2D PDEs.
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "Hamiltonian mechanics"
  - "PDE solving"
  - "functional approximation"
  - "neural fields"
  - "energy conservation"
date: 2026-05-08
content_hash: 6a1211a47a038745
---

# Hamiltonian Neural PDE Solvers through Functional Approximation

**Conference**: NeurIPS 2025
**arXiv**: [2505.13275](https://arxiv.org/abs/2505.13275)  
**Code**: [GitHub](https://github.com/anthonyzhou-1/hamiltonian_pdes)  
**Area**: Scientific Computing / Neural PDE Solvers
**Keywords**: Hamiltonian mechanics, PDE solving, functional approximation, neural fields, energy conservation

## TL;DR

Grounded in the Riesz representation theorem, this work approximates infinite-dimensional Hamiltonian functionals via learnable integral kernel functionals (IKF). Functional derivatives are obtained through automatic differentiation, yielding an energy-conserving neural PDE solver (HNS) that demonstrates superior stability and generalization on 1D/2D PDEs.

## Background & Motivation

**Background**: Neural PDE solvers (FNO, U-Net, etc.) have achieved notable progress in parametric PDE solving, yet the vast majority operate within a Newtonian framework—directly predicting the next-step state or time derivative—without exploiting the conservation structure inherent in physical systems.

**Limitations of Prior Work**: Hamiltonian Neural Networks (HNNs) have demonstrated the ability to enforce conservation laws in discrete particle systems, but HNNs are limited to finite-dimensional settings (e.g., $N$-body problems), where the Hamiltonian is a function $\mathcal{H}(\mathbf{q}, \mathbf{p}): \mathbb{R}^{2n} \to \mathbb{R}$. Most practical PDEs, however, describe continuous fields (fluids, waves, elasticity) and require **infinite-dimensional** Hamiltonian mechanics, in which the Hamiltonian is a functional $\mathcal{H}[u]: \mathcal{F}(\Omega) \to \mathbb{R}$ and evolution is governed by the functional derivative $\delta\mathcal{H}/\delta u$.

**Key Challenge**: Extending the Hamiltonian framework to PDEs presents two fundamental challenges: (a) one must approximate a mapping from a function space to a scalar (a functional), for which traditional neural networks are not designed; and (b) the approximated functional must admit accurate functional derivatives to drive the temporal evolution via Hamilton's equations.

**Goal**: Design a neural network architecture capable of learning Hamiltonian functionals and accurately computing their functional derivatives, thereby constructing a neural PDE solver that respects the Hamiltonian structure.

**Key Insight**: The **Riesz representation theorem** from functional analysis—any continuous linear functional can be expressed as an inner product $\mathcal{H}[u] = \langle u, \kappa_\theta \rangle$—reduces the problem of functional approximation to one of function approximation, the latter being a natural strength of neural networks.

**Core Idea**: Approximate the Hamiltonian functional via an integral kernel parameterized by a neural field; obtain functional derivatives through automatic differentiation; and construct a conservation-respecting PDE solver.

## Method

### Overall Architecture

The HNS (Hamiltonian Neural Solver) proceeds as follows:

1. **Forward pass**: Given the current field state $\mathbf{u}^t$ and coordinates $\mathbf{x}$, compute the Hamiltonian scalar $\mathcal{H}_\theta$ via the Integral Kernel Functional (IKF).
2. **Backward pass**: Compute the functional derivative $\delta\mathcal{H}_\theta / \delta u$ via automatic differentiation.
3. **Temporal evolution**: Apply the known linear operator $\mathcal{J}$ (e.g., $\partial_x$) to obtain the time derivative $\partial u / \partial t = \mathcal{J}(\delta\mathcal{H}_\theta / \delta u)$.
4. **Numerical integration**: Advance the state $\mathbf{u}^{t+1}$ using a second-order Adams–Bashforth scheme.

### Key Designs

#### 1. Integral Kernel Functional (IKF)

- **Function**: Represents the functional $\mathcal{H}[u]$ in an integral kernel form.
- **Mechanism**: By the Riesz representation theorem, $\mathcal{H}[u] = \int_\Omega \kappa_\theta(x, u(x)) \cdot u(x) \, dx$, discretized via Riemann summation as $\mathcal{H}_\theta \approx \sum_i \kappa_\theta(x_i, u_i) \cdot u_i \cdot \mu_i \Delta x$.
- **Design Motivation**: While superficially similar to the integral kernel operators in neural operators, the IKF is fundamentally distinct—operator outputs are functions requiring per-query summation (necessitating Fourier truncation for efficiency), whereas the IKF outputs a scalar computed by a single Riemann sum, preserving full accuracy.

#### 2. SIREN + FiLM-Conditioned Kernel Parameterization

- **Function**: Parameterizes the kernel function $\kappa_\theta$ using a neural field architecture.
- **Mechanism**: A sinusoidal representation network (SIREN) serves as the kernel backbone, with each layer conditioned via FiLM (Feature-wise Linear Modulation): $\kappa_\theta^{(l)}(x_i, u_i) = \gamma_\theta^{(l)}(u_i) \sin(Wx_i + b) + \beta_\theta^{(l)}(u_i)$.
- **Design Motivation**: SIREN not only fits functions effectively but also yields accurate gradient representations (the derivative of a sine remains a sine), which facilitates precise functional derivative computation. FiLM enables local or global conditioning, allowing the kernel to depend on the input field.

#### 3. Local vs. Global Conditioning

- **Local conditioning**: FiLM parameters depend solely on $u_i = u(x_i)$; suitable for PDEs whose Hamiltonians contain only local terms (e.g., Advection).
- **Global conditioning**: FiLM parameters depend on the entire field $\mathbf{u}^t$ (via a shallow 1D CNN); required for PDEs with nonlocal terms such as $u_{xx}$ (e.g., KdV, SWE).

#### 4. Implicit Functional Derivative Learning

- **Key Finding**: Training the IKF solely on functional values $\mathcal{H}[u]$ (scalar supervision) implicitly yields accurate functional derivatives $\delta\mathcal{H}/\delta u$ through automatic differentiation.
- **Intuition**: For a linear IKF, $\nabla_\mathbf{u} \sum_i \kappa_\theta(x_i) u_i \mu_i \Delta x = [\kappa_\theta(x_1), \ldots, \kappa_\theta(x_n)]$; the functional derivative is precisely the discretization of the learned kernel.

### Loss & Training

- Training loss: $\mathcal{L} = \|\delta\mathcal{H}_\theta / \delta u - \delta\mathcal{H} / \delta u\|^2$, optimizing directly in the functional derivative domain.
- Supervision targets are obtained from analytic functional derivative expressions of the true Hamiltonian, evaluated via finite differences.
- Training in the functional derivative domain rather than in the temporal domain is preferred, as it directly optimizes the kernel $\kappa_\theta$.

## Key Experimental Results

### Main Results

#### Toy: Linear/Nonlinear Functional Approximation (Table 1)

| Metric | MLP | FNO | IKF |
|--------|-----|-----|-----|
| Linear functional $\mathcal{F}_l[u]$ (Base) | 2.47e-5 | 2.76e-4 | **3.00e-16** |
| Linear functional derivative (Base) | 0.083 | 0.066 | **1.15e-3** |
| Linear functional (OOD) | 0.046 | 0.268 | **2.13e-7** |
| Nonlinear functional $\mathcal{F}_{nl}[u]$ (Base) | 0.029 | 0.016 | **2.05e-3** |
| Nonlinear functional derivative (Base) | 1.33 | 2.10 | **0.045** |

IKF substantially outperforms MLP and FNO on both functional values and functional derivatives, while using the fewest parameters (4.3K vs. MLP 7.5K vs. FNO 10.9K).

#### 1D PDEs: Advection + KdV (Table 2)

| Model | Adv Rollout Err ↓ | KdV Corr Time ↑ |
|-------|-------------------|-----------------|
| FNO | 0.83 ± 0.17 | 68.0 ± 10.5 |
| Unet | 0.40 ± 0.29 | 134.0 ± 10.6 |
| FNO(du/dt) | 0.048 ± 0.007 | 77.6 ± 3.6 |
| Unet(du/dt) | 0.057 ± 0.024 | 113.3 ± 15.0 |
| **HNS** | **0.0039 ± 0.0008** | **151.1 ± 3.0** |

HNS surpasses all baselines with roughly half the parameters (Adv: 32K vs. 65K; KdV: 87K vs. 135K). The Advection rollout error is an order of magnitude lower than the best baseline.

#### 2D Shallow Water Equations (Table 3)

| Model | Sines (in-dist) ↓ | Pulse (OOD) ↓ | Params |
|-------|-------------------|---------------|--------|
| Transolver | 0.084 | 0.122 | 4M |
| FNO | 0.057 | 0.117 | 7M |
| PINO | 0.053 | 0.114 | 7M |
| Unet | **0.010** | 0.042 | 3M |
| **HNS** | 0.026 | **0.021** | 3M |

HNS achieves a substantial lead in OOD generalization on Pulse initial conditions (0.021 vs. Unet 0.042) while maintaining energy conservation.

### Ablation Study

| Ablation | Key Finding |
|----------|-------------|
| Kernel type (linear vs. nonlinear) | Nonlinear kernels are necessary for complex Hamiltonians |
| Conditioning (local vs. global) | Global conditioning is required for PDEs containing $u_{xx}$ terms (e.g., KdV) |
| SIREN vs. MLP kernel | SIREN yields more accurate gradient representations and superior performance |
| Quadrature method | The trapezoidal rule provides sufficient accuracy |

### Key Findings

1. **Energy conservation**: The Hamiltonian along HNS-predicted trajectories remains nearly constant over time, while those of FNO/U-Net exhibit significant drift.
2. **Temporal extrapolation**: Trained on short time windows (Adv: 0–4 s; KdV: 0–25 s), HNS stably extrapolates to much longer horizons (Adv: 0–20 s; KdV: 0–100 s).
3. **OOD generalization**: On 2D SWE, HNS maintains Hamiltonian conservation even under OOD initial conditions, demonstrating a strong inductive bias.
4. **Parameter efficiency**: Kernel weights are shared across all input points $(x_i, u_i)$, resulting in approximately half the parameter count of baselines.

## Highlights & Insights

1. **Elegant theoretical foundation**: Reducing functional approximation to function approximation via the Riesz representation theorem is a natural and profound insight.
2. **Implicit derivative learning**: Accurate functional derivatives are attained from scalar supervision alone—a capability unavailable to MLP/FNO architectures—making the Hamiltonian framework viable for PDE solving.
3. **Unified perspective**: IKF simultaneously connects the literature on functional analysis, neural fields, and neural operators.
4. **Practical parameter efficiency**: The weight-sharing mechanism enables HNS to achieve better performance with fewer parameters.
5. **Instructive negative result**: Adding an auxiliary loss on the Hamiltonian value is detrimental (the model may learn a trivial constant mapping), revealing subtle constraints in enforcing Hamiltonian conservation.

## Limitations & Future Work

1. **Incomplete theory for nonlinear functionals**: The Riesz theorem guarantees approximation only for linear functionals; the nonlinear case lacks rigorous theoretical support.
2. **Inference overhead**: Computing functional derivatives via backpropagation and evaluating the linear operator $\mathcal{J}$ makes inference slower than direct prediction methods.
3. **Scalability**: Reliance on numerical integration (Riemann summation) raises concerns about scaling to high-resolution grids.
4. **Restricted to Hamiltonian systems**: The method cannot directly handle dissipative systems; extensions (e.g., the GENERIC framework) require additional work.
5. **Numerical error in $\mathcal{J}$**: Finite-difference approximations of $\mathcal{J}$ may introduce numerical errors, particularly for higher-order derivative terms.
6. **Future directions**: Integration with symplectic integrators; extension to 3D or more complex geometries; exploration of learning $\mathcal{J}$.

## Related Work & Insights

- **HNN family**: Greydanus et al. (2019) introduced discrete HNNs; the present work is a natural extension to the infinite-dimensional setting.
- **Neural operators**: The integral kernel operator in FNO shares a deep connection with IKF (IKF can be viewed as a functional version of an operator evaluated at a single point).
- **Neural fields** (SIREN, NeRF, etc.): Provide mature tools for kernel parameterization.
- **ML for DFT**: Learning energy functionals in density functional theory is conceptually analogous to IKF.
- **Insights**: The Hamiltonian framework not only provides conservation guarantees but also embodies a paradigm of "learning energy rather than forces," with potentially significant implications for molecular dynamics, atmospheric simulation, and related fields.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First extension of the Hamiltonian framework to infinite-dimensional neural PDE solving; theoretically novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Progressive validation from toy problems to 2D SWE, though 3D experiments and larger-scale validation are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear; the narrative from discrete to continuous is exceptionally coherent.
- Value: ⭐⭐⭐⭐ — Introduces a new paradigm for physics-informed ML, though practical applicability is limited to Hamiltonian systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[NeurIPS 2025\] High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)

</div>

<!-- RELATED:END -->
