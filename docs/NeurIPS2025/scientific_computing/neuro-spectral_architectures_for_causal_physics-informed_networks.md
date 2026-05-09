---
title: >-
  [Paper Note] Neuro-Spectral Architectures for Causal Physics-Informed Networks
description: >-
  [NeurIPS 2025][Scientific Computing][PINN] NeuSA integrates classical spectral methods with Neural ODEs: the PDE is projected onto a spectral basis (Fourier) to obtain an ODE system, which is then solved by a NODE that learns the dynamical evolution. This architecture-level design eliminates the spectral bias and causality violations inherent in conventional PINNs, achieving errors 1–2 orders of magnitude lower than baselines on wave, Burgers, and sine-Gordon equations while training faster.
tags:
  - NeurIPS 2025
  - Scientific Computing
  - PINN
  - Spectral Methods
  - Neural ODE
  - Causality
  - Wave Equation
date: 2026-05-08
content_hash: 9ce327dcd4fb4b7f
---

# Neuro-Spectral Architectures for Causal Physics-Informed Networks

**Conference**: NeurIPS 2025
**arXiv**: [2509.04966](https://arxiv.org/abs/2509.04966)
**Code**: [https://github.com/arthur-bizzi/neusa](https://github.com/arthur-bizzi/neusa)
**Area**: Scientific Computing / PDE Solving
**Keywords**: PINN, Spectral Methods, Neural ODE, Causality, Wave Equation

## TL;DR
NeuSA integrates classical spectral methods with Neural ODEs: the PDE is projected onto a spectral basis (Fourier) to obtain an ODE system, which is then solved by a NODE that learns the dynamical evolution. This architecture-level design eliminates the spectral bias and causality violations inherent in conventional PINNs, achieving errors 1–2 orders of magnitude lower than baselines on wave, Burgers, and sine-Gordon equations while training faster.

## Background & Motivation

**State of the Field**: Physics-informed neural networks (PINNs) solve PDEs by embedding physical laws into the loss function, offering a mesh-free and flexible framework. Mainstream PINNs are based on standard MLPs or enhanced variants (QRes, FLS, PINNsFormer).

**Limitations of Prior Work**: Standard MLP-based PINNs suffer from three fundamental issues: (a) **spectral bias** — sigmoid/ReLU activations favor low-frequency components, making it difficult to represent high-frequency details; (b) **causality violation** — simultaneous optimization over the entire spatiotemporal domain leads to temporal inconsistency and convergence to trivial equilibrium solutions; (c) **poor extrapolation** — predictions rapidly degrade outside the training domain.

**Root Cause**: Conventional PINNs treat spatiotemporal coordinates as network inputs without distinguishing time from space, violating the causal structure of PDEs (initial values → temporal evolution). Global optimization forces initial and boundary conditions to be imposed as soft constraints in the loss, frequently causing conflicting gradients and training instability.

**Paper Goals**: Design a PINN architecture that enforces causality at the architectural level, overcomes spectral bias, and automatically satisfies initial and boundary conditions.

**Starting Point**: In classical numerical methods, spectral methods naturally provide high-frequency expressiveness, while the method of lines discretizes PDEs into ODE systems to preserve causal structure. Combining both with Neural ODEs allows the network to learn the temporal evolution of spectral coefficients rather than directly fitting the solution.

**Core Idea**: Project the PDE onto a Fourier spectral basis to spatially discretize it into an ODE system, then use a Neural ODE to learn the temporal evolution of the spectral coefficients — guaranteeing causality, spectral fidelity, and automatic satisfaction of initial/boundary conditions at the architecture level.

## Method

### Overall Architecture
Input: initial condition $\mathbf{u}_0(\mathbf{x})$ → **Spectral decomposition**: project onto Fourier basis to obtain spectral coefficients $\hat{\mathbf{u}}(0)$ → **Neural ODE time integration**: integrate the learned vector field $\hat{\mathbf{F}}_\theta$ to the target time via 4th-order Runge-Kutta → **Spectral reconstruction**: multiply spectral coefficients by basis functions to obtain the continuous solution $\mathbf{u}_\theta(t, \mathbf{x})$ at arbitrary spatiotemporal points. A single forward pass covers the entire spatiotemporal domain.

### Key Designs

1. **Spectral Decomposition + Analytic Initialization**:

    - Function: Represent the solution $\mathbf{u}(t, \mathbf{x})$ as $\sum_k \hat{\mathbf{u}}_k(t) \mathbf{b}_k(\mathbf{x})$ and initialize the NODE vector field with the analytic solution of the linear homogeneous problem.
    - Mechanism: Choose a Fourier basis (or sine/cosine extensions) → the PDE becomes an ODE for the spectral coefficients: $\frac{d}{dt}\hat{\mathbf{u}} = \hat{\mathbf{F}}(\hat{\mathbf{u}})$. For the linear translation-invariant part, the Fourier multiplier $M$ yields $\hat{\mathbf{F}}_\theta(\hat{\mathbf{u}}) = M \odot \hat{\mathbf{u}} + \epsilon \mathcal{F}_\theta(\hat{\mathbf{u}})$, with $\epsilon$ initialized to a small value.
    - Design Motivation: The Fourier basis overcomes spectral bias by providing explicit representations of high-frequency components. Analytic initialization starts the network from the linear approximate solution, so $\mathcal{F}_\theta$ only needs to learn the residual correction for nonlinear/inhomogeneous terms, substantially reducing learning difficulty.

2. **Neural ODE Causal Time Integration**:

    - Function: Integrate the spectral coefficients in time using 4th-order Runge-Kutta, inherently guaranteeing causal structure.
    - Mechanism: The NODE encodes the time dimension as integration steps rather than input coordinates. Initial conditions are exactly satisfied through the integration starting point (rather than as soft constraints), and subsequent time steps depend on prior states.
    - Design Motivation: Theoretical guarantee — for band-limited initial conditions and globally Lipschitz vector fields, the NeuSA solution automatically satisfies initial conditions and possesses uniqueness (Theorem 1). The $\mathcal{L}_{IC}$ and $\mathcal{L}_{BC}$ loss terms are unnecessary, eliminating gradient conflicts from multi-objective optimization.

3. **Dimension-wise Layers**:

    - Function: Replace fully connected layers for processing high-dimensional spectral coefficients, avoiding parameter explosion.
    - Mechanism: For a 2D spectral coefficient matrix $\hat{\mathbf{u}} \in \mathbb{R}^{m \times n}$, apply Hadamard scaling followed by separate linear transformations along rows and columns ($O(mn)$ parameters vs. $O(m^2n^2)$ for fully connected layers).
    - Design Motivation: $100 \times 100$ frequency modes yield a 10K-dimensional representation, requiring $10^8$ parameters with full connectivity. Dimension-wise layers preserve global connectivity (required for spectral representations) while keeping parameter count tractable.

### Loss & Training
- Only the PDE residual loss is required: $\mathcal{L}_{PDE} = \sum_{t_i, \mathbf{x}_j} \|\frac{d}{dt}\mathbf{u}_\theta - \mathbf{F}(\mathbf{u}_\theta, \nabla\mathbf{u}_\theta, ...)\|^2$
- Initial condition and boundary condition loss terms are unnecessary (automatically satisfied by the architecture).
- Adam optimizer with learning rate 0.01 (10× larger than the baseline's 0.001, since the architectural prior stabilizes optimization).
- Spatial derivatives are computed analytically via basis function differentiation, without autograd (computational cost does not grow exponentially with derivative order).

## Key Experimental Results

### Main Results

| PDE Problem | PINN | QRes | FLS | PINNsFormer | **NeuSA** |
|-------------|------|------|-----|-------------|-----------|
| 2D Wave (layered) rMSE | 0.545 | 0.115 | 0.590 | 1.072 | **0.075** |
| 2D Wave (Marmousi) rMSE | 0.698 | 0.412 | 0.684 | — | **0.171** |
| 3D Wave rMSE | 0.073 | 0.021 | 0.070 | — | **0.008** |
| 2D Burgers rMSE | 0.221 | 0.073 | 0.202 | 1.053 | **0.051** |
| 1D sine-Gordon rMSE | 0.139 | 0.020 | 0.135 | 0.681 | **0.001** |

### Training Time Comparison

| PDE Problem | PINN | QRes | FLS | **NeuSA** | NeuSA steps / baseline steps |
|-------------|------|------|-----|-----------|-------------------------------|
| 2D Wave (layered) | 566s | 750s | 577s | **530s** | 2K/20K (10× fewer) |
| Marmousi | 635s | 718s | 648s | **573s** | 2K/20K |
| 1D sine-Gordon | 976s | 1315s | 1015s | **215s** | 1K/10K |

### Key Findings
- NeuSA achieves the lowest error on all benchmarks; on sine-Gordon it outperforms the second-best method QRes by one order of magnitude (0.001 vs. 0.020).
- Despite being built on a computationally intensive NODE, NeuSA's training time is comparable to or shorter than the baselines — physical priors reduce the required number of steps by 10×, and a single forward pass covers the entire domain.
- For the wave equation, NeuSA is the only method that accurately recovers second-order reflected waves.
- In the temporal extrapolation experiment (Burgers equation), NeuSA maintains accurate predictions in $[1,2]$ beyond the training domain $[0,1]$, while PINN and QRes diverge rapidly.

## Highlights & Insights
- **Addressing the root causes of PINN failure at the architecture level**: Spectral bias and causality violations are not mitigated through loss modifications or training strategies but are eliminated at the source through architectural design — a more fundamental solution than weighted losses or curriculum learning.
- **Analytic initialization is the key to training acceleration**: Starting from the linear homogeneous solution, the neural network learns only the residual correction, reducing the number of convergence steps by 10×. This strategy is transferable to other scientific computing + deep learning settings.
- **Single forward pass covers the entire spatiotemporal domain**: Conventional PINNs require one forward pass per collocation point; NeuSA obtains the complete spatiotemporal solution in one pass via spectral decomposition + ODE integration, yielding an inherent batching advantage.
- **No initial/boundary condition losses required**: This eliminates the need to tune hyperparameters such as $\lambda_{IC}$ and $\lambda_{BC}$, and avoids gradient conflicts in multi-objective optimization.

## Limitations & Future Work
- Currently limited to rectangular domains (a constraint of the Fourier basis); complex geometries require more general basis functions (e.g., spherical harmonics, finite element bases), at the cost of losing the analytic initialization convenience.
- Runge-Kutta integration may be unstable for stiff problems, necessitating implicit methods.
- Performance is strongly dependent on analytic initialization — initialization without prior knowledge leads to significantly degraded results.
- Validation on truly high-dimensional problems (>3D) is absent.

## Related Work & Insights
- **vs. standard PINN**: PINNs directly fit the spatiotemporal solution with an MLP, suffering from spectral bias and causality violations; NeuSA eliminates these issues architecturally, reducing errors by 1–2 orders of magnitude.
- **vs. FLS/Fourier Features**: FLS mitigates spectral bias via sinusoidal encoding layers but does not address causality; NeuSA resolves both problems simultaneously.
- **vs. PINNsFormer**: PINNsFormer models temporal dependencies with Transformer attention, yet is outperformed by NeuSA in both accuracy and training efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of spectral methods and Neural ODEs constitutes a fundamentally new architectural paradigm that resolves all three core PINN problems at their source.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three PDE classes and five benchmarks, though large-scale and complex-geometry validation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear method motivation, and high-quality figures.
- Value: ⭐⭐⭐⭐⭐ Introduces a theoretically grounded new architectural direction for the PINN community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/scientific_computing/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [\[NeurIPS 2025\] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)

<!-- RELATED:END -->
