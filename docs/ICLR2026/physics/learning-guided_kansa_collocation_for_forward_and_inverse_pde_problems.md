---
title: >-
  [Paper Note] Learning-guided Kansa Collocation for Forward and Inverse PDE Problems
description: >-
  [ICLR 2026][Physics & Scientific Computing][Kansa method] This work extends the meshfree radial basis function (RBF)-based Kansa collocation method from single-variable linear PDEs to coupled multi-variable and nonlinear…
tags:
  - "ICLR 2026"
  - "Physics & Scientific Computing"
  - "Kansa method"
  - "radial basis functions"
  - "nonlinear PDEs"
  - "inverse problems"
  - "neural PDE solvers"
date: 2026-05-08
content_hash: fdfb78d03a589b7b
---

# Learning-guided Kansa Collocation for Forward and Inverse PDE Problems

**Conference**: ICLR 2026
**arXiv**: [2602.07970](https://arxiv.org/abs/2602.07970)  
**Code**: None  
**Area**: Scientific Computing / PDE Solving
**Keywords**: Kansa method, radial basis functions, nonlinear PDEs, inverse problems, neural PDE solvers

## TL;DR

This work extends the meshfree radial basis function (RBF)-based Kansa collocation method from single-variable linear PDEs to coupled multi-variable and nonlinear PDE settings. It incorporates automatic shape-parameter tuning and multiple time-stepping schemes, and provides a systematic comparison against neural PDE solvers such as PINNs and FNO on both forward and inverse problems.

## Background & Motivation

**Challenges in PDE solving**: Partial differential equations are widely used in physics, graphics, and biology, yet classical numerical methods (FDM/FEM) suffer from the curse of dimensionality, high computational cost, and domain-specific discretization requirements.

**Rise of neural PDE solvers**: PINNs (Raissi et al. 2019) and FNO (Li et al. 2020) demonstrate strong generalization and high-dimensional handling capabilities, but are limited by high training costs and large data requirements.

**Advantages of the Kansa method**: The Kansa method is a meshfree RBF-based solver that requires no grid discretization and is naturally suited to complex geometries. Zhong et al. (2023) introduced the Constrained Neural Fields (CNF) framework with automatic shape-parameter optimization.

**Limitations of existing Kansa methods**: Zhong et al. (2023) address only single-variable linear PDEs and cannot handle the coupled systems and nonlinear operators common in practice.

**Lack of systematic comparison**: It remains unclear how an extended Kansa method compares to classical and neural PDE solvers across multiple quality metrics (L1/L2 error, efficiency, convergence rate, etc.).

**Importance of inverse problems**: Inferring unknown PDE parameters (e.g., diffusion coefficients, flow velocities) from observational data is critical for scientific simulation, yet existing Kansa frameworks have not addressed the inverse setting.

## Method

### Overall Architecture

The framework is built on the Kansa collocation method: the field is approximated as a linear combination of RBFs, $\hat{u}(\mathbf{x}) = \sum_k \alpha_k \psi_k(\|\mathbf{x} - \mathbf{x}_k\|)$, and the coefficients $\alpha_k$ are determined by enforcing PDE constraints and boundary conditions. The core extensions are: (1) coupled multi-variable extension, (2) nonlinear operator handling, (3) automatic hyperparameter tuning, and (4) inverse problem solving.

### Key Designs

1. **Coupled Multi-variable PDE Extension (Extension 1)**

    - **Function**: Extends a single unknown field $u$ to a multi-dimensional field $\mathbf{u} = [u_1, u_2, \ldots, u_{N_D}]$.
    - **Mechanism**: Each dimension $u_d$ has an independent RBF expansion and coefficients $\alpha_k^{(d)}$; a linear coupling operator $\mathcal{G}$ links the dimensions, yielding a horizontally stacked block-matrix system.
    - **Design Motivation**: Many physical equations (e.g., Navier–Stokes, Maxwell) are inherently coupled PDE systems requiring the simultaneous solution of multiple physical quantities.

2. **Nonlinear Operator Handling (Extension 2)**

    - **Function**: Handles nonlinear differential operators such as the $u \frac{\partial u}{\partial x}$ term in Burgers' equation.
    - **Mechanism**: Introduces the differentiation matrix $\mathbf{D}_x = \mathbf{K}_x \cdot \mathbf{K}^{-1}$ to decompose nonlinear operators into combinations of differential operations on known fields. Five solution strategies are provided: forward Euler (explicit), IMEX (semi-implicit), backward Euler (implicit Newton–Raphson), Crank–Nicolson (second-order), and full nonlinear direct optimization.
    - **Design Motivation**: Nonlinearity prevents direct separation into a linear system; temporal discretization or iterative optimization is required to circumvent this.

3. **Automatic Shape-Parameter Tuning**

    - **Function**: Automatically optimizes the RBF shape parameter $\epsilon$.
    - **Mechanism**: For linear PDEs, jointly minimizes the condition number of the operator matrix and the variation of the solution field. For nonlinear PDEs, directly minimizes a composite objective of PDE residual, solution variation, and training L2 loss.
    - **Design Motivation**: The choice of $\epsilon$ critically affects accuracy and stability (accuracy–conditioning trade-off), making manual tuning impractical.

4. **Inverse Problem Solving**

    - **Function**: Recovers unknown PDE parameters $\boldsymbol{\pi}$ from solution observations $u^{\text{obs}}$.
    - **Mechanism**: $\boldsymbol{\pi}^* = \arg\min_{\boldsymbol{\pi}} \mathcal{L}(u^{\text{obs}}, u^{\text{pred}}(\boldsymbol{\pi}))$, solved using SciPy's least-squares and root-finding algorithms.
    - **Design Motivation**: Inverse problems are critical in scientific computing (e.g., parameter estimation, material property inference); this extension broadens the Kansa framework to the inverse setting.

5. **Systematic Comparison with Other Solvers**

    - **Function**: Benchmarks Kansa, PINN, and FNO on standard PDE problems.
    - **Mechanism**: Unified evaluation metrics (L2 error, efficiency, memory, convergence rate) with fairly matched training data volumes.
    - **Design Motivation**: Provides a solver-selection guide for different PDE types.

### Loss & Training

- **Linear PDEs**: Least-squares closed-form solution $\mathbf{a}^{\text{opt}} = (\mathbf{F}^T\mathbf{F})^{-1}\mathbf{F}^T\mathbf{h}$.
- **Nonlinear PDEs**: Residual minimization $\min_\alpha \sum_i (\mathcal{F}[\hat{u}](\mathbf{x}_i) - h(\mathbf{x}_i))^2$.
- **Shape-parameter tuning**: Grid search over $\epsilon$.
- **PINN baseline**: Adam optimizer, learning rate $10^{-3}$, 3000 epochs.
- **FNO baseline**: Requires 100 PDE instances for training, 100 epochs.

## Key Experimental Results

### Main Results

Relative L2 error comparison on the 1D advection equation (forward problem):

| Method | Type | Relative L2 Error | Notes |
|--------|------|-------------------|-------|
| Kansa (linear solve) | Meshfree RBF | Low | No training; direct solve |
| Kansa (IMEX) | Meshfree RBF | Low, stable | Semi-implicit; suited for stiff problems |
| Kansa (Crank–Nicolson) | Meshfree RBF | Lowest | Second-order accuracy |
| PINN | Neural network | Moderate | Requires many training iterations |
| FNO | Operator learning | Moderate | Requires multi-instance training data |

Comparison on Burgers' equation (nonlinear):

| Method | Scheme | Accuracy | Stability |
|--------|--------|----------|-----------|
| Forward Euler Kansa | Explicit | $O(\Delta t)$ | Unstable |
| IMEX Kansa | Semi-implicit | $O(\Delta t)$ | Stable |
| Backward Euler Kansa | Implicit + Newton | $O(\Delta t)$ | Stable |
| Crank–Nicolson Kansa | Implicit | $O(\Delta t^2)$ | Stable |
| Full nonlinear Kansa | Global optimization | $O(1)$ | N/A |

### Ablation Study

| Dimension | Variable | Observation |
|-----------|----------|-------------|
| Number of collocation points $N$ | 50 → 500 | Accuracy improves but condition number deteriorates |
| Shape parameter $\epsilon$ | Manual vs. auto-tuned | Auto-tuning significantly reduces error |
| Time-stepping scheme | 5 schemes | Crank–Nicolson achieves highest accuracy; IMEX offers best overall balance |
| Coupled dimensions $N_D$ | 1 → multi | Computational cost grows linearly; accuracy is largely preserved |

### Key Findings

- The Kansa method substantially outperforms PINNs in accuracy at low collocation-point counts, with a clear advantage in settings that do not require large training datasets.
- Automatic shape-parameter tuning effectively resolves the accuracy–conditioning trade-off inherent to RBF methods.
- Among the nonlinear extensions, the IMEX scheme provides the best balance of accuracy, stability, and efficiency.
- The differentiability of the Kansa method makes parameter inference in inverse problems natural and efficient.
- FNO, while offering strong generalization, requires approximately 100 times more training data than Kansa or PINNs.

## Highlights & Insights

1. **Systematic extension**: The progression from single-variable linear to coupled nonlinear PDEs is logically structured, and the five nonlinear solution strategies cover a broad range of practical requirements.
2. **Effective use of the differentiation matrix**: $\mathbf{D}_x = \mathbf{K}_x \cdot \mathbf{K}^{-1}$ combines the flexibility of RBFs with the precision of differential operators, serving as the key enabler for extending Kansa to the nonlinear regime.
3. **Practical orientation**: The systematic solver comparison provides actionable guidance for real-world scientific computing applications.
4. **Meshfree advantages**: The Kansa method requires no mesh generation and is inherently well-suited to complex geometries and high-dimensional problems.
5. **Natural integration of inverse problems**: The RBF representation reduces parametric inversion to a standard optimization problem.

## Limitations & Future Work

1. **Ill-conditioning**: The RBF matrix condition number deteriorates rapidly at high collocation-point densities, limiting scalability.
2. **High-dimensional extension**: Experiments are restricted to 1D and simple 2D cases; validation in 3D and higher dimensions is absent.
3. **Convergence guarantees for nonlinear solvers**: Convergence of the Newton–Raphson solver depends on initialization and lacks theoretical guarantees.
4. **Incomplete comparison with modern methods**: Comparisons with more recent approaches such as DeepONet and Operator Transformers are not conducted.
5. **Limited inverse problem experiments**: Inverse problem evaluation covers only simple parameter inference; more complex scenarios (e.g., unknown source terms, unknown boundary conditions) are not tested.
6. **Insufficient theoretical error analysis**: Error bounds and convergence-order analysis for the nonlinear extensions are not adequately developed.

## Related Work & Insights

- **Zhong et al. (2023)** proposed the CNF framework that directly motivates this work; the present paper is a natural extension toward coupled and nonlinear settings.
- **Kansa (1990)** provides the theoretical foundation for meshfree RBF methods.
- **Raissi et al. (2019)** (PINN) and **Li et al. (2020)** (FNO) represent two major paradigms in neural PDE solving and serve as the primary baselines.
- **Insights**: Integrating the Kansa method with differentiable rendering pipelines for inverse physics problems is a promising direction; hybrid approaches combining the Kansa method with neural operators (e.g., using neural networks to learn optimal collocation point placement) also merit exploration.

## Rating

- **Novelty**: ⭐⭐⭐ — The extension directions are natural but largely incremental; the core ideas (differentiation matrices, temporal discretization) are combinations of classical numerical techniques.
- **Experimental Thoroughness**: ⭐⭐⭐ — Forward and inverse problems are covered across multiple PDE types, but experiments are small-scale (primarily 1D) and lack large-scale validation.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is developed in a clear, layered manner with detailed matrix derivations, though the notation is dense.
- **Value**: ⭐⭐⭐ — Offers practical contributions to the Kansa methods community; the systematic comparison has reference value, but the overall scope of impact is relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](../../AAAI2026/physics/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](../../NeurIPS2025/physics/physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](../../NeurIPS2025/physics/deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)

</div>

<!-- RELATED:END -->
