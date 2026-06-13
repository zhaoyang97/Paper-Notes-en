---
title: >-
  [Paper Note] Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics
description: >-
  [NeurIPS 2025][Neural ODE] This paper proposes Helmholtz metrics — differentiable metrics derived from the Helmholtz conditions — to quantify how closely a given ODE approximates the Euler-Lagrange equations. These metri…
tags:
  - "NeurIPS 2025"
  - "Neural ODE"
  - "Lagrangian mechanics"
  - "Helmholtz conditions"
  - "physics regularization"
  - "Euler-Lagrange equations"
date: 2026-05-08
content_hash: d89f4b38bc22a8ac
---

# Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics

**Conference**: NeurIPS 2025
**arXiv**: [2510.06367](https://arxiv.org/abs/2510.06367)  
**Code**: [GitHub](https://github.com/luwo9/LagrangianNeuralODEs)  
**Area**: Physics-Informed Learning / Neural ODE
**Keywords**: Neural ODE, Lagrangian mechanics, Helmholtz conditions, physics regularization, Euler-Lagrange equations

## TL;DR

This paper proposes Helmholtz metrics — differentiable metrics derived from the Helmholtz conditions — to quantify how closely a given ODE approximates the Euler-Lagrange equations. These metrics are incorporated as regularization terms into second-order Neural ODE training, forming Lagrangian Neural ODEs that guide the model toward true physical laws with zero additional inference overhead.

## Background & Motivation

Neural ODEs are powerful tools for learning dynamical systems from data, capable of learning ODEs of the form $\dot{s} = h_\theta(t, s)$. However, not all ODEs carry physical meaning — the stationary action principle fundamental to physics requires that system trajectories satisfy the Euler-Lagrange equations. Standard Neural ODEs provide no mechanism to ensure that the learned ODE constitutes an Euler-Lagrange equation, potentially yielding unphysical solutions.

The core problem has two aspects: (1) **the identification problem**: how to differentiably quantify how closely an ODE approximates the Euler-Lagrange equations; and (2) **the learning problem**: how to guide Neural ODEs during training to converge toward true Euler-Lagrange equations.

Existing approaches such as Lagrangian Neural Networks (LNNs) directly predict a Lagrangian and derive the ODE from it, but this requires computing the Euler-Lagrange equations in both forward and backward passes, incurring high computational cost and poor stability. This paper adopts an **inverse approach**: learn the ODE directly, then verify whether it satisfies the Lagrangian structure via the Helmholtz conditions.

## Method

### Overall Architecture

The model consists of three networks: $f_{\theta_1}$ models the acceleration $\ddot{x}$, $g_{\theta_2}$ learns the Hessian matrix of the Lagrangian, and $\text{NN}_{\theta_3}$ predicts initial velocities from initial positions. Training jointly optimizes the regression loss $\mathcal{L}_R$ and the Helmholtz metric regularization term $\mathcal{L}_H$; at inference time, only $f_{\theta_1}$ and $\text{NN}_{\theta_3}$ are used.

### Key Designs

1. **Differentiable Implementation of Helmholtz Metrics**:
    - Function: Transforms the Helmholtz conditions into loss functions optimizable via neural networks.
    - Mechanism: Defines auxiliary quantities $\Phi$, parameterizes the Hessian matrix $g$ with a neural network $g_{\theta_2}$, and minimizes the MSE of residuals from three Helmholtz conditions. The minimum absolute eigenvalue $\lambda_{\min}$ normalizes the residuals to prevent the network from "cheating" by learning small eigenvalues.
    - Design Motivation: A differentiable, trainable metric is needed to quantify whether an ODE originates from a Lagrangian while avoiding degenerate solutions.

2. **Multi-Objective Optimization Strategy**:
    - Function: Jointly optimizes the regression loss and the Helmholtz metric.
    - Mechanism: The total loss is $\mathcal{L}_{\text{tot}} = \mathcal{L}_R + \mathcal{L}_H$. Gradient clipping (clipping $\|\nabla_{\theta_1} \mathcal{L}_H\|$ to $c_1 \approx 0.05$) ensures data dominates in the early training phase, preventing convergence to an incorrect Euler-Lagrange equation.
    - Design Motivation: Overly strong regularization causes the model to converge to physical laws inconsistent with the data.

3. **Zero Additional Inference Overhead**:
    - Function: Helmholtz metrics are used only during training and are entirely absent at inference time.
    - Mechanism: $g_{\theta_2}$ is computed and optimized only during training; at inference, only $f_{\theta_1}$ is needed to evaluate the right-hand side of the ODE.
    - Design Motivation: This is a core advantage over LNNs, which require computing Euler-Lagrange equations via automatic differentiation at inference time, incurring significant overhead.

### Loss & Training

- Regression loss: $\mathcal{L}_R = \text{MSE}(x_{\text{pred}}, x_{\text{data}})$
- Helmholtz regularization: $\mathcal{L}_H = \text{MSE}(\sum_i \mathcal{R}_i / \lambda_{\min})$
- Training techniques: progressive time step inclusion (gradually increasing the number of time steps to avoid local minima); the output of $g_{\theta_2}$ is processed with a $\sinh$ transformation to handle exponential behavior.
- Network architecture: $f_{\theta_1}$ (1 layer × 16), $g_{\theta_2}$ (2 layers × 64), $\text{NN}_{\theta_3}$ (3 layers × 16), Softplus activation; RAdam optimizer, batch size 128.

## Key Experimental Results

### Main Results

| System | Helmholtz Metric Behavior | Remarks |
|--------|--------------------------|---------|
| Undamped oscillator | $\mathcal{L}_H$ decreases significantly | Lagrangian exists |
| Kepler problem | $\mathcal{L}_H$ decreases significantly | Lagrangian exists |
| Damped oscillator (time-independent $g$) | $\mathcal{L}_H$ fails to decrease | No time-independent Lagrangian exists |
| Damped oscillator (time-dependent $g$) | $\mathcal{L}_H$ decreases significantly | Time-dependent Lagrangian exists |
| Non-Lagrangian ODE | $\mathcal{L}_H$ improves only marginally | Correctly identified as having no Lagrangian |

### Ablation Study

Comparison of 40 pairs of regularized vs. unregularized models (MSE ratio $R = \exp(l_{\text{reg}} - l_{\text{unreg}})$):

| Evaluation Dimension | MSE Ratio $R$ | Significance |
|---------------------|--------------|--------------|
| Position $x$ (in-distribution) | < 1 | Significant (Welch's t-test) |
| Velocity $\dot{x}$ | << 1 | Highly significant |
| Acceleration $\ddot{x}$ | << 1 | Highly significant |
| Extrapolation (2× training time) | << 1 | Highly significant |

### Key Findings

- Helmholtz metrics accurately distinguish between Lagrangian and non-Lagrangian systems.
- The learned $g$ closely matches the analytical Lagrangian Hessian (Kepler problem: median error $3.7 \times 10^{-4}$).
- Regularization substantially improves learning accuracy for velocity and acceleration, with particularly notable gains in extrapolation performance.

## Highlights & Insights

- **Elegant inverse approach**: Rather than directly modeling the Lagrangian as in LNNs, this work learns the ODE and subsequently verifies the existence of a Lagrangian, avoiding the overhead of forward Euler-Lagrange computation.
- **Physical diagnostic capability**: Beyond improving learning, the framework can diagnose whether a system is physical — the Helmholtz metric fails to converge for damped systems under time-independent settings, correctly reflecting the non-fundamental nature of damping.
- **Solid theoretical foundation**: Grounded in Douglas's classical Helmholtz condition theory (1939/1941), the work bridges century-old mathematical tools with modern deep learning.

## Limitations & Future Work

- Validation is limited to low-dimensional (2D) toy systems; scalability to high-dimensional and more complex systems remains untested.
- No systematic quantitative comparison against LNNs or Hamiltonian Neural Networks is provided.
- Numerical stability may become problematic in higher dimensions (eigenvalue computation, robustness of gradient clipping).
- The expressive capacity of $g_{\theta_2}$ may be insufficient when the system's Lagrangian takes a highly complex form.

## Related Work & Insights

- **Lagrangian Neural Networks (LNNs)**: A forward approach — predicts the Lagrangian to derive the ODE; the present work takes the inverse approach.
- **Hamiltonian Neural Networks**: An analogous idea formulated within the equivalent Hamiltonian framework.
- **Physics-Informed Neural Networks (PINNs)**: A broader paradigm for learning with physical constraints.
- **Implications for ML in the physical sciences**: Helmholtz metrics can serve as a general diagnostic tool for learning physical systems.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovatively applies classical Helmholtz conditions as regularization for Neural ODEs.
- Experimental Thoroughness: ⭐⭐⭐ Validation systems are relatively simple; comparisons with competing methods are absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear; physical intuition is richly conveyed.
- Value: ⭐⭐⭐⭐ Introduces a new regularization paradigm for Physics-Informed ML.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/others/measuring_uncertainty_calibration.md)
- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](../../AAAI2026/others/measuring_model_performance_in_the_presence_of_an_intervention.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](../../ICLR2026/others/tabstruct_measuring_structural_fidelity_of_tabular_data.md)

</div>

<!-- RELATED:END -->
