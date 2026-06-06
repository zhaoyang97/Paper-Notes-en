---
title: >-
  [Paper Note] Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers
description: >-
  [NeurIPS 2025][Time Series][Reduced Order Modeling] This paper proposes Φ-ROM, a framework that embeds differentiable PDE solvers into the training loop of nonlinear reduced order models. By leveraging solver feedback to…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Reduced Order Modeling"
  - "Differentiable Solvers"
  - "Physics-Informed Neural Networks"
  - "Implicit Neural Representations"
  - "Partial Differential Equations"
date: 2026-05-08
content_hash: d2f2ed4c91d4bb95
---

# Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers

**Conference**: NeurIPS 2025
**arXiv**: [2505.14595](https://arxiv.org/abs/2505.14595)  
**Authors**: Nima Hosseini Dashtbayaz (UWO), Hesam Salehipour (Autodesk Research), Adrian Butscher (Autodesk Research), Nigel Morris (Autodesk Research)
**Code**: [phi-rom.github.io](https://phi-rom.github.io)  
**Area**: Time Series
**Keywords**: Reduced Order Modeling, Differentiable Solvers, Physics-Informed Neural Networks, Implicit Neural Representations, Partial Differential Equations

## TL;DR

This paper proposes Φ-ROM, a framework that embeds differentiable PDE solvers into the training loop of nonlinear reduced order models. By leveraging solver feedback to directly constrain latent space dynamics, Φ-ROM significantly outperforms purely data-driven ROMs and other physics-informed methods in generalization to unseen parameters/initial conditions, long-horizon extrapolation, and solution recovery from sparse observations.

## Background & Motivation

### State of the Field
Reduced order modeling (ROM) aims to accelerate simulation by compressing high-dimensional PDE systems onto low-dimensional latent manifolds, with broad applications in many-query engineering tasks such as design optimization, optimal control, and inverse problems. Traditional methods employ linear dimensionality reduction (e.g., PCA), while recent work adopts nonlinear manifold ROMs based on autoencoders, learning temporal evolution in latent space via Neural ODEs and related approaches.

### Limitations of Prior Work
- **Fundamental limitation of data-driven ROMs**: Existing nonlinear ROMs (e.g., DINo) rely entirely on numerical solver-generated datasets for training, discarding the solver once data generation is complete. The learned latent dynamics are not guaranteed to be physically consistent, leading to error accumulation, failure in long-horizon extrapolation, and poor generalization to new parameters or initial conditions.
- **Limitations of existing physics-informed approaches**: (i) PINN-ROM augments the loss with PDE residual terms but performs poorly on nonlinear PDEs due to spectral bias and other optimization difficulties; (ii) CROM directly solves the PDE in physical space during inference, precluding genuine dimensionality-reduction speedup and being infeasible for INRs that do not output complete physical fields.
- **Wasted solver information**: Despite encoding true discretized physics, high-fidelity numerical solvers are entirely excluded from the training process in all existing frameworks.

### Root Cause
By directly embedding the numerical solver into the ROM training procedure, the latent space dynamics can be constrained by true physics, thereby achieving better generalization, extrapolation, and data efficiency without sacrificing the benefits of dimensionality reduction.

## Method

### Overall Architecture
Φ-ROM builds upon the DINo framework and comprises two core components:

1. **Conditional INR decoder $D_\theta$**: Maps low-dimensional latent coordinates $\alpha \in \mathbb{R}^k$ to physical fields $\hat{\mathbf{u}} = D_\theta(\alpha, \mathcal{X})$, employing an auto-decoding scheme (encoder-free; latent coordinates are obtained via inversion optimization) that naturally supports arbitrary and irregular observation grids.
2. **Dynamics network $\Psi_\phi$**: Learns the temporal evolution of the latent space in Neural ODE form as $\Psi_\phi(\alpha) = \dot{\alpha}$.

### Core Innovation: Physics-Informed Dynamics Loss
The key contribution is the use of a differentiable PDE solver $\mathcal{S}$ to directly compute target time derivatives in the latent space. The procedure is as follows:

1. Compute the Jacobian of the decoder with respect to $\alpha$: $J_D(\alpha)\dot{\alpha} = d\hat{\mathbf{u}}/dt$
2. Use the solver $\mathcal{S}$ to compute the true time derivative $d\hat{\mathbf{u}}/dt$ of the reconstructed field $\hat{\mathbf{u}}$
3. Project onto the latent space via the pseudoinverse: $\dot{\alpha}^* = J_D^\dagger(\alpha) \cdot d\hat{\mathbf{u}}/dt$
4. Define the dynamics loss: $L_{dyn} = \ell(\Psi_\phi(\alpha), \dot{\alpha}^*)$

### Loss & Training
The reconstruction loss and dynamics loss are jointly optimized:

$$L_{\Phi\text{-ROM}} = \lambda L_{rec} + (1-\lambda) L_{dyn}$$

where $\lambda \in [0.5, 0.8]$ controls the regularization strength. Since $\mathcal{S}$ is differentiable, gradients of $L_{dyn}$ back-propagate through the solver to the decoder parameters $\theta$ and the latent manifold $\Gamma$, providing physics regularization.

### Hyper-reduction for Scalability
Computing the full Jacobian and solving the least-squares problem scales with the spatial grid size $N$. The paper adopts a **randomized hyper-reduction** strategy:
- For each training snapshot, randomly subsample $\gamma N$ ($\gamma = 0.1$) spatial points.
- Construct the Jacobian and solve the least-squares problem only at the subsampled points.
- Compute the Jacobian using forward-mode automatic differentiation.

### Sparse Data Training
Since the INR decoder naturally supports arbitrary grids, training data may be provided on an irregular sparse grid $\mathcal{X}_{tr}$, while the solver operates on its dedicated grid $\mathcal{X}_\mathcal{S}$. The reconstruction loss is computed on $\mathcal{X}_{tr}$ and the dynamics loss on $\mathcal{X}_\mathcal{S}$, enabling flexible data assimilation.

### Parameterized Dynamics Network
For parameterized PDEs (e.g., varying Reynolds numbers), the parameter $\beta$ is passed through a trainable linear transformation and concatenated with $\alpha$ as input to the dynamics network: $\dot{\alpha} = \Psi(\alpha, \beta)$, enabling cross-parameter generalization.

## Key Experimental Results

### Experiment 1: Comparison of Physics-Informed Strategies (Diffusion & Burgers')

On the 2D diffusion equation and 1D Burgers' equation, Φ-ROM is compared against DINo (purely data-driven), PINN-ROM, and CROM:

| Method | Diffusion $[0,T_{tr}]$ | Diffusion $[T_{tr},T_{te}]$ | Burgers' $[0,T_{tr}]$ | Burgers' $[T_{tr},T_{te}]$ |
|------|:---:|:---:|:---:|:---:|
| **Φ-ROM** | **0.080** | **0.034** | 0.021 | **0.028** |
| DINo | 0.089 | 0.051 | 0.021 | 0.060 |
| PINN-ROM | 0.081 | 0.042 | 0.088 | 0.348 |
| FD-CROM | 0.131 | 0.351 | **0.001** | 0.044 |
| AD-CROM | 0.093 | 0.106 | 0.121 | 0.196 |
| ↓AD-CROM | 0.456 | 0.856 | 0.090 | 0.212 |

Key findings: PINN-ROM and AD-CROM fail severely on the nonlinear Burgers' equation (extrapolation errors of 0.348 and 0.196, respectively). Although FD-CROM achieves high in-window accuracy, it degrades in extrapolation. Φ-ROM achieves the best performance across all extrapolation settings.

### Experiment 2: Generalization on Complex PDEs (N-S, KdV & LBM)

Results on 2D Navier-Stokes decaying turbulence (64×64 grid, 256 training trajectories), 2D KdV equation (512 trajectories), and 2D flow past a cylinder (LBM, parameterized by Reynolds number):

| Problem | Setting | Φ-ROM Test Interp. | DINo Test Interp. | Φ-ROM Test Extrap. | DINo Test Extrap. |
|------|------|:---:|:---:|:---:|:---:|
| N-S | Full grid training | **0.170** | 0.580 | **0.373** | 1.543 |
| N-S | 5% sparse training | **0.192** | 0.584 | **0.397** | 1.450 |
| N-S | 2% sparse training | **0.189** | 0.594 | **0.394** | 1.517 |
| KdV | Full grid training | **0.233** | 0.459 | **0.486** | 0.728 |
| KdV | 5% sparse training | **0.248** | 0.543 | **0.499** | 0.851 |
| LBM | Full grid (out-of-domain β) | **0.115** | 0.457 | **0.180** | 0.566 |
| LBM | 2% sparse (out-of-domain β) | **0.188** | 0.412 | **0.303** | 0.507 |

In N-S extrapolation, Φ-ROM outperforms DINo by more than 4× (0.373 vs. 1.543). Even with only 2% observed spatial points, Φ-ROM maintains accuracy close to full-grid training on N-S (0.394 vs. 0.373), whereas DINo degrades severely.

## Highlights & Insights

- **Methodological innovation**: This is the first work to embed a differentiable PDE solver into the nonlinear ROM training loop, directly imposing physical constraints on the latent space through solver gradients. The concept is elegant and the gains are substantial.
- **Consistent superiority**: Φ-ROM demonstrates consistent advantages in generalization and extrapolation across 5 different PDEs and 5 numerical methods (finite difference, spectral, finite volume, and Lattice Boltzmann), validating the robustness and generality of the framework.
- **Sparse data capability**: Training with only 2%–5% of spatial observation points suffices to recover full-field solutions, providing a practical framework for field reconstruction and data assimilation.
- **Open-source and extensible**: A JAX-based open-source implementation is provided, readily extensible to new PDEs and solvers.

## Limitations & Future Work

- **Requires differentiable solvers**: The PDE solver must be implemented in a differentiable framework such as JAX or PyTorch, limiting out-of-the-box applicability to legacy codes.
- **Increased training cost**: Compared to purely data-driven methods, each training step incurs additional solver forward/backward passes and Jacobian computation.
- **Restricted to first-order temporal PDEs**: The current framework assumes the form $\dot{u} = \mathcal{N}(u;\beta)$, and does not cover higher-order temporal PDEs (e.g., wave equations) or steady-state PDEs.
- **Large-scale 3D problems**: The scalability of hyper-reduction in high-dimensional settings has not been validated and requires further investigation.
- **Decoder accuracy bottleneck**: DINo achieves higher in-distribution accuracy (e.g., N-S: 0.036 vs. 0.064), suggesting that physics regularization trades off some in-distribution fitting accuracy.

## Related Work & Insights

- **DINo (Yin et al. 2023)**: The direct baseline for Φ-ROM, also employing an INR decoder and Neural ODE dynamics network, but purely data-driven training leads to poor generalization and accumulating extrapolation errors.
- **CROM (Chen et al. 2021)**: Solves the PDE directly in physical space during inference, but without genuine dimensionality-reduction speedup; infeasible for complex multi-field PDEs (e.g., requiring both pressure and velocity); accuracy collapses after subsampling.
- **PINN-ROM**: Incorporates PDE residuals via automatic differentiation as regularization, but fails severely on nonlinear PDEs due to spectral bias and optimization difficulties (Burgers' extrapolation error: 0.348).
- **Lee & Parish (2025)**: Introduces parameterized dynamics networks; this work improves upon it by adding a trainable linear transformation that significantly enhances parameter generalization.
- **Classical projection-based ROMs (Benner et al. 2015)**: Based on linear subspaces, unable to capture the manifold structure of nonlinear dynamics.

## Rating

- Novelty: ⭐⭐⭐⭐ — Embedding differentiable solvers into ROM training is a natural yet previously unrealized idea; the hyper-reduction projection design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five PDEs, five numerical methods, and diverse training/testing settings (sparse, parameter extrapolation, temporal extrapolation) with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with clearly articulated motivation; mathematical formulations and intuitive explanations are well balanced.
- Value: ⭐⭐⭐⭐ — Provides an effective new paradigm for physics-informed ROM; open-source code enhances practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[NeurIPS 2025\] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling](parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](../../ICML2026/time_series/generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)
- [\[ICML 2026\] IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection](../../ICML2026/time_series/impact_influence_modeling_for_open-set_time_series_anomaly_detection.md)
- [\[ACL 2026\] A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting](../../ACL2026/time_series/a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari.md)

</div>

<!-- RELATED:END -->
