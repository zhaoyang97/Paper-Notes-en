---
title: >-
  [Paper Note] Integration Matters for Learning PDEs with Backward SDEs
description: >-
  [NeurIPS 2025][Scientific Computing][BSDE] This paper identifies the root cause of why standard BSDE methods underperform PINNs — an irreducible discretization bias introduced by Euler-Maruyama integration — and proposes…
tags:
  - "NeurIPS 2025"
  - "Scientific Computing"
  - "BSDE"
  - "PDE solving"
  - "Stratonovich integration"
  - "Heun method"
  - "discretization bias"
date: 2026-05-08
content_hash: cd2528c22f4ce112
---

# Integration Matters for Learning PDEs with Backward SDEs

**Conference**: NeurIPS 2025
**arXiv**: [2505.01078](https://arxiv.org/abs/2505.01078)
**Code**: [https://github.com/sungje-park/heunbsde](https://github.com/sungje-park/heunbsde)
**Area**: Scientific Computing / PDE Solving
**Keywords**: BSDE, PDE solving, Stratonovich integration, Heun method, discretization bias

## TL;DR
This paper identifies the root cause of why standard BSDE methods underperform PINNs — an irreducible discretization bias introduced by Euler-Maruyama integration — and proposes Heun-BSDE based on the Stratonovich formulation to fully eliminate this bias, achieving competitive performance against PINNs on high-dimensional PDEs.

## Background & Motivation

**Background**: Two major deep learning approaches exist for solving high-dimensional PDEs: PINNs (directly minimizing PDE residuals) and BSDE methods (reformulating PDEs as forward-backward SDEs and simulating trajectories). BSDE methods have natural advantages for problems with underlying dynamics, such as stochastic optimal control.

**Limitations of Prior Work**: Empirically, BSDE methods perform significantly worse than PINNs, yet the underlying reason has remained unclear. Prior work [33] proposed a hybrid interpolation loss to narrow the gap, but introduced hyperparameters requiring tuning without explaining the root cause.

**Key Challenge**: When the standard BSDE discretizes the one-step consistency loss using Euler-Maruyama (EM) integration, it introduces an **irreducible bias term** independent of the step size $\tau$: $\text{Bias}(\theta) = \frac{1}{2T}\int_0^T \mathbb{E}\text{tr}((H \cdot \nabla^2 u_\theta)^2)dt$, causing the optimization objective to deviate from the true solution.

**Goal**: (1) Identify the root cause of the BSDE vs. PINNs performance gap; (2) Propose a bias-free integration scheme.

**Key Insight**: Reinterpreting the BSDE as a Stratonovich SDE (rather than an Itô SDE) and applying stochastic Heun integration (which converges to the Stratonovich solution) to eliminate the bias.

**Core Idea**: Replace Itô + EM integration with Stratonovich + Heun integration to eliminate the discretization bias in the BSDE loss.

## Method

### Overall Architecture
The forward-backward SDE system for PDE solving is reformulated in Stratonovich form and discretized using the stochastic Heun method (a two-step predictor-corrector scheme), yielding an unbiased one-step consistency loss.

### Key Designs

1. **Bias Analysis (Theorem 4.2)**:

    - Function: Theoretically proves that the bias in the EM one-step loss is irreducible.
    - Mechanism: $\tau^{-2} \cdot \ell_{\text{EM},\tau}(\theta,x,t) = (R[u_\theta])^2 + \frac{1}{2}\text{tr}((H \cdot \nabla^2 u_\theta)^2) + O(\tau^{1/2})$, where the second term is a $\tau$-independent bias that persists even as $\tau \to 0$.
    - Design Motivation: Explains why reducing the step size fails to improve EM-BSDE.

2. **Stratonovich-Heun BSDE Loss (Theorem 4.4)**:

    - Function: Proposes an unbiased loss function.
    - Mechanism: $L_{\text{Heun},\tau}(\theta) = \frac{1}{T}\int_0^T \mathbb{E}[(R[u_\theta])^2]dt + O(\tau^{1/2})$ — the bias term is only $O(\tau^{1/2})$ and can be eliminated by reducing the step size.
    - Design Motivation: The Heun method is a second-order predictor-corrector scheme that naturally converges to the Stratonovich solution.

3. **Efficient Sub-sampling Implementation**:

    - Function: Accelerates training.
    - Mechanism: A full forward SDE trajectory is rolled out (with stop gradient), followed by random sub-sampling of $B$ time steps to compute the loss, rather than using all $N$ steps.
    - Design Motivation: Reduces per-step computational cost while maintaining performance.

### Loss & Training
- Heun discretization requires one additional function evaluation (predictor step + corrector step) but permits larger step sizes.
- The one-step loss ($k=1$) suffices; multi-step loss hyperparameter tuning is unnecessary.

## Key Experimental Results

### Main Results (100D PDE, one-step loss)

| PDE | PINNs | FS-PINNs | EM-BSDE | Heun-BSDE |
|-----|-------|----------|---------|-----------|
| 100D HJB | 0.1260 | 0.0737 | 0.3626 | **0.0493** |
| 100D BSB | 1.5066 | **0.0497** | 0.3735 | 0.0535 |
| 10D BZ | 3.8566 | 0.0351 | 0.1903 | **0.0228** |

### Ablation Study (varying step count $k$)

| Steps $k$ | EM-BSDE (100D HJB) | Heun-BSDE (100D HJB) |
|---------|---------------------|----------------------|
| $k=1$ | 0.3626 | **0.0493** |
| $k=5$ | 0.2117 | 0.0640 |
| $k=50$ | 0.0858 | 0.0601 |

### Key Findings
- **The EM-BSDE bias is indeed the root cause of the performance gap**: EM-BSDE is substantially outperformed by Heun-BSDE across all experiments.
- **Heun-BSDE is competitive with PINNs**: It surpasses PINNs on HJB and BZ, and approaches PINNs on BSB.
- **EM-BSDE requires multiple steps to partially mitigate bias**, but multi-step formulations introduce optimization difficulties; Heun-BSDE achieves strong results with a single step.
- **Sub-sampling incurs negligible performance loss** while substantially accelerating training.

## Highlights & Insights
- **A neglected algorithmic detail determines overall method success**: The choice of integration scheme (EM vs. Heun) has never been seriously studied in the BSDE literature, yet it is the root cause of the performance gap. This serves as a reminder that implementation details may matter more than methodological innovations.
- **Theory-driven algorithmic improvement**: The problem is identified and the solution designed through rigorous bias analysis rather than empirical trial-and-error, with theory and experiments in perfect agreement.
- **The importance of Stratonovich vs. Itô in numerical implementation**: Although the two formulations are equivalent in the continuous limit, the Stratonovich form is more amenable to numerical discretization.

## Limitations & Future Work
- The Heun method requires two function evaluations per step (vs. one for EM), doubling computational cost.
- Experiments are conducted on only three standard PDE benchmarks and have not been validated on more complex real-world problems.
- All methods perform poorly on the 100D BZ problem (RL2 > 1.7), indicating that high-dimensional coupled FBSDEs remain highly challenging.
- Integration with adaptive step-size strategies is not discussed.

## Related Work & Insights
- **vs. PINNs**: PINNs directly minimize PDE residuals and do not suffer from integration bias, but require explicit knowledge of the PDE; BSDE methods can learn from simulation.
- **vs. interpolation loss in [33]**: [33] mitigates bias by tuning the optimal number of steps; Heun-BSDE eliminates bias fundamentally, requiring no such tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ Identifies and explains a previously overlooked yet important issue with an elegant solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theory and experiments are mutually corroborating, with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous and clearly presented.
- Value: ⭐⭐⭐⭐ Makes an important contribution to the BSDE-PDE community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[ICCV 2025\] JPEG Processing Neural Operator for Backward-Compatible Coding](../../ICCV2025/scientific_computing/jpeg_processing_neural_operator_for_backward-compatible_coding.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/scientific_computing/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)

</div>

<!-- RELATED:END -->
