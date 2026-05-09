---
title: >-
  [Paper Note] Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free
description: >-
  [NeurIPS 2025][Schrödinger bridge] This paper generalizes the Schrödinger bridge problem (SBP) from Brownian motion reference processes to multivariate Ornstein-Uhlenbeck (mvOU) reference processes, derives exact solutions for the Gaussian case, and proposes the simulation-free mvOU-OTFM algorithm for general distributions.
tags:
  - NeurIPS 2025
  - Schrödinger bridge
  - non-equilibrium diffusion
  - Ornstein-Uhlenbeck process
  - Flow Matching
  - optimal transport
date: 2026-05-08
content_hash: cacf3dc32473dd2c
---

# Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free

**Conference**: NeurIPS 2025
**arXiv**: [2505.16644](https://arxiv.org/abs/2505.16644)
**Code**: None
**Area**: Other
**Keywords**: Schrödinger bridge, non-equilibrium diffusion, Ornstein-Uhlenbeck process, Flow Matching, optimal transport

## TL;DR

This paper generalizes the Schrödinger bridge problem (SBP) from Brownian motion reference processes to multivariate Ornstein-Uhlenbeck (mvOU) reference processes, derives exact solutions for the Gaussian case, and proposes the simulation-free mvOU-OTFM algorithm for general distributions.

## Background & Motivation

**Background**: The Schrödinger bridge problem (SBP) is the theoretical backbone for reconstructing stochastic dynamics from population snapshots, with broad applications in biological cell dynamics modeling and generative models.

**Limitations of Prior Work**: Existing methods almost exclusively assume Brownian motion or scalar OU processes as reference dynamics, restricting modeling to gradient-driven (equilibrium) systems; biological systems, however, are inherently non-equilibrium.

**Key Challenge**: Non-equilibrium systems require asymmetric drift matrices (non-conservative force fields), yet methods permitting general drift (e.g., IPFP, Neural SDEs) rely on expensive numerical simulation and suffer from poor accuracy in high dimensions.

**Goal**: Efficiently and accurately solve the SBP for non-equilibrium systems within the framework of linear reference dynamics (mvOU processes).

**Key Insight**: Exploit the analytical tractability of mvOU processes to strike a balance between physical relevance and computational feasibility.

**Core Idea**: Use the mvOU process as the reference process and leverage its analytical bridge formulas to enable simulation-free score/flow matching training.

## Method

### Overall Architecture

The SBP is solved in two stages: (1) solve the static SBP via entropic optimal transport to obtain the optimal coupling $\pi$; (2) use the analytical formulas of mvOU bridges to train neural networks approximating the dynamic SBP solution via score and flow matching.

### Key Designs

1. **Analytical Characterization of mvOU Bridges (Theorem 1 & 2)**:

    - **Function**: Derive closed-form expressions for the bridge SDE, score function, and flow field of an mvOU process conditioned on its endpoints.
    - **Design Motivation**: These closed-form expressions are the foundation for simulation-free training.
    - **Mechanism**: The bridge SDE takes the form $d\mathbf{Y}_t = (\mathbf{A}(\mathbf{Y}_t - \mathbf{m}) + \mathbf{c}_{t|(\mathbf{x}_0,\mathbf{x}_1)})dt + \boldsymbol{\sigma} d\mathbf{B}_t$, where the control term is $\mathbf{c}_{t} = -\mathbf{\Lambda}_t^{-1}(\mathbf{Y}_t - \mathbf{k}_t)$; the score is $\mathbf{s}_{t|(\mathbf{x}_0,\mathbf{x}_T)}(\mathbf{x}) = \mathbf{\Sigma}_{t|(\mathbf{x}_0,\mathbf{x}_T)}^{-1}(\boldsymbol{\mu}_{t|(\mathbf{x}_0,\mathbf{x}_T)} - \mathbf{x})$.
    - **Novelty**: Reduces to the standard Brownian bridge formula when $\mathbf{A}=0$.

2. **Exact Solution for Gaussian Schrödinger Bridges (Theorem 3)**:

    - **Function**: Provide a complete analytical characterization of mvOU-GSB for Gaussian endpoint distributions.
    - **Design Motivation**: Serves as an accuracy benchmark and can be applied directly to interpolation between Gaussian distributions.
    - **Mechanism**: A coordinate transformation recasts the mvOU-SBP as a standard entropic OT problem, yielding closed-form expressions for the mean and covariance.
    - **Novelty**: Generalizes the scalar OU and Brownian motion results of Bunne et al. (2023).

3. **mvOU-OTFM Algorithm (Proposition 1 & Theorem 4)**:

    - **Function**: Provide a simulation-free training algorithm for general (non-Gaussian) distributions.
    - **Design Motivation**: Analytical solutions are not directly available for general distributions.
    - **Mechanism**: The static SBP is first solved via the Sinkhorn algorithm using the analytical mvOU transport cost; conditional score and flow matching then train the network. Loss:
      $$L(\theta,\varphi) = \mathbb{E}[\|\mathbf{u}_t^\theta(\mathbf{z}) - \mathbf{u}_{t|(\mathbf{x}_0,\mathbf{x}_T)}(\mathbf{z})\|^2 + \lambda_t \|\mathbf{s}_t^\varphi(\mathbf{z}) - \mathbf{s}_{t|(\mathbf{x}_0,\mathbf{x}_T)}(\mathbf{z})\|^2]$$
    - **Novelty**: The Brownian variant of Tong et al. (2023b) is recovered as a special case.

4. **Iterative Reference Process Refinement (Algorithm 2)**:

    - **Function**: Learn optimal mvOU reference process parameters from data.
    - **Design Motivation**: The initial reference process may be insufficiently accurate.
    - **Mechanism**: Alternately solve the SBP and update $(\mathbf{A}, \mathbf{m})$ via regularized linear regression.

### Loss & Training

- Joint score and flow matching loss (Eq. 19).
- Two-stage training decoupled by the Sinkhorn algorithm (OT coupling first, then neural network regression).
- The mvOU bridge quantities $\mathbf{\Phi}_t$ and $\mathbf{\Omega}_t$ require only a one-time numerical integration and can be cached for reuse.

## Key Experimental Results

### Main Results

**Gaussian SBP accuracy comparison** (Bures-Wasserstein marginal error):

| Dimension $d$ | mvOU-OTFM | BM-OTFM | IPML (→) | NLSB |
|---|---|---|---|---|
| 2 | **0.19±0.17** | 8.40±0.77 | 5.65±1.41 | 1.21±0.18 |
| 10 | **0.59±0.36** | 8.93±0.55 | 3.00±0.63 | 1.36±0.13 |
| 50 | **2.21±0.36** | 11.74±0.37 | 8.32±0.63 | 6.39±0.13 |
| 100 | **6.84±0.78** | 15.14±0.95 | 14.38±0.38 | 17.40±0.13 |

**Repressilator leave-one-out interpolation error**:

| Metric | Iter. 0 | Iter. 4 | SBIRR (mvOU) | SBIRR (MLP) |
|---|---|---|---|---|
| EMD | 3.38±1.52 | **1.40±0.57** | 2.10±0.74 | 1.67±0.95 |
| Energy distance | 1.86±1.06 | **0.89±0.55** | 1.39±0.82 | 1.10±0.86 |

### Ablation Study

**Cell cycle data: effect of mvOU reference process scaling parameter $\gamma$**:
- $\gamma=0$ (Brownian motion): fails to recover cyclic dynamics.
- $\gamma=30$–$70$: Bures-Wasserstein interpolation error is minimized.
- $\gamma>100$: performance degrades.
- Optimal $\gamma=50$ correctly recovers cell cycle oscillatory behavior.

### Key Findings

- mvOU-OTFM achieves the highest accuracy across all dimensions; at $d=50$, the marginal error is only one-third that of NLSB.
- Training is extremely fast: $d=50$ converges in 1–2 minutes on CPU, versus 15+ minutes on GPU for NLSB.
- Iterative reference process refinement consistently reduces error.
- The drift matrix $\mathbf{A}$ learned from repressilator data closely matches the cyclic activation–inhibition pattern of the system Jacobian.
- On real scRNA-seq data, the mvOU reference successfully recovers cell cycle oscillations, while the Brownian reference fails.

## Highlights & Insights

- **Solid theoretical contributions**: Four theorems fully cover the analytical theory of mvOU-SBP, from bridge characterization to exact Gaussian solutions to simulation-free learning.
- **Physical intuition meets mathematical rigor**: Asymmetric drift matrices naturally model non-equilibrium systems while the linear framework preserves tractability.
- **Strong practicality**: Minute-scale training on CPU significantly outperforms competing methods on GPU.
- **Elegant unification**: Brownian SBP and scalar OU-SBP are both recovered as special cases.

## Limitations & Future Work

- Matrix operations with $O(d^3)$ complexity limit scalability beyond $d>100$.
- The linear reference dynamics assumption restricts modeling of highly nonlinear systems.
- Minibatch OT coupling may introduce bias.
- Gaussian Process techniques could be integrated to extend the method to higher dimensions.

## Related Work & Insights

- This work constitutes a theoretical generalization of the Gaussian SBP framework of Bunne et al. (2023).
- The [SF]²M method of Tong et al. (2023b) is recovered as a special case under the Brownian reference.
- The iterative refinement strategy of SBIRR (Shen et al. 2024) is adopted here with a more efficient solver.
- The method provides a computationally more efficient tool for single-cell RNA-seq dynamic modeling.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Theoretically rigorous generalization, though the research direction is incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage with synthetic and biological data; exact solutions serve as baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematically rigorous, consistent notation, and clear logical flow.
- **Value**: ⭐⭐⭐⭐ Practical applicability in both computational biology and generative modeling.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Adjoint Schrödinger Bridge Sampler](adjoint_schrödinger_bridge_sampler.md)
- [\[NeurIPS 2025\] Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](modeling_cell_dynamics_and_interactions_with_unbalanced_mean_field_schrödinger_b.md)
- [\[NeurIPS 2025\] Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning](directional_non-commutative_monoidal_structures_for_compositional_embeddings_in_.md)
- [\[NeurIPS 2025\] Non-Clairvoyant Scheduling with Progress Bars](non-clairvoyant_scheduling_with_progress_bars.md)
- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)

<!-- RELATED:END -->
