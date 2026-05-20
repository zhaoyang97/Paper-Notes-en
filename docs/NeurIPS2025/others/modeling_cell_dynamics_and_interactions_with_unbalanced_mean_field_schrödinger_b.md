---
title: >-
  [Paper Note] Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge
description: >-
  [NeurIPS 2025][Schrödinger Bridge] This paper proposes the Unbalanced Mean Field Schrödinger Bridge (UMFSB) framework and the CytoBridge deep learning algorithm…
tags:
  - "NeurIPS 2025"
  - "Schrödinger Bridge"
  - "cell dynamics"
  - "cell-cell interaction"
  - "single-cell RNA sequencing"
  - "optimal transport"
date: 2026-05-08
content_hash: 395692f4611100a8
---

# Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge

**Conference**: NeurIPS 2025
**arXiv**: [2505.11197](https://arxiv.org/abs/2505.11197)  
**Code**: [GitHub](https://github.com/zhenyiizhang/CytoBridge-NeurIPS)  
**Area**: Computational Biology / Optimal Transport
**Keywords**: Schrödinger Bridge, cell dynamics, cell-cell interaction, single-cell RNA sequencing, optimal transport

## TL;DR

This paper proposes the Unbalanced Mean Field Schrödinger Bridge (UMFSB) framework and the CytoBridge deep learning algorithm, which simultaneously model unbalanced stochastic cell dynamics and cell-cell interactions from sparse temporal snapshot data.

## Background & Motivation

Reconstructing dynamics from high-dimensional distributional samples is a central challenge in both science and machine learning. In single-cell biology, inferring cell trajectories from scRNA-seq snapshot data is a fundamental problem. Prior work has made progress in several directions:

- **Optimal Transport (OT) methods**: e.g., the Benamou–Brenier formulation for inferring continuous cell dynamics
- **Unbalanced OT**: incorporating the Wasserstein–Fisher–Rao metric to account for cell proliferation and death
- **Schrödinger Bridge**: modeling the most likely stochastic transition path between distributions
- **Unbalanced stochastic methods**: e.g., DeepRUOT, which jointly handles unbalanced and stochastic effects

However, **virtually all existing methods neglect cell-cell interactions**. In realistic biological settings, intercellular communication is a fundamental life process that directly influences cell state transitions. For instance, neighboring cells may promote or inhibit each other's differentiation via signaling molecules.

**Key Challenge**: How to simultaneously model unbalanced stochastic effects (proliferation/death) and inter-particle interactions within a dynamical optimal transport framework?

**Key Insight**: Unify Mean Field Schrödinger Bridge (handling interactions) with Regularized Unbalanced OT (handling unbalanced effects) into the novel UMFSB framework, and design a deep learning solver accordingly.

## Method

### Overall Architecture

CytoBridge parameterizes the key quantities of the UMFSB problem with four neural networks: the transport velocity $\mathbf{v}_\theta$, growth rate $g_\theta$, log-density function $s_\theta$, and interaction potential $\Phi_\theta$. Training adopts a two-stage strategy (pretraining followed by joint training), with a loss function comprising an energy loss, a reconstruction loss, and a Fokker–Planck physics-informed constraint.

### Key Designs

1. **UMFSB Framework**: Unifies RUOT (for unbalanced effects) and MFSB (for interactions). The core PDE constraint is:

$$\frac{\partial \rho}{\partial t} = -\nabla_\mathbf{x} \cdot \left[\left(\mathbf{b} - \int k(\mathbf{x},\mathbf{y})\nabla_\mathbf{x}\Phi(\mathbf{x}-\mathbf{y})\rho(\mathbf{y},t)d\mathbf{y}\right)\rho\right] + \frac{\sigma^2}{2}\Delta\rho + g\rho$$

   Setting $k=0$ recovers RUOT; setting $g=0$ recovers MFSB, yielding an elegant unification.

2. **Fisher Information Regularization (Theorem 4.1)**: Transforms the original SDE-constrained problem into an ODE-constrained one, making computation more tractable. The key equivalence introduces a new vector field $\mathbf{v} = \mathbf{b} - \frac{1}{2}\sigma^2\nabla_\mathbf{x}\log\rho$ (i.e., the probability flow ODE), absorbing the diffusion term into a Fisher information regularizer $\frac{\sigma^4}{8}\|\nabla_\mathbf{x}\log\rho\|^2$.

3. **Weighted Particle Simulation (Proposition 5.1)**: Approximates the continuous density evolution via a weighted interacting particle system. Each particle has position $\mathbf{X}_t^i$ and weight $w_i(t)$, where the weight evolves as $\frac{dw_i}{dt} = g(\mathbf{X}_t^i, t)w_i(t)$ and the position follows an SDE with interaction terms. As $N \to \infty$, the weighted empirical measure weakly converges to the density solution of UMFSB.

4. **Random Batch Methods (RBM)**: Computing interaction terms naively requires $\mathcal{O}(N^2)$ operations. RBM randomly partitions particles into groups and computes only intra-group interactions, reducing complexity to $\mathcal{O}((p-1)N)$ while maintaining $\mathcal{W}_2$ convergence $\leq C\sqrt{\tau}$.

### Loss & Training

The total loss comprises three components:

- **Energy Loss $\mathcal{L}_{\text{Energy}}$**: Encourages the principle of least action; an upper-bound approximation is used to avoid coupled optimization of $\mathbf{v}_\theta$ and $s_\theta$.
- **Reconstruction Loss $\mathcal{L}_{\text{Recons}}$**: Includes local/global mass matching losses and a Wasserstein distribution matching loss to align generated densities with observed data.
- **Fokker–Planck Constraint $\mathcal{L}_{\text{FP}}$**: A PINN-style physics-informed loss that enforces all four networks to satisfy the Fokker–Planck equation.

Training proceeds in two stages: a **pretraining stage** (sequentially initializing $g_\theta, \mathbf{v}_\theta, \Phi_\theta, s_\theta$ in four steps) and a **joint training stage** (minimizing the total loss to jointly optimize all networks).

## Key Experimental Results

### Main Results: Synthetic Gene Regulatory Network (Attractive Interaction, $\sigma=0.05$)

| Model | $\mathcal{W}_1$ (t=1) | TMV (t=1) | $\mathcal{W}_1$ (t=4) | TMV (t=4) |
|------|---------|---------|---------|---------|
| SF2M | 0.146±0.002 | 0.080±0.000 | 0.554±0.005 | 0.930±0.000 |
| DeepRUOT | 0.044±0.002 | 0.014±0.007 | 0.057±0.003 | 0.075±0.044 |
| UOT-FM | 0.051±0.000 | 0.010±0.000 | 0.054±0.000 | 0.095±0.000 |
| **CytoBridge** | **0.015±0.001** | **0.013±0.009** | **0.038±0.003** | **0.058±0.061** |

### Main Results: Mouse Hematopoiesis Dataset ($\sigma=0.1$)

| Model | $\mathcal{W}_1$ (t=1) | TMV (t=1) | $\mathcal{W}_1$ (t=2) | TMV (t=2) |
|------|---------|---------|---------|---------|
| SF2M | 8.217±0.001 | 2.231±0.000 | 11.086±0.002 | 5.399±0.000 |
| MIOFlow | 6.313±0.000 | 2.231±0.000 | 6.746±0.000 | 5.399±0.000 |
| DeepRUOT | 6.052±0.002 | 0.200±0.001 | 6.757±0.006 | 0.260±0.007 |
| **CytoBridge** | **6.013±0.002** | **0.208±0.001** | **6.644±0.011** | **0.078±0.013** |

### Ablation Study

| Configuration | Key Metric | Remarks |
|------|---------|------|
| No interaction (DeepRUOT) | $\mathcal{W}_1$=0.044 (t=1) | Captures transition patterns but fails to capture variance reduction |
| No growth (SF2M) | TMV=0.930 (t=4) | Cannot match mass changes, produces erroneous transitions |
| Full CytoBridge | $\mathcal{W}_1$=0.015, TMV=0.013 | Simultaneously captures transition and interaction patterns |
| Lennard-Jones potential | Correctly identified | Recovers LJ potential with both attractive and repulsive components |
| No-interaction ground truth | Correctly identifies $k \approx 0$ | Accurately determines the absence of interactions |

### Key Findings

- CytoBridge achieves superior or comparable performance to existing SOTA in distribution matching ($\mathcal{W}_1$) and mass matching (TMV) across all datasets.
- The model automatically learns the profile of the interaction potential (attractive/repulsive/absent) from data.
- On the mouse hematopoiesis dataset, regions with high learned growth rates correspond to hematopoietic stem cell populations.
- Correlation analysis between learned interaction forces and cell differentiation directions suggests that interactions may promote early differentiation while suppressing late-stage differentiation.

## Highlights & Insights

- **Solid theoretical contributions**: The UMFSB framework elegantly unifies RUOT and MFSB; the Fisher information transform converting the SDE problem into an ODE problem is a notable technical contribution.
- **Strong interpretability**: The model directly outputs interaction potentials, growth rates, and the Waddington landscape, facilitating biological interpretation.
- **Data-driven interaction learning**: No a priori specification of the interaction potential form is required; the neural network learns it directly from snapshot data.
- The adoption of RBM enables the method to scale to large datasets (49K+ cells).

## Limitations & Future Work

- The current approach optimizes an upper bound on the energy term rather than the original UMFSB objective.
- The RBF expansion of interaction terms limits expressiveness; sparse representation methods could be considered.
- The multi-stage training procedure is relatively complex; incorporating optimality conditions from the HJB equation could simplify training.
- Biological priors (e.g., ligand–receptor information) are not exploited to constrain the interaction network.
- Developing simulation-free training methods (analogous to flow matching) is a promising direction for future work.

## Related Work & Insights

- **Relation to DeepRUOT**: CytoBridge extends DeepRUOT by incorporating interaction terms, generalizing from RUOT to UMFSB.
- **Relation to Meta Flow Matching**: The latter uses a GCN to model neighborhood cell influences but considers only the neighborhood structure at the initial time point.
- The Fisher information regularization paradigm is generalizable to other optimization problems involving SDE constraints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The UMFSB framework is a meaningful extension of OT theory, being the first to unify unbalanced effects, interactions, and stochasticity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Both synthetic and real-world data are evaluated with thorough ablation studies, though comparisons of computational efficiency are lacking.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and the overall structure is well-organized.
- Value: ⭐⭐⭐⭐ Directly valuable to the computational biology community, with broader potential for machine learning applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adjoint Schrödinger Bridge Sampler](adjoint_schrödinger_bridge_sampler.md)
- [\[NeurIPS 2025\] Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free](learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)
- [\[NeurIPS 2025\] Inferring Stochastic Dynamics with Growth from Cross-Sectional Data](inferring_stochastic_dynamics_with_growth_from_cross-sectional_data.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](normalization_in_attention_dynamics.md)
- [\[NeurIPS 2025\] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action](variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)

</div>

<!-- RELATED:END -->
