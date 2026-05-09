---
title: >-
  [Paper Note] FEAT: Free Energy Estimators with Adaptive Transport
description: >-
  [NeurIPS 2025][Free energy estimation] This paper proposes the FEAT framework, which employs stochastic interpolants to learn transport maps between two thermodynamic systems. Building on the escorted Jarzynski equality and the controlled Crooks theorem, FEAT provides consistent, minimum-variance free energy difference estimators along with variational upper and lower bounds, thereby unifying equilibrium and non-equilibrium approaches.
tags:
  - NeurIPS 2025
  - Free energy estimation
  - stochastic interpolants
  - Jarzynski equality
  - Crooks theorem
  - variational bounds
date: 2026-05-08
content_hash: c45f82408bcb7d1d
---

# FEAT: Free Energy Estimators with Adaptive Transport

**Conference**: NeurIPS 2025
**arXiv**: [2504.11516](https://arxiv.org/abs/2504.11516)
**Code**: [GitHub](https://github.com/jiajunhe98/FEAT)
**Area**: Computational Physics / Molecular Simulation
**Keywords**: Free energy estimation, stochastic interpolants, Jarzynski equality, Crooks theorem, variational bounds

## TL;DR

This paper proposes the FEAT framework, which employs stochastic interpolants to learn transport maps between two thermodynamic systems. Building on the escorted Jarzynski equality and the controlled Crooks theorem, FEAT provides consistent, minimum-variance free energy difference estimators along with variational upper and lower bounds, thereby unifying equilibrium and non-equilibrium approaches.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Free energy estimation is a fundamental challenge in statistical mechanics, chemistry, biology, and machine learning (e.g., partition function computation and ligand binding free energies).

### State of the Field

**Background**: Classical methods (FEP, BAR, TI) rely on equilibrium sampling or intermediate systems, and fail in high-dimensional spaces where distributional overlap is insufficient.

### Root Cause

**Key Challenge**: The Jarzynski equality introduces non-equilibrium trajectories, but the resulting estimators suffer from high variance.

### Solution Direction

**Solution Direction**: Recent deep learning approaches (normalizing flows combined with targeted FEP, neural TI) have shown progress; however, non-equilibrium methods remain underexplored within deep learning frameworks.

### Remarks

**Remarks**: FEAT leverages stochastic interpolants to efficiently learn transport maps, and provides more flexible, lower-variance estimators via the escorted Jarzynski equality and the Crooks theorem.

## Method

### Overall Architecture

The core pipeline of FEAT:

1. **Learning Transport**: Given samples from two endpoint systems $S_a$ and $S_b$, a velocity field $v_t^\psi$ and energy gradient $\nabla U_t^\theta$ are learned via stochastic interpolants $I_t = \alpha_t x_a + \beta_t x_b + \gamma_t \epsilon$.
2. **Computing Generalized Work**: The (corrected) generalized work $\widetilde{W}^v$ is computed along the learned SDE trajectory.
3. **Estimating Free Energy Difference**: The escorted Jarzynski equality yields variational bounds, while non-equilibrium BAR provides the minimum-variance estimate.

### Key Designs

1. **Stochastic Interpolant Transport Learning**:
    - Function: Two neural networks are used to separately learn the velocity field $v_t^\psi(x)$ and the score function $\nabla U_t^\theta(x)$.
    - Mechanism: The velocity field is trained via a regression loss $\mathcal{L}_v = \mathbb{E}[|v_t^\psi(I_t) - \dot{I}_t|^2]$; the score function is trained via denoising score matching $\mathcal{L}_U^{\text{DSM}} = \mathbb{E}[|\nabla U_t^\theta(I_t) - \gamma_t^{-1}\epsilon|^2]$.
    - Design Motivation: Stochastic interpolants simultaneously learn transport and energy interpolation without requiring a predefined energy path; training does not require Langevin dynamics simulation.

2. **Boundary Condition Correction + FB RND Estimation**:
    - Function: Handles cases where the learned $U_0^\theta \neq U_a$ and $U_1^\theta \neq U_b$, employing the forward-backward Radon-Nikodym derivative to avoid divergence computation.
    - Mechanism: The corrected generalized work incorporates endpoint energy correction terms; the FB RND formulation eliminates the need to compute $\nabla \cdot v_t$ and is robust to discretization errors.
    - Design Motivation: Boundary conditions are difficult to satisfy exactly in practice; divergence computation is computationally expensive and introduces precision issues.

### Loss & Training

- Total training loss: $\mathcal{L}_v(\psi) + \mathcal{L}_U^{\text{DSM}}(\theta) + \mathcal{L}_U^{\text{TSM,0}}(\theta) + \mathcal{L}_U^{\text{TSM,1}}(\theta)$
- Target Score Matching (TSM) uses $\nabla U_a(x_a)$ and $\nabla U_b(x_b)$ as endpoint supervision signals to improve boundary condition satisfaction.
- Estimator hierarchy: Variational lower bound (ELBO) ≤ IWAE lower bound ≤ $\Delta F$ ≤ IWAE upper bound ≤ Variational upper bound (EUBO)
- Non-equilibrium BAR: iteratively solves $C = \Delta F$ to yield a minimum-variance estimate.

## Key Experimental Results

### Main Results

| Method | GMM (2D) Error | Alanine Dipeptide Error | $\phi^4$ QFT Error |
|--------|---------------|------------------------|-------------------|
| Targeted FEP | 0.15 | Large | Large |
| Neural TI | 0.08 | Moderate | Moderate |
| **FEAT (BAR)** | **0.02** | **Smallest** | **Smallest** |

### Ablation Study

- FEAT-BAR vs. FEAT-Jarzynski: BAR consistently outperforms unidirectional Jarzynski with lower variance.
- With/without TSM loss: TSM significantly improves boundary condition matching accuracy.
- FB RND vs. divergence formulation: FB RND is more robust to discretization and incurs lower computational cost.
- ODE ($\sigma_t = 0$) vs. SDE ($\sigma_t > 0$): SDE yields lower variance in high-dimensional problems.

### Key Findings

- FEAT significantly outperforms targeted FEP and neural TI across all evaluated settings.
- Sub-framework relationships: static sampling ($\sigma_t=0, v_t=0$) = FEP; ODE transport ($\sigma_t=0$) = targeted FEP/CNF; perfect transport = TI.
- The quantum field theory experiment ($\phi^4$ theory) demonstrates the applicability of FEAT to fundamental problems in physics.

## Highlights & Insights

- **Theoretical Unification**: FEAT is the first framework to unify FEP, BAR, targeted FEP, TI, and the Jarzynski method within a single formulation.
- The non-equilibrium BAR estimator simultaneously achieves consistency and minimum-variance properties.
- The FB RND technique that eliminates divergence computation has broad applicability.
- From a machine learning perspective, FEAT bridges variational inference (ELBO/EUBO) and statistical physics (Jarzynski/Crooks).

## Limitations & Future Work

- The framework requires exact samples from both endpoint systems and is not applicable to settings where only single-endpoint samples are available.
- Estimation accuracy is directly affected by the quality of neural network training.
- Scalability to large molecular systems remains to be verified (the largest system evaluated is alanine dipeptide).
- Discretization errors are mitigated by FB RND but not entirely eliminated.

## Related Work & Insights

- Relationship to neural TI (Máté et al.): neural TI is a special case of FEAT in the perfect transport limit.
- Relationship to normalizing flow methods: CNF is a special case of FEAT as an ODE formulation when $\sigma_t=0$.
- FEAT has direct applicability to relative binding free energy calculations in drug design.

## Rating

- Theoretical Innovation: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Overall: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)
- [\[AAAI 2026\] Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](../../AAAI2026/physics/adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)
- [\[NeurIPS 2025\] Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions](dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)
- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)

<!-- RELATED:END -->
