---
title: >-
  [Paper Note] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints
description: >-
  [NeurIPS 2025][Image Generation][Flow Matching] This paper proposes Physics-Constrained Flow Matching (PCFM), a zero-shot inference framework that enforces arbitrary nonlinear equality constraints to machine precision du…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Flow Matching"
  - "Physical Constraints"
  - "Hard Constraint Satisfaction"
  - "PDE Solving"
  - "Zero-Shot Inference"
date: 2026-05-08
content_hash: 40d7f5d04660d9b3
---

# Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints

**Conference**: NeurIPS 2025
**arXiv**: [2506.04171](https://arxiv.org/abs/2506.04171)
**Code**: [Python](https://github.com/cpfpengfei/PCFM) / [Julia](https://github.com/utkarsh530/PCFM.jl)
**Area**: Diffusion Models / Image Generation
**Keywords**: Flow Matching, Physical Constraints, Hard Constraint Satisfaction, PDE Solving, Zero-Shot Inference

## TL;DR
This paper proposes Physics-Constrained Flow Matching (PCFM), a zero-shot inference framework that enforces arbitrary nonlinear equality constraints to machine precision during sampling from pretrained flow matching models. The framework alternates among forward shooting with projection, OT-interpolation backward updates, and relaxed penalty correction at each sub-step, achieving up to 99.5% improvement over baselines on PDE problems involving shocks and discontinuities.

## Background & Motivation

Deep generative models have been applied to simulation and uncertainty quantification of physical PDE systems, yet ensuring that generated samples strictly satisfy conservation laws, boundary conditions, and other physical constraints remains a central challenge.

Limitations of existing methods:

**Soft-constraint methods** (e.g., PINN-style penalty terms, DiffusionPDE) can only approximately satisfy constraints, potentially leading to critical failures in scenarios where exact constraint satisfaction is essential — particularly for hyperbolic equations with shocks.

**The ECI framework** has only been validated on simple linear, non-overlapping constraints (e.g., pointwise Dirichlet conditions) and relies on closed-form analytic projections, making it inapplicable to nonlinear or coupled constraints.

**D-Flow** requires backpropagating gradients through an ODE solver, incurring extremely high computational cost (4–6× slower than PCFM).

**PDM** projects onto the constraint manifold at every step, over-constraining the sampling trajectory for nonlinear or global constraints.

**Key Challenge**: The sampling process of generative models is inherently stochastic and progressive, while constraints need only be exactly satisfied at the final denoised solution. PCFM's key insight is that one can "look ahead" to the terminal state, project it onto the constraint manifold, and "look back" to the current step — simultaneously achieving constraint satisfaction and alignment with the generative flow.

## Method

### Overall Architecture
PCFM augments a pretrained Functional Flow Matching (FFM) model with zero-shot inference. At each sub-step $\tau \to \tau + \delta\tau$, three operations are performed: forward shooting with projection, OT backward update, and relaxed constraint correction.

### Key Designs

1. **Forward Shooting and Gauss-Newton Projection**:

    - At each sub-step, integrate forward to the terminal state $\tau=1$: $u_1 = \text{ODESolve}(u(\tau), v_\theta, \tau, 1)$
    - Apply one Gauss-Newton projection of the terminal state onto the tangent space of the constraint manifold: $u_{\text{proj}} = u_1 - J^\top(JJ^\top)^{-1}h(u_1)$
    - When constraint $h$ is affine, this projection yields the exact solution (Proposition E.1)
    - **Design Motivation**: Forward shooting only requires a low-accuracy integrator (e.g., large-step Euler), keeping costs low.

2. **OT Displacement Interpolation Backward Update**:

    - Trace the projected terminal state $u_{\text{proj}}$ back to $\tau' = \tau + \delta\tau$ along the OT straight-line path: $\hat{u}_{\tau'} = \text{ODESolve}(u_{\text{proj}}, -(u_{\text{proj}} - u_0), 1, \tau')$
    - Core theory (Proposition 3.1): As step size $\delta\tau \to 0$, the OT displacement interpolation $\bar{v}(u) = u_1 - u_0$ approximates the true vector field $v_\theta$ to $O(\delta\tau^p)$, and the backward update is unconditionally stable.
    - **Design Motivation**: Direct backward integration of the ODE is numerically unstable due to sign reversal of Jacobian eigenvalues; OT interpolation completely avoids this issue.

3. **Relaxed Constraint Correction**:

    - Apply a penalized correction to the back-traced $\hat{u}_{\tau'}$: $u_{\tau'} = \arg\min_u \|u - \hat{u}_{\tau'}\|^2 + \lambda\|h(u + \gamma v_\theta(u, \tau'))\|^2$, where $\gamma = 1-\tau'$
    - The constraint is evaluated at the extrapolated point $u + \gamma v_\theta$ rather than the current point, encouraging constraint satisfaction at the terminal state.
    - **Design Motivation**: Discretization and nonlinearity introduce residual errors; the penalty term provides robustness under coarse grids.

4. **Final Projection Guarantee**:

    - If constraint residuals still exceed threshold $\epsilon$ after sampling, a full constraint projection is applied: $\min_{u'} \|u' - u_1\|^2 \text{ s.t. } h(u') = 0$
    - Uses a custom batched differentiable solver that only solves an $m \times m$ Schur complement system ($m = \dim h \ll n$), accounting for only 1–3% of total sampling time.

### Constraint Type Coverage
- **Linear**: Dirichlet initial/boundary conditions, global mass conservation under periodic boundaries
- **Nonlinear**: Nonlinear conservation laws (e.g., nonlinear mass in reaction-diffusion equations), local conservation via Godunov fluxes
- ECI is a special case of this framework: it reduces to ECI when $\lambda = 0$ with linear non-overlapping constraints.

## Key Experimental Results

### Main Results: Multi-PDE Constrained Generation Performance

| PDE | Method | MMSE ($\times 10^{-2}$) | CE(IC) ($\times 10^{-2}$) | CE(CL) ($\times 10^{-2}$) |
|-----|--------|------------------------|--------------------------|--------------------------|
| Heat | PCFM | **0.241** | **0** | **0** |
| Heat | ECI | 0.697 | 0 | 0 |
| Heat | FFM | 4.56 | 579 | 2.11 |
| N-S | PCFM | **4.59** | **0** | **0** |
| N-S | ECI | 5.23 | 0 | 0 |
| Burgers IC | PCFM | **0.052** | **0** | **0** |
| Burgers IC | ECI | 10.0 | 0 | 205 |
| R-D | PCFM | **0.026** | **0** | **0** |
| R-D | ECI | 0.324 | 0 | 6.00 |

### Ablation Study: Effect of Number of Collocation Points (Burgers IC)

| # Collocation Points k | MMSE | CE(CL) | Notes |
|------------------------|------|--------|-------|
| 0 | ~0.3 | ~0 | IC + mass conservation only |
| 1 | ~0.15 | ~0 | Local flux constraints begin to help |
| 3 | ~0.08 | ~0 | Continued improvement |
| 5 | **~0.05** | **~0** | Best; IC/mass constraints unaffected |

### Runtime Comparison

| Method | Heat (ms/sample) | Burgers (ms/sample) |
|--------|-----------------|---------------------|
| PCFM | 291.8 | 1371.0 |
| ECI | 65.6 | 780.6 |
| D-Flow | 2770.5 | 8048.9 |
| DiffusionPDE | 128.6 | 211.2 |

### Key Findings
- PCFM is the only method capable of simultaneously satisfying nonlinear constraints (e.g., nonlinear mass conservation in the Burgers equation) exactly while capturing shock dynamics.
- Unlike PINN-style soft-constraint methods, adding more complementary constraints in PCFM improves generation quality — because hard constraints do not compete via gradient conflicts.
- Relaxed constraint correction ($\lambda > 0$) provides significant robustness gains when the number of steps is small.

## Highlights & Insights
- The work elegantly bridges classical numerical PDE ideas (projection methods, constraint satisfaction) with generative model inference.
- Proposition 3.1 provides theoretical guarantees for the OT backward update, addressing the long-standing numerical instability of backward integration in neural ODEs.

## Limitations & Future Work
- Currently only equality constraints are supported; rigorous enforcement of inequality constraints (e.g., TVD conditions) requires further investigation.
- The Gauss-Newton projection assumes full rank of the constraint Jacobian; regularization is needed in degenerate cases.
- Runtime lies between fast-but-inexact methods (ECI, DiffusionPDE) and slow-but-exact methods (D-Flow).

## Related Work & Insights
- **vs ECI**: ECI is a special case of PCFM ($\lambda=0$, linear constraints); PCFM supports arbitrary nonlinear coupled constraints.
- **vs DiffusionPDE**: DiffusionPDE uses PINN-style soft penalties, cannot satisfy constraints exactly, and requires backpropagation.
- **vs PDM**: PDM projects at every step, over-constraining trajectories for global constraints; PCFM avoids this via look-ahead–project–look-back.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Innovative integration of numerical methods and generative models; zero-shot nonlinear hard constraints are unprecedented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers four PDEs, linear and nonlinear constraints, and multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Method is clearly described with complete algorithmic pseudocode.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to constrained generation in scientific computing; framework generalizes to domains such as molecular design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[NeurIPS 2025\] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)
- [\[NeurIPS 2025\] Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch](shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[NeurIPS 2025\] Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems](equivariant_flow_matching_for_symmetry-breaking_bifurcation_problems.md)

</div>

<!-- RELATED:END -->
