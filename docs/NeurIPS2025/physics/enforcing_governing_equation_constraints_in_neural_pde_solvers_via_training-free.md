---
title: >-
  [Paper Note] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][Neural PDE Solver] Two training-free post-processing projection methods are proposed—nonlinear LBFGS optimization and local linearization projection—to project the outputs o…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "Neural PDE Solver"
  - "Constraint Projection"
  - "LBFGS"
  - "Physical Consistency"
  - "Post-processing"
date: 2026-05-08
content_hash: c02e44ba2d315526
---

# Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections

**Conference**: NeurIPS 2025
**arXiv**: [2511.17258](https://arxiv.org/abs/2511.17258)  
**Code**: None  
**Area**: Scientific Computing / PDE Solving
**Keywords**: Neural PDE Solver, Constraint Projection, LBFGS, Physical Consistency, Post-processing

## TL;DR
Two training-free post-processing projection methods are proposed—nonlinear LBFGS optimization and local linearization projection—to project the outputs of neural PDE solvers onto the feasible manifold satisfying governing equation constraints. Evaluated on Lorenz/KS/Navier-Stokes, both methods substantially reduce constraint violations and improve accuracy, markedly outperforming physics-informed training.

## Background & Motivation

**Background**: Neural PDE solvers (e.g., FNO, DeepONet) can rapidly predict PDE solutions, yet their outputs frequently violate the physical constraints of the governing equations—even when performing well on standard metrics, the predicted solutions may fail to satisfy fundamental physical laws such as mass conservation, momentum conservation, and incompressibility.

**Limitations of Prior Work**:
   - **Physics-Informed methods** (e.g., PINN/PINO): Incorporate PDE residual penalty terms into the loss function, which complicates training, leads to unstable convergence, and provides no precise control over the degree of constraint satisfaction.
   - **Architecture enforcement methods**: Guarantee certain constraints (e.g., zero divergence) through network design, but at the cost of reduced model expressiveness and requiring custom architectures for each constraint type.
   - **Auxiliary network methods**: Introduce additional networks to learn constraints, increasing training overhead and hyperparameter tuning burden.

**Key Challenge**: Constraints arising from dynamical PDEs are inherently nonlinear and induce long-range temporal dependencies—constraint violations at one time step propagate along the entire trajectory through time evolution. Existing methods either handle only simple linear constraints (e.g., divergence-free) or cannot guarantee strict satisfaction.

**Goal**: After a neural solver has been trained, how can its outputs be post-hoc projected onto the feasible manifold satisfying nonlinear dynamical PDE constraints, without any network retraining.

**Key Insight**: The constraint enforcement problem is reformulated as a numerical optimization problem—the neural network prediction serves as an initial guess, and a classical optimization algorithm (LBFGS) or linearized projection is used to find the solution closest to the prediction that satisfies the PDE constraints.

**Core Idea**: Rather than approximately satisfying physical constraints during training, numerically project predictions onto the constraint manifold after inference via optimization.

## Method

### Overall Architecture
Given a neural PDE solver prediction $\hat{u}$, the goal is to find a solution $u^*$ satisfying discretized PDE constraints $h(u) = c$ such that $\|u^* - \hat{u}\|$ is minimized. Here $h(u)$ encodes the discretized PDE operator, boundary conditions, and initial conditions, and $c$ is the corresponding right-hand side. The entire process occurs at inference time and involves no updates to the neural network parameters.

### Key Designs

1. **Nonlinear LBFGS Projection**:

    - **Function**: Directly solves the unconstrained relaxation of the constraint projection problem.
    - **Mechanism**: The constraint projection problem is relaxed to $\min_u \|u - \hat{u}\| + \lambda \|h(u) - c\|$ and solved iteratively via the L-BFGS algorithm, which leverages gradient and approximate Hessian information for efficient quasi-Newton optimization.
    - **Design Motivation**: Nonlinear constraints admit no closed-form solution, but the neural network prediction already constitutes a good initial guess, enabling rapid convergence. L-BFGS avoids constructing a full Hessian matrix, making it memory-efficient.

2. **Linearized Constrained Projection**:

    - **Function**: Applies a first-order Taylor expansion to the nonlinear constraints and solves the resulting linear constraint projection.
    - **Mechanism**: Linearizing at $\hat{u}$ yields $J_h u = b$; the closed-form solution $u = \hat{u} - \mathcal{C}^\top(\mathcal{C}\mathcal{C}^\top)^{-1}(\mathcal{C}\hat{u} - b)$ is then applied. For large systems, the inversion is avoided by employing iterative solvers such as conjugate gradient (CG) or GMRES, requiring only Jacobian-vector products (JVP) and vector-Jacobian products (VJP), which are efficiently computed via automatic differentiation.
    - **Design Motivation**: Linear projection requires only a single computation step and is fast; JVP/VJP avoid explicitly constructing large-scale Jacobian matrices.

3. **Linearized Relaxed Projection**:

    - **Function**: A soft-constraint variant following linearization.
    - **Mechanism**: Solves $u = (I + \lambda \mathcal{C}^\top\mathcal{C})^{-1}(\hat{u} + \lambda \mathcal{C}^\top b)$, with $\lambda$ trading off constraint satisfaction against proximity to the original prediction.
    - **Design Motivation**: When linearization is insufficiently accurate (highly nonlinear constraints), strict projection may introduce artifacts; the relaxed variant provides a gentler correction.

### Loss & Training
The proposed methods require no training—all projections are post-processing operations performed after inference. For baseline models, standard MSE models and physics-informed models (PINN/PINO) are trained separately; the latter augments the MSE loss with a PDE residual penalty.

## Key Experimental Results

### Main Results

**Navier-Stokes (resolution 64, most critical experiment)**:

| Method | MSE (×10⁻¹) | Residual (×10⁻²) |
|--------|-------------|-----------------|
| FNO baseline | 13.0 | 8.13 |
| FNO + Constrained | 10.5 | 4.21 |
| FNO + Relaxed | 12.2 | 3.40 |
| FNO + LBFGS | **2.63** | **0.00901** |
| PINO baseline | 13.8 | 6.48 |
| PINO + LBFGS | **3.21** | **0.00956** |

LBFGS projection reduces the MSE on the NS equations by ~80% and the constraint residual by ~900×.

### Ablation Study

| PDE System | Baseline Residual | LBFGS Residual | Reduction |
|------------|------------------|----------------|-----------|
| Lorenz (3D ODE) | 50.8×10⁻⁴ | 1.18×10⁻⁴ | ~97.7% |
| KS (1D PDE, res64) | 46.8×10⁻⁵ | 4.53×10⁻⁵ | ~90.3% |
| NS (2D PDE, res64) | 8.13×10⁻² | 0.009×10⁻² | ~99.9% |

Cross-resolution results: KS and NS exhibit similar trends at resolutions of 128/256; constraint residuals increase with resolution (as finer grids reveal more violations), yet LBFGS projection remains consistently effective.

### Key Findings
- **LBFGS substantially outperforms linearization methods**: Linear approximations are valid only locally and degrade rapidly away from the linearization point. The Taylor expansion analysis in Figure 2 clearly demonstrates this—the first-order approximation becomes noisy near the optimum, while the second-order approximation remains reliable.
- **Post-processing outperforms training-time constraints**: PINO (which optimizes PDE residuals during training) exhibits higher constraint violations than standard FNO + LBFGS post-processing, indicating that incorporating PDE penalties into the loss can harm training stability.
- **NS equations benefit the most**: NS solutions contain rich small-scale structures that MSE-trained models fail to capture; LBFGS projection recovers these fine structures by enforcing PDE constraints.
- **Physics-informed training can even be detrimental**: PINN-MLP on Lorenz achieves constraint violations comparable to or higher than vanilla MLP, as multi-objective optimization renders training more difficult.

## Highlights & Insights
- **A coarse-to-fine two-stage paradigm**: A neural network rapidly produces a coarse prediction, which is then refined to physical consistency via numerical methods. This elegantly combines the speed advantage of neural networks with the accuracy guarantees of numerical methods, and the paradigm generalizes to any AI generation task requiring hard constraint satisfaction.
- **Scalable design via JVP/VJP**: Automatic differentiation efficiently computes Jacobian-vector products without explicitly constructing Jacobian matrices, making linearized projection scalable to large systems (NS at resolution 256).
- **Constraint landscape visualization** (Figure 2): Constraint violations are plotted against path length along the LBFGS optimization trajectory, overlaid with first- and second-order Taylor approximations, intuitively explaining why LBFGS outperforms linearization. This analysis methodology is itself a valuable diagnostic tool.

## Limitations & Future Work
- **Computational overhead**: LBFGS projection requires multiple iterations (200 steps), each demanding evaluation of the PDE residual and its gradient, which may become a bottleneck for large-scale 3D problems.
- **Non-differentiability**: The current projection cannot backpropagate gradients into neural network parameters and is therefore incompatible with end-to-end training. The authors suggest developing differentiable projection operators in future work.
- **Dependence on initial prediction quality**: If the neural network's initial prediction is too far from the true solution, LBFGS may converge to an incorrect local minimum.
- **Stochasticity not addressed**: For problems requiring uncertainty quantification or multimodal solutions, deterministic projection may be inappropriate.
- **Discretization error**: Projection guarantees satisfaction of the discretized constraints; however, discretization itself introduces error, and true physical consistency depends on grid resolution.

## Related Work & Insights
- **vs. PINN/PINO**: Physical constraints as training loss vs. post-processing constraints. The experiments in this paper convincingly demonstrate that post-processing methods are far superior to training-time methods in constraint satisfaction, challenging the intuition that "physics-informed is better."
- **vs. Architecture enforcement** (e.g., divergence-free networks): Architecture enforcement handles only specific constraint types (typically linear), whereas projection methods apply to arbitrary nonlinear constraints.
- **vs. Traditional numerical methods**: The approach can be viewed as a hybrid of "neural network as a high-quality initial guess + classical numerical solving," combining the advantages of both.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The projection idea is not new, but its systematic application to nonlinear dynamical PDE constraints with a rigorous comparison against physics-informed methods is a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three PDE systems of varying complexity, multiple resolutions, and multiple baselines; 3D complex geometry experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical exposition is rigorous; the constraint landscape visualization provides highly insightful analysis.
- **Value**: ⭐⭐⭐⭐ Highly practical—physical consistency is substantially improved without retraining the model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/physics/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[NeurIPS 2025\] INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)

</div>

<!-- RELATED:END -->
