---
title: >-
  [Paper Note] Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics
description: >-
  [ICML 2026][Physics & Scientific Computing][Mesh Physics] Starting from four physical principles—"locality + permutation equivariance + orientation covariance + energy conservation/dissipation inequality"—it is proven th…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Mesh Physics"
  - "port-Hamiltonian"
  - "Topology–Metric Separation"
  - "Energy Conservation"
  - "MeshGraphNet"
date: 2026-05-08
content_hash: f6ef079a801831ba
---

# Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics

**Conference**: ICML 2026  
**arXiv**: [2605.00394](https://arxiv.org/abs/2605.00394)  
**Code**: None  
**Area**: 3D Physical Simulation / Structure-Preserving Neural Networks / Mesh Learning  
**Keywords**: Mesh Physics, port-Hamiltonian, Topology–Metric Separation, Energy Conservation, MeshGraphNet

## TL;DR
Starting from four physical principles—"locality + permutation equivariance + orientation covariance + energy conservation/dissipation inequality"—it is proven that any mesh-based physical dynamics satisfying these axioms can be locally reduced to a port-Hamiltonian form at the Jacobian level. In this formulation, the conservative interconnection structure $J$ is completely fixed by the mesh topology (the signed incidence matrix $D_k$), while metric and dissipation enter through learnable $G$ and $R$. MeshFT-Net, designed based on this theory, achieves near-zero energy drift and correct dispersion/momentum in long-term rollouts, significantly outperforming MGN and HNN.

## Background & Motivation

**Background**: The field is rapidly evolving along two paths: using GNNs/message passing to learn mesh physics (fluids, elastodynamics, acoustics) (e.g., MeshGraphNets, SPH-Net, FNO), and using explicit structure-preserving networks (HNN, LNN, port-Hamiltonian NN, GENERIC) that hard-code energy or symplectic structures into the architecture.

**Limitations of Prior Work**: Pure MGN-based methods suffer from energy drift and non-physical modes in long-term rollouts. HNN or global port-Hamiltonian NNs require manual selection of a global Hamiltonian/template, showing poor robustness when the model is misspecified. Neither path clearly explains which degrees of freedom in mesh physics are non-physical and should be eliminated by the structure.

**Key Challenge**: In exterior differential geometry, the exterior derivative $d$ is topological (metric-independent), while geometric/material properties enter only through metric operators like the Hodge $\star$. Unfortunately, existing learned simulators entangle the two, allowing metric learning to pollute the topological structure, which in turn causes topological errors to amplify metric errors.

**Goal**: (1) Provide a clean set of physical principles; (2) Formally prove that these principles force the dynamics into a port-Hamiltonian form at the Jacobian level; (3) Design a network that "fixes the topology and learns only the metric," and verify its long-term stability, dispersion, momentum, and OOD generalization.

**Key Insight**: MeshGraphNet is viewed as a superset that already satisfies Locality (L) and Permutation Equivariance (P) but lacks Orientation Covariance (O) and Energy Balance (E). By enforcing O and E, redundant non-physical degrees of freedom are structurally eliminated, leaving exactly the topological skeleton of classical Discrete Exterior Calculus (DEC) plus local metric operators.

**Core Idea**: "Physical Principles $\Rightarrow$ Jacobian Factorization $\Rightarrow$ Fixed Topology, Learnable Metric"—topology is fixed by $D_k$ (signed incidence matrix), while only the positive-definite metric $G_\theta$ and semi-positive-definite dissipation $R_\theta$ are learnable.

## Method

### Overall Architecture
The input is a fixed oriented cell complex $\mathcal{K}$ and an initial state $z^0 = (z_k^0, z_{k+1}^0)$ (cochain degrees of freedom, e.g., node potentials + edge flows). The output is the state $z^{n+1}$ at the next time step. The pipeline consists of: (1) Using the reduction theorem to constrain dynamics to a port-Hamiltonian form $\dot z = (J - R(z)) G(z) z$; (2) Setting $J = \begin{pmatrix} 0 & -D_k^\top \\ D_k & 0 \end{pmatrix}$ fixed by the mesh incidence matrix (untrained); (3) Using a Strang splitting integrator to alternate between "half-step dissipation + conservation step + half-step dissipation," where all operations are sparse matvecs with $O(N)$ complexity.

### Key Designs

1.  **Four Axioms + Local port-Hamiltonian Reduction Theorem**:
    *   **Function**: Provides an axiomatic definition for "valid mesh physical dynamics" and proves that any $F$ satisfying these axioms can be written at the Jacobian level as $\partial F / \partial z = (J(z) - R(z)) G(z)$.
    *   **Mechanism**: The four axioms are (L) Locality, (P) Permutation Equivariance, (O) Orientation Covariance (reversing cell direction flips the sign of oriented variables but keeps scalar quantities like $H$ and $e^\top \dot z$ invariant), and (E) Energy Balance—dynamics are split into a conservative part $F_\text{con}$ (satisfying $e^\top F_\text{con} = 0$) and a dissipative part $F_\text{diss}$ (satisfying $e^\top F_\text{diss} \le 0$). It is proven that the Jacobian of the conservative part must be skew-symmetric and the dissipative part must be negative semi-definite. Furthermore, the off-diagonal blocks of the conservative interconnection must take the signed-incidence structure $J_{k,k+1} = -D_k^\top C_k(z)$ and $J_{k+1,k} = C_k(z) D_k$.
    *   **Design Motivation**: Unlike "positing a port-Hamiltonian template," this work deduces "what must be fixed and what can be learned," framing the division of labor between topology and metric as a structural theorem rather than engineering heuristics.

2.  **MeshFT-Net: Fixed Topology, Learnable Metric**:
    *   **Function**: Directly implements the theorem as a neural network architecture—fixing $J$ as $\begin{pmatrix} 0 & -D_k^\top \\ D_k & 0 \end{pmatrix}$ and restricting learnable weights to the SPD metric $G_\theta$ and PSD dissipation $R_\theta$.
    *   **Mechanism**: Energy is defined as a quadratic form $H_\theta(z) = \tfrac{1}{2} z^\top G_\theta z$, with co-energy $e = G_\theta z$; $G_\theta$ is implemented using softplus diagonals or small Cholesky blocks, conditioned on local geometric/material features via permutation-equivariant + orientation-even MLPs. $R_\theta(z)$ takes a Rayleigh form $z \mapsto \gamma(\cdot) G_\theta^{-1} z$ to ensure PSD.
    *   **Design Motivation**: Moves "topology" from the training set to the "mesh itself"—since it is given by the mesh and should not be learned from data—while leaving "metric/materials/dissipation" to the network. Consequently, the topological structure remains stable even with limited or out-of-distribution data.

3.  **Strang Splitting Time Integrator + CFL Guard**:
    *   **Function**: Maintains symplecticity of the conservative part while accurately integrating dissipation within a single update layer.
    *   **Mechanism**: Algorithm 1 provides a KDK pattern: a half-step dissipation $\exp(-\tfrac{\Delta t}{2} R G) z$, followed by a symmetric leapfrog for the conservative part $z_k \leftarrow z_k - \tfrac{\Delta t}{2} D_k^\top G_{k+1} z_{k+1} \rightarrow z_{k+1} \leftarrow z_{k+1} + \Delta t D_k G_k z_k^\text{half} \rightarrow$ another half-step conservation, and finally another half-step dissipation. `CFLGuard(Δt)` scales the step size based on local maximum eigenvalues to prevent numerical explosion.
    *   **Design Motivation**: Standard Euler schemes cannot accurately conserve energy; Strang splitting allows the conservative and dissipative sub-flows to proceed without interference, yielding an analytically provable $\dot H = -e^\top R(z) e \le 0$ when paired with an exact skew-symmetric $J$.

### Loss & Training
Supervised one-step prediction: $\sum_k \text{Loss}(\hat z_k^{n+1}, z_k^{n+1})$, without using PDE residual terms. The inductive bias stems entirely from the fixed $J$ and SPD/PSD structures. Multi-step stacking with final output supervision is used for rollout tasks.

## Key Experimental Results

### Main Results
Evaluated on analytic plane waves (regular grid + Delaunay), Rayleigh damped oscillation, acoustic scattering from The Well, and OOD settings (frequency/wave speed/resolution) against MGN, MGN-HP, HNN, PI-MGN, FNO, and GraphCON.

| Task | Model | One-step MSE | TSMSE (rollout) | Energy Drift |
|------|------|--------------|-----------------|----------|
| Analytic Plane Wave (Regular) | MGN | $1.6{\times}10^{-7}$ | $1.3{\times}10^{-1}$ | $25.9$ |
| Analytic Plane Wave | HNN | $3.5{\times}10^{-8}$ | $3.0{\times}10^{-3}$ | $1.0{\times}10^{-2}$ |
| Analytic Plane Wave | **Ours** | $\mathbf{1.3{\times}10^{-9}}$ | $\mathbf{9.6{\times}10^{-5}}$ | $\mathbf{1.3{\times}10^{-4}}$ |
| Rayleigh Damping | MGN | $5.2{\times}10^{-8}$ | $1.7{\times}10^{-1}$ | NEE $2.2$ |
| Rayleigh Damping | **Ours** | $1.2{\times}10^{-7}$ | $\mathbf{2.1{\times}10^{-2}}$ | NEE $\mathbf{2.1{\times}10^{-2}}$ |

### Ablation Study

| Configuration | TSMSE | Energy Drift |
|------|-------|----------|
| Fixed $J$ + Diagonal $G$ | $4.52{\times}10^{-5}$ | $0.115$ |
| Fixed $J$ + Full $G$ | $3.28{\times}10^{-5}$ | $0.028$ |
| $z$-dependent $J$ + Diagonal $G$ | $\mathbf{6.77{\times}10^{-6}}$ | $0.025$ |
| $z$-dependent $J$ + Full $G$ | $6.17{\times}10^{-6}$ | $0.030$ |

Physical consistency diagnostics show that MeshFT-Net ranks first in wave speed error, gauge relations, PDE residuals (short/long-range), kinetic-potential energy equipartition, and momentum conservation, with momentum error as low as $4.9{\times}10^{-8}$ (vs. $0.39$ for MGN and $1.07$ for HNN).

### Key Findings
- One-step MSE is not strongly correlated with long-range rollout performance: MGN achieved the lowest one-step MSE in damping tasks but had nearly 10x higher rollout TSMSE than MeshFT-Net, indicating that short-term local accuracy does not imply long-term physical fidelity.
- Momentum conservation is not an explicit constraint, but MeshFT-Net naturally inherits action-reaction relations by enforcing orientation covariance (O). Methods without enforced (O) show momentum drift orders of magnitude higher.
- On OOD tasks, MGN/FNO/PI-MGN diverge ($>100$) when resolution or wave speed shifts, whereas MeshFT-Net maintains energy drift $<\mathcal{O}(10^{-1})$. The inductive bias of the fixed topology provides genuine generalization.
- Nonlinear shallow-water experiments show that when coefficients are state-dependent, changing $J$ to be state-dependent improves performance, though "fixed unlearnable topology + full $G$" can compensate to some extent, representing a trade-off between model capacity and structure.

## Highlights & Insights
- "Theorem-driven architecture design" is the primary methodological contribution: first proving the solution space is structurally constrained to port-Hamiltonian forms under specific axioms, then treating these constraints as hard architecture rather than working backward from a global Hamiltonian template.
- The "topology–metric separation" is abstract yet practical: topology (incidence matrix $D_k$) belongs to the mesh and is never learned; metric ($G, R$) belongs to physics and is learnable. This manually injects a layer of "physically non-generalizable information" into the GNN, leaving the learnable parts for properties truly related to material and geometry.
- The interpretation of the MGN series is very clear: MGN is not "wrong," but its axioms are too broad. By adding (O) and (E), the state space of the dynamics is refined to a physically plausible subset, providing a clear "energy + orientation" patch for future mesh-based simulators.

## Limitations & Future Work
- The main experiments utilized state-independent $G_\theta$ (quadratic storage). Strongly nonlinear PDEs (e.g., Navier-Stokes, plasticity, phase transitions) require nonlinear constitutive models $G_\theta(z)$ / $\Psi_\theta(e)$, explored only in supplementary toy experiments.
- Axioms (O) and (E) are sufficient conditions but do not guarantee correct behavior under external sources, complex boundary conditions, or multi-physics coupling; external source terms require further extension.
- The framework relies on the incidence structure $D_k$ of the cell complex, making it not directly applicable to completely unstructured or time-varying topologies (e.g., fracturing materials, adaptive meshing).
- In some OOD shifts, MeshFT-Net is only "relatively" better than FNO/GraphCON; absolute TSMSE remains non-zero, suggesting topological constraints cannot fully replace sufficient data.

## Related Work & Insights
- **vs MGN**: MGN satisfies only (L) + (P); this work enforces (O) + (E), structurally eliminating non-physical degrees of freedom and improving long-range stability by orders of magnitude.
- **vs HNN / port-Hamiltonian NN**: Those methods learn on a "global Hamiltonian template," requiring the template to be correct. This work uses "local Jacobian factorization," making it more robust to template errors.
- **vs DEC / Data-driven Exterior Calculus**: Both share the topology–metric separation idea; this work derives it from physical axioms rather than starting from differential geometry templates, making it more general.
- **vs PI-MGN / FNO**: PDE residual/operator learning is data-driven + weak physics; this work is structure-driven and does not require knowledge of the PDE form, leading to more stable generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ Strictly translates the topology–metric separation of exterior differential geometry into GNN architecture design principles.
- Experimental Thoroughness: ⭐⭐⭐⭐ Analytic + real data + physical diagnostics + OOD + nonlinear ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear theorem statements; Algorithm 1 is reproducible almost without the appendix.
- Value: ⭐⭐⭐⭐ Provides a theorem-driven design paradigm for the learned simulator community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)
- [\[NeurIPS 2025\] High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](../../NeurIPS2025/physics/high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](../../NeurIPS2025/physics/hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[CVPR 2026\] PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](../../CVPR2026/physics/physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/physics/astral_training_physics-informed_neural_networks_with_error_majorants.md)

</div>

<!-- RELATED:END -->
