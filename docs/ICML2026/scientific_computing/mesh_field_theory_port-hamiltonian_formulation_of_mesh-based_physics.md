---
title: >-
  [Paper Note] Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics
description: >-
  [ICML 2026][Scientific Computing][Mesh Physics] Starting from four physical principles—locality, permutation equivariance, orientation covariance…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Mesh Physics"
  - "Port-Hamiltonian"
  - "Topology-Metric Separation"
  - "Energy Conservation"
  - "MeshGraphNet"
date: 2026-05-08
content_hash: ab5fdfd6c6865b42
---

# Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics

**Conference**: ICML 2026  
**arXiv**: [2605.00394](https://arxiv.org/abs/2605.00394)  
**Code**: None  
**Area**: 3D Physics Simulation / Structure-Preserving Neural Networks / Mesh Learning  
**Keywords**: Mesh Physics, Port-Hamiltonian, Topology-Metric Separation, Energy Conservation, MeshGraphNet

## TL;DR
Starting from four physical principles—locality, permutation equivariance, orientation covariance, and energy conservation/dissipation inequality—this work proves that any mesh-based physical dynamics satisfying these axioms can be locally reduced to a port-Hamiltonian form at the Jacobian level. In this form, the conserved interconnection structure $J$ is entirely determined by the mesh topology (signed incidence matrix $D_k$), while the metric and dissipation are learned through $G$ and $R$. The proposed MeshFT-Net achieves near-zero energy drift, correct dispersion and momentum over long rollouts, and significantly outperforms MGN and HNN.

## Background & Motivation

**Background**: Two main approaches dominate mesh-based physics learning (e.g., fluids, elastics, acoustics): (1) GNN/message-passing methods (e.g., MeshGraphNets, SPH-Net, FNO), and (2) explicit structure-preserving networks (e.g., HNN, LNN, port-Hamiltonian NN, GENERIC), which hard-code energy/symplectic structures into the architecture.

**Limitations of Prior Work**: Pure MGN-like methods suffer from energy drift and non-physical modes during long rollouts. HNN/global port-Hamiltonian NNs require manually chosen global Hamiltonians/templates, which are not robust to model mis-specifications. Neither approach clarifies which degrees of freedom in mesh physics are non-physical and should be structurally eliminated.

**Key Challenge**: In exterior differential geometry, the exterior derivative $d$ is topological (metric-independent), while geometric/material properties enter only through metric operators like the Hodge star $\star$. Existing learned simulators conflate these, allowing metric learning to contaminate topology and amplifying metric errors through topological inaccuracies.

**Goal**: (1) Establish clean physical principles; (2) formally prove that these principles constrain dynamics to a port-Hamiltonian form at the Jacobian level; (3) design a network that "fixes topology, learns metrics," and validate its long-term stability, dispersion, momentum, and OOD generalization.

**Key Insight**: MeshGraphNet can be viewed as satisfying locality (L) and permutation equivariance (P) but lacking orientation covariance (O) and energy balance (E). Enforcing (O) and (E) eliminates non-physical degrees of freedom structurally, leaving the classical DEC (Discrete Exterior Calculus) topological skeleton plus local metric operators.

**Core Idea**: "Physical principles ⇒ Jacobian factorization ⇒ Fixed topology, learnable metrics"—topology is fixed by $D_k$ (signed incidence matrix), while only the positive-definite metric $G_\theta$ and semi-positive-definite dissipation $R_\theta$ are learnable.

## Method

### Overall Architecture
The input is a fixed directed cell complex $\mathcal{K}$ and initial state $z^0 = (z_k^0, z_{k+1}^0)$ (cochain degrees of freedom, e.g., node potentials + edge fluxes). The output is the next state $z^{n+1}$. The pipeline: (1) Use the reduction theorem to constrain dynamics to the port-Hamiltonian form $\dot z = (J - R(z)) G(z) z$; (2) $J = \begin{pmatrix} 0 & -D_k^\top \\ D_k & 0 \end{pmatrix}$ is entirely determined by the mesh incidence matrix and is not trained; (3) Use a Strang splitting integrator to alternate "half-step dissipation + conservative step + half-step dissipation," with all operations as sparse matrix-vector products, achieving $O(N)$ complexity.

### Key Designs

1. **Four Axioms + Local Port-Hamiltonian Reduction Theorem**:

    - **Function**: Provides an axiomatic definition of "valid mesh-based physical dynamics" and proves that any $F$ satisfying these axioms can be written at the Jacobian level as $\partial F / \partial z = (J(z) - R(z)) G(z)$.
    - **Mechanism**: The four axioms are (L) locality, (P) permutation equivariance, (O) orientation covariance (reversing cell orientation flips only oriented variable signs, leaving scalar quantities $H, e^\top \dot z$ invariant), and (E) energy balance—dynamics decompose into a conservative part $F_\text{con}$ (satisfying $e^\top F_\text{con} = 0$) and a dissipative part $F_\text{diss}$ (satisfying $e^\top F_\text{diss} \le 0$). The theorem proves that the Jacobian of the conservative part must be skew-symmetric, the dissipative part must be semi-negative-definite, and the off-diagonal blocks of the conservative interconnection must take the signed-incidence structure $J_{k,k+1} = -D_k^\top C_k(z)$ and $J_{k+1,k} = C_k(z) D_k$ (Corollary 3.4 shows $C_k$ can be absorbed into coordinate scaling when constant).
    - **Design Motivation**: Unlike directly positing a port-Hamiltonian template, this work deduces "what must be fixed and what can be learned," formalizing the topology-metric separation as a structural theorem rather than an engineering heuristic.

2. **MeshFT-Net: Fixed Topology, Learnable Metrics**:

    - **Function**: Implements the theorem as a neural network architecture—$J$ is fixed as $\begin{pmatrix} 0 & -D_k^\top \\ D_k & 0 \end{pmatrix}$, and learnable weights are restricted to SPD metrics $G_\theta$ and PSD dissipation $R_\theta$.
    - **Mechanism**: Energy is quadratic $H_\theta(z) = \tfrac{1}{2} z^\top G_\theta z$, with co-energy $e = G_\theta z$. $G_\theta$ is implemented via softplus diagonals or small Cholesky blocks, conditioned on local geometry/material features through permutation-equivariant and orientation-even MLPs. $R_\theta(z)$ adopts a Rayleigh form $z \mapsto \gamma(\cdot) G_\theta^{-1} z$ to ensure PSD.
    - **Design Motivation**: Moves "topology" from the training set to the "mesh itself"—since it is mesh-given and should not be learned—and delegates "metrics/materials/dissipation" to the network. This ensures that even with limited or OOD data, the topological structure remains intact.

3. **Strang Splitting Time Integrator + CFL Guard**:

    - **Function**: Maintains symplecticity of the conservative part while precisely integrating dissipation within a single update layer.
    - **Mechanism**: Algorithm 1 implements a KDK scheme: first a half-step dissipation $\exp(-\tfrac{\Delta t}{2} R G) z$, then symmetric leapfrog updates for the conservative part $z_k \leftarrow z_k - \tfrac{\Delta t}{2} D_k^\top G_{k+1} z_{k+1}$ → $z_{k+1} \leftarrow z_{k+1} + \Delta t D_k G_k z_k^\text{half}$ → another half-step conservation, and finally another half-step dissipation. `CFLGuard(Δt)` scales the step size based on the local maximum eigenvalue to prevent numerical blow-up.
    - **Design Motivation**: Conventional Euler integration cannot conserve energy precisely. Strang splitting ensures that the conservative and dissipative subflows do not interfere, enabling an analytically provable $\dot H = -e^\top R(z) e \le 0$.

### Loss & Training
Supervised one-step prediction: $\sum_k \text{Loss}(\hat z_k^{n+1}, z_k^{n+1})$, without PDE residual terms. The inductive bias comes entirely from the fixed $J$ and SPD/PSD structures. Multi-step stacking with supervision on the final output is used for rollout tasks.

## Key Experimental Results

### Main Results
Benchmarked on analytic plane waves (regular grid + Delaunay), Rayleigh damped oscillations, acoustic scattering from The Well, and OOD settings (frequency/wave speed/resolution shifts) against MGN, MGN-HP, HNN, PI-MGN, FNO, and GraphCON.

| Task | Model | One-step MSE | TSMSE (rollout) | Energy Drift |
|------|-------|--------------|-----------------|--------------|
| Plane Waves (Regular Grid) | MGN | $1.6{\times}10^{-7}$ | $1.3{\times}10^{-1}$ | $25.9$ |
| Plane Waves | HNN | $3.5{\times}10^{-8}$ | $3.0{\times}10^{-3}$ | $1.0{\times}10^{-2}$ |
| Plane Waves | **MeshFT-Net** | $\mathbf{1.3{\times}10^{-9}}$ | $\mathbf{9.6{\times}10^{-5}}$ | $\mathbf{1.3{\times}10^{-4}}$ |
| Rayleigh Damping | MGN | $5.2{\times}10^{-8}$ | $1.7{\times}10^{-1}$ | NEE $2.2$ |
| Rayleigh Damping | **MeshFT-Net** | $1.2{\times}10^{-7}$ | $\mathbf{2.1{\times}10^{-2}}$ | NEE $\mathbf{2.1{\times}10^{-2}}$ |

### Ablation Study

| Configuration | TSMSE | Energy Drift |
|---------------|-------|--------------|
| Fixed $J$ + Diagonal $G$ | $4.52{\times}10^{-5}$ | $0.115$ |
| Fixed $J$ + Full $G$ | $3.28{\times}10^{-5}$ | $0.028$ |
| $z$-Dependent $J$ + Diagonal $G$ | $\mathbf{6.77{\times}10^{-6}}$ | $0.025$ |
| $z$-Dependent $J$ + Full $G$ | $6.17{\times}10^{-6}$ | $0.030$ |

Physical consistency diagnostics (Table 3a excerpt) show MeshFT-Net ranks first in wave speed error, gauge relations, PDE residuals (short/long-range), kinetic-potential energy equipartition, and momentum conservation, with momentum error as low as $4.9{\times}10^{-8}$ (MGN: $0.39$, HNN: $1.07$).

### Key Findings
- One-step MSE does not strongly correlate with long-term rollout performance: MGN achieves the lowest one-step MSE in damping tasks but has a rollout TSMSE nearly 10× higher than MeshFT-Net, indicating short-term accuracy does not imply long-term physical fidelity.
- Momentum conservation is not explicitly constrained, but MeshFT-Net naturally inherits action-reaction symmetry by enforcing orientation covariance (O). Methods without (O) exhibit orders-of-magnitude higher momentum drift.
- Under OOD conditions, MGN/FNO/PI-MGN diverge ($>100$) under resolution or wave speed shifts, while MeshFT-Net maintains energy drift $<\mathcal{O}(10^{-1})$, demonstrating the generalization power of fixed-topology inductive bias.
- Nonlinear shallow-water toy experiments show that state-dependent $J$ significantly improves performance when coefficients depend on the state. However, "fixed topology + full $G$" compensates to some extent, balancing model capacity and structure.

## Highlights & Insights
- "Theorem-driven architecture design" is the paper's key methodological contribution: proving that a set of axioms structurally constrains the solution space to port-Hamiltonian form, then hard-coding these constraints into the architecture rather than reverse-engineering from a global Hamiltonian template.
- "Topology-metric separation" is abstract but highly practical: topology (incidence matrix $D_k$) belongs to the mesh and is never learned, while metrics ($G, R$) belong to physics and are learnable. This injects a layer of "non-generalizable physics" into GNNs, leaving learnable parts to material/geometry-related properties.
- The interpretation of MGN is clear—MGN is not wrong but overly permissive. Adding (O) and (E) shrinks the state space to a physically reasonable subset, providing a clear "energy + orientation" patching route for future mesh-based simulators.

## Limitations & Future Work
- The main experiments use state-independent $G_\theta$ (quadratic storage). Strongly nonlinear PDEs (e.g., Navier-Stokes, plasticity, phase transitions) require nonlinear constitutive $G_\theta(z)$/$\Psi_\theta(e)$, which are only explored in supplementary toy experiments.
- Axioms (O) and (E) are sufficient conditions but do not guarantee correctness under external sources, boundary conditions, or multi-physics coupling. The paper explicitly notes that external source terms require further extensions.
- The framework relies on the cell complex's incidence structure $D_k$, making adaptation to fully unstructured/time-varying topologies (e.g., fracture materials, adaptive meshes) non-trivial.
- In some OOD shifts, MeshFT-Net outperforms FNO/GraphCON only relatively, with absolute TSMSE still non-zero, indicating that topological constraints cannot fully replace sufficiently rich data.

## Related Work & Insights
- **vs MGN**: MGN satisfies (L) + (P), while this work enforces (O) + (E), structurally eliminating non-physical degrees of freedom and improving long-term stability by orders of magnitude.
- **vs HNN/port-Hamiltonian NN (Desai et al.)**: Those methods learn on a "global Hamiltonian template," requiring correct templates. This work uses "local Jacobian factorization," making it more robust to template errors.
- **vs DEC/data-driven exterior calculus (Trask et al.)**: Both share the topology-metric separation idea. This work derives it from physical axioms rather than differential geometry templates, making it more general.
- **vs PI-MGN/FNO**: PDE residual/operator learning is data-driven + weakly physical. This work is structure-driven, requiring no knowledge of PDE forms, with more stable generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizes topology-metric separation from exterior differential geometry into a strict GNN design principle, with one-to-one correspondence between theory and implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers analytic + real data, physical diagnostics, OOD, and nonlinear ablations comprehensively.
- Writing Quality: ⭐⭐⭐⭐ Theorems are clearly stated, Algorithm 1 is reproducible, and supplementary material is almost unnecessary.
- Value: ⭐⭐⭐⭐ Provides a theorem-driven design paradigm for the learned simulator community, enabling future structure-preserving networks to follow the "axioms ⇒ Jacobian constraints ⇒ fixed topology" workflow.
