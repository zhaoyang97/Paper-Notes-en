---
title: >-
  [Paper Note] Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability
description: >-
  [ICML2025 Spotlight][Hamiltonian Systems] This work proposes a geometric framework of "observation-induced fiber bundles" that internalizes observational uncertainty in partially observable systems from external perturbations into intrinsic variations of fiber coordinates. On this structure, it unifies the treatment of state and observational constraints, establishing a complete theory of symplectic geometry, integrability, symmetry, and conservation laws.
tags:
  - "ICML2025 Spotlight"
  - "Hamiltonian Systems"
  - "Fiber Bundles"
  - "Observational Constraints"
  - "Dirac Constraint Theory"
  - "Symplectic Structures"
  - "Integrability"
  - "Lax Pairs"
  - "Noether's Theorem"
  - "Safe Control"
date: 2026-05-08
content_hash: 8a363233ddbd22f5
---

# Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability

**Conference**: ICML2025 Spotlight  
**arXiv**: [2505.22824](https://arxiv.org/abs/2505.22824)  
**Code**: None  
**Area**: Physics/Geometry (Constrained Dynamics, Fiber Bundle Theory, Symplectic Geometry)  
**Keywords**: Hamiltonian Systems, Fiber Bundles, Observational Constraints, Dirac Constraint Theory, Symplectic Structures, Integrability, Lax Pairs, Noether's Theorem, Safe Control

## TL;DR
This work proposes a geometric framework of "observation-induced fiber bundles" that internalizes observational uncertainty in partially observable systems from external perturbations into intrinsic variations of fiber coordinates. On this structure, it unifies the treatment of state and observational constraints, establishing a complete theory of symplectic geometry, integrability, symmetry, and conservation laws.

## Background & Motivation

Classical constrained Hamiltonian theory (e.g., Dirac constraint quantization) assumes that system states are fully observable, meaning that the constraints $\phi(x)=0$ act directly on the precisely known state $x \in M$. However, in modern control systems, robotics, and quantum measurements, the **incompleteness of observations** (sensor noise, measurement accuracy limits) invalidates this idealized assumption.

Limitations of prior work:

| Method | Core Idea | Limitations |
|------|---------|------|
| Probabilistic constraint methods | Stochastic optimization to address uncertainty | Loss of dynamical system geometric structure, difficult to preserve symplectic structures |
| Robust control | Worst-case analysis of bounded uncertainty | Overly conservative, lacking geometric insights |
| Extended Kalman Filter | Treating uncertainty as external perturbations to estimate and eliminate | Fails to exploit the intrinsic structures of the system |

**Key Insight**: When the system state $x \in M$ can only be indirectly acquired through an observation map $h: M \to Y$ with uncertainty $\epsilon$, the observational uncertainty is not noise to be eliminated, but rather an **intrinsic component** of the system's geometric structure.

## Method

### 1. Construction of Observation-Induced Fiber Bundles

Given a Hamiltonian system $(M, \omega, H)$ and an observation map $h: M \to Y$, an observation-induced fiber bundle $\pi: E \to M$ is constructed:

- **Base Manifold**: $2n$-dimensional symplectic manifold $M$ (phase space)
- **Fiber**: The fiber corresponding to each state $x$, given by $\pi^{-1}(x) = \{\xi \in T^*_{h(x)}Y : \|\xi\|_{\rho_x} \leq \delta(x)\}$
- **Uncertainty Function** $\delta: M \to (0, \Delta]$: Encoding the differences in observation accuracy at different positions
- **Total Space**: $E = \{(x, \xi) : x \in \mathcal{W}, \|\xi\|_{\rho_x} \leq \delta(x)\}$

Key elements include:
- The operating region $\mathcal{W} \subset M$ is a connected open set, and $\bar{\mathcal{W}}$ is compact
- The observation map satisfies the local diffeomorphism condition $\text{rank}(dh) = k$ on $\mathcal{W}$
- The family of fiber metrics $\rho = \{\rho_x\}$ varies smoothly on $\bar{\mathcal{W}}$

### 2. Observation-Adaptive Connections and Curvature

A linear connection $\nabla$ preserving the fiber structure is defined, satisfying:
- **Metric Compatibility**: $\nabla \rho = 0$
- **Observation Compatibility**: Preserving the horizontal-vertical decomposition $TE = H_\nabla E \oplus VE$
- **Torsion-Freeness**: $T^\nabla = 0$ along the base manifold directions
- **Curvature Control**: $\|R^\nabla\|_{L^\infty(\mathcal{W})} < \infty$

The specific form of the connection coefficients is:
- Base manifold part: $\Gamma^k_{ij} = \Gamma^{LC,k}_{ij}$ (Levi-Civita connection)
- Fiber part: $\Gamma^a_{bc} = 0$ (the fiber metric does not depend on fiber coordinates)
- Mixed part: $\Gamma^a_{bi} = \frac{1}{2}\rho^{ac}\partial_i \rho_{bc}$

The mixed curvature simplifies to:

$$R^\nabla{}^a{}_{bij} = \partial_i \Gamma^a_{jb} - \partial_j \Gamma^a_{ib}$$

### 3. Symplectic Structure on Fiber Bundles

The symplectic form on the total space is decomposed into three parts:

$$\omega_E = \pi^* \omega_{\mathcal{W}} + \omega_{\text{fib}} + \Omega_{\text{mix}}$$

- **Symplectic Form of the Base Manifold**: $\pi^*\omega_\mathcal{W} = \sum_\alpha dq^\alpha \wedge dp_\alpha$
- **Standard Symplectic Form of the Fiber**: $\omega_{\text{fib}} = \sum_a d\xi_a \wedge d\pi_a$
- **Curvature Mixed Term**: $\Omega_{\text{mix}} = \sum_{i,a}(K_{ia}\,dx^i \wedge d\xi_a + L_{ia}\,dx^i \wedge d\pi_a)$

Boundary treatment employs a smooth cutoff function $\chi$ to achieve a degenerate transition of the symplectic structure, ensuring that the Hamiltonian equations remain well-defined near the boundary.

### 4. Integrability and Lax Pairs

Geometrical necessary and sufficient conditions for complete integrability under observational constraints are established, and the Arnold-Liouville theorem is extended to the fiber bundle setting. Taking the $n$-particle Toda lattice as an example, the observational Lax pair is constructed:

$$L(\lambda, \epsilon) = L_0(\epsilon) + \lambda L_1(\epsilon)$$

where $L_0$ is a tridiagonal matrix with observational errors (diagonal elements are momenta $p_i$, subdiagonal elements are $e^{y_i + \epsilon_i}$), and $L_1$ is an uncertainty diagonal matrix.

**Conditions for Preserving Integrability**: When the observational error satisfies $\|\epsilon\|_\infty \leq \epsilon_{\text{crit}}$, the modified integrals $I^{\text{obs}}_k = I^{\text{classical}}_k + \epsilon \delta I_k + O(\epsilon^2)$ remain in involution, and the invariant tori undergo only $O(\epsilon)$ regular deformations.

### 5. Symmetries and Noether's Theorem

The geometric characterization and principal bundle representation theory of observational symmetry groups are established. Noether's theorem on fiber bundles is proved, enabling the handling of **observation-dependent conservation laws** that traditional theories cannot cover. A momentum map theory on fiber bundles is also constructed.

### 6. Geometry-Preserving Numerical Algorithms

A numerical integration method that preserves the geometric structure of the fiber bundle is proposed:
- Utilizing the observation-adaptive connection to compute the Hamiltonian vector field
- Adopting geometric projection (rather than algebraic correction) upon constraint violation
- Adapting the time step size to local variations of the observational uncertainty $\delta(x)$
- Geometry loss function design: $L_{\text{geo}} = \|\Phi(x,\xi)\|^2 + \alpha\|\nabla_i \rho_{jk}\|^2 + \beta\|\omega_E - \omega_{\text{ref}}\|^2$

## Key Experimental Results

This paper is a purely theoretical work with no traditional numerical experiments. The effectiveness of the theory is verified through three applications from [25]:

| Application Scenario | Base Manifold $M$ | Observation Space $Y$ | Fiber Bundle Characteristics |
|---------|-----------|-------------|-----------|
| Soft robot (MPM) | Continuum configuration space | $\mathbb{R}^6$ (6 sensors) | Satisfies (C4c) decay condition |
| 7-DOF manipulator | $\mathbb{R}^{14}$ (joint phase space) | $SE(3)$ (end-effector pose) | Avoids singular configurations, $SO(3)$ symmetry |
| Quadrotor navigation | $\mathbb{R}^3 \times SO(3) \times \mathbb{R}^6$ | $\mathbb{R}^4$ (4 depth sensors) | Satisfies (C4b), $\beta=2$ (power decay) |

Convergence of the geometry-preserving algorithm (Theorem 26):
- Constraint preservation error: $|\Phi(x_n, \xi_n)| \leq C_1 h^2$
- Symplectic structure preservation error: $|\omega_{E,n} - \omega_E| \leq C_2 h^2$
- Global convergence: $\|(x_n,\xi_n) - (x(t_n),\xi(t_n))\|_E \leq Che^{Lt_n}$
- 7-DOF manipulator: step size $h \sim 10^{-3}$s, constraint error $\sim 10^{-6}$

## Highlights & Insights

1. **Geometric Internalization of Observational Uncertainty**: Turning observational errors from external perturbations into the intrinsic variations of fiber coordinates. This paradigm shift resembles the equivalence principle in General Relativity; observational uncertainty can be locally "gauged away" under an appropriate geometric structure.
2. **Unified Framework**: The same theory covers Dirac constraints, symplectic reduction, integrability, Noether's theorem, and Control Barrier Functions, bridging classical mechanics with modern safety control.
3. **Complete Characterization of the Properness Condition**: A complete set of necessary and sufficient conditions for the properness of constraints on fiber bundles is provided (Theorem 7), comprising four conditions: global bounded uncertainty, quadratic lower bound of the potential function, Lipschitz continuity, and asymptotic control.
4. **Engineering Rationality of Operating Region Settings**: The localized treatment to avoid singular configurations ensures both theoretical rigor and compatibility with the physical constraints of real-world systems.
5. **Preservation of Toda Lattice Integrability**: The critical condition $\epsilon_{\text{crit}}$ for maintaining the integrable structure under observational perturbations is explicitly given.

## Limitations & Future Work

1. **Deterministic Uncertainty Assumption**: The current framework only handles deterministic observational uncertainty and cannot directly deal with true stochastic noise such as Wiener processes, necessitating the development of stochastic differential geometry.
2. **Localization Limitation**: Theoretical rigor is primarily guaranteed within the operating region that avoids singular configurations; extending this to a global theory requires addressing complex topological and geometric issues.
3. **Strict Smoothness Requirements**: The theory requires the system to possess sufficient smoothness ($C^3$), which may be overly restrictive in certain practical applications.
4. **Lack of Numerical Experiments**: Actual computational results of the geometry-preserving algorithm and quantitative comparisons with traditional methods are not provided.
5. **Indirect Application Verification**: The three application cases originate from the companion work [25], and this paper does not independently conduct end-to-end verification.

## Related Work & Insights

- **Dirac Constraint Theory** [5,6]: This work generalizes it from constrained submanifolds to a fiber bundle setting.
- **Marsden-Weinstein Symplectic Reduction** [13]: This work extends it to a version under observational constraints.
- **Arnold-Liouville Theorem** [3]: Extended to complete integrability on fiber bundles.
- **Control Barrier Functions** [2]: This work provides a fiber bundle geometric foundation for them.
- **[25] "Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures"**: The companion ICML work for application validation.
- **Yang-Mills Theory** [24]: The observation-adaptive connection has a similar geometric status to gauge field connections.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Pioneeringly geometricizes observational uncertainty into a fiber bundle structure, presenting a brand-new theoretical paradigm.
- Experimental Thoroughness: ⭐⭐ — A purely theoretical work; application validations are from the companion paper, lacking independent numerical experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and detailed proofs, though extremely long (2000+ lines) with some redundancies.
- Value: ⭐⭐⭐⭐ — Establishes a unified geometric bridge between constrained dynamics and safety control, demonstrating outstanding theoretical depth but pending practical impact validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[CVPR 2025\] Foundations of the Theory of Performance-Based Ranking](../../CVPR2025/others/foundations_of_the_theory_of_performance-based_ranking.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2025\] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](../../NeurIPS2025/others/semi-infinite_nonconvex_constrained_min-max_optimization.md)

</div>

<!-- RELATED:END -->
