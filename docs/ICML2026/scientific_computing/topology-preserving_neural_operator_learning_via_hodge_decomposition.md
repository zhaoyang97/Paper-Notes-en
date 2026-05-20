---
title: >-
  [Paper Note] Topology-Preserving Neural Operator Learning via Hodge Decomposition
description: >-
  [ICML 2026][Scientific Computing][Hodge Decomposition] This paper proposes the Hodge Spectral Duality (HSD) neural operator, which decomposes the solution operator of manifold PDEs via Hodge orthogonal decomposition into…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Hodge Decomposition"
  - "Neural Operator"
  - "Discrete Exterior Calculus"
  - "Manifold PDE"
  - "Spectral Method"
date: 2026-05-08
content_hash: 2f38614236eb9dda
---

# Topology-Preserving Neural Operator Learning via Hodge Decomposition

**Conference**: ICML 2026  
**arXiv**: [2605.13834](https://arxiv.org/abs/2605.13834)  
**Code**: https://github.com/ContinuumCoder/Hodge-Spectral-Duality (available)  
**Area**: 3D Vision / Neural Operators / Manifold PDE  
**Keywords**: Hodge Decomposition, Neural Operator, Discrete Exterior Calculus, Manifold PDE, Spectral Method

## TL;DR
This paper proposes the Hodge Spectral Duality (HSD) neural operator, which decomposes the solution operator of manifold PDEs via Hodge orthogonal decomposition into a "low-frequency topological component (spectral basis) + high-frequency geometric component (FNO auxiliary grid)" dual-branch structure. A commutator correction term couples the two, enabling both high accuracy and conservation law fidelity on complex meshes.

## Background & Motivation

**Background**: Neural operators (FNO, DeepONet, PINN) have achieved resolution-independent operator mappings on Euclidean regular grids. However, real-world engineering PDEs often occur on Riemannian manifolds with boundaries, curvature, and nontrivial topology (e.g., automotive aerodynamic surfaces, geophysical spheres, biological organ geometries). Such physical fields are naturally differential forms: 0-form (scalar potential), 1-form (flux), 2-form (vorticity/flux), whose evolution is constrained by both de Rham cohomology and Riemannian metric structures.

**Limitations of Prior Work**: Existing methods each have structural shortcomings. GNN-based local message passing suffers from over-smoothing/over-squashing and cannot capture global topology determined by the Hodge Laplacian null space. FNO-type extrinsic spectral methods are FFT-friendly on Euclidean grids but only "softly" constrain cohomology and boundary topology; harmonic components can only be preserved via loss penalties. Intrinsic geometric methods (geodesic/tangent bundle conv) preserve manifold structure but require geometry-adaptive kernels, leading to computational explosion on large meshes and poor high-frequency detail capture.

**Key Challenge**: Topological constraints (from the Hodge Laplacian $\Delta_k = d\delta + \delta d$ kernel, corresponding to conservation laws and global circulation) and geometric constraints (from metric $g$ and material tensor $\kappa$, governing high-frequency boundary layers and anisotropic diffusion) arise from fundamentally different algebraic structures. A single representation space struggles to efficiently approximate both, leading to an "efficiency-expressiveness-topological fidelity" trade-off.

**Goal**: Construct a neural operator framework that is both resolution-independent and structure-preserving, capable of learning PDE solution operators on general Riemannian manifolds, while hard-constraining topological invariants (Betti numbers $b_k$, circulation, flux).

**Key Insight**: The authors observe that Hodge orthogonal decomposition uniquely splits any $k$-form into gradient, curl, and harmonic orthogonal subspaces. This orthogonality enables "additive approximation" at the operator level—decomposing $\mathcal{G}_\theta^k$ into a low-frequency topological branch $\mathcal{G}_{\mathrm{base},\theta}^k$ and a high-frequency geometric branch $\mathcal{G}_{\mathrm{fiber},\theta}^k$, which reside in orthogonal subspaces and do not interfere.

**Core Idea**: Use Discrete Exterior Calculus (DEC) to perform an offline spectral decomposition of the Hodge Laplacian's low-frequency eigenvectors as the "Base space" to learn topology-driven low-frequency responses; use FNO on an auxiliary Euclidean grid to learn metric-driven high-frequency residuals, and constrain them to the orthogonal complement of the Base via projection $(\mathbf{I}-\Pi_{\mathrm{base}})$. Finally, a Lie-Trotter operator splitting commutator correction $\mathcal{C}_\theta$ compensates for the splitting residual between the two non-commuting operators.

## Method

### Overall Architecture
HSD expresses each operator learning layer as an additive structure: "Base branch + Fiber branch + commutator correction":

$\boldsymbol{\omega}_k^{(\ell+1)} = \mathcal{G}_{\mathrm{base}}^{(\ell)}(\boldsymbol{\omega}_k^{(\ell)}) + (\mathbf{I}-\Pi_{\mathrm{base}}^k)\bigl[\mathcal{G}_{\mathrm{fiber}}^{(\ell)}(\boldsymbol{\omega}_k^{(\ell)}) + \mathcal{C}_\theta^{(\ell)}(\mathbf{z}^{(\ell)})\bigr]$

The input is a discrete $k$-form on a simplicial complex $K$ (0-form on nodes, 1-form on edges, 2-form on faces). Offline, a sparse eigendecomposition $\mathbf{L}_k \mathbf{\Psi}_k = \mathbf{\Psi}_k \mathbf{\Lambda}_k$ is performed, truncating to $m_k$ lowest-frequency eigenvectors to form the spectral basis $\mathbf{\Phi}_k$. Online, the field is projected onto the Base space (spectral coefficients) and, via a lift operator $\iota$, onto an auxiliary Euclidean grid for FFT. The outputs are summed after inverse projection and orthogonal complement constraint.

### Key Designs

1. **Base Branch: Hodge Spectral Coefficient Learning + Hard Constraint on Harmonic Modes**:

    - **Function**: Performs physically consistent nonlinear mapping in the low-dimensional spectral subspace $\mathcal{V}_{\mathrm{base}}^k = \mathrm{span}(\mathbf{\Phi}_k)$, while hard-constraining the corresponding topological invariants.
    - **Mechanism**: Each layer projects the field to the spectral domain via the Hodge inner product $\mathbf{c}_k^{(\ell)} = \mathbf{\Phi}_k^\top *_k \boldsymbol{\omega}_k^{(\ell)} \in \mathbb{R}^{m_k}$; uses precomputed spectral differential matrices $\mathcal{M}_d^{(k)}, \mathcal{M}_\delta^{(k)}$ to form $(k\pm1)$-order derivative features $\mathbf{q}_k^{(\ell)}$; passes through a gated MLP (gMLP) to learn quadratic nonlinear couplings in the spectral domain (e.g., convection term $\mathbf{u}\cdot\nabla\mathbf{u}$), with formula $\tilde{\mathbf{c}}_k = \mathbf{W}_{\mathrm{out}}(\phi(\mathbf{W}_g \mathbf{q}) \odot (\mathbf{W}_c \mathbf{q})) + \mathbf{c}_k$. Crucially, after updating, a diagonal projection $\mathbf{P}_H^k$ directly overwrites the zero-eigenvalue (harmonic) modes with their original values, thus **hard-constraining** the cohomology class and global flux at each layer.
    - **Design Motivation**: FNO's soft penalty for conservation laws is inaccurate, while GNNs cannot capture global circulation. The number of harmonic modes equals the Betti number $b_k$ (a few to dozens), so hard-constraining them does not affect high-frequency learnability.

2. **Fiber Branch: High-Frequency Residuals on Auxiliary Euclidean Grid + Orthogonal Projection**:

    - **Function**: Learns metric-driven high-frequency geometric corrections (anisotropic diffusion, boundary layers) in the Base orthogonal complement $\mathcal{V}_{\mathrm{fiber}}^k$, without disturbing global topology.
    - **Mechanism**: Uses a lift operator $\iota$ (Whitney form + KDE) to map discrete cochains to tensor fields on an auxiliary Euclidean grid $\Omega_{\mathrm{aux}}$, applies standard FNO spectral convolution $\mathcal{F}^{-1}\mathbf{R}_{\mathrm{loc}}\mathcal{F}$, then uses the adjoint pullback operator $\mathcal{R}$ to map back to $C^k(K)$. Finally, multiplies by $(\mathbf{I}-\Pi_{\mathrm{base}}^k)$ to zero out all low-frequency components, ensuring Fiber only corrects high frequencies.
    - **Design Motivation**: Compared to intrinsic manifold convolution, Euclidean grid FFT has complexity $\mathcal{O}(N\log N)$ and naturally expresses anisotropy. The orthogonal complement constraint prevents Fiber from altering conserved components.

3. **Commutator Correction $\mathcal{C}_\theta$: Compensating Operator Splitting Residuals**:

    - **Function**: Compensates for the systematic Lie-Trotter splitting error arising from the non-commutativity $[\mathcal{A}_{\mathrm{Topo}}^k, \mathcal{A}_{\mathrm{Geom}}^k] \neq 0$ of the topological and geometric operators.
    - **Mechanism**: Concatenates geometric lift features $\iota(\boldsymbol{\omega}_k)$ and spectral domain first derivatives $(\mathbf{c}_k, \mathcal{M}_d \mathbf{c}_k, \mathcal{M}_\delta \mathbf{c}_k)$ into interaction features $\mathbf{z}^{(\ell)}$, passes through a lightweight MLP to output the correction, and constrains it to the Fiber subspace via $(\mathbf{I}-\Pi_{\mathrm{base}})$. Gating is initialized near zero, starting from a decoupled state and gradually learning the coupling.
    - **Design Motivation**: Lie-Trotter splitting has $O(\Delta t^2)$ residuals at second order; simply adding the two branches cannot express the $AB-BA$ second-order cross term. A learnable small correction can eliminate this systematic bias.

### Loss & Training
End-to-end MSE supervision (no PDE residual loss). Offline, a single $\mathbf{L}_k$ sparse eigendecomposition is performed (about 57s on a $\sim 20k$-element tetrahedral mesh). Online training cost is $\mathcal{O}(Nk)$ for spectral projection + $\mathcal{O}(N\log N)$ for FFT. Overall training time is significantly lower than message-passing MGN-type methods.

## Key Experimental Results

### Main Results
On DrivAerNet++ automotive aerodynamics, multi-connected region magnetostatics, and toroidal advection-diffusion tasks, all methods use 207k–310k parameters. Results:

| Task | Model | MSE↓ | Spectral Fidelity↑ | $\beta_0$ Score↑ | IoU↑ |
|------|------|------|-------------------|------------------|------|
| Ext. Aero | FNO-3D | $1.80\times 10^{-2}$ | 0.7110 | 0.5584 | 0.3010 |
| Ext. Aero | HSD | $\mathbf{1.08\times 10^{-2}}$ | **0.8423** | **0.6112** | **0.3398** |
| Magnetostatics | DeepONet | $2.89\times 10^{-4}$ | 0.9468 | 0.7877 | 0.7834 |
| Magnetostatics | HSD | $\mathbf{1.84\times 10^{-4}}$ | **0.9492** | **0.8176** | **0.8110** |
| Toroidal | FNO-3D | $5.55\times 10^{-4}$ | 0.9079 | 0.6721 | 0.7515 |
| Toroidal | HSD | $\mathbf{3.56\times 10^{-4}}$ | **0.9115** | **0.7829** | **0.8131** |

HSD reduces MSE by 36%–40% compared to the next best method across all tasks; improvements in topological fidelity ($\beta_0$ score, measuring connected component consistency) are especially significant.

### Ablation Study

| Configuration | Magnetostatics | Ext. Aero | Toroidal |
|---------------|---------------|-----------|----------|
| HSD Full | $1.84\times 10^{-4}$ | $1.08\times 10^{-2}$ | $3.56\times 10^{-4}$ |
| w/o $\mathcal{C}_\theta$ (no commutator) | $2.18\times 10^{-4}$ (+18%) | $1.17\times 10^{-2}$ (+8%) | $3.79\times 10^{-4}$ (+6%) |
| w/o $\Pi_{\mathrm{base}}$ (no orthogonal projection) | $2.20\times 10^{-4}$ (+20%) | $1.45\times 10^{-2}$ (+34%) | $3.72\times 10^{-4}$ (+4%) |
| Direct FNO-3D Baseline | $8.51\times 10^{-4}$ (+363%) | $1.80\times 10^{-2}$ (+67%) | $5.55\times 10^{-4}$ (+56%) |

Experiments with spectral mode $k=64\to 256$ show MSE decreases monotonically but with diminishing returns (Magnetostatics drops only another 14%), confirming the dual design philosophy that "Base needs only a few low-frequency modes + Fiber supplements high-frequency".

### Key Findings
- Orthogonal projection $\Pi_{\mathrm{base}}$ has the greatest impact on geometrically complex domains (Ext. Aero); removing it increases MSE by 34%, as FNO spectral convolution introduces non-physical low-frequency noise contaminating conserved components.
- Commutator correction $\mathcal{C}_\theta$ has the greatest impact on multi-connected domains (Magnetostatics); removing it increases error by 18%, confirming that explicit compensation for topological-geometric operator non-commutativity is necessary.
- On the external aerodynamics task, increasing the inference mesh from 3,000 to 7,000 nodes, HSD's error fluctuates only 30%, while all baselines' errors increase by at least 10×, indicating HSD truly learns the PDE operator rather than a mesh-specific mapping.
- In terms of training efficiency, HSD is 56× faster than MGN on Ext. Aero (33s vs 1865s), and uses only 5% of MGN's time on Magnetostatics, demonstrating the engineering feasibility of offline spectral decomposition + online low-dimensional updates.

## Highlights & Insights
- **Operator-level Additive Decomposition**: Hodge orthogonality provides a strong algebraic structure of "strict orthogonality between topological and geometric modes", making the dual-branch not just an engineering trick but a mathematically justified operator splitting. This "subspace division of labor" can be transferred to other geometry-prior learning tasks (e.g., manifold diffusion models, geometric GANs).
- **Hard Constraint vs Soft Penalty**: Directly overwriting harmonic mode updates via diagonal projection is a representative approach for "hard-constraining topological invariants". Compared to PINN-type conservation laws in the loss, this structural hard constraint requires no weight tuning and has mathematical guarantees.
- **Offline-Online Decoupling**: Placing expensive geometric encoding (sparse eigendecomposition) offline, with only low-dimensional spectral updates and FFT online, is deployment-friendly—when reusing the same geometry for multiple inferences, the amortized cost is nearly zero.
- The commutator correction term $\mathcal{C}_\theta$ is an underrated design: many "dual-branch" works assume the two branches are additive, but when underlying operators are non-commutative, this assumption necessarily misses second-order terms. Explicitly modeling $[A,B]$ is valuable for multimodal fusion and hybrid architectures.

## Limitations & Future Work
- Relies on a one-time offline Hodge Laplacian sparse eigendecomposition, supporting only fixed geometry or isometric/small perturbation scenarios; time-varying geometry (requiring remeshing at each step) is currently unsupported. The authors suggest using Functional Maps or iso-spectral deformation for low-cost spectral basis transfer.
- The current framework targets Eulerian simulations on 3D and lower-dimensional manifolds; Lagrangian particle tracking and strong discontinuities (shocks, phase boundaries) are not yet applicable, as the auxiliary Euclidean grid mollification is low-pass and cannot represent discontinuities.
- All experiments are on medium-scale meshes with 3,000 nodes; scalability, eigendecomposition stability, and memory usage on industrial-scale million-node meshes remain unverified.
- The commutator correction's approximation (lightweight MLP) lacks theoretical characterization for higher-order splitting residuals and could be improved—higher-order splitting schemes (Strang splitting, Yoshida-4) or more structured learnable correction operators may be considered.

## Related Work & Insights
- **vs FNO/Geo-FNO**: FNO performs spectral convolution on Euclidean grids; Geo-FNO uses diffeomorphism to map geometry back to Euclidean space. HSD does not attempt to "straighten" geometry, but directly defines operator learning in the Hodge spectral domain, fundamentally preserving manifold cohomology structure.
- **vs DeepONet**: DeepONet uses branch-trunk inner products for global fitting, achieving good MSE on some tasks (e.g., Magnetostatics scalar fields), but with low topological fidelity (IoU 0.78 vs HSD 0.81). HSD's hard constraint on harmonic modes systematically outperforms such global fitting.
- **vs GNN/MGN**: Message passing inevitably suffers from over-smoothing/over-squashing, making it hard to capture global flux. HSD outsources global structure to the spectral basis, letting local FNO handle high frequencies, thus avoiding GNN's shortcomings by design.
- **vs Topological Deep Learning (SCN/SCNN)**: Existing TDL work mainly addresses classification/interpolation, lacking continuous operator mapping. HSD is the first to combine DEC with neural operators, advancing TDL into the operator learning domain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Hodge Spectral Duality is original both in mathematical structure (orthogonal decomposition) and engineering implementation (dual-branch + commutator), truly bridging algebraic topology and neural operators.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks cover closed surfaces, multi-connected domains, and nonzero genus tori, with comprehensive metrics (accuracy + conservation + topology) and complete ablation; however, mesh scale is small and not validated on large-scale industrial CFD.
- Writing Quality: ⭐⭐⭐⭐ Mathematical notation is rigorous, motivation is clear, and appendix derivations are complete; some formula-dense sections require DEC background to fully understand.
- Value: ⭐⭐⭐⭐⭐ Provides a new neural operator baseline that is "fast, accurate, and conservative" for scientific computing/CAE, with direct application potential in industrial CFD/electromagnetic simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)
- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](../../ICLR2026/3d_vision/learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)
- [\[ICML 2026\] Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](../../ICLR2026/3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](../../CVPR2026/3d_vision/reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)

</div>

<!-- RELATED:END -->
