---
title: >-
  [Paper Note] Topology-Preserving Neural Operator Learning via Hodge Decomposition
description: >-
  [ICML 2026][Physics & Scientific Computing][Hodge decomposition] This paper proposes the Hodge Spectral Duality (HSD) neural operator, which decomposes the solution operator of manifold PDEs into a dual-branch structure:…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Hodge decomposition"
  - "Neural operator"
  - "Discrete Exterior Calculus"
  - "Manifold PDE"
  - "Spectral methods"
date: 2026-05-08
content_hash: c66052e80a0889f7
---

# Topology-Preserving Neural Operator Learning via Hodge Decomposition

**Conference**: ICML 2026  
**arXiv**: [2605.13834](https://arxiv.org/abs/2605.13834)  
**Code**: https://github.com/ContinuumCoder/Hodge-Spectral-Duality (Available)  
**Area**: 3D Vision / Neural Operator / Manifold PDE  
**Keywords**: Hodge decomposition, Neural operator, Discrete Exterior Calculus, Manifold PDE, Spectral methods

## TL;DR
This paper proposes the Hodge Spectral Duality (HSD) neural operator, which decomposes the solution operator of manifold PDEs into a dual-branch structure: "low-frequency topological components (spectral basis) + high-frequency geometric components (FNO auxiliary grid)" based on Hodge orthogonal decomposition. A commutator correction term is applied to couple the two, achieving both high precision and conservation law fidelity on complex meshes.

## Background & Motivation

**Background**: Neural operators (FNO, DeepONet, PINN) have demonstrated the ability to learn resolution-independent solution mappings on Euclidean regular grids. However, practical engineering PDEs often occur on Riemannian manifolds with boundaries, curvature, and non-trivial topology (e.g., car aerodynamic surfaces, geophysical spheres, biological geometries). These physical fields are naturally differential forms: 0-forms (scalar potential), 1-forms (flux), 2-forms (vorticity/flux), whose evolution is constrained by de Rham cohomology structures and Riemannian metrics.

**Limitations of Prior Work**: Existing methods possess structural deficiencies. GNN-based local message passing suffers from over-smoothing/over-squashing and fails to capture global topology determined by the Hodge Laplacian null space. FNO-based extrinsic spectral methods are efficient for FFT on Euclidean grids but provide only "soft constraints" for cohomology and boundary topology, where harmonic components must be preserved via loss penalties. Intrinsic geometric methods (geodesic/tangent bundle conv) preserve manifold structures but require geometric adaptation, leading to high computational costs and inability to handle high-frequency details.

**Key Challenge**: Topological constraints (from the kernel space of the Hodge Laplacian $\Delta_k=d\delta+\delta d$, corresponding to conservation laws and global circulation) and geometric constraints (from the metric $g$ and material tensor $\kappa$, dominating high-frequency boundary layers and anisotropic diffusion) originate from two distinct algebraic structures. A single representation space struggles to efficiently approximate both components simultaneously, resulting in an "efficiency-expressivity-topology fidelity" trade-off.

**Goal**: Construct a resolution-independent and structure-preserving neural operator framework capable of learning PDE solution operators on general Riemannian manifolds while imposing hard constraints on topological invariants (Betti numbers $b_k$, circulation, flux).

**Key Insight**: The authors observe that Hodge orthogonal decomposition uniquely splits any $k$-form into gradient-type + curl-type + harmonic-type **orthogonal** subspaces. This orthogonality implies that additive approximation can be performed at the operator level—splitting $\mathcal{G}_\theta^k$ into a low-frequency topological branch $\mathcal{G}_{\mathrm{base},\theta}^k$ and a high-frequency geometric branch $\mathcal{G}_{\mathrm{fiber},\theta}^k$, which operate in orthogonal subspaces without interference.

**Core Idea**: Use Discrete Exterior Calculus (DEC) to perform offline eigen-decomposition of the Hodge Laplacian to establish a "Base space" for learning topology-driven low-frequency responses. Use FNO on an auxiliary Euclidean grid to learn metric-driven high-frequency residuals, which are forced into the orthogonal complement of the Base via a projection $(\mathbf{I}-\Pi_{\mathrm{base}})$. Finally, use a commutator correction term $\mathcal{C}_\theta$ derived from Lie-Trotter operator splitting to compensate for the splitting residual between the two non-commutative operators.

## Method

### Overall Architecture
HSD formulates each operator layer as an additive structure consisting of a Base branch, a Fiber branch, and a commutator correction:

$\boldsymbol{\omega}_k^{(\ell+1)}=\mathcal{G}_{\mathrm{base}}^{(\ell)}(\boldsymbol{\omega}_k^{(\ell)})+(\mathbf{I}-\Pi_{\mathrm{base}}^k)\bigl[\mathcal{G}_{\mathrm{fiber}}^{(\ell)}(\boldsymbol{\omega}_k^{(\ell)})+\mathcal{C}_\theta^{(\ell)}(\mathbf{z}^{(\ell)})\bigr]$

The input consists of discrete $k$-forms on a simplicial complex $K$ (0-forms on nodes, 1-forms on edges, 2-forms on faces). During the offline phase, a sparse eigen-decomposition $\mathbf{L}_k \mathbf{\Psi}_k = \mathbf{\Psi}_k \mathbf{\Lambda}_k$ is performed, and $m_k$ lowest-frequency eigenvectors are truncated to form the spectral basis $\mathbf{\Phi}_k$. During the online phase, the fields are projected onto the Base space (spectral coefficients) and lifted to an auxiliary Euclidean grid via a lift operator $\iota$ for FFT processing. The output is refined through back-projection and orthogonal complement constraints.

### Key Designs

1.  **Base Branch: Hodge Spectral Learning + Hard Topological Constraints**:
    *   **Function**: Learn physically consistent nonlinear mappings in the low-dimensional spectral subspace $\mathcal{V}_{\mathrm{base}}^k=\mathrm{span}(\mathbf{\Phi}_k)$ and hard-preserve corresponding topological invariants.
    *   **Mechanism**: The field is first projected to the spectral domain $\mathbf{c}_k^{(\ell)}=\mathbf{\Phi}_k^\top *_k \boldsymbol{\omega}_k^{(\ell)}\in\mathbb{R}^{m_k}$ using the Hodge inner product. Precomputed spectral derivative matrices $\mathcal{M}_d^{(k)},\mathcal{M}_\delta^{(k)}$ are used to construct $(k\pm1)$-order derivative features $\mathbf{q}_k^{(\ell)}$. A gated MLP (gMLP) then learns spectral nonlinear coupling (e.g., convection terms $\mathbf{u}\cdot\nabla\mathbf{u}$), formulated as $\tilde{\mathbf{c}}_k=\mathbf{W}_{\mathrm{out}}(\phi(\mathbf{W}_g \mathbf{q})\odot(\mathbf{W}_c \mathbf{q}))+\mathbf{c}_k$. Crucially, after updating, a diagonal projection $\mathbf{P}_H^k$ is used to **overwrite the zero-eigenvalue (harmonic) modes with their original values**, thereby **hard-preserving** cohomology classes and global fluxes layer-by-layer.
    *   **Design Motivation**: Soft penalties in FNO are insufficient for strict conservation laws, while GNNs fail to capture global circulation. The number of harmonic modes equals the Betti number $b_k$, which is limited (from several to dozens), allowing them to be hard-constrained without sacrificing high-frequency learnability.

2.  **Fiber Branch: High-frequency Residuals on Auxiliary Euclidean Grids + Orthogonal Projection**:
    *   **Function**: Learn metric-driven high-frequency geometric corrections (anisotropic diffusion, boundary layers) in the Base orthogonal complement $\mathcal{V}_{\mathrm{fiber}}^k$ without disrupting global topology.
    *   **Mechanism**: A lift operator $\iota$ based on Whitney forms and KDE lifts discrete cochains to tensor fields on an auxiliary Euclidean grid $\Omega_{\mathrm{aux}}$. Standard FNO spectral convolutions $\mathcal{F}^{-1}\mathbf{R}_{\mathrm{loc}}\mathcal{F}$ are applied, followed by a pullback $\mathcal{R}$ back to $C^k(K)$. Finally, multiplication by $(\mathbf{I}-\Pi_{\mathrm{base}}^k)$ filters out all low-frequency components, ensuring the Fiber branch only modifies high frequencies.
    *   **Design Motivation**: Compared to intrinsic manifold convolutions, Euclidean FFT has a complexity of $\mathcal{O}(N\log N)$ and inherent anisotropic representational power. The orthogonal complement constraint prevents the Fiber branch from perturbing conserved components.

3.  **Commutator Correction $\mathcal{C}_\theta$: Compensating Operator Splitting Residuals**:
    *   **Function**: Compensate for systematic Lie-Trotter splitting errors arising from the non-commutativity of topological operators $\mathcal{A}_{\mathrm{Topo}}^k$ and geometric operators $\mathcal{A}_{\mathrm{Geom}}^k$, i.e., $[\mathcal{A}_{\mathrm{Topo}}^k,\mathcal{A}_{\mathrm{Geom}}^k]\neq 0$.
    *   **Mechanism**: Geometric lift features $\iota(\boldsymbol{\omega}_k)$ and spectral derivatives $(\mathbf{c}_k,\mathcal{M}_d \mathbf{c}_k,\mathcal{M}_\delta \mathbf{c}_k)$ are concatenated into interaction features $\mathbf{z}^{(\ell)}$. A lightweight MLP outputs a correction term, which is similarly constrained to the Fiber subspace via $(\mathbf{I}-\Pi_{\mathrm{base}})$. Gated initialization is set near zero to start from a decoupled state and gradually learn the coupling.
    *   **Design Motivation**: Lie-Trotter splitting introduces $O(\Delta t^2)$ residuals. Simple addition of two branches cannot represent the second-order cross-terms of $AB-BA$. A learnable correction term can eliminate this systematic bias.

### Loss & Training
End-to-end MSE supervision is used (without PDE residual loss). The offline sparse eigen-decomposition of $\mathbf{L}_k$ takes approximately 57s on a $20k$-element tetrahedral mesh. Online training costs involve $\mathcal{O}(Nk)$ spectral projection and $\mathcal{O}(N\log N)$ FFT, which are significantly lower than MGN-style message passing.

## Key Experimental Results

### Main Results
Evaluations were conducted on DrivAerNet++ car aerodynamics, magnetostatics in multi-connected regions, and toroidal advection-diffusion. All methods were standardized within a 207k–310k parameter range:

| Task | Model | MSE↓ | Spectral Fidelity↑ | $\beta_0$ Score↑ | IoU↑ |
|------|------|------|----------|----------------|------|
| Ext. Aero | FNO-3D | $1.80\times 10^{-2}$ | 0.7110 | 0.5584 | 0.3010 |
| Ext. Aero | HSD | $\mathbf{1.08\times 10^{-2}}$ | **0.8423** | **0.6112** | **0.3398** |
| Magnetostatics | DeepONet | $2.89\times 10^{-4}$ | 0.9468 | 0.7877 | 0.7834 |
| Magnetostatics | HSD | $\mathbf{1.84\times 10^{-4}}$ | **0.9492** | **0.8176** | **0.8110** |
| Toroidal | FNO-3D | $5.55\times 10^{-4}$ | 0.9079 | 0.6721 | 0.7515 |
| Toroidal | HSD | $\mathbf{3.56\times 10^{-4}}$ | **0.9115** | **0.7829** | **0.8131** |

HSD reduced MSE by 36%–40% compared to the second-best methods across all tasks, with significant improvements in topological fidelity ($\beta_0$ score).

### Ablation Study

| Configuration | Magnetostatics | Ext. Aero | Toroidal |
|------|----------------|-----------|----------|
| Full HSD | $1.84\times 10^{-4}$ | $1.08\times 10^{-2}$ | $3.56\times 10^{-4}$ |
| w/o $\mathcal{C}_\theta$ (no commutator) | $2.18\times 10^{-4}$ (+18%) | $1.17\times 10^{-2}$ (+8%) | $3.79\times 10^{-4}$ (+6%) |
| w/o $\Pi_{\mathrm{base}}$ (no projection) | $2.20\times 10^{-4}$ (+20%) | $1.45\times 10^{-2}$ (+34%) | $3.72\times 10^{-4}$ (+4%) |
| FNO-3D Baseline | $8.51\times 10^{-4}$ (+363%) | $1.80\times 10^{-2}$ (+67%) | $5.55\times 10^{-4}$ (+56%) |

Experiments with spectral modes $k=64\to 256$ showed monotonically decreasing MSE with diminishing returns, validating the "Base only needs low-frequency modes + Fiber compensates high-frequency" duality philosophy.

### Key Findings
*   Orthogonal projection $\Pi_{\mathrm{base}}$ has the greatest impact on geometrically complex domains (Ext. Aero); removing it increased MSE by 34%, as FNO spectral convolutions introduce non-physical low-frequency noise.
*   The commutator correction $\mathcal{C}_\theta$ is most critical for multi-connected domains (Magnetostatics); its removal led to an 18% error increase, confirming that non-commutativity must be explicitly compensated.
*   On aerodynamics tasks, increasing the inference mesh density from 3k to 7k nodes caused HSD errors to fluctuate by only 30%, whereas all baseline errors increased at least 10x, indicating that HSD learns the PDE operator rather than a grid-specific mapping.
*   Training efficiency: HSD was 56× faster than MGN in the Ext. Aero task (33s vs 1865s), proving the feasibility of offline spectral decomposition and online low-dimensional updates.

## Highlights & Insights
*   **Operator-level Additive Decomposition**: Hodge orthogonality provides a powerful algebraic structure where topological and geometric modes are strictly orthogonal. This makes the dual-branch approach a mathematically sound operator splitting rather than just an engineering trick.
*   **Hard Constraints vs. Soft Penalties**: Directly overwriting harmonic mode updates via diagonal projection represents a structural approach to preserving topological invariants. This is superior to PINN-style loss weighting as it eliminates hyperparameter tuning while providing mathematical guarantees.
*   **Offline-Online Decoupling**: Offloading expensive geometric encoding (sparse eigen-decomposition) to an offline phase and performing only low-dimensional spectral updates and FFT online is engineering-friendly, as costs are amortized over multiple inferences.
*   **Commutator Correction Term**: The $\mathcal{C}_\theta$ design addresses an often-overlooked issue; many dual-branch studies assume simple additivity, which fails for non-commutative operators. Explicitly modeling $[A,B]$ is a valuable concept for multimodal fusion and hybrid architectures.

## Limitations & Future Work
*   Reliance on one-time offline Hodge Laplacian sparse eigen-decomposition limits the model to fixed or near-isometric geometries. Support for time-varying geometries requires low-cost spectral basis transfer via Functional Maps or iso-spectral deformation.
*   The current framework is tailored for Eulerian-perspective simulations. It is not yet suitable for Lagrangian particle tracking or strong discontinuities (shocks, phase boundaries), as the auxiliary Euclidean grid mollification is low-pass and cannot represent jumps.
*   Experiments were conducted on medium-scale meshes (~3000 nodes). Scalability to million-node industrial meshes, stability of eigen-decomposition, and memory overhead require further validation.
*   The approximation of the commutator correction term (lightweight MLP) lacks theoretical bounds regarding higher-order splitting residuals. Improvement could involve higher-order splitting schemes (Strang splitting, Yoshida-4).

## Related Work & Insights
*   **vs. FNO/Geo-FNO**: FNO performs spectral convolutions on Euclidean grids, while Geo-FNO uses diffeomorphisms to map geometry back to Euclidean space. HSD avoids "straightening" the geometry and defines operator learning directly in the Hodge spectral domain, preserving manifold cohomology.
*   **vs. DeepONet**: DeepONet uses branch-trunk inner products for global fitting. While it achieves decent MSE on scalar fields, its topological fidelity (IoU 0.78 vs HSD 0.81) is lower.
*   **vs. GNN/MGN**: Message passing inherently suffers from over-smoothing, making it difficult to capture global flux. HSD delegates global structure to the spectral basis, leaving the local FNO to handle high frequencies, thereby circumnavigating GNN weaknesses.
*   **vs. Topological Deep Learning (SCN/SCNN)**: Existing TDL work primarily focuses on classification/interpolation. HSD is the first to combine DEC with neural operators, advancing TDL into the field of operator learning.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ HSD makes original contributions in both mathematical structure (orthogonal decomposition) and engineering implementation (dual-branch + commutator), bridging algebraic topology and neural operators.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Covered closed surfaces, multi-connected regions, and non-zero genus tori. Metrics are comprehensive, though validation on large-scale industrial CFD is missing.
*   **Writing Quality**: ⭐⭐⭐⭐ Rigorous notation and clear motivation. High formula density may require a background in DEC.
*   **Value**: ⭐⭐⭐⭐⭐ Provides a new baseline for scientific computing/CAE that is "fast, accurate, and conservative," with direct potential in industrial CFD and electromagnetic simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/physics/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/physics/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)
- [\[ICCV 2025\] JPEG Processing Neural Operator for Backward-Compatible Coding](../../ICCV2025/physics/jpeg_processing_neural_operator_for_backward-compatible_coding.md)
- [\[ICML 2026\] Generative Neural Operators Through Diffusion Last Layer](generative_neural_operators_through_diffusion_last_layer.md)

</div>

<!-- RELATED:END -->
