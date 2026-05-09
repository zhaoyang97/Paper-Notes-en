---
title: >-
  [Paper Note] Neural Green's Functions
description: >-
  [NeurIPS 2025][3D Vision][Green's functions] This paper proposes Neural Green's Functions, a learnable linear PDE solution operator based on eigendecomposition: pointwise geometric features are extracted from the domain geometry to predict the eigendecomposition of the Green's function, enabling one-time training to solve for arbitrary source functions and boundary conditions via numerical integration. On mechanical part thermal analysis, the method reduces error by 13.9% over the state-of-the-art neural operator while running 350× faster than numerical solvers.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Green's functions
  - neural operators
  - PDE solving
  - eigendecomposition
  - domain generalization
date: 2026-05-08
content_hash: 1ef9d4dec47a5652
---

# Neural Green's Functions

**Conference**: NeurIPS 2025
**arXiv**: [2511.01924](https://arxiv.org/abs/2511.01924)
**Code**: None
**Area**: 3D Vision
**Keywords**: Green's functions, neural operators, PDE solving, eigendecomposition, domain generalization

## TL;DR
This paper proposes Neural Green's Functions, a learnable linear PDE solution operator based on eigendecomposition: pointwise geometric features are extracted from the domain geometry to predict the eigendecomposition of the Green's function, enabling one-time training to solve for arbitrary source functions and boundary conditions via numerical integration. On mechanical part thermal analysis, the method reduces error by 13.9% over the state-of-the-art neural operator while running 350× faster than numerical solvers.

## Background & Motivation

**Background**: Learning-based PDE solvers (PINNs, FNO, GNO, Transolver, etc.) have significantly improved efficiency, yet face fundamental difficulties in **simultaneously generalizing** across varying domain geometries, source functions, and boundary conditions.

**Limitations of Prior Work**:
- PINNs require independent training for each problem instance; any change necessitates retraining.
- Neural operators (FNO, Transolver) **couple** the input mesh with sampled function values — generalization degrades when the function changes.
- Prior methods for learning Green's functions (Boullé et al., Teng et al.) are restricted to a single domain and handle only simple geometries.

**Key Challenge**: Existing methods take function values as network inputs — changing the function demands more training samples.

**Key Insight**: The Green's function $G_D(x,y)$ **depends only on the domain geometry $D$**, independent of the source function $f$ and boundary condition $h$. Once $G_D$ is learned, the solution is obtained by integration: $u(x) = \int G_D(x,y) f(y)\, dy + \text{boundary terms}$.

**Core Idea**: A neural network operating on point clouds predicts the eigendecomposition of the Green's function $G \approx \Phi \Lambda^{-1} \Phi^T$ **from domain geometry alone**, along with the mass matrix $M$ required for integration — making the design **function-agnostic by construction**.

## Method

### Overall Architecture
**Input**: Volumetric point cloud representation of the problem domain $D$. The network extracts pointwise geometric features → predicts eigenvectors $\Phi$ and eigenvalues $\Lambda$ of the Green's function, along with the mass matrix $M$ and the boundary-interior selection matrix. For any given $f$ and $h$, the solution is obtained directly via matrix operations:
$$\mathbf{u} = \mathbf{K}^T \{\mathbf{G}(\mathbf{K}\mathbf{M}\mathbf{f} - \mathbf{K}\mathbf{L}\mathbf{S}^T\mathbf{h})\} + \mathbf{S}^T\mathbf{h}$$

### Key Designs

1. **Eigendecomposition-Based Solution Operator**:

   - **Function**: Factorizes the Green's function matrix $\mathbf{G} = (\mathbf{KLK}^T)^{-1}$ as $\mathbf{G} = \mathbf{\Phi} \mathbf{\Lambda}^{-1} \mathbf{\Phi}^T$.
   - **Mechanism**: The network predicts eigenvectors $\Phi \in \mathbb{R}^{N_{int} \times K}$ and eigenvalues $\Lambda$, where $K$ is far smaller than the number of interior vertices $N_{int}$ — a low-rank approximation.
   - **Design Motivation**: Directly predicting the $N_{int} \times N_{int}$ Green's function matrix is infeasible (too large); the low-rank eigendecomposition compresses parameters from $O(N^2)$ to $O(NK)$.

2. **Purely Geometric Feature Extraction**:

   - **Function**: Extracts pointwise features from the input point cloud without using any function value information.
   - **Mechanism**: A Transolver backbone processes the volumetric point cloud, with cross-attention interactions with latent tokens.
   - **Design Motivation**: **Forcing the network to observe only geometry** is the key design choice that aligns with the mathematical property that Green's functions depend solely on domain geometry.

3. **Joint Prediction of Integration Components**:

   - **Mass matrix $M$**: Converts source function values into integration weights for irregular meshes.
   - **Selection matrices $S$/$K$**: Distinguish boundary from interior vertices.
   - All components are predicted from geometry — mesh density and shape vary across domains, so $M$ and $S$ vary accordingly.

### Loss & Training
- Supervised loss: MSE between the predicted solution $u$ and the ground truth from a numerical solver.
- Training uses the MCB dataset of mechanical part 3D geometries across 5 categories.
- Linear PDEs considered: Poisson equation and Biharmonic equation.

## Key Experimental Results

### Main Results — Steady-State Thermal Analysis (MCB Dataset, 5 Part Categories)

| Method | Mean Relative Error ↓ | Generalizes to New Functions | Generalizes to New Domains | Speed |
|---|---|---|---|---|
| PINN | Requires retraining | ✗ | ✗ | Slow |
| FNO | Moderate | Poor | Poor | Fast |
| GINO | Moderate | Poor | Moderate | Fast |
| Transolver | Baseline | Poor | Moderate | Fast |
| **Neural Green's** | **−13.9%** | **✓ (by design)** | **✓** | **Fast (350× vs. FEM)** |

### Ablation Study

| Configuration | Error | Notes |
|---|---|---|
| Full framework | Lowest | Eigendecomposition + mass matrix |
| Replace with direct regression of $u$ | +15% | Loses function-agnosticism |
| Remove mass matrix prediction | +8% | Inaccurate integration weights |
| Reduce eigenvector dimension $K$ | +5–20% | Information loss when $K$ is too small |

### Validation on Poisson/Biharmonic Equations in Simple Domains

| PDE | Method | Error on New Source Functions | Error on New Boundary Conditions |
|---|---|---|---|
| Poisson | FNO | Moderate | Moderate |
| Poisson | **Neural Green's** | **Low** | **Low** |
| Biharmonic | Transolver | Moderate | Moderate |
| Biharmonic | **Neural Green's** | **Low** | **Low** |

### Key Findings
- **Function-agnosticism is thoroughly validated experimentally**: performance degradation on source functions and boundary conditions unseen during training is minimal (<5%), whereas baseline methods degrade by 20–50%.
- The low-rank approximation of the eigendecomposition ($K = 50$–$100$) is sufficient to capture the majority of information.
- The method is 350× faster than numerical solvers (which require meshing and linear system solving), eliminating the meshing bottleneck.
- Cross-category generalization (training on one shape category, testing on another) also holds, indicating that the network learns a general geometry-to-solution-operator mapping.

## Highlights & Insights
- **"Learning the fundamental solution rather than a specific solution"** reflects a profound problem decomposition — the Green's function is the inverse operator of the PDE; knowing it yields **all** solutions. This is fundamentally more powerful than learning a solution for a particular $f$/$h$.
- **Design aligned with mathematical structure**: Green's functions depend only on geometry → the network observes only geometry. The linear structure of eigendecomposition → the network predicts eigenvectors and eigenvalues. This methodology of letting mathematical properties guide network design generalizes naturally to other operator learning problems.
- The effectiveness of the low-rank approximation suggests that **the Green's functions of most PDEs are intrinsically low-rank** — consistent with physical intuition that the influence of distant sources decays rapidly.
- A 13.9% error reduction over Transolver (same backbone) is attributable purely to **framework design**.

## Limitations & Future Work
- **Applicable only to linear PDEs whose dense operators admit eigendecomposition** (e.g., Poisson, Biharmonic); nonlinear PDEs (e.g., Navier–Stokes) are not directly supported.
- The current formulation handles Dirichlet boundary conditions; Neumann or mixed boundary conditions require modifications to the boundary terms of the Green's function.
- Point cloud representations of 3D volumetric meshes and mass matrix prediction may suffer accuracy issues at very high resolutions.
- Only steady-state PDEs are validated; time-dependent PDEs (e.g., the heat equation) require extension to the temporal-domain version of the Green's function.

## Related Work & Insights
- **vs. Boullé et al. (2021)**: Their approach learns the Green's function for a **fixed domain** (requiring retraining per domain), whereas this work learns a geometry-to-Green's-function mapping **across domains**.
- **vs. Transolver (ICML'24)**: Same backbone but different framework — Transolver takes function values as input, while this work observes only geometry. The performance gap stems entirely from framework design.
- **vs. DeepONet**: DeepONet learns general operators but encodes source functions through a branch network — generalizing to new functions requires sufficient training samples.
- **Inspiration**: For other physical problems (e.g., electromagnetics, elasticity), similar eigendecomposition-based methods may enable function-agnostic solution operators.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ A triple innovation combining Green's functions, eigendecomposition, and geometric priors; the first Green's function learning framework that generalizes across both domains and functions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on both simple domains and complex 3D mechanical parts, with comparisons against multiple state-of-the-art methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations with clear exposition of the alignment between network design and physical/mathematical structure.
- **Value**: ⭐⭐⭐⭐⭐ Transformative potential for AI-driven scientific computing — train once, solve for anything.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](learning_neural_exposure_fields_for_view_synthesis.md)
- [\[NeurIPS 2025\] Hybrid Physical-Neural Simulator for Fast Cosmological Hydrodynamics](hybrid_physical-neural_simulator_for_fast_cosmological_hydrodynamics.md)
- [\[NeurIPS 2025\] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)
- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[NeurIPS 2025\] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)

<!-- RELATED:END -->
