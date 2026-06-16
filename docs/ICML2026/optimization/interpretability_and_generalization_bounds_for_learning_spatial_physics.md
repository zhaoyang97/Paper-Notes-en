---
title: >-
  [Paper Note] Interpretability and Generalization Bounds for Learning Spatial Physics
description: >-
  [ICML 2026][Optimization & Theory][PINN] The paper utilizes numerical analysis tools to prove that for linear PDEs (such as the 1D Poisson equation), the learned solution operator $\mathbf{W}$ converges only to the projection of the true operator $\mathbf{A}$ onto the training function space, $\mathbf{A}\mathbf{U}\mathbf{U}^\top$. Consequently, the **function
tags:
  - ICML 2026
  - Optimization & Theory
  - PINN
date: 2026-05-08
content_hash: 47fec88e1cf71393
---
# Interpretability and Generalization Bounds for Learning Spatial Physics

**Conference**: ICML 2026  
**arXiv**: [2506.15199](https://arxiv.org/abs/2506.15199)  
**Code**: To be confirmed  
**Area**: Scientific Computing / SciML / Generalization Theory  
**Keywords**: Neural Operator, Green's Function, Generalization Bound, Mechanistic Interpretability, PINN  

## TL;DR
The paper utilizes numerical analysis tools to prove that for linear PDEs (such as the 1D Poisson equation), the learned solution operator $\mathbf{W}$ converges only to the projection of the true operator $\mathbf{A}$ onto the training function space, $\mathbf{A}\mathbf{U}\mathbf{U}^\top$. Consequently, the **function space itself**—rather than data volume or grid resolution—determines out-of-distribution (OOD) generalization. The authors propose a mechanistic interpretability technique that visualizes whether the "Green's function structure" is learned by applying the weight matrix to one-hot vectors. Through 25×25 cross-dataset evaluation, they identify the failure modes of eight classes of SciML models, including PINN, DeepONet, FNO, and PI-DeepONet.

## Background & Motivation

**Background**: Applying ML to scientific computing primarily follows two routes: white-box (SINDy, symbolic regression, providing closed-form formulas) and black-box (DeepONet, Fourier Neural Operator, PINN, flexible but uninterpretable). An intermediate category, "physics-aware" models (PINN, PI-DeepONet), injects priors by incorporating PDE losses into training. These models often achieve machine-precision level MSE on their designated training distributions.

**Limitations of Prior Work**: Low training error does **not** imply that the correct physics have been learned. Prior works have sporadically observed that Neural ODEs overfit time series, PINNs fail under certain training strategies, and PINOs collapse during cross-resolution tasks. However, these are "empirical phenomena" lacking theoretical characterization to explain **why** they fail. Furthermore, there is no unified framework to simultaneously evaluate parameter learning, operator learning, and physics-aware models.

**Key Challenge**: Traditional ML intuition suggests that "more data and stronger expressivity" should monotonically improve generalization. In contrast, classical numerical analysis dictates that approximation error is determined by the **discretization order and function space**. These two perspectives clash in SciML. This paper aims to introduce numerical analysis *a priori* estimates into ML to delineate the boundaries of this conflict.

**Goal**: (1) Provide rigorous convergence and generalization bounds for parameter fitting and linear operator learning on the simplified 1D Poisson equation; (2) Incorporate "training function space" as a first-order variable in the analysis; (3) Develop a mechanistic interpretability method that is loss-independent and weight-based to intuitively judge if a model has truly learned the physics.

**Key Insight**: Starting from the Green's function $G(s, x)$ of the Poisson equation—which serves as the "standard answer" for the PDE solution operator—if the learned matrix $\mathbf{W}$ is actually approximating the discretization of $\mathbf{A} = \int G \psi$, then $\mathbf{W} \mathbf{e}_j$ should resemble the impulse response of the Green's function. This provides a unified handle for both theoretical analysis and visual diagnostics.

**Core Idea**: The training data is modeled as a stochastic process within a "sampled function space $\mathcal{F}(\mathrm{type}, p)$." The authors prove that the solution $\mathbf{W}^*$ of Gradient Descent (GD) on linear models is the result of orthogonally projecting the true operator onto the training space. This projection residual is used as an *a priori* generalization bound. Furthermore, $\mathbf{W} \mathbf{e}_j$ (or $\mathrm{Model}(\mathbf{e}_j)$ for non-linear models) is utilized as a "Green's function extractor" for mechanistic interpretability checks.

## Method

### Overall Architecture
The research focuses on the 1D Poisson equation $-k \, d^2 u / dx^2 = f(x)$ on $[0,1]$ with homogeneous Dirichlet boundaries. The corresponding solution operator can be expressed via the Green's function $G(s, x)$. The authors construct 25 datasets—polynomial, sine, and cosine $\mathcal{F}(\mathrm{type}, p)$ ($p = 1..8$) plus a FEM piecewise linear set—each consisting of 10,000 samples. They then train 8 classes of models and perform a 25×25 cross-evaluation (rows = training set, columns = test set), producing error matrix heatmaps.

The theoretical side focuses on two settings with analytical solutions:
- **Setting A** (Parameter fitting + Known PDE structure): Fix the finite difference stencil order $q$ and learn only the scalar $w \approx k$;
- **Setting B** (Black-box linear operator): Learn the entire matrix $\mathbf{u} = \mathbf{W} \mathbf{f}$.

The empirical side extends this framework to architectures without analytical solutions, such as deep linear models, MLP, DeepONet, FNO, PINN, and PI-DeepONet.

### Key Designs

**1. A Priori Estimation for Finite Difference Parameter Learning (Theorem 3.1): Debunking the "More Data is Better" Intuition**

In the analytical setting where the PDE structure is known and only a scalar $w \approx k$ is learned using a $q$-th order stencil (e.g., three-point FD-2), the paper proves that for training polynomial orders $p < q$, $w = k$ is exact. However, once $p \geq q$, an irreducible bias emerges:

$$\frac{|w-k|}{|k|} = \mu_q \Delta x^q + \sum_{m=q+1}^p \mu_m \Delta x^m \approx \mu_q \Delta x^q$$

where $\mu_m$ is the truncation error coefficient of the stencil. This contradicts ML intuition: adding data with orders higher than the stencil order introduces an additional $\mathcal{O}(\Delta x^m)$ bias. This occurs because high-order polynomials allow truncation errors to be "absorbed" into $w$. This is a hard ceiling caused by discretization order, independent of data volume, and the same trend is observed in PINN inverse problems.

**2. Subspace Projection Theorem for Linear Operators (Theorem 3.2): Encoding "Training Function Space" into Generalization Bounds**

For the black-box linear model $\mathbf{u}=\mathbf{W}\mathbf{f}$, where training forcing is sampled from $\mathbf{f}^{(n)}=\mathbf{B}\mathbf{c}^{(n)}$ ($\mathbf{B}$ is a Vandermonde-like matrix of rank $p+1$), the GD limit under zero-mean initialization is:

$$\mathbf{W}^* = \mathbf{A}\,\mathbf{U}\mathbf{U}^\top + \mathbf{W}^0(\mathbf{I}-\mathbf{U}\mathbf{U}^\top)$$

where $\mathbf{U}$ is the left orthogonal basis of $\mathbf{B}$. This result implies that the convergence point depends solely on the rank of the training space. The true operator $\mathbf{A}$ is learned if and only if $\dim\mathcal{F}_{\mathrm{train}}\geq\mathrm{rank}(\mathbf{A})$; otherwise, $\mathbf{W}$ remains a projection with residual initial noise in orthogonal directions. This explains why training error can reach machine precision while the matrix differs significantly from $\mathbf{A}$.

**3. Green's Function Mechanistic Interpretability Probe: Checking Response via One-Hot Inputs**

Training/testing MSE cannot distinguish between overfitting a function space and truly learning the operator. The authors propose a diagnostic: since $\mathbf{A}_{ij}\leftrightarrow\mathrm{Model}(\mathbf{f}=\mathbf{e}_j)_i$, feeding a one-hot input $\mathbf{f}=\mathbf{e}_j$ "scans" the model into a matrix. For linear models, one examines weight columns; for non-linear models like FNO or DeepONet, the response to 25 one-hot vectors is plotted. A correctly learned model will show columns resembling the Green's function impulse response (tent-like piecewise linear structures).

### Experimental Protocol
The authors introduce **function-space cross-evaluation**, training models on 25 datasets and testing them on the other 24, resulting in a 25×25 MSE heatmap. This redefines "OOD" from "distribution shift" to "function subspace shift."

## Key Experimental Results

### Main Results

| Model Family | Subspace Generalize? | Training MSE | OOD Failure Mode |
|---|---|---|---|
| Linear Model $\mathbf{u} = \mathbf{W}\mathbf{f}$ | Yes (Consistent with Thm 3.2) | $\sim 10^{-20}$ | Fails outside training distribution across function families |
| Deep Linear | Partial | Moderate | Inconsistent outside subspace |
| MLP | No (Strongly Diagonal) | Moderate | Minimal generalization, pure overfitting |
| FD Parameter Fitting | — | Increases with $p$ | Higher training order $p \implies$ higher bias in $w$ (Thm 3.1) |
| PINN inverse problem | — | Same trend as FD | $p$ ↑ $\implies$ $w$ error ↑ |
| DeepONet | Block Lower Triangular | Low | Slight overfitting on training distribution |
| FNO | Similar to DeepONet | Unstable | Fails to train on some function classes |
| PI-DeepONet | Block Lower Triangular | $\sim 10^{-6}$ | PDE loss raises error floor but doesn't resolve subspace limits |

> **Key Contrast**: Despite training errors differing by $10^{14}$ orders of magnitude ($10^{-20}$ vs $10^{-6}$), OOD errors across subspaces for all models converge to the same $10^{-2}$ magnitude—proving that **whether the test space falls within the training subspace is more decisive than model complexity**.

### Robustness & Extension

| Setting | Phenomenon | Conclusion |
|---|---|---|
| Measurement Noise | Floor rises from $10^{-20}$ to $10^{-9}$ | Noise raises the floor but does not blur subspace boundaries |
| 1D Biharmonic | Same block lower triangular structure | Thm 3.2 is not limited to Poisson |
| 2D Poisson (Tensor Product) | Sierpiński-triangle-like heatmap | Subspace generalization holds independently per dimension |
| FEM Data for Linear Model | Learns full $\mathbf{A}$; inverse retrieves FD stencil | Dense data space is sufficient for learning the true operator |

### Key Findings
- **Source of Irreducible Bias**: In parameter learning, adding polynomial orders higher than the stencil order $q$ increases error (Thm 3.1), contradicting "more data is better" ML wisdom.
- **Physics-Aware ≠ Physics-Correct**: PI-DeepONet incorporates PDE loss but fails to offer better out-of-subspace generalization; it merely raises the training error floor.
- **FEM Data as the "Golden Passport"**: Models trained on FEM piecewise linear data show the broadest generalization, as these bases are dense enough to span other function subspaces.

## Highlights & Insights
- **Function Space as a First-Order Variable**: Unlike traditional OOD analysis focusing on distribution or labels, this work explicitly incorporates the "function subspace spanned by the training distribution" into generalization bounds.
- **A New Hallmarking Benchmark**: The 25×25 cross-evaluation heatmap provides significantly more information than a single MSE value; it is suggested as a standard for future SciML research.
- **Green's Function Probe**: The "one-hot input, visualize response" methodology is applicable to any model mapping spatial functions to spatial functions, offering a "mechanistic" bar for PDE structure identification.

## Limitations & Future Work
- **Constraint to Linear PDEs**: Theorem 3.2 strictly applies to linear operators. Subspace projection bounds for non-linear PDEs (e.g., Navier-Stokes) remain unexplored.
- **Dimensionality Limits**: Storing and analyzing the discretization of $\mathbf{A}$ for non-separable high-dimensional PDEs is computationally expensive.
- **Optimization Impact**: The study uses GD with zero initialization; the effects of Adam, weight decay, or dropout on subspace projection behavior were not analyzed.

## Related Work & Insights
- **vs. Boullé et al.**: While they use Green's functions for data demand lower bounds, this work uses them as both a **theoretical anchor** and a **visual probe**.
- **vs. PINN Failure Literature**: This work provides a structured explanation for documented failures—PDE losses cannot expand the reachable subspace.
- **Inspiration**: The "function-space cross-evaluation" could be adapted to evaluate LLM in-context learning by treating prompt "task spaces" as $\mathcal{F}_{\mathrm{train}}$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[NeurIPS 2025\] Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines](../../NeurIPS2025/optimization/learning_at_the_speed_of_physics_equilibrium_propagation_on_oscillator_ising_mac.md)
- [\[ICML 2025\] A Generalization Result for Convergence in Learning-to-Optimize](../../ICML2025/optimization/a_generalization_result_for_convergence_in_learning-to-optimize.md)
- [\[AAAI 2026\] A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](../../AAAI2026/optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)

</div>

<!-- RELATED:END -->
