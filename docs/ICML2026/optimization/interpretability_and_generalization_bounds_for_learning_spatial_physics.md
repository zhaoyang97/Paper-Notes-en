---
title: >-
  [Paper Note] Interpretability and Generalization Bounds for Learning Spatial Physics
description: >-
  [ICML 2026][Optimization][Neural Operators] This paper employs numerical analysis tools to prove that a learned solution operator $\mathbf{W}$ for linear PDEs (e.g., 1D Poisson) converges only to the projection of the true operator $\mathbf{A}$ onto the training function space, denoted as $\mathbf{A}\mathbf{U}\mathbf{U}^\top$. Consequently, the **function space itself**—rather than data volume or grid fineness—determines OOD generalization. The authors propose a mechanistic i…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Neural Operators"
  - "Green's Functions"
  - "Generalization Bounds"
  - "Mechanistic Interpretability"
  - "PINN"
date: 2026-05-08
content_hash: 2d5ef428a5f00364
---

# Interpretability and Generalization Bounds for Learning Spatial Physics

**Conference**: ICML 2026  
**arXiv**: [2506.15199](https://arxiv.org/abs/2506.15199)  
**Code**: To be confirmed  
**Area**: Scientific Computing / SciML / Generalization Theory  
**Keywords**: Neural Operators, Green's Functions, Generalization Bounds, Mechanistic Interpretability, PINN  

## TL;DR
This paper employs numerical analysis tools to prove that a learned solution operator $\mathbf{W}$ for linear PDEs (e.g., 1D Poisson) converges only to the projection of the true operator $\mathbf{A}$ onto the training function space, denoted as $\mathbf{A}\mathbf{U}\mathbf{U}^\top$. Consequently, the **function space itself**—rather than data volume or grid fineness—determines OOD generalization. The authors propose a mechanistic interpretability technique that visualizes whether the Green's function structure is learned by applying the weight matrix to one-hot vectors. Using a 25×25 cross-dataset evaluation, they identify the failure modes of eight classes of SciML models, including PINNs, DeepONets, and FNOs.

## Background & Motivation

**Background**: Applying ML to scientific computing typically follows two paths: white-box (SINDy, symbolic regression providing closed-form formulas) and black-box (DeepONet, Fourier Neural Operator, PINN, which are flexible but lack interpretability). A middle category of "physics-aware" models (PINN, PI-DeepONet) injects priors by incorporating PDE losses into training. These models often achieve machine-precision MSE on their specific training distributions.

**Limitations of Prior Work**: Low training error **does not** equate to learning the correct physics. Existing work has sporadically observed that Neural ODEs overfit time series, PINNs fail under certain training strategies, and PINOs collapse across resolutions. However, these are "empirical phenomena" lacking theoretical characterization of **why** they fail. There is no unified framework to evaluate parameter learning, operator learning, and physics-aware models simultaneously.

**Key Challenge**: Traditional ML intuition suggests that "more data and stronger expressivity" should monotonically improve generalization. In contrast, classic numerical analysis states that approximation error is determined by **discretization order and function spaces**. These two views clash in SciML. This paper aims to introduce *a priori* estimates from numerical analysis into ML to characterize the boundaries of this conflict.

**Goal**: (1) Provide rigorous convergence and generalization bounds for parameter fitting and linear operator learning on the 1D Poisson equation. (2) Incorporate the "training function space" as a first-order variable in the analysis. (3) Provide a loss-independent, weight-only mechanistic interpretability method to visually determine if a model has truly learned the physics.

**Key Insight**: Start from the Green's function $G(s, x)$ of the Poisson equation—the "standard answer" for the PDE solution operator. If a learned matrix $\mathbf{W}$ is approximating the discretization of $\mathbf{A} = \int G \psi$, then $\mathbf{W} \mathbf{e}_j$ should resemble the impulse response of the Green's function. This provides a unified handle for both theoretical analysis and visual diagnostics.

**Core Idea**: Model training data as a stochastic process within a "sampling function space $\mathcal{F}(\mathrm{type}, p)$." Prove that the solution $\mathbf{W}^*$ of a linear model via Gradient Descent (GD) is the result of an orthogonal projection of the true operator onto the training space. Use this projection residual as an *a priori* generalization bound. Employ $\mathbf{W} \mathbf{e}_j$ (or $\mathrm{Model}(\mathbf{e}_j)$ for nonlinear models) as a "Green's function extractor" for mechanistic interpretability.

## Method

### Overall Architecture
The study focuses on the 1D Poisson equation $-k \, d^2 u / dx^2 = f(x)$ on $[0,1]$ with homogeneous Dirichlet boundaries, where the solution operator is represented by the Green's function $G(s, x)$. The authors construct 25 datasets—polynomial/sine/cosine $\mathcal{F}(\mathrm{type}, p)$ ($p = 1..8$) plus a FEM piecewise linear basis, each with 10,000 samples. They train 8 classes of models and perform a 25×25 cross-evaluation (rows = training set, columns = test set) to produce error matrix heatmaps.

The theoretical side focuses on two settings with analytical solutions:
- **Setting A** (Parameter fitting + known PDE structure): Fixed finite difference stencil order $q$, learning only the scalar $w \approx k$;
- **Setting B** (Black-box linear operator): Learning the entire matrix $\mathbf{u} = \mathbf{W} \mathbf{f}$.

The empirical side extends this framework to architectures without analytical solutions, such as deep linear networks, MLPs, DeepONet, FNO, PINN, and PI-DeepONet.

### Key Designs

**1. *A priori* estimation for finite difference parameter learning (Theorem 3.1): Debunking the "more data is better" intuition**

In Setting A, where only the scalar $w \approx k$ is learned using a $q$-th order stencil (e.g., FD-2 where $d^2 u/dx^2 \approx (u_{i-1}-2u_i+u_{i+1})/\Delta x^2 + \mathcal{O}(\Delta x^q)$), the paper proves that while $w=k$ is exact when the training polynomial degree $p<q$, an irreducible bias emerges as soon as $p \geq q$:

$$\frac{|w-k|}{|k|} = \mu_q \Delta x^q + \sum_{m=q+1}^p \mu_m \Delta x^m \approx \mu_q \Delta x^q$$

where $\mu_m$ are truncation error coefficients. This contradicts ML intuition: every additional order of polynomial data higher than the stencil order adds an $\mathcal{O}(\Delta x^m)$ bias, as high-order polynomials allow truncation errors to be "absorbed" into $w$. This is a hard ceiling caused by discretization order, independent of data volume, and is also observed in PINN inverse problems.

**2. Subspace Projection Theorem for linear operators (Theorem 3.2): Incorporating "Function Space" into generalization bounds**

In Setting B (black-box model $\mathbf{u}=\mathbf{W}\mathbf{f}$), assume training forcing is sampled from $\mathbf{f}^{(n)}=\mathbf{B}\mathbf{c}^{(n)}$ (where $\mathbf{B}$ is a Vandermonde-like matrix of rank $p+1$). Under zero-mean initialization, the GD limit is:

$$\mathbf{W}^* = \mathbf{A}\,\mathbf{U}\mathbf{U}^\top + \mathbf{W}^0(\mathbf{I}-\mathbf{U}\mathbf{U}^\top)$$

where $\mathbf{U}$ is the left orthogonal basis of $\mathbf{B}$. This result is notably pessimistic: the convergence point is independent of data volume or mesh fineness and depends solely on the rank of the training space. The true operator is learned if and only if $\dim\mathcal{F}_{\mathrm{train}}\geq\mathrm{rank}(\mathbf{A})$; otherwise, $\mathbf{W}$ remains a projection of $\mathbf{A}$ onto the training space with residual initial noise in orthogonal directions. This explains why training error can reach machine precision while the matrix differs significantly from the true $\mathbf{A}$.

**3. Green's Function Mechanistic Probe: One-hot response analysis**

Training/test MSE cannot distinguish between "overfitting a function space" and "truly learning an operator." Both scenarios yield low loss on training sets. The authors propose a diagnostic orthogonal to the loss: since any model mapping forcing to solution satisfies $\mathbf{A}_{ij}\leftrightarrow\mathrm{Model}(\mathbf{f}=\mathbf{e}_j)_i$, feeding a one-hot vector $\mathbf{f}=\mathbf{e}_j$ "scans" the model into a matrix. For linear models, one examines weight columns; for nonlinear ones (MLP/DeepONet/FNO), the response to 25 one-hot seeds is plotted. If learned correctly, the columns resemble Green's function impulse responses (tent-like piecewise linear structures). One can further invert $\hat{\mathbf{L}}=\mathbf{W}^{-1}$ to see if the tridiagonal local stencil is recovered.

### Experimental Thoroughness
The authors introduce a **function-space cross-evaluation** protocol: training a model on each of the 25 datasets and testing it on the other 24 to create a 25×25 MSE heatmap. Variations arise from both mesh $\Delta x$ and function class $\mathcal{F}(\mathrm{type}, p)$. This constitutes a new SciML benchmark where OOD is strictly defined as switching function subspaces.

## Key Experimental Results

### Main Results (Findings from 25×25 OOD cross-evaluation)

| Model Family | Subspace Generalize? | Training MSE | OOD Failure Mode |
|---|---|---|---|
| Linear $\mathbf{u} = \mathbf{W}\mathbf{f}$ | Yes (Lower-triagonal block, per Thm 3.2) | $\sim 10^{-20}$ | Fails outside training distribution (e.g., poly→sin) |
| Deep Linear | Partial (on sin/cos, not poly) | Moderate | Inconsistent outside subspace |
| MLP | No (Strongly diagonal) | Moderate | Almost no generalization, pure overfitting |
| FD Parameter Fitting | — | Increases with $p$ | Higher training degree $p$ leads to biased $w$ (Thm 3.1) |
| PINN inverse problem | — | Same trend as FD | Higher degree $p$ → Higher $w$ error |
| DeepONet | Block-low-triangular | Low | Slight overfitting on training distribution |
| FNO | Similar to DeepONet | Unstable | Fails to converge on some function classes |
| PI-DeepONet | Block-low-triangular | $\sim 10^{-6}$ | PDE loss raises the error floor but doesn't fix subspace limits |

> Key Contrast: While training errors vary by factor of $10^{14}$ ($10^{-20}$ vs $10^{-6}$), all models jump to the same $10^{-2}$ magnitude during OOD subspace shifts—proving that **whether the test space falls within the training subspace is more decisive than model complexity**.

### Robustness & Extensions

| Setting | Observation | Conclusion |
|---|---|---|
| Measurement Noise | Floor rises from $10^{-20}$ to $10^{-9}$ | Noise raises the error floor but preserves the subspace structure. |
| 1D Biharmonic | Same block-low-triangular structure | Thm 3.2 is not limited to Poisson equations. |
| 2D Poisson (Tensor Product) | Sierpiński-triangle pattern | Subspace generalization holds independently for each spatial dimension. |
| FEM Data Training | Learns full $\mathbf{A}$, recovers tridiagonal stencil | Density in data space is a sufficient condition for learning the true operator. |

### Key Findings
- **Counter-intuitive Source of Irreducible Bias**: In parameter learning, adding polynomial orders higher than the stencil order $q$ increases error (Thm 3.1), contradicting "more data is better" ML wisdom, as high-order polynomials allow truncation errors to masquerade as adjustable parameters.
- **Physics-Aware $\neq$ Physics-Correct**: PI-DeepONet incorporates PDE loss but lacks superior OOD subspace generalization; it merely raises the training floor from machine precision to $10^{-6}$.
- **FEM Data as a Golden Ticket**: Models trained on FEM piecewise linear data show the broadest cross-subspace generalization because the basis is dense enough to span other function families. This provides practical guidance for SciML data collection.

## Highlights & Insights
- **Elevating "Function Space" to a First-Order Variable**: Traditional OOD analysis focuses on distribution shifts or noise; this paper explicitly writes the "function subspace spanned by the training distribution" into generalization bounds, providing a clean closed-form $\mathbf{A} \mathbf{U} \mathbf{U}^\top$ verifiable by experiment.
- **The 25×25 Heatmap as a New Benchmark**: This evaluation is far more informative than a single MSE. Future SciML papers should adopt this visualization.
- **Transferability of Green's Function Probes**: The "one-hot response" concept applies to any space-to-space mapping model, including temporal operators or 3D solvers.
- **Constructive Pessimism**: The theory identifies hard ceilings but also provides instructions: selecting piecewise linear or wide, high-order function families for training sets essentially expands the learnable subspace.

## Limitations & Future Work
- **Limited to Linear PDEs**: Theorem 3.2 strictly applies to linear operators; bounds for nonlinear PDEs (e.g., Navier-Stokes) remain empirical observations.
- **Dimensionality**: Green's function discretizations in high dimensions may lead to memory bottlenecks for storing $\mathbf{A}$.
- **Lack of Regularization/Optimizer Analysis**: All experiments use zero-initialization and GD. The impact of Adam, weight decay, or dropout on subspace projection behavior is not yet addressed.
- **Future Directions**: Generalizing "subspace rank $\geq$ operator rank" to local linearization conditions for nonlinear operators and researching how active learning can select minimal training sets to cover target operators.

## Related Work & Insights
- **vs. Boullé et al. (Green-function ML)**: While they use Green's functions for constructing operators or proving data lower bounds, this paper uses them as both a **theoretical anchor** and a **visual probe**.
- **vs. Krishnapriyan et al. (PINN failures)**: This paper provides a structured explanation for PINN failures—PDE loss cannot expand the reachable subspace, thus models remain bound by Thm 3.2.
- **vs. mechanistic interpretability (Nanda et al.)**: Similar to identifying algorithms in Transformers, this paper translates "viewing weights → finding known algorithms" to SciML, where Green's functions serve as the "known algorithm."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Rigorously introduces numerical analysis *a priori* estimates into ML operator generalization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of 8 model types, 25 datasets, and multiple PDE extensions.
- Writing Quality: ⭐⭐⭐⭐⭐ — The contrast in Fig. 1 between training precision and operator accuracy is highly impactful.
- Value: ⭐⭐⭐⭐⭐ — Provides direct guidance for data collection and establishes a new benchmark protocol.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[NeurIPS 2025\] Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines](../../NeurIPS2025/optimization/learning_at_the_speed_of_physics_equilibrium_propagation_on_oscillator_ising_mac.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[ICML 2025\] A Generalization Result for Convergence in Learning-to-Optimize](../../ICML2025/optimization/a_generalization_result_for_convergence_in_learning-to-optimize.md)

</div>

<!-- RELATED:END -->
