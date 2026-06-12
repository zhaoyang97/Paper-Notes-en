---
title: >-
  [Paper Note] Interpretability and Generalization Bounds for Learning Spatial Physics
description: >-
  [ICML 2026][Optimization][Neural Operators] This paper uses numerical analysis tools to prove that for linear PDEs (such as 1D Poisson)…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Neural Operators"
  - "Green's Functions"
  - "Generalization Bounds"
  - "Mechanistic Interpretability"
  - "PINN"
date: 2026-05-08
content_hash: 539203abf8f24a67
---

# Interpretability and Generalization Bounds for Learning Spatial Physics

**Conference**: ICML 2026  
**arXiv**: [2506.15199](https://arxiv.org/abs/2506.15199)  
**Code**: To be confirmed  
**Area**: Scientific Computing / SciML / Generalization Theory  
**Keywords**: Neural Operators, Green's Functions, Generalization Bounds, Mechanistic Interpretability, PINN  

## TL;DR
This paper uses numerical analysis tools to prove that for linear PDEs (such as 1D Poisson), the learned solution operator $\mathbf{W}$ only converges to the projection of the true operator $\mathbf{A}$ onto the training function space $\mathbf{A}\mathbf{U}\mathbf{U}^\top$. Consequently, the **function space itself**—rather than the data volume or grid fineness—determines OOD generalization. It proposes a mechanistic interpretability technique that reveals the "Green's function structure" by applying the weight matrix to one-hot inputs, and uses a 25×25 cross-dataset cross-evaluation to identify failure modes in 8 types of SciML models (including PINN, DeepONet, FNO, and PI-DeepONet).

## Background & Motivation

**Background**: Applying ML to scientific computing follows two main paradigms: white-box (SINDy, symbolic regression, providing closed-form formulas) and black-box (DeepONet, Fourier Neural Operator, PINN, which are flexible but uninterpretable). An intermediate category of "physics-aware" models (PINN, PI-DeepONet) injects priors by including the PDE residual in the loss. These models often achieve machine-precision MSE on their specific training distributions.

**Limitations of Prior Work**: Low training error does **not** equate to learning the correct physics. Existing work has sporadically observed that Neural ODEs overfit temporal sequences, PINNs fail under certain training strategies, and PINOs collapse across resolutions—but these are "empirical phenomena" lacking theoretical characterization of **why they fail**. Furthermore, there is no unified framework to simultaneously evaluate parameter learning, operator learning, and physics-aware models.

**Key Challenge**: Traditional ML intuition suggests that "more data and higher expressivity" should monotonically improve generalization. However, classical numerical analysis states that approximation error is determined by the **discretization order and function space**. These two perspectives clash in SciML. This paper aims to integrate a priori estimates from numerical analysis into ML to characterize the boundaries of this conflict.

**Goal**: (1) Provide rigorous convergence and generalization bounds for parameter fitting and linear operator learning on simple 1D Poisson equations; (2) include the "training function space" as a first-order variable in the analysis; (3) provide a weight-based mechanistic interpretability method, independent of loss, to intuitively judge whether a model has truly learned the underlying physics.

**Key Insight**: Starting from the Green's function $G(s, x)$ of the Poisson equation—which is the "ground truth" for the PDE solution operator—if a learned matrix $\mathbf{W}$ is approximating the discretization of $\mathbf{A} = \int G \psi$, then $\mathbf{W} \mathbf{e}_j$ should resemble the impulse response of the Green's function. This provides a unified handle for both theoretical analysis and visual diagnosis.

**Core Idea**: The training data is modeled as a stochastic process over a "sampled function space $\mathcal{F}(\mathrm{type}, p)$." It is proven that the solution $\mathbf{W}^*$ of a linear model under Gradient Descent (GD) is the result of orthogonally projecting the true operator onto the training space. This projection residual serves as the a priori generalization bound. Furthermore, $\mathbf{W} \mathbf{e}_j$ (or $\mathrm{Model}(\mathbf{e}_j)$ for nonlinear models) is used as a "Green's function extractor" for mechanistic interpretability checks.

## Method

### Overall Architecture
The study focuses on the 1D Poisson equation $-k \, d^2 u / dx^2 = f(x)$ on $[0,1]$ with homogeneous Dirichlet boundaries. The solution operator is represented by the Green's function $G(s, x)$. The authors construct 25 datasets—Polynomial / Sine / Cosine $\mathcal{F}(\mathrm{type}, p)$ ($p = 1..8$) plus a piecewise linear FEM dataset, each with 10,000 samples. They train 8 types of models and perform 25×25 cross-evaluation (row=training set, column=test set) to produce error matrix heatmaps.

The theoretical side focuses on two settings with analytical solutions:
- **Setting A** (Parameter Fitting + Known PDE Structure): Fix the finite difference stencil order $q$ and learn only the scalar $w \approx k$.
- **Setting B** (Black-box Linear Operator): Learn the entire matrix $\mathbf{u} = \mathbf{W} \mathbf{f}$.

The empirical side extends this framework to architectures without analytical solutions, such as deep linear models, MLP, DeepONet, FNO, PINN, and PI-DeepONet.

### Key Designs

1.  **A Priori Estimation for Finite Difference Parameter Learning (Theorem 3.1)**:
    - **Function**: Characterizes the upper bound of error when fitting parameter $w$ using known PDE structures and finite-difference stencils on polynomial training data.
    - **Mechanism**: Solves MSE minimization with a $q$-th order stencil (e.g., three-point FD-2, $d^2 u / dx^2 \approx (u_{i-1} - 2u_i + u_{i+1}) / \Delta x^2 + \mathcal{O}(\Delta x^q)$). Proof shows: when training polynomial degree $p < q$, $w=k$ is exact; when $p \geq q$, an irreducible bias exists: $|w - k| / |k| = \mu_q \Delta x^q + \sum_{m=q+1}^p \mu_m \Delta x^m \approx \mu_q \Delta x^q$, where $\mu_m$ are stencil truncation error coefficients.
    - **Design Motivation**: To debunk the ML intuition that "more data is always better." Adding higher-order polynomial data adds another $\mathcal{O}(\Delta x^m)$ bias term because high-order polynomials allow the stencil's truncation error to be "absorbed" into $w$. This is a hard ceiling caused by the discretization order, independent of the infinite data assumption, and is also observed in PINN inverse problems.

2.  **Subspace Projection Theorem for Linear Operators (Theorem 3.2)**:
    - **Function**: Precisely writes the convergence point of a black-box linear model $\mathbf{W}$ under GD, explicitly incorporating the "training function space" into the generalization bound.
    - **Mechanism**: Assume training forcing $\mathbf{f}^{(n)} = \mathbf{B} \mathbf{c}^{(n)}$ is sampled from a space (where $\mathbf{B}$ is like a Vandermonde matrix with rank $p+1$). Under zero-mean initialization $\mathbf{W}^0$, the GD limit is $\mathbf{W}^* = \mathbf{A} \mathbf{U} \mathbf{U}^\top + \mathbf{W}^0 (\mathbf{I} - \mathbf{U} \mathbf{U}^\top)$, where $\mathbf{U}$ is the left orthogonal basis of $\mathbf{B}$ ($N_{\mathrm{grid}} \times (p+1)$).
    - **Design Motivation**: This is a **highly pessimistic** result—it is independent of data volume or grid fineness and depends only on the rank of the training space. The true operator is learned if and only if $\dim \mathcal{F}_{\mathrm{train}} \geq \mathrm{rank}(\mathbf{A})$; otherwise, $\mathbf{W}$ remains a projection of $\mathbf{A}$, with residual initial noise in orthogonal directions. This explains the counter-intuitive phenomenon where training error reaches machine precision but the matrix differs significantly from the true $\mathbf{A}$. The predicted MSE lower bound is $\|\mathbf{A} - \mathbf{A} \mathbf{U} \mathbf{U}^\top\|_F^2$.

3.  **Green's Function Mechanistic Interpretability Probe**:
    - **Function**: Judges whether a model has learned the PDE solution operator by feeding it one-hot inputs $\mathbf{f} = \mathbf{e}_j$ instead of checking the loss.
    - **Mechanism**: $\mathbf{A}_{ij} \leftrightarrow \mathrm{Model}(\mathbf{f} = \mathbf{e}_j)_i$ holds for any model mapping forcing to solution. For linear models, one examines the columns of the weight matrix; for nonlinear operators like MLP/DeepONet/FNO, one visualizes the matrix "scanned" by 25 one-hot inputs. If learned correctly, columns look like Green's function impulse responses and the matrix shows a "piecewise linear" structure; otherwise, it is noise. One can also invert the matrix $\hat{\mathbf{L}} = \mathbf{W}^{-1}$ to see if the tri-diagonal local stencil is recovered.
    - **Design Motivation**: Traditional training/test MSE cannot distinguish between "overfitting to a specific function space" and "truly learning the operator," as both can yield low training loss. The Green's function probe provides a diagnostic signal **orthogonal to the loss**.

## Key Experimental Results

### Main Results (Key Findings from 25×25 OOD Cross-Evaluation)

| Model Family | Subspace Generalize? | Training MSE Magnitude | OOD Failure Mode |
| :--- | :--- | :--- | :--- |
| Linear Model $\mathbf{u} = \mathbf{W}\mathbf{f}$ | Yes (3 block lower-triangular, matches Thm 3.2) | $\sim 10^{-20}$ | Fails outside training distribution across function families |
| Deep Linear | Partial (Yes for sin/cos, No for poly) | Medium | Inconsistent outside subspace |
| MLP | No, strongly diagonal | Medium | Little generalization, pure overfitting |
| FD Parameter Fitting | — | Increases as $p$ increases | Higher training degree $p$ leads to higher $w$ bias (Thm 3.1) |
| PINN Inverse Problem | — | Same trend as FD | $w$ error increases as $p$ increases |
| DeepONet | Block lower-triangular, diagonal bias | Lower | Slight overfitting on training distribution |
| FNO | Similar to DeepONet | Unstable; fails on some function classes | — |
| PI-DeepONet | Block lower-triangular, high baseline | $\sim 10^{-6}$ | PDE loss raises the error floor but doesn't remove subspace limits |

> Key Contrast: Training errors differ by 14 orders of magnitude ($10^{-20}$ vs $10^{-6}$), yet all models jump to the same $10^{-2}$ magnitude when crossing function subspaces OOD—proving that **whether the test space falls within the training subspace is more decisive than model complexity**.

### Key Findings
- **Counter-intuitive Source of Irreducible Bias**: In parameter learning, adding polynomial orders higher than the stencil order $q$ increases error (Thm 3.1), contradicting the ML experience that "richer data is better," because high-order polynomials allow truncation errors to "masquerade" as tunable parameters.
- **Physics-Aware $\neq$ Physics-Correct**: PI-DeepONet includes the PDE in the loss but does not show better out-of-subspace generalization; it merely raises the training error floor from machine precision to $10^{-6}$. Prior terms cannot overcome function space limitations.
- **FEM Data as a Golden Ticket**: All black-box models exhibit the broadest cross-subspace generalization when trained on piecewise linear FEM data, as the piecewise linear basis is sufficiently dense to span the subspaces of other function families. This provides practical guidance for SciML data collection.

## Highlights & Insights
- **Elevating "Function Space" to a First-Order Variable**: While traditional OOD analysis focuses on distribution shifts and noise, this paper explicitly incorporates the "function subspace spanned by the training distribution" into generalization bounds, providing a clean closed-form $\mathbf{A} \mathbf{U} \mathbf{U}^\top$ that is empirically verifiable.
- **The Three-Block Lower-Triangular Heatmap as a New SciML Benchmark**: This 25×25 cross-evaluation is far more informative than a single test MSE.
- **Transferability of the Green's Function Probe**: The "one-hot input, observe response" methodology is applicable to any model mapping spatial functions to spatial functions.
- **Constructive Nature of Pessimistic Conclusions**: The theory provides constructive guidance for data collection—selecting piecewise linear or broad function families (like high-degree dense polynomials) for training sets can essentially expand the learnable subspace.

## Limitations & Future Work
- **Limited to Linear PDEs**: Theorem 3.2 strictly applies to linear operators; a subspace projection bound for nonlinear PDEs (e.g., Navier-Stokes) is yet to be established.
- **Limited to 1D / Tensor-Product 2D**: For high-dimensional non-separable PDEs, Green's function discretization may suffer from the curse of dimensionality.
- **No Analysis of Regularization/Optimizers**: Experiments primarily use zero-initialization and GD; the impact of Adam, weight decay, or dropout on subspace projection behavior remains unanswered.
- **Future Directions**: Generalizing "subspace rank $\geq$ operator rank" to local linearization conditions for nonlinear operators and researching how active learning can select minimal training sets to cover target operators.

## Related Work & Insights
- **vs. Boullé et al. (Green-function ML)**: They use Green's functions to construct operators or prove lower bounds for data requirements. This paper uses the Green's function as both a **theoretical anchor** and a **visual diagnostic probe**.
- **vs. Krishnapriyan's PINN Failure Series**: Those works show PINN failures under specific strategies; this paper provides a more structural explanation—PINN's PDE loss cannot expand the subspace, thus remains constrained by Thm 3.2.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Rigorously brings a priori estimates from numerical analysis into SciML generalization analysis with verifiable closed-form predictions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 model types × 25 datasets, including noise, 2D, and biharmonic extensions, plus comprehensive visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ — The contrast in Fig. 1 is highly impactful; theory and experiments are tightly integrated.
- Value: ⭐⭐⭐⭐⭐ — Provides direct guidance for SciML data collection and establishes the cross-evaluation protocol as a new benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[NeurIPS 2025\] Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines](../../NeurIPS2025/optimization/learning_at_the_speed_of_physics_equilibrium_propagation_on_oscillator_ising_mac.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](../../ICLR2026/optimization/generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)

</div>

<!-- RELATED:END -->
