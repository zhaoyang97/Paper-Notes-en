---
title: >-
  [Paper Note] Separable Neural Networks: Approximation Theory, NTK Regime, and Preconditioned Gradient Descent
description: >-
  [ICLR 2026][learning_theory][Paper Note] The paper systematically establishes the theoretical foundation for Separable Neural Networks (SepNN): it proves that CP/TT/Tucker-type SepNNs possess universal approximation capabilities, derives their NTK regimes under infinite-width/infinite-rank and fixed-rank settings, and proposes SepPGD. This method utilizes low
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 4af43b64db6096f1
---
# Separable Neural Networks: Approximation Theory, NTK Regime, and Preconditioned Gradient Descent

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FlcMckO6x5](https://openreview.net/forum?id=FlcMckO6x5)  
**Paper**: OpenReview  
**Code**: https://github.com/YisiLuo/SepPGD  
**Area**: Learning Theory / NTK / Optimization  
**Keywords**: Separable Neural Networks, Universal Approximation, Neural Tangent Kernel, Spectral Bias, Preconditioned Gradient Descent  

## TL;DR
The paper systematically establishes the theoretical foundation for Separable Neural Networks (SepNN): it proves that CP/TT/Tucker-type SepNNs possess universal approximation capabilities, derives their NTK regimes under infinite-width/infinite-rank and fixed-rank settings, and proposes SepPGD. This method utilizes low-dimensional separable preconditioning matrices to adjust the NTK spectrum, accelerating training convergence in grid-coordinate tasks such as Implicit Neural Representations (INR) and Physics-Informed Neural Networks (PINN).

## Background & Motivation
**Background**: The fundamental concept of SepNN is to decompose a multivariable function $f(x_1,\dots,x_D)$ into a combination of several 1D factor functions. For example, the CP form is written as $f_\Theta(x)=\sum_{r=1}^R\prod_{d=1}^D (f_{\Theta_d}(x_d))_r$. This architecture is natural for INR, separable PINNs, and coordinate networks like NeRF or tensor factorization, as training points often lie on regular grids. It allows for calculating factor outputs for each dimension separately before combining them via tensor operations.

**Limitations of Prior Work**: In practice, the appeal of SepNN is clear: for a $D$-dimensional grid with $n$ points per dimension, a standard MLP must process $n^D$ coordinate points, whereas a SepNN only requires forward passes for each dimension's 1D coordinates. However, existing work remains largely at the level of structural design and applications. Fundamental questions regarding whether SepNN can represent any continuous multivariable function, whether it follows NTK dynamics similar to standard wide networks during training, and why it converges slowly on high-frequency details have not been systematically addressed.

**Key Challenge**: The efficiency of SepNN stems from its strong separable structure, which naturally raises concerns about its expressive power. Furthermore, the spectral bias of NTK suggests slow convergence in directions corresponding to small eigenvalues. In INR/PINN tasks, SepNN is often required to fit high-frequency components like image textures, surface details, and PDE solutions. Thus, SepNN must maintain the $O(nD)$ computational advantage while avoiding bottlenecks in representation and optimization.

**Goal**: The authors break the problem into three layers: first, representation capability, proving that CP, TT, and Tucker SepNNs can approximate any continuous multivariable function on compact sets; second, training dynamics, deriving the NTK form for CP SepNN and distinguishing between infinite-rank and fixed-rank regimes; third, algorithmic improvement, using the separable structure to design low-complexity preconditioned gradient descent to mitigate spectral bias.

**Key Insight**: The crucial observation is that while SepNN is not a standard MLP, its structure aligns perfectly with tensor decomposition and compositions of 1D MLPs. Representation theory can utilize the Stone-Weierstrass theorem to show that separable function families are dense in $C(X)$. Training theory can decompose the overall kernel into a weighted sum of products between individual factor NTKs and other factor outputs.

**Core Idea**: **Uniformly explain why SepNN can represent, how it trains, and how it accelerates using "density of separable function families + factor NTK decomposition + separable preconditioning."**

## Method
### Overall Architecture
Rather than proposing a single new network, the paper builds a complete theory-algorithm chain around SepNN. It first proves that the separable structure does not sacrifice universal approximation, then uses NTK to characterize error convergence directions under gradient descent, and finally proposes SepPGD based on the NTK spectral bias. The methodology follows a conceptual flow rather than a complex pipeline diagram.

The logic can be summarized as: Given CP/TT/Tucker SepNNs, the authors treat them as algebras formed by 1D continuous functions. After proving these algebras satisfy Stone-Weierstrass conditions, factor functions are replaced by 1D MLPs. For CP SepNN with $1/\sqrt{R}$ scaling, the expansion of $K_\Theta(x,x')=\langle\nabla_\Theta f_\Theta(x),\nabla_\Theta f_\Theta(x')\rangle$ shows the total NTK is a weighted sum of individual factor NTKs. Finally, SepPGD replaces the $n^D\times n^D$ preconditioning matrix with $n\times n$ factor preconditioning matrices $S_d$, feeding preconditioned residuals back to factor networks via mode products.

### Key Designs
**1. From Stone-Weierstrass to SepNN: Proving Density before MLP Fitting**

The primary doubt regarding SepNN is expressivity: does decomposing into 1D factors limit it to low-rank structures? The authors respond that as rank $R$ grows, this is not an issue for universal approximation. For CP SepNN, the function class is defined as $\mathcal{A}=\{g(x)=\sum_{r=1}^R\prod_{d=1}^D (g_d(x_d))_r\}$. 

The goal is to prove $\mathcal{A}$ is dense in $C(X)$. The authors verify the Stone-Weierstrass conditions: it contains constant functions, separates points via 1D functions, and remains closed under addition and multiplication. After approximating a multivariable function with a separable continuous function, factor networks replace 1D functions. Using error bounds, if each factor error is less than $\delta$, the total error is controlled by $DM^{D-1}\delta$, accumulating linearly with rank.

**2. Two NTK Regimes: Deterministic for Infinite Rank, Stochastic for Fixed Rank**

For CP SepNN:
$$f_\Theta(x_1,\dots,x_D)=\frac{1}{\sqrt{R}}\sum_{r=1}^R\prod_{d=1}^D (f_{\Theta_d}(x_d))_r,$$
the overall NTK is expressed as:
$$K_\Theta(x,x')=\frac{1}{R}\sum_{d=1}^D a_d(x)^\top K_{\Theta_d}(x_d,x'_d)a_d(x'),$$
where $K_{\Theta_d}$ is the multi-output NTK of the $d$-th factor MLP, and $a_d(x)$ collects rank-wise products of other factor outputs. 

In the infinite-width ($W\to\infty$) and infinite-rank ($R\to\infty$) regime, factor NTKs converge to a deterministic kernel $k(x_d,x'_d)$, and rank averages converge by the law of large numbers. The total NTK converges to:
$$K(x,x')=\sum_{d=1}^D k(x_d,x'_d)\prod_{d'\ne d}c_{d'}(x_{d'},x'_{d'}).$$
However, in the fixed-rank ($R$) and infinite-width ($W\to\infty$) regime, the rank average cannot eliminate randomness, leading to a stochastic kernel. This explains why small-rank SepNN training behavior remains stochastic.

**3. Spectral Bias Diagnosis: Explaining Slow Convergence via NTK Eigenvalues**

Under gradient flow with $L_2$ loss, the prediction dynamics follow $du(t)/dt=-K(u(t)-y)$. For $K=\sum_i \lambda_i v_iv_i^\top$, the residual in each eigen-direction is:
$$v_i^\top(u(t)-y)=\exp(-\lambda_i t)v_i^\top(u(0)-y).$$
Components projected onto large eigenvalues are learned first, while small eigenvalues lead to slow convergence. In INR, this corresponds to high-frequency textures; in PINNs, it corresponds to complex local structures. The authors verify that SepNN exhibits significant spectral bias, making "slow training on details" an analytically tractable problem.

**4. SepPGD: Decomposing Large Preconditioning into Dimension-wise Matrices**

Standard NTK preconditioning requires $S\in\mathbb{R}^{n^D\times n^D}$, which is computationally prohibitive for grids. SepPGD constructs $n\times n$ factor preconditioning matrices $S_d$ for each factor's pseudo-NTK. Using a "spectral flattening" approach: if eigenvalues are $\lambda_i$, and $g(\lambda_i)=\lambda_k$ for $i \leq k$, then:
$$S_d=I-\sum_{i=1}^k\left(1-\frac{g(\lambda_i)}{\lambda_i}\right)v_iv_i^\top.$$
During training, the residual tensor $\mathcal{R}$ is multiplied by $S_d$ along each mode. In 2D, SepPGD is equivalent to a Kronecker-sum preconditioning $\tilde{S}=S_1\otimes I_n+I_n\otimes S_2$. Using tensor identities, this maintains the $O(nD)$ efficiency.

### A Complete Example
For a $512\times512$ image INR, inputs are grid coordinates $(x,y)$. A standard MLP processes pixels as $512^2$ samples. SepNN feeds $x$-coordinates and $y$-coordinates into separate 1D factor MLPs to get matrices $F_x, F_y$, then predicts via $Z=F_x^\top F_y$. 

Early in training, SepNN captures low-frequency blocks. SepPGD calculates factor pseudo-NTKs to get $S_x, S_y$, reweights the image residual $R=Z-Y$ along each axis to construct target matrices $M_x, M_y$ for the factor MLPs. This allows high-frequency directions, suppressed by the NTK spectrum, to participate effectively in updates earlier.

### Loss & Training
The paper focuses on $L_2$ loss optimization. For grid points, the label is reshaped into a $D$-order tensor $Y$. SepPGD applies preconditioning to the data-fitting term. In PINN experiments, it is applied to data, initial, and boundary terms but not derivative terms. Factor networks typically use SIREN architectures.

## Key Experimental Results

### Main Results
The experiments cover KRR, image INR, 3D surface occupancy, and 3D separable PINNs.

| Task | Metric | SepNN | SepNN + SepPGD | Conclusion |
|------|------|-------|----------------|----------|
| Image (Plane) | PSNR | 26.48 | 33.30 | Significant detail recovery |
| 3D Surface (Thai statue) | IoU | 0.983 | 0.992 | Better texture capture |
| 3D Diffusion PINN | MSE | 0.042 | 0.037 | Further error reduction |
| 3D Klein-Gordon PINN | MSE | 0.029 | 0.018 | Accelerates PDE convergence |

| Method | Complexity | Objects | Application |
|------|------------------|----------------|----------|
| Hessian-based | $O(P)$ | Parameter Hessian | Expensive for large $P$ |
| Global NTK | $O(n^D)$ | Standard NTK | Prohibitive for grids |
| SepPGD | $O(nD)$ | $D$ factor NTKs | Efficient for SepNN |

### Ablation Study
- **Rank $R$**: SepPGD shows stable Gains across $R=100\sim700$.
- **Modulation $k$**: Effective for $k \ge 60$; insensitive in a reasonable range.
- **Update Frequency**: PSNR drops minimally when reducing update frequency, as NTK evolves slowly.
- **Activations**: Effective across Sin, Cos, and Fourier+ReLU.
- **Noise**: Robust at low/medium noise, but may fit noise faster at high noise levels.

### Key Findings
- **Theory-Experiment Alignment**: Infinite width alone does not yield a deterministic kernel for SepNN; infinite rank is also required.
- **Spectral Adjustment**: SepPGD improves convergence for textures and PDE details by smoothing the NTK spectrum.
- **Grid Efficiency**: SepPGD's advantages depend on grid structures; non-grid samples lose the Kronecker efficiency.

## Highlights & Insights
- **Theoretical Foundation**: Moves SepNN from an empirical structure to a theoretical object covering CP, TT, and Tucker forms via Stone-Weierstrass.
- **Fixed-rank Nuance**: highlights that infinite width is insufficient for a deterministic kernel, emphasizing the role of rank $R$ in dynamics.
- **Structural Alignment**: SepPGD is designed specifically for separable architectures, maintaining efficiency while improving optimization.
- **Addressing Real Pain Points**: Links spectral bias to specific failure modes in INR and PINN (e.g., blurred details).

## Limitations & Future Work
- **Approximation Rates**: Does not specify how error scales with rank, width, and dimension.
- **NTK Focus**: Primarily analyzed for CP SepNN; TT/Tucker extensions are provided but less detailed.
- **Fixed-rank Theory**: Lacks non-asymptotic convergence or generalization bounds for the fixed-rank regime.
- **Noise Sensitivity**: Faster high-frequency fitting may lead to overfitting under high noise levels.
- **PDE Residuals**: SepPGD is not currently applied to derivative-based PDE residuals.

## Related Work & Insights
- **Compared to CoordX**: Explains Why separable structures are expressive enough and provides NTK analysis.
- **Compared to Separable PINN**: Complements the existing efficiency with rigorous theory and accelerated convergence via SepPGD.
- **Compared to Global NTK Modulation**: SepPGD offers a computationally feasible alternative for large grid-based datasets.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Unifies approximation, NTK regimes, and preconditioning specifically for SepNN.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid coverage of INR and PINN; lacks large-scale scientific applications.
- Writing Quality: ⭐⭐⭐⭐☆ Clear logic, though tensor math might be challenging for some.
- Value: ⭐⭐⭐⭐⭐ Essential for users of separable coordinate networks and PINNs.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime](scaling_laws_and_spectra_of_shallow_neural_networks_in_the_feature_learning_regi.md)
- [\[ICLR 2026\] Interactive Learning of Single-Index Models via Stochastic Gradient Descent](interactive_learning_of_single-index_models_via_stochastic_gradient_descent.md)
- [\[ICLR 2026\] Gradient Descent Dynamics of Rank-One Matrix Denoising](gradient_descent_dynamics_of_rank-one_matrix_denoising.md)
- [\[ICLR 2026\] A New Initialization to Control Gradients in Sinusoidal Neural Networks](a_new_initialization_to_control_gradients_in_sinusoidal_neural_networks.md)
- [\[ICLR 2026\] Transformers Trained via Gradient Descent Can Provably Learn a Class of Teacher Models](transformers_trained_via_gradient_descent_can_provably_learn_a_class_of_teacher_.md)

</div>

<!-- RELATED:END -->
