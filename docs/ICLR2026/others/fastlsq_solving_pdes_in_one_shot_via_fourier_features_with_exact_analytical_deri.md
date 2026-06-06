---
title: >-
  [Paper Note] FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives
description: >-
  [ICLR2026][PDE solving] By exploiting the cyclic closed-form derivative structure of sinusoidal basis functions, this work presents a one-shot PDE solver that requires neither automatic differentiation nor iterative trai…
tags:
  - "ICLR2026"
  - "PDE solving"
  - "random Fourier features"
  - "physics-informed computing"
  - "one-shot solver"
  - "Newton-Raphson"
  - "inverse problems"
date: 2026-05-08
content_hash: 21dfed65c68a6020
---

# FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives

**Conference**: ICLR2026
**arXiv**: [2602.10541](https://arxiv.org/abs/2602.10541)  
**Code**: [sulcantonin/FastLSQ](https://github.com/sulcantonin/FastLSQ) (`pip install fastlsq`)  
**Area**: Other
**Keywords**: PDE solving, random Fourier features, physics-informed computing, one-shot solver, Newton-Raphson, inverse problems

## TL;DR
By exploiting the cyclic closed-form derivative structure of sinusoidal basis functions, this work presents a one-shot PDE solver that requires neither automatic differentiation nor iterative training. It achieves $10^{-7}$ accuracy in 0.07s for linear PDEs and $10^{-8}$–$10^{-9}$ accuracy in under 9s for nonlinear PDEs, outperforming PINNs by thousands of times in speed and several orders of magnitude in accuracy.

## Background & Motivation
- **Classical numerical methods** (FEM, FDM, spectral methods) dominate scientific computing, but for high-dimensional problems ($d \geq 5$) the computational cost scales as $h^{-d}$, and their implementation is highly problem-specific.
- **PINNs** offer a mesh-free alternative but suffer from serious drawbacks: training takes minutes to hours, and they are prone to spectral bias, causality violations, and sensitivity to loss weighting.
- **Random feature methods** (e.g., PIELM, RF-PDE) represent a middle ground by freezing random parameters and training only a linear output layer. However, PIELM uses $\tanh$ activations, which lack a closed-form cyclic derivative structure and require manual symbolic calculus derivation for each PDE operator; RF-PDE still requires 600–2000 iterations of optimization.
- **Core observation**: The sinusoidal feature $\phi_j(\mathbf{x}) = \sin(\mathbf{W}_j \cdot \mathbf{x} + b_j)$ admits closed-form derivatives of arbitrary order with a cyclic structure ($\sin \to \cos \to -\sin \to -\cos$), enabling $\mathcal{O}(1)$ assembly of any linear differential operator matrix without automatic differentiation or computational graphs.

## Core Problem
How to construct an **operator-agnostic** PDE solving framework that avoids the iterative training overhead of PINNs while also eliminating the burden of manually deriving derivative formulas for each new PDE operator as required by PIELM?

## Method

### 1. Random Fourier Feature Approximation
The PDE solution $u(\mathbf{x})$ is approximated using sinusoidal random features:

$$u_N(\mathbf{x}) = \frac{1}{\sqrt{N}} \sum_{j=1}^{N} \beta_j \sin(\mathbf{W}_j^\top \mathbf{x} + b_j)$$

where $\mathbf{W}_j \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_d)$ and $b_j \sim \mathcal{U}(0, 2\pi)$ are frozen; only the linear coefficients $\boldsymbol{\beta}$ are trained. The $1/\sqrt{N}$ normalization ensures the empirical kernel converges to a Gaussian RBF kernel and prevents coefficient inflation to $\mathcal{O}(10^6)$–$10^8$, which would cause ill-conditioning. A multi-block architecture with $B$ blocks using different bandwidths $\sigma_b$ is adopted to capture multi-scale features.

### 2. Exact Analytical Derivatives of Sinusoidal Bases (Key Innovation)
For any multi-index $\alpha = (\alpha_1, \dots, \alpha_d)$:

$$D^\alpha \phi_j(\mathbf{x}) = \left(\prod_{k=1}^d W_{jk}^{\alpha_k}\right) \cdot \Phi_{|\alpha| \bmod 4}(\mathbf{W}_j^\top \mathbf{x} + b_j)$$

where $\Phi_0 = \sin$, $\Phi_1 = \cos$, $\Phi_2 = -\sin$, $\Phi_3 = -\cos$. This implies:
- **Laplacian**: $\Delta \phi_j = -\|\mathbf{W}_j\|^2 \sin(\mathbf{W}_j^\top \mathbf{x} + b_j)$
- **Biharmonic**: $\Delta^2 \phi_j = \|\mathbf{W}_j\|^4 \sin(\mathbf{W}_j^\top \mathbf{x} + b_j)$
- **Advection**: $\mathbf{v} \cdot \nabla \phi_j = (\mathbf{v} \cdot \mathbf{W}_j) \cos(\mathbf{W}_j^\top \mathbf{x} + b_j)$

Each term requires only a single trigonometric evaluation multiplied by a monomial in the weights, with no automatic differentiation or computational graph needed. The $\tanh$ function does not admit an analogous closed-form pattern, as its $n$-th order derivative is a polynomial of degree $n+1$.

### 3. Linear PDEs: One-Shot Least-Squares Solve
Substituting the features into the linear PDE $\mathcal{L}[u] = f$ and boundary conditions $\mathcal{B}[u] = g$ yields an augmented linear system:

$$\begin{pmatrix} \mathbf{A}^{\text{pde}} \\ \lambda \mathbf{A}^{\text{bc}} \end{pmatrix} \boldsymbol{\beta} = \begin{pmatrix} \mathbf{f} \\ \lambda \mathbf{g} \end{pmatrix}$$

The solution $\boldsymbol{\beta}^* = \mathbf{A}^\dagger \mathbf{b}$ is obtained in a single shot via QR or SVD decomposition, requiring no iteration whatsoever.

### 4. Nonlinear PDEs: Newton-Raphson Extension
For nonlinear PDEs $\mathcal{L}[u] + \mathcal{N}[u] = f$, Newton-Raphson iteration is applied:

$$\mathbf{J}^{(k)} \delta\boldsymbol{\beta} = -\mathbf{R}^{(k)}, \quad \boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} + \alpha \delta\boldsymbol{\beta}$$

The Jacobian inherits the analytical closed-form structure. Four key algorithmic enhancements ensure robust convergence:
- **Warm-start**: the linear part is solved first to provide an initial guess
- **Backtracking line search**: Armijo-type sufficient decrease condition prevents excessively large steps
- **Solution-level convergence criterion**: $\|\Delta u\| / \|u\|$ is used instead of coefficient-level changes
- **Continuation (homotopy)**: for advection-dominated problems (e.g., Burgers), viscosity is gradually reduced as $\nu = 1.0 \to 0.5 \to 0.2 \to 0.1$

### 5. Downstream Applications
- **PDE discovery**: The analytical derivative dictionary is approximately 6000× cleaner than finite differences (RMSE 0.4 vs. 2500), substantially expanding the noise tolerance of SINDy-type methods.
- **Inverse problems**: Gradients propagate analytically through the pre-factored linear solve, enabling recovery of 4 anisotropic Gaussian heat sources (24 parameters) from 4 sensors, or hidden coil locations from 8 sparse magnetic field measurements (error $< 0.02$).

## Key Experimental Results

### Linear PDEs (Solver Mode)

| Problem | FastLSQ Time | FastLSQ $L^2$ | PINNacle Time | PINNacle $L^2$ | Speedup |
|------|------------|-------------|-------------|--------------|--------|
| Poisson 5D | 0.07s | 4.8e-7 | ~1780s | 4.7e-4 | 25000× |
| Wave 1D | 0.06s | 1.3e-6 | ~272s | 9.8e-2 | 4500× |
| Helmholtz 2D | 0.08s | 1.9e-6 | N/A | N/A | — |
| Maxwell 2D | 0.05s | 6.7e-7 | N/A | N/A | — |

### Nonlinear PDEs (Newton Solver Mode)
- NL-Poisson: $L^2 = 6.1 \times 10^{-8}$ (8.2s), surpassing even the regression baseline fitted to the exact solution ($1.9 \times 10^{-7}$)
- Burgers ($\nu=0.1$): $L^2 = 3.9 \times 10^{-9}$ (7.4s, 48 iterations with homotopy)
- Compared to scikit-fem P2 FEM: FastLSQ achieves $10^{-7}$–$10^{-9}$ with 1500 features, while FEM achieves $10^{-6}$ at ~4000 DoF

### Ablation Study
- Removing $1/\sqrt{N}$ normalization: accuracy degrades by 4 orders of magnitude or diverges
- Removing Tikhonov regularization: accuracy degrades by 3 orders of magnitude
- Removing warm-start: accuracy degrades by 1 order of magnitude or diverges
- Removing continuation: Burgers problem diverges

### sin vs. tanh Basis Comparison (Same Solver Protocol)
The accuracy gap of 10×–1000× is entirely attributable to the choice of basis function. For gradient accuracy, FastLSQ's gradient error is typically within one order of magnitude of its value error, whereas PIELM's gradient error is 10×–100× worse.

## Highlights & Insights
- **Elegantly simple core insight**: The cyclic derivative property of sinusoidal functions ($\sin \to \cos \to -\sin \to -\cos$) is elementary, yet its practical implications for PDE solving have long been overlooked. Systematizing it into a general-purpose solver framework is a remarkably clever contribution.
- **Operator-agnostic design**: A single formula applies to any linear differential operator, whereas PIELM requires a fresh manual derivation for every new PDE.
- **Speed and accuracy simultaneously**: Linear PDEs are solved in 0.07s to $10^{-7}$ accuracy—25,000× faster and 1000× more accurate than the fastest PINN variants.
- **Practical downstream applications**: PDE discovery (analytical derivative dictionaries) and inverse problems (heat source and coil localization) demonstrate the real engineering value of the framework.
- **Complete reproducible package**: `pip install fastlsq` with publicly available code.

## Limitations & Future Work
- The bandwidth $\sigma$ requires grid search tuning; no automatic selection strategy exists yet (though differentiable optimization is demonstrated in the appendix).
- For high-order PDEs or large $\sigma$, the monomial prefactors amplify the condition number, limiting achievable accuracy.
- The current framework supports only simple box domains; irregular geometries require additional boundary sampling strategies.
- The $1/\sqrt{N}$ normalization implies that increasing $N$ does not straightforwardly improve accuracy, as the kernel approximation saturates and conditioning deteriorates at large $N$.
- The Newton extension, while effective, is 40–100× slower than the linear mode (4–9s vs. $<$0.1s).
- For solutions with discontinuities (e.g., shocks), sinusoidal bases produce Gibbs oscillations; in such cases $\tanh$ bases are more robust.

## Related Work & Insights

| Method | Type | Iterations | Operator Derivation | Typical Accuracy | Typical Time |
|------|------|------|---------|---------|---------|
| **FastLSQ** | Sinusoidal random features | One-shot (linear) / Newton (NL) | Closed-form, universal | $10^{-7}$–$10^{-9}$ | 0.07–9s |
| PIELM | tanh random features | One-shot | Manual, per-operator | $10^{-3}$–$10^{-6}$ | ~0.07s |
| PINNs | Neural network | SGD, thousands of steps | Automatic differentiation | $10^{-2}$–$10^{-4}$ | 270–7500s |
| RF-PDE | Random features | 600–2000 rounds | Automatic differentiation | $10^{-3}$–$10^{-5}$ | 38–51s |
| RBF Kansa | Radial basis functions | One-shot | Analytical | ~$10^{-5}$ | Problem-dependent |
| FEM | Finite elements | Direct | Weak form | ~$10^{-6}$ | Infeasible for $d \geq 5$ |

This work demonstrates that **"returning to elementary mathematics for closed-form structure"** retains tremendous value even in the era dominated by deep learning—the cyclic derivatives of sine are high-school knowledge, yet systematically exploiting them can decisively outperform complex neural network approaches. Random Fourier features (Rahimi & Recht 2007) are well established in the kernel methods literature; this paper precisely bridges them with the requirements of PDE solving, constituting an elegant cross-domain transfer. The 6000× improvement in SINDy-type methods via analytical derivative dictionaries suggests that in scientific discovery tasks, **the quality of differentiability of the representation** may matter more than model capacity. The inverse problem applications (heat source localization, coil recovery) chart a clear path from "solving PDEs" to "differentiable digital twins."

## Rating
- Novelty: ⭐⭐⭐⭐ (The core insight is elementary but is systematized into a general framework, with a clear differentiation from PIELM)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (17 PDEs, multiple baselines, ablations, inverse problems, and gradient accuracy analysis—extremely comprehensive)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear exposition, information-dense tables, and a very fair comparison with related work)
- Value: ⭐⭐⭐⭐ (High practical value for the PDE solving community; the pip package significantly lowers the barrier to adoption)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Latent Fourier Transform](latent_fourier_transform.md)
- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[ICLR 2026\] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](../../AAAI2026/others/online_linear_regression_with_paid_stochastic_features.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](../../AAAI2026/others/certified_branch-and-bound_maxsat_solving_extended_version.md)

</div>

<!-- RELATED:END -->
