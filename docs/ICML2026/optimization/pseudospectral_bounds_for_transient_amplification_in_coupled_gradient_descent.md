---
title: >-
  [Paper Note] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] This paper establishes sharp Kreiss constant bounds $K(J) \leq 2/(1-\gamma) + \|C\|/(4(1-\gamma))$ for the block-triangular Jacobian $J = \begin{bmatrix} A & 0 \\ C & D \end{bmatrix}$ in coupled gradient descent, providing matching lower bounds—revealing that transient amplification can be arbitrarily large even when t
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: fba372c9575b6e4a
---
# Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent

**Conference**: ICML 2026  
**arXiv**: [2606.04031](https://arxiv.org/abs/2606.04031)  
**Code**: TBD  
**Area**: Optimization / Learning Dynamics / Bilevel Optimization  
**Keywords**: Pseudospectrum, Kreiss constant, Coupled gradient descent, Bilevel optimization, Two-time-scale

## TL;DR
This paper establishes sharp Kreiss constant bounds $K(J) \leq 2/(1-\gamma) + \|C\|/(4(1-\gamma))$ for the block-triangular Jacobian $J = \begin{bmatrix} A & 0 \\ C & D \end{bmatrix}$ in coupled gradient descent, providing matching lower bounds—revealing that transient amplification can be arbitrarily large even when the spectral radius is less than 1. This theory serves as a scaling law for high-dimensional learning dynamics, yielding $O(K(J)^2 \log(1/\delta))$ finite-time iteration complexity and extending to nearly self-referential systems.

## Background & Motivation

**Background**: Coupled gradient descent is ubiquitous in modern ML—bilevel optimization (HyperNet, MAML), two-time-scale stochastic approximation, and GANs (generator vs. discriminator). Linearized dynamics are given by $\begin{bmatrix}x_{t+1} \\ y_{t+1}\end{bmatrix} = J \begin{bmatrix}x_t \\ y_t\end{bmatrix}$, where $A = I - \alpha \nabla^2_{xx}F$ and $D = I - \beta \nabla^2_{yy}G$.

**Limitations of Prior Work**: (1) When $B = 0$ (block-triangular), asymptotic stability only considers $\rho(A)$ and $\rho(D)$, yet even if $\rho(A), \rho(D) < 1$, the transient $\|J^t\|$ can be arbitrarily large (**transient amplification of non-normal matrices**). (2) Kreiss theorem and pseudospectral theory are established tools in numerical linear algebra to characterize transients, but they are seldom used in optimization literature. (3) Existing optimization analyses (e.g., IQC) provide Lyapunov certificates but lack quantitative transient bounds. (4) In high-dimensional learning, condition numbers grow $\rightarrow \gamma \to 1^- \rightarrow \|C\|/(1-\gamma)$ explodes $\rightarrow$ transient amplification becomes particularly severe.

**Key Challenge**: Asymptotic stability ($\rho < 1$) does not imply stable training—transients may amplify by orders of magnitude. This issue is especially pronounced in high-dimensional learning but is entirely ignored by existing analyses focusing solely on the spectral radius.

**Goal**: (1) Establish sharp upper and lower Kreiss constant bounds for block-triangular Jacobians; (2) Characterize the critical coupling threshold; (3) Extend to nearly self-referential ($B \neq 0$ but small) systems; (4) Provide non-asymptotic iteration complexity scaling laws.

**Key Insight**: Leveraging pseudospectral theory $\Lambda_\varepsilon(M) = \{z : \|(zI-M)^{-1}\| > 1/\varepsilon\}$ and the Kreiss constant $K(M) = \sup_{|z|>1}(|z|-1)\|(zI-M)^{-1}\|$. The Kreiss theorem $K(M) \leq \sup_t \|M^t\| \leq enK(M)$ precisely controls transient amplification. For block-triangular matrices, the block resolvent formula is used to decompose the problem, where symmetric diagonal blocks yield $\|(zI-A)^{-1}\| \leq 1/(r-\gamma)$ and the off-diagonal block contributes $\|C\|/(r-\gamma)^2$.

**Core Idea**: Formalize "transient amplification of non-normal matrices" using the Kreiss constant, provide closed-form upper and lower bounds for block-triangular structures, and introduce these numerical analysis tools into the non-asymptotic analysis of coupled optimization.

## Method

### Overall Architecture

The paper is structured as a **sequential chain of theorems**. After linearizing coupled gradient descent into $J=\begin{bmatrix} A & B \\ C & D\end{bmatrix}$ near a fixed point, the paper progresses in four steps: it first quantifies transient amplification as closed-form Kreiss constant bounds in the simplest $B=0$ (block-triangular) case (Design 1); it then proves this bound exhausts all information from $(\rho(A),\rho(D),\|C\|)$ and characterizes the critical threshold where coupling becomes dangerous (Design 2); next, it uses Neumann series to extend conclusions from strictly triangular perturbations to nearly self-referential systems where $B\neq 0$ (Design 3); finally, it translates the Kreiss constant into iteration complexity scaling laws for reaching a target accuracy $\delta$ in stochastic settings (Design 4).

### Key Designs

**1. Block-triangular Kreiss bounds (Theorem 4 & 5): Quantifying transient amplification as a function of $\gamma$ and $\|C\|$**

The difficulty with non-normal matrices is that $\|J^t\|$ can grow even if the spectral radius is $<1$; the Kreiss constant is required to characterize this. The block-triangular structure allows the resolvent to be decomposed by blocks:

$$(zI-J)^{-1}=\begin{bmatrix}(zI-A)^{-1} & 0 \\ (zI-D)^{-1}C(zI-A)^{-1} & (zI-D)^{-1}\end{bmatrix}.$$

Symmetric diagonal blocks satisfy $\|(zI-A)^{-1}\|\le 1/(r-\gamma)$, while the off-diagonal term satisfies $\|(zI-D)^{-1}C(zI-A)^{-1}\|\le\|C\|/(r-\gamma)^2$. Optimizing for $r>1$ yields $K(J)\le\sup_r[2(r-1)/(r-\gamma)+(r-1)\|C\|/(r-\gamma)^2]$. This approach separates symmetric and non-normal components and provides matched upper and lower bounds (within a factor-of-2), indicating the sharpness of the bound.

**2. Minimax lower bound + Critical coupling threshold (Theorem 7 & 10): Proving bound sharpness and boundary conditions**

To verify if $(\rho(A),\rho(D),\|C\|)$ are sufficient descriptors, the authors construct a family of worst-case Jacobians. They show that any estimator using only these quantities has a minimax error of at least $c/(8(1-\gamma)^2)$, declaring the derived bounds optimal. Simultaneously, a critical coupling threshold compares $\|C\|$ directly with $(1-\gamma)^2$; exceeding this threshold shifts the system from "transient amplification" to "spectral instability," providing a design guideline for coupling limits.

**3. Neumann perturbation extension to $B\neq 0$ (Theorem 9): Generalizing to nearly self-referential systems**

Actual systems are often weakly self-referential (e.g., GAN generators indirectly observe themselves). The Jacobian is defined as $J_\varepsilon=J_0+\varepsilon B_0$, where $J_0$ is block-triangular. As long as $\varepsilon\|B_0\|K_0<(1-\gamma)$, the Neumann series $(zI-J_\varepsilon)^{-1}=(zI-J_0)^{-1}\sum_k(\varepsilon B_0(zI-J_0)^{-1})^k$ converges uniformly for $|z|>1$, leading to:

$$K(J_\varepsilon)\le \frac{K_0}{1-\varepsilon\|B_0\|K_0/(1-\gamma)}.$$

This allows conclusions from block-triangular models to extend smoothly to realistic near-triangular scenarios.

**4. Sample-complexity scaling law (Theorem 11): Translating Kreiss constants to training duration**

The authors express the iteration complexity $T(\delta)$ of stochastic coupled descent (with gradient noise variance $\sigma^2$) to reach accuracy $\delta$ as a function of the Kreiss constant: $T(\delta) = O(K(J)^2 \log(1/\delta)/(1-\gamma)^2)$. This **instance-dependent** scaling law exposes a regime hidden to spectral radius analysis: in high-dimensional learning where $\gamma\to 1$, $K(J)$ can be very large, causing complexity to explode quadratically.

## Key Experimental Results

### Transient validation in Linear-Quadratic problems
Empirical measurements of $\sup_t \|J^t\|$ align with the proposed bound $2/(1-\gamma) + $\|C\|/(4(1-\gamma))$ (Fig 1). The bound accurately tracks measured transient peaks across various values of $\gamma$.

### Main Results vs. IQC
Comparison on a set of coupled LQ problems:

| Method | Transient Bound | Tightness |
| :--- | :--- | :--- |
| Spectral radius only | Asymptotic only ($\rho < 1$) | Fails completely |
| IQC Lyapunov | $\geq$ 10x measured peak | Conservative |
| **Pseudospectral (Ours)** | **~1.5x measured peak** | **Tight** |

While IQC provides certificates, it is 10x conservative; the pseudospectral bound is more than 6x tighter.

### Neural network training validation
Tracking the effective $K(J)$ during GAN training reveals that the predicted "high-$K$ phase = unstable training" corresponds precisely with empirical training collapse, providing a tool for predicting failure via dynamical spectra.

### Key Findings
- **Transient amplification is a significant risk in high-dimensional learning**: As $\gamma \to 1$, $K(J)$ can reach values in the hundreds, meaning $\|J^t\|$ can amplify initial errors by orders of magnitude.
- **Block-triangular structures are ubiquitous**: Bilevel optimization naturally follows this structure when the inner-loop has negligible effects on outer-loop Hessians.
- **Superiority over IQC**: The framework provides quantitative transient bounds whereas IQC only yields qualitative certificates.
- **GAN Training Prediction**: The framework can serve as an early warning system for training instability.

## Highlights & Insights
- **Introduction of Kreiss Theorem + Pseudospectral Theory**: These mature numerical linear algebra tools have been largely overlooked in ML optimization; this paper defines their utility for large-scale learning.
- **Underestimated Block-triangular Structure**: This structural assumption allows for a clean separation of symmetric and non-normal components.
- **Scaling Law Perspective**: The $O(K(J)^2 \log(1/\delta)/(1-\gamma)^2)$ instance-dependent complexity reveals failure modes that spectral radius analysis misses.
- **Theoretical Rigor + Numerical Validation**: The logic holds from minimax lower bounds to practical neural network experiments.

## Limitations & Future Work
- A factor-of-2 gap remains in the leading term of the bound.
- The assumption of symmetric $A, D$ is restrictive; non-symmetric cases (e.g., GANs with complex regularizers) require further study.
- Extension to self-referential systems only covers small $\varepsilon$; strongly coupled systems are unaddressed.
- Experiments focus on LQ and toy GANs; validation on large-scale LLM training is missing.
- The scaling law is based on worst-case formulation and might be conservative for benign instances.

## Related Work & Insights
- **vs. IQC (Lessard 2016)**: IQC provides certificates; this work provides quantitative transient bounds.
- **vs. Two-time-scale SA (Konda-Tsitsiklis)**: Their analysis focuses on asymptotic convergence; this work focuses on non-asymptotic transients.
- **vs. Pseudospectra (Trefethen-Embree)**: This work adapts these numerical tools specifically for ML optimization analysis.
- **Insight**: Any "non-normal linearized dynamics" scenario (GANs, Actor-Critic RL, bilevel meta-learning) can benefit from Kreiss-based analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](../../ICML2025/optimization/quantum_optimization_via_gradient-based_hamiltonian_descent.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)

</div>

<!-- RELATED:END -->
