---
title: >-
  [Paper Note] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent
description: >-
  [ICML 2026][Optimization][Pseudospectra] This work establishes sharp Kreiss constant bounds $K(J) \leq 2/(1-\gamma) + \|C\|/(4(1-\gamma))$ for the block-triangular Jacobian $J = \begin{bmatrix} A & 0 \\ C & D \end{bmatri…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Pseudospectra"
  - "Kreiss Constant"
  - "Coupled Gradient Descent"
  - "Bilevel Optimization"
  - "Two-time-scale"
date: 2026-05-08
content_hash: ac4871f867c4222c
---

# Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent

**Conference**: ICML 2026  
**arXiv**: [2606.04031](https://arxiv.org/abs/2606.04031)  
**Code**: To be confirmed  
**Area**: Optimization / Learning Dynamics / Bilevel Optimization  
**Keywords**: Pseudospectra, Kreiss Constant, Coupled Gradient Descent, Bilevel Optimization, Two-time-scale

## TL;DR
This work establishes sharp Kreiss constant bounds $K(J) \leq 2/(1-\gamma) + \|C\|/(4(1-\gamma))$ for the block-triangular Jacobian $J = \begin{bmatrix} A & 0 \\ C & D \end{bmatrix}$ in coupled gradient descent and provides matching lower bounds. It reveals that transient amplification can be arbitrarily large even when the spectral radius is $< 1$. As a scaling law for high-dimensional learning dynamics, this theory provides a finite-time iteration complexity of $O(K(J)^2 \log(1/\delta))$ and extends to nearly self-referential systems.

## Background & Motivation

**Background**: Coupled gradient descent is ubiquitous in modern ML—bilevel optimization (HyperNet, MAML), two-time-scale stochastic approximation, and GANs (generator vs. discriminator). The linearized dynamics are given by $\begin{bmatrix}x_{t+1} \\ y_{t+1}\end{bmatrix} = J \begin{bmatrix}x_t \\ y_t\end{bmatrix}$, where $A = I - \alpha \nabla^2_{xx}F$ and $D = I - \beta \nabla^2_{yy}G$.

**Limitations of Prior Work**: (1) When $B = 0$ (block-triangular), asymptotic stability depends only on $\rho(A)$ and $\rho(D)$, but even if $\rho(A), \rho(D) < 1$, the transient $\|J^t\|$ can be arbitrarily large (**transient amplification in non-normal matrices**). (2) In numerical linear algebra, the Kreiss theorem and pseudospectral theory are known to characterize transients, but they are seldom used in optimization literature. (3) Existing optimization analyses (e.g., IQC) provide Lyapunov certificates but lack quantitative transient bounds. (4) In high-dimensional learning, the condition number grows $\to \gamma \to 1^-$ $\to \|C\|/(1-\gamma)$ explodes, making transient amplification particularly severe.

**Key Challenge**: Asymptotic stability ($\rho < 1$) does not imply training stability—transients can be amplified by orders of magnitude. This problem is especially acute in high-dimensional learning but is completely ignored by existing analyses that only consider the spectral radius.

**Goal**: (1) Establish sharp upper and lower Kreiss constant bounds for block-triangular Jacobians; (2) Characterize critical coupling thresholds; (3) Extend to nearly self-referential systems (where $B \neq 0$ but is small); (4) Provide a non-asymptotic iteration complexity scaling law.

**Key Insight**: Utilize pseudospectral theory $\Lambda_\varepsilon(M) = \{z : \|(zI-M)^{-1}\| > 1/\varepsilon\}$ and the Kreiss constant $K(M) = \sup_{|z|>1}(|z|-1)\|(zI-M)^{-1}\|$. The Kreiss theorem $K(M) \leq \sup_t \|M^t\| \leq enK(M)$ precisely controls transient amplification. For block-triangular matrices, the block resolvent formula is used to decompose the analysis; for symmetric diagonal blocks, $\|(zI-A)^{-1}\| \leq 1/(r-\gamma)$ is used, while the off-diagonal block contributes $\|C\|/(r-\gamma)^2$.

**Core Idea**: Formalize "transient amplification of non-normal matrices" using the Kreiss constant, provide closed-form upper and lower bounds for block-triangular structures, and introduce these numerical analysis tools into the non-asymptotic analysis of coupled optimization.

## Method

### Overall Architecture

Linearized coupled gradient descent $J = \begin{bmatrix} A & B \\ C & D \end{bmatrix}$, with a focus on:
- **Section 4**: Main results for $B = 0$ block-triangular structures.
- **Section 5**: Extension to $B \neq 0$ self-referential systems via Neumann series perturbation.
- **Section 6**: Sample-complexity scaling law.

### Key Designs

1. **Upper and Lower Kreiss Bounds for Block-triangular (Theorem 4 & 5)**:
    - **Function**: Rigorously quantify transient amplification as a function of $\gamma$ and $\|C\|$.
    - **Mechanism**: Use the block resolvent $(zI - J)^{-1} = \begin{bmatrix}(zI-A)^{-1} & 0 \\ (zI-D)^{-1} C (zI-A)^{-1} & (zI-D)^{-1}\end{bmatrix}$. Symmetric $A, D$ yield $\|(zI-A)^{-1}\| \leq 1/(r-\gamma)$, while the off-diagonal term yields $\|(zI-D)^{-1} C (zI-A)^{-1}\| \leq \|C\|/(r-\gamma)^2$. Optimizing $r > 1$ gives $K(J) \leq \sup_r [2(r-1)/(r-\gamma) + (r-1)\|C\|/(r-\gamma)^2]$.
    - **Design Motivation**: The block resolvent allows separate analysis of symmetric and non-normal components. Optimizing $r$ provides a closed-form solution. The matching bounds (modulo a factor-of-2 gap) demonstrate the sharpness of the bound.

2. **Minimax Lower Bound + Critical Coupling Threshold (Theorem 7 & 10)**:
    - **Function**: Prove that any estimator using only $(\rho(A), \rho(D), \|C\|)$ has a worst-case distance from the true $K(J)$ of at least $c/(8(1-\gamma)^2)$. Characterize the critical coupling at which spectral instability occurs.
    - **Mechanism**: Construct a family of worst-case Jacobians such that any estimator has $\Omega(c/(1-\gamma)^2)$ error on this family. The critical coupling threshold compares $\|C\|$ with $(1-\gamma)^2$; exceeding this threshold shifts the system from transient amplification to spectral instability.
    - **Design Motivation**: The minimax lower bound proves that the proposed bound cannot be fundamentally improved. The threshold provides direct design guidelines for practitioners (at what coupling level instability begins).

3. **Neumann Perturbation Extension to $B \neq 0$ (Theorem 9)**:
    - **Function**: Generalize block-triangular results to nearly self-referential systems.
    - **Mechanism**: Let $J_\varepsilon = J_0 + \varepsilon B_0$, where $J_0$ is block-triangular. If $\varepsilon \|B_0\| K_0 < (1-\gamma)$, the Neumann series $(zI - J_\varepsilon)^{-1} = (zI - J_0)^{-1} \sum_k (\varepsilon B_0 (zI - J_0)^{-1})^k$ converges uniformly for $|z| > 1$, resulting in $K(J_\varepsilon) \leq K_0 / (1 - \varepsilon\|B_0\|K_0/(1-\gamma))$.
    - **Design Motivation**: Real systems often exhibit weak self-reference (e.g., a GAN generator indirectly observes itself). The perturbation framework maintains the validity of block-triangular results under small coupling.

### Sample-complexity scaling law (Theorem 11)

Stochastic coupled descent requires $T(\delta) = O(K(J)^2 \log(1/\delta)/(1-\gamma)^2)$ steps to reach $\delta$ precision. This is a non-asymptotic scaling law for high-dimensional learning dynamics that is instance-dependent (on the specific $J$), rather than purely worst-case.

## Key Experimental Results

### Transient Verification in Linear-Quadratic Problems

As $\|C\|$ increases, the measured $\sup_t \|J^t\|$ aligns with the proposed bound $2/(1-\gamma) + $\|C\|/(4(1-\gamma))$ (Figure 1). For different values of $\gamma$, the bound accurately tracks the measured transient peaks.

### Comparison vs. IQC

On the same set of coupled LQ problems:

| Method | Transient Bound | Tightness |
|------|---------|-----|
| Spectral radius only | Asymptotic only ($\rho < 1$) | Fails completely |
| IQC Lyapunov | $\geq$ 10× measured peak | Conservative |
| **Pseudospectral (Ours)** | **~1.5× measured peak** | **Tight** |

IQC provides safety certificates but is 10× conservative; the proposed bound is over 6× tighter.

### Neural Network Training Verification

Tracking the effective $K(J)$ of linearized dynamics during GAN training: the predicted "high-K phase = unstable training" precisely corresponds to measured training collapses. This provides a usable tool for **predicting training failure from a dynamical spectral perspective**.

### Key Findings
- **Transient amplification is a real risk in high-dimensional learning**: For $\gamma \to 1$ (high condition number), $K(J)$ can reach several hundreds, meaning $\|J^t\|$ transients can amplify initial errors by hundreds of times.
- **Block-triangular structures are common**: Bilevel optimization (where the inner-loop does not affect the outer-loop Hessian) is naturally block-triangular.
- **Significantly tighter than IQC**: This work provides a quantitative transient bound, whereas IQC provides only qualitative certificates.
- **GAN training prediction**: The framework can serve as an early warning system for training collapse.

## Highlights & Insights
- **Introduction of Kreiss Theorem and Pseudospectral Theory to Optimization**: These established tools from numerical linear algebra have been long ignored by the ML community. This work introduces them systematically with LLM/GAN-scale implications, opening a new research direction.
- **Block-triangular is an undervalued structure**: It describes bilevel optimization and TTS approximation. Separating diagonal symmetric blocks from off-diagonal components simplifies the analysis significantly.
- **Scaling Law Perspective**: The instance-dependent complexity $T(\delta) = O(K(J)^2 \log(1/\delta)/(1-\gamma)^2)$ exposes regimes invisible to spectral-radius-only analysis.
- **Theoretical Rigor + Numerical Validation**: A complete chain of logic—upper/lower bounds, minimax results, critical thresholds, perturbation extensions, scaling laws, and experiments.

## Limitations & Future Work
- The factor-of-2 gap in the leading term remains unclosed; whether the bound can be further tightened is open.
- The assumption of symmetric $A, D$ is strong; non-symmetric cases (e.g., GANs with regularizers) require re-analysis.
- The self-referential extension is limited to small $\varepsilon$; strong coupling scenarios like certain GANs remain uncovered.
- Experiments are biased toward LQ + toy GANs; validation on large-scale LLM training is missing.
- The scaling law is in a worst-case form and might be conservative on benign instances.

## Related Work & Insights
- **vs. IQC (Lessard 2016)**: IQC provides qualitative Lyapunov certificates; this work provides quantitative transient bounds.
- **vs. Two-time-scale SA (Konda-Tsitsiklis)**: That analysis focuses on asymptotic convergence; this work is non-asymptotic and addresses transients.
- **vs. Pseudospectra (Trefethen-Embree)**: That is the foundation in numerical linear algebra; this work is the first systematic application to ML optimization analysis.
- **Insight**: All "non-normal linearized dynamics" scenarios (GANs, actor-critic RL, bilevel meta-learning) can benefit from Kreiss analysis; these pseudospectral tools can be generalized to various aspects of optimization algorithm stability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bringing Kreiss theorem and pseudospectra into coupled optimization is a truly new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ LQ + IQC comparison + NN validation, though somewhat toy; lacks large-scale LLM/GAN validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous with a complete chain of theorems; the scaling law framing is very persuasive.
- Value: ⭐⭐⭐⭐ High theoretical value for bilevel, GAN, and TTS RL communities; significant implications for high-dimensional learning dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](../../ICLR2026/optimization/nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)

</div>

<!-- RELATED:END -->
