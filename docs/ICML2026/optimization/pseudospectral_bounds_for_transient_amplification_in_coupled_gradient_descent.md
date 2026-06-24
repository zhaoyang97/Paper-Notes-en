---
title: >-
  [Paper Note] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent
description: >-
  [ICML 2026][Optimization][Pseudospectra] This paper establishes sharp Kreiss constant bounds $K(J) \leq 2/(1-\gamma) + \|C\|/(4(1-\gamma))$ for block-triangular Jacobians $J = \begin{bmatrix} A & 0 \\ C & D \end{bmatrix}$ in coupled gradient descent, providing matching lower bounds. It reveals that transient amplification can be arbitrarily large even when the spectral radius is $< 1$. This theory serves as a scaling law for high-dimensional learning dynamics…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Pseudospectra"
  - "Kreiss Constant"
  - "Coupled Gradient Descent"
  - "Bilevel Optimization"
  - "Two-Time-Scale"
date: 2026-05-08
content_hash: 9d0b3c08bea63344
---

# Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent

**Conference**: ICML 2026  
**arXiv**: [2606.04031](https://arxiv.org/abs/2606.04031)  
**Code**: TBD  
**Area**: Optimization / Learning Dynamics / Bilevel Optimization  
**Keywords**: Pseudospectra, Kreiss Constant, Coupled Gradient Descent, Bilevel Optimization, Two-Time-Scale

## TL;DR
This paper establishes sharp Kreiss constant bounds $K(J) \leq 2/(1-\gamma) + \|C\|/(4(1-\gamma))$ for block-triangular Jacobians $J = \begin{bmatrix} A & 0 \\ C & D \end{bmatrix}$ in coupled gradient descent, providing matching lower bounds. It reveals that transient amplification can be arbitrarily large even when the spectral radius is $< 1$. This theory serves as a scaling law for high-dimensional learning dynamics, providing a finite-time iteration complexity of $O(K(J)^2 \log(1/\delta))$ and extending the results to nearly self-referential systems.

## Background & Motivation

**Background**: Coupled gradient descent is ubiquitous in modern ML—ranging from bilevel optimization (HyperNet, MAML) and two-time-scale stochastic approximation to GANs (generator vs. discriminator). The linearized dynamics are represented as $\begin{bmatrix}x_{t+1} \\ y_{t+1}\end{bmatrix} = J \begin{bmatrix}x_t \\ y_t\end{bmatrix}$, where $A = I - \alpha \nabla^2_{xx}F$ and $D = I - \beta \nabla^2_{yy}G$.

**Limitations of Prior Work**: (1) When $B = 0$ (block-triangular), asymptotic stability depends only on $\rho(A)$ and $\rho(D)$; however, even if $\rho(A), \rho(D) < 1$, the transient $\|J^t\|$ can be arbitrarily large (**transient amplification in non-normal matrices**). (2) While the Kreiss theorem and pseudospectral theory in numerical linear algebra are known to characterize transients, they are rarely utilized in optimization literature. (3) Existing optimization analyses (e.g., IQC) provide Lyapunov certificates but lack quantitative transient bounds. (4) In high-dimensional learning, the condition number grows ($\gamma \to 1^-$), causing $\|C\|/(1-\gamma)$ to explode and making transient amplification particularly severe.

**Key Challenge**: Asymptotic stability ($\rho < 1$) does not guarantee training stability—transients may amplify by orders of magnitude. This issue is exacerbated in high-dimensional learning but is entirely overlooked by existing analyses that only focus on the spectral radius.

**Goal**: (1) Establish sharp upper and lower Kreiss constant bounds for block-triangular Jacobians; (2) Characterize the critical coupling threshold; (3) Extend the theory to nearly self-referential systems ($B \neq 0$ but small); (4) Provide a non-asymptotic iteration complexity scaling law.

**Key Insight**: The study utilizes pseudospectral theory $\Lambda_\varepsilon(M) = \{z : \|(zI-M)^{-1}\| > 1/\varepsilon\}$ and the Kreiss constant $K(M) = \sup_{|z|>1}(|z|-1)\|(zI-M)^{-1}\|$. The Kreiss theorem $K(M) \leq \sup_t \|M^t\| \leq enK(M)$ precisely controls transient amplification. For block-triangular structures, the block resolvent formula is decomposed; the symmetric diagonal blocks yield $\|(zI-A)^{-1}\| \leq 1/(r-\gamma)$, while the off-diagonal block contributes $\|C\|/(r-\gamma)^2$.

**Core Idea**: Formalize "transient amplification of non-normal matrices" using the Kreiss constant, provide closed-form upper and lower bounds for block-triangular cases, and introduce these numerical analysis tools into the non-asymptotic analysis of coupled optimization.

## Method

### Overall Architecture

This paper is not a data pipeline but an **interlocking chain of theorems**. After linearizing the coupled gradient descent near a fixed point as $J=\begin{bmatrix} A & B \\ C & D\end{bmatrix}$, the research progresses in four steps: first, quantifying transient amplification as closed-form Kreiss constant bounds in the cleanest $B=0$ (block-triangular) case (Design 1); second, proving that this bound extracts all relevant information from $(\rho(A), \rho(D), \|C\|)$ and characterizing the "critical red line" where coupling becomes dangerous (Design 2); third, using Neumann series to extend conclusions from strict triangular perturbations to nearly self-referential systems where $B\neq0$ (Design 3); and finally, translating the Kreiss constant into a scaling law for iteration complexity required to reach $\delta$-accuracy in stochastic settings (Design 4). These four steps advance systematically—from "how large is the transient" to "is the bound tight," "when does instability occur," "does it hold for non-ideal systems," and finally "how many steps to train."

### Key Designs

**1. Block-triangular Kreiss Upper and Lower Bounds (Theorem 4 & 5): Quantifying transient amplification as a function of $\gamma$ and $\|C\|$**

The difficulty with non-normal matrices is that $\|J^t\|$ might not be suppressed even if the spectral radius is $<1$; the Kreiss constant is required to characterize the transient. The block-triangular structure allows the resolvent to be decomposed by blocks:

$$(zI-J)^{-1}=\begin{bmatrix}(zI-A)^{-1} & 0 \\ (zI-D)^{-1}C(zI-A)^{-1} & (zI-D)^{-1}\end{bmatrix}.$$

The symmetric diagonal blocks yield $\|(zI-A)^{-1}\|\le 1/(r-\gamma)$, while the off-diagonal term gives $\|(zI-D)^{-1}C(zI-A)^{-1}\|\le\|C\|/(r-\gamma)^2$. Optimizing over $r>1$ yields $K(J)\le\sup_r[2(r-1)/(r-\gamma)+(r-1)\|C\|/(r-\gamma)^2]$. The advantage of this block decomposition is the separate treatment of symmetric and non-normal components, resulting in clean bounds. The matching upper and lower bounds (within a factor-of-2 gap) indicate that the bound is sharp rather than a loose estimate.

**2. Minimax Lower Bound + Critical Coupling Threshold (Theorem 7 & 10): Proving the bound cannot be essentially improved and defining the hazard line**

Upper bounds alone are insufficient; it must be shown whether $(\rho(A), \rho(D), \|C\|)$ provide enough information. The authors construct a family of worst-case Jacobians such that any estimator using only these parameters incurs an error of at least $\Omega(c/(1-\gamma)^2)$, meaning the distance to the true $K(J)$ is at least $c/(8(1-\gamma)^2)$. This minimax lower bound confirms that the proposed bound fully utilizes the available information. Concurrently, the critical coupling threshold directly compares $\|C\|$ with $(1-\gamma)^2$; exceeding this threshold pushes the system from "transient amplification" toward "spectral instability," providing a clear design red line for practitioners.

**3. Neumann Perturbation Extension to $B\neq 0$ (Theorem 9): Generalizing to nearly self-referential systems**

Real-world systems are often weakly self-referential (e.g., a GAN generator indirectly observes itself), making strict block-triangular structures idealized. The authors express the Jacobian as $J_\varepsilon=J_0+\varepsilon B_0$, where $J_0$ is block-triangular. As long as $\varepsilon\|B_0\|K_0<(1-\gamma)$, the Neumann series $(zI-J_\varepsilon)^{-1}=(zI-J_0)^{-1}\sum_k(\varepsilon B_0(zI-J_0)^{-1})^k$ converges uniformly for $|z|>1$, leading to:

$$K(J_\varepsilon)\le \frac{K_0}{1-\varepsilon\|B_0\|K_0/(1-\gamma)}.$$

This allows block-triangular conclusions to transition smoothly to real-world nearly self-referential scenarios under small coupling, rather than being restricted to idealized triangular cases.

**4. Sample-complexity Scaling Law (Theorem 11): Translating the Kreiss constant into "training steps"**

The first three designs characterize the transient itself, but practitioners care about convergence steps. The authors formulate the iteration complexity for stochastic coupled descent (with gradient noise variance $\sigma^2$) to reach $\delta$-accuracy as a function of the Kreiss constant: $T(\delta) = O(K(J)^2 \log(1/\delta)/(1-\gamma)^2)$. Crucially, this is **instance-dependent** (relying on the specific $J$), revealing a regime invisible to spectral radius analysis: in high-dimensional learning where $\gamma\to 1$, $K(J)$ can soar to the hundreds, causing iteration complexity to explode quadratically. This step grounds the transient theory into actionable conclusions regarding computational cost.

## Key Experimental Results

### Transient Verification in Linear-Quadratic Problems
As $\|C\|$ increases, the measured $\sup_t \|J^t\|$ aligns with the proposed bound $2/(1-\gamma) + \|C\|/(4(1-\gamma))$ (Figure 1 in the paper). Under different values of $\gamma$, the bound accurately tracks the measured transient peaks.

### Comparison vs. IQC
On the same set of coupled LQ problems:

| Method | Transient Bound | Tightness |
|------|---------|-----|
| Spectral radius only | Asymptotic only ($\rho < 1$) | Fails completely |
| IQC Lyapunov | $\geq$ 10× Measured Peak | Conservative |
| **Pseudospectral (Ours)** | **~1.5× Measured Peak** | **Tight** |

IQC provides safety certificates but is conservative by 10×; the proposed bound is over 6× tighter.

### Neural Network Training Verification
The effective $K(J)$ of linearized dynamics was tracked during GAN training. The predicted "high-K phase = unstable training" corresponds precisely with measured training collapses, providing a tool to **predict training failure from the perspective of dynamical spectra**.

### Key Findings
- **Transient amplification is a real risk in high-dimensional learning**: For $\gamma \to 1$ (high condition number), $K(J)$ can reach hundreds, meaning $\|J^t\|$ can amplify initial errors by hundreds of times during the transient phase.
- **Block-triangular structures are common**: Bilevel optimization (where the inner-loop does not affect the outer-loop Hessian) is naturally block-triangular.
- **Significantly tighter than IQC**: Ours provides a quantitative transient bound, whereas IQC only gives qualitative certificates.
- **GAN training prediction**: The framework can serve as an early warning system for training collapse.

## Highlights & Insights
- **Introducing Kreiss Theorem + Pseudospectral Theory to Optimization Analysis**: Mature tools from numerical linear algebra have been long ignored by the ML community; this paper introduces them systematically, revealing consequences at LLM/GAN scales and opening a new research direction.
- **The Undervalued Block-Triangular Structure**: Recognizing that bilevel optimization and TTS approximation share this structure allows for elegant analysis by separating diagonal symmetric components from off-diagonal ones.
- **Scaling Law Perspective**: The instance-dependent complexity $T(\delta) = O(K(J)^2 \log(1/\delta)/(1-\gamma)^2)$ exposes regimes that spectral-radius analysis cannot see.
- **Theoretical Rigor + Numerical Validation**: The complete chain from upper/lower bounds and minimax limits to perturbation extensions and scaling laws is thoroughly validated by experiments.

## Limitations & Future Work
- The factor-of-2 gap in the leading term remains unclosed; whether the bound can be further tightened is an open question.
- The assumption of symmetric $A, D$ is strong; asymmetric cases (e.g., GANs with certain regularizations) require fresh analysis.
- The self-referential extension only covers small $\varepsilon$; strong coupling scenarios (e.g., certain GANs) are not yet addressed.
- Experiments are biased toward LQ and toy GANs; validation on large-scale LLM training is missing.
- The scaling law is presented in a worst-case form and might be conservative on benign instances.

## Related Work & Insights
- **vs. IQC (Lessard 2016)**: IQC provides qualitative Lyapunov certificates; this work provides quantitative transient bounds.
- **vs. Two-time-scale SA (Konda-Tsitsiklis)**: That analysis focuses on asymptotic convergence; this work focuses on non-asymptotic transient behavior.
- **vs. Pseudospectra (Trefethen-Embree)**: That is a foundation of numerical linear algebra; this work is the first to systematically apply it to ML optimization analysis.
- **Insight**: All "non-normal linearized dynamics" scenarios (GANs, actor-critic RL, bilevel meta-learning) can benefit from Kreiss analysis; these pseudospectral tools can be extended to various aspects of optimization algorithm stability analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bringing Kreiss theorem + pseudospectra into coupled optimization is a truly new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong LQ + IQC comparison + NN validation, but slightly toy-oriented; lacks large-scale LLM/GAN validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous with a complete theorem chain; the scaling law framing is very persuasive.
- Value: ⭐⭐⭐⭐ High theoretical value for bilevel, GAN, and TTS RL communities; significant implications for high-dimensional learning dynamics theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] Flatland: The Adventures of Gradient Descent with Large Step Sizes](flatland_the_adventures_of_gradient_descent_with_large_step_sizes.md)
- [\[ICLR 2026\] Corner Gradient Descent](../../ICLR2026/optimization/corner_gradient_descent.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[ICLR 2026\] Fast Data Mixture Optimization via Gradient Descent](../../ICLR2026/optimization/fast_data_mixture_optimization_via_gradient_descent.md)

</div>

<!-- RELATED:END -->
