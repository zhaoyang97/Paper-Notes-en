---
title: >-
  [Paper Note] Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning
description: >-
  [ICML2026][Optimization][Single-Index Models] This paper provides a rigorous proof in Gaussian single-index models with quadratic activation: while full-batch gradient descent (GD) "reusing all data" under naive quadratic activation is no more sample-efficient than one-pass SGD (both requiring $n\gtrsim d\log d$), simply **truncating** the activation allows full-batch GD to achieve weak or even strong recovery with $n\gtrsim d$ (linear sample size). This establishes a $\log d…
tags:
  - "ICML2026"
  - "Optimization"
  - "Single-Index Models"
  - "Sample Complexity"
  - "Full-Batch Gradient Descent"
  - "Phase Retrieval"
  - "BBP Phase Transition"
date: 2026-05-08
content_hash: 8019c537545e679a
---

# Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning

**Conference**: ICML2026  
**arXiv**: [2602.02431](https://arxiv.org/abs/2602.02431)  
**Code**: To be confirmed  
**Area**: Learning Theory / Optimization  
**Keywords**: Single-Index Models, Sample Complexity, Full-Batch Gradient Descent, Phase Retrieval, BBP Phase Transition

## TL;DR
This paper provides a rigorous proof in Gaussian single-index models with quadratic activation: while full-batch gradient descent (GD) "reusing all data" under naive quadratic activation is no more sample-efficient than one-pass SGD (both requiring $n\gtrsim d\log d$), simply **truncating** the activation allows full-batch GD to achieve weak or even strong recovery with $n\gtrsim d$ (linear sample size). This establishes a $\log d$ sample complexity separation from one-pass SGD, which still requires $d\log d$.

## Background & Motivation

**Background**: There is a common folklore in machine learning that reusing training data multiple times (multi-pass / full-batch) improves the statistical efficiency of gradient methods. While this is well-understood in linear regression, the specific advantages and magnitude of improvement for multi-pass GD over one-pass online SGD in **nonlinear, non-convex** feature learning remain unclear.

**Limitations of Prior Work**: Single-index models $y=\sigma(\langle x,\theta^\star\rangle)$ serve as standard sandboxes for analyzing non-convex feature learning in shallow networks. For activations with an information exponent of 2 (including all even link functions like quadratic activation or phase retrieval), it is known that one-pass SGD requires $n\gtrsim d\log d$ samples to achieve weak recovery of $\theta^\star$, whereas the information-theoretic limit is only $n\gtrsim d$. Spectral methods and Approximate Message Passing (AMP) succeed at $n\gtrsim d$, but they are not gradient-based methods.

**Key Challenge**: Researchers aim to determine whether **standard full-batch gradient descent** (without modifying the algorithm or loss) can bridge this $\log d$ gap. The challenges are twofold: ① Most existing analyses of full-batch GD still require $n\gtrsim d\,\mathrm{polylog}\,d$, failing to prove superiority over one-pass SGD; ② Prior mechanisms explaining multi-pass advantages (e.g., Dandi et al. 2024b), which relate data reuse to label pre-processing/loss modification, do not apply to even activations as their generative exponent remains $\ge 2$, meaning pre-processing cannot eliminate the $\log d$ factor.

**Goal**: To rigorously characterize when full-batch GD can achieve weak and strong recovery with **linear sample complexity** in the proportional $n, d$ regime, and to specify the required iteration complexity.

**Key Insight**: The authors focus on the "minimizing correlation loss on a sphere" setting, where the $\log d$ barrier under SGD is considered inevitable. They first prove a negative result showing that naive quadratic activation is insufficient, then demonstrate how a minimal modification (truncating the activation) flips the situation.

**Core Idea**: Unbounded naive quadratic activation causes the empirical matrix spectrum to be dominated by noise, leading the top eigenvector to decorrelate with the signal. By **truncating the activation to bound the measurements**, the empirical Hessian along the optimization trajectory exhibits a consistent BBP phase transition (spectral gap + signal alignment). This enables recovery at $n\gtrsim d$, while one-pass SGD remains constrained by the $d\log d$ lower bound for the same truncated link, thus creating the separation.

## Method

### Overall Architecture
This is a purely theoretical paper where the "method" consists of three sets of theorems that dissect whether data reuse is beneficial. The setting is fixed: $n$ i.i.d. samples $x_i\sim\mathcal N(0,I_d)$, $y_i=\sigma(\langle x_i,\theta^\star\rangle)$, $\|\theta^\star\|=1$, with activation being quadratic $\sigma(z)=z^2$ or its truncated version. Evaluation is based on two levels: **weak recovery** $\lim_{n\to\infty}\frac{|\langle\hat\theta,\theta^\star\rangle|}{\|\hat\theta\|\,\|\theta^\star\|}=\epsilon>0$ (obtaining non-trivial correlation), and **strong recovery** $\lim_{n\to\infty}\min_{s\in\{\pm1\}}\|\hat\theta-s\theta^\star\|=0$ (exact recovery up to sign ambiguity). The baseline is one-pass SGD, which has a lower bound of $n\gtrsim d\log d$ for information exponent 2 (Ben Arous et al. 2021).

Two key quantities are central to the analysis. First, the **information exponent**: it characterizes the difficulty of "escaping the mediocre direction" near random initialization. For index 2 (e.g., $z^2$), the population gradient flow takes $T\gtrsim\log d$ time to escape, translating to $n\gtrsim d\log d$ for online SGD. Second, the **generative exponent** (Damian et al. 2024): it characterizes the remaining difficulty after any label pre-processing. For the quadratic/truncated quadratic activations here, this exponent is still 2, explaining why previous "multi-pass ≈ loss modification" mechanisms fail in this setting.

The authors proceed in three steps: a **negative result** on the sphere with correlation loss (naive quadratic activation), a **positive result** for the same setting with **truncation** (weak recovery @ $n\gtrsim d$), and finally a switch to Euclidean space with squared loss and small initialization to achieve **strong recovery + iteration complexity**.

| Theorem | Loss / Geometry | Activation | Conclusion | Sample Complexity |
|------|------------|------|------|--------|
| Thm 3.1 (Neg.) | Correlation / Spherical Flow | $z^2$ | Overlap $\to 0$ when $n=o(d\log d)$ | Req. $\gtrsim d\log d$ |
| Thm 3.2 (Pos.) | Correlation / Spherical Flow | Truncated $z^2$ | Weak recovery, overlap $\to 1$ | $n\gtrsim d$ |
| Thm 4 (Pos.) | Squared Loss / Euclidean GD | Truncated $z^2$ | Strong (Exact) recovery | $n\gtrsim d$, $T\gtrsim\log d$ steps |

### Key Designs

**1. Negative Result: Under naive quadratic activation, full-batch GD degrades to power iteration on a fixed matrix, still requiring $d\log d$**

This addresses the intuition that reusing data must be beneficial. For $\sigma(z)=z^2$, the spherical gradient flow of the correlation loss $\hat L(\theta)=-\frac1n\sum_i y_i\sigma(\langle x_i,\theta\rangle)$ is exactly equivalent to power iteration $\frac{d\theta}{dt}=(I-\theta\theta^\top)A_\star\theta$ on a **fixed matrix** $A_\star=\frac{2}{n}\sum_i y_i x_i x_i^\top$. Thus, $\theta(t)$ converges to the principal eigenvector $v_1(A_\star)$. The authors prove (Thm 3.1) via random matrix theory that when $n=o(d\log d)$, the **unbounded** quadratic activation causes the noise in the matrix to overwhelm the rank-one signal update, resulting in $|\langle v_1(A_\star),\theta^\star\rangle|\to 0$. In this case, full-batch GD offers **no statistical advantage** over one-pass SGD.

**2. Truncation triggers consistent BBP phase transitions, reducing the weak recovery threshold to $d$**

The failure of the naive case stems from the spectrum being dominated by heavy-tailed noise. The proposed fix is minimal: **truncate** the activation such that $\sigma(z)=\min\{z^2,M\}$. After truncation, the spherical gradient flow becomes power iteration on a **time-varying matrix** $A(\theta(t))$. The key observation is decomposing it as $A(\theta)=A_\star-B(\theta)$, where $B(\theta)\succeq0$ is supported only on the rare event $\{\langle x_i,\theta\rangle^2>M\}$ and can be controlled uniformly via VC-dimension arguments. Since measurements $y_i$ are now bounded, the concentration of the sub-Gaussian covariance matrix yields a **consistent BBP phase transition**: with high probability, $A(\theta)$ has a constant spectral gap for all $\theta$, and the top eigenvector aligns with $\theta^\star$ as $n/d$ grows. This allows $n\gtrsim d$ to suffice for recovery—a clear **separation** from one-pass SGD, which still requires $d\log d$ for the same link.

**3. Small initialization + squared loss upgrades weak recovery to strong recovery with $T\gtrsim\log d$ complexity**

The spherical analysis has two blind spots: it cannot specify iteration steps due to the time-varying nature of $A(\theta(t))$, and it cannot give strong recovery because the norm is constant. By switching to Euclidean GD on squared loss with **small initialization**, the authors show that GD first enters a "search phase" for $O(\log d/\eta)$ steps before geometrically converging to $\theta^\star$. Thm 4 provides the **first** strong recovery and convergence rate guarantee for unmodified full-batch GD in the information-theoretically optimal $n, d$ regime.

## Key Experimental Results

### Main Results: Threshold Behavior across Dimensions

| Setting | Phenomenon | Theorem Corroborated |
|------|-----------|-----------|
| Quadratic $\sigma(z)=z^2$ | $n/d$ threshold for fixed overlap increases with $d$ | Thm 3.1 (Requires $d\log d$) |
| Truncated $\sigma=\min\{z^2,M\}$ | Curves for different $d$ collapse; threshold $\delta=\Theta(1)$ | Thm 3.2 ($n\gtrsim d$ is sufficient) |
| Threshold Fitting | Required $\delta$ for target overlap is highly linear with $\log d$ | Quantifies the $\log d$ barrier |

### Key Findings
- **Truncation is the watershed**: Simply truncating the activation (a seemingly trivial change) shifts the weak recovery threshold from growing with $d$ to being constant, aligning with the "unbounded noise vs. consistent BBP" theory.
- **$\log d$ factor is quantitatively visible**: Empirical fitting confirms that the sample complexity for naive quadratic activation is indeed $\simeq \log d$.
- **Advantages stem from data reuse**: Unlike prior work focusing on over-parameterization, the separation here arises purely from full-batch data reuse in a model that is not over-parameterized.

## Highlights & Insights
- **Paired "Negative + Positive" Results**: Presenting both the failure of naive GD and the success of truncated GD makes the causal role of truncation undeniable.
- **Consistent BBP Transition as Technical Core**: Decomposing the time-varying $A(\theta)$ and using VC-dimension to control the perturbation $B(\theta)$ is a powerful technique for analyzing any non-convex problem where the spectrum changes along the trajectory.
- **Stable Manifold Theorem to exclude bad fixed points**: Combining Lyapunov functions with the Center Stable Manifold Theorem to prove that the basin of attraction for "bad" fixed points is a zero-measure set is a robust framework for non-convex analysis.
- **Truncation as "Cheap Regularization"**: Truncation does not change the information exponent (so SGD gains nothing), yet it makes the empirical spectrum for full-batch GD well-behaved.
- **Closing the Multi-epoch Theory Gap**: The motivation aligns with the empirical observation that training LLMs for multiple epochs is beneficial when data is limited, providing a clean nonlinear case where data reuse saves samples.

## Limitations & Future Work
- **Model Constraints**: Results are limited to Gaussian single-index models and quadratic-like activations. Generalization to information exponent $k$, multi-index models, or real-world distributions is unknown.
- **Coupling of $M$ and $n$**: Thm 3.2 requires $n\ge CM^4 d$. If $M$ is too small, truncation error dominates; if $M$ is too large, the required $n$ increases. 
- **Gradient Flow vs. GD**: Much of the work uses continuous-time flow; discrete-step effects are mainly addressed in the squared loss section under specific conditions.
- **Asymptotic Nature**: Results are stated in the $n,d\to\infty$ proportional limit; finite-dimensional fluctuations are not fully covered.

## Related Work & Insights
- **vs. One-pass Online SGD (Ben Arous et al. 2021)**: This paper establishes the separation where full-batch GD reaches $d$ while SGD is stuck at $d\log d$.
- **vs. "Two-pass = Modified Loss" (Dandi et al. 2024)**: Their mechanism fails for even link functions; this paper proposes a different mechanism based on consistent BBP transitions.
- **vs. Spectral Methods/AMP**: While these methods were known to succeed at $n\gtrsim d$, this work proves that **standard GD** can achieve the same sample optimality.
- **vs. Over-parameterization (Sarao Mannelli et al. 2020b)**: This paper clarifies that the advantage originates from data reuse itself rather than model over-parameterization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Single-Index Models via Harmonic Decomposition](../../NeurIPS2025/optimization/learning_single-index_models_via_harmonic_decomposition.md)
- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](../../ICLR2026/optimization/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)
- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](../../ICML2025/optimization/nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)
- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](../../ICML2025/optimization/improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)

</div>

<!-- RELATED:END -->
