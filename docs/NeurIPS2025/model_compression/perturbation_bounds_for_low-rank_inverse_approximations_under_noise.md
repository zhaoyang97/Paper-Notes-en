---
title: >-
  [Paper Note] Perturbation Bounds for Low-Rank Inverse Approximations under Noise
description: >-
  [NeurIPS 2025][Model Compression][low-rank inverse approximation] This work derives the first non-asymptotic spectral norm perturbation bounds for low-rank inverse approximations $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ under noise, via a novel *contour bootstrapping* technique that handles the non-entire function $f(z) = 1/z$. Under favorable conditions, the proposed bounds improve upon classical bounds by a factor of $\sqrt{n}$.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "low-rank inverse approximation"
  - "matrix perturbation theory"
  - "spectral norm bounds"
  - "contour integration"
  - "eigengap"
date: 2026-05-08
content_hash: f81d0610b0d37404
---

# Perturbation Bounds for Low-Rank Inverse Approximations under Noise

**Conference**: NeurIPS 2025
**arXiv**: [2510.25571](https://arxiv.org/abs/2510.25571)  
**Code**: None  
**Area**: Signal & Communications / Numerical Linear Algebra
**Keywords**: low-rank inverse approximation, matrix perturbation theory, spectral norm bounds, contour integration, eigengap

## TL;DR

This work derives the first non-asymptotic spectral norm perturbation bounds for low-rank inverse approximations $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ under noise, via a novel *contour bootstrapping* technique that handles the non-entire function $f(z) = 1/z$. Under favorable conditions, the proposed bounds improve upon classical bounds by a factor of $\sqrt{n}$.

## Background & Motivation

1. **Background**: Low-rank matrix approximation is a foundational tool in machine learning and scientific computing. The inverse $A^{-1}$ of a large symmetric matrix $A$ is commonly approximated by its low-rank surrogate $A_p^{-1}$, with widespread applications in kernel methods, Gaussian processes, and covariance inference.

2. **Limitations of Prior Work**: In practice, only a noisy observation $\tilde{A} = A + E$ is available. Classical perturbation theory (Neumann expansion) provides bounds only for the full inverse $\|A^{-1} - \tilde{A}^{-1}\|$, without accounting for the effect of truncation to rank $p$. Bounds derived from the Eckart–Young–Mirsky theorem (the EYM-N bound) contain a constant term $2/\lambda_{n-p}$ that does not vanish even as $\|E\| \to 0$.

3. **Key Challenge**: It remains unclear how the error in low-rank inverse approximation depends on the noise level $\|E\|$, the eigengap $\delta_{n-p}$, and the spectral decay, in an interacting manner. Classical bounds are overly pessimistic and do not distinguish spectral structure.

4. **Goal**: Derive tight, spectrally adaptive, non-asymptotic bounds for $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$.

5. **Key Insight**: Apply contour integration techniques to the non-entire function $f(z) = 1/z$ (previously restricted to entire functions such as the matrix exponential), and reduce higher-order terms to first order via contour bootstrapping.

6. **Core Idea**: Design carefully chosen contours encircling the smallest $p$ eigenvalues, and exploit perturbation control of Riesz projectors to obtain tight bounds that depend on the eigengap and noise alignment.

## Method

### Overall Architecture

The proof follows a three-step framework:
1. Express the perturbation $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ via contour integration.
2. Apply the contour bootstrapping lemma to establish $F \leq 2F_1$, reducing the problem to first-order terms.
3. Construct tailored contours so that the resulting integral can be evaluated explicitly.

### Key Designs

**1. Contour Integral Representation and Bootstrapping Lemma (Lemma 3.1)**

- **Function**: Reduces the low-rank inverse perturbation problem to a tractable first-order integral.
- **Mechanism**: A contour $\Gamma$ is chosen in the complex plane to encircle the smallest $p$ eigenvalues of $A$, avoiding $z = 0$ and the larger eigenvalues. By the residue theorem, $A_p^{-1} = \frac{1}{2\pi i}\oint_\Gamma z^{-1}(zI-A)^{-1}dz$. The key lemma establishes that under the condition $4\|E\| \leq \min\{\lambda_n, \delta_{n-p}\}$, one has $F \leq 2F_1$, where $F_1$ involves only the first-order resolvent expansion $z^{-1}(zI-A)^{-1}E(zI-A)^{-1}$.
- **Design Motivation**: Traditional Neumann-series estimates of each $F_s$ yield only $O(\|E\|/\delta_{n-p}^2)$, which is suboptimal. The bootstrapping lemma shows that the dominant term $F_1$ suffices to control the full perturbation, since higher-order terms are suppressed geometrically on the contour.

**2. Main Theorem (Theorem 2.1)**

- **Function**: Provides an interpretable two-term bound.
- **Mechanism**: For a positive definite matrix $A$, under the condition $4\|E\| \leq \min\{\lambda_n, \delta_{n-p}\}$:
$$\|(\tilde{A}^{-1})_p - A_p^{-1}\| \leq \frac{4\|E\|}{\lambda_n^2} + \frac{5\|E\|}{\lambda_{n-p}\delta_{n-p}}$$
The first term corresponds to the classical scaling of the full inverse $\|E\|/\lambda_n^2$; the second captures the additional sensitivity from projection onto the subspace of smallest eigenvalues.
- **Design Motivation**: The bound vanishes as $\|E\| \to 0$ (which the EYM-N bound cannot achieve) and depends explicitly on the eigengap $\delta_{n-p}$.

**3. Tailored Contour Construction**

- **Function**: Enables explicit evaluation of the first-order integral $F_1$.
- **Mechanism**: Contours $\Gamma$ of a specific geometry are constructed such that the distances to the smallest $p$ eigenvalues and to $z = 0$ are controllable. Weyl's inequality ensures that the eigenvalue ordering of $\tilde{A}$ is preserved under perturbation.
- **Design Motivation**: The geometry of the contour governs the tightness of the integral bound, and must simultaneously satisfy: (i) encircling the correct eigenvalues, (ii) maintaining sufficient distance from $z = 0$ (the singularity of the non-entire function), and (iii) being tight enough to yield optimal constants.

### Loss & Training

(This is a purely theoretical work; no training procedure is involved.)

## Key Experimental Results

### Main Results

**Synthetic matrices (spectrum $\{n, 2n, \ldots, 10n, 20n, \ldots, 20n\}$, $p = 10$, Wigner noise)**

| Method | Bound |
|--------|-------|
| EYM-N bound | $O(1/n)$ |
| **Ours** | $O(n^{-3/2})$ |
| Improvement factor | $\sqrt{n}$ |

**Real matrix experiments**

| Dataset | Ours / True Error | EYM-N / True Error |
|---------|------------------|--------------------|
| 1990 US Census covariance | ≤10× | 100–1000× |
| BCSSTK09 stiffness matrix | ≤10× | 100–1000× |
| Synthetic covariance matrix | ≤10× | 10–100× |
| Elliptic operator discretization | ≤5× | 50–500× |

### Ablation Study

| Condition | Effect on Bound |
|-----------|----------------|
| $p = n$ (full inverse) | Second term vanishes; recovers Neumann bound $O(\|E\|/\lambda_n^2)$ |
| $\delta_{n-p} \gg \|E\|$ | Second term negligible; dominant term is $O(\|E\|/\lambda_n^2)$ |
| Wigner noise $\|E\| = O(\sqrt{n})$ | Bound is $O(\sqrt{n}/\lambda_n^2 + \sqrt{n}/\lambda_{n-p}\delta_{n-p})$ |
| Differential privacy noise | Safety margin is sufficient; condition easily satisfied |

### Key Findings

- The proposed bound tracks the true error closely on real data (within a constant factor ≤10), whereas the EYM-N bound overestimates by 1–2 orders of magnitude.
- A $\sqrt{n}$-fold improvement is achieved under favorable spectral conditions (large eigengap).
- The gap condition $4\|E\| < \delta_{n-p}$ is widely satisfied for practical matrices.
- Applied to preconditioned conjugate gradient (PCG), the bound implies an $n^{1/4}$ improvement in iteration count.

## Highlights & Insights

- This work derives the first non-asymptotic spectral norm bounds for low-rank inverse approximations under general noise.
- The contour bootstrapping technique elegantly unifies the non-entire function case $f(z) = 1/z$ with the treatment of entire functions.
- The two-term structure of the bound has a clear physical interpretation: the first term captures full-inverse sensitivity, and the second captures subspace sensitivity.
- The practical utility is concrete: the bounds provide a quantitative criterion for assessing the reliability of low-rank inverses in noisy settings.

## Limitations & Future Work

- The gap condition $4\|E\| < \delta_{n-p}$ may fail for nearly degenerate matrices with clustered eigenvalues.
- The current analysis is restricted to symmetric matrices; the non-symmetric case is not covered.
- The asymptotically refined bounds in Section 5 are not experimentally validated.
- The constants 4 and 5 may admit further optimization.

## Related Work & Insights

- The classical Neumann expansion and the Davis–Kahan $\sin(\theta)$ theorem serve as direct foundations.
- The contour bootstrapping technique of Tran & Vishnoi (2025) is extended from entire functions to rational functions.
- For machine learning: when using Nyström approximations or sketch-based preconditioners, the proposed bounds can provide tighter error guarantees.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Applying contour bootstrapping to non-entire functions is a novel mathematical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The tightness of the bounds is validated on both synthetic and real data.
- **Writing Quality**: ⭐⭐⭐⭐ The three-step proof framework is clearly organized, though the mathematical density is high.
- **Value**: ⭐⭐⭐⭐ Fills an important theoretical gap in the analysis of noisy low-rank inverse approximations; the PCG application demonstrates practical relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FiRA: Can We Achieve Full-Rank Training of LLMs Under Low-Rank Constraint?](fira_can_we_achieve_full-rank_training_of_llms_under_low-rank_constraint.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)

</div>

<!-- RELATED:END -->
