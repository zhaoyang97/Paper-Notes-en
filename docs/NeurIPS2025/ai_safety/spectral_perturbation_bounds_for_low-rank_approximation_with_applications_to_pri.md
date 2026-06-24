---
title: >-
  [Paper Note] Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy
description: >-
  [NeurIPS 2025][AI Safety][low-rank approximation] This paper establishes novel high-probability perturbation bounds for low-rank approximation of symmetric matrices under the spectral norm, improving upon the classical Eckart–Young–Mirsky theorem, and resolves an open problem in differentially private PCA.
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "low-rank approximation"
  - "spectral norm perturbation bounds"
  - "differentially private PCA"
  - "contour bootstrapping"
  - "matrix perturbation theory"
date: 2026-05-08
content_hash: 4bba90d62ffa1248
---

# Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy

**Conference**: NeurIPS 2025
**arXiv**: [2510.25670](https://arxiv.org/abs/2510.25670)  
**Code**: None  
**Area**: AI Safety / Differential Privacy
**Keywords**: low-rank approximation, spectral norm perturbation bounds, differentially private PCA, contour bootstrapping, matrix perturbation theory

## TL;DR

This paper establishes novel high-probability perturbation bounds for low-rank approximation of symmetric matrices under the spectral norm, improving upon the classical Eckart–Young–Mirsky theorem, and resolves an open problem in differentially private PCA.

## Background & Motivation

Low-rank approximation is a foundational technique in machine learning and data science. When a matrix $A$ is perturbed by noise $E$, the central question is to quantify the effect of the perturbation on the top-$p$ approximation $A_p$, i.e., how large is $\|\tilde{A}_p - A_p\|$.

Limitations of existing metrics:
- **Frobenius norm**: may overestimate the noise effect by a factor of $\sqrt{p}$
- **Reconstruction error**: may severely underestimate subspace deviation — it can remain small even when the top-$p$ eigenspace undergoes a large rotation
- **Classical EYM bound**: $\|\tilde{A}_p - A_p\| \leq 2(\lambda_{p+1} + \|E\|)$, which is overly pessimistic

The spectral norm captures worst-case directional error and provides the strongest practical guarantee for downstream tasks such as PCA and clustering. Mangoubi and Vishnoi posed an open problem in Remark 5.3 of their NeurIPS paper: can high-probability spectral norm bounds on $\|\tilde{A}_p - A_p\|$ be obtained under natural conditions?

## Method

### Overall Architecture

The paper represents the top-$p$ eigenspace projector via contour integrals from complex analysis, and derives tighter perturbation bounds through a technique termed *contour bootstrapping*.

### Key Designs

1. **Main spectral norm bound (Theorem 2.1)**: Under the eigengap condition $\delta_p := \lambda_p - \lambda_{p+1} \geq 4\|E\|$, the paper proves $\|\tilde{A}_p - A_p\| \leq O(\|E\| \cdot \lambda_p / \delta_p)$. This improves over the EYM bound by a factor of $\min\{\lambda_{p+1}/\|E\|, \delta_p/\|E\|\}$. For example, when the spectrum is $\{10n, 9n, \ldots, n, n/2, 1, \ldots, 1\}$, EYM yields $O(n)$ whereas this paper yields $O(\sqrt{n})$.

2. **Interaction-aware refined bound (Theorem 2.2)**: By introducing a *half-decay distance* $r$ and a noise–eigenspace interaction term $x = \max_{i,j \leq r} |u_i^\top E u_j|$, the paper obtains $\|\tilde{A}_p - A_p\| \leq \tilde{O}(\|E\| + r^2 x \cdot \lambda_p / \delta_p)$. When the matrix has low stable rank or the noise is approximately orthogonal to the principal eigenspace, this yields a further improvement of up to $\sqrt{n}$.

3. **Generalization to spectral functions (Theorem 2.3)**: The results are extended to general spectral functions $f(A)$, including polynomials, exponentials, and trigonometric functions, via a natural generalization of contour integration. This covers $f(z) = z^k, \exp(z), \cos(z)$, and related cases.

### Loss & Training

This is a theoretical paper. The core technical contribution is **contour bootstrapping** (Lemma 3.1):
- The rank-$p$ projector is represented via contour integrals in the complex plane
- Iterative bootstrapping steps progressively tighten the bound
- This avoids classical power-series expansions or Davis–Kahan-style arguments
- The technique generalizes the approach of tran2025davis for handling eigenspace perturbations

## Key Experimental Results

### Main Results

| Method | Norm Type | Noise Model | Probabilistic Guarantee | Extra Factor vs. $\|E\|$ |
|--------|-----------|-------------|------------------------|--------------------------|
| Classical EYM | Spectral | Arbitrary | High-probability | $\lambda_{p+1}/\|E\|$ |
| Mangoubi–Vishnoi | Frobenius | Complex Gaussian | In expectation | $\sqrt{p} \cdot \lambda_p/\delta_p$ |
| **Ours, Thm 2.1** | Spectral | Arbitrary symmetric | High-probability | $\lambda_p/\delta_p$ |
| **Ours, Thm 2.2** | Spectral | Arbitrary symmetric | High-probability | $r^2 x \cdot \lambda_p/\delta_p$ |

### Ablation Study

| Configuration | Metric | Notes |
|---------------|--------|-------|
| Gaussian noise | Spectral error | Predicted values closely track actual error |
| Rademacher noise | Spectral error | Equally accurate, validating non-Gaussian applicability |
| Varying dimension $n$ | Relative error | Improvement becomes more pronounced as $n$ grows |

### Key Findings

- The proposed bounds closely match actual spectral errors on real covariance matrices
- Improvements over the classical EYM baseline are consistent across datasets and noise regimes
- Corollary 2.4 provides the first high-probability spectral norm guarantee for differentially private PCA

## Highlights & Insights

- Resolves an open problem in the literature (Mangoubi–Vishnoi, Remark 5.3)
- Contour bootstrapping is a novel analytical technique that brings complex analysis into matrix perturbation theory
- Results apply to both real- and complex-valued Wigner noise, generalizing beyond the complex Gaussian setting of prior work
- Elimination of the $\sqrt{p}$ factor has practical significance for high-dimensional differentially private applications

## Limitations & Future Work

- The eigengap condition $4\|E\| \leq \delta_p$, while mild, remains a necessary assumption
- Theorem 2.3 does not apply to non-entire functions such as $f(z) = z^c$ for non-integer $c$, due to singularity issues
- Specific privacy mechanisms such as quantized Gaussian or Laplace noise are not analyzed
- Constants are not optimized (bound $<7$) and could be tightened in future work

## Related Work & Insights

- **Relation to Davis–Kahan**: This paper offers a complementary perspective — Davis–Kahan controls subspace angles, whereas this work controls matrix approximation error
- **Implications for differential privacy**: The spectral norm is a more appropriate utility metric than the Frobenius norm or reconstruction error
- **Potential applications of contour bootstrapping**: The technique may extend to perturbation analysis of a broader class of matrix functions

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Resolves an open problem and introduces a fundamentally new analytical technique
- **Experimental Thoroughness**: ⭐⭐⭐ Empirical validation is present but limited in scale (primarily a theoretical contribution)
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, precise theorem statements, and thorough comparisons
- **Value**: ⭐⭐⭐⭐ Significant contributions to both matrix perturbation theory and differential privacy

## Supplementary Details

- The eigengap condition $4\|E\| \leq \delta_p$ holds in several classical models: spiked covariance, stochastic block model, and kernel matrices
- Empirical analysis (citing DBM NeurIPS Section B) shows that the 1990 Census and UCI Adult datasets satisfy the eigengap condition
- Corollary 2.4 further derives high-probability Frobenius norm and reconstruction error bounds
- Compared to Mangoubi–Vishnoi, the paper also eliminates the $\log^{\log\log n} n$ factor and the restrictive assumption $\lambda_1 \leq n^{50}$

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Video Unlearning via Low-Rank Refusal Vector](../../ICLR2026/ai_safety/video_unlearning_via_low-rank_refusal_vector.md)
- [\[NeurIPS 2025\] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](sequentially_auditing_differential_privacy.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
