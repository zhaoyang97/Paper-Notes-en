---
title: >-
  [Paper Note] Information-Computation Tradeoffs for Noiseless Linear Regression with Oblivious Contamination
description: >-
  [NeurIPS 2025][Information-computation tradeoff] For noiseless linear regression under the oblivious contamination model, this paper formally proves that any efficient Statistical Query algorithm requires VSTAT complexit…
tags:
  - "NeurIPS 2025"
  - "Information-computation tradeoff"
  - "robust linear regression"
  - "oblivious contamination"
  - "statistical query"
  - "computational lower bounds"
date: 2026-05-08
content_hash: 7a62149b9e3ee620
---

# Information-Computation Tradeoffs for Noiseless Linear Regression with Oblivious Contamination

**Conference**: NeurIPS 2025
**arXiv**: [2510.10665](https://arxiv.org/abs/2510.10665)  
**Code**: None  
**Area**: Statistical Learning Theory / Robust Optimization
**Keywords**: Information-computation tradeoff, robust linear regression, oblivious contamination, statistical query, computational lower bounds

## TL;DR

For noiseless linear regression under the oblivious contamination model, this paper formally proves that any efficient Statistical Query algorithm requires VSTAT complexity at least $\tilde{\Omega}(d^{1/2}/\alpha^2)$, providing evidence that the quadratic dependence on $1/\alpha$ constitutes an essential computational lower bound for efficient algorithms.

## Background & Motivation

### Problem Setup

Consider the following noiseless linear regression problem: given $n$ i.i.d. samples $(x_i, y_i)$ where:
- $x \sim \mathcal{N}(0, \mathbf{I}_d)$ (Gaussian covariates)
- $y = x^\top \beta + z$, with $z$ independent of $x$, drawn from an unknown distribution $E$
- **Key assumption**: $\mathbb{P}_E[z=0] = \alpha > 0$ (a fraction $\alpha$ of samples are "clean")

The goal is to recover the regression coefficient $\beta$ with small $\ell_2$ error.

### Information-Computation Gap

This problem exhibits a significant **information-computation gap**:

- **Information-theoretic upper bound**: Without computational constraints, $O(d/\alpha)$ samples suffice to recover $\beta$
- **Known efficient algorithms**: Polynomial-time algorithms require $\Omega(d/\alpha^2)$ samples
- **Gap**: $1/\alpha$ vs. $1/\alpha^2$, a factor of $1/\alpha$

The central question is whether this quadratic gap is accidental (existing algorithms are suboptimal) or inherent (no efficient algorithm can do better).

### Motivation

The oblivious contamination model is one of the foundational models in robust statistics, lying between the uncontaminated and adversarial contamination settings. Understanding its computational complexity helps to:
1. Delineate the boundary of computational feasibility in robust estimation
2. Guide algorithm design — if the lower bound is tight, there is no need to seek better polynomial-time algorithms
3. Deepen understanding of information-computation tradeoffs as a fundamental phenomenon

## Method

### Overall Architecture

The core technical contribution is establishing **Statistical Query (SQ) lower bounds**, proving that in the SQ computational model, recovering $\beta$ requires VSTAT complexity (query complexity) of at least $\tilde{\Omega}(d^{1/2}/\alpha^2)$.

Technical approach:
1. Reduce the regression problem to a hypothesis testing problem
2. Apply the standard SQ lower bound framework (based on statistical distance / chi-squared divergence)
3. Construct a "hard" family of oblivious contamination distributions such that distinguishing them requires a large number of SQ queries

### Key Designs

**1. SQ Model and VSTAT**

The SQ model is the standard tool for analyzing computational-statistical tradeoffs:
- Algorithms cannot access samples directly; they may only submit statistical queries of the form "what is $\mathbb{E}[\phi(x,y)]$?"
- VSTAT($n$) oracle: responds to each query with accuracy $O(1/n)$
- VSTAT complexity measures how many "effective samples" an algorithm requires
- Many known polynomial-time algorithms can be implemented within the SQ model

**2. Construction of Hard Distribution Families**

The core technical challenge is constructing a family of oblivious contamination distributions such that:
- Each distribution corresponds to a different $\beta$
- Any two distributions are "hard to distinguish" in the SQ sense
- The VSTAT complexity required for distinguishing achieves the target lower bound

The construction exploits high-dimensional geometry and special properties of Gaussian distributions:
- $\beta$ is embedded on a high-dimensional sphere
- The moment structure of the contamination distribution $E$ is carefully designed to "hide" information about $\beta$
- Key lemma: when $\alpha$ is small, distinguishing joint distributions corresponding to different $\beta$ requires precision $\Omega(1/\alpha^2)$

**3. Chi-Squared Divergence Analysis**

The core of the lower bound proof is bounding the average chi-squared divergence over the hard distribution family:
- Show that $\chi^2$-divergence is sufficiently small under low-precision SQ queries
- Thereby derive the VSTAT complexity lower bound
- Technically requires careful analysis of high-dimensional Gaussian integrals and moment estimates

### Loss & Training

This is a theoretical paper with no training involved. The core theorem is:

**Theorem (Main Result)**: Any SQ algorithm that recovers $\beta$ to constant $\ell_2$ error from the above model requires VSTAT complexity at least:

$$\tilde{\Omega}\left(\frac{d^{1/2}}{\alpha^2}\right)$$

## Key Experimental Results

### Theoretical Results Summary

This is a purely theoretical paper; "experiments" are presented as theorems. Core results:

| Algorithm Type | Sample Complexity | Polynomial Time? | Source |
|---|---|---|---|
| Information-theoretically optimal | $O(d/\alpha)$ | No (exhaustive search) | Known result |
| Filter-based | $O(d/\alpha^2)$ | Yes | Diakonikolas et al. |
| Robust regression | $O(d/\alpha^2)$ | Yes | Known result |
| **SQ lower bound** | $\tilde{\Omega}(d^{1/2}/\alpha^2)$ | — | **Ours** |

### Comparison with Related Lower Bounds

| Problem | Information-Theoretic | Known Efficient | SQ Lower Bound | Gap |
|---|---|---|---|---|
| Gaussian mean estimation (oblivious) | $O(d/\alpha)$ | $O(d/\alpha^2)$ | $\tilde{\Omega}(d/\alpha^2)$ | Closed |
| Linear regression (adversarial) | $O(d/\epsilon)$ | $O(d/\epsilon^2)$ | $\tilde{\Omega}(d/\epsilon^2)$ | Closed |
| **Linear regression (oblivious)** | $O(d/\alpha)$ | $O(d/\alpha^2)$ | $\tilde{\Omega}(d^{1/2}/\alpha^2)$ | **Partially closed** |

The lower bound matches the known upper bound in the $1/\alpha$ dependence (quadratic), but a gap of $d^{1/2}$ vs. $d$ remains in the dependence on $d$.

### Key Findings

1. **The quadratic dependence on $1/\alpha^2$ is essential**: No polynomial-time SQ algorithm can solve this problem with $O(d/\alpha^{2-\delta})$ samples.
2. **Gap remains in the $d$ dependence**: The $d^{1/2}$ factor in the lower bound is weaker than the $d$ factor in the upper bound; closing this gap is an open problem.
3. **Oblivious vs. adversarial contamination**: Despite the oblivious model being strictly weaker than adversarial contamination, the computational complexity exhibits the same quadratic dependence.

## Highlights & Insights

1. **Fills a theoretical gap**: The information-computation gap for oblivious linear regression previously lacked formal evidence; this paper provides the first SQ lower bound.
2. **Technical novelty in the proof**: Handling the complex dependencies inherent in the product structure $y = x^\top\beta + z$ is substantially more difficult than standard mean estimation lower bound proofs.
3. **Foundational significance**: Oblivious contamination is a classical model in semiparametric statistics; the results offer guidance for a broader class of robust estimation problems.
4. **Clear open problems**: The paper explicitly identifies the gap in the $d$ dependence, providing direction for future work.

## Limitations & Future Work

1. **Gap in the $d$ dependence**: Lower bound $d^{1/2}/\alpha^2$ vs. upper bound $d/\alpha^2$ leaves a $\sqrt{d}$ factor to be closed.
2. **Limitations of the SQ model**: SQ lower bounds do not rule out non-SQ algorithms (e.g., based on the SOS hierarchy) that might circumvent the lower bound.
3. **Restriction to Gaussian covariates**: Whether the lower bounds hold for non-Gaussian or sub-Gaussian covariates remains open.
4. **Constant $\ell_2$ error only**: No fine-grained sample-error tradeoff is given for recovery to precision $\epsilon$.
5. Theoretical results are not validated by numerical experiments.

## Related Work & Insights

- **Diakonikolas et al. (2019)**: SQ lower bounds for robust mean estimation under adversarial contamination; the technical foundation of this paper.
- **Gao et al. (2020)**: Information-theoretic upper bounds for mean estimation under oblivious contamination.
- **Brennan & Bresler (2020)**: A general framework for computational-statistical gaps in planted problems.
- **Diakonikolas et al. (2023)**: SQ lower bounds for regression under adversarial contamination.
- Insight: **Information hiding in the product structure $x^\top\beta + z$** is the key to understanding the computational complexity of robust regression.

## Rating

- Novelty: ⭐⭐⭐⭐ (first SQ lower bound for oblivious linear regression)
- Technical Depth: ⭐⭐⭐⭐⭐ (highly involved proof techniques)
- Experimental Thoroughness: ⭐⭐⭐ (purely theoretical; no numerical validation)
- Practical Impact: ⭐⭐ (purely theoretical results)
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[NeurIPS 2025\] AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking](acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](regression_trees_know_calculus.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](../../AAAI2026/others/online_linear_regression_with_paid_stochastic_features.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)

</div>

<!-- RELATED:END -->
