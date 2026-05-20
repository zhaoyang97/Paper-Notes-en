---
title: >-
  [Paper Note] Kernel Conditional Tests from Learning-Theoretic Bounds
description: >-
  [NeurIPS 2025][conditional hypothesis testing] A unified framework is proposed for converting confidence bounds of learning algorithms into conditional hypothesis tests. Built upon kernel ridge regression…
tags:
  - "NeurIPS 2025"
  - "conditional hypothesis testing"
  - "kernel ridge regression"
  - "confidence bounds"
  - "conditional distribution functionals"
  - "bootstrapping"
date: 2026-05-08
content_hash: 09b35c582f99877d
---

# Kernel Conditional Tests from Learning-Theoretic Bounds

**Conference**: NeurIPS 2025
**arXiv**: [2506.03898](https://arxiv.org/abs/2506.03898)  
**Code**: Unavailable  
**Area**: Statistical Testing / Kernel Methods / Learning Theory
**Keywords**: conditional hypothesis testing, kernel ridge regression, confidence bounds, conditional distribution functionals, bootstrapping
**Authors**: Pierre-François Massiani, Christian Fiedler, Lukas Haverbeck, Friedrich Solowjow, Sebastian Trimpe
**Affiliation**: RWTH Aachen University, TU Munich

## TL;DR

A unified framework is proposed for converting confidence bounds of learning algorithms into conditional hypothesis tests. Built upon kernel ridge regression, the framework yields conditional two-sample tests with finite-sample guarantees and, for the first time, supports non-i.i.d. data and online sampling scenarios.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Determining whether two systems share the same conditional relationship given a common input is a fundamental problem in science and engineering—arising, for instance, in detecting dynamical changes, assessing equipment performance under different operating conditions, or comparing treatment responses stratified by patient characteristics. Existing conditional testing methods suffer from three key limitations: (i) they detect only marginal distribution differences rather than conditional ones; (ii) they yield only global discrepancy decisions without identifying *at which covariate values* differences occur; (iii) they provide only asymptotic guarantees, lacking finite-sample validity. Moreover, nearly all existing methods rely on i.i.d. assumptions and cannot handle online or sequential sampling.

The paper's core insight is that **learning precedes testing**—confidence bounds from learning algorithms can directly yield guarantees for conditional tests. This establishes a systematic connection between source conditions in learning theory and the "prior sets" used in testing. Kernel ridge regression (KRR) is chosen to instantiate the framework, and the Uniformly Block-Diagonal (UBD) kernel assumption is introduced to handle infinite-dimensional output spaces.

## Method

### Overall Architecture

The framework operates at three levels: (1) formalizing conditional testing with covariate rejection regions and defining the types of finite-sample guarantees; (2) converting confidence bounds of an arbitrary learning method into a two-sample test for conditional expectations (Theorem 4.2); (3) extending the expectation test to more general conditional distribution functionals via kernel mean embeddings (Section 4.3).

### Key Designs

1. **Covariate Rejection Regions and Guarantee Types (Section 3)**: A conditional hypothesis $H: \mathcal{X} \times \Theta \to \{0,1\}$ and a covariate rejection region $\chi(D)$ are defined. Unlike conventional methods that take the maximum over the entire space, the proposed approach preserves spatial resolution, enabling precise localization of discrepancies. An $(\mathcal{S}, \mathcal{N})$-guarantee is defined, unifying pointwise, regional, and time-uniform guarantees. This allows the same test to be applied either at a single covariate point or across the entire rejection region with consistent validity.

2. **From Confidence Bounds to Tests (Theorem 4.2)**: If learning method $\mathfrak{L}_i$ admits a confidence bound satisfying $\|f_{D_p}^{(i)}(x) - \mathbb{E}(p)(x)\| \leq B_i(D,x)$, then the test statistic is $\|f_{D_1}^{(1)}(x) - f_{D_2}^{(2)}(x)\|$ with threshold $B_1(D_1,x) + B_2(D_2,x)$. This theorem fully formalizes the relationship between the estimation problem and the testing problem.

3. **Vector-Valued KRR Confidence Bounds—UBD Kernel (Theorem 4.3)**: This constitutes the primary technical contribution. For the KRR estimator $f_{D,\lambda}$, the bound $\|f_{D_{:n}}(x) - \mathbb{E}[p](x)\| \leq \beta_\lambda \cdot \sigma_{D_{:n},\lambda}(x)$ is established. The key innovation is the introduction of the **Uniformly Block-Diagonal (UBD) kernel** $K = \iota_\mathcal{G}(K_0 \otimes \mathrm{id}_\mathcal{V})\iota_\mathcal{G}^{-1}$, which requires only the base block $K_0$ to be trace-class rather than $K$ itself, substantially relaxing the assumptions. Two variants of the bound are provided for UBD kernels: a time-uniform bound (Case 1) and an independent-data bound (Case 2).

4. **Functional Testing and Kernel Mean Embeddings (Section 4.3)**: Via the representation map $\Phi_\mathcal{F}: z \mapsto \kappa(\cdot,z)$, data are transformed as $Y_n = \Phi_\mathcal{F}(Z_n)$, reducing the problem to a conditional expectation test. Different choices of kernel $\kappa$ target different properties: polynomial kernels test moments, Gaussian kernels test the full distribution. Even when $\mathcal{G}$ is infinite-dimensional, all computations remain tractable under the diagonal kernel $K = k \cdot \mathrm{id}_{\mathcal{H}_\kappa}$.

### Loss & Training

The standard KRR objective is $f_{D,\lambda} = \arg\min_f \sum_n \|y_n - f(x_n)\|^2 + \lambda\|f\|_K^2$. The theoretical expression for the threshold $\beta_\lambda$ depends on inaccessible parameters (RKHS norm bound $S$, sub-Gaussian constant $\rho$). Two bootstrapping schemes are provided: (a) Naive resampling—repeated sampling of pairs on the same dataset to compute the minimum consistent $\beta$; (b) Wild bootstrap—random perturbation of residuals to avoid repeated KRR refitting, resulting in lower computational cost.

## Key Experimental Results

### Main Results

| Experimental Setting | Metric | Ours (bootstrap) | Baseline (Hu & Lei 2024) | Notes |
|---|---|---|---|---|
| 2D function discrepancy, n=100 | Type I error | ≤α (well-controlled) | ≤α | Both control Type I error |
| 2D function discrepancy, ξ=2 | Type II error | ~0.15 | ~0.15 | Comparable; proposed method requires no knowledge of noise distribution |
| Low-sampling region, θ=0.05 | Type II error | ~0.30 | ~0.55 | Local testing substantially superior |
| Process monitoring, d=4, ξ=0.5 | Detection rate | Reliable detection post-perturbation | N/A | Supports correlated sampling |

### Ablation Study

| Configuration | Type I error | Type II error | Notes |
|---|---|---|---|
| Analytic threshold | ≪α (overly conservative) | Nearly independent of α | Extremely low power |
| Naive bootstrap | Tracks α well | Significantly reduced | Large power improvement |
| Wild bootstrap | Slightly below α (conservative) | Moderate | Robust but slightly lower power |
| Gaussian κ for mean difference | ~α | Higher than linear kernel | Overly powerful kernel reduces efficiency for low-order moments |
| Linear κ for mean difference | ~α | Low | Targeted kernel more efficient |

### Key Findings

- **Locality is the core advantage**: The proposed method substantially outperforms global methods in low-sampling regions, as the confidence bounds are inherently local.
- **Bootstrapping vs. analytic threshold**: Bootstrap reduces Type II error from near 1 to a meaningful level, which is critical for practical applicability.
- **Choice of kernel κ is fundamentally a power-generality trade-off**: Characteristic kernels (e.g., Gaussian) can detect arbitrary distributional differences but may be less powerful than specialized kernels for detecting specific moments.

## Highlights & Insights

- The conceptual link from learning to testing is a substantive original contribution, enabling any learning method with confidence bounds to automatically yield conditional tests.
- The UBD kernel assumption elegantly resolves the technical challenge of infinite-dimensional output spaces, making conditional two-sample tests computationally tractable.
- Time-uniform guarantees naturally support sequential testing and drift detection.
- The bootstrapping schemes leverage the parametric structure identified by theory, balancing theoretical motivation with practical usability.

## Limitations & Future Work

- The bootstrapping schemes currently carry only heuristic guarantees; rigorous finite-sample coverage analysis is lacking.
- The RKHS membership assumption (i.e., the conditional expectation belongs to the RKHS) restricts the class of detectable functions.
- In the process monitoring experiment, performance degrades in high dimensions, possibly because the perturbed dynamical system drives the data into sparse regions.
- Concrete instantiation for conditional independence testing is left for future work.

## Related Work & Insights

- The work generalizes the scalar KRR confidence bounds of Abbasi-Yadkori (2013) to the UBD kernel setting, with direct applicability to Bayesian optimization.
- It integrates seamlessly with conditional mean embeddings (Grünewälder et al. 2012), enriching the distributional regression toolbox.
- The bootstrapping strategy draws on the anti-symmetric multiplier approach of Singh & Vijaykumar (2023), with modifications to the standard error term.

## Key Formula Reference

- Test statistic: $\|f_{D_1}^{(1)}(x) - f_{D_2}^{(2)}(x)\|$
- Test threshold: $\sum_{i=1}^2 \beta_i \cdot \sigma_{D_i,\lambda_i}(x)$
- UBD kernel definition: $K = \iota_\mathcal{G}(K_0 \otimes \mathrm{id}_\mathcal{V})\iota_\mathcal{G}^{-1}$
- Confidence bound: $\|f_{D_{:n}}(x) - \mathbb{E}[p](x)\| \leq \beta_\lambda \cdot \sigma_{D_{:n},\lambda}(x)$ w.p. $\geq 1-\delta$

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The learning-to-testing connection is an original conceptual contribution; the UBD kernel assumption is a significant technical innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic data, parameter ablations, and a process monitoring application, with systematic analysis of kernel selection.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theory, algorithms, and experiments are highly integrated, with clear organization of material.
- **Value**: ⭐⭐⭐⭐ Provides a unified and practically actionable framework for conditional testing, with both theoretical and practical merit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Tight Bounds On the Distortion of Randomized and Deterministic Distributed Voting](tight_bounds_on_the_distortion_of_randomized_and_deterministic_distributed_votin.md)
- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](../../ICLR2026/others/predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ICCV 2025\] HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding](../../ICCV2025/others/hytip_hybrid_temporal_information_propagation_for_masked_conditional_residual_vi.md)

</div>

<!-- RELATED:END -->
