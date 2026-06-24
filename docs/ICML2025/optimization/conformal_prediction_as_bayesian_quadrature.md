---
title: >-
  [Paper Note] Conformal Prediction as Bayesian Quadrature
description: >-
  [ICML 2025 Oral][Optimization][Conformal prediction] Revisiting conformal prediction from a Bayesian perspective—proving that both split conformal prediction and conformal risk control are special cases of the Bayesian Quadrature framework, proposing practical Bayesian alternatives, and providing interpretable guarantees as well as a richer representation of future loss ranges.
tags:
  - "ICML 2025 Oral"
  - "Optimization"
  - "Conformal prediction"
  - "Bayesian quadrature"
  - "uncertainty quantification"
  - "probabilistic numerical methods"
  - "distribution-free"
date: 2026-05-08
content_hash: b65fcc2cc32dcb87
---

# Conformal Prediction as Bayesian Quadrature

**Conference**: ICML 2025 Oral  
**arXiv**: [2502.13228](https://arxiv.org/abs/2502.13228)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Conformal prediction, Bayesian quadrature, uncertainty quantification, probabilistic numerical methods, distribution-free

## TL;DR
Revisiting conformal prediction from a Bayesian perspective—proving that both split conformal prediction and conformal risk control are special cases of the Bayesian Quadrature framework, proposing practical Bayesian alternatives, and providing interpretable guarantees as well as a richer representation of future loss ranges.

## Background & Motivation

**Background**: Distribution-free uncertainty quantification (such as conformal prediction) provides statistical guarantees for the deployment of black-box models—without requiring knowledge of how the model was trained. Conformal prediction guarantees that the prediction set/interval contains the true value with a probability of at least $1-\alpha$.

**Limitations of Prior Work**:
   - Conformal prediction is based on frequentist probability—making it difficult to incorporate potential prior knowledge (such as partial information about the data distribution).
   - Frequentist guarantees control the "expected loss averaged over many datasets"—rather than "the loss on the actually observed dataset".
   - It only produces a single quantile estimate—without quantifying the uncertainty of the quantile itself.
   - It is challenging to incorporate additional structural assumptions (such as monotonicity or symmetry) to tighten the guarantees.

**Key Challenge**: Bayesian methods are thought to require prior distributions (hence not being "distribution-free"), whereas conformal prediction is distribution-free but inflexible. Is this conflict truly irreconcilable?

**Goal**: Re-unify and extend conformal prediction using Bayesian probability.

**Key Insight**: The core computation of conformal prediction—estimating quantiles from calibration data—is essentially a numerical integration problem (the inverse of the distribution function), which can be executed via Bayesian quadrature (a probabilistic numerical method).

**Core Idea**: Treat the empirical distribution of calibration scores as observations on a probability measure $\rightarrow$ model the unobserved distribution function using a Gaussian Process (GP) prior $\rightarrow$ the posterior provides a complete representation of uncertainty for the quantile (rather than just a point estimate). Without prior knowledge, this is equivalent to conformal prediction; with prior knowledge, it automatically leverages it.

## Method

### Overall Architecture
1. Given the losses/scores $s_1, \ldots, s_n$ of the calibration set.
2. Model the quantile estimation as a Bayesian quadrature assignment.
3. Place a Gaussian Process prior on the distribution function $F$.
4. Condition on observations $\rightarrow$ posterior quantile distribution.
5. Extract prediction sets/risk guarantees from the posterior.

### Key Designs

1. **Quantile Estimation as Bayesian Quadrature**:

    - Function: Re-model the quantile calculation of conformal prediction as probabilistic numerical integration.
    - Mechanism: $\hat{q} = F^{-1}(1-\alpha)$, where $F$ is the CDF of the calibration scores $\rightarrow$ $F$ is unknown but has $n$ observations $\rightarrow$ model $F$ with a GP prior $\rightarrow$ the posterior yields a distribution of $F^{-1}(1-\alpha)$ (rather than a point estimate).
    - Equivalence to Conformal Prediction: When using a "step function" prior, the posterior median is exactly equal to the quantile of conformal prediction—proving that conformal prediction is a special case of Bayesian quadrature.
    - Design Motivation: A unified framework allows leveraging prior knowledge when available (e.g., knowing the distribution is unimodal), while automatically degrading to standard conformal prediction in the absence of priors.

2. **Posterior Distribution of Quantiles**:

    - Function: Provide a complete representation of uncertainty for the quantile.
    - Mechanism: Since the GP posterior yields the complete distribution of $F \rightarrow$ the distribution of $F^{-1}(1-\alpha)$ can also be calculated $\rightarrow$ yielding a credible interval for the quantile.
    - Value:
        - Standard conformal prediction states: "Under 95% confidence, the prediction set covers the true value."
        - The Bayesian version states: "The 90% credible interval of the quantile is [q_low, q_high], corresponding to a coverage rate between [93%, 97%]."
    - Design Motivation: Richer information helps decision-makers understand the reliability of the guarantees.

3. **Incorporation of Prior Knowledge**:

    - Function: Tighten guarantees when additional information is available.
    - Mechanism: The kernel function of the GP prior encodes assumptions about $F$—e.g., a smoothness kernel $\rightarrow$ assuming $F$ is smooth; a monotonicity constraint $\rightarrow$ ensuring $F$ is monotonically increasing.
    - Concrete Example: If the loss distribution is known to be symmetric $\rightarrow$ imposing a symmetry constraint on the prior $\rightarrow$ effectively doubles the sample size $\rightarrow$ tightening the guarantees.
    - Design Motivation: Distribution-free guarantees come with a cost—standard conformal prediction yields very loose guarantees when $n$ is small, whereas prior knowledge can yield significant improvements.

### Loss & Training
- No training—purely an inference/post-processing method.
- The GP posterior has an analytical solution (for standard kernels).
- Computational complexity is $O(n^3)$ (the standard cost of GP), which is practical for small/medium calibration sets ($n < 10000$).

## Key Experimental Results

### Main Results
Quality of coverage estimation under different calibration set sizes:

| Method | n=50 coverage error ↓ | n=200 coverage error ↓ | Provides posterior? |
|------|-------------|--------------|---------|
| Split Conformal | 4.2% | 1.8% | ✗ (Point estimate) |
| **Bayesian Quadrature (No Prior)** | **4.2%** | **1.8%** | **✓** |
| **Bayesian Quadrature (Smooth Prior)** | **2.8%** | **1.2%** | **✓** |
| **Bayesian Quadrature (Monotonic Prior)** | **2.1%** | **0.9%** | **✓** |

Unified Conformal Risk Control:

| Method | Framework | Loss Guarantee at n=100 |
|------|------|-------------|
| Standard Conformal Risk Control | Frequentist | $\hat{\lambda}$ Point estimate |
| **Bayesian Quadrature Version** | **Bayesian** | **$\hat{\lambda}$ Distribution + Credible interval** |

### Ablation Study

| Type of Prior Knowledge | Coverage Error Improvement (n=50) | Description |
|-----------|------------------|------|
| No Prior (Default) | 0% (Equivalent to conformal prediction) | Baseline |
| **Smoothness Assumption** | **-33%** | Smooth CDF |
| **Monotonicity Constraint** | **-50%** | Monotonically increasing CDF (always holds) |
| **Symmetry Assumption** | **-45%** | Applicable to symmetric loss distributions |
| Misspecified Prior (Incorrect Assumptions) | +15% | Prior misspecification carries a cost |

### Key Findings
- In the absence of priors, Bayesian quadrature is exactly equivalent to standard conformal prediction—proving theoretical unification.
- Reasonable priors can halve the coverage error on small calibration sets—the value of prior knowledge is greatest when $n$ is small.
- The posterior distribution provides much richer information—decision-makers can see "how certain" the guarantee is.
- Misspecified priors lead to performance degradation—but the benefit of the Bayesian approach is that prior plausibility can be checked via the posterior.
- Monotonicity constraints are almost always safe (the CDF is inherently monotonic) $\rightarrow$ recommended as a default.

## Highlights & Insights
- **"Bayesian probability is not in conflict with being distribution-free"**—this epistemological insight is the most profound contribution of this paper. A Bayesian prior is an assumption about "the shape of $F$", not "the data distribution".
- Standard conformal prediction is recovered as a special case of Bayesian quadrature—a unified perspective that eliminates the artificial divide between the two schools of thought.
- The posterior distribution of quantiles is a highly practical new tool—upgrading from "whether 95% coverage is achieved" to "the 90% interval of coverage is [93%, 97%]".
- The intersection of Probabilistic Numerics and uncertainty quantification is a novel and promising research direction.
- For practitioners deploying safety-critical ML systems, posterior quantiles offer more responsible guarantees than point-estimate quantiles.

## Limitations & Future Work
- GP computational complexity $O(n^3)$ is impractical for large calibration sets—requiring sparse/approximate GPs.
- Choosing prior kernel functions requires domain knowledge.
- Extensions to non-exchangeable data (such as time series) require conditional Bayesian quadrature.
- Integration with active/online conformal prediction (adaptive conformal) remains unexplored.
- Bayesian quadrature in multivariate/high-dimensional settings is more challenging.

## Related Work & Insights
- **vs Standard Conformal Prediction**: An upgrade from frequentist point estimation to Bayesian posterior distribution, equivalent when no prior is used.
- **vs Bayesian approaches to conformal**: Previous Bayesian conformal methods typically modify the predictor, whereas this paper keeps the predictor as a black box and only modifies the calibration step.
- **vs Probabilistic Numerics**: A new direction applying probabilistic numerical quadrature to statistical inference.
- **Insight**: Frequentist methods can often be re-understood and naturally extended within a Bayesian framework—incorporating prior knowledge is merely optional rather than mandatory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifying conformal prediction into the Bayesian quadrature framework is a profound theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + real-world data, comparing multiple priors.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant theory and highly intuitive.
- Value: ⭐⭐⭐⭐⭐ Has a foundational impact on uncertainty quantification methodologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/optimization/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/optimization/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)

</div>

<!-- RELATED:END -->
