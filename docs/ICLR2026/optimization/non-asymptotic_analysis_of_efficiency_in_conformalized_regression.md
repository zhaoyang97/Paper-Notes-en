---
title: >-
  [Paper Note] Non-Asymptotic Analysis of Efficiency in Conformalized Regression
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] Establish the first non-asymptotic efficiency bounds for Conformalized Quantile Regression (CQR) and Conformalized Median Regression (CMR) under SGD training, explicitly characterizing the joint dependence of prediction set length deviation on training sample size $n$, calibration sample size $m$, and miscoverage rate
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 0be04b9dead2eede
---
# Non-Asymptotic Analysis of Efficiency in Conformalized Regression

**Conference**: ICLR 2026  
**arXiv**: [2510.07093](https://arxiv.org/abs/2510.07093)  
**Code**: None  
**Area**: Optimization / Statistical Learning Theory  
**Keywords**: Conformal Prediction, Quantile Regression, Non-asymptotic Analysis, Prediction Set Efficiency, Uncertainty Quantification

## TL;DR

Establish the first non-asymptotic efficiency bounds for Conformalized Quantile Regression (CQR) and Conformalized Median Regression (CMR) under SGD training, explicitly characterizing the joint dependence of prediction set length deviation on training sample size $n$, calibration sample size $m$, and miscoverage rate $\alpha$.

## Background & Motivation

Conformal Prediction is a distribution-free framework that provides prediction sets with coverage guarantees for black-box models. In regression tasks, the **efficiency** of conformal prediction is typically measured by the expected length of the prediction interval—the smaller the prediction set, the better, provided the coverage condition $\mathbb{P}[Y \in \mathcal{C}(X)] \geq 1-\alpha$ is met.

Existing efficiency analyses mainly face two issues:

**Asymptotic Analysis**: Proving that prediction sets converge to the oracle set as the sample size goes to infinity, but failing to provide finite-sample guarantees.

**Prior Non-asymptotic Bounds**: Typically treating $\alpha$ as a constant and only considering the impact of the calibration set size $m$, while ignoring the effects of the training set size $n$ and the value of $\alpha$.

In practical safety-critical applications (healthcare, finance, autonomous driving), $\alpha$ is usually set very small to ensure high coverage, yet the behavior of efficiency in these cases remains unclear. This paper fills this theoretical gap by providing the first non-asymptotic characterization of efficiency with respect to the triplet $(n, m, \alpha)$.

## Method

### Overall Architecture

The paper analyzes two regression variants under the split conformal prediction framework: Conformalized Quantile Regression (CQR), which uses a linear model to estimate conditional upper and lower quantiles to construct adaptive asymmetric intervals, and Conformalized Median Regression (CMR), which only estimates the conditional median and uses the absolute residual as the non-conformity score to construct symmetric intervals. Both are trained using SGD. The objective of the analysis is the expected absolute deviation between the prediction set length and the oracle interval length $\mathbb{E}[\,||\mathcal{C}(X)| - |\mathcal{C}^*(X)|\,|]$, explicitly decomposing this deviation into training sample size $n$, calibration sample size $m$, and miscoverage rate $\alpha$.

### Key Designs

**1. Four-term Efficiency Bound for CQR-SGD: Decomposing Length Deviation into $(n, m, \alpha)$ Dimensions**

Previous non-asymptotic analyses often treated $\alpha$ as a constant and focused solely on the calibration set size $m$, which fails in safety-critical scenarios where $\alpha$ is very small. Theorem 3.2 provides a complete bound $\mathbb{E}[\,||\mathcal{C}(X)| - |\mathcal{C}^*(X)|\,|] \leq \mathcal{O}(n^{-1/2} + (\alpha^2 n)^{-1} + m^{-1/2} + \exp(-\alpha^2 m))$, where the four terms have distinct origins: $n^{-1/2}$ is the standard training error of quantile regression, $(\alpha^2 n)^{-1}$ represents the amplification of training error as $\alpha$ decreases, $m^{-1/2}$ is the finite-sample effect of the calibration set, and $\exp(-\alpha^2 m)$ characterizes the exponential decay of calibration when $\alpha$ is small. This bound holds under the assumptions of a well-defined linear model (Assumption 3.1), bounded covariance (Assumption 3.2), and conditional density regularity (Assumption 3.3). The terms $(\alpha^2 n)^{-1}$ and $\exp(-\alpha^2 m)$ explicitly incorporate the role of $\alpha$ into the efficiency bound for the first time.

**2. Comparable Bounds for CMR-SGD: Symmetric Assumption for Simplified Analysis**

The prediction set length of CMR in homoscedastic tasks does not vary with input and is a constant. While analysis should be simpler, it only estimates the median and lacks direct control over the upper and lower quantiles. Theorem 4.1 shows that under an additional quantile symmetry assumption (Assumption 4.2, where quantiles are symmetric about the median), CMR achieves an efficiency bound of the same order in $(n, m, \alpha)$ as CQR. This indicates that the triplet characterization is not unique to CQR but is a common property of conformalized regression under SGD training, with symmetry being the key trade-off for using point median estimates to control intervals.

**3. Phase Transition Analysis: Guiding Data Allocation based on $\alpha$**

In safety-critical applications, how small must $\alpha$ be to change the dominant term of efficiency? Section 3.2.1 sets $n = \Theta(m)$ to simplify the bound to $\mathcal{O}(n^{-1/2} + (\alpha^2 n)^{-1})$ and characterizes its behavior in three segments: when $\alpha = \Omega(n^{-1/4})$, the bound is $\mathcal{O}(n^{-1/2})$ and independent of $\alpha$; when $n^{-1/2} \ll \alpha \ll n^{-1/4}$, the bound transitions to being dominated by $\mathcal{O}((\alpha^2 n)^{-1})$; and in the extreme case of $\alpha = \Theta(n^{-1/2})$, $\alpha$ completely dominates. This provides actionable data allocation guidance—when $\alpha$ is large enough, the training and calibration sets can be of the same order, but when $\alpha$ is small, more budget must be allocated to the training data to suppress the amplified training error.

**4. Three-Step Proof Framework: Layered Propagation of Parameter Deviations**

The difficulty in anchoring the length deviation to $(n, m, \alpha)$ lies in the fact that the parameter error learned by SGD must first affect the score distribution, then the empirical quantile, and finally the interval length. The proof propagates through three Propositions: first, Prop B.5 controls the impact of parameter deviation on the population quantile of the score distribution; then, Prop B.7 controls the gap between the finite-sample quantile $(1-\alpha)_m$ and the population $(1-\alpha)$ quantile; finally, Prop B.11 uses the DKW inequality to concentrate the empirical quantile to the population quantile. Two core techniques support this: leveraging the strong convexity of pinball loss to obtain an $\mathcal{O}(1/n)$ center error bound for SGD (Theorem 3.1), and utilizing conditional density regularity to ensure the score distribution has a lower-bounded density near the quantile, preventing small translations of the quantile from being infinitely amplified.

## Key Experimental Results

### Synthetic Data Experiments

Experiments used piecewise affine conditional densities to verify the three dimensions of the theoretical bound:

| Experimental Dimension | Theoretical Prediction | Experimental Observation |
|---------|---------|---------|
| Training set size $n$ | log Δ vs log n slope from -1 to -0.5 | As α increases, slope indeed transitions from -1 to -0.5 |
| Calibration set size $m$ | log-log slope approaches -0.5 | Measured slope is near -0.5 |
| Miscoverage rate α | Δ ~ α^{-2} | Fitted coefficient b₁ = -2.24, close to theoretical value -2 |

### Real Data Experiments

Using the MEPS Panel 19/20 dataset:
- Increasing calibration set $m$ consistently reduces length deviation.
- For a fixed sample size, larger $\alpha$ leads to smaller deviation and lower variance.
- Both CQR and CMR exhibit trends consistent with theoretical predictions.

### Key Findings

- In the region dominated by $\mathcal{O}((\alpha^2 n)^{-1})$, the measured log-log regression slope of Δ vs $1/(n\alpha^2)$ is approximately 0.92, close to the theoretical value of 1.
- The phase transition of the efficiency bound is clearly verified in experiments.

## Highlights & Insights

1.  **First Triplet $(n, m, \alpha)$ Efficiency Bound**: Previous work either only considered $m$ or treated $\alpha$ as a constant; this paper reveals the critical role of $\alpha$ for the first time.
2.  **Direct Assumptions on Data Distribution**: Unlike prior work that assumed properties of the score distribution, assumptions here are applied directly to the data distribution, which is more natural and verifiable.
3.  **Optimizer-Agnostic Analytical Framework**: While SGD is used as an example, the framework can be extended to other optimizers (e.g., SAGA/SVRG for exponential convergence rates) by replacing the training error bound.
4.  **Practical Data Allocation Guidance**: Depending on the magnitude of $\alpha$, it guides how to distribute data between training and calibration sets to minimize redundant prediction set length.

## Limitations & Future Work

1.  **Linear Model Assumption**: Theoretical results are limited to linear quantile regression; extending to non-linear models (e.g., neural networks) requires new technical tools.
2.  **Conditional Density Regularity**: Requires conditional density to be bounded ($f_{\min} \leq f_{Y|X} \leq f_{\max}$), excluding heavy-tailed distributions and cases where density approaches zero.
3.  **Symmetry Assumption for CMR**: Assumption 4.2 requires quantiles to be symmetric about the median, limiting the scope of CMR analysis.
4.  **Lack of Lower Bounds**: Only upper bounds are provided, making it impossible to determine the tightness of the bounds; whether a matching lower bound exists remains an important open question.

## Related Work & Insights

- **Romano et al. (2019)**: Proposers of CQR; this paper characterizes its efficiency from a theoretical perspective.
- **Lei et al. (2018)**: Incorporated training error into efficiency analysis but treated $\alpha$ as a constant.
- **Bars & Humbert (2025)**: Non-asymptotic analysis of volume-minimizing conformal methods; results align with this paper when the function class is finite and $\alpha$ is fixed.
- **Rakhlin et al. (2012)**: Provided optimal SGD convergence rates under strong convexity, serving as the theoretical foundation for Theorem 3.1.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to establish a triplet $(n, m, \alpha)$ non-asymptotic bound.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic and real data verify theoretical predictions from multiple angles.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical clarity, structured proof framework, and intuitive experiments.
- Value: ⭐⭐⭐⭐ — Fills a significant gap in conformal prediction efficiency theory and provides practical data allocation guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting](hinge_regression_tree_a_newton_method_for_oblique_regression_tree_splitting.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](../../ICML2026/optimization/taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[ICLR 2026\] Distributionally Robust Linear Regression with Block Lewis Weights](distributionally_robust_linear_regression_with_block_lewis_weights.md)
- [\[ICLR 2026\] SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration](sgd_with_adaptive_preconditioning_unified_analysis_and_momentum_acceleration.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)

</div>

<!-- RELATED:END -->
