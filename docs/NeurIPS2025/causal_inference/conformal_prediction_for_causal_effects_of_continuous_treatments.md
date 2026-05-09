---
title: >-
  [Paper Note] Conformal Prediction for Causal Effects of Continuous Treatments
description: >-
  [NeurIPS 2025][Causal Inference][conformal prediction] This work is the first to construct conformal prediction intervals for causal effects of continuous treatments (e.g., drug dosage) by parameterizing intervention-induced propensity shifts via a tilting function family and employing quantile regression, providing finite-sample $1-\alpha$ coverage guarantees under both known and unknown propensity score settings.
tags:
  - NeurIPS 2025
  - Causal Inference
  - conformal prediction
  - Continuous Treatment
  - Propensity Score
  - Uncertainty Quantification
  - Potential Outcomes
date: 2026-05-08
content_hash: 90d558307a832cbe
---

# Conformal Prediction for Causal Effects of Continuous Treatments

**Conference**: NeurIPS 2025
**arXiv**: [2407.03094](https://arxiv.org/abs/2407.03094)
**Code**: [GitHub](https://github.com/m-schroder/ContinuousCausalCP)
**Area**: Causal Inference / Conformal Prediction
**Keywords**: conformal prediction, Continuous Treatment, Causal Inference, Propensity Score, Uncertainty Quantification, Potential Outcomes

## TL;DR

This work is the first to construct conformal prediction intervals for causal effects of continuous treatments (e.g., drug dosage) by parameterizing intervention-induced propensity shifts via a tilting function family and employing quantile regression, providing finite-sample $1-\alpha$ coverage guarantees under both known and unknown propensity score settings.

## Background & Motivation

**Background**: Conformal Prediction (CP) has attracted growing attention in uncertainty quantification due to its model-agnostic, distribution-free, finite-sample coverage guarantees. However, existing causal CP methods are exclusively designed for binary or discrete treatments (e.g., treated vs. untreated) and cannot handle continuous treatments (e.g., drug dosage).

**Limitations of Prior Work**: Adapting CP from standard prediction tasks to causal inference poses three fundamental challenges:
   - **Challenge ①**: Interventions alter the propensity function $\pi(a|x)$, inducing a distributional shift between the observational and interventional distributions, thereby violating the exchangeability assumption central to CP.
   - **Challenge ②**: In observational data, propensity scores are typically unknown and must be estimated, introducing additional uncertainty from estimation error.
   - **Challenge ③**: Under continuous treatments, no two observations share identical treatment values, making direct conditional calibration infeasible.

**Key Challenge**: CP requires exchangeability to guarantee valid coverage, yet causal interventions inherently violate exchangeability. The core question is how to maintain rigorous finite-sample coverage guarantees under distributional shift.

**Limitations of Prior Work**:
   - MC Dropout: Poor posterior approximation quality, unreliable coverage rates.
   - Deep Ensembles: No theoretical coverage guarantees.
   - Bayesian methods: Require prior distribution assumptions and are not robust to model misspecification.
   - Existing causal CP methods (e.g., Lei & Candès 2021): Restricted to binary treatments; discretizing continuous treatments leads to ill-defined causal estimands.

**Key Insight**: The paper models the intervention-induced propensity shift as a tilting function family $\mathcal{F}$, and constructs CP intervals via robust optimization over this family, ensuring coverage guarantees hold under all possible distributional shifts.

## Method

### Problem Formulation

Let the data $(X_i, A_i, Y_i)_{i=1}^n$ consist of confounders $X \in \mathcal{X}$, a continuous treatment $A \in \mathcal{A}$, and an outcome $Y \in \mathcal{Y}$. The goal is to construct a prediction interval $C(X_{n+1}, \Diamond)$ for a new sample $X_{n+1}$ under intervention $\Diamond$ (hard intervention $a^*$ or soft intervention $A^*(X_{n+1})$) such that:

$$P(Y_{n+1}(\Diamond) \in C(X_{n+1}, \Diamond)) \geq 1 - \alpha$$

The data are split into a training set $D_T$ (for training the prediction model $\phi$) and a calibration set $D_C$ (for constructing CP intervals). The non-conformity score takes the residual form: $S_i = |Y_i - \phi(X_i, A_i)|$.

### Overall Architecture

A two-stage pipeline:
1. Train an arbitrary causal effect prediction model $\phi$ on the training set and compute non-conformity scores on the calibration set.
2. Solve a quantile regression problem subject to propensity shift constraints to obtain the CP interval threshold $S^*$ with coverage guarantees.

### Key Design 1: Propensity Shift Parameterization via Tilting Functions

An intervention shifts the observational propensity $\pi(a|x)$ to the interventional propensity $\tilde{\pi}(a|x)$, linked through a non-negative tilting function $f$:

$$\tilde{\pi}(a|x) = \frac{f(a,x)}{\mathbb{E}_P[f(A,X)]} \pi(a|x)$$

This parameterization reformulates the distributional shift problem in causal inference as a robust optimization problem within the CP framework. Rather than assuming exchangeability, calibration is performed conditionally for $f \in \mathcal{F}$, maintaining coverage under all possible shifts.

### Key Design 2: Known Propensity Setting (Theorem 4.2)

For a soft intervention $A^* = A + \Delta_A$, the function family is defined as $\mathcal{F} = \{\theta \frac{\pi(a+\Delta_A|x)}{\pi(a|x)} \mid \theta \in \mathbb{R}^+\}$. Direct optimization over all $S \in \mathbb{R}$ is computationally intractable. By exploiting **strong duality**, the problem is converted to its dual form:

$$\max_{\eta_i} \min_{\theta > 0} \sum_{i=m+1}^{n} \eta_i(S_i - \theta \frac{\pi(a_i + \Delta_A | x_i)}{\pi(a_i | x_i)}) + \eta_{n+1}(S - \theta \frac{\pi(a^* | x_{n+1})}{\pi(a_{n+1} | x_{n+1})})$$

subject to $-\alpha \leq \eta_i \leq 1-\alpha$. Defining $S^*$ as the largest $S$ satisfying $\eta_{n+1}^S < 1-\alpha$, the interval $C(x_{n+1}, a^*) = \{y \mid S_{n+1}(y) \leq S^*\}$ provides the coverage guarantee.

### Key Design 3: Unknown Propensity Setting (Theorem 4.5)

A hard intervention $\text{do}(a^*)$ corresponds to a Dirac-delta propensity $\delta_{a^*}(a)$, which is intractable to handle directly. The solution proceeds in three steps:

1. **Gaussian kernel smoothing**: Approximate the Dirac-delta via a Gaussian limit, $\delta_{a^*}(a) = \lim_{\sigma \to 0} \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(a-a^*)^2}{2\sigma^2})$.

2. **Bounded estimation error assumption (Assumption 1)**: Assume the propensity estimation error satisfies $c_{a_i} = \hat{\pi}(a_i|x_i) / \pi(a_i|x_i) \in [1/M, M]$, where $M$ is specified by domain experts.

3. **Type-I Invexity guarantees global optimality (Lemma 4.4)**: Although the optimization problem is non-convex, it satisfies Type-I invexity and the Linear Independence Constraint Qualification (LICQ), ensuring that KKT conditions are both necessary and sufficient, enabling globally optimal solutions.

The resulting CP interval is $C(X_{n+1}, a^*) = \{y \mid S_{n+1}(y) \leq S^*\}$, where $S^*$ is the largest $S$ for which $v_{n+1}^S > 0$.

### Loss & Training

- Stage 1: A standard MLP is trained as the causal prediction model $\phi$ with MC Dropout regularization rate 0.1.
- Stage 2: Quantile regression via the pinball loss: $l_\alpha(\theta, S) = (\alpha - \mathbf{1}[\theta - S < 0])(\theta - S)$.
- In the unknown propensity setting, conditional normalizing flows are used to estimate $\hat{\pi}(a|x)$.

## Key Experimental Results

### Synthetic Data Experiments

Evaluation on two synthetic datasets (Dataset 1: piecewise propensity + concave outcome function; Dataset 2: Gaussian propensity + oscillating outcome function), averaged over 50 random runs:

| Method | Coverage ($\alpha=0.05$, target 0.95) | Coverage ($\alpha=0.1$) | Coverage ($\alpha=0.2$) |
|------|:---:|:---:|:---:|
| **CP (Ours)** | **1.00** | **0.90–0.94** | **0.83–0.88** |
| MC-Dropout | 0.02–0.28 | 0.02–0.23 | 0.02–0.11 |
| Gaussian Process | 0.125 | 0.125 | 0.083 |
| Deep Ensemble | Worse | — | — |

The proposed CP method meets or exceeds the target coverage rate across all settings, while MC-Dropout achieves only 2%–30% of the target coverage.

### MIMIC-III Clinical Data

Evaluation on real-world intensive care data (14,719 patients, 8 clinical confounders, predicting the effect of mechanical ventilation duration on blood pressure):

- CP intervals automatically widen in high-treatment regions with sparse training data, accurately reflecting true uncertainty.
- MC-Dropout intervals are uniformly narrow across all regions, suggesting insufficient coverage.
- The behavior of CP intervals aligns with clinical intuition: uncertainty is larger in rare dosage regions.

### Key Findings

- **Substantial coverage gap**: MC-Dropout coverage is only 2%–30% of the target, which is entirely unacceptable for safety-critical applications.
- **Semantically meaningful interval width**: CP interval width is inversely correlated with data support—wider intervals where data are scarce.
- **Correct sensitivity to significance level $\alpha$**: Intervals narrow as $\alpha$ increases, consistent with theoretical expectations.

## Highlights & Insights

- **Elegance of the tilting function framework**: The complex distributional shift problem in causal inference is unified as a robust optimization over a function family, yielding a theoretically natural and general framework applicable to both soft and hard interventions.
- **Mathematical ingenuity of Gaussian kernel approximation of the Dirac-delta**: Since the propensity of a hard intervention is a Dirac-delta (a non-integrable generalized function), the Gaussian limit sidesteps this mathematical obstacle; moreover, the non-convex optimization is shown to admit a global solution under Type-I invexity.
- **Practical value of Assumption 1**: The error bound $M$ is specified by domain experts rather than inferred from data, granting practitioners direct control. This is more realistic than assuming fully known propensities and more constrained than assuming complete ignorance.
- **Model-agnosticism**: CP intervals are compatible with any causal effect prediction model, without being tied to a specific network architecture.

## Limitations & Future Work

- **No automated method for specifying the error bound $M$**: An overly conservative choice (large $M$) produces excessively wide intervals, while an optimistic choice (small $M$) leads to under-coverage. The paper recommends conservative selection but provides no data-driven strategy.
- **Sample splitting reduces data efficiency**: The three-way train/calibration/test split may result in insufficient calibration data in small-sample settings.
- **Restricted to univariate continuous treatments**: Multi-dimensional continuous treatments (e.g., combination drug regimens) are not addressed.
- **Sensitivity to Gaussian kernel bandwidth $\sigma$**: While the theory requires $\sigma \to 0$, the practical impact of choosing a finite $\sigma$ is not thoroughly ablated.
- **Computational complexity**: The optimization procedure may be slow for large-scale CATE vectors.

## Related Work & Insights

| Method Category | Representative Work | Distinction from This Work |
|---------|---------|----------|
| Discrete treatment CP | Lei & Candès 2021; Alaa et al. 2023 | Restricted to binary/discrete treatments; discretizing continuous treatments leads to ill-defined causal estimands. |
| Known propensity CP | Jin et al. 2023 | Assumes known propensity scores, which are typically unavailable in observational data. |
| MC Dropout | Gal & Ghahramani 2016 | Poor posterior approximation quality, unreliable coverage, no finite-sample guarantees. |
| Bayesian methods | Alaa & van der Schaar 2017 | Require prior assumptions, not robust to model misspecification. |
| Distributional shift CP | Barber et al. 2023; Gibbs & Candès 2021 | Shifts are assumed known or only asymptotic guarantees are provided; not applicable to causal intervention settings. |

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to address conformal prediction for continuous-treatment causal effects; innovative use of tilting function families and Type-I invexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic data + MIMIC-III real-world data, multiple baselines, multiple $\alpha$ values, 50 random seeds.
- Writing Quality: ⭐⭐⭐⭐ Theorem-lemma chain is clear; the two-scenario divide-and-conquer structure is well-organized; proofs deferred to appendix.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to safety-critical domains such as healthcare (e.g., chemotherapy dose selection).

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Transferring Causal Effects using Proxies](transferring_causal_effects_using_proxies.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](../../AAAI2026/causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)

<!-- RELATED:END -->
