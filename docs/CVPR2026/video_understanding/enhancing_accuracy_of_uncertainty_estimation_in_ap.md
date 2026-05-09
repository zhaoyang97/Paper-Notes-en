---
title: >-
  [Paper Note] Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration
description: >-
  [CVPR 2026][Video Understanding][Gaze tracking] This paper proposes an efficient post-hoc calibration method based on isotonic regression that aligns the output distribution of uncertainty models with the observed distribution, addressing inaccurate uncertainty estimation caused by domain shift in gaze tracking. It also introduces Coverage Probability Error (CPE) as a more reliable uncertainty evaluation metric than EUC.
tags:
  - CVPR 2026
  - Video Understanding
  - Gaze tracking
  - uncertainty estimation
  - post-hoc calibration
  - domain shift
  - evaluation metrics
date: 2026-05-08
content_hash: 0c5576d47877db35
---

# Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration

**Conference**: CVPR 2026
**arXiv**: [2501.14894](https://arxiv.org/abs/2501.14894)
**Code**: Available (project page)
**Area**: Video Understanding / Human Pose
**Keywords**: Gaze tracking, uncertainty estimation, post-hoc calibration, domain shift, evaluation metrics

## TL;DR
This paper proposes an efficient post-hoc calibration method based on isotonic regression that aligns the output distribution of uncertainty models with the observed distribution, addressing inaccurate uncertainty estimation caused by domain shift in gaze tracking. It also introduces Coverage Probability Error (CPE) as a more reliable uncertainty evaluation metric than EUC.

## Background & Motivation

1. **State of the Field**: Appearance-based gaze tracking uses deep learning to directly predict gaze angles from eye images. Existing uncertainty-aware methods estimate prediction uncertainty via probabilistic modeling (heteroscedastic regression), quantile regression, or contrastive learning. However, the uncertainty estimates of these models are reliable only within the training domain.

2. **Limitations of Prior Work**:
   - Domain shift (cross-subject, cross-dataset) leads to severely inaccurate uncertainty estimates numerically — the magnitude of model-predicted uncertainty does not match the actual error distribution.
   - Existing methods use uncertainty only for **relative ranking** (e.g., outlier detection) rather than providing reliable **absolute values** (e.g., 95% confidence intervals).
   - The widely used Error-Uncertainty Correlation (EUC) metric is based on a spurious causal assumption: the true sources of uncertainty are epistemic and aleatoric factors, not prediction error itself, making EUC an unreliable measure of uncertainty quality.

3. **Root Cause**: Uncertainty estimation models learn data-specific conditional distribution biases during training. Under domain shift, the learned mapping from "input → uncertainty magnitude" becomes inaccurate, while parameter-level adaptation (e.g., transfer learning, meta-learning) requires substantial target-domain data to re-learn the conditional distribution.

4. **Paper Goals**:
   - How to efficiently correct uncertainty estimates under domain shift using a small number of calibration samples?
   - How to design a metric that correctly evaluates uncertainty quality (as a replacement for the problematic EUC)?

5. **Starting Point**: Framing uncertainty correction as an output-level conditional distribution matching problem — without modifying model parameters, an isotonic regression is learned at the output level to map nominal probabilities to actual probabilities.

6. **Core Idea**: Apply isotonic regression for post-hoc calibration, requiring only ~50 calibration samples to align the predicted distribution with the observed distribution, while replacing the spuriously causal EUC with CPE for correct uncertainty quality evaluation.

## Method

### Overall Architecture
The overall pipeline consists of two components: (1) **CPE evaluation metric** — quantifying the deviation between the predicted distribution and the observed distribution across the full probability range; and (2) **post-hoc calibration** — using isotonic regression to learn a probability mapping function $R: [0,1] \to [0,1]$ that maps nominal probabilities of an uncalibrated model to corrected probabilities at inference time. The calibration process does not modify any parameters of the original uncertainty model; it only adds a lightweight mapping layer at the output.

### Key Designs

1. **Coverage Probability Error (CPE) Metric**:
   - Function: Correctly evaluates the calibration quality of uncertainty models.
   - Mechanism: For an ideal uncertainty model, the quantile at nominal cumulative probability $p$ should cover exactly a $p$ fraction of the ground-truth labels. CPE evaluates this deviation across the entire $[0,1]$ probability range: $CPE = \sqrt{\frac{1}{n}\sum_{i=0}^{n} p_{err}(\frac{i}{n})^2}$, where $p_{err}(p) = |p - \hat{P}(p)|$ and $\hat{P}(p)$ is the empirical coverage probability (the fraction of ground-truth labels falling below the $p$-th quantile). $n=11$ equally spaced checkpoints are used to balance accuracy and efficiency.
   - Design Motivation: EUC assumes uncertainty is caused by prediction error (spurious causality), meaning even a perfect uncertainty model cannot achieve EUC=1. CPE directly measures whether the predicted distribution matches the observed distribution, making it a proper scoring metric.

2. **Post-hoc Calibration via Isotonic Regression**:
   - Function: Corrects inaccurate probability outputs of an uncalibrated model into accurate probability estimates.
   - Mechanism: A mapping $R: [0,1] \to [0,1]$ is trained to map nominal probabilities to actual probabilities. For example, if the nominal 0.9 quantile actually covers only 80% of ground-truth values, then the nominal 0.9 quantile is used when 80% coverage is needed. The optimization objective is $\min \sum_{i=1}^T \|\hat{P}(p_i) - R(p_i)\|$, subject to $R(p_i) \leq R(p_{i+1})$ (monotonicity, preserving CDF properties). Isotonic regression is used as a non-parametric method that preserves CDF monotonicity without making parametric assumptions about miscalibration patterns. The corrected quantile at inference is $\tilde{\theta}_{t,quant} = F_t^{-1}(R(p))$.
   - Design Motivation: Parameter-level adaptation (transfer learning, etc.) requires large amounts of target-domain data to re-learn the conditional distribution; post-hoc calibration requires only ~50 samples to learn the output mapping, achieving extremely high data efficiency. Isotonic regression is more flexible than parametric methods such as temperature scaling, capturing non-linear miscalibration patterns.

3. **Analysis of Calibration Sample Size**:
   - Function: Determining the minimum number of calibration samples required.
   - Mechanism: Experiments show the most significant improvement occurs with 10–20 samples, and performance saturates at around 50 samples, reducing CPE from ~40% to ~5%. Thus 50 calibration samples are used by default.
   - Design Motivation: The cost of acquiring calibration samples in practice determines the feasibility of the method; the requirement of 50 samples is highly practical.

### Loss & Training
- Base uncertainty model: Heteroscedastic Gaussian regression with NLL loss $NLL_t = \frac{1}{2}ln(\hat{\sigma}_t^2) + \frac{l_{n,t}}{2\hat{\sigma}_t^2}$, where $l_{n,t}$ is the smooth L1 loss.
- Calibration model: Isotonic regression, requiring only nominal probability–empirical probability pairs as training data.
- Two backbone networks, ResNet-18 and ResNet-50, are used to verify generality.

## Key Experimental Results

### Main Results — CPE Calibration Effect

| Test Scenario | Train Set | Test Set | Backbone | CPE (Uncalibrated) | CPE (Calibrated) | Gain |
|---|---|---|---|---|---|---|
| Cross-subject | MPII | MPII | ResNet18 | 23.17% | 5.18% | ↓78% |
| Cross-subject | RTGene | RTGene | ResNet18 | 19.60% | 5.26% | ↓73% |
| Cross-dataset | MPII | RTGene | ResNet18 | 20.60% | 4.75% | ↓77% |
| Cross-dataset | RTGene | MPII | ResNet18 | 27.21% | 4.84% | ↓82% |
| Cross-dataset | MPII | RTGene | ResNet50 | 20.10% | 4.63% | ↓77% |
| Cross-dataset | RTGene | MPII | ResNet50 | 26.36% | 4.79% | ↓82% |

### 95% Confidence Interval Coverage Probability

| Test Scenario | Quantile Regression | Uncalibrated | Calibrated | Ideal |
|---|---|---|---|---|
| Case 1 | 40.5% | 41.1% | **88.0%** | 95% |
| Case 5 | 34.3% | 47.8% | **86.7%** | 95% |
| Case 8 | 16.4% | 46.2% | **88.6%** | 95% |

### Incidental Improvement in Angular Error

| Test Scenario | Angular Error (Uncalibrated) | Angular Error (Calibrated) | Gain |
|---|---|---|---|
| Cross-dataset MPII→RTGene (R18) | 13.71° | 10.12° | ↓26% |
| Cross-dataset RTGene→MPII (R18) | 18.46° | 14.50° | ↓21% |
| Cross-dataset MPII→RTGene (R50) | 13.89° | 9.50° | ↓32% |

### Key Findings
- **All calibrated models achieve >70% CPE improvement**: Statistically significant under the Mann-Whitney U test (p<0.05).
- **Post-calibration CPE stabilizes at ~5%**: Regardless of the degree of domain shift (from 8–45% → ~5%), demonstrating the robustness of the calibration approach.
- **EUC completely fails**: Even when CPE is near-perfect (~5%), EUC remains close to 0 (indicating no correlation), confirming the absence of a causal relationship between error and uncertainty.
- **Calibration also incidentally reduces angular error**: Using the median (rather than the mean) as the point estimate improves most scenarios by 7–32%.
- **50 calibration samples suffice for saturation**: The extremely low data requirement makes the method highly practical in real-world scenarios.

## Highlights & Insights
- **Deep insight into the spurious causality of EUC**: The paper argues that "correlation between error and uncertainty does not imply causation," a point with broad implications for the entire uncertainty estimation community. The sources of uncertainty are epistemic and aleatoric factors, not prediction error itself.
- **Elegant simplicity of post-hoc calibration**: Model parameters are not modified; only a probability mapping is learned via isotonic regression from 50 samples. This "minimal intervention" philosophy can be generalized to any model that outputs a probability distribution.
- **CPE as a proper scoring metric**: It directly measures the alignment between the predicted distribution and the observed distribution, which is scientifically far superior to EUC. The visualization approach (nominal vs. observed probability plot) is also highly intuitive.

## Limitations & Future Work
- The current calibration is applied independently to yaw and pitch, without considering the joint distribution of the two dimensions.
- The calibration model assumes that calibration samples and test samples are drawn from the same distribution — if the intra-target-domain distribution varies significantly, a single global calibration may be insufficient.
- Validation is limited to CNN models (ResNet-18/50); uncertainty models based on Transformer architectures remain untested.
- As a non-parametric method, isotonic regression's calibration accuracy is limited by the number of calibration samples and the range of probabilities covered.
- No systematic comparison is made against other post-hoc calibration methods (e.g., Platt scaling, temperature scaling).

## Related Work & Insights
- **vs. TMASS/GIMO and similar gaze tracking models**: These models use uncertainty only for relative ranking, so low EUC values do not affect their utility. This paper demonstrates that uncertainty estimation with reliable absolute values is both feasible and valuable.
- **vs. Kellnhofer (quantile regression)**: Quantile regression does not produce a complete distributional prediction and therefore cannot be evaluated with CPE; moreover, its 95% CI coverage is extremely low (as low as 16.4%).
- **vs. Monte Carlo Dropout / Ensembles**: These methods are rarely used in gaze tracking due to high computational cost; the proposed post-hoc calibration incurs virtually zero additional computation.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing post-hoc calibration into uncertainty estimation for gaze tracking is a novel application, and the CPE metric has broad applicability.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four domain-shift scenarios, two backbone networks, calibration sample size analysis, and 95% CI case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logical reasoning; the mathematical derivation and visual explanation of CPE are both clear.
- Value: ⭐⭐⭐⭐ The CPE metric and post-hoc calibration method are generalizable to other uncertainty-aware visual tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)

<!-- RELATED:END -->
