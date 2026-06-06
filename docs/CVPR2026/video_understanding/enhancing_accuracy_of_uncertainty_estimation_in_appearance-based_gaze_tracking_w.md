---
title: >-
  [Paper Note] Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration
description: >-
  [CVPR 2026][Video Understanding][Gaze tracking] A data-efficient post-hoc calibration method is proposed that aligns the predictive distribution of uncertainty-aware gaze tracking models with the true observational distr…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Gaze tracking"
  - "uncertainty estimation"
  - "post-hoc calibration"
  - "domain shift"
  - "coverage probability error"
date: 2026-05-08
content_hash: 1794afe32dc24cdf
---

# Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration

**Conference**: CVPR 2026
**arXiv**: [2501.14894](https://arxiv.org/abs/2501.14894)  
**Code**: Available (project page)  
**Area**: Video Understanding
**Keywords**: Gaze tracking, uncertainty estimation, post-hoc calibration, domain shift, coverage probability error

## TL;DR
A data-efficient post-hoc calibration method is proposed that aligns the predictive distribution of uncertainty-aware gaze tracking models with the true observational distribution via isotonic regression, and introduces Coverage Probability Error (CPE) as a replacement for the unreliable Error-Uncertainty Correlation (EUC) metric for evaluating uncertainty quality.

## Background & Motivation
Appearance-based gaze tracking leverages deep learning to directly predict gaze angles from eye/face images, serving as a core technology in safety-critical applications such as driver monitoring. Point estimates alone are insufficient; systems must know "how reliable is this estimate," making uncertainty estimation essential.

**Limitations of Prior Work**:

**Domain shift causes miscalibrated uncertainty**: Existing uncertainty-aware models (heteroskedastic regression, quantile regression, etc.) produce reliable uncertainty estimates only within the training domain. When test-time conditions differ in illumination, camera, or subject characteristics, the predicted variance becomes unreliable—models may assign high confidence to incorrect predictions.

**Flawed evaluation metrics**: The widely used EUC (Error-Uncertainty Correlation) assumes a correlation between uncertainty and prediction error, but this is a spurious association—uncertainty stems from aleatoric and epistemic factors rather than prediction error itself, rendering EUC an unreliable measure of uncertainty quality.

**High cost of parameter-level adaptation**: Meta-learning or transfer learning can correct miscalibrated uncertainty but require substantial target-domain data to re-learn conditional distributions.

**Core Idea**: Without modifying model parameters, apply post-hoc output calibration—learn a monotonic mapping function that maps nominal probabilities to actual coverage probabilities, aligning the calibrated distribution with the true distribution. Only approximately 50 calibration samples are required.

## Method

### Overall Architecture
Given a trained uncertainty-aware gaze tracking model $H$ that outputs a Gaussian distribution (mean = gaze angle, variance = uncertainty) for input image $x_t$, the calibration pipeline appends a calibration regressor $R$ to the model output, mapping the miscalibrated CDF to a calibrated CDF $R \circ F$.

### Key Designs
1. **Coverage Probability Error (CPE) Metric**:

    - Function: Evaluates the calibration quality of uncertainty models.
    - Mechanism: For each nominal probability $p$, compute the actual proportion $\hat{P}(p)$ of true labels falling below the corresponding quantile; ideally $\hat{P}(p) = p$. CPE is defined as the RMSE of deviations over the full probability range:
    $CPE = \sqrt{\frac{1}{n}\sum_{i=0}^{n} p_{err}\left(\frac{i}{n}\right)^2}, \quad p_{err}(p) = \left|p - \hat{P}(p)\right|$
    - Design Motivation: EUC relies on a spurious correlation between error and uncertainty; CPE directly measures the agreement between the predictive distribution and the observational distribution, following the philosophy of proper scoring rules. For example, CPE = 0.05 implies that a nominal 80% confidence interval actually covers approximately 70–90% of observations.

2. **Isotonic Regression Calibrator**:

    - Function: Learns a monotonically increasing mapping $R: [0,1] \to [0,1]$ that transforms nominal probabilities into error-compensated probabilities.
    - Mechanism: Collect calibration dataset $D = \{(p_i, \hat{P}(p_i))\}$ and fit $R$ via isotonic regression subject to the monotonicity constraint $R(p_i) \leq R(p_{i+1})$. At test time, the nominal probability $p$ is replaced by $\tilde{p} = R(p)$ when computing quantiles:
    $\frac{\sum_{t=1}^{T} I\{\theta_t \leq F_t^{-1}(R(p))\}}{T} \to p$
    - Design Motivation: (1) As a non-parametric method, it imposes no specific functional form on the calibration mapping (unlike temperature scaling, which assumes linearity), enabling it to capture nonlinear miscalibration. (2) Isotonic regression naturally satisfies the monotonicity requirement of a valid CDF. (3) Only approximately 50 calibration samples are needed, yielding high data efficiency. (4) Original model parameters remain unchanged, making the approach applicable to any model that outputs a probability distribution.

3. **Per-axis Independent Calibration**:

    - Function: Applies calibration separately to the yaw and pitch components of gaze angle.
    - Mechanism: The horizontal (yaw) and vertical (pitch) components exhibit distinct error distribution characteristics; independent calibration more precisely matches the true distribution of each axis.
    - Design Motivation: The severity of domain shift may differ between axes (e.g., horizontal direction is more strongly affected by head rotation), making separate calibration more flexible.

### Loss & Training
- The base model is trained with heteroskedastic NLL loss: $NLL_t = \frac{1}{2}\ln(\hat{\sigma}_t^2) + \frac{l_{n,t}}{2\hat{\sigma}_t^2}$, where $l_{n,t}$ is the smooth L1 loss.
- The calibrator requires no gradient-based training; it is fitted with isotonic regression (a single line of sklearn code).
- Two backbone architectures are evaluated: ResNet-18 and ResNet-50.

## Key Experimental Results

### Main Results

| Test Scenario | Train→Test | Backbone | CPE (Uncalibrated) | CPE (Calibrated) | Improvement | Angular Error (Uncalibrated) | Angular Error (Calibrated) |
|---|---|---|---|---|---|---|---|
| Cross-subject | MPII→MPII | ResNet18 | 23.17% | 5.18% | ↓78% | 5.77° | 5.09° (↓12%) |
| Cross-subject | RTGene→RTGene | ResNet18 | 19.60% | 5.26% | ↓73% | 12.36° | 10.55° (↓15%) |
| Cross-dataset | MPII→RTGene | ResNet18 | 20.60% | 4.75% | ↓77% | 13.71° | 10.12° (↓26%) |
| Cross-dataset | RTGene→MPII | ResNet18 | 27.21% | 4.84% | ↓82% | 18.46° | 14.50° (↓21%) |
| Cross-dataset | MPII→RTGene | ResNet50 | 20.10% | 4.63% | ↓77% | 13.89° | 9.50° (↓32%) |

CPE improvements exceed 70% across all scenarios, with statistical significance (Mann-Whitney U test, p < 0.05).

### Ablation Study

| Number of Calibration Samples | CPE (%) | Trend |
|---|---|---|
| 0 (uncalibrated) | ~40% | Baseline |
| 10 | ~15% | Large improvement |
| 20 | ~10% | Continued improvement |
| 50 | ~5% | Near saturation |
| 100 | ~5% | Negligible further improvement |

| 95% CI Coverage | Quantile Regression | Uncalibrated Model | Calibrated Model | Ideal |
|---|---|---|---|---|
| Case 1 | 40.5% | 41.1% | **88.0%** | 95% |
| Case 5 | 34.3% | 47.8% | **86.7%** | 95% |
| Case 8 | 16.4% | 46.2% | **88.6%** | 95% |

### Key Findings
- CPE consistently decreases from the 8–45% range to approximately 5%, demonstrating strong stability.
- Saturation is achieved with only 50 calibration samples, far more data-efficient than meta-learning and related approaches.
- Calibration yields a side benefit of reducing angular error by 7–32%, as the calibrated median is more robust than the original mean.
- EUC remains nearly unchanged before and after calibration (consistently ~0.1–0.26), entirely failing to reflect genuine improvements, further confirming that EUC is an unreliable metric.

## Highlights & Insights
- The paper precisely identifies a long-overlooked problem in the gaze tracking community: uncertainty estimates become severely miscalibrated under domain shift, while the community has been relying on the flawed EUC metric for evaluation.
- CPE is a general-purpose uncertainty evaluation metric, not limited to gaze tracking.
- The post-hoc calibration approach is extremely simple (isotonic regression + ~50 samples), plug-and-play, and requires no modification to the original model.
- The experimental design is highly systematic: 4 domain shift scenarios × 2 backbones × 3-fold cross-validation.

## Limitations & Future Work
- Calibration assumes that the calibration set and test set share the same distribution; when domain shift continuously evolves (e.g., from indoor to outdoor), online updating of the calibrator may be necessary.
- Validation is limited to ResNet-18/50 with Gaussian distribution assumptions; more complex architectures (e.g., Transformers) or non-parametric distributions have not been tested.
- MPIIGaze and RTGene are relatively controlled datasets; performance under larger domain shifts (e.g., extreme illumination, diverse ethnicities) remains unknown.
- The calibrator itself does not provide uncertainty estimates about calibration reliability.
- No comparison with Bayesian neural networks, MC Dropout, or similar methods is conducted.

## Related Work & Insights
- **Platt Scaling / Temperature Scaling**: These classical calibration methods are designed for classification tasks and assume a linear/affine mapping from logits to probabilities. The proposed isotonic regression calibrator is non-parametric, more expressive for nonlinear miscalibration in regression tasks, and requires no gradient-based optimization on a validation set.
- **Quantile Regression**: Quantile regression directly learns conditional quantiles and theoretically avoids distributional assumptions. However, experiments show that quantile regression is equally severely miscalibrated under domain shift (95% CI coverage of only 16–40%), indicating that its learned quantiles are also domain-dependent. The proposed post-hoc calibration can be stacked on top of quantile regression for further improvement.
- **MC Dropout / Deep Ensembles**: These epistemic uncertainty estimation methods require multiple forward passes or training multiple models, incurring high inference costs. The proposed method requires only a single forward pass plus a post-processing lookup, making deployment overhead virtually negligible.
- **Meta-learning Domain Adaptation**: Methods such as FADA and MAML-based approaches can fine-tune uncertainty estimates on target domains but require substantial annotated target-domain data and additional training. The proposed method requires only ~50 samples with no gradient updates.

## Highlights & Insights (Connections)
- **Generality of CPE**: CPE is task-agnostic and can be directly applied to evaluate uncertainty quality in any model that outputs probability distributions (e.g., pose estimation, depth estimation, weather forecasting). Introducing CPE for uncertainty evaluation in medical image segmentation is worth considering.
- **Paradigm value of post-hoc calibration**: The principle of "correcting the output distribution without modifying the model" is highly practical for deployed systems—when a model already in production is found to produce miscalibrated uncertainty in a new scenario, collecting a small number of samples from the new scenario suffices for recalibration.
- **Impact on downstream gaze tracking applications**: Calibrated uncertainty can inform attention allocation decisions in driving scenarios—automatically switching to a more conservative driving strategy when gaze estimation uncertainty is high.

## Rating
- Novelty: ⭐⭐⭐⭐ The CPE metric and calibration approach are conceptually clear, though the technical contribution lies primarily in combination rather than fundamental innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four domain shift scenarios + two backbones + cross-validation + sample size ablation + case studies; highly systematic.
- Writing Quality: ⭐⭐⭐⭐ Logically clear with intuitive figures and tables, though some notation is slightly redundant.
- Value: ⭐⭐⭐⭐ Highly practical methodology; the CPE metric offers reference value for the broader uncertainty estimation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)
- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)

</div>

<!-- RELATED:END -->
