---
title: >-
  [Paper Note] TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation
description: >-
  [ICML2025 Spotlight][AI Safety][Integrated Gradients] The authors propose TIMING, which improves Integrated Gradients by introducing a temporality-aware segmented random masking baseline. Additionally, they design new evaluation metrics, CPD and CPP, to address the issue of positive and negative attributions canceling each other out in current time series XAI evaluations, outperforming existing baselines across multiple real-world datasets.
tags:
  - "ICML2025 Spotlight"
  - "AI Safety"
  - "Integrated Gradients"
  - "Time Series Explanation"
  - "Feature Attribution"
  - "Explainable AI"
  - "Temporal Masking"
date: 2026-05-08
content_hash: e7bc15ce79aca74b
---

# TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation

**Conference**: ICML2025 Spotlight  
**arXiv**: [2506.05035](https://arxiv.org/abs/2506.05035)  
**Code**: [drumpt/TIMING](https://github.com/drumpt/TIMING)  
**Area**: Time Series XAI  
**Keywords**: Integrated Gradients, Time Series Explanation, Feature Attribution, Explainable AI, Temporal Masking

## TL;DR

The authors propose TIMING, which improves Integrated Gradients by introducing a temporality-aware segmented random masking baseline. Additionally, they design new evaluation metrics, CPD and CPP, to address the issue of positive and negative attributions canceling each other out in current time series XAI evaluations, outperforming existing baselines across multiple real-world datasets.

## Background & Motivation

Existing time series XAI methods (such as FIT, Dynamask, ContraLSP, TimeX++, etc.) generally employ **unsigned attribution** schemes, focusing only on the magnitude of change in output after feature removal while ignoring the direction (i.e., whether the feature enhances or suppresses model predictions). This is undesirable in practical applications where users typically expect to distinguish between positively and negatively contributing features.

A more critical issue is that **the evaluation metrics themselves are flawed**: the conventional approach measures prediction discrepancy after simultaneously removing the top-K important points. However, this "simultaneous removal" strategy leads to a cancel-out problem where positive and negative attributions offset each other. As shown in Figure 1 of the paper, an attribution method with poor ranking but consistent signs can score higher than a method with perfect ranking but inconsistent signs. This inadvertently biases the research community toward "sign-aligned" methods, while underestimating the true capability of signed methods like IG.

Furthermore, directly applying standard $IG$ to time series data presents two problems:

**Neglect of Temporal Relations**: Under a zero baseline, all points are scaled proportionally, which preserves the original temporal relationships during gradient computation. Consequently, this fails to capture the impact when temporal patterns are disrupted.

**Out-of-Distribution (OOD) Problem**: Intermediate points along the integration path are scaled as a whole, potentially falling into out-of-distribution regions never seen during model training, which leads to unreliable gradients.

## Method

### 1. New Evaluation Metrics: CPD and CPP

**Cumulative Prediction Difference (CPD)**: Sequentially removes feature points from highest to lowest absolute attribution value, accumulating the prediction changes at each step (higher is better):

$$\text{CPD}(x) = \sum_{k=0}^{K-1} \| F(x_k^{\uparrow}) - F(x_{k+1}^{\uparrow}) \|_1$$

**Cumulative Prediction Preservation (CPP)**: Sequentially removes feature points from lowest to highest absolute attribution value, accumulating the prediction changes at each step (lower is better), to verify that low-attribution points are indeed unimportant:

$$\text{CPP}(x) = \sum_{k=0}^{K-1} \| F(x_k^{\downarrow}) - F(x_{k+1}^{\downarrow}) \|_1$$

Core advantage: **Step-by-step accumulation** instead of simultaneous removal eliminates the cancel-out effect of positive and negative attributions, enabling a fair comparison between signed and unsigned methods under the same framework.

### 2. MaskingIG: Partial Preservation Strategy

The zero baseline is modified to $(1-M) \odot x$, where $M \in \{0,1\}^{T \times D}$ is a binary mask. Masked points are integrated from zero to their original values, while unmasked points retain their original values:

$$\text{MaskingIG}_{t,d}(x, M) = x_{t,d} M_{t,d} \times \int_{\alpha=0}^{1} \frac{\partial F_{\hat{y}}(\alpha(M \odot x) + (1-M) \odot x)}{\partial x_{t,d}} \, d\alpha$$

This ensures that the intermediate points along the integration path are closer to the original input, thereby mitigating the OOD problem.

### 3. RandIG → TIMING: Segmented Masking

**RandIG** takes the expectation over multiple random masks: $\text{RandIG}_{t,d}(x;p) = \mathbb{E}_{M_p}[\text{MaskingIG}_{t,d}(x, M_p) | (M_p)_{t,d}=1]$

However, point-wise independent random masking cannot preserve meaningful temporal structures. **TIMING** upgrades the masks to **segmented masks**:

$$\text{TIMING}_{t,d}(x; n, s_{min}, s_{max}) = \mathbb{E}_{M \sim G(n, s_{min}, s_{max})}[\text{MaskingIG}_{t,d}(x, M) | M_{t,d}=1]$$

where $G(n, s_{min}, s_{max})$ generates $n$ continuous segment masks with lengths in the range of $[s_{min}, s_{max}]$. Segmented preservation allows the model to perceive the presence or absence of local temporal dependencies.

### 4. Theoretical Properties

- **Effectiveness (Prop 4.1)**: Masks can be randomly sampled along the IG path, eliminating the need to repeatedly compute standard IG and then average the results.
- **Sensitivity (Prop 4.2)**: If a change in a single feature leads to a change in prediction, the TIMING attribution for this feature is non-zero.
- **Implementation Invariance (Prop 4.3)**: Functionally equivalent models produce identical attributions.
- **Incompleteness (Prop 4.4)**: TIMING does not satisfy completeness (the sum of attributions does not equal the prediction difference), which is the cost of introducing multi-baseline context.

## Key Experimental Results

### MIMIC-III Mortality Prediction (Table 2, Zero Imputation)

| Method | CPD (K=50) ↑ | CPD (K=100) ↑ | Acc ↓ |
|------|-------------|--------------|-------|
| Extrmask | 0.204 | 0.281 | **0.932** |
| ContraLSP | 0.013 | 0.028 | 0.921 |
| TimeX++ | 0.027 | 0.051 | 0.987 |
| GradSHAP | 0.327 | 0.447 | 0.975 |
| IG | 0.342 | 0.469 | 0.974 |
| **TIMING** | **0.366** | **0.505** | 0.975 |

### CPD Comparison across Multiple Datasets (Table 3, 10% Zero Imputation)

| Method | MIMIC-III | PAM | Boiler | Epilepsy | Wafer |
|------|-----------|-----|--------|----------|-------|
| IG | 0.549 | 0.573 | 0.752 | 0.054 | 0.500 |
| GradSHAP | 0.522 | 0.518 | 0.747 | 0.054 | 0.485 |
| Extrmask | 0.305 | 0.380 | 0.400 | 0.029 | 0.202 |
| **TIMING** | **0.597** | **0.602** | **1.578** | **0.060** | **0.674** |

TIMING's improvement relative to IG: MIMIC-III +8.7%, PAM +5.1%, Boiler +109.8%, Wafer +34.8%.

## Highlights & Insights

1. **Contributions at the evaluation metric level are as important as the method itself**: The CPD/CPP metrics reveal a fundamental flaw in the existing evaluation system—namely, that positive and negative attribution cancel-out causes signed methods to be heavily underestimated, allowing methods to "exploit the leaderboard" simply through ReLU sign-alignment.
2. **IG has been underestimated**: Under the new metrics, classic IG already outperforms most recently proposed time series XAI methods (e.g., ContraLSP, TimeX++), demonstrating that the bottleneck lies in the evaluation rather than the methodology itself.
3. **Ingenious segmented mask design**: It preserves local temporal patterns to mitigate OOD issues while simultaneously allowing observation of the effects when temporal relationships are broken, addressing two issues with a single solution.
4. **Both theoretical and practical**: It retains the Sensitivity and Implementation Invariance of IG while providing an efficient single-sample approximation scheme.
5. **109.8% improvement on the Boiler dataset**: This indicates that on industrial data with stronger temporal dependencies, the advantage of temporality-awareness is exceptionally prominent.

## Limitations & Future Work

1. **Does not satisfy Completeness**: The sum of attributions does not equal the prediction difference, which is suboptimal in scenarios requiring attribution budget allocation.
2. **Segmented hyperparameters (n, s_min, s_max)**: These need to be tuned based on the dataset. Although the paper demonstrates insensitivity to these hyperparameters, it still increases the barrier to implementation.
3. **Evaluated only on classification tasks**: The method has not yet been extended to other time series tasks such as time series forecasting or anomaly detection.
4. **Limited coverage of model architectures**: The main experiments are based on a single-layer GRU. Although the appendix includes CNNs and Transformers, their depth and scale remain limited.
5. **Masking strategy can be further explored**: The current implementation uses uniform random segmentation without considering data-driven adaptive segmentations.

## Rating

- Novelty: ⭐⭐⭐⭐ — Insightful contributions in both the rethinking of evaluation metrics and the design of segmented masking IG.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 13 baselines, 6 real-world datasets + 2 synthetic datasets, and multi-metric evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and highly intuitive visualization of the cancel-out problem in Figure 1.
- Value: ⭐⭐⭐⭐ — The CPD/CPP metrics are poised to become new standards for time series XAI evaluation, and TIMING is simple yet effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](../../ICML2026/ai_safety/exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)
- [\[NeurIPS 2025\] Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing](../../NeurIPS2025/ai_safety/generating_multi-table_time_series_ehr_from_latent_space_with_minimal_preprocess.md)
- [\[NeurIPS 2025\] Incentivizing Time-Aware Fairness in Data Sharing](../../NeurIPS2025/ai_safety/incentivizing_time-aware_fairness_in_data_sharing.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[ICML 2026\] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](../../ICML2026/ai_safety/timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)

</div>

<!-- RELATED:END -->
