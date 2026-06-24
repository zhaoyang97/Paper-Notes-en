---
title: >-
  [Paper Note] On Temperature Scaling and Conformal Prediction of Deep Classifiers
description: >-
  [ICML2025][Optimization][Temperature Scaling] This paper presents the first systematic study on the impact of Temperature Scaling (TS) calibration on Conformal Prediction (CP) methods, revealing the counter-intuitive phenomenon where TS improves class-conditional coverage of APS/RAPS-like methods at the expense of increasing the prediction set size. It establishes a complete non-monotonic theoretical explanation and proposes practical guidelines.
tags:
  - "ICML2025"
  - "Optimization"
  - "Temperature Scaling"
  - "Conformal Prediction"
  - "Uncertainty Quantification"
  - "Calibration"
  - "Prediction Set Size"
  - "Conditional Coverage"
date: 2026-05-08
content_hash: 6d67dcac781975ea
---

# On Temperature Scaling and Conformal Prediction of Deep Classifiers

**Conference**: ICML2025  
**arXiv**: [2402.05806](https://arxiv.org/abs/2402.05806)  
**Code**: Not provided  
**Area**: LLM Evaluation  
**Keywords**: Temperature Scaling, Conformal Prediction, Uncertainty Quantification, Calibration, Prediction Set Size, Conditional Coverage

## TL;DR
This paper presents the first systematic study on the impact of Temperature Scaling (TS) calibration on Conformal Prediction (CP) methods, revealing the counter-intuitive phenomenon where TS improves class-conditional coverage of APS/RAPS-like methods at the expense of increasing the prediction set size. It establishes a complete non-monotonic theoretical explanation and proposes practical guidelines.

## Background & Motivation

Uncertainty quantification for deep classifiers mainly relies on two types of post-processing methods:

**Calibration**: Adjusting softmax values so that the maximum value better estimates the correct probability. Temperature Scaling (TS) is the most popular method, which divides logits by a temperature $T$ before applying softmax: $\hat{\pi}_T(\mathbf{x}) = \sigma(\mathbf{z}(\mathbf{x})/T)$

**Conformal Prediction (CP)**: Generating a prediction set of candidate labels that is guaranteed to contain the true label with a user-specified probability (marginal coverage guarantee).

Both are required in critical applications—calibration provides confidence estimation, while CP provides coverage guarantees. However, their interaction has rarely been studied. Although papers on APS and RAPS perform TS calibration before applying CP, no prior work has investigated exactly how TS calibration affects the performance of CP.

**Core Problem**: What impact does TS calibration have on the prediction set size and conditional coverage of CP methods (LAC, APS, RAPS)?

## Method

### Research Framework

The paper progresses in three steps:
1. **Empirical Study**: Investigating the effects of TS calibration on CP across multiple dataset-model pairs.
2. **Extended Exploration**: Extending the temperature $T$ from the calibrated value to a wide range of $[0.5, 5]$.
3. **Theoretical Analysis**: Establishing mathematical theories to explain the observed non-monotonic phenomena.

### Score Functions of Three CP Methods

- **LAC**: $s(\mathbf{x}, y) = 1 - \hat{\pi}_y(\mathbf{x})$, directly using softmax values.
- **APS**: $s(\mathbf{x}, y) = \sum_{i=1}^{L_y} \hat{\pi}_{(i)}(\mathbf{x})$, accumulating the sorted softmax values.
- **RAPS**: $s(\mathbf{x}, y) = \sum_{i=1}^{L_y} \hat{\pi}_{(i)}(\mathbf{x}) + \lambda(L_y - k_{reg})_+$, which adds a regularization term on top of APS.

Where $\hat{\pi}_{(i)}$ is the $i$-th softmax value sorted in descending order, and $L_y$ is the rank of the true label.

### Key Theoretical Results

**Theorem 4.1 (Monotonically Decreasing Threshold)**: For a sorted logit vector $\mathbf{z}$, as temperature $T$ increases, the cumulative sum of the top $L$ sorted softmax values strictly decreases:

$$T > \tilde{T} > 0 \implies \sum_{j=1}^{L} \pi_{\tilde{T},j} \geq \sum_{j=1}^{L} \pi_{T,j}$$

**Corollary 4.2**: The thresholds $\hat{q}_T$ of APS and RAPS decrease monotonically with temperature $T$.

**Theorem 4.4 (Non-monotonicity Explanation)**: Define the gap function $g(\mathbf{z}; T, M) = \sum_{i=1}^M \sigma_i(\mathbf{z}) - \sum_{i=1}^M \sigma_i(\mathbf{z}/T)$, and the boundary function $b(T)$. When $\Delta z = z_1 - z_2 > b(T)$:

- If $T > 1$: $\nabla_{z_1} g < 0$, meaning the gap for high-score samples is smaller $\rightarrow$ prediction set becomes larger.
- If $0 < T < 1$: $\nabla_{z_1} g > 0$, meaning the gap for high-score samples is larger $\rightarrow$ prediction set becomes smaller.

The boundary function $b(T)$ has a local minimum $\tilde{T}_c$ when $T > 1$, which explains the non-monotonic trend where the prediction set size increases first and then decreases.

### Mechanism Intuition

When TS increases temperature ($T > 1$):
- **Threshold Effect**: $\hat{q}_T$ decreases (which helps reduce the prediction set).
- **Dispersion Effect**: The softmax distribution becomes flatter, and the cumulative values of the top classes decrease faster (requiring more classes to reach the threshold).
- **Competing Results**: When $T$ is moderate, the dispersion effect dominates $\rightarrow$ prediction set becomes larger; when $T$ is very large, the two effects reach an equilibrium $\rightarrow$ prediction set begins to shrink.

## Key Experimental Results

### Main Table: Impact of TS Calibration on Prediction Set Size ($\alpha=0.1$)

| Dataset-Model | $T^*$ | Accuracy | APS Original | APS Calibrated | RAPS Original | RAPS Calibrated | LAC Original | LAC Calibrated |
|---|---|---|---|---|---|---|---|---|
| ImageNet-ResNet152 | 1.227 | 78.3% | 6.34 | **11.11** | 2.71 | **4.30** | 1.95 | 1.92 |
| ImageNet-ViT-B/16 | 1.180 | 83.9% | 10.10 | **19.27** | 1.93 | **2.34** | 2.22 | 2.23 |
| CIFAR100-ResNet50 | 1.524 | 80.9% | 5.31 | **9.14** | 2.88 | **4.96** | 1.62 | 1.57 |
| CIFAR100-DenseNet121 | 1.469 | 76.1% | 4.26 | **6.51** | 2.98 | **4.27** | 2.13 | 2.06 |
| CIFAR10-ResNet50 | 1.761 | 94.6% | 1.04 | 1.13 | 0.98 | 1.05 | 0.91 | 0.91 |

**Key Findings**: After TS calibration, the prediction set size of APS/RAPS **increases significantly** (e.g., ImageNet-ViT APS increases from 10.10 to 19.27), while LAC is barely affected.

### Conditional Coverage Improvement (Lower TopCovGap $\downarrow$ is Better)

| Dataset-Model | APS Original | APS Calibrated | RAPS Original | RAPS Calibrated |
|---|---|---|---|---|
| ImageNet-ResNet152 | 16.0% | **13.8%** | 17.6% | **15.2%** |
| ImageNet-ViT-B/16 | 14.2% | **12.2%** | 14.7% | **12.5%** |
| CIFAR100-ResNet50 | 12.6% | **9.0%** | 11.7% | **7.9%** |

**Key Findings**: TS calibration improves the class-conditional coverage of APS/RAPS, particularly when $T^*$ is large.

### Non-monotonic Trends (Wide-range Temperature Experiments)

- **Prediction Set Size (AvgSize)**: Increases first and then decreases as $T$ increases, peaking at $T_c$.
- **Conditional Coverage (TopCovGap)**: Decreases first and then increases as $T$ increases, reaching optimal conditional coverage at the minimum point.
- **Threshold ($\hat{q}$)**: APS/RAPS thresholds decrease monotonically with $T$ (validating Theorem 4.1).
- This pattern is **consistently reproduced** across 7 dataset-model pairs.

### Theoretical Validation

Taking CIFAR100-ResNet50 as an example: for the median sample, $\Delta z \approx 8$ and $C = 100$. The theoretical boundary yields an effective temperature range of $1.25 < T < 2.33$. The calibration temperature $T^* = 1.524$ falls precisely within this range, rigorously proving that the prediction set of the median sample will increase after calibration.

## Highlights & Insights

1. **Counter-intuitive Discovery**: Although TS calibration improves conditional coverage, it surprisingly increases the prediction set size of APS/RAPS—a trade-off that has previously gone unnoticed.
2. **Complete Non-monotonic Landscape**: Investigated not only at the calibration point $T^*$ but also across a wide range of $[0.5, 5]$, revealing the complete trade-off curve between prediction set size and conditional coverage.
3. **Rigorous Theoretical Support**: Theorems 4.1 to 4.4 construct a complete mathematical explanation chain, where the shape of the boundary function $b(T)$ perfectly explains the non-monotonic trend.
4. **Practical Value**: Provides actionable guidelines for practitioners—controlling the trade-off between prediction set size and conditional coverage by tuning the temperature $T$.
5. **Differentiated Contribution**: Clearly distinguishes the impact of TS on different CP methods (LAC is unaffected vs APS/RAPS are significantly affected), offering recommendations for selection.

## Limitations & Future Work

1. **Lack of Theoretical Analysis for Conditional Coverage**: The paper only establishes theory for prediction set size; the theoretical analysis of conditional coverage is left for future work.
2. **Limited to Classification Tasks**: Not extended to regression or other prediction tasks in CP.
3. **Only Considering TS**: Does not investigate the impact of other calibration methods (e.g., Platt Scaling, Histogram Binning) on CP.
4. **Technical Assumptions**: The theoretical analysis relies on the assumption that "the quantile sample remains the same before and after calibration," which, although supported by experiments, is not completely rigorous.
5. **Lack of Code**: The paper does not provide a public code repository, limiting reproducibility.
6. **Simplistic Practical Guidelines**: The control of the trade-off lacks automated strategies for temperature selection.

## Related Work & Insights

- **CP Foundations**: Founded by Vovk et al. (1999, 2005); APS (Romano et al., 2020) and RAPS (Angelopoulos et al., 2021) are the most popular adaptive methods.
- **Calibration**: Temperature Scaling by Guo et al. (2017) is the de facto standard.
- **Parallel Work**: Xi et al. (2024) also studied the impact of TS on CP, but only covered a small subset of the results in this paper (without considering conditional coverage and with a limited temperature range).
- **Insights**: This work suggests that we cannot simply "calibrate first and then apply CP" when designing CP pipelines; the influence of calibration on downstream CP properties must be carefully considered.

## Rating
- Novelty: ⭐⭐⭐⭐ — The first to systematically reveal the interaction between TS and CP; the counter-intuitive findings are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 dataset-model pairs, various values of $\alpha$ and CP set proportions, 100 runs of median-mean statistics.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, good progressive logic from experiments to theory, and comprehensive theoretical derivations.
- Value: ⭐⭐⭐⭐ — Direct guidance for CP practitioners, with solid theoretical contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](conformal_prediction_as_bayesian_quadrature.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/optimization/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/optimization/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[CVPR 2025\] Test-Time Augmentation Improves Efficiency in Conformal Prediction](../../CVPR2025/optimization/test-time_augmentation_improves_efficiency_in_conformal_prediction.md)

</div>

<!-- RELATED:END -->
