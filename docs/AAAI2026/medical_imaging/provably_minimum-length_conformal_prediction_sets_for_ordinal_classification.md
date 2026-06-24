---
title: >-
  [Paper Note] Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification
description: >-
  [AAAI 2026][Medical Imaging][Conformal Prediction] This paper proposes min-CPS and its regularized variant min-RCPS, a model-agnostic conformal prediction method for ordinal classification. By solving the minimum-length prediction interval for each sample via a linear-time sliding window algorithm, the method reduces average prediction set size by 15% while maintaining coverage guarantees, with theoretical optimality guarantees at the instance level.
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Conformal Prediction"
  - "Ordinal Classification"
  - "Prediction Set Size"
  - "Uncertainty Quantification"
  - "Sliding Window Algorithm"
  - "Model-Agnostic"
date: 2026-05-08
content_hash: 44fca0d20a6505e9
---

# Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification

**Conference**: AAAI 2026
**arXiv**: [2511.16845](https://arxiv.org/abs/2511.16845)  
**Code**: [github.com/xrty/OCP](https://github.com/xrty/OCP)  
**Area**: Medical Imaging / Uncertainty Quantification / Ordinal Classification
**Keywords**: Conformal Prediction, Ordinal Classification, Prediction Set Size, Uncertainty Quantification, Sliding Window Algorithm, Model-Agnostic

## TL;DR
This paper proposes min-CPS and its regularized variant min-RCPS, a model-agnostic conformal prediction method for ordinal classification. By solving the minimum-length prediction interval for each sample via a linear-time sliding window algorithm, the method reduces average prediction set size by 15% while maintaining coverage guarantees, with theoretical optimality guarantees at the instance level.

## Background & Motivation

**Background**: Ordinal classification is widely used in high-stakes applications (medical diagnosis grading, credit assessment, age estimation), where labels have a natural ordering but unknown spacing. Conformal Prediction (CP) is a general uncertainty quantification framework that provides distribution-free coverage guarantees.

**Limitations of Prior Work** in ordinal CP:
- **Ordinal APS** (Lu et al. 2022): Greedily expands from the highest-probability label to neighboring labels, but this greedy search is **heuristic** with no optimality guarantee and may produce unnecessarily large prediction intervals.
- **COPOC** (Dey et al. 2023): Requires the model to output a unimodal distribution via a learned auxiliary module, which breaks the **model-agnostic** property of CP.
- Neither approach provides theoretical analysis from the perspective of prediction efficiency.

**Key Challenge**: Existing methods are either heuristic (no optimality guarantee) or model-dependent (not model-agnostic), and neither rigorously analyzes the coverage-efficiency trade-off.

**Key Insight**: Formalize ordinal CP as an **instance-level minimum-length covering problem** and design provably optimal algorithms.

## Method

### Problem Setup

- Input $X \in \mathcal{X}$, ordinal label $Y \in \{1, 2, \ldots, K\}$
- Model $f(X)$ outputs a probability distribution over classes
- Goal: Construct a contiguous interval $\hat{C}_\tau(X) = \{y: l(X;\tau) \le y \le u(X;\tau)\}$ satisfying marginal coverage $\mathbb{P}\{Y \in \hat{C}_\tau(X)\} \ge 1-\alpha$

### Instance-Level Minimum-Length Covering

For any input $X$ and threshold $\tau$, solve:

$$\min_{(l,u) \in \mathcal{U}(X;\tau)} \ell(l,u) \triangleq u - l$$

subject to:
1. **Coverage probability**: $\sum_{k=l}^{u} f(X)_k \ge \tau$
2. **Anchor inclusion**: $l \le \hat{y}^*(X) \le u$ (where $\hat{y}^*$ is the mode of the predicted distribution)

### Sliding Window Algorithm (Algorithm 1)

**Key Designs**:
1. Precompute prefix sums $P_k = \sum_{i=1}^k f(X)_i$ to enable $O(1)$ interval probability queries.
2. Outer loop: $u$ increases monotonically from $\hat{y}^*$ to $K$.
3. Inner loop: $l$ moves from left to right as long as coverage $P_u - P_{l-1} \ge \tau$ and $l \le \hat{y}^*$.
4. Record the shortest interval satisfying all constraints.

**Theoretical Guarantee (Theorem 1)**:
- Algorithm 1 exactly solves the minimum-length covering problem.
- Time complexity $O(K)$ (linear! vs. brute-force $O(K^2)$).
- Model-agnostic: holds for any predicted distribution (no unimodal or other distributional assumption required).

### min-CPS: Calibrating Threshold $\tau$

Binary search determines $\tau$ such that empirical coverage $F(\tau) \ge 1-\alpha$:

$$F(\tau) = \frac{1}{n} \sum_{i=1}^n \mathbb{1}[l^*(X_i;\tau) \le Y_i \le u^*(X_i;\tau)]$$

**Lemma 1**: If $f(X)$ satisfies **radial monotonicity** (i.e., probabilities decrease from the mode outward), then $F(\tau)$ is monotonically non-decreasing in $\tau$, guaranteeing binary search convergence.

**Theorem 2**: Under radial monotonicity, the calibrated threshold of min-CPS provides a $(1-\alpha)$ marginal coverage guarantee.

Total time complexity: $O(\log(1/\epsilon) \cdot n \cdot K)$.

### min-RCPS: Length-Regularized Variant

**Design Motivation**: Samples with larger prediction intervals have higher uncertainty and should be penalized to reduce unnecessary expansion.

The feasibility constraint is modified to:
$$\sum_{k=l}^u f(X)_k - \lambda \cdot \ell(l,u) \ge \tau$$

A linear penalty $\lambda$ on interval length is introduced, requiring larger intervals to accumulate higher probability mass to satisfy the constraint. Setting $\lambda=0$ recovers min-CPS.

**Corollary 1**: min-RCPS preserves exchangeability and retains standard CP coverage guarantees.

## Key Experimental Results

### Datasets (4, across multiple modalities)

| Dataset | Type | # Classes | Description |
|--------|------|--------|------|
| UTKFace | Image | ~100 | Facial age estimation |
| Avocado Price | Tabular | ~50 | Avocado price grading |
| Electric Motor Temp. | Time series | ~80 | Motor temperature monitoring |
| IMDB | Image | ~100 | Facial age estimation (large-scale) |

### Baselines
- Naive CDF: Constructs intervals directly from cumulative probabilities
- Ordinal APS: Adaptive prediction sets for ordinal classification
- WCRC (Weighted CRC): Weighted conformal risk control

### Main Results ($\alpha=0.1$)

| Dataset | min-CPS Set Size↓ | min-RCPS Set Size↓ | vs. Ordinal APS |
|--------|----------------|-----------------|----------------|
| Temperature | 5.09 | 4.99 | -5.11% / -6.96% |
| UTKFace | 29.77 | 29.77 | -7.00% / -7.01% |
| Avocado Price | 9.12 | 8.92 | **-39.93%** / **-41.21%** |
| IMDB | 28.18 | 28.26 | -3.49% / -3.16% |
| **Average** | — | — | **-14%** / **-15%** |

Coverage $\ge 1-\alpha$ is achieved by all methods, confirming the theoretical guarantees.

### Sensitivity Analysis over $\alpha$ (Avocado Price)

| $\alpha$ | min-CPS Reduction | min-RCPS Reduction |
|----------|--------------|---------------|
| 0.10 | -39.93% | -41.21% |
| 0.05 | -36.22% | -35.58% |
| 0.01 | -24.80% | -26.38% |

Averaged across all $\alpha$: min-CPS -33.65%, min-RCPS -34.39%.

### Key Findings

1. **Largest gains on Avocado Price** (~40%), as the irregular label distribution in this dataset causes greedy methods to produce many unnecessarily large intervals.
2. **Empirical monotonicity holds**: The monotonicity of $F(\tau)$ is verified across all experiments (Figure 2), supporting the theoretical assumption.
3. **min-RCPS marginally outperforms min-CPS** in most settings; the $\lambda$ regularization effectively compresses intervals for high-uncertainty samples.

## Highlights & Insights

1. **From heuristic to provably optimal**: This work is the first to provide instance-level optimality guarantees for prediction efficiency in ordinal CP.
2. **Linear-time algorithm**: The sliding window reduces complexity from $O(K^2)$ to $O(K)$, which is critical for tasks with many classes (e.g., age estimation with $K=100+$).
3. **Fully model-agnostic**: No unimodal or distributional assumptions are required (unlike COPOC), fully preserving the distribution-free advantage of CP.
4. **Elegant intuition for min-RCPS**: Larger interval $\Rightarrow$ higher uncertainty $\Rightarrow$ higher probability mass required for coverage $\Rightarrow$ natural compression.
5. **Closed loop between theory and experiments**: Theorem 1 (optimality) $\to$ Lemma 1 (monotonicity condition) $\to$ Theorem 2 (coverage guarantee) $\to$ empirical validation of monotonicity.

## Limitations & Future Work

1. Coverage guarantees rely on the **radial monotonicity assumption** (Lemma 1), which holds in all experiments but may fail for severely multimodal distributions.
2. Only marginal coverage is guaranteed; class-conditional coverage is not addressed.
3. The $\lambda$ hyperparameter of min-RCPS requires tuning on a validation set, introducing additional sensitivity.
4. Training details of the base models are not fully described, making it difficult to assess min-CPS performance with weak models.
5. The method has not been validated in real medical diagnosis applications, despite medical imaging being mentioned in the title; experiments primarily use facial age and commodity price datasets.

## Related Work & Insights

- **Conformal Prediction**: Standard CP (Vovk 2005), APS (Romano 2020), RAPS, SAPS, RC3P
- **Ordinal CP**: Ordinal APS (Lu 2022, greedy heuristic), COPOC (Dey 2023, requires unimodality), WCRC (Xu 2023, weighted risk control)
- **Ordinal Classification**: Facial age estimation, diabetic retinopathy grading, aesthetic quality assessment

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐⭐ — Formalizes ordinal CP as a minimum-length covering problem and provides a provably optimal solution; strong theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets and multiple $\alpha$ values; improvements are significant and consistent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The theorem–lemma–corollary logical chain is clear; algorithm pseudocode is complete.
- **Value**: ⭐⭐⭐⭐ — Linear-time, plug-and-play; applicable to any ordinal classification model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](../../ICML2026/medical_imaging/cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[ICLR 2026\] CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition](../../ICLR2026/medical_imaging/consign_conformal_segmentation_informed_by_spatial_groupings_via_decomposition.md)
- [\[CVPR 2026\] LATA: Laplacian-Assisted Transductive Adaptation for Conformal Uncertainty in Medical VLMs](../../CVPR2026/medical_imaging/lata_laplacian-assisted_transductive_adaptation_for_conformal_uncertainty_in_med.md)
- [\[AAAI 2026\] Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction](cross-sample_augmented_test-time_adaptation_for_personalized_intraoperative_hypo.md)

</div>

<!-- RELATED:END -->
