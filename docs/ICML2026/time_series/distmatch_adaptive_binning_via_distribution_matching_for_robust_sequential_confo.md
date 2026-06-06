---
title: >-
  [Paper Note] DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal
description: >-
  [ICML 2026][Time Series][Sequential Conformal Prediction] DistMatch proposes a recursive binning method based on **KS statistics**—it **discards weight reassignment** by grouping residuals into approximately exchangeable…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Sequential Conformal Prediction"
  - "Distribution Shift"
  - "Kolmogorov-Smirnov"
  - "Adaptive Binning"
date: 2026-05-08
content_hash: 6ac60f537bb7c4fe
---

# DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal

**Conference**: ICML 2026  
**arXiv**: [2606.00690](https://arxiv.org/abs/2606.00690)  
**Code**: To be confirmed  
**Area**: Time Series / Uncertainty Quantization / Conformal Prediction  
**Keywords**: Sequential Conformal Prediction, Distribution Shift, Kolmogorov-Smirnov, Adaptive Binning

## TL;DR
DistMatch proposes a recursive binning method based on **KS statistics**—it **discards weight reassignment** by grouping residuals into approximately exchangeable leaf nodes, providing effective conformal prediction intervals under distribution shift; it achieves the smallest interval width across 5 datasets while maintaining valid coverage.

## Background & Motivation

**Background**: Sequential conformal prediction provides valid uncertainty quantification by constructing prediction intervals, but traditional methods assume residual exchangeability—an assumption often violated in real-world time series. Existing methods primarily approximate exchangeability through residual weight reassignment.

**Limitations of Prior Work**:
- Weight reassignment schemes (time-weighting methods) struggle to accurately estimate weights and easily discard informative early samples during distribution jumps.
- Similarity retrieval methods are highly sensitive to retrieval quality; even small similarity estimation errors may assign excessive weights to irrelevant or noisy samples.
- Continuous weight allocation distorts the empirical distribution of residuals, leading to inaccurate quantile estimation.

**Key Challenge**: How to handle distribution shifts in time series and ensure conformal coverage without relying on precise weight estimation.

**Goal**: Design a binning method that does not require weight reassignment, inducing approximate local exchangeability by grouping similar samples to achieve robustness against distribution shifts.

**Key Insight**: Using non-parametric KS statistics for distribution similarity measurement avoids dependence on temporal assumptions like global stationarity; compared to weight reassignment, binning methods better preserve the statistical properties of residuals by maintaining the integrity of the empirical distribution.

**Core Idea**: Replace weight schemes with a recursive binary tree driven by KS statistics—recursively grouping residuals into leaves with bounded distribution distances, where each leaf independently applies online quantile regression for locally adaptive robust inference.

## Method

### Overall Architecture
Two stages—**Training Stage**: Given calibration set residuals, construct residual patches $\tilde{\epsilon}_t = \{\epsilon_{t - w + 1}, \ldots, \epsilon_t\}$ paired with target residuals $\epsilon_{t+1}$; recursively select split anchors by maximizing matching scores to group patches into leaves satisfying KS distance bounds. **Inference Stage**: For a new patch $\tilde{\epsilon}_T$, route it to the corresponding leaf by recursively comparing its KS distance with split anchors, then use a Quantile Regression Forest (QRF) within the leaf to estimate quantiles and construct prediction intervals; simultaneously update the QRF within the leaf online to adapt to new observations.

### Key Designs

1. **Recursive Binning via KS Statistics**:

    - Function: Group residuals into independent subgroups through distribution similarity, inducing approximately exchangeable leaf nodes.
    - Mechanism: Define matching gain score $\text{MG}(\tilde{\epsilon}_i) = \sum_j \mathbb{1}\{D_{\text{KS}}(\tilde{\epsilon}_i, \tilde{\epsilon}_j) \leq \gamma\}$, selecting the anchor $s$ that maximizes this score at each split node. KS distance $D_{\text{KS}} = \sup_x |F_i(x) - F_j(x)|$ measures the maximum deviation between two empirical distribution functions and is a density-independent non-parametric metric. Recursively split until no better split satisfying the minimum sample size $n_{\min}$ can be found.
    - Design Motivation: Compared to weight reassignment, binning maintains the integrity of the empirical distribution; compared to alternatives like TV distance, KS statistics are computationally efficient ($O(w)$ vs $O(n)$) and possess strong discriminative power for skewed distributions.

2. **Residual Patch + Target-level Exchangeability**:

    - Function: Establish a theoretical bridge between patch-level grouping and target-level exchangeability to ensure valid prediction coverage.
    - Mechanism: Define $\gamma^*$-approximate local exchangeability as $\max_{t \in \mathcal{L}_{k^*}} D_{\text{KS}}(P_{t+1}, P_{T+1}) \leq \gamma^*$, meaning the KS distance between the cumulative distributions of all target residuals in a leaf and the unseen target distribution is bounded. Under local stationarity and $\beta$-mixing assumptions, it can be proven that if the patch-level KS bound is $2\gamma$, the target-level bound is $2 C \gamma + \mathcal{O}(\sigma_{\text{mix}})$.
    - Design Motivation: Compared to global exchangeability, local exchangeability allows for mild distribution shifts; achieving target-level guarantees through patch-level tests avoids the difficulty of directly estimating future distributions.

3. **Online Adaptation + Ensemble Robustness**:

    - Function: Maintain valid coverage in long sequences and handle severe distribution shifts.
    - Mechanism: An ensemble of $B$ bootstrap trees, each built with bootstrap samples at a sampling ratio $\theta$. For each unseen patch, at least one tree routes it to the matching leaf with probability $p_{\min}$; robust predictions are obtained by averaging quantile estimates $\bar{q} = \frac{1}{B} \sum_b q^{(b)}$ from $B$ trees. Update the QRF of the corresponding leaf after each new residual observation without changing the tree structure.
    - Design Motivation: Ensemble strategies provide multiple fallback routing paths under extreme shifts; online QRF updates allow local quantiles to adaptively evolve without expensive tree reconstruction, maintaining an online cost of $O(T w \log n)$.

## Key Experimental Results

### Main Results (5 Real-world Datasets, α = 0.1)

| Dataset | Method | Coverage ↑ | Interval Width ↓ | Winkler Score ↓ |
|--------|------|---------|----------|------------|
| Elec. | **Ours** | 0.92 | **0.27** | **1.97** |
| Elec. | SPCI | 0.90 | 0.28 | 2.54 |
| Solar | **Ours** | 0.91 | **60.00** | **1.54** |
| Solar | SPCI | 0.85 | 47.36 | 1.98 |
| Wind | **Ours** | 0.90 | **69.04** | **2.15** |
| Wind | SPCI | 0.83 | 63.14 | 2.19 |

DistMatch achieves the smallest interval width across all 5 datasets while maintaining valid coverage.

### Ablation Study

| Configuration | Elec. | Solar | Wind | Avg. Winkler |
|------|-------|-------|-------|-----------|
| Full Model (γ = 0.1, w = 100) | 0.92 | 0.91 | 0.90 | 1.95 |
| w/o KS (using Wasserstein) | 0.91 | 0.90 | 0.89 | 3.42 |
| w/o KS (using KL Divergence) | 0.88 | 0.86 | 0.82 | Coverage Failed |
| w/o Ensemble (Single Tree) | 0.91 | 0.89 | 0.88 | 2.34 |

### Key Findings
- KS statistics perform better than Wasserstein and KL divergence while having the lowest computational cost ($O(w)$ vs $O(n^2)$).
- Ensemble mechanisms are critical in severe shift scenarios.
- The hyperparameter $\gamma$ shows good stability within 0.1, effectively managing the bias-variance trade-off.

## Highlights & Insights
- **Innovative theoretical framework**: Establishes theoretical guarantees for binning-based sequential CP using approximate local exchangeability for the first time; derives target-level bounds from patch-level KS bounds to avoid direct modeling of future distributions.
- **Elegant design choices**: KS statistics as a distribution matching criterion are simple and effective (non-parametric, density-independent) with clear geometric interpretations (maximum deviation of empirical CDF); compared to continuous weights, discrete binning naturally preserves the integrity of empirical distributions.
- **Online robustness mechanism**: The combination of ensembles and online updates enables DistMatch to maintain valid coverage under extreme shifts; computational efficiency is improved by $T$ times compared to SPCI and HopCPT.
- **Transferable ideas**: The concept of distribution-matching binning can be extended to other uncertainty quantification tasks requiring handling of distribution shifts (risk calibration, probabilistic forecasting).

## Limitations & Future Work
- Actual fulfillment of theoretical assumptions: Relies on local stationarity and $\beta$-mixing, which may fail in sequences with long memory or extreme non-stationarity.
- Hyperparameter sensitivity: Although stable for $\gamma \in [0.05, 0.15]$, optimization via greedy search is still required for new datasets.
- Sample size requirements: Calibration set size $n$ affects tree construction cost $O(n^2 w \log n)$.
- Future work: Adaptive $\gamma$ selection mechanisms; expansion to multi-dimensional outputs or multi-step forecasting; exploring other time-series tasks (anomaly detection, demand forecasting).

## Related Work & Insights
- **vs SPCI**: Relies on sliding window model updates to capture distribution changes, which is prone to failure under extreme shifts; DistMatch achieves adaptation through distribution-matching binning without retraining the predictor.
- **vs HopCPT**: Uses Hopfield networks to retrieve similar past residuals, making retrieval quality sensitive; DistMatch avoids amplification of similarity estimation errors via global KS matching.
- **vs KOWCPI**: Depends on kernel methods for weight calculation, which are sensitive to kernel choice; DistMatch avoids weights entirely in favor of discrete binning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First to introduce KS statistics into sequential CP and replace weights with binning, featuring an innovative theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐  Validated across 5 real datasets with 8 baselines plus ablation and theoretical verification; lacks experiments on extreme data and cross-domain generalization.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logic, standardized notation, rigorous theoretical derivation, and thorough experimental analysis.
- Value: ⭐⭐⭐⭐  Sequential CP is a practical problem, and DistMatch shows significant performance; the theoretical framework is highly relevant to the uncertainty quantification community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Time Series Reasoning via Segment Selection](adaptive_time_series_reasoning_via_segment_selection.md)
- [\[ICML 2026\] Doubly Outlier-Robust Online Infinite Hidden Markov Model](doubly_outlier-robust_online_infinite_hidden_markov_model.md)
- [\[ICML 2026\] Divide and Contrast: Learning Robust Temporal Features Without Augmentation](divide_and_contrast_learning_robust_temporal_features_without_augmentation.md)
- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](../../ICLR2026/time_series/rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
