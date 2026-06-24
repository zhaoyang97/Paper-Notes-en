---
title: >-
  [Paper Note] Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning
description: >-
  [CVPR 2025][Optimization][Federated Semi-Supervised Learning] This paper proposes TABASCO, a two-stage two-dimensional sample selection framework to address federated semi-supervised learning under joint label noise and long-tailed distributions. It utilizes two complementary metrics, Weighted JSD (WJSD) and Adaptive Centroid Distance (ACD), to identify clean samples. After GMM clustering, the remaining noisy data is leveraged in a semi-supervised manner…
tags:
  - "CVPR 2025"
  - "Optimization"
  - "Federated Semi-Supervised Learning"
  - "Noisy Labels"
  - "Long-Tailed Distribution"
  - "WJSD"
  - "Adaptive Centroid"
date: 2026-05-08
content_hash: f0023d6ebc79344f
---

# Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.13227](https://arxiv.org/abs/2503.13227)  
**Code**: [https://github.com/Wakings/TABASCO](https://github.com/Wakings/TABASCO)  
**Area**: Optimization / Federated Semi-Supervised Learning  
**Keywords**: Federated Semi-Supervised Learning, Noisy Labels, Long-Tailed Distribution, WJSD, Adaptive Centroid

## TL;DR

This paper proposes TABASCO, a two-stage two-dimensional sample selection framework to address federated semi-supervised learning under joint label noise and long-tailed distributions. It utilizes two complementary metrics, Weighted JSD (WJSD) and Adaptive Centroid Distance (ACD), to identify clean samples. After GMM clustering, the remaining noisy data is leveraged in a semi-supervised manner, achieving 85.53% accuracy on CIFAR-10 (0.1 imbalance + 0.4 noise).

## Background & Motivation

**Background**: Real-world federated learning data simultaneously faces three challenges: label noise (annotation errors), long-tailed distributions (extreme imbalance in sample sizes between head and tail classes), and distribution skew (non-IID). Existing methods typically address only one or two of these issues.

**Limitations of Prior Work**: (1) Under label noise, the standard JSD (Jensen-Shannon divergence) metric fails on imbalanced data, as the "normal" JSD of tail classes may overlap with the "noisy" JSD of head classes; (2) feature distance-based centroid methods suffer from centroid contamination when noise is present.

**Key Challenge**: Symmetric and asymmetric noise require different detection metrics—in symmetric noise, the true categories are diverse (making JSD more effective), while in asymmetric noise, there is only one target class (making feature distance more effective).

**Key Insight**: Leveraging both WJSD and ACD as complementary metrics, the method automatically selects the more suitable dimension for GMM clustering based on the current class.

**Core Idea**: WJSD (symmetric noise) + ACD (asymmetric noise) + adaptive dimension selection = clean sample identification under both noise and long-tailed distributions.

## Method

### Key Designs

1. **WJSD (Weighted Jensen-Shannon Divergence)**:

    - **Function**: Detects noisy labeled samples, being more robust to long-tailed imbalance
    - **Mechanism**: $WJSD(x_i) = W(x_i) \times JSD(x_i)$, where weights $W(x_i) = \min(\max(\mathbf{p}_i)/p_i^c, \max(\bar{\mathbf{p}}_c)/\bar{p}_c^c)$. This weight amplifies the JSD values of samples where the "predicted top class is not the annotated class"—noisy samples are usually more confident in the wrong category than the annotated one.
    - **Design Motivation**: Standard JSD yields different optimal thresholds for tail and head classes, whereas WJSD normalizes the values to make them comparable across different categories.

2. **ACD (Adaptive Centroid Distance)**:

    - **Function**: Detects asymmetric noise in the feature space
    - **Mechanism**: $ACD(x_i) = \cos(\mathbf{f}_i, \mathbf{o}_c)$, where the centroid $H_c$ is constructed using high-confidence weighted samples. The high-confidence weight $w_i = \max(1, p_i^{t_c}/\bar{p}_c^{t_c})$ enhances the contribution of true samples.
    - **Design Motivation**: Directly calculating the centroid from all samples gets contaminated by noisy samples; a purity-enhanced centroid is more reliable.

3. **Two-Dimensional Adaptive Selection**:

    - **Function**: Automatically selects the more effective dimension between WJSD or ACD for each class
    - **Mechanism**: Algorithm 1 automatically makes decisions based on the GMM clustering quality in the two dimensions

### Loss & Training

The selected clean samples are trained using standard cross-entropy and Lovász-Softmax, while noisy samples are treated as unlabeled data for semi-supervised learning.

## Key Experimental Results

| Setting | TABASCO | Prev. SOTA | Baseline |
|------|---------|---------|------|
| CIFAR-10 (0.1 imbalance, 0.4 symmetric noise) | **85.53%** | 84.25% | 71.67% |
| CIFAR-100 (0.1 imbalance, 0.4 asymmetric noise) | **59.39%** | 55.99% | 44.45% |
| CIFAR-10N (real-world noise) | Consistently +2-5% | — | — |

### Ablation Study

| Improvement | CIFAR-100 Gain |
|------|---------------|
| JSD → WJSD | +1.77% |
| CD → ACD (Purity-enhanced) | Significantly improves asymmetric scenarios |
| Single dimension → 2D adaptive | More robust under mixed noise |

### Key Findings
- WJSD dominates under symmetric noise, while ACD dominates under asymmetric noise—validating their complementarity.
- Purity enhancement brings the centroid accuracy of tail classes close to 100% (Fig. A2).
- The training cost is 4.49$\times$ the baseline (on CIFAR-100), representing a trade-off with efficiency.

## Highlights & Insights
- **First to simultaneously address federated learning under joint label noise and long-tailed distributions**—this is the most realistic setting.
- **Complementarity of 2D metrics**—instead of trying to solve all noise types with a single metric, it adaptively selects the most appropriate one.

## Limitations & Future Work
- Assumes the asymmetric noise rate is $<50\%$.
- High training cost ($4.49\times$).
- Class-level filtering only; instance-level noise within classes is not handled.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The complementary design of WJSD+ACD is effective for practical problems
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets + real-world noise
- **Writing Quality**: ⭐⭐⭐⭐ Clear
- **Value**: ⭐⭐⭐⭐ Provides a practical framework for federated learning under noise and long-tailed distributions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/optimization/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[ICLR 2026\] Enhancing Communication Compression via Discrepancy-aware Calibration for Federated Learning](../../ICLR2026/optimization/enhancing_communication_compression_via_discrepancy-aware_calibration_for_federa.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](../../NeurIPS2025/optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)
- [\[CVPR 2025\] Federated Learning with Domain Shift Eraser](federated_learning_with_domain_shift_eraser.md)
- [\[CVPR 2025\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)

</div>

<!-- RELATED:END -->
