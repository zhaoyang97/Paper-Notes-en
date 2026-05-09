---
title: >-
  [Paper Note] Test-Time Adaptation by Causal Trimming
description: >-
  [NeurIPS 2025][LLM Evaluation][test-time adaptation] This paper proposes TACT, a method that identifies non-causal directions in the representation space via data augmentation and PCA, then removes the projections of both test representations and class prototypes along these directions at test time. This reduces model reliance on non-causal features and significantly improves prediction performance under distribution shift.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - test-time adaptation
  - causal features
  - PCA
  - non-causal feature trimming
  - distribution shift
date: 2026-05-08
content_hash: 9a9c264d58efb919
---

# Test-Time Adaptation by Causal Trimming

**Conference**: NeurIPS 2025
**arXiv**: [2510.11133](https://arxiv.org/abs/2510.11133)
**Code**: [GitHub](https://github.com/NancyQuris/TACT)
**Area**: Test-Time Adaptation / Distribution Shift Robustness
**Keywords**: test-time adaptation, causal features, PCA, non-causal feature trimming, distribution shift

## TL;DR

This paper proposes TACT, a method that identifies non-causal directions in the representation space via data augmentation and PCA, then removes the projections of both test representations and class prototypes along these directions at test time. This reduces model reliance on non-causal features and significantly improves prediction performance under distribution shift.

## Background & Motivation

Test-time adaptation (TTA) aims to improve model robustness under distribution shift when only unlabeled target samples are available. Most existing TTA methods rely on model-generated pseudo-labels to guide adaptation, but when predictions are influenced by non-causal features — features without a direct causal relationship to the prediction target — pseudo-labels become unreliable, leading to suboptimal adaptation.

Non-causal features exhibit inconsistent correlations between training and test distributions and are a key factor in performance degradation. While methods such as DeYO acknowledge this issue, they only selectively use causal-feature-based predictions for model updates rather than actively suppressing non-causal influence — early predictions remain contaminated by non-causal signals, requiring many adaptation steps to mitigate.

The core motivation of TACT is to **actively identify and remove non-causal components from representations**, rather than passively waiting for them to be suppressed. Prior work shows that representations learned under standard training contain a mixture of causal and non-causal features, and that the causal components are often already well-learned but obscured by non-causal ones. Removing the non-causal directions can therefore immediately improve predictions.

## Method

### Overall Architecture

TACT operates in three steps: (1) generate augmented samples that preserve causal features while altering non-causal features; (2) apply PCA to the representations of the augmented samples to identify the directions of maximum variance as non-causal directions; (3) remove the projections of both the test representation and classifier class prototypes along these directions ("causal trimming"), then make predictions using the trimmed representations and prototypes.

### Key Designs

1. **Non-Causal Feature Identification**: Given a test sample $x$, $n$ augmented samples $\{\tilde{x}_i\}_{i=1}^n$ are generated such that causal content is preserved while non-causal attributes are varied. The representation matrix $\mathbf{Z}$ is collected, and the covariance matrix $\mathbf{\Sigma_Z} = (\mathbf{Z} - \bar{\mathbf{Z}})^\top(\mathbf{Z} - \bar{\mathbf{Z}})$ is decomposed as $\mathbf{\Sigma_Z} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$. The eigenvector $e_1$ corresponding to the largest eigenvalue is taken as the non-causal direction, since causal content remains invariant under augmentation while non-causal attributes exhibit maximum variance.

2. **Causal Trimming**: The projection of the representation onto the top $m$ principal components is removed: $\hat{z} = z - \sum_{i=1}^{m}(z \cdot e_i)e_i$. The same operation is applied to classifier class prototypes $q_j$: $\hat{q}_j = q_j - \sum_{i=1}^{m}(q_j \cdot e_i)e_i$. If causal features are invariant under augmentation and orthogonal to the non-causal directions, causal information is preserved after trimming.

3. **Moving Average Prototypes**: Since non-causal direction estimates from individual samples are noisy, a moving average of the trimmed class prototypes is maintained throughout adaptation: $\bar{\hat{q}}_j = \frac{i-1}{i}\bar{\hat{q}}_j + \frac{1}{i}\hat{q}_j^{(i)}$. Final predictions are based on the similarity between the trimmed representation and the moving average prototypes: $y = \arg\max_j \frac{\exp(\hat{z} \cdot \bar{\hat{q}}_j)}{\sum_i \exp(\hat{z} \cdot \bar{\hat{q}}_i)}$.

### Loss & Training

TACT itself is a backpropagation-free method. The authors also propose a TACT-adapt variant that uses TACT predictions as pseudo-labels to guide gradient updates: $\mathcal{L} = \mathcal{L}_{CE}(\hat{y}, y_{\text{TACT}}) + \lambda \mathcal{L}_{IM}(\hat{y})$, where $\mathcal{L}_{IM}$ is the information maximization loss from SHOT.

### Theoretical Analysis

Three propositions characterize the conditions under which TACT is effective:
- **Proposition 1**: When the top $m$ principal components cause incorrect predictions and their contribution exceeds that of the remaining components, removing them corrects the prediction.
- **Proposition 2** (Causal Preservation): Three conditions under which the trimmed representation maintains correct predictions under the causal decision boundary.
- **Proposition 3**: Conditions under which trimming does not negatively affect already correctly classified samples.

## Key Experimental Results

### Main Results

Experiments span five real-world distribution shift datasets across image (Camelyon17, ImageNet-R, ImageNet-V2), audio (Birdcalls), and text (CivilComments) modalities.

| Method | Birdcalls (F1) | Camelyon17 (Acc) | CivilComments (WG Acc) | ImageNet-R (Acc) | ImageNet-V2 (Acc) |
|------|------|------|------|------|------|
| No TTA | 22.74 | 62.31 | 55.38 | 41.83 | 62.97 |
| T3A (BP-free) | 26.16 | 69.96 | 56.43 | 41.78 | 62.93 |
| LAME (BP-free) | 23.66 | 62.38 | 56.24 | 41.77 | 63.00 |
| **TACT (BP-free)** | **31.14** | **70.17** | **71.80** | **43.59** | **63.33** |
| SHOT (BP-based) | 26.82 | 80.28 | 13.93 | 48.79 | 63.32 |
| DeYO (BP-based) | 23.29 | 69.64 | - | 46.87 | 62.96 |
| TAST (BP-based) | 26.08 | 83.01 | 56.56 | 41.09 | 62.84 |
| **TACT-adapt** | **31.25** | **83.70** | **71.98** | **48.81** | **63.44** |

TACT outperforms all BP-free methods across all benchmarks, and TACT-adapt achieves the best overall performance. Notably, on CivilComments, TACT improves over the second-best method by approximately 15%.

### Ablation Study

| Configuration | Birdcalls | Camelyon17 | CivilComments | ImageNet-R | ImageNet-V2 |
|------|------|------|------|------|------|
| No TTA | 22.74 | 62.31 | 55.38 | 41.83 | 62.97 |
| Trim z only | 25.91 | 69.43 | 67.84 | 43.21 | 63.24 |
| Trim q + avg q̂ only | 27.36 | 64.74 | 62.41 | 42.24 | 63.03 |
| Trim z + trim q + avg q̂ (full TACT) | **31.14** | **70.17** | **71.80** | **43.59** | **63.33** |

Both representation trimming and prototype trimming contribute individually, and combining them yields the best performance.

### Key Findings

- TACT takes **immediate effect** at test time without iterative updates, yielding predictions less influenced by non-causal features from the first step (Figure 1 shows it substantially outperforms DeYO in the first 100 steps).
- GradCAM visualizations show that TACT shifts model attention from non-causal regions such as backgrounds toward core causal features of the objects.
- An augmentation count of $n \in \{128, 256, 512\}$ is generally sufficient; removing 1 principal component is effective in most cases, while more complex shift scenarios may require removing additional components.
- The method is validated on Transformer architectures (ViT, DistilBERT) across image, audio, and text modalities.

## Highlights & Insights

- The approach is elegant and effective: non-causal features are identified as directions of maximum variance under augmentation and removed via linear projection, with no additional training required.
- This is the first TTA method to be evaluated across audio and text modalities, going beyond the image-only scope of prior work.
- The theoretical analysis provides clear sufficient conditions for the method's effectiveness.
- As a pseudo-label generator (TACT-adapt), TACT can enhance any gradient-based TTA method.

## Limitations & Future Work

- Domain knowledge is required to select appropriate data augmentations that alter non-causal features while preserving causal ones.
- PCA assumes that causal and non-causal features are linearly separable and orthogonal in representation space, which may not fully hold in practice.
- Future work could explore non-causal feature identification without prior knowledge and develop methods that go beyond the orthogonality assumption.

## Related Work & Insights

- Consistent with the observation in DFR that models have already learned causal features, which are merely obscured by non-causal components.
- The linear representation hypothesis is empirically validated here: semantic concepts are indeed linearly encoded in representation space.
- The method is orthogonal to other TTA approaches and can serve as a general-purpose non-causal feature mitigation module.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The integration of causal inference with PCA for TTA is novel and practically motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets, three modalities, comprehensive ablations, and theoretical analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with theoretical support.
- **Value**: ⭐⭐⭐⭐ Simple, efficient, and readily applicable to existing systems.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[NeurIPS 2025\] Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking](time_travel_is_cheating_going_live_with_deepfund_for_real-time_fund_investment_b.md)
- [\[AAAI 2026\] Test-time Diverse Reasoning by Riemannian Activation Steering](../../AAAI2026/llm_evaluation/test-time_diverse_reasoning_by_riemannian_activation_steering.md)
- [\[NeurIPS 2025\] Decoupled Entropy Minimization](decoupled_entropy_minimization.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)

<!-- RELATED:END -->
