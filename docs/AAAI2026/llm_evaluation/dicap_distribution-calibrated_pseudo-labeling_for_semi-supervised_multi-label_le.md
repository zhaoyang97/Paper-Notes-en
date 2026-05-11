---
title: >-
  [Paper Note] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning
description: >-
  [AAAI 2026][LLM Evaluation][Semi-supervised learning] This paper proposes DiCaP (Distribution-Calibrated Pseudo-labeling), which estimates the posterior correctness rate of pseudo-labels to calibrate their weights…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Semi-supervised learning"
  - "multi-label learning"
  - "pseudo-labeling"
  - "calibrated weighting"
  - "contrastive learning"
date: 2026-05-08
content_hash: ae76617e1eee99cb
---

# DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning

**Conference**: AAAI 2026
**arXiv**: [2511.20225](https://arxiv.org/abs/2511.20225)
**Code**: [github.com/hb-studying/DiCaP](https://github.com/hb-studying/DiCaP)
**Area**: LLM Evaluation
**Keywords**: Semi-supervised learning, multi-label learning, pseudo-labeling, calibrated weighting, contrastive learning

## TL;DR

This paper proposes DiCaP (Distribution-Calibrated Pseudo-labeling), which estimates the posterior correctness rate of pseudo-labels to calibrate their weights, introduces a dual-threshold mechanism to separate confident and ambiguous regions with differentiated strategies, and surpasses the state of the art by up to 4.27% in semi-supervised multi-label learning.

## Background & Motivation

### Challenges in Semi-Supervised Multi-Label Learning

Multi-label learning (MLL) requires a model to predict multiple relevant labels per sample and is widely applied in image annotation, text classification, facial expression recognition, and related domains. However, obtaining complete multi-label annotations is prohibitively expensive, making **semi-supervised multi-label learning (SSMLL)**—which leverages a small amount of labeled data alongside large quantities of unlabeled data—an active research direction.

### Core Problems with Existing Methods

Current mainstream SSMLL methods are based on pseudo-labeling but suffer from key limitations:

**Uniform weighting**: Methods such as CAP and D2L assign identical weights to all pseudo-labels regardless of confidence level. Low-quality pseudo-labels are treated on par with high-quality ones, amplifying the adverse effects of noise.

**Poor confidence calibration**: Deep networks tend to produce **overconfident** predictions, leading to a significant discrepancy between predicted probabilities and actual correctness rates.

**Distribution mismatch between labeled and unlabeled data**: The correctness-rate distributions of labeled and unlabeled data differ due to different training signals, so the correctness rate estimated from labeled data cannot be directly applied to unlabeled data.

### Core Observation

**Key finding**: On the same dataset, the pseudo-label correctness-rate distribution of unlabeled data remains **stable and consistent** even as the number of labeled training samples varies.

This implies that a small subset of labeled data can be held out as an "estimation set," treated as unlabeled data to estimate the correctness-rate distribution, and that distribution can then be applied to all unlabeled data.

## Method

### Overall Architecture

DiCaP consists of five stages:

```
Labeled data D_l → Split into D_sup (80%) + D_est (20%)
                        ↓                        ↓
                Supervised training    Merged as unlabeled data into D_unsup = D_u ∪ D_est
                        ↓                        ↓
                Generate predictions  → Estimate correctness-rate distribution
                                         using ground-truth labels of D_est
                        ↓
                Dual-threshold partition: Confident samples → Weighted pseudo-labels
                                          Ambiguous samples → Contrastive learning
                        ↓
                Joint training: L_sup + L_pseudo + L_uncer
                        ↓
                Fine-tuning stage: Freeze backbone, fine-tune classifier head on D_est
```

### Key Designs

#### 1. **Distribution-Calibrated Weighting (DCW)**: Theoretically Optimal Pseudo-label Weighting

**Theoretical derivation**: By minimizing the BCE loss between pseudo-label weights and the correctness indicator function, the optimal weight is derived as the posterior correctness rate:

$$w^*(p_i) = P(\hat{y} = y \mid p_i)$$

i.e., the probability that the pseudo-label is correct given confidence $p_i$.

**Practical estimation**: The confidence interval $[0,1]$ is uniformly divided into $K=20$ bins; for each bin $\mathcal{B}_k$:

$$r_k^{pos} = \frac{n_k^{pos}}{n_k^{pos} + n_k^{neg} + \epsilon}$$

where $n_k^{pos}$ and $n_k^{neg}$ are the counts of true positives and true negatives in that bin, computed on the estimation set $\mathcal{D}_{est}$ (which has ground-truth labels).

**Linear interpolation smoothing**: For any confidence $p$, the weight is obtained via linear interpolation between adjacent bins:

$$w(p, \hat{y}=1) = \left(\frac{k+1}{K} - p\right) r_k^{pos} + \left(p - \frac{k}{K}\right) r_{k+1}^{pos}$$

**Design Motivation**: This approach is more reliable than directly using prediction confidence as weights (experiments show that direct confidence weighting performs even worse than uniform weighting) and more accurate than estimating from labeled data (due to distribution mismatch).

#### 2. **Dual-Threshold Pseudo-labeling (DTH)**: Separating Confident and Ambiguous Regions

For each class $c$, dynamic thresholds are computed from the median prediction scores of positive and negative labeled samples:

$$\tau_c^{pos} = \frac{\max(\mathcal{P}_c^{pos}) + \min(\mathcal{P}_c^{pos})}{2}, \quad \tau_c^{neg} = \frac{\max(\mathcal{P}_c^{neg}) + \min(\mathcal{P}_c^{neg})}{2}$$

The prediction score of each unlabeled sample is partitioned into three regions:

$$\hat{y}_{uc} = \begin{cases} 1 & \text{if } p_{uc} > \tau_c^{pos} \quad \text{(confident positive)} \\ 0 & \text{if } p_{uc} < \tau_c^{neg} \quad \text{(confident negative)} \\ -1 & \text{if } \tau_c^{neg} \leq p_{uc} \leq \tau_c^{pos} \quad \text{(ambiguous)} \end{cases}$$

- **Confident samples**: Assigned pseudo-labels with calibrated weights, trained with weighted ASL loss.
- **Ambiguous samples**: No hard labels assigned; contrastive learning regularization is applied instead.

#### 3. **Uncertain-sample Robust Representation Learning (URRL)**

**Class-level contrastive learning** is applied to ambiguous samples, extending standard InfoNCE to the multi-label setting:

- Each sample $x_i$ generates a weakly augmented view $x_i^w$ and a strongly augmented view $x_i^s$.
- Class-level feature embeddings $\{z_{ic}^w\}, \{z_{ic}^s\}$ are extracted.
- The two views of the same sample and same class form positive pairs; all others are negative pairs.

$$\mathcal{L}_{uncer} = -\frac{1}{2B}\sum_{i=1}^{2B}\log\frac{\exp(z_i \cdot z_i^+ / \tau)}{\sum_{j=1}^{2B}\mathbb{I}_{i\neq j}\exp(z_i \cdot z_j / \tau)}$$

**Design Motivation**: Ambiguous samples are ill-suited for hard-label supervision, yet their feature representations remain valuable. Contrastive learning exploits these samples to improve representations without introducing noisy gradients.

### Loss & Training

**Overall loss**:

$$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{pseudo} + \mathcal{L}_{uncer}$$

- $\mathcal{L}_{sup}$: Supervised ASL loss on $\mathcal{D}_{sup}$
- $\mathcal{L}_{pseudo}$: Weighted pseudo-label ASL loss on confident samples
- $\mathcal{L}_{uncer}$: Class-level contrastive loss on ambiguous samples

**Fine-tuning stage**: The backbone is frozen and only the classifier head is fine-tuned on $\mathcal{D}_{est}$ using its ground-truth labels for 20 epochs:

$$\mathcal{L}_{ft} = \frac{1}{|\mathcal{D}_{est}|}\sum_{(x,y)\in\mathcal{D}_{est}} \ell(f_\theta(x), y)$$

**Training details**: ResNet-50 backbone + ML-Decoder, AdamW optimizer, OneCycleLR scheduler, EMA (decay rate 0.9997), RandAugment + Cutout data augmentation.

## Key Experimental Results

### Main Results

Comprehensive comparison against 11 methods across 4 datasets × 4 labeling ratios (mAP%):

| Method | VOC 5% | VOC 10% | COCO 5% | COCO 10% | NUS 5% | AWA 5% |
|--------|--------|---------|---------|----------|--------|--------|
| BCE | 65.40 | 75.48 | 57.09 | 62.34 | 40.12 | 61.33 |
| ASL | 71.41 | 77.81 | 57.87 | 62.95 | 42.04 | 60.40 |
| CAP | 75.90 | 81.83 | 62.88 | 67.18 | 44.98 | 63.90 |
| PCLP | 77.25 | 82.21 | 64.43 | 69.02 | 46.39 | 64.30 |
| BBAM | 78.66 | 83.45 | 63.54 | 67.41 | 33.15 | 64.19 |
| D2L | 79.26 | 84.06 | 69.30 | 73.06 | 46.86 | 64.66 |
| **DiCaP** | **83.53** | **87.92** | **70.07** | **73.55** | **48.37** | **66.32** |
| Δ vs D2L | **+4.27** | **+3.86** | **+0.77** | **+0.49** | **+1.51** | **+1.66** |

### Ablation Study

Incremental component addition on COCO and NUS (average mAP):

| Configuration | Avg. mAP (%) | Gain |
|---------------|-------------|------|
| Baseline (labeled data only) | 55.82 | — |
| + DCW (calibrated weighting) | 58.73 | **+2.91** |
| + DTH (dual threshold) | 59.91 | +1.18 |
| + URRL (contrastive learning) | 60.25 | +0.34 |
| + WCL (warm-up contrastive) | 60.40 | +0.15 |
| + FTE (fine-tune on estimation set) | **60.81** | +0.41 |

**Comparison of weighting strategies** (COCO, averaged across labeling ratios):

| Weighting Strategy | Avg. mAP (%) | Notes |
|--------------------|-------------|-------|
| Uniform | 72.04 | Equal weights |
| Confidence | 71.84 | Direct predicted probability (**worse** than uniform) |
| Labeled | 72.42 | Estimated from labeled data |
| **DiCaP** | **73.01** | Distribution-calibrated via estimation set |
| Optimal | 73.09 | Computed with ground-truth labels (upper bound) |

### Key Findings

1. **DiCaP approaches the theoretical optimum**: Only 0.08% below Optimal (which uses ground-truth labels).
2. **Using confidence directly as weights is harmful**: Deep networks are overconfident, causing severe calibration bias.
3. **DCW contributes the most** (average +2.91%), validating the central importance of correctness-rate calibration.
4. **Clear efficiency advantage**: Compared to D2L, GPU memory is reduced by ~68% (4.44 vs. 14.17 GB on COCO) and training speed improves by ~15%.
5. **Estimation distribution is highly accurate**: Even with only 57 estimation samples (VOC 5%), the estimated distribution nearly perfectly matches the true distribution.

## Highlights & Insights

- **Closed loop from theory to practice**: The paper derives the optimal weight theoretically → discovers distributional stability → designs a practical estimation strategy, forming a complete logical chain.
- **The empirical "stability" finding is highly valuable**: The correctness-rate distribution of unlabeled data does not change with the amount of labeled data—a seemingly simple observation that provides a solid foundation for the entire method.
- **Dual threshold + contrastive learning**: Elegantly handles the "ambiguous zone" problem, avoiding hard binary boundaries.
- **Larger gains under extreme label scarcity**: A 4.27% improvement at 5% labeling ratio (VOC) demonstrates that the method is most valuable when labels are extremely scarce.

## Limitations & Future Work

1. **Fixed estimation set split ratio**: Using 20% as the estimation set may not be optimal across all scenarios.
2. **Fixed bin count $K=20$**: Finer-grained or adaptive binning strategies may yield further improvements.
3. **Validation limited to image tasks**: Other modalities such as text classification and video annotation have not been explored.
4. **Diminishing returns from contrastive learning**: URRL contributes only +0.34%, suggesting that stronger unsupervised signals may be needed.
5. **Dependence on ResNet-50 backbone**: Performance on architectures such as ViT has not been verified.

## Related Work & Insights

- **Pseudo-labeling methods**: Single-label semi-supervised methods such as FixMatch and FlexMatch.
- **Multi-label specific**: CAP (class-level thresholding), D2L (metric-adaptive thresholding), PCLP (causal priors).
- **Calibration methods**: Temperature Scaling, Mixup Calibration.
- **Inspiration**: The distributional stability observation may generalize to other tasks, such as pseudo-bounding-box quality estimation in semi-supervised object detection.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of distribution-calibrated weighting and dual thresholding is novel; the core observation is valuable)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets × 4 labeling ratios × 11 baselines + complete ablation + efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Theoretical derivations are clear; visualizations are intuitive)
- Value: ⭐⭐⭐⭐ (Direct contribution to the semi-supervised multi-label community, though generalizability awaits further verification)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](../../NeurIPS2025/llm_evaluation/keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/llm_evaluation/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/llm_evaluation/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[AAAI 2026\] SpikCommander: A High-Performance Spiking Transformer with Multi-View Learning for Efficient Speech Command Recognition](spikcommander_a_high-performance_spiking_transformer_with_multi-view_learning_fo.md)

</div>

<!-- RELATED:END -->
