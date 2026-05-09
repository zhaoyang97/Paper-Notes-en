---
title: >-
  [Paper Note] FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity
description: >-
  [CVPR 2026][federated continual learning] FEAT is proposed to address the underutilization of replay exemplars in federated continual learning (FCL), mitigating cross-client heterogeneity and task-level data imbalance via geometric structure alignment (angular distillation based on ETF prototypes) and energy-based geometric correction (inference-time debiasing).
tags:
  - CVPR 2026
  - federated continual learning
  - exemplar replay
  - equiangular tight frame
  - geometric correction
  - class imbalance
date: 2026-05-08
content_hash: d67418cacfba7ef5
---

# FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity

**Conference**: CVPR 2026
**arXiv**: [2604.08617](https://arxiv.org/abs/2604.08617)
**Code**: None
**Area**: Other
**Keywords**: federated continual learning, exemplar replay, equiangular tight frame, geometric correction, class imbalance

## TL;DR

FEAT is proposed to address the underutilization of replay exemplars in federated continual learning (FCL), mitigating cross-client heterogeneity and task-level data imbalance via geometric structure alignment (angular distillation based on ETF prototypes) and energy-based geometric correction (inference-time debiasing).

## Background & Motivation

In FCL, exemplar replay is the dominant strategy for alleviating catastrophic forgetting. Existing work primarily focuses on selecting representative samples (e.g., Re-Fed, FedCBDR), while neglecting how to effectively utilize these limited exemplars. Replay introduces two persistent challenges: (1) replay data exacerbates cross-client heterogeneity; and (2) severe distributional imbalance exists between historical tasks (tail classes) and current tasks (head classes), causing tail-class features to drift toward head-class directions.

Although ETF classifiers encourage globally consistent class directions, cross-client feature alignment for tail classes remains significantly weaker than for head classes under continual dynamic heterogeneity.

## Method

### Overall Architecture

FEAT comprises two modules that are orthogonal to exemplar selection strategies and can be seamlessly integrated with existing replay methods such as Re-Fed+ and FedCBDR.

### Key Designs

1. **Geometric Structure Alignment (GSA)**: Constructs batch-level cosine similarity matrices $M_F$ for features and $M_P$ for corresponding ETF prototypes, then computes KL divergence after row-wise softmax normalization. Class-balanced aggregation is adopted—samples within each class are averaged independently before averaging across classes—ensuring tail classes receive sufficient geometric supervision.

2. **Energy-based Geometric Correction (EGC)**: At inference time, features are projected onto head-class and tail-class ETF subspaces, and the respective normalized energies are computed. During training, EMA is used to collect energy statistics of tail-class samples as priors. At inference time, components biased toward the head-class subspace are removed from the features, reducing overconfidence toward majority classes and improving sensitivity to minority classes.

3. **ETF Subspace Partitioning**: Current-task classes are treated as head classes and historical-task classes as tail classes. Orthogonal projection operators are constructed from ETF prototypes to compute the energy in head-class and tail-class subspaces separately.

### Loss & Training

$\mathcal{L} = \mathcal{L}_\text{CLS} + \lambda \cdot \mathcal{L}_\text{GSA}$. $\mathcal{L}_\text{CLS}$ is a cross-entropy loss using the similarity between ETF prototypes and features as logits. After each communication round, the server aggregates model parameters and global energy statistics. EGC is applied only at inference time, introducing no additional training cost.

## Key Experimental Results

### Main Results

| Dataset | Heterogeneity | FEAT | Prev. SOTA | Gain |
|--------|--------|------|---------|------|
| CIFAR-100 ($\alpha=0.1$) | High | Best | Multiple methods | Consistent Top-1 improvement |
| Tiny-ImageNet | Medium | Best | Multiple methods | Consistent improvement |
| Mini-ImageNet | Low | Best | Multiple methods | Consistent improvement |

### Ablation Study

| Configuration | Top-1 Accuracy | Note |
|------|-----------|------|
| Baseline (w/o FEAT) | Lower | Severe tail-class drift |
| + GSA | Improved | Better cross-client alignment |
| + EGC | Further improved | Effective inference debiasing |
| + Both | Best | Complementary effect |

### Key Findings

- GSA effectively improves cross-client feature consistency for tail classes.
- Inference-time debiasing via EGC significantly improves tail-class accuracy without additional training cost.
- FEAT is orthogonal to exemplar selection strategies and consistently improves performance when combined with Re-Fed+ and FedCBDR.

## Highlights & Insights

- Addresses the underexplored question of *how* to use replay exemplars rather than *which* to select, filling a notable research gap.
- The class-balanced KL distillation in GSA ensures fair alignment supervision for tail classes.
- EGC functions as an inference-time post-processing step with zero additional training cost, offering strong practical utility.
- The orthogonal design with respect to replay strategies enables broad applicability.

## Limitations & Future Work

- The number of ETF prototypes scales with the number of classes, potentially posing challenges in high-dimensional settings.
- The energy statistics used in EGC rely on priors collected during training, which may become inaccurate under distribution shift.

## Rating

- Novelty: ⭐⭐⭐⭐ — A fresh perspective focusing on replay utilization rather than selection.
- Technical Depth: ⭐⭐⭐⭐ — A complete design integrating ETF, angular distillation, and energy correction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across three datasets under multiple heterogeneity levels.
- Value: ⭐⭐⭐⭐ — Plug-and-play design with zero-cost inference debiasing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](../../AAAI2026/others/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)
- [\[CVPR 2026\] Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correcti.md)
- [\[ICLR 2026\] Federated ADMM from Bayesian Duality](../../ICLR2026/others/federated_admm_from_bayesian_duality.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)

</div>

<!-- RELATED:END -->
