---
title: >-
  [Paper Note] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Time-series classification] STaRFormer is proposed, which employs Dynamic Attention-based Regional Masking (DAReM) to identify task-critical regions and apply masking perturbation…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Time-series classification"
  - "contrastive learning"
  - "dynamic masking"
  - "non-stationarity"
  - "irregular sampling"
date: 2026-05-08
content_hash: ff3adf09b9d6a07a
---

# STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking

**Conference**: NeurIPS 2025  
**arXiv**: [2504.10097](https://arxiv.org/abs/2504.10097)  
**Code**: [https://star-former.github.io](https://star-former.github.io)  
**Area**: Self-Supervised Learning  
**Keywords**: Time-series classification, contrastive learning, dynamic masking, non-stationarity, irregular sampling

## TL;DR
STaRFormer is proposed, which employs Dynamic Attention-based Regional Masking (DAReM) to identify task-critical regions and apply masking perturbations, coupled with intra-batch and intra-class semi-supervised contrastive learning to embed task information into latent representations. The method comprehensively outperforms state-of-the-art baselines across 56 datasets spanning non-stationary, irregularly sampled, classification, anomaly detection, and regression settings.

## Background & Motivation

**Background**: Time-series modeling methods typically assume complete, stationary, and uniformly sampled data. Self-supervised contrastive learning approaches (e.g., TS2Vec, TimesURL) are decoupled from downstream tasks.

**Limitations of Prior Work**: Real-world sensor data frequently exhibit non-stationarity and irregular sampling (e.g., UWB ranging: 79% non-stationary); pretrained contrastive learning methods are insufficiently coupled with downstream tasks.

**Key Challenge**: Contrastive learning requires effective augmentation strategies, yet conventional random augmentations disregard task relevance. Masking task-critical regions is necessary to compel the model to learn robust representations.

**Goal**: Design a framework that couples representation learning with downstream tasks while handling non-stationarity and irregular sampling.

**Key Insight**: Dynamic attention-based masking identifies task-critical regions → masking → reconstruction → intra-batch/intra-class contrastive learning.

**Core Idea**: DAReM identifies task-critical regions → masking perturbs statistical properties → semi-supervised contrastive learning couples downstream tasks = task-aware robust time-series representations.

## Method

### Overall Architecture
A Siamese architecture is adopted: the left branch processes the original sequence for downstream tasks, while the right branch processes the masked sequence for reconstruction. Both branches share parameters and generate unmasked/masked latent representations, which are aligned via a semi-supervised contrastive loss.

### Key Designs

1. **DAReM**:

    - Function: Dynamically identifies and masks task-critical regions.
    - Mechanism: Collects attention weights → attention rollout → computes global importance scores → applies regional masking to high-importance regions.
    - Design Motivation: Masking critical regions prevents the model from relying on any single feature, thereby improving robustness.

2. **Semi-Supervised Contrastive Learning**:

    - Function: Integrates self-supervised (intra-batch) and supervised (intra-class) contrastive objectives.
    - Mechanism: Intra-batch — masked and unmasked representations of the same sequence form positive pairs; intra-class — sequences of the same class also form positive pairs.
    - Design Motivation: The semi-supervised formulation strikes a balance between purely self-supervised and purely supervised contrastive learning.

### Loss & Training
- $\mathfrak{L} = \mathfrak{L}_{Task} + \lambda_{CL}[\lambda_{fuse}\mathfrak{L}_{bw} + (1-\lambda_{fuse})\mathfrak{L}_{cw}]$

## Key Experimental Results

### Main Results (56 Datasets)

| Data Type | Task | STaRFormer | Best Baseline | Gain |
|-----------|------|-----------|---------------|------|
| Non-stationary + spatiotemporal (DKT) | Classification | **0.852** | 0.849 (Transformer) | +0.3% |
| Non-stationary + spatiotemporal (GeoLife) | Classification | **0.932** | 0.913 (ST-GRU) | +2.1% |
| Irregular sampling (P19/P12/PAM) | Classification | AUROC/Acc improved | Multiple baselines | Significant |
| 30 UEA datasets | Classification | Best on 14/30 | TARNet et al. | Overall best |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| w/o DAReM | Degraded | Masking task-critical regions is more effective |
| w/o intra-class contrastive | Degraded | Supervised signal is important for representation quality |
| w/o intra-batch contrastive | Degraded | Self-supervised objective provides additional regularization |

### Key Findings
- DAReM is particularly effective on non-stationary data.
- Semi-supervised > purely self-supervised > purely supervised.
- Generality demonstrated across 56 datasets.

## Highlights & Insights
- Masking task-critical regions is substantially more informative than masking random regions.
- Task-coupled contrastive learning more directly benefits downstream tasks.

## Limitations & Future Work
- DAReM hyperparameters require tuning.
- Validation is conducted primarily on Transformer backbones.

## Related Work & Insights
- **vs. TARNet**: TARNet performs point masking with reconstruction. STaRFormer generalizes to regional masking with semi-supervised contrastive learning.
- **vs. TS2Vec/TimesURL**: These methods adopt decoupled self-supervised learning. STaRFormer couples representation learning with downstream tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of DAReM and semi-supervised contrastive learning is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 56 datasets × 3 task types.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly articulated.
- Value: ⭐⭐⭐⭐ Provides important reference for time-series representation learning.

### Supplementary Technical Details
- DAReM involves three hyperparameters: $\varphi$ (maximum masking ratio), $\zeta$ (attention score threshold), and $\gamma$ (regional boundary).
- The contrastive learning temperature $\tau$ is set to 0.1 across all experiments.
- Validation on the BMW industrial dataset (Digital Key Trajectories) demonstrates practical deployment value.
- For regression tasks, k-means clustering is used to generate pseudo-labels, where $k$ is a hyperparameter requiring optimization.
- For anomaly detection tasks, element-wise contrastive learning is employed, incorporating both intra-class and inter-class positive pair combinations.
- Experiments in large-scale federated learning settings have been conducted in a parallel work [forstenhausler_leveraging_2025].
- Achieving the best performance on 14 out of 30 datasets in the UEA multivariate benchmark demonstrates the generality of the framework.
- The method is particularly effective on irregularly sampled medical datasets (P19/P12/PAM).
- Comparisons with ViTST indicate that directly modeling time-series is more effective than converting them to images for ViT-based processing.
- Multiple state-of-the-art F1 scores are achieved on 5 anomaly detection datasets.
- Regression tasks are validated in industrial equipment predictive maintenance scenarios.
- Parameter sharing in the Siamese network incurs only approximately 30% additional computational overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](../../AAAI2026/self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)

</div>

<!-- RELATED:END -->
