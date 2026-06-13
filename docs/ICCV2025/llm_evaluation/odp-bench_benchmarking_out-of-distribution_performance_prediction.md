---
title: >-
  [Paper Note] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction
description: >-
  [ICCV 2025][LLM Evaluation][OOD performance prediction] This paper presents ODP-Bench, the first comprehensive benchmark for OOD performance prediction, covering 29 OOD datasets, 10 prediction algorithms, and 1…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "OOD performance prediction"
  - "distribution shift"
  - "benchmark evaluation"
  - "robustness assessment"
  - "model selection"
date: 2026-05-08
content_hash: bd9e741547e72a52
---

# ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction

**Conference**: ICCV 2025
**arXiv**: [2510.27263](https://arxiv.org/abs/2510.27263)  
**Code**: [https://github.com/h-yu16/Performance_Prediction/](https://github.com/h-yu16/Performance_Prediction/)  
**Area**: LLM Evaluation
**Keywords**: OOD performance prediction, distribution shift, benchmark evaluation, robustness assessment, model selection

## TL;DR
This paper presents ODP-Bench, the first comprehensive benchmark for OOD performance prediction, covering 29 OOD datasets, 10 prediction algorithms, and 1,444 pretrained models. It reveals a key finding that existing algorithms perform reasonably well on synthetic corruptions but consistently fail under natural distribution shifts.

## Background & Motivation

**Background**: OOD performance prediction aims to estimate how well a trained model performs on an unlabeled OOD test set, enabling safer deployment in risk-sensitive scenarios such as autonomous driving and medical imaging. Recent methods approach this from multiple angles, including model confidence, distributional discrepancy, and model agreement.

**Limitations of Prior Work**:
- **Inconsistent evaluation**: Evaluation protocols vary substantially across the literature, with no consensus on training models, test datasets, or evaluation metrics.
- **Insufficient coverage**: Most works rely on a limited number of OOD datasets, focusing primarily on synthetic corruption shifts while rarely addressing important OOD scenarios such as domain generalization and subpopulation shift.
- **Narrow shift types**: Real-world distribution shifts arising from camera location, image background, and demographic attributes are seldom studied.

**Key Challenge**: Without fair and unified evaluation conditions, it is impossible to accurately characterize the capability boundaries and applicable scope of each algorithm. Although OOD generalization methods (invariant learning, domain generalization, etc.) have advanced, experiments consistently show that no algorithm substantially improves OOD performance, making direct prediction and model selection based on predicted performance increasingly important.

**Goal**: Establish a unified, comprehensive, and fair benchmark for OOD performance prediction, enabling fair comparison of different algorithms under identical conditions while thoroughly analyzing their capability boundaries.

**Key Insight**: Simultaneously expand along three dimensions — dataset coverage, diversity of distribution shifts, and number of pretrained models — and provide 1,444 ready-to-use pretrained models as a testbench to avoid redundant training.

**Core Idea**: Expose the fundamental limitations of existing OOD performance prediction algorithms on natural distribution shifts through a unified large-scale benchmark.

## Method

### Overall Architecture
ODP-Bench consists of three components: (1) 29 OOD datasets covering diverse distribution shift types; (2) 1,444 pretrained models with varying architectures, initializations, and training algorithms, serving as a testbench; and (3) 10 practical performance prediction algorithms. Given a trained model $f_{\theta_0}$, a labeled validation set $\{x_i^{va}, y_i^{va}\}_{i=1}^{n_{va}}$, and an unlabeled OOD test set $\{x_i^{te}\}_{i=1}^{n_{te}}$, the goal is to predict the model's performance on the test set or to compute a proxy score positively correlated with the true performance.

### Key Designs

1. **Dataset Design (29 OOD Datasets)**:

    - **Synthetic corruption**: CIFAR-10-C, CIFAR-100-C, ImageNet-C, TinyImageNet-C
    - **Style shift**: ImageNet-S, ImageNet-R, PACS
    - **Background shift**: NICO++, Waterbirds
    - **Data collection shift**: CIFAR-10.1/10.2, CINIC-10, STL-10, ImageNet-V2, VLCS
    - **Camera location shift**: iWildCam, TerraInc, ObjectNet
    - **Demographic shift**: CelebA, CivilComments, CheXpert
    - **Other**: FMoW (temporal + geographic), RxRx1 (batch effects), Amazon, DomainNet, OfficeHome

2. **Model Training Strategy**:

    - ImageNet variants: 109 open-source models directly from Torchvision
    - CIFAR variants: trained from scratch with 3 random seeds per architecture (57 for CIFAR-10, 108 for CIFAR-100)
    - WILDS: initialized from ImageNet pretrained weights, 1 model per architecture across 30 architectures
    - Domain generalization / subpopulation shift: supervised, MoCo, and CLIP pretrained weights; ResNet-50 and ViT-B/16 backbones; leave-one-domain/group-out protocol; 5 random seeds per configuration

3. **Evaluation Metric — Spearman Rank Correlation**:
    $$\rho = 1 - \frac{6\sum_{i=1}^{n}(R(\hat{S}_i) - R(Acc_i))^2}{n(n^2-1)}$$
   where $\hat{S}_i$ is the predicted score and $Acc_i$ is the true accuracy. This metric is preferred over $R^2$ because the latter is sensitive to outliers and cannot handle nonlinear correlations. Metrics are computed across architectures (rather than within a single architecture as in prior work), making the evaluation more challenging and more realistic.

4. **10 Prediction Algorithms**: Covering confidence-based methods (ATC, DoC), distribution discrepancy-based methods (COT, COTT), feature-based methods (Nuclear Norm, MaNo, Dispersion, MDE), data augmentation-based methods (NI), and model agreement-based methods (Agreement).

## Key Experimental Results

### Main Results: 29 OOD Datasets × 10 Algorithms

| Dataset Type | Representative Dataset | Mean $\rho$ | Effective Algorithms ($\rho>0.7$) |
|:---|:---|:---|:---|
| Synthetic corruption | CIFAR-10-C | 0.746 | 9/10 |
| Synthetic corruption | CIFAR-100-C | 0.712 | 9/10 |
| Natural shift — style | ImageNet-S | 0.559 | 4/10 |
| Natural shift — background | NICO++ | 0.705 | 9/10 |
| Natural shift — fine-grained category | DomainNet | 0.451 | 2/10 |
| Natural shift — camera location | iWildCam | 0.328 | 0/10 |
| Natural shift — adversarial | ImageNet-A | 0.270 | 1/10 |

**Key Finding 1**: On synthetic corruption datasets, most algorithms achieve rank correlation >0.7 (9/10 effective), whereas the number of effective algorithms drops sharply on natural distribution shifts (0/10 on iWildCam).

**Key Finding 2**: Agreement and ATC are the most consistently stable algorithms overall. Agreement achieves rank correlations >0.9 on multiple datasets (CIFAR-10-C: 0.991, ImageNet-V2: 0.996), and ATC ranks in the top three on 15 out of 29 datasets.

### Ablation Study

| Analysis Dimension | Finding |
|:---|:---|
| Pretrained weights | CLIP pretraining leads to more concentrated model performance, making prediction harder; MoCo and supervised pretraining show little difference |
| Model architecture | Within-architecture $\rho$ is generally high (>0.95), but performance degrades significantly across architectures |
| $R^2$ vs. $\rho$ | $R^2$ is more sensitive to outliers; on some datasets, $R^2<0$ while $\rho>0.5$ |
| Shift type | Camera location and adversarial shifts are the hardest to predict; corruption is the easiest |

### Key Findings
1. Existing algorithms are effective on synthetic corruptions but broadly fail on natural shifts — this is the central challenge in OOD performance prediction.
2. No single algorithm performs well across all shift types.
3. Cross-architecture performance prediction is substantially more challenging than within-architecture prediction.
4. CLIP pretrained weights make performance prediction more difficult.

## Highlights & Insights

1. **Large-scale unified benchmark**: For the first time, datasets from OOD sub-fields such as domain generalization and subpopulation shift are incorporated into a performance prediction benchmark, with 29 datasets covering 7 types of distribution shifts.
2. **Reusable testbench**: The 1,444 pretrained models allow future researchers to fairly compare methods without repeated training.
3. **Revealing critical blind spots**: The benchmark clearly demonstrates the capability boundary of existing algorithms — easy on synthetic shifts, hard on natural shifts — and points the way for future research.
4. **Importance of cross-architecture evaluation**: Prior within-architecture evaluations likely overestimate the practical utility of existing algorithms.

## Limitations & Future Work

1. The benchmark focuses exclusively on classification tasks; OOD performance prediction for detection, segmentation, and other tasks is not addressed.
2. Although all algorithms perform poorly on natural distribution shifts, no improvements are proposed.
3. The relationships between different shift types, or the effects of compound shifts, are not analyzed.
4. The 1,444 models are predominantly CNN-based (ResNet, VGG, etc.); coverage of large models such as ViT-L and Swin Transformer is limited.

## Related Work & Insights

- **OOD generalization**: Invariant learning (IRM, IRMv1), DRO, domain generalization (SWAD, CORAL), stable learning, etc.
- **OOD performance prediction**: Model confidence-based (ATC, DoC), distribution discrepancy-based (COT), model agreement-based (Agreement-on-the-line), feature separation-based (Dispersion Score), etc.
- **Related benchmarks**: WILDS, DomainBed, and MetaShift provide partial OOD dataset coverage but lack a comprehensive benchmark specifically targeting performance prediction.

## Rating
- Novelty: ★★★☆☆ (benchmark paper; contribution lies primarily in systematic integration)
- Experimental Thoroughness: ★★★★★ (29 datasets, 10 algorithms, 1,444 models)
- Value: ★★★★★ (open-source code and model testbench, immediately usable)
- Writing Quality: ★★★★☆ (clear structure, in-depth analysis)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](../../NeurIPS2025/llm_evaluation/leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](../../ACL2026/llm_evaluation/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)
- [\[NeurIPS 2025\] Predicting the Performance of Black-Box LLMs through Follow-Up Queries](../../NeurIPS2025/llm_evaluation/predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](../../NeurIPS2025/llm_evaluation/ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)

</div>

<!-- RELATED:END -->
