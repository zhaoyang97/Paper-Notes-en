---
title: >-
  [Paper Note] ConSurv: Multimodal Continual Learning for Survival Analysis
description: >-
  [AAAI 2026][Medical Imaging][Continual Learning] This paper proposes ConSurv, the first multimodal continual learning framework for survival analysis. Through two core components — Multi-Stage Mixture-of-Experts (MS-MoE) and Feature-Constrained Replay (FCR) — ConSurv effectively mitigates catastrophic forgetting in settings that integrate whole slide pathology images and genomic data, comprehensively outperforming existing methods on the newly constructed MSAIL benchmark.
tags:
  - AAAI 2026
  - Medical Imaging
  - Continual Learning
  - Multimodal Fusion
  - Survival Analysis
  - Whole Slide Images
  - Catastrophic Forgetting
date: 2026-05-08
content_hash: b73fcb13dd4f3cd9
---

# ConSurv: Multimodal Continual Learning for Survival Analysis

**Conference**: AAAI 2026
**arXiv**: [2511.09853](https://arxiv.org/abs/2511.09853)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: Continual Learning, Multimodal Fusion, Survival Analysis, Whole Slide Images, Catastrophic Forgetting

## TL;DR

This paper proposes ConSurv, the first multimodal continual learning framework for survival analysis. Through two core components — Multi-Stage Mixture-of-Experts (MS-MoE) and Feature-Constrained Replay (FCR) — ConSurv effectively mitigates catastrophic forgetting in settings that integrate whole slide pathology images and genomic data, comprehensively outperforming existing methods on the newly constructed MSAIL benchmark.

## Background & Motivation

**State of the Field**: Cancer survival prediction is critical to clinical practice, informing mortality risk and guiding treatment decisions. Recent multimodal approaches combining whole slide images (WSI) and genomic data have demonstrated strong survival prediction capability, as different modalities provide complementary patient information.

**Limitations of Prior Work**: (1) In real clinical environments, data accumulates continuously — new patient data and new hospital data sources arrive persistently — yet existing survival analysis models are typically trained on a single static dataset and cannot adapt to dynamically changing clinical settings. (2) Directly fine-tuning on new data leads to catastrophic forgetting, where the model loses knowledge acquired from previous data. (3) Existing continual learning methods are primarily designed for unimodal classification tasks and cannot effectively handle the unique challenges of multimodal survival analysis — namely, the extreme scale of WSIs (gigapixel-level) and the complex cross-modal interactions with high-dimensional genomic data.

**Root Cause**: Continual learning must simultaneously satisfy two conflicting objectives — acquiring knowledge from new tasks (plasticity) and retaining knowledge from old tasks (stability). In multimodal survival analysis, this tension is exacerbated: the encoders for each modality and the fusion module must both adapt to new data while preserving prior knowledge, and the inter-modal interactions also shift across tasks.

**Paper Goals**: (1) Enable effective continual learning for multimodal (WSI + genomic) survival analysis; (2) Design an architecture capable of capturing both task-shared and task-specific knowledge; (3) Mitigate catastrophic forgetting at multiple levels; (4) Establish a standardized evaluation benchmark.

**Starting Point**: The authors observe that knowledge in multimodal survival analysis can be decomposed at multiple levels — unimodal knowledge at the encoder level and cross-modal knowledge at the fusion level — and that both task-shared and task-specific components coexist. Based on this observation, knowledge sharing and specialization can be handled separately at different stages of the network.

**Core Idea**: Separate task-shared and task-specific knowledge by deploying Mixture-of-Experts mechanisms in both the encoders and the fusion module (MS-MoE), and mitigate forgetting by constraining feature drift at multiple levels — encoder-level and fusion-level (FCR).

## Method

### Overall Architecture

ConSurv takes as input a pair of multimodal data: a gigapixel-level whole slide pathology image (WSI) and a genomic feature vector. The two modalities are processed by their respective encoders to extract features, which are then integrated by a fusion module into a unified representation for survival risk prediction. The entire network is trained sequentially under a continual learning setting, with each task corresponding to a dataset. The core objective is to achieve strong performance on each new task without forgetting prior tasks.

### Key Designs

1. **Multi-Stage Mixture-of-Experts (MS-MoE)**:

    - **Function**: Captures task-shared and task-specific knowledge at different learning stages of the network, covering both unimodal and cross-modal levels.
    - **Mechanism**: MoE structures are introduced into both modality encoders and the modality fusion module. Each MoE layer contains shared experts (used across all tasks) and task-specific experts (exclusive to each task). A gating network dynamically routes inputs to an appropriate combination of experts. At the encoder stage, MoE learns shared and task-specific intra-modal patterns; at the fusion stage, MoE learns shared and task-specific cross-modal interaction relationships. The term "multi-stage" reflects the deployment of MoE at multiple positions in the network (encoders + fusion module), rather than at a single location.
    - **Design Motivation**: WSIs across different datasets (tasks) originate from different scanners and staining protocols, and genomic data distributions also vary. Shared experts capture universal patterns across tasks (e.g., fundamental histological features), while task-specific experts adapt to the unique distribution of each task. This separation prevents shared knowledge from being overwritten by new tasks.

2. **Feature-Constrained Replay (FCR)**:

    - **Function**: Further mitigates catastrophic forgetting through multi-level constraints on feature drift.
    - **Mechanism**: A small experience replay buffer stores representative samples from previous tasks. When training on a new task, old samples are replayed simultaneously, and feature drift is constrained at three levels: (a) output features of the WSI encoder; (b) output features of the genomic encoder; (c) the unified representation after fusion. Concretely, the L2 distance between the current model's and the old model's features on replayed samples is computed and incorporated as a regularization loss into the training objective.
    - **Design Motivation**: Standard experience replay constrains consistency only at the output layer (predictions), neglecting drift in intermediate features. In multimodal settings, both unimodal feature drift at the encoder level and cross-modal representation drift at the fusion level contribute to forgetting. Multi-level constraints comprehensively cover all channels through which knowledge is lost.

3. **MSAIL Benchmark (Multimodal Survival Analysis Incremental Learning)**:

    - **Function**: Provides a standardized evaluation platform for multimodal continual learning in survival analysis.
    - **Mechanism**: Four publicly available cancer survival analysis datasets are integrated and organized under a task-incremental learning setting — the model learns the four datasets sequentially, and is evaluated on comprehensive performance across all previously seen datasets. Evaluation metrics include the C-index (concordance index) and other standard survival analysis measures.
    - **Design Motivation**: No standardized benchmark previously existed for multimodal continual learning survival analysis, leaving the field without a unified comparison platform. The construction of MSAIL fills this gap.

### Loss & Training

The total loss consists of three components: (1) a survival analysis loss (e.g., negative log partial likelihood of the Cox model) for survival prediction; (2) an FCR regularization loss (multi-level feature drift penalty) to mitigate forgetting; and (3) a load-balancing loss for MoE to ensure uniform expert utilization. The training strategy follows task-incremental learning — four datasets are learned sequentially, with a small subset of samples from previous tasks replayed during each new task's training.

## Key Experimental Results

### Main Results

Comparison with multiple continual learning methods on the MSAIL benchmark.

| Method | Avg. C-index | Forgetting Rate | Notes |
|--------|-------------|-----------------|-------|
| ConSurv | Best | Lowest | Full method |
| EWC | Moderate | High | Classic regularization; not multimodal-aware |
| ER (Experience Replay) | Moderate | Moderate | Standard replay; lacks multi-level constraints |
| Fine-tuning | Best on latest task | Severe | No continual learning strategy |
| Joint Training | Upper bound | None | Idealized joint training on all data |

### Ablation Study

| Configuration | C-index | Notes |
|---------------|---------|-------|
| ConSurv (Full) | Best | MS-MoE + FCR complete model |
| w/o MS-MoE | Significant drop | Without MoE, shared/specific knowledge cannot be separated |
| w/o FCR | Drop | Forgetting worsens without feature-constrained replay |
| w/o encoder-level constraints | Minor drop | Fusion-level constraints alone are insufficient |
| w/o fusion-level constraints | Minor drop | Encoder-level constraints alone neglect cross-modal drift |

### Key Findings

- MS-MoE is the most critical component — without it, the model cannot effectively separate task-shared and task-specific knowledge, and catastrophic forgetting increases substantially.
- FCR's multi-level constraints further reduce forgetting compared to prediction-level-only constraints, validating the importance of preserving intermediate features.
- Both encoder-level and fusion-level constraints contribute complementarily; their combination yields the best overall performance.
- The gap between ConSurv and Joint Training (ideal upper bound) is small, demonstrating that the method effectively approximates forgetting-free ideal performance.

## Highlights & Insights

- **Pioneering the introduction of continual learning into multimodal survival analysis** is a foundational contribution; the problem formulation itself carries significant value — clinical data does accumulate continuously, and the limitations of static models are real.
- **The multi-stage MoE design** is particularly elegant: knowledge separation is applied not only at the encoder level but also at the fusion level. This pervasive separation strategy comprehensively covers all channels of knowledge loss throughout the multimodal network.
- **The construction of the MSAIL benchmark** provides a standardized platform for future research and carries long-term value for the community.

## Limitations & Future Work

- Experiments are conducted on four datasets, representing a relatively small number of tasks; performance on longer task sequences remains unverified.
- Task-specific experts in MoE grow linearly with the number of tasks, which may cause parameter bloat in scenarios with a very large number of tasks.
- Data privacy constraints are not considered — in real clinical settings, replay of old data may be prohibited due to privacy regulations.
- Integration with federated learning could be explored to enable privacy-preserving continual learning in multi-hospital collaborative scenarios.

## Related Work & Insights

- **vs. EWC/SI and other regularization methods**: These methods mitigate forgetting by constraining parameter changes but lack modality-awareness. ConSurv's FCR constrains at the feature level, making it more suitable for multimodal settings.
- **vs. Standard MoE**: Standard MoE does not distinguish between shared and task-specific experts. ConSurv's MS-MoE explicitly introduces two types of experts, which is better suited for continual learning.
- **vs. PackNet/ProgressiveNet and other architectural methods**: These methods accommodate new knowledge through network expansion but do not consider multimodal interactions. ConSurv's introduction of MoE at the fusion level is a key distinguishing feature.

## Rating

- Novelty: ⭐⭐⭐⭐ Pioneering introduction of continual learning into multimodal survival analysis, with innovation in both problem formulation and method design
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies, multi-method comparisons, and construction of a new benchmark
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated and method description is detailed
- Value: ⭐⭐⭐⭐ Provides practical guidance for continuous updating of clinical AI systems

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](learning_with_preserving_for_continual_multitask_learning.md)
- [\[ICLR 2026\] SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis](../../ICLR2026/medical_imaging/survhte-bench_a_benchmark_for_heterogeneous_treatment_effect_estimation_in_survi.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[AAAI 2026\] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction](grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](../../CVPR2026/medical_imaging/residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)

<!-- RELATED:END -->
