---
title: >-
  [Paper Note] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning
description: >-
  [ICCV 2025][Object Detection][Moment Retrieval] This paper proposes the AMR framework, which leverages a Splice-and-Boost data augmentation strategy and a cold-start–distillation two-stage training pipeline to substantially improve boundary awareness and semantic discriminability in video moment retrieval—without relying on any external data or pretrained models—surpassing the previous SOTA by +5% on QVHighlights.
tags:
  - ICCV 2025
  - Object Detection
  - Moment Retrieval
  - Data Augmentation
  - Knowledge Distillation
  - DETR
  - Video Temporal Grounding
date: 2026-05-08
content_hash: 26f7732787fd7f69
---

# Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning

**Conference**: ICCV 2025
**arXiv**: [2510.19622](https://arxiv.org/abs/2510.19622)
**Code**: [https://github.com/SooLab/AMR](https://github.com/SooLab/AMR)
**Area**: Object Detection
**Keywords**: Moment Retrieval, Data Augmentation, Knowledge Distillation, DETR, Video Temporal Grounding

## TL;DR

This paper proposes the AMR framework, which leverages a Splice-and-Boost data augmentation strategy and a cold-start–distillation two-stage training pipeline to substantially improve boundary awareness and semantic discriminability in video moment retrieval—without relying on any external data or pretrained models—surpassing the previous SOTA by +5% on QVHighlights.

## Background & Motivation

**State of the Field**: Moment Retrieval aims to precisely localize the start and end timestamps of a target segment in a long video given a natural language query. Mainstream approaches adopt the DETR framework for end-to-end grounding, yet remain constrained by annotation scarcity and label ambiguity.

**Limitations of Prior Work**:
   - **Data Scarcity**: Dense annotation of individual videos is costly, resulting in insufficient training samples. Models tend to learn shallow associations between textual keywords and local visual features rather than understanding the temporal completeness of actions.
   - **Boundary Ambiguity**: Transition regions between adjacent events (e.g., "preparing to shoot" and "completing the shot") yield unclear boundary signals.
   - **Fine-Grained Semantic Confusion**: Existing methods struggle to distinguish semantically similar yet logically distinct segments (e.g., "kicking a ball" vs. "throwing a ball" both involve ball motion).

**Limitations of Existing Attempts**: Prior work has explored three directions—multimodal interaction modules (limited attentive discriminability), large-scale pretraining transfer (prohibitive parameter count and deployment cost), and external large-model data synthesis (introducing external dependencies). None addresses the root cause: how to elicit discriminative spatiotemporal reasoning from limited annotations under zero external dependencies.

**Starting Point**: The authors observe that structurally decomposing and recombining existing data can yield augmented samples with explicit boundaries and semantic distractors. A two-stage training strategy is then designed to transfer knowledge from augmented data back to the real data distribution.

**Core Idea**: Splice-and-Boost augmentation + cold-start distillation two-stage training = a breakthrough in moment retrieval performance with zero external dependencies.

## Method

### Overall Architecture

AMR follows the standard DETR pipeline: video features are extracted via CLIP + SlowFast ($V \in \mathbb{R}^{L \times C}$), text features via the CLIP text encoder ($T \in \mathbb{R}^{K \times C}$), cross-modal interaction is performed through a Transformer encoder, and the decoder employs $N$ learnable queries to produce temporal span predictions. The innovations are concentrated along two dimensions: data augmentation and training strategy.

### Key Designs

1. **Splice (Temporal Splicing) Augmentation**:

    - **Function**: Constructs augmented training samples with explicit temporal boundaries.
    - **Mechanism**: A foreground target segment is extracted from an original sample and randomly downsampled. A window of equal length is then randomly selected and removed from the background timeline of a different video, and the target segment is seamlessly inserted into the resulting gap.
    - **Design Motivation**: By decoupling segment semantics from background context, the model is compelled to localize events independently of surrounding context, thereby strengthening boundary awareness.

2. **Boost (Semantic Augmentation) Mechanism**:

    - **Function**: Introduces semantically ambiguous hard negatives to improve the model's ability to discriminate between similar events.
    - **Mechanism**: A cross-validation paradigm is employed—the training set is split into two mutually exclusive subsets $\mathcal{D}_1$ and $\mathcal{D}_2$. A model trained on one subset identifies high-confidence false positives on the other ($\text{IoU}(W_i, W_j^{gt}) = 0$ and confidence $S_i > \theta$). The roles of the subsets are then swapped iteratively to collect ambiguous segments, which are spliced alongside ground-truth target segments into background videos.
    - **Design Motivation**: Jointly optimizes localization of true segments and suppression of ambiguous ones, balancing diversity and discriminability without introducing external data.

3. **Cold-Start Phase**:

    - **Function**: Establishes foundational boundary localization and semantic discriminability on augmented data.
    - **Mechanism**: A curriculum learning strategy trains exclusively on augmented data for 40 epochs. The explicit foreground/background distinction and challenging segments in the augmented data build basic discriminative capacity.
    - **Design Motivation**: Provides a strong knowledge baseline for the distillation phase; however, the sharp boundaries in augmented data may cause the model to over-rely on abrupt visual transitions.

4. **Distillation Adaptation Phase (Dual-Path Distillation)**:

    - **Function**: Transfers cold-start knowledge to the real data distribution while preventing forgetting.
    - **Mechanism**: Dual-path queries are introduced—original queries $Q^{\text{ori}}$ preserve DETR localization capacity, while active queries $Q^{\text{act}}$ dynamically adapt to real data, each with independent FFNs. A distillation loss constrains the distribution of original queries to remain consistent with a frozen base query: $\mathcal{L}_{\text{dill}} = 1 - \frac{1}{N}\sum_{i}\frac{(\hat{Q}^{\text{ori}}_i)^T \hat{Q}^{\text{base}}_i}{\|\hat{Q}^{\text{ori}}_i\| \|\hat{Q}^{\text{base}}_i\|}$
    - **Design Motivation**: Bridges the synthetic-to-real domain gap through parameter isolation and similarity-driven regularization.

5. **Discriminative Contrastive Loss (DCL)**:

    - **Function**: Preserves the model's discriminability against ambiguous segments.
    - **Mechanism**: Frame-level text–video cosine similarity $M_i$ is used to construct relative ranking constraints rather than absolute score suppression: $\mathcal{L}_{\text{disc}} = -\log\frac{\exp(p/\tau)}{\exp(p/\tau) + \exp(n/\tau)}$, where $p$ and $n$ denote the mean matching scores for the ground-truth and ambiguous regions, respectively.
    - **Design Motivation**: Encourages a larger discriminative margin between true events and their semantic neighbors.

### Loss & Training

- **Cold-Start Phase**: $\mathcal{L} = \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{loc}}\mathcal{L}_{\text{loc}} + \lambda_{\text{sal}}\mathcal{L}_{\text{sal}}$
- **Distillation Phase**: $\mathcal{L} = \mathcal{L}_{\text{stage1}} + 0.5 \cdot \mathcal{L}_{\text{dill}} + 0.5 \cdot \mathcal{L}_{\text{disc}}$
- Hungarian matching establishes prediction–ground-truth correspondences; Stage 1: 40 epochs, Stage 2: 100 epochs; AdamW, lr=1e-4.

## Key Experimental Results

### Main Results

**QVHighlights Dataset (Validation / Test Sets)**:

| Method | R1@0.5 (Val) | R1@0.7 (Val) | mAP Avg. (Val) | R1@0.5 (Test) | mAP Avg. (Test) |
|--------|-------------|-------------|---------------|--------------|----------------|
| M-DETR | 53.94 | 34.84 | 32.20 | 52.89 | 30.73 |
| QD-DETR | 62.68 | 46.66 | 41.22 | 62.40 | 39.86 |
| TR-DETR | 67.10 | 51.48 | 45.09 | 64.66 | 42.62 |
| BAM-DETR | 65.10 | 51.61 | 47.61 | 62.71 | 45.36 |
| **AMR (Ours)** | **70.13** | **56.65** | **51.66** | **68.22** | **48.43** |

On the validation set, AMR surpasses BAM-DETR by +5.03% on R1@0.5, +5.04% on R1@0.7, and +4.05% on average mAP.

**Charades-STA & TACoS**:

| Method | Charades R1@0.5 | Charades mIoU | TACoS R1@0.5 | TACoS mIoU |
|--------|----------------|--------------|-------------|-----------|
| UniVTG | 58.01 | 50.10 | 34.97 | 33.60 |
| UVCOM | 59.25 | - | 36.39 | - |
| **AMR** | **62.02** | **52.58** | **40.91** | **38.22** |

### Ablation Study

| Setting | Splice | Boost | Two-Stage | Dill | DCL | mAP Avg. |
|---------|--------|-------|-----------|------|-----|---------|
| (a) Baseline | - | - | - | - | - | 44.82 |
| (c) Aug. Only | ✓ | ✓ | - | - | - | 43.88 |
| (f) Aug. + Two-Stage | ✓ | ✓ | ✓ | - | - | 48.07 |
| (l) Full Model | ✓ | ✓ | - | ✓ | ✓ | **51.66** |

- Applying data augmentation alone (without two-stage training) actually falls below the baseline (43.88 vs. 44.82), validating the distribution gap between augmented and real data.
- Two-stage training combined with augmentation substantially improves performance to 48.07; distillation and contrastive losses further push it to 51.66.

## Personal Reflections

- **Highlights**: The zero-external-dependency design philosophy is practically appealing; the Boost mechanism's use of cross-validation to mine hard negatives is elegant; the dual-path query distillation paradigm is conceptually clean.
- **Limitations**: The two-stage training pipeline introduces additional complexity (140 epochs total); Splice augmentation may introduce unnatural visual discontinuities.
- **Insights**: The synthetic-to-real knowledge transfer paradigm based on dual-path queries and distillation is generalizable to other video understanding tasks that benefit from data augmentation.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)
- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[ICCV 2025\] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)

<!-- RELATED:END -->
