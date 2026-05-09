---
title: >-
  [Paper Note] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions
description: >-
  [NeurIPS 2025][Video Understanding][Multi-Moment Retrieval] This paper proposes the QV-M2 dataset (the first fully human-annotated multi-moment retrieval benchmark) and the FlashMMR framework (incorporating a Post-Verification Module), extending video moment retrieval from single-moment to multi-moment scenarios and establishing a standardized evaluation protocol for multi-moment retrieval.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Multi-Moment Retrieval
  - Video Temporal Grounding
  - Dataset
  - Post-Verification Module
  - Moment Retrieval
date: 2026-05-08
content_hash: a6ca3b5076890df6
---

# When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions

**Conference**: NeurIPS 2025
**arXiv**: [2510.17218](https://arxiv.org/abs/2510.17218)
**Code**: [GitHub](https://github.com/Zhuo-Cao/QV-M2)
**Area**: Video Understanding
**Keywords**: Multi-Moment Retrieval, Video Temporal Grounding, Dataset, Post-Verification Module, Moment Retrieval

## TL;DR

This paper proposes the QV-M2 dataset (the first fully human-annotated multi-moment retrieval benchmark) and the FlashMMR framework (incorporating a Post-Verification Module), extending video moment retrieval from single-moment to multi-moment scenarios and establishing a standardized evaluation protocol for multi-moment retrieval.

## Background & Motivation

### From Single-Moment to Multi-Moment: The Need

Video Moment Retrieval (MR) aims to localize relevant temporal segments in a video given a natural language query. Existing methods almost universally rely on the **Single-Moment Retrieval (SMR)** assumption: each query corresponds to exactly one video segment. In practice, however, a single query often corresponds to multiple disjoint moments. For example, "chopping vegetables" may occur multiple times in a cooking tutorial, and "a successful three-pointer" recurs throughout a sports broadcast.

### Three Core Gaps

**No dataset**: Existing MR datasets (Charades-STA, QVHighlights, etc.) are predominantly single-moment annotated, with an average of only 1–1.8 moments per query.

**No evaluation metrics**: Standard mAP and IoU metrics cannot measure coverage in multi-target retrieval.

**No methods**: Existing model architectures are inherently constrained to single-moment prediction and perform poorly in multi-moment scenarios.

### Failure of SMR Methods in Multi-Moment Settings

The core issue with SMR methods on multi-moment queries is that models tend to optimize for the single highest-confidence moment prediction, discarding other equally valid segments. Even when multiple candidate predictions are generated, there is no mechanism to ensure semantic consistency and redundancy removal across moments.

## Method

### Overall Architecture

FlashMMR builds upon the base architecture of FlashVTG, consisting of three stages: feature extraction and fusion, multi-scale temporal processing, and prediction. A **Post-Verification Module (PVM)** is newly introduced to handle boundary refinement and semantic consistency control specific to multi-moment scenarios.

### Key Designs

#### 1. QV-M2 Dataset Construction

**Construction approach**: Based on the original videos from QVHighlights, with additional human annotations targeting multi-moment scenarios.

- Annotation guidelines: (i) create detailed queries that precisely capture actors, actions, and context; (ii) include context-dependent queries requiring understanding of temporal relationships; (iii) design negative queries marking segments where specific actions do not occur.
- Quality control: After every 100 annotated videos, 5% are randomly sampled for review by a second annotator; if temporal boundary overlap falls below 90%, a third annotator re-labels the sample.
- Statistics: 2,212 queries, 6,384 annotated segments, covering 1,341 videos, with an **average of 2.9 moments per query** (substantially higher than QVHighlights' 1.8).

#### 2. Multi-Moment Evaluation Metrics

**Generalized mAP (G-mAP)**: Average AP across multiple IoU thresholds $\mathcal{T} = \{0.5, 0.55, \dots, 0.9\}$:

$$\text{G-mAP} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{AP}(\tau)$$

Results are also reported grouped by the number of ground-truth moments per query (mAP@1\_tgt, mAP@2\_tgt, mAP@3+\_tgt).

**Mean IoU@k**: Average IoU between the top-$k$ predictions and their best-matching ground truths:

$$\text{mIoU}@k = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{k} \sum_{i=1}^{k} \max_{\text{gt} \in \mathcal{G}(q)} \text{IoU}(\text{pred}_i, \text{gt})$$

**Mean Recall@k**: Recall of ground truths covered by the top-$k$ predictions, computed only for queries with at least $k$ ground-truth moments.

All three metric groups reduce to standard SMR metrics when $k=1$, ensuring backward compatibility.

#### 3. Post-Verification Module

**Design Motivation**: Initial predictions in multi-moment scenarios are prone to redundant or irrelevant moment proposals, necessitating a mechanism for boundary refinement and low-confidence proposal filtering.

**Post-processing and Feature Refinement**: Structured constraints are applied to the initial predictions $\hat{B} \in \mathbb{R}^{3 \times n}$:

$$\tilde{B} = \mathcal{F}(\hat{B}, \lambda_{\text{clip}}, \lambda_{\text{round}})$$

This includes minimum/maximum window length constraints, temporal clipping, and discretization. Multi-modal feature representations $\mathbf{I}_i$ are then extracted from the fused features $F$ for each predicted interval.

**Semantic Consistency Verification**: A GRU recurrent network $\mathcal{P}_{\text{GRU}}$ models contextual dependencies among retrieved intervals:

$$p_i = \sigma(\mathcal{P}_{\text{GRU}}(\mathbf{I}_i))$$

The output refined confidence $p_i$ is supervised via tIoU:

$$\hat{\mathbf{IoU}} = \max(\text{tIoU}(\tilde{B}, B^*), \text{dim}=-1)$$

### Loss & Training

The total loss consists of FlashVTG's base loss plus the post-verification loss:

$$\mathcal{L}_{\text{PV}} = \|\mathbf{p} - \hat{\mathbf{IoU}}\|_2^2 + \mathcal{L}_{\text{repr}}$$

The representation learning loss $\mathcal{L}_{\text{repr}} = \sum_i \text{CE}(\mathbf{S}_i, \mathbf{T}_i)$ enforces high semantic consistency among temporally adjacent frames via cross-entropy between the cosine similarity matrix $\mathbf{S}_i$ and the ground-truth segment consistency matrix $\mathbf{T}_i$. Loss weights: 9 for $\mathcal{L}_{\text{PV}}$ and 7 for $\mathcal{L}_{\text{repr}}$. NMS threshold: 0.7.

## Key Experimental Results

### Main Results on QV-M2 Test Set

| Method | G-mAP | mAP@3+tgt | mIoU@2 | mIoU@3 | mR@2 | mR@3 |
|--------|-------|-----------|--------|--------|------|------|
| M-DETR (NeurIPS'21) | 20.65 | 10.95 | 38.98 | 34.34 | 30.95 | 26.24 |
| QD-DETR (CVPR'23) | 28.95 | 18.30 | 46.79 | 40.50 | 40.58 | 36.05 |
| FlashVTG (WACV'25) | 32.14 | 20.19 | 47.85 | 40.92 | 41.30 | 35.94 |
| **FlashMMR** | **35.14** | **22.89** | **49.64** | **42.92** | **44.33** | **38.50** |
| Gain (vs. FlashVTG) | +3.00 | +2.70 | +1.79 | +2.00 | +3.03 | +2.56 |

### Ablation Study

| Configuration | G-mAP | mAP@2_tgt | mAP@3+tgt | mIoU@2 | mR@2 |
|---------------|-------|-----------|-----------|--------|------|
| FlashMMR w/o PV (QV-M2) | 32.14 | 39.48 | 20.19 | 47.85 | 41.30 |
| FlashMMR w/ PV (QV-M2) | **35.14** | **42.52** | **22.89** | **49.64** | **44.33** |
| FlashMMR w/o PV (QVHighlights) | 48.02 | 35.08 | 13.85 | 43.80 | 38.98 |
| FlashMMR w/ PV (QVHighlights) | **48.07** | **35.78** | **15.15** | **45.32** | **40.63** |

### Cross-Dataset Results (QV-M2 Training Improves Performance)

| Method | Training Set | G-mAP | mR@3 |
|--------|-------------|-------|------|
| M-DETR | QVHighlights | 32.79 | 19.55 |
| M-DETR | QV-M2 Train | **34.70** | **23.67** |
| FlashMMR | QVHighlights | 48.07 | 36.68 |
| FlashMMR | QV-M2 Train | **48.42** | **39.29** |

### Key Findings

1. The Post-Verification Module yields consistent gains on both datasets, with a 3% G-mAP improvement on QV-M2, confirming its necessity for multi-moment scenarios.
2. Training on QV-M2 improves all methods on both SMR and MMR tasks, validating the dataset quality.
3. Multi-moment queries (3+ targets) pose the greatest challenge to all methods; FlashMMR improves mAP@3+tgt from 20.19 to 22.89.
4. Existing SMR methods exhibit significant performance drops under MMR evaluation, confirming the necessity of dedicated MMR frameworks.

## Highlights & Insights

1. **Valuable problem formulation**: The paradigm shift from SMR to MMR reflects real-world requirements and represents an important advance in natural language video understanding.
2. **Comprehensive evaluation protocol**: The G-mAP, mIoU@k, and mR@k metrics are backward compatible with SMR metrics, offering an elegant design.
3. **High-quality dataset**: Fully human-annotated with rigorous quality control, averaging 2.9 moments per query, making it more challenging than existing datasets.
4. **Concise and effective method design**: The Post-Verification Module requires only a single GRU network, with minimal parameter overhead yet significant performance gains.

## Limitations & Future Work

- The Post-Verification Module is relatively straightforward; reinforcement learning or contrastive learning could be explored for more fine-grained moment discrimination.
- QV-M2 is limited in scale (2,212 queries); larger-scale annotation may be needed as models advance.
- Feature extraction based on SlowFast+CLIP is fixed; stronger visual backbones remain unexplored.
- The method assumes non-overlapping moments; handling partially overlapping scenarios is not explicitly addressed.
- The NMS threshold of 0.7 is fixed; adaptive thresholding strategies may further improve performance.

## Related Work & Insights

- **Single-Moment Retrieval**: Moment-DETR, QD-DETR, FlashVTG
- **Multi-Moment Retrieval**: SFABD, NExT-VMR (concurrent work, dataset not publicly released)
- **Datasets**: QVHighlights, DiDeMo, CharadesSTA
- **Insights**: The evaluation metric design for MMR is generalizable to other one-to-many matching problems.

## Rating
- Novelty: ⭐⭐⭐⭐☆ — Multi-moment retrieval is a natural extension, yet prior systematic work is lacking.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Cross-dataset comparisons with 6 baselines, ablations, and a new metric suite.
- Writing Quality: ⭐⭐⭐⭐☆ — Clear structure with well-documented dataset construction procedures.
- Value: ⭐⭐⭐⭐☆ — The dataset and metric contributions may have greater long-term impact than the method itself.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](../../ICCV2025/video_understanding/moment_quantization_for_video_temporal_grounding.md)
- [\[NeurIPS 2025\] Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding](empower_words_dualground_for_structured_phrase_and_sentencel.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](token_bottleneck_one_token_to_remember_dynamics.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)

<!-- RELATED:END -->
