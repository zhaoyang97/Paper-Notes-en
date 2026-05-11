---
title: >-
  [Paper Note] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation
description: >-
  [ICCV 2025][Segmentation][Open-vocabulary segmentation] This paper exposes an evaluation bias in existing open-vocabulary segmentation (OVS) benchmarks…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Open-vocabulary segmentation"
  - "benchmark evaluation"
  - "semantic space"
  - "CLIP fine-tuning"
  - "gradient-free aggregation"
  - "proxy calibration"
date: 2026-05-08
content_hash: c94f9fd0ac53e41d
---

# Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation

**Conference**: ICCV 2025
**arXiv**: [2506.16058](https://arxiv.org/abs/2506.16058)
**Code**: Not released
**Area**: Image Segmentation
**Keywords**: Open-vocabulary segmentation, benchmark evaluation, semantic space, CLIP fine-tuning, gradient-free aggregation, proxy calibration

## TL;DR

This paper exposes an evaluation bias in existing open-vocabulary segmentation (OVS) benchmarks, where test sets exhibit high semantic similarity to training spaces. It proposes a new benchmark, OpenBench, and a method, OVSNet, that integrates heterogeneous features via Gradient-Free Aggregation (GFA) and expands the training semantic space at zero cost through Proxy Calibration (PC), achieving state-of-the-art performance on both existing benchmarks and OpenBench.

## Background & Motivation

Open-vocabulary segmentation (OVS) aims to segment objects in images based on arbitrary text inputs. Existing methods primarily leverage large-scale pretrained vision-language models such as CLIP, following several technical paradigms: two-stage segment-then-classify, frozen CLIP feature integration with segmentation decoders, and fine-tuning of CLIP encoders.

This paper presents a critical observation: **fine-tuning CLIP consistently improves performance on existing benchmarks, yet this contradicts the fundamental objective of open-vocabulary tasks.** Fine-tuning CLIP causes the model to overfit a specific training semantic space, thereby reducing generalization — which is the core requirement of OVS.

Through statistical analysis, the authors identify the root cause:

- **High semantic overlap between existing test sets and training sets**: The average similarity between VOC and the training set reaches 0.97, PC-59 reaches 0.95, and even ADE-847, considered a challenging benchmark, reaches 0.79.
- **Evaluation bias**: Under such high overlap, performance gains from fine-tuning CLIP reflect overfitting to training semantics rather than genuine open-vocabulary understanding.

This finding motivates work in two directions:

**Evaluation**: Construction of OpenBench, which exhibits substantially lower semantic similarity to training data (average similarity 0.61, maximum 0.79), comprising 286 fine-grained categories and 6,056 images without semantic duplication.

**Method**: Design of OVSNet, which improves segmentation performance without sacrificing CLIP's generalization capability.

## Method

### Overall Architecture

OVSNet builds upon the CLIP visual and text encoders, combined with a trained segmentation decoder (Mask2Former):
1. The CLIP image encoder extracts visual features; the segmentation decoder generates mask proposals and query embeddings.
2. CLIP features are extracted from predicted mask regions via Mask Pooling.
3. Gradient-Free Aggregation (GFA) fuses query embeddings and CLIP features.
4. The aggregated visual embeddings are aligned with text embeddings via vision-language alignment.
5. During training, Proxy Calibration (PC) is applied to expand the training semantic space.

### Key Design 1: Gradient-Free Aggregation (GFA)

The core tension is that the segmentation decoder's query embeddings carry strong region-aligned priors for training categories but generalize poorly to novel categories, whereas CLIP mask-pooled features exhibit strong generalization but lack region-level alignment.

Learned fusion mechanisms (e.g., self-attention, cross-attention) tend to over-rely on query embeddings during training while neglecting CLIP features, thereby harming generalization. Inspired by the Random Walk algorithm, a gradient-free formulation is adopted instead.

The affinity matrix is initialized as: $\mathcal{Z} = \lambda F_C^0 (F_Q^0)^\top$

The iterative update equations are:

$$F_Q^t = \omega \text{Norm}(\mathcal{Z})^\top F_C^{t-1} + (1-\omega) F_Q^0$$

$$F_C^t = \omega \mathcal{Z} F_Q^t + (1-\omega) F_C^0$$

A Neumann series approximation yields the closed-form solution as $t \to \infty$:

$$F_C^\infty = (1-\omega)(I - \omega^2 A)^{-1}(\omega \mathcal{Z} F_Q^0 + F_C^0)$$

where $A = \mathcal{Z} \text{Norm}(\mathcal{Z})^\top$ and $\omega \in (0,1)$ controls the degree of fusion. The entire process introduces no gradients, avoiding overfitting to training semantics.

### Key Design 2: Proxy Calibration (PC)

A broader semantic training space encourages more generalizable representations, but expanding the training space for segmentation tasks is prohibitively expensive. PC generates proxy embeddings via convex combinations of training embeddings, simulating unseen semantics at zero cost:

$$F'_{Qmn} = \alpha \cdot F_{Qm} + (1-\alpha) \cdot F_{Qn}$$

where $\alpha \sim \text{Beta}(\gamma, \gamma)$. Convex combinations are applied synchronously to query embeddings $F_Q$, CLIP features $F_C$, and text embeddings $F_T$, followed by cosine distance supervision:

$$\mathcal{L}_{PQ} = 1 - \frac{\mathbf{F'_Q} \cdot \mathbf{F'_T}}{\|\mathbf{F'_Q}\|_2 \|\mathbf{F'_T}\|_2}, \quad \mathcal{L}_{PC} = 1 - \frac{\mathbf{F'_C} \cdot \mathbf{F'_T}}{\|\mathbf{F'_C}\|_2 \|\mathbf{F'_T}\|_2}$$

### Loss & Training

Total loss = Segmentation loss (dice loss + CE loss, weight 5) + Classification loss (CE loss, weight 2) + Proxy loss ($\mathcal{L}_{PQ} + \mathcal{L}_{PC}$)

## Key Experimental Results

### Main Results

Comprehensive comparison on existing benchmarks and OpenBench (Base-level CLIP Backbone):

| Method | ADE-150 | ADE-847 | PC-59 | PC-459 | VOC | OpenBench | Avg. |
|--------|---------|---------|-------|--------|-----|-----------|------|
| SAN | 27.5 | 10.1 | 53.8 | 12.6 | 94.0 | 39.6 | 39.6 |
| CATSeg | 31.8 | 12.0 | 57.5 | 19.0 | 94.6 | 36.1 | 41.8 |
| MAFT+ | 34.6 | 13.8 | 57.5 | 16.2 | 95.4 | 43.7 | 43.5 |
| **OVSNet** | **35.8** | **14.5** | **58.6** | **19.1** | **95.7** | **44.9** | **44.8** |

Under the Large-level CLIP Backbone, OVSNet achieves an average of 47.4, surpassing MAFT+ (46.0) and SED (45.6).

Key finding: CATSeg, which fine-tunes CLIP, performs competitively on existing benchmarks (ADE-150: 31.8) but achieves only 36.1 on OpenBench, significantly below methods that freeze CLIP. This validates that fine-tuning CLIP constitutes overfitting to training semantics.

### Ablation Study

| Method | ADE-150 | PC-459 | OpenBench |
|--------|---------|--------|-----------|
| Baseline | 33.1 | 14.3 | 42.3 |
| + GFA | 34.7 (+1.6) | 16.0 (+1.7) | 43.7 (+1.4) |
| + PC | 33.9 (+0.8) | 17.2 (+2.9) | 44.3 (+2.0) |
| + Both | **35.8** (+2.7) | **19.1** (+4.8) | **44.9** (+2.6) |

GFA vs. learned fusion (OpenBench):

| Method | ADE-150 | PC-459 | OpenBench |
|--------|---------|--------|-----------|
| Self Attention | 34.0 | 14.7 | 40.6 |
| Cross Attention | 34.2 | 14.8 | 41.2 |
| GFA | **34.7** | **16.0** | **43.7** |

Key findings:
1. GFA improves over Cross Attention by 2.5 mIoU on OpenBench, confirming that learned fusion overfits to training semantics.
2. PC yields the largest gains on PC-459 (+2.9) and OpenBench (+2.0), i.e., scenarios with many categories and large distributional discrepancy from training data.
3. $\gamma=2$ (higher probability density at the midpoint of the Beta distribution) yields the best performance.

### Effect of Number of Inference Categories

An interesting finding: given the same model and images, performance degrades as the number of categories provided at inference time increases. The performance of SAN and MAFT+ on ADE-150 decreases monotonically as the number of irrelevant categories grows.

## Highlights & Insights

1. **Fundamental reflection on evaluation**: The first systematic quantification of semantic overlap between existing OVS test sets and training sets, revealing the intrinsic nature of evaluation bias.
2. **OpenBench design**: Addresses the semantic duplication issue present in PC-459 and ADE-847 (fine-grained yet duplication-free), and incorporates an "others" category to better reflect real-world scenarios.
3. **Theoretical elegance of GFA**: The Neumann series closed-form solution eliminates iterative computation while removing the risk of overfitting.
4. **Zero-cost PC**: Operates purely through convex combinations in embedding space, requiring no additional data or annotations.

## Limitations & Future Work

1. OpenBench is primarily curated from existing segmentation datasets, limiting its coverage to the scope of the source datasets.
2. The Neumann series approximation in GFA requires matrix inversion, which incurs non-negligible computational cost at large input scales.
3. On VOC, which exhibits high semantic similarity to the training set, performance is slightly below methods that fine-tune CLIP, as the proposed approach preserves CLIP's generalization space rather than maximizing training-set performance.
4. Hyperparameters such as $\omega$ and $\gamma$ require tuning effort.

## Related Work & Insights

- **Two-stage OVS**: SimSeg and OVSeg generate mask proposals followed by CLIP-based classification.
- **Unified-space methods**: SAN and FCCLIP integrate CLIP features into the segmentation decoder.
- **CLIP fine-tuning methods**: CATSeg and SED adopt an early fusion paradigm based on cost maps.
- **Vision-language pretraining**: Large-scale contrastive learning models such as CLIP and ALIGN.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The insight into evaluation bias and the contribution of OpenBench carry significant value for the field.
- **Technical Quality**: ⭐⭐⭐⭐ — GFA and PC are elegantly designed and effective; ablations are thorough.
- **Practicality**: ⭐⭐⭐⭐ — OpenBench is available for community evaluation; the method yields concrete improvements.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is compellingly argued and supported by rigorous data analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)
- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_openvocabulary_remote_sensing.md)

</div>

<!-- RELATED:END -->
