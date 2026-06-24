---
title: >-
  [Paper Note] BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos
description: >-
  [ECCV2024][Object Detection][Temporal Sentence Grounding] Proposes the Boundary-Aligned Moment Detection Transformer (BAM-DETR), which models moments using an anchor-boundary triplet $(p, d_s, d_e)$ instead of the traditional center-length duplet $(c, l)$. Combined with a dual-pathway decoder and a quality-based ranking mechanism, it effectively addresses the issue of imprecise localization caused by center ambiguity.
tags:
  - "ECCV2024"
  - "Object Detection"
  - "Temporal Sentence Grounding"
  - "Detection Transformer"
  - "Boundary Alignment"
  - "video understanding"
  - "Moment Retrieval"
date: 2026-05-08
content_hash: b849f8c345904a20
---

# BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos

**Conference**: ECCV2024  
**arXiv**: [2312.00083](https://arxiv.org/abs/2312.00083)  
**Code**: [GitHub](https://github.com/Pilhyeon/BAM-DETR)  
**Area**: Object Detection  
**Keywords**: Temporal Sentence Grounding, Detection Transformer, Boundary Alignment, video understanding, Moment Retrieval

## TL;DR

Proposes the Boundary-Aligned Moment Detection Transformer (BAM-DETR), which models moments using an anchor-boundary triplet $(p, d_s, d_e)$ instead of the traditional center-length duplet $(c, l)$. Combined with a dual-pathway decoder and a quality-based ranking mechanism, it effectively addresses the issue of imprecise localization caused by center ambiguity.

## Background & Motivation

Temporal Sentence Grounding (TSG) aims to localize target moment segments in untrimmed videos given a natural language query. Recently, DETR-inspired query-based methods (such as Moment-DETR, QD-DETR) have made significant progress by decoding learned queries to generate moment predictions.

However, existing methods suffer from two core limitations:

1. **Center Ambiguity**: Traditional methods represent moments using $(c, l)$ (center, length), which implicitly assumes that the boundaries are equidistant from the center. However, in practice, the center frame of a moment is often unrelated to the sentence semantics (e.g., the center position in the video might be a low-salience irrelevant scene). Inaccurate center predictions directly lead to shifts in the entire segment.
2. **Score Ranking**: Traditional methods rank candidates using classification scores (sentence-segment matching scores). However, an incomplete segment may still match the sentence well, causing incomplete predictions to be erroneously ranked higher.

The authors validate these issues through diagnostic experiments: on QVHighlights, when the center prediction shift increases (error from [0, 0.1) to [0.4, 0.5)), the IoUs of Moment-DETR and QD-DETR drop sharply from 83-87% to 35-36%, whereas BAM-DETR maintains a stable IoU of ~77% across all error groups.

## Core Problem

How to eliminate the dependency on precise center predictions in temporal sentence grounding to achieve more robust boundary alignment? How to enable the model to prioritize predictions with high localization quality over those with merely high semantic matching scores when ranking candidates?

## Method

### 1. Boundary-Oriented Moment Modeling

Core change: Replace $(c, l)$ with the triplet $(p, d_s, d_e)$, where:

- $p$ is the anchor point, which can be any salient point within the target moment and does not need to be the exact center
- $d_s$ is the distance from the anchor point to the start boundary
- $d_e$ is the distance from the anchor point to the end boundary

The moment is represented as $\hat{\varphi} = (p - d_s, p + d_e)$. This asymmetric design allows the model to only need to find any salient anchor point within the target, greatly reducing localization difficulty.

### 2. Dual-pathway Decoder

Based on the intuition that anchor updates require a global scan to find potential locations, while boundary updates need to focus on fine-grained features near the boundaries, two parallel pathways are designed:

**Anchor Update Pathway**:

- Self-attention layer: Interaction among anchor queries $\mathbf{C}_p$ to eliminate redundancy, utilizing positional encodings based on the current prediction.
- Global cross-attention layer: Performs global attention aggregation over the multimodal memory features $\hat{\mathcal{V}}$, employing Q-K concatenation instead of summation to decouple the roles of features and positional encodings.
- FFN + sigmoid to refine anchor locations.

**Boundary Update Pathway**:

- Locality-enhanced memory: Generates boundary-sensitive features $\hat{\mathcal{V}}_s$ and $\hat{\mathcal{V}}_e$ via 1D convolutional layers, applying regularization to high-activate them near the start and end boundaries, which are then concatenated with the original features to form locality-enhanced memories.
- Boundary-Focused Attention: Uses deformable attention to sample $K=3$ neighboring points around current boundary positions as origins, predicting offsets and weights to perform local feature aggregation.
- FFN + sigmoid to refine boundary distances.

Both pathways undergo $L_D = 2$ layers of iterative decoding.

### 3. Quality-Based Score Ranking

Instead of using classification scores, the system directly predicts the localization quality (maximum IoU with the ground-truth moment) for each candidate:

$$\mathbf{q} = \sigma(\text{MLP}([\mathbf{C}_p \| \mathbf{C}_s \| \mathbf{C}_e]))$$

The quality loss is the L1 distance between the predicted quality score and the actual IoU. The matching cost function also omits the classification term, containing only L1 distance and GIoU loss to enable localization-oriented matching.

### 4. Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{loc}} + \lambda_{\text{qual}}\mathcal{L}_{\text{qual}} + \lambda_{\text{sal}}\mathcal{L}_{\text{sal}} + \lambda_{\text{regul}}\mathcal{L}_{\text{regul}}$$

where $\mathcal{L}_{\text{sal}}$ includes the margin ranking loss, contrastive loss, and negative relationship loss, and $\mathcal{L}_{\text{regul}}$ is the regularization loss for boundary-sensitive features.

## Key Experimental Results

### QVHighlights Test Set (w/o Pre-training)

| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg. |
|------|--------|--------|---------|----------|----------|
| Moment-DETR | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |
| QD-DETR | 62.40 | 44.98 | 62.52 | 39.88 | 39.86 |
| **BAM-DETR** | **62.71** | **48.64** | **64.57** | **46.33** | **45.36** |
| **BAM-DETR†** | **64.07** | **48.12** | **65.61** | **47.51** | **46.91** |

### Charades-STA Test Set

| Method | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|------|--------|--------|--------|------|
| UniVTG | 70.81 | 58.01 | 35.65 | 50.10 |
| **BAM-DETR** | **72.93** | **59.95** | **39.38** | **52.33** |

### TACoS Test Set

| Method | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|------|--------|--------|--------|------|
| QD-DETR | - | 36.77 | 21.07 | 35.76 |
| **BAM-DETR** | **56.69** | **41.54** | **26.77** | **39.31** |

### Ablation Study (QVHighlights val)

| Component | R1@0.5 | R1@0.7 | mAP Avg. |
|------|--------|--------|----------|
| Baseline | 62.39 | 47.87 | 41.75 |
| + Boundary-oriented modeling | 63.42 | 49.23 | 42.42 |
| + Dual-pathway decoder | 63.61 | 50.26 | 44.16 |
| + Quality score | **65.10** | **51.61** | **47.61** |

All three components provide clear contributions, with the quality score yielding the largest gain (+3.45 mAP Avg.).

## Highlights & Insights

1. **Elegant Problem Identification & Modeling**: Pinpoints the overlooked "center ambiguity" issue via comprehensive diagnostic experiments. The proposed anchor-boundary triplet modeling serves as a simple yet effective solution.
2. **Global Search vs. Local Refinement**: The dual-pathway design decouples global search (finding anchors) from local refinement (adjusting boundaries), which is highly intuitive and incurs minimal extra computational overhead.
3. **Quality Scores over Classification Scores**: Radically alters the ranking logic, forcing the model to prioritize high-quality localization results during inference.
4. **Strong Robustness**: Outperforms competitors on the de-biased version of Charades-STA, demonstrating a pronounced advantage particularly under the moment-length bias setting (R1@0.7: 40.74 vs QD-DETR 32.87), indicating that boundary-oriented modeling is inherently robust to biases.
5. **Comprehensive Experiments**: Reaches state-of-the-art performance across three benchmark datasets, with ablation studies clearly dissecting the contribution of each component.

## Limitations & Future Work

1. **Limited to 1D Temporal Space**: The current design targets temporal sentence grounding; extending it to 2D spatial dimensions (e.g., spatial localization in videos) requires additional adaptation.
2. **Dependency on Anchor Quality**: Although it alleviates the dependency on exact centers, the anchor point must still reside inside the target moment, which can still pose challenges for extremely short moments or highly similar context scenes.
3. **Fixed Sampling Points ($K=3$) for Deformable Attention**: A fixed $K=3$ may not be optimal for moments with varying lengths/complexities; an adaptive $K$ could bring further improvements.
4. **Unexplored Synergy with Large Pre-trained Models (e.g., InternVideo)**: Integrating stronger visual backbones could potentially boost performance further.

## Related Work & Insights

- **vs Moment-DETR / QD-DETR**: Also query-based methods in the DETR family, but differ in moment representation (triplet vs. duplet) and decoder architecture (dual-pathway vs. single-pathway). The improvements are particularly prominent under strict IoU thresholds.
- **vs UniVTG**: UniVTG follows a unified framework paradigm and requires 4.2M pre-training data to reach 43.63 mAP, whereas BAM-DETR achieves 45.36 mAP without pre-training and reaches 46.67 mAP using only 236K pre-training data.
- **vs Dense Regression Methods (e.g., SSTG)**: Dense regression predicts boundaries treating each frame as an anchor, but anchor positions are fixed. BAM-DETR employs dynamic anchors that adjust iteratively, achieving accurate localization with fewer predictions.
- **vs 2D Object Detection DETR Variants (DAB-DETR, DINO)**: Temporal moments present distinct challenges compared to spatial objects (e.g., center ambiguity, fuzzy boundaries). BAM-DETR's boundary-oriented modeling is a tailored refinement of the DETR paradigm for the temporal domain.

## Related Work & Insights

1. **Generality of Boundary-Oriented Modeling**: Shifting from center-length to anchor-boundary representation can be extended to other regression tasks (e.g., temporal action localization, audio event detection). Any scenario where "the center is difficult to define precisely" can benefit from this paradigm.
2. **Generalization of Quality Score**: Replacing classification scores with IoU predictions for ranking is a prevailing trend in the detection community (e.g., IoU-Net, FCOS). This work successfully introduces and validates this concept for temporal localization.
3. **Design Philosophy of Dual-Pathway Decoding**: The decoupling of global and local features is applicable to various "coarse-to-fine" localization scenarios, such as rough interval localization followed by boundary refinement in event detection.

## Rating
- Novelty: ⭐⭐⭐⭐ — The boundary-oriented modeling and dual-pathway decoder designs are novel, though the individual technical components are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covered three benchmark datasets, robustness tests, exhaustive ablation studies, and diagnostic experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive figures/tables, and the diagnostic analysis in Table 1 is particularly outstanding.
- Value: ⭐⭐⭐⭐ — Provides a fresh perspective on the DETR paradigm for temporal localization, with a simple yet highly effective method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sim-DETR: Unlock DETR for Temporal Sentence Grounding](../../ICCV2025/object_detection/sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [\[ECCV 2024\] SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)
- [\[ECCV 2024\] DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)

</div>

<!-- RELATED:END -->
