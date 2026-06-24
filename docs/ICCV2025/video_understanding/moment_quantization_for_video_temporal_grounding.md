---
title: >-
  [Paper Note] Moment Quantization for Video Temporal Grounding
description: >-
  [ICCV 2025][Video Understanding][Video temporal grounding] This paper proposes MQVTG, which for the first time introduces vector quantization into video temporal grounding (VTG) by mapping video clips to discrete vectors via a moment codebook and soft quantization, thereby enhancing foreground/background discriminability and achieving state-of-the-art performance on 6 benchmarks.
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Video temporal grounding"
  - "vector quantization"
  - "moment codebook"
  - "highlight detection"
  - "discrete representation learning"
date: 2026-05-08
content_hash: d300ae8ccb5dcc14
---

# Moment Quantization for Video Temporal Grounding

**Conference**: ICCV 2025
**arXiv**: [2504.02286](https://arxiv.org/abs/2504.02286)  
**Code**: None  
**Area**: Video Understanding
**Keywords**: Video temporal grounding, vector quantization, moment codebook, highlight detection, discrete representation learning

## TL;DR

This paper proposes MQVTG, which for the first time introduces vector quantization into video temporal grounding (VTG) by mapping video clips to discrete vectors via a moment codebook and soft quantization, thereby enhancing foreground/background discriminability and achieving state-of-the-art performance on 6 benchmarks.

## Background & Motivation

Video temporal grounding (VTG) aims to localize relevant moments in a video given natural language descriptions. The core challenge lies in distinguishing relevant from irrelevant moments.

Limitations of existing methods:

**Weak discriminability in continuous feature space**: Taking TR-DETR as an example, foreground and similar background features are close to each other in the feature space, making them difficult to separate.

**Insufficient foreground feature aggregation**: Features from different foreground regions are scattered in the feature space without effective aggregation.

**Redundancy in video data**: Videos contain substantial redundant information, while the essence of VTG is to extract discriminative features that distinguish foreground from background.

Key insight: foreground moments can be described concisely in discrete natural language (e.g., "stirring curry with a spoon"). This raises the question of whether discrete vectors can describe continuous video moments to enhance discriminability. The clustering nature of vector quantization naturally aligns with the foreground/background separation requirement in VTG.

## Method

### Overall Architecture

MQVTG progresses from simple Clip Quantization to Moment Quantization, with three core improvements: (1) quantization is placed after temporal modeling to accommodate cross-clip characteristics; (2) soft quantization is employed to preserve visual diversity; (3) a moment codebook with prior initialization and joint projection is designed.

### Key Designs

1. **Evolution from Clip Quantization to Moment Quantization**:

    - Clip Quantization, analogous to image quantization, directly quantizes individual clips and ignores two properties of video moments: **cross-clip nature** (an action spans multiple clips) and **visual diversity** (the same description has multiple visual manifestations).
    - Moment Quantization performs quantization after the temporal encoder $E_t$, so that the quantization operates on features $z_t = E_t(z_s)$ that already encode temporal relationships.

2. **Soft Quantization**: Rather than directly replacing continuous features with discrete codewords (hard quantization), the quantization process serves as a clustering regularization. A codebook loss $\mathcal{L}_{cb} = \|C(z_t) - \text{sg}(E_t(z_s))\|_2^2$ and a commitment loss $\mathcal{L}_{cmt} = \|\text{sg}(C(z_t)) - E_t(z_s)\|_2^2$ drive feature–codeword clustering, while the downstream grounding module continues to use continuous features $z_t$. This avoids information loss caused by a limited-capacity codebook.

3. **Moment Codebook**:

    - **Prior initialization**: CLIP features are extracted for all training video clips, and k-means clustering is applied; the cluster centers are used to initialize the codebook, ensuring that codewords are effective from the start.
    - **Joint projection**: A trainable projection layer $C' = P(C)$ (linear layer) is introduced to replace direct optimization of codebook vectors, enabling the exploration of temporal semantic relationships among different codewords.

4. **Plug-and-play property**: The quantization module can be integrated into both encoder-only and encoder-decoder (DETR) architectures. During training, only codebook parameters are added; at inference, there is zero additional cost.

### Loss & Training

Overall loss: $\mathcal{L}_{overall} = \mathcal{L}_{mr} + \lambda_{hd}\mathcal{L}_{hd} + \lambda_{mq}\mathcal{L}_{mq} + \lambda_{align}\mathcal{L}_{align}$

- $\mathcal{L}_{mr}$: moment retrieval loss (L1 + Focal)
- $\mathcal{L}_{hd}$: highlight detection loss (intra-video contrastive learning)
- $\mathcal{L}_{mq} = \mathcal{L}_{cb} + \lambda_{cmt}\mathcal{L}_{cmt}$: quantization supervision
- $\mathcal{L}_{align}$: InfoNCE video–text alignment loss

## Key Experimental Results

### Main Results

QVHighlights validation set (MR + HD):

| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP Avg. | HD mAP | HD HIT@1 |
|--------|--------|--------|---------|----------|--------|----------|
| TR-DETR | 67.10 | 51.48 | 66.27 | 45.09 | 40.55 | 64.77 |
| CG-DETR | 67.35 | 52.06 | 65.57 | 44.93 | **40.79** | **66.71** |
| R²-Tuning | **68.71** | 52.06 | - | 47.59 | 40.59 | 64.32 |
| **MQVTG** | 67.94 | **53.03** | **68.54** | **48.81** | 40.23 | 65.29 |

Moment retrieval on Charades-STA / TACoS / Ego4D-NLQ:

| Method | Charades R1@0.7 | TACoS R1@0.7 | Ego4D mIoU |
|--------|-----------------|--------------|------------|
| R²-Tuning | 37.02 | 25.12 | 4.94 |
| **MQVTG** | **38.84** | **25.82** | **5.08** |

### Ablation Study

Core component ablation (QVHighlights val):

| Configuration | R1@0.5 | R1@0.7 | mAP@0.5 | mAP Avg. |
|---------------|--------|--------|---------|----------|
| Baseline (no quantization) | 65.35 | 49.42 | 66.99 | 45.63 |
| + Quantization after temporal modeling (QATM) | 66.37 | 51.11 | 67.43 | 47.02 |
| + QATM + Soft Quantization (SQ) | 66.52 | 51.23 | 68.18 | 47.54 |
| + QATM + SQ + Moment Codebook (MC) | **67.94** | **53.03** | **68.54** | **48.81** |

Comparison of quantization strategies:

| Quantization | R1@0.7 | mAP Avg. | Note |
|--------------|--------|----------|------|
| Image Quantization | 51.03 | 46.55 | Image-level quantization |
| Clip Quantization | 51.61 | 46.93 | Clip-level quantization |
| **Moment Quantization** | **53.03** | **48.81** | Moment-level quantization |
| Hard Quantization | 50.90 | 47.46 | Direct replacement with discrete vectors |

### Key Findings

- Plug-and-play validation: consistent improvements are observed when integrated into DETR-based models including QD-DETR, TR-DETR, and TaskWeave.
- Highlight detection surpasses the prior state of the art by 2.1% on YouTube HL and 1.4% on TVSum.
- Low codebook utilization (< 10%) is the current bottleneck, limiting performance on fine-grained scenes.

## Highlights & Insights

- This work is the first to transfer vector quantization from image/audio domains to video temporal grounding, filling a gap in this research direction.
- The soft quantization strategy is elegant: the quantization process serves as a regularizer that drives feature clustering without directly using discrete codewords, balancing discriminability and information completeness.
- The k-means prior initialization is simple yet effective, addressing the cold-start problem in codebook training.

## Limitations & Future Work

- Codebook utilization is low (< 10%), with a large proportion of codewords remaining inactive, limiting performance on fine-grained scenes.
- Improvements on highlight detection are less pronounced than on moment retrieval, reflecting a conflict between global and local information requirements.
- Future work may explore dynamic codebook sizes or hierarchical codebook structures.

## Related Work & Insights

- The idea of using quantization as a regularization tool (rather than for compression or reconstruction purposes) is generalizable to other video tasks requiring foreground/background separation.
- The joint projection strategy of the moment codebook offers insights into establishing associations among codebook vectors.
- The plug-and-play nature allows this method to be combined with future stronger baseline models.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](../../NeurIPS2025/video_understanding/when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[NeurIPS 2025\] DualGround: Structured Phrase and Sentence-Level Temporal Grounding](../../NeurIPS2025/video_understanding/dualground_phrase_temporal.md)

</div>

<!-- RELATED:END -->
