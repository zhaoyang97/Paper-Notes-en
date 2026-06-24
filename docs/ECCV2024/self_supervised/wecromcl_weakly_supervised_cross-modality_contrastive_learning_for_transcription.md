---
title: >-
  [Paper Note] WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting
description: >-
  [ECCV 2024][Self-Supervised Learning][Text Detection] The WeCromCL framework is proposed to achieve scene text spotting using only transcription annotations (without location annotations) through weakly supervised atomic cross-modality contrastive learning. The detected anchor points are utilized as pseudo-labels to train a single-point supervised text detector, achieving performance close to fully supervised methods without bounding box annotations.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Text Detection"
  - "Weakly Supervised Learning"
  - "Cross-Modality Contrastive Learning"
  - "Transcription Supervision"
  - "Scene Text Recognition"
date: 2026-05-08
content_hash: be747accc01f8095
---

# WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting

**Conference**: ECCV 2024  
**arXiv**: [2407.19507](https://arxiv.org/abs/2407.19507)  
**Code**: Yes ([https://github.com/ZhengyaoFang/WeCromCL](https://github.com/ZhengyaoFang/WeCromCL))  
**Area**: Self-Supervised  
**Keywords**: Text Detection, Weakly Supervised Learning, Cross-Modality Contrastive Learning, Transcription Supervision, Scene Text Recognition

## TL;DR

The WeCromCL framework is proposed to achieve scene text spotting using only transcription annotations (without location annotations) through weakly supervised atomic cross-modality contrastive learning. The detected anchor points are utilized as pseudo-labels to train a single-point supervised text detector, achieving performance close to fully supervised methods without bounding box annotations.

## Background & Motivation

Scene text spotting typically requires precise text boundary annotations (polygons/rectangles), which are extremely expensive to obtain. **Transcription-only supervision** offers an attractive alternative, requiring only text content annotations without any location labels.

Limitations of existing transcription-only supervised methods:

**NPTS**: Models text spotting as a sequence prediction task, concatenating all text instances into a single sequence for autoregressive prediction. However, since there is no predefined order among text instances, the model must fit all permutations, making training convergence extremely difficult and requiring substantial computational resources.

**TOSS**: Leverages pre-learned queries as in DETR to locate text, but DETR is inherently designed to rely on positional supervision, which limits its performance when location labels are absent.

The core insight of this work is to decompose transcription-only supervised text spotting into two stages: first, locating anchor points through weakly supervised cross-modality contrastive learning, and second, training a single-point supervised detector using these anchors as pseudo-labels.

## Method

### Overall Architecture

WeCromCL adopts a two-stage pipeline:

**Stage 1: Weakly Supervised Anchor Point Detection**
- Input: Scene images + text transcriptions (no location annotations)
- Output: The anchor position of each transcription inside the image
- Method: Atomic cross-modality contrastive learning

**Stage 2: Anchor-guided Text Spotting**
- Input: Images + anchor pseudo-labels
- Output: Text detection and recognition results
- Method: Single-point detectors based on SPTS or adapted SRSTS v2

### Key Designs

**Atomic Contrastive Learning vs. Holistic Contrastive Learning**:

| Dimension | Holistic (e.g., CLIP/oCLIP) | Atomic (WeCromCL) |
|------|------------------------|-------------------|
| Objective | Global semantic relevance of image-text pairs | Character-level visual appearance consistency between transcriptions and local image regions |
| Granularity | Whole image vs. full text | Pixel-level activation maps vs. character-wise matching |
| Localization Capability | Cannot locate precisely | Enables anchor localization via activation map peaks |

**Character-Wise Text Encoder**:

- Learns an independent vector embedding $\mathbf{E} \in \mathbb{R}^{|\Sigma| \times C}$ for each character in the alphabet
- Learns a positional embedding $\mathbf{P} \in \mathbb{R}^{L \times C}$ to preserve the sequential information of characters
- Models relations between characters via a Transformer Encoder after fusion
- Finally averages all character representations to obtain the text representation $\mathbf{F}_T \in \mathbb{R}^C$

**Soft-modeled Activation Map**:

The activation map is calculated via cross-modality cross-attention, using the text representation as the query and the visual feature of each pixel as the key/value:

$$\mathbf{M}_{(i,j)} = (\mathbf{W}_T^\top \mathbf{F}_T) \cdot (\mathbf{W}_I^\top \mathbf{F}_{I,(i,j)})$$

After normalized by softmax, the peak locations represent the anchor points. The activation map is further used to aggregate the visual features corresponding to the transcription from the image.

**Negative Sample Mining**:

Unpaired transcriptions are randomly selected to construct more negative sample pairs. By increasing the number of negative samples $N_{\text{aug}}$ in the image-to-text direction, discriminative capability is enhanced with almost negligible overhead.

### Loss & Training

The contrastive learning loss comprises two directions:

**Text-to-Image (T2I) Direction**:

$$\mathcal{L}_i^{T2I} = -\log\frac{\exp(\text{Cosine}(\mathbf{F}_{I_i,T_i}^c, \mathbf{F}_{T_i})/\tau)}{\sum_{j=0}^{N-1}\exp(\text{Cosine}(\mathbf{F}_{I_j,T_i}^c, \mathbf{F}_{T_i})/\tau)}$$

**Image-to-Text (I2T) Direction** (including negative sample mining):

$$\mathcal{L}_i^{I2T} = -\log\frac{\exp(\text{Cosine}(\mathbf{F}_{I_i,T_i}^c, \mathbf{F}_{T_i})/\tau)}{\sum_{j=0}^{N+N_{\text{aug}}-1}\exp(\text{Cosine}(\mathbf{F}_{I_i,T_j}^c, \mathbf{F}_{T_j})/\tau)}$$

The overall loss is the average of both directions.

## Key Experimental Results

### Main Results

WeCromCL anchor point detection performance (F-measure, single-point metric):

| Dataset | Train Set | Test Set |
|--------|--------|--------|
| ICDAR 2013 | 93.2 | 90.5 |
| ICDAR 2015 | 88.6 | 83.4 |
| Total-Text | 84.3 | 80.3 |
| CTW1500 | 66.3 | 77.7 |

WeCromCL + SPTS vs. NPTS (Edit Distance Metric):

| Method | ICDAR2015 S | W | G | Total-Text None | Full |
|------|------------|---|---|----------------|------|
| NPTS | 70.3 | 62.7 | 57.0 | 61.6 | 70.6 |
| WeCromCL + SPTS | **71.8** | **64.7** | **59.7** | **63.2** | **70.7** |

### Ablation Study

Character-wise vs. Word-wise Text Encoder (Test set F-measure):

| Encoder Type | IC13 | IC15 | Total-Text | CTW1500 |
|-----------|------|------|------------|---------|
| Token-wise (CLIP) | 78.6 | 64.4 | 64.9 | 65.5 |
| **Character-wise** | **90.5** | **83.4** | **80.3** | **77.7** |

WeCromCL vs. oCLIP (Test set F-measure):

| Method | IC13 | IC15 | Total-Text | CTW1500 |
|------|------|------|------------|---------|
| oCLIP (Holistic Contrastive) | 72.5 | 41.7 | 42.8 | 45.9 |
| **WeCromCL (Atomic Contrastive)** | **90.5** | **83.4** | **80.3** | **77.7** |

### Key Findings

1. The character-wise encoder improves performance by over 10 F1 points compared to the word-wise encoder across all datasets, suggesting that text spotting relies more on visual appearance matching rather than semantic matching.
2. Atomic contrastive learning (WeCromCL) significantly outperforms holistic contrastive learning (oCLIP), with a performance gap of 31.8% on CTW1500.
3. Negative sample mining increases the F-measure on the CTW1500 test set by 10.2%.
4. The pseudo-labels generated by WeCromCL can enhance fully supervised detectors, showing particularly significant improvements when annotated data is scarce.

## Highlights & Insights

1. **Elegant Problem Decomposition**: Decomposing the challenging transcription-only supervised task into two approachable sub-problems (weakly supervised localization + single-point supervised detection) significantly reduces optimization difficulty.
2. **Introduction of Atomic Contrastive Learning**: Unlike models like CLIP that focus on semantic correlation, WeCromCL learns character-level visual appearance consistency, which addresses the fundamental requirement of text spotting.
3. **Analogy of Clustering Centers**: Transcriptions act as cluster centers that associate all images containing them, allowing the model to learn the common visual appearance patterns of each transcription across numerous images, which is a highly intuitive interpretation.
4. **Low-cost Negative Sample Augmentation**: Generating additional negative samples only on the text side (incurring almost zero computation cost) provides substantial performance gains.

## Limitations & Future Work

1. The two-stage pipeline allows anchor localization errors to propagate to the detection stage, suggesting future explorations in end-to-end joint optimization.
2. When multiple identical transcriptions appear within the same image, the activation maps may suffer from ambiguity.
3. The applicability of the character-wise encoder to non-Latin alphabet languages (e.g., Chinese characters) remains to be validated.
4. Comparisons with the latest large vision-language models (e.g., pipeline combining SAM + OCR) have not been conducted.

## Related Work & Insights

- **oCLIP / VLPT**: Representatives of holistic contrastive learning, focusing on global semantic matching.
- **SPTS**: A single-point supervised text detector, serving as an ideal partner for WeCromCL.
- **NPTS**: Also a transcription-only supervised method, but its single-stage design poses challenging optimization difficulties.
- **Insight**: Weakly supervised localization tasks can be formulated as cross-modality contrastive learning, where the peaks of activation maps correspond to the localization results.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| **Overall** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)
- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[ECCV 2024\] InfMAE: A Foundation Model in the Infrared Modality](infmae_a_foundation_model_in_the_infrared_modality.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](../../ICLR2026/self_supervised/on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)

</div>

<!-- RELATED:END -->
