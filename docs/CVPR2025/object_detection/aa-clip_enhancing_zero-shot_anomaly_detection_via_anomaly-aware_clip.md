---
title: >-
  [Paper Note] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP
description: >-
  [CVPR 2025][Object Detection][Zero-Shot Anomaly Detection] Proposed AA-CLIP, which enhances anomaly discriminability while preserving the generalization ability of CLIP through a two-stage training strategy (first adapting the text encoder to establish anomaly-aware anchors, then aligning patch-level visual features). It achieves SOTA zero-shot anomaly detection performance across multiple industrial and medical datasets with minimal training samples.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Zero-Shot Anomaly Detection"
  - "CLIP Adaptation"
  - "Text Anchor Disentanglement"
  - "Residual Adapter"
  - "Industrial/Medical Anomalies"
date: 2026-05-08
content_hash: 95a573d70492e539
---

# AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP

**Conference**: CVPR 2025  
**arXiv**: [2503.06661](https://arxiv.org/abs/2503.06661)  
**Code**: [https://github.com/Mwxinnn/AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)  
**Area**: Anomaly Detection / Medical Imaging  
**Keywords**: Zero-Shot Anomaly Detection, CLIP Adaptation, Text Anchor Disentanglement, Residual Adapter, Industrial/Medical Anomalies

## TL;DR

Proposed AA-CLIP, which enhances anomaly discriminability while preserving the generalization ability of CLIP through a two-stage training strategy (first adapting the text encoder to establish anomaly-aware anchors, then aligning patch-level visual features). It achieves SOTA zero-shot anomaly detection performance across multiple industrial and medical datasets with minimal training samples.

## Background & Motivation

Anomaly Detection (AD) aims to model normal distributions to identify outlier samples, which is widely applied in industrial defect detection and medical lesion detection. Traditional AD methods rely heavily on sufficient labeled data, exhibiting limited generalization capabilities. Leveraging large-scale image-text contrastive pre-training, CLIP has demonstrated powerful zero-shot transfer capabilities, becoming a popular solution for few/zero-shot AD.

However, existing CLIP-based AD methods face a core issue: **CLIP is inherently "anomaly unaware" (Anomaly Unawareness)**. The reason lies in the fact that CLIP is trained on general data, lacking a fine-grained semantic understanding of defects/anomalies. Specifically, this manifests as:
1. Normal and abnormal text embeddings are highly intertwined in the feature space, and t-SNE visualization shows they are almost inseparable.
2. Even if an image has obvious defects, the similarity between its visual features and the "normal" description remains higher than that with the "anomaly" description.
3. Directly using original CLIP text embeddings as AD anchors yields poor performance.

Existing solutions either adapt visual features exclusively while overlooking the anomaly unawareness in the text space (e.g., VAND, MVFA-AD), or employ prompt learning to reform the text encoder at the cost of disrupting class-specific information (e.g., AnomalyCLIP, AdaCLIP).

**Core Idea**: First establish anomaly-aware "anchors" in the text space to clearly separate normal and abnormal semantics, and then guide visual features to align with these anchors for precise localization. This realizes a two-stage sequential adaptation, utilizing residual adapters to safeguard the original knowledge of CLIP.

## Method

### Overall Architecture

AA-CLIP adopts a two-stage training workflow, where the original parameters of CLIP remain frozen throughout:
- **Input**: Image + normal/anomaly text prompts
- **First Stage**: Freeze the visual encoder, adapt the text encoder → Output anomaly-aware text anchors $T_N$, $T_A$
- **Second Stage**: Freeze the text encoder, adapt the visual encoder → Output patch-level visual features aligned with the anchors
- **Inference**: Compare the cosine similarity between visual features and anchors → Pixel-level anomaly segmentation maps + image-level anomaly scores

### Key Designs

1. **Residual Adapter**:

    - **Function**: Injects trainable modules into the shallow layers of the encoders to adapt features while retaining the original knowledge.
    - **Mechanism**: Passes the $i$-th Transformer layer output $x^i$ through a linear transformation, activation, and normalization to generate a residual $x^i_{residual}$, which is then weighted and fused with the original feature:
    $$x^i_{enhanced} = \lambda \cdot x^i_{residual} + (1-\lambda) \cdot x^i$$
      where $\lambda=0.1$ to ensure that the original information remains dominant.
    - **Design Motivation**: Directly inserting ordinary adapters severely degrades the generalization capability of CLIP (pixel-AUROC drops by 40 points in ablation studies). A residual connection blends new information at a small ratio to achieve a "gentle adaptation".

2. **Stage 1: Text Anchor Disentanglement**:

    - **Function**: Inserts residual adapters into the first $K_T=3$ layers of the text encoder to generate text embeddings capable of distinguishing between normal and abnormal semantics.
    - **Mechanism**: The average embeddings of normal prompts and anomaly prompts are designated as anchors $T_N$ and $T_A$, respectively. The cosine similarity between these anchors and visual features is calculated to obtain classification prediction $p_{cls}$ and segmentation prediction $p_{seg}$.
    - Simultaneously, a **Disentangle Loss** is introduced to enforce orthogonality between normal and abnormal anchors:
    $$\mathcal{L}_{dis} = |\langle T_N, T_A \rangle|^2$$
    - **Design Motivation**: To ensure that the two types of anchors are sufficiently separated in the feature space, reducing confusion. t-SNE shows that normal/abnormal embeddings for each category are clearly decoupled after adaptation and that this capability generalizes well to unseen classes.

3. **Stage 2: Patch Feature Alignment**:

    - **Function**: Inserts residual adapters into the first $K_I=6$ layers of the visual encoder to align multi-granularity patch features with text anchors.
    - **Mechanism**: Extracts intermediate features $F^i$ from the 6th, 12th, 18th, and 24th layers of the visual encoder, sums and aggregates them after mapping via trainable projectors:
    $$V_{patch} = \sum_{i=1}^{4} Proj_i(F^i)$$
    - **Design Motivation**: Multi-granularity feature fusion enables anomalies of different scales to be captured. The text encoder is frozen in this stage to avoid category information collapse caused by joint training.

### Loss & Training

- Alignment Loss: $\mathcal{L}_{align} = \mathcal{L}_{cls} + \mathcal{L}_{seg}$
    - $\mathcal{L}_{cls}$: Image-level BCE loss
    - $\mathcal{L}_{seg}$: Pixel-level Dice + Focal loss
- Total Loss: $\mathcal{L}_{total} = \mathcal{L}_{align} + \gamma \mathcal{L}_{dis}$, $\gamma=0.1$
- Stage 1: 5 epochs, lr $1\times10^{-5}$; Stage 2: 20 epochs, lr $5\times10^{-4}$
- Key reason for two-stage decoupled training: Single-stage joint training easily leads to category information collapse (verified by ablation), destroying zero-shot generalization.

## Key Experimental Results

### Main Results

| Dataset | Metric | AA-CLIP (full) | AnomalyCLIP | AdaCLIP | Gain |
|--------|------|----------------|-------------|---------|------|
| 11-Dataset Avg | Pixel-AUROC | **93.4** | 91.3 | 90.4 | +2.1 |
| 7-Dataset Avg | Image-AUROC | **83.1** (64-shot) | 78.4 | 80.6 | +2.5 |
| Liver CT | Pixel-AUROC | **97.8** | 93.9 | 94.5 | +3.3 |
| Retina OCT | Pixel-AUROC | **95.5** | 92.6 | 88.5 | +2.9 |
| ClinicDB | Pixel-AUROC | **89.9** | 85.0 | 85.9 | +4.0 |

- Most prominent finding: Trained with only **2-shot** (1 normal + 1 anomaly per class), the pixel-level AUROC reaches 92.0%, surpassing all prior methods trained on full-shot.
- The advantage is more pronounced in the medical field: reaching 97.8% on liver CT and 96.5% on brain MRI.

### Ablation Study

| Configuration | Pixel-AUROC | Image-AUROC | Description |
|------|-----------|-----------|------|
| Original CLIP | 50.3 | 69.3 | Baseline |
| + Linear Proj (VAND) | 88.9 | 69.3 | Visual adaptation only |
| + Ordinary Adapter | 48.9 (-40.0) | 53.4 (-15.9) | Destroys original knowledge |
| + Residual Adapter | 91.3 (+2.4) | 80.7 (+11.4) | Protects generalization ability |
| + Text Residual Adapter | 92.1 (+3.2) | 82.6 (+13.3) | Effective adaptation of text space |
| + Disentangle Loss | **92.7** (+3.8) | **83.3** (+14.0) | Anchor disentanglement necessary |

### Key Findings

- Directly inserting ordinary adapters into Transformer layers leads to a catastrophic drop in zero-shot performance, validating the necessity of the residual design.
- Text-space adaptation is more key than visual-space adaptation: text anchor disentanglement provides a more precise semantic foundation for visual alignment.
- Single-stage joint training (e.g., AdaCLIP strategy) leads to category information collapse; the two-stage strategy is key to preserving generalization capability.

## Highlights & Insights

- **Precise Problem Definition**: Systematically analyzes the "anomaly unawareness" limitation of CLIP for the first time, clearly highlighting the issue using t-SNE, heatmaps, and counterexamples.
- **Minimalist & Efficient Method**: The core only consists of residual adapters + two-stage training, free of complex architectures, reproducible on a single RTX 3090.
- **High Data Efficiency**: 2-shot adaptation alone outperforms previous full-shot methods, carrying significant value for data-scarce medical scenarios.
- **Cross-Domain Generalization**: Trained on VisA, yet seamlessly transfers to completely different datasets such as MVTec-AD, brain MRI, liver CT, and retina OCT.

## Limitations & Future Work

- Signs of overfitting emerge in full-shot training, indicating a saturation point in CLIP adaptation, which warrants research into better regularization strategies.
- Validation is restricted to industrial and medical domains; generalization to natural images, remote sensing, and other fields remains unexplored.
- Prompt designs are relatively fixed; automated prompt search could be explored for further improvement.

## Related Work & Insights

- The fine-grained semantic perception of CLIP is both its bottleneck and a research hotspot. The logic of AA-CLIP (first correcting the text space, then guiding the visual space) can be extended to other tasks requiring fine-grained understanding.
- The design philosophy of the residual adapter resembles LoRA but is simpler, achieving a sound balance between preserving pre-trained knowledge and injecting new capabilities.
- Insight from the two-stage strategy: Decoupling the adaptation of different modalities prevents mutual interference.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel problem formulation (Anomaly Unawareness); the methodological design is reasonable, although the components themselves are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets, diverse shot configurations, thorough ablation studies, and dual-domain validation (industry + medical).
- Writing Quality: ⭐⭐⭐⭐⭐ Rich visualizations (t-SNE, heatmaps, qualitative outcomes) with coherent logic.
- Value: ⭐⭐⭐⭐ Practically valuable to the zero-shot AD field, particularly in addressing low-data requirements in medical contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DLVP-CLIP: Enhancing Fine-Grained Zero-Shot Anomaly Detection via Dynamic Local Visual Prompting](../../CVPR2026/object_detection/dlvp-clip_enhancing_fine-grained_zero-shot_anomaly_detection_via_dynamic_local_v.md)
- [\[CVPR 2025\] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](towards_zero-shot_anomaly_detection_and_reasoning_with_multimodal_large_language.md)
- [\[CVPR 2026\] FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement](../../CVPR2026/object_detection/fb-clip_fine-grained_zero-shot_anomaly_detection_with_foreground-background_dise.md)
- [\[CVPR 2025\] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting](t2icount_enhancing_cross-modal_understanding_for_zero-shot_counting.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](../../CVPR2026/object_detection/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)

</div>

<!-- RELATED:END -->
