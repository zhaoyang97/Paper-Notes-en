---
title: >-
  [Paper Note] ResCLIP: Residual Attention for Training-free Dense Vision-language Inference
description: >-
  [CVPR 2025][Segmentation][Open-vocabulary Semantic Segmentation] Discovers that cross-correlation self-attention in the intermediate layers of CLIP possesses localization properties, and proposes two plug-and-play modules: Residual Cross-correlation Self-attention (RCS) and Semantic Feedback Refinement (SFR), significantly improving CLIP's dense inference capabilities in open-vocabulary semantic segmentation.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Open-vocabulary Semantic Segmentation"
  - "CLIP"
  - "Training-free"
  - "Residual Attention"
  - "Dense Inference"
date: 2026-05-08
content_hash: 9cce084fdebf522c
---

# ResCLIP: Residual Attention for Training-free Dense Vision-language Inference

**Conference**: CVPR 2025  
**arXiv**: [2411.15851](https://arxiv.org/abs/2411.15851)  
**Code**: [GitHub](https://github.com/yvhangyang/ResCLIP)  
**Area**: Semantic Segmentation  
**Keywords**: Open-vocabulary Semantic Segmentation, CLIP, Training-free, Residual Attention, Dense Inference

## TL;DR

Discovers that cross-correlation self-attention in the intermediate layers of CLIP possesses localization properties, and proposes two plug-and-play modules: Residual Cross-correlation Self-attention (RCS) and Semantic Feedback Refinement (SFR), significantly improving CLIP's dense inference capabilities in open-vocabulary semantic segmentation.

## Background & Motivation

- Vision-language models like CLIP perform exceptionally in image-level open-vocabulary tasks, but underperform on pixel-level dense prediction tasks (such as semantic segmentation).
- Prior research has found that the self-attention in the final layer of CLIP exhibits spatial invariance, where the attention patterns of all patches become similar, losing spatial discriminative ability.
- Existing training-free methods (e.g., SCLIP, ClearCLIP, NACLIP) obtain spatially covariant features by replacing the query-key attention of the final layer with self-correlation attention (query-query or key-key).
- However, these methods overlook the rich spatial correspondences captured by cross-correlation self-attention (query-key).
- The authors discover a key phenomenon: Cross-Correlation Self-Attention (C2SA) in the **non-final layers** of CLIP also exhibits localization attributes and class specificity.
- There is a need for a method to introduce the localization information from intermediate layers into the final layer, while enhancing the semantic consistency of regions belonging to the same category.

## Method

### Overall Architecture

ResCLIP consists of two modules: Residual Cross-correlation Self-attention (RCS) and Semantic Feedback Refinement (SFR). RCS extracts and aggregates cross-correlation attention $\mathcal{A}_c$ from intermediate layers of CLIP, and performs residual fusion with the self-correlation attention $\mathcal{A}_s$ of the final layer. SFR utilizes initial segmentation results as semantic feedback to further adjust attention scores, thereby enhancing focus and local consistency in regions of the same category. Both modules can be seamlessly integrated into existing methods such as SCLIP, ClearCLIP, and NACLIP as plug-and-play solutions.

### Key Designs

**1. Residual Cross-correlation Self-attention (RCS)**

- **Function**: Restores the spatial invariance issue of the final-layer attention using cross-correlation attention from intermediate layers.
- **Mechanism**: Extracts standard query-key attention $\mathcal{A}_{qk}^i$ from the $s$-th to the $e$-th layer, and averages them to obtain the aggregated attention $\mathcal{A}_c = \frac{1}{N}\sum_{i=s}^{e}\mathcal{A}_{qk}^i$. It then performs weighted fusion with the self-correlation attention of the final layer: $\mathcal{A}_{rcs} = (1-\lambda_{rcs})\cdot\mathcal{A}_s + \lambda_{rcs}\cdot\mathcal{A}_c$.
- **Design Motivation**: Although Self-Correlation Self-Attention (SCSA) resolves spatial invariance, it lacks cross-feature dynamics. In contrast, cross-correlation attention in intermediate layers retains rich class-specific localization information (visualizations clearly show that intermediate-layer attention can focus on regions of objects belonging to the same class).

**2. Semantic Feedback Refinement (SFR)**

- **Function**: Explicitly enhances attention focus on regions of the same category, maintaining local spatial consistency.
- **Mechanism**: First uses RCS to obtain an initial segmentation map $\mathcal{M}$, then constructs a semantic mask: if patch $(m,n)$ belongs to the same class as the target patch, the attention is kept; otherwise, it is decayed. Connected component analysis is used to distinguish connected and disconnected regions of the same category, applying a Chebyshev distance-based decay function $D(p,q) = \exp(-\frac{d(p,q)}{\max(d(\cdot,\cdot))})$ to disconnected regions, followed by Gaussian kernel smoothing. The final fusion formula is $S_r = (1-\lambda_{sfr})\cdot S_s + \lambda_{sfr}\cdot\hat{S}$.
- **Design Motivation**: Previous neighborhood priors (such as NACLIP's Gaussian kernel) are isotropic and fail to adapt to objects of different shapes. SFR provides an adaptive, category-aware prior through the semantic segmentation map to guide attention more precisely.

**3. Layer Fusion Strategy**

- **Function**: Determines which intermediate layers to aggregate cross-correlation attention from.
- **Mechanism**: Explores two strategies: cumulative aggregation (from layer 1 to layer $n$) and sliding window aggregation (window size of 4). The optimal configuration is sliding window aggregation on layers 6 $\rightarrow$ 9.
- **Design Motivation**: Different layers capture different granularities of spatial correspondence, and appropriate layer selection can balance local details and global semantics.

### Loss & Training

- Completely training-free, involving no extra training or fine-tuning.
- Only modifies the calculation of the final-layer attention in CLIP.
- Hyperparameters are set to $\lambda_{rcs}=0.5$ and $\lambda_{sfr}=0.7$ (stable within the 0.6-0.8 range).
- Uses standard ImageNet prompt templates without additional text augmentation strategies.
- Sliding window inference: $224 \times 224$ window shape, stride of 112.

## Key Experimental Results

### Main Results

mIoU on datasets without background classes (ViT-B/16):

| Method | VOC20 | Context59 | Stuff | Cityscape | ADE20k | Average |
|------|-------|-----------|-------|-----------|--------|------|
| SCLIP | 80.4 | 34.2 | 22.4 | 32.2 | 16.1 | 37.1 |
| +ResCLIP | **84.6** | 35.8 | 23.9 | 34.4 | 17.6 | 39.3(+2.2) |
| ClearCLIP | 80.9 | 35.9 | 23.9 | 30.0 | 16.7 | 37.5 |
| +ResCLIP | **87.1** | 36.4 | 24.3 | 34.5 | 17.8 | 40.0(+2.5) |
| NACLIP | 79.7 | 35.2 | 23.3 | 35.5 | 17.4 | 38.2 |
| +ResCLIP | **86.0** | **36.8** | **24.7** | **35.9** | **18.0** | **40.3(+2.1)** |

### Ablation Study

Based on NACLIP (ViT-B/16, VOC20):

| Configuration | RCS | SFR | mIoU | Δ |
|------|-----|-----|------|---|
| Baseline (NACLIP) | - | - | 79.7 | - |
| +RCS | ✓ | - | 85.5 | +5.8 |
| +SFR | - | ✓ | 81.5 | +1.8 |
| +RCS+SFR | ✓ | ✓ | **86.0** | **+6.3** |

### Key Findings

1. ResCLIP consistently improves all baseline methods, with up to +13.1% mIoU improvement on SCLIP using ViT-L/14.
2. The RCS module contributes the most (+5.8%), validating the localization value of intermediate-layer attention.
3. The two modules are complementary: using RCS alone yields +5.8%, SFR alone +1.8%, and the combination +6.3%.
4. The sliding window aggregation strategy (layers 6 $\rightarrow$ 9) outperforms cumulative aggregation, indicating that intermediate layer information closer to the final layer is more valuable.
5. Performance is insensitive to hyperparameters, with $\lambda_{rcs}$ and $\lambda_{sfr}$ demonstrating stable performance across a wide range.

## Highlights & Insights

- First to discover that cross-correlation attention in CLIP intermediate layers possesses localization properties, serving as an overlooked yet highly valuable source of information.
- Specifically designed to be plug-and-play, completely compatible with existing methods like SCLIP/ClearCLIP/NACLIP.
- Utilizing segmentation results as feedback to adjust attention in SFR is a highly clever self-guidance strategy.
- Existing methods suffer severe performance degradation on ViT-L/14 (e.g., SCLIP drops by 13.5%), which ResCLIP effectively mitigates.

## Limitations & Future Work

- The SFR module depends on the quality of the initial segmentation; large errors in the initial segmentation may introduce noise.
- The method increases computational overhead during inference (requiring traversal of intermediate-layer attention).
- Only validated on 2D semantic segmentation tasks; future work can explore extending it to other dense tasks such as 3D understanding and object detection.
- Trainable attention adjustment strategies could be integrated in the future to further enhance performance.
- Layer selection strategies could explore more adaptive schemes to replace manual sliding windows.

## Related Work & Insights

- **SCLIP / ClearCLIP / NACLIP**: Training-free methods that replace final-layer attention with self-correlation versions. ResCLIP is orthogonally complementary to them.
- **MaskCLIP**: The earliest work applying CLIP to dense inference, directly using value features for segmentation.
- **ProxyCLIP**: Combines attention from vision foundation models like SAM, sharing a similar idea of fusing external information with ResCLIP.
- **Insight**: In foundation models, features/attentions from different layers may carry different types of valuable information, and fully utilizing inter-layer information is an important direction for improving dense predictions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The discovery of intermediate-layer attention is insightful, and the design of RCS is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablation with 8 datasets, 3 baseline methods, and 2 kinds of backbones.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, intuitive illustrations, and rigorous formulation.
- **Value**: ⭐⭐⭐⭐ — A practical, plug-and-play solution that significantly advances dense inference for CLIP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](../../ECCV2024/segmentation/sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [\[CVPR 2025\] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)

</div>

<!-- RELATED:END -->
