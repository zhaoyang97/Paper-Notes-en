---
title: >-
  [Paper Note] EAFormer: Scene Text Segmentation with Edge-Aware Transformers
description: >-
  [ECCV 2024][Segmentation][Scene Text Segmentation] Proposes Edge-Aware Transformer (EAFormer), which filters non-text region edges using a Text Edge Extractor, and fuses text edge information in the encoder using symmetric cross-attention. This significantly improves the segmentation accuracy of text edge regions. Furthermore, the COCO_TS and MLT_S datasets are re-annotated for fairer evaluation.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Scene Text Segmentation"
  - "Edge-Aware"
  - "Transformer"
  - "Symmetric Cross-Attention"
  - "Dataset Re-annotation"
date: 2026-05-08
content_hash: fc13c7feeedb5ad1
---

# EAFormer: Scene Text Segmentation with Edge-Aware Transformers

**Conference**: ECCV 2024  
**arXiv**: [2407.17020](https://arxiv.org/abs/2407.17020)  
**Code**: [https://hyangyu.github.io/EAFormer/](https://hyangyu.github.io/EAFormer/)  
**Area**: Image Segmentation  
**Keywords**: Scene Text Segmentation, Edge-Aware, Transformer, Symmetric Cross-Attention, Dataset Re-annotation

## TL;DR
Proposes Edge-Aware Transformer (EAFormer), which filters non-text region edges using a Text Edge Extractor, and fuses text edge information in the encoder using symmetric cross-attention. This significantly improves the segmentation accuracy of text edge regions. Furthermore, the COCO_TS and MLT_S datasets are re-annotated for fairer evaluation.

## Background & Motivation

**Background**: Scene text segmentation aims to distinguish foreground text from the background at the pixel level, and is widely applied to downstream tasks such as text removal, document analysis, and scene understanding. In recent years, deep learning-based methods have emerged continuously, such as TexRNet introducing character-level supervision and TextFormer adding a recognition head, leading to continuous performance improvements.

**Limitations of Prior Work**:

**Neglecting text edges**: Existing methods improve overall segmentation accuracy but ignore the accuracy of text edge regions. Imprecise text masks lead to residual or over-erased pixels in downstream tasks like text removal.

**Difficulty in introducing edge information**: Traditional edge detection (Canny) can precisely capture edges but cannot distinguish between text and non-text regions. Using the global edge map directly introduces substantial noise.

**Coarse annotations in evaluation datasets**: Datasets like COCO_TS and MLT_S rely on box-generated annotations, suffering from poor quality (missing annotations, imprecise edges), which hinders fair evaluation of methods.

**Key Challenge**: How to effectively leverage precise edge information from traditional edge detection to enhance text segmentation while avoiding negative interference from non-text edges?

**Key Insight**: Designing a two-stage "filter-then-fuse" strategy: first filtering edges of non-text regions using a lightweight text detector, then integrating the filtered text edges into the early stages of the encoder via symmetric cross-attention.

**Core Idea**: Accurate segmentation of text edges can be achieved through a three-step pipeline: "text edge extraction $\to$ edge filtering $\to$ edge-guided encoding", leveraging Canny's strong edge detection, a lightweight detector's regional filtering, and symmetric cross-attention for feature fusion.

## Method

### Overall Architecture
EAFormer consists of three modules:
1. **Text Edge Extractor**: Extracts and filters text edges.
2. **Edge-Guided Encoder**: A SegFormer-based 4-stage encoder that integrates edge guidance in the first stage.
3. **Text Segmentation Decoder**: An MLP-based decoder that fuses multi-scale features to predict text masks.

Given an input scene image $\mathbf{X} \in \mathbb{R}^{3 \times H \times W}$, it outputs the text mask $\mathbf{M}_t$.

### Key Designs

1. **Text Edge Extractor**:

    - Extracts global edges $\mathbf{E}_w$ using the Canny algorithm (thresholds 100/200).
    - Extracts multi-scale features $\{\mathbf{F}_1^d, ..., \mathbf{F}_4^d\}$ using a lightweight ResNet backbone, and predicts the text region mask via a $1\times 1$ convolution:
    $\mathbf{M}_a = \text{Conv}_{1\times 1}(\text{Concat}(\{\mathbf{F}_1^d, \mathbf{F}_2^d, \mathbf{F}_3^d, \mathbf{F}_4^d\}))$
    - Filters non-text edges via element-wise multiplication: $\mathbf{E}_t = \mathbf{M}_a \odot \text{SoftArgmax}(\mathbf{E}_w)$
    - Employs SoftArgmax to enable joint end-to-end optimization of both the text detection and segmentation branches.

2. **Edge-Guided Encoder**:

    - Based on the SegFormer 4-stage hierarchical Transformer encoder.
    - Introduces **Symmetric Cross-Attention** after the **first stage**:
        - Edge as Query, visual features as Key/Value $\to$ extracts edge-aware visual information $\mathbf{F}^{ev}$.
        - Visual features as Query, edge as Key/Value $\to$ extracts text edge information $\mathbf{F}^{te}$.
    $\hat{\mathbf{F}}_1^s = \mathbf{F}^{ev} \oplus \mathbf{F}^{te} \oplus \mathbf{F}_1^s$
    - Design motivation for **fusing edges only in the first stage**: K-Means clustering visualization shows that only early-stage features attend to edge information, while high-level features discard edge details. Experiments also verify that introducing edge information in higher stages degrades performance.

3. **MLP Text Segmentation Decoder**:

    - Multi-stage features are unified in channel dimensions via MLPs and upsampled to the same resolution.
    - Features are concatenated and fused via another MLP to predict the binary text mask.

### Loss & Training
Employs only two cross-entropy losses to avoid complex hyperparameter tuning:
$$\mathcal{L} = \underbrace{\text{CE}(\mathbf{M}_t, \hat{\mathbf{M}}_t)}_{\mathcal{L}_{seg}} + \lambda \underbrace{\text{CE}(\mathbf{M}_a, \hat{\mathbf{M}}_a)}_{\mathcal{L}_{det}}$$
$\lambda = 1.0$; bounding-box level supervision for text detection can be directly derived from semantic annotations, requiring no additional annotation. Optimized with AdamW, lr=$6\times10^{-5}$, batch size 4 on 8x RTX 4090 GPUs.

## Key Experimental Results

### Main Results

**English Text Segmentation Benchmarks**:

| Method | TextSeg fgIoU | TextSeg F-score | COCO_TS fgIoU | MLT_S fgIoU | BTS fgIoU |
|------|--------------|-----------------|---------------|-------------|-----------|
| SegFormer (Baseline) | 84.59 | 0.916 | 63.17 | 78.77 | 84.99 |
| TextFormer | 87.42 | 0.933 | 73.20 | 86.66 | 86.97 |
| TFT | 87.11 | 0.931 | 73.40 | 87.80 | 87.84 |
| **EAFormer (Ours)** | **88.06** | **0.939** | **81.03** | **89.02** | **88.08** |

Outperforms TFT by 7.63% fgIoU on COCO_TS, and surpasses TextFormer by 0.64% fgIoU on TextSeg.

**Results on Re-annotated Datasets**:

| Method | COCO_TS (Re-annotated) fgIoU | COCO_TS F-score | MLT_S (Re-annotated) fgIoU | MLT_S F-score |
|------|------------------------|-----------------|---------------------|---------------|
| TextFormer | 52.73 | 0.688 | 74.83 | 0.861 |
| **EAFormer** | **64.82** | **0.786** | **81.92** | **0.900** |

When trained and tested using more precise annotations, the advantage of EAFormer is more pronounced (+12.09% on COCO_TS).

### Ablation Study

**Edge Filtering and Edge Guidance**:

| Edge Filtering (EF) | Edge Guidance (EG) | TextSeg fgIoU | BTS fgIoU |
|------|------|-------------|-----------|
| ✗ | ✗ | 84.59 | 84.99 |
| ✓ | ✗ | 86.85 | 87.35 |
| ✗ | ✓ | 81.03 | 80.35 |
| ✓ | ✓ | **88.06** | **88.08** |

**Hyperparameter $\lambda$ Selection**:

| $\lambda$ | TextSeg fgIoU | TextSeg F-score |
|---|--------------|-----------------|
| 0.1 | 84.03 | 0.910 |
| 0.5 | 87.33 | 0.926 |
| 1.0 | **88.06** | **0.939** |
| 5.0 | 87.67 | 0.934 |
| 10.0 | 87.94 | 0.937 |

### Key Findings
- **Using edge guidance without filtering degrades performance** (TextSeg drops from 84.59 to 81.03), since the edges of non-text regions introduce severe interference.
- Edges should be integrated in the **first stage**; incorporating them in the third or fourth stages yields performance even lower than the baseline.
- Replacing the lightweight detector with a pre-trained text detector (DBNet) achieves 90.16% / 95.2% fgIoU / F-score on TextSeg.
- Controllable parameter increase: parameters grow from 85M (TextFormer) to 92M, while inference time changes from 0.42s to 0.47s/image.

## Highlights & Insights
- The **"filter-then-guide" strategy** is key to introducing edge information in text segmentation; using raw Canny edges directly is counterproductive.
- Bidirectional information exchange of **Symmetric Cross-Attention** is more effective than unidirectional fusion, allowing visual features to perceive edges while enabling edge features to capture visual context.
- **Re-annotated datasets** are a major contribution: the original annotations of COCO_TS and MLT_S are of poor quality; re-annotation enables a fairer evaluation of methods.
- Integrating edges in low-level features aligns with the visual perception hierarchy: low levels focus on texture/edges, while high levels capture semantics.

## Limitations & Future Work
- Introducing a lightweight text detector increases the parameter count (+7M).
- Only Canny edge detection is used; employing deep learning-based edge detection methods (e.g., HED/BDCN) might yield further improvements.
- Edge detection remains challenging for blurred text, especially in low-resolution scenes.
- Future work can explore extending edge guidance to instance-level text segmentation.

## Related Work & Insights
- Hierarchical Transformer design of SegFormer $\to$ provides a natural integration point for edge guidance.
- Edge-guided segmentation methods like BCANet/BSNet require edge annotations $\to$ EAFormer eliminates this requirement by leveraging Canny edges.
- Differentiable binarization in DBNet $\to$ inspires the design of the text detection branch.
- Insight: The design of symmetric cross-attention can be generalized to other segmentation tasks requiring auxiliary guidance (e.g., contour guidance in medical images).

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of edge filtering and symmetric cross-attention is novel and addresses a practical problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive testing on 6 datasets, thorough ablation studies, and re-annotated datasets enhance the credibility of the evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, and the visualization analyses (K-Means clustering, multi-stage features) are convincing.
- Value: ⭐⭐⭐⭐ The re-annotated dataset contribution holds lasting value, and the method is highly practical for the text segmentation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Detection: A Structure-Aware Framework for Scene Text Tracking](../../ICML2026/segmentation/beyond_detection_a_structure-aware_framework_for_scene_text_tracking.md)
- [\[ECCV 2024\] Occlusion-Aware Seamless Segmentation](occlusion-aware_seamless_segmentation.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)
- [\[ECCV 2024\] OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](../../ICCV2025/segmentation/enhancing_transformers_through_conditioned_embedded_tokens.md)

</div>

<!-- RELATED:END -->
