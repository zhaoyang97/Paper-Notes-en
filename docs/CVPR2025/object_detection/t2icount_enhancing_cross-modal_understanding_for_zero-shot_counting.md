---
title: >-
  [Paper Note] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting
description: >-
  [CVPR 2025][Object Detection][Zero-Shot Object Counting] This paper proposes T2ICount, which leverages one-step denoising features from pre-trained text-to-image diffusion models for zero-shot object counting. It addresses the lack of text sensitivity in one-step denoising through a Hierarchical Semantic Correction Module (HSCM) and a Representational Regional Coherence loss ($\mathcal{L}_{RRC}$).
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Zero-Shot Object Counting"
  - "Diffusion Models"
  - "Text Sensitivity"
  - "Cross-Modal Alignment"
  - "Density Estimation"
date: 2026-05-08
content_hash: efcb6e3dad7f43e8
---

# T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting

**Conference**: CVPR 2025  
**arXiv**: [2502.20625](https://arxiv.org/abs/2502.20625)  
**Code**: [https://github.com/cha15yq/T2ICount](https://github.com/cha15yq/T2ICount)  
**Area**: Image Generation/Zero-Shot Counting  
**Keywords**: Zero-Shot Object Counting, Diffusion Models, Text Sensitivity, Cross-Modal Alignment, Density Estimation

## TL;DR

This paper proposes T2ICount, which leverages one-step denoising features from pre-trained text-to-image diffusion models for zero-shot object counting. It addresses the lack of text sensitivity in one-step denoising through a Hierarchical Semantic Correction Module (HSCM) and a Representational Regional Coherence loss ($\mathcal{L}_{RRC}$).

## Background & Motivation

Zero-shot object counting aims to count objects of arbitrary categories in an image based on text descriptions without visual exemplars. Existing methods mainly rely on CLIP vision-language models but suffer from fundamental issues:

1. **Text Insensitivity**: The CLIP image encoder operates at a global semantic level and naturally tends to focus on the majority class of objects in the image. When the text specifies a minority class, the model fails to respond correctly.
2. **Dataset Bias**: In benchmark datasets like FSC-147, the annotated categories are almost always the majority classes in the images, hiding the models' text insensitivity issue.
3. **Potential of Diffusion Models**: Text-to-image diffusion models possess rich pixel-level semantic understanding capabilities and are naturally suited for pixel-level tasks like counting. However, the multi-step denoising process is computationally prohibitive.

However, when utilizing one-step denoising to improve efficiency, the text-visual correspondence is insufficiently established, and cross-attention maps exhibit severe semantic misalignment—irrelevant regions are highlighted, and attention on relevant objects is inconsistent.

## Method

### Overall Architecture

Based on Stable Diffusion, T2ICount takes an image and text prompts as input, performs one-step denoising, extracts multi-scale feature maps from the U-Net decoder, refines the text-image alignment layer-by-layer using HSCM, and finally generates a density map for counting.

### Key Designs

**Design 1: Hierarchical Semantic Correction Module (HSCM)**

- **Function**: Compensate for the insufficient text-image interaction caused by one-step denoising.
- **Mechanism**: A three-stage cascaded design. In each stage, adjacent scale features are first fused as $F_i' = \text{Conv}(\text{Concat}(\text{Up}(V_{i+1}), F_i))$, and then alternately refined through SEM (bidirectional cross-modal attention + text-image similarity computation) and SCM (guiding feature correction using the similarity map from the previous stage).
- **Design Motivation**: The cross-modal alignment of one-step denoising is too weak, requiring additional multi-stage correction. SEM learns to generate a text-image similarity map $S_i$ similar to a segmentation mask, while SCM redirects attention to text-relevant regions using $V_{i+1} \odot S_{i+1}$.

$$S_i = \frac{V_i \cdot c'}{\|V_i\| \|c'\|}$$

**Design 2: Representational Regional Coherence Loss ($\mathcal{L}_{RRC}$)**

- **Function**: Leverage the cross-attention maps of the diffusion model to generate reliable supervisory signals for positive and negative samples.
- **Mechanism**: Although one-step attention maps are insensitive to specific categories, they can effectively capture the overall foreground region. Multi-scale cross-attention maps $\mathcal{A}^{cross}$ are fused and combined with the ground-truth density map to generate a Positive-Negative-Ambiguous (PNA) ternary mask: high-density areas are positive, low-attention areas are negative, and the rest are ambiguous (where no constraint is applied).
- **Design Motivation**: Counting datasets only provide point annotations, and traditional methods distinguish between foreground and background using density thresholds, which often misclassify substantial foreground regions as background. The PNA mask identifies the background using attention maps, preventing misclassification of the foreground.

$$p_{jk} = \begin{cases} 1, & \text{if } D_{jk}^{gt} \geq \tau \\ 0, & \text{if } \mathcal{A}_{jk}^{cross} \leq \theta \\ -1, & \text{otherwise} \end{cases}$$

**Design 3: FSC-147-S Evaluation Benchmark**

- **Function**: Provide a more rigorous evaluation protocol for text-guided counting.
- **Mechanism**: Select and re-annotate images from FSC-147 so that the text-specified category differs from the majority class, specifically testing the model's ability to count minority classes.
- **Design Motivation**: In existing benchmarks, the annotated class is almost always the majority class, allowing models to score well even if they completely ignore the text, which fails to truly evaluate text sensitivity.

### Loss & Training

The overall loss consists of the regression loss and the regional coherence loss:

$$\mathcal{L} = \mathcal{L}_{reg} + \gamma \mathcal{L}_{RRC}$$

where $\mathcal{L}_{RRC} = \lambda \mathcal{L}_{pos} + \mathcal{L}_{neg}$. The positive sample loss pulls the similarity close to 1, while the negative sample loss uses a hinge loss to push it below 0.

## Key Experimental Results

### FSC-147 Test Set

| Method | Type | MAE↓ | RMSE↓ |
|------|------|------|-------|
| FamNet (3-shot) | Few-shot | 22.56 | 101.54 |
| BMNet (3-shot) | Few-shot | 14.62 | 91.83 |
| CLIP-Count | Zero-shot | 17.78 | 112.09 |
| VLCounter | Zero-shot | 17.05 | 106.16 |
| **T2ICount** | **Zero-shot** | **15.89** | **94.32** |

### FSC-147-S (Text Sensitivity Evaluation)

| Method | MAE↓ | RMSE↓ |
|------|------|-------|
| CLIP-Count | 32.41 | 48.75 |
| VLCounter | 29.83 | 45.12 |
| **T2ICount** | **18.56** | **31.27** |

### Ablation Study

| Component | Val MAE | Test MAE |
|------|---------|----------|
| Baseline (One-step U-Net features) | 22.15 | 21.35 |
| + HSCM | 18.42 | 17.89 |
| + $\mathcal{L}_{RRC}$ | 16.73 | 16.41 |
| + HSCM + $\mathcal{L}_{RRC}$ (Full) | **15.56** | **15.89** |

### Key Findings

1. As a zero-shot method, T2ICount achieves performance close to or even surpassing some few-shot methods (e.g., FamNet 3-shot).
2. On FSC-147-S, T2ICount achieves an approximately 43% reduction in MAE compared to CLIP-Count, demonstrating a significant advantage in text sensitivity.
3. HSCM and $\mathcal{L}_{RRC}$ each contribute to an improvement of about 3-4 in MAE, showing mutual complementarity.
4. Although the cross-attention maps are insensitive to specific classes, they function well as foreground detectors—a highly novel observation.

## Highlights & Insights

1. **Identifying and Resolving Text Insensitivity**: This work identifies a fundamental and overlooked issue in the field of zero-shot counting and addresses it through both a new method and a new evaluation protocol.
2. **Creative Utilization of Diffusion Model Attention Maps**: Although the one-step attention maps exhibit poor class sensitivity, employing them as a foreground detector to generate supervision signals is highly ingenious.
3. **PNA Ternary Mask Design**: This design elegantly addresses the challenge of partitioning positive and negative samples under point-annotation scenarios, avoiding the misclassification of foreground as background.

## Limitations & Future Work

1. It relies on the pre-trained Stable Diffusion model, resulting in a large model size and inference speeds that remain limited by the U-Net forward pass.
2. The information from one-step denoising is limited; multi-step denoising combined with distillation could potentially further enhance performance.
3. The scale of FSC-147-S is limited, highlighting the need for a larger-scale evaluation benchmark for text sensitivity.
4. Whether newer diffusion models (such as SDXL or SD3) can provide superior features remains unexplored.

## Related Work & Insights

- **CLIP-Count/VLCounter**: Pioneers of CLIP-based zero-shot counting; this paper identifies their fundamental limitation regarding text insensitivity.
- **CounTR/LOCA**: Few-shot counting methods; T2ICount achieves impressive performance close to theirs under a zero-shot setting.
- **DiffusionDet**: An exploration of diffusion models for detection, inspiring the application of diffusion features to counting tasks.
- **Insight**: The intermediate features of diffusion models have demonstrated powerful capabilities across various vision tasks, warranting further exploration in different scenarios.

## Rating

⭐⭐⭐⭐ — Well-defined problem formulation (text insensitivity), creative solutions (PNA mask + HSCM), and comprehensive experimental design (the proposed FSC-147-S). This work makes a substantial contribution as a representative study introducing diffusion models to zero-shot counting. The main drawbacks lie in model efficiency and the limited scale of FSC-147-S.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[CVPR 2026\] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](../../CVPR2026/object_detection/boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)
- [\[CVPR 2025\] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](towards_zero-shot_anomaly_detection_and_reasoning_with_multimodal_large_language.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](../../NeurIPS2025/object_detection/detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](../../ICLR2026/object_detection/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)

</div>

<!-- RELATED:END -->
