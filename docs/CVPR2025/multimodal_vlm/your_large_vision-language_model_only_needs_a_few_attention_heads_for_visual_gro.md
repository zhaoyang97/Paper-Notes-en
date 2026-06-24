---
title: >-
  [Paper Note] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding
description: >-
  [CVPR 2025][Multimodal VLM][Visual Grounding] It is discovered that frozen LVLMs naturally contain a small number of "localization heads" that consistently capture object locations corresponding to textual semantics. Using the attention maps of only 3 attention heads, training-free visual grounding achieves 86.5% on RefCOCO val, outperforming the fine-tuned LISA-7B.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Visual Grounding"
  - "Attention Head Discovery"
  - "Training-free Method"
  - "Localization Heads"
  - "LVLM Interpretability"
date: 2026-05-08
content_hash: a0b3275a07d3a502
---

# Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding

**Conference**: CVPR 2025  
**arXiv**: [2503.06287](https://arxiv.org/abs/2503.06287)  
**Code**: None (Project page to be confirmed)  
**Area**: Multimodal VLM  
**Keywords**: Visual Grounding, Attention Head Discovery, Training-free Method, Localization Heads, LVLM Interpretability

## TL;DR
It is discovered that frozen LVLMs naturally contain a small number of "localization heads" that consistently capture object locations corresponding to textual semantics. Using the attention maps of only 3 attention heads, training-free visual grounding achieves 86.5% on RefCOCO val, outperforming the fine-tuned LISA-7B.

## Background & Motivation

**Background**: Visual grounding (VG) requires models to locate objects in an image corresponding to text descriptions. Existing methods either train specialized grounding heads on large amounts of annotated data (e.g., LISA, CogVLM) or use external detectors combined with CLIP matching.

**Limitations of Prior Work**: Training-based grounding methods require extensive bounding box annotations and specialized fine-tuning. Existing training-free methods (e.g., ReCLIP, GroundVLP) rely on external modules and have limited accuracy. The key question is: Do LVLMs already possess internal spatial grounding capabilities?

**Key Challenge**: While LVLMs are expected to understand spatial relationships after large-scale image-text training, their average attention maps are usually blurry and struggle to yield localization information.

**Goal**: Explore the internal spatial grounding mechanism of LVLMs—specifically, whether there are specific attention heads that naturally capture object locations.

**Key Insight**: Instead of averaging across all attention heads, they are analyzed individually. "Localization heads" are filtered using two metrics: attention sum (the sum of attention weights from text tokens to image tokens) and spatial entropy (the centralization of the attention map).

**Core Idea**: Among thousands of attention heads in a frozen LVLM, the attention maps of only 3 specific heads are sufficient for high-precision visual grounding, without any training or external modules.

## Method

### Overall Architecture
Input image-text pair → LVLM forward inference → Individual analysis of each attention head → Two-stage filtering (attention sum threshold + spatial entropy ranking) → Selection of top-$k$ low spatial entropy heads → Assembly of their attention maps → Prediction of bounding box / mask.

### Key Designs

1. **Attention Sum Filtering**:

    - **Function**: Filter out attention heads that do not focus on image tokens.
    - **Mechanism**: Calculate the sum of attention weights, $S_{img}$, from text tokens to image tokens for each head. Set a threshold $\tau$ at the maximum curvature; heads with $S_{img} < \tau$ are excluded.
    - **Design Motivation**: Many attention heads primarily focus on relations between text tokens, which are useless for localization.

2. **Spatial Entropy Filtering**:

    - **Function**: Identify the heads with the most concentrated (most localized) attention distribution.
    - **Mechanism**: Binarize the attention map → Connected component analysis → Calculate entropy $H$. Low entropy indicates concentration in a few regions (precise localization), while high entropy indicates dispersion (blurriness). The frequency of each head falling into the top-10 lowest entropy is compiled across 1000 samples.
    - **Design Motivation**: Localization heads should exhibit spatial concentration across different samples; ranking by selection frequency ensures consistency.

3. **Stability of Selection Frequency**:

    - **Function**: Identify localization heads that are consistent across samples.
    - **Mechanism**: Spearman correlation coefficient > 0.7 between selection frequency ranking and IoU demonstrates that heads with higher selection frequencies are indeed better at localization. Only 3 heads are sufficient (with diminishing marginal returns for more heads).
    - **Design Motivation**: Prove that localization heads are intrinsic properties of LVLMs rather than accidental anomalies on specific samples.

### Loss & Training
Completely training-free—solely utilizing the attention maps of the frozen LVLM.

## Key Experimental Results

### Main Results

| Method | Type | RefCOCO val | testA | testB | RefCOCO+ val |
|------|------|-----------|-------|-------|-------------|
| ReCLIP | Training-free | 45.8 | 46.1 | 47.1 | 47.9 |
| GroundVLP | Training-free | 65.0 | 73.5 | 55.0 | 68.8 |
| LISA-7B | Fine-tuned | 74.1 | 76.5 | 71.1 | 62.4 |
| **Ours (LLaVA-7B)** | **Training-free** | **86.5** | **89.8** | **80.2** | **80.1** |
| **Ours (LLaVA-13B)** | **Training-free** | **87.2** | **90.0** | **83.3** | **82.7** |

### Ablation Study

| Configuration | Performance |
|------|------|
| Averaging all attention heads | Extremely poor (blurry) |
| Top-1 localization head | Insufficient stability |
| Top-3 localization heads | **Optimal** |
| Top-10 localization heads | Slight decline |

### Key Findings
- **Training-free beats fine-tuning**: Using only 3 attention heads outperforms LISA-7B on RefCOCO (86.5 vs 74.1), despite the latter being fine-tuned on extensive grounding data.
- **Universal existence across architectures**: Localization heads are consistently found across 10+ different LVLMs (including LLaVA, InternVL, Qwen-VL, etc.).
- **Average attention is completely useless**: Averaging thousands of heads generates blurry, noisy maps. Only a few specific heads generate sharp and accurate attention maps.

## Highlights & Insights
- **The discovery of "LVLMs are born to localize"** is profoundly important—it indicates that large-scale image-text training has already encoded spatial understanding within attention mechanisms, eliminating the need for specialized grounding training.
- **Only 3 heads are sufficient**: Distilling down to just 3 heads from thousands shows remarkable efficiency in information utilization. This suggests that localization heads could be utilized for highly efficient attention pruning.

## Limitations & Future Work
- Filtering localization heads requires a small validation set (1000 samples) and is not entirely data-free.
- It might be insufficient for complex reasoning-based grounding (e.g., "the larger object on the left"), which requires deeper semantic comprehension.
- The evaluation is limited to the REC task; extending to RES (referring expression segmentation) requires an additional mask generation pipeline.

## Related Work & Insights
- **vs LISA**: LISA appends a decoder head and fine-tunes the LVLM. This work proves that fine-tuning is unnecessary—the attention of the frozen model is sufficient.
- **vs CLIP Heatmap Methods**: CLIP's global contrastive features are unsuited for precise localization. Autoregressive attention in LVLMs naturally encodes spatial relationships.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of "localization heads" provides significant insights into the internal mechanisms of LVLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 10+ LVLMs, 3 REC datasets, and complete with extensive analyses of filtering metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical progression from hypothesis to discovery to verification is extremely clear.
- Value: ⭐⭐⭐⭐⭐ Makes vital contributions to VLM interpretability and efficient grounding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)

</div>

<!-- RELATED:END -->
