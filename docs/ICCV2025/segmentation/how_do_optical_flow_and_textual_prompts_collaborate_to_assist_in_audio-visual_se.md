---
title: >-
  [Paper Note] How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?
description: >-
  [ICCV 2025][Segmentation][Audio-Visual Semantic Segmentation] This paper proposes the SSP (Stepping Stone Plus) framework, which employs optical flow as auxiliary mask prompts in conjunction with two types of textual pro…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Audio-Visual Semantic Segmentation"
  - "Optical Flow"
  - "Textual Prompts"
  - "Cross-Modal Alignment"
  - "AVSS"
date: 2026-05-08
content_hash: f23f7adea12d9970
---

# How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?

**Conference**: ICCV 2025
**arXiv**: [2601.08133](https://arxiv.org/abs/2601.08133)  
**Code**: None  
**Area**: Audio-Visual Segmentation
**Keywords**: Audio-Visual Semantic Segmentation, Optical Flow, Textual Prompts, Cross-Modal Alignment, AVSS

## TL;DR

This paper proposes the SSP (Stepping Stone Plus) framework, which employs optical flow as auxiliary mask prompts in conjunction with two types of textual prompts and a Visual-Textual Alignment (VTA) module, achieving state-of-the-art performance on the audio-visual semantic segmentation task.

## Background & Motivation

Audio-Visual Semantic Segmentation (AVSS) requires models to identify sounding objects at the pixel level and assign semantic labels, extending the AVS task. Existing methods fall into two categories:

**Fusion-based methods**: These fuse audio and visual modalities into a unified representation, but may lose critical information from one modality if the model is poorly designed.

**Prompt-based methods**: These include three sub-categories based on object queries, masks, and textual prompts. Textual prompts are typically static macro-level descriptions that lack dynamic temporal information; moreover, using separate encoders for different modalities results in embeddings residing in isolated latent spaces.

The core insight of this paper is that **sounding objects are generally associated with motion**. Optical flow can therefore capture motion dynamics and provide valuable temporal context. For stationary sounding objects (e.g., alarm clocks), textual prompts serve as a complementary source of information.

## Method

### Overall Architecture

The SSP framework is built upon the AAVS baseline and decomposes the AVSS task into two stages: AVS followed by semantic segmentation. The framework introduces four key components:
- Pre-mask technique (optical-flow-assisted mask generation)
- Two types of textual prompts (scene description + sounding object identification)
- VTA Visual-Textual Alignment module
- Post-mask training objective

### Key Designs

1. **Pre-mask with Optical Flow**:

    - Perceiver IO is used to extract optical flow $O^{\mathcal{T}-1}$ between adjacent frames.
    - Flow deviation is smoothed by averaging adjacent frames: $O^{\mathcal{T}} = \text{Stack}\{O^1; \text{Mean}(O^t, O^{t+1}); O^{\mathcal{T}}\}$
    - The optical flow is converted into a binary mask $\mathcal{M}_O$, which is combined with the GT mask $\mathcal{M}_{GT}$ to produce a ternary mask $\mathcal{M}_{Pre}$: the intersection region is assigned 1 (definite foreground), the symmetric difference region is assigned 0.5 (uncertain), and the remainder is assigned 0 (background).
    - Design Motivation: The intersection with optical flow ensures partial segmentation accuracy, while uncertain regions are left for textual prompts to resolve.

2. **Textual Prompts**:

    - MiniCPM-o-2.6 (MLLM) is used to generate two types of text for each complete video:
        - **A₁ (Scene Description)**: Responds to the query "describe the location and characteristics of each object in the video," providing holistic semantic understanding.
        - **A₂ (Sounding Object Identification)**: Based on A₁, responds to "which nouns are likely to produce sound," localizing stationary sounding objects.
    - A₁ serves as the foundation for A₂, forming a hierarchical relationship that progressively refines semantic information.
    - Design Motivation: Addresses the inability of optical flow to detect stationary sounding objects (e.g., a piano); textual prompts compensate for the gray uncertain regions.

3. **Visual-Textual Alignment (VTA) Module**:

    - Built on BERT as the backbone to achieve cross-modal alignment between visual and textual representations.
    - Visual features are encoded with CLIP and text is tokenized with BLIP; attention masks are merged for unified processing.
    - BERT is invoked twice: the first pass fuses visual and textual features, and the second pass further refines the textual representation using the fused features.
    - The outputs Align₁ and Align₂ are added to the final features of the visual decoder and normalized.
    - Design Motivation: Avoids the modality isolation problem caused by using independent encoders.

### Loss & Training

The total loss consists of two components:

$$\mathcal{L} = \mathcal{L}_{AVS} + \lambda'_{mask} \cdot \mathcal{L}'_{mask}$$

- $\mathcal{L}_{AVS} = 5\mathcal{L}_{mask} + 5\mathcal{L}_{dice} + 2\mathcal{L}_{bce}$ (standard AVS loss)
- **Post-mask technique**: An additional $\mathcal{L}'_{mask}$ is introduced, with supervision label $\mathcal{M}_{Post} = \mathcal{M}_O \cap \mathcal{M}_{GT}$ (intersection of optical flow and GT masks), and $\lambda'_{mask}=10$ (higher weight for stronger penalization).
- Training settings: S4/MS3 trained for 30 epochs, AVSS for 60 epochs, batch size=2, learning rate decayed from 1e-3 to 1e-4.

## Key Experimental Results

### Main Results

| Method | Type | S4 mIoU | S4 F | MS3 mIoU | MS3 F | AVSS mIoU | AVSS F |
|--------|------|---------|------|----------|-------|-----------|--------|
| AVSBench | Fusion | 78.7 | 87.9 | 54.0 | 64.5 | 29.8 | 35.2 |
| AVSegFormer | Fusion | 82.1 | 89.9 | 58.4 | 69.3 | 36.7 | 42.0 |
| AAVS | Prompt | 83.2 | 91.3 | 67.3 | 77.6 | 48.5 | 53.2 |
| COMBO | Prompt | 84.7 | 91.9 | 59.2 | 71.2 | 42.1 | 46.1 |
| AVS-Mamba | Prompt | 85.0 | 92.6 | 68.6 | 78.8 | 39.7 | 45.1 |
| TeSO | Prompt | 83.2 | 93.3 | 66.0 | 80.1 | 38.9 | 45.1 |
| **SSP (Ours)** | **Prompt** | **85.4** | **93.3** | **72.3** | **84.6** | **50.1** | **54.5** |

Compared to the AAVS baseline, SSP achieves: S4 +2.2%/+1.9%, MS3 +5.0%/+7.0%, AVSS +1.6%/+1.3%.

### Ablation Study

| Configuration | S4 mIoU | MS3 mIoU | AVSS mIoU |
|---------------|---------|----------|-----------|
| AAVS baseline | 83.2 | 67.3 | 48.5 |
| + Pre-mask | 84.1 | 69.5 | 49.2 |
| + Pre-mask + Post-mask | 85.0 | 70.2 | 49.4 |
| + Textual prompts (w/o VTA) | 83.7 | 68.1 | 48.6 |
| + Textual prompts + VTA | 84.3 | 69.8 | 49.0 |
| w/o Post-mask | 84.5 | 70.4 | 49.7 |
| **Full model** | **85.4** | **72.3** | **50.1** |

### Key Findings

- Optical flow pre-mask alone yields a substantial improvement of +2.2% mIoU on MS3, demonstrating the effectiveness of optical flow as a dynamic prompt.
- The VTA module outperforms simple cross-attention by an average of 1.1% mIoU, validating the necessity of the alignment module.
- The quality of textual prompts significantly affects performance, with a gap of 2.3% mIoU between the best and worst configurations.
- Performance is optimal at $\lambda'_{mask}=10$; values either higher or lower degrade results.

## Highlights & Insights

- **Pioneer use of optical flow as an AVS prompt**: This is the first work to introduce optical flow into audio-visual segmentation, leveraging the prior that sounding objects are typically in motion.
- **Complementary dual-prompt design**: Optical flow captures dynamic objects while textual prompts compensate for stationary sounding objects, achieving comprehensive coverage.
- **Elegant ternary mask design**: Encoding certainty as 0/0.5/1 carries richer information than a simple binary mask.
- **Dual BERT invocation in VTA**: The first pass performs cross-modal fusion, and the second progressively refines the textual representation, enhancing alignment quality in a coarse-to-fine manner.

## Limitations & Future Work

- Optical flow extraction relies on Perceiver IO, introducing additional computational overhead.
- Textual prompts depend on an external MLLM (MiniCPM-o-2.6), requiring an additional model at inference time.
- GT masks are used in pre-mask construction during training but are unavailable at inference, creating a train-test discrepancy.
- The quality of textual prompts is bounded by MLLM capability, and prompt engineering requires careful tuning.

## Related Work & Insights

- The two-stage paradigm of AAVS (AVS + SS) provides an effective task decomposition strategy.
- The use of optical flow in video understanding can be generalized to other audio-visual collaborative tasks.
- The approach of using BERT as a cross-modal alignment backbone in VTA is applicable to other multimodal fusion scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing optical flow as an auxiliary AVS prompt is a genuine innovation; the complementary dual-prompt design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on three datasets with comprehensive ablation studies, textual quality analysis, and visualizations.
- **Writing Quality**: ⭐⭐⭐ Structure is clear, though some notation definitions are redundant.
- **Value**: ⭐⭐⭐⭐ Provides an effective multimodal collaborative framework for the AVSS task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICCV 2025\] Refer to Any Segmentation Mask Group With Vision-Language Prompts](refer_to_any_segmentation_mask_group_with_vision-language_prompts.md)
- [\[ICML 2026\] LightAVSeg: Lightweight Audio-Visual Segmentation](../../ICML2026/segmentation/lightavseg_lightweight_audio-visual_segmentation.md)

</div>

<!-- RELATED:END -->
