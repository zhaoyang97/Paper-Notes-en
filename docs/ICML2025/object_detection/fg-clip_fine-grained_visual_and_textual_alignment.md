---
title: >-
  [Paper Note] FG-CLIP: Fine-Grained Visual and Textual Alignment
description: >-
  [ICML 2025][Object Detection][CLIP] FG-CLIP systematically addresses the three major bottlenecks of fine-grained understanding in CLIP: capturing global semantic details with 1.6B long-description-image pairs, achieving precise regional alignment with 12M images and 40M region annotations, and training models to distinguish subtle semantic differences with 10M hard negatives, achieving comprehensive leading performance in fine-grained understanding, open-vocabulary detection…
tags:
  - "ICML 2025"
  - "Object Detection"
  - "CLIP"
  - "Fine-Grained Understanding"
  - "Long Captions"
  - "Regional Alignment"
  - "Hard Negatives"
  - "Open-Vocabulary Detection"
date: 2026-05-08
content_hash: 35563e1705d61d86
---

# FG-CLIP: Fine-Grained Visual and Textual Alignment

**Conference**: ICML 2025  
**arXiv**: [2505.05071](https://arxiv.org/abs/2505.05071)  
**Code**: [https://github.com/360CVGroup/FG-CLIP](https://github.com/360CVGroup/FG-CLIP)  
**Area**: Object Detection  
**Keywords**: CLIP, Fine-Grained Understanding, Long Captions, Regional Alignment, Hard Negatives, Open-Vocabulary Detection

## TL;DR

FG-CLIP systematically addresses the three major bottlenecks of fine-grained understanding in CLIP: capturing global semantic details with 1.6B long-description-image pairs, achieving precise regional alignment with 12M images and 40M region annotations, and training models to distinguish subtle semantic differences with 10M hard negatives, achieving comprehensive leading performance in fine-grained understanding, open-vocabulary detection, and image-text retrieval.

## Background & Motivation

**Background**: CLIP, trained on large-scale image-text pairs via contrastive learning, has achieved immense success in downstream tasks like zero-shot classification and image-text retrieval. However, CLIP's fine-grained understanding is severely limited—it can distinguish "birds" from "cars", but struggles to differentiate a "red-winged blackbird" from a "blue tit".

**Limitations of Prior Work**: CLIP's fine-grained bottleneck stems from three levels: (1) the text encoder is limited to 77 tokens, preventing it from handling detailed descriptions; (2) image-text alignment occurs at the global image level, making it unable to extract region-specific representations—preventing individual alignment when "there is a red sports car and a blue sedan in the image"; (3) the training data is predominated by positive examples with a lack of hard negatives, making the model unable to distinguish "red sports car" vs "blue sports car". Existing improvements (such as LongCLIP for extending text length, RegionCLIP for introducing region data, and FineCLIP for self-distillation) address parts of these problems but fail to offer a systematic integration.

**Key Challenge**: Fine-grained understanding requires the simultaneous execution of three capabilities: understanding long text, aligning local regions, and distinguishing subtle differences. Existing methods cover only one or two of these.

**Goal**: Concurrently overcome all three fine-grained bottlenecks of CLIP through a comprehensive upgrade of data scale, data quality, and training strategies.

**Key Insight**: A three-pronged approach—1.6B long captions (global semantics) + 40M region annotations (local alignment) + 10M hard negatives (discriminative power).

**Core Idea**: Data-driven systematic enhancement—the FineHARD dataset + two-stage training = fine-grained CLIP.

## Method

### Overall Architecture

FG-CLIP employs a two-stage training paradigm:
- **Stage 1**: Perform global contrastive learning on 1.6B long caption-image pairs to build the foundation for fine-grained textual understanding.
- **Stage 2**: Introduce regional contrastive learning and hard negative learning on top of global contrastive learning, utilizing the FineHARD dataset for fine-grained alignment.

Text position embedding expansion: Keep original position embeddings for $\le 20$ tokens; for $>20$ tokens, linearly interpolate by a factor of 4, increasing the maximum length from 77 to 248 tokens.

### Key Designs

1. **1.6B Long Caption Data Construction**: CogVLM2-19B is used to recaption LAION-2B, generating 1.6B long caption-image pairs. For example, "a bird" becomes "a red-winged blackbird perched on a tree branch in a park". This was completed in a production environment with 160×910B NPUs over 30 days. The scale is 1600 times that of LongCLIP (1M) and 640 times that of FineCLIP (2.5M). **Design Motivation**: Large-scale detailed descriptions allow the model to understand fine-grained semantics in global scenes.

2. **FineHARD Dataset**: Based on GRIT images, CogVLM2-19B generates captions $\to$ SpaCy parses referring expressions $\to$ Yolo-World localizes bboxes (confidence $>0.4$, NMS deduplication) $\to$ yielding 12M images with 40M region annotations. Then, Llama-3.1-70B is leveraged to generate 10 hard negative captions for each positive sample (only altering attribute words while keeping the object name intact). Manual inspection of 3,000 samples showed a 98.9% qualification rate. **Design Motivation**: (a) Detailed regional descriptions instead of category labels offer richer semantics; (b) Hard negatives force the model to learn attribute-level distinctions.

3. **Three-in-One Training Loss**: $L = L_{\text{global}} + \alpha L_{\text{regional}} + \beta L_{\text{hard}}$ ($\alpha=0.1, \beta=0.5$). $L_{\text{global}}$ lies in InfoNCE global contrastive learning, where each image is paired with both short and long captions simultaneously. $L_{\text{regional}}$ performs contrast against $K$ region-text pairs (using RoIAlign to extract region features). $L_{\text{hard}}$ is a unidirectional classification loss over $M$ descriptions (1 positive + $M-1$ negatives) per region. **Design Motivation**: A coarse-to-fine progressive alignment from global $\to$ regional $\to$ hard negatives.

### Loss & Training

- **Stage 1**: 1.6B images, batch size 384 per NPU, lr=1e-4, AdamW, 1 epoch, DeepSpeed Zero-2 + BF16.
- **Stage 2**: 12M images, batch size 512 per GPU, lr=1e-6 (two orders of magnitude lower to prevent forgetting), 1 epoch, TF32 + BF16.
- Learnable temperature $\tau$ is initialized to 0.07.
- Model weights are initialized from original CLIP.

## Key Experimental Results

### Main Results: Fine-Grained Understanding (FG-OVD Benchmark, Accuracy %)

| Method | Backbone | hard | medium | easy | trivial |
|------|---------|------|--------|------|---------|
| CLIP | ViT-B/16 | 12.0 | 23.1 | 22.2 | 58.5 |
| EVA-CLIP | ViT-B/16 | 14.0 | 30.1 | 29.4 | 58.3 |
| FineCLIP | ViT-B/16 | 26.8 | 49.8 | 50.4 | 71.9 |
| **FG-CLIP** | **ViT-B/16** | **46.1** | **66.6** | **68.7** | **83.4** |
| CLIP | ViT-L/14 | 15.4 | 25.3 | 25.7 | 38.8 |
| FineCLIP | ViT-L/14 | 22.8 | 46.0 | 46.0 | 73.6 |
| **FG-CLIP** | **ViT-L/14** | **48.4** | **69.5** | **71.2** | **89.7** |

### BBox Classification & Open-Vocabulary Detection

| Method | Backbone | COCO BBox | LVIS BBox | OV-COCO $AP_{50}^{novel}$ |
|------|---------|-----------|-----------|----------------------|
| CLIP | ViT-B/16 | 44.2 | 20.9 | 17.5(F-ViT) |
| FineCLIP | ViT-B/16 | 48.4 | 23.3 | 29.8(F-ViT) |
| CLIPSelf | ViT-B/16 | 43.7 | 7.8 | 33.6(F-ViT) |
| **FG-CLIP** | **ViT-B/16** | **52.3** | **28.6** | **35.1**(F-ViT) |
| **FG-CLIP** | **ViT-L/14** | **63.2** | **38.3** | **41.2**(F-ViT) |

### Ablation Study

| Configuration | DCI I2T/T2I | BBox Top-1 | FG-OVD hard/med/easy |
|------|------------|-----------|---------------------|
| CLIP Baseline | 45.5/43.0 | 44.2 | 12.0/23.1/22.2 |
| Stage1 (Long Captions) | 58.3/57.5 | 47.2 | 21.8/41.6/36.2 |
| +$L_g$ | 62.7/61.2 | 46.8 | 25.4/46.8/42.9 |
| +$L_g$+$L_r$ | 62.4/61.1 | **53.7** | 24.5/47.1/49.5 |
| +$L_g$+$L_r$+$L_h$ | 61.8/60.6 | 52.3 | **46.1/66.6/68.7** |

### Key Findings

- The impact of hard negatives is most significant: performance on the hard subset increases from 24.5% to 46.1% (+88%), demonstrating that it is the key to distinguishing objects in the same category with different attributes.
- Regional contrastive learning substantially improves BBox classification: Top-1 increases from 46.8 to 53.7 (+14.7%).
- The positive effect of the 1.6B scale long captions is already conspicuous in Stage 1: DCI I2T increases from 45.5 to 58.3.
- Replacing the CLIP vision encoder in LLaVA with FG-CLIP: RefCOCO testA/testB scores improve by +3.1 and +7.0, respectively.

## Highlights & Insights

- The unprecedented scale of the 1.6B long-description dataset and the FineHARD dataset represent significant resource contributions to the community.
- A three-pronged system that systematically addresses all three of CLIP's fine-grained bottlenecks: global, regional, and discriminative.
- Fully open-sourced (data, models, and code), available on both GitHub and Hugging Face.
- Effective as a "plug-and-play" replacement for LLaVA's vision encoder, demonstrating the generalizability of its representation quality.

## Limitations & Future Work

- The training cost is extremely high (requiring 160×910B NPUs for 30 days of recaptioning + 7 days for FineHARD construction), raising a high barrier to replication.
- The text position embedding expansion strategy (cutoff at 20 tokens + 4x interpolation) is relatively ad-hoc and lacks comparison with methods like RoPE.
- Hard negative samples rely on LLM generation; a 98.9% qualification rate implies approximately 110,000 noisy samples.
- The paradigm does not address fine-grained temporal understanding in video scenarios.

## Related Work & Insights

- vs LongCLIP: Only executes text length expansion (1M data), whereas FG-CLIP employs a three-pronged approach (1.6B + 40M + 10M).
- vs RegionCLIP: Align regions using category labels, which limits semantic diversity; FG-CLIP uses detailed descriptions.
- vs FineCLIP: Uses 2.5M long descriptions + self-distillation; FG-CLIP represents a 640x larger scale and incorporates hard negatives.

## Rating

⭐⭐⭐⭐ — An engineering-driven systematic solution that comprehensively enhances fine-grained understanding through large-scale, high-quality data and a three-in-one training strategy. The methodology's innovation lies in the rational integration of data construction and training paradigms, rather than algorithmic breakthroughs. Being fully open-sourced substantially increases its practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DLVP-CLIP: Enhancing Fine-Grained Zero-Shot Anomaly Detection via Dynamic Local Visual Prompting](../../CVPR2026/object_detection/dlvp-clip_enhancing_fine-grained_zero-shot_anomaly_detection_via_dynamic_local_v.md)
- [\[CVPR 2026\] FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement](../../CVPR2026/object_detection/fb-clip_fine-grained_zero-shot_anomaly_detection_with_foreground-background_dise.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](../../ICCV2025/object_detection/visual-rft_visual_reinforcement_fine-tuning.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[ICML 2025\] Understanding the Emergence of Multimodal Representation Alignment](understanding_the_emergence_of_multimodal_representation_alignment.md)

</div>

<!-- RELATED:END -->
