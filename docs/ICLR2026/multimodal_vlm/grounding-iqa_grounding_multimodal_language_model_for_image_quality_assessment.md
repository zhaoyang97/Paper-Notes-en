---
title: >-
  [Paper Note] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment
description: >-
  [ICLR 2026][Multimodal VLM][Image Quality Assessment] This paper integrates spatial grounding (referring + grounding) with image quality assessment (IQA)…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Image Quality Assessment"
  - "Spatial Grounding"
  - "Multimodal LLM"
  - "Fine-Grained Perception"
  - "Grounding"
date: 2026-05-08
content_hash: ff643437de7e879a
---

# Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment

**Conference**: ICLR 2026
**arXiv**: [2411.17237](https://arxiv.org/abs/2411.17237)
**Code**: [https://github.com/zhengchen1999/Grounding-IQA](https://github.com/zhengchen1999/Grounding-IQA)
**Area**: Object Detection / Multimodal VLM / Image Quality Assessment
**Keywords**: Image Quality Assessment, Spatial Grounding, Multimodal LLM, Fine-Grained Perception, Grounding

## TL;DR
This paper integrates spatial grounding (referring + grounding) with image quality assessment (IQA), constructs the GIQA-160K dataset to fine-tune a multimodal LLM that generates quality descriptions with bounding boxes and spatial VQA, achieving significantly superior fine-grained quality perception over general-purpose MLLMs.

## Background & Motivation

**Background**: IQA has evolved from traditional metrics (PSNR/SSIM) to multimodal LLM-based semantic IQA (e.g., Q-Instruct), enabling natural language quality descriptions.

**Limitations of Prior Work**: Existing IQA methods only produce image-level quality descriptions (e.g., "the image is generally blurry") without identifying which specific regions exhibit which quality issues. For complex images with spatially varying quality, such global descriptions are insufficiently granular.

**Key Challenge**: IQA requires fine-grained spatial localization capabilities, yet existing IQA datasets lack spatial annotations, and the spatial perception capacity of MLLMs remains underutilized for low-level vision tasks.

**Goal**: (a) Construct an IQA dataset with spatial annotations; (b) train an MLLM to perform quality assessment and spatial grounding jointly.

**Key Insight**: Two new sub-tasks are defined—GIQA-Description (quality descriptions with bounding boxes) and GIQA-VQA (quality QA with spatial information).

**Core Idea**: Enable IQA models to not only state "the image is blurry" but also specify "the billiard table region (bbox) is sharp, while the background region (bbox) is blurry."

## Method

### Overall Architecture
A four-stage automatic annotation pipeline constructs the GIQA-160K dataset, upon which an MLLM (e.g., mPLUG-Owl2) is fine-tuned.

### Key Designs

1. **Automatic Annotation Pipeline (4 Stages)**:

    - Stage 1: Llama3 extracts object label triplets (descriptive phrase, quality, effect) from quality descriptions.
    - Stage 2: Grounding DINO detects bounding boxes using descriptive phrases (rather than category names) for higher precision.
    - Stage 3: IQA-Filter uses Q-Instruct to verify whether detected boxes genuinely exhibit the specified quality issues; Box-Merge consolidates fragmented boxes.
    - Stage 4: Coordinates are discretized into grid indices (20×20 grid), representing each box with at most 9 tokens.

2. **GIQA-VQA Generation**:

    - **Function**: Automatically generates spatially-grounded QA pairs from GIQA-DES descriptions.
    - **Mechanism**: An LLM generates two types of questions—Yes/No questions (~50K) and open-ended What/Why/How questions (~50K)—ensuring that questions reference spatially grounded entities.
    - **Design Motivation**: The VQA format supports both referring (querying quality given a location) and grounding (querying location given a quality description).

3. **Multi-Task Training**:

    - **Function**: Jointly trains on description and VQA tasks.
    - **Mechanism**: Standard SFT fine-tuning on GIQA-160K with autoregressive LM loss.
    - **Design Motivation**: Ablations show that multi-task training (DES + VQA) outperforms single-task training on both tasks.

## Key Experimental Results

### Main Results (GIQA-Bench, mPLUG-Owl2-7B)

| Metric | Before Fine-Tuning | After Fine-Tuning | Gain |
|---|---|---|---|
| BLEU@4 | 3.62 | 22.87 | +19.25 |
| LLM-Score | 48.25 | 63.00 | +14.75 |
| mIoU (Box Localization) | N/A | 0.5955 | - |
| VQA Overall Accuracy | 56.3% | 74.2% | +17.9% |

### Cross-Model Comparison

| Model | mIoU | BLEU@4 | VQA Overall Accuracy |
|---|---|---|---|
| LLaVA-v1.5-7B | 0.528 | 19.02 | 68.5% |
| LLaVA-v1.6-7B | 0.598 | 19.17 | 72.5% |
| **mPLUG-Owl2-7B** | **0.596** | **22.87** | **74.2%** |

### Ablation Study

| Configuration | Tag-Recall | LLM-Score | VQA Accuracy |
|---|---|---|---|
| Only-DES | 0.550 | 61.75 | 59.0% |
| Only-VQA | 0.328 | 38.50 | 72.2% |
| **GIQA-160K (DES+VQA)** | **0.547** | **63.00** | **74.2%** |

### Key Findings
- Multi-task training improves VQA accuracy by 2.0% over Only-VQA, and description quality by 1.25 LLM-Score over Only-DES.
- Box refinement (IQA-Filter + Box-Merge) improves mIoU from 0.562 to 0.585.
- Coordinate discretization into a 20×20 grid requires only 9 tokens per box, offering high efficiency.

## Highlights & Insights
- **Cross-Domain Innovation (IQA + Grounding)**: Introducing referring/grounding into IQA represents a natural yet previously unexplored intersection.
- **Automatic Annotation Pipeline**: The four-stage pipeline is highly automated and potentially applicable to other low-level vision tasks requiring spatial annotation.
- **Dataset Contribution**: GIQA-160K contains 167K annotated samples and is the first IQA dataset with spatial grounding annotations.

## Limitations & Future Work
- The annotation pipeline depends on multiple models (Llama3, Grounding DINO, Q-Instruct), allowing errors to propagate across stages.
- The 20×20 grid offers limited spatial resolution, reducing localization accuracy for small-region quality issues.
- Evaluation is limited to 7B-scale models; performance with larger models remains unexplored.
- Quality descriptions are sourced from existing IQA datasets, limiting coverage of quality distortion types.

## Related Work & Insights
- **vs. Q-Instruct**: Produces text-only IQA without spatial grounding; this work adds spatial annotations on top of its outputs.
- **vs. Grounding DINO**: Used within the annotation pipeline but cannot directly perform IQA.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The task formulation of IQA + Grounding is novel, though the method itself (SFT fine-tuning of MLLMs) is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model validation and ablations are provided, but comparisons against specialized IQA methods are absent.
- **Writing Quality**: ⭐⭐⭐⭐ The annotation pipeline is described in thorough detail.
- **Value**: ⭐⭐⭐⭐ The dataset and task definition contribute more significantly than the methodology itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ICLR 2026\] How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images](how_do_medical_mllms_fail_a_study_on_visual_grounding_in_medical_images.md)
- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](../../CVPR2026/multimodal_vlm/groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](../../CVPR2026/multimodal_vlm/timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)

</div>

<!-- RELATED:END -->
