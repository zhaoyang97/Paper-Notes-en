---
title: >-
  [Paper Note] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Visual Prompting] VP-Bench introduces the first systematic two-stage benchmark for evaluating MLLMs' understanding of visual prompts (VPs): Stage 1 covers 30K+ images across 8 VP shape types × 355 attribute combinations to assess VP perception ability, while Stage 2 evaluates the practical effectiveness of VPs on 6 downstream tasks. Experiments on 28 MLLMs reveal the critical impact of VP shape selection on model performance.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Visual Prompting
  - MLLM Evaluation
  - Visual Prompt Perception
  - Region Referring
  - Benchmark Design
date: 2026-05-08
content_hash: 84a87512169b7a86
---

# VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.11438](https://arxiv.org/abs/2511.11438)
**Code**: [https://github.com/Endlinc/VP-Bench](https://github.com/Endlinc/VP-Bench)
**Area**: Multimodal VLM / Benchmark
**Keywords**: Visual Prompting, MLLM Evaluation, Visual Prompt Perception, Region Referring, Benchmark Design

## TL;DR
VP-Bench introduces the first systematic two-stage benchmark for evaluating MLLMs' understanding of visual prompts (VPs): Stage 1 covers 30K+ images across 8 VP shape types × 355 attribute combinations to assess VP perception ability, while Stage 2 evaluates the practical effectiveness of VPs on 6 downstream tasks. Experiments on 28 MLLMs reveal the critical impact of VP shape selection on model performance.

## Background & Motivation
**Background**: When humans wish to direct MLLMs' attention to specific regions in an image, they naturally employ visual prompts (VPs) such as bounding boxes, arrows, and circled annotations. Such usage is increasingly common in interactive AI applications.

**Limitations of Prior Work**: Existing VP-related benchmarks (e.g., ViP-Bench with only 303 images; SoV with only 119 images) are too small in scale with limited VP type coverage. They neither systematically study the effect of VP shape/attribute choices on model performance nor evaluate the practical effectiveness of VPs on downstream tasks.

**Key Challenge**: Different VP shapes (bounding box vs. scribble vs. point marker) and attributes (color/thickness/style) vary greatly in perceptibility to MLLMs, yet no systematic study exists to guide practitioners on "which VP to use."

**Goal**: Systematically evaluate MLLMs' VP perception capabilities and the impact of VPs on downstream tasks.

**Key Insight**: VP shapes are categorized into 8 types (tag/bbox/arrow/mask/contour/oval/point/scribble), each further decomposed by attributes (color/line width/vertex shape, etc.), forming the most comprehensive VP evaluation framework to date.

**Core Idea**: Systematically evaluate VP understanding capabilities of 28 MLLMs using 34K+ images across 355 VP attribute combinations.

## Method

### Overall Architecture
A two-stage evaluation paradigm: Stage 1 (VP Perception) assesses models' ability to detect, count, localize, and interpret different VP shapes and attributes on natural scene images; Stage 2 (VP Downstream Effectiveness) applies each model's best-performing VP configuration from Stage 1 to measure practical gains across 6 real-world application tasks.

### Key Designs

1. **Stage 1: VP Perception Evaluation**

   - **Function**: Evaluate MLLMs' perception of 8 VP shape types across 355 attribute combinations.
   - **Mechanism**: 30K+ images are generated from MS-COCO annotations, each overlaid with a specific VP. Four question types are included — existence (presence of VP), counting (number of VPs), coarse localization (VP position in the image), and reference (object pointed to by VP). All questions are multiple-choice. Debiasing questions (no VP present but VP mentioned in the question) are included to probe hallucination.
   - **Design Motivation**: The 355 attribute combinations represent more than 40× the coverage of existing benchmarks, enabling precise analysis of which color/thickness/style is most effective.

2. **Stage 2: VP Downstream Task Evaluation**

   - **Function**: Assess whether VPs are more beneficial than pure text-based spatial descriptions in real-world application scenarios.
   - Six downstream tasks: Medical Image Analysis (MIA), 3D object recognition, facial expression recognition, street scene recognition, GUI element recognition, and Scene Graph Generation (SGG).
   - **Comparative Design**: R-BVP (randomly selected globally best VP from Stage 1) vs. BVP (best VP for the specific model), as well as VP vs. pure text spatial descriptions.

3. **VP Description**

   - **Function**: Augment text instructions with natural language descriptions of the VP shape (e.g., "the red bounding box marks the target region").
   - **Core Idea**: Making VP semantics explicit in both visual and textual modalities simultaneously reduces ambiguity in the model's interpretation of VP meaning.
   - **Key Finding**: Adding VP descriptions yields substantial performance gains — InternVL3-78B improves on average from 81.3% → 88.0%, with the Mask shape showing a 29% improvement.

## Key Experimental Results

### Stage 1 — VP Perception (28 MLLMs)

| Model | BBox | Oval | Tag | Mask | Point | Scribble | Avg |
|-------|------|------|-----|------|-------|----------|-----|
| Human | 97.3 | 94.9 | 89.0 | 85.3 | 90.7 | 82.8 | 90.0 |
| InternVL3-78B | 94.3 | 95.8 | 93.9 | 80.0 | 81.6 | 80.9 | 88.0 |
| GPT-4o | 74.2 | 79.8 | 70.0 | 65.3 | 49.3 | 64.5 | 68.8 |
| Qwen2.5-VL-72B | 92.9 | 92.8 | 92.3 | 68.7 | 69.6 | 74.8 | 82.8 |

### Stage 2 — VP Downstream Tasks (Top Models)

| Model | MIA | 3D | Street | GUI | SGG | Avg |
|-------|-----|----|--------|-----|-----|-----|
| InternVL3-78B | 53.4 | 87.0 | 67.5 | 97.8 | 95.0 | 79.1 |
| InternVL3-38B | 48.4 | 88.7 | 59.8 | 99.0 | 94.2 | 77.2 |
| Molmo-72B | 62.8 | 78.0 | 60.8 | 96.3 | 91.8 | 76.2 |

### Key Findings
- **Regular shapes >> irregular shapes**: BBox/Oval/Tag average 85%+, while Mask/Point/Scribble reach only ~69% — MLLMs exhibit severely insufficient perception of irregular VPs.
- **Color is critical**: High-contrast colors (maximizing contrast against the background) represent the optimal choice for the majority of models.
- **VP descriptions yield substantial gains**: Adding descriptions to the Mask shape improves InternVL3-78B by 29.2% and Qwen2.5-VL-72B by 27.3%.
- **GPT-4o unexpectedly lags in VP perception**: Its 68.8% average accuracy is far below open-source models (InternVL3-78B: 88.0%), with particularly poor spatial localization (57.8%).
- **VP-specific fine-tuning is not always beneficial**: ViP-LLaVA, after training on VP data, improves in Stage 1 but degrades in Stage 2 — underlying capability matters more than task-specific training.
- **VP vs. text spatial description**: VPs outperform text-based spatial descriptions by 1.64% (MIA) and 1.15% (3D) on average, but text performs better in tasks such as facial expression recognition — VP effectiveness is task-dependent.

## Highlights & Insights
- **Largest VP evaluation to date**: 34K+ images, 355 attribute combinations, 28 models — approximately 100× larger than ViP-Bench, making this the most comprehensive VP perception benchmark available.
- **Practical design guidelines**: High-contrast color + medium-thickness bounding box constitutes a "universally optimal VP" — providing direct guidance for VP design in real-world applications.
- **Debiasing question design**: Approximately 12.5% of questions are debiasing samples (no VP in image but VP mentioned in question), effectively probing VP hallucination in models.
- **Importance of VP descriptions**: Simply stating "this red box indicates the target region" in the instruction yields substantial performance gains — a near-zero-cost improvement strategy.

## Limitations & Future Work
- Stage 2 covers a limited number of downstream tasks (6), with only 200 images per task.
- VP rendering is based on square crops from MS-COCO, which may not fully reflect VP usage in real-world scenarios.
- Dynamic or interactive VPs (e.g., mouse drag, gesture guidance) are not evaluated.
- Model performance under multiple simultaneous VPs is not examined.

## Related Work & Insights
- **vs. ViP-Bench**: ViP-Bench uses 303 images / 8 VP types / 1 domain; VP-Bench uses 34K images / 355 combinations / 4 domains — over 100× scale improvement.
- **vs. SoV**: SoV contains only 119 images for Set-of-Mark validation; VP-Bench covers substantially more shapes and attributes.
- **vs. VipAct (in this note collection)**: VipAct's VP description agent design aligns with VP-Bench's finding that explicitly describing VP semantics in instructions facilitates MLLM comprehension.
- **Insight**: VP shape selection should be task-driven rather than based on universal preference — the optimal VP may differ across tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First large-scale systematic VP evaluation, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 28 models, 355 attribute combinations, two-stage evaluation, debiasing design, VP description ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed and substantive data.
- **Value**: ⭐⭐⭐⭐⭐ Significant reference value for VP design practice and MLLM region-understanding research.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)
- [\[AAAI 2026\] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](../../CVPR2026/multimodal_vlm/vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)

<!-- RELATED:END -->
