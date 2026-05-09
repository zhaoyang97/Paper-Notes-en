---
title: >-
  [Paper Note] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods
description: >-
  [CVPR 2026][Multimodal VLM][HOI Detection] This paper proposes CrossHOI-Bench, the first unified multiple-choice HOI benchmark for evaluating both VLMs and HOI-specific models. Through carefully curated positive and negative examples that eliminate erroneous penalties from incomplete annotations, the benchmark reveals that large VLMs under zero-shot settings surpass state-of-the-art HOI methods by +5.18% in Instance-F1, while still exhibiting systematic weaknesses in multi-action recognition and cross-person action attribution.
tags:
  - CVPR 2026
  - Multimodal VLM
  - HOI Detection
  - VLM Evaluation
  - Multiple-Choice Benchmark
  - Cross-Paradigm Comparison
date: 2026-05-08
content_hash: 0c3d78e37774e655
---

# CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods

**Conference**: CVPR 2026
**arXiv**: [2508.18753](https://arxiv.org/abs/2508.18753)
**Code**: [github.com/ChelsieLei/CrossHOI-Bench](https://github.com/ChelsieLei/CrossHOI-Bench)
**Area**: Multimodal VLM / Human-Object Interaction Detection
**Keywords**: HOI Detection, VLM Evaluation, Multiple-Choice Benchmark, Cross-Paradigm Comparison

## TL;DR

This paper proposes CrossHOI-Bench, the first unified multiple-choice HOI benchmark for evaluating both VLMs and HOI-specific models. Through carefully curated positive and negative examples that eliminate erroneous penalties from incomplete annotations, the benchmark reveals that large VLMs under zero-shot settings surpass state-of-the-art HOI methods by +5.18% in Instance-F1, while still exhibiting systematic weaknesses in multi-action recognition and cross-person action attribution.

## Background & Motivation

**Background**: HOI detection has long been dominated by task-specific models (ADA-CM, CMMP, HOLa, etc.), yet large generative VLMs (Qwen2.5-VL, InternVL3) have demonstrated strong open-scene understanding, raising the central question of whether VLMs can directly perform HOI detection.

**Limitations of Prior Work**: (1) Existing HOI benchmarks (HICO-DET) rely on exact label matching combined with incomplete annotations—any correct prediction not covered by annotations is penalized; (2) annotation ambiguity cannot be resolved through exhaustive labeling, as a single image often lacks sufficient visual evidence to distinguish actions (e.g., "boarding" vs. "exiting an aircraft"); (3) the HICO-DET train/test distributions are highly similar (KL = 0.088), with a large proportion of simple head-category scenes that fail to genuinely challenge model capability.

**Key Challenge**: A unified evaluation protocol is needed to fairly compare both paradigms under the same standard, yet the methodology of existing benchmarks is itself flawed, systematically underestimating true capability: HOI-specific methods achieve <50% mAP, while VLMs reach only ~15% on HICO-DET.

**Goal**: Design an unbiased cross-paradigm HOI evaluation benchmark that reveals the true capability gaps and complementary strengths between VLMs and HOI-specific methods.

**Key Insight**: Reformulate HOI detection as a multi-answer multiple-choice task, using carefully curated positive and negative examples to eliminate the incomplete-annotation problem.

**Core Idea**: A multiple-choice format with curated negatives and three evaluation settings enables, for the first time, a fair and direct comparison between VLMs and HOI-specific methods.

## Method

### Overall Architecture

CrossHOI-Bench reformulates HOI detection as a multi-answer multiple-choice task: each question contains four options with explicitly defined positives (from annotations plus human supplements) and curated negatives (excluding actions that may be correct but are unannotated), thereby avoiding erroneous penalties caused by incomplete annotations.

### Key Designs

1. **Dataset Construction Pipeline (Three Stages)**:

    - **Task Reformulation**: A four-option format allowing multiple correct answers per question (e.g., simultaneously "hold knife" and "cut with knife"), with options randomly shuffled to prevent position bias.
    - **Coarse Filtering (Automatic)**: GPT-4.1 performs an initial pass to classify candidate actions as semantically consistent or inconsistent → dual-model consistency verification using Qwen2.5-VL-32B and GPT-4o, retaining only negatives that both models unanimously identify as incorrect.
    - **Manual Refinement**: (a) Removal of overly simple scenes (single person, plain background, etc.); (b) addition of hard positives (ambiguous actions where both "boarding" and "exiting" are valid); (c) addition of hard negatives (actions performed by surrounding persons, fine-grained similar actions such as "holding" vs. "hugging").
    - **Redistributed Test Set**: KL divergence from the HICO-DET training set is increased from 0.088 to 0.629, reducing prior distribution bias.
    - Output: 1,274 images in the main benchmark + 647 questions in the V-COCO extension (multi-person scenes) + 1,852 questions in the SWiG-HOI extension (person–person interactions), totaling 3,773 questions.

2. **Three Complementary Evaluation Settings**:

    - **Setting 1** (Full HOI Detection): The model must first localize the target person (IoU ≥ 0.5) and then recognize the interaction. Evaluates core HOI detection capability.
    - **Setting 2** (Diagnostic Recognition): Given the target person bounding box, only interaction recognition is evaluated. Eliminates detection error and isolates the localization bottleneck.
    - **Setting 3** (Image-Level Recognition): Recognize interactions for all persons in the image, evaluating global scene understanding.

### Evaluation Metrics

Macro-F1 (class-balanced), Instance-F1 (per-question performance), Micro-F1 (global aggregation), Exact Match (EM), average precision, and average recall.

## Key Experimental Results

### Main Results: Setting 1 (Full HOI Detection)

| Method | Type | Macro-F1 | Instance-F1 | EM | Avg.Prec | Avg.Rec |
|--------|------|----------|-------------|-----|----------|---------|
| ADA-CM | HOI | 43.02 | 47.76 | 19.15 | 76.25 | 51.80 |
| HOLa | HOI | 43.61 | 47.12 | 19.78 | 74.31 | 52.15 |
| CMD-SE | HOI | 47.49 | 44.66 | 20.33 | 78.33 | 46.96 |
| InternVL3-38B | VLM | 38.04 | 38.68 | 20.33 | 84.72 | 33.56 |
| **Qwen2.5-VL-32B** | **VLM** | **50.71** | **52.94** | **26.06** | 75.03 | 51.97 |
| Qwen2.5-VL-7B | VLM | 29.73 | 30.53 | 14.29 | 75.92 | 24.89 |

### Setting 2 (Given Person Bounding Box, Recognition Only)

| Method | Type | Macro-F1 | Instance-F1 | EM | Avg.Prec | Avg.Rec |
|--------|------|----------|-------------|-----|----------|---------|
| InternVL3-38B | VLM | 58.94 | 67.41 | 35.64 | 81.90 | 57.85 |
| Qwen2.5-VL-32B | VLM | 62.90 | 69.52 | 35.01 | 75.30 | 66.61 |
| Qwen2.5-VL-7B | VLM | 48.93 | 57.25 | 25.98 | 74.49 | 46.87 |

### Ablation Study

| Dimension | VLM Advantage | HOI-Specific Advantage |
|-----------|--------------|------------------------|
| Overall F1 | Large models surpass SOTA zero-shot | Small models match HOI methods |
| Multi-action recognition | Tendency to predict only one action | Better recognition of co-occurring actions |
| Cross-person attribution | Frequently attributes surrounding persons' actions to the target | More accurate person–action binding |
| Precision | Generally higher (84.72%) | More stable (~76%) |
| Recall | High variance (33%–52%) | More balanced (~52%) |

### Key Findings

- Qwen2.5-VL-32B zero-shot surpasses all HOI-specific methods in Instance-F1 by +5.18% (52.94 vs. 47.76).
- Small VLMs (7–8B) match HOI-specific methods in recognition-only settings, but performance degrades substantially when detection is required.
- Core VLM weaknesses: insufficient multi-action prediction (predicting only a single interaction) and cross-person action misattribution.
- Qwen3-VL-30B exhibits extreme behavior: Avg.Prec = 100% but Recall ≈ 0%, effectively making no predictions.

## Highlights & Insights

- This is the first work to compare VLMs and HOI-specific methods under a unified protocol, yielding convincing and exemplary conclusions.
- The benchmark reveals the fundamental reasons why existing HOI benchmarks systematically underestimate model capability.
- The methodology of multiple-choice format combined with carefully curated negative examples is transferable to other evaluation tasks with incomplete annotations.
- The three complementary settings and sub-benchmarks provide a rigorous multi-dimensional capability analysis.

## Limitations & Future Work

- The benchmark scale is relatively small (1,274 images in the main benchmark), which may be insufficient to cover all HOI scenarios.
- The multiple-choice format may simplify the true difficulty of open-world HOI understanding.
- Negative example curation relies on VLM consistency judgments, which may introduce model-specific biases.
- The impact of fine-tuning VLMs on benchmark performance is not analyzed in depth.

## Related Work & Insights

- **vs. HICO-DET/V-COCO**: The fundamental distinction lies in multi-answer multiple-choice vs. exact matching, eliminating the systematic penalty from incomplete annotations.
- **vs. SWiG-HOI**: SWiG-HOI expands the label space (5,500+ HOI categories), whereas CrossHOI-Bench focuses on fair cross-paradigm comparison.
- **Methodological Implications**: VLM capabilities across many tasks may be systematically underestimated due to benchmark design flaws; similar evaluation reformulations may yield new findings in other domains.
- **VLM Capability Ceiling**: The HOI understanding capacity of large VLMs has been substantially underestimated; future work should prioritize multi-action recognition and cross-person attribution.

## Rating

⭐⭐⭐⭐⭐ (4.5/5)

- **Novelty** ⭐⭐⭐⭐: The benchmark design for unified cross-paradigm evaluation fills an important gap.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Multiple models, multiple settings, sub-benchmarks, and multi-dimensional analysis make the evaluation exceptionally comprehensive.
- **Writing Quality** ⭐⭐⭐⭐⭐: Problem formulation is clear, experimental design is rigorous, and conclusions are compelling.
- **Value** ⭐⭐⭐⭐⭐: Reveals important evaluation biases and paradigm complementarity with meaningful community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](../../ICLR2026/multimodal_vlm/zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)
- [\[CVPR 2026\] UNICBench: UNIfied Counting Benchmark for MLLM](unicbench_unified_counting_benchmark_for_mllm.md)
- [\[CVPR 2026\] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)
- [\[CVPR 2026\] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)

</div>

<!-- RELATED:END -->
