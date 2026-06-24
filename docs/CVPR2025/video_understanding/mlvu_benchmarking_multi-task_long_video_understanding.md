---
title: >-
  [Paper Note] MLVU: Benchmarking Multi-task Long Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Long Video Understanding] This work proposes the MLVU benchmark, which systematically evaluates the capability of multimodal large language models in long video understanding through 9 diverse evaluation tasks, various video types, and flexible duration settings, revealing significant limitations of existing models in processing long videos.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Video Benchmarking"
  - "Multimodal Large Language Models"
  - "Multi-task Evaluation"
  - "Video QA"
date: 2026-05-08
content_hash: 26e864b3271a8135
---

# MLVU: Benchmarking Multi-task Long Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2406.04264](https://arxiv.org/abs/2406.04264)  
**Code**: [https://github.com/JUNJIE99/MLVU](https://github.com/JUNJIE99/MLVU)  
**Area**: Video Understanding  
**Keywords**: Long Video Understanding, Video Benchmarking, Multimodal Large Language Models, Multi-task Evaluation, Video QA

## TL;DR

This work proposes the MLVU benchmark, which systematically evaluates the capability of multimodal large language models in long video understanding through 9 diverse evaluation tasks, various video types, and flexible duration settings, revealing significant limitations of existing models in processing long videos.

## Background & Motivation

### Background

**Background**: Existing video understanding benchmarks exhibit three major issues: (1) Insufficient video length, where videos in most benchmarks are only a few to tens of seconds long, failing to reflect true long video understanding capabilities; (2) Lack of diversity in video types and evaluation tasks, which typically focus on a single type (e.g., egocentric videos) or a single task (e.g., captioning); (3) Suboptimal evaluation design, where many questions can be answered directly without watching the video, such as relying on common knowledge of famous movies or focusing solely on single-frame information.

The core objective of MLVU is to construct a long video understanding benchmark that is **sufficiently long, rich in types, and diverse in tasks**, ensuring that evaluation tasks can only be completed based on an in-depth understanding of the long videos.

### Proposed Approach

**Goal**: ### Overall Architecture

MLVU is a multi-task long video understanding benchmark consisting of 3,102 questions and 9 types of tasks, constructed based on 1,730 videos.

## Method

### Overall Architecture

MLVU is a multi-task long video understanding benchmark consisting of 3,102 questions and 9 types of tasks, constructed based on 1,730 videos. Video lengths range from 3 minutes to 2 hours, with an average of approximately 15 minutes. The videos are further segmented into incremental clips (e.g., first 3 minutes, first 6 minutes, full-length), enabling evaluation across different durations.

### Key Designs

1. **Three-level Evaluation Task Hierarchy**: Long video understanding is divided into three levels—(a) Holistic LVU: including Topic Reasoning (TR), Anomaly Recognition (AR), and Video Summarization (VS), which require global information; (b) Single-Detail LVU: including Needle QA (NQA), Ego Reasoning (ER), Plot QA (PQA), and Sub-Scene Captioning (SSC), which require locating and understanding specific segments; (c) Multi-Detail LVU: including Action Ordering (AO) and Action Counting (AC), which require jointly utilizing information from multiple locations.

2. **Innovative Design of Needle QA**: Inspired by the Needle-In-the-Haystack-Search in the text domain, a short video (needle) is randomly inserted into a long background video. Models must infer the location of the needle based on the question and answer it. This effectively tests the model's ability to locate and utilize local information within long videos.

3. **MLVU Time-ladder Derivative Dataset**: Evaluations are created for the same task under different video lengths (180s, 360s, 600s) to systematically study the impact of video length on model performance, achieving a flexible analysis in the temporal dimension.

### Loss & Training

Since this paper introduces a benchmark, it does not involve model training. The evaluation strategies include: Multiple-choice questions are measured using Accuracy; generation tasks (VS and SSC) utilize GPT-4 scoring to compare generated content with human annotations. All models are evaluated in a zero-shot manner.

## Key Experimental Results

### Main Results

| Model | M-Avg (MCQ) | G-Avg (Gen) | TR | NQA | AO | AC |
|------|-------------|-------------|-----|-----|-----|-----|
| GPT-4o | 54.5% | 5.87% | 83.7% | 42.9% | 46.2% | 35.0% |
| LLaVA-OneVision | 51.7% | 4.42% | 83.5% | 46.7% | 35.7% | 23.3% |
| Video-XL | 46.3% | 4.21% | 78.0% | 50.0% | 48.6% | 31.7% |
| VideoLLaMA2 | 48.4% | 3.95% | 80.2% | 36.7% | 42.9% | 16.7% |
| InternVL-2 | 47.5% | 3.90% | 85.7% | 48.3% | 32.9% | 15.0% |
| Random Baseline | 16.7% | - | 16.7% | 16.7% | 16.7% | 16.7% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Context Length: GPT-4o 16 frames → 256 frames | M-Avg: 45.8 → 54.5 (+8.7) | Longer inputs significantly improve LVU performance |
| Context Length: MGV 16 frames → 90 frames | M-Avg: 24.2 → 31.7 (+7.5) | Open-source models also benefit |
| LLM Backbone: Vicuna-7B → 13B | M-Avg: 13.3 → 18.8 (+5.5) | Larger backbones help |
| LLM Backbone: LLaMA-7B → Mistral-7B | M-Avg: 20.6 → 31.7 (+11.1) | Stronger backbones bring significant improvements |
| Image Understanding (MMMU): GPT-4V → 4o | 58.1 → 63.8 (MMMU) / 43.3 → 45.8 (M-Avg) | Image understanding capability is highly correlated with LVU |

### Key Findings

1. Long video understanding remains a massive challenge for current MLLMs: GPT-4o achieves an M-Avg of only 54.5%, and most models perform close to random on tasks like NQA, AO, and AC.
2. Performance for all models continuously degrades as video length increases: models designed for short videos exhibit near-random performance on 10-minute videos.
3. Multi-detail tasks (AO, AC) are far more difficult than single-detail tasks, with performance dropping sharply as the number of probes increases.
4. Advanced long-video models (LongVA, Video-XL) are less sensitive to the temporal location of reference segments, demonstrating more stable performance.
5. Context length, image understanding capability, and LLM backbone are the three key factors influencing LVU performance.

## Highlights & Insights

- Clever design of evaluation tasks: Needle QA effectively adapts experiences from the text domain to test information retrieval capabilities in long videos.
- The Time-ladder design allows quantitative analysis of the impact of video length on performance, which is rare in previous benchmarks.
- Covers a wide variety of video types including movies, surveillance, egocentric videos, cartoons, and gaming, closely aligning with real-world application scenarios.
- Dual-track evaluation with both open-ended and multiple-choice questions systematically assesses different dimensions of model capabilities.

## Limitations & Future Work

- Generation tasks (VS, SSC) rely on GPT-4 evaluation, which may introduce assessment bias.
- There is still room to expand the diversity of video sources (e.g., fewer videos in specialized fields like medicine and education).
- Only evaluated sub-sampled frame inputs without exploring continuous video stream processing.
- Lacks consideration of audio information, which is crucial for many video understanding tasks.

## Related Work & Insights

- Complementary to concurrent works such as Video-MME and LongVideoBench, MLVU holds an advantage in video length and task diversity.
- The design methodology of Needle QA can be extended to long-context evaluations in other modalities.
- The Time-ladder concept can be applied to evaluate how efficiently models utilize their context windows.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-level task hierarchy and Needle QA designs are innovative, though the novelty of benchmarks is typically moderate.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 23 models, with highly sufficient multi-dimensional ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in systematic evaluations for long video understanding, offering valuable references to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)
- [\[CVPR 2025\] Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](holmes-vau_towards_long-term_video_anomaly_understanding_at_any_granularity.md)
- [\[CVPR 2025\] VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video SpatioTemporal Augmentation](vista_enhancing_long-duration_and_high-resolution_video_understanding_by_video_s.md)

</div>

<!-- RELATED:END -->
