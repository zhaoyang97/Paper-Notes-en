---
title: >-
  [Paper Note] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering
description: >-
  [CVPR 2025][Video Understanding][Egocentric Video Question Answering] This paper proposes the EgoTextVQA benchmark, which contains 1.5K egocentric videos and 7K scene-text-related QA pairs, revealing that existing MLLMs exhibit severe deficiencies in real-time scene text-aware QA assistance from an egocentric perspective (the best model, Gemini 1.5 Pro, achieves only ~33% accuracy).
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Egocentric Video Question Answering"
  - "Scene Text Recognition"
  - "Multimodal Large Language Models (MLLMs)"
  - "Benchmark Dataset"
  - "Egocentric Vision"
date: 2026-05-08
content_hash: ee967754a40b9339
---

# EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering

**Conference**: CVPR 2025  
**arXiv**: [2502.07411](https://arxiv.org/abs/2502.07411)  
**Code**: [https://github.com/zhousheng97/EgoTextVQA](https://github.com/zhousheng97/EgoTextVQA)  
**Area**: Video Understanding  
**Keywords**: Egocentric Video Question Answering, Scene Text Recognition, Multimodal Large Language Models (MLLMs), Benchmark Dataset, Egocentric Vision

## TL;DR

This paper proposes the EgoTextVQA benchmark, which contains 1.5K egocentric videos and 7K scene-text-related QA pairs, revealing that existing MLLMs exhibit severe deficiencies in real-time scene text-aware QA assistance from an egocentric perspective (the best model, Gemini 1.5 Pro, achieves only ~33% accuracy).

## Background & Motivation

### Background

**Background**: Existing scene text VQA datasets (e.g., TextVQA, ST-VQA) assume that users can take clear, focused images and that the questions directly refer to the text regions, which is impractical in real-world applications. When considering scenarios such as assisting visually impaired individuals, users find it difficult to capture clear images or point directly at text regions. Meanwhile, existing video text-QA datasets (e.g., RoadTextVQA) still assume that users know the location of the scene text, resulting in overly simplistic question designs that only require OCR extraction to answer.

The motivation of this paper is to build an egocentric scene text-aware video question answering benchmark that is closer to real-world demands:

### Limitations of Prior Work

**Limitations of Prior Work**: Questions must reflect **real user needs** instead of directly pointing to the scene text.

### Key Challenge

**Key Challenge**: Support for **real-time streaming QA** (each question is timestamped, and the model can only access video content prior to the timestamp).

### Proposed Solution

**Proposed Solution**: Coverage of **diverse indoor and outdoor scenarios** (outdoor driving + indoor chores).

## Method

### Overall Architecture

EgoTextVQA is an **evaluation benchmark** rather than a methodological paper. Its core contributions lie in the meticulous construction of the dataset, comprehensive model evaluations, and heuristic discovery.

### Key Designs

1. **Dataset Construction Pipeline**:
    - Function: To filter, generate, and refine high-quality QA pairs from existing egocentric video datasets.
    - Mechanism: Automatically filter videos containing text using a scene text detection system (thresholds: 15% for RoadTextVQA, 5% for Ego4D), then generate QA pairs using GPT-4o with carefully designed prompts, and finally perform 5 rounds of progressive filtering, correction, and refinement by 9 annotators.
    - Design Motivation: Automated generation ensures diversity, while multiple rounds of human verification ensure high quality. Ultimately, only about 30% of the automatically generated QA pairs are retained.

2. **Real-Time Streaming QA Setting**:
    - Function: To simulate real-time visual assistance in real-world scenarios.
    - Mechanism: Assign a timestamp to each question; the model is only allowed to access the video content prior to the associated timestamp.
    - Design Motivation: Existing benchmarks allow access to the entire video, which deviates from real-world assistive applications. The highest accuracy of closed-loop models on the real-time QA subset is only 20.2%, substantially lower than the 33.4% on the full set.

3. **Multi-Dimensional Question Classification System**:
    - Function: To support fine-grained analysis of model behavior.
    - Mechanism: Out-of-door scenarios are categorized into Location/Description/Direction/Intention Reasoning, etc.; indoor scenarios are categorized into Hands-on/Shopping/Kitchen/Book-related/Gameplay, etc.
    - Design Motivation: Different question categories examine disparate capabilities of the models, facilitating the identification of model weaknesses.

### Loss & Training

This is a benchmark paper and does not involve model training. GPT-4o mini is employed as a semantic similarity evaluator to output two metrics: Accuracy (0-100%) and Score (0-5).

## Key Experimental Results

### Main Results

| Model | EgoTextVQA-Outdoor Acc. | EgoTextVQA-Indoor Acc. | Type |
|------|------------------------|----------------------|------|
| Gemini 1.5 Pro | **33.4%** | **34.4%** | Closed-source |
| Gemini 1.5 Flash | 30.1% | 32.0% | Closed-source |
| GPT-4o | 30.3% | 28.3% | Closed-source |
| Qwen2-VL | 28.2% | 23.3% | Open-source |
| LLaVA-NeXT-Video | 19.5% | 25.4% | Open-source |
| Human | 43.1% | 27.7% | - |

### Heuristic Exploration Experiments

| Strategy | Outdoor Acc. Change | Indoor Acc. Change | Description |
|------|------------------|-----------------|------|
| Video + Scene Text Assistance (GPT-4o) | 30.3 -> 52.9 (+22.6) | 28.3 -> 37.9 (+9.6) | OCR assistance yields massive gains |
| High-Resolution QA Frame (Qwen2-VL) | 28.2 -> 46.8 (+18.6) | - | High resolution is critical |
| Single-Frame vs. Video (Gemini 1.5 Pro) | 33.4 -> 30.4 (-3.0) | 34.4 -> 15.8 (-18.6) | Indoor requires multi-frame reasoning |
| Scene Text Super-Resolution 1.5x (Qwen2-VL) | 28.2 -> 34.1 (+5.9) | 23.3 -> 22.3 (-1.0) | Effective for outdoor scenarios |

### Key Findings

1. **All models perform poorly on EgoTextVQA**, with the best closed-source model achieving only ~33%, while human performance in indoor scenes is even lower (27.7%), demonstrating that scene text recognition poses a significant challenge for both humans and models.
2. **Assisting with OCR input is the most effective approach**: After incorporating auxiliary scene text information, GPT-4o's outdoor accuracy jumps from 30.3% to 52.9%.
3. **Indoor scenarios are exceptionally reliant on multi-frame temporal reasoning**: Utilizing single-frame inputs in indoor scenarios results in a drastic performance drop of up to 18.6% for Gemini 1.5 Pro.
4. **High-resolution images are vital for scene text recognition**, yet there exists a necessary trade-off with computational efficiency.

## Highlights & Insights

- The dataset design is highly rigorous: with 5 rounds of human verification, questions that do not point directly to text, and a simulated real-time streaming QA setting, it reflects profound consideration for real-world applications.
- Human performance is lower than that of closed-source models (in indoor scenarios), which exposes the true difficulty of the task—it is not merely about text recognition, but also necessitates external knowledge.
- The heuristic exploration phase is highly systematic, delivering a comprehensive multi-angle analysis across temporal localization, resolution, OCR assistance, and multimodal inputs.

## Limitations & Future Work

- The diversity of open-ended answers leads to relatively low human evaluation scores; the ground truth (GT) answers can be further enriched in the future.
- Currently, the benchmark only supports open-ended QA, and could be extended to formats like multiple-choice questions in the future.
- Relying on GPT-4o mini as an evaluator may introduce systematic evaluation biases.
- The resolution of indoor videos is relatively low ($480 \times 360$), which limits the effectiveness of scene text recognition.

## Related Work & Insights

- Unlike static image VQA benchmarks (e.g., TextVQA and ST-VQA), this work emphasizes an **egocentric perspective in dynamic videos**.
- Unlike general egocentric VQA benchmarks (e.g., QAEgo4D and AssistQ), this work specifically **focuses on scene text**.
- Insights: Future video understanding models must simultaneously possess high-resolution text recognition, temporal reasoning, and user intention comprehension capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ The first benchmark to focus on real-time, egocentric scene-text QA, with highly precise target scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models evaluated alongside heuristic exploration in 4 distinct dimensions, providing exceptionally detailed analyses.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with highly transparent details on the dataset construction process.
- Value: ⭐⭐⭐⭐ Exposes critical bottlenecks for MLLMs in real-world visual assistance applications, offering valuable guidance for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[CVPR 2025\] BIMBA: Selective-Scan Compression for Long-Range Video Question Answering](bimba_selective-scan_compression_for_long-range_video_question_answering.md)
- [\[CVPR 2026\] Ego-Grounding for Personalized Question-Answering in Egocentric Videos](../../CVPR2026/video_understanding/ego-grounding_for_personalized_question-answering_in_egocentric_videos.md)
- [\[CVPR 2025\] EgoLife: Towards Egocentric Life Assistant](egolife_towards_egocentric_life_assistant.md)

</div>

<!-- RELATED:END -->
