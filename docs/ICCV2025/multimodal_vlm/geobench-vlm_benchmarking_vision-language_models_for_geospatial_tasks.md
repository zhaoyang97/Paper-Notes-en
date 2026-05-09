---
title: >-
  [Paper Note] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks
description: >-
  [ICCV 2025][Multimodal VLM][Vision-Language Models] This paper introduces GEOBench-VLM, a comprehensive benchmark designed to evaluate VLMs on geospatial tasks, encompassing 31 sub-tasks across 8 major categories and over 10,000 manually verified instructions. The benchmark reveals that current state-of-the-art VLMs, including GPT-4o, still perform poorly on geospatial tasks, with the highest accuracy reaching only 41.7%.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Vision-Language Models
  - Geospatial
  - Remote Sensing Benchmark
  - Multimodal Evaluation
  - Temporal Analysis
date: 2026-05-08
content_hash: cee42d7d8b271035
---

# GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks

**Conference**: ICCV 2025
**arXiv**: [2411.19325](https://arxiv.org/abs/2411.19325)
**Code**: [https://github.com/The-AI-Alliance/GEO-Bench-VLM](https://github.com/The-AI-Alliance/GEO-Bench-VLM)
**Area**: Multimodal VLM
**Keywords**: Vision-Language Models, Geospatial, Remote Sensing Benchmark, Multimodal Evaluation, Temporal Analysis

## TL;DR

This paper introduces GEOBench-VLM, a comprehensive benchmark designed to evaluate VLMs on geospatial tasks, encompassing 31 sub-tasks across 8 major categories and over 10,000 manually verified instructions. The benchmark reveals that current state-of-the-art VLMs, including GPT-4o, still perform poorly on geospatial tasks, with the highest accuracy reaching only 41.7%.

## Background & Motivation

Existing VLM evaluation benchmarks (e.g., SEED-Bench, MMMU, MMBench) focus primarily on general vision-language tasks and fail to adequately address the unique challenges of geospatial applications:

**Temporal Change Detection**: Monitoring urban development and environmental degradation requires temporal analysis capabilities.

**Large-Scale Object Counting**: Remote sensing imagery demands precise counting of buildings, vehicles, and similar objects.

**Small Object Detection**: Object scales vary considerably in satellite imagery.

**Non-Optical Data Interpretation**: Understanding unconventional imagery such as SAR and multispectral data.

Existing remote sensing VLM benchmarks (e.g., VLEO) lack coverage of temporal analysis, segmentation tasks, and non-optical data evaluation. GEOBench-VLM aims to fill this gap by providing a comprehensive geospatial evaluation framework for both general-purpose and remote sensing-specific VLMs.

## Method

### Overall Architecture

GEOBench-VLM is an evaluation benchmark suite rather than a novel model; its core contributions lie in data construction, task design, and the evaluation framework. A multiple-choice question (MCQ) format is adopted to enable objective and scalable automated evaluation, mitigating biases and hallucination issues associated with open-ended responses.

### Key Designs

1. **Task taxonomy of 8 categories and 31 sub-tasks**:

    - **Scene Understanding**: Scene classification, land-use classification, crop classification.
    - **Object Classification**: Fine-grained classification of ship types, aircraft types, etc.
    - **Object Localization & Counting**: Referring expression detection and counting of various objects (vehicles, aircraft, buildings, water bodies, trees, marine debris, etc.).
    - **Event Detection**: Fire risk assessment, disaster type classification.
    - **Caption Generation**: Image captioning to assess scene and detail description capabilities.
    - **Semantic Segmentation**: Referring expression segmentation, generating binary masks for specific targets.
    - **Temporal Understanding**: Change detection, disaster damage assessment, long-term crop classification.
    - **Non-Optical Data**: SAR-based ship detection, flood detection, earthquake magnitude estimation.
    - **Design Motivation**: To cover the full spectrum of remote sensing application scenarios.

2. **Data construction pipeline**:

    - Open-source remote sensing datasets are integrated, with each task sampling from multiple datasets to ensure diversity.
    - For classification tasks, GPT-4o generates five-option MCQs consisting of one correct answer, one semantically similar "nearest distractor" (manually verified), and three plausible distractors.
    - Counting tasks convert detection data into questions, providing the correct count alongside options with ±20% and ±40% deviations.
    - Spatial relationship tasks are annotated by human annotators with cross-validation.
    - Caption generation combines GPT-4o generation with manual refinement.
    - **Design Motivation**: Combining automatic generation with manual verification to ensure data quality.

3. **Comprehensive VLM evaluation framework**:

    - Thirteen state-of-the-art VLMs are evaluated, including general-purpose models (GPT-4o, LLaVA-OneVision, Qwen2-VL, InternVL2, etc.) and remote sensing-specific models (GeoChat, RS-LLaVA, EarthDial, etc.).
    - Multi-dimensional metrics are employed: MCQ accuracy, detection precision, segmentation mIoU, and caption BERTScore.
    - **Design Motivation**: To simultaneously evaluate both general-purpose and domain-specific models, comprehensively revealing capability gaps.

### Loss & Training

As a benchmark paper, no training is involved. All VLMs are evaluated in a zero-shot inference setting directly on GEOBench-VLM.

## Key Experimental Results

### Main Results — VLM Accuracy Across Task Categories

| Model | Event Detection | Object Classification | Counting | Scene Understanding | Caption (BERT) |
|---|---|---|---|---|---|
| GPT-4o | 0.473 | **0.586** | 0.397 | 0.711 | 0.642 |
| EarthDial | **0.542** | 0.404 | 0.363 | **0.771** | 0.538 |
| Qwen2-VL | 0.464 | 0.456 | 0.402 | 0.676 | 0.590 |
| LLaVA-OneVision | 0.406 | 0.459 | **0.438** | 0.664 | 0.632 |
| InternVL-2 | 0.346 | 0.306 | 0.328 | 0.573 | 0.597 |
| GeoChat | 0.337 | 0.313 | 0.292 | 0.609 | 0.440 |
| SPHINX | 0.236 | 0.205 | 0.186 | 0.217 | **0.645** |

### Ablation Study — Referring Expression Detection Precision

| Model | Prec@0.5 | Prec@0.25 |
|---|---|---|
| **SPHINX** | **0.341** | **0.529** |
| EarthDial | 0.243 | 0.414 |
| Qwen2-VL | 0.152 | 0.252 |
| GeoChat | 0.115 | 0.210 |
| GPT-4o | 0.009 | 0.039 |

### Key Findings

- **Best-performing models remain limited**: LLaVA-OneVision ranks first with an average MCQ accuracy of 41.7%, only slightly above twice the random-guess baseline.
- **No model achieves comprehensive dominance**: GPT-4o excels at object classification, EarthDial at scene understanding and event detection, and LLaVA-OneVision at counting.
- **Remote sensing-specific models do not always outperform**: General-purpose models surpass domain-specific models on multiple tasks.
- **Counting tasks pose major challenges**: Accuracy drops substantially for all models in high-density scenes (>50 objects).
- **Temporal information is underutilized**: Multi-temporal data degrades performance on some tasks, indicating that current VLMs struggle to leverage temporal dependencies.
- **GPT-4o performs worst on localization** (Prec@0.5 of only 0.009) while excelling at object classification.
- **Prompt sensitivity**: GPT-4o and InternVL2 are most sensitive to prompt variations, whereas EarthDial and SkySenseGPT are comparatively stable.

## Highlights & Insights

1. **Filling a critical gap**: This is the first comprehensive geospatial VLM benchmark spanning 8 categories and 31 sub-tasks, including previously absent categories such as temporal analysis, non-optical data, and segmentation.
2. **Manual verification ensures quality**: Over 10,000 instructions are manually verified, and the MCQ format reduces evaluation bias.
3. **In-depth analyses are informative**: Analyses of object density vs. counting accuracy, prompt sensitivity, and single- vs. multi-temporal comparisons reveal deeper limitations of current VLMs.
4. **Open-source and extensible**: The benchmark is publicly available, facilitating iterative follow-up research.

## Limitations & Future Work

- The MCQ format limits assessment of VLMs' open-ended generation capabilities.
- Only a subset of models supports certain tasks (e.g., segmentation), restricting the scope of comparison.
- Inference latency and computational efficiency are not evaluated.
- Data sources are predominantly public remote sensing datasets, which may introduce distributional bias.
- Future work could incorporate 3D geospatial understanding and multimodal fusion tasks (e.g., joint optical and SAR reasoning).

## Related Work & Insights

- Complements the VLEO benchmark with significant extensions in temporal analysis, segmentation, and non-optical data.
- Reveals limitations of fine-tuning remote sensing-specific VLMs, as domain-specific models do not consistently outperform general-purpose ones across tasks.
- In-depth analyses of counting and localization tasks can guide architectural improvements for remote sensing VLMs.
- Provides concrete performance targets for the development of next-generation geospatial-specific VLMs.

## Rating

- **Novelty**: ⭐⭐⭐ A benchmark paper with creative task design and data construction, though methodological innovation is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluates 13 VLMs across 31 tasks with rich analytical dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, though some tables are data-dense and moderately difficult to read.
- **Value**: ⭐⭐⭐⭐ Provides the geospatial AI community with a much-needed standardized evaluation tool of high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](../../ACL2026/multimodal_vlm/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](../../NeurIPS2025/multimodal_vlm/mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)

<!-- RELATED:END -->
