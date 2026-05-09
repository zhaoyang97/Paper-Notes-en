---
title: >-
  [Paper Note] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video
description: >-
  [NeurIPS 2025][Multimodal VLM][real-time video understanding] This paper proposes RTV-Bench, a fine-grained evaluation benchmark for assessing the continuous real-time video analysis capabilities of MLLMs. Comprising 552 videos and 4,608 QA pairs, it comprehensively evaluates model perception, understanding, and reasoning in dynamic video streams through a multi-timestamp QA mechanism, hierarchical question structure, and multi-dimensional assessment.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - real-time video understanding
  - multimodal large model evaluation
  - continuous analysis
  - multi-timestamp QA
  - video benchmark
date: 2026-05-08
content_hash: 4db29fdfa73fd00c
---

# RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video

**Conference**: NeurIPS 2025
**arXiv**: [2505.02064](https://arxiv.org/abs/2505.02064)
**Code**: [https://ljungang.github.io/RTV-Bench](https://ljungang.github.io/RTV-Bench)
**Area**: Multimodal VLM
**Keywords**: real-time video understanding, multimodal large model evaluation, continuous analysis, multi-timestamp QA, video benchmark

## TL;DR
This paper proposes RTV-Bench, a fine-grained evaluation benchmark for assessing the continuous real-time video analysis capabilities of MLLMs. Comprising 552 videos and 4,608 QA pairs, it comprehensively evaluates model perception, understanding, and reasoning in dynamic video streams through a multi-timestamp QA mechanism, hierarchical question structure, and multi-dimensional assessment.

## Background & Motivation

**State of the Field**: MLLMs have advanced rapidly in perception, understanding, and reasoning, yet existing benchmarks primarily evaluate static or offline video understanding, making it difficult to measure model performance on continuous dynamic video streams.

**Limitations of Prior Work**: While benchmarks such as VStream, StreamingBench, and OVOBench have improved upon video length and evaluation types, they remain insufficient for assessing real-time responsiveness — in particular, they overlook a model's ability to capture transitional and momentary details within visual input.

**Root Cause**: Real-world video is continuously evolving, and the correct answer to a given question may differ depending on the timestamp at which it is posed. Existing benchmarks typically ask questions at a single point in time and are therefore unable to test model sensitivity to dynamic state changes.

**Paper Goals**: To design a benchmark that comprehensively evaluates MLLM continuous analysis capabilities in real-time video scenarios, spanning three levels: perception, understanding, and reasoning.

**Starting Point**: Three core innovations are introduced — a multi-timestamp QA mechanism, a hierarchical question structure, and a multi-dimensional evaluation system — to construct a more rigorous assessment of real-time video understanding.

**Core Idea**: The same conceptual question is posed repeatedly at different timestamps throughout a video, with the correct answer changing as the scene evolves. This directly tests a model's ability to continuously track temporal states and update its understanding accordingly.

## Method

### Overall Architecture
RTV-Bench consists of 552 diverse videos (total duration 167.2 hours, average 18.2 minutes per video) and 4,608 carefully annotated QA pairs. Videos are drawn primarily from three domains — autonomous driving, sports events, and egocentric footage — spanning 16 subcategories.

### Key Designs

1. **Multi-Timestamp QA Mechanism (MTQA)**:

    - Function: Evaluates a model's ability to track dynamic video changes in real time.
    - Mechanism: The same conceptual question is asked at multiple timestamps throughout the video. For example, "What is the goalkeeper doing?" may have different correct answers — "diving," "standing," or "kicking" — as the match progresses. Annotators assign the earliest valid timestamp to each answer option.
    - Design Motivation: Unlike OVOBench, which poses different questions at different timestamps, MTQA reuses the same question across time, imposing a more rigorous test of continuous analysis capability.

2. **Hierarchical Question Structure**:

    - Function: Ensures models possess reliable sequential reasoning ability.
    - Mechanism: Each question group contains approximately three multiple-choice questions. The first two address basic perception and understanding, while the third requires higher-order reasoning that integrates contextual information. The advanced question is logically dependent on correct responses to the foundational ones.
    - Design Motivation: Prevents models from arriving at correct answers via cognitive shortcuts, ensuring that higher-order reasoning is grounded in solid foundational understanding.

3. **Multi-Dimensional Evaluation System**:

    - Function: Provides fine-grained diagnostic assessment of model capabilities.
    - Mechanism: Evaluation is structured across eight dimensions — Temporal Perception (TP), Scene Perception (SP), Visual Perception (VP), Future Prediction (FP), Phenomenon Understanding (PU), Intent Analysis (IA), Global Understanding (GU), and Spatiotemporal Reasoning (SR). A conditional Score metric is introduced: credit for advanced questions is awarded only when all prerequisite foundational questions are answered correctly.
    - Design Motivation: To go beyond a single aggregate score and provide a more informative view of model capabilities and limitations.

### Loss & Training
RTV-Bench is an evaluation benchmark and does not involve model training. The annotation pipeline employs DeepSeek to generate initial question templates, which are subsequently refined by human annotators to reflect the demands of dynamic scenes, ensuring high annotation quality.

## Key Experimental Results

### Main Results

| Model | Type | Overall Acc (%) | Score | MTQA Acc (%) |
|-------|------|-----------------|-------|--------------|
| GPT-4o | Closed-source | 50.02 | 22.10 | 44.73 |
| IXC2.5-OL | Online 7B | 47.33 | 15.40 | 38.21 |
| VITA-1.5 | Online 7B | 44.51 | 11.80 | 36.32 |
| Qwen2.5-VL | Offline 7B | 40.41 | 7.13 | 37.46 |
| VideoLLaMA2 | Offline 7B | 39.55 | 7.90 | 34.95 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Online vs. offline models | +7.78% Acc | Online models significantly outperform offline models |
| Increasing frame count | Non-monotonic improvement | More frames do not consistently improve performance |
| Model scale | Positive correlation | Larger models generally perform better |

### Key Findings
- Most models achieve accuracy below 50%, indicating that real-time video understanding remains a substantial challenge.
- Online models (e.g., IXC2.5-OL) significantly outperform offline models on MTQA tasks, yet still fall considerably short of GPT-4o.
- The benefit of increasing frame sampling density is non-monotonic, suggesting that simply adding more frames cannot resolve the problem and that purpose-built streaming architectures are needed.

## Highlights & Insights
- The MTQA design is notably innovative: the same question posed at different timestamps yields different correct answers, which tests continuous analysis capability more rigorously than the conventional paradigm of asking different questions at different timestamps. This design principle is transferable to other dynamic scenario evaluations.
- The conditional Score metric elegantly prevents "spurious success," where a model correctly answers an advanced question while failing the foundational ones, thereby improving the reliability of the evaluation.

## Limitations & Future Work
- Video sources are skewed toward three domains — driving, sports, and egocentric footage — limiting diversity.
- Evaluation relies exclusively on multiple-choice questions; open-ended QA is not considered.
- Future work may extend coverage to a broader range of scenario types and incorporate evaluation of model latency and real-time response speed.

## Related Work & Insights
- **vs. StreamingBench**: Although StreamingBench evaluates real-time scenarios, its question design is relatively simple and lacks a mechanism for reusing questions across multiple timestamps.
- **vs. OVOBench**: OVOBench poses different questions at different timestamps, whereas RTV-Bench uses the same question to test dynamic tracking, resulting in a more rigorous evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-timestamp QA and hierarchical evaluation designs are genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers a broad range of online, offline, and closed-source models.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with well-articulated motivation.
- Value: ⭐⭐⭐⭐ Fills an important gap in real-time video understanding evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] rtv-bench benchmarking mllm continuous perception understanding and reasoning th](rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)
- [\[NeurIPS 2025\] in the eye of mllm benchmarking egocentric video intent understanding with gaze-](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[NeurIPS 2025\] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images](gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] DynamicVL: Benchmarking MLLMs for Dynamic City Understanding](dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)

<!-- RELATED:END -->
