---
title: >-
  [Paper Note] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?
description: >-
  [ICCV 2025][Multimodal VLM][MLLM benchmark] This paper proposes STI-Bench, a benchmark for evaluating the precise spatial-temporal understanding capabilities of multimodal large language models (MLLMs), covering three scene categories (desktop/indoor/outdoor), eight static and dynamic task types, and over 2,000 QA pairs. The benchmark reveals that the current state-of-the-art MLLM (Gemini-2.5-Pro) achieves an average accuracy of only 41.4%, exposing fundamental deficiencies in precise spatial quantification and temporal dynamic understanding.
tags:
  - ICCV 2025
  - Multimodal VLM
  - MLLM benchmark
  - spatial-temporal understanding
  - embodied AI
  - video QA
  - autonomous driving
date: 2026-05-08
content_hash: b47001871c8dbee4
---

# STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?

**Conference**: ICCV 2025
**arXiv**: [2503.23765](https://arxiv.org/abs/2503.23765)
**Code**: [Project Page](https://mint-sjtu.github.io/STI-Bench.io/)
**Area**: Multimodal VLM
**Keywords**: MLLM benchmark, spatial-temporal understanding, embodied AI, video QA, autonomous driving

## TL;DR

This paper proposes STI-Bench, a benchmark for evaluating the precise spatial-temporal understanding capabilities of multimodal large language models (MLLMs), covering three scene categories (desktop/indoor/outdoor), eight static and dynamic task types, and over 2,000 QA pairs. The benchmark reveals that the current state-of-the-art MLLM (Gemini-2.5-Pro) achieves an average accuracy of only 41.4%, exposing fundamental deficiencies in precise spatial quantification and temporal dynamic understanding.

## Background & Motivation

MLLMs are increasingly adopted as end-to-end solutions for embodied AI and autonomous driving, yet a critical gap has been largely overlooked:

**The gap between semantic understanding and precise spatial-temporal understanding**: Existing evaluations primarily focus on 2D visual perception and semantic QA, whereas embodied tasks require precise 3D spatial measurements and physical motion understanding.

**Limitations of prior benchmarks**:
- VSI-Bench covers only limited scenes and task types
- EmbodiedBench and similar works rely on simulated environments
- No benchmark simultaneously covers static spatial measurement and dynamic motion analysis using real-world data

**Core question**: **Are MLLMs truly ready for precise spatial-temporal world understanding?**

Design rationale: Video (rather than point clouds) is used as input because (1) mainstream models such as GPT-4o and Gemini accept image/video input, and (2) video is more prevalent in everyday settings and contains sufficient information for spatial-temporal reasoning.

## Method

### Overall Architecture

STI-Bench comprises 300+ real-world videos and 2,064 QA pairs, spanning three scene categories (Desktop/Indoor/Outdoor) and eight task types. Questions are presented in a five-choice format, with a random baseline accuracy of 20%.

### Key Designs

1. **Eight-task taxonomy**: Tasks are divided into static understanding (3 types) and dynamic understanding (5 types):

   **Static Understanding**:
   - **Dimension Measurement (Dim. Meas.)**: Inferring physical dimensions from 2D pixels, e.g., "How tall is this box?"
   - **Spatial Relation**: Determining relative positions among objects, e.g., "Is the chair to the left or right of the table?"
   - **3D Video Grounding**: Retrieving 3D bounding boxes given semantic descriptions

   **Dynamic Understanding**:
   - **Displacement & Path Length (Disp. & P.L.)**: Tracking motion distances across frames
   - **Speed & Acceleration (Speed & Acc.)**: Computing motion parameters by integrating spatial displacement and temporal intervals
   - **Egocentric Orientation (Ego Orient.)**: Understanding horizontal azimuth changes of the camera
   - **Trajectory Description (Traj. Desc.)**: Abstracting motion paths into natural language
   - **Pose Estimation (Pose Est.)**: Estimating camera pose at a specific moment given an initial pose

2. **Data sources and multi-scale design**: Three datasets covering different spatial scales:

   - **Waymo** (outdoor): Autonomous driving scenes, decimeter-to-meter precision
   - **ScanNet** (indoor): Indoor 3D scene reconstruction, centimeter-to-decimeter precision
   - **Omni6DPose** (desktop): 6D object pose estimation, millimeter precision

3. **Refined distractor design**: Different error ranges are specified per scene type (Desktop: 0.5–5 cm, Indoor: 5–50 cm, Outdoor: 0.5–5 m), with distractors generated via logarithmic sampling:

   $$e = E_{min} \cdot (E_{max}/E_{min})^u, \quad u \sim \mathcal{U}(0,1)$$

   Weighted averaging is applied to adjust the distance between distractors and the correct answer, ensuring the minimum distance equals the sampled error value $e$. Design motivation: precision requirements vary greatly across scenes, and a uniform distractor generation strategy would render some scenes trivially easy or prohibitively difficult.

4. **Construction pipeline**: Dataset annotation → automatic QA generation (MLLM-assisted) → multi-round human quality control → option randomization

### Loss & Training

This is a benchmark paper and does not involve model training. Evaluation protocol: 30 frames are uniformly sampled (20 for Claude), the sampling FPS is specified in the prompt, and model selections are directly compared against ground truth.

## Key Experimental Results

### Main Results (All Models)

| Model | Rank | Avg | Dim.Meas. | Spatial Rel. | 3D Ground. | Disp.&PL | Speed&Acc | Ego Orient. | Traj.Desc. | Pose Est. |
|------|------|-----|-----------|-------------|------------|----------|-----------|-------------|------------|-----------|
| Gemini-2.5-Pro | 1 | 41.4 | 38.7 | 53.8 | 36.9 | 33.9 | 33.1 | **52.5** | 47.4 | 50.4 |
| Qwen2.5-VL-72B | 2 | 40.7 | 31.5 | 47.6 | 39.1 | 25.1 | 38.4 | 43.8 | 51.3 | 60.6 |
| Claude-3.7-Sonnet | 3 | 40.5 | 29.8 | 45.5 | 35.7 | 28.9 | 38.8 | 40.0 | 47.4 | **62.6** |
| GPT-4o | 8 | 34.8 | 27.1 | 51.8 | 29.0 | 23.2 | 35.4 | 33.7 | 32.0 | 53.6 |
| MiniCPM-V-2.6 | 10 | 26.9 | 27.7 | 44.5 | 29.0 | 19.0 | 25.7 | 7.0 | 30.8 | 35.6 |

- The best model, Gemini-2.5-Pro, achieves an average of only **41.4%** (random baseline: 20%)
- Precise quantification tasks are the most challenging: Dim. Meas. 38.7%, Disp. & P.L. 33.9%, Speed & Acc. 33.1%
- Relative understanding tasks fare better: Spatial Relation 53.8%, Ego Orient. 52.5%

### Cross-Scene Comparison

| Model | Outdoor | Indoor | Desktop | Overall |
|------|---------|--------|---------|---------|
| Qwen2.5-VL-72B | **50.6** | 35.0 | 33.5 | 40.7 |
| Gemini-2.5-Pro | 48.7 | 37.1 | **35.8** | **41.4** |
| Claude-3.7-Sonnet | 47.2 | **38.2** | 32.3 | 40.5 |
| GPT-4o | 41.4 | 33.1 | 27.3 | 35.2 |

- All models perform best on **outdoor** scenes and worst on **desktop** scenes
- This is likely attributable to training data distribution (autonomous driving data vastly outnumbers desktop manipulation data)

### Ablation / Error Analysis

| Error Type | Proportion | Affected Tasks |
|---------|------|---------|
| Inaccurate spatial quantification | ~40% | Dim. Meas., distance estimation, 3D grounding |
| Temporal dynamic understanding errors | ~35% | Displacement, speed, trajectory description |
| Insufficient cross-modal fusion | ~25% | All tasks relying on instructions or initial conditions |

### Key Findings

- **MLLMs are far from mature in "precise spatial-temporal understanding"**: The strongest model achieves only 41.4%, with many tasks approaching random guessing
- **Three fundamental deficiencies**:
  1. **Inaccurate spatial quantification**: Lack of visual scale references; difficulty mapping 2D pixels to physical measurements
  2. **Temporal dynamic understanding errors**: Inability to effectively track cross-frame motion or distinguish camera motion from object motion
  3. **Insufficient cross-modal fusion**: Misinterpretation of temporal constraints (e.g., "from 1s to 18s"); inability to integrate initial conditions
- **Significant model scale effect**: 72B/78B models (40%+) substantially outperform 7B models (26–35%)
- **Outdoor > Indoor > Desktop**: Likely reflects training data bias

## Highlights & Insights

- **Fills the gap in precise spatial-temporal evaluation**: Prior benchmarks focus either on semantics or on only a subset of tasks; STI-Bench is the first to systematically cover eight precise spatial-temporal task types
- **Carefully designed multi-scale scene coverage**: From millimeter-scale desktop to meter-scale outdoor scenes, comprehensively testing precision across different spatial scales
- **In-depth error analysis**: Beyond reporting numbers, the paper analyzes three fundamental error patterns through Gemini-2.5-Pro's reasoning traces, providing actionable directions for future improvement
- **Logarithmic distractor sampling**: Ensures uniform distribution of option differences, avoiding artificially trivial or excessively difficult questions

## Limitations & Future Work

- The scale of 2,000+ QA pairs is relatively small and may be insufficient for statistically significant comparisons
- Only video input with uniform 30-frame sampling is evaluated; the impact of different sampling strategies remains unexplored
- Multi-sensor fusion scenarios (e.g., LiDAR + Camera) are not included
- Desktop scene data derives from a single source (Omni6DPose), limiting diversity
- Additional dynamic tasks such as object interaction prediction and collision prediction could be incorporated

## Related Work & Insights

- **VSI-Bench** is the closest predecessor but covers only limited scenes and spatial tasks
- Results indicate serious deficiencies in MLLMs' "world model" capabilities, serving as a cautionary signal for research directions that rely on MLLMs for end-to-end autonomous driving or robotic manipulation
- Future improvement directions: incorporating 3D geometric priors (depth estimation pretraining), physics-engine-assisted training, and denser frame sampling

## Rating

- Novelty: ⭐⭐⭐⭐ — First benchmark to comprehensively cover precise spatial-temporal understanding
- Technical Depth: ⭐⭐⭐ — Limited by the nature of benchmark papers, though the task design and distractor generation are thoughtfully conceived
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 models (including 4 closed-source SOTAs) + cross-scene + cross-task + in-depth error analysis
- Value: ⭐⭐⭐⭐ — Provides an important evaluation tool and improvement directions for the embodied AI and autonomous driving communities

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ICCV 2025\] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?](advdreamer_unveils_are_visionlanguage_models_truly_ready_for.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[AAAI 2026\] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](../../AAAI2026/multimodal_vlm/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)

</div>

<!-- RELATED:END -->
