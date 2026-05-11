---
title: >-
  [Paper Note] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark
description: >-
  [NeurIPS 2025][Video Understanding][Egocentric Video] This paper introduces EgoGazeVQA, the first egocentric video question answering benchmark that incorporates user eye-gaze data. Through gaze-guided prompting strategi…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Egocentric Video"
  - "Gaze Guidance"
  - "Video Question Answering"
  - "User Intent Understanding"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: c44d9789cd4444b5
---

# EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark

**Conference**: NeurIPS 2025  
**arXiv**: [2509.07447](https://arxiv.org/abs/2509.07447)  
**Code**: [https://taiyi98.github.io/projects/EgoGazeVQA](https://taiyi98.github.io/projects/EgoGazeVQA)  
**Area**: Video Understanding  
**Keywords**: Egocentric Video, Gaze Guidance, Video Question Answering, User Intent Understanding, Multimodal Large Language Models

## TL;DR
This paper introduces EgoGazeVQA, the first egocentric video question answering benchmark that incorporates user eye-gaze data. Through gaze-guided prompting strategies (textual, visual, and salience map), the benchmark demonstrates substantial improvements in MLLMs' ability to understand user intent. The Gaze Salience Map strategy raises MiniCPM-o's accuracy from 35.9% to 53.7%.

## Background & Motivation

**Background**: MLLMs have achieved significant progress in video understanding; however, existing benchmarks are predominantly based on third-person perspectives and cannot directly capture user attention focus or behavioral intent.

**Limitations of Prior Work**: Existing egocentric video QA benchmarks (e.g., QaEgo4D, EgoSchema) overlook a critical first-person signal — gaze. Gaze directly reflects user attention and intent, and the majority of user queries are fundamentally dependent on what the user is looking at.

**Key Challenge**: MLLMs construct visual tokens from global image frames, providing broad context while failing to capture the explicit intent signals of the camera wearer, making it difficult for models to accurately infer what the user is viewing or intending to do.

**Goal**: To construct the first egocentric VQA benchmark that integrates gaze data, and to evaluate whether MLLMs can leverage gaze information to enhance understanding of user intent.

**Key Insight**: Gaze coordinate information is incorporated into MLLM prompts via three distinct modalities — textual, visual markers, and salience maps.

**Core Idea**: Gaze signals represent a critically missing modality for understanding user intent in egocentric video; gaze-guided prompting can substantially compensate for MLLMs' deficiencies in intent understanding.

## Method

### Overall Architecture
Video clips and gaze coordinates are extracted from three egocentric video datasets with eye-tracking data — Ego4D, EgoExo4D, and EGTEA Gaze+. Qwen2.5-VL is used to generate spatially and temporally aware, intent-related QA pairs, which are then manually reviewed to form a benchmark dataset comprising 913 videos and 1,757 QA pairs.

### Key Designs

1. **Data Construction Pipeline**:

    - Function: Generate high-quality gaze-guided QA pairs.
    - Mechanism: Every 9 frames constitute a keyframe segment, paired with normalized gaze coordinates and fed into Qwen2.5-VL to generate 3 QA pairs (each with 5 choices). Manual review covers six dimensions: relevance, answerability, fluency, accuracy, conciseness, and difficulty.
    - Design Motivation: Distractor options include counterfactual choices, spatially proximate traps, and high-salience distractors, ensuring that correct reasoning requires leveraging gaze information.

2. **Three Gaze-Guided Prompting Strategies**:

    - Function: Encode gaze information into MLLM inputs via different representations.
    - Mechanism: (1) Textual prompt (T): gaze coordinates are directly provided as text; (2) Visual prompt (V): gaze points are annotated on video frames; (3) Salience map (S): gaze trajectories are rendered as heatmaps and provided as additional visual context.
    - Design Motivation: Different MLLMs may exhibit varying sensitivity to different forms of gaze encoding, necessitating systematic comparison.

3. **LoRA Fine-Tuning Experiments**:

    - Function: Assess whether fine-tuning can bridge the gap in MLLMs' understanding of gaze signals.
    - Mechanism: Since gaze signals are rarely present in MLLM training data, LoRA fine-tuning is applied to help models learn to utilize gaze cues.
    - Design Motivation: To determine whether gaze comprehension capability can be acquired through lightweight fine-tuning.

### Loss & Training
No training strategy is involved in the benchmark evaluation. LoRA fine-tuning experiments employ the standard instruction-following loss.

## Key Experimental Results

### Main Results

| Model | No Gaze | +Text (T) | +Visual (V) | +Salience Map (S) |
|-------|---------|-----------|-------------|-------------------|
| InternVL2.5-8B | 58.3% | 60.1% | **60.6%** | 59.9% |
| GPT-4o mini | 57.0% | 58.8% | 58.5% | 58.7% |
| MiniCPM-o 2.6 | 35.9% | 50.0% | 50.2% | **53.7%** |
| Human Baseline | - | - | - | 83.8% |

### Ablation Study

| Dimension | No Gaze | Best Gaze Strategy | Gain |
|-----------|---------|-------------------|------|
| Spatial | 36.1–50.4% | 49.4–55.0% | +3–13% |
| Temporal | 34.5–51.1% | 41.6–52.7% | +2–7% |
| Causal | 32.5–75.6% | 64.4–80.3% | +5–32% |

### Key Findings
- MiniCPM-o achieves the largest gain (+17.8%), indicating that weaker models benefit more substantially from gaze signals.
- The salience map strategy performs best overall, as it simultaneously encodes both spatial and temporal information of the gaze trajectory.
- The causal reasoning dimension benefits most markedly, since gaze directly points to the causes and intentions behind user actions.
- The substantial gap between the human baseline (83.8%) and the best model (60.6%) indicates considerable room for improvement in gaze-based understanding.

## Highlights & Insights
- This work is the first to incorporate gaze data into a VQA benchmark, filling an important gap in egocentric video understanding. Gaze as direct evidence of user intent has long been undervalued.
- The success of the salience map strategy suggests that converting sparse point coordinates into dense spatial priors (heatmaps) can be more effectively leveraged by vision-based models.

## Limitations & Future Work
- The dataset scale is relatively small (1,757 QA pairs), which may limit the reliability of statistical conclusions.
- Gaze data requires specialized hardware for collection, constraining practical applicability.
- Future work may explore automated pipelines that estimate gaze direction from standard video and apply it to intent understanding.

## Related Work & Insights
- **vs. EgoSchema**: EgoSchema focuses on long-video reasoning but does not leverage gaze data; EgoGazeVQA treats gaze as a core signal.
- **vs. GazeGPT**: GazeGPT demonstrates the utility of gaze for MLLM user interfaces but lacks a standardized benchmark and evaluation protocol; this work addresses that gap.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First gaze-guided VQA benchmark with a novel entry point.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across multiple models, strategies, and dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with convincing motivation.
- Value: ⭐⭐⭐⭐ Offers important guidance for the development of egocentric AI assistants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](../../CVPR2026/video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[CVPR 2026\] EgoPointVQA: Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[NeurIPS 2025\] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)

</div>

<!-- RELATED:END -->
