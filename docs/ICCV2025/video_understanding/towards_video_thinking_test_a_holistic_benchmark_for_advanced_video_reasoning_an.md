---
title: >-
  [Paper Note] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding
description: >-
  [ICCV 2025][Video Understanding][Video Understanding Benchmark] This paper introduces Video Thinking Test (Video-TT), a benchmark for evaluating both the correctness and robustness of video large language models (Video LLMs). It comprises 1,000 YouTube Shorts videos and 5,000 questions, designed around visual/narrative complexity factors and natural adversarial question variants. The benchmark reveals a substantial gap between the best-performing model (GPT-4o, 36.6%) and humans (84.3%).
tags:
  - ICCV 2025
  - Video Understanding
  - Video Understanding Benchmark
  - Video LLM
  - Adversarial Robustness
  - Visual Complexity
  - Narrative Complexity
date: 2026-05-08
content_hash: e74a6f50d9b290d0
---

# Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding

**Conference**: ICCV 2025
**arXiv**: [2507.15028](https://arxiv.org/abs/2507.15028)
**Code**: [https://github.com/zhangyuanhan-ai/video-tt](https://github.com/zhangyuanhan-ai/video-tt)
**Area**: Video Understanding
**Keywords**: Video Understanding Benchmark, Video LLM, Adversarial Robustness, Visual Complexity, Narrative Complexity

## TL;DR

This paper introduces Video Thinking Test (Video-TT), a benchmark for evaluating both the correctness and robustness of video large language models (Video LLMs). It comprises 1,000 YouTube Shorts videos and 5,000 questions, designed around visual/narrative complexity factors and natural adversarial question variants. The benchmark reveals a substantial gap between the best-performing model (GPT-4o, 36.6%) and humans (84.3%).

## Background & Motivation

Video LLMs are approaching human-level intelligence in many respects, yet existing benchmarks fail to accurately characterize the gap between models and humans, for two fundamental reasons:

**Misleading correctness evaluation**: Existing benchmarks (e.g., VideoMME, MVBench) cannot distinguish between errors caused by insufficient frame sampling and errors caused by genuine comprehension deficits. Since Video LLMs typically sample a limited number of frames before reasoning, large performance gaps on long-video benchmarks may merely reflect limitations in sampling strategies. Conversely, on short-video benchmarks (e.g., VideoMME-Short), model performance approaches human-level ceilings, creating the false impression that models have reached human capability — when in fact the questions are simply not challenging enough.

**Distorted robustness evaluation**: Existing robustness studies primarily test artificially introduced perturbations (e.g., pixel modifications, spelling errors), which are overly contrived and do not reflect real-world complexity. What needs to be evaluated is robustness under **natural adversarial conditions** — specifically, whether a model maintains consistent answers when the same question is posed from different angles.

The core design philosophy of Video-TT is: (1) ensure each question is sufficiently complex to differentiate human and model comprehension; (2) ensure questions are answerable under a fixed frame budget, decoupling frame sampling from comprehension; and (3) evaluate robustness via natural adversarial question variants.

## Method

### Overall Architecture

Video-TT consists of 1,000 YouTube Shorts videos, each paired with one primary open-ended question and four adversarial variant questions (5,000 QA pairs in total). All questions are designed around eight visual/narrative complexity factors and are verified to be answerable under uniform 80-frame sampling.

### Key Designs

1. **Visual Complexity Factors**:

    - Function: Define visual factors that make video content difficult to understand.
    - Mechanism: Drawing on cognitive science theories of visual complexity, four factor categories are identified — (a) **Unclear/atypical content**: content that deviates from common scenes, or contains noise, blur, or occlusion; (b) **Motion speed**: video or camera motion too fast to track; (c) **Spatiotemporal arrangement**: large numbers of objects, complex interactions, and high cognitive load; (d) **Visual illusions**: techniques that deliberately create perceptual illusions.
    - Design Motivation: Question difficulty depends not only on question type (e.g., "object color" vs. "plot comprehension") but critically on **contextual and scene conditions**. For example, "What is the color of the second car?" becomes extremely difficult when the car is moving rapidly and partially occluded.

2. **Narrative Complexity Factors**:

    - Function: Define comprehension challenges that go beyond linear narrative.
    - Mechanism: Four factor categories — (a) **Complex plots**: containing twists or unexpected endings; (b) **Narrative editing**: complex shot combinations such as montage; (c) **Technical editing**: special filming techniques or post-production effects; (d) **World knowledge**: requiring prior knowledge for full comprehension.
    - Design Motivation: These forms of complexity demand deeper viewer engagement with the video content.

3. **Natural Adversarial Question Design**:

    - Function: Test model robustness to different phrasings of the same question.
    - Mechanism: Each primary question yields four variants — (a) **Rephrased open-ended**: semantic paraphrase; (b) **Correctly-led open-ended**: provides a correct cue; (c) **Wrongly-led open-ended**: provides a misleading cue; (d) **Multiple-choice**: mixes correct and incorrect options.
    - Design Motivation: A model that truly understands a video should be robust to different formulations of the same question. Robustness is defined as the proportion of videos where all five questions are answered correctly, divided by the proportion where only the primary question is answered correctly.

4. **Annotation Quality Control**:

    - Function: Ensure annotation quality and consistency.
    - Mechanism: Multi-layer quality assurance — (a) each question must involve at least one complexity factor; (b) difficulty is validated using GPT-4o, LLaVA-Video-7B, and Qwen2.5-VL-7B (at least one model must answer incorrectly at least once across three attempts); (c) annotators must provide reasoning traces and explain model errors; (d) questions must be answerable under 80-frame sampling; (e) three-annotator agreement checks.
    - Design Motivation: Strictly decouple frame sampling limitations from comprehension deficits.

### Evaluation Metrics

- **Correctness Score**: Open-ended responses are scored by Qwen2.5-72B on a 0–5 scale (score > 3 is counted as correct).
- **Robustness Score**: Among videos where the primary question is answered correctly, the proportion where all 5 questions are also answered correctly.

## Key Experimental Results

### Main Results

| Model | Primary | Rephrased | Correctly-Led | Wrongly-Led | Multi-Choice | Avg | Robustness |
|-------|---------|-----------|---------------|-------------|-------------|-----|------------|
| Qwen2.5-VL-7B | 20.9 | 22.5 | 45.3 | 39.3 | 39.9 | 33.6 | 14.4 |
| LLaVA-Video-72B | 24.4 | 25.7 | 57.7 | 32.6 | 47.5 | 37.6 | 19.7 |
| Qwen2.5-VL-72B | 26.6 | 25.7 | 31.1 | 49.8 | 45.6 | 35.8 | 22.2 |
| Gemini Pro | 28.8 | 29.7 | 50.2 | 29.2 | 42.3 | 38.2 | 20.5 |
| **GPT-4o** | **36.6** | **35.4** | **67.5** | **39.8** | **46.6** | **45.2** | **36.0** |
| **Human** | **84.3** | **83.9** | **83.9** | **76.2** | **87.5** | **83.2** | **64.4** |

### Ablation Study (GPT-4o Error Analysis and Augmentation Experiments)

| Analysis Dimension | Result | Notes |
|-------------------|--------|-------|
| Frame count 8→64 (Human) | Accuracy rises steadily toward ceiling | Human comprehension improves consistently with more frames |
| Frame count 8→64 (Model) | Saturates at ~8 frames | Additional frames do not aid model comprehension |
| CoT on Wrongly-Led | +6.8% relative gain | Structured reasoning helps resist misleading cues |
| CoT on Multi-Choice | No notable gain | Structured format yields limited benefit |
| Audio transcription on robustness | +15% relative gain | Multimodal information enhances robustness |
| Spatiotemporal confusion error rate (element/event localization) | 79% / 88% | Largest model weakness |
| World knowledge deficit (character motivation) | 44% errors | Lack of contextual knowledge |
| Complex plot confusion (attribute/causality) | 55% errors | Insufficient causal reasoning |

### Key Findings

- **Large human–model gap**: GPT-4o correctness 36.6% vs. human 84.3%; robustness 36.0% vs. 64.4% — a gap rarely observed in existing benchmarks.
- **Open-source vs. closed-source robustness gap**: Qwen2.5-VL-72B robustness (22.2%) is 13.8 points below GPT-4o (36.0%), a gap far exceeding the difference on correctness alone.
- **More frames do not help models**: Model performance saturates at approximately 8 frames, confirming that the bottleneck lies in comprehension rather than sampling.
- **Multiple-choice overestimates model capability**: LLaVA-Video-72B on multiple-choice (47.5%) is comparable to GPT-4o (46.6%), yet the open-ended gap is substantial (24.4% vs. 36.6%).

## Highlights & Insights

- **Decoupling frame sampling from comprehension**: A neglected but critical design choice — by ensuring all questions are answerable within 80 frames, the benchmark isolates comprehension as the variable of interest.
- **Value of natural adversarial questions**: Rather than perturbing pixels or introducing spelling errors, the benchmark rephrases the same question from a different angle — a setting far closer to real user interactions.
- **GPT-4o error patterns**: Spatiotemporal confusion (inability to track action participants across multiple scenes), world knowledge deficits (failure to infer implicit intent and social dynamics), and complex plot confusion (inability to maintain causal chains across events).
- **Fine-grained analysis across 18 question types**: From elemental attributes to plot causality, the benchmark provides a detailed profile of model capabilities.

## Limitations & Future Work

- Only YouTube Shorts (< 65 seconds) are used; long-video understanding is not addressed.
- The annotation process is labor-intensive and difficult to scale.
- Robustness is evaluated as a binary measure (all correct / not all correct); intermediate states are not modeled.
- Error analysis covers only GPT-4o; error patterns of other models may differ.
- Systematic evaluation incorporating audio/speech modalities has not been conducted (only preliminary experiments were performed).

## Related Work & Insights

- **VideoMME**: Collects videos from YouTube, but performance gaps on the long-video track primarily reflect insufficient frame sampling.
- **MVBench**: Integrates existing benchmarks, but source datasets have been widely used in training, raising data leakage concerns.
- **TemporalBench**: Evaluates fine-grained temporal understanding, but is limited to specific temporal question types.
- **FunQA**: Tests counter-intuitive and humorous content, but is restricted in domain coverage.
- **Insight**: The core principle of benchmark design should be to control confounding variables (e.g., frame sampling) so that the target variable (e.g., comprehension ability) is accurately measured.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[CVPR 2026\] VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](../../CVPR2026/video_understanding/videoauto-r1_video_auto_reasoning_via_thinking_once_answering_twice.md)
- [\[CVPR 2026\] MINERVA-Cultural: A Benchmark for Cultural and Multilingual Long Video Reasoning](../../CVPR2026/video_understanding/minerva-cultural_a_benchmark_for_cultural_and_multilingual_long_video_reasoning.md)
- [\[ICCV 2025\] Breaking the Encoder Barrier for Seamless Video-Language Understanding](breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)

<!-- RELATED:END -->
