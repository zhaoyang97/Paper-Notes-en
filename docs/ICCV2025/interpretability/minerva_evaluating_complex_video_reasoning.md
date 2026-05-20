---
title: >-
  [Paper Note] Minerva: Evaluating Complex Video Reasoning
description: >-
  [ICCV 2025][Interpretability][video reasoning] This paper introduces Minerva — a manually annotated benchmark of 1,515 complex video reasoning QA pairs, each with 5 answer choices and a detailed reasoning trace…
tags:
  - "ICCV 2025"
  - "Interpretability"
  - "video reasoning"
  - "reasoning trace evaluation"
  - "video QA"
  - "benchmark"
  - "reasoning error taxonomy"
date: 2026-05-08
content_hash: d696cb1ffed0da1a
---

# Minerva: Evaluating Complex Video Reasoning

**Conference**: ICCV 2025
**arXiv**: [2505.00681](https://arxiv.org/abs/2505.00681)  
**Code**: [GitHub](https://github.com/google-deepmind/neptune?tab=readme-ov-file#minerva)  
**Area**: Interpretability
**Keywords**: video reasoning, reasoning trace evaluation, video QA, benchmark, reasoning error taxonomy

## TL;DR

This paper introduces Minerva — a manually annotated benchmark of 1,515 complex video reasoning QA pairs, each with 5 answer choices and a detailed reasoning trace, designed to evaluate the video reasoning capabilities of multimodal large language models. It further establishes a video reasoning error taxonomy (Temporal / Perceptual / Logical / Completeness) and the MiRA automated evaluation framework.

## Background & Motivation

The core problem with existing video benchmarks: **they evaluate only the final answer, not the reasoning process**.

**Correct answers do not imply correct reasoning**: models may arrive at the right answer through linguistic bias, elimination, or sheer luck rather than genuine video understanding.

**Wrong answers do not imply complete failure**: a model may be only one step away from the correct answer, yet be penalized entirely for the final incorrect response.

**The unique nature of video reasoning**: unlike text-based reasoning, video reasoning requires multi-step collaboration among temporal localization, visual perception (recognizing objects/actions/events), and logical inference — each step potentially demanding different skills and modalities.

**Limitations of existing datasets**: most rely on semi-automated LLM annotation pipelines and provide no intermediate reasoning steps; the few datasets that do offer auxiliary information either use very short videos (VideoCoT on Kinetics700) or generate low-quality reasoning traces (auto-generated, containing substantial irrelevant content).

Core need: a benchmark with **fully manual annotation, high-quality reasoning traces, and long videos across diverse domains** that not only assesses final answer correctness but also diagnoses which step in the reasoning chain fails.

## Method

### Overall Architecture

Minerva construction and evaluation involves three layers:
1. **Dataset construction**: video selection → manual annotation → quality review → adversarial filtering
2. **MCQ evaluation**: accuracy of multimodal models on 5-way multiple-choice questions
3. **Reasoning trace evaluation**: scoring reasoning processes using the Minerva Rubric (human + LLM-as-Judge)

### Key Designs

1. **Video selection strategy** — four domains to ensure complexity:

    - **Short Films**: complex storylines, relationship arcs, and event arcs; mainstream films are excluded to reduce training data contamination.
    - **Sports & Board Games**: require rule-based reasoning, fine-grained action recognition, and piece/player position judgment.
    - **Educational / STEM Lectures**: mathematical and scientific reasoning, constituting only 8% of the dataset as speech often dominates.
    - **Lifestyle**: cooking and travel videos featuring causal event chains, egocentric perspectives, and spatial reasoning.

2. **Annotation design** — ensuring complex multi-step reasoning:

    - Each question requires at least 2 distinct skills: temporal reasoning, counting, causal inference, goal reasoning, situational awareness, event detection, state change, OCR, speech understanding, spatial perception, numerical reasoning, object recognition, and counterfactual reasoning.
    - Reasoning traces are detailed and free-form, including timestamps (99.6% of questions, averaging 4 per question) and descriptions of key actions.
    - Average reasoning trace length: 92 words; video duration ranges from 2 minutes to 100+ minutes (mean: 12 minutes).

3. **Adversarial filtering — removing textual bias**:

    - Text-only tests (QAD-only and ASR-only) are conducted using Deepseek, GPT-4o, Gemini-Flash, and Qwen2.5-VL.
    - Multi-model consensus is used to filter questions answerable from text alone, while avoiding the removal of questions that happen to be answered correctly by chance.

4. **Reasoning error taxonomy (Minerva Rubric)**:

    - **Perceptual Correctness**: visual perception errors (object/action/event recognition, OCR, ASR parsing).
    - **Temporal Localization**: pointing to incorrect temporal segments in the video.
    - **Logical Reasoning**: logical inference errors (including arithmetic and numerical reasoning).
    - **Completeness**: reasoning trace missing critical steps.

5. **MiRA (Minerva Reasoning Assessment)**:

    - LLM-as-Judge automated evaluation based on the Minerva Rubric.
    - Supports both Reference-based (with ground-truth reasoning trace) and Reference-free modes.
    - Scores using a 3-point Likert scale.

### Loss & Training

This paper presents an evaluation benchmark and does not involve model training. Prompt design encompasses three strategies: direct answer, step-by-step reasoning, and reasoning augmented with the Minerva Rubric.

## Key Experimental Results

### Main Results (MCQ Accuracy)

| Model | Frames | ASR | MCQ-Acc% |
|-------|--------|-----|----------|
| Random | - | - | 20.0 |
| Qwen2.5-VL (open-source) | 768 | ✓ | 35.05 |
| VideoLLaMA3 (open-source) | 180 | ✓ | 35.91 |
| InternVideo2.5 (open-source) | 256 | ✓ | 35.18 |
| Claude 3.5 Sonnet v2 | 64 | ✓ | 31.28 |
| GPT-4o | 250 | ✓ | 45.54 |
| GPT-4.1 | 256 | ✓ | 53.99 |
| Gemini 2.0 Flash | 256 | ✓ | 53.47 |
| Gemini 2.5 Pro Thinking | 1024 | ✓ | **66.2** |
| **Human** | all | ✓ | **92.54** |

### Reasoning Trace Evaluation (MiRA + Human)

| Evaluation | Temporal | Perceptual | Logical | Completeness |
|-----------|----------|-----------|---------|-------------|
| Human | 0.440 | 0.625 | 0.770 | 0.725 |
| RF-MiRA (Pearson r) | 0.711 (0.56) | 0.684 (0.45) | 0.920 (0.21) | 0.871 (0.07) |
| RB-MiRA (Pearson r) | 0.434 (0.79) | 0.484 (0.59) | 0.848 (0.17) | 0.748 (0.24) |

### Ablation Study (Prompt)

| Prompt Strategy | MCQ Accuracy | MiRA |
|----------------|-------------|------|
| Direct answer | 46.47 | - |
| + Step-by-step reasoning | 51.22 | 0.65 |
| + Minerva Rubric | **53.47** | **0.75** |

### Key Findings

- **Large human–model gap**: the strongest model (Gemini 2.5 Pro Thinking) achieves 66.2% vs. human performance of 92.5%, a gap of nearly 30%.
- **Narrowing open- vs. closed-source gap**: Qwen2.5-VL and InternVideo2.5 already surpass Claude Sonnet.
- **Temporal localization is the primary bottleneck**: Temporal scores are the lowest in reasoning trace evaluation (human score: 0.440), far below Logical (0.770).
- **Correct answers ≠ correct reasoning**: Table 6 presents cases in which models produce correct answers through severely flawed reasoning traces — fabricating content absent from the video to arrive at the right answer.
- **Providing the Rubric improves performance**: simply informing the model of the evaluation criteria in the prompt (without additional computation) raises MCQ accuracy from 51.22% to 53.47%.
- **Reference-based MiRA achieves the highest correlation with human judgment on the Temporal dimension ($r = 0.79$)**, indicating that LLM scoring is more reliable when a reference reasoning trace is provided.
- **Thinking mode is beneficial**: enabling thinking mode for Gemini 2.5 Pro at 1,024 frames improves accuracy from 63.9% to 66.2%.

## Highlights & Insights

- **The reasoning trace annotation strategy is well-justified**: fully manual annotation with free-form text (rather than structured templates) balances quality with expressive flexibility.
- **The error taxonomy (Rubric) offers dual value**: it serves both as an evaluation tool and as a prompt-level mechanism to improve model performance.
- **The true bottleneck in video understanding is revealed**: the limitation lies not in logical reasoning ability but in temporal localization and visual perception — the distinctive challenges of video reasoning compared to text-based reasoning.
- **The case studies in Table 6 are highly compelling**: models can arrive at correct answers through fabricated reasoning, demonstrating that MCQ accuracy alone is insufficient for evaluating video understanding.

## Limitations & Future Work

- The dataset scale is limited (1,515 questions); although quality is high, coverage of all video understanding scenarios may be incomplete.
- LLM-as-Judge for reasoning trace evaluation shows relatively low correlation with human judgment on the Logical and Completeness dimensions ($r < 0.25$), indicating that automated evaluation still requires improvement.
- No training split is provided, precluding direct use for fine-tuning model reasoning capabilities.
- Video sources are predominantly from YouTube, leaving certain specialized domains (e.g., medical imaging, industrial inspection) uncovered.

## Related Work & Insights

- The reasoning trace annotation methodology can be generalized to other multimodal tasks (e.g., document understanding, 3D scene reasoning).
- The four-dimensional error taxonomy of the Minerva Rubric can serve as a general diagnostic framework for video AI systems.
- The phenomenon that "providing evaluation criteria in the prompt improves model performance" warrants further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first video understanding benchmark with reasoning traces; original error taxonomy)
- Technical Depth: ⭐⭐⭐⭐ (rigorous annotation design; complete adversarial filtering and multi-level evaluation system)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (10+ models, human baselines, multi-dimensional ablations over frame count / ASR / prompt / thinking mode)
- Value: ⭐⭐⭐⭐⭐ (directly applicable to diagnosing reasoning bottlenecks in video models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](../../NeurIPS2025/interpretability/a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](../../NeurIPS2025/interpretability/vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/interpretability/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)

</div>

<!-- RELATED:END -->
