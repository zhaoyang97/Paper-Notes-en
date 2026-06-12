---
title: >-
  [Paper Note] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations
description: >-
  [Multimodal VLM] This paper introduces VisJudge-Bench, the first comprehensive benchmark for aesthetics and quality assessment of data visualizations (3,090 samples, 32 chart types), and trains the VisJudge model…
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: 1eee0930292a4968
---

# VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2510.22373](https://arxiv.org/abs/2510.22373)
- **Code**: [GitHub](https://github.com/HKUSTDial/VisJudgeBench)
- **Area**: Multimodal Large Language Models / Visualization Quality Assessment
- **Keywords**: visualization assessment, aesthetic quality, MLLM-as-a-Judge, data visualization, benchmark

## TL;DR

This paper introduces VisJudge-Bench, the first comprehensive benchmark for aesthetics and quality assessment of data visualizations (3,090 samples, 32 chart types), and trains the VisJudge model, which reduces MAE by 23.9% compared to GPT-5 and improves agreement with human experts by 60.5%.

## Background & Motivation

Data visualization is an effective means of transforming complex data into intuitive insights, with quality determined by three dimensions: **Fidelity** (whether data is accurately represented), **Expressiveness** (whether information is clearly conveyed), and **Aesthetics** (whether the design is visually appealing). However, significant gaps exist in prior work:

**Chart QA benchmarks** (e.g., ChartQA, ChartInsights) focus solely on chart content understanding, without evaluating design quality.

**Natural image aesthetics benchmarks** (e.g., AVA, ArtiMuse) target artistic beauty only, neglecting the core purpose of visualization — effective data communication.

**Visualization evaluation benchmarks** (e.g., VisEval) primarily assess NL2VIS generation accuracy rather than the intrinsic design quality of visualizations themselves.

Consequently, **a systematic framework for measuring the comprehensive capability of MLLMs in visualization aesthetics and quality assessment is lacking**. Even the state-of-the-art GPT-5 achieves an MAE of 0.553 on this task, with a correlation of only 0.428 against human ratings.

## Method

### Overall Architecture

VisJudge-Bench is constructed following a three-stage methodology: **(1) Data Collection and Processing** → **(2) Adaptive Question Generation** → **(3) Expert Annotation and Quality Control**. A domain-specific model, VisJudge, is further trained on top of the benchmark.

### The "Fidelity–Expressiveness–Aesthetics" Evaluation Framework

Drawing on the classical Chinese translation principle of *Xìn–Dá–Yǎ* (faithfulness, expressiveness, elegance), the framework establishes six measurable evaluation dimensions:

- **Fidelity**: *Data Fidelity* — whether visual encodings accurately reflect the underlying data, detecting misleading designs such as improper axis settings, scale distortion, and truncated baselines.
- **Expressiveness**:
    - *Semantic Readability*: whether users can clearly decode the visual elements in the chart.
    - *Insight Discovery*: whether the visualization reveals deeper data patterns, trends, or outliers.
- **Aesthetics**:
    - *Design Style*: the innovativeness and distinctiveness of the design.
    - *Visual Composition*: the rationality of spatial layout and the balance and order of element positioning.
    - *Color Harmony*: whether the color scheme achieves a balance between aesthetic appeal and information communication.

### Dataset Construction

Starting from 300,000+ initial images, a three-stage filtering process is applied:

1. **Initial Filtering**: automated scripts + perceptual hashing for deduplication → 80,210 candidates.
2. **Automated Classification**: GPT-4o classification + manual verification → 13,220 valid samples.
3. **Stratified Sampling**: final 3,090 samples covering single charts (1,041), multi-charts (1,024), and dashboards (1,025), across 32 subtypes.

### Adaptive Question Generation

GPT-4o is used to extract metadata (type, visual elements) from charts, and customized scoring questions along with five-level rubrics are generated based on predefined templates. For instance, under the Data Fidelity dimension, rubric scores range from "1 = truncated axes or misleading scales present" to "5 = bar lengths strictly proportional to displayed values."

### Expert Annotation and Quality Control

- **Stage 1**: 603 crowdworkers independently rate each sample; 3 annotators × 6 dimensions per sample.
- **Stage 2**: Variance-based conflict identification and resolution, including outlier removal and malicious rating detection.
- **Stage 3**: 3 visualization analysis experts independently review samples; complex cases are resolved through discussion to reach consensus.

### VisJudge Model Training

- **Data Split**: 70%/10%/20% for training/validation/testing (2,163/279/648 samples).
- **Base Models**: Qwen2.5-VL (3B/7B), InternVL3-8B, Llava-v1.6-mistral-7B.
- **Training Method**: GRPO reinforcement learning with a composite reward function = accuracy reward (minimizing prediction error) + format reward (ensuring structured output).
- **Parameter-Efficient Fine-Tuning**: LoRA, 5 epochs, learning rate $1 \times 10^{-5}$.

## Key Experimental Results

### Main Results (MAE ↓)

| Model | Overall | Fidelity | Readability | Insight | Design | Composition | Color |
|------|---------|----------|-------------|---------|--------|-------------|-------|
| GPT-5 | 0.553 | 0.862 | 0.781 | 0.778 | 0.649 | 0.699 | 0.682 |
| GPT-4o | 0.610 | 0.988 | 0.806 | 0.744 | 0.609 | 0.695 | 0.657 |
| VisJudge (Qwen2.5-VL-7B) | **0.421** | **0.661** | **0.648** | **0.677** | **0.580** | **0.545** | **0.604** |

### Key Findings

1. **GPT-5 remains insufficient**: Even the strongest closed-source model achieves an MAE of 0.553 and a correlation of only 0.428, indicating that general-purpose MLLMs cannot automatically acquire the specialized capability required for visualization assessment.
2. **VisJudge substantially closes the gap**: The best-performing VisJudge (Qwen2.5-VL-7B) reduces MAE to 0.421 (↓23.9%) and raises correlation to 0.687 (↑60.5%).
3. **Open-source models lag further behind**: Open-source models generally yield MAE > 0.7, with the worst performance on the Fidelity and Expressiveness dimensions.
4. **Aesthetics assessment is relatively easier**: All models perform better on the three Aesthetics sub-dimensions than on Fidelity and Expressiveness.

### Ablation Study

- GRPO reinforcement learning significantly outperforms pure SFT training.
- The composite reward design (accuracy + format) outperforms single-objective reward.
- Models of varying architectures and parameter scales all benefit from fine-tuning, validating cross-architecture generalizability.

## Highlights & Insights

- Introduces the first comprehensive benchmark targeting visualization aesthetics and quality assessment, filling an important gap in the field.
- The three-dimensional evaluation framework inspired by *Xìn–Dá–Yǎ* is elegantly designed, with six sub-dimensions providing comprehensive coverage.
- 3,090 expert-annotated samples spanning 32 chart types ensure high data quality.
- GRPO fine-tuning is highly effective; small models after fine-tuning substantially surpass GPT-5.
- Reveals critical deficiencies of current MLLMs in visualization evaluation.

## Limitations & Future Work

- Fidelity assessment is based solely on visual appearance, without access to source data for true data–visual consistency verification.
- Samples are primarily web-crawled, which may introduce distributional bias.
- Only small-to-medium open-source models are fine-tuned; larger-scale models remain unexplored.
- Human expert annotation is inherently subjective and inter-annotator disagreement may exist.

## Related Work & Insights

- **Visualization Recommendation**: Voyager, Draco (rule-driven); VizML, DeepEye (learning-driven).
- **NL2VIS Evaluation**: nvBench, MatPlotAgent — focus on code generation rather than design quality.
- **MLLM-as-a-Judge**: General aesthetics (AVA), chart understanding (ChartQA), and visualization evaluation (VisEval) all have notable limitations.
- **Image Aesthetics Assessment**: AVA, ArtiMuse — designed for natural images and not applicable to visualizations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First dedicated benchmark for visualization quality assessment.
- **Technical Depth**: ⭐⭐⭐⭐ — Comprehensive evaluation framework design and rigorous annotation pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic evaluation of 12 models with sufficient ablation studies.
- **Value**: ⭐⭐⭐⭐ — Directly advances the field of automated visualization assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](../../NeurIPS2025/multimodal_vlm/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)

</div>

<!-- RELATED:END -->
