---
title: >-
  [Paper Note] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models
description: >-
  [ICCV 2025][Multimodal VLM][graph analysis] GRAB is a graph analysis benchmark for large multimodal models (LMMs), comprising 3,284 synthetically generated questions spanning 5 tasks and 23 graph properties. The strongest model evaluated, Claude 3.5 Sonnet, achieves only 21.0% accuracy, revealing critical deficiencies in LMMs' capacity for visual analytical reasoning.
tags:
  - ICCV 2025
  - Multimodal VLM
  - graph analysis
  - benchmark
  - large multimodal models
  - synthetic data
  - visual reasoning
date: 2026-05-08
content_hash: af7d6d64895f282f
---

# GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models

**Conference**: ICCV 2025
**arXiv**: [2408.11817](https://arxiv.org/abs/2408.11817)
**Code**: [https://grab-benchmark.github.io](https://grab-benchmark.github.io)
**Area**: Multimodal VLM
**Keywords**: graph analysis, benchmark, large multimodal models, synthetic data, visual reasoning

## TL;DR

GRAB is a graph analysis benchmark for large multimodal models (LMMs), comprising 3,284 synthetically generated questions spanning 5 tasks and 23 graph properties. The strongest model evaluated, Claude 3.5 Sonnet, achieves only 21.0% accuracy, revealing critical deficiencies in LMMs' capacity for visual analytical reasoning.

## Background & Motivation

### State of the Field & Limitations of Prior Work

The capabilities of large multimodal models (LMMs) are advancing rapidly, yet existing benchmarks are becoming saturated at an equally rapid pace. GPT-4o has surpassed 88–90% on mainstream benchmarks such as MGSM, HumanEval, and MMLU, rendering these evaluations increasingly unable to discriminate between models. Additionally, annotation errors in existing benchmarks have been widely reported, further narrowing the usable evaluation space.

**Graph analysis as a core application scenario**: Interpreting scientific and mathematical graphs underpins a broad range of analytical tasks. In many practical contexts, the underlying data is inaccessible (e.g., figures embedded in documents or hand-drawn sketches), and numerical values can only be inferred through visual interpretation. This places high demands on the precise visual reasoning capabilities of LMMs.

**Limitations of existing chart benchmarks**:
- Existing benchmarks (e.g., ChartQA, MathVista) lack sufficient difficulty; GPT-4o already exceeds 60% on MathVista.
- Many questions target simple OCR-like tasks (reading legends, axis labels) rather than genuine analytical reasoning.
- Annotation quality is inconsistent, with manual labeling introducing noise.

### Root Cause & Starting Point

The authors argue that next-generation benchmarks must satisfy three key properties: **sufficient difficulty** (leaving substantial headroom for even the strongest current models), **noise-free, high-quality annotations**, and **resistance to data contamination**. Synthetic data is the most effective means of satisfying these requirements—it enables precise control over question difficulty, automatic generation of noise-free ground-truth answers, and reduced likelihood of overlap with pretraining corpora.

Motivated by this reasoning, the authors design GRAB, a challenging benchmark centered on synthetic graphs that comprehensively evaluates graph analysis capabilities.

## Method

### Overall Architecture

GRAB consists of 3,284 questions covering 5 core tasks and 23 graph properties. All synthetic graphs are generated using the Matplotlib and Seaborn libraries. A lightweight 500-question variant, GRAB-Lite, is also included to facilitate rapid evaluation.

### Key Designs

#### 1. Graph Property Taxonomy (23 Properties across 9 Categories)

- **Intercepts & Gradients**: x-intercept, y-intercept, gradient
- **Stationary Points**: coordinates of stationary points
- **Trigonometric**: amplitude, vertical shift, period
- **Function Equations**: identification of functional expressions
- **Counting**: number of points, number of series
- **Correlation**: Pearson, Spearman, and Kendall correlation coefficients
- **Bounded Area**: total bounded area, net bounded area
- **Dispersion**: mean, median, interquartile range, variance
- **Range & Extrema**: maximum/minimum values, domain length, range

**Design Motivation**: The taxonomy covers the majority of typical tasks an analyst might perform when interpreting a graph, while deliberately excluding simple OCR-type questions (e.g., reading titles or legends) to focus evaluation on visual analytical reasoning.

#### 2. Five Task Categories

- **Properties (660 questions)**: Attribute derivation for single functions or series, serving as a foundational task.
- **Functions (710 questions)**: Computing the mean of a given property across up to 10 overlapping functions, where function overlap increases difficulty.
- **Series (490 questions)**: Computing the mean of a given property across up to 10 data series, where data noise increases difficulty.
- **Transforms (310 questions)**: Applying up to 10 sequential transformations (rotation, translation, scaling, reflection) to a single function before querying a property.
- **Real (1,114 questions)**: Introducing real-world elements—hand-drawn whiteboard diagrams, sketches on paper, screenshots embedded in emails/presentations/video conferences, and various forms of degradation (blurring, flipping, artifacts).

**Design Motivation**: Tasks are structured to enable progressive evaluation from simple to complex scenarios. The Real task specifically assesses model robustness to the forms of degradation encountered in practical settings.

#### 3. Data Generation & Quality Control

- **Generation pipeline**: For each graph property, 250 candidate questions are initially generated, then downsampled to ensure a uniform answer distribution and avoid bias toward values near zero.
- **Precision design**: Approximately 75% of questions require integer-precision answers; ~25% require one decimal place to increase difficulty.
- **Graph aesthetics**: Properties task graphs sample visual appearance parameters randomly; other tasks use a unified appearance to control for confounding variables.
- **Quality control**: Multiple rounds of manual review are conducted to ensure each question is answerable, the ground-truth answer is correct, and the graph is clearly readable.

#### 4. Evaluation Protocol

- **Strict exact-match scoring**: No lenient post-processing is applied; model outputs must exactly match the ground-truth answer.
- **Joint assessment of reasoning and instruction-following**: Extraneous preambles (e.g., "The answer is...") result in the response being marked incorrect.
- **Design Motivation**: A model that can reason correctly but fails to produce a precisely formatted output is equally unusable in real-world applications.

### Loss & Training

GRAB is an evaluation benchmark; no model training is involved.

## Key Experimental Results

### Main Results

| Model | Properties | Functions | Series | Transforms | Real | Overall |
|-------|-----------|-----------|--------|-----------|------|---------|
| Claude 3.5 Sonnet | 41.8 | 15.5 | 11.0 | 10.0 | 19.6 | **21.0** |
| Gemini 1.5 Pro | 34.2 | 11.4 | 13.3 | 6.5 | 20.3 | 18.8 |
| Gemini 1.5 Flash | 28.5 | 11.5 | 8.4 | 9.0 | 17.1 | 16.1 |
| GPT-4o | 24.7 | 10.8 | 9.2 | 3.5 | 17.3 | 14.9 |
| GPT-4 Turbo | 18.5 | 8.5 | 4.9 | 3.5 | 7.5 | 9.2 |
| LLaVA-1.5 13b | 5.0 | 7.7 | 8.4 | 3.9 | 8.9 | 7.3 |
| CogVLM-Chat | 7.0 | 4.9 | 5.1 | 3.9 | 10.5 | 7.2 |

All 20 evaluated LMMs perform extremely poorly. The strongest model, Claude 3.5 Sonnet, achieves only 21.0%, while most open-source models perform near chance level.

### Ablation Study (Task Difficulty Analysis)

| Real Subset | Whiteboard | Paper | Screenshots | Noise | Overall |
|-------------|-----------|-------|------------|-------|---------|
| Claude 3.5 Sonnet | 14.6 | 17.0 | 18.6 | 21.1 | 19.6 |
| GPT-4o | 29.3 | 9.8 | 18.0 | 16.3 | 17.3 |
| Gemini 1.5 Pro | 34.4 | 24.4 | 21.9 | 17.2 | 20.3 |
| Gemini 2.5 Flash | 36.6 | 22.0 | 29.5 | 29.3 | 29.4 |

| Category | Claude 3.5 Sonnet | GPT-4o | Gemini 1.5 Pro | Notes |
|----------|------------------|--------|----------------|-------|
| Counting | 30.0 | 30.0 | 33.3 | Easiest category |
| Intercepts & Gradients | 25.5 | 14.1 | 17.9 | Moderate difficulty |
| Correlation | 9.2 | 15.8 | 26.7 | High difficulty |
| Area Bounded | 2.7 | 2.0 | 4.7 | Near-total failure |
| Functions | 0.0 | 0.0 | 0.0 | Zero score across all models |

### Key Findings

- **Transforms is the hardest task**: All models perform worst on this task, indicating that LMMs struggle with multi-step visual transformation reasoning.
- **Complete failure on the Functions category**: No model achieves a non-zero score on function equation identification.
- **Area Bounded is extremely challenging**: Requiring complex integral estimation, nearly all models score near zero.
- **Closed-source models consistently outperform open-source models**: With the exception of Reka Core, all closed-source models surpass the best open-source alternatives.
- **Real task performance is comparable to other tasks**: Noise and degradation have minimal impact on performance, indicating that the core bottleneck is reasoning capability rather than image quality.
- **Instruction following contributes to scoring**: Some models produce correct computations but are marked incorrect due to non-compliant output formatting.
- **Significant generational improvements in Gemini**: Performance improves consistently from Gemini 1.5 Flash to 2.0 Flash to 2.5 Flash.

## Highlights & Insights

- **Forward-looking design**: Substantial headroom is preserved for future models; with current models at only 21%, the benchmark is likely to remain relevant for years.
- **Methodological contribution of synthetic data**: The paper demonstrates how synthetic data can be used to construct high-quality, controllable, contamination-resistant benchmarks.
- **Philosophy of exact-match evaluation**: The principle that instruction-following ability is as important as reasoning ability in evaluation is a valuable methodological insight.
- **Comprehensive exposure of LMM weaknesses**: Graph analysis—seemingly a straightforward task—proves to be a significant blind spot for even the strongest current models.

## Limitations & Future Work

- Synthetic data may exhibit distribution shift relative to real-world graphs due to the constrained visual style of Matplotlib.
- Strict exact-match scoring may underestimate the true reasoning capabilities of certain models.
- The effects of advanced prompting strategies such as chain-of-thought are not examined.
- The number of hand-drawn images in the Real task is limited (only 41 whiteboard and 41 paper examples), reducing representativeness.
- More recent models (e.g., GPT-4.5, Claude 4) are not included in the evaluation.
- A systematic taxonomy of model failure modes is absent.

## Related Work & Insights

- **MathVista**: Covers a broader range of mathematical reasoning but lacks sufficient difficulty (GPT-4o exceeds 60%); GRAB focuses specifically on graph analysis at a substantially higher difficulty level.
- **ChartQA, PlotQA**: Existing chart understanding benchmarks that are now approaching saturation.
- **FigureQA**: A binary-classification chart understanding task with insufficient difficulty.
- **Implications for the LMM community**: Precise numerical reasoning represents a substantially harder challenge than standard VQA and requires dedicated capability development.

## Rating

- Novelty: ⭐⭐⭐⭐ — The benchmark itself is well-designed, though constructing a high-difficulty benchmark is not an entirely novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 20 models with detailed per-category analysis and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rich figures, and rigorous argumentation.
- Value: ⭐⭐⭐⭐ — An important reference for understanding and advancing LMM development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging](finmmr_make_financial_numerical_reasoning_more_multimodal_comprehensive_and_chal.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
