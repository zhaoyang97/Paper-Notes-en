---
title: >-
  [Paper Note] LTD-Bench: Evaluating Large Language Models by Letting Them Draw
description: >-
  [NeurIPS 2025][LLM Evaluation][spatial reasoning] LTD-Bench evaluates the spatial reasoning capabilities of LLMs by having them draw (via dot-matrix output or code-based rendering)…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "spatial reasoning"
  - "visual generation"
  - "benchmark"
  - "drawing ability"
date: 2026-05-08
content_hash: b8c806a9d0636b07
---

# LTD-Bench: Evaluating Large Language Models by Letting Them Draw

**Conference**: NeurIPS 2025
**arXiv**: [2511.02347](https://arxiv.org/abs/2511.02347)  
**Code**: [walktaster/LTD-Bench](https://github.com/walktaster/LTD-Bench)  
**Area**: LLM Evaluation
**Keywords**: LLM evaluation, spatial reasoning, visual generation, benchmark, drawing ability

## TL;DR

LTD-Bench evaluates the spatial reasoning capabilities of LLMs by having them draw (via dot-matrix output or code-based rendering), transforming abstract evaluation metrics into intuitive visual outputs. The benchmark reveals critical deficiencies in current state-of-the-art LLMs regarding bidirectional mapping between linguistic and spatial concepts.

## Background & Motivation

A critical blind spot exists in current LLM evaluation paradigms: reliance on opaque numerical metrics that obscure fundamental limitations in spatial reasoning. When a model achieves 85% on a benchmark, what specific capabilities and limitations does that figure actually reveal? Traditional evaluation focuses predominantly on symbolic manipulation tasks such as language understanding (MMLU), mathematical reasoning (GSM8K), and code generation (HumanEval), leaving several gaps:

**Absence of spatial reasoning evaluation**: No systematic benchmark assesses LLMs' spatial perception and imagination capabilities.

**Unintuitive evaluation results**: Numerical scores fail to convey what a model can and cannot do in a transparent manner.

**Importance of physical world understanding**: LLMs are increasingly deployed in domains requiring spatial reasoning, such as robotics, autonomous driving, and design tools.

**Support from cognitive science**: Studies on congenitally blind individuals demonstrate that spatial cognition can be established through non-visual modalities (e.g., linguistic description), suggesting that text-only LLMs should in principle possess spatial understanding.

## Method

### Overall Architecture

LTD-Bench comprises **183 tasks** designed around three core principles:

1. **Visual interpretability**: All generation task outputs are rendered as images to intuitively illustrate model capabilities.
2. **Dual-pathway evaluation**: Both generation (spatial imagination) and recognition (spatial perception) directions are assessed simultaneously.
3. **Progressive complexity**: Three difficulty levels systematically locate the threshold of model capability.

### Key Designs

**Easy Level: Discrete Grid Spatial Understanding**
- Generation task: Given a description (e.g., "draw the letter H as a 3×3 binary matrix"), the model outputs a dot-matrix representation.
- Recognition task: Given a dot-matrix, the model identifies the character it represents.
- 86 questions total (50 generation + 36 recognition).

**Normal Level: Curve Composition in Continuous Coordinate Space**
- Generation task: The model generates Python code to draw a specified character using curves (text rendering functions are prohibited).
- Recognition task: Given Python code that draws a character, the model identifies the depicted character.
- 72 questions total (36 generation + 36 recognition).

**Hard Level: Real-World Object Drawing**
- Generation task: Open-ended instructions to draw complex real-world objects (e.g., "draw a cat with pointed ears, long whiskers, and round eyes").
- Generation only; evaluated automatically by GPT-4.1 on a scale of 0.0–1.0.
- 25 questions total.

| Level | Generation | Recognition | Total |
|-------|-----------|-------------|-------|
| Easy | 50 | 36 | 86 |
| Normal | 36 | 36 | 72 |
| Hard | 25 | - | 25 |
| **Total** | **111** | **72** | **183** |

### Loss & Training

This work is an evaluation benchmark and does not involve model training. Evaluation strategies are as follows:
- Easy and Normal generation tasks: dual-track assessment combining human evaluation and GPT-4.1 automated scoring.
- Easy and Normal recognition tasks: accuracy computed by direct comparison against ground-truth answers.
- Hard generation tasks: GPT-4.1 evaluation only, given the subjectivity of open-ended outputs.
- Generation tasks with code execution failures are assigned a score of 0 directly.

## Key Experimental Results

### Main Results

Comprehensive performance of 7 state-of-the-art LLMs on LTD-Bench:

| Model | Easy Gen. | Easy Rec. | Normal Gen. | Normal Rec. | Hard Gen. | Avg. |
|-------|-----------|-----------|-------------|-------------|-----------|------|
| Deepseek-r1 | 82.0 | 69.4 | 65.3 | 77.8 | 63.2 | **71.5** |
| GPT-4.1-mini | 85.0 | 38.9 | 70.8 | 55.6 | 71.6 | 64.4 |
| Deepseek-v3 | 72.0 | 36.1 | 54.2 | 63.9 | 66.4 | 58.5 |
| GPT-4o | 81.0 | 41.7 | 45.8 | 44.4 | 48.0 | 52.2 |
| QwQ-32B | 65.0 | 36.1 | 38.9 | 58.3 | 42.0 | 48.1 |
| Qwen2.5-72B | 56.0 | 13.9 | 18.1 | 25.0 | 40.8 | 30.8 |
| Llama3.3-70B | 46.0 | 11.1 | 23.6 | 19.4 | 35.2 | 27.1 |

Breakdown by generation vs. recognition ability:

| Model | Generation | Recognition | Avg. |
|-------|-----------|-------------|------|
| Deepseek-r1 | 72.9 | **73.6** | **73.2** |
| GPT-4.1-mini | **77.5** | 47.2 | 62.3 |
| Deepseek-v3 | 64.7 | 50.0 | 57.4 |
| GPT-4o | 62.1 | 43.1 | 52.6 |

### Ablation Study

Effect of deep reasoning distillation on Llama3.3-70B:

| Model | Generation | Recognition | Avg. |
|-------|-----------|-------------|------|
| Llama3.3-70B-Instruct | 36.6 | 15.3 | 26.0 |
| Deepseek-r1-distill-Llama3.3-70B | 33.7 | 33.3 | 33.5 |
| Change Δ | ↓2.9 | **↑18.1** | ↑7.6 |

Deep reasoning distillation substantially improves recognition performance (+18%) but yields a slight decline in generation performance (−2.9%), indicating that enhanced reasoning benefits spatial perception but is ineffective—or even detrimental—for spatial imagination.

### Key Findings

1. **LLMs exhibit overall weak spatial reasoning**: Only Deepseek-r1 achieves an average above 70%; most models fall below 60%, whereas human experts approach perfect scores on Easy and Normal tasks.
2. **Deep reasoning improves recognition but not generation**: Enhanced reasoning strengthens spatial perception, while spatial imagination appears to require a distinct set of underlying capabilities.
3. **Multimodal LLMs show no consistent advantage**: Multimodal models such as GPT-4.1-mini and GPT-4o do not consistently outperform text-only models on purely textual spatial tasks.
4. **Model similarity analysis**: Two models from the Qwen2.5 family produce highly similar image styles (12 out of 22 samples rated as more similar), offering a novel perspective for model similarity assessment.

## Highlights & Insights

1. **Evaluation paradigm innovation**: By converting LLM evaluation from abstract numbers into intuitive visual outputs, the benchmark allows non-experts to directly perceive the capability boundaries of models—a highly valuable evaluation philosophy.
2. **Dual-pathway and progressive design**: Simultaneously testing language→space (generation) and space→language (recognition) mappings, combined with three difficulty levels, yields a comprehensive profile of spatial cognitive ability.
3. **Dissociation between deep reasoning and spatial imagination**: The finding that reasoning ability and spatial imagination are orthogonal offers important insights into the cognitive structure of LLMs.
4. **Model style similarity**: Visual outputs from Hard-level generation tasks can be leveraged to analyze stylistic similarity across models—a dimension that traditional evaluation approaches struggle to capture.

## Limitations & Future Work

1. **Small dataset scale**: With only 183 tasks, the benchmark may be insufficient to draw statistically robust conclusions.
2. **Limited evaluation dimensions**: The benchmark focuses exclusively on spatial perception and imagination, leaving other spatial reasoning capabilities unaddressed (e.g., spatial transformation, 3D reasoning).
3. **Reliability of GPT-4.1-based evaluation**: Hard-level assessment relies entirely on GPT-4.1, introducing potential bias from the evaluator model itself.
4. **Preliminary model similarity analysis**: The analysis is based solely on visual style comparison and lacks a systematic quantitative methodology.
5. **Failure causes not thoroughly analyzed**: The benchmark does not investigate which specific spatial reasoning sub-skills cause model failures (e.g., rotation, scale, topological relations).

## Related Work & Insights

- **Existing LLM benchmarks**: MMLU, GSM8K, HumanEval, and similar benchmarks emphasize symbolic manipulation while lacking spatial evaluation.
- **Neuroscience of spatial cognition**: Studies such as Striem-Amit et al. (2018) demonstrate that spatial cognition does not depend on visual experience, providing a theoretical basis for expecting text-based LLMs to possess spatial capabilities.
- **ARC Prize**: Chollet et al. (2024)'s abstract reasoning challenge is related to this paper's focus but employs a fundamentally different methodology.
- **Insights**: The "let the model perform visual tasks" paradigm can be extended to more complex spatial reasoning evaluations such as physical simulation and 3D scene construction; the "infer capabilities from output visualization" evaluation framework is transferable to other domains requiring structured output.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of having LLMs draw is highly novel; transforming abstract scores into intuitive visual outputs is a creative and valuable contribution.
- Experimental Thoroughness: ⭐⭐⭐ Seven models are evaluated, but the dataset of only 183 questions limits statistical significance.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with well-articulated motivation; visual examples are persuasive.
- Value: ⭐⭐⭐⭐ The benchmark reveals significant capability gaps in LLM spatial reasoning and provides a reference point for future evaluation directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)

</div>

<!-- RELATED:END -->
