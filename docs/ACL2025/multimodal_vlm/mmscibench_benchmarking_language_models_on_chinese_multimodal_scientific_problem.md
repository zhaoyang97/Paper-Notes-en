---
title: >-
  [Paper Note] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems
description: >-
  [ACL 2025][Multimodal VLM][scientific reasoning benchmark] This work proposes MMSciBench, a multimodal scientific reasoning benchmark containing 4,482 Chinese high school mathematics and physics problems. It covers both multiple-choice and question-answering formats, across text-only and multimodal (text-image) settings, complete with human-annotated difficulty levels and a three-level knowledge taxonomy. Evaluation shows that the strongest model, Gemini 1.5 Pro 002…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "scientific reasoning benchmark"
  - "multimodal evaluation"
  - "Chinese science problems"
  - "vision-language models"
  - "mathematical and physical reasoning"
date: 2026-05-08
content_hash: ce34f1e6bdc891b5
---

# MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems

**Conference**: ACL 2025  
**arXiv**: [2503.01891](https://arxiv.org/abs/2503.01891)  
**Code**: [GitHub](https://github.com/xinwuye/MMSciBench-code) | [HuggingFace](https://huggingface.co/datasets/XinwuYe/MMSciBench)  
**Area**: Multimodal VLM  
**Keywords**: scientific reasoning benchmark, multimodal evaluation, Chinese science problems, vision-language models, mathematical and physical reasoning

## TL;DR

This work proposes MMSciBench, a multimodal scientific reasoning benchmark containing 4,482 Chinese high school mathematics and physics problems. It covers both multiple-choice and question-answering formats, across text-only and multimodal (text-image) settings, complete with human-annotated difficulty levels and a three-level knowledge taxonomy. Evaluation shows that the strongest model, Gemini 1.5 Pro 002, only achieves 63.77% accuracy, with a significant performance drop on multimodal problems (a gap of 36.28 percentage points).

## Background & Motivation

Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) have demonstrated strong capabilities across numerous tasks, but their scientific reasoning abilities—especially in multimodal scenarios—remain insufficiently evaluated. Existing scientific benchmarks exhibit three major limitations:

**Lack of multimodal evaluation**: Most scientific benchmarks only contain text-only problems, failing to assess the joint vision-text reasoning capabilities of models.

**Limited area coverage**: Existing datasets are either overly focused on a single discipline or cross-disciplinary but lack systematicity, making it difficult to evaluate the understanding of core concepts within specific subjects.

**Insufficient evaluation granularity**: The lack of human-annotated difficulty levels and structured knowledge classification makes it difficult to analyze performance discrepancies across different complexities and knowledge domains.

Furthermore, Chinese scientific reasoning benchmarks are particularly scarce. GAOKAO-Bench and GAOKAO-MM contain only 3K and 650 problems respectively, and lack fine-grained knowledge classification. This motivates the authors to construct a Chinese scientific benchmark that balances scale, quality, multimodality, and fine-grained evaluation.

## Method

### 1. Data Collection and Quality Control

MMSciBench's data is originally annotated by K-12 teachers, with each problem containing:
- Problem text (Chinese)
- Detailed step-by-step solution process and final answer
- Human-annotated difficulty score (0–1 normalized)
- Knowledge point tags
- Metadata (problem type, modality, subject)

Quality control workflow:
- Filter problems with incomplete information or duplicates.
- Keep only highly challenging problems with difficulty scores $\geq 0.7$.
- Limit each problem to at most one image to maintain evaluation consistency.
- Employ GPT-4o for three-level knowledge classification, followed by manual validation by curriculum experts.
- Obtain a final set of 4,482 problem-solution pairs.

### 2. Dataset Structural Design

**Problem Type Dimension**:

| Problem Type | Math | Physics | Total |
|------|------|------|------|
| Multiple-Choice (MCQ) | 760 | 2,707 | 3,467 |
| Question-Answering (Q&A) | 516 | 499 | 1,015 |

**Modality Dimension**:

| Modality | Math | Physics | Total |
|------|------|------|------|
| Multimodal (Text-Image) | 457 | 710 | 1,167 |
| Text-Only | 819 | 2,496 | 3,315 |

### 3. Three-Level Knowledge Taxonomy

- **Domain**: Core disciplinary areas, such as "Set" and "Function" in mathematics, and "Classical Mechanics", "Electrodynamics", and "Quantum Mechanics" in physics.
- **Module**: Key themes under domains, such as "Probability & Statistics" and "Mechanical Motion & Physical Models".
- **Chapter**: The finest granularity, such as "Exponential Function", "Trigonometric Function", "Hooke's Law", and "Equilibrium Conditions of Coplanar Forces".

### 4. Evaluation Framework

- **Metric**: Accuracy (only evaluating the correctness of the final answer).
- **Evaluation Protocol**: GPT-4o is employed as an automatic evaluator to compare model outputs against standard answers. Following iterative calibration over 180 problems, the agreement rate between GPT-4o judgment and human evaluation reached 97.22%.
- **Prompt Design**: Zero-shot setting, utilizing a unified prompt template without optimization for specific models and without providing auxiliary knowledge point information.

## Key Experimental Results

### Table 1: Overall and Subject-Specific Accuracy of Models

| Model | Math | Physics | Overall |
|------|------|------|------|
| Gemini 1.5 Pro 002 | 56.74% | 66.56% | **63.77%** |
| Qwen2-VL-72B-Instruct | 35.50% | 64.32% | 56.11% |
| Claude 3.5 Sonnet | 37.38% | 60.54% | 53.95% |
| GPT-4o | 35.97% | 56.89% | 50.94% |
| Llama-3.2-90B-Vision-Instruct | 16.69% | 36.96% | 31.19% |
| Qwen2.5-Math-72B-Instruct | 57.39%* | — | — |
| DeepSeekMath-7B-Instruct | 21.86%* | — | — |
| o1 | 67.40%† | — | — |
| Claude 3.7 Sonnet | 37.64%† | — | — |

\*Text-only math problems only; †Multimodal math problems only

### Table 2: Accuracy Comparison between Text-Only vs. Multimodal Settings

| Model | Math-Text | Math-Multimodal | Physics-Text | Physics-Multimodal | Overall-Text | Overall-Multimodal |
|------|------------|----------|------------|----------|------------|----------|
| Gemini 1.5 Pro 002 | 69.60% | 33.70% | 74.40% | 39.01% | **73.21%** | **36.93%** |
| Qwen2-VL-72B-Instruct | 41.39% | 24.95% | 72.48% | 35.63% | 64.80% | 31.45% |
| Claude 3.5 Sonnet | 44.57% | 24.51% | 67.75% | 35.21% | 62.02% | 31.02% |
| GPT-4o | 44.69% | 20.35% | 64.10% | 31.55% | 59.31% | 27.16% |
| Llama-3.2-90B-Vision | 19.54% | 11.60% | 42.83% | 16.34% | 37.07% | 14.48% |

**Key Finding**: All LVLMs perform significantly worse on multimodal problems than on text-only problems. Gemini 1.5 Pro 002 achieves an overall accuracy of 73.21% on text-only problems, but only 36.93% on multimodal problems, resulting in a gap of **36.28 percentage points**.

### Table 3: Accuracy Comparison between MCQ and Q&A (excluding random guessing baseline)

| Model | MCQ (Overall) | Q&A (Overall) | MCQ beyond Random Baseline |
|------|------------|------------|----------------|
| Gemini 1.5 Pro 002 | 68.82% | 46.50% | +47.96% |
| Qwen2-VL-72B-Instruct | 65.71% | 23.35% | +44.85% |
| Claude 3.5 Sonnet | 61.55% | 27.98% | +40.69% |
| GPT-4o | 57.51% | 28.47% | +36.65% |
| Llama-3.2-90B-Vision | 37.96% | 8.08% | +17.10% |

Gemini 1.5 Pro 002's accuracy on Q&A problems is **22.32 percentage points** lower than on MCQs, indicating that open-ended question answering is significantly more challenging.

### Table 4: Effects of Chain-of-Thought Prompting

| Model | Default (Chinese) | CoT (Chinese) | CoT (English) |
|------|------------|---------|---------|
| Llama-3.2-90B-Vision | 31.19% | 33.24% | **38.00%** |
| GPT-4o | 50.94% | 50.85% | **52.86%** |
| Claude 3.5 Sonnet | 53.95% | 54.42% | **55.40%** |
| Gemini 1.5 Pro 002 | **63.77%** | 63.61% | 62.25% |

Most models display improved performance when using English CoT for reasoning on Chinese problems. However, Gemini's performance declines slightly, potentially due to its higher sensitivity to consistency in the prompt-response language.

## Key Findings

### Error Type Analysis

An in-depth analysis of problems where all models failed (240 cases) reveals the error distribution:
- **Reasoning Errors**: 77.1% (the primary bottleneck)
- **Calculation Errors**: 11.3%
- **Visual Misinterpretations**: 7.5%
- **Information Integration Failures**: 2.5%
- **Textual Misunderstandings**: 1.7%

Reasoning errors overwhelmingly dominate, indicating a fundamental deficiency of current models in complex multi-step scientific reasoning.

### Cross-Knowledge-Point Analysis

- Model performance varies significantly across different subfields: Gemini leads in most domains but lags behind Claude and GPT-4o in the "Electrodynamics-Magnetic Field" subfield.
- Universally weak areas for all models: "Electromagnetic Induction and its Applications" in physics, and "Geometry & Algebra" as well as "Function-Introductory Knowledge" in mathematics.
- The overall accuracy in physics is higher than that in mathematics, partially because the proportion of text-only problems in physics is higher.

## Highlights & Insights

- **Multimodal + Multi-format**: Covers four combinations (text-only vs. multimodal, MCQ vs. Q&A) to support comprehensive cross-analysis.
- **Three-Level Knowledge Taxonomy**: The hierarchical classification from Domain $\to$ Module $\to$ Chapter enables fine-grained capability diagnosis, allowing localized analysis of model weaknesses in specific knowledge points.
- **Human-Annotated Difficulty**: All problems are accompanied by standardized difficulty scores annotated by K-12 teachers, with filtering $\geq 0.7$ ensuring the challenging nature of the benchmark.
- **Detailed Solution Process**: Each problem is equipped with step-by-step solution explanations, facilitating error localization and future research on model improvement.
- **Multi-Dimensional Analysis**: Not only reports overall accuracy but also deeply analyzes results across disciplines, formats, modalities, knowledge points, CoT effects, and error types.

## Limitations & Future Work

1. **Limited Subject Coverage**: Only covers high school mathematics and physics, excluding other subjects like chemistry and biology, and does not feature university-level or Olympiad-level content.
2. **Final Answer-Only Evaluation**: Ignores the correctness of intermediate reasoning steps, which might mask crucial differences in models' internal reasoning processes.
3. **Single Language**: Predominantly in Chinese, which might disadvantage models primarily trained on English data; cultural and linguistic biases may impact fairness.
4. **Moderate Dataset Size**: 4,482 problems is relatively small compared to large-scale benchmarks; the strict difficulty filtering ($\geq 0.7$) might exclude valuable boundary cases.
5. **Limitations of GPT-4o Judging**: Despite the 97.22% agreement rate, the automatic evaluator may introduce systematic biases, such as being too lenient on incomplete answers or inaccurate in judging the equivalence of complex mathematical expressions.

## Related Work & Insights

| Benchmark | Subject | Modality | Language | Difficulty | Scale | Knowledge Classification | Solution Process |
|------|------|------|------|------|------|---------|---------|
| GAOKAO-Bench | Math/Phys/Other | Text-only | Chinese | High School | 3K | ✗ | ✓ |
| GAOKAO-MM | Math/Phys/Other | Text+Image | Chinese | High School | 650 | ✗ | ✓ |
| OlympiadBench | Math/Phys | Text+Image | CN/EN | Olympiad | 8K | ✓ | ✓ |
| SciBench | Math/Phys/Other | Text+Image | English | College | 869 | ✓ | ✓ |
| M3Exam | Multi-subject | Text+Image | Multilingual | K-12 | 12K | ✓ | Answer Only |
| EXAMS-V | Multi-subject (20) | Text+Image | Multilingual | Middle School | 21K | ✓ | Answer Only |
| **MMSciBench** | **Math/Phys** | **Text+Image** | **Chinese** | **High School** | **4.5K** | **✓ (Three-level)** | **✓** |

MMSciBench uniquely combines multimodal evaluation, a three-level knowledge taxonomy, human-annotated difficulty, and detailed solution processes among Chinese scientific benchmarks, filling the gap in fine-grained capability evaluation.

## Rating

- Novelty: ⭐⭐⭐ — The benchmark design philosophy is solid, but the primary contribution lies in the dataset, with limited methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely comprehensive, analyzing 9 models across multiple dimensions (discipline, problem type, modality, knowledge point, CoT, and error type).
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich tables and figures, and diverse analytical perspectives.
- Value: ⭐⭐⭐⭐ — Provides a high-quality benchmark for Chinese scientific reasoning evaluation; the three-level classification system facilitates fine-grained diagnostic analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification](sciver_evaluating_foundation_models_for_multimodal_scientific_claim_verification.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](../../ICCV2025/multimodal_vlm/geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)
- [\[ACL 2025\] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)
- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)

</div>

<!-- RELATED:END -->
