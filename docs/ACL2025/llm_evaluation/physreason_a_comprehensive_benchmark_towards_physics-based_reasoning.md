---
title: >-
  [Paper Note] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning
description: >-
  [ACL 2025][LLM Evaluation][Physics reasoning] This paper proposes PhysReason, a physics reasoning benchmark containing 1,200 physics problems (averaging 8.1 solving steps per problem). It designs a two-tier automatic evaluation framework, PSAS, covering both answer-level and step-level evaluations. The benchmark reveals that top-tier models (such as DeepSeek-R1 and o3-mini) achieve less than 60% accuracy on physics reasoning and identifies four major reasoning bottlenecks.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Physics reasoning"
  - "benchmark"
  - "step-level evaluation"
  - "multimodal"
  - "large language models"
date: 2026-05-08
content_hash: 216cb7e5c7691241
---

# PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning

**Conference**: ACL 2025  
**arXiv**: [2502.12054](https://arxiv.org/abs/2502.12054)  
**Code**: [dxzxy12138/PhysReason](https://dxzxy12138.github.io/PhysReason/)  
**Area**: LLM Evaluation  
**Keywords**: Physics reasoning, benchmark, step-level evaluation, multimodal, large language models

## TL;DR

This paper proposes PhysReason, a physics reasoning benchmark containing 1,200 physics problems (averaging 8.1 solving steps per problem). It designs a two-tier automatic evaluation framework, PSAS, covering both answer-level and step-level evaluations. The benchmark reveals that top-tier models (such as DeepSeek-R1 and o3-mini) achieve less than 60% accuracy on physics reasoning and identifies four major reasoning bottlenecks.

## Background & Motivation

While Large Language Models (LLMs) perform exceptionally well in mathematical and logical reasoning, evaluation in physics reasoning—a domain closer to real-world applications—remains severely insufficient. Existing physics benchmarks (such as ScienceQA, SciBench, GPQA, and OlympiadBench) suffer from two key limitations of prior work:

**Over-simplified reasoning processes**: Problems in existing benchmarks typically involve only 3–4 physical formulas, failing to truly test multi-step reasoning capabilities.

**Neglecting step-level evaluation**: By focusing solely on final answers, they cannot reveal where and why models make mistakes.

The unique challenge of physics reasoning lies in the need to integrate multiple theorems and adhere to physical constraints, which aligns closer to real-world application scenarios (e.g., robotics, autonomous driving) than pure mathematical reasoning. Therefore, a comprehensive benchmark with complex reasoning processes and step-level evaluation is highly demanded.

## Method

### Overall Architecture

PhysReason consists of two core components:
1. **Benchmark Dataset**: 1,200 meticulously curated physics problems, covering knowledge-oriented (25%) and reasoning-oriented (75%) tasks, with the latter further divided into three difficulty levels.
2. **PSAS Evaluation Framework**: A two-tier evaluation framework containing answer-level (PSAS-A) and step-level (PSAS-S) metrics.

### Key Designs

1. **Data Collection Pipeline (Five Stages)**:

    - **Acquisition**: Problems are collected from global college entrance examinations, mock exams, and international physics olympiads, with sources including Gaokao (China), JEE (India), Unified State Exam (Russia), as well as IPhO, APhO, EPhO, etc. Over 20,000 raw problems are gathered from 1,254 PDFs.
    - **Standardization**: Utilizing the MinerU framework to parse PDFs, followed by deduplication, filtering, and format standardization.
    - **Translation**: A two-stage translation pipeline, with accuracy verified by engineering postdocs.
    - **Search Prevention**: Problems whose answers can be easily found within a 5-minute Google search are excluded.
    - **Classification**: Problems are classified into knowledge-oriented and three levels of reasoning-oriented tasks based on solving time and theorems used.

2. **Annotation Framework (8 Dimensions)**: Diagram, Context, Sub-questions, Solution, Step Analysis, Answer, Theorem, and Difficulty. Each step must contain formula derivations from physical theorems and associated computations.

3. **PSAS-A (Answer-Level Evaluation)**: Extracting model answers for each sub-question and comparing them semantically with ground-truth answers. Sub-question scores are weighted by the length of annotated solutions: $\text{Score}(M) = \frac{\sum_{q_i}|s_i| \times C(\hat{a}_i, a_i)}{\sum_{q_i}|s_i|}$

4. **PSAS-S (Step-Level Evaluation, Four Stages)**:

    - **Data Extraction**: LLMs extract content corresponding to the annotated steps from the model output.
    - **Scoring**: Each step is evaluated across two dimensions—theorem application (ScoreFormula) and numerical calculation (ScoreValue), each with a weight of 0.5.
    - **First-Error Step Detection**: Pinpointing the step where the model's reasoning first deviates from the correct path.
    - **Error Analysis**: Classifying errors into 7 categories (diagrammatic analysis error, physical theorem application error, physical condition analysis error, physical process understanding error, variable relation error, calculation process error, and boundary condition analysis error).

5. **Three Principles of Step Definition**: Completeness (a complete logical reasoning unit), Independence (can be understood and evaluated independently), and Progressiveness (substantially advances the problem-solving process).

### Loss & Training

This work functions as an evaluation benchmark and does not involve training. DeepSeek-V3 is utilized as the grader model, and 15 mainstream LLMs/VLMs are evaluated under the zero-shot CoT setting.

## Key Experimental Results

### Main Results

| Model | Type | Knowledge | Easy | Medium | Hard | Average |
|------|------|--------|------|------|------|------|
| GPT-4o | Non-O | 50.71/65.82 | 33.87/51.98 | 22.73/42.36 | 11.03/24.71 | 29.58/47.23 |
| Gemini-2.0-Pro | Non-O | 67.99/79.01 | 55.43/71.47 | 44.29/57.74 | 23.81/42.66 | 47.88/62.74 |
| o3-mini-high | O-like | 70.67/83.61 | 67.20/81.95 | 45.31/64.57 | 30.12/47.23 | 53.32/69.34 |
| Deepseek-R1 | O-like | **75.11/85.91** | **65.08/79.81** | **54.84/72.02** | **31.95/51.50** | **56.75/73.26** |

(Format: Answer-Level / Step-Level)

### PSAS Evaluation Reliability

| Method | Answer Accuracy | Step Accuracy |
|------|----------|----------|
| Deepseek-R1 Direct Evaluation | 93.31% | 37.54% |
| PSAS (Deepseek-V3) | **99.35%** | **98.04%** |

The evaluation accuracy of the PSAS framework exceeds 98%, which significantly outperforms direct evaluation by LLMs.

### Ablation Study

| Dimension | Key Metric | Description |
|------|---------|------|
| Knowledge → Hard | 75.11% → 31.95% | Performance drops severely as difficulty increases |
| O-like vs Non-O | 50%+ vs <48% | O-like models significantly outperform non-O-like models |
| Step-level vs Answer-level | Step-level scores are higher | Models can complete some correct steps |
| Multimodality | 81% of problems contain diagrams | Image understanding poses an extra challenge |

### Error Type Analysis

| Error Type | Proportion | Description |
|---------|------|------|
| Physical Theorem Application Error (PTAE) | Highest | Selecting or applying the wrong theorem |
| Physical Process Understanding Error (PPUE) | Second Highest | Improper understanding of physical scenarios |
| Calculation Process Error (CPE) | Medium | Algebraic calculation errors |
| Physical Condition Analysis Error (PCAE) | Relatively High | Missing or misunderstanding physical conditions |

### Key Findings

1. **Top-tier models still fail**: DeepSeek-R1 achieves an average answer-level score of only 56.75%, and only 31.95% on hard problems.

2. **Difficulty correlates positively with step count, causing sharp performance drops**: From knowledge-oriented problems (75.11%) to hard problems (31.95%), models struggle to maintain accuracy across continuous reasoning steps.

3. **Step-level evaluation is more discriminative**: Step-level score differences are more pronounced than answer-level differences on high-difficulty problems, allowing for more precise differentiation of model capabilities.

4. **Positive correlation between knowledge and reasoning**: DeepSeek-R1 and Gemini-2.0-Flash-Thinking perform outstandingly in both knowledge and reasoning. However, when knowledge scores are similar, O-like models perform better on reasoning problems, indicating that reinforcement learning and Chain-of-Thought (CoT) training help enhance reasoning capabilities.

5. **Four major reasoning bottlenecks**: Physical theorem application, physical process understanding, calculation processes, and physical condition analysis are key bottlenecks limiting model performance.

## Highlights & Insights

1. **True complex reasoning**: The requirement of an average of 8.1 steps (15.6 steps for hard problems) to solve problems goes far beyond existing benchmarks, aligning closer to the complexity of realistic physics reasoning.
2. **Pioneering step-level evaluation**: PSAS-S not only scores but also localizes the first-error step and analyzes key error categories, offering clear pathways for model improvement.
3. **Highly reliable evaluation framework**: An evaluation accuracy exceeding 98% addresses the unreliability of using LLMs directly for step-level reasoning evaluation.
4. **Layered difficulty design**: The four levels (knowledge, easy, medium, and hard) facilitate fine-grained evaluation of reasoning capabilities at different levels.
5. **Multimodal integration**: 81% of the problems contain diagrams, truly reflecting the multimodal nature of physical problems.

## Limitations & Future Work

1. Mainly focusing on classical physics and olympiad physics, leaving cutting-edge physical research topics unexplored.
2. Problem sources are predominantly exam and competition questions, presenting a gap with real-world scientific research scenarios.
3. The efficacy of image captions as a visual alternative requires further validation.
4. Step definitions rely heavily on annotators' domain expertise in physics and judgment standards.
5. The exploration of Test-Time Compute Scaling is relatively preliminary.

## Related Work & Insights

- **Mathematical Reasoning Benchmarks**: GSM8K, MATH, etc., focus on mathematical reasoning.
- **Evolution of Physics Benchmarks**: ScienceQA (K-12) → SciBench (college) → GPQA (expert-level) → PhysReason (complex reasoning).
- **LLM Evaluation Methods**: The proposed PSAS framework can be generalized to other evaluation scenarios requiring multi-step reasoning (e.g., mathematical proofs, program debugging).
- **Insight**: The paradigm of step-level evaluation paired with error analysis holds valuable implications for all multi-step reasoning tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The step-level evaluation framework and error analysis present clear innovations, and the benchmark itself fills a gap in physics reasoning evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 15 models, solid validation of evaluation framework reliability, and in-depth error type analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-founded benchmark design, though some LaTeX formulas are closely packed.
- **Value**: ⭐⭐⭐⭐⭐ Fills an important gap in physics reasoning evaluation, and the PSAS framework holds broad reference value for multi-step reasoning evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](../../ICCV2025/llm_evaluation/3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](financereasoning_benchmarking_financial_numerical_reasoning_more.md)
- [\[ICLR 2026\] PRISM-Physics: Causal DAG-Based Process Evaluation for Physics Reasoning](../../ICLR2026/llm_evaluation/prism-physics_causal_dag-based_process_evaluation_for_physics_reasoning.md)
- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)

</div>

<!-- RELATED:END -->
