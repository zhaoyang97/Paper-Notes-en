---
title: >-
  [Paper Note] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection
description: >-
  [ACL 2026][vlm_reasoning][Paper Note] This paper formally defines the multimodal error detection task and constructs the ErrorRadar benchmark—comprising 2,500 K-12 multimodal math problems derived from real student responses. It evaluates MLLM capabilities in two subtasks: error step localization (STEP) and error type classification (CATE), finding that th
tags:
  - ACL 2026
  - vlm_reasoning
date: 2026-05-08
content_hash: dcd458e8f4e16398
---
# ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection

**Conference**: ACL 2026 Findings  
**arXiv**: [2410.04509](https://arxiv.org/abs/2410.04509)  
**Code**: None  
**Area**: Multimodal VLM / Mathematical Reasoning Evaluation  
**Keywords**: Multimodal Error Detection, Mathematical Reasoning Benchmark, K-12 Education, Error Step Localization, Error Classification

## TL;DR

This paper formally defines the multimodal error detection task and constructs the ErrorRadar benchmark—comprising 2,500 K-12 multimodal math problems derived from real student responses. It evaluates MLLM capabilities in two subtasks: error step localization (STEP) and error type classification (CATE), finding that the strongest model, GPT-4o, still trails human evaluation by approximately 10-15%.

## Background & Motivation

**Background**: Current mathematical reasoning benchmarks (e.g., MathVista, MathVerse, MATH-V) primarily evaluate the problem-solving capabilities of MLLMs, focusing on whether the models can correctly solve mathematical problems. MLLMs have achieved significant progress on these benchmarks.

**Limitations of Prior Work**: (1) Existing benchmarks focus only on "solution accuracy," ignoring the critical user requirement in educational scenarios—error detection; (2) In real educational settings, it is necessary not only to find the first erroneous step in a student's derivation but also to determine the error type (visual perception/calculation/reasoning/knowledge/misunderstanding), which is a complex task requiring deep understanding of mathematical concepts and cognitive processes; (3) Existing benchmarks lack real student response data, failing to reflect actual pedagogical needs.

**Key Challenge**: High scores on problem-solving benchmarks do not imply an understanding of erroneous reasoning—error detection requires deeper mathematical understanding and multi-step reasoning verification capabilities, dimensions not covered by current evaluation systems.

**Goal**: (1) Formally define the multimodal error detection task; (2) Construct a high-quality benchmark based on real student data; (3) Systematically evaluate the error detection capabilities of 20+ MLLMs.

**Key Insight**: Starting from the practical needs of educational scenarios—when a student submits an incorrect solution, a teacher needs to locate the error step and determine the error type. This is more challenging than simple problem-solving because it requires simultaneous understanding of the correct solution and the erroneous reasoning path.

**Core Idea**: Elevate mathematical reasoning evaluation from "can it solve" to "can it diagnose errors"—the latter requires stronger reasoning verification and cognitive understanding capabilities, which more authentically reflects the depth of MLLM mathematical reasoning.

## Method

### Overall Architecture

ErrorRadar defines two subtasks: given a multimodal math problem $\mathcal{I}_i = \{Q_{text,i}, Q_{image,i}, A_{correct,i}, A_{incorrect,i}, \{S_{k,i}\}_{k=1}^{n_i}\}$, (1) the STEP task locates the first erroneous step $x_i = \arg\min_k \{S_{k,i} \text{ is incorrect}\}$; (2) the CATE task classifies the error into one of five categories: VIS/CAL/REAS/KNOW/MIS. Data is sourced from real K-12 math question banks of global educational institutions and constructed through expert annotation.

### Key Designs

**1. Collection and Annotation of Real Student Data: Replacing Artificial Errors with High-Frequency Errors from Million-Scale Repositories**

The success of an error detection benchmark first depends on the "source of errors." Artificial errors are often systematic and predictable, failing to reflect real cognitive biases of students. Thus, this work started with a million-scale repository from educational institutions, filtering approximately 180,000 single-image math problems based on content universality and clarity, then selecting the most frequent incorrect answers as student responses (while excluding noise such as system input errors). This ensures every error represents a mistake actually made by students.

The annotation process involved 10 educational experts performing two rounds of cross-checking to identify the first error step and error type for each problem, with disputes resolved by an annotation lead. This process of "real high-frequency errors + expert double-blind cross-checking" ensures that ErrorRadar measures the diagnostic power of models regarding real cognitive biases rather than fitting synthetic patterns.

**2. Five-Category Error Taxonomy: Mapping the Error Spectrum from Low-Level Perception to High-Level Cognition**

Merely judging "correct or incorrect" is insufficient; educational scenarios require identifying the "type of error." This paper defines five error categories along cognitive levels: Visual Perception (VIS, failure to interpret image information), Calculation (CAL, arithmetic errors), Reasoning (REAS, improper logic), Knowledge (KNOW, incomplete understanding of concepts), and Misunderstanding (MIS, failure to understand problem requirements). This spectrum covers everything from low-level visual processing to high-level comprehension.

In real data, these five categories are naturally imbalanced: REAS (38.0%) and CAL (36.5%) constitute the majority, while KNOW (4.8%) and MIS (4.9%) are sparse. This skew serves as a signal—in subsequent experiments, weak models tended to guess CAL for all answers, leading to hollow F1 scores, which evidenced bias caused by this distribution.

**3. Three-Stage Evaluation Protocol: Rule-Based Answer Extraction + Three-Round Averaging to Eliminate Noise**

To ensure comparability across 20+ models, the evaluation process must be standardized. ErrorRadar splits it into three steps: MLLM response generation, answer extraction via template matching rules, and finally scoring. The STEP subtask uses accuracy $Acc_{step} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(x_i = G_{step,i})$ to measure the hit rate of error step localization, while the CATE subtask uses Precision/Recall/F1 and their macro-averages to evaluate classification quality. In task definition, STEP seeks the first error step $x_i = \arg\min_k \{S_{k,i} \text{ is incorrect}\}$, and CATE classifies the error into VIS/CAL/REAS/KNOW/MIS.

Two engineering details determine the credibility of results: using template rules instead of LLM-as-Judge for extraction avoids preference bias of the judge model; each model is run for three rounds and averaged to suppress random fluctuations. By using expert performance as the upper bound reference, the protocol fairly places models and humans on the same scale.

### Loss & Training

ErrorRadar is an evaluation benchmark and does not involve training. Evaluation was conducted on 20+ models (including open-source and closed-source), with educational expert performance as the human ceiling reference.

## Key Experimental Results

### Main Results

**Performance Comparison of Major Models**

| Model Type | Model | STEP Acc↑ | CATE F1↑ |
|---------|------|----------|----------|
| Closed-source | GPT-4o | **55.1** | **53.1** |
| Closed-source | Gemini-Pro-1.5 | 52.3 | 47.8 |
| Closed-source | Claude-3.5-Sonnet | 50.7 | 45.2 |
| Open-source | InternVL2-76B | 54.4 | 49.6 |
| Open-source | LLaVA-NEXT-72B | 51.8 | 46.3 |
| Human | Educational Expert | **69.8** | **60.7** |

### Scaling Analysis

| Model Series | Scale | STEP Acc↑ | CATE Acc↑ |
|---------|------|----------|----------|
| InternVL2 | 2B (Tiny) | 9.8 | - |
| InternVL2 | 8B (Small) | 30.4 | - |
| InternVL2 | 26B (Middle) | 42.1 | - |
| InternVL2 | 76B (Large) | **54.4** | - |
| LLaVA-NEXT | 7B (Small) | 30.3 | - |
| LLaVA-NEXT | 72B (Large) | **51.8** | - |

### Key Findings

- Closed-source models generally outperform open-source models; GPT-4o performs strongest but still trails humans by approximately 15% (STEP) and 8% (CATE).
- Weak models over-rely on the CAL category—for instance, MiniCPM-LLaMA3-v2.5 achieved 100% recall on CAL, but over 80% of its total predictions were CAL, exposing the issue of overfitting to simple categories.
- The STEP task is generally easier than the CATE task—locating an error step requires a lower cognitive level than determining the error type, similar to how localization is simpler than classification in object detection.
- STEP performance follows a scaling-law-like trend as model size increases, but CATE performance may actually decrease at large scales—indicating that error classification requires specialized training rather than just scale.
- Math-specific models (e.g., G-LLaVA) performed worse, suggesting that problem-solving ability does not equate to error diagnosis ability.

## Highlights & Insights

- Real student data is the core value—unlike synthetic errors, real errors reflect specific cognitive bias patterns, giving the benchmark educational practical significance.
- The discovery that "solving ability $\neq$ error diagnosis ability" is a critical warning for educational AI deployment—high scores of current MLLMs on solving benchmarks may lead to misleading deployment decisions.
- The phenomenon of weak models overfitting the CAL category provides a direction for improvement—category preferences could be corrected during training through weighting strategies like Focal Loss.

## Limitations & Future Work

- The dataset size (2,500 problems) is relatively limited; the problem types and visual representations in K-12 mathematics extend far beyond this.
- Currently a static evaluation, it does not account for interactive error correction (e.g., guiding students to correct errors).
- Only single-turn error detection is evaluated, without involving multi-turn diagnostic dialogues.
- Imbalanced error type distribution (KNOW and MIS account for only about 5%) may affect evaluation fairness.

## Related Work & Insights

- **vs MathVista/MathVerse**: These benchmarks evaluate solving ability, while ErrorRadar evaluates error diagnosis ability—the latter is more critical for educational applications.
- **vs EIC (ACL Findings)**: EIC also involves error detection but is limited to plain text; ErrorRadar performs this in a multimodal setting for the first time.
- **vs MR-GSM8K**: MR-GSM8K evaluates reasoning verification ability but uses synthetic data, whereas ErrorRadar uses real student data.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic formalization of multimodal error detection tasks, filling an evaluation gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20+ model evaluations + human baseline + scaling analysis + multi-dimensional findings.
- Writing Quality: ⭐⭐⭐⭐ Clear task formalization and well-summarized findings.
- Value: ⭐⭐⭐⭐ Direct practical significance for educational AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AXG-Reasoner: Error Detection and Explanation in Long Task Videos with Vision-Language Models](../../CVPR2026/vlm_reasoning/axg-reasoner_error_detection_and_explanation_in_long_task_videos_with_vision-lan.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[CVPR 2026\] EduDiag: A Benchmark for Educational Diagnostic Reasoning with Error Tracing and Correction on Large Multimodal Models](../../CVPR2026/vlm_reasoning/edudiag_a_benchmark_for_educational_diagnostic_reasoning_with_error_tracing_and_.md)

</div>

<!-- RELATED:END -->
