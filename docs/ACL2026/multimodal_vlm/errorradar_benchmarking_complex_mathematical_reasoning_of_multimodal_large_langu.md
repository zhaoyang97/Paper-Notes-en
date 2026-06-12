---
title: >-
  [Paper Note] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection
description: >-
  [ACL 2026][Multimodal VLM][Multimodal Error Detection] This paper formally defines the task of multimodal error detection and constructs the ErrorRadar benchmark—comprising 2…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal Error Detection"
  - "Mathematical Reasoning Benchmark"
  - "K-12 Education"
  - "Error Step Localization"
  - "Error Classification"
date: 2026-05-08
content_hash: 61f3dde233543271
---

# ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection

**Conference**: ACL 2026  
**arXiv**: [2410.04509](https://arxiv.org/abs/2410.04509)  
**Code**: None  
**Area**: Multimodal VLM / Mathematical Reasoning Evaluation  
**Keywords**: Multimodal Error Detection, Mathematical Reasoning Benchmark, K-12 Education, Error Step Localization, Error Classification

## TL;DR

This paper formally defines the task of multimodal error detection and constructs the ErrorRadar benchmark—comprising 2,500 K-12 multimodal math problems derived from real student responses. It evaluates MLLM capabilities in two sub-tasks: error step identification (STEP) and error category classification (CATE), finding that even the strongest model, GPT-4o, still lags behind human evaluators by approximately 10-15%.

## Background & Motivation

**Background**: Current mathematical reasoning benchmarks (e.g., MathVista, MathVerse, MATH-V) primarily evaluate the problem-solving capabilities of MLLMs, focusing on whether models can correctly solve mathematical problems. MLLMs have made significant progress on these benchmarks.

**Limitations of Prior Work**: (1) Existing benchmarks only focus on "accuracy of solutions," ignoring a more critical user need in educational scenarios—error detection; (2) In real educational settings, it is necessary not only to find the first erroneous step in a student's solution process but also to determine the error type (visual perception/calculation/reasoning/knowledge/misunderstanding), which is a complex task requiring a deep understanding of mathematical concepts and cognitive processes; (3) Existing benchmarks lack data from real student responses, failing to reflect actual pedagogical needs.

**Key Challenge**: High scores on problem-solving benchmarks do not imply that MLLMs understand erroneous reasoning—error detection requires deeper mathematical understanding and multi-step reasoning verification capabilities, dimensions not covered by the current evaluation system.

**Goal**: (1) Formally define the multimodal error detection task; (2) Construct a high-quality benchmark based on real student data; (3) Systematically evaluate the error detection capabilities of over 20 MLLMs.

**Key Insight**: Starting from the practical needs of educational scenarios—when a student submits an incorrect solution, a teacher needs to locate the error step and determine the error type. This is more challenging than simple problem-solving, as it requires simultaneous understanding of both the correct method and the erroneous reasoning path.

**Core Idea**: Elevate mathematical reasoning evaluation from "can solve problems" to "can diagnose errors"—the latter requires stronger reasoning verification and cognitive understanding capabilities, providing a more authentic reflection of the depth of MLLM mathematical reasoning.

## Method

### Overall Architecture

ErrorRadar defines two sub-tasks: given a multimodal math problem $\mathcal{I}_i = \{Q_{text,i}, Q_{image,i}, A_{correct,i}, A_{incorrect,i}, \{S_{k,i}\}_{k=1}^{n_i}\}$, (1) the STEP task locates the first error step $x_i = \arg\min_k \{S_{k,i} \text{ is incorrect}\}$; (2) the CATE task categorizes the error into one of five types: VIS/CAL/REAS/KNOW/MIS. Data is sourced from real K-12 math problem banks of global educational institutions and constructed through expert annotation.

### Key Designs

1. **Data Collection and Annotation Pipeline**:

    - Function: Build a high-quality benchmark from real student interaction data.
    - Mechanism: Approximately 180,000 single-image math problems were filtered from a million-scale question bank of educational institutions, refined by content universality and clarity of expression. For each problem, the most frequent incorrect answer was selected as the student response (excluding system input errors). About 10 educational experts performed two rounds of cross-check annotation for error steps and types, with discrepancies resolved by an annotation lead.
    - Design Motivation: Using real student error data rather than synthetic errors ensures the benchmark reflects actual cognitive bias patterns.

2. **Five-Category Error Classification System**:

    - Function: Cover the primary cognitive dimensions of mathematical errors.
    - Mechanism: Defines visual perception errors VIS (failure to interpret image information), calculation errors CAL (arithmetic operation errors), reasoning errors REAS (inappropriate logical reasoning), knowledge errors KNOW (incomplete understanding of knowledge points), and misunderstanding MIS (incorrectly understanding problem requirements). In the data distribution, REAS (38.0%) and CAL (36.5%) dominate, while KNOW (4.8%) and MIS (4.9%) are less frequent.
    - Design Motivation: The classification system covers a complete error spectrum from perception to higher-order cognition, with each category corresponding to different cognitive ability requirements.

3. **Evaluation Protocol Design**:

    - Function: Standardize the evaluation process to ensure comparability.
    - Mechanism: Three-stage evaluation—MLLM generates a response, answer extraction, and score calculation. STEP uses accuracy $Acc_{step} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(x_i = G_{step,i})$, while CATE uses Precision/Recall/F1 and their macro-average. Each model undergoes three evaluation rounds to obtain an average.
    - Design Motivation: Template matching rules for answer extraction avoid LLM-as-Judge bias, and the three-round average reduces random fluctuation.

### Loss & Training

ErrorRadar is an evaluation benchmark and does not involve training. Over 20 models (including open-source and closed-source) were evaluated, with human performance from educational experts used as an upper-bound reference.

## Key Experimental Results

### Main Results

**Primary Model Performance Comparison**

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

- Closed-source models generally outperform open-source models; GPT-4o is the strongest but still lags behind humans by about 15% in STEP and 8% in CATE.
- Weak models over-rely on the CAL category—for instance, MiniCPM-LLaMA3-v2.5 achieved 100% recall on CAL, but over 80% of its actual predictions were CAL, exposing the issue of overfitting to simple categories.
- The STEP task is generally easier than CATE—locating an error step requires a lower cognitive level than judging the error type, similar to how localization is easier than classification in object detection.
- STEP performance follows a scaling-law-like trend as model size increases, but CATE performance may actually decrease at large scales—indicating that error classification requires specialized training rather than just scale.
- Specialized math models (e.g., G-LLaVA) performed worse—demonstrating that problem-solving ability does not equal error diagnosis ability.

## Highlights & Insights

- Real student data is the core value—unlike synthetic errors, real errors reflect specific cognitive bias patterns, making the benchmark pedagogically significant.
- The finding that "problem-solving ability $\neq$ error diagnosis ability" is a critical warning for educational AI deployment—high scores on current problem-solving benchmarks may mislead deployment decisions.
- The phenomenon of weak models overfitting to the CAL category provides a direction for improvement—category preferences could be corrected during training through weighting strategies like Focal Loss.

## Limitations & Future Work

- The dataset size (2,500 questions) is relatively limited; the problem types and visual representations covered by K-12 mathematics extend far beyond this.
- Currently a static evaluation, not accounting for interactive error correction (e.g., guiding students to correct errors).
- Evaluation only covers single-turn error detection, without involving multi-turn diagnostic dialogues.
- Imbalanced error type distribution (KNOW and MIS account for only about 5%) may affect evaluation fairness.

## Related Work & Insights

- **vs MathVista/MathVerse**: These benchmarks evaluate problem-solving ability, whereas ErrorRadar evaluates error diagnosis ability—the latter is more critical for educational applications.
- **vs EIC (ACL Findings)**: EIC also involves error detection but is limited to pure text; ErrorRadar is the first to perform this in a multimodal setting.
- **vs MR-GSM8K**: MR-GSM8K evaluates reasoning verification ability but uses synthetic data, whereas ErrorRadar uses real student data.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematize the multimodal error detection task, filling an evaluation gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 20+ models + human baseline + scaling analysis + multi-dimensional findings.
- Writing Quality: ⭐⭐⭐⭐ Clear task formalization and well-summarized findings.
- Value: ⭐⭐⭐⭐ Direct practical significance for educational AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](../../ICLR2026/multimodal_vlm/visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)

</div>

<!-- RELATED:END -->
