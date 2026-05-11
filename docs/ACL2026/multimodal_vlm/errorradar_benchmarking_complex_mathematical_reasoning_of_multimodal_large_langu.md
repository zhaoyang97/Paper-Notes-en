---
title: >-
  [Paper Note] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection
description: >-
  [ACL 2026][Multimodal VLM][Multimodal error detection] This paper formally defines the multimodal error detection task and constructs the ErrorRadar benchmark — comprising 2…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal error detection"
  - "mathematical reasoning benchmark"
  - "K-12 education"
  - "error step localization"
  - "error classification"
date: 2026-05-08
content_hash: fa71170fa6a49517
---

# ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection

**Conference**: ACL 2026
**arXiv**: [2410.04509](https://arxiv.org/abs/2410.04509)
**Code**: None
**Area**: Multimodal VLM / Mathematical Reasoning Evaluation
**Keywords**: Multimodal error detection, mathematical reasoning benchmark, K-12 education, error step localization, error classification

## TL;DR

This paper formally defines the multimodal error detection task and constructs the ErrorRadar benchmark — comprising 2,500 K-12 multimodal math problems drawn from real student responses — to evaluate MLLMs on two subtasks: error step identification (STEP) and error type classification (CATE). The strongest model, GPT-4o, still lags behind human evaluators by approximately 10–15%.

## Background & Motivation

**Background**: Existing mathematical reasoning benchmarks (e.g., MathVista, MathVerse, MATH-V) primarily assess the problem-solving ability of MLLMs, focusing on whether models can correctly solve mathematical problems. MLLMs have achieved substantial progress on these benchmarks.

**Limitations of Prior Work**: (1) Existing benchmarks focus solely on "solution correctness," overlooking error detection — a more critical requirement in educational scenarios. (2) In real educational settings, it is necessary not only to locate the first erroneous step in a student's solution process but also to identify the error type (visual perception / computation / reasoning / knowledge / misunderstanding), a complex task that requires deep understanding of mathematical concepts and cognitive processes. (3) Existing benchmarks lack real student response data and thus fail to reflect actual pedagogical needs.

**Key Challenge**: High scores on problem-solving benchmarks do not imply that MLLMs can understand erroneous reasoning — error detection demands deeper mathematical comprehension and multi-step reasoning verification, a dimension not covered by current evaluation frameworks.

**Goal**: (1) Formally define the multimodal error detection task; (2) construct a high-quality benchmark grounded in real student data; (3) systematically evaluate 20+ MLLMs on error detection capability.

**Key Insight**: The approach is motivated by real educational needs — when a student submits an incorrect solution, a teacher must locate the erroneous step and identify the error type. This is more challenging than mere problem solving, as it requires simultaneously understanding the correct solution and the erroneous reasoning path.

**Core Idea**: Elevating mathematical reasoning evaluation from "can the model solve problems" to "can the model diagnose errors" — the latter demands stronger reasoning verification and cognitive understanding, more faithfully reflecting the depth of MLLMs' mathematical reasoning.

## Method

### Overall Architecture

ErrorRadar defines two subtasks: given a multimodal math problem $\mathcal{I}_i = \{Q_{text,i}, Q_{image,i}, A_{correct,i}, A_{incorrect,i}, \{S_{k,i}\}_{k=1}^{n_i}\}$, (1) the STEP task localizes the first erroneous step $x_i = \arg\min_k \{S_{k,i} \text{ is incorrect}\}$; (2) the CATE task classifies the error into one of five categories: VIS/CAL/REAS/KNOW/MIS. Data are sourced from real K-12 mathematics problem repositories provided by global educational institutions and annotated by domain experts.

### Key Designs

1. **Data Collection and Annotation Pipeline**:

    - Function: Constructing a high-quality benchmark from real student interaction data.
    - Mechanism: Approximately 180,000 single-image math problems are filtered from a million-scale educational repository, refined by content generalizability and expression clarity. For each problem, the most frequently occurring incorrect answer is selected as the student response (excluding system input errors). Approximately 10 education experts conduct two rounds of cross-checking to annotate error steps and error types, with disagreements adjudicated by the annotation lead.
    - Design Motivation: Using real student errors rather than artificially constructed errors ensures the benchmark reflects authentic cognitive bias patterns.

2. **Five-Category Error Taxonomy**:

    - Function: Covering the primary cognitive dimensions of mathematical errors.
    - Mechanism: Five error types are defined — visual perception errors VIS (failure to interpret image information), calculation errors CAL (arithmetic mistakes), reasoning errors REAS (faulty logical inference), knowledge errors KNOW (incomplete understanding of concepts), and misinterpretation errors MIS (failure to correctly understand the problem requirements). In terms of data distribution, REAS (38.0%) and CAL (36.5%) dominate, while KNOW (4.8%) and MIS (4.9%) are less frequent.
    - Design Motivation: The taxonomy spans the full error spectrum from perceptual to higher-order cognitive failures, with each error type corresponding to distinct cognitive capability requirements.

3. **Evaluation Protocol Design**:

    - Function: Standardizing the evaluation procedure to ensure comparability.
    - Mechanism: A three-stage evaluation pipeline — MLLM response generation, answer extraction, and score computation. STEP is evaluated by accuracy $Acc_{step} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(x_i = G_{step,i})$; CATE is evaluated by Precision/Recall/F1 and their macro averages. Each model is evaluated over three runs and averaged.
    - Design Motivation: Template-matching rules for answer extraction avoid biases inherent to LLM-as-Judge approaches; three-run averaging reduces stochastic variance.

### Loss & Training

ErrorRadar is an evaluation benchmark and does not involve training. More than 20 models (both open-source and proprietary) are evaluated, with human performance by education experts serving as an upper-bound reference.

## Key Experimental Results

### Main Results

**Performance Comparison of Major Models**

| Model Type | Model | STEP Acc↑ | CATE F1↑ |
|-----------|-------|-----------|----------|
| Proprietary | GPT-4o | **55.1** | **53.1** |
| Proprietary | Gemini-Pro-1.5 | 52.3 | 47.8 |
| Proprietary | Claude-3.5-Sonnet | 50.7 | 45.2 |
| Open-source | InternVL2-76B | 54.4 | 49.6 |
| Open-source | LLaVA-NEXT-72B | 51.8 | 46.3 |
| Human | Education Experts | **69.8** | **60.7** |

### Scaling Analysis

| Model Family | Scale | STEP Acc↑ | CATE Acc↑ |
|-------------|-------|-----------|----------|
| InternVL2 | 2B (Tiny) | 9.8 | - |
| InternVL2 | 8B (Small) | 30.4 | - |
| InternVL2 | 26B (Middle) | 42.1 | - |
| InternVL2 | 76B (Large) | **54.4** | - |
| LLaVA-NEXT | 7B (Small) | 30.3 | - |
| LLaVA-NEXT | 72B (Large) | **51.8** | - |

### Key Findings

- Proprietary models generally outperform open-source models; GPT-4o achieves the best performance but still trails humans by approximately 15% (STEP) and 8% (CATE).
- Weaker models exhibit excessive reliance on the CAL category — for instance, MiniCPM-LLaMA3-v2.5 achieves 100% recall on CAL, yet over 80% of its predictions are classified as CAL, revealing overfitting to the most common category.
- The STEP task is generally easier than CATE — localizing an erroneous step requires a lower level of cognitive abstraction than identifying the error type, analogous to how localization is simpler than classification in object detection.
- STEP performance follows a scaling-law-like trend with increasing model size, whereas CATE performance may actually decline at larger scales — suggesting that error classification requires specialized training rather than scale alone.
- Math-specialized models (e.g., G-LLaVA) perform worse than expected — problem-solving ability does not equate to error diagnosis capability.

## Highlights & Insights

- Real student data is the core contribution — unlike artificially constructed errors, authentic errors reflect specific cognitive bias patterns, lending the benchmark direct educational relevance.
- The finding that "problem-solving ability ≠ error diagnosis capability" carries an important warning for educational AI deployment — high scores of current MLLMs on problem-solving benchmarks may mislead deployment decisions.
- The phenomenon of weaker models overfitting to the CAL category suggests a concrete improvement direction — category preference can be corrected during training via weighting strategies such as Focal Loss.

## Limitations & Future Work

- The dataset scale (2,500 problems) is relatively limited; K-12 mathematics encompasses far more problem types and visual representations.
- The current framework is static and does not account for interactive error correction (e.g., guiding students to correct their mistakes).
- Only single-turn error detection is evaluated; multi-turn diagnostic dialogue is not addressed.
- The uneven distribution of error types (KNOW and MIS each accounting for only ~5%) may affect evaluation fairness.

## Related Work & Insights

- **vs. MathVista/MathVerse**: These benchmarks assess problem-solving ability, whereas ErrorRadar assesses error diagnosis capability — the latter is more critical for educational applications.
- **vs. EIC (ACL Findings)**: EIC also involves error detection but is restricted to pure text; ErrorRadar is the first to conduct this evaluation in a multimodal setting.
- **vs. MR-GSM8K**: MR-GSM8K evaluates reasoning verification capability but uses synthetic data; ErrorRadar employs real student data.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic formulation of multimodal error detection, filling a gap in evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 20+ models + human baseline + scaling analysis + multi-dimensional findings.
- Writing Quality: ⭐⭐⭐⭐ Task formalization is clear; findings are well summarized.
- Value: ⭐⭐⭐⭐ Direct practical implications for educational AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](../../ICLR2026/multimodal_vlm/visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)

</div>

<!-- RELATED:END -->
