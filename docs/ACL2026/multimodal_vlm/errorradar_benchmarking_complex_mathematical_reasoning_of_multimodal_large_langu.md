---
title: >-
  [Paper Note] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection
description: >-
  [ACL 2026][Multimodal VLM][Paper Note] This paper formally defines the multimodal error detection task and constructs the ErrorRadar benchmark—comprising 2,500 K-12 multimodal math problems from real-world student responses—to evaluate MLLM performance in error step identification (STEP) and error category classification (CATE). The results show that the st
tags:
  - ACL 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 232e3f4de93d36da
---
# ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection

**Conference**: ACL 2026 Findings  
**arXiv**: [2410.04509](https://arxiv.org/abs/2410.04509)  
**Code**: None  
**Area**: Multimodal VLM / Mathematical Reasoning Evaluation  
**Keywords**: Multimodal Error Detection, Mathematical Reasoning Benchmark, K-12 Education, Error Step Location, Error Classification

## TL;DR

This paper formally defines the multimodal error detection task and constructs the ErrorRadar benchmark—comprising 2,500 K-12 multimodal math problems from real-world student responses—to evaluate MLLM performance in error step identification (STEP) and error category classification (CATE). The results show that the strongest model, GPT-4o, still lags behind human experts by approximately 10-15%.

## Background & Motivation

**Background**: Current mathematical reasoning benchmarks (e.g., MathVista, MathVerse, MATH-V) primarily evaluate the problem-solving capabilities of MLLMs, focusing on whether models can correctly solve mathematical problems. MLLMs have achieved significant progress on these benchmarks.

**Limitations of Prior Work**: (1) Existing benchmarks focus solely on "problem-solving accuracy," ignoring a critical user need in educational scenarios—error detection; (2) In real educational settings, it is necessary not only to locate the first error in a student's solution process but also to diagnose the error type (visual perception/calculation/reasoning/knowledge/misunderstanding), which is a complex task requiring deep understanding of mathematical concepts and cognitive processes; (3) Existing benchmarks lack data from real student responses, failing to reflect actual pedagogical requirements.

**Key Challenge**: High scores on problem-solving benchmarks do not imply that MLLMs understand erroneous reasoning—error detection requires deeper mathematical comprehension and multi-step reasoning verification, which are dimensions not covered by current evaluation frameworks.

**Goal**: (1) Formally define the multimodal error detection task; (2) Construct a high-quality benchmark based on real student data; (3) Systematically evaluate the error detection capabilities of 20+ MLLMs.

**Key Insight**: The research starts from the practical needs of educational scenarios—after students submit incorrect solutions, teachers need to locate error steps and judge the error types. This is more challenging than simple problem solving because it requires simultaneous understanding of the correct solution and the erroneous reasoning path.

**Core Idea**: Elevate mathematical reasoning evaluation from "ability to solve problems" to "ability to diagnose errors"—the latter requires stronger reasoning verification and cognitive understanding, more accurately reflecting the depth of MLLM mathematical reasoning.

## Method

### Overall Architecture

ErrorRadar defines two sub-tasks: given a multimodal math problem $\mathcal{I}_i = \{Q_{text,i}, Q_{image,i}, A_{correct,i}, A_{incorrect,i}, \{S_{k,i}\}_{k=1}^{n_i}\}$, (1) the STEP task locates the first error step $x_i = \arg\min_k \{S_{k,i} \text{ is incorrect}\}$; (2) the CATE task classifies the error into one of five categories: VIS/CAL/REAS/KNOW/MIS. Data is sourced from real K-12 math problem banks of global educational institutions and constructed through expert annotation.

### Key Designs

**1. Collection and Annotation of Real Student Data: Replacing artificial errors with high-frequency incorrect answers from a million-problem database**

The success of an error detection benchmark depends primarily on the source of the errors. Artificial errors are often systematic and predictable, failing to reflect real cognitive biases of students. Thus, this paper starts from an educational database of millions of problems, first filtering approximately 180,000 single-image math problems based on content universality and clarity, then selecting the most frequent incorrect answers for each problem as student responses (while excluding system input noise). This ensures every error represents a mistake actually made by a student.

The annotation phase involved approximately 10 educational experts performing two rounds of cross-checking to identify the first error step and error type for each problem, with disagreements resolved by an annotation lead. This process of "real high-frequency errors + expert double-blind cross-checking" ensures that ErrorRadar measures the diagnostic capability regarding real cognitive biases rather than fitting synthetic patterns.

**2. Five-category Error Taxonomy: Mapping the error spectrum from low-level perception to high-level cognition**

Simply judging "correct or incorrect" is insufficient; educational scenarios require identifying the specific "category of error." This paper defines five error types based on cognitive levels: Visual Perception VIS (failure to interpret image information), Calculation CAL (arithmetic errors), Reasoning REAS (inappropriate logical reasoning), Knowledge KNOW (incomplete understanding of knowledge points), and Misunderstanding MIS (failure to understand problem requirements). This spectrum covers everything from bottom-level image interpretation to top-level problem comprehension, each corresponding to a different cognitive capability gap.

In real data, these five categories are naturally imbalanced: REAS (38.0%) and CAL (36.5%) represent the majority, while KNOW (4.8%) and MIS (4.9%) are sparse. This skew acts as a signal—subsequent experiments showed that weak models tended to guess CAL for all answers, leading to artificially high F1 scores, which serves as evidence of being biased by this distribution.

**3. Three-stage Evaluation Protocol: Rule-based answer extraction + three-round averaging to eliminate evaluation noise**

To ensure comparability across 20+ models, the evaluation process is standardized. ErrorRadar divides it into three steps: the MLLM generates a response, answer extraction is performed using template matching rules, and finally, scores are calculated. The STEP sub-task uses accuracy $Acc_{step} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(x_i = G_{step,i})$ to measure the success of error step localization, while the CATE sub-task utilizes Precision/Recall/F1 and their macro-averages to evaluate classification quality. By definition, STEP identifies the first erroneous step $x_i = \arg\min_k \{S_{k,i} \text{ is incorrect}\}$, and CATE assigns the error to one of VIS/CAL/REAS/KNOW/MIS.

Two engineering details determine the credibility of the results: utilizing template rules instead of LLM-as-Judge for extraction avoids preference bias from the judge model; each model is run three times and averaged to suppress random fluctuations. By using educational experts' human performance as the upper bound reference, the protocol fairly compares models and humans on a unified scale.

### Loss & Training

ErrorRadar is an evaluation benchmark and does not involve training. 20+ models (including open and closed source) are evaluated, with educational experts' human performance used as the reference upper bound.

## Key Experimental Results

### Main Results

**Comparison of Main Model Performance**

| Model Type | Model | STEP Acc↑ | CATE F1↑ |
|:---|:---|:---|:---|
| Closed-source | GPT-4o | **55.1** | **53.1** |
| Closed-source | Gemini-Pro-1.5 | 52.3 | 47.8 |
| Closed-source | Claude-3.5-Sonnet | 50.7 | 45.2 |
| Open-source | InternVL2-76B | 54.4 | 49.6 |
| Open-source | LLaVA-NEXT-72B | 51.8 | 46.3 |
| Human | Educational Experts | **69.8** | **60.7** |

### Scaling Analysis

| Model Series | Scale | STEP Acc↑ | CATE Acc↑ |
|:---|:---|:---|:---|
| InternVL2 | 2B (Tiny) | 9.8 | - |
| InternVL2 | 8B (Small) | 30.4 | - |
| InternVL2 | 26B (Middle) | 42.1 | - |
| InternVL2 | 76B (Large) | **54.4** | - |
| LLaVA-NEXT | 7B (Small) | 30.3 | - |
| LLaVA-NEXT | 72B (Large) | **51.8** | - |

### Key Findings

- Closed-source models overall outperform open-source models; GPT-4o shows the strongest performance but still lags behind humans by approximately 15% (STEP) and 8% (CATE).
- Weak models over-rely on the CAL category—for instance, MiniCPM-LLaMA3-v2.5 achieves a 100% recall on CAL, but over 80% of its total predictions are CAL, exposing the problem of overfitting to simple categories.
- The STEP task is generally easier than CATE—locating an error step requires lower cognitive levels than judging the error type, similar to how localization is simpler than classification in object detection.
- STEP performance follows a scaling law trend as model size increases, but CATE performance may actually decrease at large scales—indicating that error classification requires specialized training rather than scale alone.
- Math-specific models (e.g., G-LLaVA) performed poorly—problem-solving ability is not equivalent to error diagnostic ability.

## Highlights & Insights

- Real student data provides core value—unlike artificial errors, real errors reflect specific cognitive bias patterns, making the benchmark pedagogically significant.
- The finding that "Problem-Solving Ability $\neq$ Error Diagnostic Ability" serves as a warning for educational AI deployment—high scores on current problem-solving benchmarks could lead to misleading deployment decisions.
- The phenomenon of weak models overfitting the CAL category indicates an improvement direction—category preferences could be corrected during training through weighting strategies like Focal Loss.

## Limitations & Future Work

- The dataset size (2,500 problems) is relatively limited; K-12 mathematics covers more problem types and visual representations than included.
- The evaluation is currently static and does not consider interactive error correction (e.g., guiding students to fix errors).
- Only single-turn error detection is evaluated, without involving multi-turn diagnostic dialogues.
- Uneven distribution of error types (KNOW and MIS account for only ~5%) may affect evaluation fairness.

## Related Work & Insights

- **vs MathVista/MathVerse**: These benchmarks evaluate problem-solving ability, whereas ErrorRadar evaluates error diagnostic ability—the latter is more critical for educational applications.
- **vs EIC (ACL Findings)**: EIC also addresses error detection but is limited to text only; ErrorRadar is the first to do so in a multimodal setting.
- **vs MR-GSM8K**: MR-GSM8K evaluates reasoning verification ability using synthetic data, while ErrorRadar uses real student data.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic multimodal error detection task, filling an evaluation gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20+ model evaluations + human baseline + scaling analysis + multi-dimensional findings.
- Writing Quality: ⭐⭐⭐⭐ Clear task formalization and insightful result summaries.
- Value: ⭐⭐⭐⭐ Direct practical significance for the deployment of educational AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[CVPR 2026\] EduDiag: A Benchmark for Educational Diagnostic Reasoning with Error Tracing and Correction on Large Multimodal Models](../../CVPR2026/multimodal_vlm/edudiag_a_benchmark_for_educational_diagnostic_reasoning_with_error_tracing_and_.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[CVPR 2026\] DiGraphHal-Bench: Evaluating Multimodal Large Language Models on Complex Directed Graphs](../../CVPR2026/multimodal_vlm/digraphhal-bench_evaluating_multimodal_large_language_models_on_complex_directed.md)

</div>

<!-- RELATED:END -->
