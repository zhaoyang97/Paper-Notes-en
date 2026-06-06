---
title: >-
  [Paper Note] EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions
description: >-
  [ACL 2026][Multimodal VLM][STEM handwritten understanding] The authors release the EDU-CIRCUIT-HW dataset, containing 1,334 real-world university circuit course handwritten assignments…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "STEM handwritten understanding"
  - "MLLM evaluation"
  - "auto-grading"
  - "recognition error propagation"
  - "human-in-the-loop"
date: 2026-05-08
content_hash: dfaefc1db09de2e8
---

# EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions

**Conference**: ACL 2026  
**arXiv**: [2602.00095](https://arxiv.org/abs/2602.00095)  
**Code**: Project Website + GitHub (Links provided in the paper)  
**Area**: Multimodal VLM / Educational Evaluation  
**Keywords**: STEM handwritten understanding, MLLM evaluation, auto-grading, recognition error propagation, human-in-the-loop

## TL;DR
The authors release the EDU-CIRCUIT-HW dataset, containing 1,334 real-world university circuit course handwritten assignments, and propose a dual-layer evaluation protocol consisting of "upstream recognition + downstream grading." It is found that even the strongest MLLMs (GPT-5.1 / Gemini-3-Preview) contain recognition errors in 37–85% of samples, though only 7–20% propagate to the grading stage. Through an error-pattern-based LLM-judge regrading module with only 3.3% human intervention, point-agreement is improved from 70% to 76%.

## Background & Motivation

**Background**: Using MLLMs as "automated grading teaching assistants" has emerged as a new trend in AI education: first utilizing Gemini/GPT/Claude to recognize handwritten assignments, followed by LLM-based grading according to a rubric (Kortemeyer 2024, Liu 2024, Yang 2025, etc.). However, most existing evaluations focus on simple K-12 mathematics (DrawEduMath) or isolated formulas (CROHME, MathWriting), failing to reflect the complex handwritten text found in university STEM courses that intertwines formulas, derivations, and hand-drawn circuit diagrams.

**Limitations of Prior Work**: The authors highlight two fundamental issues: (1) **Data Scarcity**: Lack of benchmarks containing "mixed text-image content + university-level difficulty + real student handwriting"; (2) **Evaluation Paradigm Mismatch**: Prior works often only evaluate downstream results (mostly coarse-grained binary auto-grading), causing recognition errors outside the rubric to be "shielded," leading developers to overestimate the visual understanding capabilities of MLLMs. For example, in Figure 1, items ① and ② are misrecognized but hidden because they are not part of the grading points.

**Key Challenge**: The "latency rate" of recognition errors is much higher than the "manifestation rate"—once rubrics are tightened or downstream tasks like circuit-to-netlist are performed, these latent errors will explode. Traditional evaluation protocols focus solely on "grading agreement" and fail to identify them.

**Goal**: To establish a dual-metric system for "upstream recognition fidelity + downstream grading" to quantitatively answer (i) the prevalence of recognition errors, (ii) which types are most critical, and (iii) whether error patterns can be used for defense.

**Key Insight**: A dual-split approach is used, creating an "observation set" (513 snapshots with word-for-word expert verifications for training/analysis) and a "test set" (821 snapshots with only ground-truth scores to simulate generalization/deployment). An LLM-as-a-judge is employed to automatically list and then categorize recognition errors.

**Core Idea**: Use "expert word-for-word transcription" as an oracle to calculate recognition errors, define high-level Error Impact Rate (EIR) to map recognition errors to grading errors, and finally implement a regrading pipeline via "error patterns $\to$ low-confidence routing $\to$ human-in-the-loop" to transform recognition vulnerability into a controllable cost.

## Method

### Overall Architecture
The entire benchmark and diagnostic workflow is as follows: (1) **Data Collection**: 1,334 handwritten solutions from 29 students across 62 textbook problems in a Spring 2025 undergraduate circuit course at a US research university. Experts provided 5-dimensional rubric scores (E / M / U / C / NC). The observation set (11 students, 513 snapshots) provides expert transcriptions in Markdown and natural language descriptions of diagrams; the test set (18 students, 821 snapshots) contains only ground-truth scores. (2) **Recognition Evaluation**: 6 MLLMs (Gemini-3-Pro-Preview, Gemini-2.5-Pro, GPT-5.1, Claude-4.5-Sonnet, Qwen3-VL-Plus/8B-Thinking) perform recognition. Gemini-2.5-Pro serves as the LLM-judge comparing against the oracle to list discrepant items, which are then classified into four categories (Symbolic & Character / Structural & Notational / Diagrammatic / Textual & Logical). (3) **Downstream Grading**: GPT-5.1 is fixed as the grader, outputting scores based on problem + reference + rubric. Binary / Type / Point agreement is calculated against expert reports. (4) **Impact Analysis**: EIR is defined as the ratio of recognition errors causing scoring differences to total recognition errors. (5) **Regrading Case Study**: Error patterns summarized from the observation set are injected into prompts to detect potential recognition errors in the test set, routing low-confidence samples to humans for regrading.

### Key Designs

1.  **Dual-layer Evaluation Protocol ($\text{SER}$ / $\text{AEC} + \text{EIR} + \text{Binary/Type/Point Agreement}$)**:
    *   **Function**: Decouples "recognition" and "grading" and quantifies error propagation.
    *   **Mechanism**: The recognition side uses Sample Error Rate $\text{SER}=\frac{\#\{s: \text{errors}(s)>0\}}{|S|}$ and Average Error Count $\text{AEC}=\frac{1}{|S|}\sum_s \#\text{errors}(s)$. The grading side uses three progressive levels: Binary $\to$ Type $\to$ Point agreement. These are bridged by $\text{EIR}=\frac{\text{number of recognition errors causing grading discrepancies}}{\text{total recognition errors}}$.
    *   **Design Motivation**: Relying solely on task-centric metrics like "auto-grading accuracy" allows many "silent errors" to escape. EIR quantitatively answers "how poor must recognition be to harm downstream tasks," a metric missing in most vision-to-reasoning pipelines outside of education.

2.  **LLM-as-a-Judge Recognition Error Listing + Four-category Taxonomy**:
    *   **Function**: Automatically identifies semantic differences between MLLM recognition results and expert transcriptions, categorizing them from "surface to deep."
    *   **Mechanism**: Oracle Markdown and tested Markdown are fed to Gemini-2.5-Pro to list discrepant items at the sentence/formula level. Semantically equivalent formatting differences (e.g., `KCL: out` $\equiv$ `KCL: @ out`) are aligned. Items are then archived into Symbolic & Character, Structural & Notational, Diagrammatic, and Textual & Logical categories. Human validation on 186 samples shows sample-level accuracy $\ge 0.95$ and item-level $F1 \ge 0.90$.
    *   **Design Motivation**: Manual annotation of every item is unscalable. Defining the judge task as "listing differences + classification" with an oracle restricts the LLM to "comparative checking" rather than "open-ended scoring," minimizing hallucination errors.

3.  **Error-Pattern-Driven Human-in-the-Loop Regrading Module**:
    *   **Function**: Uses error patterns from the observation set as "risk features" to scan the test set, intercepting suspicious samples for human intervention.
    *   **Mechanism**: Common confusion patterns (e.g., $-V\to V$, $\frac{1/8}{1/8+1/16}\to \frac{8}{8+16}$) are extracted from the observation set and placed into a detector prompt. The detector scans samples with initial point deductions and flags them as high/low confidence. Low confidence samples are sent to TAs; high confidence samples are regraded by the LLM. Samples with no deduction are passed (as recognition errors primarily cause false-positive deductions).
    *   **Design Motivation**: Full automation is unacceptable in high-stakes education, while full manual grading is too expensive. This design assumes error patterns are statistically consistent, suppressing manual labor to $\le 5\%$ while pushing point-agreement toward the "expert-transcription" ceiling.

### Loss & Training
This work involves no model training; it is a **prompt-only evaluation + LLM-judge pipeline**. the grader is fixed as GPT-5.1. The recognition side covers 5 commercial models and 1 open-source 8B model. GPT-5.1 is used consistently for the detector/regrader/grader in the regrading module to eliminate inter-model heterogeneity interference.

## Key Experimental Results

### Main Results
Recognition quality of six MLLMs on the observation set and its impact on 5-dimensional rubric grading (GPT-5.1 as grader; Graduate TA as baseline; Human Expert row represents oracle grader with expert transcriptions):

| Recognizer | $\text{SER} \downarrow$ | $\text{AEC} \downarrow$ | Binary $\uparrow$ | Type $\uparrow$ | Point $\uparrow$ | $\text{EIR} \downarrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Graduate (Human) | – | – | 83.63 | 82.46 | 81.29 | – |
| Human Expert (Oracle) | – | – | **89.47** | 78.36 | 74.46 | – |
| Gemini-3-Preview | **37.62** | **0.61** | 87.91 | **78.17** | **74.27** | **7.60** |
| Gemini-2.5-Pro | 53.52 | 1.23 | 85.58 | 73.68 | 69.40 | 14.72 |
| Qwen3-VL-Plus | 61.72 | 1.38 | 80.90 | 68.62 | 65.11 | 16.67 |
| GPT-5.1 | 71.54 | 2.05 | 77.78 | 65.50 | 61.99 | 17.89 |
| Claude-4.5-Sonnet | 80.70 | 2.76 | 77.58 | 63.16 | 59.84 | 18.05 |
| Qwen3-VL-8B-Thinking| 85.43 | 2.79 | 75.05 | 61.01 | 56.92 | 19.60 |

Key points: (1) Even the strongest Gemini-3-Preview has recognition errors in 37.6% of samples, yet its EIR is only 7.6%, indicating downstream grading hides significant recognition issues. (2) From Gemini-3-Preview to Qwen3-VL-8B-Thinking, stricter rubrics lead to larger performance gaps (12.86% in Binary vs. 17.35% in Point), confirming that rubric tightening manifests recognition errors. (3) MLLMs can exceed Graduate TAs in Binary agreement but still lag in Type/Point, suggesting LLMs are more lenient while humans are more precise.

### Ablation Study
Comparison between the vanilla pipeline and the regrading module on the test set:

| Workflow | Visual Recognizer | Binary | Type | Point | LLM Regrade | Human Regrade |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Vanilla | Gemini-2.5-Pro | 85.02 | 74.91 | 69.91 | – | – |
| Vanilla | GPT-5.1 | 82.34 | 72.23 | 66.87 | – | – |
| **+ Regrading** | Gemini-2.5-Pro | **86.48** | **77.34** | **74.42** | 20.6% | 3.3% |
| **+ Regrading** | GPT-5.1 | **86.60** | **78.93** | **75.76** | 25.1% | 4.4% |

Key points: With $\le 5\%$ human intervention, Point agreement improved from ~70% to 76%, slightly exceeding the "expert recognition" ceiling of 74.46% because the detector proactively helps the grader avoid pitfalls.

### Key Findings
*   **Symbolic & Character errors are most common**, and their EIR is also the highest (≈20%) because the grader relies heavily on symbol matching. Diagrammatic and Textual & Logical errors involve higher cognitive levels, but current rubrics barely cover them, resulting in EIR $< 10\%$—a form of "survivorship bias" in auto-grading.
*   **Finer rubrics better distinguish models**: The gap between models widens from ~13% to ~17% across Binary/Type/Point levels, implying that AI education evaluations must use point-level rubrics for diagnostic value.
*   **Small models are not necessarily worse at diagrams**: Qwen3-VL-8B-Thinking had 98 Diagrammatic errors, outperforming Gemini-2.5-Pro's 103, reflecting that commercial models excel in textual reasoning rather than graphic understanding.
*   **Regrading yields gains even without top-tier MLLMs**: Even using Gemini-2.5-Pro as a recognizer, the detector + 3.3% human effort improved Point agreement by +4.5%, proving the high ROI of "error pattern + human-in-the-loop."

## Highlights & Insights
*   **Dual-layer evaluation implemented in 'high-stakes' scenarios**: Previous handwriting evaluations stopped at OCR digits. This work repositioned "recognition fidelity" as the bottleneck for reliability and used EIR to quantify "silent errors." This "decouple then bridge" philosophy can be applied to any perception-to-reasoning pipeline.
*   **Observation/Test Dual-split data design**: Concentrating "expert word-for-word verification" costs on ~40% of data for diagnostic learning while using score oracles for the remaining 60% provides a balanced, high-density benchmark paradigm.
*   **LLM-as-a-Judge "Difference Listing" mode**: Defining the judge's task as enumerating and categorizing differences rather than direct scoring restricted LLM freedom and stabilized $F1$ above 0.9.
*   **Error pattern $\to$ Routing $\to$ Human-in-the-loop deployment framework**: Transforms the "recognition reliability" problem into a "controllable human-labor ratio" problem, offering a blueprint for actual AI grading deployment.

## Limitations & Future Work
*   The dataset only covers one circuit analysis course; diagrams are circuit-specific. Generalization to geometry, chemical structures, or flowcharts requires caution.
*   Downstream tasks are limited to auto-grading; sensitivity to recognition errors may vary significantly for VQA, circuit-to-netlist, or tutoring tasks.
*   Rubrics and ground-truth are provided by a small pool of doctoral experts; open-ended STEM grading is subjective and may contain systematic bias.
*   The use of GPT-5.1 for the detector, regrader, and grader may involve implicit self-confirmation bias, requiring cross-validation with heterogeneous models.
*   Future work could expand across disciplines and downstream tasks, incorporating continuous learning for error patterns.

## Related Work & Insights
*   **vs. DrawEduMath (Baral 2025)**: They focus on K-12 math VQA; this work moves to university STEM with complex solutions and explicit point-level rubrics.
*   **vs. CROHME / MathWriting**: These evaluate isolated formula OCR; this work evaluates "formulas + derivations + diagrams" mixed text, covering the "long tail" of recognition failure.
*   **vs. Pensieve Grader (Yang 2025) / Liu (2024)**: They focus on end-to-end grading; this work evaluates recognition separately and uses EIR to explain downstream error sources.
*   **vs. HTR Correction (Pavlopoulos 2023)**: They perform post-hoc correction; this work uses error patterns for pre-emptive filtering and routing, which is lighter and more suitable for LLM-only pipelines.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ The combination of dual-layer evaluation, EIR, and error pattern routing is novel and addresses real pain points.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage with 6 MLLMs, 4 error types, and 3 rubric tiers.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear progression from argument to evidence to solution; some sections are slightly wordy.
*   **Value**: ⭐⭐⭐⭐⭐ Directly addresses the reliability issue in AI grading with a deployable engineering solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)
- [\[ACL 2026\] GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance](guidedog_a_real-world_egocentric_multimodal_dataset_for_blind_and_low-vision_acc.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)

</div>

<!-- RELATED:END -->
