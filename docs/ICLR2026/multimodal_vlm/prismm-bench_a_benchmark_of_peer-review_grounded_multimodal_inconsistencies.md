---
title: >-
  [Paper Note] PRISMM-Bench: A Benchmark of Peer-Review Grounded Multimodal Inconsistencies
description: >-
  [ICLR 2026][Multimodal VLM][Multimodal Inconsistency] This work introduces PRISMM-Bench, the first benchmark grounded in genuine reviewer-annotated multimodal inconsistencies in scientific papers. Mining 18,009 ICLR open reviews yields 384 cross-modal inconsistencies, evaluated across three tasks—identification, remediation, and paired matching—with a JSON-structured debiasing scheme for answer representation. Among 21 state-of-the-art LMMs, the best achieves only 53.9%, systematically exposing severe deficiencies in cross-modal reasoning over scientific documents.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Multimodal Inconsistency
  - Peer Review
  - Scientific Papers
  - LMM Benchmark
  - JSON Debiasing
date: 2026-05-08
content_hash: b0a8cb6b97cf226c
---

# PRISMM-Bench: A Benchmark of Peer-Review Grounded Multimodal Inconsistencies

**Conference**: ICLR 2026
**arXiv**: [2510.16505](https://arxiv.org/abs/2510.16505)
**Code**: [Project Page](https://da-luggas.github.io/prismm-bench/)
**Area**: Multimodal Evaluation / Scientific Documents
**Keywords**: Multimodal Inconsistency, Peer Review, Scientific Papers, LMM Benchmark, JSON Debiasing

## TL;DR

This work introduces PRISMM-Bench, the first benchmark grounded in genuine reviewer-annotated multimodal inconsistencies in scientific papers. Mining 18,009 ICLR open reviews yields 384 cross-modal inconsistencies, evaluated across three tasks—identification, remediation, and paired matching—with a JSON-structured debiasing scheme for answer representation. Among 21 state-of-the-art LMMs, the best achieves only 53.9%, systematically exposing severe deficiencies in cross-modal reasoning over scientific documents.

## Background & Motivation

**Background**: Large multimodal models (LMMs) are increasingly employed to assist scientific research—interpreting figures, summarizing papers, and detecting errors. A fundamental question, however, remains unresolved: can LMMs genuinely understand and reason over the complex multimodal structure of scientific papers spanning text, figures, and equations?

**Limitations of Prior Work**:
- Existing document QA benchmarks (DocVQA, ChartQA, etc.) evaluate individual modalities in isolation, ignoring cross-modal dependencies among text, figures, and formulas.
- Synthetic datasets (e.g., MMIR) inject artificial errors, which tend to be overly conspicuous and fail to represent the subtle, domain-knowledge-demanding inconsistencies found in real-world scientific writing.
- Multiple-choice evaluation suffers from severe linguistic bias—models can achieve well above chance accuracy by reading answer choices alone without the question (e.g., Gemini 2.5 Flash reaches 57.6% without context).

**Key Challenge**: A benchmark that is simultaneously *authentic* and *systematic* is needed to assess cross-modal reasoning, yet real inconsistencies are rare, scattered, and costly to verify; moreover, evaluation itself is compromised by linguistic shortcuts.

**Goal**: (1) How can real cross-modal inconsistencies be systematically collected? (2) How can evaluation tasks be designed to be fair and unbiased?

**Key Insight**: Open peer review provides a natural solution—inconsistencies flagged by reviewers in real papers constitute expert-level annotations that are organically produced and unpredictable.

**Core Idea**: Reviewer criticisms are the best test cases for multimodal reasoning.

## Method

### Overall Architecture: Six-Stage Construction Pipeline

PRISMM-Bench is constructed through six stages: (1) *Review Acquisition*—18,009 ICLR 2024/2025 reviews are scraped from OpenReview, restricted to rejected or withdrawn papers without rebuttals to ensure inconsistencies remain uncorrected; (2) *LLM Filtering*—Mistral Nemo at low temperature filters down to 6,056 candidate inconsistency mentions; (3) *Human Annotation*—a custom web annotation tool is used to verify each entry, labeling inconsistency type, modalities involved, and location metadata, yielding 384 inconsistencies across 353 papers and 15 categories; (4) *LMM Task Generation*—Gemini 2.5 Flash automatically generates four-way multiple-choice questions; (5) *Human Verification*—automatically generated errors are corrected; (6) *LLM Debiasing*—natural-language answers are converted to JSON format to eliminate linguistic shortcuts.

### Key Design 1: Three-Task Progressive Evaluation Framework

Three multiple-choice tasks of increasing difficulty (4 options each) are combined with three levels of context granularity, forming seven evaluation configurations:

- **Inconsistency Identification (Ident, 384 items)**: Given paper context, answer "What inconsistency exists across these sections?"→detection ability.
- **Inconsistency Remediation (Remedy, 384 items)**: Answer "What action is needed to fix the inconsistency?"→requires deeper reasoning.
- **Paired Matching (Match, 192 items)**: Given a visual element, identify which of four candidates conflicts with it→pure visual cross-modal reasoning.

Three context granularity levels: Focused (key excerpts only) → Page (full page rendered at 144 DPI) → Document (entire paper concatenated into five images), with increasing difficulty.

**Design Motivation**: The three tasks progress from *detection* to *remediation* to *relational reasoning*, while the three context levels move from noise-free to highly distracting, jointly covering the full capability spectrum of scientific document understanding.

### Key Design 2: JSON-Structured Debiasing Answer Representation

To counter models exploiting choice-only shortcuts, answer options are converted from natural language to structured JSON:

- Ident task: **Evidence–Claim** JSON format (evidence + assertion).
- Remedy task: **Target–Action** JSON format (target element + corrective action).

The core mechanism is to remove stylistic cues (length variation, phrasing habits, positional patterns) while retaining only semantic content. The visual dependency ratio $R$ quantifies the effect:

$$R = \frac{Acc_{\text{with\_context}} - Acc_{\text{without\_context}}}{1 - Acc_{\text{without\_context}}}$$

A higher $R$ indicates greater reliance on visual evidence. Human $R = 69.0\%$, while the best model achieves only $R = 53.5\%$, indicating that humans rely more genuinely on visual reasoning than current models.

## Key Experimental Results

### Main Results: 21 LMMs Benchmarked (Accuracy %)

| Model | Params | Ident-Focused | Remedy-Focused | Match | Ident-Page | Ident-Doc | Avg. |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemma 3 4B | 4B | 27.9 | 29.9 | 39.6 | 25.0 | 26.6 | 27.8 |
| InternVL3.5 8B (R) | 8B | 49.5 | 35.9 | 45.8 | 38.3 | 36.7 | 37.7 |
| Ovis2 34B | 34B | 50.0 | 41.1 | 37.0 | 40.6 | 33.3 | 38.7 |
| GLM 4.5V 106B (R) | 106B | 51.8 | 43.2 | 52.1 | 45.8 | 40.9 | 42.6 |
| GPT-5 minimal (R) | — | 53.6 | 43.5 | 63.0 | 47.1 | 40.9 | 44.0 |
| Gemini 2.5 Pro (R) | — | 65.9 | 61.2 | 66.7 | 54.7 | 39.8 | 52.8 |
| **GPT-5 high (R)** | — | **63.8** | **54.4** | **70.3** | **58.1** | **46.9** | **53.9** |

### Ablation Study: Effect of Disabling CoT Reasoning (Ident-Focused)

| Model | Reasoning On | Reasoning Off | Drop |
|------|:---:|:---:|:---:|
| GLM 4.5V 106B | 51.8% | 43.2% | −16.6% |
| InternVL3.5 8B | 49.5% | 40.6% | −18.0% |
| InternVL3.5 38B | 54.4% | 40.4% | **−25.7%** |

### JSON Debiasing Effect (User Study Subset)

| Model | NL w/o Context | JSON w/o Context | Visual Dep. R (NL) | Visual Dep. R (JSON) |
|------|:---:|:---:|:---:|:---:|
| InternVL3.5 38B | 53.7% | **25.3%** | 22.5 | 38.1 |
| Gemini 2.5 Pro | 70.1% | **37.3%** | 43.8 | 45.2 |
| Human | **27.5%** | — | **69.0** | — |

### Key Findings

- Even the strongest model, GPT-5 (high), achieves only 53.9%, falling far short of what would be required for a reliable scientific assistant.
- Performance consistently degrades from Focused → Page → Document, indicating that long-document distraction is a critical bottleneck.
- Remedy scores are systematically lower than Ident scores, confirming that "fixing" requires deeper reasoning than "detecting."
- Enabling CoT reasoning improves performance by 5–14 percentage points on average, highlighting the importance of structured reasoning for scientific document understanding.
- 17% of ICLR 2025 submissions contain at least one reviewer-flagged inconsistency, demonstrating that cross-modal inconsistency is a pervasive problem.
- High-resolution specialist models (VILA HD 4K, InternLM XC 2.5) show no advantage under extended context.

## Highlights & Insights

- **"Reviewer Criticism as Test Case" Data Philosophy**: Rather than artificially injecting errors, the benchmark exploits problems naturally identified by experts during peer review, maximizing ecological validity and proximity to real-world application scenarios.
- **Elegant Simplicity of JSON Debiasing**: The idea of style homogenization—borrowed from NLP security—is transferred to multimodal evaluation, using uniform structured representations to eliminate answer-style variation and addressing a systemic problem in MCQ-based benchmarking.
- **Sustainable Live Benchmark**: The pipeline can be applied to new conference review data, continuously generating fresh samples and fundamentally avoiding data contamination.
- **Scale vs. Architecture**: Gemma 3 12B achieves 63.5% on the Match task, surpassing many 70B+ models, suggesting that architectural design matters more than raw parameter count.

## Limitations & Future Work

- Coverage is limited to the AI domain (ICLR 2024/2025); inconsistencies in chemistry, biology, physics, and other fields may exhibit different characteristics.
- Samples are biased toward rejected papers; persistent inconsistencies in accepted papers remain unevaluated.
- The 384-sample scale is limited, providing insufficient statistical power for fine-grained per-category analyses.
- The benchmark evaluates identification of inconsistencies at known locations, without assessing the ability to proactively search for them across an entire paper.

## Related Work & Insights

- **vs. MMIR (Yan et al., 2025)**: MMIR uses synthetically injected inconsistencies, enabling easier scaling but at the cost of realism; PRISMM-Bench uses genuine reviewer annotations, which are harder to collect but offer higher ecological validity. The two approaches are complementary.
- **vs. QASA / SciDQA**: The former is text-only QA; the latter has a similar data source but lacks visual elements. PRISMM-Bench is unique in combining both *authentic sourcing* and *multimodal evaluation*.
- **Insights**: Future work could extend the pipeline to arXiv preprints and reviews from additional venues to construct a large-scale cross-domain version. Integration with automated review tools (e.g., AI reviewers) could establish a closed-loop system for proactive inconsistency detection.

## Rating

⭐⭐⭐⭐⭐ (5/5)

Overall assessment: The first benchmark grounded in authentic reviewer-annotated inconsistencies, combined with JSON debiasing, evaluated across 21 models × three tasks × three context levels in an exceptionally thorough experimental design. The pipeline supports sustainable expansion, establishing infrastructure-level contributions to the evaluation of scientific AI assistants and setting a new standard for multimodal benchmarking.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](../../CVPR2026/multimodal_vlm/enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[ICLR 2026\] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification](lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve.md)

<!-- RELATED:END -->
