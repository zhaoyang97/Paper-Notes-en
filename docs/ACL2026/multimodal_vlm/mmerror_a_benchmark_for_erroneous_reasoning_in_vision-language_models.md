---
title: >-
  [Paper Note] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][erroneous reasoning diagnosis] This paper presents MMErroR, a multimodal erroneous reasoning benchmark comprising 1,997 samples, each containing exactly one deliberately injected reasoning error, spanning 6 domains and 4 error types. The benchmark requires VLMs to not only detect the presence of errors in reasoning chains but also classify the error type (Visual Perception Error / Knowledge Deficiency Error / Question Comprehension Error / Reasoning Error). Evaluation of 12 representative VLMs reveals that even the strongest model, Gemini-3-Pro-Preview, achieves only 66.65% accuracy.
tags:
  - ACL 2026
  - Multimodal VLM
  - erroneous reasoning diagnosis
  - vision-language model evaluation
  - process-level assessment
  - error taxonomy
  - multimodal reasoning
date: 2026-05-08
content_hash: 8ab457930f4d941b
---

# MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2601.03331](https://arxiv.org/abs/2601.03331)
**Code**: [https://mmerror-benchmark.github.io](https://mmerror-benchmark.github.io)
**Area**: Multimodal VLM
**Keywords**: erroneous reasoning diagnosis, vision-language model evaluation, process-level assessment, error taxonomy, multimodal reasoning

## TL;DR

This paper presents MMErroR, a multimodal erroneous reasoning benchmark comprising 1,997 samples, each containing exactly one deliberately injected reasoning error, spanning 6 domains and 4 error types. The benchmark requires VLMs to not only detect the presence of errors in reasoning chains but also classify the error type (Visual Perception Error / Knowledge Deficiency Error / Question Comprehension Error / Reasoning Error). Evaluation of 12 representative VLMs reveals that even the strongest model, Gemini-3-Pro-Preview, achieves only 66.65% accuracy.

## Background & Motivation

**Background**: Vision-language models (VLMs) continue to set new records on benchmarks such as MMMU and MathVista, creating the impression that models are approaching human-level understanding. However, existing evaluations almost exclusively adopt an answer-oriented paradigm—verifying only whether the final answer is correct while ignoring the soundness of the reasoning process that leads to it.

**Limitations of Prior Work**: (1) A correct final answer does not imply a correct reasoning process—models may arrive at the right result through shortcuts or pattern matching, masking deficiencies in genuine reasoning ability. (2) Existing error-localization benchmarks (e.g., ProcessBench, ErrorRadar) focus solely on identifying *which step* is erroneous, without diagnosing the type or root cause of the error. (3) These benchmarks are either limited to a single modality (pure text) or lack domain diversity and a systematic error taxonomy.

**Key Challenge**: A significant gap exists between VLMs' high scores on standard benchmarks and their ability to diagnose errors when presented with faulty reasoning chains. Models can produce apparently coherent reasoning chains yet fail to identify errors within them, indicating that *generative ability* and *introspective ability* are fundamentally distinct capabilities.

**Goal**: To construct a multimodal, multi-domain, process-level reasoning evaluation benchmark with error-type classification, enabling systematic assessment of whether VLMs can identify erroneous reasoning and diagnose its underlying cause.

**Key Insight**: The paper approaches the problem from the perspective of *error classification* rather than *error localization*—the task is not merely to detect which step is wrong, but to diagnose whether the root cause is a failure of visual perception, knowledge application, question comprehension, or logical reasoning.

**Core Idea**: A controlled benchmark is designed in which each sample contains exactly one error with a clearly attributable root cause. Errors are injected via GPT-5, followed by three rounds of human verification and quality scoring, ensuring uniqueness and traceability of error-type labels. Two evaluation modes are supported: Error Type Classification (ETC) and Error Presence Detection (EPD).

## Method

### Overall Architecture

The construction pipeline of MMErroR consists of four stages: (1) *Question curation*—stratified sampling from existing benchmarks (MMMU, MathVista, MathVerse, ScienceQA, AI2D) with complexity filtering to retain multi-step reasoning instances; (2) *Error injection*—GPT-5 is used to inject one semantically coherent error into a correct reasoning chain, constrained to one of four predefined error types; (3) *Data validation*—20 domain experts (6 professors and 14 doctoral students) conduct three rounds of manual inspection, narrowing an initial pool of 10,000 samples to 3,148; (4) *Quality assurance*—at least two linguistics experts score each sample along four dimensions (coherence, step clarity, error localizability, and semantic consistency), retaining only samples with a mean score above 0.5, yielding the final 1,997 samples.

### Key Designs

1. **Four-Category Error Taxonomy**

    - **Function**: Provides a fine-grained root-cause diagnostic framework.
    - **Mechanism**: Four mutually exclusive error types are defined—Visual Perception Error (VPE: e.g., object misidentification, spatial relation misjudgment, symbol misreading), Knowledge Deficiency Error (KDE: e.g., applying incorrect formulas or physical laws; the largest category at 44.07%), Question Comprehension Error (QCE: e.g., misinterpreting the question's intent or overlooking key constraints), and Reasoning Error (RE: e.g., logical fallacies, missing premises, invalid inference steps). Each reasoning chain contains exactly one error, ensuring unambiguous attribution.
    - **Design Motivation**: Different error types reflect weaknesses at distinct stages of a model's multimodal understanding pipeline; merely localizing an erroneous step cannot reveal the fundamental cause of failure.

2. **Dual Evaluation Protocol (ETC + EPD)**

    - **Function**: Assesses models' error-diagnosis capability at two levels of difficulty.
    - **Mechanism**: In the ETC (Error Type Classification) mode, the model is informed that the reasoning chain contains an error and is asked to classify its type. In the EPD (Error Presence Detection) mode, the model must first determine whether an error exists before classifying it. Since all samples in the current dataset contain errors, EPD serves as a controlled stress test of error sensitivity and attribution ability. A naive strategy of always reporting an error does not yield a score, because credit is awarded only when the error type is also classified correctly.
    - **Design Motivation**: ETC measures diagnostic precision under the known-error condition; EPD further tests the model's ability to proactively detect errors. Together, the two modes provide a comprehensive assessment of introspective capability.

3. **Single-Error Controlled Design with Multi-Stage Quality Filtering**

    - **Function**: Ensures the benchmark's attributability and high overall quality.
    - **Mechanism**: Each reasoning chain contains exactly one root-cause error, while all remaining steps maintain local coherence and logical validity. Three rounds of expert review (retention requires unanimous agreement) are combined with four-dimensional quality scoring (coherence / step clarity / error localizability / semantic consistency, threshold 0.5). Cohen's Kappa $\kappa = 0.796$; the third-round rejection rate is only 2.81% (inter-annotator agreement 97.19%).
    - **Design Motivation**: Interactions among multiple errors would severely confound attribution. Although the single-error design sacrifices some ecological validity, it yields unambiguous diagnostic labels and interpretable evaluation results.

### Loss & Training

This paper presents an evaluation benchmark and does not involve model training. Evaluation is conducted in a multiple-choice format; all models are decoded at temperature 0 to ensure determinism and reproducibility.

## Key Experimental Results

### Main Results

| Model | ML | PE | CM | BH | EE | DA | Overall (ETC) |
|------|-----|-----|-----|-----|-----|-----|---------|
| Gemini-3-Pro-Preview | 66.37 | 66.88 | 69.81 | 64.43 | 65.39 | 69.26 | 66.65 |
| Doubao-Seed-2.0-pro | 65.47 | 67.32 | 61.01 | 59.94 | 66.16 | 66.22 | 64.80 |
| GPT-5.2 (xhigh) | 64.56 | 63.62 | 62.26 | 60.50 | 65.14 | 69.59 | 64.30 |
| Claude-Opus-4.5 | 62.76 | 61.00 | 61.64 | 57.70 | 56.74 | 68.58 | 61.04 |
| Kimi-K2.5 | 63.66 | 55.56 | 51.57 | 58.82 | 66.67 | 61.15 | 60.19 |
| Qwen3-VL-32B-Thinking | 59.46 | 54.90 | 52.20 | 65.83 | 60.81 | 59.80 | 59.29 |
| Human Expert (High) | 91.07 | 88.65 | 87.50 | 90.15 | 88.96 | 90.18 | 89.52 |
| Random Choice | 22.10 | 23.62 | 24.18 | 24.06 | 21.50 | 25.53 | 23.45 |

| Model | ETC Overall | EPD Overall | EPD Drop |
|------|---------|---------|------------|
| Gemini-3-Pro-Preview | 66.65 | 61.39 | −5.26 |
| GPT-5.2 (xhigh) | 64.30 | 58.54 | −5.76 |
| Claude-Opus-4.5 | 61.04 | 55.18 | −5.86 |
| Kimi-K2.5 | 60.19 | 51.63 | −8.56 |
| LLaMA-4-Maverick | 39.46 | 18.13 | −21.33 |

### Ablation Study

| Input Condition | Gemini-3-Pro | GPT-5.2 | Doubao-Seed | Qwen3-VL-32B |
|---------|-------------|---------|------------|-------------|
| VQA (original Q&A) | 81.0 | 80.0 | 80.5 | 78.5 |
| VQA + erroneous chain | 82.5 | 80.5 | 81.5 | 80.0 |
| VQA + erroneous chain + error step | 84.0 | 82.0 | 83.0 | 82.5 |
| VQA + erroneous chain + error type | 90.5 | 89.5 | 88.5 | 84.5 |

### Key Findings

- **Large gap between all models and human experts**: The strongest VLM (66.65%) trails the high-group human experts (89.52%) by nearly 23 percentage points, indicating that erroneous reasoning diagnosis is a major weakness of current VLMs.
- **EPD is substantially harder than ETC**: All models exhibit significant score drops from ETC to EPD; LLaMA-4-Maverick collapses from 39.46% to 18.13%, demonstrating that *proactively detecting errors* is far more difficult than *classifying errors once informed of their existence*.
- **Error-diagnosis ability correlates with question-answering ability**: Samples on which models correctly diagnose the error type also yield higher raw VQA accuracy (Gemini: 85.5% vs. 74.5%), suggesting that error-diagnosis capability reflects genuine depth of understanding.
- **Error-type information is more valuable than error-step information**: Providing the error type improves VQA accuracy by approximately 9.5 points, whereas providing only the erroneous step yields only a 2–3-point improvement, demonstrating that knowing *why* something is wrong is more correctively valuable than knowing *where* it is wrong.
- **No single model dominates across all domains**: Different models excel in different domains, indicating that error diagnosis draws on diverse underlying capabilities including domain knowledge, visual grounding, and procedural reasoning.

## Highlights & Insights

- **Paradigm shift from answer correctness to process diagnosis**: MMErroR is the first benchmark to advance multimodal reasoning evaluation from "is the answer correct?" to "can the model diagnose the error type in a reasoning process?", offering a fundamentally new perspective on VLMs' true reasoning capability.
- **Error type is more correctively valuable than error location**: Ablation results clearly show that knowing *why* something is wrong (error type) is more effective for error correction than knowing *where* it is wrong (error step), with important implications for the design of future VLM self-correction mechanisms.
- **Logit lens visualization analysis**: Logit lens analysis on Qwen3-VL-32B-Instruct intuitively demonstrates precise semantic alignment between visual and textual tokens during correct diagnosis, and the collapse of cross-modal alignment during incorrect diagnosis.
- **Rigorous quality control**: The pipeline reduces an initial pool of 10,000 samples through three rounds of expert review to 3,148, and further through quality scoring to 1,997, yielding a retention rate of approximately 20% and a Cohen's Kappa of 0.796, ensuring high benchmark reliability.

## Limitations & Future Work

- Each sample contains only one error, whereas real-world reasoning failures often involve cascading or multiple co-occurring errors.
- The current version contains only erroneous reasoning chains; the EPD task therefore cannot measure false-positive rates on correct reasoning chains (i.e., the tendency to over-report errors).
- Initial erroneous reasoning chains are generated by GPT-5, which may introduce biases specific to that model's error patterns or linguistic style.
- Future work may extend the benchmark to open-ended generation evaluation (beyond the multiple-choice format) and to multi-error cascading scenarios.

## Related Work & Insights

- **vs. ProcessBench / PRISM-Bench**: These benchmarks localize erroneous steps but do not classify error types; MMErroR requires models to diagnose the root cause of errors.
- **vs. ErrorRadar**: ErrorRadar focuses on error localization but lacks multi-domain coverage and a systematic error taxonomy.
- **vs. POPE / HallusionBench**: These hallucination benchmarks primarily target visual perception errors; MMErroR additionally covers higher-order failure modes such as knowledge application, question comprehension, and logical reasoning.
- **vs. MMMU / MathVista**: These benchmarks employ outcome-oriented evaluation; MMErroR shifts to process-level diagnostic evaluation, providing a complementary assessment perspective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic evaluation of VLMs' erroneous reasoning diagnosis capability; the error taxonomy is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 models, 6 domains, dual evaluation modes, supplemented by reasoning consistency analysis, multimodal alignment analysis, and error-perception ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rigorous experimental design, and transparent quality control procedures.
- **Value**: ⭐⭐⭐⭐ — Provides an important benchmark and insights for understanding and improving VLMs' introspective capability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)
- [\[ACL 2026\] GeoRC: A Benchmark for Geolocation Reasoning Chains](georc_a_benchmark_for_geolocation_reasoning_chains.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)

</div>

<!-- RELATED:END -->
