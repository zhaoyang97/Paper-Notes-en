---
title: >-
  [Paper Note] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Erroneous reasoning diagnosis] This paper introduces MMErroR, a multimodal erroneous reasoning benchmark containing 1…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Erroneous reasoning diagnosis"
  - "Vision-Language Model evaluation"
  - "Process-level evaluation"
  - "Error classification"
  - "Multimodal reasoning"
date: 2026-05-08
content_hash: 003353edba862717
---

# MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.03331](https://arxiv.org/abs/2601.03331)  
**Code**: [https://mmerror-benchmark.github.io](https://mmerror-benchmark.github.io)  
**Area**: Multimodal VLM  
**Keywords**: Erroneous reasoning diagnosis, Vision-Language Model evaluation, Process-level evaluation, Error classification, Multimodal reasoning

## TL;DR

This paper introduces MMErroR, a multimodal erroneous reasoning benchmark containing 1,997 samples. Each sample embeds a single reasoning error across 6 major domains and 4 error types. It requires VLMs to not only detect the presence of errors in reasoning chains but also classify the error types (Visual Perception Error / Knowledge Application Error / Question Understanding Error / Reasoning Error). Evaluation of 12 representative VLMs reveals that the strongest model, Gemini-3-Pro-Preview, achieves only 66.65% accuracy.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) continuously break records on benchmarks such as MMMU and MathVista, creating the impression that "models have approached human-level understanding." However, existing evaluations almost entirely adopt a "result-oriented" paradigm—checking only whether the final answer is correct without considering if the reasoning process is sound.

**Limitations of Prior Work**: (1) Correct final answers do not guarantee correct reasoning processes; models may reach correct results through shortcuts or pattern matching, masking deficiencies in intrinsic reasoning capabilities. (2) Existing error localization benchmarks (e.g., ProcessBench, ErrorRadar) only focus on "which step went wrong" without diagnosing the type or root cause of the error. (3) These benchmarks are either limited to a single modality (text-only) or lack domain diversity and an error classification taxonomy.

**Key Challenge**: There is a significant gap between high VLM scores on various benchmarks and their diagnostic capabilities when faced with erroneous reasoning chains. Models can generate plausible reasoning chains but fail to judge errors within them, suggesting that "generation capability" and "introspection capability" are distinct skills.

**Goal**: To construct a multimodal, multi-domain, process-level reasoning evaluation benchmark with error classification to systematically assess whether VLMs possess the ability to "identify erroneous reasoning and diagnose error types."

**Key Insight**: Approaching the problem through "error classification" rather than just "error localization." The goal is not just to detect where a step fails, but to diagnose whether the root cause is visual perception failure, knowledge application failure, question understanding bias, or logical reasoning fallacy.

**Core Idea**: Design a controlled benchmark where each sample contains exactly one clear root-cause error. Errors are injected via GPT-5, followed by three rounds of human verification and quality score filtering to ensure uniqueness and attributability of error type labels. The benchmark supports two evaluation modes: Error Type Classification (ETC) and Error Presence Detection (EPD).

## Method

### Overall Architecture

The construction process of MMErroR follows four steps: (1) Question Curation—stratified sampling from benchmarks like MMMU, MathVista, MathVerse, ScienceQA, and AI2D, with complexity filtering to retain multi-step reasoning instances. (2) Error Injection—using GPT-5 to inject one semantically coherent error into a correct reasoning chain, restricted to one of four predefined types. (3) Data Verification—20 experts (6 professors + 14 PhD students) performed three rounds of manual inspection, filtering initial 10,000 samples down to 3,148. (4) Quality Assurance—at least two linguistics experts scored samples based on coherence, step clarity, error localizability, and semantic consistency, retaining 1,997 samples with a mean score $> 0.5$.

### Key Designs

1. **Four-class Error Taxonomy**:
    - Function: Provides a fine-grained diagnostic framework for root-cause errors.
    - Mechanism: Defines four mutually exclusive error types—Visual Perception Error (VPE, e.g., misidentification, spatial misjudgment, symbol misreading), Knowledge Application Error (KDE, e.g., using wrong formulas, misapplying physical laws, the largest category at 44.07%), Question Understanding Error (QCE, e.g., misunderstanding intent, ignoring constraints), and Reasoning Error (RE, e.g., logical fallacies, missing premises, invalid steps). Each reasoning chain contains only one error to ensure unique attribution.
    - Design Motivation: Different error types reflect weaknesses in different stages of the model's multimodal understanding pipeline; locating a step alone cannot reveal the nature of the failure.

2. **Dual-mode Evaluation Protocol (ETC + EPD)**:
    - Function: Evaluates error diagnostic capabilities across two difficulty levels.
    - Mechanism: The ETC (Error Type Classification) mode informs the model that the reasoning chain is definitely wrong and asks for the error type. The EPD (Error Presence Detection) mode requires the model to first determine "if there is an error" before classifying it. Since the current dataset only contains erroneous samples, EPD acts as a controlled stress test for error sensitivity and attribution. In EPD, an "always report error" strategy is ineffective because points are only awarded if the error type is also correctly classified.
    - Design Motivation: ETC tests diagnostic precision under known error conditions, while EPD further tests the model's ability to actively discover errors. Together, they provide a comprehensive assessment of introspection.

3. **Single-error Controlled Design + Multi-stage Quality Filtering**:
    - Function: Ensures attributability and high quality of the benchmark.
    - Mechanism: Only one root-cause error is injected into each reasoning chain; other steps remain locally coherent and logically valid. Three rounds of expert review (retention only upon unanimous approval) + four-dimension quality scoring (coherence/clarity/localizability/consistency, threshold 0.5). Cohen's Kappa $\kappa = 0.796$, with a third-round rejection rate of only 2.81% (observed agreement 97.19%).
    - Design Motivation: Multiple interacting errors would severely confound attribution. The single-error design sacrifices some realism for label clarity and evaluation interpretability.

### Loss & Training

This is a benchmark paper and does not involve model training. Evaluation is conducted using a multiple-choice format where the model outputs corresponding labels. All model decoding temperatures are set to 0 to ensure determinism and reproducibility.

## Key Experimental Results

### Main Results

| Model | ML | PE | CM | BH | EE | DA | Total (ETC) |
|------|-----|-----|-----|-----|-----|-----|---------|
| Gemini-3-Pro-Preview | 66.37 | 66.88 | 69.81 | 64.43 | 65.39 | 69.26 | 66.65 |
| Doubao-Seed-2.0-pro | 65.47 | 67.32 | 61.01 | 59.94 | 66.16 | 66.22 | 64.80 |
| GPT-5.2 (xhigh) | 64.56 | 63.62 | 62.26 | 60.50 | 65.14 | 69.59 | 64.30 |
| Claude-Opus-4.5 | 62.76 | 61.00 | 61.64 | 57.70 | 56.74 | 68.58 | 61.04 |
| Kimi-K2.5 | 63.66 | 55.56 | 51.57 | 58.82 | 66.67 | 61.15 | 60.19 |
| Qwen3-VL-32B-Thinking | 59.46 | 54.90 | 52.20 | 65.83 | 60.81 | 59.80 | 59.29 |
| Human Expert (High) | 91.07 | 88.65 | 87.50 | 90.15 | 88.96 | 90.18 | 89.52 |
| Random Choice | 22.10 | 23.62 | 24.18 | 24.06 | 21.50 | 25.53 | 23.45 |

| Model | ETC Total | EPD Total | EPD Drop |
|------|---------|---------|------------|
| Gemini-3-Pro-Preview | 66.65 | 61.39 | -5.26 |
| GPT-5.2 (xhigh) | 64.30 | 58.54 | -5.76 |
| Claude-Opus-4.5 | 61.04 | 55.18 | -5.86 |
| Kimi-K2.5 | 60.19 | 51.63 | -8.56 |
| LLaMA-4-Maverick | 39.46 | 18.13 | -21.33 |

### Ablation Study

| Input Condition | Gemini-3-Pro | GPT-5.2 | Doubao-Seed | Qwen3-VL-32B |
|---------|-------------|---------|------------|-------------|
| VQA (Original QA) | 81.0 | 80.0 | 80.5 | 78.5 |
| VQA + Erroneous Chain | 82.5 | 80.5 | 81.5 | 80.0 |
| VQA + Erroneous Chain + Error Step | 84.0 | 82.0 | 83.0 | 82.5 |
| VQA + Erroneous Chain + Error Type | 90.5 | 89.5 | 88.5 | 84.5 |

### Key Findings

- **Significant gap between all models and human experts**: The strongest VLM (66.65%) lags nearly 23 percentage points behind high-group human experts (89.52%), indicating that erroneous reasoning diagnosis is a major weakness for VLMs.
- **EPD is much harder than ETC**: All models showed significant performance drops from ETC to EPD; LLaMA-4-Maverick plummeted from 39.46% to 18.13%, suggesting that "actively discovering errors" is far more difficult than "classifying errors when informed they exist."
- **Error diagnostic capability correlates positively with QA capability**: Samples where models correctly diagnosed error types also showed higher original VQA accuracy (Gemini: 85.5% vs 74.5%), indicating that error diagnosis reflects true depth of understanding.
- **Error type information is more useful than error step information**: Providing the error type increased VQA accuracy by ~9.5 points, whereas providing only the error step increased it by only ~2-3 points. This proves that "knowing why it is wrong" has more corrective value than "knowing where it is wrong."
- **No single model dominates all domains**: Different models show varied strengths across domains, highlighting that error diagnosis depends on diverse underlying abilities like domain knowledge, visual grounding, and procedural reasoning.

## Highlights & Insights

- **Paradigm shift from "Answer Correctness" to "Process Diagnosis"**: MMErroR is the first to push multimodal reasoning evaluation from "is the result correct" to "can the error type in the reasoning process be diagnosed," providing a fresh perspective on the true reasoning capabilities of VLMs.
- **Error types hold more corrective value than error locations**: Ablation experiments clearly demonstrate that knowing the "why" (error type) is more effective for correcting mistakes than knowing the "where" (error step), which has significant implications for future VLM self-correction mechanisms.
- **Logit Lens Visualization Analysis**: Through logit lens analysis of Qwen3-VL-32B-Instruct, the paper intuitively demonstrates precise semantic alignment between visual and text tokens during correct diagnosis, and the collapse of cross-modal alignment during incorrect diagnosis.
- **Extremely strict quality control**: The retention rate was approximately 20% (filtering 10,000 down to 1,997) through three rounds of expert review and quality scoring. A Cohen's Kappa of 0.796 ensures high reliability of the benchmark.

## Limitations & Future Work

- Each sample contains only one error, whereas real-world reasoning failures often involve cascading or multiple simultaneous errors.
- The current version only contains erroneous reasoning chains; the EPD task cannot test the false positive rate (i.e., "over-reporting errors") on correct reasoning chains.
- Initial erroneous reasoning chains were generated by GPT-5, which might introduce biases (error patterns or linguistic styles) specific to that model.
- Future work could extend to open-ended generation evaluation (rather than multiple-choice) and multi-error cascading scenarios.

## Related Work & Insights

- **vs ProcessBench/PRISM-Bench**: These benchmarks only locate error steps without classifying error types; MMErroR requires diagnosing the root cause.
- **vs ErrorRadar**: ErrorRadar focuses on error localization but lacks multi-domain coverage and an error classification taxonomy.
- **vs POPE/HallusionBench**: These hallucination benchmarks primarily target visual perception errors; MMErroR covers higher-order failures like knowledge application, question understanding, and logical reasoning.
- **vs MMMU/MathVista**: These benchmarks use result-oriented evaluation; MMErroR shifts toward process-level diagnostic evaluation, acting as a complement.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically evaluates VLM error reasoning diagnosis for the first time with a well-designed taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 models, 6 domains, dual evaluation modes, supplemented by reasoning consistency, multimodal alignment, and error perception ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous experimental design, and transparent quality control.
- Value: ⭐⭐⭐⭐ Provides a crucial benchmark and insights for understanding and improving VLM introspection capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ACL 2026\] GeoRC: A Benchmark for Geolocation Reasoning Chains](georc_a_benchmark_for_geolocation_reasoning_chains.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)

</div>

<!-- RELATED:END -->
