---
title: >-
  [Paper Note] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life
description: >-
  [ACL 2026][LLM Safety][Paper Note] This paper proposes the SaLAD benchmark, comprising 2013 real-world image-text samples across 10 daily life categories. It evaluates the ability of Multimodal Large Language Models (MLLMs) to identify implicit safety risks and provide safety warnings during daily assistance, revealing that even the strongest models ach
tags:
  - ACL 2026
  - LLM Safety
date: 2026-05-08
content_hash: b49fad30f6a54907
---
# When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.04043](https://arxiv.org/abs/2601.04043)  
**Code**: [GitHub](https://github.com/xinyuelou/SaLAD)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Safety, Benchmarking, Daily Life Safety, Safety Warning Evaluation, MLLM Alignment

## TL;DR
This paper proposes the SaLAD benchmark, comprising 2013 real-world image-text samples across 10 daily life categories. It evaluates the ability of Multimodal Large Language Models (MLLMs) to identify implicit safety risks and provide safety warnings during daily assistance, revealing that even the strongest models achieve only 57.2% accuracy on unsafe queries.

## Background & Motivation

**Background** Multimodal Large Language Models (MLLMs) have become indispensable assistants in human life, helping users solve problems and providing guidance. However, when users rely on MLLM responses as behavioral references, inaccurate or biased content can lead to unsafe decisions.

**Limitations of Prior Work** Existing multimodal safety benchmarks primarily focus on explicit malicious behaviors (e.g., jailbreak attacks). In real-world scenarios, users usually do not intentionally guide models to generate unsafe content but are instead misled while seeking routine information in fields such as healthcare, transportation, and nutrition. Existing implicit safety benchmarks (e.g., SIUO, MSSBench) suffer from small sample sizes, unrealistic scenarios, and incomplete categories.

**Key Challenge** Traditional safety evaluation uses "refusal to answer" as the criterion. However, in daily assistance scenarios, a simple refusal fails to protect the user—models need to identify hidden risks and provide constructive safety warnings. Existing safety alignment methods are effective on traditional benchmarks but are nearly ineffective in implicit safety scenarios like SaLAD.

**Goal** To construct a multimodal safety benchmark close to real daily life to comprehensively evaluate the ability of MLLMs to identify implicit safety risks while assisting human daily activities.

**Key Insight** Mine queries from real user interactions paired with real images (non-AI generated) to ensure that safety risks cannot be inferred from text alone (requiring cross-modal reasoning), and introduce a "safety warning" evaluation framework to replace the simple "refusal/pass" dichotomy.

**Core Idea** Safety evaluation should shift from "whether the model refuses to answer" to "whether the model identifies risks and provides meaningful safety warnings," which better aligns with the actual needs of daily assistance scenarios.

## Method

### Overall Architecture
The construction of SaLAD follows a three-step pipeline: (1) Mining potential unsafe behaviors from 100K real user queries; (2) Retrieving matching real images for each text query to construct image-text pairs; (3) Writing safety warnings for unsafe samples and normal advice for safe samples. Evaluation is conducted using a GPT-4o-based automatic judge for safety warnings.

### Key Designs

**1. Implicit Risk Mining: Sourcing samples from real interactions rather than fabricated extreme cases**

Many safety benchmarks fail due to scenario distortion—inserting extreme examples like "using an umbrella as a parachute," which no one would actually do. In contrast, SaLAD starts from 100K real user queries, using Sentence-BERT for encoding followed by K-Means clustering into 10 daily categories like home, sports, study, and office. Subsequently, LLMs filter out semantically redundant queries and extract potential unsafe behaviors, which are then manually verified by annotators against authoritative safety manuals. The risks identified are "traps common people might actually fall into," ensuring credibility and broad coverage across 10 categories.

**2. No Visual Safety Information Leakage (No VSIL): Forcing the model to actually "see" the image rather than guessing risks from text**

A hidden flaw in existing benchmarks is VSIL: safety clues are included in the text, allowing models to judge whether to warn based solely on reading. SaLAD deliberately avoids disclosing any image-specific information in the text queries, keeping risk signals only within the fine-grained visual details. The model must integrate both visual and textual evidence to respond correctly. This constraint ensures the benchmark measures cross-modal safety understanding; the finding that "unsafe detection drops by approximately 10% without images" serves as evidence.

**3. Safety Warning Evaluation Framework: Changing assessment from "refusal" to "user protection"**

In daily assistance, simple refusals are unhelpful. When users ask routine questions about medicine or nutrition, a "cannot answer" response leaves them exposed to risk. SaLAD evaluates the quality of safety warnings: for unsafe samples, a qualified response must clearly point out the risk and explain "why it is dangerous and how to be safe"; for safe samples, the model should answer normally without over-sensitive refusal. Using GPT-4o as a judge achieved 93.85% consistency with human evaluation. The balanced design of safe and unsafe sets also prevents "gaming the score" via blanket refusals.

### Loss & Training
This work focuses on benchmark construction and involves no training process. Data was cross-validated by 6 manual annotators, with each sample checked by 5 others to ensure: (1) risks cannot be inferred from text alone; (2) safety warnings are clear and coherent after combining text and images.

## Key Experimental Results

### Main Results

| Model | Safety Set Accuracy | Unsafe Set Accuracy | Overall Accuracy |
|------|------------|--------------|----------|
| Claude3.7-Sonnet | 99.58 | **57.20** | **77.05** |
| Gemini2.5-Flash | 99.68 | 55.05 | 75.96 |
| GPT-4o | 99.79 | 53.83 | 75.36 |
| LLaVA-OneVision | 99.89 | 37.10 | 66.52 |
| Qwen2.5-VL-7B | 98.41 | 31.59 | 62.89 |
| Deepseek-VL2-Tiny | 89.08 | 10.93 | 47.54 |

### Ablation Study

| Config | Safety Set | Unsafe Set | Overall | Description |
|------|--------|---------|------|------|
| Qwen2.5-VL-7B Vanilla | 100.00 | 33.00 | 66.50 | Baseline |
| w/o image | 98.50 | 23.50 | 61.00 | Unsafe detection drops significantly without images |
| w/ image caption | 100.00 | 27.50 | 63.75 | Image captions cannot replace original images |
| w/ Safety Prompt | 100.00 | 41.50 | 70.75 | Limited improvement from safety prompts |
| + VLGuard | 94.50 | 43.50 | 69.00 | Limited effect of alignment methods |
| + SPA-VL | 100.00 | 35.00 | 67.50 | Only 1% improvement |

### Key Findings
- Even the strongest closed-source model (Claude 3.7) has an accuracy of only 57.2% on unsafe queries; open-source models average only 30.65%.
- Removing images leads to an approximately 10% drop in unsafe detection, confirming the effectiveness of the cross-modal dataset design.
- Image captions cannot replace original images because implicit safety risks are hidden in fine-grained visual details.
- Existing safety alignment methods (VLGuard, MIS, SPA-VL) have limited effectiveness: VLGuard increases the refusal rate on the safety set, while MIS fails to identify risks despite not refusing.
- Multiple-choice tests indicate that models possess over 80% of safety knowledge but fail to apply it correctly in multimodal scenarios.

## Highlights & Insights
- The "safety warning" evaluation framework is more practical than traditional refusal rate assessments, driving a paradigm shift from "can it refuse" to "can it protect the user."
- The No Visual Safety Information Leakage (No VSIL) design ensures the benchmark truly tests cross-modal safety understanding.
- Discovered a "knowledge-application gap": models possess safety knowledge but cannot apply it in visual contexts.
- The balanced design of safety and allergy subsets prevents models from "padding scores" through blanket refusals.

## Limitations & Future Work
- The dataset size is 2013 samples, which is sufficient to reveal vulnerabilities but limited in coverage.
- Evaluation relies on LLM-as-a-judge; while highly consistent with human evaluation, it has inherent limitations.
- Only covers English scenarios; safety standards may vary across different cultural backgrounds.
- Future work necessitates the development of more fine-grained and generalizable multimodal safety alignment strategies.

## Related Work & Insights
- Contrasts with findings from VLSBench: the latter suggested "using image captions is safer than using images," but SaLAD proves image captions cannot replace original images in implicit safety scenarios.
- A warning to the safety alignment field: methods effective on traditional benchmarks nearly fail in implicit safety scenarios.
- Provides a more realistic evaluation platform for future multimodal safety alignment research.

## Rating
- Novelty: ⭐⭐⭐⭐ Implicit safety + safety warning evaluation is a meaningful new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models, multiple safety alignment methods, and detailed modal analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-justified motivation.
- Value: ⭐⭐⭐⭐ Significant reference value for multimodal safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)

</div>

<!-- RELATED:END -->
