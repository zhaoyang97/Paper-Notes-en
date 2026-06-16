---
title: >-
  [Paper Note] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life
description: >-
  [ACL 2026][LLM Safety][Paper Note] The authors propose the SaLAD benchmark, comprising 2013 real-world image-text samples across 10 daily scenarios. It evaluates the ability of Multimodal Large Language Models (MLLMs) to identify implicit safety risks and provide safety warnings during daily assistance, revealing that even the strongest models achieve o
tags:
  - ACL 2026
  - LLM Safety
date: 2026-05-08
content_hash: c7859d6cbd40cc5f
---
# When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.04043](https://arxiv.org/abs/2601.04043)  
**Code**: [GitHub](https://github.com/xinyuelou/SaLAD)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Safety, Benchmarking, Daily Life Safety, Safety Warning Evaluation, MLLM Alignment

## TL;DR
The authors propose the SaLAD benchmark, comprising 2013 real-world image-text samples across 10 daily scenarios. It evaluates the ability of Multimodal Large Language Models (MLLMs) to identify implicit safety risks and provide safety warnings during daily assistance, revealing that even the strongest models achieve only 57.2% accuracy on unsafe queries.

## Background & Motivation

**Background** Multimodal Large Language Models (MLLMs) have become indispensable assistants in human life, helping users solve problems and providing guidance. However, when users rely on MLLM responses for behavioral reference, inaccurate or biased content can lead to unsafe decision-making.

**Limitations of Prior Work** Existing multimodal safety benchmarks primarily focus on explicit malicious behaviors (e.g., jailbreak attacks). In real scenarios, users typically do not intentionally guide models to generate unsafe content but are instead misled while seeking routine information in fields such as medicine, transportation, and nutrition. Existing implicit safety benchmarks (e.g., SIUO, MSSBench) suffer from issues such as small sample sizes, unrealistic scenarios, and incomplete categories.

**Key Challenge** Traditional safety assessments use "refusal to answer" as the gold standard. However, in daily assistance scenarios, a simple refusal fails to protect the user—the model needs to identify hidden risks and provide constructive safety warnings. Existing safety alignment methods are effective on traditional benchmarks but prove almost ineffective in implicit safety scenarios like SaLAD.

**Goal** To construct a multimodal safety benchmark that closely mirrors real daily life to comprehensively evaluate the ability of MLLMs to identify implicit safety risks while assisting human activities.

**Key Insight** The authors mine queries from real user interactions and pair them with real images (non-AI generated) to ensure that safety risks cannot be inferred from text alone (requiring cross-modal reasoning). Furthermore, a "safety warning" evaluation framework is introduced to replace the simple "refusal/pass" binary approach.

**Core Idea** Safety evaluation should shift from "whether the model refuses to answer" to "whether the model identifies risks and provides meaningful safety warnings," which is more aligned with the practical requirements of daily assistance scenarios.

## Method

### Overall Architecture
The construction of SaLAD follows a three-step pipeline: (1) mining potential unsafe behaviors from 100K real user queries; (2) retrieving matching real images for each text query to construct image-text pairs; (3) writing safety warnings for unsafe samples and normal suggestions for safe samples. Evaluation is conducted using a GPT-4o-based automated judge focused on safety warnings.

### Key Designs

**1. Implicit Risk Mining: Utilizing samples from real interactions rather than fabricated extreme cases**

Many safety benchmarks fail due to scenario distortion—artificially inserting extreme examples like "using an umbrella as a parachute," which no one would actually attempt. SaLAD instead starts with 100K real user queries, uses Sentence-BERT encoding followed by K-Means clustering to categorize them into 10 daily domains (e.g., home, sports, study, office). LLMs then filter semantic duplicates and extract potential unsafe behaviors, which are finally verified by human annotators against authoritative safety manuals. This ensures the risks are genuine "traps" ordinary people might encounter, guaranteeing both credibility and coverage.

**2. No Visual Safety Information Leakage (No VSIL): Forcing models to genuinely process images rather than guessing risks from text**

A hidden defect in existing benchmarks is VSIL: risk clues are embedded in the text, allowing models to judge whether to warn based solely on textual input, reducing "multimodal safety" to text-only safety. SaLAD avoids disclosing any image-specific information in the text queries, keeping risk signals strictly within fine-grained visual details. Models must integrate both visual and textual evidence to respond correctly. This constraint ensures the benchmark measures cross-modal safety understanding; the ~10% drop in unsafe detection when images are removed proves its effectiveness.

**3. Safety Warning Evaluation Framework: Shifting assessment from "refusal" to "user protection"**

In daily assistance, a simple refusal is often unhelpful—when users ask routine questions about medicine, transportation, or nutrition, a response like "I cannot answer" leaves them exposed to risk. Therefore, SaLAD evaluates the quality of safety warnings: for unsafe samples, a qualified response must explicitly identify the risk and explain "why it is dangerous and how to be safe." For safe samples, the model should answer normally without over-sensitive refusals. GPT-4o acts as the judge, achieving a 93.85% consistency rate with human evaluation. The balanced design of safe and unsafe sets also prevents models from "gaming" the benchmark through blanket refusals.

### Loss & Training
Ours is a benchmark construction effort and does not involve a training process. Data was cross-validated by 6 human annotators, where each sample was checked by 5 others to ensure it met two criteria: (1) the risk cannot be inferred from text alone; (2) the safety warning is clear and coherent after combining text and image.

## Key Experimental Results

### Main Results

| Model | Safe Set Accuracy | Unsafe Set Accuracy | Overall Accuracy |
|------|------------|--------------|----------|
| Claude3.7-Sonnet | 99.58 | **57.20** | **77.05** |
| Gemini2.5-Flash | 99.68 | 55.05 | 75.96 |
| GPT-4o | 99.79 | 53.83 | 75.36 |
| LLaVA-OneVision | 99.89 | 37.10 | 66.52 |
| Qwen2.5-VL-7B | 98.41 | 31.59 | 62.89 |
| Deepseek-VL2-Tiny | 89.08 | 10.93 | 47.54 |

### Ablation Study

| Configuration | Safe Set | Unsafe Set | Overall | Description |
|------|--------|---------|------|------|
| Qwen2.5-VL-7B Vanilla | 100.00 | 33.00 | 66.50 | Baseline |
| w/o image | 98.50 | 23.50 | 61.00 | Significant drop in unsafe detection without images |
| w/ image caption | 100.00 | 27.50 | 63.75 | Captions cannot replace original images |
| w/ Safety Prompt | 100.00 | 41.50 | 70.75 | Limited Gain from safety prompts |
| + VLGuard | 94.50 | 43.50 | 69.00 | Alignment methods show limited effectiveness |
| + SPA-VL | 100.00 | 35.00 | 67.50 | Only 1% Gain |

### Key Findings
- Even the strongest closed-source model (Claude 3.7) achieves only 57.2% accuracy on unsafe queries, while open-source models average only 30.65%.
- Removing images leads to a decrease in unsafe detection of approximately 10%, confirming the effectiveness of the cross-modal dataset design.
- Image captions cannot replace original images because implicit safety risks are hidden in fine-grained visual details.
- Existing safety alignment methods (VLGuard, MIS, SPA-VL) show limited effectiveness: VLGuard increases refusal rates on safe sets, while MIS fails to identify risks.
- Multiple-choice tests indicate models possess over 80% safety knowledge but cannot correctly apply it in multimodal scenarios.

## Highlights & Insights
- The "safety warning" evaluation framework is more practically significant than traditional refusal rate assessments, driving a paradigm shift from "can it refuse" to "can it protect the user."
- The No Visual Safety Information Leakage (No VSIL) design ensures the benchmark truly tests cross-modal safety understanding.
- The "knowledge-application gap" is identified: models possess safety knowledge but fail to apply it in visual contexts.
- The balanced design of safety and allergy subsets prevents models from "gaming" scores through blanket refusals.

## Limitations & Future Work
- The dataset scale of 2013 samples, while sufficient to reveal vulnerabilities, has limited coverage.
- Evaluation relies on LLM-as-a-judge; despite high human consistency, inherent limitations exist.
- Only English scenarios are covered; safety standards may vary across different cultural backgrounds.
- Future work should focus on developing finer-grained and more generalizable multimodal safety alignment strategies.

## Related Work & Insights
- Contrasts with the findings of VLSBench: while the latter suggests "using captions is safer than using images," SaLAD proves that captions cannot replace original images in implicit safety scenarios.
- Serves as a warning to the safety alignment field: methods effective on traditional benchmarks almost entirely fail in implicit safety scenarios.
- Provides a more realistic evaluation platform for future research in multimodal safety alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ Implicit safety + safety warning evaluation is a significant new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes 18 models, various alignment methods, and detailed modal analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-justified motivation.
- Value: ⭐⭐⭐⭐ High reference value for multimodal safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[CVPR 2026\] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](../../CVPR2026/llm_safety/demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)

</div>

<!-- RELATED:END -->
