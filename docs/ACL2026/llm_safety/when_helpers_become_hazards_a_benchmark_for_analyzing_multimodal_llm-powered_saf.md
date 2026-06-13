---
title: >-
  [Paper Note] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life
description: >-
  [ACL 2026][LLM Safety][Multimodal Safety] Proposes the SaLAD benchmark, comprising 2013 real-world image-text samples across 10 daily life scenarios…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Multimodal Safety"
  - "Benchmark"
  - "Daily Life Safety"
  - "Safety Warning Evaluation"
  - "MLLM Alignment"
date: 2026-05-08
content_hash: 0b90839c17bde4c8
---

# When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life

**Conference**: ACL 2026  
**arXiv**: [2601.04043](https://arxiv.org/abs/2601.04043)  
**Code**: [GitHub](https://github.com/xinyuelou/SaLAD)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Safety, Benchmark, Daily Life Safety, Safety Warning Evaluation, MLLM Alignment

## TL;DR
Proposes the SaLAD benchmark, comprising 2013 real-world image-text samples across 10 daily life scenarios, to evaluate the ability of multimodal large language models to identify implicit safety risks and provide safety warnings in daily assistance; reveals that even the strongest models achieve only 57.2% accuracy on unsafe queries.

## Background & Motivation

**Background** Multimodal Large Language Models (MLLMs) have become indispensable assistants in human life, capable of helping users solve problems and providing guidance. However, when users reference MLLM responses for action, inaccurate or biased content may lead to unsafe decisions.

**Limitations of Prior Work** Existing multimodal safety benchmarks primarily focus on explicit malicious behaviors (e.g., jailbreak attacks). In real scenarios, users usually do not intentionally induce unsafe content but are misled while seeking routine information in fields like medicine, transportation, and nutrition. Existing implicit safety benchmarks (e.g., SIUO, MSSBench) suffer from small sample sizes, unrealistic scenarios, and incomplete categories.

**Key Challenge** Traditional safety evaluation uses "refusal to answer" as the standard. However, in daily assistance scenarios, simple refusal fails to protect users—models need to identify hidden risks and provide constructive safety warnings. Existing safety alignment methods are effective on traditional benchmarks but nearly ineffective in implicit safety scenarios like SaLAD.

**Goal** Construct a multimodal safety benchmark close to actual daily life to comprehensively evaluate the capability of MLLMs to identify implicit safety risks during daily human assistance.

**Key Insight** Mine queries from real user interactions and pair them with real images (non-AI generated) to ensure safety risks cannot be inferred solely from text (requiring cross-modal reasoning), and introduce a "safety warning" evaluation framework to replace the simple "refusal/pass" dichotomy.

**Core Idea** Safety evaluation should shift from "whether the model refuses to answer" to "whether the model identifies risks and provides meaningful safety warnings," which aligns better with the actual needs of daily assistance scenarios.

## Method

### Overall Architecture
The construction of SaLAD follows a three-step pipeline: (1) mining potential unsafe behaviors from 100K real user queries; (2) retrieving matching real images for each text query to construct image-text pairs; (3) writing safety warnings for unsafe samples and normal suggestions for safe samples. Evaluation utilizes an automated GPT-4o judge based on safety warnings.

### Key Designs

1. **Data Construction—Implicit Risk Mining**:
    - **Function**: Identify potential unsafe behaviors in daily scenarios from real user interactions.
    - **Mechanism**: Use 100K real user queries as a candidate pool, cluster them into 10 categories (Home, Sports, Learning, Office, etc.) using K-Means + Sentence-BERT, filter similar queries and generate potential unsafe behaviors using LLMs, and verify with human annotators based on authoritative safety manuals.
    - **Design Motivation**: Ensure scenarios are authentic and credible, avoiding unrealistic extreme cases (e.g., "flying with an umbrella"), while ensuring broad coverage.

2. **No Visual Safety Information Leakage (No VSIL) Design**:
    - **Function**: Ensure safety risks cannot be inferred from text alone and require cross-modal reasoning.
    - **Mechanism**: Do not disclose image-specific information in text queries; models must integrate both visual and textual modalities to respond correctly.
    - **Design Motivation**: Many safety risks in existing benchmarks can be inferred from text alone (VSIL problem), turning the evaluation into a measure of text safety rather than multimodal safety capability.

3. **Safety Warning Evaluation Framework**:
    - **Function**: Use safety warning quality rather than refusal rate as the evaluation standard.
    - **Mechanism**: For unsafe samples, correct responses should explicitly identify risks and provide explanations; for safe samples, correct responses should provide normal answers rather than over-sensitive refusals. Using GPT-4o as a judge achieved 93.85% consistency with human evaluation.
    - **Design Motivation**: Simple refusal cannot protect users in daily assistance; models should act like an experienced assistant, pointing out "why it is dangerous" and "how to do it safely."

### Loss & Training
This work focuses on benchmark construction; no training process is involved. Data was cross-validated by 6 human annotators, each sample checked by 5 others, ensuring: (1) risk cannot be inferred from text alone; (2) safety warnings are clear and coherent after combining image and text.

## Key Experimental Results

### Main Results

| Model | Safe Set Acc | Unsafe Set Acc | Overall Acc |
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
| w/o image | 98.50 | 23.50 | 61.00 | Unsafe detection drops significantly without images |
| w/ image caption | 100.00 | 27.50 | 63.75 | Image captions cannot replace original images |
| w/ Safety Prompt | 100.00 | 41.50 | 70.75 | Limited improvement with safety prompts |
| + VLGuard | 94.50 | 43.50 | 69.00 | Limited effect of alignment methods |
| + SPA-VL | 100.00 | 35.00 | 67.50 | Only 1% improvement |

### Key Findings
- Even the strongest closed-source model (Claude 3.7) has an accuracy of only 57.2% on unsafe queries, while open-source models average only 30.65%.
- Removing images reduces unsafe detection by about 10%, confirming the effectiveness of the cross-modal design.
- Image captions cannot replace original images because implicit safety risks are hidden in fine-grained visual details.
- Existing safety alignment methods (VLGuard, MIS, SPA-VL) show limited effectiveness: VLGuard increases the refusal rate of safe sets, and while MIS does not refuse, it fails to identify risks.
- Multiple-choice tests indicate that models possess 80%+ safety knowledge but cannot apply it correctly in multimodal scenarios.

## Highlights & Insights
- The "safety warning" evaluation framework is more practical than traditional refusal rate metrics, driving a paradigm shift in safety evaluation from "can it refuse" to "can it protect users."
- The No Visual Safety Information Leakage (No VSIL) design ensures the benchmark truly tests cross-modal safety understanding.
- Discovered the "knowledge-application gap": models have safety knowledge but cannot apply it in visual contexts.
- Balanced design of safe and unsafe subsets prevents models from "gaming" the score through blanket refusals.

## Limitations & Future Work
- The dataset size is 2013 samples; while sufficient to reveal vulnerabilities, the coverage is limited.
- Evaluation relies on LLM-as-a-judge, which, despite high human consistency, has inherent limitations.
- Only covers English scenarios; safety standards may vary across different cultural backgrounds.
- Future work needs to develop more fine-grained and generalizable multimodal safety alignment strategies.

## Related Work & Insights
- Contrasts with findings from VLSBench: the latter suggests "using image captions is safer than using images," but SaLAD proves image captions cannot replace original images in implicit safety scenarios.
- A warning to the safety alignment field: methods effective on traditional benchmarks nearly fail in implicit safety scenarios.
- Provides a more realistic evaluation platform for future multimodal safety alignment research.

## Rating
- Novelty: ⭐⭐⭐⭐ Implicit safety + safety warning evaluation is a meaningful new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models, multiple safety alignment methods, detailed modal analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated.
- Value: ⭐⭐⭐⭐ Significant reference value for multimodal safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)

</div>

<!-- RELATED:END -->
