---
title: >-
  [Paper Note] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life
description: >-
  [ACL 2026][Multimodal VLM][Multimodal safety] This paper introduces SaLAD, a benchmark comprising 2,013 real image-text samples spanning 10 daily-life categories…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal safety"
  - "benchmark"
  - "daily life safety"
  - "safety warning evaluation"
  - "MLLM alignment"
date: 2026-05-08
content_hash: 0b8a3fc8e564f345
---

# When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life

**Conference**: ACL 2026
**arXiv**: [2601.04043](https://arxiv.org/abs/2601.04043)
**Code**: [GitHub](https://github.com/xinyuelou/SaLAD)
**Area**: Multimodal VLM
**Keywords**: Multimodal safety, benchmark, daily life safety, safety warning evaluation, MLLM alignment

## TL;DR
This paper introduces SaLAD, a benchmark comprising 2,013 real image-text samples spanning 10 daily-life categories, designed to evaluate the ability of multimodal large language models to identify implicit safety risks and provide safety warnings during everyday assistance. Results reveal that even the strongest model achieves only 57.2% accuracy on unsafe queries.

## Background & Motivation

**Background** Multimodal large language models (MLLMs) have become indispensable assistants in human daily life, helping users solve problems and providing guidance. However, when users treat MLLM responses as behavioral references, inaccurate or biased content may lead to unsafe decisions.

**Limitations of Prior Work** Existing multimodal safety benchmarks primarily focus on explicit malicious behaviors such as jailbreak attacks. In real-world scenarios, however, users typically do not intentionally elicit unsafe content; instead, they are misled while seeking routine information in domains such as medicine, transportation, or nutrition. Existing implicit safety benchmarks (e.g., SIUO, MSSBench) suffer from limited sample sizes, unrealistic scenarios, and incomplete category coverage.

**Key Challenge** Traditional safety evaluation uses "refusal to answer" as the criterion, but in daily assistance scenarios, simple refusal fails to protect users—models must identify hidden risks and provide constructive safety warnings. Existing safety alignment methods perform well on conventional benchmarks but are nearly ineffective in the implicit safety scenarios exemplified by SaLAD.

**Goal** To construct a multimodal safety benchmark that closely reflects real daily life, comprehensively evaluating MLLMs' ability to identify implicit safety risks when assisting humans in everyday activities.

**Key Insight** Queries are sourced from real user interactions and paired with authentic (non-AI-generated) images, ensuring that safety risks cannot be inferred from text alone (requiring cross-modal reasoning). A "safety warning" evaluation framework is introduced to replace the simple binary "refuse/pass" paradigm.

**Core Idea** Safety evaluation should shift from "whether the model refuses to answer" to "whether the model identifies risks and provides meaningful safety warnings," which better reflects the practical needs of daily assistance scenarios.

## Method

### Overall Architecture
SaLAD is constructed via a three-stage pipeline: (1) mining potentially unsafe behaviors from 100K real user queries; (2) retrieving matched real images for each text query to form image-text pairs; and (3) annotating safety warnings for unsafe samples and normal advice for safe samples. Evaluation is performed using a GPT-4o-based automatic judge grounded in safety warnings.

### Key Designs

1. **Data Construction — Implicit Risk Mining**:
    - Function: Identify potentially unsafe behaviors in daily scenarios from real user interactions.
    - Mechanism: A pool of 100K real user queries is collected and clustered into 10 categories (home, sports, study, office, etc.) using K-Means + Sentence-BERT. An LLM filters similar queries and generates potentially unsafe behaviors, which are then verified by human annotators against authoritative safety guidelines.
    - Design Motivation: To ensure scenario authenticity and avoid unrealistic extreme cases (e.g., "flying with an umbrella"), while maintaining broad coverage.

2. **No Visual Safety Information Leakage (No VSIL) Design**:
    - Function: Ensure that safety risks cannot be inferred from text alone and require cross-modal reasoning over both image and text.
    - Mechanism: Text queries do not disclose image-specific information; models must integrate both visual and textual modalities to respond correctly.
    - Design Motivation: In existing benchmarks, many safety risks can be inferred from text alone (the VSIL problem), reducing evaluation to a purely textual safety assessment rather than a multimodal one.

3. **Safety Warning Evaluation Framework**:
    - Function: Use safety warning quality rather than refusal rate as the evaluation criterion.
    - Mechanism: For unsafe samples, a correct response should explicitly identify the risk and provide an explanation; for safe samples, a correct response should answer normally rather than refusing due to over-sensitivity. GPT-4o serves as the judge, achieving 93.85% agreement with human evaluation.
    - Design Motivation: Simple refusal does not protect users in daily assistance contexts; models should, like experienced advisors, indicate "why it is dangerous" and "how to do it safely."

### Loss & Training
This work is a benchmark construction study with no training procedure. Data is cross-validated by 6 human annotators, with each sample verified by the remaining 5, ensuring two criteria are met: (1) the risk cannot be inferred from text alone; and (2) the safety warning is clear and coherent when image and text are combined.

## Key Experimental Results

### Main Results

| Model | Safe Accuracy | Unsafe Accuracy | Overall Accuracy |
|-------|--------------|----------------|-----------------|
| Claude3.7-Sonnet | 99.58 | **57.20** | **77.05** |
| Gemini2.5-Flash | 99.68 | 55.05 | 75.96 |
| GPT-4o | 99.79 | 53.83 | 75.36 |
| LLaVA-OneVision | 99.89 | 37.10 | 66.52 |
| Qwen2.5-VL-7B | 98.41 | 31.59 | 62.89 |
| Deepseek-VL2-Tiny | 89.08 | 10.93 | 47.54 |

### Ablation Study

| Configuration | Safe | Unsafe | Overall | Note |
|---------------|------|--------|---------|------|
| Qwen2.5-VL-7B Vanilla | 100.00 | 33.00 | 66.50 | Baseline |
| w/o image | 98.50 | 23.50 | 61.00 | Large drop in unsafe detection without image |
| w/ image caption | 100.00 | 27.50 | 63.75 | Caption cannot substitute original image |
| w/ Safety Prompt | 100.00 | 41.50 | 70.75 | Safety prompts yield limited improvement |
| + VLGuard | 94.50 | 43.50 | 69.00 | Alignment methods show limited effect |
| + SPA-VL | 100.00 | 35.00 | 67.50 | Only ~1% gain |

### Key Findings
- Even the strongest closed-source model (Claude3.7) achieves only 57.2% accuracy on unsafe queries; open-source models average only 30.65%.
- Removing images reduces unsafe detection by approximately 10%, validating the effectiveness of the benchmark's cross-modal design.
- Image captions cannot substitute for the original image, as implicit safety risks are embedded in fine-grained visual details.
- Existing safety alignment methods (VLGuard, MIS, SPA-VL) show limited effectiveness: VLGuard increases refusal rates on safe samples, while MIS avoids refusal but fails to identify risks.
- Multiple-choice testing reveals that models possess safety knowledge at 80%+ accuracy, yet cannot correctly apply it in multimodal scenarios.

## Highlights & Insights
- The "safety warning" evaluation framework is more practically meaningful than traditional refusal-rate metrics, driving a paradigm shift in safety evaluation from "can it refuse" to "can it protect the user."
- The No VSIL design ensures the benchmark genuinely tests cross-modal safety understanding.
- A "knowledge-application gap" is identified: models possess safety knowledge but cannot apply it in visual contexts.
- The balanced design of safe and unsafe subsets prevents models from gaming the benchmark through indiscriminate refusal.

## Limitations & Future Work
- The dataset contains 2,013 samples, which, while sufficient to reveal vulnerabilities, offers limited coverage.
- Evaluation relies on LLM-as-a-judge, which, despite high agreement with human evaluation, has inherent limitations.
- Only English-language scenarios are covered; safety standards may differ across cultural contexts.
- Future work should develop more fine-grained and generalizable multimodal safety alignment strategies.

## Related Work & Insights
- The findings contrast with VLSBench, which argues that "using image captions is safer than using images directly"; SaLAD demonstrates that in implicit safety scenarios, captions cannot replace the original image.
- This work serves as a cautionary signal for the safety alignment community: methods effective on conventional benchmarks nearly fail in implicit safety scenarios.
- SaLAD provides a more practically grounded evaluation platform for future multimodal safety alignment research.

## Rating
- Novelty: ⭐⭐⭐⭐ Implicit safety combined with safety warning evaluation represents a meaningful new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models, multiple safety alignment methods, and detailed modality analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated contributions.
- Value: ⭐⭐⭐⭐ Significant reference value for multimodal safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive](../../AAAI2026/multimodal_vlm/oida-qa_a_multimodal_benchmark_for_analyzing_the_opioid_industry_documents_archi.md)
- [\[ICLR 2026\] MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](../../ICLR2026/multimodal_vlm/mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning](when_slower_isn39t_truer_inverse_scaling_law_of_truthfulness_in_multimodal_reaso.md)

</div>

<!-- RELATED:END -->
