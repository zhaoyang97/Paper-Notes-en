---
title: >-
  [Paper Note] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning
description: >-
  [ACL 2026][Multimodal VLM][Ancient Chinese Character Evolution] This paper constructs an ancient Chinese character evolution analysis benchmark containing 11 tasks and over 130,000 instances. After evaluating 19 MLLMs…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Ancient Chinese Character Evolution"
  - "MLLM"
  - "Glyph-driven Fine-tuning"
  - "Oracle Bone Inscriptions"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: 147b11db0da5c2a7
---

# Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning

**Conference**: ACL 2026  
**arXiv**: [2604.11299](https://arxiv.org/abs/2604.11299)  
**Code**: [https://github.com/songruiecho/GEVO](https://github.com/songruiecho/GEVO)  
**Area**: Multimodal VLM / Digital Humanities  
**Keywords**: Ancient Chinese Character Evolution, MLLM, Glyph-driven Fine-tuning, Oracle Bone Inscriptions, Curriculum Learning

## TL;DR
This paper constructs an ancient Chinese character evolution analysis benchmark containing 11 tasks and over 130,000 instances. After evaluating 19 MLLMs, it was discovered that current models possess limited capabilities in glyph-level recognition and evolutionary reasoning. To address this, the glyph-driven contrastive fine-tuning framework (GEVO) is proposed, achieving comprehensive task improvements on a 2B model.

## Background & Motivation

**Background**: With the rapid development of MLLMs, increasing research has begun to leverage them for analyzing ancient characters (e.g., Oracle bone and Bronze inscriptions), showing potential in tasks ranging from character recognition to cultural relic interpretation. The evolutionary analysis of ancient characters (from Oracle bone to Regular script) is a fundamental path for understanding cultural transformation and historical inheritance.

**Limitations of Prior Work**: (1) There is a lack of systematic benchmarks to evaluate the capabilities of MLLMs in ancient character evolution analysis; (2) existing MLLMs perform poorly in cross-era font style identification and ancient character recognition; (3) while some studies explore ancient scripts, systematically enhancing MLLM capabilities in evolution analysis remains an open problem.

**Key Challenge**: Ancient character evolution involves subtle glyph variations and structural changes across eras. Existing MLLMs are primarily trained on modern data and lack an understanding of ancient glyph features. However, the observation that minor fine-tuning can significantly improve era attribution indicates that MLLMs possess latent potential but require targeted guidance.

**Goal**: (1) Construct a comprehensive benchmark for ancient Chinese character evolution analysis; (2) systematically evaluate the capability boundaries of existing MLLMs; (3) propose an effective fine-tuning method to enhance evolution analysis capabilities.

**Key Insight**: It was observed that MLLMs can significantly improve era attribution after minimal fine-tuning. This inspired the design of a glyph-driven contrastive fine-tuning method—enabling the model to distinguish subtle differences in glyph changes caused by era transitions versus character identity.

**Core Idea**: Utilizing the philosophy of curriculum learning, positive and negative glyph pairs are constructed to guide the model in capturing glyph transformation patterns within evolutionary consistency through contrastive learning.

## Method

### Overall Architecture
The GEVO framework consists of two stages: (1) Benchmark construction—utilizing 7,740 characters and nearly 30,000 transcription images across five stages from Oracle bone to Regular script, involving 11 sub-tasks in 3 categories; (2) Glyph-driven fine-tuning—constructing glyph contrastive data to train the model to distinguish glyph differences from simple to complex through curriculum learning.

### Key Designs

1.  **Ancient Character Evolution Benchmark Construction**:
    *   **Function**: Provides 11 sub-tasks and over 130,000 test instances for systematic evaluation of MLLM capabilities in evolution analysis.
    *   **Mechanism**: Divides the evolutionary process into five stages: Oracle Bone, Bronze, Seal, Clerical, and Regular scripts. Three task categories: (T1) Basic Recognition—font style and era judgment; (T2) Glyph Understanding—image-level character recognition and structural analysis; (T3) Evolution Analysis—cross-era comparison and evolutionary path reasoning. All tasks are designed in a QA format with mixed image-text input and text output.
    *   **Design Motivation**: Existing ancient script benchmarks mostly focus on a single stage (Oracle bone) or single task. This benchmark covers the complete evolutionary chain with multi-dimensional evaluation to fully reveal MLLM capability boundaries.

2.  **Glyph-Driven Contrastive Fine-Tuning (GEVO)**:
    *   **Function**: Guides the model to capture evolutionary consistency and era-specific differences in glyph transformations through contrastive learning.
    *   **Mechanism**: Constructs positive and negative glyph pairs—positive pairs are variants of the same character across different eras (capturing evolutionary consistency), while negative pairs are glyphs of different characters (learning to distinguish differences). A curriculum learning strategy is employed, starting from simple pairs with large visual differences and gradually transitioning to difficult pairs with subtle differences. The training objective is a mixture of glyph recognition and contrastive judgment tasks.
    *   **Design Motivation**: Glyph changes in ancient character evolution are often subtle (e.g., stroke simplification, structural adjustments). Direct fine-tuning on recognition tasks might only capture surface features. Contrastive learning forces the model to attend to fine-grained differences, while curriculum learning prevents the model from being overwhelmed by difficult samples initially.

3.  **Multi-dimensional Evaluation Protocol**:
    *   **Function**: Systematically evaluates MLLM evolution analysis capabilities across different granularities and dimensions.
    *   **Mechanism**: 11 sub-tasks cover dimensions from single-image recognition to cross-era reasoning. Evaluation combines accuracy with domain-expert validation. 19 MLLMs (ranging from 1B to 72B, including closed-source models like GPT-4o-mini and GPT-5-mini) were evaluated.
    *   **Design Motivation**: Different tasks place varying demands on a model's visual understanding, knowledge reasoning, and cross-era association capabilities. Multi-dimensional evaluation enables precise localization of MLLM strengths and weaknesses.

### Loss & Training
Standard cross-entropy loss is combined with contrastive learning through positive and negative sample construction. Curriculum learning ranks training data based on the visual saliency of glyph differences. Fine-tuning is performed on a 2B model.

## Key Experimental Results

### Main Results (19 MLLM Evaluations)

| Model | Avg. Score | Font Recognition (T1) | Character Recognition (T2) | Evolution Analysis (T3) |
| :--- | :--- | :--- | :--- | :--- |
| GPT-5-mini | 24.88 | Low | Extremely Low (0.07) | Low |
| Gemini-3-Flash | 27.89 | Low | Extremely Low | Low |
| Qwen2.5-VL-7B | **47.65** | Medium | 23.51 | Medium |
| Qwen2.5-VL-72B | 46.30+ | Medium | 24.45 | Medium |
| **Ours** (GEVO-2B) | Overall Gain | Significant Gain | Significant Gain | Significant Gain |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| GEVO Full | Improvement across all 11 tasks | Contrastive + Curriculum Learning |
| w/o Curriculum | Diminished gains in some tasks | Simple-to-hard order is beneficial |
| w/o Contrastive | Limited improvement | Recognition training alone is insufficient |
| Recognition-only FT | Period gain but weak reasoning | Validates the necessity of contrastive learning |

### Key Findings
*   All existing MLLMs (including GPT-5-mini) perform poorly in ancient character evolution analysis, with average scores not exceeding 50.
*   Character-level recognition (T2.1) is the primary bottleneck for all models, with most scoring near 0%.
*   Unexpected discovery: Minor fine-tuning significantly improves era attribution, but reasoning tasks require the support of contrastive learning.
*   GEVO achieves consistent improvements across all 11 tasks on a 2B model.
*   Open-source 7B models (e.g., Qwen2.5-VL-7B) outperform closed-source LLMs, possibly because safety constraints in the latter affect performance on non-standard tasks.

## Highlights & Insights
*   **Cultural Value of the Benchmark**: An AI evaluation benchmark covering the full evolutionary chain from Oracle bone to Regular script is a significant contribution to digital humanities, potentially driving the development of computational palaeography.
*   **Capturing Invariance via Contrastive Learning**: Using variants of the same character across eras as positive pairs to learn evolutionary laws is an approach that can be generalized to any visual task requiring cross-temporal or cross-style understanding.
*   **Potential of Small Models**: A 2B model can improve across all tasks through targeted fine-tuning, suggesting that the injection of domain knowledge is more critical than model size for specific fields.

## Limitations & Future Work
*   The dataset only covers approximately 7,740 characters with recorded evolution; many characters lack complete evolutionary paths.
*   The absolute performance of 2B models remains limited and requires validation on larger models.
*   The benchmark is primarily based on transcription images (not actual artifact photos), which may differ from real-world recognition scenarios.
*   The use of evolutionary knowledge to assist in interpreting undeciphered characters has not yet been explored.

## Related Work & Insights
*   **vs TongGu-VL**: A VLM designed for ancient scripts, but limited to a 2B scale and weak in evolution analysis. GEVO is more effective through its fine-tuning strategy.
*   **vs Traditional Ancient Script OCR**: Focused CNN models lack reasoning and association capabilities. MLLMs possess this potential but require guidance.
*   **vs General VLM Fine-Tuning**: Standard SFT improves recognition but is insufficient for evolutionary reasoning. Contrastive learning provides additional structural learning signals.

## Rating
*   Novelty: ⭐⭐⭐⭐ First systematic MLLM benchmark for ancient character evolution; novel glyph-contrastive fine-tuning approach.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 19 models, 11 sub-tasks, and sufficient ablation.
*   Writing Quality: ⭐⭐⭐⭐ Clear benchmark construction process and in-depth analysis of evaluation results.
*   Value: ⭐⭐⭐⭐ Unique contribution to digital humanities and ancient character research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DRIFT: Transferring Reasoning Priors for Efficient MLLM Fine-Tuning](drift_transferring_reasoning_priors_for_efficient_mllm_fine-tuning.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[ACL 2026\] Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry](thinking_like_a_botanist_challenging_multimodal_language_models_with_intent-driv.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)

</div>

<!-- RELATED:END -->
