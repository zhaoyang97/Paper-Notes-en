---
title: >-
  [Paper Note] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly
description: >-
  [NeurIPS 2025][Multimodal VLM][long-context VLM] This paper introduces MMLongBench, the first comprehensive benchmark for evaluating long-context vision-language models (LCVLMs), comprising 13…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "long-context VLM"
  - "benchmark"
  - "multi-task evaluation"
  - "cross-modal tokenization"
  - "NIAH"
date: 2026-05-08
content_hash: 3cffad7ee42a0061
---

# MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.10610](https://arxiv.org/abs/2505.10610)  
**Code**: [GitHub](https://github.com/EdinburghNLP/MMLongBench)  
**Area**: Multimodal VLM / Long-Context Evaluation
**Keywords**: long-context VLM, benchmark, multi-task evaluation, cross-modal tokenization, NIAH

## TL;DR
This paper introduces MMLongBench, the first comprehensive benchmark for evaluating long-context vision-language models (LCVLMs), comprising 13,331 samples spanning 5 downstream task categories, mixed image types, and 5 standardized input length levels (8K–128K tokens). Evaluation of 46 models reveals that single-task performance is a weak proxy for overall capability, and that stronger reasoning ability positively correlates with long-context performance.

## Background & Motivation
**Background**: The context windows of VLMs have been extended to 128K+ tokens (e.g., GPT-4o, Gemini-2.5), giving rise to long-context VLMs (LCVLMs) capable of processing hundreds of images and thousands of interleaved text tokens. However, evaluation benchmarks have lagged significantly behind.

**Limitations of Prior Work**:
- **Limited task coverage**: Existing benchmarks focus on a single task type (e.g., MM-NIAH targets only needle-in-a-haystack retrieval; MMLongBench-Doc covers only document VQA), and no single task can reflect overall long-context capability.
- **Narrow image type coverage**: Most benchmarks include only natural photographs or synthetic document screenshots, lacking comprehensiveness.
- **Inconsistent input length definitions**: Different benchmarks define "length" differently—some by image count, others by token count—and most provide only a single length level.
- **Absence of important tasks**: Practically relevant scenarios such as Visual RAG, many-shot ICL, and summarization are entirely absent from prior benchmarks.

**Key Challenge**: Model developers need to identify performance strengths and weaknesses across specific length levels and task types, yet existing benchmarks do not support such fine-grained analysis.

**Goal**: To construct a unified evaluation benchmark covering multiple task types, image modalities, and length levels.

**Key Insight**: A unified cross-modal token counting scheme (vision patches + text tokens), combined with 5 standardized length levels and 5 downstream task categories.

**Core Idea**: Provide the missing evaluation infrastructure for LCVLMs through unified length control and diverse task coverage.

## Method

### Overall Architecture
MMLongBench encompasses 5 task categories × 5 length levels × mixed image types:
- **Visual RAG**: Retrieve information from long-context Wikipedia passages to answer visual questions.
- **NIAH**: Locate a "needle" inserted into a sequence of "haystack" images.
- **Many-Shot ICL**: Perform image classification based on hundreds of in-context examples.
- **Summarization**: Summarize PDF documents.
- **DocVQA**: Conduct visual question answering over long documents.

### Key Designs

1. **Cross-Modal Tokenization**:
    - **Function**: Unify the counting of vision patches and text tokens.
    - **Mechanism**: The image token count equals the number of patches produced by the visual encoder (after $2\times2$ pixel unshuffle), which are summed with text tokens to form the total sequence length.
    - **Design Motivation**: Aligns with implementations of recent models such as Qwen2.5-VL and InternVL3, ensuring length metrics are comparable across models.

2. **5 Standardized Length Levels (8K/16K/32K/64K/128K)**:
    - **Function**: Provide each sample with context versions at five distinct lengths.
    - **Mechanism**: Total token counts are precisely controlled by padding or truncating context materials.
    - **Design Motivation**: Enables systematic analysis of performance trends as a function of context length, following established practices in text-domain long-context evaluation.

3. **Diverse Image Type Coverage**:
    - **Function**: Include both natural images (photographs, scenes) and synthetic images (document screenshots, webpages, application screenshots).
    - **Mechanism**: Different tasks naturally introduce different image types—NIAH and ICL use natural images; DocVQA and Summarization use synthetic images; VRAG incorporates both.
    - **Design Motivation**: Prevents evaluation blind spots arising from image-type bias.

4. **Comprehensive Model Evaluation (46 Models)**:
    - **Function**: Evaluate both closed-source (GPT-4o, Gemini, etc.) and open-source (LLaVA, Qwen-VL, InternVL, etc.) models.
    - **Mechanism**: A unified evaluation protocol with controlled variables for fair comparison.
    - **Design Motivation**: Provide a panoramic view of current LCVLM capabilities.

## Key Experimental Results

### Key Findings

| Finding | Details |
|---------|---------|
| Single-task → Overall | **Weak proxy**: Strong performance on NIAH **does not imply** strong performance on VRAG or ICL. |
| Closed-source vs. Open-source | Closed-source models lead overall, but both categories exhibit substantial degradation at 128K. |
| Reasoning vs. Long-Context | **Positive correlation**: Gemini thinking variants substantially outperform their standard counterparts. |
| OCR Bottleneck | OCR capability and cross-modal retrieval are the primary bottlenecks for current LCVLMs. |
| Length Sensitivity | Most models exhibit significant performance degradation beginning at 32K tokens. |

### Task-Level Results (Representative Models)

| Model | VRAG | NIAH | ICL | Summ | DocVQA | Overall |
|-------|------|------|-----|------|--------|---------|
| GPT-4o | High | High | Mid | Mid | High | **Top tier** |
| Gemini-2.5-Flash | Mid | High | High | High | Mid | High tier |
| Qwen2.5-VL-72B | Mid | Mid | Mid | Mid | Mid | Mid tier |
| InternVL3-8B | Low | Mid | Low | Low | Low | Low tier |

### Length Sensitivity

| Length | Average Performance (Normalized) |
|--------|----------------------------------|
| 8K | 1.00 (baseline) |
| 16K | ~0.95 |
| 32K | ~0.85 |
| 64K | ~0.70 |
| 128K | ~0.55 |

### Key Numbers
- 13,331 evaluation samples
- 5 downstream task categories
- 46 evaluated models
- 5 standardized length levels (8K–128K)
- Mixed image types (natural + synthetic)

## Highlights & Insights
- **"Single-task performance is a weak proxy for overall capability"**: This finding highlights the insufficiency of using NIAH alone to evaluate long-context ability, motivating the necessity of multi-task evaluation.
- **Reasoning capability ≈ Long-context capability**: The advantage of Gemini thinking variants suggests that explicit reasoning ability facilitates processing of long contexts—an insight with implications for LCVLM design.
- **Unified length control**: This work is the first in the vision-language domain to enforce length control standards as rigorous as those in the text domain.
- **Extensibility**: All datasets are designed to be readily extended to longer contexts (256K+).

## Limitations & Future Work
- **No intra-task difficulty stratification**: Easy and hard subsets are not distinguished within individual tasks.
- **English-only evaluation**: Multilingual long-context evaluation is absent.
- **Video excluded**: Long-video understanding is an important long-context scenario not covered in this benchmark.
- **No human baseline**: Human performance on the same tasks is not provided as an upper-bound reference.

## Related Work & Insights
- **vs. MM-NIAH**: Covers only the NIAH task; MMLongBench covers 5 task categories.
- **vs. MMLongBench-Doc**: Focuses solely on document VQA without length control; this work provides comprehensive task coverage with rigorous length control.
- **vs. MileBench**: Claims comprehensiveness but has an average context length of only ~9K tokens, making it unsuitable as a genuine long-context benchmark.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First comprehensive multi-task long-context evaluation benchmark for LCVLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 46 models × 5 tasks × 5 length levels, with detailed error analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated; comparative tables with prior benchmarks are highly persuasive.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a critical gap in LCVLM evaluation and is poised to become a standard benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[NeurIPS 2025\] Context Informs Pragmatic Interpretation in Vision-Language Models](context_informs_pragmatic_interpretation_in_vision-language_models.md)
- [\[NeurIPS 2025\] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)

</div>

<!-- RELATED:END -->
