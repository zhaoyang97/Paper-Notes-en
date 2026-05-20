---
title: >-
  [Paper Note] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models
description: >-
  [ICCV 2025][Multimodal VLM][factuality evaluation] SimpleVQA is the first VQA benchmark designed for comprehensive multimodal factuality evaluation of MLLMs. It spans 9 task types and 9 thematic domains…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "factuality evaluation"
  - "multimodal benchmark"
  - "visual question answering"
  - "large language models"
  - "hallucination detection"
date: 2026-05-08
content_hash: b3e84e7b868a5739
---

# SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models

**Conference**: ICCV 2025
**arXiv**: [2502.13059](https://arxiv.org/abs/2502.13059)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: factuality evaluation, multimodal benchmark, visual question answering, large language models, hallucination detection

## TL;DR

SimpleVQA is the first VQA benchmark designed for comprehensive multimodal factuality evaluation of MLLMs. It spans 9 task types and 9 thematic domains, and employs a short-answer design with deterministic references alongside an LLM-as-a-judge scoring protocol to systematically assess the factual capabilities of 18 MLLMs and 8 text-only LLMs.

## Background & Motivation

**Background**: MLLMs are increasingly deployed across diverse domains—from medical diagnosis to autonomous driving—where output accuracy and reliability are critical. Existing benchmarks (VQAv2, TextVQA, MMBench, etc.) primarily assess perceptual understanding, leaving factuality (i.e., whether a model can generate correct answers grounded in real-world knowledge) largely unevaluated in a systematic manner.

**Limitations of Prior Work**: (1) Answers in existing VQA benchmarks often require subjective judgment or admit multiple valid responses, making objective factuality assessment difficult. (2) Many benchmark questions become invalid over time (e.g., "Who is the current president?"), undermining long-term stability. (3) Evaluation granularity is insufficient to identify in which knowledge domains or task types models are more prone to errors.

**Key Challenge**: Factuality evaluation demands answers that are definitive, objective, and time-invariant—constraints that existing visual question answering datasets were not designed to satisfy.

**Goal**: To construct a multi-dimensional, high-quality, and easily evaluable factuality benchmark for MLLMs that can precisely localize where models tend to produce factual errors across tasks and domains.

**Key Insight**: The benchmark centers on objective facts, formulating short-answer natural-language questions for which each question has a unique correct answer, is unaffected by temporal changes, and can be automatically scored with low variance using an LLM-based judge.

**Core Idea**: A matrix-structured benchmark is built from 9 task types covering objective events and common knowledge $\times$ 9 thematic domains. Rigorous quality control generates concise, unambiguous reference answers, and LLM-as-a-judge is adopted for low-variance automated scoring.

## Method

### Overall Architecture

The construction pipeline of SimpleVQA consists of: (1) defining 9 task types (e.g., entity recognition, attribute judgment, relational reasoning, counting, OCR, geolocalization) and 9 thematic domains (e.g., people, landmarks, art, nature, food, sports); (2) human annotators creating question–answer pairs from real images, ensuring answers are short, deterministic, and time-invariant; (3) multiple rounds of quality control including answer verification, ambiguity detection, and temporal validity checks; (4) automated evaluation via an LLM-as-a-judge scoring system.

### Key Designs

1. **Matrix-Structured Task–Domain Organization**:

    - Function: Systematically covers multiple dimensions of factuality evaluation.
    - Mechanism: Evaluation items are organized into a matrix of 9 task types $\times$ 9 thematic domains. Task types span from low-level perception (OCR, counting) to high-level reasoning (relational reasoning, commonsense judgment); thematic domains cover people, landmarks, artworks, and more. This matrix design enables precise identification of model weaknesses.
    - Design Motivation: Single-dimensional evaluation cannot reveal the patterns of factual errors. For example, a model may perform well on person identification but poorly on geographic knowledge, or excel at OCR while struggling with relational reasoning.

2. **Static Deterministic Answer Design**:

    - Function: Ensures objectivity and long-term validity of evaluation results.
    - Mechanism: All reference answers must satisfy three conditions—brevity (typically 1–3 words), determinism (a unique correct answer), and time-invariance (excluding questions such as "Who is the current president?"). Annotators must confirm that each answer is grounded in objective fact or widely accepted common knowledge.
    - Design Motivation: Long or open-ended answers introduce evaluation ambiguity, and temporally sensitive questions cause benchmark degradation. This design ensures stable, long-term usability.

3. **LLM-as-a-Judge Scoring System**:

    - Function: Achieves low-variance, scalable automated evaluation.
    - Mechanism: Both the model-generated answer and the reference answer are provided to a strong LLM (e.g., GPT-4), which judges whether the generated answer is correct. Because reference answers are short and unambiguous, the LLM judge only needs to assess semantic equivalence rather than subjective quality, resulting in very low scoring variance.
    - Design Motivation: Human evaluation is costly and non-scalable, while traditional exact matching is overly strict with respect to paraphrases. LLM-as-a-judge achieves near-human agreement in short-answer deterministic settings.

### Loss & Training

SimpleVQA is an evaluation benchmark rather than a training method; no loss function is involved. The core contribution lies in dataset construction and evaluation protocol design.

## Key Experimental Results

### Main Results

Comprehensive evaluation results across 18 MLLMs and 8 text-only LLMs:

| Model | Overall Accuracy | Image Understanding | Knowledge Reasoning | Rank |
|-------|-----------------|--------------------|--------------------|------|
| GPT-4V | Top tier | Strong | Strong | Top-3 |
| Gemini Pro Vision | High | Strong | Moderate | Top-5 |
| LLaVA-v1.6 | Moderate | Moderate | Weak | Mid |
| Qwen-VL | Moderate | Moderate | Moderate | Mid |
| Small open-source models | Lower | Weak | Weak | Lower |

### Ablation Study: Task Type and Domain Analysis

| Task Type | Avg. Accuracy | Difficulty | Notes |
|-----------|--------------|------------|-------|
| Entity Recognition | High | Low | Most fundamental perceptual task |
| OCR / Text Reading | High | Low | Well-supported by current MLLMs |
| Counting | Moderate | Moderate | Complex scenes remain challenging |
| Relational Reasoning | Low | High | Requires combining vision and knowledge |
| Commonsense Judgment | Low | High | Heavily dependent on world knowledge |

### Key Findings

- Closed-source commercial models (GPT-4V, Gemini) substantially outperform open-source models in factuality, with the gap stemming primarily from knowledge reasoning tasks rather than perceptual tasks.
- All models exhibit the greatest variance on "landmark recognition" and "person identification," indicating uneven knowledge coverage in these domains.
- Text-only LLMs given image captions sometimes achieve higher factuality accuracy than certain MLLMs in end-to-end settings, suggesting that visual encoders may introduce noise.
- LLM-as-a-judge agreement with human evaluation exceeds 95%, validating the reliability of this scoring protocol.
- Model scale and factuality are not strictly correlated; architecture and training data quality have greater influence.

## Highlights & Insights

- **"Simple but Effective" Design Philosophy**: Rather than pursuing complex evaluation protocols, SimpleVQA reduces evaluation noise by strictly constraining answer format (short, deterministic, non-outdated). The elegance of this design lies in front-loading complexity into the data construction stage rather than the evaluation stage.
- **Cross-Dimensional Error Analysis**: The matrix structure enables cross-analysis of "task type × domain," allowing developers to precisely localize model weaknesses. This systematic evaluation framework is transferable to other modalities.
- **Complementarity with MLLM Hallucination Research**: SimpleVQA targets factuality in the sense of "not asserting what one does not know," whereas hallucination research addresses "perceiving what is not present"—the two perspectives are complementary.

## Limitations & Future Work

- Only English-language questions are covered; multilingual factuality is not assessed.
- The short-answer design, while favorable for evaluation, excludes complex reasoning tasks that require extended explanations.
- Although the $9 \times 9$ matrix is systematic, per-cell sample sizes may be insufficient for statistically significant conclusions.
- The time-invariant answer design excludes real-world scenarios that require temporal awareness (e.g., "In what year did this news event occur?").
- Future directions include: multilingual extensions, video factuality evaluation, and integration with retrieval-augmented generation.

## Related Work & Insights

- **vs. MMMU / MMBench**: These benchmarks focus on comprehensive capability assessment, while SimpleVQA concentrates specifically on factuality; the two are complementary.
- **vs. POPE**: POPE evaluates object-level hallucination; SimpleVQA evaluates knowledge-level factuality—the two address distinct phenomena.
- **vs. TruthfulQA**: TruthfulQA is a text-only factuality benchmark; SimpleVQA is the first multimodal factuality benchmark.
- The benchmark reveals that visual encoders may introduce factuality noise, a phenomenon worth attention in VLM architectural design.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic multimodal factuality benchmark, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 26 models across rich analytical dimensions.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline is clearly described; evaluation protocol is rigorous.
- Value: ⭐⭐⭐⭐ Provides practical guidance for MLLM factuality research and model selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)

</div>

<!-- RELATED:END -->
