---
title: >-
  [Paper Note] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity
description: >-
  [ACL 2026][Multimodal VLM][Chinese art] This paper constructs CArtBench, a multi-task benchmark based on collections from the Palace Museum…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Chinese art"
  - "museum benchmark"
  - "vision-language models"
  - "appreciation capability"
  - "authenticity discrimination"
date: 2026-05-08
content_hash: 8bb39cfe5f240c60
---

# CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity

**Conference**: ACL 2026  
**arXiv**: [2604.11632](https://arxiv.org/abs/2604.11632)  
**Code**: [https://github.com/Big-Sid/CARTBENCH-Chinese-Artwork-Benchmark](https://github.com/Big-Sid/CARTBENCH-Chinese-Artwork-Benchmark)  
**Area**: Multimodal VLM/Cultural Understanding  
**Keywords**: Chinese art, museum benchmark, vision-language models, appreciation capability, authenticity discrimination

## TL;DR

This paper constructs CArtBench, a multi-task benchmark based on collections from the Palace Museum, to evaluate four capabilities of VLMs in Chinese art understanding (evidentiary QA, structured appreciation, defensible re-interpretation, and authenticity discrimination). It finds that even the strongest models exhibit significant performance degradation in evidence association and style-period reasoning, while authenticity discrimination remains near random levels.

## Background & Motivation

**Background**: VLMs are increasingly utilized as general-purpose multimodal assistants, yet their evaluation is largely dominated by web images and Western-centric concepts. Although Chinese and culture-focused benchmarks have expanded, they primarily focus on short-text recognition and basic QA.

**Limitations of Prior Work**: (1) Existing benchmarks lack assessments of expert-oriented explanation capabilities—deep understanding that requires cultural anchoring and explicit support from visual evidence; (2) Many visual conventions in Chinese art are era-sensitive, and curation-level understanding requires linking observable clues to historical contexts; (3) Authenticity judgment is a core workflow of cultural heritage, yet current VLM capabilities in this area have never been evaluated.

**Key Challenge**: VLMs may perform well on short-text QA, but high accuracy might mask severe deficiencies in deep capabilities such as evidence association, structured appreciation, and authenticity verification.

**Goal**: Construct a unified benchmark to comprehensively evaluate the curation-level capabilities of VLMs in Chinese art understanding.

**Key Insight**: Aligning Wikidata entities of the Palace Museum collections with authoritative catalog pages to build a museum benchmark spanning multiple dynasties and five major art categories.

**Core Idea**: Expanding from short-text QA to four progressive task levels (evidence-anchored QA, structured appreciation, defensible interpretation, and authenticity discrimination) to reveal systematic failure modes of VLMs in cultural understanding.

## Method

### Overall Architecture

CArtBench is constructed through a three-stage pipeline: (1) Retrieving image collections of the Palace Museum from Wikidata; (2) Aligning collections with official catalog descriptions; (3) Expert-guided filtering and classification. Four complementary tasks are instantiated based on the constructed data.

### Key Designs

1.  **CuratorQA (Curation-level QA)**:
    *   **Function**: Evaluates the evidence-anchored identification and reasoning capabilities of VLMs.
    *   **Mechanism**: Includes 14,421 questions covering 1,589 artworks, categorized into P1 (visual evidence only) and P2 (integration with art knowledge required). Six question types include subject identification, scene classification, composition format, technique style, iconographic detection, and style-period reasoning. QA pairs were generated using GPT-5.2, with an expert-verified error rate of only 0.47% across 1,000 samples.
    *   **Design Motivation**: The stratification of P1/P2 difficulty and the classification of six question types allow the evaluation to precisely pinpoint the capability bottlenecks of the models.

2.  **CatalogCaption (Structured Appreciation)**:
    *   **Function**: Evaluates the ability of VLMs to generate four-paragraph expert-level appreciation texts.
    *   **Mechanism**: For 86 artworks, models are required to generate structured appreciation texts containing basic information, technical analysis, historical background, and aesthetic evaluation, which are compared against authoritative catalog descriptions.
    *   **Design Motivation**: Long-text generation is a more challenging task than QA, requiring the model to synthesize visual understanding with cultural knowledge.

3.  **ConnoisseurPairs (Authenticity Discrimination)**:
    *   **Function**: Evaluates the ability of VLMs to discriminate between visually similar authentic-fake pairs.
    *   **Mechanism**: 10 pairs of visually similar authentic artworks and imitations are used, requiring the model to judge which is authentic based on global consistency and subtle clues. This serves as a diagnostic stress test.
    *   **Design Motivation**: Authenticity discrimination is a core skill of connoisseurs, testing whether VLMs can move beyond superficial recognition to deep reasoning.

### Loss & Training

No model training is involved. Evaluation employs a unified protocol combining automatic metrics, format compliance checks, and expert scoring.

## Key Experimental Results

### Main Results

**Overall Accuracy on CuratorQA (9 VLMs)**

| Model | Overall Accuracy | QA6 (Style-Period Reasoning) |
| :--- | :--- | :--- |
| Qwen3-VL-235B | 0.84 | 0.56 |
| Qwen3-VL-30B | 0.80 | 0.42 |
| Qwen2.5-VL-72B | 0.81 | 0.53 |
| Qwen2.5-VL-32B | 0.80 | 0.53 |

### Ablation Study

*   High overall accuracy masks significant performance drops in evidence association (QA5) and style-period reasoning (QA6).
*   Long-text appreciation (CatalogCaption) remains far below the level of expert references.
*   Authenticity discrimination (ConnoisseurPairs) is near random levels for all models, highlighting the extreme difficulty of connoisseur-level reasoning.

### Key Findings

*   High scores in short-text recognition by VLMs may mask severe deficiencies in evidence association and cultural reasoning.
*   Style-period reasoning is the most difficult sub-task, with the strongest model achieving only 56%.
*   The near-random performance in authenticity discrimination indicates that current VLMs lack connoisseur-level visual reasoning capabilities.
*   Significant performance differences exist across different art categories.

## Highlights & Insights

*   The first museum-grade Chinese art VLM benchmark, spanning four levels: recognition, appreciation, interpretation, and authenticity verification.
*   Alignment with authoritative Palace Museum catalogs ensures data authenticity and authority.
*   The unique design of the authenticity discrimination task directly targets the blind spots of deep reasoning in VLMs.
*   The evaluation protocol is rigorously designed, combining automatic metrics and expert scoring.

## Limitations & Future Work

*   The scales of ReInterpret and ConnoisseurPairs are relatively small (25/10), serving primarily as diagnostic evaluations.
*   Data is primarily sourced from the Palace Museum, which may introduce collection bias.
*   The cost of expert annotation for authenticity discrimination is extremely high, making large-scale expansion difficult.
*   Future work could expand to involve more museums and diverse artistic traditions.

## Related Work & Insights

*   Complements cultural-aware benchmarks like CVLUE and CulturalVQA, but probes deeper into expert-level evaluation.
*   Forms task complementarity with ArtEmis (emotion) and MuseumQA (fact-based).
*   Provides more rigorous evaluation standards for AI applications in the field of cultural heritage.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ The first museum-grade Chinese art VLM benchmark covering authenticity.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 9 VLMs across four tasks.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure with well-justified task design motivations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)
- [\[ACL 2026\] Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations](beyond_screenshots_evaluating_vlms_understanding_of_ui_animations.md)

</div>

<!-- RELATED:END -->
