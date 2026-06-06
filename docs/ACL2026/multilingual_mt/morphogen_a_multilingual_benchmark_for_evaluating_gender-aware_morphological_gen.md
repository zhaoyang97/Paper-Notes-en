---
title: >-
  [Paper Note] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Gender-aware morphological generation] This paper proposes MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Gender-aware morphological generation"
  - "multilingual benchmark"
  - "French Arabic Hindi"
  - "LLM morphological reasoning"
  - "gender bias evaluation"
date: 2026-05-08
content_hash: 35e081c9e56c8ee1
---

# MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation

**Conference**: ACL 2026  
**arXiv**: [2604.18914](https://arxiv.org/abs/2604.18914)  
**Code**: [GitHub](https://github.com/) (Code + Dataset links provided)  
**Area**: Multilingual Translation  
**Keywords**: Gender-aware morphological generation, multilingual benchmark, French Arabic Hindi, LLM morphological reasoning, gender bias evaluation

## TL;DR

This paper proposes MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic, and Hindi (comprising 20,328 sentence pairs). It defines the GENFORM task (rewriting first-person sentences into the opposite gender) and introduces three evaluation metrics: SGA, GIoU, and CGA. Benchmarking results on 15 multilingual LLMs reveal systematic deficiencies in complex morphological reasoning, gender bias, and multi-entity interference.

## Background & Motivation

**Background**: Multilingual LLMs perform well in high-level tasks such as translation and question answering, but their capabilities regarding grammatical gender and morphological consistency have not been systematically evaluated. Existing multilingual benchmarks (e.g., XTREME, Global-MMLU) focus on semantic and lexical levels, failing to isolate fine-grained weaknesses at the morphological level.

**Limitations of Prior Work**: (1) In morphologically rich languages early like French, Arabic, and Hindi, gender affects verb conjugation, pronouns, adjectives, and even first-person constructions, yet existing benchmarks do not directly evaluate this capability; (2) existing gender bias datasets (e.g., WinoMT, MT-GenEval) rely on rigid templates and are small in scale, lacking a systematic evaluation that covers the full range of gender marking phenomena; (3) there is a lack of rigorous indicators that can simultaneously evaluate the correctness of gender conversion and penalize erroneous modifications.

**Key Challenge**: Success at the semantic level masks deficiencies at the morphological level—models may "know" the meaning of a word but cannot correctly apply gender morphology rules, especially in multi-entity scenarios where gender interference occurs.

**Goal**: To build the first benchmark for systematically evaluating the gender-aware morphological generation capabilities of multilingual LLMs, covering three typologically diverse languages and providing fine-grained evaluation metrics.

**Key Insight**: French, Arabic, and Hindi were selected as they employ distinct gender marking strategies—French combines phonological, morphological, and semantic cues; Arabic features a highly regular suffix system; and Hindi utilizes natural gender with partial morphological marking—forming a complementary testing platform.

**Core Idea**: The GENFORM task (gender-reversal rewriting) is used to isolate and evaluate the morphological reasoning capabilities of LLMs. Combined with the GIoU metric, which penalizes both under-generation and over-generation, systemic weaknesses across different languages and constructions are revealed.

## Method

### Overall Architecture

The construction of MORPHOGEN follows four steps: (1) Designing 12–14 morphological rules per language based on grammatical characteristics (covering verb tenses, adjectives, pronouns, passive voice, multi-entity, etc.); (2) Generating English source sentences using controlled templates and LLMs, then translating them into the target languages; (3) Manual correction by multilingual annotators into male and female versions; (4) Cross-validation to ensure data quality (validation score 0.9705, inter-annotator agreement 0.9495).

### Key Designs

1.  **GENFORM Task Design**:
    - **Function**: To isolate and evaluate the gender-aware morphological reasoning capability of LLMs, eliminating interference from semantic understanding.
    - **Mechanism**: Given a first-person sentence and the speaker's gender, the model must rewrite the sentence into the opposite gender while maintaining semantics, fluency, and syntactic structure. "Gendered words" are defined as the differing words between the source and its gender-counterpart, averaging 1.43–2.02 words per sentence.
    - **Design Motivation**: First-person sentences provide the most natural scenario for gender morphology testing—the speaker's gender is marked explicitly or implicitly through verb conjugation and adjective endings, requiring compositional morphological reasoning rather than surface-level replacement.

2.  **GIoU (Gender IoU) Metric**:
    - **Function**: To provide a more rigorous evaluation than sentence-level accuracy by penalizing both under-generation and over-generation.
    - **Mechanism**: Inspired by IoU in object detection, it calculates the ratio of correctly converted gendered words to the union of gendered words plus mismatched ones: $$\text{GIoU} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Gendered}_i \cap \text{Mismatch}_i^c|}{|\text{Gendered}_i \cup \text{Mismatch}_i|}$$
    - **Design Motivation**: SGA only evaluates whether the words that should be changed are handled correctly, but does not penalize the model for incorrectly modifying non-gendered words. GIoU is particularly important in multi-entity scenarios—if a model changes the gender of all entities (rather than just the speaker), the GIoU will drop significantly.

3.  **Systematic Morphological Rule System**:
    - **Function**: To ensure the evaluation covers the full spectrum of gender marking phenomena in each language.
    - **Mechanism**: 12–14 rules per language cover five categories: (1) Verbs and Tenses—differences in conjugation; (2) Adjectives and Occupational Nouns—morphological marking; (3) Pronouns and Possessives—language-specific strategies; (4) Clause-level Effects—syntactic impact on gender marking (e.g., passive voice); (5) Multi-entity and Gender Interference—scenarios where only the speaker's gender should change despite the presence of two human referents.
    - **Design Motivation**: Different rules examine various aspects of morphological reasoning, ranging from simple rules (pronoun replacement) to complex rules (multi-entity interference), creating a difficulty gradient.

### Loss & Training

MORPHOGEN is an evaluation benchmark and does not involve model training. Fifteen LLMs were evaluated in a zero-shot setting.

## Key Experimental Results

### Main Results

**Cross-lingual GIoU Performance Comparison**

| Model | French GIoU | Arabic GIoU | Hindi GIoU |
|------|----------|-------------|-----------|
| Gemma2-2B | 39.73 | 14.73 | 71.41 |
| LLAMA-3.1-8B | 67.89 | 43.51 | 83.12 |
| Phi4-14B | 79.84 | 57.08 | 82.77 |
| LLAMA-3.3-70B | 76.68 | 59.16 | 93.33 |
| GPT-4o-mini | **86.43** | **71.02** | 88.81 |

### Ablation Study

**Gender Bias Analysis (△SGA = SGA_M - SGA_F)**

| Model | French △SGA | Arabic △SGA | Hindi △SGA |
|------|----------|-------------|-----------|
| LLAMA-3.3-70B | +15.15 | +7.50 | +3.67 |
| Gemma3-4B | -14.16 | -8.20 | -14.32 |
| Qwen3-32B | +10.10 | +11.94 | +5.14 |

### Key Findings

- Model scale is critical for complex morphology—in Arabic (the strictest morphological system), the CGA of the Gemma series improved from 14.10% (2B) to 74.74% (27B).
- A persistent masculine bias exists in French and Arabic (larger models tend to default to masculine forms), whereas some Hindi models exhibit a feminine bias.
- The gap between GIoU and SGA exposes multi-entity gender interference—models frequently and incorrectly modify the gender of entities other than the speaker.
- Due to its relatively simpler morphology, small models achieve better performance in Hindi (LLAMA-3.1-8B CGA=89.21%).

## Highlights & Insights

- The design of the GENFORM task is ingenious—isolating the evaluation of morphological reasoning through gender-reversal rewriting avoids interference from semantic understanding. This task design can be transferred to evaluate other grammatical dimensions in LLMs.
- The GIoU metric fills a significant gap—in multi-entity scenarios, it is insufficient to only measure "how many were corrected"; one must also check "whether things that should not have been changed were modified." This is instructive for any NLP task involving precise editing.
- The selection of three languages covers three distinct gender marking strategies (phonological-morphological-semantic/regular suffixes/natural gender), providing typological breadth to the findings.

## Limitations & Future Work

- Coverage is limited to only three languages and lacks dialectal variations (e.g., Arabic dialects have different gender marking patterns).
- Only binary gender systems are considered, excluding non-binary expressions.
- The Arabic dataset is smaller in scale (2,719 sentences vs. 9,999 for French).
- Multi-entity scenarios are restricted to two human referents; more complex multi-entity discourse is not covered.

## Related Work & Insights

- **vs WinoMT**: Relies on rigid templates (~1K samples); MORPHOGEN covers broader morphological phenomena with a larger volume of data.
- **vs MT-GenEval**: Lacks first-person sentences and speaker gender labels; MORPHOGEN focuses on speaker-gender-driven morphological changes.
- **vs MuST-SHE**: Provides speaker annotations but is not public; MORPHOGEN is open under CC BY-NC 4.0.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic multilingual gender morphology benchmark; ingenious GENFORM task design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models + 3 languages + 3 metrics + directional bias analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed documentation of linguistic rules.
- Value: ⭐⭐⭐⭐ Fills the gap in morphological evaluation of multilingual LLMs; GIoU metric is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks](mitigating_extrinsic_gender_bias_for_bangla_classification_tasks.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)

</div>

<!-- RELATED:END -->
