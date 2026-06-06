---
title: >-
  [Paper Note] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation
description: >-
  [ACL 2026][Multilingual & Machine Translation][gender-aware morphological generation] This paper introduces MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic, and Hindi (20…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "gender-aware morphological generation"
  - "multilingual benchmark"
  - "French/Arabic/Hindi"
  - "LLM morphological reasoning"
  - "gender bias evaluation"
date: 2026-05-08
content_hash: bb022f6f96697e9c
---

# MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation

**Conference**: ACL 2026
**arXiv**: [2604.18914](https://arxiv.org/abs/2604.18914)  
**Code**: [GitHub](https://github.com/) (Code + Dataset links provided)  
**Area**: Multilingual Translation
**Keywords**: gender-aware morphological generation, multilingual benchmark, French/Arabic/Hindi, LLM morphological reasoning, gender bias evaluation

## TL;DR

This paper introduces MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic, and Hindi (20,328 sentence pairs in total). It defines the GENFORM task (rewriting first-person sentences into the opposite gender), proposes three evaluation metrics—SGA, GIoU, and CGA—and benchmarks 15 multilingual LLMs, revealing systematic deficiencies in complex morphological reasoning, gender bias, and multi-entity interference.

## Background & Motivation

**Background**: Multilingual LLMs perform well on high-level tasks such as translation and question answering, yet their ability to handle grammatical gender and morphological agreement has not been systematically evaluated. Existing multilingual benchmarks (XTREME, Global-MMLU) focus on semantic and lexical levels and cannot isolate fine-grained morphological weaknesses.

**Limitations of Prior Work**: (1) In morphologically rich languages such as French, Arabic, and Hindi, gender affects verb conjugation, pronouns, adjectives, and even first-person constructions, yet existing benchmarks do not directly assess this capability; (2) existing gender bias datasets (WinoMT, MT-GenEval) rely on rigid templates and are small in scale, lacking systematic evaluation of the full range of gender-marking phenomena; (3) no rigorous metric exists that simultaneously rewards correct gender conversion and penalizes erroneous modifications.

**Key Challenge**: LLMs' success at the semantic level masks their deficiencies at the morphological level—a model may "know" the meaning of a word yet fail to apply gender morphology rules correctly, with gender interference becoming particularly pronounced in multi-entity scenarios.

**Goal**: To construct the first benchmark that systematically evaluates gender-aware morphological generation in multilingual LLMs, covering three typologically diverse languages and providing fine-grained evaluation metrics.

**Key Insight**: Three languages with distinct gender-marking strategies are selected—French, which combines phonological, morphological, and semantic cues; Arabic, which employs a highly regular suffix system; and Hindi, which uses natural gender with partial morphological marking—forming a complementary test platform.

**Core Idea**: The GENFORM task (gender-reversal rewriting) isolates the evaluation of LLMs' morphological reasoning ability. Combined with the GIoU metric, which simultaneously penalizes under-generation and over-generation, the benchmark reveals systematic weaknesses across languages and constructions.

## Method

### Overall Architecture

MORPHOGEN is constructed in four steps: (1) designing 12–14 morphological rules per language based on each language's grammatical properties, covering verb tense, adjectives, pronouns, passive voice, and multi-entity scenarios; (2) generating English source sentences using controlled templates and LLM-assisted generation, then translating them into the target languages; (3) having multilingual annotators manually correct each sentence into male and female versions; and (4) applying cross-validation to ensure data quality (validation score: 0.9705; annotator agreement: 0.9495).

### Key Designs

1. **GENFORM Task Design**:

    - Function: Isolates the evaluation of LLMs' gender-aware morphological reasoning by eliminating confounds from semantic understanding.
    - Mechanism: Given a first-person sentence and the speaker's gender, the model must rewrite the sentence in the opposite gender while preserving semantics, fluency, and syntactic structure. "Gendered words" are defined as the differing tokens between the source sentence and its gender-opposite counterpart, averaging 1.43–2.02 gendered words per sentence.
    - Design Motivation: First-person sentences represent the most natural test scenario for gender morphology—the speaker's gender is marked explicitly or implicitly through verb conjugations, adjectival endings, and similar features, requiring the model to perform compositional morphological reasoning rather than surface-level substitution.

2. **GIoU (Gender IoU) Metric**:

    - Function: Provides a stricter evaluation than sentence-level accuracy by simultaneously penalizing under-generation (missed changes) and over-generation (incorrect changes).
    - Mechanism: Inspired by IoU in object detection, GIoU computes the ratio of correctly converted gendered words to the union of gendered words and mismatched words: $\text{GIoU} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Gendered}_i \cap \text{Mismatch}_i^c|}{|\text{Gendered}_i \cup \text{Mismatch}_i|}$
    - Design Motivation: SGA only assesses whether words that should be changed are changed correctly, without penalizing the model for erroneously modifying non-gendered words. GIoU is especially important in multi-entity scenarios—if a model changes the gender of all entities rather than only the speaker, GIoU decreases substantially.

3. **Systematic Morphological Rule Framework**:

    - Function: Ensures that evaluation covers the complete gender-marking phenomena of each language.
    - Mechanism: 12–14 rules are designed per language, spanning five categories: (1) verbs and tense—gender agreement variation across tenses; (2) adjectives and occupational nouns—gender morphological marking; (3) pronouns and possessives—language-specific gender-marking strategies; (4) clause-level effects—the impact of syntactic constructions such as passive voice on gender marking; and (5) multi-entity and gender interference—scenarios with two human referents where only the speaker's gender should change.
    - Design Motivation: Different rules probe distinct aspects of morphological reasoning—from simple rules (e.g., pronoun substitution) to complex ones (e.g., multi-entity interference)—forming a difficulty gradient.

### Loss & Training

MORPHOGEN is an evaluation benchmark and does not involve model training. All 15 LLMs are evaluated in a zero-shot setting.

## Key Experimental Results

### Main Results

**Cross-Lingual GIoU Performance Comparison**

| Model | French GIoU | Arabic GIoU | Hindi GIoU |
|-------|-------------|-------------|------------|
| Gemma2-2B | 39.73 | 14.73 | 71.41 |
| LLAMA-3.1-8B | 67.89 | 43.51 | 83.12 |
| Phi4-14B | 79.84 | 57.08 | 82.77 |
| LLAMA-3.3-70B | 76.68 | 59.16 | 93.33 |
| GPT-4o-mini | **86.43** | **71.02** | 88.81 |

### Ablation Study

**Gender Bias Analysis (△SGA = SGA_M - SGA_F)**

| Model | French △SGA | Arabic △SGA | Hindi △SGA |
|-------|-------------|-------------|------------|
| LLAMA-3.3-70B | +15.15 | +7.50 | +3.67 |
| Gemma3-4B | -14.16 | -8.20 | -14.32 |
| Qwen3-32B | +10.10 | +11.94 | +5.14 |

### Key Findings

- Model scale is critical for complex morphology—on Arabic (the most demanding morphological system), CGA for the Gemma series improves from 14.10% at 2B to 74.74% at 27B.
- French and Arabic exhibit a persistent masculine bias (larger models tend to default to masculine forms), whereas some models show a feminine bias in Hindi.
- The gap between GIoU and SGA exposes multi-entity gender interference—models erroneously modify the gender of non-speaker entities.
- Hindi's relatively simpler morphology allows even small models to achieve strong performance (LLAMA-3.1-8B CGA = 89.21%).

## Highlights & Insights

- The GENFORM task design is particularly elegant—by framing evaluation as gender-reversal rewriting, it isolates morphological reasoning from semantic understanding. This task design paradigm is transferable to evaluating LLMs on other grammatical dimensions.
- The GIoU metric fills an important gap: in multi-entity scenarios, measuring only "how many words were correctly changed" is insufficient; one must also account for "whether non-target words were incorrectly modified." This has broad implications for any NLP task involving precise editing.
- The selection of three languages covering three distinct gender-marking strategies (phonological-morphological-semantic / regular suffixes / natural gender) gives the findings typological breadth.

## Limitations & Future Work

- Only three languages are covered, with no dialectal variation (e.g., Arabic dialects have different gender-marking patterns).
- Only binary gender systems are considered; non-binary gender expression is not addressed.
- The Arabic dataset is smaller in scale (2,719 sentences vs. 9,999 for French).
- Multi-entity scenarios are limited to two human referents; more complex multi-entity discourse is not covered.

## Related Work & Insights

- **vs. WinoMT**: Relies on rigid templates (~1K samples); MORPHOGEN covers a broader range of morphological phenomena at a much larger scale.
- **vs. MT-GenEval**: Lacks first-person sentences and speaker gender labels; MORPHOGEN focuses on morphological variation driven by speaker gender.
- **vs. MuST-SHE**: Provides speaker annotations but is not publicly released; MORPHOGEN is openly available under CC BY-NC 4.0.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic multilingual gender morphology benchmark with an elegantly designed GENFORM task.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models × 3 languages × 3 metrics + directional bias analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough documentation of language-specific morphological rules.
- Value: ⭐⭐⭐⭐ Fills a gap in morphological evaluation for multilingual LLMs; the GIoU metric is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks](mitigating_extrinsic_gender_bias_for_bangla_classification_tasks.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](../../AAAI2026/multilingual_mt/bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)
- [\[ACL 2026\] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)

</div>

<!-- RELATED:END -->
