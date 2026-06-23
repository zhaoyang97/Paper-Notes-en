---
title: >-
  [Paper Note] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation
description: >-
  [ACL 2026][Multilingual & Translation][Paper Note] This paper proposes MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic, and Hindi (20,328 sentence pairs). It defines the GENFORM task (rewriting first-person sentences to the opposite gender) and introduces three evaluation metrics: SGA, GIoU, and CGA. Benchmarking 15 mult
tags:
  - ACL 2026
  - Multilingual & Translation
date: 2026-05-08
content_hash: 8bec6d7e64caec75
---
# MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation

**Conference**: ACL 2026  
**arXiv**: [2604.18914](https://arxiv.org/abs/2604.18914)  
**Code**: [GitHub](https://github.com/) (Code + Dataset link provided)  
**Area**: Multilingual Translation  
**Keywords**: Gender-aware morphological generation, multilingual benchmark, French Arabic Hindi, LLM morphological reasoning, gender bias evaluation

## TL;DR

This paper proposes MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic, and Hindi (20,328 sentence pairs). It defines the GENFORM task (rewriting first-person sentences to the opposite gender) and introduces three evaluation metrics: SGA, GIoU, and CGA. Benchmarking 15 multilingual LLMs reveals systematic deficiencies in complex morphological reasoning, gender bias, and multi-entity interference.

## Background & Motivation

**Background**: Multilingual LLMs perform well on high-level tasks like translation and QA, but their capabilities regarding grammatical gender and morphological consistency have not been systematically evaluated. Existing multilingual benchmarks (e.g., XTREME, Global-MMLU) focus on semantic and lexical levels, failing to isolate fine-grained morphological weaknesses.

**Limitations of Prior Work**: (1) In morphologically rich languages (e.g., French, Arabic, Hindi), gender affects verb conjugation, pronouns, adjectives, and even first-person structures, yet existing benchmarks do not directly evaluate this capability; (2) Existing gender bias datasets (e.g., WinoMT, MT-GenEval) rely on rigid templates and are small in scale, lacking systematic evaluation of the full range of gender-marking phenomena; (3) There is a lack of rigorous metrics capable of simultaneously evaluating the correctness of gender conversion and penalizing erroneous modifications.

**Key Challenge**: Success at the semantic level masks deficiencies at the morphological level—models may "know" the meaning of a word but cannot correctly apply gender morphology rules, especially in multi-entity scenarios where gender interference occurs.

**Goal**: To build the first benchmark for systematically evaluating the gender-aware morphological generation capabilities of multilingual LLMs, covering three typologically diverse languages and providing fine-grained evaluation metrics.

**Key Insight**: Choosing French, Arabic, and Hindi—three languages with distinct gender-marking strategies (French combining phonological/morphological/semantic cues, Arabic's highly regular suffix system, and Hindi's natural gender plus partial morphological marking)—creates a complementary testing platform.

**Core Idea**: Isolate the morphological reasoning capability of LLMs through the GENFORM task (gender-reversal rewriting), paired with the GIoU metric to simultaneously penalize missed and incorrect modifications, revealing systematic weaknesses across different languages and constructions.

## Method

### Overall Architecture

The construction of MORPHOGEN follows four steps: (1) Designing 12-14 morphological rules per language based on grammatical characteristics (covering verb tenses, adjectives, pronouns, passive voice, multi-entity, etc.); (2) Generating English source sentences using controlled templates + LLMs, then translating them into target languages; (3) Manual correction by multilingual annotators into male/female versions; (4) Cross-validation to ensure data quality (validation score 0.9705, inter-annotator agreement 0.9495).

### Key Designs

**1. GENFORM Task: Isolating morphological reasoning from semantic understanding via first-person gender-reversal rewriting**

Existing benchmarks test either semantics or lexicon, failing to examine whether a model truly masters gender morphology rules—a model might "know" word meanings but use incorrect verb conjugations or adjective endings. The GENFORM task provides a first-person sentence and the speaker's gender, requiring the model to rewrite it into the opposite gender while maintaining semantics, fluency, and syntactic structure. "Gendered words" are precisely defined as the differences between the source sentence and its gender-counterpart, averaging 1.43-2.02 words per sentence. The first-person is chosen because it is the most natural scenario for gender morphology testing: the speaker's gender is marked explicitly or implicitly through verb conjugations and adjective endings, forcing the model to perform compositional morphological reasoning rather than surface substitution.

**2. GIoU (Gender IoU) Metric: Penalizing both missed and erroneous modifications strictly**

Sentence-level Accuracy (SGA) only checks if the words that should be changed are correct, ignoring cases where the model randomly modifies non-gendered words—especially in multi-entity sentences where models often change the gender of all entities instead of only the speaker. GIoU adopts the IoU concept from object detection, calculating the ratio of correctly converted gendered words to the "union of gendered words and mismatched words":

$$\text{GIoU} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Gendered}_i \cap \text{Mismatch}_i^c|}{|\text{Gendered}_i \cup \text{Mismatch}_i|}$$

By including both the union and mismatched words in the denominator, any modification to entities that should remain unchanged significantly lowers the score. Consequently, GIoU effectively exposes problems in multi-entity gender interference scenarios where SGA might fail to detect flaws.

**3. Systematic Morphological Rule System: Covering complete gender-marking phenomena**

Gender-marking strategies vary significantly across languages; a few templates cannot cover them all. The authors designed 12-14 rules for each language, categorized into five groups: Verbs and Tenses (differences in gender conjugation across tenses), Adjectives and Occupational Nouns (morphological gender marking), Pronouns and Possessives (language-specific marking strategies), Clause-level Effects (syntactic structures like passive voice affecting gender marking), and Multi-entity and Gender Interference (where only the speaker should be modified when two human referents coexist). These rules range from simple pronoun replacement to complex multi-entity interference, naturally forming a difficulty gradient that allows the benchmark to examine different facets of morphological reasoning.

### Loss & Training

MORPHOGEN is an evaluation benchmark and does not involve model training. 15 LLMs were evaluated in a zero-shot manner.

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

**Gender Bias Analysis ($\Delta$SGA = SGA_M - SGA_F)**

| Model | French $\Delta$SGA | Arabic $\Delta$SGA | Hindi $\Delta$SGA |
|------|----------|-------------|-----------|
| LLAMA-3.3-70B | +15.15 | +7.50 | +3.67 |
| Gemma3-4B | -14.16 | -8.20 | -14.32 |
| Qwen3-32B | +10.10 | +11.94 | +5.14 |

### Key Findings

- Model scale is critical for complex morphology—in Arabic (the strictest morphological system), the Gemma series improved CGA from 14.10% to 74.74% when scaling from 2B to 27B.
- A persistent masculine bias exists in French and Arabic (large models prefer default masculine forms), while some Hindi models exhibit a feminine bias.
- The gap between GIoU and SGA exposes the multi-entity gender interference problem, where models incorrectly modify the gender of non-speaker entities.
- Due to its relatively simpler morphology, small models achieve better performance in Hindi (LLAMA-3.1-8B CGA=89.21%).

## Highlights & Insights

- The design of the GENFORM task is ingenious—isolating morphological reasoning via gender-reversal rewriting avoids interference from semantic understanding. This task design can be adapted to evaluate LLM capabilities in other grammatical dimensions.
- The GIoU metric fills a critical gap—in multi-entity scenarios, checking "how many were changed correctly" is insufficient; one must also check "whether things that shouldn't change were modified." This is instructive for any NLP task involving precise editing.
- The selection of three languages covers three distinct gender-marking strategies (phonological-morphological-semantic/regular suffixes/natural gender), providing typological breadth to the findings.

## Limitations & Future Work

- Only three languages are covered, and dialectal variations (e.g., Arabic dialects with different gender-marking patterns) are missing.
- Only binary gender systems are considered, excluding non-binary gender expressions.
- The Arabic dataset is smaller in size (2,719 sentences vs. 9,999 for French).
- Multi-entity scenarios are limited to two human referents; more complex multi-entity discourse is not covered.

## Related Work & Insights

- **vs. WinoMT**: Relies on rigid templates (~1K samples); MORPHOGEN covers broader morphological phenomena with a larger volume of data.
- **vs. MT-GenEval**: Lacks first-person sentences and speaker gender labels; MORPHOGEN focuses on morphological changes driven by the speaker's gender.
- **vs. MuST-SHE**: Provides speaker annotations but is not public; MORPHOGEN is open under CC BY-NC 4.0.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic multilingual gender morphology benchmark; ingenious GENFORM task design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models + 3 languages + 3 metrics + directional bias analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed documentation of linguistic rules.
- Value: ⭐⭐⭐⭐ Fills the gap in morphological evaluation of multilingual LLMs; GIoU metric is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](../../ACL2025/multilingual_mt/lexgen_domain-aware_multilingual_lexicon_generation.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media](plurule_a_benchmark_for_moderating_pluralistic_communities_on_social_media.md)

</div>

<!-- RELATED:END -->
