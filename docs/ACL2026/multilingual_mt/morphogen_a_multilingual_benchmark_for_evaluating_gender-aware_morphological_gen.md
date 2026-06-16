---
title: >-
  [Paper Note] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation
description: >-
  [ACL 2026][Multilingual & Translation][Paper Note] Ours presents MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic, and Hindi (totaling 20,328 sentence pairs). It defines the GENFORM task (rewriting first-person sentences into the opposite gender) and proposes three evaluation metrics: SGA, GIoU, and CGA. Benchmarking 15 m
tags:
  - ACL 2026
  - Multilingual & Translation
date: 2026-05-08
content_hash: bf2aec4ef32fbedc
---
# MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation

**Conference**: ACL 2026  
**arXiv**: [2604.18914](https://arxiv.org/abs/2604.18914)  
**Code**: [GitHub](https://github.com/) (Code + Dataset link provided)  
**Area**: Multilingual Translation  
**Keywords**: Gender-aware morphological generation, multilingual benchmark, French/Arabic/Hindi, LLM morphological reasoning, gender bias evaluation

## TL;DR

Ours presents MORPHOGEN, a large-scale gender-aware morphological generation benchmark covering French, Arabic, and Hindi (totaling 20,328 sentence pairs). It defines the GENFORM task (rewriting first-person sentences into the opposite gender) and proposes three evaluation metrics: SGA, GIoU, and CGA. Benchmarking 15 multilingual LLMs reveals systematic deficiencies in complex morphological reasoning, gender bias, and multi-entity interference.

## Background & Motivation

**Background**: Multilingual LLMs perform well on high-level tasks such as translation and QA, but their capabilities regarding grammatical gender and morphological consistency have not been systematically evaluated. Existing multilingual benchmarks (XTREME, Global-MMLU) focus on semantic and lexical levels, failing to isolate fine-grained morphological weaknesses.

**Limitations of Prior Work**: (1) In morphologically rich languages (e.g., French, Arabic, Hindi), gender affects verb conjugation, pronouns, adjectives, and even first-person structures, yet existing benchmarks do not directly evaluate this capability; (2) Existing gender bias datasets (WinoMT, MT-GenEval) rely on rigid templates and small scales, lacking systematic evaluation of comprehensive gender marking phenomena; (3) There is a lack of rigorous metrics that simultaneously evaluate gender conversion correctness and penalize erroneous modifications.

**Key Challenge**: Success at the semantic level masks deficiencies at the morphological level—models may "know" the meaning of a word but cannot correctly apply gender morphological rules, particularly in multi-entity scenarios where gender interference occurs.

**Goal**: Build the first benchmark for systematically evaluating the gender-aware morphological generation capabilities of multilingual LLMs, covering three typologically diverse languages and providing fine-grained evaluation metrics.

**Key Insight**: Select French, Arabic, and Hindi—three languages with distinct gender marking strategies: French (phonetic/morphological/semantic cues), Arabic (highly regular suffix system), and Hindi (natural gender + partial morphological marking)—to form a complementary testing platform.

**Core Idea**: Isolate the morphological reasoning capability of LLMs through the GENFORM task (gender-reversal rewriting), paired with the GIoU metric to simultaneously penalize missed and incorrect modifications, revealing systematic weaknesses across languages and constructions.

## Method

### Overall Architecture

The construction of MORPHOGEN follows four steps: (1) Designing 12-14 morphological rules per language (covering verb tenses, adjectives, pronouns, passive voice, multi-entity, etc.); (2) Using controlled templates + LLM to generate English source sentences, then translating them into target languages; (3) Manual correction by multilingual annotators into male/female versions; (4) Cross-validation to ensure data quality (verification score 0.9705, inter-annotator agreement 0.9495).

### Key Designs

**1. GENFORM Task: Isolate morphological reasoning from semantic understanding via first-person gender-reversal rewriting**

Existing benchmarks evaluate either semantics or lexicon, failing to examine whether a model truly masters gender morphological rules—a model might understand a word's meaning but use incorrect verb conjugations or adjective endings. GENFORM is designed given a first-person sentence and the speaker's gender, requiring the model to rewrite it into the opposite gender while maintaining semantics, fluency, and syntactic structure. "Gendered words" are precisely defined as the differences between the source sentence and its gender-counterpart, averaging 1.43-2.02 per sentence. First-person is chosen as it is the most natural scenario for gender morphology: the speaker's gender is marked explicitly or implicitly, forcing the model to perform compositional morphological reasoning rather than surface-level replacement.

**2. GIoU (Gender IoU) Metric: Penalizing both missed and erroneous modifications, more rigorous than sentence accuracy**

Sentence-level accuracy (SGA) only checks if the words that should be changed are correct, ignoring cases where the model incorrectly modifies non-gendered words—especially in multi-entity sentences where models often change the gender of all entities instead of just the speaker. GIoU borrows the IoU concept from object detection, calculating a score as the ratio of correctly converted gendered words to the "union of gendered words plus mismatched words":

$$\text{GIoU} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Gendered}_i \cap \text{Mismatch}_i^c|}{|\text{Gendered}_i \cup \text{Mismatch}_i|}$$

By including mismatched words in the denominator, any unnecessary modification of entities results in a significant score drop. Consequently, GIoU is particularly effective at exposing issues in multi-entity gender interference scenarios where SGA might fail to detect errors.

**3. Systemic Morphological Rule System: A comprehensive grid covering gender marking phenomena**

Gender marking strategies vary significantly across languages. The authors design 12-14 rules per language, categorized into five groups: Verbs & Tense (conjugation differences), Adjectives & Occupation Nouns (morphological markers), Pronouns & Possessives (language-specific strategies), Clause-level effects (syntactic influences like passive voice), and Multi-entity & Gender interference (ensuring only the speaker's gender is modified when two human referents coexist). These rules range from simple substitutions to complex interference, forming a difficulty gradient for itemized evaluation.

### Loss & Training

MORPHOGEN is an evaluation benchmark and does not involve model training. 15 LLMs were evaluated in a zero-shot setting.

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

- Model scale is crucial for complex morphology—in Arabic (the most rigorous morphological system), the CGA of the Gemma series improved from 14.10% (2B) to 74.74% (27B).
- Persistent masculine bias exists in French and Arabic (large models favor default masculine forms), while some models show feminine bias in Hindi.
- The gap between GIoU and SGA exposes multi-entity gender interference—models incorrectly modified the gender of non-speaker entities.
- Due to its relatively simpler morphology, small models achieve better performance in Hindi (LLAMA-3.1-8B CGA=89.21%).

## Highlights & Insights

- The design of the GENFORM task is sophisticated—it isolates morphological reasoning by gender reversal, avoiding interference from semantic understanding. This design can be transferred to evaluate other grammatical dimensions in LLMs.
- The GIoU metric fills an important gap—in multi-entity scenarios, checking "how many were corrected" is insufficient; one must also check "if anything was incorrectly changed." This is applicable to any NLP task involving precise editing.
- The selection of three languages covers three distinct marking strategies (phonetic-morphological-semantic / regular suffix / natural gender), providing typological breadth to the findings.

## Limitations & Future Work

- Coverage is limited to three languages and lacks dialectal variations (e.g., Arabic dialects have different gender marking patterns).
- Only binary gender systems are considered, excluding non-binary gender expressions.
- The Arabic dataset scale is smaller (2,719 sentences vs. 9,999 for French).
- Multi-entity scenarios are limited to two human referents; more complex multi-entity discourse is not covered.

## Related Work & Insights

- **vs WinoMT**: Relies on rigid templates (~1K samples); MORPHOGEN covers broader morphological phenomena with larger data volume.
- **vs MT-GenEval**: Lacks first-person sentences and speaker gender labels; MORPHOGEN focuses on speaker-driven morphological changes.
- **vs MuST-SHE**: Provides speaker annotations but is not public; MORPHOGEN is open under CC BY-NC 4.0.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic multilingual gender morphology benchmark; clever GENFORM task design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models + 3 languages + 3 metrics + directional bias analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed morphological rule documentation.
- Value: ⭐⭐⭐⭐ Fills the gap in morphological evaluation for multilingual LLMs; GIoU metric is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](../../ACL2025/multilingual_mt/lexgen_domain-aware_multilingual_lexicon_generation.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)

</div>

<!-- RELATED:END -->
