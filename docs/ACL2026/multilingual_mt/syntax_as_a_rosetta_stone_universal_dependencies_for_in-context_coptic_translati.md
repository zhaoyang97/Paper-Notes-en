---
title: >-
  [Paper Note] Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation
description: >-
  [ACL 2026][Low-resource machine translation] This paper is the first to explore Universal Dependencies (UD) syntactic information as an augmentation source for in-context learning (ICL) in low-resource Coptic-to-English machine translation. While syntactic information alone is less effective than a bilingual lexicon, combining lexicon with syntactic information (LEX+SYN) achieves the best results across all tested models, with Gemma-27B reaching a BERTScore F1 of 0.8746 (+0.0361).
tags:
  - ACL 2026
  - Low-resource machine translation
  - Coptic
  - Universal Dependencies
  - in-context learning
  - syntax-augmented translation
date: 2026-05-08
content_hash: 0fefe16721eabc07
---

# Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation

**Conference**: ACL 2026
**arXiv**: [2604.18758](https://arxiv.org/abs/2604.18758)
**Code**: [GitHub](https://github.com/gucorpling/in-context-coptic-translation)
**Area**: Low-Resource Machine Translation / Multilingual NLP
**Keywords**: Low-resource machine translation, Coptic, Universal Dependencies, in-context learning, syntax-augmented translation

## TL;DR

This paper is the first to explore Universal Dependencies (UD) syntactic information as an augmentation source for in-context learning (ICL) in low-resource Coptic-to-English machine translation. While syntactic information alone is less effective than a bilingual lexicon, combining lexicon with syntactic information (LEX+SYN) achieves the best results across all tested models, with Gemma-27B reaching a BERTScore F1 of 0.8746 (+0.0361).

## Background & Motivation

**Background**: LLMs have approached practical usability for high-resource language translation, but low-resource languages (LRLs) have barely benefited — models lack fundamental language modeling capacity for these languages. Augmenting ICL prompts with bilingual lexicons has been shown to be effective, yet lexicons only provide word-by-word translations and cannot encode grammatical relations.

**Limitations of Prior Work**: (1) Coptic is an agglutinative language whose grammatical constructions (e.g., the auxiliary system, postponed subjects) carry meaning distinctions that cannot be inferred from content words alone; (2) even large models such as GPT-4.1 produce fluent but fundamentally incorrect Coptic translations without augmentation; (3) simple word lists face an information ceiling — they cannot inform the model about grammatical structure.

**Key Challenge**: Lexicon augmentation covers only the lexical dimension; the information gap in the grammatical dimension limits the upper bound of translation quality. A mechanism is needed to inject grammatical information into in-context prompts.

**Goal**: To verify whether UD syntactic information can provide complementary gains beyond a bilingual lexicon for low-resource translation in an ICL setting.

**Key Insight**: Coptic already has an available UD treebank (60K tokens, 2,387 sentences). The paper designs multiple syntactic information representations — raw CoNLL-U, natural-language verbalization, and construction rules derived from error analysis — and tests their combined effect with the lexicon.

**Core Idea**: Syntactic information and lexical information are orthogonally complementary: the lexicon addresses "what does each word mean," while syntax addresses "how are words related to one another." Combining both can break through the information ceiling imposed by lexicon-only augmentation.

## Method

### Overall Architecture

Four information components are designed: LEX (bilingual lexicon mapping), DEP (natural-language verbalization of UD dependency relations), CON (26 construction rules derived from error analysis), and CoNLLU (raw UD parse output). Single-component and combination settings are evaluated, with BERTScore F1 as the primary metric.

### Key Designs

1. **LEX Component**:

    - **Function**: Maps Coptic vocabulary to English translations, providing word-level semantic information.
    - **Mechanism**: Uses part-of-speech and morphological information from syntactic parsing (lemmas and tokenization) to retrieve dialect-specific translations from a Coptic lexicon. Entries are further filtered to retain the most relevant hierarchical information for prompt-length control.
    - **Design Motivation**: An established baseline component proven effective for low-resource translation, serving as the reference point for subsequent syntactic augmentation.

2. **CON Component**:

    - **Function**: Provides translation instructions for specific grammatical constructions targeting typical model errors observed on the development set.
    - **Mechanism**: GPT-4.1 mini's translation errors on the development set are manually analyzed; 26 common error patterns are identified in conjunction with syntactic parsing, and DepEdit templates are used to match dependency subtrees. Rules range from simple (e.g., imperative identification) to complex (e.g., postponed subject constructions), automatically detecting patterns and generating translation guidance.
    - **Design Motivation**: The most customized component — it directly targets the model's known weaknesses. Unlike generic syntactic information, construction rules encode domain-expert understanding of translation pitfalls.

3. **DEP Component**:

    - **Function**: Translates UD syntactic structure into brief English statements, enabling LLMs to understand inter-word relations.
    - **Mechanism**: Head–dependent relations are extracted for each sentence and verbalized as plain English statements (e.g., "n is the case marking of you"). Controllable parameters include the selected UD label set, selected part-of-speech tags, and disambiguation of repeated tokens.
    - **Design Motivation**: Raw CoNLL-U format may be "recognized" by LLMs through training exposure but not necessarily understood semantically. Verbalization converts structural information into a natural-language form that LLMs can more readily exploit.

### Loss & Training

Pure ICL setting; no training is performed. Three models are evaluated: Gemma-12B, Gemma-27B, and GPT-4.1. Evaluation uses the standard dev (380 sentences) and test (405 sentences) splits of the UD treebank.

## Key Experimental Results

### Main Results

**Test Set BERTScore F1**

| Setting | Gemma-12B | Gemma-27B | GPT-4.1 |
|---------|-----------|-----------|---------|
| Baseline | 0.8363 | 0.8385 | 0.9012 |
| LEX | 0.8551 (+0.019) | 0.8565 (+0.018) | 0.9152 (+0.014) |
| LEX+SYN | **0.8707 (+0.034)** | **0.8746 (+0.036)** | **0.9195 (+0.018)** |

### Ablation Study

| Setting | BERTScore Δ (Gemma-27B test) | Note |
|---------|------------------------------|------|
| Baseline | 0.0000 | No augmentation |
| DEP alone | +0.0033 | Dependency verbalization only; marginal gain |
| CON alone | +0.0133 | Construction rules effective in isolation |
| CoNLLU alone | +0.0162 | Raw parse output performs reasonably well |
| LEX alone | +0.0181 | Most effective single component |
| LEX+SYN | +0.0361 | Combined effect far exceeds any single component |

### Key Findings

- Syntactic information alone is less effective than the lexicon, but in combination it provides substantial gains beyond the lexicon alone (LEX+SYN exceeds LEX alone by +0.018 BERTScore on Gemma-27B).
- The combination effect is approximately additive — the individual gains of LEX and SYN sum close to the total gain of LEX+SYN.
- The gap between automatic and gold-standard parses is small, indicating that syntactic augmentation is robust to parser errors.
- Gains are larger on non-biblical text, suggesting that syntactic information is more valuable when the model cannot rely on memorized scripture.

## Highlights & Insights

- Applying UD syntactic information to ICL-based translation is a first — demonstrating that grammatical knowledge can be injected into prompts as "another reference source."
- The CON component embodies an error-driven engineering philosophy: rather than providing all grammatical information generically, it offers targeted guidance directed at the model's actual failure modes.
- The approach has clear practical value for LRL translation workflows, with the potential to reduce the burden of expert post-editing.

## Limitations & Future Work

- Limited to the single direction of Coptic-to-English translation.
- Passage-level translation restricts evaluation of discourse-level phenomena.
- The CON component requires linguistic expert involvement in error analysis.
- The use of parallel few-shot examples remains unexplored.

## Related Work & Insights

- **vs. Dictionary-only ICL**: LEX serves as the baseline; this paper demonstrates that syntactic augmentation provides complementary information not covered by the lexicon.
- **vs. Grammar excerpt approaches**: Rules extracted from grammar books are general-purpose, whereas CON rules are customized to the model's specific weaknesses.
- **vs. Fine-tuning approaches**: Fine-tuning requires parallel data and still yields suboptimal results; ICL with augmentation is more flexible.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of UD syntactic information to ICL-based translation; the CON component design is particularly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model evaluation, multi-component ablation, gold vs. automatic parse comparison, and biblical vs. non-biblical text analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear paper structure with thorough introduction of linguistic background.
- **Value**: ⭐⭐⭐⭐ Makes a concrete contribution to low-resource translation research; the method is transferable to other LRLs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](../../AAAI2026/multilingual_mt/gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)
- [\[NeurIPS 2025\] How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](../../NeurIPS2025/multilingual_mt/how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] Just Use XML: Revisiting Joint Translation and Label Projection](just_use_xml_revisiting_joint_translation_and_label_projection.md)
- [\[ACL 2026\] Lost in Translation: Do LVLM Judges Generalize Across Languages?](lost_in_translation_do_lvlm_judges_generalize_across_languages.md)

</div>

<!-- RELATED:END -->
