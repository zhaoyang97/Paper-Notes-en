---
title: >-
  [Paper Note] Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Low-resource Machine Translation] This paper performs the first exploration of using Universal Dependencies (UD) syntactic information as an augmentation source for In-Conte…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Low-resource Machine Translation"
  - "Coptic"
  - "Universal Dependencies"
  - "In-Context Learning"
  - "Syntactic Augmentation"
date: 2026-05-08
content_hash: bd2eae7df9f4f485
---

# Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation

**Conference**: ACL 2026  
**arXiv**: [2604.18758](https://arxiv.org/abs/2604.18758)  
**Code**: [GitHub](https://github.com/gucorpling/in-context-coptic-translation)  
**Area**: Low-resource Machine Translation / Multilingual NLP  
**Keywords**: Low-resource Machine Translation, Coptic, Universal Dependencies, In-Context Learning, Syntactic Augmentation

## TL;DR

This paper performs the first exploration of using Universal Dependencies (UD) syntactic information as an augmentation source for In-Context Learning (ICL) in low-resource Coptic-to-English machine translation. The authors find that while syntactic information alone is less effective than dictionaries, combining dictionaries with syntax (LEX+SYN) achieves the best performance across all models, with Gemma-27B reaching a BERTScore F1 of 0.8746 (+0.0361).

## Background & Motivation

**Background**: LLMs have approached practical utility in high-resource language translation, but low-resource languages (LRLs) have barely benefited, as models lack basic language modeling capabilities for these languages. Augmenting ICL prompts with bilingual dictionaries has proven effective, but dictionaries only provide word-for-word translation and cannot encode grammatical relationships.

**Limitations of Prior Work**: (1) Coptic is an agglutinative language where grammatical constructions (e.g., auxiliary systems, post-posed subjects) carry semantic nuances that cannot be inferred from content words alone; (2) Even powerful models like GPT-4.1 produce fluent but fundamentally incorrect Coptic translations without augmentation; (3) Simple lexicons have an information ceiling—they cannot inform the model about grammatical structure.

**Key Challenge**: Dictionary augmentation only covers the lexical dimension; the information gap in the grammatical dimension limits the upper bound of translation quality. A method is needed to inject grammatical information into in-context prompts.

**Goal**: To verify whether UD syntactic information can provide complementary gains beyond dictionaries for low-resource translation in an ICL setting.

**Key Insight**: Leveraging existing Coptic UD treebanks (60K tokens, 2,387 sentences), the authors design multiple ways to represent syntactic information (raw CoNLL-U, natural language verbalization, and construction rules based on error analysis) to test their combined effects with dictionaries.

**Core Idea**: Syntactic and lexical information are orthogonal and complementary—dictionaries solve "what each word means," while syntax solves "how words relate to each other." Combining both allows for breaking through the information ceiling of dictionary-only augmentation.

## Method

### Overall Architecture

The authors design four information components: LEX (bilingual dictionary mapping), DEP (natural language verbalization of UD dependency relations), CON (26 construction rules based on error analysis), and CoNLLU (raw UD parse output). Single-component and combined settings are tested, with BERTScore F1 as the primary metric.

### Key Designs

1.  **LEX Dictionary Component**:
    - **Function**: Maps Coptic vocabulary to English translations, providing word-level semantic information.
    - **Mechanism**: Utilizes POS and morphological information (lemma and segmentation) from syntactic analysis to retrieve dialect-specific translations in Coptic dictionaries. Entries are further filtered to retain the most relevant hierarchical information to control prompt length.
    - **Design Motivation**: A foundational component proven effective for low-resource translation, providing a baseline for subsequent syntactic augmentation.

2.  **CON Construction Rules Component**:
    - **Function**: Provides translation instructions for specific grammatical constructions targeting typical model errors identified on the development set.
    - **Mechanism**: Manual analysis of GPT-4.1 mini translation errors on the dev set, combined with syntactic analysis to identify 26 common error patterns. DepEdit templates are used to match dependency subtrees. Strategies range from simple rules (e.g., imperative identification) to complex ones (e.g., post-posed subject constructions), enabling automatic detection and generation of translation guidance.
    - **Design Motivation**: The most customized component—directly addressing model weaknesses. Unlike general syntax info, construction rules encode domain experts' understanding of translation pitfalls.

3.  **DEP Dependency Verbalization Component**:
    - **Function**: Translates UD syntactic structures into short English statements so that LLMs can understand inter-word relationships.
    - **Mechanism**: Extracts head-dependent relations for each sentence and verbalizes them into plain English statements (e.g., "n is the case marking of you"). Controllable parameters include the selected UD tag set, POS selection, and disambiguation of repeated tokens.
    - **Design Motivation**: Raw CoNLL-U formats might be "recognized" by LLMs due to exposure in training data, but their semantics are not necessarily understood. Verbalization transforms structural information into a natural language form that LLMs can utilize more easily.

### Loss & Training

A pure ICL setting with no training is used. The study evaluates three models: Gemma-12B/27B and GPT-4.1. Evaluation is conducted on the standard dev (380 sentences) and test (405 sentences) sets of the UD treebank.

## Key Experimental Results

### Main Results

**Test Set BERTScore F1**

| Setup | Gemma-12B | Gemma-27B | GPT-4.1 |
|-------|-----------|-----------|---------|
| Baseline | 0.8363 | 0.8385 | 0.9012 |
| LEX | 0.8551 (+0.019) | 0.8565 (+0.018) | 0.9152 (+0.014) |
| LEX+SYN | **0.8707 (+0.034)** | **0.8746 (+0.036)** | **0.9195 (+0.018)** |

### Ablation Study

| Setup | BERTScore Δ (Gemma-27B test) | Description |
|-------|---------------------------|-------------|
| Baseline | 0.0000 | No augmentation |
| DEP alone | +0.0033 | Dependency verbalization only; slight improvement |
| CON alone | +0.0133 | Construction rules are effective individually |
| CoNLLU alone | +0.0162 | Raw parse output works well |
| LEX alone | +0.0181 | Dictionary is the most effective single component |
| LEX+SYN | +0.0361 | Combined effect far exceeds single components |

### Key Findings

- Syntactic information alone is less effective than the dictionary, but combined settings provide significant gains beyond the dictionary (LEX+SYN improves 0.018 over LEX alone on Gemma-27B).
- The combined effects are additive—the individual gains of LEX and SYN nearly sum up to the total gain of LEX+SYN.
- The gap between automatic parsing and gold standards is minimal, suggesting that syntactic augmentation is robust to parser errors.
- Greater gains are observed on non-biblical texts, indicating that syntactic information is more valuable when the model cannot rely on memorized scripture.

## Highlights & Insights

- The use of UD syntactic information for ICL translation is a first—demonstrating that grammatical knowledge can be injected into prompts as "another form of reference material."
- The design of the CON component reflects an "error-oriented" engineering philosophy—providing targeted guidance for actual model errors rather than generalized grammatical information.
- This method has clear value in practical LRL translation scenarios by reducing the workload required for expert post-editing.

## Limitations & Future Work

- Limited to the single direction of Coptic-to-English.
- Sentential translation limits the evaluation of discourse-level phenomena.
- The CON component requires linguistic experts for error analysis.
- The effect of leveraging parallel few-shot examples was not explored.

## Related Work & Insights

- **vs Dictionary-only ICL**: LEX serves as the baseline; this paper proves that syntactic augmentation provides complementary info that dictionaries cannot cover.
- **vs Grammar excerpt approaches**: Rules extracted from grammar books are generic, whereas CON is customized for model weaknesses.
- **vs Fine-tuning approaches**: Fine-tuning requires large parallel data and remains suboptimal; ICL+augmentation is more flexible.

## Rating

- Novelty: ⭐⭐⭐⭐ First use of UD syntax for ICL translation; clever CON component design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model + multi-component ablation + gold vs. silver parsing + biblical vs. non-biblical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure and sufficient linguistic background.
- Value: ⭐⭐⭐⭐ Significant contribution to low-resource translation research; transferable method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](../../AAAI2026/multilingual_mt/gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)
- [\[NeurIPS 2025\] How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](../../NeurIPS2025/multilingual_mt/how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)
- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)

</div>

<!-- RELATED:END -->
