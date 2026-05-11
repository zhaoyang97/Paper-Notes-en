---
title: >-
  [Paper Note] How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective
description: >-
  [AAAI 2026][Multilingual & Machine Translation][multilingual LLM] This paper proposes a ternary neuron classification scheme (language-specific / language-related / universal) and decomposes multilingual LLM inference in…
tags:
  - "AAAI 2026"
  - "Multilingual & Machine Translation"
  - "multilingual LLM"
  - "language neurons"
  - "alignment mechanism"
  - "ternary classification"
  - "spontaneous alignment"
date: 2026-05-08
content_hash: 492b9a7bce11a661
---

# How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective

**Conference**: AAAI 2026
**arXiv**: [2505.21505](https://arxiv.org/abs/2505.21505)
**Code**: [https://github.com/NJUNLP/Language-Neurons-Alignment](https://github.com/NJUNLP/Language-Neurons-Alignment)
**Area**: Multilingual Translation
**Keywords**: multilingual LLM, language neurons, alignment mechanism, ternary classification, spontaneous alignment

## TL;DR
This paper proposes a ternary neuron classification scheme (language-specific / language-related / universal) and decomposes multilingual LLM inference into a four-stage framework. It finds that multilingual alignment improves performance by increasing language-related neurons (while reducing language-specific ones), and further demonstrates a "spontaneous multilingual alignment" effect on untrained languages.

## Background & Motivation

**Background**: LLMs exhibit large disparities in multilingual capability due to imbalanced pretraining corpora. MAPO improves low-resource language performance by aligning non-English capabilities toward English. Tang et al. categorize neurons into "language-specific" and "universal" types.

**Limitations of Prior Work**: The binary classification ignores neurons activated across multiple (but not all) languages — neurons that are neither language-specific nor universal.

**Key Challenge**: Existing taxonomies fail to capture nuanced cross-lingual neuron sharing patterns, leaving the multilingual alignment mechanism incompletely understood.

**Goal**: Which types of neurons does alignment actually enhance? Why does alignment also benefit untrained languages?

**Key Insight**: Introduce "language-related neurons" as a third category and analyze layer-wise changes before and after alignment.

**Core Idea**: A ternary classification scheme combined with a four-stage analysis framework reveals the neuron-level mechanism underlying multilingual alignment.

## Method

### Overall Architecture

Multilingual alignment is performed via MAPO (a DPO variant). The ternary classification scheme is then applied to analyze changes in neuron distribution before and after alignment, within a four-stage functional decomposition of multilingual inference.

### Key Designs

1. **Ternary Neuron Classification**

    - Function: Categorizes activated neurons into language-specific (only 1 language), language-related (2–9 languages), and universal (all 10 languages).
    - Mechanism: $\text{score}_{i,j} = -\sum_k p'^k \log p'^k - \lambda \max_k p^k$; the bottom 1% of neurons by score are selected and subdivided by the number of activating languages $N_{i,j}$.
    - Design Motivation: Jointly considering language specificity (entropy) and activation strength (maximum activation probability) yields more accurate classification than entropy alone.

2. **Four-Stage Multilingual Inference Model**

    - Stage 1 (lower layers): Multilingual understanding — peaks in language-specific and language-related neurons.
    - Stage 2 (middle layers): Reasoning in a shared semantic space — universal neurons dominate.
    - Stage 3 (upper layers): Multilingual output space transformation — language-specific and language-related neurons increase again.
    - Stage 4 (final layers): Vocabulary-space output — universal neurons unexpectedly increase again, correlating with a shared vocabulary.

3. **Neuron Change Analysis Before and After Alignment**

    - After alignment, language-specific neurons decrease and language-related neurons increase; the model learns to reuse cross-lingual shared neurons.
    - Untrained languages exhibit similar change patterns, providing a mechanistic explanation for spontaneous alignment.

### Loss & Training

MAPO-DPO training. Alignment scores are computed using the NLLB-200 translation model; 10,000 preference pairs per target language; LoRA fine-tuning.

## Key Experimental Results

### Main Results (MGSM Multilingual Math Reasoning)

| Setting | bn | th | sw | ja | zh | ru | de | es | fr | en | Avg |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Base | 43.6 | 53.2 | 50.4 | 55.6 | 59.6 | 59.2 | 61.2 | 62.8 | 56.8 | 75.6 | 57.8 |
| zh/de⇒en | 46.4 | 55.6 | 59.2 | 56.8 | 64.0 | 71.2 | 66.8 | 71.2 | 69.2 | 75.2 | 63.6 |
| sw/th⇒en | 48.8 | 58.8 | 59.2 | 56.4 | 68.4 | 68.4 | 69.2 | 69.6 | 70.4 | 77.6 | **64.7** |

### Neuron Changes Under Spontaneous Alignment

| Language Type | Language-Specific Change | Language-Related Change |
|---------------|--------------------------|-------------------------|
| Trained languages | −37 | +232 |
| Untrained languages | −36 | +205 |

### English Distinctiveness

| Language | Language-Specific | Language-Related |
|----------|-------------------|------------------|
| English | 46 | 603 |
| Non-English (mean) | 613 | 2006 |

### Key Findings
- **Alignment is essentially neuron sharing**: language-specific neurons decrease while language-related neurons increase.
- **Spontaneous alignment mechanism**: Untrained languages exhibit similar change patterns (+205 vs. +232 for trained languages); newly added language-related neurons serve unseen languages as well.
- **English distinctiveness**: English has very few language-specific/related neurons (46/603) because its "language-related" neurons are shared with nearly all languages and are thus classified as universal.
- **Four-stage vs. three-stage model**: The increase in universal neurons at the final layer is a new finding not captured by prior three-stage frameworks.

## Highlights & Insights
- **Necessity of ternary classification**: Language-related and language-specific neurons exhibit opposite trends under alignment; merging them into one category would obscure this critical pattern.
- **Elegant explanation of spontaneous alignment**: Language-related neurons generated by training on two languages happen to be reused by other languages as well.
- **English as a hub**: English language-related neurons are shared with so many languages that they are classified as universal, explaining why deactivating English-specific neurons does not impair English performance.

## Limitations & Future Work
- Validation is limited to Mistral/MetaMath variants; additional architectures are needed.
- Analysis covers only mathematical reasoning; patterns for translation and question answering may differ.
- Thresholds (top 5%, 1%) are set empirically.
- Observed correlations do not establish causality.

## Related Work & Insights
- **vs. Tang et al.**: Binary classification cannot capture the increase in language-related neurons induced by alignment.
- **vs. Zhao et al.**: The three-stage model does not distinguish the behavior of universal neurons in the final layer.
- **vs. Zhang et al.**: Their work observes the spontaneous alignment phenomenon; this paper provides a neuron-level mechanistic explanation.
- Inspiration: The ternary classification framework can be applied to neuron analysis of other capabilities, such as detecting "specialized neurons" in code generation and mathematical reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The ternary classification, four-stage framework, and mechanistic explanation of spontaneous alignment together offer multi-level novel contributions to multilingual LLM interpretability.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multilingual, multi-model analysis is comprehensive, but is limited to mathematical reasoning; validation on translation and QA tasks is absent.
- Writing Quality: ⭐⭐⭐⭐ — The analytical framework is systematic, and figures clearly illustrate neuron distribution changes before and after alignment.
- Value: ⭐⭐⭐⭐⭐ — Provides deep insights into multilingual LLM mechanisms; the ternary classification framework is broadly applicable to other multilingual capability analyses.

## Additional Notes
- The methodology and experimental design of this work offer valuable reference for related fields.
- Future work may validate the generalizability and scalability of the approach across more scenarios and larger scales.
- Combining this work with recent related approaches (e.g., RL/MCTS or multimodal methods) presents potential research opportunities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](../../ACL2026/multilingual_mt/language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](../../ACL2026/multilingual_mt/location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](../../ACL2026/multilingual_mt/no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)

</div>

<!-- RELATED:END -->
