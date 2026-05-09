---
title: >-
  [Paper Note] Exploring the Translation Mechanism of Large Language Models
description: >-
  [NeurIPS 2025][translation mechanism] This paper proposes a subspace-intervened path patching method for fine-grained causal analysis of the translation mechanism in LLMs. The study finds that translation is driven by a sparse set of attention heads comprising fewer than 5% of all heads, categorized into three functional roles: source heads, indicator heads, and positional heads. MLP layers integrate these features into an English-centric intermediate representation, and fine-tuning only 64 critical heads achieves performance comparable to full-parameter fine-tuning.
tags:
  - NeurIPS 2025
  - translation mechanism
  - mechanistic interpretability
  - attention head
  - path patching
  - subspace intervention
date: 2026-05-08
content_hash: b36067dc32257541
---

# Exploring the Translation Mechanism of Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.11806](https://arxiv.org/abs/2502.11806)
**Code**: Available (link provided in paper)
**Area**: Multilingual Translation
**Keywords**: translation mechanism, mechanistic interpretability, attention head, path patching, subspace intervention

## TL;DR
This paper proposes a subspace-intervened path patching method for fine-grained causal analysis of the translation mechanism in LLMs. The study finds that translation is driven by a sparse set of attention heads comprising fewer than 5% of all heads, categorized into three functional roles: source heads, indicator heads, and positional heads. MLP layers integrate these features into an English-centric intermediate representation, and fine-tuning only 64 critical heads achieves performance comparable to full-parameter fine-tuning.

## Background & Motivation
**State of the Field**: LLMs exhibit strong multilingual translation capabilities, yet the internal core translation mechanism—even for basic word-level translation—remains poorly understood. Prior analyses have largely remained at the level of surface observations (neuron activation patterns, intermediate representation visualization) rather than revealing causal computational mechanisms.

**Limitations of Prior Work**: (1) Conventional path patching intervenes on entire activation vectors, resulting in coarse granularity and noisy estimates; (2) systematic study of the translation mechanism in decoder-only LLMs is lacking (prior work focused on encoder-decoder architectures); (3) it is unclear which attention heads serve which functions or how MLP layers participate in translation.

**Root Cause**: There is a need to precisely localize translation-relevant causal effects within the high-dimensional activation space of LLMs while filtering out activation dimensions unrelated to translation.

**Paper Goals**: Systematically answer three questions: which components are critical for translation? What behavioral patterns do these components exhibit? Can fine-tuning these components improve translation performance?

**Starting Point**: The linear representation hypothesis—linear subspaces of activation vectors constitute the most interpretable model components. A "translation-oriented subspace" is extracted by contrasting positive/negative data pairs (with/without translation logic), and interventions are applied exclusively within this subspace.

**Core Idea**: Perform path patching within the translation-oriented subspace to precisely localize critical translation components, uncovering a sparse translation circuit consisting of three functionally differentiated attention head types and English-centric MLP processing.

## Method

### Overall Architecture
A three-stage systematic framework: (1) identify critical translation components via subspace-intervened path patching; (2) analyze the functional roles of critical attention heads and the representational properties of MLP layers; (3) design and validate a targeted fine-tuning strategy based on the findings.

### Key Designs

1. **Subspace-Intervened Path Patching**:

    - **Function**: Performs causal intervention within the "translation-oriented subspace" of component activations rather than on the entire activation vector.
    - **Mechanism**: Contrastive data pairs (with translation logic $X_+$ vs. without translation logic $X_-$) are used to compute an activation difference matrix $\mathbf{M}_c$. Orthogonal decomposition (optimization objective Eq. 1) partitions it into a general translation-oriented subspace $\mathbf{S}_c$ and a dataset-specific subspace $\mathbf{E}_c$. Intervention replaces only the component along the $\mathbf{S}_c$ direction: $\tilde{\mathbf{a}}_c = W_cW_c^T\mathbf{a}_c(X_-) + (I-W_cW_c^T)\mathbf{a}_c(X_+)$
    - **Design Motivation**: Standard path patching, which replaces the entire activation vector, introduces interference from dimensions unrelated to translation; subspace projection precisely isolates the translation signal.

2. **Three Functionally Differentiated Attention Head Types**:

    - **Function**: Critical heads are categorized into source heads (attending to source-language tokens), indicator heads (attending to translation instruction tokens such as "Chinese:"), and positional heads (maintaining sequential position information) based on attention weight analysis.
    - **Mechanism**: The attention distribution of each critical head is analyzed for its alignment with source-language words, instruction tokens, and positional indices.
    - **Design Motivation**: To understand *why* these heads are important—each extracts a different type of information required for the translation task.

3. **English-Centric MLP Processing**:

    - **Function**: Demonstrates that MLP layers integrate multilingual features extracted by attention heads into an English-centric intermediate representation.
    - **Mechanism**: The correlation between hidden representations at each MLP layer and the token embeddings of English, source-language, and target-language tokens is measured; intermediate MLP representations are found to be highly correlated with English embeddings.
    - **Design Motivation**: Provides causal-level confirmation of the previously proposed hypothesis that LLMs use English as an implicit computational pivot.

### Validation: Targeted Fine-Tuning
Fine-tuning only the 64 identified critical attention heads (<5% of parameters) matches or exceeds full-parameter fine-tuning performance on both word-level and sentence-level translation.

## Key Experimental Results

### Critical Component Statistics (LLaMA2-7B)

| Metric | Value |
|--------|-------|
| Proportion of translation-critical attention heads | <5% |
| Head overlap rate across language pairs (same source/target) | >70% |
| Head overlap rate across bidirectional translation pairs | >60% |
| Layer range of critical head concentration | Layers 12–20 + last 2 layers |

### Targeted Fine-Tuning vs. Full-Parameter Fine-Tuning

| Configuration | Parameters | Translation Performance |
|---------------|-----------|------------------------|
| Full-parameter fine-tuning | 100% | Baseline |
| Fine-tune 64 critical heads only | <5% | **Comparable or better** |
| Fine-tune top-5 shared heads | — | Word-level −39% logits |

### Knockout Validation

| Operation | Change in Translation Accuracy |
|-----------|-------------------------------|
| Incrementally knock out critical heads | Significant drop (90% → <30%) |
| Incrementally knock out random heads | Fluctuation <2% |
| Knock out critical MLPs | Similarly significant drop |

### Key Findings
- Translation-critical heads exhibit strong cross-lingual transferability—different translation directions share a large proportion of critical heads.
- The translation criticality of MLP layers is concentrated in layers 15 and beyond; the final MLP layer accounts for up to 50% of the change in target token logits.
- Low-resource languages (Swahili, Bengali, Arabic) exhibit the same sparsity and transferability patterns.

## Highlights & Insights
- **Subspace-projected path patching**: By intervening within a translation-specific subspace, this approach substantially improves the precision and interpretability of causal analysis, and is generalizable to mechanistic analysis of other tasks.
- **Functional differentiation among three head types**: The translation process is clearly decomposed into three sub-functions: extracting source-language content, recognizing translation task signals, and maintaining positional information.
- **<5% parameter fine-tuning experiment**: Mechanistic understanding is directly translated into a practical parameter-efficient fine-tuning strategy.

## Limitations & Future Work
- **Analysis is primarily conducted at the word level**: Although sentence-level transfer is validated, the core analysis remains in a simplified word-level setting.
- **Only LLaMA2-7B is analyzed**: The translation mechanism in larger models (70B+) may differ.
- **Counterfactual template design**: The negative sample construction (replacing translation instruction tokens) may not fully isolate the translation signal.

## Related Work & Insights
- **vs. Voita et al. (2019) encoder-decoder head pruning**: This paper performs analogous analysis on decoder-only LLMs for the first time, finding that the conclusions regarding sparsity and functional differentiation are consistent across architectures.
- **vs. Wendler et al. (2024) English pivot hypothesis**: The English-centric processing hypothesis is confirmed at the causal level (rather than merely at the correlational level).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Subspace path patching and the three-category head taxonomy are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six translation directions + low-resource languages + sentence-level validation + mathematical reasoning transfer + knockout experiments + fine-tuning.
- **Writing Quality**: ⭐⭐⭐⭐ Analysis proceeds in a well-structured, layered manner with clear figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Represents a significant advance in understanding the LLM translation mechanism; the targeted fine-tuning strategy has direct practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[NeurIPS 2025\] ParallelPrompt: Extracting Parallelism from Large Language Model Queries](parallelprompt_extracting_parallelism_from_large_language_model_queries.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[NeurIPS 2025\] Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](reflective_translation_improving_low-resource_machine_translation_via_structured.md)

<!-- RELATED:END -->
