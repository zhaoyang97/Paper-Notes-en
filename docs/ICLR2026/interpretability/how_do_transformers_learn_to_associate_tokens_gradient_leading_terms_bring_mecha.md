---
title: >-
  [Paper Note] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding
description: >-
  [ICLR 2026][Interpretability][Transformer interpretability] By analyzing the leading terms of training gradients, this paper derives closed-form expressions for each Transformer weight matrix during the early training ph…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Transformer interpretability"
  - "training dynamics"
  - "gradient leading terms"
  - "semantic association"
  - "closed-form weight expressions"
date: 2026-05-08
content_hash: dd47c6fd854815f9
---

# How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding

**Conference**: ICLR 2026
**arXiv**: [2601.19208](https://arxiv.org/abs/2601.19208)
**Code**: None
**Area**: LLM / NLP (Mechanistic Interpretability)
**Keywords**: Transformer interpretability, training dynamics, gradient leading terms, semantic association, closed-form weight expressions

## TL;DR
By analyzing the leading terms of training gradients, this paper derives closed-form expressions for each Transformer weight matrix during the early training phase. Each matrix decomposes into a simple combination of three basis functions (bigram, token-interchangeability, and context mapping), revealing how Transformers learn semantic associations such as "bird"↔"flew" from natural language data. The theoretical predictions align closely with the weights learned by real LLMs.

## Background & Motivation

### State of the Field

**Background**: Semantic associations between tokens such as "bird" and "flew" are fundamental to language modeling—models must generalize beyond memorization to produce coherent text. Understanding how these associations are learned and represented in language models is critical for bridging deep learning with linguistic theory and establishing a mechanistic foundation for large language models.

**Current Issues**:

### Limitations of Prior Work

**Limitations of Prior Work**: Existing Transformer interpretability research follows two main directions: (1) analyzing internal representations of trained models (e.g., attention visualization, probing); and (2) theoretical analysis of simplified models or tasks (e.g., single-head attention on synthetic data).

### Root Cause

**Key Challenge**: A critical gap exists in the mechanistic understanding of how Transformers progressively learn semantic associations **during training** from real natural language data.

### Resolution

**Resolution**: Existing theoretical analyses based on simplifying assumptions (e.g., linear attention, single-layer models) cannot directly explain multi-layer, multi-head Transformers trained on real data.

**Key Insight**: This paper adopts a **training dynamics** perspective, using **leading-term approximations** of gradients to derive analytically tractable and empirically verifiable closed-form weight expressions. Rather than assuming a simplified architecture, the approach characterizes the formation of model weights during early training by mathematically identifying the dominant contributions to gradient updates.

**Core Idea**: Each group of Transformer weights can be expressed as a simple combination of three basis functions that reflect statistical properties of the text corpus, with each basis function corresponding to a distinct mechanism for learning semantic associations.

## Method

### Overall Architecture
- **Input/Object**: Attention-based language models (Transformers) trained with standard autoregressive or MLM objectives on natural language corpora.
- **Analysis Tool**: Leading-term analysis of the Taylor expansion of per-step gradient updates.
- **Output**: Closed-form expressions for each Transformer weight matrix ($W_Q$, $W_K$, $W_V$, embeddings, etc.) during the early training phase.

### Key Designs

1. **Gradient Leading-Term Approximation**:

    - Gradients are decomposed to identify the dominant terms contributing most to weight updates.
    - After discarding higher-order terms, the leading terms can be expressed in terms of corpus statistics.
    - This approximation is particularly accurate during early training, when weights remain small.

2. **Three Basis Functions**:
   The derivation shows that each group of Transformer weights can be expressed as a combination of the following three basis functions:

    - **Bigram Mapping**: Captures co-occurrence statistics of adjacent token pairs. For example, the high-frequency co-occurrence of "the"→"cat" leads to an increase in the corresponding weights. This is the most direct form of sequence statistics—when token $A$ frequently precedes token $B$, the model learns the association $A \rightarrow B$.

    - **Token-Interchangeability Mapping**: Captures "interchangeability"—relationships between tokens that appear in similar contexts. For example, "dog" and "cat" are not adjacent co-occurring tokens, but they are interchangeable in contexts such as "The ___ sat on the mat." This reflects the distributional semantics hypothesis.

    - **Context Mapping**: Captures broader contextual patterns—how a specific context influences predictions of subsequent tokens. This basis function encodes the statistical regularity of "given a context window, what token is most likely to follow."

3. **Closed-Form Weight Expressions**:

    - Each weight matrix ($W_Q$, $W_K$, $W_V$, etc.) can be written during early training as a linear combination of the three basis functions above.
    - The combination coefficients depend on architectural details (number of layers, heads) and training hyperparameters.
    - These closed-form expressions reveal the **functional specialization** of Transformer components:
        - Query-Key weights rely primarily on bigram and token-interchangeability mappings → determine "what to attend to."
        - Value weights rely primarily on context mapping → determine "what information to propagate."

### Levels of Theoretical Contribution
1. **Descriptive**: Provides a mathematical expression for the weights.
2. **Explanatory**: Explains how each component captures semantic associations based on corpus statistics.
3. **Predictive**: Enables quantitative comparison between theoretical expressions and weights learned by real LLMs.

## Key Experimental Results

### Theory vs. Real Weights

| Validation Dimension | Result | Notes |
|----------------------|--------|-------|
| Weight approximation accuracy | Strong alignment | Theoretical closed-form expressions match patterns of actual trained weights closely |
| Validation on real LLMs | Successful | Theoretical predictions verified not only on small models but also on practical-scale LLMs |
| Qualitative analysis | Interpretable | Closed-form expressions explain specific semantic association patterns learned by the model |

### Ablation Study

| Analysis Dimension | Key Finding | Notes |
|--------------------|-------------|-------|
| Early vs. late training | Early-phase approximation is more accurate | Consistent with theoretical expectations of the leading-term approximation |
| Contribution of each basis function | All three contribute significantly | Removing any single basis function substantially degrades approximation quality |
| Different weight matrices | Functional differentiation | Q/K weights depend more on bigram; V weights depend more on context mapping |
| Behavior across layers | Layer-wise variation | Shallower layers favor bigram; deeper layers favor context mapping |

### Key Findings
- **Decomposability of Transformer weights**: Complex weight matrices can be decomposed into combinations of only three simple basis functions derived from corpus statistics, substantially simplifying the understanding of Transformer learning mechanisms.
- **Emergence mechanism of semantic associations**: Semantic associations are not directly encoded but emerge through the interplay of bigram co-occurrence statistics, distributional interchangeability, and contextual patterns.
- **Functional specialization**: Q/K matrices and V matrices play fundamentally different roles in encoding semantic associations—an intuitive result now supported by theoretical grounding for the first time.
- **Practical utility of theoretical predictions**: The closed-form expressions are validated on real LLMs, demonstrating that these results are not merely theoretically elegant but carry genuine explanatory power.

## Highlights & Insights
- **Understanding Transformers from a training dynamics perspective**: Unlike most interpretability work that analyzes "what a trained model does," this paper analyzes "how the model learned to do it"—providing a more fundamental understanding.
- **Elegance of the three-basis-function decomposition**: Reducing high-dimensional, complex weight matrices to three basis functions with clear statistical semantics is both elegant and interpretable, serving as a bridge between deep learning and computational linguistics.
- **Theoretical evidence for the distributional semantics hypothesis**: The token-interchangeability basis function directly corresponds to the linguistic hypothesis that "words appearing in similar contexts have similar meanings." This paper derives this conclusion naturally from a purely mathematical analysis of training dynamics.
- **Quantitative validation beyond qualitative description**: The work goes beyond theoretical derivation to provide systematic quantitative validation on real LLMs.

## Limitations & Future Work
- The leading-term approximation loses accuracy in later training stages (as weights grow larger), limiting its explanatory power for fully trained models.
- The analysis focuses primarily on the early training phase; nonlinear effects and feature complexification in mid-to-late training are not fully captured.
- Whether the three-basis-function decomposition generalizes to much larger models (billions of parameters) requires further investigation.
- Only semantic association is analyzed; whether other Transformer capabilities (e.g., reasoning, planning) can be studied under a similar framework remains an open question.
- Practical applications of the closed-form expressions (e.g., guiding model initialization, architecture design, or knowledge editing) are not yet fully explored.
- The impact of inter-layer interactions and residual connections in multi-layer Transformers warrants deeper analysis.

## Related Work & Insights
- **Mechanistic Interpretability** (Olah et al., 2020): Circuit-level interpretability analysis of Transformers.
- **Induction Heads** (Olsson et al., 2022): Specific computational patterns at the attention head level.
- **Transformer training dynamics** (Li et al., 2023, etc.): Analysis of dynamic properties of the Transformer training process.
- **Distributional Semantics** (Harris, 1954; Firth, 1957): Word meaning is determined by distributional context—this paper provides theoretical support for this classical hypothesis from the perspective of neural network training dynamics.
- **Feature Learning Theory**: Recent theoretical work on how neural networks learn features (lazy training vs. feature learning regime).
- **Inspiration**: Gradient leading-term analysis is a powerful yet underutilized analytical tool that may be broadly applicable to understanding other neural network architectures and tasks. The methodology of directly linking training dynamics to corpus statistics offers a new analytical framework for understanding how data shapes models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Gradient leading-term analysis + three-basis-function decomposition = entirely new theoretical framework)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated on real LLMs, but extensibility to larger models remains to be shown)
- Writing Quality: ⭐⭐⭐⭐ (High theoretical depth with rigorous mathematical derivations)
- Value: ⭐⭐⭐⭐⭐ (Important theoretical contribution to mechanistic understanding of Transformers, bridging deep learning and linguistics)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [\[NeurIPS 2025\] Base Models Know How to Reason, Thinking Models Learn When](../../NeurIPS2025/interpretability/base_models_know_how_to_reason_thinking_models_learn_when.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](../../NeurIPS2025/interpretability/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[ICLR 2026\] Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance](stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan.md)
- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)

</div>

<!-- RELATED:END -->
