---
title: >-
  [Paper Note] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding
description: >-
  [ICLR 2026][Interpretability][Transformer Interpretability] Through leading-term approximation analysis of training gradients, this study derives closed-form expressions for Transformer weights during early training. These weights are decomposable into a simple combination of three basis functions (bigram, token-interchangeability, context mapping), revealing how Transformers learn semantic associations like "bird"↔"flew" from natural language data. Theoretical predictions al…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Transformer Interpretability"
  - "Training Dynamics"
  - "Gradient Leading Terms"
  - "Semantic Association"
  - "Closed-form Weight Expression"
date: 2026-05-08
content_hash: 02b401ebb18ae3d2
---

# How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding

**Conference**: ICLR 2026  
**arXiv**: [2601.19208](https://arxiv.org/abs/2601.19208)  
**Code**: None  
**Area**: LLM / NLP (Mechanistic Interpretability)  
**Keywords**: Transformer Interpretability, Training Dynamics, Gradient Leading Terms, Semantic Association, Closed-form Weight Expression

## TL;DR
Through leading-term approximation analysis of training gradients, this study derives closed-form expressions for Transformer weights during early training. These weights are decomposable into a simple combination of three basis functions (bigram, token-interchangeability, context mapping), revealing how Transformers learn semantic associations like "bird"↔"flew" from natural language data. Theoretical predictions align closely with weights learned in real LLMs.

## Background & Motivation

**Background**: Semantic association between tokens (e.g., "bird" and "flew") is fundamental to language modeling, enabling models to transcend rote memorization for generalization and coherent generation. Understanding how these associations are learned and represented is key to linking deep learning with linguistic theory and establishing a mechanistic foundation for LLMs.

**Limitations of Prior Work**: Existing Transformer interpretability research generally follows two paths: (1) analyzing representations of trained models (e.g., attention visualization, probing); (2) theoretical analysis on simplified models or synthetic tasks (e.g., single-head attention, synthetic languages, removed positional encodings or residual connections). These either focus on static post-training results without explaining the "how," or rely on assumptions distant from real-world training, making conclusions difficult to generalize.

**Key Challenge**: There is a lack of mechanistic understanding regarding how Transformers learn semantic associations from **real natural language data using standard training procedures**—a process that occurs during training and is often bypassed by static analysis and oversimplified theories.

**Key Insight**: By taking a **training dynamics** perspective on standard architectures (including relative positional encoding, causal masking, and residual streams), this work utilizes **leading-term approximation** of gradients to derive analytical, verifiable closed-form weight expressions characterizing how weights form in early training.

**Core Idea**: Each set of Transformer weights in early training can be represented as a simple combination of three basis functions reflecting corpus statistics, with each basis function corresponding to a specific learning mechanism for semantic associations.

## Method

### Overall Architecture

This work focuses on the **training process itself** rather than training new models or dissecting trained networks. It tracks how each weight matrix ($W_Q$, $W_K$, $W_V$, embedding, etc.) is shaped by gradients during auto-regressive training of a standard Transformer on natural language corpora. The core tool is a leading-term approximation of gradient updates, which expresses weights in the early training phase as a closed-form combination of three statistical basis functions.

### Key Designs

**1. Gradient Leading-Term Approximation: Transforming Intractable Dynamics into Analytical Statistics**

Directly tracking full gradient updates is intractable due to high-order coupling between weights. This approach performs an expansion of the gradient and retains only the leading terms that contribute most to weight updates. This approximation is valid in early training when weights are small and high-order terms (powers of weights) are negligible. consequently, gradients depend primarily on corpus co-occurrence statistics rather than initialization details. Since core capabilities like induction heads and linear semantic relations form early and persist, analyzing this stage is both crucial and tractable.

**2. Three Basis Function Decomposition: Decoding Semantic Associations into Three Statistical Pathways**

The leading-term analysis concludes that weights can be expressed as a combination of three basis functions: **Bigram mapping** captures adjacent token dependencies (e.g., "the" → "cat"). **Token-interchangeability mapping** captures functional similarity; words like "car" and "truck" appear in similar contexts and play similar syntactic roles, thus receiving similar representations—embodying the distributional semantics hypothesis. **Context mapping** encodes long-range prefix-suffix conditional statistics. Semantic associations like "bird"↔"flew" emerge from the superposition of these three pathways.

**3. Closed-form Weight Expression and Functional Division: Verifying Theory with Real Weights**

By substituting these basis functions, weight matrices (Output, Value, Query–Key) in early training are expressed as linear combinations with coefficients determined by architecture (layers, heads) and hyperparameters. This reveals a functional division: $W_{QK}$ weights are dominated by bigram and interchangeability mappings to determine "where to attend," while $W_V$ weights are dominated by context mapping to determine "what information to pass." This quantification allows direct comparison between theoretical predictions and weights of real LLMs.

## Key Experimental Results

### Theory vs. Real Weight Comparison

| Validation Dimension | Result | Description |
|---------|------|------|
| Weight Approximation Accuracy | High Alignment | Pattern of theoretical closed-form expressions matches actual weights. |
| Verification on Real LLMs | Successful | Theoretical predictions verified on actual large-scale LLMs, not just small models. |
| Qualitative Analysis | Interpretable | Closed-form expressions explain specific learned semantic association patterns. |

### Ablation Study

| Analysis Dimension | Key Findings | Description |
|---------|---------|------|
| Early vs. Late Training | Early approximation is more precise | Consistent with theoretical expectations of leading-term approximation. |
| Contribution of Bases | All three contribute significantly | Removing any basis function significantly degrades approximation quality. |
| Weight Matrix Differences | Functional Differentiation | Q/K weights rely more on bigram; V weights rely more on context. |
| Behavior Across Layers | Inter-layer Variation | Shallow layers favor bigram; deeper layers favor context. |

### Key Findings
- **Transformer Weight Decomposability**: Complex weight matrices can be decomposed into combinations of just 3 simple basis functions based on corpus statistics.
- **Emergence of Semantic Association**: Associations emerge through the interaction of bigram statistics, distributional interchangeability, and context patterns.
- **Functional Differentiation**: $W_{QK}$ and $W_V$ matrices play essentially different roles in encoding associations, providing theoretical support for intuitive divisions.
- **Utility of Theoretical Predictions**: Verification on real LLMs proves the expressions are practical tools for explanation beyond purely theoretical interest.

## Highlights & Insights
- **Standardizing Training Dynamics**: Shifts the focus from "what a model does" to "how a model learns it" through a fundamental analytical lens.
- **Simplicity of Three-Basis Decomposition**: Reduces high-dimensional complexity to three statistical basis functions, bridging deep learning and computational linguistics.
- **Theoretical Evidence for Distributional Semantics**: Naturally derives the linguistic hypothesis that "words in similar contexts have similar meanings" from purely mathematical analysis of training.
- **Quantitative over Qualitative**: Provides systematic quantitative validation on real-world models rather than just qualitative descriptions.

## Limitations & Future Work
- Leading-term approximation accuracy declines as weights grow in later training stages.
- Analysis focuses on early training; non-linear effects and feature complexity in mid-to-late stages are not fully captured.
- Scalability of the three-basis decomposition to models with parameters in the tens of billions requires further validation.
- Focus is limited to semantic association; other capabilities like reasoning or planning need similar frameworks.
- Potential applications in guided initialization, architecture design, or knowledge editing remain unexplored.

## Related Work & Insights
- **Mechanistic Interpretability** (Olah et al., 2020): Circuit-level analysis of Transformers.
- **Induction Heads** (Olsson et al., 2022): Specific computational patterns at the attention head level.
- **Distributional Semantics** (Harris, 1954): This work provides theoretical support from a neural training perspective for classic linguistic hypotheses.
- **Feature Learning Theory**: Connects to recent work regarding the lazy training vs. feature learning regimes.
- **Insight**: Leading-term gradient analysis is a powerful, underutilized tool that correlates training dynamics directly to corpus statistics, providing a framework for how data shapes models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Gradient leading-term analysis + 3-basis decomposition = New theoretical framework)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Verified on real LLMs, though scalability to largest models is pending)
- Writing Quality: ⭐⭐⭐⭐ (High theoretical depth and rigorous derivation)
- Value: ⭐⭐⭐⭐⭐ (Significant contribution to mechanistic understanding of Transformers)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How Do Transformers Learn Implicit Reasoning?](../../NeurIPS2025/interpretability/how_do_transformers_learn_implicit_reasoning.md)
- [\[ICLR 2026\] How Transformers Learn Causal Structures In-Context: Explainable Mechanism Meets Theoretical Guarantee](how_transformers_learn_causal_structures_in-context_explainable_mechanism_meets_.md)
- [\[ICLR 2026\] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning](from_tokens_to_thoughts_how_llms_and_humans_trade_compression_for_meaning.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept_a.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)

</div>

<!-- RELATED:END -->
