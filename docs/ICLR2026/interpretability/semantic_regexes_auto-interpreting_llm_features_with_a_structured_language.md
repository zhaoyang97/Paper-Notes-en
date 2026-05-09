---
title: >-
  [Paper Note] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language
description: >-
  [ICLR 2026][mechanistic interpretability] This paper proposes *semantic regexes*—a structured language for automatically describing LLM features—using three primitives (symbol/lexeme/field) and three modifier types (context/composition/quantification). The approach achieves accuracy on par with natural language descriptions while producing more concise and consistent feature descriptions, and enables quantitative analysis of how feature complexity evolves across layers.
tags:
  - ICLR 2026
  - mechanistic interpretability
  - feature description
  - structured language
  - sparse autoencoder
  - automated interpretability
date: 2026-05-08
content_hash: f1bda795ab8323eb
---

# Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language

**Conference**: ICLR 2026
**arXiv**: [2510.06378](https://arxiv.org/abs/2510.06378)
**Code**: [https://github.com/apple/ml-semantic-regex](https://github.com/apple/ml-semantic-regex)
**Area**: Interpretability
**Keywords**: mechanistic interpretability, feature description, structured language, sparse autoencoder, automated interpretability

## TL;DR
This paper proposes *semantic regexes*—a structured language for automatically describing LLM features—using three primitives (symbol/lexeme/field) and three modifier types (context/composition/quantification). The approach achieves accuracy on par with natural language descriptions while producing more concise and consistent feature descriptions, and enables quantitative analysis of how feature complexity evolves across layers.

## Background & Motivation
Automated interpretability aims to translate internal LLM features into human-understandable descriptions. Existing methods (e.g., Bills et al. 2023; Paulo et al. 2024) rely on natural language, which suffers from three core limitations:

**Ambiguity**: Natural language descriptions tend to be verbose or vague, with significant variation across annotators for the same feature.

**Inconsistency**: Functionally equivalent features may receive entirely different natural language descriptions, hindering redundant feature detection and circuit analysis.

**Lack of structure**: Natural language cannot directly encode feature complexity, making model-level systematic analysis difficult.

The root cause lies in the tension between natural language's flexibility—sufficient for describing individual features—and its inherent ambiguity, which obstructs large-scale, systematic feature analysis. Inspired by regular expressions and programming languages, the authors approach the problem by designing a structured language that is both **precise** and **expressive**, capable of accurately characterizing feature behavior while providing a structured interface for model-level analysis.

## Method

### Overall Architecture
Semantic regexes are embedded within a standard automated interpretability pipeline: given a subject model's features and their activation data, an explainer model (GPT-4o-mini) generates semantic regex descriptions, which are then scored by an evaluator model. A key design choice is to decouple **description format** from **generation procedure**—replacing the description language requires only updating the language specification in the prompt.

### Key Designs
1. **Three-level Primitives**: Corresponding to three levels of feature abstraction:

    - `[:symbol X:]`: matches exact strings, e.g., `[:symbol color:]` matches only "color"
    - `[:lexeme X:]`: matches morphological variants, e.g., `[:lexeme color:]` matches "color/colors/coloring"
    - `[:field X:]`: matches semantically related words, e.g., `[:field color:]` matches "red/blue/green"

   The three levels progress from precise to abstract, reflecting the continuum from low-level token detection to high-level semantic concept encoding in LLM features.

2. **Three Modifier Types**: Extending the expressive power of primitives:

    - **Context**: `@{:context X:}(semantic regex)` constrains the semantic context, e.g., `@{:context politics:}([:symbol color:])` matches "color" only in political contexts
    - **Composition**: supports sequential composition and alternation `|`, e.g., `[:field color:]([:symbol and:]|[:symbol or:])[:field color:]`
    - **Quantification**: uses the regex quantifier `?` to indicate optionality, e.g., `[:symbol a:][:field color:]?[:field flower:]`

3. **Language Design Methodology**: A grounded-theory approach is adopted, iteratively introducing new primitives and modifiers through manual inspection of thousands of features on Neuronpedia, until saturation is reached—i.e., all observed feature patterns can be described.

### Loss & Training
- Subject models: GPT-2-Small and Gemma-2-2B, with features extracted from residual layers via SAEs
- Explainer/Evaluator: GPT-4o-mini
- Semantic regex generation prompts are adapted from the max-acts approach: instructions are updated with semantic regex syntax, grammar definitions are added, and few-shot examples are revised
- Top-10 activating examples are presented; the model is instructed to first produce a brief natural language explanation, then output the semantic regex

## Key Experimental Results

### Main Results (Accuracy Comparison)
100 features per layer are evaluated on GPT-2-RES-25k, Gemma-2-2B-RES-16k, and Gemma-2-2B-RES-65k:

| Method | Clarity (Gen.) | Detection (Disc.) | Fuzzing (Disc.) | Responsiveness (Disc.) | Faithfulness |
|------|:-:|:-:|:-:|:-:|:-:|
| token-act-pair | baseline | baseline | baseline | baseline | baseline |
| max-acts | mid | mid | mid | mid | mid |
| **semantic-regex** | **≥ max-acts** | **≥ token-act-pair** | **≥ token-act-pair** | **≥ max-acts** | on par |

- Semantic regex significantly outperforms token-act-pair on clarity across all models (p<0.05)
- Significantly outperforms token-act-pair on detection/fuzzing/responsiveness for GPT-2 and Gemma-65k
- Non-inferiority tests confirm no significant accuracy gap between semantic regex and natural language

### Ablation Study (Conciseness and Consistency)

| Metric | semantic-regex | max-acts | token-act-pair |
|------|:-:|:-:|:-:|
| Median description length (chars) | **41** (IQR: 19–59) | 139 (IQR: 119–166) | 55 (IQR: 46–66) |
| Consistency (fraction of identical descriptions) | **33.6%** | 0.0% | 12.2% |

- Semantic regex is **3.4×** shorter than max-acts and **1.3×** shorter than token-act-pair
- Consistency is **2.8×** higher than token-act-pair

### Key Findings
1. **Feature complexity increases with layer depth**: Early layers are dominated by simple symbol primitives; later layers require more compositional and field-level primitives, with average component count increasing across layers. Symbol usage decreases while field usage increases with depth.
2. **User study (N=24)**: Semantic regex helped users build more accurate mental models for 9 out of 12 features (larger gap between positive activation examples and counter-examples).
3. Users required minimal instruction to understand semantic regex, while natural language descriptions prompted more clarification requests.
4. Extra detail in natural language descriptions frequently misled users, whereas the conciseness of semantic regex reduced cognitive load.

## Highlights & Insights
- **Structure does not sacrifice expressivity**: Constraining the language reduces noise and improves usability.
- **From individual features to model-level analysis**: The distribution of primitive types serves as a proxy for layer-wise complexity, requiring no additional probes or tests.
- **An elegant analogy with regular expressions**: Just as regular expressions describe character patterns, semantic regexes describe semantic patterns, naturally bridging symbolic systems and neural network representations.
- **Engineering modularity**: Integration into existing pipelines requires only prompt modification, ensuring compatibility with future methods.

## Limitations & Future Work
- Overly concise descriptions may introduce ambiguity (e.g., does `[:field musician:]` match "guitarist" or only "renowned musicians"?).
- The mapping is non-unique: a single feature may admit multiple valid semantic regexes, and no canonical "style guide" exists.
- Certain components are underspecified (e.g., case sensitivity), potentially causing inconsistent model behavior.
- Support for polysemantic features is limited; highly entangled concepts still yield incoherent descriptions.
- Models must learn the new language from very few examples, occasionally producing syntactic errors.

## Related Work & Insights
- Orthogonally complementary to feature extraction methods such as SAEs and transcoders: semantic regex handles description while SAEs handle discovery.
- Deeply integrated with the Neuronpedia platform, enabling direct interactive feature exploration.
- Raises the question of whether **domain-specific** structured languages could be designed (e.g., for safety features, attention heads, or multimodal features).
- Draws an analogy to the history of programming languages: different interpretability tasks may call for different "interpretability programming languages."
- Directly supports circuit analysis (circuit tracing): consistent descriptions make automated identification of redundant features feasible.
- Complementary to the output-centric approach of Gur-Arieh et al. (2025): semantic regex focuses on activation patterns, whereas the latter focuses on output effects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to introduce a structured language into automated interpretability; conceptually original with a rigorous methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-metric evaluation with a user study, though limited to GPT-2 and Gemma-2-2B.
- Writing Quality: ⭐⭐⭐⭐⭐ Polished visualizations, clear argumentation, and well-chosen analogies.
- Value: ⭐⭐⭐⭐ Provides new tools and analytical dimensions for mechanistic interpretability, though broader adoption will require ecosystem support.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)
- [\[ICLR 2026\] A Cortically Inspired Architecture for Modular Perceptual AI](a_cortically_inspired_architecture_for_modular_perceptual_ai.md)
- [\[ICLR 2026\] Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)

<!-- RELATED:END -->
