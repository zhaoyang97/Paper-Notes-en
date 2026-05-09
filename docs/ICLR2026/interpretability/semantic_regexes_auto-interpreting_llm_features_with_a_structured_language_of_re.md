---
title: >-
  [Paper Note] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language
description: >-
  [ICLR 2026][mechanistic_interpretability] This paper proposes **Semantic Regexes**, a structured language for automatically describing LLM features. By combining primitives (symbol / lexeme / field) with modifiers (context / composition / quantification), it produces feature descriptions that are equally accurate to natural language, yet more concise, consistent, and amenable to programmatic analysis.
tags:
  - ICLR 2026
  - mechanistic_interpretability
  - automated_interpretability
  - sparse_autoencoders
  - structured_language
  - feature_description
date: 2026-05-08
content_hash: 87fa2b924f63a5c1
---

# Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language

**Conference**: ICLR 2026
**arXiv**: [2510.06378](https://arxiv.org/abs/2510.06378)
**Code**: [apple/ml-semantic-regex](https://github.com/apple/ml-semantic-regex)
**Area**: LLM NLP / Mechanistic Interpretability
**Keywords**: mechanistic_interpretability, automated_interpretability, sparse_autoencoders, structured_language, feature_description

## TL;DR

This paper proposes **Semantic Regexes**, a structured language for automatically describing LLM features. By combining primitives (symbol / lexeme / field) with modifiers (context / composition / quantification), it produces feature descriptions that are equally accurate to natural language, yet more concise, consistent, and amenable to programmatic analysis.

## Background & Motivation

**State of the Field**:
- Methods such as sparse autoencoders (SAEs) can extract monosemantic features from LLMs.
- Automated interpretability pipelines use LLMs to translate these features into human-readable descriptions.
- Such descriptions help researchers understand what concepts a model encodes and enable feature circuit tracing.

**Limitations of Prior Work**:
- **Verbosity**: Descriptions are often excessively wordy (e.g., "The presence of the sequence 54 indicating a year, time, or numeric reference frequently associated with events").
- **Inconsistency**: Functionally identical features may receive entirely different descriptions across runs.
- **Ambiguity**: Natural language is inherently ambiguous, which hinders analysis tasks that require compositional reasoning.
- **Manual re-annotation**: Even in recent feature circuit work, researchers must still manually re-label features.

**Root Cause**:
- A well-defined grammar and semantics reduce ambiguity.
- Compositional rules enable precise expressions ranging from simple to complex.
- Uniform expression facilitates comparison and aggregation across features.

## Method

### Overall Architecture

Semantic Regex is a structured language induced from thousands of real LLM features via grounded theory methodology. It is integrated into a standard automated interpretability pipeline (explainer + evaluator) without altering the pipeline architecture.

### Key Designs: Language Specification

**Three Primitives**:

1. **Symbol** `[:symbol X:]` — matches the exact string X
   - Example: `[:symbol color:]` matches the token "color"
   - Describes features that activate on specific tokens

2. **Lexeme** `[:lexeme X:]` — matches syntactic variants of X (tense, plural, etc.)
   - Example: `[:lexeme color:]` matches "color", "colors", "coloring", etc.
   - Describes features that capture word meaning

3. **Field** `[:field X:]` — matches semantic variants of X (words within the same conceptual domain)
   - Example: `[:field color:]` matches "red", "blue", "green", etc.
   - Describes features that activate on conceptual categories

**Three Modifiers**:

1. **Context** `@{:context X:}(regex)` — matches within context X
   - Example: `@{:context politics:}([:symbol color:])` matches "color" only in political contexts

2. **Composition** — sequential concatenation and alternation (|)
   - Example: `[:field color:]([:symbol and:]|[:symbol or:])[:field color:]`

3. **Quantification** — standard regex quantifier `?` (zero or one occurrence)
   - Example: `[:symbol a:][:field color:]?[:field flower:]`

### Automated Interpretability Pipeline

- **Subject models**: GPT-2-Small, Gemma-2-2B
- **Feature source**: SAE-extracted latent features (GPT-2-RES-25k, Gemma-2-2B-RES-16k/65k)
- **Explainer model**: GPT-4o-mini (generates natural language or semantic regex descriptions given activation data)
- **Evaluator model**: GPT-4o-mini (assesses the match between description and feature behavior)

Semantic regexes alter only the description language, not the pipeline architecture. They are incorporated by injecting the semantic regex grammar rules and few-shot examples into the max-acts prompt.

### Evaluation Metrics

- **Generation metric (Clarity)**: Whether the description can generate highly activating examples (analogous to precision)
- **Discrimination metrics (Detection / Fuzzing / Responsiveness / Purity)**: Whether the description matches known activating examples (analogous to recall)
- **Faithfulness metric**: Whether the description reflects the effects of causal interventions

### Loss & Training

This paper does not involve model training; it presents an evaluation framework. Feature extraction uses pre-trained SAEs.

## Key Experimental Results

### Main Results: Accuracy Comparison (100 features per layer)

| Metric | Semantic Regex | max-acts (NL) | token-act-pair (NL) |
|--------|---------------|---------------|---------------------|
| Clarity (GPT-2) | **Significantly better** | Baseline | Lowest |
| Detection (GPT-2) | **Significantly better than tap** | On par with SR | Lowest |
| Fuzzing (GPT-2) | **Significantly better than tap** | On par with SR | Lowest |
| Clarity (Gemma-16k) | Non-inferior | Baseline | Lowest |
| Clarity (Gemma-65k) | **Significantly better than tap** | Baseline | Lowest |

Core finding: **Semantic regexes perform at least on par with natural language across all models, and significantly outperform token-act-pair on multiple metrics**, demonstrating that structural constraints do not reduce descriptive capacity.

### Ablation Study: Conciseness and Consistency

| Measure | Semantic Regex | max-acts | token-act-pair |
|---------|---------------|----------|----------------|
| Median description length (characters) | **41** (IQR: 19–59) | 139 (IQR: 119–166) | 55 (IQR: 46–66) |
| Identical description rate (5 samples) | **33.6%** | 0.0% | 12.2% |

- Semantic regexes are **3.4× shorter** than max-acts descriptions.
- Across repeated sampling of the same feature, semantic regexes yield identical descriptions 33.6% of the time (vs. 0% for max-acts).

### Feature Complexity Analysis

| Layer Position | Mean Component Count | Low-level Primitive Ratio | Field Ratio | Modifier Ratio |
|----------------|---------------------|--------------------------|-------------|----------------|
| Early layers | Low | High (symbol/lexeme dominant) | Low | Low |
| Middle layers | Moderate | Decreasing | Increasing | Increasing |
| Late layers | High | Lowest | Highest | Highest |

The structure of semantic regexes naturally encodes feature complexity: features in later layers require longer, more abstract descriptions. This is consistent with the known phenomenon that later layers encode more complex representations, but here it can be read directly from the feature descriptions for the first time.

### User Study (N = 24)

| Measure | Features where Semantic Regex outperforms NL / 12 |
|---------|--------------------------------------------------|
| Decision boundary understanding (positive–counterfactual activation gap) | **9 / 12** |

- Participants generated better positive and counterfactual examples using semantic regexes for 9 out of 12 features.
- Natural language descriptions often introduced irrelevant details that led to misunderstanding (e.g., "expected to indicates anticipation" caused participants to incorrectly treat unexpected contexts as counterexamples).
- Participants demonstrated better comprehension of semantic regexes than anticipated; more comprehension issues were observed with natural language descriptions.

### Key Findings

1. **Structured ≠ reduced expressiveness**: Semantic regexes match or exceed natural language in accuracy.
2. **3.4× conciseness gain**: Substantially reduces the interpretive burden.
3. **Consistency improves from 0% to 33.6%**: Facilitates redundant feature detection and circuit analysis.
4. **Complexity is readable**: The description format directly reflects a feature's level of abstraction.
5. **Human-friendly**: User study confirms that semantic regexes help users build more accurate mental models of features.

## Highlights & Insights

- **Methodological innovation**: Extends the concept of regular expressions to the semantic domain, achieving both formalism and readability.
- **Grounded theory-driven**: The language design is inductively derived from thousands of real features rather than constructed a priori.
- **Plug-and-play**: Integration into existing interpretability pipelines requires only modifying the description specification in the prompt.
- **Critical capability for scalable analysis**: While natural language descriptions suffice for understanding individual features, the structure of semantic regexes enables macro-level analysis across models and layers.
- Developed by **Apple**, with open-source code and an interactive interface.

## Limitations & Future Work

1. **Non-unique mapping**: A single feature may admit multiple equivalent semantic regex descriptions; a standardized style guide is lacking.
2. **Risk of excessive conciseness**: `[:field musician:]` may lead users to expect "guitarist" to produce strong activations, whereas the actual feature activates only on musician names.
3. **Polysemy unresolved**: Highly polysemous features yield incoherent semantic regex descriptions.
4. **Limits of in-context learning**: LLMs learn the semantic regex grammar from brief descriptions and few-shot examples, resulting in occasional errors.
5. Validation is currently limited to GPT-2 and Gemma-2; applicability to larger models and more complex SAEs remains to be verified.

## Related Work & Insights

- **SAE / Transcoders (Bricken et al., 2023; Dunefsky et al., 2024)**: Extract monosemantic features from LLMs.
- **Bills et al. (2023)**: Pioneers of the automated interpretability pipeline (token-act-pair method).
- **Paulo et al. (2024)**: The max-acts method, which improves the presentation of activation data.
- **Ameisen et al. (2025)**: Feature circuit tracing, which requires manual re-annotation—the consistency of semantic regexes could reduce this need.
- **Jin et al. (2025)**: Hierarchical feature complexity—semantic regexes independently validate this finding from the description side.
- Key insight: Interpretability is not merely about generating descriptions; the **language of description itself** is worthy of deliberate design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Structured language × automated interpretability is a novel combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-model, multi-metric evaluation with a user study; comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, open-sourced by Apple, immediately usable.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear argumentation and excellent visualizations.
- **Overall**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [\[ICLR 2026\] Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)

<!-- RELATED:END -->
