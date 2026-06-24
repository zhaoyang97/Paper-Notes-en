---
title: >-
  [Paper Note] Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing
description: >-
  [ACL 2025][NLP Understanding][Ambiguity resolution] A modular approach of "disambiguate first, parse later" is proposed, which leverages LLMs to generate default interpretations and trains a specialized infilling model to complete missing ones, thereby transforming ambiguous natural language queries into multiple explicit interpretations before parsing them into SQL individually.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Ambiguity resolution"
  - "Semantic parsing"
  - "Text-to-SQL"
  - "Natural language explanations"
  - "LLM preference"
date: 2026-05-08
content_hash: 6a0e72f7ce9109ce
---

# Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing

**Conference**: ACL 2025  
**arXiv**: [2502.18448](https://arxiv.org/abs/2502.18448)  
**Code**: [github.com/saparina/disambiguate-then-parse](https://github.com/saparina/disambiguate-then-parse)  
**Area**: NLP Understanding  
**Keywords**: Ambiguity resolution, Semantic parsing, Text-to-SQL, Natural language explanations, LLM preference  

## TL;DR

A modular approach of "disambiguate first, parse later" is proposed, which leverages LLMs to generate default interpretations and trains a specialized infilling model to complete missing ones, thereby transforming ambiguous natural language queries into multiple explicit interpretations before parsing them into SQL individually.

## Background & Motivation

**Background**: Natural language interfaces (such as Text-to-SQL systems) face significant challenges when handling ambiguous and under-specified queries. For example, in "return the rating for each hotel," "rating" could refer to a star rating, a customer review rating, or both.

**Limitations of Prior Work**: Existing LLMs exhibit strong systematic preferences when encountering ambiguity—they tend to predict only a single "default" interpretation while ignoring other plausible readings. Studies by Liu et al. (2023) and Floratou et al. (2024) have confirmed this issue.

**Key Challenge**: Ambiguous queries have multiple valid SQL formulations (each yielding different execution results), but models can only cover a fraction of these interpretations. Directly prompting LLMs to generate all interpretations has limited efficacy because the inherent biases resulting from LLM training data distribution are difficult to eliminate.

**Goal**: How to completely cover all valid interpretations of ambiguous queries in Text-to-SQL scenarios, rather than outputting only a single preferred interpretation.

**Key Insight**: Rather than trying to correct LLM bias, this work **exploits** this bias—first utilizing the LLM to generate the "default" interpretation, and then training a specialized infilling model to discover and complete the missing interpretations.

**Core Idea**: The task of ambiguity resolution is decomposed into a three-step modular pipeline: "default interpretation generation + missing interpretation infilling + semantic parsing," using natural language explanations as an intermediate representation to bridge ambiguous inputs and SQL outputs.

## Method

### Overall Architecture

The system consists of three modular components:
- **Default Interpretation Generator**: Uses zero-shot prompting of an LLM to generate an initial set of interpretations $\{\hat{u}_1, ..., \hat{u}_k\}$ for an ambiguous query $u$ under database context $\mathcal{C}$
- **Infilling Model**: Inspects the ambiguous query and the initial interpretations to generate the missing interpretations $\{\hat{u}_l, ..., \hat{u}_m\}$
- **Text-to-SQL Parser**: Translates each explicit interpretation independently into a SQL query

### Key Designs

**Problem Formalization**: For an ambiguous query $u$, there exist multiple valid SQL expressions $\{e_1, ..., e_n\}$, where $\llbracket e_i \rrbracket \neq \llbracket e_j \rrbracket$ (yielding different execution results). The goal is to first generate a set of explicit natural language interpretations and then map each to SQL.

**Default Interpretation Generation**: Instructions-tuned Llama-3.1 8B is used with a carefully designed prompt to handle both ambiguous and unambiguous cases. Redundancies in the generated interpretations are removed by comparing the SQL execution results (interpretations with identical execution results are considered synonymous).

**Infilling Training Data Construction**: This is the core innovation of the methodology. Since existing datasets (like AmbiQT) only provide gold SQLs without natural language explanations, the authors creatively:
1. Utilize the synonym replacement strategy of AmbiQT and leverage an LLM (Llama 3.1 8B) to generate reference interpretations by replacing column/table name synonyms in the query.
2. Verify using a code generation LLM (Qwen2.5-Coder 32B): only keeping samples where both interpretations can generate correct SQL queries within 5 attempts.
3. Compare SQL execution results instead of direct natural language comparison to determine if the default interpretations have covered a reference interpretation.

**Infilling Model Training**: The input consists of the ambiguous query, the database context, and the set of default interpretations, and the output is the missing interpretations. If all interpretations are already covered, the model outputs "All interpretations are covered." It is trained on Llama-3.1 8B using LoRA adapters.

### Loss & Training

- The infilling model is trained using standard supervised sequence generation.
- Approximately 5K synthetic data points derived from a subset of the Spider training set are used for training.
- Only the infilling component requires training; the other two modules are plug-and-play pretrained models.

## Key Experimental Results

### Main Results

| Method | AmbiQT Single↑ | AmbiQT Full↑ | Ambrosia Single↑ | Ambrosia Full↑ |
|------|:---:|:---:|:---:|:---:|
| 0-shot Prompt (End-to-End) | 62.3 | 12.3 | 29.4 | 0.9 |
| 3-shot Prompt (End-to-End) | 44.3 | 10.9 | 35.7 | 1.3 |
| SFT (End-to-End) | 82.1 | 63.2 | 38.0 | 0.4 |
| Interpretation Gen (Prompt) | 81.8 | 26.0 | 81.9 | 16.9 |
| **Ours (w. Infilling)** | **92.3** | **53.2** | **84.4** | **18.8** |

### Ablation Study

| Default Interpretation Model | AmbiQT Single/Full | w. Infilling Single/Full |
|---|:---:|:---:|
| Llama 3.1 8B | 81.8/26.0 | 92.3/53.2 |
| Qwen 2.5 7B | 65.0/25.9 | 88.7/48.6 |
| Gemma 2 9B | 77.3/17.3 | 91.1/55.9 |

Effect of Text-to-SQL models: For Qwen2.5-Coder 32B vs 7B, the AmbiQT Full coverage decreases from 53.2% to 40.2%.

### Key Findings

1. **Infilling is effective for all generator models**: Regardless of whether Llama, Qwen, or Gemma is used as the default generator, infilling significantly improves coverage.
2. **End-to-end SFT performs well in-domain but generalizes poorly**: It achieves 63.2% Full Coverage on AmbiQT but only 0.4% on Ambrosia.
3. **Self-correction degrades recall**: It filters out some valid interpretations, dropping Single coverage from 81.8% to 77.4%.
4. **Error Analysis**: SQL generation errors (~30%) and interpretations failing to match intentions (~27-30%) are the primary sources of error.

## Highlights & Insights

- **Cleverly Leverages LLM Biases**: Instead of eliminating LLM preferences, this approach turns them into an advantage—securing the "easy" default interpretations first, then focusing purely on filling the missing details.
- **Advantages of Modular Design**: The three components can be independently replaced and optimized; improvements in any module directly boost overall performance.
- **Natural Training Data Construction**: Using SQL execution results as an "automatic annotator" to validate interpretation correctness avoids expensive manual annotation.
- **Natural Handling of Unambiguous Queries**: For unambiguous queries, only a single interpretation is produced, and the infilling module naturally outputs "all covered," bypassing the need for a separate ambiguity detection module.

## Limitations & Future Work

- Full Coverage on Ambrosia remains low (18.8%), indicating that cross-domain generalization capability needs further improvement.
- Infilling only adds new interpretations but cannot filter out erroneous ones, which might cause precision to decrease.
- The generation of reference interpretations relies on synonym replacement, which has limited applicability to more complex ambiguity types, such as scope ambiguity.
- The generalizability to other semantic parsing tasks (e.g., code generation) beyond Text-to-SQL has not been validated.

## Related Work & Insights

- This work complements AmbiQT by Bhaskar et al. (2023) and Ambrosia by Saparina & Lapata (2024a)—where the former injects ambiguity and the latter incorporates human-annotated interpretations.
- The "explain before execution" paradigm can be extended to other domains requiring ambiguity resolution, such as code generation and mathematical reasoning.
- The concept of infilling is analogous to boosting in ensemble learning—acquiring "easy" solutions first, then focusing on the "difficult" parts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The paradigm of "exploiting biases rather than eliminating them" is novel, and the training data construction method for the infilling model is clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The validation across two datasets, various baselines, and ablation studies is solid, though cross-domain generalization experiments could be further elaborated.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The paper is clearly structured, the motivation is well-articulated, and the illustrations are intuitive.
- **Value**: ⭐⭐⭐⭐ — The modular approach is highly practical and provides a new paradigm for ambiguity resolution in semantic parsing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)
- [\[ACL 2025\] BookCoref: Coreference Resolution at Book Scale](bookcoref_book_scale.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2025\] End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)

</div>

<!-- RELATED:END -->
