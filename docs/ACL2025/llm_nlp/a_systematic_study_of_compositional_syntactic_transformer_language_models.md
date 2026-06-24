---
title: >-
  [Paper Note] A Systematic Study of Compositional Syntactic Transformer Language Models
description: >-
  [ACL 2025][LLM (Other)][Syntactic Language Models] This paper proposes a unified framework to systematically study four key design dimensions of compositional syntactic Transformer language models (SLMs): tree format, linearization strategy, composition function, and sub-constituent masking. Covering existing models and 13 new variants, this work provides multiple design recommendations for SLMs through comprehensive evaluations across five dimensions: language modeling…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Syntactic Language Models"
  - "Compositionality"
  - "Constituency Parse Trees"
  - "Transformer"
  - "Syntactic Generalization"
date: 2026-05-08
content_hash: ccf125a1a65d1e08
---

# A Systematic Study of Compositional Syntactic Transformer Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.22978](https://arxiv.org/abs/2506.22978)  
**Code**: [GitHub](https://github.com/zhaoyd1/compositional_SLMs)  
**Area**: Natural Language Processing / Language Modeling  
**Keywords**: Syntactic Language Models, Compositionality, Constituency Parse Trees, Transformer, Syntactic Generalization

## TL;DR

This paper proposes a unified framework to systematically study four key design dimensions of compositional syntactic Transformer language models (SLMs): tree format, linearization strategy, composition function, and sub-constituent masking. Covering existing models and 13 new variants, this work provides multiple design recommendations for SLMs through comprehensive evaluations across five dimensions: language modeling, syntactic generalization, summarization, dialogue, and inference efficiency.

## Background & Motivation

**Background**: Although Transformer language models are powerful, they lack inductive biases for syntactic structure. Syntactic language models (SLMs) introduce syntactic bias by jointly modeling syntactic parse trees and surface sentences, showing potential in syntactic generalization and downstream tasks.

**Limitations of Prior Work**: Existing compositional SLMs (such as Transformer Grammars and GPST) make diverse choices regarding tree format, linearization method, composition function, and attention masking schemes. However, the specific impacts of these design dimensions on SLM performance lack systematic investigation.

**Key Challenge**: Different compositional SLMs report performance under their respective experimental setups, lacking a fair and comparable unified evaluation. Consequently, researchers cannot clearly identify which design choices are truly critical.

**Goal**: To systematically identify the key design dimensions of compositional SLMs, construct a unified framework, comprehensively evaluate the performance of 16 variants under unified conditions, and provide design recommendations.

**Key Insight**: By combining four binary design dimensions (binary/non-binary trees, top-down/bottom-up linearization, internal/external composition functions, and masking/non-masking of sub-constituents) into $2^4=16$ variants, this work trains and evaluates them under identical data and parameter scales.

**Core Idea**: Through a systematic experimental comparison of 16 compositional SLM variants within a unified framework, this study reveals the independent and interactive effects of each design dimension, proposing design guidelines such as "sub-constituent masking is not recommended, while external composition function + binary trees are recommended".

## Method

### Overall Architecture

Compositional SLMs jointly model the sentence $x$ and the constituency parse tree $y$, autoregressively generating the linearized $(x, y)$ via an action sequence $a$. The framework offers two options across four dimensions, yielding 16 variants, all implemented uniformly using Transformers.

### Key Designs

1. **Parse Tree Binarization (Binary vs Non-binary)**
    - **Function**: Determines whether the modeled tree is the original non-binary tree or a binary tree converted via Chomsky Normal Form (CNF).
    - **Mechanism**: Non-binary trees preserve linguistic structures but make composition difficult, whereas binary trees simplify composition but increase tree depth.
    - **Design Motivation**: In practice, binary trees facilitate learning effective compositional representations more easily.

2. **Linearization Strategy (Top-down vs Bottom-up)**
    - **Function**: Determines how to convert the tree structure into an action sequence.
    - **Mechanism**: Top-down uses preorder traversal, while bottom-up uses postorder traversal. Bottom-up sequences are shorter (eliminating the need for the "(" action).
    - **Design Motivation**: Bottom-up introduces a new starting position prediction problem for non-binary trees, which is a combination investigated for the first time in this work.

3. **Composition Function (Internal vs External)**
    - **Function**: Determines how to compute the compositional representation of constituents.
    - **Mechanism**: Internal composition (In) reuses the Transformer's own parameters through attention masking; external composition (Ex) uses independent small Transformer modules.
    - **Design Motivation**: Internal composition is simple to implement but suffers from receptive field limitations, while external composition introduces extra parameters but offers stronger expressiveness.

### Loss & Training

- The action sequences are trained using the standard autoregressive cross-entropy loss.
- All models are trained from scratch on the BLLIP-LG dataset, with the parameter scale aligned to GPT-2 small (768 dimensions, 12 layers, 12 heads).
- The external composition function uses a small 4-layer, 256-dimensional Transformer, increasing the parameter count by only 5%.
- During training, a CRF syntactic parser is used to generate silver-standard constituency trees.

## Key Experimental Results

### Main Results

| Model | PPL†(↓) | Syntactic Generalization SG(↑) | Xsum R-AVG(↑) | DailyDialog R-AVG(↑) |
|------|---------|--------------|---------------|---------------------|
| GPT2-token | 17.31 | 64.1 | 18.82 | 10.38 |
| GPT2-tree | 19.97 | 73.1 | 20.88 | 11.04 |
| Bi-Up-Ex-Nm | 20.51 | 80.1 | 20.33 | 10.59 |
| Bi-Up-Ex-M | 24.15 | **82.4** | 16.02 | 9.04 |
| Bi-Up-In-Nm | 19.99 | 77.5 | 20.29 | 9.51 |
| Nb-Dn-In-Nm | **18.11** | 78.1 | 20.81 | 10.40 |

### Ablation Study

| Design Dimension | LM Preference | Syntactic Generalization Preference | Downstream Generation Preference |
|---------|------------|------------|------------|
| Masking vs Non-masking | Nm >> M | Bi: M > Nm; Nb: Nm > M | Nm >> M |
| Internal vs External | In > Ex | Bi-Ex > Bi-In | Marginal difference |
| Binary vs Non-binary | Nb-In alignment slightly superior | Bi >> Nb (with external composition) | Marginal difference |

### Key Findings

- Compositional SLMs do not outperform standard Transformers in language modeling PPL, but significantly improve syntactic generalization (up to 82.4 vs 64.1).
- Nb-#-Ex-# (non-binary + external composition) fails catastrophically in syntactic generalization (only scoring 40-52), because the small external model struggles with variable-length sub-constituents.
- Sub-constituent masking (M) severely degrades performance in language modeling and generation tasks, but helps in binary-tree syntactic generalization.
- GPT2-tree (SLM without composition) surprisingly performs best in downstream tasks, suggesting that explicit composition is not critical for generation.

## Highlights & Insights

- **Value of the Unified Framework**: By incorporating fragmented existing works (TG, CAG, GPST) into a unified framework for comparison, this study reveals for the first time the true impact of individual design options, particularly exposing the critical issue of the Nb-Ex combination.
- **Discrepant Insights Across Tasks**: The same design choice can perform drastically differently across different tasks (e.g., masking helps syntactic generalization but harms generation), reminding researchers of the necessity for multi-dimensional evaluation.
- **Clear Practical Recommendations**: Binary trees + external composition perform best for syntactic generalization, and non-masking is essential for downstream tasks, providing clear guidelines for future SLM designs.

## Limitations & Future Work

- The model scale (GPT-2 small) is relatively small, and conclusions might differ at larger scales.
- Only English data and silver-standard parse trees were utilized; the impacts of multilingual data or gold-standard trees remain unverified.
- The module for external composition is small (4 layers); larger or better composition modules might improve the performance of Nb-Ex.
- Combining unsupervised grammar induction (latent trees) with compositional SLMs has not yet been explored.

## Related Work & Insights

- **Transformer Grammars (Sartran 2022)**: First proposed an SLM with internal composition + masking, corresponding to Nb-Dn-In-M. This paper finds that the non-masking version performs better.
- **GPST (Hu 2024)**: First proposed external composition, corresponding to Bi-Up-Ex-Nm. This paper validates its advantages in syntactic generalization.
- **Insight**: The concept of compositional SLMs can be applied to LLM pre-training to explore whether syntactic bias remains beneficial at a larger scale.

## Rating

- **Novelty**: ⭐⭐⭐ (Unified framework is valuable, though the individual components are not newly designed)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (5 evaluation dimensions, 16 variants; highly systematic experimental design)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure and rich tables/figures, but formulas are relatively dense)
- **Value**: ⭐⭐⭐⭐ (Provides significant guidance for the SLM field; practical recommendations)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)
- [\[ACL 2025\] Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](cadllm_cad_modeling_from_text.md)
- [\[ACL 2025\] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)
- [\[ACL 2025\] Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models](circuit_compositions_modular_structures.md)
- [\[ACL 2025\] An Empirical Study of Large Language Models for Automated Review Generation](an_empirical_study_of_large_language_models_for_automated_review_generation.md)

</div>

<!-- RELATED:END -->
