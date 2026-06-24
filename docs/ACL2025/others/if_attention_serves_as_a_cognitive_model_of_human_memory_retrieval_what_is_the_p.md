---
title: >-
  [Paper Note] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?
description: >-
  [ACL 2025][Attention] This paper investigates whether the attention mechanism of Transformer Grammar (TG) can serve as a cognitive model for human memory retrieval. By relating the model to human reading times using Normalized Attention Entropy (NAE), the study reveals that syntax-based attention explains human sentence processing behavior better than token-based attention, and both make independent, complementary contributions.
tags:
  - "ACL 2025"
  - "Attention"
  - "Human Memory Retrieval"
  - "Transformer Grammar"
  - "Syntactic Structure"
  - "Reading Time Prediction"
date: 2026-05-08
content_hash: d070e351e1ebaabd
---

# If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?

**Conference**: ACL 2025  
**arXiv**: [2502.11469](https://arxiv.org/abs/2502.11469)  
**Code**: [Available (GitHub)](https://github.com/osekilab/TG-NAE)  
**Area**: Computational Psycholinguistics / NLP  
**Keywords**: Attention, Human Memory Retrieval, Transformer Grammar, Syntactic Structure, Reading Time Prediction

## TL;DR

This paper investigates whether the attention mechanism of Transformer Grammar (TG) can serve as a cognitive model for human memory retrieval. By relating the model to human reading times using Normalized Attention Entropy (NAE), the study reveals that syntax-based attention explains human sentence processing behavior better than token-based attention, and both make independent, complementary contributions.

## Background & Motivation

In computational psycholinguistics, whether language models can serve as cognitive models of human sentence processing is a core question. Historically, this question has been primarily explored from the perspective of **expectation-based theory**—i.e., whether the prediction of the next token (surprisal) can model human predictive processing. In recent years, the success of the attention mechanism in Transformers has unexpectedly opened a new path for another major class of theories—**memory-based theory**.

Researchers have found an intriguing parallel between the weighted reference patterns in attention weights and the way humans retrieve elements during online sentence comprehension. In particular, **cue-based retrieval** theory posits that when processing sentences, humans retrieve prior elements from working memory using cues provided by the current input word, and retrieval becomes more difficult as the number of distractors increases.

However, prior work has focused on **vanilla Transformers** (operating on token-level representations), overlooking an important fact: psycholinguistic research has long shown that **syntactic structure** provides explanations for human sentence processing that cannot be fully explained by token-level factors. This leads to the core question of this paper: **If attention can serve as a general algorithm for memory retrieval, can attention operating over syntactic structures also capture human memory retrieval?**

## Method

### Overall Architecture

This paper connects the attention mechanism of **Transformer Grammar (TG)** with human reading time data. TG is a syntactic language model that jointly generates token sequences and their corresponding syntactic structures. The linking hypothesis utilizes **Normalized Attention Entropy (NAE)**: the more dispersed the attention weights (higher entropy), the greater the interference during retrieval, corresponding to longer reading times.

### Key Designs

1. **The COMPOSE/STACK Mechanism in Transformer Grammar (TG)**: The core innovation of TG lies in its handling of closed constituents. When a closing bracket X) is generated, a vector representation of the closed constituent is computed via COMPOSE attention; subsequent STACK operations refer to this vector as the representation of the constituent for the next step of prediction. This means TG's attention operates on units of syntactic structure (closed constituents as a whole) rather than on token sequences as in the vanilla Transformer.

2. **Computation of NAE**: For each word, the NAE values of each attention head at the top layer of TG are summed. NAE is obtained by calculating the normalized entropy after renormalizing the attention weights, with a range of $[0, 1]$. For TG, only attention triggered by lexical tokens (terminals) is considered, excluding attention to non-lexical symbols.

3. **TG-comp Variant**: To determine whether TG's advantage arises simply from considering syntactic structures or specifically from COMPOSE attention (representing a closed constituent as a single representation), the authors constructed a TG-comp variant. It treats each action in the action sequence as an independent token and does not use COMPOSE attention.

### Loss & Training

- **Model**: A 16-layer, 8-head TG and Transformer (252M parameters), trained on the BLLIP-lg corpus (42M tokens, 1.8M sentences)
- **Evaluation**: Linear Mixed-Effects Models are used, with baseline predictors (word length, n-gram frequency, surprisal, stack count, etc.) plus NAE as fixed effects. The contribution of NAE is evaluated via $\Delta\text{LogLik}$ (increment in log-likelihood)
- **Reading Time Data**: Natural Stories corpus, containing various syntactically complex constructions of interest to psycholinguistics

## Key Experimental Results

### Main Results: Contribution of NAE to Reading Time Prediction

| Model | $\Delta\text{LogLik}$ (↑) | NAE Effect Size (ms) | NAE_so Effect Size (ms) | Significant Seeds |
|------|-------------|----------------|-------------------|---------|
| TG | 76.6 (±8.1) | 1.42 (±0.2)*** | 2.26 (±0.1)*** | 3/3 |
| Transformer | 42.8 (±9.5) | 1.32 (±0.2)*** | 1.46 (±0.2)*** | 3/3 |

### Ablation Study: Contribution of COMPOSE Attention

| Model | $\Delta\text{LogLik}$ | NAE Significance | NAE_so Significance |
|------|---------|-----------|-------------|
| TG | 46.1 (±9.1) | ** (2/3) | *** (3/3) |
| TG-comp | 18.1 (±9.3) | ** (1/3) | *** (3/3) |

### Key Findings

1. **The contribution of TG's NAE is significantly higher than that of Transformer** (76.6 vs 42.8 $\Delta\text{LogLik}$), indicating that syntactic structure-based memory retrieval plays a more dominant role in human sentence processing.
2. **The two models provide independent contributions**: Likelihood ratio tests show that a regression model including both NAEs is significantly better than either single model, implying that humans use dual memory representations—syntactic structures and token sequences.
3. **POS analysis reveals complementarity**: TG's NAE is better on verbs (VB, VBG, VBN, VBP), while the Transformer's NAE is better on nouns (NN, NNP). This aligns with the hypothesis that "verb-triggered retrieval depends on syntactic features, while noun-triggered retrieval depends on semantic features."
4. **COMPOSE attention is crucial**: TG significantly outperforms TG-comp (46.1 vs 18.1), and TG-comp fails to capture the variance already explained by TG. COMPOSE contributes the most to verb processing.
5. **NAE captures interference effects rather than decay effects**: When both NAE and Category Locality Theory (CLT) are included in the model, their contributions remain independent, confirming that NAE quantifies interference during memory retrieval.

## Highlights & Insights

- **Interdisciplinary Bridge**: This study bridges the attention mechanism in NLP with syntactic structure theories in linguistics, providing a broad-coverage candidate implementation for human memory retrieval.
- **From the "Computational Level" to the "Algorithmic Level"**: Compared to surprisal theory, which operates at the most abstract computational level of Marr's tri-level hypothesis, interpreting attention as memory retrieval shifts the investigation down to the more concrete algorithmic level.
- **Empirical Support for the Dual-Memory System Hypothesis**: One system is based on syntactic structure and the other on token sequences, with attention serving as the general retrieval algorithm.

## Limitations & Future Work

- The computation of NAE (only at the top layer, summing across heads, and summing across subwords) follows legacy methods; alternative approaches warrant exploration.
- Validation is restricted to English self-paced reading time corpora; the generalizability to other languages and other cognitive metrics (e.g., eye-tracking, EEG/fMRI) remains unknown.
- The study utilizes an "oracular" syntactic structure, thus failing to resolve local ambiguities actually encountered by humans.
- The model adopts a top-down parsing strategy, whereas psycholinguistic evidence generally favors a left-corner strategy.

## Related Work & Insights

- Ryu and Lewis (2021) first proposed using attention entropy as a linking hypothesis for cue-based retrieval.
- Oh and Schuler (2022) generalized it to naturalistic text and proposed NAE.
- The Transformer Grammar by Sartran et al. (2022) outperforms the vanilla Transformer on grammatical judgment and brain activity.
- Category Locality Theory (Isono, 2024) quantifies memory decay based on the distance between syntactic phrases.
- This study integrates these directions to systematically compare structure-level vs. token-level attention for modeling human memory retrieval for the first time.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to interpret TG's attention as a cognitive memory retrieval model, filling the gap at the syntactic structure level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Main experiments + independent contribution tests + POS analysis + COMPOSE ablation + separation of interference vs. decay; the analytical pipeline is highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Exceptionally clear logic, natural motivation build-up, and smooth integration between technical concepts and cognitive theories.
- **Value**: ⭐⭐⭐⭐ — Provides significant empirical and theoretical contributions to the intersection of computational psycholinguistics and NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](developmentally-plausible_working_memory_shapes_a_critical_period_for_language_a.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ACL 2025\] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents](in_prospect_and_retrospect_reflective_memory_management_for_long-term_personaliz.md)
- [\[ICML 2025\] Modern Methods in Associative Memory](../../ICML2025/others/modern_methods_in_associative_memory.md)
- [\[NeurIPS 2025\] Dense Associative Memory with Epanechnikov Energy](../../NeurIPS2025/others/dense_associative_memory_with_epanechnikov_energy.md)

</div>

<!-- RELATED:END -->
