---
title: >-
  [Paper Note] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] Proposes the first modular analysis framework for query-level RAG robustness. Through 1092+ experiments across 5 perturbation types $\times$ 4 retrievers $\times$ 3 LLMs $\times$ 3 datasets, the study reveals the complementary robustness of dense and sparse retrievers against different perturbation types and provides actionable engineering recommendations.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Query Perturbation"
  - "Robustness Evaluation Framework"
  - "Dense/Sparse Retrievers"
  - "Modular Decoupling Analysis"
date: 2026-05-08
content_hash: bf5b05147e5a10de
---

# Investigating the Robustness of Retrieval-Augmented Generation at the Query Level

**Conference**: ACL 2025  
**arXiv**: [2507.06956](https://arxiv.org/abs/2507.06956)  
**Code**: Coming soon  
**Area**: Information Retrieval / RAG Robustness  
**Keywords**: RAG, Query Perturbation, Robustness Evaluation Framework, Dense/Sparse Retrievers, Modular Decoupling Analysis  

## TL;DR

Proposes the first modular analysis framework for query-level RAG robustness. Through 1092+ experiments across 5 perturbation types $\times$ 4 retrievers $\times$ 3 LLMs $\times$ 3 datasets, the study reveals the complementary robustness of dense and sparse retrievers against different perturbation types and provides actionable engineering recommendations.

## Background & Motivation

- **Background**: RAG has become a mainstream solution for mitigating LLM hallucinations and reducing knowledge updating costs. By retrieving external documents to provide factual context for generation, it is widely applied in scenarios such as enterprise QA and knowledge management.
- **Limitations of Prior Work**: Existing research on RAG robustness mainly focuses on noise in the retrieved documents (irrelevant passages, misinformation) or evaluates the robustness of the retriever/LLM module in isolation. There is a lack of systematic, decoupled evaluation of all modules across the entire RAG pipeline starting from the query end.
- **Key Challenge**: Real-world user queries exhibit natural diversity—different phrasing habits, typos, redundant expressions, and differences in tone—but it remains unclear how much impact these variations have on each individual module in the RAG pipeline, making targeted optimization difficult.
- **Goal**: Quantify the sensitivity of different components in the RAG pipeline (retriever, generator, end-to-end) to various query perturbations, and provide a module-level diagnostic methodology to help practitioners pinpoint pipeline bottlenecks.
- **Key Insight**: Design five semantic-preserving query perturbations (redundant information, formal tone, introducing ambiguity, and 10%/25% typos) to measure the performance changes of each module under isolated and joint settings, and then decouple the dominant factors through Pearson correlation analysis.
- **Core Idea**: Through large-scale controlled experiments and modular decoupling correlation analysis, this work is the first to reveal the differentiated impact of query-level perturbations on each module of the RAG pipeline and proposes a systematic diagnostic framework.

## Method

### Overall Architecture

The analysis framework in this paper consists of three progressive layers of experiments:

1. **Retriever Isolation Analysis**: Applies 5 types of perturbations to 4 retrievers (BGE-base-en-v1.5 and Contriever as dense retrievers; BM25 Flat and BM25 Multi-field as sparse retrievers), measuring retrieval performance changes using Recall@k.
2. **Generator Isolation Analysis**: Evaluates 3 LLMs (Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, Qwen2.5-7B-Instruct) under two extreme settings: Closed-book (pure parametric knowledge, no retrieval) and Oracle (assuming perfect retrieval, providing only correct documents).
3. **End-to-End Pipeline Analysis**: Combines retrievers and generators into 12 RAG pipelines, measuring joint performance and decoupling each module's contribution to end-to-end performance variations via Pearson correlation coefficients.

The datasets cover three QA datasets with different characteristics: NQ (general single-hop QA, 2.68M documents), HotpotQA (multi-hop QA, 5.23M documents), and BioASQ (biomedical domain, 14.91M documents). Evaluation metrics include Recall@k (retrieval side) and Match (generation side, checking if the output contains the answer span), utilizing model-free evaluation to prevent instability introduced by LLM evaluators.

### Key Design 1: Five Query Perturbation Generation Strategies

- **Redundant Information Insertion (Redundancy)**: Uses GPT-4o to insert relevant redundant descriptions without answer clues into the query, simulating scenarios where users input excessive background information.
- **Formal Tone Change (Formal)**: Rewrites queries into more formal expressions using GPT-4o, altering only the surface form.
- **Ambiguity Introduction (Ambiguity)**: Introduces vague expressions (e.g., inserting "might", replacing specific terms with more general ones) via GPT-4o to simulate imprecise user formulations.
- **Typos 10%/25%**: Uses the TextAttack library to replace characters in words based on QWERTY keyboard proximity, keeping stop words intact to preserve core semantics.
- Each perturbation type generates 5 perturbed versions for each original sample, and the quality of the perturbed samples is verified using GPT2-Large's perplexity and multilingual-e5-base's semantic similarity.

### Key Design 2: Modular Decoupling Pearson Correlation Analysis

- **Function**: Quantitative differentiation of the respective contribution ratios of the retriever and generator to the end-to-end RAG performance fluctuations.
- **Mechanism**: First calculates the performance difference between each perturbed sample and its original counterpart on a sample-by-sample basis ($\Delta Recall@5$ for the retriever and $\Delta Match$ for the generator), and then calculates the Pearson correlation coefficients for "$\Delta Retriever - \Delta RAG$" and "$\Delta Generator - \Delta RAG$" respectively.
- **Design Motivation**: End-to-end RAG performance is a coupled outcome of the retriever and the generator, making it impossible to identify bottlenecks through direct observation. Through decoupling analysis, practitioners can accurately target the problematic module for specific perturbation types.

### Key Design 3: LLM Internal Representation Visualization

- **Function**: Reveals how query perturbations interfere with the LLM's reasoning process from the perspective of its internal representations.
- **Mechanism**: Extracts the average representation of all attention heads in the final hidden layer of Llama-3.1-8B (corresponding to the last non-padding token) and visualizes it in two dimensions using PCA dimensionality reduction.
- **Design Motivation**: Reveals that on BioASQ, even under the Oracle setting, redundancy and ambiguity perturbations still cause significant dispersion in the LLM's internal representation, explaining why correct documents cannot fully compensate for the negative impact of query perturbations.

### Loss & Training

This work is purely analytical; all retrievers and LLMs directly utilize pretrained weights without fine-tuning. Generation employs greedy decoding (temperature=0) with a maximum input length of 4096 tokens and a maximum generation length of 128 tokens, and each retrieved document is truncated to 100 words. Experiments were completed on RTX 3090 GPUs using the vLLM inference framework.

## Key Experimental Results

### Main Results: Retriever Recall@5 (%) Performance under Different Perturbations

| Dataset | Retriever | Original | Redundancy | Formal | Ambiguity | Typo 10% | Typo 25% |
|--------|--------|----------|------------|--------|-----------|----------|----------|
| HotpotQA | BGE Base | **71.82** | 66.92 | 69.34 | 64.45 | 62.94 | 47.75 |
| HotpotQA | Contriever | **60.84** | 59.12 | 59.34 | 56.42 | 53.37 | 39.06 |
| HotpotQA | BM25 Flat | **60.81** | 44.72 | 54.20 | 49.38 | 54.34 | 41.92 |
| HotpotQA | BM25 MF | **58.00** | 47.20 | 53.90 | 50.17 | 50.59 | 37.36 |
| NQ | BGE Base | **64.59** | 55.10 | 61.65 | 51.60 | 50.04 | 34.35 |
| NQ | Contriever | **58.60** | 52.11 | 56.58 | 47.33 | 45.39 | 30.95 |
| BioASQ | BGE Base | **36.06** | 33.01 | 34.82 | 30.24 | 30.43 | 27.83 |
| BioASQ | BM25 Flat | **45.22** | 25.01 | 37.89 | 33.37 | 35.94 | 29.87 |

### Ablation Study: Pearson Correlation Decoupling Analysis (BGE Base + Llama-3.1-8B)

| Dataset | Correlation Type | Redundancy | Formal | Ambiguity | Typo 10% | Typo 25% |
|--------|---------|------------|--------|-----------|----------|----------|
| BioASQ | Retriever-RAG | 0.05 | 0.04 | 0.15 | **0.21** | **0.23** |
| BioASQ | Closed-book-RAG | 0.21 | 0.08 | 0.23 | 0.05 | 0.10 |
| BioASQ | Oracle-RAG | **0.35** | **0.15** | **0.33** | 0.04 | 0.12 |
| NQ | Retriever-RAG | **0.31** | **0.27** | **0.30** | **0.35** | **0.40** |
| NQ | Closed-book-RAG | 0.03 | 0.04 | 0.11 | 0.08 | 0.16 |
| NQ | Oracle-RAG | 0.11 | 0.14 | 0.15 | 0.06 | 0.03 |

### Key Findings

- Dense retrievers are more robust to redundant information but sensitive to spelling errors (BGE on NQ under Typo 25% causes Recall to plummet from 64.59% to 34.35%), whereas sparse retrievers show the opposite behavior (BM25 Flat on BioASQ under redundancy drops from 45.22% to 25.01%).
- On NQ, end-to-end performance is strongly correlated with changes in retriever performance (Pearson 0.27-0.40), indicating that the retriever is the dominant factor; on BioASQ under redundancy/ambiguity, the correlation with Oracle is higher (0.33/0.35), making the generator's context utilization capability the bottleneck.
- Closed-book performance does not predict RAG performance—Mistral has limited parametric knowledge but performs best under the Oracle setting, demonstrating that generator evaluation in RAG cannot be decoupled from the retrieved context.
- Formal tone mapping has the least impact across all modules and datasets, whereas Typo 25% is the most destructive.

## Highlights & Insights

- The first modular diagnostic framework for query-level RAG robustness; the three-layer progressive analysis provides a highly structured and reproducible set of conclusions.
- The discovery of dense/sparse complementarity offers direct engineering value—hybrid retrieval strategies can be considered to hedge against the risks of query variations.
- The modular decoupling methodology is highly practical: practitioners can run this framework on their own pipelines and data to rapidly pinpoint bottleneck modules.
- The conclusion of "Closed-book $\neq$ RAG robustness" corrects a common misconception.
- The experimental scale is large (1092+ experiments) yet cost-effective (7-8B models + RTX 3090), making the methodology widely reproducible.

## Limitations & Future Work

- Does not include a reranker module; in practical RAG pipelines, rerankers might alter the impact distribution of perturbations.
- Only employs 7-8B parameter LLMs; the robustness performance of larger models (70B+) might differ.
- Does not explore the practical effects of mitigation strategies (e.g., query rewriting/expansion, adversarial training, query-perturbation-aware training).
- The analysis of LLM internal representations is limited to PCA visualization, lacking quantitative mechanical explanations.
- Evaluation metrics are restricted to surface matching (Match), without adopting semantic-level metrics such as BERTScore.
- The pipeline hyperparameter space (context length, document truncation strategies) is not fully explored.

## Related Work & Insights

- **Retriever Robustness**: Zhuang & Zuccon (2022) investigate encoding strategies and training methods of BERT retrievers against spelling errors; Arabzadeh et al. (2023) directly evaluate stability by perturbing dense representations. This work extends these to a comprehensive comparison across multiple perturbation types.
- **LLM Robustness**: Shi et al. (2023) find that LLMs are easily distracted by irrelevant context; Zhu et al. (2024a,b) perform prompt perturbation analysis at multi-granular levels. The unique contribution of this work lies in distinguishing between closed-book and oracle settings, revealing the inconsistency in conclusions between the two.
- **Pipeline Robustness**: Chen et al. (2023) test the ability of RAG to handle irrelevant/incorrect information; Fang et al. (2024) and Yoran et al. (2024) propose training methods to enhance noise robustness. This paper offers a complementary perspective by starting from the query end and providing a modular decoupling analysis.
- **Insights**: Hybrid dense + sparse retrieval might be the most direct strategy to improve robustness; incorporating query-perturbation augmentation (perturbation-aware training) in RAG joint training is a direction worth exploring.

## Rating

- Novelty: ⭐⭐⭐ The first modular decoupling analysis framework for query-level RAG robustness, but its core methodology consists of standard controlled experiments and correlation analysis, yielding limited methodological novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1092+ experiments covering 4 retrievers $\times$ 3 LLMs $\times$ 5 perturbations $\times$ 3 datasets, providing a comprehensive and deep analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich tables/figures, and highly practical engineering recommendations.
- Value: ⭐⭐⭐⭐ Highly instructive for RAG engineering practices, and the modular decoupling methodology has high reusability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)
- [\[ACL 2025\] Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness](multilingual_retrieval_augmented_generation_for_culturally-sensitive_tasks_a_ben.md)
- [\[ACL 2025\] Investigating Language Preference of Multilingual RAG Systems](investigating_language_preference_of_multilingual_rag_systems.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)

</div>

<!-- RELATED:END -->
