---
title: >-
  [Paper Note] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home
description: >-
  [ACL 2025][Adaptive Retrieval] This work conducts a comprehensive evaluation of 35 adaptive retrieval methods (including 8 state-of-the-art methods and 27 uncertainty estimation methods), revealing that classic uncertainty estimation techniques often outperform complex, specialized pipelines in terms of efficiency and self-knowledge capability, while maintaining comparable QA performance.
tags:
  - "ACL 2025"
  - "Adaptive Retrieval"
  - "RAG"
  - "Uncertainty Estimation"
  - "Self-knowledge"
  - "Question Answering System"
date: 2026-05-08
content_hash: 2c31e949cb9c23e9
---

# Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home

**Conference**: ACL 2025  
**arXiv**: [2501.12835](https://arxiv.org/abs/2501.12835)  
**Code**: [https://github.com/s-nlp/AdaRAGUE](https://github.com/s-nlp/AdaRAGUE)  
**Area**: Other  
**Keywords**: Adaptive Retrieval, RAG, Uncertainty Estimation, Self-knowledge, Question Answering System

## TL;DR

This work conducts a comprehensive evaluation of 35 adaptive retrieval methods (including 8 state-of-the-art methods and 27 uncertainty estimation methods), revealing that classic uncertainty estimation techniques often outperform complex, specialized pipelines in terms of efficiency and self-knowledge capability, while maintaining comparable QA performance.

## Background & Motivation

Retrieval-Augmented Generation (RAG) can improve the accuracy of LLM responses and alleviate hallucinations, but it faces two key challenges:

**Retrieval is not always necessary**: When the LLM already possesses the relevant knowledge, retrieval may introduce irrelevant information, leading to error propagation and external hallucinations.

**High computational cost**: Each retrieval call requires invoking the retriever and performing additional LM inference.

To address this, adaptive retrieval methods have emerged, which leverage the LLM's **self-knowledge** (i.e., the model's ability to recognize its own knowledge boundaries) to decide when retrieval is needed. However, existing research has three blind spots:

- **Neglecting efficiency evaluation**: Focusing only on the number of retrievals without tracking the number of LM calls (which can be more expensive).
- **Lack of comparison with uncertainty estimation methods**: Mature methods like Mean Entropy have never been systematically evaluated in adaptive retrieval scenarios.
- **Lack of self-knowledge capability evaluation**: Fails to measure the capability of the method itself in predicting "whether retrieval is needed".

This paper bridges these gaps through a comprehensive and unified evaluation framework.

## Method

### Overall Architecture

A unified evaluation of 35 methods across 6 datasets is conducted, using 10 metrics covering three dimensions: QA performance, self-knowledge, and efficiency. Uncertainty estimation methods are integrated into the adaptive retrieval framework of AdaptiveRAG to ensure comparability.

### Key Designs

1. **End-to-end adaptive retrieval methods (8 methods)**:

    - **IRCoT**: Dynamically adds retrieved passages during Chain-of-Thought reasoning until the answer is generated.
    - **AdaptiveRAG**: Uses a T5-large classifier to predict three outcomes (no retrieval/single retrieval/multiple retrieval).
    - **FLARE**: Triggers retrieval and regenerates responses when token probabilities fall below a threshold.
    - **DRAGIN**: Similar to FLARE but filters stop words and reconstructs queries using attention weights.
    - **Rowen**: Determines whether to retrieve based on consistency (cross-lingual/cross-model).
    - **SeaKR**: Monitors internal states using an uncertainty module to trigger retrieval and rerank candidate passages.

2. **Uncertainty estimation methods (27 methods, highlighting 5)**:

    - **Lexical Similarity**: Based on the average similarity among sampled responses.
    - **Max/Mean Entropy**: Computes the entropy for each token and aggregates them using maximum/mean values.
    - **SAR**: Entropy aggregation weighted by token relevance.
    - **EigValLaplacian**: Constructs a weighted graph of sampled responses and computes the sum of Laplacian eigenvalues.
    - Uncertainty scores are converted into retrieval decisions using a classifier.

3. **Evaluation framework**:

    - **QA metrics**: In-Accuracy (InAcc), Exact Match (EM), F1.
    - **Efficiency metrics**: Retrieval Calls (RC), LM Calls (LMC).
    - **Self-knowledge metrics**: ROC-AUC, Spearman correlation coefficient, accuracy, overconfidence rate, underconfidence rate.
    - **Cross-dataset ranking aggregation**: Uses reciprocal rank fusion for fair comparison.

### Loss & Training

- All methods uniformly use the LLaMA 3.1-8b-instruct model.
- BM25 + Elasticsearch is uniformly used as the retriever.
- Classifiers for uncertainty methods are trained on the training set, and the best classifier is selected to report test set results.
- Baseline methods retain their respective original settings.

## Key Experimental Results

### Main Results

**QA performance and efficiency (selected key methods)**:

| Method | NQ InAcc | NQ LMC | NQ RC | HotPot InAcc | HotPot LMC | HotPot RC |
|------|----------|--------|-------|-------------|------------|-----------|
| Never RAG | 0.446 | 1.0 | 0.00 | 0.286 | 1.0 | 0.00 |
| Always RAG | 0.496 | 1.0 | 1.00 | 0.410 | 1.0 | 1.00 |
| IRCoT | 0.478 | 2.7 | 2.70 | 0.438 | 4.4 | 4.38 |
| DRAGIN | 0.480 | 4.5 | 2.24 | 0.430 | 5.1 | 2.56 |
| RowenHybrid | 0.494 | **55.0** | 7.27 | 0.354 | **59.8** | 7.63 |
| **Mean Entropy** | 0.498 | **1.9** | **0.88** | 0.410 | **2.0** | **0.99** |
| **Best UE** | **0.512** | **1.8** | **0.81** | **0.414** | **2.0** | **0.99** |
| Ideal Oracle | 0.608 | 1.6 | 0.55 | 0.460 | 1.7 | 0.71 |

### Ablation Study

**Self-knowledge evaluation (ROC-AUC)**:

| Method | NQ | SQUAD | TQA | 2Wiki | HotPot | Musique |
|------|-----|-------|-----|-------|--------|---------|
| AdaptiveRAG | 0.54 | 0.58 | 0.49 | 0.71 | 0.62 | 0.64 |
| FLARE | 0.59 | 0.58 | 0.57 | 0.62 | 0.54 | 0.51 |
| SeaKR | 0.64 | **0.77** | **0.78** | 0.37 | 0.55 | 0.56 |
| **Max Entropy** | 0.73 | 0.72 | 0.72 | **0.73** | **0.66** | **0.68** |

### Key Findings

1. **Uncertainty estimation methods exhibit an overwhelming advantage in efficiency**: Each question requires only ~2 LM calls and <1 retrieval call, whereas RowenHybrid requires 55-80 LM calls.
2. **Uncertainty methods achieve comparable QA performance to complex pipelines**: Best UE even outperforms most end-to-end methods on single-hop datasets.
3. **Uncertainty methods are consistently superior in self-knowledge capability**: The ROC-AUC of Max Entropy significantly outperforms all end-to-end methods on the majority of datasets.
4. **No single method dominates across all dimensions**: Efficiency $\neq$ Performance $\neq$ Self-knowledge; selection should depend on the application scenario.
5. **A clear gap remains between current methods and optimal performance**: The Ideal Oracle demonstrates substantial room for improvement.

## Highlights & Insights

- **Significant framework contribution**: This study systematically introduces 27 uncertainty estimation methods to the adaptive retrieval scenario for the first time, filling an important comparative gap.
- **Challenging the "complex is better" assumption**: The simple Mean Entropy method outperforms carefully designed multi-step pipelines in many scenarios.
- **LM calls as an overlooked critical cost**: The Rowen series requires 30-80 LM calls per question, which is cost-prohibitive when using commercial APIs.
- **Multi-dimensional evaluation perspective**: It unifies triple dimensions of QA performance, efficiency, and self-knowledge capability into a single evaluation for the first time, providing a comprehensive reference for researchers.
- **OOD Analysis**: It further evaluates the performance of uncertainty methods under out-of-distribution scenarios, adding practical value.

## Limitations & Future Work

- Only a single model (LLaMA 3.1-8b-instruct) is evaluated; different models may exhibit different optimal methods.
- Only BM25 is utilized as the retriever; stronger retrievers (such as dense retrievers) might alter the relative ranking of methods.
- Uncertainty methods require training a classifier to determine thresholds, introducing additional data and hyperparameter tuning demands.
- The definition of self-knowledge capacity relies on binary In-Accuracy labels, which may be too coarse.
- The quality of retrieved documents is not factored into final answer evaluation; poor retrieval might cause "Always RAG" to be underestimated.

## Related Work & Insights

- Unifies two previously isolated research domains: Adaptive RAG (Su et al., 2024; Jeong et al., 2024) and Uncertainty Estimation (Fadeeva et al., 2023; Duan et al., 2023).
- Provides a practical evaluation framework for LLM self-knowledge capacity research (Yin et al., 2023, 2024).
- Practical takeaway: In resource-constrained scenarios, instead of building complex adaptive retrieval pipelines, it is often better to directly employ simple methods like Mean Entropy.

## Rating

- Novelty: ⭐⭐⭐ The method itself has limited innovation; the core contribution lies in the comprehensive comparative framework and counterintuitive findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 35 methods, 6 datasets, 10 metrics, OOD analysis, and classifier complexity analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-organized paper with highly intuitive multi-dimensional visualization in Figure 1.
- Value: ⭐⭐⭐⭐⭐ Highly valuable systematic evaluation for the adaptive retrieval field; the findings have direct practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](../../AAAI2026/others/leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[ACL 2025\] Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity](cramming_tokens_embedding_capacity.md)

</div>

<!-- RELATED:END -->
