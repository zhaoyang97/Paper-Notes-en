---
title: >-
  [Paper Note] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection
description: >-
  [ACL 2026][fraud detection] This paper proposes FinFRE-RAG, a two-stage framework that serializes high-dimensional tabular transaction data into natural language via importance-guided feature reduction, and combines label-aware retrieval-augmented in-context learning to substantially improve F1/MCC of open-source LLMs on financial fraud detection, narrowing the performance gap with specialized tabular classifiers.
tags:
  - ACL 2026
  - fraud detection
  - tabular data
  - retrieval-augmented generation
  - feature selection
  - in-context learning
date: 2026-05-08
content_hash: f503a824e99eeb62
---

# Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection

**Conference**: ACL 2026
**arXiv**: [2512.13040](https://arxiv.org/abs/2512.13040)
**Code**: None
**Area**: Information Retrieval / Financial NLP
**Keywords**: fraud detection, tabular data, retrieval-augmented generation, feature selection, in-context learning

## TL;DR

This paper proposes FinFRE-RAG, a two-stage framework that serializes high-dimensional tabular transaction data into natural language via importance-guided feature reduction, and combines label-aware retrieval-augmented in-context learning to substantially improve F1/MCC of open-source LLMs on financial fraud detection, narrowing the performance gap with specialized tabular classifiers.

## Background & Motivation

**Background**: Financial fraud detection primarily relies on tabular models such as XGBoost and LightGBM, which require extensive feature engineering and offer limited interpretability. LLMs can generate human-readable explanations and assist in feature analysis, but perform poorly when directly applied to tabular fraud detection.

**Limitations of Prior Work**: (1) Tabular input mismatch — transaction data consists of high-dimensional numerical and categorical features, whereas LLMs are pretrained on natural language and handle the semantics and numerical precision of structured features poorly; (2) Fraud ambiguity and scarcity — fraud definitions vary across institutions, products, and regions, and fraud rates are extremely low (<1%), making it difficult for LLMs to identify subtle discriminative patterns.

**Key Challenge**: LLMs possess reasoning and interpretability generation capabilities, but lack knowledge of "what constitutes fraud" — they must be taught which feature patterns are associated with fraudulent behavior.

**Goal**: Design a fine-tuning-free framework that enables LLMs to understand and detect tabular financial fraud through feature reduction and retrieval augmentation.

**Key Insight**: Reframe fraud detection as an instance-based reasoning problem — by retrieving semantically similar historical transactions as few-shot examples, the LLM reasons by analogy to make predictions.

**Core Idea**: Offline feature reduction (ranking features via random forest to retain the top-$k$) combined with online retrieval-augmented in-context learning (categorical filtering → numerical similarity search → natural language serialization).

## Method

### Overall Architecture

FinFRE-RAG operates in two stages: (1) offline feature reduction — a random forest is trained to extract feature importance rankings, the top-$k$ features are retained, and normalized representations are precomputed; (2) online retrieval-augmented inference — for each query transaction, candidates are first filtered by categorical attributes, then the nearest neighbors are retrieved by numerical cosine similarity, and the retrieved examples together with the query are serialized into a natural language prompt, from which the LLM outputs a 5-point risk score.

### Key Designs

1. **Importance-Guided Feature Reduction**:

    - Function: Compress high-dimensional tabular data into a compact representation processable by LLMs.
    - Mechanism: A random forest is trained on an external dataset (without hyperparameter tuning) to extract feature importance rankings; the top-$k$ features (default $k=10$) are retained. Z-score normalization is precomputed for all numerical features for subsequent retrieval.
    - Design Motivation: Rather than optimizing the classifier itself, this approach obtains a coarse but effective feature ranking at minimal cost; reducing prompt length avoids exceeding the context window and removes noisy features so that the LLM focuses on the most informative attributes.

2. **Hybrid Retrieval Strategy (Categorical Filtering + Numerical Similarity)**:

    - Function: Identify historical transactions that are structurally and numerically most similar to each query.
    - Mechanism: Equality constraints are progressively added based on importance-ranked categorical attributes (with a fallback mechanism to avoid empty candidate sets); within the candidate pool, the top-$n$ nearest neighbors (default $n=20$) are retrieved by cosine similarity over normalized feature vectors. Retrieved results are organized by label distribution as few-shot examples.
    - Design Motivation: Categorical filtering ensures structural-semantic consistency (e.g., same transaction type), while numerical similarity provides fine-grained matching — leveraging the LLM's strength in analogical reasoning.

3. **Natural Language Serialization and Risk Scoring**:

    - Function: Transform tabular data into natural language understandable by LLMs and obtain fine-grained predictions.
    - Mechanism: Features are embedded into natural language templates (rather than simple key-value lists), and each retrieved example is accompanied by a label description. The LLM outputs a 5-point risk score (Score $\geq 4$ is treated as fraud) rather than a direct binary classification.
    - Design Motivation: A 5-point scoring scheme provides a more granular risk signal than binary classification, allowing the LLM to express uncertainty rather than being forced into a hard decision.

### Loss & Training

FinFRE-RAG requires no LLM training. Six open-source models are evaluated: Qwen3-14B/80B, Gemma 3-12B/27B, and GPT-OSS-20B/120B. The feature reduction stage trains a simple random forest. Fine-tuning comparisons employ LoRA.

## Key Experimental Results

### Main Results

**FinFRE-RAG vs. Direct Prompting (F1 Improvement, Gemma 3-12B)**

| Dataset | Direct Prompting F1 | + FinFRE-RAG F1 | Gain |
|---------|---------------------|-----------------|------|
| ccf | 0.00 | 0.79 | +0.79 |
| ccFraud | 0.13 | 0.59 | +0.46 |
| IEEE-CIS | 0.01 | 0.59 | +0.58 |
| PaySim | 0.00 | 0.71 | +0.71 |

### Ablation Study

**Effect of Feature Count and Retrieval Count**

| Configuration | Description |
|---------------|-------------|
| $k=10$ features | Optimal balance; more features introduce noise |
| $n=20$ neighbors | Sufficient to provide analogical reasoning context |
| 5-point scoring | Outperforms binary classification output |

### Key Findings

- Under direct prompting, LLMs perform near random (F1 $\approx$ 0); FinFRE-RAG improves F1 to 0.5–0.8.
- Gemma 3-12B + FinFRE-RAG achieves competitive performance against XGBoost/LightGBM on multiple datasets.
- LLM fine-tuning (LoRA) underperforms FinFRE-RAG in most settings, indicating that ICL is more suitable than parameter updating for this task.
- The 5-point risk scoring consistently outperforms binary classification, as the model can express uncertainty.
- Despite approaching the performance of specialized classifiers, the distinctive value of LLMs lies in generating interpretable fraud analysis rationales.

## Highlights & Insights

- Reframing financial fraud detection as "instance-based reasoning" rather than "classification" fully exploits LLMs' capacity for analogical reasoning.
- No LLM parameter training is required; the approach relies entirely on ICL, which is well-suited to the strict data privacy requirements of the financial domain.
- The two-stage design combining feature reduction and retrieval augmentation is highly generalizable and can be extended to other tabular data tasks.

## Limitations & Future Work

- LLMs still lag behind specialized tabular classifiers (e.g., XGBoost), particularly in large-scale, high-dimensional settings.
- Feature reduction relies on random forest importance rankings, which may not generalize to all data distributions.
- Evaluation is conducted on only four public datasets; validation on real production-level fraud detection systems is absent.
- The threshold for the 5-point scoring scheme (Score $\geq 4$) is fixed and no optimal threshold search is performed.

## Related Work & Insights

- **vs. XGBoost/LightGBM**: Specialized classifiers retain an advantage in pure predictive performance but provide no interpretable rationales.
- **vs. Financial LLMs (FinGPT, etc.)**: Domain-specific LLMs are based on older architectures with insufficient instruction-following capabilities.
- **vs. Direct LLM Prompting**: Without feature selection and retrieval, LLMs fail almost completely, demonstrating the necessity of the proposed framework.

## Rating

- Novelty: ⭐⭐⭐ The framework is relatively straightforward; the combination of RAG applied to tabular data is moderately novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 6 models + fine-tuning comparisons + multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear; experiments follow a research-question-driven design.
- Value: ⭐⭐⭐⭐ Provides a practical baseline methodology for LLM applications in the financial domain.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)

<!-- RELATED:END -->
