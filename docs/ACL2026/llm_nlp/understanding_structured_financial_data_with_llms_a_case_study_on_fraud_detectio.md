---
title: >-
  [Paper Note] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection
description: >-
  [ACL 2026][LLM/NLP][Fraud Detection] This paper proposes FinFRE-RAG, a two-stage framework that serializes high-dimensional tabular transaction data into natural language through importance-guided feature reduction and c…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Fraud Detection"
  - "Tabular Data"
  - "Retrieval-Augmented Generation"
  - "Feature Selection"
  - "In-Context Learning"
date: 2026-05-08
content_hash: 09650ea55e612df3
---

# Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection

**Conference**: ACL 2026  
**arXiv**: [2512.13040](https://arxiv.org/abs/2512.13040)  
**Code**: None  
**Area**: Information Retrieval / Financial NLP  
**Keywords**: Fraud Detection, Tabular Data, Retrieval-Augmented Generation, Feature Selection, In-Context Learning

## TL;DR

This paper proposes FinFRE-RAG, a two-stage framework that serializes high-dimensional tabular transaction data into natural language through importance-guided feature reduction and combines it with label-aware retrieval-augmented in-context learning. This significantly improves the F1/MCC of open-source LLMs in financial fraud detection, narrowing the performance gap with specialized tabular classifiers.

## Background & Motivation

**Background**: Financial fraud detection primarily relies on tabular models such as XGBoost and LightGBM, which require extensive feature engineering and offer limited interpretability. While LLMs can generate human-readable explanations and assist in feature analysis, they perform poorly when directly applied to tabular fraud detection.

**Limitations of Prior Work**: (1) Tabular input mismatch—transaction data consists of high-dimensional tables with numerical/categorical features, whereas LLMs are pre-trained on natural language and struggle with the semantics of structured features and numerical precision; (2) Fraud ambiguity and scarcity—fraud definitions vary across institutions, products, and regions, and fraud ratios are extremely low (<1%), making it difficult for LLMs to identify subtle discriminatory patterns.

**Key Challenge**: LLMs possess the potential for reasoning and generating interpretable analyses but lack knowledge of "what constitutes fraud"—it is necessary to teach LLMs which feature patterns correlate with fraudulent behavior.

**Goal**: Design a tuning-free framework that enables LLMs to understand and detect tabular financial fraud through feature reduction and retrieval augmentation.

**Key Insight**: Reframe fraud detection as an instance-based reasoning problem—using retrieved semantically similar historical transactions as few-shot examples to allow the LLM to judge via analogical reasoning.

**Core Idea**: Offline feature reduction (utilizing Random Forest ranking to retain top-k features) + Online retrieval-augmented in-context learning (categorical filtering → numerical similarity search → natural language serialization).

## Method

### Overall Architecture

FinFRE-RAG consists of two stages: (1) Offline feature reduction—training a Random Forest to extract feature importance rankings, retaining top-k features, and pre-computing normalized representations; (2) Online retrieval-augmented inference—for each query transaction, candidates are first filtered by categorical attributes and then searched for nearest neighbors based on numerical cosine similarity. The retrieved results and the query are serialized into a natural language prompt, and the LLM outputs a 5-point risk score.

### Key Designs

1. **Importance-guided Feature Reduction**:
    - **Function**: Compresses high-dimensional tabular data into a compact representation processable by LLMs.
    - **Mechanism**: Trains a Random Forest on external datasets (without hyperparameter optimization) to extract feature importance rankings, retaining the top-k (default $k=10$) features. Pre-computes z-score normalization for all numerical features for subsequent retrieval.
    - **Design Motivation**: Aims for a rough but effective feature ranking at minimal cost rather than an optimal classifier; reduces prompt length to avoid exceeding context windows; removes noisy features to focus the LLM on the most informative attributes.

2. **Hybrid Retrieval Strategy (Categorical Filtering + Numerical Similarity)**:
    - **Function**: Finds historical transactions most similar in structure and value to each query.
    - **Mechanism**: Progressively adds equality constraints based on importance-ranked categorical attributes (with a fallback mechanism to avoid empty sets), then retrieves top-n (default $n=20$) nearest neighbors based on the cosine similarity of normalized feature vectors within the candidate pool. Retrieval results are organized into few-shot examples by label distribution.
    - **Design Motivation**: Categorical filtering ensures structural semantic consistency (e.g., same transaction type), while numerical similarity provides fine-grained matching—leveraging the LLM's strength in analogical reasoning.

3. **Natural Language Serialization and Risk Scoring**:
    - **Function**: Transforms tabular data into natural language understandable by LLMs and obtains fine-grained predictions.
    - **Mechanism**: Features are embedded into natural language templates (rather than simple key-value lists), and each retrieval example includes a label description. The LLM outputs a 5-point risk score (Score $\geq 4$ is considered fraud) instead of a direct binary classification.
    - **Design Motivation**: The 5-point scale provides finer-grained risk signals than binary classification, allowing the LLM to express uncertainty rather than being forced into a hard decision.

### Loss & Training

FinFRE-RAG does not require training the LLM. Six open-source models are used: Qwen3-14B/80B, Gemma 3-12B/27B, and GPT-OSS-20B/120B. A simple Random Forest is trained during the feature reduction stage. LoRA is used for fine-tuning comparisons.

## Key Experimental Results

### Main Results

**FinFRE-RAG vs. Direct Prompting (F1 Gain, Gemma 3-12B)**

| Dataset | Direct Prompting F1 | + FinFRE-RAG F1 | Gain |
|---------|--------------------|-----------------|------|
| ccf | 0.00 | 0.79 | +0.79 |
| ccFraud | 0.13 | 0.59 | +0.46 |
| IEEE-CIS | 0.01 | 0.59 | +0.58 |
| PaySim | 0.00 | 0.71 | +0.71 |

### Ablation Study

**Impact of Feature Count and Retrieval Count**

| Configuration | Description |
|---------------|-------------|
| $k=10$ Features | Optimal balance point; more features introduce noise |
| $n=20$ Neighbors | Sufficient for providing analogical reasoning context |
| 5-point Scoring | Outperforms binary classification output |

### Key Findings

- Under direct prompting, LLMs perform close to random guessing (F1 $\approx 0$); with FinFRE-RAG, F1 improves to 0.5–0.8.
- Gemma 3-12B + FinFRE-RAG is competitive with XGBoost/LightGBM across multiple datasets.
- LLM fine-tuning (LoRA) is inferior to FinFRE-RAG in most settings, suggesting that ICL is more suitable than parameter updates for this task.
- The 5-point risk score consistently outperforms binary classification as it allows the model to express uncertainty.
- Despite performance nearing specialized classifiers, the unique value of LLMs lies in generating interpretable fraud analysis rationales.

## Highlights & Insights

- Reframes financial fraud detection as "instance-based reasoning" rather than "classification"—fully exploiting the analogical reasoning capabilities of LLMs.
- Requires no training of LLM parameters and relies entirely on ICL, making it suitable for strict data privacy requirements in the financial sector.
- The two-stage design of feature reduction + retrieval augmentation is highly generalizable and can be extended to other tabular data tasks.

## Limitations & Future Work

- LLMs still lag behind specialized tabular classifiers (e.g., XGBoost), especially in large-scale, high-dimensional scenarios.
- Feature reduction depends on Random Forest importance rankings, which may not suit all data distributions.
- Evaluated only on 4 public datasets; not yet validated in real-world production-level fraud detection systems.
- The threshold for the 5-point score (Score $\geq 4$) is fixed without an optimal threshold search.

## Related Work & Insights

- **vs. XGBoost/LightGBM**: Specialized classifiers still hold an advantage in pure predictive performance but do not provide interpretable reasons.
- **vs. Financial LLMs (FinGPT, etc.)**: Domain-specific LLMs based on older architectures lack sufficient instruction-following capabilities.
- **vs. Direct LLM Prompting**: LLMs fail almost completely without feature selection and retrieval, proving the necessity of the framework.

## Rating

- Novelty: ⭐⭐⭐ The framework approach is relatively direct; the combination of applying RAG to tabular data is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 6 models + fine-tuning comparison + multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem analysis and RQ-driven experimental design.
- Value: ⭐⭐⭐⭐ Provides a practical baseline method for LLM applications in the financial domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Study of LLMs' Preferences for Libraries and Programming Languages](a_study_of_llms39_preferences_for_libraries_and_programming_languages.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[AAAI 2026\] TEMPLE: Incentivizing Temporal Understanding of Video LLMs via Progressive Pre-SFT Alignment](../../AAAI2026/llm_nlp/temple_incentivizing_temporal_understanding_of_video_large_language_models_via_p.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](../../ICML2026/llm_nlp/token-efficient_change_detection_in_llm_apis.md)

</div>

<!-- RELATED:END -->
