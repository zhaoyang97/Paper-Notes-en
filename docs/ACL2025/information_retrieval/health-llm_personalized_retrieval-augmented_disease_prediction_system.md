---
title: >-
  [Paper Note] Health-LLM: Personalized Retrieval-Augmented Disease Prediction System
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] The Health-LLM framework is proposed, which integrates feature score extraction from health reports via LLM + Llama Index, RAG-augmented medical knowledge retrieval, and CAAFE automated feature engineering with an XGBoost classifier. It achieves an Accuracy of 0.833 and an F1 score of 0.762 for disease prediction on the IMCS-21 Chinese telemedicine dataset, significantly outperforming GPT-4 few-shot+RAG (Acc 0.68) and fine-tuned LL…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Disease Prediction"
  - "Llama Index"
  - "Feature Engineering"
  - "Personalized Health Management"
date: 2026-05-08
content_hash: 46b66620f75480db
---

# Health-LLM: Personalized Retrieval-Augmented Disease Prediction System

**Conference**: ACL 2025  
**arXiv**: [2402.00746](https://arxiv.org/abs/2402.00746)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: RAG, Disease Prediction, Llama Index, Feature Engineering, Personalized Health Management

## TL;DR

The Health-LLM framework is proposed, which integrates feature score extraction from health reports via LLM + Llama Index, RAG-augmented medical knowledge retrieval, and CAAFE automated feature engineering with an XGBoost classifier. It achieves an Accuracy of 0.833 and an F1 score of 0.762 for disease prediction on the IMCS-21 Chinese telemedicine dataset, significantly outperforming GPT-4 few-shot+RAG (Acc 0.68) and fine-tuned LLaMA-2-13B (Acc 0.73).

## Background & Motivation

**Background**: LLMs have demonstrated great potential in the medical domain (e.g., GPT-4, AMIE). However, traditional health management methods are limited by static data and uniform standards, making it difficult to meet personalized needs. Directly employing LLMs for disease prediction (e.g., zero-shot GPT-4) yields limited accuracy ($< 40\%$).

**Limitations of Prior Work**: (1) Utilizing LLMs alone for clinical prediction lacks domain-specific, fine-grained feature extraction; (2) health report data is rich but challenging to convert into actionable prediction features; (3) LLMs lack a deep understanding of professional medical knowledge and require external knowledge augmentation.

**Key Challenge**: How to combine the language understanding capabilities of LLMs with the predictive accuracy of structured machine learning to achieve personalized disease prediction in medical scenarios that outperforms pure LLM or pure traditional methods?

**Goal**: Build a disease prediction system that combines LLM feature extraction and machine learning classification.

**Key Insight**: Instead of using the LLM as the final classifier, it is treated as an intelligent feature extractor—extracting structured scores from health reports via question-answering (QA) to serve as input features for XGBoost.

**Core Idea**: LLM for feature extraction + RAG-augmented knowledge + XGBoost for classification = disease prediction superior to pure LLM and pure traditional methods.

## Method

### Overall Architecture

The pipeline of Health-LLM consists of four steps: (1) leverages the in-context learning capability of LLMs to generate disease-symptom features in batches; (2) extracts feature scores ($0\text{-}1$ confidence) from health reports through Llama Index + RAG; (3) applies CAAFE for automated feature engineering optimization; and (4) performs XGBoost model training and disease prediction, complemented by LLM-generated personalized health advice.

### Key Designs

1. **LLM In-Context Learning for Symptom Feature Generation**:

    - **Function**: Enabling LLMs to batch-generate symptom description lists for various diseases via in-context learning.
    - **Mechanism**: Providing several "disease $\rightarrow$ symptom list" examples (e.g., "common cold $\rightarrow$ runny nose, sore throat, cough"). After learning the pattern, the LLM batch-generates symptom descriptions for additional diseases.
    - **Design Motivation**: Automatically constructing a disease-symptom knowledge base to avoid manually authoring feature lists for each disease.

2. **Llama Index + RAG Feature Scoring**:

    - **Function**: Splitting health report documents into text chunks, embedding and storing them in a vector database; designing 152 medical-related questions (e.g., "Does this person have good sleep habits?"), and retrieving relevant text chunks through Llama Index's search-then-synthesize pipeline to output a $0\text{-}1$ confidence score by the LLM.
    - **Mechanism**: The RAG mechanism retrieves the 3 most relevant pieces of information from a professional medical knowledge base to embed into the prompt, enhancing the domain knowledge of the LLM. The score of each question becomes a feature dimension for the downstream classifier, resulting in a 152-dimensional feature vector.
    - **Design Motivation**: Direct LLM answers may lack specialist knowledge, whereas RAG provides context to make scoring more accurate; this process transforms unstructured health reports into structured numerical features.

3. **CAAFE Automated Feature Engineering + XGBoost Classification**:

    - **Function**: Utilizing Context-Aware Automated Feature Engineering (CAAFE) to allow LLMs to automatically generate new features based on the dataset semantics, followed by multi-label disease classification (61 diseases) using XGBoost.
    - **Mechanism**: CAAFE leverages the LLM to understand the dataset context, iteratively generating semantically related derived features (e.g., combining multiple symptom scores into a comprehensive index). XGBoost outputs binary classifications ($0/1$), and some diseases support fine-grained grading (e.g., mild/severe fatty liver).
    - **Design Motivation**: Automated feature engineering overcomes the limitations of manual feature design, and XGBoost offers more stable and accurate classification than using the LLM directly.

### Interactive Health Consultation

Users can interact with the system in two ways: (1) submitting a health report to obtain predictions and recommendations; (2) describing symptoms through conversation, where the system records the conversation and makes predictions based on the dialogue content. Dialogue and recommendation generation are powered by GPT-4 Turbo.

## Key Experimental Results

### Main Results (IMCS-21 Chinese Telemedicine Dataset)

| Model/Method | Accuracy | F1 |
|----------|----------|-----|
| GPT-3.5 (zero-shot) | 0.333 | 0.361 |
| GPT-4 (zero-shot) | 0.390 | 0.312 |
| GPT-3.5 (few-shot + RAG) | 0.451 | 0.451 |
| TextCNN | 0.437 | 0.429 |
| RoBERT | 0.585 | 0.543 |
| GPT-4 (few-shot) | 0.620 | 0.671 |
| GPT-4 (few-shot + RAG) | 0.680 | 0.718 |
| Fine-tuned LLaMA-2-7B | 0.710 | 0.593 |
| Fine-tuned LLaMA-2-13B | 0.730 | 0.671 |
| **Health-LLM (Ours)** | **0.833** | **0.762** |

### Ablation Study

| Configuration | Accuracy | F1 |
|------|----------|-----|
| Health-LLM without Retrieval | 0.78 | 0.714 |
| Health-LLM without CAAFE | 0.77 | 0.721 |
| Health-LLM (Full) | 0.83 | 0.762 |

### Key Findings
- Health-LLM improves Accuracy by $+15.3\%$ compared to the strongest pure LLM solution (GPT-4 few-shot + RAG).
- RAG and CAAFE each contribute approximately $5\text{-}6\%$ to the Accuracy improvement, making both indispensable.
- Traditional text classification methods (TextCNN, RoBERT) perform significantly worse than LLM-based solutions, indicating that long-text understanding is critical.
- Although fine-tuned LLaMA-2 achieves higher Accuracy, its F1 score is lower, suggesting inferior generalization compared to Health-LLM.
- The system can cover 61 types of diseases, ranging from common ailments (cold, indigestion) to complex disorders (endocrine disorders).

## Highlights & Insights
- **Design paradigm of LLM as a feature extractor rather than a classifier**: Stepping away from the paradigm of "using LLMs directly for prediction," this approach converts the language understanding capabilities of LLMs into structured features for traditional ML. This LLM+ML hybrid paradigm provides valuable insights for various application domains.
- **Ingenious design of QA-based feature scoring**: Unstructured health reports are transformed into 152-dimensional numerical features via 152 medical questions, which leverages the semantic understanding of LLMs while preserving the interpretability and stability of ML.

## Limitations & Future Work
- Evaluated only on a single Chinese dataset (IMCS-21, 10 pediatric diseases), leading to questionable generalizability.
- Reliance on OpenAI APIs (GPT-4 Turbo) raises data privacy and cost concerns.
- Constructing the 152 questions and the disease-symptom knowledge base still requires the participation of domain experts.
- High system latency (multiple LLM calls + RAG retrieval) makes it unsuitable for real-time diagnostic scenarios.
- Only two ablation configurations were tested, without analyzing the independent contributions and interaction effects of each module.

## Related Work & Insights
- **vs CPLLM**: CPLLM directly uses fine-tuned LLMs for clinical prediction; Health-LLM uses LLMs for feature extraction and ML for prediction, achieving superior performance.
- **vs AMIE (Google)**: AMIE acts as an interactive diagnostic agent; Health-LLM focuses on batch prediction from report data.
- **vs fine-tuned LLaMA-2**: Fine-tuning requires large amounts of labeled data and necessitates re-training for each new task, whereas Health-LLM's QA-based featurization provides greater flexibility.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combined pipeline of LLM+RAG+XGBoost has practical value, though the individual components are not original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The baselines cover a wide range (traditional ML + LLM + fine-tuning), but are limited by a single dataset and insufficient ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The system description is clear, but preprocessing details for IMCS-21 are not sufficiently transparent.
- **Value**: ⭐⭐⭐⭐ Offers significant engineering reference value for the design of medical AI systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System](moc_mixtures_of_text_chunking_learners_for_retrieval-augmented_generation_system.md)
- [\[ACL 2025\] Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction](empaths_at_semeval-2025_task_11_retrieval-augmented_approach_to_perceived_emotio.md)
- [\[AAAI 2026\] PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](../../AAAI2026/information_retrieval/precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)

</div>

<!-- RELATED:END -->
