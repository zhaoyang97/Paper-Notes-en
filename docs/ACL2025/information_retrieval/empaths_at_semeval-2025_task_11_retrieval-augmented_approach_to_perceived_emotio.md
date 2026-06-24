---
title: >-
  [Paper Note] Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction
description: >-
  [ACL 2025 (SemEval Workshop)][Information Retrieval & RAG][Emotion Detection] This paper proposes the EmoRAG system, which combines a retrieval-augmented generation (RAG) pipeline with multi-LLM ensemble aggregation. Without any additional training, it achieves competitive results across 28 languages on the SemEval-2025 Task 11 multi-label emotion detection task, with an average F1-micro score of 0.638.
tags:
  - "ACL 2025 (SemEval Workshop)"
  - "Information Retrieval & RAG"
  - "Emotion Detection"
  - "RAG"
  - "Multilingual"
  - "LLM Ensembling"
  - "Multi-label Classification"
date: 2026-05-08
content_hash: 697032a37a9d5509
---

# Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction

**Conference**: ACL 2025 (SemEval Workshop)  
**arXiv**: [2506.04409](https://arxiv.org/abs/2506.04409)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Emotion Detection, RAG, Multilingual, LLM Ensembling, Multi-label Classification

## TL;DR

This paper proposes the EmoRAG system, which combines a retrieval-augmented generation (RAG) pipeline with multi-LLM ensemble aggregation. Without any additional training, it achieves competitive results across 28 languages on the SemEval-2025 Task 11 multi-label emotion detection task, with an average F1-micro score of 0.638.

## Background & Motivation

SemEval-2025 Task 11 focuses on **perceived emotion detection**: determining which emotions the majority of readers would infer the speaker is experiencing from a given text (joy, sadness, fear, anger, surprise, disgust, neutral). This does not analyze the reader's evoked emotions, nor does it infer the speaker's true emotions; instead, it focuses on the **socially consensual understanding** of emotions.

**Key Challenge**:

**Multilingual Coverage**: Covers 28 languages, including many low-resource languages (Hausa, Kinyarwanda, Emakhuwa, isiZulu, etc.).

**Multi-label Task**: Each text segment may contain multiple emotions simultaneously.

**Cultural Differences**: Significant differences exist in emotion expression and interpretation across different cultural backgrounds.

**Traditional methods** (finetuning pre-trained Transformers + linear classification heads) perform well in monolingual scenarios, but cross-lingual generalization faces additional challenges from linguistic variability and cultural differences.

**Mechanism of EmoRAG**: Utilizing the training-free nature of RAG, the annotated training data is used as a retrieval corpus. This allows the model to refer to relevant emotion instances during inference, thereby improving cross-lingual and cross-cultural robustness.

## Method

### Overall Architecture

EmoRAG consists of four components connected in series: Database → Retriever → Generators (LLM Ensemble) → Aggregation Model.

### Key Designs

1. **Database**:

    - Built directly using annotated training data.
    - Based on the BRIGHTER dataset (multi-label emotion annotations in 28 languages) and the EthioEmo dataset (4 Ethiopian languages).

2. **Retriever**:

    - Comparison of two retrievers:
        - **n-gram Retriever** (from the LangChain module): Hypothesized to be better for low-resource languages because it relies on surface text features.
        - **BGE-M3 Sentence Embedding Retriever**: A multilingual embedding model.
    - K-value setting: K=30 for low-resource languages (due to high token consumption), K=100 for high-resource languages.
    - Retrieved samples are used as few-shot prompts for the LLMs.

3. **Generators (Decoder Models)**:

    - Uses an ensemble of four LLMs:
        - Llama-3.1-70B
        - Qwen2.5-72B-Instruct
        - gpt-4o-mini
        - gemma-2-27b-it
    - System prompts are all in English (experiments showed that English prompts outperform target language prompts).
    - Each model independently outputs emotion predictions.

4. **Aggregation Strategies**:

    - **Single Model**: Direct use of a single model's output.
    - **Majority Vote**: The prediction for each label is determined by a majority vote of all models.
    - **Macro/Micro Weighted Vote**: Weighted by the model's F1 score on the development set.
    - **Label-F1 Weighted Vote**: Weighted independently for each label, based on the F1 score of that label across different models.
    - **GPT-4o Aggregation**: Feeds all model results and few-shot examples into gpt-4o-mini for aggregation.

### Loss & Training

- **Training-Free**: No model finetuning is performed; it relies entirely on retrieval and generation during inference.
- The optimal aggregation strategy and retriever configuration are determined solely on the dev set.

## Key Experimental Results

### Main Results — Performance on Representative Languages in Test Set (Table)

| Language | Best Model | Dev F1-micro | Dev F1-macro | Test F1-micro | Test F1-macro |
|------|----------|-------------|-------------|--------------|--------------|
| English | L-F1 Vote | 0.821 | 0.818 | 0.807 | 0.789 |
| Spanish | L-F1 Vote | 0.813 | 0.809 | 0.820 | 0.817 |
| Russian | L-F1 Vote | 0.880 | 0.880 | 0.883 | 0.879 |
| Hindi | L-F1 Vote | 0.842 | 0.849 | 0.866 | 0.866 |
| Marathi | L-F1 Vote | 0.943 | 0.947 | 0.856 | 0.864 |
| Hausa | L-F1 Vote | 0.735 | 0.731 | 0.704 | 0.695 |
| Swahili | L-F1 Vote | 0.440 | 0.409 | 0.430 | 0.386 |
| Emakhuwa | gpt-4o-mini | 0.300 | 0.211 | 0.256 | 0.216 |

### Average Performance of Each Model on Dev Set (Table)

| Model/Strategy | F1-micro | F1-macro |
|-----------|----------|----------|
| llama-3.1-70b | 0.563 | 0.515 |
| qwen2.5-70b | 0.590 | 0.556 |
| gpt-4o-mini | 0.631 | 0.590 |
| gpt-4o-mini + n-gram | 0.641 | 0.601 |
| gemma-2-27b | 0.617 | 0.576 |
| majority_vote | 0.661 | 0.617 |
| **majority_vote_by_label_f1** | **0.678** | **0.634** |

### Key Findings

- **Label-F1 Weighted Vote** is the optimal aggregation strategy for the vast majority of languages (the best choice for 21 out of 28 languages).
- High-resource languages perform well (Russian 0.883, Hindi 0.866), while performance on low-resource languages varies significantly (Emakhuwa is only 0.256, Tigrinya is 0.260).
- The n-gram retriever is more effective for some low-resource languages (Oromo, Sundanese, Mandarin, and Kinyarwanda selected the n-gram configuration).
- gpt-4o-mini performs best among single models (F1-micro 0.631), but the ensemble aggregation (0.678) significantly outperforms any single model.
- English system prompts perform better than target-language prompts, possibly because LLMs have stronger English instruction-following capabilities.
- Some languages show a large disparity between dev and test performance (German: dev 0.745 $\rightarrow$ test 0.269; Brazilian Portuguese: dev 0.766 $\rightarrow$ test 0.481), suggesting distribution shift issues.

## Highlights & Insights

- **Zero-Training Paradigm**: The RAG + LLM ensemble completely bypasses finetuning, making it attractive for resource-constrained scenarios.
- **Label-F1 Weighting Strategy** is more fine-grained than simple majority voting or global weighting: different models perform optimally for different emotion labels.
- The choice of n-gram vs. embedding retriever reveals distinct characteristics of low-resource vs. high-resource languages: the embedding quality of low-resource languages is poor, making surface-level feature matching potentially more reliable.
- Strong complementarity among multiple LLMs: individual models excel on different languages.

## Limitations & Future Work

- Only covers 6 basic emotions plus neutral; finer-grained emotion classification was not tested.
- Insufficient handling of highly imbalanced class distributions and significant distribution shifts (e.g., German, Portuguese).
- The configuration of K values (30 vs. 100) is somewhat coarse, with no systematic hyperparameter search conducted.
- High inference cost (4 large models + retrieval), making it unsuitable for low-latency scenarios.
- No direct comparison with finetuning baselines, making it difficult to judge whether the RAG approach is truly superior to finetuning.

## Related Work & Insights

- The limitations of traditional methods (finetuning Transformers + classification heads) on multilingual emotion tasks serve as the starting point of Ours.
- The success of RAG in knowledge-intensive NLP tasks is extended to the field of emotion classification.
- The BRIGHTER dataset covers emotion annotations for 28 languages for the first time, providing a foundation for multilingual emotion research.
- This echoes LLM-based approaches to cross-lingual emotion detection, but Ours emphasizes practicality and scalability.

## Rating

- **Novelty**: 5/10 — The combination of RAG and LLM ensembles is an application of mature paradigms and lacks technological innovation.
- **Experimental Thoroughness**: 7/10 — Comprehensive coverage of 28 languages, but lacks a comparison with finetuning baselines.
- **Writing Quality**: 6/10 — The content is clearly organized but short, lacking in-depth analysis.
- **Value**: 6/10 — As a SemEval system description paper, it demonstrates the feasibility of RAG in multilingual emotion analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Health-LLM: Personalized Retrieval-Augmented Disease Prediction System](health-llm_personalized_retrieval-augmented_disease_prediction_system.md)
- [\[NeurIPS 2025\] RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition](../../NeurIPS2025/information_retrieval/rmit-adms_at_the_mmu-rag_neurips_2025_competition.md)
- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets](genie_worksheets_tod_agent.md)
- [\[NeurIPS 2025\] Learning Task-Agnostic Representations through Multi-Teacher Distillation](../../NeurIPS2025/information_retrieval/learning_task-agnostic_representations_through_multi-teacher_distillation.md)

</div>

<!-- RELATED:END -->
