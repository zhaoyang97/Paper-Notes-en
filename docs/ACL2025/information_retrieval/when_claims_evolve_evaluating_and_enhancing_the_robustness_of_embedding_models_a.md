---
title: >-
  [Paper Note] When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits
description: >-
  [ACL 2025 Findings][Information Retrieval & RAG][Fact-checking] A perturbation framework is proposed to systematically evaluate the robustness of sentence embedding models when handling edited misinformation claims. Standard embedding models exhibit significant performance degradation, which can be mitigated using two approaches: knowledge distillation and claim normalization, bringing up to a 17 percentage point improvement in in-domain robustness and a 10 percentage point i…
tags:
  - "ACL 2025 Findings"
  - "Information Retrieval & RAG"
  - "Fact-checking"
  - "claim matching"
  - "embedding model robustness"
  - "misinformation variants"
  - "retrieval systems"
date: 2026-05-08
content_hash: 671c44e62c74c54c
---

# When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits

**Conference**: ACL 2025 Findings  
**arXiv**: [2503.03417](https://arxiv.org/abs/2503.03417)  
**Code**: [https://github.com/JabezNzomo99/claim-matching-robustness](https://github.com/JabezNzomo99/claim-matching-robustness)  
**Area**: Information Retrieval  
**Keywords**: Fact-checking, claim matching, embedding model robustness, misinformation variants, retrieval systems

## TL;DR

A perturbation framework is proposed to systematically evaluate the robustness of sentence embedding models when handling edited misinformation claims. Standard embedding models exhibit significant performance degradation, which can be mitigated using two approaches: knowledge distillation and claim normalization, bringing up to a 17 percentage point improvement in in-domain robustness and a 10 percentage point improvement in cross-domain generalization.

## Background & Motivation

**Background**: Fact-checkers increasingly rely on automated "claim matching" systems—when a new claim arises, the system uses a sentence embedding model to retrieve the most relevant fact-check from a database of verified claims, helping checkers quickly determine if the claim has already been debunked. This pipeline typically consists of two stages: a first-stage preliminary retrieval (recall) using embedding models, and a second-stage fine ranking using a reranker.

**Limitations of Prior Work**: Web users often make various edits to the original claim when spreading misinformation—rewriting the phrasing (rewrite), adding negation (negation), introducing typos (typos), amplifying or minimizing the tone (amplify/minimize), substituting entity names (entity replacement), or even converting to dialects (dialect). Although these edited claims still semantically refer to the same fact-check, whether embedding models can maintain robust retrieval performance on these variations remains unclear.

**Key Challenge**: Embedding models are trained to map semantically similar sentences to nearby locations in the vector space, but subtle textual edits (e.g., adding a negation word or using a dialect) can cause vector shifts, causing claims that should match to "drift" to unretrievable locations in the vector space.

**Goal**: (1) To design a systematic perturbation framework to generate plausible and natural claim variants; (2) to evaluate the robustness of mainstream embedding models under diverse edit types; and (3) to propose training-time and inference-time mitigation techniques to enhance robustness.

**Key Insight**: Utilize GPT-4o as a "perturbator" to generate claim variants, verify that these variations still point to the same fact-check, and then systematically evaluate embedding models on these verified variants.

**Core Idea**: Automatically generate multiple types of claim editing variants using an LLM to stress-test embedding models, followed by training more robust embedding models via knowledge distillation, and applying inference-time claim normalization using LLMs to eliminate noise.

## Method

### Overall Architecture

The methodology is divided into three stages: (1) Perturbation generation—editing original claims using GPT-4o with multiple edit types and verifying them; (2) robustness evaluation—evaluating multiple models in a multi-stage retrieval pipeline (embedding retrieval + reranking); and (3) mitigation methods—improving robustness both at training time (knowledge distillation) and inference time (claim normalization). The evaluation utilizes two in-domain datasets, CheckThat22 and FactCheckTweet, and one OOD dataset.

### Key Designs

1. **LLM-driven Perturbation Framework**:

    - **Function**: Automatically and systematically generates diverse, semantic-preserving claim variants.
    - **Mechanism**: GPT-4o is used to generate five types of editing variants for each claim, including rewrite, negation, typos, amplify/minimize, entity replacement, and dialect. This is followed by a verification step using GPT-4o to check whether each variant still corresponds to the original fact-check, conveys the same core claim, and reads naturally. Validated variants are retained, and two subsets—"baseline" (minimal edit) and "worst-case" (maximal edit)—are selected for evaluation.
    - **Design Motivation**: Manually writing claim variants is unscalable and fails to systematically cover all edit types. The two-step process of LLM generation followed by LLM verification ensures both diversity and quality. Distinguishing between baseline and worst-case allows the evaluation of models under different editing intensities.

2. **Multi-Stage Retrieval Pipeline Robustness Evaluation**:

    - **Function**: Comprehensively evaluates the robustness of embedding models in real-world deployment scenarios.
    - **Mechanism**: The evaluation utilizes a two-stage retrieval pipeline. The first stage uses 10+ embedding models (including all-mpnet-base-v2, all-MiniLM-L12-v2, sentence-t5, INSTRUCTOR, NV-Embed-v2, SFR-Embedding-Mistral, etc.) to retrieve top-k candidates, and the second stage uses rerankers (like BGE-LLM) to fine-rank them. Metrics like MAP@k and Recall@k are calculated for both original and edited claims, with performance gaps ($\Delta$) measuring robustness. BM25 is also included as a sparse retrieval baseline.
    - **Design Motivation**: Evaluating embedding models in isolation is insufficient; they must be assessed within the complete pipeline because rerankers might compensate for losses in the first stage. In practice, these two stages are typically deployed sequentially.

3. **Training-time Mitigation: Robustness Knowledge Distillation**:

    - **Function**: Fine-tunes embedding models to be more robust against claim edits.
    - **Mechanism**: "Original-variant" claim pairs generated by the perturbation framework are used as training data for knowledge distillation of the embedding model. The teacher model's embeddings of the original claims serve as targets, and the student model is trained to align its embeddings of the edited claims to the teacher's outputs. The training set consists of approximately 70K claim pairs (or a lite version of 11.5K pairs).
    - **Design Motivation**: Direct fine-tuning on edited claims may lead to overfitting to specific edit types. Knowledge distillation encourages the model to learn to "ignore surface edits while preserving semantics," yielding better generalization.

### Inference-time Mitigation: Claim Normalization

During inference, GPT-4o is used to normalize noisy input claims (containing typos, dialects, etc.) into a standard format before feeding them into the retrieval pipeline. This training-free method only adds a preprocessing step on the input side.

## Key Experimental Results

### Main Results

| Embedding Model | Original MAP@5 | Edited MAP@5 | Δ (pp) | Post-Rerank Δ |
|---------|-----------|-------------|--------|-----------|
| all-mpnet-base-v2 | 78.4 | 64.2 | -14.2 | -8.7 |
| all-MiniLM-L12-v2 | 74.8 | 59.1 | -15.7 | -9.3 |
| sentence-t5-large | 76.2 | 63.5 | -12.7 | -7.8 |
| INSTRUCTOR-large | 79.1 | 68.3 | -10.8 | -6.5 |
| NV-Embed-v2 | 83.6 | 75.1 | -8.5 | -4.2 |
| SFR-Embedding-Mistral | 85.2 | 78.4 | -6.8 | -3.5 |
| BM25 | 62.3 | 48.7 | -13.6 | - |

### Mitigation Performance

| Method | In-domain Δ Improvement (pp) | Cross-domain Δ Improvement (pp) | Description |
|-----|-------------------|-----------------|------|
| No Mitigation (baseline) | 0 | 0 | Original performance degradation |
| Knowledge Distillation (full, 70K) | +17.0 | +10.0 | Training-time method, performs the best |
| Knowledge Distillation (lite, 11.5K) | +12.3 | +7.2 | Lightweight training data is also effective |
| Claim Normalization (GPT-4o) | +9.5 | +6.8 | Inference-time method, no retraining required |
| Reranker (BGE-LLM) | +5.8 | +3.4 | Mitigates but cannot fully compensate |

### Key Findings

- LLM-distilled embedding models (e.g., NV-Embed-v2, SFR-Embedding-Mistral) are inherently more robust than traditional embedding models, but at higher computational costs.
- Rerankers can partially recover the performance loss from the first stage, but they cannot fully compensate—if the correct result is missed during the first-stage recall, the reranker is powerless.
- Knowledge distillation is the most effective mitigation method, yielding a 17pp improvement in-domain and a 10pp improvement cross-domain.
- Among the different edit types, negation impacts embedding models the most, as it directly alters the semantic direction; dialect edits also exhibit a surprisingly large negative impact.

## Highlights & Insights

- **Employing LLMs for both attack (generating perturbations) and defense (claim normalization)** is a clever design—using the same tool to build a complete evaluation-and-resolution framework.
- The knowledge distillation method is simple yet remarkably effective, and its training data can be auto-generated by the perturbation framework, establishing a self-sustaining "generate perturbations $\rightarrow$ train robust models" loop.
- The discovery that "rerankers cannot fully compensate for first-stage losses" has valuable practical implications—highlighting that deploying a fact-checking system requires focusing not just on reranker accuracy but also on the robustness of first-stage retrieval.

## Limitations & Future Work

- Perturbation generation relies on GPT-4o, which is costly and constrained by model capabilities.
- Only English claims were evaluated; claim-editing patterns in multilingual settings may differ.
- The OOD dataset is sourced from a single origin (Meedan WhatsApp Tiplines), requiring further validation for cross-domain generalization conclusions.
- Knowledge distillation requires separate training for each embedding model, causing deployment costs to scale linearly with the number of models.

## Related Work & Insights

- **vs Adversarial NLI/TextFooler**: These works focus on adversarial attacks on text classification models, whereas this work targets the robustness of retrieval systems—possessing different adversarial goals (classification flip vs retrieval failure) and perturbation methods closer to real-user behaviors.
- **vs CheckThat Lab**: While the CheckThat series evaluations contain claim-matching tasks, they do not consider robustness issues induced by claim editing, a gap this paper addresses.
- **vs Sentence-BERT AdvTraining**: Prior adversarial training on embedding models focused primarily on synonym replacement; this work systematically covers 6+ edit types for more comprehensive evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically evaluate the robustness of embedding models to multiple edit types in claim-matching scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 10+ embedding models, 6 edit types, 3 datasets, and various mitigation methods.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, systematic experimental design, and practical insights.
- Value: ⭐⭐⭐⭐ Directly beneficial for the practical deployment of fact-checking systems, with open-sourced code and data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](../../ACL2026/information_retrieval/quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](length-induced_embedding_collapse_in_plm-based_models.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](semantic_outlier_removal_with_embedding_models_and_llms.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)

</div>

<!-- RELATED:END -->
