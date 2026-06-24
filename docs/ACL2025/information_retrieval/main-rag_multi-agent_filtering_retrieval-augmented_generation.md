---
title: >-
  [Paper Note] MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][Multi-Agent Filtering] This paper proposes MAIN-RAG, a training-free multi-agent RAG filtering framework that collaborates via three LLM agents (Predictor $\rightarrow$ Judge $\rightarrow$ Final-Predictor) to evaluate the relevance of retrieved documents. It designs an adaptive threshold (based on the score mean and standard deviation) to dynamically filter noisy documents, achieving a 2-11% accuracy improvement across 4 QA benchmarks.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Multi-Agent Filtering"
  - "Document Noise"
  - "Adaptive Threshold"
  - "Training-Free"
  - "Relevance Scoring"
date: 2026-05-08
content_hash: 8b8b26e5938fa991
---

# MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2501.00332](https://arxiv.org/abs/2501.00332)  
**Code**: Unreleased  
**Authors**: Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, Menghai Pan, Chin-Chia Michael Yeh, Guanchu Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahashweta Das, Na Zou  
**Affiliations**: Texas A&M University, Visa Research, WPI, University of Utah, University of Houston  
**Area**: Retrieval-Augmented Generation (RAG)  
**Keywords**: Multi-Agent Filtering, Document Noise, Adaptive Threshold, Training-Free, Relevance Scoring

## TL;DR

This paper proposes MAIN-RAG, a training-free multi-agent RAG filtering framework that collaborates via three LLM agents (Predictor $\rightarrow$ Judge $\rightarrow$ Final-Predictor) to evaluate the relevance of retrieved documents. It designs an adaptive threshold (based on the score mean and standard deviation) to dynamically filter noisy documents, achieving a 2-11% accuracy improvement across 4 QA benchmarks.

## Background & Motivation

**Noise in RAG**:
   - Documents returned by the retriever often contain irrelevant or noisy content.
   - Noisy documents can mislead LLMs and reduce response accuracy.
   - Existing studies (Chen et al., 2024; Yu et al., 2024) show that LLMs lack robustness to noise.

**Limitations of Prior Work**:
   - **Training-based RAG** (Self-RAG, REALM): Effective but requires substantial computational resources and training data.
   - **Training-free RAG**: Simple and efficient but highly sensitive to noise, often directly concatenating top-k documents into the prompt.
   - Lacks an effective post-processing filtering mechanism.

**Impact of Document Order**:
   - LLMs exhibit a "lost in the middle" phenomenon—tending to focus on the beginning and end of the input context.
   - Randomly shuffling document order leads to high performance variance (max is much higher than min), suggesting the existence of an optimal permutation.

**Core Motivation**: To design a training-free multi-agent framework that improves the noise robustness of RAG via collaborative evaluation and adaptive filtering.

## Method

### Overall Architecture

MAIN-RAG introduces a multi-agent filtering layer after the retrieval stage of the standard RAG pipeline, which is collaboratively executed by three LLM agents:

### Agent Definition

#### Agent-1: Predictor

- For each query $q$, reads each retrieved document $d_i$ one by one.
- Generates a preliminary answer $a_i$ based on each document.
- Forms a Document-Query-Answer triplet $(d_i, q, a_i)$.

#### Agent-2: Judge

- Receives each $(d_i, q, a_i)$ triplet.
- Determines whether the document provides relevant supporting information for the query and answer.
- Outputs a judgment of "Yes" or "No".

**Key Innovation—Relevance Score Quantization**:
- Instead of using discrete "Yes"/"No" judgments, the continuous relevance score is calculated as:
  $$\text{Score} = \log P(\text{"Yes"}) - \log P(\text{"No"})$$
- This difference serves as the continuous relevance score $r_i$ of the document.
- This continuous score allows documents to be sorted and provides a basis for filtering.

#### Agent-3: Final-Predictor

- Receives the filtered and sorted list of documents.
- Generates the final answer based on the high-quality documents.

### Adaptive Threshold $\tau_q$

**Core Observations**:
- Score distribution of relevant documents: higher scores with small standard deviation (the LLM is more confident).
- Score distribution of noisy documents: uniform with larger standard deviation (the LLM is uncertain and prone to misjudgment).
- The optimal filtering threshold varies across different queries.

**Design**:
- For each query $q$, the average relevance score of all candidate documents is computed as the adaptive threshold $\tau_q$.
- Documents with a score $r_i \geq \tau_q$ are retained.
- Flexibility can be introduced via $\tau_q - n \cdot \sigma$ (where $n$ is the only hyperparameter).
- **Intuition**: When there are many relevant documents, the average score is high, thus filtering out low-score outliers. When there are few relevant documents, the average score is low, preserving about half of the documents.

### Document Ordering

Filtered documents are sorted in **descending order** of relevance scores—placing high-scoring documents at the beginning to exploit the LLM's primacy bias.

## Key Experimental Results

### Dataset and Setup

- **4 QA Benchmarks**: TriviaQA (Open Domain), PopQA (Long-tail Entities), ARC-Challenge (Science QA), ALCE-ASQA (Long-form QA).
- **Agent Instantiation**: Pre-trained Mistral-7B or Llama3-8B (without fine-tuning).
- **Retriever**: Contriever-MS MARCO, retrieving up to 20 documents per query.
- **Evaluation**: Zero-shot.

### Main Results

| Method | TriviaQA | PopQA | ARC-C | ASQA (em/rg/mau) |
|------|:---:|:---:|:---:|:---:|
| **No Retrieval** |||||
| Mistral-7B | 54.8 | 26.2 | 55.5 | 11.2/18.1/27.6 |
| Llama3-8B | 68.4 | 29.2 | 58.8 | 19.4/30.3/54.3 |
| **Standard RAG** |||||
| Mistral-7B + RAG | 69.4 | 55.5 | 57.1 | 32.4/34.8/54.3 |
| Llama3-8B + RAG | 73.1 | 61.8 | 55.6 | 37.1/36.5/63.0 |
| **Training-based Methods** |||||
| Self-RAG-7B | 66.4 | 54.9 | 67.3 | 30.0/35.7/74.3 |
| **MAIN-RAG (Ours)** |||||
| MAIN-RAG (Mistral) | **71.0** | **58.9** | **58.9** | 35.7/36.2/60.0 |
| MAIN-RAG (Llama3) | **74.1** | **64.0** | **61.9** | **39.2/42.0/70.6** |

**Key Findings**:
1. MAIN-RAG outperforms training-free baselines across all benchmarks, with improvements of 2-11%.
2. On TriviaQA and PopQA, training-free MAIN-RAG achieves performance close to or exceeding the training-based Self-RAG.
3. The advantage is most pronounced on PopQA (long-tail entities)—where the retriever is not fine-tuned on the target data, introducing more noise and highlighting the value of filtering.

### Ablation Study (Figure 7)

| Variant | Role |
|------|------|
| Naïve Multi-agent RAG | The Judge uses discrete "Yes"/"No" judgments instead of continuous scores $\rightarrow$ Performance drops, demonstrating the necessity of score quantization. |
| MAIN-RAG (Random) | Filtered documents are randomly ordered $\rightarrow$ Performance drops, demonstrating the importance of descending order sorting. |
| Standard RAG | No filtering $\rightarrow$ Baseline. |

### Adaptive Threshold Ablation

| Method | TriviaQA | PopQA | ARC-C |
|------|:---:|:---:|:---:|
| $\tau_q$ (Default) | **71.0** | **58.9** | **58.9** |
| $\tau_q - 0.5\sigma$ | 71.2 | 58.6 | 59.0 |
| $\tau_q - 1.0\sigma$ | 70.8 | 58.0 | 58.5 |
| $\tau_q - 1.5\sigma$ | 70.4 | 58.4 | 57.7 |
| Ascending Order | 70.2 | 53.5 | 57.4 |

**Key Findings**:
- The default $\tau_q$ ranks at least second on all benchmarks, making it the most robust choice.
- Descending order consistently outperforms ascending order, validating the presence of LLM primacy bias.
- The effect of adjusting $\sigma$ varies by dataset; the default setting remains the most versatile.

### Case Study: Intuition behind $\tau_q$ values

- **Case 1** ($\tau_q = 9.575$): High confidence, most documents are relevant $\rightarrow$ strict filtering $\rightarrow$ correct answer.
- **Case 2** ($\tau_q = -8.425$): Low confidence, most documents are noisy $\rightarrow$ loose retention $\rightarrow$ finds the answer from a few informative documents.
- **Case 3** ($\tau_q = 0.4875$): Medium confidence $\rightarrow$ some documents are relevant but lack target information $\rightarrow$ incorrect answer.

## Highlights & Insights

1. **Training-free Competitiveness**: Without fine-tuning or extra annotated data, the proposed method approaches the performance of training-based methods like Self-RAG purely through inference-time multi-agent collaboration.
2. **Clever Design of Relevance Score Quantization**: Transforming binary judgments into continuous scores using $\log P(\text{"Yes"}) - \log P(\text{"No"})$ bridges agent evaluations and traditional IR ranking.
3. **Robustness of Adaptive Threshold**: The single hyperparameter $n$ works stably with its default value (0), adapting dynamically to different queries without requiring a fixed threshold.
4. **Empirical Validation of Document Ordering**: Systematically validates the significant impact of document order on RAG performance, showing that descending order is consistently optimal.
5. **Scalability**: The three agents can be instantiated with different LLMs, posing no specific requirements on the model choices.

## Limitations & Future Work

1. Each of the three agents requires an LLM inference step, causing computational overhead to be roughly 3x that of standard RAG (each document requires Agent-1 inference once + Agent-2 evaluation once).
2. Verified only on QA tasks, without testing other RAG scenarios such as summarization or dialogue.
3. Only evaluated on two models, Mistral-7B and Llama3-8B.
4. Orthogonal optimizations such as document compression and advanced decoding strategies were not considered.
5. The adaptive threshold is based on the score mean, which may be suboptimal for highly skewed score distributions.
6. The impact of retriever selection and re-rankers was not considered.

## Related Work & Insights

- **Training-based RAG**: Self-RAG (Asai et al., 2024) learns retrieval and self-criticism via reflection tokens; REALM (Guu et al., 2020) introduces retrieval during pre-training.
- **Training-free RAG**: In-context RALM (Ram et al., 2023) retrieves dynamically; FLARE (Jiang et al., 2023) actively determines when to retrieve.
- **Noise Robustness**: The RGB benchmark by Chen et al. (2024) evaluates RAG noise robustness; Yu et al. (2024) leverage context ranking to improve robustness.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ The combination of multi-agent filtering and adaptive threshold is highly novel and practical. Quantizing relevance through log-prob differences is simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough evaluations across 4 benchmarks, comparisons with multiple baselines, complete ablation studies, and intuitive case studies.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, minimal hyperparameters ($n=0$ by default), providing direct value to practical RAG applications.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architectural diagram; the motivation for the adaptive threshold is naturally derived from observations of the score distribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](../../ACL2026/information_retrieval/mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)
- [\[ACL 2025\] Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation](typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)

</div>

<!-- RELATED:END -->
