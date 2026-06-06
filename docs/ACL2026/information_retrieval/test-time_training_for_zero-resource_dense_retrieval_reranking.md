---
title: >-
  [Paper Note] Test-Time Training for Zero-Resource Dense Retrieval Reranking
description: >-
  [ACL2026][Information Retrieval & RAG][Zero-Shot Reranking] DART is proposed to adaptively adjust the scoring function of a dense retriever using a bilinear matrix during inference. By leveraging the retrieval results th…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Zero-Shot Reranking"
  - "Test-Time Adaptation"
  - "Dense Retrieval"
  - "Bilinear Scoring Matrix"
date: 2026-05-08
content_hash: 7ba2f8eb4f97d8b1
---

# Test-Time Training for Zero-Resource Dense Retrieval Reranking

**Conference**: ACL2026  
**arXiv**: [2606.01070](https://arxiv.org/abs/2606.01070)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Zero-Shot Reranking, Test-Time Adaptation, Dense Retrieval, Bilinear Scoring Matrix

## TL;DR
DART is proposed to adaptively adjust the scoring function of a dense retriever using a bilinear matrix during inference. By leveraging the retrieval results themselves as pseudo-labels for zero-shot unlabeled reranking, it achieves an average improvement of 2.1% NDCG@10 on the BEIR benchmark while maintaining latency under 10ms.

## Background & Motivation

**Background**: In modern information retrieval systems, a two-stage cascaded architecture has become standard practice: the first stage uses a fast dense retriever (bi-encoder) to retrieve candidates from a full corpus, while the second stage employs an accurate but slow reranker (cross-encoder or LLM) to refine the ranking. Dense retrievers are preferred for their millisecond-level latency and strong recall, yet the reranking stage faces severe zero-resource challenges.

**Limitations of Prior Work**: Supervised reranking methods (cross-encoder, LLM rerankers) require expensive human-annotated data and massive computational resources. While methods like ColBERT perform well, their latency often reaches 200–500ms or higher, severely limiting real-time applications. In unlabeled settings, practitioners are often forced to skip the reranking step and use raw dense retrieval scores, especially in vector database systems that only index embeddings. Meanwhile, unsupervised Pseudo-Relevance Feedback (PRF) does not require training data but is unstable and even degrades results on most BEIR datasets.

**Key Challenge**: To achieve zero-resource unlabeled reranking, one must either choose supervised methods (requiring data and time) or rely on unreliable unsupervised heuristics. It is difficult to balance efficiency and effectiveness.

**Goal**: To find a lightweight, inexpensive, fast, and reliable zero-resource reranking solution that requires neither external resources nor offline training.

**Key Insight**: A crucial but overlooked signal is observed—the ranked list from the retriever itself contains task-relevant information: top-ranked documents are likely relevant (pseudo-positives), and bottom-ranked documents are likely irrelevant (pseudo-negatives). Although these pseudo-labels are noisy, they are query-specific and readily available.

**Core Idea**: Rather than altering query or document representations, the scoring function is personalized for each query at inference time. This preserves the capabilities of the pre-trained dense retriever while learning query-specific adjustments. This marks the first application of Test-Time Training (TTT) principles in retrieval reranking.

## Method

### Overall Architecture

DART models zero-resource reranking as an online optimization problem: for each incoming query $q$, it first retrieves top-$K$ documents using an initial scoring function $s(q,d)=\phi(q)^\top\psi(d)$. Based on pseudo-labels within these documents (top $n_{\text{pos}}$ as pseudo-positives, bottom $n_{\text{neg}}$ as pseudo-negatives), a bilinear transformation matrix $W$ is optimized via gradient steps, upgrading the scoring function to $s_W(q,d)=\phi(q)^\top W\psi(d)$. After optimization, the updated matrix is used to rerank the retrieval results. To enhance stability and generalization, cross-query momentum states (MetaInit and EMA) are maintained, allowing subsequent queries to benefit from the adaptation experience of preceding ones.

### Key Designs

1. **Confidence-weighted Pseudo-labels + Bilinear Scoring Matrix Optimization with Adaptive Margin**:

    - **Function**: Transitions the fixed cosine similarity scoring function to $\phi(q)^\top W\psi(d)$ by learning an adjustable $d \times d$ transformation matrix $W$, allowing the importance of different semantic dimensions to change dynamically for each query.
    - **Mechanism**: $W$ is initialized as $I$ to ensure the starting point is standard cosine similarity. Weights for pseudo-positives are $w_i^+ = \exp(s_i/T) / \sum_{i'}\exp(s_{i'}/T)$, and for pseudo-negatives $w_j^- = \exp(-s_j/T) / \sum_j\exp(-s_j/T)$, automatically focusing on high-confidence pseudo-labels. The margin is designed as $\text{margin}(q) = \alpha_{\text{mar}} + \beta_{\text{mar}}(1-s_{\text{top1}})$, where hard queries require a larger margin while easy queries reduce the requirement.
    - **Design Motivation**: Pseudo-labels are noisy; treating them equally amplifies noise. Confidence weighting automatically distinguishes reliability. The adaptive margin addresses the misalignment of fixed margins across varying query difficulties—easy queries do not need a large margin, while difficult ones require more aggressive adjustments.

2. **Cross-Query Momentum and Smoothing Mechanism (MetaInit + EMA)**:

    - **Function**: Maintains two complementary matrix states to smooth parameter evolution.
    - **Mechanism**: MetaInit learns a global matrix $W_{\text{meta}}$, updated via the Reptile rule after each query: $W_{\text{meta}}^{(t)} = W_{\text{meta}}^{(t-1)} + \beta_{\text{meta}}(W^\star(t) - W_{\text{meta}}^{(t-1)})$, which serves as the starting point for the next query. EMA maintains $W_{\text{ema}} = \alpha_{\text{ema}}W_{\text{ema}} + (1-\alpha_{\text{ema}})W^\star$ for final reranking to reduce single-query variance.
    - **Design Motivation**: Optimization signals from a single query are weak (only top-100 documents), leading to potential overfitting. Cross-query momentum aggregates learning signals across multiple queries, accelerating convergence and avoiding over-adaptation to single-query noise. Experiments show EMA is most effective, providing positive gains across all datasets.

3. **Adaptive Optimizer Selection Strategy (SGD vs Lion)**:

    - **Function**: Automatically chooses between SGD with momentum and the Lion optimizer based on the pseudo-label quality of the dataset.
    - **Mechanism**: In practice, both optimizers are run for the first 50–100 queries to compare their average pseudo-label loss; the one with lower loss is selected for subsequent queries. SGD is suitable for datasets with high pseudo-label noise, while Lion, based on gradient signs, is better for datasets with high-quality pseudo-labels.
    - **Design Motivation**: Differences in dataset sparsity and domain result in varied pseudo-label quality. No single optimizer is optimal for all scenarios.

## Key Experimental Results

### Main Results

Evaluated on six BEIR benchmark datasets:

| Dataset | NFCorpus | SCIDOCS | FiQA | ArguAna | TREC-COVID | SciFact | Avg. | Avg. Gain | Latency |
|--------|----------|---------|------|---------|------------|---------|------|-----------|------|
| Dense Retrieval (BGE-small) | 0.337 | 0.197 | 0.385 | 0.595 | 0.665 | 0.720 | 0.483 | 0.0% | <1ms |
| BM25 Reranking | 0.302 | 0.156 | 0.220 | 0.371 | 0.685 | 0.588 | 0.387 | −21.2% | <2ms |
| PRF-Vec (n=3) | 0.347 | 0.203 | 0.371 | 0.602 | 0.663 | 0.710 | 0.483 | +0.3% | <2ms |
| **DART (Ours)** | **0.354** | **0.205** | **0.389** | **0.605** | **0.670** | **0.719** | **0.490** | **+2.1%** | **<10ms** |

Ours outperforms the dense retrieval baseline on 5/6 datasets, with the highest gain on NFCorpus (+5.0%). The disastrous performance of BM25 reranking (−21.2%) highlights the mismatch of lexical methods. Compared to recent unsupervised LLM methods, DART achieves peak performance with <10ms latency (over 20x faster than their 200ms).

### Ablation Study

| Configuration | NFCorpus | SCIDOCS | FiQA | ArguAna | Avg. Gain |
|------|----------|---------|------|---------|---------|
| Dense Retrieval | 0.337 | 0.197 | 0.385 | 0.595 | 0.0% |
| Base (Confidence weight only) | 0.346 | 0.199 | 0.363 | 0.595 | +0.5% |
| + AdaMargin | 0.350 | 0.201 | 0.362 | 0.595 | +3.9% |
| + EMA | 0.351 | 0.199 | 0.378 | 0.596 | +4.0% |
| + MetaInit | 0.348 | 0.197 | 0.362 | 0.599 | +3.3% |
| + EMA + AdaMargin | 0.355 | 0.203 | 0.378 | 0.597 | +5.3% |
| + All (inc. Lion) | 0.354 | 0.205 | 0.389 | 0.605 | +5.0% |

**Key Findings**:

- **EMA is most effective**, yielding positive gains across all four datasets.
- **AdaMargin contributes most to NFCorpus**, which has a wide distribution of query difficulty.
- **Lion provides a +4.1% single-step boost on SCIDOCS**, confirming its advantage when pseudo-labels are clean.
- **The three components are complementary**, with the full combination achieving the optimal average gain.

## Highlights & Insights

- **Sophisticated Pseudo-label Reliability**: Instead of crude binary pseudo-labels, soft confidence weights $\exp(s_i/T)$ are used for automatic weighting—a transferable concept for other pseudo-label scenarios (domain adaptation, active learning).
- **Difficulty-Adaptive Margin**: $\text{margin}(q) = \alpha_{\text{mar}} + \beta_{\text{mar}}(1-s_{\text{top1}})$ elegantly quantifies query difficulty into a scalar to regulate learning intensity.
- **Discovery of Low-Rank Structure**: The transformation matrix $\Delta W$ learned by DART exhibits significant low-rank properties (the top three singular values explain 28.4% of variance), indicating the network automatically adapts within a small task-relevant dimensional subspace.
- **Practical Innovation under Strict Latency**: achieving results with only 5 gradient steps and matrix multiplications under a <10ms constraint demonstrates a perfect balance between efficient computation and effectiveness.
- **New Heights in Zero-Resource Settings**: Reaching performance comparable to strongly supervised methods in "absolute forbidden zones" (no labels, no external resources, no offline training).

## Limitations & Future Work

**Limitations acknowledged by the authors**:

- **Warm-up Cost for Optimizer Selection**: Requires 50–100 queries to compare optimizers; authors suggest SGD as the default.
- **Scalability Bottleneck**: The current implementation optimizes a $d \times d$ matrix; for $d \geq 768$, memory and computational overhead grow quadratically. The paper proposes low-rank parameterization $W = I + AB^\top$ for future implementation.

**Own observations**:

- In domains where the retriever fails severely (e.g., −0.1% on SciFact), pseudo-label quality is poor, and improvements are limited.
- Cross-query momentum assumes similarity in query streams; it may fail in scenarios with drastic conversational topic shifts.
- Other loss function designs, such as listwise losses, have not been explored.

**Concrete Ideas for Improvement**:

- Implement low-rank parameterization to support larger embedding dimensions.
- Study adaptation at the session or session-cluster level.
- Explore distilling the knowledge of matrix $W$ into fixed parameters for edge systems that do not support gradients.

## Related Work & Insights

- **vs Traditional Pseudo-Relevance Feedback (PRF)**: PRF utilizes pseudo-relevant documents by modifying query representations, while DART keeps representations fixed and adjusts the scoring function. These are complementary approaches, with DART being more precise and flexible.
- **vs Unsupervised Domain Adaptation (GPL, AugTriever)**: These require offline training and data generation; DART is entirely online with zero offline cost.
- **vs LLM Rerankers**: LLMs have strong text understanding but 200–500ms latency is unsuitable for real-time systems. DART trades lightweight parameter adaptation for low latency.
- **vs TTT in Computer Vision**: TTT++ validated test-time parameter adaptation for image classification; DART successfully migrates this to retrieval ranking for the first time.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of Test-Time Training to retrieval reranking, cleverly utilizing retrieval results as pseudo-labels for zero-resource adaptation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Cross-domain validation on six BEIR datasets, complete ablation experiments, and in-depth low-rank structure analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, sufficient motivation, precise methodology, and complete algorithm pseudo-code.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses extremely common industry scenarios with a simple, low-overhead, and stable solution, possessing strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](../../NeurIPS2025/information_retrieval/retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[ACL 2026\] The Dilemma of Low-Resource Languages in Multilingual Retrieval: Evidence from Amharic](the_multilingual_curse_at_the_retrieval_layer_evidence_from_amharic.md)
- [\[ACL 2026\] CRAFT: Training-Free Cascaded Retrieval for Tabular QA](craft_training-free_cascaded_retrieval_for_tabular_qa.md)
- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[ACL 2026\] ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals](chunqiutr_time-keyed_temporal_retrieval_in_classical_chinese_annals.md)

</div>

<!-- RELATED:END -->
