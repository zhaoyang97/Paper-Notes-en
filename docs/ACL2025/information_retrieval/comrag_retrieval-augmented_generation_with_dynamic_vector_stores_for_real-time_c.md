---
title: >-
  [Paper Note] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry
description: >-
  [ACL 2025][Information Retrieval & RAG][Retrieval-Augmented Generation] ComRAG is proposed — a retrieval-augmented generation framework for real-time community question answering (CQA) in industry. By utilizing a tri-store architecture (**static knowledge vector store + high-/low-quality dynamic QA vector stores**) and a **centroid-based memory mechanism**, it achieves up to a 25.9% improvement in vector similarity across three CQA datasets while reducing latency by 8.7%-23.3…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Community Question Answering"
  - "Dynamic Vector Stores"
  - "Centroid-Based Memory Mechanism"
  - "Industrial Deployment"
date: 2026-05-08
content_hash: 552bda69f2a87951
---

# ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry

**Conference**: ACL 2025  
**arXiv**: [2506.21098](https://arxiv.org/abs/2506.21098)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Retrieval-Augmented Generation, Community Question Answering, Dynamic Vector Stores, Centroid-Based Memory Mechanism, Industrial Deployment

## TL;DR

ComRAG is proposed — a retrieval-augmented generation framework for real-time community question answering (CQA) in industry. By utilizing a tri-store architecture (**static knowledge vector store + high-/low-quality dynamic QA vector stores**) and a **centroid-based memory mechanism**, it achieves up to a 25.9% improvement in vector similarity across three CQA datasets while reducing latency by 8.7%-23.3%.

## Background & Motivation

**Background**: Community Question Answering (CQA) platforms (such as Stack Overflow) serve as crucial knowledge bases. Existing methods are divided into retrieval-based (selecting answers from community history) and generative (using LLMs as "community experts" to answer directly).

**Limitations of Prior Work**:
   - **Neglecting external domain knowledge**: Pure CQA methods lack support from professional documentation.
   - **Static perspective**: They cannot handle the real-time inflow of questions or leverage dynamically accumulated historical QA.
   - **Lack of memory mechanism**: Historical QA grows continuously without efficient storage management strategies.

**Key Challenge**: Real-time CQA needs to simultaneously utilize both **static domain knowledge** and **dynamic historical QA with varying quality**, while controlling storage growth to suit industrial deployment.

**Goal**: (Q1) How to integrate static knowledge with dynamic historical QA? (Q2) How to manage rapidly growing historical data and varying answer quality?

**Key Insight**: Extending the RAG framework to feature three vector stores (static knowledge store + high-quality CQA store + low-quality CQA store), integrated with centroid-based clustering memory management.

**Core Idea**: Realizing efficient real-time community question answering through high-/low-quality separated dynamic vector stores combined with centroid-based memory.

## Method

### Overall Architecture

ComRAG consists of three vector stores and three query paths:
- **Static Knowledge Vector Store $\mathcal{V}_{\text{kb}}$**: Embeddings of domain documents.
- **High-Quality CQA Vector Store $\mathcal{V}_{\text{high}}$**: Historical QA pairs with score $\geq \gamma$.
- **Low-Quality CQA Vector Store $\mathcal{V}_{\text{low}}$**: Historical QA pairs with score $< \gamma$.

### Key Designs

1. **Centroid-Based Memory Mechanism**:

    - Cluster similar historical questions into $m$ clusters $\{C_1, C_2, \dots, C_m\}$.
    - Maintain a centroid $\mathbf{c} = \frac{1}{|C|}\sum_{q_i \in C} \text{Emb}(q_i)$ for each cluster.
    - Upon arrival of a new question: if its similarity to a cluster centroid is $\geq \tau$, assign it to that cluster and update the centroid; otherwise, create a new cluster.
    - **Intra-cluster replacement strategy**: If the new question shows a similarity of $> \delta$ to an existing question within the cluster and possesses a higher-quality answer, the old one is replaced.
    - Design Motivation: **Controlling storage growth** — only retaining representative questions for each cluster to avoid memory overflow.

2. **Three Query Paths**:

    - **Path ①**: If an extremely similar question exists in the high-quality store ($\text{CosSim} \geq \delta$), directly reuse the historical answer.
    - **Path ②**: If there is a high-quality historical QA with moderate similarity ($\tau \leq \text{CosSim} < \delta$), generate an answer using it as a reference.
    - **Path ③**: If no match is found in the high-quality store, retrieve documents from the static knowledge store + retrieve historical QA from the low-quality store (as a **negative reference** to prompt the LLM to avoid similar mistakes).

3. **Adaptive Temperature Adjustment**:

    - Dynamically adjust the generation temperature based on the variance of the retrieved historical answer scores.
    - Low score variance (consistent historical answers) $\rightarrow$ higher temperature to encourage exploration.
    - High score variance (varying historical answers) $\rightarrow$ lower temperature to guarantee consistency.
    - Equation: $T(\Delta) = |\exp(-k \cdot \min_{1 \leq i \leq l-1}(s_{i+1} - s_i))|_{[T_{min}, T_{max}]}$
    - Hyperparameters: $k=250$, $T_{min}=0.7$, $T_{max}=1.2$

### Scoring & Updating

- Utilize BERT-Score as the answer quality scorer.
- Score after each answer generation, and partition into high-quality or low-quality CQA stores based on the threshold $\gamma$.

## Key Experimental Results

### Main Results (Three Datasets)

| Method | MSQA SIM | MSQA Avg Time | ProCQA SIM | ProCQA Avg Time | PolarDBQA BERT-Score |
|------|----------|---------------|------------|-----------------|---------------------|
| Raw LLM | 80.58 | 12.70s | 74.88 | 12.77s | 60.34 |
| Vanilla RAG | 80.73 | 13.86s | 75.59 | 16.97s | 64.78 |
| RAG+DPR | 80.50 | 14.08s | 74.83 | 13.79s | 66.55 |
| LLM+EXP | 76.70 | 20.23s | 67.78 | 22.69s | 67.00 |
| **ComRAG** | **94.70** | **11.60s** | **95.31** | **10.42s** | **67.39** |

- ComRAG achieves up to a **25.9%** improvement in the SIM metric (ProCQA: 74.88 $\rightarrow$ 95.31).
- Latency is reduced by 8.7%-23.3%.
- Both BERT-Score and BLEU/ROUGE-L demonstrate competitive performance.

### Ablation Study (PolarDBQA, 10 Iterations)

- Removing high-quality CQA store: Latency increases by 4.9s, BERT-Score decreases by 2.6.
- Removing centroid-based memory mechanism: Latency increases by 2.2s, BERT-Score decreases by 0.5.
- Removing static knowledge store and adaptive temperature: The proportion of directly reusable answers drops significantly.

### Efficiency & Storage Growth

- Query latency continuously decreases with iterations: ProCQA drops from 10.42s to 4.95s (-52.5%).
- BERT-Score continuously improves with iterations: A cumulative gain of 2.25% on MSQA.
- Storage growth rate rapidly drops from 20.23% in the first round to 2.06% in the 10th round (ProCQA).

### Key Findings

- **High-quality CQA store contributes the most**: Its removal causes the most significant impact on latency and quality.
- **Centroid-based memory mechanism effectively controls storage inflation**: Growth rate drops from 20.23% $\rightarrow$ 2.06%.
- **Low-quality store offers unique value as a counterexample**: It guides the LLM to steer clear of known errors via Path ③.
- **The system "gets smarter" over time**: Accumulated historical QA continuously boosts both response efficiency and quality.

## Highlights & Insights

- **Ingenious design of the tri-store architecture**: High-quality for reuse/reference, low-quality for contrast/avoidance, and static store for supplementing professional domain knowledge.
- **Centroid-based memory is core to industrial deployability**: Instead of simply "saving everything", it selectively groups and replaces.
- **Adaptive temperature adjustment**: Simple yet highly intuitive—encourages diversity when historical answers are consistent, and tightens when they diverge.
- **Genuinely oriented towards industrial production**: Directly considers real-world deployment challenges like latency and storage growth.

## Limitations & Future Work

1. **Fixed Thresholds**: $\tau$, $\delta$, and $\gamma$ are manually set fixed values, lacking adaptive tuning.
2. **Simplistic Handling of Low-Quality QA**: It only tells the LLM to "avoid similar answers" via prompt guidance; more advanced filtering or rectification mechanisms could be designed.
3. **Rule-Based Routing Strategy**: The selection among the three query paths is based on hard thresholds; a learning-based router could be introduced.
4. **Centroid-Based Memory Ignores Recency and Frequency**: Inactive clusters might occupy storage for a long time.
5. **Limited Evaluation Settings**: PolarDBQA is a proprietary dataset, and only one LLM backbone (Qwen2.5-14B) was utilized.

## Related Work & Insights

- **Difference from Standard RAG**: RAG typically relies only on static corpora; ComRAG's core contribution is the introduction of dynamically quality-graded historical QA management.
- **Comparison with LLM+EXP (Original MSQA Method)**: Although ComRAG's framework is more complex, its latency is actually lower because directly reusing high-quality answers bypasses the generation step.
- **Insights**: For any scenarios requiring continuous responses to similar queries (e.g., customer service, technical support), the "answer reuse + quality grading" paradigm is a highly valuable design prototype.

## Rating

- **Novelty**: ⭐⭐⭐ — The separation of high/low quality and centroid-based memory show some novelty, but the overall framework is an engineering extension of RAG.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three datasets, ablations, iterative evaluations, and storage analysis thoroughly.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and rigorous formal formulation.
- **Value**: ⭐⭐⭐⭐ — High practical utility in industrial scenarios, backed by real-world deployment in Alibaba's PolarDB environment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering](graf_graph_retrieval_augmented_by_facts_for_romanian_legal_multi-choice_question.md)
- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[NeurIPS 2025\] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](../../NeurIPS2025/information_retrieval/cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)

</div>

<!-- RELATED:END -->
