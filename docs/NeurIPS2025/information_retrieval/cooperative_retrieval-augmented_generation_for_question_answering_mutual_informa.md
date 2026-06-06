---
title: >-
  [Paper Note] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][RAG] CoopRAG is a framework that achieves bidirectional cooperation between the retriever and the LLM through query expansion, retriever layer-contrastive reranking…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "query expansion"
  - "layer-contrastive ranking"
  - "multi-hop QA"
  - "cooperative mechanism"
date: 2026-05-08
content_hash: 21b91d932cf1537a
---

# Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers

**Conference**: NeurIPS 2025
**arXiv**: [2512.10422](https://arxiv.org/abs/2512.10422)  
**Code**: [GitHub](https://github.com/meaningful96/CoopRAG)  
**Area**: Retrieval-Augmented Generation & Question Answering
**Keywords**: RAG, query expansion, layer-contrastive ranking, multi-hop QA, cooperative mechanism

## TL;DR
CoopRAG is a framework that achieves bidirectional cooperation between the retriever and the LLM through query expansion, retriever layer-contrastive reranking, and reasoning chain completion. It surpasses HippoRAG2 by 5.3% on multi-hop QA and by 35.2% on single-hop QA.

## Background & Motivation
**Insufficient query information**: Original questions lack sufficient information to adequately guide retrieval and reasoning.

**Shallow retrieval**: Existing retrievers tend to rely on surface-level pattern matching rather than deep semantic understanding.

**Hallucination**: LLMs tend to generate incorrect information when faced with uncertain knowledge.

**Bidirectional enhancement**: A mechanism enabling mutual feedback and correction between the retriever and the LLM is needed.

## Method

### Overall Architecture
CoopRAG consists of five stages:

1. **Query Expansion**: The LLM decomposes the question into sub-questions and an uncertain reasoning chain.
2. **Expansion-Augmented Retrieval**: The expanded query is used to retrieve top-$n$ documents.
3. **Ranking by Contrasting Layers (RaLa)**: Documents are reranked to top-$k$.
4. **Reasoning Chain Completion**: The LLM fills in uncertain parts of the reasoning chain.
5. **Final Inference**: The answer is generated.

### Key Designs

**Query Expansion (Section 3.2)**:

The LLM generates:
- A sub-question set $S = \{s_1, s_2, \ldots, s_{|S|}\}$
- An uncertain reasoning chain $R = \{(e_1, r_1, e_1'), \ldots, (e_{|R|}, r_t, \texttt{<FILL>})\}$

Core innovation: `<UNCERTAIN>` masks are used in place of uncertain entities to avoid hallucination:

$$U = Q \,\|\, S \,\|\, R$$

**Ranking by Contrasting Layers (RaLa, Section 3.4)**:

Motivation: Lower Transformer layers capture syntactic information, while higher layers capture semantics.
Approach: The representational difference between intermediate and top layers is compared.

$$\text{score}(U, D) = \text{avg}_{i=0}^{|U|} \max_{j}\, g(q_i, d_j)$$

where $g(q_i, d_j) = \max_{l \in C} \left(\langle q_i, d_j^{(L)}\rangle - \langle q_i, d_j^{(l)}\rangle\right)$

Practical implementation (for cost reduction):

$$\text{score}_o(U, D) = \omega_{U,D} \cdot \text{avg}_i \max_j \langle q_i, d_j^{(L)}\rangle$$

where $\omega_{U,D} = g(q_0, d_0)$ is the gap weight.

**Reasoning Chain Completion (Section 3.5)**:

The LLM leverages top-$k$ documents to fill in `<UNCERTAIN>` and `<FILL>` placeholders and reconstruct the reasoning chain.

**Difficulty-Aware Training (Section 3.6)**:

Weighted loss: $\alpha_{U_i} = \log(1 + |S_{U_i}|)$

Harder questions (with more sub-questions) receive higher loss weights.

## Key Experimental Results

### Retrieval Performance (Table 2 — Multi-hop QA)

| Method | HotpotQA R@2 | MuSiQue R@2 | 2Wiki R@2 | Key Improvement |
|---|---|---|---|---|
| HippoRAG2 (L3.3) | 83.5% | 56.1% | 76.2% | baseline |
| HippoRAG2 (GPT) | 80.5% | 53.5% | 74.6% | reference |
| SiReRAG (GPT) | 80.0% | 52.5% | 60.6% | inferior |
| CoopRAG (G2-9B) | 87.9% | **59.4%** | 80.1% | +5.9% |
| CoopRAG (G2-27B) | 88.3% | 59.4% | 80.8% | +6.3% |
| **CoopRAG (L3.3)** | 86.9% | 58.2% | 80.6% | +3.4% |
| **CoopRAG (GPT)** | **88.8%** | **59.6%** | **80.4%** | **+8.3%** |

### Single-hop QA Performance (Table 2 right, NaturalQuestions)

| Method | R@2 | R@5 | Improvement |
|---|---|---|---|
| HippoRAG2 (L3.3) | 45.6% | 78.0% | baseline |
| HippoRAG2 (GPT) | 44.4% | 76.4% | reference |
| CoopRAG (G2-9B) | 71.6% | **88.9%** | **+27.2%** |
| CoopRAG (G2-27B) | 72.8% | **89.5%** | **+28.4%** |
| **CoopRAG (L3.3)** | 77.2% | **90.8%** | **+31.6%** |
| **CoopRAG (GPT)** | **80.8%** | **92.1%** | **+35.2%** |

### QA Performance (Table 3 — EM/F1)

| Method | HotpotQA | MuSiQue | 2Wiki | NQ |
|---|---|---|---|---|
| HippoRAG2 (L3.3) | 62.7/75.5 | 37.2/48.6 | 65.0/71.0 | 48.6/63.3 |
| HippoRAG2 (GPT) | 56.3/71.1 | 35.0/49.3 | 60.5/69.7 | 43.4/60.0 |
| CoopRAG (G2-9B) | **64.4**/78.1 | **52.2**/65.2 | 70.0/78.1 | 63.8/72.7 |
| CoopRAG (L3.3) | 64.7/79.0 | 52.6/66.6 | **71.2**/78.8 | 70.9/80.3 |
| **CoopRAG (GPT)** | 65.5/79.2 | **52.8**/66.7 | 70.8/78.6 | **71.3**/80.5 |

### Performance on Recent Benchmarks (Table 4 — MMLU-Pro)

| Task | Baseline | CoopRAG-Algo | CoopRAG-Math | Best Gain |
|---|---|---|---|---|
| Math | 54.63% | 53.89% | **60.25%** | +5.62% |
| CS | 37.80% | **40.73%** | 42.20% | +4.40% |
| Physics | 38.49% | 39.26% | **44.19%** | +5.70% |
| Average | — | +1.35% | **+3.02%** | best |

## Highlights & Insights
1. **Expansion–completion loop**: Query expansion combined with reasoning chain completion forms a complete bidirectional cooperative cycle, in which the LLM guides retrieval while retrieval supplements the LLM's knowledge.
2. **Layer-contrastive ingenuity**: Leveraging the multi-layer properties of Transformer internal representations breaks the limitations of single-vector representation.
3. **Uncertainty masking**: Explicitly masking uncertain entities to prevent hallucination is a seemingly simple yet practically effective design choice.
4. **Scale efficiency**: Gemma2-9B outperforms HippoRAG2 based on Llama3.3-70B, demonstrating the superiority of the proposed method.

## Limitations & Future Work
1. **Question type constraints**: The method primarily targets factual QA; its applicability to open-ended questions remains unexplored.
2. **LLM dependency**: The approach relies on the LLM's expansion capability; weaker models may lead to failure cases.
3. **Computational cost**: Multiple LLM calls (expansion → completion → inference) result in relatively high inference latency.
4. **Heuristic layer selection**: The dynamic layer selection via $\omega$ weights remains heuristic, with limited theoretical grounding.

## Related Work & Insights
- **RAG methods**: HippoRAG, SiReRAG, HopRAG, GraphRAG, LightRAG
- **Query augmentation**: HyDE, GAR, Step-Back-Prompting
- **Dense retrieval**: ColBERT, DPR, GTR
- **Reasoning chains**: CoT, ToT, ReAct
- **QA systems**: KBQA, multi-hop QA

## Rating
⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
