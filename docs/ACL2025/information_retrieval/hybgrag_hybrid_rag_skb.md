---
title: >-
  [Paper Note] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This paper proposes HybGRAG, a method that leverages both textual and relational information through a Retriever Bank, coupled with a Critic module's self-reflection to iteratively correct question routing errors, achieving an average Hit@1 improvement of 51% on hybrid question answering tasks over semi-structured knowledge bases.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Graph RAG"
  - "hybrid question answering"
  - "semi-structured knowledge base"
  - "self-reflection"
date: 2026-05-08
content_hash: 1a48da17a71bd0da
---

# HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases

**Conference**: ACL 2025  
**arXiv**: [2412.16311](https://arxiv.org/abs/2412.16311)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: RAG, Graph RAG, hybrid question answering, semi-structured knowledge base, self-reflection

## TL;DR
This paper proposes HybGRAG, a method that leverages both textual and relational information through a Retriever Bank, coupled with a Critic module's self-reflection to iteratively correct question routing errors, achieving an average Hit@1 improvement of 51% on hybrid question answering tasks over semi-structured knowledge bases.

## Background & Motivation
**Background**: Retrieval-Augmented Generation (RAG) retrieves information from unstructured documents to assist LLMs in answering questions, while Graph RAG (GRAG) retrieves relational information from knowledge graphs.

**Limitations of Prior Work**: Many real-world questions require both textual and relational information ("hybrid questions"), such as "Find a paper written by John Smith on nanofluid heat transfer" — which involves both author relationships (relational information) and paper topics (textual information). Existing methods can only handle one of these.

**Key Challenge**: (C1) Text retrievers and graph retrievers are individually effective but their retrieved information does not overlap, requiring coordination; (C2) LLMs often misidentify textual aspects as relational aspects on their first attempt, leading to retrieval failure and requiring iterative correction.

**Goal**: How to perform effective retrieval for hybrid questions on semi-structured knowledge bases (SKB, consisting of documents and knowledge graphs)?

**Key Insight**: Empirical analysis reveals that the theoretical upper bound of "optimal routing" is much higher than that of any single retriever (indicating information complementarity), and LLMs' subgraph extraction hit rate increases from 67.7% to 92.3% given corrective feedback.

**Core Idea**: Use a Retriever Bank to coordinate text and graph retrieval to solve hybrid questions, and employ a Critic module to iteratively correct routing errors through self-reflection.

## Method

### Overall Architecture
The input consists of a user question $q$ and a semi-structured knowledge base SKB (knowledge graph $G$ + document collection $\mathcal{D}$). The Router (LLM) first routes the question — identifying subject entities $\hat{\mathcal{E}}$ and useful relations $\hat{\mathcal{R}}$, and selecting a retrieval module. The Critic module validates the retrieval results; if validation fails, it generates feedback to guide the Router's modifications in the next round, with a maximum of $T$ iterations.

### Key Designs
1. **Retriever Bank (Addressing C1)**:

    - **Function**: Contains a text retrieval module and a hybrid retrieval module, selected for use by the Router.
    - **Mechanism**: The text retrieval module directly retrieves from documents using Vector Similarity Search (VSS). The hybrid retrieval module first uses a graph retriever to extract entities from subgraphs based on $\hat{\mathcal{E}}$ and $\hat{\mathcal{R}}$, and then employs a VSS ranker to sort the associated documents of these entities by semantic similarity to the question, achieving coordination of relational and textual information.
    - **Design Motivation**: Empirical results demonstrate that the strengths of the text and graph retrievers barely overlap (optimal routing Hit@1=0.45 vs. 0.29/0.25 individually), necessitating the utilization of both types of information.

2. **Critic Module (Addressing C2)**:

    - **Function**: Validates whether the retrieval results are correct, and if incorrect, generates structured corrective feedback to help the Router make modifications.
    - **Mechanism**: Split into two LLMs—Validator and Commenter. The Validator judges whether the retrieval results satisfy the question requirements (using reasoning paths as validation context). The Commenter generates corrective feedback (e.g., "entity X is incorrect, please replace it"), using about 30 successful experiences for In-Context Learning (ICL).
    - **Design Motivation**: A divide-and-conquer strategy reduces task difficulty (more accurate than a single LLM performing both validation and feedback generation simultaneously) and avoids the "lost in the middle" problem. Corrective feedback is more precise and directive than natural language feedback.

3. **Router**:

    - **Function**: Identifies subject entities and useful relations based on the question and entity/relation type information, and selects the retrieval module.
    - **Mechanism**: Employs an LLM with few-shot prompting, which first extracts relational aspects before deciding on the retrieval module selection. It modifies decisions in the next iteration upon receiving feedback from the Critic.
    - **Design Motivation**: Extracting entities/relations prior to decision-making (rather than the reverse) improves routing quality; for instance, when no entity is extracted, text retrieval is naturally selected.

## Key Experimental Results

### Main Results

| Dataset | Metric | HybGRAG | Prev. SOTA (AvaTaR) | Gain |
|--------|------|---------|-------------------|---------|
| STaRK-MAG | Hit@1 | 0.654 | 0.444 | +47.4% |
| STaRK-MAG | MRR | 0.698 | 0.512 | +36.5% |
| STaRK-Prime | Hit@1 | 0.286 | 0.184 | +54.9% |
| STaRK-Prime | MRR | 0.345 | 0.267 | +29.0% |

### Ablation Study

| Configuration | Hit@1 (MAG) | Description |
|------|-------------|------|
| Hybrid RM only | 0.503 | No Critic (No-Agent) |
| Router for SR | 0.621 | Single-Agent Self-Reflection |
| HybGRAG (Multi-Agent) | 0.654 | Full Method |
| w/o validation context | ~0.60 | Validator without reasoning paths |
| Commenter 5-shot | ~0.62 | Reduced ICL examples |

### Key Findings
- Self-reflection yields the most significant performance gain from 1 to 2 rounds, converging in a few iterations.
- The Multi-Agent (Router + Validator + Commenter) setup outperforms Single-Agent self-reflection (0.654 vs. 0.621).
- Switching to the smaller Claude 3 Haiku maintains strong performance (Hit@1=0.602) while speeding up by 1.96×.
- HybGRAG is training-free, whereas AvaTaR requires 500+ API calls for training.
- It also achieves the best Score on the end-to-end CRAG benchmark (Sonnet: 0.336 vs. second best 0.240).

## Highlights & Insights
- Practical methodology-driven design based on empirical analysis (C1: non-overlapping information, C2: error-prone first-attempt extraction) exemplifies a commendable research paradigm of "identifying problems before solving them".
- The design of corrective feedback (tabular error types + revision suggestions) is more effective than free-form natural language feedback, an insight that is transferrable to other agentic systems.

## Limitations & Future Work
- Reliance on LLM API calls, with up to 14 calls per question, which can be costly.
- Evaluated only on two benchmarks: STaRK and CRAG.
- The ~30 ICL examples for the Critic module require manual collection, leading to some cold-start workload.

## Related Work & Insights
- **vs. RAG**: RAG ignores relations between documents, failing to satisfy relational constraints.
- **vs. Think-on-Graph**: Pure graph querying ignores textual information, performing poorly on hybrid question answering (HQA).
- **vs. AvaTaR**: Requires substantial API calls for training, whereas HybGRAG is training-free and achieves 47% higher Hit@1.
- **vs. Reflexion**: General self-reflection lack targeted corrective feedback, yielding limited improvement on HQA.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hybrid retrieval and structured corrective feedback is innovative, though individual component ideas have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive major experiments, ablations, end-to-end evaluations, and cost analyses are provided, but the number of benchmarks is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The question-driven narrative structure is clear, and diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Holds direct reference value for practical RAG systems, as semi-structured KB scenarios are common.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[ACL 2025\] Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](toward_structured_knowledge_reasoning_contrastive_retrieval-augmented_generation.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)

</div>

<!-- RELATED:END -->
