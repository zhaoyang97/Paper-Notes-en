---
title: >-
  [Paper Note] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs
description: >-
  [AAAI 2026][Multi-hop QA] N2N-GQA is proposed as the first zero-shot framework for open-domain hybrid table-text question answering. Its core mechanism transforms noisy retrieved documents into a dynamic evidence graph (documents as nodes, TF-IDF shared-term weights as edges), and employs graph centrality-based pruning to identify "bridging documents" that connect multi-hop reasoning chains. On OTT-QA, it achieves +39.6 EM over Vanilla RAG (8.0 → 48.8), approaching the fine-tuned system CORE (49.0 EM) in a zero-shot setting.
tags:
  - AAAI 2026
  - Multi-hop QA
  - Graph-based Retrieval
  - Table-Text Hybrid QA
  - Zero-shot
  - GraphRank
date: 2026-05-08
content_hash: 335e9149b648d3a2
---

# N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs

**Conference**: AAAI 2026
**arXiv**: [2601.06603](https://arxiv.org/abs/2601.06603)
**Code**: None
**Area**: LLM Reasoning / RAG
**Keywords**: Multi-hop QA, Graph-based Retrieval, Table-Text Hybrid QA, Zero-shot, GraphRank

## TL;DR
N2N-GQA is proposed as the first zero-shot framework for open-domain hybrid table-text question answering. Its core mechanism transforms noisy retrieved documents into a dynamic evidence graph (documents as nodes, TF-IDF shared-term weights as edges), and employs graph centrality-based pruning to identify "bridging documents" that connect multi-hop reasoning chains. On OTT-QA, it achieves +39.6 EM over Vanilla RAG (8.0 → 48.8), approaching the fine-tuned system CORE (49.0 EM) in a zero-shot setting.

## Background & Motivation

**Background**: Multi-hop QA (e.g., "What is the capital of the birth country of the 2019 Atlantic Hockey Player of the Year?") requires retrieval and reasoning across multiple documents and tables. Standard RAG pipelines pass retrieved results to LLMs as flat ranked lists. Open-domain hybrid QA benchmarks such as OTT-QA require simultaneous retrieval of tables and text passages from large corpora.

**Limitations of Prior Work**:
   - List-based retrieval scores each document independently, failing to discover inter-document reasoning chain relationships — a document may appear irrelevant in isolation yet serve as a bridge connecting two highly relevant documents.
   - Retrieval noise is amplified in multi-hop settings, as errors in the first hop propagate through the entire reasoning chain.
   - Competitive open-domain methods (CORE, COS) require extensive task-specific fine-tuning.
   - Zero-shot methods (e.g., ODYSSEY) are evaluated only in closed-domain settings where gold-standard evidence is provided.

**Key Challenge**: Multi-hop reasoning requires understanding relationships among evidence fragments (i.e., which documents form a reasoning chain), whereas ranked lists evaluate each document in isolation — relational information is discarded at the retrieval stage.

**Goal**: To achieve zero-shot open-domain multi-hop QA without any task-specific training, by organizing retrieved results via graph structures to recover inter-document relationships.

**Key Insight**: Retrieved documents are modeled as graph nodes, with TF-IDF shared-term weights as edges, and graph centrality is used to identify structurally important bridging documents.

**Core Idea**: Transform the flat retrieval list of RAG into a structured evidence graph, using graph centrality-based pruning to filter noise and preserve reasoning chains.

## Method

### Overall Architecture
The N2N-GQA pipeline consists of: (1) LLM-based structured query planning — decomposing complex questions into multi-hop sub-queries and conditional templates; (2) hop-by-hop iterative retrieval — ColBERTv2 retrieval → temporary evidence graph construction → GraphRank pruning → LLM intermediate entity extraction → entity substitution into the next-hop template; (3) global evidence aggregation — evidence pooled across all hops → bridge-aware hybrid selector → final evidence graph → GraphRank pruning → LLM final answer synthesis.

### Key Designs

1. **Dynamic Evidence Graph Construction**:

    - Function: Organizes retrieved documents into a query-specific knowledge graph.
    - Mechanism: Each retrieved document (passage or serialized table row) serves as a node; edge weights are defined as the sum of TF-IDF scores of shared terms. Edges represent semantic overlap between documents — documents sharing important entities are more likely to be adjacent links in a reasoning chain.
    - Design Motivation: TF-IDF is preferred over dense embeddings because it requires explicit lexical overlap, reducing "semantic drift" (conceptually similar but factually distinct), and is training-free, computationally efficient, and interpretable.

2. **GraphRank Scoring**:

    - Function: Re-ranks documents by combining semantic relevance with structural importance.
    - Mechanism: $\text{Score}_{GR}(v) = S_{sem}(v) \times (1 + (1-\alpha) \times S_{struct}(v))$, with $\alpha = 0.85$. The multiplicative formulation ensures semantic dominance — noise nodes with high centrality but low semantic relevance are suppressed, and structural information acts only as a "confidence amplifier" for semantically relevant documents.
    - Design Motivation: Under additive combination, high-centrality noise may outweigh genuinely relevant documents with lower semantic scores; the multiplicative formulation naturally resolves this issue.

3. **Bridge-Aware Hybrid Selector**:

    - Function: Identifies and prioritizes bridging documents that connect tables and text passages during the final evidence aggregation stage.
    - Mechanism: Checks for entity linkage between the top-ranked passage and the top-ranked table: $\phi(p,t) = \mathbb{1}[E(t) \cap T(p) \neq \emptyset]$. If a link is found, only the table score is boosted; otherwise, both scores are boosted.
    - Design Motivation: Reasoning chains in hybrid QA typically require cross-modal inference (table→text or text→table), and bridging documents serve as indispensable intermediaries.

4. **Structured Query Planning**:

    - Function: The LLM decomposes complex questions into executable multi-hop plans.
    - Mechanism: Outputs structured JSON encoding question complexity classification (1/2/3 hops), initial query with expected entity type, conditional templates for subsequent hops (with placeholders), and query alternatives.
    - Design Motivation: Predictable, machine-readable output provides a clear execution path for each stage of the pipeline.

### Loss & Training
- Fully zero-shot — no training or fine-tuning is performed.
- Reader models: GPT-4o, GPT-4.1, Llama3-70B.
- Retrieval: Pre-indexed ColBERTv2; top-100 retrieval followed by graph pruning to 12–25 nodes.

## Key Experimental Results

### Main Results
OTT-QA (open-domain, 500 samples), zero-shot:

| Method | Reader | EM↑ | F1↑ | BERTScore-F1↑ |
|--------|--------|-----|-----|---------------|
| Vanilla RAG | GPT-4o | 8.00 | 16.09 | 8.85 |
| RAG + Query Decomposition | GPT-4o | 31.40 | 43.07 | 37.84 |
| N2N-GQA w/o GraphRank | GPT-4.1 | 48.50 | 56.90 | 58.22 |
| **N2N-GQA w/ GraphRank** | **GPT-4.1** | **48.80** | **57.26** | **58.76** |
| N2N-GQA w/ GraphRank | Llama3-70B | 40.80 | 48.08 | 49.38 |

Fine-tuned systems (non-zero-shot) for reference: CORE 49.0 EM, COS 56.9 EM.

### Ablation Study

| Component | EM Gain | Notes |
|-----------|---------|-------|
| + Query Decomposition | +23.4 | 8.0→31.4; largest single-component contribution |
| + Graph Construction & Pruning | +16.0 | 31.4→47.4; evidence graph is the core contributor |
| + GraphRank | +0.2–0.3 | 47.4→47.6; modest but consistent |
| Full N2N-GQA | +39.6 | 8.0→47.6 (GPT-4o) |

### Key Findings
- **Graph-structured organization is the critical breakthrough**: Graph construction and pruning alone yield +16 EM, far exceeding GraphRank's +0.3 — "organizing documents as a graph" matters more than "how to rank nodes."
- **Query decomposition is the second-largest contributor**: +23.4 EM; explicit decomposition of multi-hop questions is essential for zero-shot systems.
- **Zero-shot performance approaches fine-tuned systems**: 48.8 EM vs. CORE 49.0 EM, a gap of less than one point.
- **Stronger readers benefit more**: GPT-4.1 > GPT-4o > Llama3-70B.

## Highlights & Insights
- **The "list → graph" paradigm shift** is the central contribution — simply organizing retrieved results as a graph followed by pruning yields +16 EM, demonstrating that conventional RAG pipelines discard substantial relational information.
- **The deliberate simplicity of TF-IDF edge weights**: In a zero-shot framework, "the right structure" outweighs "sophisticated edge weight computation."
- **Multiplicative GraphRank** ensures semantic dominance and prevents structural scores from introducing noise — an underappreciated but important design detail.

## Limitations & Future Work
- Evaluation is conducted on only 500 samples due to LLM inference costs.
- TF-IDF may miss bridging relationships between semantically synonymous but lexically distinct terms.
- An ~8-point gap remains relative to COS (56.9 EM).
- Table serialization may lose structural information (column types, cross-row relationships).

## Related Work & Insights
- **vs. CORE**: Fine-tuned DPR with entity linking. N2N-GQA achieves comparable zero-shot performance (48.8 vs. 49.0 EM).
- **vs. COS**: Wikipedia-scale pretraining. N2N-GQA trails by ~8 points but is entirely zero-shot.
- **vs. ODYSSEY**: Zero-shot but closed-domain. N2N-GQA addresses the more challenging open-domain setting.
- Graph-structured evidence organization is broadly applicable to any multi-hop RAG scenario.

## Rating
- Novelty: ⭐⭐⭐⭐ The "list → graph" paradigm is simple yet effective; first zero-shot open-domain hybrid QA framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, three readers, and well-designed incremental ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is precise; ablation design is clear.
- Value: ⭐⭐⭐⭐ Demonstrates the importance of graph structure for RAG; directly applicable for zero-shot deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](../../ACL2026/information_retrieval/dqa_diagnostic_question_answering_for_it_support.md)
- [\[AAAI 2026\] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/information_retrieval/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
