---
title: >-
  [Paper Note] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models
description: >-
  [ICML 2025][Graph Learning][RAG] HippoRAG 2 is proposed to comprehensively outperform standard RAG across factual memory, semantic understanding, and associative reasoning. This is achieved by integrating passage nodes into knowledge graphs, utilizing query-to-triple deep contextualized linking, and employing LLM-driven recognition memory filtering, moving a step closer to non-parametric continual learning for LLMs.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "RAG"
  - "Knowledge Graphs"
  - "Continual Learning"
  - "Personalized PageRank"
  - "Long-Term Memory"
date: 2026-05-08
content_hash: e9df69ab78721bef
---

# From RAG to Memory: Non-Parametric Continual Learning for Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2502.14802](https://arxiv.org/abs/2502.14802)  
**Code**: [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)  
**Area**: Graph Learning  
**Keywords**: RAG, Knowledge Graphs, Continual Learning, Personalized PageRank, Long-Term Memory

## TL;DR

HippoRAG 2 is proposed to comprehensively outperform standard RAG across factual memory, semantic understanding, and associative reasoning. This is achieved by integrating passage nodes into knowledge graphs, utilizing query-to-triple deep contextualized linking, and employing LLM-driven recognition memory filtering, moving a step closer to non-parametric continual learning for LLMs.

## Background & Motivation

Continual Learning is a core capability of human intelligence. However, LLMs face two major challenges when absorbing new knowledge: **catastrophic forgetting** and **high knowledge update costs**. RAG, as a non-parametric alternative, has become the mainstream continual learning solution in production systems. Nonetheless, standard RAG relies on simple vector retrieval and suffers from deficiencies in two aspects:

**Sense-making**: Inability to integrate complex contextual information across passages.

**Associativity**: Inability to perform multi-hop reasoning and connect scattered knowledge fragments.

Although existing structurally augmented RAG methods (such as RAPTOR, GraphRAG, LightRAG, and HippoRAG) improve on their respective target tasks, the authors discover a key issue through comprehensive experimentation: **these methods exhibit significant performance degradation outside of their target tasks**. For instance, HippoRAG excels in multi-hop QA but declines in passage understanding tasks, while RAPTOR regresses significantly on simple QA and multi-hop QA. This "trade-off" phenomenon indicates a significant gap between existing methods and a true human long-term memory system.

The goal of HippoRAG 2 is: **to comprehensively improve performance across factual memory, semantic understanding, and associative reasoning without Corporate-style trade-offs**.

## Method

### Overall Architecture

HippoRAG 2 builds upon the neurobiologically inspired architecture of HippoRAG, comprising two major phases:

**Offline Indexing**:

1. Using an LLM (Llama-3.3-70B-Instruct) via OpenIE to extract triples (subject, relation, object) from each passage to construct a schema-free knowledge graph.
2. Utilizing a retrieval encoder (NV-Embed-v2) to encode phrase nodes, detect synonymy, and add synonym edges.
3. **New**: Integrating original passages as passage nodes into the graph, linked to their derived phrase nodes via "contains" edges (Dense-Sparse Integration).

**Online Retrieval**:

1. Using the encoder to match queries with triples and passages, identifying seed nodes (Query-to-Triple).
2. **New**: Employing an LLM as a recognition memory filter to screen out irrelevant triples (Recognition Memory).
3. Utilizing the filtered seed nodes to set the restart probability of Personalized PageRank (PPR) and performing graph search.
4. Ranking passages by their PageRank scores and retrieving the top-k as the context for QA.

### Key Designs

#### Design 1: Dense-Sparse Integration

Inspired by the theories of dense and sparse coding in the human brain, HippoRAG 2 simultaneously maintains two types of nodes:

- **Phrase Node**: Corresponds to sparse coding, representing conceptual entities extracted from passages; concise but with information loss.
- **Passage Node**: Corresponds to dense coding, preserving the complete context of the original passages.

Specifically, each passage acts as a passage node, connected to all phrase nodes extracted from it via contextual edges labeled "contains". Compared to the document integration in HippoRAG (which simply aggregates graph search and embedding matching scores), this design allows contextual information to **directly participate in the graph search process** rather than being spliced post-processing. Ablation studies show that removing passage nodes drops the average Recall@5 from 87.1% to 81.0%.

#### Design 2: Deeper Contextualization

The original HippoRAG extracts entities from queries via NER and matches them to knowledge graph nodes (NER-to-Node). This process is overly concept-centric and ignores contextual signals. HippoRAG 2 explores three linking strategies:

| Strategy | Description | Performance |
|------|------|------|
| NER to Node | Extracts entities from the query and matches them to KG nodes | Baseline method, Avg Recall 74.6% |
| Query to Node | Matches the entire query directly to KG nodes | Mismatched granularity, Avg Recall 59.6% |
| **Query to Triple** | Matches the entire query to KG triples | **Optimal**, Avg Recall 87.1% |

The core advantage of Query-to-Triple is that triples inherently encapsulate the basic contextual relationships between concepts, capturing the query intent more comprehensively. The informational granularity of queries and triples is better aligned compared to queries and individual nodes.

#### Design 3: Recognition Memory

Drawing inspiration from the complementary mechanisms of recall and recognition in human memory:

1. **Recall Phase**: Retrieving the top-k triples $T$ using an embedding model.
2. **Recognition Phase**: Filtering the retrieved triples utilizing an LLM to generate a subset $T' \subseteq T$, discarding triples irrelevant to the query.

The filtering prompt is automatically tuned using DSPy's MIPROv2 optimizer (including instructions and exemplars). Ablation studies indicate that the filtering mechanism yields an average improvement of approximately 0.7% (86.4% $\to$ 87.1%), which, while incremental, is consistently effective across multiple datasets.

### Loss & Training

HippoRAG 2 does not involve end-to-end training. Its core algorithm is **Personalized PageRank (PPR)**, with key hyperparameters controlled as follows:

- **Seed Node Selection**: Phrase nodes are selected based on their average ranking score in the filtered triples (up to k); all passage nodes are also used as seed nodes, as wider activation facilitates multi-hop reasoning.
- **Restart Probability Allocation**: Phrase nodes are allocated based on ranking scores, while passage nodes are based on embedding similarity multiplied by a weight factor.
- **Weight Factor**: Controls the balance of influence between phrase nodes and passage nodes, empirically set to **0.05**.
- **Triple Filtering Prompt Optimization**: Automatically tuned using DSPy MIPROv2 + Llama-3.3-70B.

## Key Experimental Results

### Main Results

Evaluating three categories of tasks: single-hop QA (factual memory), multi-hop QA (associative reasoning), and long-passage understanding (semantic understanding).

**QA Performance (F1 score, with Llama-3.3-70B as the reader)**:

| Dataset | Type | HippoRAG 2 | NV-Embed-v2 (Strongest Baseline) | Gain |
|--------|------|-------------|----------------------|------|
| NQ | Single-hop QA | **63.3** | 61.9 | +1.4 |
| PopQA | Single-hop QA | 56.2 | 55.7 | +0.5 |
| MuSiQue | Multi-hop QA | **48.6** | 45.7 | +2.9 |
| 2Wiki | Multi-hop QA | 71.0 | 61.5 | +9.5 |
| HotpotQA | Multi-hop QA | **75.5** | 75.3 | +0.2 |
| LV-Eval | Multi-hop QA | **12.9** | 9.8 | +3.1 |
| NarrativeQA | Passage Understanding | **25.9** | 25.7 | +0.2 |
| **Average** | - | **59.8** | 57.0 | **+2.8** |

**Retrieval Performance (Recall@5)**:

| Dataset | HippoRAG 2 | NV-Embed-v2 | HippoRAG | Gain (vs NV) |
|--------|------------|-------------|----------|------------|
| NQ | **78.0** | 75.4 | 44.4 | +2.6 |
| PopQA | 51.7 | 51.0 | **53.8** | +0.7 |
| MuSiQue | **74.7** | 69.7 | 53.2 | +5.0 |
| 2Wiki | **90.4** | 76.5 | **90.4** | +13.9 |
| HotpotQA | **96.3** | 94.5 | 77.3 | +1.8 |
| **Average** | **78.2** | 73.4 | 63.8 | **+4.8** |

### Ablation Study

| Configuration | MuSiQue | 2Wiki | HotpotQA | Avg | Description |
|------|---------|-------|----------|-----|------|
| HippoRAG 2 (Full) | **74.7** | 90.4 | **96.3** | **87.1** | - |
| w/ NER to Node | 53.8 | **91.2** | 78.8 | 74.6 | Concept-level matching loses context |
| w/ Query to Node | 44.9 | 65.5 | 68.3 | 59.6 | Mismatched granularity between query and node |
| w/o Passage Node | 63.7 | 90.3 | 88.9 | 81.0 | Lacks dense coding |
| w/o Filter | 73.0 | 90.7 | 95.4 | 86.4 | Unfiltered noisy triples |

### Key Findings

1. **Comprehensively Outperforming the Strongest Embedding Models**: HippoRAG 2 is the only structurally augmented method that outperforms NV-Embed-v2 across all three dimensions: factual memory, semantic understanding, and associative reasoning.
2. **Query-to-Triple is the Most Crucial Design**: It outperforms NER-to-Node by an average of 12.5% Recall@5, as the information granularity at the triple level matches the query best.
3. **Universal Enhancement for Dense Retrievers**: It consistently brings improvements (+5.0% to 5.6%) across three retrievers: GTE-Qwen2-7B, GritLM-7B, and NV-Embed-v2.
4. **Robustness in Continual Learning**: In simulation experiments with a continuously expanding corpus, HippoRAG 2's advantage over NV-Embed-v2 remains stable.
5. **Fatal Flaws of Other Structurally Augmented Methods**: LightRAG suffers from F1 < 12% on most tasks, while GraphRAG and RAPTOR degrade significantly on single-hop QA.

## Highlights & Insights

1. **Practical Value of Neurobiological Analogy**: Mapping human memory mechanisms with a three-component architecture of the hippocampus (KG + PPR), neocortex (LLM), and parahippocampal gyrus (encoder) is not merely conceptually appealing but also effectively guides design decisions at the engineering level.
2. **Elegant Solution to the Concept-Context Trade-off**: Unlike GraphRAG/LightRAG which utilize KGs to expand retrieval corpora (introducing LLM-generated noise), HippoRAG 2 uses the KG to assist the retrieval process itself while retaining the original context via passage nodes.
3. **Practical Application of DSPy Automatic Prompt Optimization**: Rather than manual design, the prompt for the recognition memory module is automatically tuned using MIPROv2, demonstrating the potential of deploying automated prompt engineering in RAG systems.
4. **Counter-Intuitive Discovery of the 0.05 Weight Factor**: The restart probability of passage nodes needs to be significantly scaled down ($\times 0.05$), indicating that concept-level signals are far more crucial than context-level signals in PPR, where passage nodes function more as "anchors" rather than "dominant elements".

## Limitations & Future Work

1. **Computational Overhead**: Offline indexing requires an LLM to perform OpenIE extraction, and online retrieval requires an LLM for triple filtering, significantly increasing costs compared to pure vector RAG.
2. **Degradation Trend in Associative Reasoning**: In continual learning experiments, multi-hop QA performance continuously declines as the corpus expands. HippoRAG 2 and NV-Embed-v2 decline at similar rates, indicating that the KG structure does not fundamentally solve the information overload issue.
3. **Dependency on OpenIE Quality**: The quality of triple extraction directly affects the KG structure and retrieval effectiveness, showing a strong dependency on the LLM's OpenIE capacity.
4. **Single Graph Search Algorithm**: Only PPR is utilized, leaving the potential of other graph search or graph neural network methods unexplored.
5. **Evaluation Limitations**: NarrativeQA evaluated only 10 documents and 293 queries, leaving the evaluation scale for the semantic understanding dimension relatively small.

## Related Work & Insights

- **HippoRAG** (Gutiérrez et al., 2024): Direct predecessor, which achieves associative reasoning via OpenIE + PPR, but suffers from context loss due to being concept-centric.
- **GraphRAG** (Edge et al., 2024): Uses graph community detection to generate summaries for enhanced semantic understanding, but degrades on single-hop QA and multi-hop QA.
- **RAPTOR** (Sarthi et al., 2024): Organizes corpora via hierarchical summaries, but introduces noise leading to QA degradation.
- **LightRAG** (Guo et al., 2024): Dual-level retrieval but performs extremely poorly in experiments (average F1 of only 6.6%).
- **NV-Embed-v2** (Lee et al., 2025): A SOTA embedding model with 7B parameters, serving as the strongest pure vector RAG baseline.

**Insights**: This work demonstrates that the integration approach of structured knowledge and vector retrieval is crucial — using structure to assist retrieval is superior to using structure to expand corpora. Future RAG system designs should pursue multi-dimensional robustness evaluation rather than metric improvements on a single task.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The three improved designs are each backed by theory and experiments; the overall architecture evolves naturally |
| Experimental Thoroughness | 5 | 7 datasets $\times$ 3 task types, multiple ablation groups, verification across multiple retrievers, and continual learning tests |
| Writing Quality | 4 | Clear structure with its neurobiological analogy consistently threaded throughout |
| Value | 4 | Open-source code that can be directly integrated into existing RAG systems, though computational cost remains a barrier for practical deployment |
| Total Score | 4.25 | Represents a systematic and comprehensive upgrade in the RAG field; the experimental design is highly noteworthy |

## Rating
- Novelty: Pending Review
- Experimental Thoroughness: Pending Review
- Writing Quality: Pending Review
- Value: Pending Review

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation](../../NeurIPS2025/graph_learning/elastic_weight_consolidation_for_knowledge_graph_continual_learning_an_empirical.md)
- [\[ICML 2025\] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [\[AAAI 2026\] Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces](../../AAAI2026/graph_learning/beyond_fact_retrieval_episodic_memory_for_rag_with_generative_semantic_workspace.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)

</div>

<!-- RELATED:END -->
