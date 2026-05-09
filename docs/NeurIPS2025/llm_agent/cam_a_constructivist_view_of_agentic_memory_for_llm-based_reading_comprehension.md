---
title: >-
  [Paper Note] CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension
description: >-
  [NeurIPS 2025][LLM Agent][Agentic memory] Inspired by Piaget's constructivist theory, this paper proposes CAM — an agentic memory system characterized by three properties: structuredness (hierarchical schema), flexibility (assimilation via overlapping clustering), and dynamism (incremental adaptation). CAM comprehensively outperforms baselines such as RAPTOR and GraphRAG across six long-document reading comprehension benchmarks.
tags:
  - NeurIPS 2025
  - LLM Agent
  - Agentic memory
  - long-document understanding
  - hierarchical memory
  - incremental clustering
  - constructivism
date: 2026-05-08
content_hash: c4ff4c4149be37eb
---

# CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension

**Conference**: NeurIPS 2025
**arXiv**: [2510.05520](https://arxiv.org/abs/2510.05520)
**Code**: [https://github.com/rui9812/CAM](https://github.com/rui9812/CAM)
**Area**: Agent
**Keywords**: Agentic memory, long-document understanding, hierarchical memory, incremental clustering, constructivism

## TL;DR
Inspired by Piaget's constructivist theory, this paper proposes CAM — an agentic memory system characterized by three properties: structuredness (hierarchical schema), flexibility (assimilation via overlapping clustering), and dynamism (incremental adaptation). CAM comprehensively outperforms baselines such as RAPTOR and GraphRAG across six long-document reading comprehension benchmarks.

## Background & Motivation

**Background**: When LLMs process extremely long documents (e.g., novels, meeting transcripts), they face information overload. The mainstream approach is to equip LLMs with external memory modules that store documents in chunks for subsequent retrieval. Existing methods fall into two categories: unstructured memory (MemGPT, ReadAgent) and structured memory (RAPTOR, GraphRAG, MemTree).

**Limitations of Prior Work**: (a) Unstructured memory (tabular storage) fails to capture inter-information associations, making recall difficult when key information is scattered; (b) Although structured memory is preferable, RAPTOR and GraphRAG are offline methods requiring full reconstruction upon each update; MemTree supports online updates but only allows one-by-one insertion and does not support overlapping clustering; (c) There is a lack of principled design guidelines for memory systems.

**Key Challenge**: How can a memory system simultaneously achieve structuredness, flexible assimilation (where a single unit can belong to multiple higher-level abstractions), and dynamic structural adaptation (incremental updates rather than full reconstruction)?

**Goal**: To provide a design blueprint for memory systems (three key properties) and implement a prototype that satisfies all three simultaneously.

**Key Insight**: Drawing from Piaget's constructivist theory in cognitive science — memory is a cognitive structure that continuously evolves through *assimilation* (incorporating new information into existing schemas) and *accommodation* (adjusting schemas to fit new information).

**Core Idea**: Unify the constructivist operations of assimilation (flexibility) and accommodation (dynamism) through an incremental overlapping clustering algorithm, thereby constructing a hierarchical memory that supports batched online updates.

## Method

### Overall Architecture

CAM consists of two phases: **memory construction** (Reading Phase) and **memory retrieval** (Inference Phase).

During the construction phase, text chunks are organized into a multi-level hierarchical structure: the bottom layer is a semantic network $G_0$ composed of raw text chunks, while upper-level nodes are LLM-generated summaries of tightly related lower-level nodes. Batched online updates are enabled by an incremental overlapping clustering algorithm.

During the retrieval phase, a Prune-and-Grow strategy is employed: global semantic matching first rapidly locates relevant nodes, followed by recursive relational exploration along the memory structure.

### Key Designs

1. **Foundational Network Expansion**:

    - Function: Integrates new text chunks into the bottom-level semantic network $G_0$.
    - Mechanism: A composite similarity score $s(v_i, v_j) = \alpha \cdot \cos(f_{emb}(v_i), f_{emb}(v_j)) + (1-\alpha) \cdot \exp(-\frac{(i-j)^2}{2\sigma^2})$ is computed for each pair of text chunks, combining semantic similarity and positional proximity. For each new chunk, top-$k$ related nodes exceeding threshold $\theta$ are selected to establish edges.
    - Design Motivation: Pure semantic similarity ignores narrative coherence (adjacent passages may be semantically dissimilar yet logically related); positional proximity compensates for this deficiency.

2. **Ego-Centric Disentanglement**:

    - Function: Enables **flexible assimilation** through overlapping clustering — allowing a single bottom-level node to contribute to multiple higher-level abstractions simultaneously.
    - Mechanism: For each node $v$, its ego-network (neighbor subgraph) is extracted and decomposed into connected components $\{C_v^1, ..., C_v^{t_v}\}$. A replica of $v$ is created for each component, forming a replica network $\tilde{G}_0$. This transforms the overlapping clustering problem into a non-overlapping clustering problem on the replica network.
    - Design Motivation: Particularly elegant — overlapping clustering algorithms typically incur high computational complexity. By combining ego-network decomposition with node replication, overlapping behavior can be realized using a simple and efficient label propagation algorithm. The approach also naturally supports incremental updates (only replicas of affected nodes need to be updated) and parallelization.

3. **Online Clustering Updates**:

    - Function: Incrementally updates cluster assignments on the replica network, realizing **dynamic accommodation**.
    - Mechanism: An incremental label propagation algorithm is used — new replicas are initialized with unique labels, while affected existing replicas retain their previous labels. Labels are iteratively propagated within the affected subgraph until convergence, with label changes recursively propagating to neighbors (ripple effect). For modified clusters, LLM regenerates summary nodes, upper-level connections are updated, and disentanglement and clustering updates are recursively triggered at higher layers.
    - Design Motivation: A locality-first strategy — only affected portions are updated rather than globally reconstructing, achieving $O(\text{local})$ rather than $O(n)$ update complexity.

4. **Prune-and-Grow Retrieval Strategy**:

    - Function: Adaptively retrieves query-relevant information from the memory structure.
    - Mechanism: Two stages — (1) **Rapid localization**: globally computes embedding similarity between the query and all memory nodes, selecting the top-$s$ nodes to form candidate set $D$; (2) **Relational exploration**: the LLM selects useful nodes from $D$ to form an activation set $P$, collects same-level neighbors and lower-level children of nodes in $P$ as new candidates, and iteratively expands $P$ until no further growth occurs.
    - Design Motivation: Combines the coverage of global retrieval with the precision of local structural exploration, avoiding the risk of missing relevant bottom-level information that may occur with pure hierarchical traversal.

### Loss & Training
- No training is required — GPT-4o-mini serves as the LLM backbone and text-embedding-3-small as the encoder.
- Text chunk size: 512 tokens.
- All baselines use the same LLM and embedding model to ensure fair comparison.

## Key Experimental Results

### Main Results

| Method | Type | NovelQA ACC-L | QMSum ACC-L | FABLES F1P | MH-RAG F1 | ODSum-S ACC-L | ODSum-M ACC-L |
|--------|------|--------------|-------------|-----------|-----------|--------------|--------------|
| MemGPT | Unstructured | 40.8 | 40.3 | 81.3 | 67.1 | 36.8 | 33.2 |
| ReadAgent | Unstructured | 42.3 | 45.5 | 84.2 | 65.5 | 39.4 | 35.7 |
| RAPTOR | Structured | 47.8 | 50.7 | 86.5 | 73.6 | 48.7 | 44.3 |
| GraphRAG | Structured | 45.3 | 53.9 | 87.2 | 70.3 | 50.2 | 45.8 |
| MemTree | Structured | 43.5 | 48.3 | 83.1 | 71.2 | 46.0 | 42.5 |
| **CAM** | **Structured** | **52.3** | **57.6** | **91.5** | **77.5** | **54.6** | **50.7** |

CAM achieves comprehensive superiority on all metrics across all six datasets, with an average improvement of 3.0%.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Without overlapping clustering (flexibility removed) | Performance drops ~2%, validating the necessity of overlapping clustering |
| RAPTOR retrieval vs. CAM retrieval | CAM's Prune-and-Grow outperforms RAPTOR's global retrieval/traversal |
| Online batched update efficiency | RAPTOR/GraphRAG require >1 hour for reconstruction; MemTree scales linearly with batch size; CAM exhibits sublinear growth and is 4× faster than offline methods |
| Inference stability across batch sizes | CAM maintains stable performance across different batch sizes, approaching offline-level performance |

### Key Findings
- **Structured > Unstructured**: Structured methods comprehensively outperform unstructured ones, with ACC-L gaps of approximately 5–10%.
- **Flexibility is critical**: RAPTOR (with overlapping support) consistently outperforms MemTree (without overlapping support), with an average gap of 2.1%.
- **Dynamism enables practical viability**: CAM is the only method that simultaneously supports batched online updates while maintaining offline-level performance.
- Knowledge graph-based modeling (GraphRAG/HippoRAG) is less effective than hierarchical clustering with summarization for narrative-style texts.

## Highlights & Insights
- **Ego-centric disentanglement** is the most elegant design in this paper: it reduces the overlapping clustering problem to non-overlapping clustering on a replica network — a solution that is both concise and efficient, while naturally supporting incremental updates and parallelization. This technique is transferable to any scenario requiring incremental overlapping clustering.
- The introduction of the **constructivist theoretical framework** is not merely conceptual — the three properties (structuredness, flexibility, dynamism) are precisely mapped to technical designs (hierarchical clustering, overlapping clustering, incremental clustering), and experiments validate the independent contribution of each property.
- **Prune-and-Grow retrieval** combines the advantages of both global and local approaches, offering greater flexibility than pure global retrieval or hierarchical traversal alone.

## Limitations & Future Work
- The system relies on GPT-4o-mini for summarization and retrieval judgment; effectiveness with open-source models remains unverified.
- The similarity threshold $\theta$ and top-$k$ for the bottom-level semantic network require tuning; robustness across different document types is insufficiently analyzed.
- For extremely long documents (e.g., millions of tokens), the rapid localization stage — which globally computes embedding similarity across all memory nodes — may become a bottleneck.
- The Gaussian positional proximity assumption may be inappropriate for non-sequential documents (e.g., web page collections).
- Replacing LLM calls with lighter-weight summarization models to reduce cost is a promising direction.

## Related Work & Insights
- **vs. RAPTOR**: RAPTOR also uses hierarchical clustering with overlapping, but is an offline method requiring full reconstruction upon each update. CAM extends RAPTOR with incremental and online capabilities.
- **vs. GraphRAG**: GraphRAG models documents as knowledge graphs, which is advantageous for entity-rich texts but less effective for narrative ones. CAM's clustering-based summarization approach is more general.
- **vs. MemTree**: MemTree supports online updates but lacks overlapping clustering and only allows one-by-one insertion. CAM comprehensively surpasses MemTree.
- The paper offers broadly applicable insights for agentic memory system design: structuredness, flexibility, and dynamism can serve as a three-dimensional evaluation framework for memory systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of mapping constructivist theory to a technical blueprint is novel; ego-centric disentanglement is a key contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets, seven baselines, online/offline comparisons, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative chain from theory to blueprint to implementation to validation is clear and coherent.
- Value: ⭐⭐⭐⭐ Offers practical guidance for the design of long-document agentic memory systems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[NeurIPS 2025\] Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents](agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)
- [\[NeurIPS 2025\] Agentic NL2SQL to Reduce Computational Costs](agentic_nl2sql_to_reduce_computational_costs.md)
- [\[NeurIPS 2025\] Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading](orchestration_framework_for_financial_agents_from_algorithmic_trading_to_agentic.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)

<!-- RELATED:END -->
