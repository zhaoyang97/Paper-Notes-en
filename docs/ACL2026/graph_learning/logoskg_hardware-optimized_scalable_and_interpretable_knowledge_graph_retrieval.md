---
title: >-
  [Paper Note] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval
description: >-
  [ACL 2026][Graph Learning][Knowledge Graph Retrieval] This paper proposes LogosKG, a hardware-aligned knowledge graph (KG) retrieval framework. By transforming graph traversal into multiplication operations of ternary sp…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graph Retrieval"
  - "Hardware-Aligned Optimization"
  - "Multi-Hop Traversal"
  - "Sparse Matrix Operations"
  - "KG-LLM Interaction"
date: 2026-05-08
content_hash: 2779c3815f69af14
---

# LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval

**Conference**: ACL 2026  
**arXiv**: [2604.18913](https://arxiv.org/abs/2604.18913)  
**Code**: [GitHub](https://github.com/LARK-NLP-Lab/LogosKG)  
**Area**: Graph Learning/Knowledge Graph  
**Keywords**: Knowledge Graph Retrieval, Hardware-Aligned Optimization, Multi-Hop Traversal, Sparse Matrix Operations, KG-LLM Interaction

## TL;DR

This paper proposes LogosKG, a hardware-aligned knowledge graph (KG) retrieval framework. By transforming graph traversal into multiplication operations of ternary sparse matrices (SUB/OBJ/REL), combined with degree-aware graph partitioning, cross-graph routing, and on-demand caching, LogosKG achieves scalable and interpretable multi-hop retrieval on billion-edge KGs using a single device. Downstream KG-LLM interaction experiments reveal the impact of graph topology on LLM diagnostic reasoning.

## Background & Motivation

**Background**: The integration of KGs and LLMs is increasingly critical, as KGs provide structured and verifiable reasoning support, particularly in high-stakes fields like medical diagnosis. Multi-hop retrieval is a fundamental KG operation, but traditional traversal algorithms (DFS/BFS) incur $O(|V|+|E|)$ computational costs on large-scale KGs, and reachable entities grow exponentially with the number of hops.

**Limitations of Prior Work**: (1) Biomedical KGs (e.g., UMLS with 407K nodes/3.4M edges; PubMedKG with 54.4M nodes/86.5M edges) consume 1.5–23.5 GB of memory just for loading, and 2-hop expansion can involve $10^9$ reachable edges. (2) Existing systems can only optimize one or two dimensions among matrix representation, scalability, and path reconstruction—GraphBLAS supports matrices but is neither scalable nor capable of path reconstruction; Neo4j is scalable and supports path reconstruction but lacks matrix optimization. (3) GPU frameworks (DGL, PyG) prioritize training over retrieval.

**Key Challenge**: High-hop retrieval requires three attributes simultaneously: digitized matrix operations (to leverage hardware parallelism), scalability (to handle large graphs exceeding memory), and path reconstruction (to support interpretable reasoning). Existing systems fail to satisfy all three.

**Goal**: Construct a unified framework that satisfies these three attributes on single-device hardware and utilize its high-hop retrieval capability to systematically study the impact of KG topology on LLM reasoning.

**Key Insight**: Decomposing the KG into three sparse adjacency matrices (SUB, OBJ, REL) transforms graph traversal into sparse matrix multiplication, which naturally aligns with the parallel computing architectures of CPUs and GPUs.

**Core Idea**: By utilizing KG ternary matrix decomposition + degree-aware partitioning + cross-graph routing + LRU on-demand caching, the theoretical formalization is translated into a practical large-scale retrieval system, achieving a partitioning complexity of $\mathcal{O}(|\mathcal{E}| \log |\mathcal{E}| + |\mathcal{T}|)$.

## Method

### Overall Architecture

LogosKG decomposes a KG into three sparse matrices: SUB (entity $\to$ triplet), OBJ (triplet $\to$ entity), and REL (triplet $\to$ relation). A 1-hop retrieval is represented as $\mathbf{q}^{(1)} = \mathbf{q}^{(0)} \cdot \mathbf{SUB} \cdot \mathbf{OBJ}$, and k-hop retrieval iterates $k$ times. For large graphs exceeding memory, LogosKG divides the KG into balanced subgraphs via degree-aware partitioning, distributes queries through cross-graph routing, and manages loading strategies via LRU caching.

### Key Designs

1.  **Ternary Matrix Decomposition and Path Reconstruction**:
    - **Function**: Transforms graph traversal into hardware-friendly sparse matrix operations while preserving complete path information.
    - **Mechanism**: The KG is decomposed into $\mathbf{SUB} \in \{0,1\}^{|\mathcal{E}| \times |\mathcal{T}|}$, $\mathbf{OBJ} \in \{0,1\}^{|\mathcal{T}| \times |\mathcal{E}|}$, and $\mathbf{REL} \in \{0,1\}^{|\mathcal{T}| \times |\mathcal{R}|}$. In each hop $h$, triplets are activated via $\mathbf{t}^{(h)} = \mathbf{q}^{(h-1)} \cdot \mathbf{SUB}$. For each activated triplet $t$, the subject, relation, and object are read directly from the associated matrices to construct the path $s_h \xrightarrow{r_h} e_h$.
    - **Design Motivation**: Matrix methods like GraphBLAS lose edge provenance during aggregation, making path reconstruction impossible. By retaining the REL matrix and recording activated triplets per hop, LogosKG achieves full path reconstruction without extra overhead.

2.  **Degree-Aware Graph Partitioning and Cross-Graph Routing**:
    - **Function**: Enables LogosKG to process large-scale KGs exceeding memory limits on a single device.
    - **Mechanism**: Triplets are sorted by the degree of the subject entity and distributed alternately to subgraphs (similar to card shuffling), ensuring high-degree nodes are distributed evenly to avoid hotspots. All triplets of the same subject entity are assigned to the same subgraph to maintain structural integrity. A global metadata mapping $P: \mathcal{E} \to \{1, \ldots, m\}$ is maintained for routing. After each hop, results from subgraphs are merged into a global query vector and redistributed to relevant subgraphs for the next hop.
    - **Design Motivation**: Random partitioning might concentrate high-degree nodes in specific subgraphs, causing load imbalance. Degree-aware allocation ensures the computational load of each subgraph is roughly equal.

3.  **LRU On-demand Caching and Batching Optimization**:
    - **Function**: Minimizes disk I/O overhead to keep retrieval latency close to in-memory computation.
    - **Mechanism**: A memory cache with fixed capacity $n$ is maintained using an LRU strategy for subgraph loading/eviction. The expected retrieval cost is $\mathbb{E}[\tau_{\text{retrieval}}] = h \cdot \tau_{\text{mm}} + (1-h) \cdot (\tau_{\text{mm}} + \tau_{\text{io}})$. Temporal locality is optimized through query batching—grouping queries with similar subgraph requirements to improve the cache hit rate.
    - **Design Motivation**: Even with partitioning, loading subgraphs per query triggers frequent I/O. Batching exploits subgraph sharing across queries to reduce cache misses.

### Loss & Training

LogosKG is a deterministic retrieval system and does not involve training. It provides three backend implementations: Numba, SciPy, and Torch (supporting both CPU and GPU).

## Key Experimental Results

### Main Results

**UMLS KG Retrieval Efficiency (Query Time in ms, Timeout Rate %)**

| Method | 1-hop QT | 3-hop QT | 5-hop QT | 5-hop Timeout |
| :--- | :--- | :--- | :--- | :--- |
| NetworkX | 0.21 | 93.92 | 1511.28 | 0.00 |
| igraph | 1.15 | 309.90 | - | - |
| LogosKG (CPU) | ~0.1 | ~10 | ~100 | 0.00 |
| LogosKG (GPU) | ~0.05 | ~5 | ~50 | 0.00 |

### Ablation Study

**PubMedKG (100×UMLS) Scalability**

| Configuration | Description |
| :--- | :--- |
| No Partitioning | Out of Memory (23.5 GB raw data) |
| Partitioning + On-demand Cache | Successfully completed 5-hop retrieval on single device |
| Partitioning + Batching | Further reduced I/O overhead |

### Key Findings

- LogosKG is approximately one order of magnitude faster than NetworkX on UMLS, with the performance gap widening at higher hops (the parallel advantage of matrix operations is more pronounced in deep retrieval).
- On PubMedKG (54.4M nodes/86.5M edges), other single-machine systems failed to complete high-hop retrieval, while LogosKG succeeded via partitioning and caching.
- KG-LLM interaction experiments revealed a structural gap—systematic biases exist between KG topology (hop distribution, connectivity) and LLM diagnostic reasoning.
- Retrieval fidelity is 100%—LogosKG's deterministic matrix operations yield results identical to traditional traversal.

## Highlights & Insights

- Transforming KG traversal into sparse matrix multiplication is simple yet effective—it leverages modern hardware optimizations for matrix operations (SIMD, GPU tensor cores), shifting the system bottleneck from graph traversal to memory management.
- Retaining the REL matrix is a key innovation for path reconstruction—unlike other matrix-based methods (e.g., GraphBLAS) that lose edge information during multiplication, LogosKG maintains interpretability at zero additional overhead via independent triplet-relation mapping.
- The "shuffling" strategy for degree-aware partitioning is both concise and efficient—the $\mathcal{O}(|\mathcal{E}| \log |\mathcal{E}| + |\mathcal{T}|)$ complexity makes preprocessing costs negligible.

## Limitations & Future Work

- The experiments focus primarily on biomedical KGs; although the framework is domain-agnostic, it lacks validation in other domains.
- LRU cache efficiency may decrease when query distributions are highly non-uniform.
- The KG-LLM interaction experiment is a preliminary exploration; the choice of downstream reasoning components (learned vs. non-learned) requires further research.
- No fair comparison was made with distributed systems (e.g., TigerGraph clusters) in extreme large-scale scenarios.

## Related Work & Insights

- **vs Neo4j/TigerGraph**: Database engines provide rich query languages but require distributed infrastructure; LogosKG achieves comparable scalability on a single device through matrix operations.
- **vs GraphBLAS**: Supports matrix operations but lacks path reconstruction and scalability; LogosKG addresses these with the REL matrix and partitioning.
- **vs DGL/PyG**: GPU frameworks prioritize training over retrieval; LogosKG focuses on retrieval efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of ternary matrix decomposition and degree-aware partitioning is an innovative system design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baseline comparisons + scalability validation + KG-LLM interaction, though the domain is somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐ The system paper is clearly written, with complete algorithmic pseudocode and complexity analysis.
- Value: ⭐⭐⭐⭐ Addresses practical system bottlenecks for high-hop retrieval on large KGs; open-source and reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[ACL 2026\] Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval](autonomous_knowledge_graph_exploration_with_adaptive_breadth-depth_retrieval.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[ICLR 2026\] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](../../ICLR2026/graph_learning/ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)

</div>

<!-- RELATED:END -->
