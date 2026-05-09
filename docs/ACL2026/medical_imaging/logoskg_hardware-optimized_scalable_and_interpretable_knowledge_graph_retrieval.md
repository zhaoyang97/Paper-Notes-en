---
title: >-
  [Paper Note] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval
description: >-
  [ACL 2026][Medical Imaging][Knowledge graph retrieval] This paper proposes LogosKG, a hardware-aligned knowledge graph retrieval framework that reformulates graph traversal as multiplications over three sparse associative matrices (SUB/OBJ/REL). Combined with degree-aware graph partitioning, cross-graph routing, and on-demand LRU caching, LogosKG enables scalable and interpretable high-hop retrieval over billion-edge KGs on a single device. Downstream KG-LLM interaction experiments further reveal the systematic influence of graph topology on LLM diagnostic reasoning.
tags:
  - ACL 2026
  - Medical Imaging
  - Knowledge graph retrieval
  - hardware-aligned optimization
  - multi-hop traversal
  - sparse matrix operations
  - KG-LLM interaction
date: 2026-05-08
content_hash: 810f603cb01f60c4
---

# LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval

**Conference**: ACL 2026
**arXiv**: [2604.18913](https://arxiv.org/abs/2604.18913)
**Code**: [GitHub](https://github.com/LARK-NLP-Lab/LogosKG)
**Area**: Medical Imaging
**Keywords**: Knowledge graph retrieval, hardware-aligned optimization, multi-hop traversal, sparse matrix operations, KG-LLM interaction

## TL;DR

This paper proposes LogosKG, a hardware-aligned knowledge graph retrieval framework that reformulates graph traversal as multiplications over three sparse associative matrices (SUB/OBJ/REL). Combined with degree-aware graph partitioning, cross-graph routing, and on-demand LRU caching, LogosKG enables scalable and interpretable high-hop retrieval over billion-edge KGs on a single device. Downstream KG-LLM interaction experiments further reveal the systematic influence of graph topology on LLM diagnostic reasoning.

## Background & Motivation

**Background**: The integration of knowledge graphs with LLMs is increasingly important—KGs provide structured, verifiable reasoning support, particularly in high-stakes domains such as medical diagnosis. Multi-hop retrieval is a fundamental KG operation; however, traditional graph traversal algorithms (DFS/BFS) incur $O(|V|+|E|)$ computational cost on large-scale KGs, and the set of reachable entities grows exponentially with hop depth.

**Limitations of Prior Work**: (1) Biomedical KGs such as UMLS (407K nodes/3.4M edges) and PubMedKG (54.4M nodes/86.5M edges) consume 1.5–23.5 GB of memory at load time, and 2-hop expansion can involve up to $10^9$ reachable edges. (2) Existing systems can optimize at most two of three desiderata—matrix-based representation, scalability, and path reconstruction—simultaneously: GraphBLAS supports matrix operations but lacks scalability and path reconstruction; Neo4j offers scalability and path reconstruction but is not matrix-based; (3) GPU frameworks (DGL, PyG) prioritize training over retrieval.

**Key Challenge**: High-hop retrieval requires all three properties simultaneously—matrix-based operations (to exploit hardware parallelism), scalability (to handle graphs exceeding memory capacity), and path reconstruction (to support interpretable reasoning)—yet no existing system satisfies all three.

**Goal**: To construct a unified framework that simultaneously satisfies all three properties on single-device hardware, and to leverage its high-hop retrieval capability to systematically investigate the effect of KG topology on LLM reasoning.

**Key Insight**: Decomposing a KG into three sparse associative matrices (SUB, OBJ, REL) and reformulating graph traversal as sparse matrix multiplication naturally aligns with the parallel computing architectures of modern CPUs and GPUs.

**Core Idea**: By combining KG ternary matrix decomposition, degree-aware partitioning, cross-graph routing, and LRU on-demand caching, LogosKG translates a formal theoretical representation into a practical large-scale retrieval system, achieving $\mathcal{O}(|\mathcal{E}| \log |\mathcal{E}| + |\mathcal{T}|)$ partitioning complexity.

## Method

### Overall Architecture

LogosKG decomposes a KG into three sparse matrices: SUB (entity→triple), OBJ (triple→entity), and REL (triple→relation). A single-hop retrieval is computed as $\mathbf{q}^{(1)} = \mathbf{q}^{(0)} \cdot \mathbf{SUB} \cdot \mathbf{OBJ}$, and $k$-hop retrieval iterates this operation $k$ times. For graphs exceeding memory capacity, degree-aware partitioning divides the KG into balanced subgraphs; cross-graph routing dispatches queries across partitions; and LRU caching governs subgraph loading and eviction.

### Key Designs

1. **Ternary Matrix Decomposition and Path Reconstruction**:

    - Function: Reformulates graph traversal as hardware-friendly sparse matrix operations while preserving complete path information.
    - Mechanism: The KG is decomposed into $\mathbf{SUB} \in \{0,1\}^{|\mathcal{E}| \times |\mathcal{T}|}$, $\mathbf{OBJ} \in \{0,1\}^{|\mathcal{T}| \times |\mathcal{E}|}$, and $\mathbf{REL} \in \{0,1\}^{|\mathcal{T}| \times |\mathcal{R}|}$. At each hop $h$, triples are activated via $\mathbf{t}^{(h)} = \mathbf{q}^{(h-1)} \cdot \mathbf{SUB}$; for each activated triple $t$, the subject, relation, and object are read directly from the associative matrices to construct the path $s_h \xrightarrow{r_h} e_h$.
    - Design Motivation: Matrix-based methods such as GraphBLAS lose edge provenance during aggregation, precluding path reconstruction. By retaining the REL matrix and recording activated triples at each hop, LogosKG achieves complete path reconstruction at no additional overhead.

2. **Degree-Aware Graph Partitioning and Cross-Graph Routing**:

    - Function: Enables LogosKG to process large-scale KGs that exceed single-device memory on a single machine.
    - Mechanism: Subject entities are sorted by degree and assigned to subgraphs in an interleaved (round-robin) fashion—analogous to dealing cards—ensuring that high-degree nodes are evenly distributed and hotspots are avoided. All triples sharing the same subject entity are assigned to the same subgraph to preserve structural integrity. A global metadata mapping $P: \mathcal{E} \to \{1, \ldots, m\}$ is maintained for cross-partition routing. After each hop, results from all subgraphs are merged into a global query vector and redistributed to the relevant subgraphs for the subsequent hop.
    - Design Motivation: Random partitioning may concentrate high-degree nodes in certain subgraphs, causing load imbalance. Degree-aware assignment ensures approximately equal computational load across subgraphs.

3. **LRU On-Demand Caching and Batch Processing Optimization**:

    - Function: Minimizes disk I/O overhead so that retrieval latency approaches that of purely in-memory computation.
    - Mechanism: A fixed-capacity in-memory cache of size $n$ is maintained, with subgraph loading and eviction governed by an LRU policy. The expected retrieval cost is $\mathbb{E}[\tau_{\text{retrieval}}] = h \cdot \tau_{\text{mm}} + (1-h) \cdot (\tau_{\text{mm}} + \tau_{\text{io}})$. Query batching improves temporal locality by grouping queries with similar subgraph access patterns, thereby increasing cache hit rates.
    - Design Motivation: Even with partitioning, per-query subgraph loading would frequently trigger I/O. Batching exploits inter-query subgraph sharing to reduce cache misses.

### Loss & Training

LogosKG is a deterministic retrieval system and involves no training. Three backend implementations are provided: Numba, SciPy, and Torch (supporting both CPU and GPU).

## Key Experimental Results

### Main Results

**UMLS KG Retrieval Efficiency (Query Time in ms, Timeout Rate in %)**

| Method | 1-hop QT | 3-hop QT | 5-hop QT | 5-hop Timeout |
|--------|---------|---------|---------|--------------|
| NetworkX | 0.21 | 93.92 | 1511.28 | 0.00 |
| igraph | 1.15 | 309.90 | - | - |
| LogosKG (CPU) | ~0.1 | ~10 | ~100 | 0.00 |
| LogosKG (GPU) | ~0.05 | ~5 | ~50 | 0.00 |

### Ablation Study

**PubMedKG (100×UMLS) Scalability**

| Configuration | Description |
|---------------|-------------|
| No partitioning | Out-of-memory (23.5 GB raw data) |
| Partitioning + on-demand caching | Successfully completes 5-hop retrieval on a single device |
| Partitioning + batch optimization | Further reduces I/O overhead |

### Key Findings

- LogosKG is approximately one order of magnitude faster than NetworkX on UMLS, with the performance gap widening as hop depth increases, reflecting the growing parallelism advantage of matrix operations at higher hops.
- On PubMedKG (54.4M nodes/86.5M edges), all other single-machine systems fail to complete high-hop retrieval, whereas LogosKG succeeds via partitioning and caching.
- KG-LLM interaction experiments reveal a structural gap: systematic discrepancies exist between KG topology (hop distribution, connectivity) and LLM diagnostic reasoning.
- Retrieval fidelity is 100%—the deterministic matrix operations of LogosKG produce results fully consistent with traditional traversal methods.

## Highlights & Insights

- Reformulating KG traversal as sparse matrix multiplication is deceptively simple yet remarkably effective: it exploits deep hardware optimizations for matrix operations (SIMD, GPU tensor cores) and shifts the system bottleneck from graph traversal to memory management.
- The retention of the REL matrix is the key innovation enabling path reconstruction. Other matrix-based methods (e.g., GraphBLAS) lose edge information during matrix multiplication; LogosKG achieves interpretability at zero additional cost by independently maintaining a triple-to-relation mapping.
- The "card-dealing" strategy for degree-aware partitioning is elegant and efficient, with $O(|E|\log|E|+|T|)$ complexity rendering preprocessing overhead negligible.

## Limitations & Future Work

- Experiments focus primarily on biomedical KGs; although the framework is domain-agnostic, validation on other domains is lacking.
- LRU caching may become inefficient when query distributions are highly skewed.
- The KG-LLM interaction experiments are exploratory; the choice of downstream reasoning components (learned vs. non-learned) warrants deeper investigation.
- No fair comparison with distributed systems (e.g., TigerGraph clusters) is conducted at large scale.

## Related Work & Insights

- **vs. Neo4j/TigerGraph**: Database engines provide rich query languages but require distributed infrastructure; LogosKG achieves comparable scalability on a single device through matrix-based operations.
- **vs. GraphBLAS**: Matrix-based but lacking path reconstruction and scalability; LogosKG addresses both limitations via the REL matrix and partitioning.
- **vs. DGL/PyG**: GPU frameworks prioritize training over retrieval; LogosKG is specifically designed for retrieval efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of ternary matrix decomposition and degree-aware partitioning constitutes a novel system design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-baseline comparisons, scalability validation, and KG-LLM interaction experiments are included, though the domain coverage is relatively narrow.
- Writing Quality: ⭐⭐⭐⭐ The system paper is clearly written, with complete algorithmic pseudocode and complexity analysis.
- Value: ⭐⭐⭐⭐ Addresses practical system bottlenecks in large-scale KG high-hop retrieval; open-source and reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[NeurIPS 2025\] MedMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph](../../NeurIPS2025/medical_imaging/medmkg_benchmarking_medical_knowledge_exploitation_with_multimodal_knowledge_gra.md)
- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](../../AAAI2026/medical_imaging/nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](../../AAAI2026/medical_imaging/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[ACL 2026\] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)

</div>

<!-- RELATED:END -->
