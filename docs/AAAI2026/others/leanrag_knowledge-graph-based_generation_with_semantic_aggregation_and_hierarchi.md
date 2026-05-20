---
title: >-
  [Paper Note] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval
description: >-
  [AAAI 2026][Retrieval-Augmented Generation] This paper proposes LeanRAG, a framework that employs a semantic aggregation algorithm to automatically construct explicit relations among summary nodes in a hierarchical knowl…
tags:
  - "AAAI 2026"
  - "Retrieval-Augmented Generation"
  - "Knowledge Graph"
  - "Hierarchical Aggregation"
  - "Lowest Common Ancestor"
  - "Semantic Network"
date: 2026-05-08
content_hash: ca9a9fc1d3430f89
---

# LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval

**Conference**: AAAI 2026
**arXiv**: [2508.10391](https://arxiv.org/abs/2508.10391)  
**Code**: [GitHub](https://github.com/RaZzzyz/LeanRAG)  
**Area**: Other
**Keywords**: Retrieval-Augmented Generation, Knowledge Graph, Hierarchical Aggregation, Lowest Common Ancestor, Semantic Network

## TL;DR

This paper proposes LeanRAG, a framework that employs a semantic aggregation algorithm to automatically construct explicit relations among summary nodes in a hierarchical knowledge graph, thereby breaking "semantic islands." Combined with a bottom-up retrieval strategy based on the Lowest Common Ancestor (LCA), LeanRAG efficiently navigates the hierarchical structure, achieving state-of-the-art performance on four QA benchmarks while reducing retrieval redundancy by 46%.

## Background & Motivation

Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by coupling them with external knowledge bases. However, naive RAG's chunk-based retrieval suffers from the "chunking dilemma": fine-grained chunks lose context, while coarse-grained chunks introduce noise. Knowledge-graph-based RAG methods have emerged to address this—GraphRAG organizes documents into community knowledge graphs to preserve local context, and HiRAG further introduces hierarchical structures that cluster entities into multi-level summaries.

Nevertheless, existing hierarchical KG-RAG methods leave two **critical challenges unresolved**:

**Semantic Island Problem**: High-level summary nodes lack explicit relational connections among themselves, leaving them mutually isolated. Cross-community reasoning between different concept communities is impossible, turning high-level knowledge into a collection of disconnected "islands."

**Structure–Retrieval Decoupling Problem**: The retrieval process is topology-agnostic and often degenerates into flat semantic search over all nodes. The carefully constructed hierarchical index is "flattened" during retrieval; rich structural information is used only for post-retrieval context expansion rather than guiding retrieval itself.

The core idea of this paper is to **co-design knowledge structure construction and retrieval strategy**: the aggregation phase not only clusters entities but also automatically infers inter-summary relations (eliminating semantic islands), while the retrieval phase natively exploits the hierarchical structure through LCA path traversal for precise navigation (eliminating structure–retrieval decoupling).

## Method

### Overall Architecture

LeanRAG consists of two core innovations: (1) **Hierarchical Knowledge Graph Aggregation**, which recursively constructs a multi-layer semantic network $\mathcal{H} = \{\mathcal{G}_0, \mathcal{G}_1, \ldots, \mathcal{G}_k\}$ bottom-up from a base KG $\mathcal{G}_0$, with each layer containing progressively more abstract entities and explicit inter-entity relations; and (2) **LCA-Based Structured Retrieval**, which, given a query, first anchors the most relevant fine-grained entities at the base layer, then traverses upward through the hierarchy to the lowest common ancestor, constructing a compact and contextually coherent subgraph as input to the LLM.

### Key Designs

1. **Recursive Semantic Clustering**:

    - Function: Groups entities at each layer by semantic similarity.
    - Mechanism: Entity description texts are encoded into dense vectors $\mathbf{E}_{i-1} = \{\Phi(d_v) \mid v \in V_{i-1}\}$ using a pretrained embedding model, and a Gaussian Mixture Model (GMM) partitions entities into $m$ disjoint clusters $\mathcal{C}_{i-1} = \{C_1, C_2, \ldots, C_m\}$.
    - Design Motivation: GMM is more flexible than hard clustering and can capture soft boundaries between entities. Clustering based on textual descriptions rather than graph structural features ensures semantic coherence.

2. **Aggregated Entity Generation**:

    - Function: Generates a more abstract summary entity for each cluster.
    - Mechanism: For each cluster $C_j$, an LLM-driven generation function $\mathcal{F}_{\text{entity}}$ synthesizes the intra-cluster entities and their relations to produce a new abstract entity and description: $(\alpha_j, d_{\alpha_j}) = \mathcal{F}_{\text{entity}}(C_j, R_{C_j})$. The new entity $\alpha_j$ becomes the parent node of all entities in the cluster.
    - Design Motivation: Rather than simple clustering, intra-cluster relations are considered when generating summary entities, thereby preserving relational information in semantically meaningful abstractions.

3. **Aggregated Relation Generation**:

    - Function: Infers and creates new high-level relations between aggregated entities to break semantic islands.
    - Mechanism: For any pair of aggregated entities $(\alpha_j, \alpha_k)$, the cross-cluster relations $R_{<C_j, C_k>}$ between their underlying clusters $C_j$ and $C_k$ are counted. A connectivity strength $\lambda_{j,k}$ is defined as the number of cross-cluster relations. If $\lambda_{j,k}$ exceeds a dynamic threshold $\tau$, an LLM generates a semantic high-level relation description $r_{<\alpha_j, \alpha_k>} = \mathcal{F}_{\text{rel}}(\alpha_j, \alpha_k, R_{<C_j, C_k>})$; otherwise, the underlying relation texts are directly concatenated.
    - Design Motivation: This is the core innovation distinguishing LeanRAG from all prior methods. Approaches such as HiRAG cluster entities without establishing inter-summary relations, leaving high-level nodes as islands. Explicit relations ensure the entire hierarchical graph remains navigable at every layer.

4. **Initial Entity Anchoring**:

    - Function: Anchors the user query to the most relevant fine-grained entities at the base layer.
    - Mechanism: Dense retrieval is performed exclusively over the entity set of $\mathcal{G}_0$ to identify the top-$n$ semantically most similar "seed entities": $V_{\text{seed}} = \text{Top-n}_{v \in V_0}(\text{sim}(q, d_v))$.
    - Design Motivation: Starting from the finest granularity ensures retrieval precision and avoids imprecise matching directly against high-level summaries.

5. **LCA Path Traversal**:

    - Function: Constructs the minimal compact subgraph connecting the seed entities along the hierarchical structure.
    - Mechanism: The lowest common ancestor $v_{\text{lca}}$ (the common ancestor of minimum depth) of the seed entity set $V_{\text{seed}}$ in $\mathcal{H}$ is identified; all entities and relations along the shortest paths from each seed entity to $v_{\text{lca}}$ are then collected: $\mathcal{P}_{\text{lca}} = \bigcup_{v \in V_{\text{seed}}} \text{ShortestPath}_\mathcal{H}(v, v_{\text{lca}})$.
    - The final retrieved subgraph $\mathcal{G}_{\text{ret}}$ contains path entities $V_{\text{ret}}$, path relations $R_{\text{lca}}$, and inter-cluster relations among aggregated entities at the same layer $R_{\text{inter-cluster}}$, along with the original text chunks corresponding to the base entities as supplementary evidence.
    - Design Motivation: Compared to finding all paths on a flat graph (which introduces numerous noisy intermediate nodes), LCA path traversal constructs a "narrative structure" from specifics to shared concepts, yielding retrieval content that is both compact and semantically coherent. The hierarchical nature ensures complete coverage from concrete facts to high-level concepts.

### Loss & Training

LeanRAG is a training-free retrieval framework with no loss function design. Key hyperparameters include the GMM cluster count, the threshold $\tau$ (controlling the strictness of high-level relation generation, adaptively adjusted per layer), and the number of retrieval anchors $n$.

## Key Experimental Results

### Main Results (Four QA Datasets, 1–10 Scoring, LLM-as-Judge)

| Dataset | Metric | LeanRAG | HiRAG | GraphRAG | NaiveRAG | LightRAG |
|---------|--------|---------|-------|----------|----------|----------|
| Mix | Overall | **8.59** | 8.08 | 7.87 | 7.47 | 7.61 |
| CS | Overall | **8.82** | 8.77 | 8.37 | 8.77 | 8.59 |
| Legal | Overall | **8.49** | 8.00 | 8.44 | 8.21 | 7.74 |
| Agriculture | Overall | **8.87** | 8.87 | 8.85 | 8.69 | 8.56 |
| Mix | Diversity | **7.73** | 7.21 | 7.04 | 6.65 | 6.69 |
| Legal | Empowerment | **8.42** | 8.18 | 8.33 | 8.28 | 7.83 |

### Ablation Study

**RQ3: Effect of Cross-Cluster Relations (Win Rate With vs. Without Relation Paths)**

| Dataset | Metric | LeanRAG win | LeanRAG w/o Relation win |
|---------|--------|-------------|--------------------------|
| Mix | Overall | **53.8%** | 46.2% |
| CS | Overall | **58.5%** | 41.5% |
| Legal | Overall | **56.5%** | 43.5% |
| Agriculture | Overall | **58.0%** | 42.0% |
| CS | Diversity | **66.0%** | 34.0% |

**RQ4: Necessity of Original Text Context**

| Dataset | Metric | LeanRAG | LeanRAG w/o Context |
|---------|--------|---------|---------------------|
| Mix | Overall | **8.59** | 7.93 ↓ |
| CS | Overall | **8.82** | 8.34 ↓ |
| Legal | Overall | **8.49** | 8.00 ↓ |
| Agriculture | Overall | **8.87** | 8.53 ↓ |

**RQ2: Retrieval Redundancy Analysis**

LeanRAG's retrieved context is on average 46% fewer tokens than baseline methods, substantially reducing information redundancy while maintaining or improving answer quality.

### Key Findings
- LeanRAG achieves optimal or comparable performance on nearly all metrics across four datasets, with the most pronounced advantage on the Diversity dimension, reflecting the effect of cross-cluster relations in broadening information sources.
- Removing cross-cluster relations causes the sharpest drop in Diversity (CS dataset: 66% vs. 34% win rate), validating the core value of relational bridging in breaking semantic islands.
- Removing original text leads to the most notable declines in Comprehensiveness and Empowerment, indicating that the graph structure serves as a "semantic indexing and navigation system" while true informational richness comes from the raw text.
- A 46% reduction in retrieval context without quality degradation confirms that LCA path traversal localizes relevant information more precisely than flat search.
- On the Agriculture dataset, LeanRAG is on par with HiRAG, suggesting that for domain-focused, structurally simple scenarios, the additional relation construction in hierarchical aggregation yields limited marginal benefit.

## Highlights & Insights
- The definition of the "semantic island" problem is highly precise and targets the core shortcoming of existing hierarchical KG-RAG methods.
- The aggregation algorithm's principle of "building not only entities but also relations" is simple yet far-reaching—it transforms a fragmented hierarchical tree into a fully connected semantic network.
- LCA path traversal is an elegant retrieval strategy that applies a classical tree-algorithm concept to the RAG retrieval problem.
- The co-design philosophy for retrieval and indexing structures is broadly applicable—the pain point of prior methods lies precisely in their decoupling.
- The experimental design is clear, with four targeted RQs and ablation analyses that quantify the contribution of each framework component.

## Limitations & Future Work
- The hierarchical aggregation process requires multiple LLM calls to generate aggregated entities and relation descriptions, resulting in high construction costs, especially for large-scale knowledge bases.
- The selection of GMM cluster count and threshold $\tau$ relies on validation-set tuning, lacking an automated hyperparameter determination method.
- LCA retrieval assumes an approximately tree-structured hierarchy, which may be less suitable for highly interconnected graph structures.
- Evaluation relies primarily on LLM judgment (DeepSeek-V3), without human evaluation to validate scoring reliability.
- Dynamic knowledge graph update scenarios remain unexplored—when underlying documents change, must the hierarchical structure be fully rebuilt or can it be updated incrementally?
- All experiments use a single LLM (DeepSeek-V3) as the generator, leaving the framework's performance across LLMs of different parameter scales unverified.

## Related Work & Insights
- The evolutionary lineage GraphRAG → LightRAG → HiRAG → LeanRAG is clear: from community graphs to dual-layer frameworks to hierarchical clustering to fully connected hierarchical networks.
- RAPTOR's recursive summarization tree provides early inspiration for hierarchical retrieval but lacks inter-entity relation modeling—LeanRAG supplements this with an explicit relational dimension.
- The LCA concept may inspire other hierarchical retrieval scenarios, such as file system retrieval and ontological reasoning.
- The paradigm of "structure-guided retrieval" also holds reference value for other structured knowledge sources, such as code graphs and mathematical knowledge bases.

## Rating
- Novelty: ⭐⭐⭐⭐ (The semantic island problem definition is novel and the LCA retrieval strategy is creative, though the core technical components are relatively standard.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four datasets, four RQs, and multiple ablations, but human evaluation is absent.)
- Writing Quality: ⭐⭐⭐⭐ (Framework diagrams are clear and problem motivation is well-developed, though some notation definitions could be more concise.)
- Value: ⭐⭐⭐⭐⭐ (RAG is a prominent research area; this paper addresses practical pain points with a practical, open-source framework.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[ICLR 2026\] Exchangeability of GNN Representations with Applications to Graph Retrieval](../../ICLR2026/others/exchangeability_gnn_representations.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](../../ICLR2026/others/learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)

</div>

<!-- RELATED:END -->
