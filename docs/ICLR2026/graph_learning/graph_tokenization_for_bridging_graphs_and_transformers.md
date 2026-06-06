---
title: >-
  [Paper Note] Graph Tokenization for Bridging Graphs and Transformers
description: >-
  [ICLR 2026][Graph Learning][graph tokenization] This paper proposes GraphTokenizer, a framework that converts graphs into symbol sequences via invertible frequency-guided serialization…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "graph tokenization"
  - "BPE"
  - "graph serialization"
  - "Transformer"
  - "graph classification"
date: 2026-05-08
content_hash: 72ba8292f0837a78
---

# Graph Tokenization for Bridging Graphs and Transformers

**Conference**: ICLR 2026
**arXiv**: [2603.11099](https://arxiv.org/abs/2603.11099)  
**Code**: Available (provided in supplementary material)  
**Area**: Graph Learning
**Keywords**: graph tokenization, BPE, graph serialization, Transformer, graph classification

## TL;DR
This paper proposes GraphTokenizer, a framework that converts graphs into symbol sequences via invertible frequency-guided serialization, then applies BPE to learn a substructure vocabulary, enabling standard Transformers (e.g., BERT/GTE) to process graph data directly without any architectural modification, achieving state-of-the-art results on 14 benchmarks.

## Background & Motivation

**Background**: Graph-structured data learning primarily relies on GNNs (aggregating neighbor information via message passing) or specially designed Graph Transformers (incorporating attention mechanisms for graphs). Another line of work maps graphs to continuous embeddings for Transformer consumption.

**Limitations of Prior Work**: (a) Graph Transformers require graph-specific architectural designs, precluding direct reuse of pretrained models and training recipes from the LLM ecosystem; (b) mapping graphs to continuous embeddings often incurs information loss or representation instability; (c) existing graph serialization methods (Random Walk, BFS/DFS) are either non-invertible (losing edge connectivity information) or non-deterministic (producing different sequences for isomorphic graphs).

**Key Challenge**: Text is naturally a path graph with fixed neighborhood structure and ordering—tokenization is straightforward. In general graphs, however, neighborhoods can branch in multiple directions, there is no canonical node ordering, and co-occurrence statistics such as n-grams cannot be directly applied.

**Goal**: Design a general graph tokenizer that faithfully converts any labeled graph into a discrete token sequence, enabling standard sequence models to process graph data directly.

**Key Insight**: Decompose graph tokenization into two steps—(1) invertible and deterministic graph serialization, and (2) learning a substructure vocabulary from the serialized corpus using BPE. The key insight is that by guiding serialization with global frequency statistics, common substructures appear adjacently in the sequence, which aligns perfectly with BPE's greedy merge strategy.

**Core Idea**: Frequency-guided invertible graph serialization + BPE = a graph tokenizer that reformulates graph learning as sequence modeling.

## Method

### Overall Architecture
GraphTokenizer $\Phi = T \circ f$: an input labeled graph $\mathcal{G}$ is passed through frequency-guided invertible serialization $f_g(\mathcal{G}, F)$ to produce a symbol sequence, which is then compressed by BPE tokenizer $T$ into a token sequence $S_T$ fed into a standard Transformer (BERT/GTE) for downstream tasks. The original graph can be fully reconstructed via the inverse operation $f^{-1} \circ T^{-1}$.

### Key Designs

1. **Local Structural Pattern Statistics**:

    - Function: Computes the frequency $F(p)$ of all local edge patterns $p = (l_u, l_e, l_v)$ over the training corpus.
    - Mechanism: Edge patterns are the minimal substructures preserving the typed relationship between two entities; they are computationally efficient and invariant under node permutations. The frequency map $F$ serves as global statistical information to guide subsequent serialization.
    - Design Motivation: Provides deterministic priority for serialization while ensuring that high-frequency substructures appear adjacently in the sequence, creating ideal input for BPE.

2. **Frequency-Guided Invertible Serialization (Frequency-Guided Eulerian Circuit)**:

    - Function: Traverses every edge of the graph exactly once along an Eulerian circuit, selecting the next edge according to frequency priority.
    - Mechanism: Each undirected edge is split into two directed edges, making any connected graph admit an Eulerian circuit. At each node $u$, among unvisited outgoing edges $\mathcal{E}_u$, the algorithm selects $e^* = \arg\max_{e_i \in \mathcal{E}_u} \pi(e_i, F)$ with priority $\pi(e_i, F) = F(p_i)$. Traversal outputs an alternating node-edge-node sequence, guaranteeing invertibility.
    - Design Motivation: (a) Eulerian circuits are inherently invertible since every edge is visited; (b) frequency guidance resolves the non-determinism of the classical Hierholzer algorithm, producing identical sequences for isomorphic graphs; (c) time complexity is only $O(|\mathcal{E}|)$.
    - Distinction from Prior Methods: Random Walk is non-invertible; BFS/DFS lose edge information; SMILES is applicable only to molecular graphs.

3. **BPE Vocabulary Learning**:

    - Function: Trains BPE on the serialized corpus, iteratively merging the most frequent adjacent symbol pair into a new token.
    - Mechanism: Because serialization has arranged high-frequency substructures as adjacent symbols, BPE's greedy merge strategy naturally discovers semantically meaningful graph substructure tokens. Each merged token corresponds to a decodable subgraph fragment.
    - Design Motivation: (a) Compresses sequence length (~10× compression ratio), reducing Transformer computational overhead; (b) the learned vocabulary is hierarchical and semantically meaningful (e.g., functional groups in molecules); (c) entirely data-driven, requiring no domain knowledge.

4. **Frequency-Guided CPP (Chinese Postman Problem)**:

    - Function: An alternative approach that covers all edges with minimum-weight traversal.
    - Mechanism: Frequency information is encoded into edge weights $w(e) = \alpha \cdot 1 + (1-\alpha) \cdot g(F(p_e))$, making high-frequency edges more likely to be traversed consecutively.
    - Design Motivation: CPP already produces highly structured sequences; the marginal gain from frequency guidance is limited. However, CPP has complexity $O(|\mathcal{V}|^3)$, far exceeding Feuler's $O(|\mathcal{E}|)$.

### Loss & Training
The tokenizer itself requires no gradient-based training (BPE is a statistical algorithm). Downstream training uses standard Transformer objectives:
- GT+BERT: BERT-small architecture with a prepended [CLS] token.
- GT+GTE: Larger GTE model (~BERT-base parameter count).
- Standard classification/regression losses.

## Key Experimental Results

### Main Results

| Model | molhiv (AUC↑) | p-func (AP↑) | mutag (Acc↑) | zinc (MAE↓) | qm9 (MAE↓) |
|-------|--------------|-------------|-------------|------------|------------|
| GCN | 74.0 | 53.2 | 79.7 | 0.399 | 0.134 |
| GIN | 76.1 | 61.4 | 80.4 | 0.379 | 0.176 |
| GraphGPS | 78.5 | 53.5 | 84.3 | 0.310 | 0.084 |
| GraphMamba | 81.2 | 67.7 | 85.0 | 0.209 | 0.083 |
| GCN+ | 80.1 | **72.6** | 88.7 | 0.116 | 0.077 |
| GT+BERT | 82.6 | 68.5 | 87.5 | 0.241 | 0.122 |
| **GT+GTE** | **87.4** | 73.1 | **90.1** | **0.131** | **0.071** |

GT+GTE achieves state-of-the-art on the majority of 14 benchmarks using a standard off-the-shelf Transformer with no graph-specific architectural modifications.

### Ablation Study

| Serialization | molhiv (w/ BPE) | molhiv (w/o BPE) | zinc (w/ BPE) | zinc (w/o BPE) |
|--------------|----------------|-----------------|-------------|--------------|
| BFS | 72.3 | 81.2 | 0.453 | 0.696 |
| DFS | 76.0 | 79.1 | 0.446 | 0.705 |
| Eulerian | 84.5 | 81.0 | 0.164 | 0.160 |
| **Feuler** | **87.4** | 81.3 | **0.131** | 0.171 |
| CPP | 86.9 | 81.2 | 0.141 | 0.145 |

### Key Findings
- **Invertible serialization is critical**: Edge-traversal methods (Eulerian/CPP) substantially outperform node-traversal methods (BFS/DFS/TOPO) by preserving complete edge connectivity information.
- **Frequency guidance is effective**: Feuler consistently outperforms unguided Eulerian serialization with lower variance.
- **BPE is a key component**: It improves performance in nearly all configurations while delivering ~10× sequence compression and 2.5× training speedup.
- **The model scales well**: Scaling from GT+BERT to GT+GTE yields consistent performance gains, unlike GNNs that tend to degrade due to over-smoothing.
- The vocabulary learned by BPE carries chemical semantics: token size distribution peaks at 4–6 nodes (corresponding to functional groups), with atom-level tokens comprising only 7.1%.

## Highlights & Insights
- **Paradigm shift in graph learning**: Graph learning is fully reformulated as sequence modeling, decoupling data representation from model architecture. This means the graph domain can directly benefit from all advances in the LLM ecosystem (longer context, more efficient attention, pretrained models, etc.).
- **Co-design of frequency-guided serialization and BPE**: The two components are not simply concatenated but carefully co-designed—serialization creates ideal compressible input for BPE, while BPE in turn discovers meaningful substructures. This co-design principle is transferable to tokenization of other non-sequential data.
- **Invertibility as a quality guarantee**: The invertibility of serialization is not merely a theoretical virtue; it directly correlates with better downstream performance—lossless information preservation is the foundation of high-quality tokenization.

## Limitations & Future Work
- Validation is limited to graph-level tasks (classification/regression); node-level and edge-level prediction tasks remain unexplored.
- Serialization assumes connected graphs; disconnected graphs must be serialized separately and concatenated.
- Frequency statistics are based on the simplest edge pattern $(l_u, l_e, l_v)$; statistical guidance derived from larger substructure patterns has not been explored.
- BPE is a greedy algorithm and may not find a globally optimal substructure decomposition.
- The pretraining-then-finetuning paradigm has not been explored—pretraining the tokenizer and Transformer on large-scale graph corpora could yield further improvements.

## Related Work & Insights
- **vs. GraphGPS**: GraphGPS is a purpose-built Graph Transformer requiring graph-specific components such as positional encodings; GT uses off-the-shelf Transformers and achieves superior performance.
- **vs. GraphMamba**: GraphMamba combines serialization with the Mamba architecture, but its serialization is neither invertible nor deterministic; GT's serialization has stronger theoretical properties.
- **vs. GCN+**: GCN+ is a strengthened GNN baseline that approaches GT+GTE on some datasets but lags substantially on molhiv (80.1 vs. 87.4).
- **vs. SMILES**: SMILES is a domain-specific serialization for molecular graphs that does not generalize to arbitrary graphs; GT is applicable to any labeled graph.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Fully transplants the NLP tokenization paradigm to the graph domain; conceptually original and thoroughly executed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 14 benchmarks, comprehensive ablations, efficiency analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is precise, theoretical framework is rigorous, and figures are well-designed.
- Value: ⭐⭐⭐⭐⭐ — Potentially transformative for the graph learning research paradigm, enabling standard Transformers to be applied directly to graph data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[ICML 2026\] What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA](../../ICML2026/graph_learning/what_structural_inductive_bias_helps_transformers_reason_over_knowledge_graphs_a.md)
- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)
- [\[NeurIPS 2025\] Relieving the Over-Aggregating Effect in Graph Transformers](../../NeurIPS2025/graph_learning/relieving_the_over-aggregating_effect_in_graph_transformers.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)

</div>

<!-- RELATED:END -->
