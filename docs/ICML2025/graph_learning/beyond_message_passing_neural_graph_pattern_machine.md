---
title: >-
  [Paper Note] Beyond Message Passing: Neural Graph Pattern Machine
description: >-
  [ICML2025][Graph Learning][Graph Patterns] This paper proposes the Neural Graph Pattern Machine (GPM), which samples graph patterns using random walks. It utilizes dual encoders for semantic and anonymous paths to capture node features and topological structures, respectively. A Transformer is then employed to identify task-relevant pattern tokens, bypassing the message-passing paradigm entirely and outperforming SOTA methods across node-, edge-, and graph-level tasks.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Graph Patterns"
  - "Random Walks"
  - "Message Passing"
  - "Substructure Encoding"
  - "Graph Transformer"
date: 2026-05-08
content_hash: b7242d35811745db
---

# Beyond Message Passing: Neural Graph Pattern Machine

**Conference**: ICML2025  
**arXiv**: [2501.18739](https://arxiv.org/abs/2501.18739)  
**Code**: [GitHub - GPM](https://github.com/Zehong-Wang/GPM)  
**Area**: Graph Learning  
**Keywords**: Graph Patterns, Random Walks, Message Passing, Substructure Encoding, Graph Transformer

## TL;DR
This paper proposes the Neural Graph Pattern Machine (GPM), which samples graph patterns using random walks. It utilizes dual encoders for semantic and anonymous paths to capture node features and topological structures, respectively. A Transformer is then employed to identify task-relevant pattern tokens, bypassing the message-passing paradigm entirely and outperforming SOTA methods across node-, edge-, and graph-level tasks.

## Background & Motivation

### Key Challenge

**Key Challenge**: Standard GNNs rely on message passing, leaving them limited by the 1-WL test in terms of expressive power:
- Inability to distinguish basic substructures like triangles, k-cliques, and cycles.
- Multi-hop message passing suffers from over-squashing, making it difficult to capture long-range dependencies.
- Poor interpretability.

### Why Substructures/Graph Patterns Matter

- Triangular closure in social networks acts as an indicator of community stability.
- Benzene rings in molecular graphs serve as core substructures for chemical reactivity.
- Inductive biases for downstream tasks are inherently embedded within graph patterns.

### Limitations of Prior Work

**Limitations of Prior Work**: 1. Lack of a general-purpose graph tokenizer.
2. Encoders still rely on message passing, inheriting its expressive limitations.
3. Lack of mechanisms to identify "which patterns are most important for the task".

## Method

### Overall Architecture
1. **Pattern Tokenizer**: Random walk sampling
2. **Pattern Encoder**: Dual encoding of semantic and anonymous paths
3. **Important Pattern Identifier**: Transformer attention-based selection

### Step 1: Tokenizer Based on Random Walks
Each random walk naturally corresponds to a graph pattern:
- **Semantic path** $(v_0,...,v_L)$: Retains node identities and features.
- **Anonymous path** $(\gamma_0,...,\gamma_L)$: Replaces nodes with the index of their first appearance.

For example, "A-B-C-A-D" and "C-D-E-C-A" both map to "0-1-2-0-3" (the same topology).

Theoretical Guarantee: The distribution of anonymous paths is sufficient to reconstruct $k$-hop subgraphs (Proposition 3.2).

### Step 2: Dual Encoder
- Semantic paths use a Transformer to encode feature sequences.
- Anonymous paths use a GRU to encode loop-based adjacency.
- Final representation: $p = \rho_s(w) + \lambda \cdot \rho_a(\phi)$

### Step 3: Transformer Pattern Recognition
Self-attention learns the relative importance of patterns. Optional class tokens can be used to improve interpretability. After mean pooling, the output is fed into a prediction head.

## Key Experimental Results

### Comprehensive Evaluation Across Tasks


### Main Results

| Task Type | Example Dataset | Performance of GPM |
|----------|-----------|----------|
| Node Classification | Cora, ogbn-arxiv | Top-1 in most cases |
| Link Prediction | Collab, PPA | Outperforms GNN+GT |
| Graph Classification | MUTAG, ogbg-molhiv | Top-1 in most cases |
| Graph Regression | ZINC | Outperforms all message-passing methods |

### Comparison with Different Paradigms


### Ablation Study

| Method Type | Representative | Relative Performance of GPM |
|----------|------|-------------|
| Standard GNNs | GCN, GAT, GIN | Significantly superior |
| Higher-order GNNs | GSN, CWN | Superior or comparable |
| Graph Transformers | GPS, Graphormer | Superior in most cases |
| Random Walk Methods | CRaWl | Superior |

### Key Findings
1. Consistently outperforms message passing across all four task categories, showing stronger OOD generalization.
2. Capable of distinguishing non-isomorphic graphs that GNNs fail to differentiate.
3. Performs exceptionally well on datasets with long-range dependencies.
4. Scales to large graphs and supports distributed training.
5. Class token attention provides interpretable rankings of pattern importance.

## Highlights & Insights

1. Bypasses the message-passing paradigm entirely to propose a brand-new paradigm for graph learning.
2. The dual-path decomposition of random walks into "semantic + anonymous" representations is highly elegant.
3. Exceptionally versatile: the same framework handles node-, edge-, and graph-level tasks.
4. Interpretability is built-in rather than an after-the-fact patch.
5. Solid theoretical foundation: formal proof is provided for the sufficient statistics of the anonymous path distribution.

## Limitations & Future Work

1. Randomness introduced by pattern sampling may cause higher variance in small graphs.
2. The length of random walks and the number of samples require fine-tuning.
3. Adaptation to directed and heterogeneous graphs warrants further exploration.
4. GRU encoding of anonymous paths might suffer from information decay on extremely long walks.

## Related Work & Insights

- Difference from Graph Transformers: Tokens are graph patterns (substructures) rather than individual nodes.
- Difference from higher-order GNNs: Operates directly in the pattern space without executing message passing.
- Inspiration: Pattern-level masked pre-training or adaptive walking strategies could be developed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (5.0/5) — Completely breaks away from the message-passing paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5.0/5) — Across four task categories + OOD + scalability
- Writing Quality: ⭐⭐⭐⭐⭐ (5.0/5)
- Value: ⭐⭐⭐⭐⭐ (5.0/5) — A paradigm-shifting contribution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative Graph Pattern Machine](../../NeurIPS2025/graph_learning/generative_graph_pattern_machine.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [\[NeurIPS 2025\] What Expressivity Theory Misses: Message Passing Complexity for GNNs](../../NeurIPS2025/graph_learning/what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)
- [\[ICML 2025\] GlycanAA: Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training](modeling_all-atom_glycan_structures_via_hierarchical_message_passing_and_multi-s.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)

</div>

<!-- RELATED:END -->
