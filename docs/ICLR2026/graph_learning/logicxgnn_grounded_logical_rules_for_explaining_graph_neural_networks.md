---
title: >-
  [Paper Note] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks
description: >-
  [ICLR 2026][Graph Learning][GNN explainability] LogicXGNN proposes a post-hoc framework for extracting interpretable first-order logical rules from trained GNNs. The framework identifies predicates via graph structural h…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "GNN explainability"
  - "logical rule extraction"
  - "decision trees"
  - "knowledge discovery"
  - "graph generation"
date: 2026-05-08
content_hash: 71411807b4ef41a7
---

# LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2503.19476](https://arxiv.org/abs/2503.19476)
**Code**: None
**Area**: Graph Learning
**Keywords**: GNN explainability, logical rule extraction, decision trees, knowledge discovery, graph generation

## TL;DR

LogicXGNN proposes a post-hoc framework for extracting interpretable first-order logical rules from trained GNNs. The framework identifies predicates via graph structural hashing and hidden-layer embedding pattern recognition, determines discriminative DNF rule structures using decision trees, and grounds abstract predicates into the input space. The resulting rule-based classifier can serve as a substitute for the original GNN and also functions as a controllable graph generation model.

## Background & Motivation

Graph Neural Networks (GNNs) have achieved remarkable success in drug discovery, fraud detection, and recommendation systems, yet their black-box nature hinders deployment in high-reliability domains such as healthcare. Existing GNN explainability methods exhibit notable shortcomings:

- **Local attribution methods** (e.g., GNNExplainer, PGExplainer) explain predictions for individual instances and cannot provide a global description of model behavior.
- **Concept-based global methods** (e.g., GCNeuron, GLGExplainer) rely on predefined concepts and produce rules that characterize patterns within a single class without effectively discriminating across classes.
- GLGExplainer suffers from severely limited expressiveness—for example, it uses only 2 predicates on the Mutagenicity dataset, yielding performance close to a random classifier.

The core problem addressed by this paper: can logical rules that are simultaneously interpretable, discriminative, and model-agnostic be extracted from GNNs?

## Method

### Overall Architecture

LogicXGNN decomposes the problem into three sequential subtasks: (1) identifying the hidden predicate set $P$; (2) determining the DNF logical rule structure $\phi_M$; and (3) grounding the rules into the input feature space. The final output comprises descriptive rules $\bar{\phi}_M$ (for graph generation) and discriminative rules $\hat{\phi}_M$ (for classification).

### Key Designs

1. **Hidden Predicate Identification (Section 3.1)**:

    - **Structural patterns**: For each node $v$, the receptive-field subgraph spanning its 1-to-$L$-hop neighborhood is hashed to produce a structural signature $\text{Pattern}_{struct}(v) = \text{Hash}(\text{ReceptiveField}(v, A, L))$.
    - **Embedding patterns**: A decision tree is trained on final-layer graph embeddings to identify the most discriminative dimension set $K$ and thresholds $T$; these thresholds are broadcast to node embedding layers to generate binary activation states per node per dimension.
    - **Predicate definition**: Predicate $p_j$ is defined as a combined tuple of structural and embedding patterns; it evaluates to true for node $v$ if and only if both components match.
    - **Design Motivation**: Message passing in GNNs causes over-smoothing that blurs structural information; explicit structural pattern extraction is therefore introduced as a complement.

2. **Logical Rule Structure Determination (Section 3.2)**:

    - For each class $c$, predicate activation patterns from all correctly classified instances are collected to form a binary matrix $\Phi_c$.
    - Each row encodes a conjunctive clause (AND rule); the disjunction (OR) of all rows constitutes the descriptive rule $\bar{\phi}_M^c$.
    - Inter-predicate connectivity patterns $\psi_M$ (in adjacency matrix form) are jointly learned for graph generation and motif-level rule grounding.
    - $\Phi$ and labels $Y$ are fed into a second decision tree to produce discriminative rules $\hat{\phi}_M$ for cross-class distinction.

3. **Input-Space Grounding (Section 3.3)**:

    - For each node $v$, subgraph input features are constructed as $Z_{v,L} = \text{CONCAT}(X_v, \text{Encode}(\{f(u) | u \in N^{(1)}(v)\}), \ldots, \text{Encode}(\{f(u) | u \in N^{(L)}(v)\}))$.
    - For predicate pairs that are structurally isomorphic but differ in embedding patterns, a decision tree learns discriminating rules based on input features.
    - For predicates without isomorphic counterparts, feature dimensions with minimum variance are extracted as representative descriptors.
    - Motif-level rules can be derived by combining connectivity patterns $\psi_M$—for instance, inferring "whether a cycle exists" from "two nodes of degree 2."

4. **LogicXGNN as a Generative Model**:

    - A conjunctive clause is sampled from the descriptive rule $\bar{\phi}_M$.
    - A predicate-level graph template is constructed by combining the connectivity pattern $\psi_M$.
    - Node features are assigned to each predicate according to grounding rules $R$.
    - New graph instances preserving class-specific properties can be generated in a controllable and transparent manner.

### Loss & Training

LogicXGNN is a post-hoc method and does not modify the original GNN training. Computational overhead stems primarily from decision tree training (applied at three stages), comparable to the complexity of training a DT. All experiments were conducted on Ubuntu 22.04 with 32 GB RAM and a 2.7 GHz processor. GNNs use 2 layers with hidden dimension 32. Datasets are split 8:2 for training and testing. The CART algorithm is used for all decision trees.

## Key Experimental Results

### Main Results

| Dataset | $|P|$ | $|\hat{\phi}_M^0|$ | $|\hat{\phi}_M^1|$ | $\phi_M$ Train Acc. | $\phi_M$ Test Acc. | GNN Test Acc. | GLGExplainer Test Acc. |
|---------|-------|-------|-------|-------|-------|-------|-------|
| HIN | 2293 | 146 | 62 | 89.12 | 85.23 | 86.93 | 49.99 |
| BBBP | 254 | 20 | 22 | 81.18 | **81.37** | 81.13 | 42.90 |
| Mutagenicity | 314 | 198 | 167 | 78.12 | 76.50 | 76.04 | 54.26 |

### Ablation Study

| Configuration | Key Feature | Description |
|--------------|-------------|-------------|
| Descriptive rules $\bar{\phi}_M$ | BBBP: 146+1044 clauses | Full coverage of intra-class variation; usable for generation |
| Discriminative rules $\hat{\phi}_M$ | BBBP: 20+22 clauses | Compact yet achieves accuracy comparable to or exceeding the GNN |
| GLGExplainer | 2 predicates | Overly simplified; performance near random |
| GCNeuron | Class-conditioned input | No discriminative power; produces entirely different explanations per class |

### Key Findings

- **Rule-based models can outperform neural networks**: On BBBP, $\hat{\phi}_M$ achieves a test accuracy of 81.37%, surpassing the original GNN's 81.13%; performance is also competitive on Mutagenicity.
- **Accurate chemical knowledge discovery**:
    - BBBP dataset: Molecules with oxygen-rich functional groups are found to be less likely to cross the blood-brain barrier, consistent with increased hydrophilicity and reduced lipophilicity.
    - MUTAG dataset: Methyl groups (−CH3) are associated with non-mutagenicity, while the combination of aromatic rings and nitro groups is identified as critical to mutagenicity—consistent with findings reported in the original study (Debnath et al., 1991).
    - Methods such as GCNeuron and GLGExplainer treat nitro groups and carbon rings as independent entities, overlooking their synergistic interaction.
- **Controllable graph generation outperforms XGNN**: XGNN (an RL-based generation method) produces graph structures inconsistent with real molecular distributions (e.g., bipartite graphs), whereas LogicXGNN adheres to learned rules, enabling reconstruction of original molecules and generation of new instances that preserve key structural properties.

## Highlights & Insights

- This is the first interpretable rule-based model proposed as a functional equivalent of a GNN—not merely an explanation, but a substitute for the original model.
- The three-stage design is highly modular: predicate discovery, rule organization, and input-space grounding are each decoupled.
- Decision trees are employed three times with distinct purposes: (1) extracting key embedding dimensions, (2) generating discriminative rules, and (3) grounding rules into the input space.
- Motif-level rule grounding is a notable strength: connectivity patterns elevate node-level rules to subgraph-level semantic understanding.
- The application as a generative model is particularly promising, especially for combinatorial generation in drug design.

## Limitations & Future Work

- The experimental scope is limited (three datasets), lacking validation on large-scale graph benchmarks.
- Graph hashing may produce an explosion of unique structural patterns on large graphs, causing the predicate set to scale poorly.
- The number of descriptive rules can be very large (e.g., 1,044 clauses for class 1 on BBBP), limiting human readability.
- The method currently supports only classification tasks and has not been extended to regression (e.g., molecular property prediction, 3D structure prediction).
- No analysis is provided on the scalability of the method as the number of nodes, classes, or layers increases.
- A systematic comparison with local explanation methods such as GNNExplainer in terms of explanation quality is absent.

## Related Work & Insights

- The work draws inspiration from NeuroLogic (Geng et al., 2025), which extracts logical rules from hidden-layer activation patterns in fully connected networks and CNNs.
- A direct comparison is drawn with GLGExplainer (Azzolin et al., ICLR 2023): the latter relies on local explanations from PGExplainer, whereas LogicXGNN is entirely data-driven.
- Insight: Combining the translation of neural networks into logic programs with program synthesis could potentially further improve the performance of rule-based models.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](../../ACL2026/graph_learning/from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](../../NeurIPS2025/graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
