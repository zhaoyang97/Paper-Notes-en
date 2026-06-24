---
title: >-
  [Paper Note] NN-Former: Rethinking Graph Structure in Neural Architecture Representation
description: >-
  [CVPR 2025][Graph Learning][Architecture Prediction] NN-Former proposes a hybrid GNN-Transformer architecture predictor, revealing that existing methods overlook the topological information of "sibling nodes" (nodes sharing parent/child nodes). By introducing Adjacency-Sibling Multihead Attention (ASMA) and Bidirectional Graph Isomorphism FFN (BGIFFN), it achieves a Kendall's Tau of $0.877$/$0.890$ on NAS-Bench-101/201 and reduces the MAPE of latency prediction by 48-64%.
tags:
  - "CVPR 2025"
  - "Graph Learning"
  - "Architecture Prediction"
  - "NAS"
  - "Graph Transformer"
  - "Sibling Nodes"
  - "Latency Prediction"
date: 2026-05-08
content_hash: 4bc87be91f5161f3
---

# NN-Former: Rethinking Graph Structure in Neural Architecture Representation

**Conference**: CVPR 2025  
**arXiv**: [2507.00880](https://arxiv.org/abs/2507.00880)  
**Code**: [https://github.com/XuRuihan/NNFormer](https://github.com/XuRuihan/NNFormer)  
**Area**: Graph Learning  
**Keywords**: Architecture Prediction, NAS, Graph Transformer, Sibling Nodes, Latency Prediction

## TL;DR

NN-Former proposes a hybrid GNN-Transformer architecture predictor, revealing that existing methods overlook the topological information of "sibling nodes" (nodes sharing parent/child nodes). By introducing Adjacency-Sibling Multihead Attention (ASMA) and Bidirectional Graph Isomorphism FFN (BGIFFN), it achieves a Kendall's Tau of $0.877$/$0.890$ on NAS-Bench-101/201 and reduces the MAPE of latency prediction by 48-64%.

## Background & Motivation

1. **Background**: Performance predictors in neural architecture search (NAS) need to predict accuracy or latency from architecture graphs. Existing methods are divided into GNN-based (GATES, GMAE-NAS) and Transformer-based (NAR-Former V2, PINAT).
2. **Limitations of Prior Work**: (1) GNNs only propagate adjacency information, ignoring "sibling" relationships (e.g., two parallel branches sharing input/output); (2) The global attention of Transformers does not distinguish between different types of graph structural relationships; (3) Existing predictors lack accuracy under low-data regimes (0.02% training data).
3. **Key Challenge**: "Sibling relationships" in graph structures (such as two branches of a residual connection) contain crucial design information but cannot be directly observed in the adjacency matrix $A$ (requiring $AA^T$ or $A^TA$), which are entirely missed by existing methods.
4. **Goal**: Design a hybrid predictor that simultaneously utilizes adjacency and sibling topologies.
5. **Key Insight**: Use $AA^T$ to extract siblings sharing a parent node (e.g., two paths of a residual block) and $A^TA$ to extract siblings sharing a child node (e.g., multi-input aggregation).
6. **Core Idea**: ASMA (four-head attention, each using a different topological mask) + BGIFFN (Bidirectional Graph Isomorphism Feed-Forward Network).

## Method

### Overall Architecture

Architecture graph $\to$ one-hot operation encoding + sinusoidal attribute encoding $\to$ $L$-layer NN-Former (ASMA + BGIFFN) $\to$ global pooling $\to$ MLP predicting accuracy or latency.

### Key Designs

1. **Adjacency-Sibling Multihead Attention (ASMA)**

    - **Function**: Simultaneously models four topological relationships: adjacency and sibling relationships.
    - **Mechanism**: Four attention heads utilize different topological masks—Head 1: $(I+A)$ forward adjacency; Head 2: $(I+A^T)$ backward adjacency; Head 3: $(I+AA^T)$ sibling sharing a parent; Head 4: $(I+A^TA)$ sibling sharing a child. Each head calculates: $X_i = \sigma((Q_iK_i^T \circ Mask_i)/\sqrt{h})V_i$
    - **Design Motivation**: Global attention in standard Transformers treats all node pairs equivalently, but in a computational graph, the importance of adjacent, parent-child, and sibling relationships varies significantly.

2. **Bidirectional Graph Isomorphism FFN (BGIFFN)**

    - **Function**: Injects graph structural information into the feed-forward network.
    - **Mechanism**: $H_g = \text{Concat}(\text{GC}(H, A), \text{GC}(H, A^T))$, which concatenates graph convolutions in both forward and backward directions, and then fuses them with the original features: $\text{BGIFFN}(H, A) = \text{ReLU}(HW_1 + H_g)W_2$. No positional encodings are required.
    - **Design Motivation**: Standard FFNs process each node independently. Injecting graph convolution allows the FFN to be topographically aware.

3. **Positional Encoding-Free Design**

    - **Function**: BGIFFN naturally provides positional information.
    - **Mechanism**: Since BGIFFN performs forward and backward graph convolutions at each layer, node positional information is implicitly encoded through multi-layer propagation.
    - **Design Motivation**: Nodes in a graph do not have a fixed order (unlike sequences), making traditional positional encodings inapplicable.

### Loss & Training

L1 loss (for accuracy prediction) or MAPE loss (for latency prediction). AdamW optimizer.

## Key Experimental Results

### Main Results

| Setting | NAR-Former V2 | **NN-Former** | Gain |
|------|--------------|-------------|------|
| NAS-101 (0.02%) | 0.663 | **0.709** | +6.9% |
| NAS-101 (1%) | 0.861 | **0.877** | +1.9% |
| NAS-201 (1%) | 0.752 | **0.804** | +6.9% |
| NAS-201 (10%) | 0.888 | **0.890** | +0.2% |
| Latency-EfficientNet | 13.20 | **4.81** | -63.6% |
| Latency-MnasNet | 7.16 | **2.54** | -64.5% |

### Ablation Study

| Configuration | NAS-101 (0.1%) Tau | Description |
|------|-------------------|------|
| w/o Siblings (Adjacency only) | ~0.77 | Information loss |
| w/o BGIFFN | ~0.78 | FFN lacks graph awareness |
| Cross-attention alternative | 0.804 | Suboptimal |
| **Full NN-Former** | **0.809** | Optimal |

### Key Findings

- The performance gain is most significant under low-data regimes (0.02% scale) (+6.9%), showing that sibling information is vital for data efficiency.
- The improvements in latency prediction are the most remarkable (48-64% MAPE reduction), as latency is directly influenced by parallel structures, and sibling relationships precisely encode parallelism.
- The positional encoding-free design is an engineering advantage—simplifying the pipeline and naturally adapting to graphs of varying sizes.

## Highlights & Insights

- **Discovery of Sibling Nodes**: The topological relationships extracted by $AA^T$ and $A^TA$ were overlooked by all prior work in NAS—simple yet highly impactful.
- **4-Head Topological Attention**: Designing each head to focus on one specific structural relationship is more efficient and effective than global attention.
- **Practical Value of Latency Prediction**: While accuracy prediction is research-oriented, latency prediction is deployment-oriented. The massive advantage of NN-Former in latency makes it highly valuable for real-world model deployment.

## Limitations & Future Work

- Sibling calculation requires $O(n^2)$ matrix multiplications ($AA^T$, $A^TA$), which might become a bottleneck for exceptionally large graphs.
- Evaluated primarily on cell-based architectures; custom architectures (such as MoE, branching networks) have not been extensively verified.
- The design of 4 heads (each mapping to one topology) is hardcoded—more heads or different topological combinations have not been explored.

## Related Work & Insights

- **vs NAR-Former V2**: A pure Transformer approach that does not differentiate between topological types. NN-Former explicitly encodes four relationships via ASMA.
- **vs GATES/GMAE-NAS**: GNN approaches propagate adjacency information only. NN-Former bridges the blind spot of GNNs through sibling relationships.

## Rating

- Novelty: ⭐⭐⭐⭐ Sibling node discovery and 4-head topological attention are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ NAS-101/201 accuracy + NNLQ latency + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Offers direct improvement value for NAS predictors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Banyan: Improved Representation Learning with Explicit Structure](../../ICML2025/graph_learning/banyan_improved_representation_learning_with_explicit_structure.md)
- [\[ICLR 2026\] Structure-Aware Graph Hypernetworks for Neural Program Synthesis](../../ICLR2026/graph_learning/structure-aware_graph_hypernetworks_for_neural_program_synthesis.md)
- [\[NeurIPS 2025\] From Sequence to Structure: Uncovering Substructure Reasoning in Transformers](../../NeurIPS2025/graph_learning/from_sequence_to_structure_uncovering_substructure_reasoning_in_transformers.md)
- [\[ICML 2025\] Beyond Message Passing: Neural Graph Pattern Machine](../../ICML2025/graph_learning/beyond_message_passing_neural_graph_pattern_machine.md)
- [\[ICML 2026\] Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](../../ICML2026/graph_learning/rethinking_feature_alignment_in_generalist_graph_anomaly_detection_a_relational_.md)

</div>

<!-- RELATED:END -->
