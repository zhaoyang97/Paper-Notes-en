---
title: >-
  [Paper Note] Learning Repetition-Invariant Representations for Polymer Informatics
description: >-
  [NeurIPS 2025][Graph Learning][Polymer Informatics] This paper proposes GRIN (Graph Repetition-Invariant Network), which achieves invariance to the number of repeated monomer units in polymer representations via Max aggregation and a specialized graph construction strategy, addressing a fundamental symmetry problem in polymer representation learning.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Polymer Informatics
  - Repetition Invariance
  - Graph Neural Networks
  - Max Aggregation
  - Molecular Representation
date: 2026-05-08
content_hash: 5a3ea91c903148f5
---

# Learning Repetition-Invariant Representations for Polymer Informatics

**Conference**: NeurIPS 2025
**arXiv**: [2505.10726](https://arxiv.org/abs/2505.10726)
**Code**: Available
**Area**: Graph Learning / Materials Science
**Keywords**: Polymer Informatics, Repetition Invariance, Graph Neural Networks, Max Aggregation, Molecular Representation

## TL;DR

This paper proposes GRIN (Graph Repetition-Invariant Network), which achieves invariance to the number of repeated monomer units in polymer representations via Max aggregation and a specialized graph construction strategy, addressing a fundamental symmetry problem in polymer representation learning.

## Background & Motivation

### State of the Field

**State of the Field**: Polymers are long-chain molecules composed of repeating monomer units, and property prediction is a central problem in materials science.

**Limitations of Prior Work**: Standard GNNs produce different predictions for different repeat-unit representations of the same polymer (1 repeat vs. 3 repeats vs. N repeats), violating a fundamental symmetry of polymers — repetition invariance.

**Root Cause**: The aggregation operations of GNNs (Sum/Mean) are sensitive to graph size and cannot naturally handle "representations of the same molecule at different scales."

**Starting Point**: Max aggregation is naturally invariant to the number of repetitions (taking the maximum is unaffected by repetition count), but requires a specialized graph construction to function correctly.

**Core Idea**: Max aggregation + cyclization (connecting chain head and tail to form a ring) + a sufficient number of message-passing layers.

## Method

### Overall Architecture

Input SMILES → Construct polymer graph (repeat unit + cyclization) → Max-GNN encoding → Global Max pooling → Property prediction.

### Key Designs

1. **Theoretical Foundation for Repetition Invariance**

    - **Function**: Proves that Max aggregation is a necessary and sufficient condition for achieving repetition invariance.
    - **Mechanism**: Theorem 1 establishes that a GNN is fully invariant to repetition count if and only if two conditions are met: (1) the number of message-passing layers $\geq$ half the graph diameter; and (2) Max aggregation is used.
    - **Design Motivation**: Sum/Mean aggregation varies linearly or constantly with repetition count, violating invariance.

2. **Cyclization Strategy**

    - **Function**: Connects the head and tail of a linear polymer chain to form a cyclic graph.
    - **Mechanism**: The terminal attachment points (denoted by *) are linked, making the graph of 1 repeat unit topologically equivalent to that of N repeat units.
    - **Design Motivation**: Without cyclization, the neighborhood of terminal atoms differs across different repetition counts, breaking invariance.

3. **GRIN Architecture**

    - **Function**: A complete repetition-invariant GNN architecture.
    - **Mechanism**: Any MPNN backbone (GIN, GAT, etc.) with Max aggregation replacing Sum/Mean, cyclic graph input, and global Max pooling.
    - **Design Motivation**: Modular design allows any MPNN variant to be used as a plug-and-play backbone.

### Loss & Training

Standard regression/classification losses (MSE/BCE) are used with no special training strategy. The key contribution lies in producing consistent predictions at inference time regardless of the number of repeat units.

## Key Experimental Results

### Main Results (Polymer Property Prediction)

| Method | Glass Transition Temp. MAE↓ | Band Gap MAE↓ | Dielectric Constant MAE↓ | Repetition-Invariant? |
|---|---|---|---|---|
| GNN-Sum | 15.3 | 0.45 | 0.12 | ✗ |
| GNN-Mean | 14.8 | 0.42 | 0.11 | ✗ |
| Fingerprint | 16.2 | 0.48 | 0.14 | ✓ |
| **GRIN** | **13.1** | **0.38** | **0.09** | **✓** |

### Ablation Study

| Configuration | Glass Transition Temp. MAE | Repetition Invariance Error |
|---|---|---|
| Sum aggregation | 14.8 | 8.5% |
| Mean aggregation | 15.1 | 3.2% |
| Max aggregation, no cyclization | 14.2 | 1.8% |
| **Max aggregation + cyclization** | **13.1** | **<0.01%** |

### Key Findings

- GRIN achieves optimal performance simultaneously in prediction accuracy and repetition invariance.
- Sum/Mean aggregation incurs repetition-count errors of 8.5% and 3.2%, respectively; GRIN achieves <0.01%.
- Cyclization is a necessary condition for achieving full invariance.
- GRIN consistently outperforms baselines across all three property prediction benchmarks.

## Highlights & Insights

- **Theory-Driven Design**: The method is derived from symmetry principles rather than empirical heuristics. The formal proof of Max aggregation's repetition invariance constitutes the core contribution.
- **Plug-and-Play**: Replacing Sum/Mean with Max and adding cyclization upgrades any existing GNN to a repetition-invariant variant with minimal modification.
- **Materials Science Significance**: Resolves a longstanding representational ambiguity in the field, enabling more reliable polymer property predictions.

## Limitations & Future Work

- Max aggregation may discard count-based information (e.g., atom count), which could be detrimental for certain properties.
- Only homopolymers (single repeat unit) are considered; extension to copolymers remains future work.
- With very large numbers of message-passing layers, Max aggregation may suffer from over-smoothing.

## Related Work & Insights

- **vs. SchNet/DimeNet**: These are 3D molecular GNNs that do not account for the specific nature of polymers; GRIN specifically addresses repetition invariance.
- **vs. Fingerprint Methods**: Traditional fingerprints are naturally repetition-invariant but have limited expressive power; GRIN achieves both invariance and expressiveness.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Theory-driven method design
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-property validation + theoretical verification
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous proofs, clear exposition
- Value: ⭐⭐⭐⭐ A foundational contribution to the materials science domain

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[NeurIPS 2025\] Principled Data Augmentation for Learning to Solve Quadratic Programming Problems](principled_data_augmentation_for_learning_to_solve_quadratic_programming_problem.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)

<!-- RELATED:END -->
