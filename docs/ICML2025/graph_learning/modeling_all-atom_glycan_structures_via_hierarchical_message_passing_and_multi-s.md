---
title: >-
  [Paper Note] GlycanAA: Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training
description: >-
  [ICML 2025][Graph Learning][Glycan modeling] This work proposes GlycanAA, the first all-atom glycan modeling approach. Glycans are represented as heterogeneous graphs containing atom nodes and monosaccharide nodes. A hierarchical message passing scheme captures multi-scale information ranging from local atomic interactions to global monosaccharide interactions. This is further enhanced by multi-scale masked prediction pre-training (PreGlycanAA)…
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Glycan modeling"
  - "all-atom graphs"
  - "hierarchical message passing"
  - "multi-scale pre-training"
  - "GNN"
date: 2026-05-08
content_hash: 44e56e6ff697cec7
---

# GlycanAA: Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training

**Conference**: ICML 2025  
**arXiv**: [2506.01376](https://arxiv.org/abs/2506.01376)  
**Code**: [https://github.com/kasawa1234/GlycanAA](https://github.com/kasawa1234/GlycanAA)  
**Area**: Graph Learning  
**Keywords**: Glycan modeling, all-atom graphs, hierarchical message passing, multi-scale pre-training, GNN

## TL;DR

This work proposes GlycanAA, the first all-atom glycan modeling approach. Glycans are represented as heterogeneous graphs containing atom nodes and monosaccharide nodes. A hierarchical message passing scheme captures multi-scale information ranging from local atomic interactions to global monosaccharide interactions. This is further enhanced by multi-scale masked prediction pre-training (PreGlycanAA), achieving top performance across 11 tasks on the GlycanML benchmark.

## Background & Motivation

### 1. Importance of Glycans

Glycans are complex macromolecules composed of sugar molecules, playing critical roles in biological processes such as extracellular matrix formation, cell-cell communication, immune response, and cell differentiation.

### Solution Approach

**Goal**: Prior methods modeled glycans as monosaccharide-level graphs, ignoring atom-level structures. Small molecule encoders applied directly to glycans yield poor results because of the scale mismatch, leading to insufficient representation capability.

### Key Challenge

**Key Challenge**: Utilizing the natural hierarchical structure of glycans: atoms constitute the local structures of monosaccharides, and different monosaccharides constitute the global backbone. A hierarchical message passing model is designed to simultaneously capture both scales.

## Method

### Overall Architecture

1. Represent glycans as heterogeneous graphs: atom nodes + monosaccharide nodes + different types of edges.
2. Hierarchical message passing: three levels of interactions (atom-atom, atom-monosaccharide, monosaccharide-monosaccharide).
3. Multi-scale masked prediction pre-training: self-supervised learning on 40,781 unlabeled glycans.

### Key Designs

#### 1. Heterogeneous Graph Representation

- Atom nodes: Encode atom types, charges, and other properties.
- Monosaccharide nodes: Encode monosaccharide types (e.g., Glucose, GlcNAc).
- Edge types: Covalent bonds between atoms, atom-monosaccharide assignment edges, and glycosidic bonds between monosaccharides.

#### 2. Hierarchical Message Passing

- **Atom-Atom**: Propagation within monosaccharides to capture local covalent bond information.
- **Atom-Monosaccharide**: Aggregation of atom features into monosaccharide representations, enabling local-to-global information flow.
- **Monosaccharide-Monosaccharide**: Propagation along the glycan backbone to capture global topological information.

#### 3. Multi-Scale Masked Prediction Pre-training

- Filtered 40,781 high-quality glycans from the GlyTouCan database.
- Randomly masked a portion of atom and monosaccharide nodes, and trained the model to recover them, thereby learning multi-scale dependencies.

## Key Experimental Results

### Main Results: GlycanML Benchmark Ranking

| Method | Type | Average Rank across 11 Tasks | Description |
|------|------|-------------|------|
| **PreGlycanAA** | All-atom + Pre-training | **1st** | Ours |
| **GlycanAA** | All-atom | **2nd** | Without pre-training |
| SweetNet | Monosaccharide-level GNN | 3rd | Prev. SOTA |
| SchNet | Small molecule encoder | 8th | Scale mismatch |

### Ablation Study

| Configuration | Ranking Trend | Description |
|------|---------|------|
| Full PreGlycanAA | Optimal | Hierarchical passing + Pre-training |
| w/o Pre-training | Decrease | GlycanAA still ranks 2nd |
| w/o Atom-level Passing | Significant decrease | Degenerates to monosaccharide-level |
| w/o Monosaccharide-level Passing | Decrease | Loses global topology |
| Single-scale Masking | Decrease | Multi-scale outperforms single-scale |

### Key Findings

- All-atom modeling significantly outperforms monosaccharide-level modeling, validating the value of atomic information.
- Pre-training delivers consistent improvements; multi-scale masking is more effective than single-scale masking.
- Small molecule encoders perform poorly on glycans.

## Highlights & Insights

- **Domain-Specific Architecture Design**: Leverages the natural hierarchical structure of glycans to design heterogeneous graphs and multi-level message passing.
- **Filling the Gap**: Proposes the first effective all-atom level glycan encoder.
- **Value of Self-Supervised Pre-training**: Multi-scale masking allows the model to understand dependencies across different levels.

## Limitations & Future Work

- The caching truncation occurred in the latter part of the method section, and the complete experimental data tables could not be obtained.
- Modeling glycan-protein interactions could serve as a next step for extension.
- 3D spatial coordinates are not utilized, which could be integrated with geometric GNNs.
- The scale of the pre-training dataset (40K) is still relatively small compared to proteins.

## Related Work & Insights

- **vs SweetNet**: Monosaccharide-level GNN that ignores atomic details.
- **vs Protein Pre-training (ESM)**: Applies a similar self-supervised approach to the brand-new glycan domain.
- **vs SchNet**: Performs poorly when applied directly to glycans; the proposed method is a custom solution.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First all-atom glycan modeling
- Experimental Thoroughness: ⭐⭐⭐⭐ Full coverage of GlycanML 11 tasks
- Writing Quality: ⭐⭐⭐⭐ Clear structure
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for glycan computational biology

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond Message Passing: Neural Graph Pattern Machine](beyond_message_passing_neural_graph_pattern_machine.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [\[NeurIPS 2025\] What Expressivity Theory Misses: Message Passing Complexity for GNNs](../../NeurIPS2025/graph_learning/what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](../../AAAI2026/graph_learning/mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[ICML 2025\] EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions](evomesh_adaptive_physical_simulation_with_hierarchical_graph_evolutions.md)

</div>

<!-- RELATED:END -->
