---
title: >-
  [Paper Note] TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation
description: >-
  [ICML 2025][Graph Learning][GNN distillation] The authors propose TINED, which directly injects the parameters of feature transformations (FT) in GNNs into MLPs (Teacher Injection) and transfers the opposing smoothing properties of FT and graph propagation (GP) in GNN layers using Dirichlet energy distillation. TINED outperforms GNN teachers across 7 datasets while achieving a 94x speedup in inference.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "GNN distillation"
  - "GNN-to-MLP"
  - "Teacher Injection"
  - "Dirichlet Energy"
  - "inference acceleration"
date: 2026-05-08
content_hash: 71d7891707611547
---

# TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation

**Conference**: ICML 2025  
**arXiv**: [2412.11180](https://arxiv.org/abs/2412.11180)  
**Code**: [https://github.com/scottjiao/TINED_ICML25/](https://github.com/scottjiao/TINED_ICML25/)  
**Area**: Graph Learning  
**Keywords**: GNN distillation, GNN-to-MLP, Teacher Injection, Dirichlet Energy, inference acceleration

## TL;DR

The authors propose TINED, which directly injects the parameters of feature transformations (FT) in GNNs into MLPs (Teacher Injection) and transfers the opposing smoothing properties of FT and graph propagation (GP) in GNN layers using Dirichlet energy distillation. TINED outperforms GNN teachers across 7 datasets while achieving a 94x speedup in inference.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: GNN message passing requires multi-hop neighborhood data, making it difficult to deploy in latency-sensitive scenarios.

### 2. Limitations of Existing Distillation

GLNN only uses soft labels for distillation, ignoring fine-grained knowledge within GNN layers.

### 3. Key Observations

- FT is computationally equivalent to the FC layers of MLPs.
- FT and GP exhibit opposing effects on smoothing: GP performs aggressive smoothing, while FT is conservative or even diversifying.

## Method

### Overall Architecture

1. **Teacher Injection**: Directly transfer parameters of the GNN's FT to the MLP's FC layers, followed by fine-tuning.
2. **Dirichlet Energy Distillation**: Transfer the opposing smoothing properties of FT/GP to the MLP using Dirichlet energy (DE) ratio.

### Key Designs

#### 1. Teacher Injection

- FT and FC share the same mathematical form: $h' = \sigma(Wh + b)$.
- Directly copy parameters and utilize another FC layer to simulate GP.
- Under theoretically proven conditions, GP can be approximated by FC, where the error bounds are related to the eigenvalues of the graph Laplacian.

#### 2. Dirichlet Energy Distillation

- DE ratio > 1 indicates conservative behavior (diversifying), while < 1 indicates aggressive behavior (smoothing).
- The distillation loss matches the DE ratio of each MLP layer with its corresponding GNN layer.

## Key Experimental Results

### Main Results: Node Classification

| Method | Citeseer | Cora | PubMed | Speed |
|------|---------|------|--------|-----|
| GCN Teacher | 73.1% | 81.5% | 79.0% | 1x |
| MLP | 61.2% | 60.0% | 71.4% | 94x |
| GLNN | 74.0% | 81.6% | 79.8% | 94x |
| NOSMOG | 75.5% | 82.3% | 80.5% | 94x |
| **TINED** | **77.0%**| **83.2%** | **81.3%** | **94x** |

### Ablation Study

| Configuration | Citeseer | Description |
|------|---------|------|
| TINED Full | 77.0% | TI + DE |
| w/o TI | 74.8% | Degenerates to soft labels |
| w/o DE | 75.6% | Loses smoothing transfer |
| Soft Label Only (GLNN) | 74.0% | Baseline |

### Key Findings

- Teacher Injection contributes +2.2%, while DE Distillation contributes +1.4%.
- Through distillation, the MLP can outperform the GNN teacher — achieving both high speed and high accuracy ("fast and good").
- Inference speed is accelerated by 94x.

## Highlights & Insights

- **Ingenious Parameter Transfer**: Discovering the equivalence of FT = FC and directly migrating parameters rather than doing indirect distillation.
- **Discovery of Opposing Smoothing**: An internal structural property of GNNs that was previously unnoticed.
- **Theoretical Guarantees**: Error bounds for the GP -> FC approximation.
- **Outperforming the Teacher**: MLP students outperform GNN teachers across multiple datasets.

## Limitations & Future Work

- Only tested on node classification tasks; graph-level and edge-level tasks remain to be validated.
- Adaptation to attention-based GNNs like GAT is left for future work.
- The behavior of DE ratio in deep GNNs has not been fully explored.
- Further improvements can be achieved by integrating structure-aware methods like VQGraph.

## Related Work & Insights

- **vs GLNN**: GLNN only employs soft label distillation, whereas TINED incorporates parameter injection and energy distillation.
- **vs NOSMOG**: NOSMOG considers graph structure but performs legacy overall distillation, whereas TINED performs layer-by-layer distillation.
- **vs VQGraph**: VQGraph learns structure-aware tokenizers, whereas TINED directly migrates parameters.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ FT=FC equivalence + opposing smoothing + layer-by-layer distillation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 datasets x various teachers
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theory, novel observations
- Value: ⭐⭐⭐⭐⭐ An efficient solution for GNN acceleration

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs](../../ICML2026/graph_learning/eralign_energy-based_representation_alignment_of_gnns_and_llms_on_text-attribute.md)
- [\[ICML 2025\] Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality](balancing_efficiency_and_expressiveness_subgraph_gnns_with_walk-based_centrality.md)
- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [\[ICML 2025\] Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes](machines_and_mathematical_mutations_using_gnns_to_characterize_quiver_mutation_c.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)

</div>

<!-- RELATED:END -->
