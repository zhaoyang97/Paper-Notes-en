---
title: >-
  [Paper Note] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction
description: >-
  [NeurIPS 2025][Graph Learning][Link Prediction] This paper identifies redundancy and over-smoothing issues in higher-order common neighbors (CN) for link prediction…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Link Prediction"
  - "Higher-Order Common Neighbors"
  - "Orthogonalization"
  - "Gram-Schmidt"
  - "Resource Allocation Heuristic"
date: 2026-05-08
content_hash: 303c0e5f06ec2119
---

# OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2505.19719](https://arxiv.org/abs/2505.19719)
**Code**: [GitHub](https://github.com/qingpingmo/OCN)
**Area**: Graph Learning / Link Prediction
**Keywords**: Link Prediction, Higher-Order Common Neighbors, Orthogonalization, Gram-Schmidt, Resource Allocation Heuristic

## TL;DR
This paper identifies redundancy and over-smoothing issues in higher-order common neighbors (CN) for link prediction, and proposes orthogonalization (Gram-Schmidt to remove inter-order linear dependence) combined with normalization (dividing by path count, a generalized resource allocation heuristic) as a solution. The method achieves an average improvement of 7.7% in HR@100 across 7 datasets, with a 13.3% gain on the DDI dataset.

## Background & Motivation
**Background**: Link prediction is a core task in graph learning. Common neighbors (CN) serve as a classic strong baseline — the more neighbors two nodes share, the more likely an edge exists between them. Higher-order CNs (reachable via 2-hop, 3-hop paths) can capture structural information at greater distances.

**Limitations of Prior Work**: (a) CNs of different orders are highly redundant — 2-hop and 1-hop CN counts are strongly correlated, yielding diminishing returns when combined; (b) higher-order CNs cause over-smoothing — CN features at the 3-hop level and beyond become nearly identical across different node pairs, losing discriminability due to excessively broad neighborhood coverage.

**Key Challenge**: Higher-order CNs theoretically contain richer structural information, yet redundancy and over-smoothing limit or even harm their practical utility.

**Goal**: Design a method that extracts **incremental information independent of lower-order CNs** from higher-order CNs, while controlling over-smoothing.

**Key Insight**: A signal processing perspective — treating CNs of different orders as signals, orthogonalization extracts the independent component of each order; normalization (dividing by path count) controls the divergence of higher-order signals.

**Core Idea**: Gram-Schmidt orthogonalization + path normalization = extracting clean, incremental link prediction signals from higher-order CNs.

## Method

### Overall Architecture
Two modules are added on top of existing MPNN+CN frameworks (e.g., NCN, NCNC): (1) orthogonalization — removing redundancy between higher-order and lower-order CNs; (2) normalization — dividing each CN coefficient by the path count. The output is integrated with MPNN features for link prediction.

### Key Designs

1. **Gram-Schmidt Orthogonalization**:

    - Function: For the $k$-order CN coefficient vector, subtract its projection onto all lower-order CN coefficients to obtain the orthogonal component.
    - Mechanism: Let $\mathbf{c}_k$ be the $k$-hop CN coefficient vector over all node pairs; the orthogonal component is $\mathbf{c}_k^{\perp} = \mathbf{c}_k - \sum_{j<k} \frac{\langle \mathbf{c}_k, \mathbf{c}_j^{\perp} \rangle}{\langle \mathbf{c}_j^{\perp}, \mathbf{c}_j^{\perp} \rangle} \mathbf{c}_j^{\perp}$
    - Scalable variant: Running inner products from batch normalization approximate global orthogonalization, avoiding full-graph storage.
    - OCNP variant: Replaces Gram-Schmidt with orthogonal polynomial bases (e.g., Chebyshev), which is faster but not strictly orthogonal.
    - Design Motivation: The Pearson correlation between 2-hop and 1-hop CNs frequently exceeds 0.8, indicating substantial redundancy.

2. **Path Normalization (Generalized Resource Allocation)**:

    - Function: The $k$-hop CN coefficient is divided by the path count $|P_k(c)|$ — the more paths through which a node is reachable in $k$ steps, the lower the informational value of each individual path.
    - Mechanism: For $k=1$, this reduces to the classical Resource Allocation (RA) index: $\sum_{c \in CN} 1/d(c)$.
    - Design Motivation: Higher-order CN coefficients grow exponentially with order (path counts increase exponentially with each additional hop); normalization controls this divergence.

3. **Integration with MPNN**:

    - Function: Orthogonalized and normalized CN features are concatenated with MPNN node features and fed into the link prediction head.
    - Formula: OCN is incorporated into the NCN framework via Equation 7.

### Loss & Training
Standard binary cross-entropy loss for link prediction. Negative sampling is used during training.

## Key Experimental Results

### Main Results (HR@100, %)

| Dataset | OCN | OCNP | Prev. SOTA (NCNC) | Gain |
|--------|-----|------|-----------------|------|
| Cora | 89.82 | **90.06** | 89.65 | +0.5% |
| Citeseer | **93.62** | — | 93.47 | +0.2% |
| Pubmed | **83.96** | — | 81.29 | +3.3% |
| Collab | **72.43** | — | 66.61 | +8.7% |
| PPA | **69.79** | — | 61.42 | +13.6% |
| DDI | **97.42** | — | 84.11 | **+15.8%** |
| Citation2 | 88.57 | — | 89.12 | -0.6% |
| **Average** | — | — | — | **+7.7%** |

### Ablation Study

| Configuration | Key Findings | Notes |
|------|---------|------|
| w/ vs. w/o orthogonalization | Orthogonalization yields significant gains on PPA/DDI | Redundancy removal is most critical |
| w/ vs. w/o normalization | Normalization is essential at higher orders (3-hop) | Controls over-smoothing |
| 1-hop only vs. 1+2-hop | Adding 2-hop is beneficial | Incremental information is effective |
| 1+2+3-hop | 3-hop introduces instability | Only 1+2-hop used in practice |
| OCN vs. OCNP | OCNP is slightly worse but faster | Trade-off between strict orthogonality and efficiency |

### Key Findings
- DDI and PPA datasets benefit most (+13–16%), as these are sparse graphs with rich higher-order structural information.
- 3-hop CNs introduce instability — using only 1-hop and 2-hop is optimal in practice.
- Normalization generalizes the Resource Allocation index, establishing a theoretical connection to classical graph heuristics.

## Highlights & Insights
- **Quantifying Redundancy**: Pearson correlations between CN orders are quantified (>0.8), explaining why naively stacking higher-order CNs yields limited gains.
- **Unification with Classical Heuristics**: The reduction of normalization to the RA index is an elegant theoretical connection — classical link prediction heuristics essentially perform "normalized 1-hop CN."
- **Simplicity and Effectiveness**: Both orthogonalization and normalization are linear operations that do not increase model complexity, yet yield an average improvement of 7.7%.

## Limitations & Future Work
- Gram-Schmidt orthogonalization requires global inner product computation, resulting in relatively high time complexity.
- Only 1-hop and 2-hop are used in practice — the instability of 3-hop CN remains unresolved.
- Theoretical analysis is grounded in random graph and BA models, with limited applicability to real-world graphs.
- Performance slightly decreases on Citation2 (-0.6%), indicating that orthogonalization is not universally beneficial.

## Related Work & Insights
- **vs. NCN/NCNC (Wang et al., 2023)**: NCN directly uses CN features; OCN additionally applies orthogonalization to remove redundancy.
- **vs. SEAL (Zhang & Chen, 2018)**: SEAL extracts local structure via subgraph extraction; OCN captures the same information more efficiently via CN coefficients.
- **vs. Resource Allocation / Adamic-Adar**: OCN's normalization generalizes these classical methods to arbitrary hop counts.

## Rating
- Novelty: ⭐⭐⭐⭐ Orthogonalizing higher-order CNs is a simple yet overlooked improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 datasets with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The narrative connecting theory to classical methods is clear.
- Value: ⭐⭐⭐⭐ Directly applicable value for the link prediction community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](../../AAAI2026/graph_learning/unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[NeurIPS 2025\] Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention](solar-geco_perovskite_solar_cell_property_prediction_with_geometric-aware_co-att.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)

</div>

<!-- RELATED:END -->
