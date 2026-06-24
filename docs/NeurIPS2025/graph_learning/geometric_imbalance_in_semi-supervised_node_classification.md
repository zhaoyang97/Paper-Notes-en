---
title: >-
  [Paper Note] Geometric Imbalance in Semi-Supervised Node Classification
description: >-
  [NeurIPS 2025][Graph Learning][geometric imbalance] This work formally introduces the concept of "geometric imbalance" in semi-supervised node classification for the first time—showing that message passing on class-imbalanced graphs causes minority-class nodes to exhibit geometric ambiguity in Riemannian manifold embedding spaces—and proposes the UNREAL framework to systematically address this issue via three modules: dual-path pseudo-label alignment, node reordering…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "geometric imbalance"
  - "semi-supervised node classification"
  - "self-training"
  - "pseudo-label"
  - "GNN"
  - "Riemannian manifold"
date: 2026-05-08
content_hash: 25affcf6d31c79e4
---

# Geometric Imbalance in Semi-Supervised Node Classification

**Conference**: NeurIPS 2025
**arXiv**: [2303.10371](https://arxiv.org/abs/2303.10371)  
**Authors**: Liang Yan, Shengzhong Zhang, Bisheng Li, Menglin Yang, Chen Yang, Min Zhou, Weiyang Ding, Yutong Xie, Zengfeng Huang (Fudan University, HKUST(GZ), MBZUAI, Logs AI)
**Code**: [yanliang3612/UNREAL](https://github.com/yanliang3612/UNREAL)  
**Area**: Graph Learning
**Keywords**: geometric imbalance, semi-supervised node classification, self-training, pseudo-label, GNN, Riemannian manifold

## TL;DR

This work formally introduces the concept of "geometric imbalance" in semi-supervised node classification for the first time—showing that message passing on class-imbalanced graphs causes minority-class nodes to exhibit geometric ambiguity in Riemannian manifold embedding spaces—and proposes the UNREAL framework to systematically address this issue via three modules: dual-path pseudo-label alignment, node reordering, and geometric imbalance sample discarding.

## Background & Motivation

Class imbalance in graph data severely impairs GNN performance in node classification, particularly in semi-supervised settings where only a small number of nodes are labeled, making information propagation even more challenging.

**Limitations of Prior Work**:
- **Oversampling methods (GraphSMOTE, GraphENS)**: Require synthesizing new nodes/edges, incurring high computational cost and difficulty ensuring structural consistency.
- **Loss function adjustment (ReNode, TAM)**: Rely on labeled nodes to infer topological bias and lack a universal quantification framework.
- **Self-training methods (GraphSR, BIM)**: Depend on heuristic selection strategies, lack theoretical guarantees, and are unstable under severe imbalance.
- **Topological vs. geometric imbalance**: Prior work addresses topological imbalance defined over the raw graph structure, whereas this paper identifies geometric imbalance defined in the Riemannian manifold embedding space—a previously overlooked problem.

**Core Insight**: Message passing amplifies embedding ambiguity for minority-class nodes on class-imbalanced graphs—these nodes become nearly equidistant from multiple class centers on the hypersphere, rendering pseudo-label assignment highly unreliable. MLPs are unaffected (no structural aggregation), confirming that geometric imbalance is a GNN-specific phenomenon.

## Method

### Theoretical Foundations of Geometric Imbalance

**Formal Definition**: Analysis is conducted on the unit hypersphere using the von Mises-Fisher (vMF) distribution. GNN embeddings are $\ell_2$-normalized and mapped to the Riemannian manifold $\mathbb{S}^{d-1}$, with each class parameterized by a vMF distribution (concentration $\kappa_i$ and mean direction $\tilde{\mu}_i$).

**Geometric Imbalance Score** (Definition 1): For an unlabeled node $u^j$, its geometric imbalance score is defined as:
$$G(u^j) = \frac{\|\tilde{h}_{u^j}^{(l)} - \tilde{\mu}_{y^{u^j}}\|^2}{\sum_{\mathcal{C}_1 \neq \mathcal{C}_2}\|\tilde{\mu}_{\mathcal{C}_1} - \tilde{\mu}_{\mathcal{C}_2}\|^2}$$
This represents the ratio of the distance from the node to its true class center relative to the total inter-class separation.

**Theorem 1**: The average information entropy $\hat{H} \propto D_{\text{intra}} / D_{\text{inter}}$, i.e., prediction uncertainty increases with intra-class dispersion and decreases with inter-class separation.

**Theorem 2**: The geometric imbalance score $\bar{G}_{\text{minor}}$ increases monotonically with the class imbalance ratio $\rho$, formally characterizing the direct relationship between data imbalance and geometric ambiguity in the embedding space.

### UNREAL Framework

The framework consists of three complementary modules that can be used independently or in combination.

#### Module 1: DualPath PseudoLabeler (DPAM)
- **Dual paths**: (1) Unsupervised clustering—applies over-clustering K-Means ($K > C$) on unlabeled node embeddings, assigning pseudo-labels based on distances between cluster centers and class centers; (2) Supervised classification—GNN directly predicts pseudo-labels.
- **Alignment mechanism**: Only nodes for which both paths agree are retained: $\mathcal{U}_i^{\text{final}} = \tilde{\mathcal{U}}_i \cap \mathcal{U}_i$
- **Role**: The clustering path mitigates majority-class bias in the classifier; the classification path eliminates geometric ambiguity noise from clustering.

#### Module 2: Node-Reordering (NR)
- Defines two rankings: **Confidence Ranking (CR)** based on classifier softmax probabilities, and **Geometric Ranking (GR)** based on distances from node embeddings to class centers.
- Measures ranking consistency $r_m$ via Rank-Biased Overlap (RBO) and adaptively combines the two: $\mathcal{N}_m^{\text{New}} = \max\{r_m, 1-r_m\} \cdot \mathcal{S}_m + \min\{r_m, 1-r_m\} \cdot \mathcal{T}_m$
- **Dynamic behavior**: Early in training, the two rankings diverge and geometric ranking dominates (more reliable); as training progresses and consistency improves, classifier confidence receives greater weight.

#### Module 3: Discarding Geometric Imbalanced Samples (DGIS)
- Defines a lightweight geometric imbalance index: $\text{GI}_u = (\beta_u - \delta_u) / \delta_u$, where $\delta_u$ is the distance to the nearest class center and $\beta_u$ is the distance to the second nearest.
- Low GI values indicate high geometric ambiguity between multiple class centers; such nodes are filtered out via a threshold.
- Executed after DPAM and NR, when the candidate pool has already been refined, ensuring computational efficiency.

## Key Experimental Results

### Experimental Setup
- **Datasets**: 8 benchmarks—Cora, CiteSeer, PubMed, Amazon-Computers (artificially imbalanced with $\rho$=10, 20, 50, 100); Computers-Random ($\rho \approx 17.7$), CS-Random ($\rho \approx 41.0$), Flickr ($\rho \approx 10.8$), Ogbn-arxiv ($\rho \approx 775.4$) (naturally imbalanced).
- **Backbone networks**: GCN, GAT, GraphSAGE.
- **Baselines**: 10+ methods including GraphSMOTE, GraphENS, ReNode, TAM, GraphSR, and BIM.
- **Metrics**: Balanced Accuracy (bAcc.) and Macro-F1.

### Table 2: Artificial Imbalance ρ=10, GCN Backbone

| Method | Cora bAcc. | Cora F1 | CiteSeer bAcc. | CiteSeer F1 |
|---|---|---|---|---|
| Vanilla | 62.82±1.43 | 61.67±1.59 | 38.72±1.88 | 28.74±3.21 |
| BIM | 72.19±0.42 | 72.67±0.48 | 58.54±0.61 | 56.81±0.98 |
| GE(w TAM) | 71.69±0.36 | 72.14±0.51 | 58.01±0.68 | 56.32±1.03 |
| GSR | 70.85±0.44 | 71.37±0.63 | 59.28±0.72 | 55.96±0.95 |
| **UNREAL** | **78.33±1.04** | **76.44±1.06** | **65.63±1.38** | **64.94±1.38** |
| Δ Gain | **+6.14** | **+3.77** | **+6.35** | **+8.13** |

### Table 3: Artificial Imbalance ρ=100, SAGE Backbone

| Method | Cora bAcc. | Cora F1 | CiteSeer bAcc. | CiteSeer F1 | PubMed bAcc. | PubMed F1 |
|---|---|---|---|---|---|---|
| Vanilla | 52.65±0.24 | 43.79±0.47 | 36.63±0.09 | 24.12±0.09 | 62.29±0.25 | 47.02±0.38 |
| BIM | 67.75±2.13 | 64.68±1.95 | 53.83±1.62 | 53.29±1.80 | 74.38±2.08 | 73.24±1.85 |
| **UNREAL** | **73.47±2.31** | **68.30±2.11** | **59.77±2.98** | **58.92±3.07** | **77.11±0.59** | **74.03±0.81** |
| Δ Gain | **+5.72** | **+3.62** | **+6.04** | **+5.63** | **+2.73** | **+0.79** |

Under extreme imbalance (ρ=100), UNREAL maintains substantial advantages, surpassing the best baseline by 5.72 percentage points in Cora bAcc.

### Natural Imbalance and Large-Scale Datasets
- **Computers-Random** (ρ≈17.7): bAcc. 85.32% on GCN, outperforming BIM by +1.29; bAcc. 82.52% on GAT, outperforming BIM by **+5.51**.
- **Ogbn-arxiv** (ρ≈775.4): Multiple baselines run out of memory (OOM); UNREAL remains executable and achieves F1 51.36 (SAGE), surpassing BIM by +1.80.
- **Flickr** (ρ≈10.8): GraphSMOTE, ReNode, and GraphENS all OOM; UNREAL achieves F1 30.60 on GCN, surpassing BIM by **+6.85**.

### Ablation Study (Table 1)
Using Cora + GCN (ρ=10) as example:
- Full UNREAL (CR+GR+DGIS, without NR): F1 76.44
- Without DGIS: drops to 75.34
- Without CR (GR+NR+DGIS only): drops to 73.93
- The three modules exhibit strong complementarity; the NR+DGIS combination achieves the best results in most settings.

## Highlights & Insights

- **Strong theoretical contribution**: The paper is the first to formally define geometric imbalance on a Riemannian manifold, establishing rigorous mathematical relationships between geometric imbalance, information entropy (Theorem 1), and class imbalance ratio (Theorem 2), elevating previously vague empirical observations to a quantifiable theoretical framework.
- **Elegant framework design**: The three modules (DPAM/NR/DGIS) each have clear theoretical motivation and practical function. They can be used independently or flexibly combined; DGIS serves as a lightweight approximation that avoids the high cost of directly computing geometric imbalance scores.
- **Comprehensive and scalable experiments**: Covering 8 datasets, 3 GNN backbones, and both artificial and natural imbalance settings, UNREAL remains effective under extreme imbalance (ρ=100, ρ≈775) without suffering from OOM issues that affect multiple baselines.

## Limitations & Future Work

- **Limited to node classification**: The framework has not been extended to other graph tasks such as link prediction or graph classification (the authors acknowledge this as future work in the conclusion).
- **K-Means clustering assumption**: Euclidean K-Means is applied on hyperspherical embeddings rather than spherical K-Means, which is theoretically inconsistent, though the authors discuss robustness in the appendix.
- **Hyperparameter sensitivity**: The number of clusters $k'$ and the DGIS threshold $\gamma$ require tuning; while sensitivity analysis suggests relative stability within reasonable ranges, optimal values vary across datasets.
- **Performance degradation on SAGE + Computers-Random**: A decrease of −3.17 bAcc. indicates that the framework is not universally optimal across all backbone–dataset combinations.
- **Limited gains on large-scale datasets**: Overall improvement on Ogbn-arxiv is modest (<1.5 bAcc.), possibly because the extreme imbalance ratio (ρ≈775) exceeds the effective range of the current approach.

## Related Work & Insights

- **Node generation methods**: GraphSMOTE (minority embedding interpolation + predicted new edges), ImGAGN (joint feature and topology generation), GraphENS (ego-network mixing for diversity augmentation) → high computational cost, difficulty ensuring structural consistency.
- **Topology-aware adjustment**: ReNode (topology-distance-based reweighting), TAM (logit calibration via local topology and class statistics) → rely on labeled nodes and cannot quantify bias in the embedding space.
- **Self-training methods**: GraphSR (similarity filtering + reinforcement learning for pseudo-label selection), BIM (influence maximization to balance class coverage), IceBerg (dual balancing + decoupled propagation) → heuristic selection without theoretical guarantees.
- **Positioning of UNREAL**: By approaching the problem from the geometric perspective of the embedding space, this work provides the first theoretical quantification of geometric imbalance, filling the gap in theoretical foundations for graph-based self-training.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The concept of geometric imbalance is an entirely novel theoretical contribution; rigorous Riemannian manifold analysis elevates empirical observations to a quantifiable framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 8 datasets, 3 backbones, diverse imbalance settings, complete ablation and sensitivity analyses; gains are limited in some settings.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and the structure is clear; the notation system is extensive but well-defined.
- Value: ⭐⭐⭐⭐ — Provides a new theoretical perspective and practical tools for imbalanced graph learning; the self-training framework exhibits strong extensibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Forest-Based Graph Learning for Semi-Supervised Node Classification](../../ICLR2026/graph_learning/forest-based_graph_learning_for_semi-supervised_node_classification.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [\[NeurIPS 2025\] Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention](solar-geco_perovskite_solar_cell_property_prediction_with_geometric-aware_co-att.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)

</div>

<!-- RELATED:END -->
