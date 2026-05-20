---
title: >-
  [Paper Note] Deep Modularity Networks with Diversity-Preserving Regularization
description: >-
  [NEURIPS2025][Interpretability][graph clustering] This work augments Deep Modularity Networks (DMoN) with three diversity-preserving regularization terms—distance-based, variance-based…
tags:
  - "NEURIPS2025"
  - "Interpretability"
  - "graph clustering"
  - "modularity maximization"
  - "diversity regularization"
  - "GNN pooling"
date: 2026-05-08
content_hash: 74c102e81140fdcf
---

# Deep Modularity Networks with Diversity-Preserving Regularization

**Conference**: NEURIPS2025
**arXiv**: [2501.13451](https://arxiv.org/abs/2501.13451)  
**Code**: [YasminSalehi/DMoN-DPR](https://github.com/YasminSalehi/DMoN-DPR)  
**Area**: Interpretability
**Keywords**: graph clustering, modularity maximization, diversity regularization, GNN pooling

## TL;DR

This work augments Deep Modularity Networks (DMoN) with three diversity-preserving regularization terms—distance-based, variance-based, and entropy-based—to explicitly promote inter-cluster separation and assignment diversity in feature space, achieving significant clustering quality improvements on feature-rich graph datasets.

## Background & Motivation

Graph clustering is a central problem in graph representation learning, with broad applications in social network community detection and biological network functional module identification. GNN-based graph pooling methods (e.g., DiffPool, MinCutPool) have made notable progress in recent years, yet they suffer from high computational cost or convergence difficulties. DMoN combines spectral modularity maximization with collapse regularization to achieve effective community detection within an end-to-end framework.

However, DMoN's optimization objective has two critical shortcomings:

1. **Lack of feature-space separation**: The objective contains no term that directly rewards inter-cluster feature separation, causing structurally distinct clusters to overlap significantly in feature space.
2. **Lack of assignment confidence control**: Without an explicit entropy or temperature control mechanism, soft assignments may harden prematurely, impairing cluster balance during the exploration phase.

## Core Problem

How to introduce explicit diversity-preserving mechanisms into DMoN's modularity maximization framework, so that clustering results are not only structurally meaningful but also achieve well-separated inter-cluster distributions and diverse assignments in feature space.

## Method

### Base Framework: DMoN Revisited

DMoN employs a GCN encoder to produce a soft assignment matrix $C = \text{softmax}(\text{GCN}(\tilde{A}, X))$, with an objective consisting of a modularity term and a collapse regularization term:

$$L_{\text{DMoN}} = -\frac{1}{2m}\text{Tr}(C^\top B C) + \frac{\sqrt{k}}{n}\left\|\sum_i C_i^\top\right\|_F - 1$$

where $B = A - \frac{dd^\top}{2m}$ is the modularity matrix.

### DMoN-DPR: Three Diversity-Preserving Regularization Terms

Three regularization terms are added on top of DMoN:

$$L_{\text{DMoN-DPR}} = L_{\text{DMoN}} + W_{\text{dist}} L^{\text{distance}} + W_{\text{var}} L^{\text{variance}} + W_{\text{entropy}} L^{\text{entropy}}$$

#### 1. Distance-Based Regularization

Inspired by SimCLR contrastive learning, this term penalizes cluster centroid pairs that are too close in feature space:

$$L^{\text{distance}} = \frac{1}{k(k-1)} \sum_{i=1}^{k} \sum_{j \neq i}^{k} \text{ReLU}(\epsilon - \|\mu_i - \mu_j\|_2^2)$$

where $\mu_i$ is the weighted centroid of cluster $i$ and $\epsilon$ is a minimum distance threshold. A penalty is incurred when two centroids are closer than $\epsilon$, pushing clusters apart.

#### 2. Variance-Based Regularization

This term maximizes the variance of each cluster's assignment probabilities across all nodes, preventing uniform assignments:

$$L^{\text{variance}} = -\frac{1}{k} \sum_{i=1}^{k} \text{Var}(C_{:i})$$

High variance implies that each cluster has a clear preference for certain nodes, promoting specialized assignments.

#### 3. Entropy-Based Regularization

With a small positive weight, this term minimizes the Shannon entropy of each node's assignment distribution:

$$L^{\text{entropy}} = -\frac{1}{n} \sum_{v=1}^{n} \sum_{i=1}^{k} C_{vi} \log(C_{vi} + \delta)$$

Small weights (0.001–0.1) ensure entropy decreases gradually, avoiding premature hardening of assignments and preserving uncertainty during early training to support exploration.

### Design Motivation

- The **distance term** drives cluster centroids apart in feature space.
- The **variance term** ensures sufficient discriminability in each cluster's assignment distribution.
- The **entropy term** gently guides assignments toward greater certainty without disrupting exploration.
- Together, the three terms achieve a synergistic effect of inter-cluster separation, intra-cluster focus, and progressive confidence.

## Key Experimental Results

Comparisons against DiffPool, MinCutPool, and DMoN are conducted on 5 benchmark datasets using averages over 10 random seeds.

**Feature-sparse datasets** (Cora / CiteSeer / PubMed):

- DPR variants marginally outperform DMoN on NMI and F1 (e.g., DPR(DV) NMI 44.40% vs. DMoN 43.92% on Cora), but improvements are not statistically significant ($p > 0.10$).
- Graph structural metrics (Conductance, Modularity) are virtually unaffected.

**Feature-rich datasets** (Coauthor CS / Coauthor Physics) — significant improvements:

| Method | CS NMI↑ | CS F1↑ | Physics NMI↑ | Physics F1↑ |
|--------|---------|--------|-------------|-------------|
| DMoN | 69.26% | 59.26% | 53.50% | 47.51% |
| DPR(DVE) | 71.28% | **62.67%** | 53.50% | **57.96%** |
| DPR(E) | **71.58%** | 61.33% | 52.83% | 51.09% |
| DPR(DV) | 70.72% | 61.35% | **55.84%** | 57.99% |

- F1 improves by more than 3 percentage points on Coauthor CS and more than 10 percentage points on Coauthor Physics.
- Paired t-tests confirm that NMI and F1 improvements are statistically significant at $p \leq 0.05$.

## Highlights & Insights

1. **Concise and effective regularization design**: Each of the three regularization terms has a clear intuition and can be plugged into DMoN without modifying the model architecture.
2. **Theory-practice consistency**: Significant improvements on feature-rich datasets validate the hypothesis that feature-space diversity enhances clustering quality.
3. **Structural metrics preserved**: Adding regularization has negligible impact on Conductance and Modularity, demonstrating that the method balances graph structure and feature space simultaneously.
4. **Rigorous statistical validation**: Paired two-tailed t-tests under matched random seeds provide stronger evidence than simply reporting mean values.

## Limitations & Future Work

1. **Limited gains on feature-sparse graphs**: Improvements on Cora, CiteSeer, and PubMed are not significant, indicating that the method relies on the richness of node features.
2. **Hyperparameter sensitivity**: The three weights $W_{\text{dist}}, W_{\text{var}}, W_{\text{entropy}}$ and the threshold $\epsilon$ require per-dataset tuning.
3. **Scalability unverified**: The largest dataset contains approximately 34K nodes (Coauthor Physics); performance on large-scale graphs remains unknown.
4. **Fixed number of clusters $k$**: The method requires a pre-specified cluster count and does not explore adaptive determination of $k$.
5. **Centroid computation in distance regularization** relies on soft-assignment weighting, leading to low centroid discriminability when assignments approach uniformity.

## Related Work & Insights

| Method | Characteristics | Limitations |
|--------|----------------|-------------|
| DiffPool | End-to-end soft assignment learning | Quadratic computational cost; unstable on large graphs (collapse on PubMed) |
| MinCutPool | Normalized cut + orthogonality constraint | May impede convergence |
| DMoN | Modularity maximization + collapse regularization | No feature-space separation mechanism |
| **DMoN-DPR** | DMoN + distance/variance/entropy regularization | Requires hyperparameter tuning; limited gains on feature-sparse graphs |

**Broader implications:**

- **Generality of diversity regularization**: The distance and variance regularization ideas are transferable to other settings requiring cluster or representation diversity, such as diversified recommendation and base learner diversity in ensemble learning.
- **Feature richness as a method selection criterion**: The paper reveals a practical principle—when node features are rich and heterogeneous, investing in additional regularization to exploit feature information is worthwhile; otherwise, the cost may outweigh the benefit.
- **Connection to contrastive learning**: The distance regularization shares the same pushing-apart mechanism as contrastive learning losses, suggesting that graph clustering may benefit from more deeply integrating contrastive learning objectives.

## Rating

- Novelty: 3/5 (regularization design is well-motivated, though individual components are not novel)
- Experimental Thoroughness: 4/5 (multiple datasets, statistical testing, and ablation studies are all present)
- Writing Quality: 4/5 (clear structure with well-articulated motivation)
- Value: 3/5 (effective on feature-rich graphs, but applicability is limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)
- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](../../ACL2026/interpretability/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](../../ICLR2026/interpretability/modal_logical_neural_networks_for_financial_ai.md)
- [\[ICLR 2026\] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](../../ICLR2026/interpretability/initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)
- [\[AAAI 2026\] FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](../../AAAI2026/interpretability/fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)

</div>

<!-- RELATED:END -->
