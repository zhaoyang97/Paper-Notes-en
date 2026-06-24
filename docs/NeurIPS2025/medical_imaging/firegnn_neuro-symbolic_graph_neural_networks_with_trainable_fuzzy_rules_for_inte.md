---
title: >-
  [Paper Note] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification
description: >-
  [NeurIPS 2025][Medical Imaging][Graph Neural Networks] This paper proposes FireGNN, which for the first time embeds trainable fuzzy rules into the GNN forward pass. Using three topological descriptors—node degree, clustering coefficient, and label consistency—FireGNN achieves endogenous interpretability for medical image classification, outperforming standard GCN/GAT/GIN and auxiliary-task baselines on 5 MedMNIST datasets and MorphoMNIST.
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Graph Neural Networks"
  - "Fuzzy Rules"
  - "Interpretability"
  - "Neuro-Symbolic Reasoning"
  - "MedMNIST"
  - "Topological Descriptors"
date: 2026-05-08
content_hash: 8ea15a9f3a7b534f
---

# FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification

**Conference**: NeurIPS 2025
**arXiv**: [2509.10510](https://arxiv.org/abs/2509.10510)  
**Code**: [GitHub](https://github.com/basiralab/FireGNN)  
**Area**: Medical Image Classification / Interpretability
**Keywords**: Graph Neural Networks, Fuzzy Rules, Interpretability, Neuro-Symbolic Reasoning, MedMNIST, Topological Descriptors

## TL;DR

This paper proposes FireGNN, which for the first time embeds trainable fuzzy rules into the GNN forward pass. Using three topological descriptors—node degree, clustering coefficient, and label consistency—FireGNN achieves endogenous interpretability for medical image classification, outperforming standard GCN/GAT/GIN and auxiliary-task baselines on 5 MedMNIST datasets and MorphoMNIST.

## Background & Motivation

**Interpretability requirements in medical AI**: Clinical settings demand not only high accuracy but also transparent and understandable reasoning to establish clinical trust.

**Black-box problem of GNNs**: Standard GNNs (GCN, GAT) achieve strong relational modeling on medical images, but lack explanations for their predictions—e.g., when a tumor patch is classified as malignant, it is unclear which node features or edges are most influential.

**Limitations of existing interpretability methods**:
   - Post-hoc methods (e.g., GNNExplainer) operate outside the model and may not faithfully reflect the model's internal reasoning.
   - Existing fuzzy-GNN hybrid methods rely on fixed rule templates and predefined thresholds that cannot adapt to different datasets.

**Underutilization of topological signals**: Many GNNs construct graphs using simple heuristics (e.g., spatial proximity), ignoring biologically meaningful topological structures.

## Method

### Graph Construction

Each medical image is treated as a node $v_i$ with feature $f_i = F(x_i)$ and label $y_i$. An adjacency matrix $A$ is constructed via top-$k$ cosine similarity. The standard GCN layer update is:

$$H^{(\ell+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(\ell)}W^{(\ell)}\right)$$

where $\tilde{A} = A + I$ and $\tilde{D}$ is the degree matrix.

### Topological Feature Extraction

A fact vector $f_u \in \mathbb{R}^3$ is constructed for each node $u$:

$$f_u = [d(u),\; C(u),\; L(u)]$$

- $d(u)$: node degree (number of connections)
- $C(u)$: clustering coefficient (ratio of actual edges among neighbors to the maximum possible)
- $L(u)$: 2-hop label consistency (proportion of nodes in the 2-hop neighborhood sharing the same label)

### Trainable Fuzzy Rules

Each rule $i$ has a learnable threshold $\theta_i$ and sharpness parameter $\alpha_i$. Activation strength is computed via sigmoid:

$$r_i(u) = \sigma\left(\alpha_i(f_u[i] - \theta_i)\right) \in [0, 1]$$

The three rules correspond to:
- **Rule 1**: IF degree $\geq \theta_1$ THEN high_connectivity
- **Rule 2**: IF clustering $\geq \theta_2$ THEN high_cliquishness
- **Rule 3**: IF 2-hop label agreement $\geq \theta_3$ THEN high_label_consistency

### Gated Fusion

The rule vector is projected into the GNN embedding space and fused with the GNN embedding:

$$e_u = W_r r(u) + b_r \in \mathbb{R}^d$$

$$g_u = \sigma(W_g[h_u \| e_u] + b_g) \in [0,1]^d$$

$$h_u' = g_u \odot h_u + (1 - g_u) \odot e_u$$

The gate $g_u$ adaptively balances contributions from the GNN embedding and the rule embedding, enabling interpretations such as "this node is classified as bladder tissue due to high connectivity and strong label consistency."

### End-to-End Training

GCN weights, fuzzy rule parameters $\{\theta_i, \alpha_i\}$, and fusion weights $(W_r, W_g)$ are jointly optimized with cross-entropy loss.

## Key Experimental Results

### Main Results: GNN Variant Comparison across Six Datasets

| Method | OrganCMNIST ACC | OrganAMNIST ACC | OrganSMNIST ACC | TissueMNIST ACC | BloodMNIST ACC |
|------|----------------|----------------|----------------|----------------|---------------|
| GCN | 88.20±0.61 | 91.85±0.30 | 78.62±0.82 | 50.90±0.32 | — |
| GCN + Aux | 88.41±0.44 | 93.11±0.24 | 79.19±0.74 | 52.70±0.22 | — |
| **GCN + FR** | **91.41±0.61** | **94.32±0.18** | **85.05±0.43** | **65.73±0.88** | — |
| GAT | 90.31±0.28 | 93.69±0.36 | 81.80±0.68 | 51.53±0.35 | — |
| GAT + Aux | 90.88±0.49 | 93.70±0.46 | 81.69±0.68 | OOM | — |
| **GAT + FR** | **91.66±0.48** | **94.52±0.31** | **84.82±0.52** | OOM | — |
| GIN | 87.96±0.59 | 91.54±0.71 | 77.23±0.62 | 50.51±1.09 | — |
| **GIN + FR** | **89.12±1.18** | **92.48±1.82** | — | — | — |

> +FR = Fuzzy Rules; +Aux = auxiliary tasks (homophily prediction + similarity entropy). Fuzzy rules consistently outperform both baselines and auxiliary-task methods across all backbones and datasets.

### Learned Fuzzy Rules Example (OrganCMNIST)

| Rule | Threshold $\theta$ | Semantics |
|------|--------------|------|
| Rule 1 (Degree) | 7.28 | IF degree ≥ 7.28 → high_connectivity |
| Rule 2 (Clustering) | 0.18 | IF clustering ≥ 0.18 → high_cliquishness |
| Rule 3 (Label Consistency) | 0.67 | IF 2-hop agreement ≥ 0.67 → high_label_consistency |

**Interpretation example**: A node predicted as "bladder" has degree = 10 (≥ 7.28, activation 0.60), clustering coefficient = 0.18 (= $\theta_2$, activation 0.50), and label consistency = 0.73 (≥ 0.67, activation 0.56) → the model makes this classification based on high connectivity and strong label consistency.

### Key Performance Gains

- GCN+FR vs. GCN on OrganSMNIST: **+6.43% ACC** (78.62 → 85.05)
- GCN+FR vs. GCN on TissueMNIST: **+14.83% ACC** (50.90 → 65.73) — a remarkably substantial improvement
- GAT+FR achieves the highest accuracy of **91.66%** on OrganCMNIST

## Highlights & Insights

1. ⭐⭐⭐ **First trainable fuzzy rule integration into GNNs**: Thresholds and sharpness parameters are learned from data without manual definition, realizing data-driven symbolic reasoning.
2. ⭐⭐⭐ **Endogenous interpretability**: Rule activations directly provide human-readable classification rationales (e.g., "high connectivity + high label consistency → bladder"), surpassing post-hoc explanation methods.
3. ⭐⭐ **Generality**: The fuzzy rule module is plug-and-play across multiple GNN backbones (GCN/GAT/GIN) without modifying the underlying architecture.
4. ⭐⭐ **Auxiliary tasks as a principled baseline**: The paper systematically compares two approaches to injecting topological information (fuzzy rules vs. auxiliary tasks), demonstrating the superiority of symbolic reasoning.

## Limitations & Future Work

1. **Only three rules**: Topological descriptors are limited to degree, clustering coefficient, and label consistency; richer graph features (e.g., betweenness centrality, motif counts) may provide additional information.
2. **Small dataset scale**: MedMNIST images have low resolution (28×28), which differs substantially from high-resolution medical images in real clinical settings.
3. **Scalability in inductive settings**: Graph construction relies on top-$k$ cosine similarity; computational cost and graph quality at large scale require further validation.
4. **OOM for GAT+FR on TissueMNIST**: Memory issues arising from the combination of attention mechanisms and fuzzy rules on large graphs remain unresolved.
5. **Semantic granularity of rules**: Current rules capture global topological properties and cannot explain local visual features within images.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐ |

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SIC: Similarity-Based Interpretable Image Classification with Neural Networks](../../ICCV2025/medical_imaging/sic_similarity-based_interpretable_image_classification_with_neural_networks.md)
- [\[ICLR 2026\] Neuro-Symbolic Decoding of Neural Activity](../../ICLR2026/medical_imaging/neuro-symbolic_decoding_of_neural_activity.md)
- [\[NeurIPS 2025\] Interpretable Next-token Prediction via the Generalized Induction Head](interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[NeurIPS 2025\] DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging](dison_decentralized_isolation_networks_for_out-of-distribution_detection_in_medi.md)

</div>

<!-- RELATED:END -->
