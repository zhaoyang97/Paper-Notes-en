---
title: >-
  [Paper Note] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments
description: >-
  [ICLR 2026][Optimization][Clustered Federated Learning] FedDAG is a clustered federated learning framework that performs more accurate client clustering via weighted class-wise similarity fusion of data and gradient signals, and enables cross-cluster feature transfer through a dual-encoder architecture, consistently outperforming existing baselines across diverse heterogeneity settings.
tags:
  - ICLR 2026
  - Optimization
  - Clustered Federated Learning
  - Data Heterogeneity
  - Dual-Encoder Architecture
  - Cross-Cluster Knowledge Sharing
  - Adaptive Clustering
date: 2026-05-08
content_hash: caf3c3ed3cdde990
---

# FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments

**Conference**: ICLR 2026
**arXiv**: [2602.23504](https://arxiv.org/abs/2602.23504)
**Code**: [https://tinyurl.com/2rbkb3zu](https://tinyurl.com/2rbkb3zu)
**Area**: Optimization / Federated Learning
**Keywords**: Clustered Federated Learning, Data Heterogeneity, Dual-Encoder Architecture, Cross-Cluster Knowledge Sharing, Adaptive Clustering

## TL;DR
FedDAG is a clustered federated learning framework that performs more accurate client clustering via weighted class-wise similarity fusion of data and gradient signals, and enables cross-cluster feature transfer through a dual-encoder architecture, consistently outperforming existing baselines across diverse heterogeneity settings.

## Background & Motivation

**Background**: Federated learning (FL) enables collaborative model training without sharing raw data, but client data heterogeneity (non-IID) leads to slow convergence and degraded accuracy. Clustered FL addresses this by grouping similar clients, with each cluster training its own model.

**Limitations of Prior Work**: Existing clustered FL methods suffer from four key limitations: 1) similarity computation relies on either data or gradient signals alone, lacking comprehensiveness; 2) knowledge sharing is restricted within clusters, failing to leverage diverse representations across clusters; 3) only label skew is typically addressed, while concept drift and quantity imbalance are neglected; 4) the number of clusters must be specified in advance.

**Key Challenge**: Data-based and gradient-based similarity measures each have blind spots — gradient similarity in high-dimensional spaces can produce false positives, while data similarity is insensitive to concept drift. Neither signal alone can accurately characterize true inter-client similarity.

**Goal**: How to dynamically cluster clients by jointly leveraging data and gradient information, while enabling cross-cluster representation sharing?

**Key Insight**: Refine similarity computation to the class level, learn adaptive weights for data and gradient signals, and employ a dual-encoder architecture to facilitate cross-cluster feature transfer.

**Core Idea**: Perform more accurate client clustering via class-wise weighted fusion of data and gradient similarity, and enable cross-cluster knowledge sharing while preserving cluster specialization through a dual-encoder architecture.

## Method

### Overall Architecture
FedDAG consists of two core components: (1) **Similarity Computation and Adaptive Clustering** — fuses data and gradient information to compute a weighted inter-client similarity matrix, generates candidate groupings via hierarchical clustering, and automatically determines the optimal number of clusters using a federated-aware metric; (2) **Dual-Encoder Training** — each cluster model contains a primary encoder (updated with intra-cluster data) and a secondary encoder (updated with gradients from complementary clusters), whose outputs are concatenated before being passed to the classifier.

### Key Designs

1. **Weighted Class-wise Data-Gradient Fusion Similarity**:

    - **Function**: Computes a more accurate inter-client similarity matrix by jointly leveraging data and gradient information.
    - **Mechanism**: Extends the data similarity approach of PACFL to the class level — comparing per-class data subspaces rather than global subspaces. Each client learns a weight $w_i$ controlling the contribution of data vs. gradient signals, yielding a final similarity of $S_{ij} = w_i \cdot S_{ij}^{data} + (1 - w_i) \cdot S_{ij}^{grad}$. Weights are optimized by minimizing an entropy-based loss to sharpen the adjacency matrix.
    - **Design Motivation**: Class-level comparison naturally handles concept drift (same label with different semantics), while weighted fusion allows each client to adaptively emphasize the most informative signal source.

2. **Dual-Encoder Cross-Cluster Knowledge Sharing**:

    - **Function**: Enables each cluster model to simultaneously learn intra-cluster specialized features and inter-cluster complementary features.
    - **Mechanism**: Each cluster model comprises a primary encoder $\phi^{(1)}$ and a secondary encoder $\phi^{(2)}$. The primary encoder is updated with aggregated gradients $\Theta_z^{1f}$ from clients within the cluster; the secondary encoder is updated with gradients $\Theta_z^{2f}$ from complementary clusters. Their outputs are concatenated along the feature dimension and fed into the classifier: $F_z(\cdot) = \psi(\phi^{(1)}(\cdot; \Theta_z^{1f}), \phi^{(2)}(\cdot; \Theta_z^{2f}); \Theta_z^c)$.
    - **Design Motivation**: Existing methods either restrict knowledge sharing within clusters or employ soft clustering that introduces noisy mixing. The dual-encoder design preserves cluster specialization via the primary encoder while introducing a complementary perspective via the secondary encoder, without mutual contamination.

3. **Federated-Aware Adaptive Clustering**:

    - **Function**: Automatically determines the optimal number of clusters without requiring it to be specified in advance.
    - **Mechanism**: Hierarchical clustering generates candidate groupings at multiple granularities. A proposed federated-aware metric evaluates each candidate by rewarding compact clusters and penalizing over-partitioning (degenerate clusters with too few clients). The grouping with the highest metric score is selected as the final partition.
    - **Design Motivation**: Prespecifying the number of clusters is impractical in real-world deployments, and hierarchical clustering in FL settings tends to over-split.

### Loss & Training
Standard cross-entropy loss is aggregated within each cluster. Weight optimization employs entropy-based regularization to promote binarization of the similarity matrix. Gradients are transmitted in compressed form to reduce communication overhead; each client computes gradients for at most one model per round.

## Key Experimental Results

### Main Results

| Algorithm | Technique | CIFAR-10 | FMNIST |
|-----------|-----------|----------|--------|
| PACFL | Data (D) | 90.45±0.30 | 94.41±0.31 |
| CFL | Gradient (G) | 72.80±0.66 | 86.97±0.23 |
| IFCA | Gradient (G) | 89.68±0.17 | 94.03±0.09 |
| **FedDAG** | **D+G+Global Feature Sharing** | **94.53±0.12** | **96.82±0.18** |

### Ablation Study

| Configuration | CIFAR-10 | Note |
|---------------|----------|------|
| FedDAG (Full) | 94.53 | Complete framework |
| Data similarity only | ~91.0 | Degenerates to PACFL++ |
| Gradient similarity only | ~88.5 | Degenerates to improved CFL |
| Without dual-encoder | ~92.0 | No cross-cluster features |
| Without adaptive cluster count | ~93.0 | Uses predefined cluster count |

### Key Findings
- FedDAG outperforms the strongest baseline PACFL by over 4 percentage points on CIFAR-10.
- Fusing data and gradient signals consistently surpasses either signal alone, especially under concept drift.
- The dual-encoder architecture yields a 2–3% improvement over single-encoder variants, confirming the value of cross-cluster knowledge sharing.
- The framework is effective across all four heterogeneity types: label skew, feature skew, concept drift, and quantity imbalance.

## Highlights & Insights
- **Class-wise Similarity Computation**: Refining similarity to the class level is a natural approach to handling concept drift and is more robust than global subspace comparison.
- **Responsibility Separation in the Dual-Encoder Design**: Assigning distinct signal sources to the primary and secondary encoders avoids the noisy mixing problem inherent in soft-clustering methods.

## Limitations & Future Work
- The dual-encoder architecture increases model parameter count and computational overhead.
- Class-level comparison incurs growing computational cost as the number of classes increases.
- Clients must upload a small amount of information for similarity computation; though compressed, this still poses privacy risks.
- The approach has not been evaluated in real-world federated scenarios such as cross-device FL.

## Related Work & Insights
- **vs. PACFL**: PACFL compares global subspaces via principal angles; FedDAG refines this to class-level comparison with weighted fusion, yielding a more comprehensive similarity measure.
- **vs. FedSoft/FedRC**: These methods employ soft clustering to allow clients to mix multiple cluster models, potentially introducing noise. FedDAG's dual-encoder structurally separates the two signal sources, avoiding this issue.

## Rating
- **Novelty**: ⭐⭐⭐ Class-wise fusion and the dual-encoder design represent reasonable but incremental contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation across four heterogeneity types is comprehensive.
- **Writing Quality**: ⭐⭐⭐ Content is thorough but the structure is somewhat complex.
- **Value**: ⭐⭐⭐ Offers practical improvements for clustered FL, though the scope is relatively specific.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](../../CVPR2026/optimization/enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)

</div>

<!-- RELATED:END -->
