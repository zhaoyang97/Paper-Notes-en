---
title: >-
  [Paper Note] Federated Learning with Domain Shift Eraser
description: >-
  [CVPR 2025][Optimization][Federated Learning] This paper proposes the FDSE method, which decomposes each network layer into a domain-free feature extractor (DFE, globally aggregated to enhance consensus) and a domain-specific shift eraser (DSE, personalized aggregated to retain local characteristics). Combined with BN consistency regularization, it achieves 76.77% on DomainNet (outperforming Ditto by 1.6%) and 91.58% on Office-Caltech10 (outperforming FedBN by 4.6%).
tags:
  - "CVPR 2025"
  - "Optimization"
  - "Federated Learning"
  - "Domain Shift"
  - "Layer Decomposition"
  - "Consistency Regularization"
  - "Dual Aggregation"
date: 2026-05-08
content_hash: 78c06c6ca3096b2b
---

# Federated Learning with Domain Shift Eraser

**Conference**: CVPR 2025  
**arXiv**: [2503.13063](https://arxiv.org/abs/2503.13063)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Federated Learning, Domain Shift, Layer Decomposition, Consistency Regularization, Dual Aggregation

## TL;DR
This paper proposes the FDSE method, which decomposes each network layer into a domain-free feature extractor (DFE, globally aggregated to enhance consensus) and a domain-specific shift eraser (DSE, personalized aggregated to retain local characteristics). Combined with BN consistency regularization, it achieves 76.77% on DomainNet (outperforming Ditto by 1.6%) and 91.58% on Office-Caltech10 (outperforming FedBN by 4.6%).

## Background & Motivation

**Background**: Federated learning faces feature space misalignment caused by domain shift—differences in data distributions across clients make the global model unsuitable for certain clients. Existing methods either enhance global consensus (e.g., FedAvg series) or enhance personalization (e.g., FedBN), but rarely balance both simultaneously.

**Limitations of Prior Work**: FedAvg-like methods promote consensus through weight averaging but ignore personalization needs; FedBN-like methods retain local BN, but globally shared layers are still interfered with by domain shift. The advantages of both strategies have not been unified at a fine-grained level.

**Key Challenge**: Federated learning requires a balance between global consensus (learning shared knowledge across clients) and local personalization (adapting to local data distributions), but existing methods can only make a single choice at the layer level.

**Goal**: To simultaneously decouple and optimize domain-free features (promoting consensus) and domain-specific features (retaining personalization) within each layer.

**Key Insight**: To decompose each convolutional layer into two sub-modules: DFE, which extracts domain-invariant features with channels $\lceil T/G \rceil$ using global aggregation, and DSE, which captures domain-specific shifts using cheap 1×1 convolutions expanded to $T$ channels using personalized aggregation.

**Core Idea**: To decouple domain-free and domain-specific features into two sub-modules within each layer, updating them using global consensus aggregation and similarity-aware personalized aggregation strategies, respectively.

## Method

### Overall Architecture
Each network layer is divided into DFE (a lightweight backbone to extract shared features) and DSE (1×1 convolution to erase domain shift). During training, both are jointly optimized but aggregated separately: DFE uses FedAvg-style global aggregation, and DSE uses self-attention personalized aggregation based on inter-client similarity. BN consistency regularization pulls local BN statistics toward global statistics.

### Key Designs

1. **Layer Decomposition (DFE + DSE)**:

    - **Function**: Separating domain-free and domain-specific features within the same layer.
    - **Mechanism**: DFE outputs $\lceil T/G \rceil$ channels ($G$ is the group number) to extract domain-invariant base features; DSE uses 1×1 convolutions to expand the channels to $T$, capturing and "erasing" domain-specific shift patterns. The two are cascaded: input $\rightarrow$ DFE $\rightarrow$ DSE $\rightarrow$ output. The proportion of DFE parameters is controlled by $G$; a larger $G$ makes DSE more lightweight.
    - **Design Motivation**: 1×1 convolutions contain very few parameters ("cheap operation"). DSE can encode domain-specific information with minimal parameters, leaving most parameters for the globally shareable DFE.

2. **Dual Aggregation Strategy**:

    - **Function**: Maximizing global consensus for DFE and retaining local personalization for DSE.
    - **Mechanism**: DFE uses FedAvg aggregation (fair consensus via $L_2$ norm minimization); DSE uses similarity-aware self-attention aggregation, which calculates the cosine similarity of DSE parameters between clients, allowing similar clients' DSEs to learn more from each other. A temperature parameter $\tau$ controls the level of personalization.
    - **Design Motivation**: Domain-free features should be fully shared (making FedAvg optimal); domain-specific features should not be simply averaged (which loses personalization), but rather allow clients in similar domains to learn from each other.

3. **BN Consistency Regularization**:

    - **Function**: Reducing deviation between local and global BN statistics.
    - **Mechanism**: The regularization loss is defined as $\lambda \sum_l (\|\mu_l^{local} - \mu_l^{global}\|^2 + \|\sigma_l^{local} - \sigma_l^{global}\|^2)$, with weight exponentially decaying across layers ($\beta=0.001$). This pulls local BN statistics closer to global ones without full alignment (leaving space for personalization).
    - **Design Motivation**: BN statistics directly reflect the data distribution; slight alignment reduces domain shift, but over-alignment hurts personalization.

### Loss & Training
Total loss = Task cross-entropy + $\lambda \cdot$ BN consistency regularization. 500 communication rounds are used, with the learning rate decaying by 0.998 per round. DomainNet/PACS uses 5 local epochs, and Office-Caltech10 uses 1.

## Key Experimental Results

### Main Results

| Dataset | FDSE (All/Avg) | Ditto | FedBN | FedAvg |
|--------|---------------|-------|-------|--------|
| DomainNet | 76.77/74.50 | 75.18/72.82 | 74.75/- | 69.17/- |
| Office-Caltech10 | 87.15/91.58 | -/- | 83.08/87.01 | -/- |
| PACS | 83.81/82.17 | 82.02/80.03 | -/- | -/- |

FDSE consistently achieves the best performance across all datasets, outperforming 20+ methods on DomainNet.

### Key Findings
- T-SNE visualization confirms that FDSE achieves better category separation and domain alignment in the feature space.
- Improvements are observed across almost all clients (shown in the spider plot), indicating that performance gains do not come at the expense of certain clients.
- Although the convergence speed is not the fastest, the final accuracy is the highest.

## Highlights & Insights
- **Intra-layer Decoupling instead of Inter-layer Allocation**: Prior methods decide whether to share or personalize at the layer level, whereas FDSE performs both functions within each layer, achieving a finer granularity.
- **1×1 Convolution as a Domain Shift Eraser**: This extremely lightweight parameter overhead captures domain-specific information, leaving the majority of parameters for the shareable DFE.

## Limitations & Future Work
- The grouping parameter $G$ needs to be set manually.
- The BN consistency weight $\lambda$ is sensitive to performance.
- It has only been validated on computer vision (CV) classification tasks.

## Related Work & Insights
- **vs FedBN**: FedBN only localizes BN layers, whereas FDSE performs decoupling in every layer, which is more thorough.
- **vs Ditto**: Ditto regularizes local models using the global model but does not perform intra-layer decomposition, whereas FDSE operates at a finer granularity.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of intra-layer decoupling is novel, and the DFE+DSE design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 20+ baselines, 3 datasets, and thorough visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation and methods are clearly articulated.
- **Value**: ⭐⭐⭐⭐ Provides a substantial push for federated domain generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](../../CVPR2026/optimization/fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)
- [\[ICLR 2026\] Bayesian Evidence-Driven Prototype Evolution for Federated Domain Adaptation](../../ICLR2026/optimization/bayesian_evidence-driven_prototype_evolution_for_federated_domain_adaptation.md)
- [\[CVPR 2025\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[CVPR 2025\] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)
- [\[CVPR 2026\] HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation](../../CVPR2026/optimization/hfedatm_hierarchical_federated_domain_generalization_via_optimal_transport_and_r.md)

</div>

<!-- RELATED:END -->
