---
title: >-
  [Paper Note] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation
description: >-
  [NeurIPS 2025][Social Computing][Graph Incremental Learning] GraphKeeper is proposed to address catastrophic forgetting in **Graph Domain-Incremental Learning (Graph Domain-IL)** through three components: domain-specific LoRA parameter isolation, intra/inter-domain disentanglement, and ridge regression-based deviation-free knowledge preservation. It outperforms the second-best method by 6.5%–16.6% and can be seamlessly integrated with graph foundation models.
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "Graph Incremental Learning"
  - "Domain-Incremental Learning"
  - "LoRA"
  - "Knowledge Disentanglement"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: 22cb79bf8913db97
---

# GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation

**Conference**: NeurIPS 2025
**arXiv**: [2511.00097](https://arxiv.org/abs/2511.00097)  
**Code**: [GitHub](https://github.com/RingBDStack/GraphKeeper)  
**Area**: Social Computing
**Keywords**: Graph Incremental Learning, Domain-Incremental Learning, LoRA, Knowledge Disentanglement, Catastrophic Forgetting

## TL;DR

GraphKeeper is proposed to address catastrophic forgetting in **Graph Domain-Incremental Learning (Graph Domain-IL)** through three components: domain-specific LoRA parameter isolation, intra/inter-domain disentanglement, and ridge regression-based deviation-free knowledge preservation. It outperforms the second-best method by 6.5%–16.6% and can be seamlessly integrated with graph foundation models.

## Background & Motivation

Graph Incremental Learning (GIL) requires models to continually update as new graph data arrives. Existing methods focus on **Task-IL** and **Class-IL** settings, both operating within a single graph domain. However, with the rise of Graph Foundation Models (GFMs), models must integrate graph data from **multiple heterogeneous domains**, i.e., the **Domain-IL** setting.

**Unique challenges of Domain-IL**:

**Embedding Shifts**: Learning new domains requires large parameter changes, causing embeddings of graphs from previous domains to shift.

**Decision Boundary Deviations**: In end-to-end training, the classifier is updated jointly with the encoder, corrupting the decision boundaries of old domains.

Empirical validation shows that SSM, a representative GIL method, performs well under Class-IL but fails severely under Domain-IL. The structural and semantic divergence across domains far exceeds intra-domain class differences, rendering existing methods inadequate.

## Method

### Overall Architecture

GraphKeeper consists of three modules targeting the two sources of catastrophic forgetting:

1. **Multi-domain Graph Disentanglement**: Prevents embedding shifts and cross-domain confusion.
2. **Deviation-Free Knowledge Preservation**: Maintains stable decision boundaries.
3. **Domain-aware Distribution Discrimination**: Matches test graphs to unknown domains at inference time.

### Key Designs

**Module 1: Multi-domain Graph Disentanglement**

*Multi-domain Feature Alignment*: Since graph features across domains have different dimensionalities, truncated SVD is applied to project all features into a unified $\bar{d}$-dimensional space:
$$\tilde{F}_i = \text{Proj}(F_i), \quad \tilde{F}_i \in \mathbb{R}^{|G_i| \times \bar{d}}$$

*Domain-Specific LoRA*: Independent LoRA modules are attached to the pretrained GNN for each domain:
$$h^l = \xi^l(h^{l-1}, W_i^l) + \phi_i^l(h^{l-1}, W_{i,\text{down}}^l W_{i,\text{up}}^l)$$
LoRA parameters of previous domains are frozen when learning a new domain, structurally preventing embedding shifts in old domains.

*Intra-domain Disentanglement*: Contrastive learning is employed to enhance discriminability among different classes within the same domain:
$$\mathcal{L}_{\text{intra}} = -\sum_{j=1}^{|G_i|} \log \frac{\sum_{o \in S_j^{\text{pos}}} \exp(\text{sim}(x_j, x_o^{\text{aug}}))}{\sum_{o' \in S_j^{\text{pos}} \cup S_j^{\text{neg}}} \exp(\text{sim}(x_j, x_{o'}^{\text{aug}}))}$$
where $S^{\text{pos}}$ denotes same-class nodes, $S^{\text{neg}}$ denotes different-class nodes, and $x^{\text{aug}}$ is drawn from augmented views.

*Inter-domain Disentanglement*: Current-domain node embeddings are pushed away from prototype embeddings of previous domains (obtained via clustering):
$$\mathcal{L}_{\text{inter}} = \frac{1}{|G_i|} \sum_{j=1}^{|G_i|} \sum_{k=1}^{|P|} \frac{1}{\|x_j - P_k\|_2^2 + \epsilon}$$
Minimizing this objective encourages sufficient separation between domains in the embedding space.

**Module 2: Deviation-Free Knowledge Preservation**

Core Idea: **Decouple the classifier from the embedding model** by replacing gradient-based updates with a closed-form ridge regression solution, thereby avoiding decision boundary deviations caused by backpropagation.

The optimal classifier weights after the $i$-th incremental domain are:
$$W_i = (X_{(1:i)}^T X_{(1:i)} + \lambda I)^{-1} X_{(1:i)}^T Y_{(1:i)}$$

Since historical data is inaccessible, a **recursive update** is applied:
$$W_i = [W_{i-1} - M_i X_i^T X_i W_{i-1} \| M_i X_i^T Y_i]$$
$$M_i = M_{i-1} - M_{i-1} X_i^T (I + X_i M_{i-1} X_i^T)^{-1} X_i M_{i-1}$$

This guarantees an **exact** update equivalent to the full-data closed-form solution, requiring no storage of any historical graph data.

**Module 3: Domain-aware Distribution Discrimination**

When the domain of a test graph is unknown, it must be matched to the correct domain-specific LoRA module. The procedure is:
1. Map features to a high-dimensional space using a randomly initialized and frozen GNN (to separate prototypes of similar domains).
2. Determine the domain of the test graph via nearest-prototype matching:
$$c_{\text{test}} = \arg\max_k \exp(-\|D_{\text{test}} - D_k\|_2^2)$$

### Loss & Training

The overall optimization objective is:
$$\mathcal{L} = \gamma_1 \mathcal{L}_{\text{intra}} + \gamma_2 \mathcal{L}_{\text{inter}}$$

Note: The decision module (ridge regression) is not updated through backpropagation of $\mathcal{L}$; instead, it is computed directly via the closed-form solution after embedding learning is complete. This decoupled design is key to preventing decision boundary deviations.

## Key Experimental Results

### Main Results

**Average results over 6 domain sequences under Domain-IL**:

| Method | Group 1 AA↑ | Group 3 AA↑ | Group 5 AA↑ |
|--------|------------|------------|------------|
| Fine-Tune | 23.9 | 20.9 | 19.7 |
| Joint (upper bound) | 66.6 | 78.0 | 74.5 |
| EWC | 23.3 | 20.8 | 20.6 |
| ER-GNN | 23.3 | 28.7 | 24.8 |
| DeLoMe | 49.3 | 70.2 | 63.2 |
| PDGNNs | 52.4 | 65.5 | 64.3 |
| TPP | 52.6 | 57.1 | 56.7 |
| **GraphKeeper** | **69.2** | **80.6** | **75.5** |

GraphKeeper surpasses the second-best method by 6.5%–16.6% and **exceeds the Joint baseline** (which has access to all historical data).

**Integration with Graph Foundation Models** (few-shot Domain-IL):

| Method | Group 1 AA↑ | Group 3 AA↑ |
|--------|------------|------------|
| GCOPE (original) | 20.6 | 13.2 |
| GCOPE + GraphKeeper | **significant gain** | **significant gain** |
| MDGPT (original) | low AA / high AF | low AA / high AF |
| MDGPT + GraphKeeper | **high AA / low AF** | **high AA / low AF** |

### Ablation Study

Ablation of individual modules (inferred from paper analysis):
- Removing inter-domain disentanglement → increased domain embedding confusion, significant performance drop.
- Removing domain-specific LoRA → uncontrolled embedding shifts, severe degradation.
- Replacing ridge regression with gradient-based classifier → decision boundary deviation, aggravated forgetting.
- Removing high-dimensional random projection → domain prototype confusion, increased domain misclassification at test time.

### Key Findings

1. **Existing GIL methods fail comprehensively under Domain-IL**: EWC, GEM, LWF, and similar methods show negligible improvement over Fine-Tune.
2. **GraphKeeper surpasses the Joint upper bound**: This indicates that a single GNN struggles to effectively fuse knowledge from multiple domains, making parameter isolation necessary.
3. **Seamless integration with GFMs**: GraphKeeper endows pretrained graph foundation models with continual learning capability while preserving their few-shot advantages.
4. **Near-zero forgetting**: The AF metric approaches 0, far outperforming all baselines including those with memory replay.
5. The relatively stronger performance of DeLoMe/PDGNNs relies on SGC/APPNP backbones (trading plasticity for stability); replacing them with GCN leads to significant performance degradation.

## Highlights & Insights

- **Novel problem formulation**: The first systematic study of Graph Domain-Incremental Learning (Domain-IL), distinguishing it from conventional Task-IL/Class-IL settings.
- **Thorough disentanglement analysis**: Forgetting is decomposed into two orthogonal dimensions—embedding shifts and decision boundary deviations—with targeted solutions for each.
- **Elegant application of closed-form ridge regression**: The recursive update formula guarantees exact equivalence to the full-data solution without storing any historical data, achieving $O(1)$ space complexity.
- **High-dimensional random projection for domain discrimination**: Simple yet effective, leveraging random projections to separate domain prototypes.
- **Promising integration with GFMs**: Provides a viable path toward continual updating of graph foundation models.

## Limitations & Future Work

1. Each new domain requires an additional LoRA module, incurring linearly growing storage and inference overhead as the number of domains increases.
2. Domain prototypes are obtained via clustering; clustering quality directly affects inter-domain disentanglement performance.
3. Domain-aware discrimination relies on prototype distances and may fail when domain distributions overlap substantially.
4. Feature alignment via truncated SVD inevitably incurs information loss.
5. Validation is limited to node classification; graph-level tasks (graph classification, link prediction) are not addressed.

## Related Work & Insights

- **Graph Incremental Learning**: Representative methods such as SSM and ER-GNN are shown to fail under Domain-IL in this work.
- **Multi-domain Graph Pretraining**: GFMs such as GCOPE and MDGPT are augmented with incremental learning capability through GraphKeeper.
- **LoRA for Graph Learning**: The domain-specific LoRA design draws inspiration from continual learning approaches in NLP.
- **Insights**: The recursive ridge regression knowledge preservation mechanism may be applicable to other incremental learning scenarios that admit closed-form updates, such as incremental updates in recommender systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to define and systematically address Graph Domain-Incremental Learning.
- **Technical Depth**: ⭐⭐⭐⭐ — Each of the three modules is theoretically grounded; the recursive update derivation is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 15 datasets, multiple domain sequences, extensive baseline comparisons, and GFM integration experiments.
- **Practicality**: ⭐⭐⭐⭐ — Directly integrable with existing GFMs; code is publicly available.
- **Overall**: ⭐⭐⭐⭐

## Background & Motivation

## Core Problem

## Method

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Inspiration & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[CVPR 2025\] Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [\[AAAI 2026\] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](../../AAAI2026/social_computing/from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](../../ACL2026/social_computing/bayesian_social_deduction_with_graph-informed_language_models.md)

</div>

<!-- RELATED:END -->
