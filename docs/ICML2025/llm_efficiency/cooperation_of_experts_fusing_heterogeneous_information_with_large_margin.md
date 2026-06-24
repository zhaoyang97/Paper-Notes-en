---
title: >-
  [Paper Note] Cooperation of Experts: Fusing Heterogeneous Information with Large Margin
description: >-
  [ICML2025][LLM Efficiency][Expert Cooperation] This paper proposes the Cooperation of Experts (CoE) framework, which encodes heterogeneous information into multiplex networks. Through a two-level expert design and large-margin confidence tensor optimization, CoE achieves expert **cooperation** (rather than competition), comprehensively outperforming existing MoE and multiplex network methods in node classification tasks.
tags:
  - "ICML2025"
  - "LLM Efficiency"
  - "Expert Cooperation"
  - "Heterogeneous Multiplex Networks"
  - "Large Margin Optimization"
  - "Mutual Information Maximization"
  - "Graph Structure Learning"
date: 2026-05-08
content_hash: 1c8854185360b7e8
---

# Cooperation of Experts: Fusing Heterogeneous Information with Large Margin

**Conference**: ICML2025  
**arXiv**: [2505.20853](https://arxiv.org/abs/2505.20853)  
**Code**: [strangeAlan/CoE](https://github.com/strangeAlan/CoE)  
**Area**: Ensemble Learning / Graph Neural Networks  
**Keywords**: Expert Cooperation, Heterogeneous Multiplex Networks, Large Margin Optimization, Mutual Information Maximization, Graph Structure Learning

## TL;DR

This paper proposes the Cooperation of Experts (CoE) framework, which encodes heterogeneous information into multiplex networks. Through a two-level expert design and large-margin confidence tensor optimization, CoE achieves expert **cooperation** (rather than competition), comprehensively outperforming existing MoE and multiplex network methods in node classification tasks.

## Background & Motivation

Real-world data is typically heterogeneous, such as multimodal data (images + text) and diverse relationships in social networks (friendship, family, professional relations). The core problems faced by existing methods are:

**Limitations of a Single Predictor**: Traditional methods train a single predictor on the entire multiplex network, ignoring the inherent heterogeneity of node patterns across different relational layers. Experiments show significant variance in performance when classifiers are trained independently on each layer (e.g., on ACM and Yelp datasets).

**Defects in the Competition Mechanism of MoE**: Mixture of Experts (MoE) only activates a subset of experts through a gating mechanism, which limits the full utilization of the rich information in heterogeneous data.

**Two Key Challenges**: (a) How to design a framework to effectively extract and integrate complex information across networks? (b) How do the trained experts collaboratively contribute to the final prediction?

## Method

### Overall Architecture

The CoE framework consists of four core steps: Heterogeneous Information Encoding $\to$ Two-Level Expert Design $\to$ Expert Collaboration Strategy $\to$ Confidence Tensor Optimization.

### 1. Encoding Heterogeneous Information as Multiplex Networks

Multi-type information is encoded into a heterogeneous multiplex network $G = \{G_1, \ldots, G_V\}$, where each layer contains the same nodes but different types of connections. A Graph Structure Learning (GSL) strategy is adopted to optimize the network topology, utilizing Simple Graph Convolution (SGC) as the network learner:

$$H_v = \sigma\left((\tilde{D}_v^{-1/2}\tilde{A}_v\tilde{D}_v^{-1/2})^r X^v \odot W_1^v\right) \odot W_2^v$$

Subsequently, the adjacency matrix is reconstructed via KNN, followed by non-negativity, symmetry, and normalization post-processing.

### 2. Two-Level Expert Design

- **Low-level Experts**: Learn specific relational patterns on a single network by maximizing $I(G_i'; Y)$.
- **High-level Experts**: Capture high-order, cross-network dependencies on fused networks.

Network fusion is achieved by maximizing the cross-network mutual information $I(G_i'; G_j')$. The expert training loss is formulated as:

$$\hat{\mathcal{L}}_E = \sum_{i=1}^{V}\mathcal{L}_{cls}(Z^i; Y) - \sum_{i=1}^{V}\sum_{j=i+1}^{V}I_{lb}(Z^i; Z^j) - \sum_{i=1}^{V}I_{lb}(Z^i; Z^{tot}) - \sum_{i=1}^{V}\sum_{j \neq i}I_{lb}(Z^i; Z^{ij})$$

where the mutual information lower bound $I_{lb}$ is estimated via contrastive learning.

### 3. Large-Margin Collaboration Mechanism

Define a confidence tensor $\Theta \in \mathbb{R}^{c \times c \times k}$ (where $c$ is the number of classes and $k$ is the number of experts), where $\Theta_{rst}$ quantifies the confidence of the $t$-th expert's prediction that a sample belongs to the $r$-th class given that its true class is $s$. The final prediction is given by:

$$\hat{y}_i = \underset{j=1\ldots c}{\arg\max}\; \mathcal{S}(\Theta g_i)_j$$

The core innovation is the **large-margin loss**, which maximizes the margin between the highest and second-highest predictions:

$$\mathcal{M} = \sum_{i=1}^{N}\left[Y_i^\top(Y_i \odot \mathcal{S}(\Theta g_i)) - \frac{1}{\alpha}\log\sum_{j=1}^{c} e^{\alpha(\mathcal{S}(\Theta g_i) - Y_i \odot \mathcal{S}(\Theta g_i))_j}\right]$$

The logsumexp function is utilized to smoothly approximate the non-convex and non-smooth $\max_2$ operation. The overall loss is $\mathcal{L} = \mathcal{C} - \eta\mathcal{M} + \hat{\mathcal{L}}_E$.

### 4. Theoretical Guarantees

- **Partial Convexity**: $\mathcal{L}(\Theta g_i)$ is convex with respect to $\Theta g_i$.
- **Lipschitz Continuity**: $L \leq 2\sqrt{c}\,k(1 + \gamma + \frac{\gamma}{c}e^\alpha)$.
- **Convergence**: Gradient descent converges to a critical point when the step size satisfies $\eta \leq 1/L$.
- **Generalization Bound**: $\mathbb{E}[\ell_{0\text{-}1}(f)] \leq \frac{1}{n}\sum_i \ell_\gamma(f; x_i, y_i) + \frac{2B_\Theta G_e\sqrt{k}}{\gamma\sqrt{n}} + 3\sqrt{\frac{\log(2/\delta)}{2n}}$.

## Key Experimental Results

### Node Classification (Multiplex Networks, 5 Datasets)

| Method | ACM | DBLP | Yelp | MAG | Amazon |
|------|-----|------|------|-----|--------|
| GCN | 89.04 | 80.70 | 74.03 | 74.60 | 93.12 |
| HAN | 91.30 | 81.28 | 52.04 | OOM | OOM |
| InfoMGF | 92.81 | 91.45 | 92.01 | 77.32 | 97.78 |
| GMoE | 90.29 | 91.18 | 91.92 | 77.27 | 97.78 |
| Mowst | 85.69 | 89.69 | 91.31 | 77.40 | 97.89 |
| **CoE** | **94.21** | **92.27** | **93.40** | **78.37** | **98.01** |

CoE achieves the best performance across all 5 datasets, exhibiting the lowest standard deviation (e.g., ACM ±0.14, Amazon ±0.09).

### Multimodal Classification (4 Datasets, Without Initial Graph Structure)

| Method | ESP | Flickr | IAPR | NUS |
|------|-----|--------|------|-----|
| QMF | 80.14 | 69.24 | 69.08 | 65.42 |
| CPM-Nets | 80.09 | 69.49 | 67.33 | 65.34 |
| **CoE** | **81.11** | **70.24** | **71.04** | **66.80** |

### Ablation Study

| Variant | ACM | DBLP | Yelp |
|------|-----|------|------|
| RF (Random Forest replacement) | 93.39 | 91.48 | 91.61 |
| WRF (Weighted Random Forest) | 93.64 | 91.97 | 93.05 |
| w/o High-level Experts | 91.25 | 90.71 | 68.27 |
| w/o GSL | 93.60 | 91.13 | 93.14 |
| **CoE (Full)** | **94.21** | **92.27** | **93.40** |

Removing high-level experts has the most significant impact (Yelp drops from 93.40 to 68.27), proving that cross-network fusion is crucial.

## Highlights & Insights

1. **Paradigm Shift from Competition to Cooperation**: This work is the first to propose expert cooperation (instead of MoE competition) in multiplex networks. All experts contribute to the decision-making process, avoiding information loss caused by gating mechanisms.
2. **Ingenious Design of Confidence Tensor**: $\Theta \in \mathbb{R}^{c \times c \times k}$ encodes both expert expertise and class relationships simultaneously, providing significantly greater expressiveness than simple weighting.
3. **Theoretically Sound Large-Margin Optimization**: The smooth approximation of $\max_2$ using logsumexp makes non-convex optimization feasible, with solid convergence and generalization guarantees.
4. **Outstanding Robustness**: On the ACM dataset, CoE maintains stable performance even when 90% of the edges are perturbed.
5. **High Versatility**: The same framework handles both multi-relational networks and multimodal data simultaneously, requiring no structural modifications.

## Limitations & Future Work

1. **Scalability Concerns**: The confidence tensor $\Theta \in \mathbb{R}^{c \times c \times k}$ scales with the number of classes and experts. Large-scale scenarios may face memory bottlenecks (some baselines suffered Out-Of-Memory (OOM) on MAG; although CoE did not, its computational overhead is not discussed).
2. **Limitation to Classification Tasks**: Experiments are limited to node classification. The framework's performance on other graph tasks like link prediction and graph classification remains unverified.
3. **Dependency on KNN Graph Construction**: For multimodal data without an initial graph structure, KNN is used to construct adjacency matrices, making the model potentially sensitive to the choice of $K$.
4. **Limited Number of Experts**: Due to the two-level design, the number of experts is constrained by the number of network layers, making it difficult to scale as flexibly as large-scale MoEs.
5. **Hyperparameter $\alpha$ Sensitivity**: Although experiments show low sensitivity to $\alpha$, an excessively large $\alpha$ within logsumexp can lead to numerical instability.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of expert cooperation and large-margin confidence tensor is novel, and the shift from competition to cooperation offers a meaningful perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Analysis is comprehensive, featuring 9 datasets, ablations, robustness studies, and hyperparameter sensitivity checks.
- Writing Quality: ⭐⭐⭐⭐ — The theoretical analysis is rigorous, the framework explanation is clear, and the top-down narrative style is easy to grasp.
- Value: ⭐⭐⭐⭐ — It provides valuable insights for both multiplex network learning and expert mechanisms. Code has been open-sourced, ensuring high reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](../../ICML2026/llm_efficiency/skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)
- [\[ICML 2025\] Mixture of Lookup Experts](mixture_of_lookup_experts.md)
- [\[ICML 2025\] Autonomy-of-Experts Models (AoE)](autonomy-of-experts_models.md)
- [\[ACL 2025\] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](../../ACL2025/llm_efficiency/dive_moe_reconstruction.md)
- [\[ACL 2025\] Boosting Long-Context Information Seeking via Query-Guided Activation Refilling](../../ACL2025/llm_efficiency/boosting_long-context_information_seeking_via_query-guided_activation_refilling.md)

</div>

<!-- RELATED:END -->
