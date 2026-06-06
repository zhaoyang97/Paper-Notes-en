---
title: >-
  [Paper Note] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections
description: >-
  [NeurIPS 2025][Optimization][Federated Learning] This paper proposes FedAux, a framework that introduces differentiable Auxiliary Projection Vectors (APVs) to map node embeddings into a one-dimensional space and perform…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Federated Learning"
  - "Graph Neural Networks"
  - "Personalized Aggregation"
  - "Subgraph Heterogeneity"
  - "Auxiliary Projection Vector"
date: 2026-05-08
content_hash: 55bc133ed6ce1639
---

# Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections

**Conference**: NeurIPS 2025  
**arXiv**: [2505.23864](https://arxiv.org/abs/2505.23864)  
**Code**: [GitHub](https://github.com/JhuoW/FedAux)  
**Area**: Optimization  
**Keywords**: Federated Learning, Graph Neural Networks, Personalized Aggregation, Subgraph Heterogeneity, Auxiliary Projection Vector

## TL;DR

This paper proposes FedAux, a framework that introduces differentiable Auxiliary Projection Vectors (APVs) to map node embeddings into a one-dimensional space and perform soft-ranking aggregation via Gaussian kernels. The APV simultaneously serves as a compact, privacy-preserving summary of the local subgraph for server-side similarity computation and participates in joint client-side optimization, enabling personalized subgraph federated learning.

## Background & Motivation

In Subgraph Federated Learning (Subgraph FL), each client holds a subgraph of a global graph, and severe non-IID heterogeneity exists across subgraphs. For instance, in multi-region social platforms, user interaction patterns and interests differ substantially across regions, making direct FedAvg aggregation of GNN models highly ineffective.

The central challenge in personalized FL is **how to measure client similarity without sharing raw data**:

- **Directly comparing parameter matrices**: Distance metrics in high-dimensional parameter spaces are unreliable (curse of dimensionality).
- **Comparing gradients**: Information is limited and the approach tends to be heuristic.
- **Sharing embeddings**: Violates privacy constraints.
- **Anchor graph methods**: A public graph is generated on the server as a testbed, but cannot explicitly model subgraph heterogeneity.

The key insight of this paper is that a **compact low-dimensional proxy** can be derived directly from model parameters to faithfully summarize local subgraph characteristics without leaking private information. This proxy should be compact enough to avoid the pitfalls of high-dimensional distance metrics, yet expressive enough to reflect meaningful differences across clients.

## Method

### Overall Architecture

The FedAux workflow proceeds as follows:
1. The server maintains global GNN parameters $\theta$ and an Auxiliary Projection Vector (APV) $\mathbf{a}$.
2. Each communication round: broadcast $(\theta, \mathbf{a})$ → local client training → upload $(\theta_k, \mathbf{a}_k)$ → server performs personalized aggregation.

### Key Designs

1. **Auxiliary Projection Vector (APV) and One-Dimensional Space Mapping**: Each client projects GNN-produced node embeddings $h_{k,i}$ onto the APV direction to obtain a scalar similarity score $s_{k,i} = \langle \hat{h}_{k,i}, \mathbf{a}_k \rangle$, mapping each node into the one-dimensional $\mathbf{a}_k$-space. The APV is learnable; clients adaptively adjust this space during training to capture inter-node relationships.

2. **Differentiable Kernel Aggregation as a Substitute for Hard Sorting**: Early methods employed hard sorting combined with 1D convolution to aggregate information from neighboring nodes, but sorting is non-differentiable, preventing APV optimization via backpropagation. This paper proposes soft sorting via a Gaussian kernel:

    $z_{k,i} = \frac{1}{M_i} \sum_{j=1}^{N_k} \kappa(s_{k,i}, s_{k,j}) h_{k,j}, \quad \kappa(s_i, s_j) = \exp\left(-\frac{(s_i - s_j)^2}{\sigma^2}\right)$

   This continuous aggregator is fully differentiable with respect to the APV — changes in the APV smoothly adjust each $s_{k,i}$, which in turn adjusts the kernel weights.

3. **Theoretical Guarantee for APV (Theorem 3.1)**: It is proven that under Gaussian kernel aggregation, the gradient of the loss with respect to the APV converges to $-\frac{2}{\sigma^2} \mathbf{C}\mathbf{a}$ (where $\mathbf{C}$ is the embedding covariance matrix) in the limit $\sigma \to 0$. With unit-norm renormalization, the update rule reduces to the **Oja learning rule**, whose global attractor is the principal eigenvector of $\mathbf{C}$. This implies that the APV is not an arbitrary trainable parameter, but rather a **statistically optimal, variance-maximizing summary of the local embeddings**.

4. **Server-Side Personalized Aggregation**: Cosine similarity between client APVs is computed, and temperature-scaled softmax yields aggregation weights:

    $w_{k,l} = \frac{\exp(\alpha \text{Sim}(\mathbf{a}_k, \mathbf{a}_l))}{\sum_{r=1}^K \exp(\alpha \text{Sim}(\mathbf{a}_k, \mathbf{a}_r))}$

   Personalized parameters are then generated for each client as $\theta_k = \sum_l w_{k,l} \theta_l$, amplifying contributions from similar clients and reducing interference from dissimilar ones.

### Loss & Training

- Client loss: cross-entropy $\mathcal{L}_k = \frac{1}{N_k} CE(\text{CLF}(\Gamma_k), Y_k)$, where $\Gamma_k = [h_{k,i} \| z_{k,i}]$
- Joint optimization of GNN parameters $\theta_k$, APV $\mathbf{a}_k$, and classifier $\Phi_k$
- Kernel bandwidth $\sigma = 1$ is effective across all datasets
- Global linear convergence guarantee (Theorem 3.3): $E[\mathcal{L}(\Psi^{(T)}) - \mathcal{L}^\star] \leq (1-\eta\mu)^{QT}(\mathcal{L}(\Psi^{(0)}) - \mathcal{L}^\star) + \frac{\eta\mathscr{L}\zeta^2}{2\mu} + \frac{2\eta\mathscr{L}\rho^2}{\mu(1-\rho)^2}$

## Key Experimental Results

### Main Results (Federated Node Classification)

| Dataset | # Clients | FedAux | FED-PUB (SOTA) | FedAvg | Local |
|---|---|---|---|---|---|
| Cora | 5 | **84.57±0.39** | 83.72±0.18 | 74.45±5.64 | 81.30±0.21 |
| Cora | 10 | **82.05±0.71** | 81.45±0.12 | 69.19±0.67 | 79.94±0.24 |
| Cora | 20 | **81.60±0.64** | 81.10±0.64 | 69.50±3.58 | 80.30±0.25 |
| CiteSeer | 5 | **72.99±0.82** | 72.40±0.26 | 71.06±0.60 | 69.02±0.05 |
| CiteSeer | 10 | **73.16±0.29** | 71.83±0.61 | 63.61±3.59 | 67.82±0.13 |
| CiteSeer | 20 | **68.10±0.35** | 66.89±0.14 | 64.68±1.83 | 65.98±0.17 |
| Pubmed | 5 | **88.10±0.16** | 86.81±0.12 | 79.40±0.11 | 84.04±0.18 |
| Pubmed | 10 | **86.43±0.20** | 86.09±0.17 | 82.71±0.29 | 82.81±0.39 |
| Pubmed | 20 | **84.87±0.42** | 84.66±0.54 | 80.97±0.26 | 82.65±0.03 |

### Additional Datasets

| Dataset | # Clients | FedAux | GCFL | FedPer | FedAvg |
|---|---|---|---|---|---|
| Amazon-Computer | 10 | 90.50+ | 90.03 | 89.73 | 79.54 |
| Amazon-Photo | 10 | 92.50+ | 92.06 | 91.76 | 83.15 |
| ogbn-arxiv | 5 | 67.00+ | 66.80 | 66.87 | 65.54 |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---|---|---|
| Hard sorting vs. kernel aggregation | Kernel aggregation is superior | Differentiability enables full APV optimization |
| With APV vs. without APV aggregation | With APV is significantly better | Validates APV effectiveness as a subgraph summary |
| $\sigma=1$ vs. other values | $\sigma=1$ is optimal | Acts as regularization to prevent overfitting |
| Verification of Theorem 3.2 | Converges to hard sorting as $\sigma \to 0$ | Theory and experiment are consistent |

### Key Findings

- FedAux **consistently outperforms** all baselines across all 6 datasets and 3 client scales.
- The most notable gain is on CiteSeer with 10 clients (73.16 vs. 71.83, absolute improvement of 1.33%).
- The APV is only a low-dimensional vector, posing minimal privacy leakage risk.
- On Pubmed, FedAux improves upon local-only training by 4%, demonstrating effective cross-client knowledge transfer.

## Highlights & Insights

- The APV design achieves three goals simultaneously: it participates in local training to improve the model, serves as a client signature for similarity computation, and protects privacy.
- Theorem 3.1 reveals that the APV corresponds to the principal component of the embedding covariance matrix, providing theoretical grounding for the design choice.
- Replacing hard sorting with a Gaussian kernel is critical — it resolves the gradient-blocking problem and enables end-to-end training.
- Complexity analysis is clear: $O(|E_k|d' + N_k^2 d')$ per client, $O(K^2 d')$ on the server.

## Limitations & Future Work

- The $O(N_k^2)$ complexity of kernel aggregation may become a bottleneck for large-scale subgraphs.
- The APV as a principal component may fail to capture critical structural differences across subgraphs under certain data distributions.
- Evaluation is limited to node classification; graph-level and link prediction tasks are not addressed.
- Sensitivity analysis of the temperature parameter $\alpha$ is insufficiently thorough.

## Related Work & Insights

- Contrasts with FED-PUB's subgraph masking approach: FedAux replaces discrete masks with continuous projections.
- The connection to the Oja learning rule offers a new perspective on representation alignment in federated learning.
- Inspiration: similar differentiable projection mechanisms could be applied in other FL scenarios requiring client signatures or fingerprints, such as heterogeneity detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The APV concept is original; the differentiable kernel aggregation design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Six datasets, three client scales, and extensive baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear; theoretical analysis is rigorous.
- **Value**: ⭐⭐⭐⭐ — Provides a concise and effective personalization solution for subgraph FL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](../../AAAI2026/optimization/personalized_federated_learning_with_bidirectional_communication_compression_via.md)
- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)

</div>

<!-- RELATED:END -->
