---
title: >-
  [Paper Note] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients
description: >-
  [NeurIPS 2025][Optimization][federated learning] This paper proposes Fed-NGA, an algorithm that aggregates client gradients via weighted averaging after $\ell_2$ normalization…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "federated learning"
  - "Byzantine Robustness"
  - "Gradient Normalization"
  - "Non-IID Data"
  - "Non-Convex Optimization"
date: 2026-05-08
content_hash: 2115beccf94e316c
---

# Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients

**Conference**: NeurIPS 2025
**arXiv**: [2408.09539](https://arxiv.org/abs/2408.09539)  
**Code**: Not provided  
**Area**: Optimization
**Keywords**: federated learning, Byzantine Robustness, Gradient Normalization, Non-IID Data, Non-Convex Optimization

## TL;DR

This paper proposes Fed-NGA, an algorithm that aggregates client gradients via weighted averaging after $\ell_2$ normalization, achieving Byzantine robustness and resilience to data heterogeneity simultaneously at an extremely low time complexity of $\mathcal{O}(pM)$. Under non-convex loss functions, it is the first to prove zero optimality gap convergence under certain mild conditions.

## Background & Motivation

Federated Learning (FL) enables multiple clients to collaboratively train a model without sharing raw data, but faces two core challenges:

1. **Byzantine Attacks**: Malicious clients can send arbitrarily forged gradient vectors to the server, severely disrupting the global model training process. The server neither knows the number of Byzantine clients nor can identify them.
2. **Data Heterogeneity (Non-IID)**: Significant differences in local data distributions across clients lead to biased local optima, hindering global model convergence.

Existing Byzantine-robust aggregation methods (Median, Krum, Geometric Median, CClip, etc.) address these issues to some extent, but incur **high computational overhead during aggregation** — with time complexities of $\mathcal{O}(pM\log M)$, $\mathcal{O}(pM^2)$, or even $\mathcal{O}(Mp^{3.5})$, which severely slow down training. Moreover, most existing theoretical analyses only guarantee convergence to a neighborhood of a stationary point, falling short of zero optimality gap.

## Core Problem

How can one minimize aggregation complexity while maintaining Byzantine robustness and accommodating data heterogeneity? Specifically:

- Can an aggregation algorithm be designed with time complexity reduced to $\mathcal{O}(pM)$ (linear in model dimension and number of clients)?
- Under non-convex loss functions, can convergence to a true stationary point (zero optimality gap) be theoretically guaranteed?

## Method

### Fed-NGA Algorithm Framework

The core idea of Fed-NGA is remarkably simple: **normalize each client's uploaded gradient vector by its $\ell_2$ norm, then compute a weighted sum to update the global model**.

The procedure is as follows:

1. **Local Update**: Each honest client $m$ computes a local gradient $g_m^t = \nabla F_m(w^t, \xi_m^t)$ based on its local sub-dataset $\xi_m^t$; Byzantine clients may send arbitrary vectors.
2. **Normalized Aggregation**: The server normalizes each received vector by its $\ell_2$ norm and computes a weighted sum with weights $\alpha_m$:

$$w^{t+1} = w^t - \eta^t \sum_{m \in \mathcal{M}} \alpha_m \cdot \frac{g_m^t}{\|g_m^t\|}$$

3. **Broadcast**: The updated global parameters are broadcast to all clients.

### Key Role of Normalization

The normalization operation maps all client gradients onto the unit sphere, thereby:

- **Bounding the influence of Byzantine attacks**: Regardless of the magnitude of gradients sent by malicious clients, their contribution after normalization is constrained to a unit vector, preventing amplified gradients from dominating the aggregation.
- **Mitigating data heterogeneity**: Normalization eliminates gradient magnitude differences, making aggregation focus on gradient directions rather than magnitudes, thereby reducing bias induced by non-IID data.
- **Computational efficiency**: Normalization requires only $\mathcal{O}(p)$ operations per client (norm computation and division), totaling $\mathcal{O}(pM)$ across $M$ clients.

### Theoretical Convergence Guarantees

The paper provides two sets of convergence results under non-convex loss functions:

**Theorem 3.7 (Under General Assumptions)**: Under standard Lipschitz continuity (Assumption 3.1), unbiased gradients (Assumption 3.2), bounded inner error (Assumption 3.3), and bounded data heterogeneity (Assumption 3.4), when the Byzantine fraction satisfies the required constraint, Fed-NGA converges to an $\mathcal{O}(\sigma + \theta)$ neighborhood of a stationary point at a rate of $\mathcal{O}(1/T^{1/2-\delta})$.

**Theorem 3.9 (Zero Gap Under Mild Assumptions)**: Under Type II bounded error assumptions (Assumptions 3.5 and 3.6), when the inner error and data heterogeneity are sufficiently small, Fed-NGA achieves zero optimality gap — i.e., convergence to a true stationary point — at the same rate of $\mathcal{O}(1/T^{1/2-\delta})$. This is the first such result established in the Byzantine-robust FL literature.

The key constraint requires that the weight fraction of Byzantine clients satisfies $\bar{C}_\alpha < 0.5$, meaning malicious clients account for less than half of the total data volume.

## Key Experimental Results

Experiments are conducted on TinyImageNet (MobileNetV3), CIFAR10 (LeNet), and MNIST (MLP) under a Dirichlet($\beta=0.6$) non-IID setting, evaluating five Byzantine attack types: Gaussian, Same-value, Sign-flip, LIE, and FoE.

### Test Accuracy

| Dataset | No Attack | Gaussian | Same-value | Sign-flip | LIE | FoE |
|---|---|---|---|---|---|---|
| TinyImageNet (Fed-NGA) | **56.95** | **55.77** | **49.31** | **45.38** | **55.82** | **45.23** |
| TinyImageNet (Best Baseline) | 55.68 | 54.22 | 48.98 | 43.14 | 54.34 | 42.88 |
| CIFAR10 (Fed-NGA) | **54.48** | **52.07** | **29.16** | **51.16** | **51.82** | **51.16** |
| MNIST (Fed-NGA) | **96.72** | **94.98** | **83.66** | **94.71** | 94.92 | **94.71** |

### Aggregation Time

Fed-NGA's aggregation time is substantially lower than other robust baselines:

- On TinyImageNet: approximately 0.19–0.34 seconds, versus ~50 seconds for Krum (>100× difference)
- On CIFAR10: approximately 0.04–0.06 seconds, versus ~22–32 seconds for Median/Krum (>100× difference)
- On MNIST: approximately 0.04–0.05 seconds, versus ~1.2–324 seconds for other methods

Fed-NGA's aggregation time is only approximately 3× that of FedAvg, which provides no robustness guarantees.

## Highlights & Insights

- **Minimal yet effective design**: Gradient normalization followed by weighted summation achieves an optimal aggregation complexity of $\mathcal{O}(pM)$, far below existing methods
- **First proof of zero optimality gap**: Under mild conditions, this is the first work to prove that Byzantine-robust FL can converge to a true stationary point rather than a neighborhood thereof
- **Simultaneous resolution of two challenges**: Normalization naturally addresses both Byzantine attacks and data heterogeneity
- **Comprehensive empirical validation**: Covering three datasets, three model architectures, and five attack types, Fed-NGA demonstrates advantages in both accuracy and efficiency

## Limitations & Future Work

- The zero optimality gap guarantee relies on Type II assumptions (Assumptions 3.5 and 3.6), which require bounded error and heterogeneity after normalization; these conditions may not hold in late-stage training when gradients approach zero
- The convergence rate $\mathcal{O}(1/T^{1/2-\delta})$ is slightly slower than standard SGD's $\mathcal{O}(1/\sqrt{T})$, where $\delta \in (0, 1/2)$ introduces an additional suboptimal factor
- Only single-step local updates are considered (one gradient computation per client per round); the method is not extended to multi-step local updates (e.g., multiple local SGD rounds as in FedAvg)
- Under the Same-value attack, CIFAR10 accuracy drops to only 29.16%; while superior to baselines, this indicates certain attack patterns remain challenging

## Related Work & Insights

| Method | Time Complexity | Non-Convex | Non-IID | Zero Gap |
|---|---|---|---|---|
| Median | $\mathcal{O}(pM\log M)$ | ✗ | ✗ | ✗ |
| Krum | $\mathcal{O}(pM^2)$ | ✓ | ✗ | ✗ |
| GM (RFA) | $\mathcal{O}(pM\log^3(M\epsilon^{-1}))$ | ✗ | ✓ | ✗ |
| CClip | $\mathcal{O}(\tau Mp)$ | ✓ | ✗ | ✗ |
| MCA | $\mathcal{O}(pM\log^3(M\epsilon^{-1}))$ | ✗ | ✓ | ✗ |
| **Fed-NGA** | $\mathcal{O}(pM)$ | **✓** | **✓** | **✓** |

Fed-NGA is the only method that simultaneously satisfies support for non-convex loss, non-IID robustness, linear aggregation complexity, and zero optimality gap guarantee.

Gradient normalization as a general technique has been applied in settings such as heavy-tailed noise mitigation and sharpness-aware minimization; this paper demonstrates its value through cross-domain transfer to Byzantine-robust FL. The work also suggests a design philosophy: **minimal operations coupled with rigorous theoretical analysis** often yield greater practical value than complex aggregation mechanisms, particularly in edge computing scenarios with limited resources. Future directions include combining normalization with other robust techniques (e.g., clipping, momentum), or extending the framework to asynchronous FL and decentralized FL settings.

## Rating
- Novelty: 3.5/5 — Normalization itself is not new, but using it as the sole robust aggregation mechanism with complete theoretical analysis is novel
- Experimental Thoroughness: 4/5 — Comprehensive coverage across three datasets and five attacks, though large-scale model experiments and broader non-IID settings are lacking
- Writing Quality: 4/5 — Clear structure, rigorous theoretical derivations, and intuitive comparison tables
- Value: 4/5 — Highly practical, particularly well-suited for resource-constrained federated learning deployments

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[ICML 2026\] Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation](../../ICML2026/optimization/delayed_momentum_aggregation_communication-efficient_byzantine-robust_federated_.md)
- [\[NeurIPS 2025\] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)

</div>

<!-- RELATED:END -->
