---
title: >-
  [Paper Note] Efficient Adaptive Federated Optimization
description: >-
  [NeurIPS 2025][Optimization][Federated Learning] FedAda2/FedAda2++ proposes efficient server–client joint adaptive optimization for federated learning: client-side local preconditioners are initialized from zero (elimina…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Federated Learning"
  - "Adaptive Optimization"
  - "Communication Efficiency"
  - "Memory Efficiency"
  - "Joint Adaptivity"
date: 2026-05-08
content_hash: 2e9818e9a4f4f0d8
---

# Efficient Adaptive Federated Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2410.18117](https://arxiv.org/abs/2410.18117)
**Code**: None (no public code mentioned)
**Area**: Federated Learning / Distributed Optimization
**Keywords**: Federated Learning, Adaptive Optimization, Communication Efficiency, Memory Efficiency, Joint Adaptivity

## TL;DR
FedAda2/FedAda2++ proposes efficient server–client joint adaptive optimization for federated learning: client-side local preconditioners are initialized from zero (eliminating server-to-client transmission), with an optional SM3-based memory-efficient compression of local statistics. The method is theoretically shown to achieve the same $O(T^{-1/2})$ convergence rate as full joint adaptivity, while incurring the same communication cost as FedAvg.

## Background & Motivation

**Background**: In federated learning, adaptive optimizers (Adam, AdaGrad) can accelerate convergence and improve accuracy. Adaptivity can be applied server-side (e.g., FedAdam), client-side, or jointly on both ends. Joint adaptivity has been shown to yield the best performance.

**Limitations of Prior Work**: A naïve implementation of joint adaptivity requires the server to transmit the global preconditioner (e.g., second-moment statistics) to clients at each communication round, incurring (a) doubled communication overhead—per-round transmission increases from $2d$ to $3d$ (where $d$ is the model dimension); and (b) increased client memory—clients must maintain local preconditioners.

**Key Challenge**: Joint adaptivity improves performance, but its communication and memory costs conflict with the resource-constrained environments typical of cross-device federated learning.

**Goal**: To retain the benefits of joint adaptivity while eliminating the associated communication and memory overhead.

**Key Insight**: A simple yet unverified idea—initializing local preconditioners from zero at each round, without transmitting the global preconditioner from the server. This paper provides the first formal convergence guarantees for this strategy.

**Core Idea**: Zero-initializing client local preconditioners, with optional SM3-based memory compression, suffices to achieve convergence and performance equivalent to costly joint adaptivity.

## Method

### Overall Architecture
Each round of FedAda2 proceeds as follows: (1) the server broadcasts the current model $x_t$ to a sampled subset of clients; (2) each client initializes its local preconditioner to zero and performs $K$ local update steps using an adaptive optimizer; (3) clients transmit only the model update $\Delta_i^t$ (not the preconditioner) back to the server; (4) the server aggregates updates using an adaptive optimizer. FedAda2++ further applies SM3 compression to client-side preconditioners to reduce memory usage.

### Key Designs

1. **Zero Local Preconditioner Initialization**:

    - **Function**: Clients re-initialize local gradient statistics from zero at each round.
    - **Mechanism**: In cross-device federated learning, clients are stateless; long intervals between participation render stale states unreliable. Zero initialization eliminates server-to-client preconditioner transmission, keeping communication at $2d$ (matching FedAvg).
    - **Design Motivation**: Empirical results show that zero initialization performs no worse than transmitting the global preconditioner, while halving the communication volume.
    - **Additional Benefit**: Allows the server and clients to use different optimizers (e.g., server Adam + client AdaGrad).

2. **SM3 Memory-Efficient Client Optimization (FedAda2++)**:

    - **Function**: Compresses client-side local preconditioners using the SM3 algorithm.
    - **Mechanism**: SM3 decomposes second-moment statistics into low-rank approximations along parameter group dimensions. For ViT models, the additional preconditioner memory is only 0.48% of the full preconditioner—a 99% memory reduction.
    - **Design Motivation**: Resource-constrained devices (mobile phones, IoT) cannot maintain full Adam second-moment states.

3. **Delayed Preconditioner Updates**:

    - **Function**: Allows clients to update local statistics only once every $z$ steps.
    - **Mechanism**: Reduces local computation overhead; theoretical analysis shows convergence rate is robust to the choice of $z$.
    - **Design Motivation**: Further alleviates the computational burden on clients.

4. **Blended Optimization Framework**:

    - **Function**: Allows each client to employ a different optimizer strategy per round.
    - **Mechanism**: The framework unifies homogeneous and heterogeneous optimizer configurations; optimizer selection can be dynamically adjusted based on device resources.
    - **Design Motivation**: In real-world deployments, device resource heterogeneity is significant.

### Theoretical Guarantees

**Theorem 4.1**: Under $L$-smoothness and bounded gradient assumptions, FedAda2/FedAda2++ satisfies:
$$\min_{t \in [T]} \|\nabla f(x_{t-1})\|^2 \leq \frac{\Psi_1 + \Psi_2 + \Psi_3 + \Psi_4 + \Psi_5}{\Psi_6}$$

**Corollary 4.3**: The convergence rate is $O(T^{-1/2})$, identical to that of full joint adaptive methods (e.g., FedAdam with preconditioner transmission).

## Key Experimental Results

### Main Results (ViT Fine-tuning)

| Dataset | Method | Final Test Accuracy | Communication | Client Memory |
|--------|------|------------|------|-----------|
| CIFAR-100 | FedAvg | Lowest | 2d | d |
| CIFAR-100 | FedAdam (server-only) | Moderate | 2d | d |
| CIFAR-100 | Costly Joint Adap. | High | 3d | 2d |
| CIFAR-100 | **FedAda2** | **High** | **2d** | 2d |
| CIFAR-100 | **FedAda2++** | **High** | **2d** | **~d** |

Consistent trends are observed across three datasets (CIFAR-100, GLD-23K, FEMNIST): FedAda2/FedAda2++ matches the performance of costly joint adaptivity while maintaining FedAvg-level communication costs. On FEMNIST, the advantage of joint adaptivity is most pronounced, with substantial gains over both FedAvg and server-only methods.

### Communication Efficiency Comparison

| Method | Joint Adaptivity | Communication | Computation (gradient calls) | Client Memory |
|------|----------|------|-------------|-----------|
| FedAvg | No | 2d | 1 | d |
| FedAdam | No | 2d | 1 | d |
| MIME | No | 5d | 3 | 4d |
| Costly Joint Adap. | Yes | 3d | 1 | 2d |
| **FedAda2** | **Yes** | **2d** | 1 | 2d |
| **FedAda2++** | **Yes** | **2d** | 1 | **~d** |

FedAda2 is the only method that achieves joint adaptivity with communication cost equal to FedAvg.

### Differential Privacy Setting

| Method | StackOverflow Accuracy |
|------|-------------------|
| FedAvg | Lowest |
| FedAdam | Moderate |
| **FedAda2/FedAda2++** | **Significantly highest** |

Under a privacy budget of $(\varepsilon, \delta) = (13.1, 0.0025)$, the advantage of FedAda2 is even more pronounced.

### Key Findings
- Zero-initialized local preconditioners perform no worse than transmitting the global preconditioner across all settings, and occasionally outperform it.
- Cross-optimizer configurations (e.g., server Adam + client AdaGrad) are robust with no significant performance degradation.
- SM3-compressed preconditioners are robust to hyperparameter choices, likely due to a denoising effect of the projection.
- FedAda2++ shows the greatest advantage when the number of local epochs is small; the advantage diminishes but remains stable as local epochs increase.
- The benefit of joint adaptivity is amplified under privacy constraints.

## Highlights & Insights
- **Remarkably simple yet effective core idea**: Zero-initializing local preconditioners—an approach previously used informally but lacking theoretical support—is here given rigorous convergence guarantees for the first time.
- **Communication savings are "free"**: Joint adaptivity communication is reduced from $3d$ to $2d$ without any performance loss, and with identical theoretical convergence rates.
- **99% memory reduction via SM3**: Preconditioner storage is compressed from $d$ to $0.0048d$, greatly extending the feasibility of adaptive optimization on mobile devices.
- **Blended optimization framework**: Supports heterogeneous devices running different optimizers, constituting a practically deployable and realistic design.

## Limitations & Future Work
- Theoretical analysis assumes full-batch gradients; extension to stochastic gradients is identified as explicit future work.
- Experiments are conducted only on ViT and logistic regression models, without covering LLM-scale settings.
- The advantage of zero initialization diminishes with a large number of local epochs; smarter initialization strategies may warrant further investigation.
- No automated method exists for selecting the optimal strategy in blended optimization.

## Related Work & Insights
- **vs. FedAdam/FedAdaGrad**: These methods apply adaptivity only on the server side, missing the acceleration from client-side adaptivity; FedAda2 addresses this gap.
- **vs. MIME/MIMELite**: MIME transmits additional optimizer states to simulate centralized adaptivity, incurring $5d$/$4d$ communication and $3\times$/$2\times$ computation; FedAda2 uses $2d$ communication and $1\times$ computation.
- **vs. Local AdaAlter/Local AMSGrad**: These methods apply adaptivity only on the client side and require transmitting client preconditioners to the server for aggregation; FedAda2 achieves dual-side adaptivity without any additional transmission.

## Rating
- **Novelty**: ⭐⭐⭐ The core idea (zero initialization) is simple and has been used informally before; the contribution lies in the theoretical proof and systematic study.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple datasets, diverse optimizer configurations, privacy settings, hyperparameter sensitivity analysis, and 20 repeated runs.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical sections are rigorous and experimental details are thorough, though the main text is highly technical.
- **Value**: ⭐⭐⭐⭐ Provides direct guidance for practical federated learning deployment and fills a gap in the theoretical literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Adaptive Experimentation with Noncompliance](efficient_adaptive_experimentation_with_noncompliance.md)
- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)

</div>

<!-- RELATED:END -->
