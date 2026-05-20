---
title: >-
  [Paper Note] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System
description: >-
  [NeurIPS 2025][Optimization][Federated Learning] This paper proposes MAR-FL, a system that reduces the communication complexity of P2P federated learning from $O(N^2)$ to $O(N \log N)$ via a Moshpit All-Reduce mechanism…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Federated Learning"
  - "P2P Communication"
  - "Moshpit All-Reduce"
  - "Differential Privacy"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: 75445b06d144543d
---

# MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System

**Conference**: NeurIPS 2025
**arXiv**: [2512.05234](https://arxiv.org/abs/2512.05234)  
**Code**: [https://github.com/felix-fjm/mar-fl](https://github.com/felix-fjm/mar-fl)  
**Area**: Optimization / Federated Learning
**Keywords**: Federated Learning, P2P Communication, Moshpit All-Reduce, Differential Privacy, Knowledge Distillation

## TL;DR

This paper proposes MAR-FL, a system that reduces the communication complexity of P2P federated learning from $O(N^2)$ to $O(N \log N)$ via a Moshpit All-Reduce mechanism and dynamic group aggregation, while maintaining robustness against network jitter.

## Background & Motivation

**Background**: Federated learning (FL) is transitioning from centralized to P2P architectures to eliminate central server bottlenecks and single points of failure.

**Limitations of Prior Work**: Existing P2P FL approaches incur high communication costs (e.g., RDFL: $O(N^2)$), making them difficult to scale to large deployments; they are also vulnerable to node churn.

**Key Challenge**: Global model aggregation requires all nodes to communicate, yet P2P settings lack a central coordinator — the key challenge is how to efficiently achieve global averaging in a fully distributed setting.

**Key Insight**: The paper draws inspiration from Moshpit SGD's dynamic grouping strategy, using a distributed hash table (DHT) to coordinate only metadata rather than model parameters.

**Core Idea**: Global aggregation is decomposed into $\lceil\log_M N\rceil$ rounds of local group-wise aggregation, with deterministic group keys used to avoid redundant pairings.

## Method

### Overall Architecture

Each peer performs: local Momentum-SGD update → multiple rounds of Moshpit All-Reduce (MAR) group aggregation → optional Moshpit Knowledge Distillation (MKD) → globally averaged model.

### Key Designs

1. **Moshpit All-Reduce Aggregation (MAR)**

    - **Function**: Partitions $N$ peers into groups of size $M$, with each peer communicating with $M-1$ others per round.
    - **Mechanism**: Global averaging is achieved in $\lceil\log_M N\rceil \approx \log N$ rounds, yielding a total communication complexity of $O(N \log N)$.
    - **Design Motivation**: A single peer dropout affects only its local group and does not block the overall process.

2. **Distributed Coordination (DHT)**

    - **Function**: Uses Hivemind Kademlia DHT to coordinate lightweight information only.
    - **Mechanism**: Model weights are never transmitted through the DHT; only barrier and group metadata are exchanged.
    - **Design Motivation**: Control-plane overhead is $O(N \log N)$, which is negligible relative to model exchange traffic.

3. **Moshpit Knowledge Distillation (MKD)**

    - **Function**: Accelerates convergence and further reduces the number of required communication rounds.
    - **Mechanism**: Automatically selects top-$\ell$ teachers based on minimum KL divergence, combining KL divergence and cross-entropy losses with a smooth weight decay schedule $\lambda = \max(0, 1 - (t-1)/K)$.
    - **Design Motivation**: Reduces communication requirements by an additional 2–3×.

4. **Differential Privacy Adaptation**

    - **Function**: Adapts the DP mechanism from FedAvg to a serverless P2P architecture.
    - **Mechanism**: Each peer performs local gradient clipping and noise addition; aggregation is performed within groups via MAR.
    - **Design Motivation**: DP guarantees are fully decentralized, requiring no central server.

### Loss & Training

Knowledge distillation loss: $L = \lambda\tau^2 D_{KL}(\text{teacher} \| \text{student}) + (1-\lambda)\,CE(y, \text{student})$. Convergence analysis shows that the mixing error satisfies exponential convergence.

## Key Experimental Results

### Main Results

| Method | Communication Cost (Relative) | Model Performance | Scalability |
|---|---|---|---|
| FedAvg (Centralized) | 1.0× | 99%+ | Central bottleneck |
| RDFL (P2P, 125 nodes) | 10.0× | 99%+ | $O(N^2)$ |
| **MAR-FL** | **1.0×** | **99%+** | **$O(N\log N)$** |

### Ablation Study

| Configuration | Communication Rounds to 50% Accuracy | Relative Improvement |
|---|---|---|
| MAR-FL (base) | 8 rounds | Baseline |
| + MKD (K=20) | 3.5 rounds | 2.3× |
| + Teacher selection | 3.2 rounds | 2.5× |
| + Progressive decay | 3.1 rounds | 2.6× |

### Key Findings

- Communication efficiency improves by 10× over RDFL ($O(N\log N)$ vs. $O(N^2)$).
- Performance degrades by less than 5% at 50% participation rate, demonstrating robustness to partial participation.
- DP-MAR-FL exhibits the same privacy-utility tradeoff curve as centralized FedAvg with DP.
- MKD further reduces the communication required for convergence by 56%.

## Highlights & Insights

- **Communication Complexity Breakthrough**: The reduction from $O(N^2)$ to $O(N\log N)$ represents a significant advance for P2P FL, enabling scalability to thousands of nodes.
- **Modular Serverless Design**: Fully decentralized with no single point of failure; DHT transmits only metadata.
- **Natural Extension to Privacy**: DP adaptation requires no central coordination, preserving privacy guarantees end-to-end.

## Limitations & Future Work

- A performance gap relative to centralized FedAvg remains, though the advantage over P2P baselines is clear.
- Performance degrades noticeably when participation rate falls below 50%.
- Experiments are conducted only on MNIST and 20NG; deployment over real wireless networks is not simulated.

## Related Work & Insights

- **vs. RDFL**: RDFL uses a closed topology with $O(N^2)$ communication and lacks fault tolerance; MAR-FL employs dynamic grouping with strong fault tolerance.
- **vs. Moshpit SGD**: This work is the first to apply dynamic grouping in the FL setting, additionally incorporating differential privacy and knowledge distillation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Communication complexity breakthrough + knowledge distillation integration
- Experimental Thoroughness: ⭐⭐⭐⭐ Optimization experiments are thorough; real-network deployment is absent
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic and rigorous derivations
- Value: ⭐⭐⭐⭐⭐ Directly addresses the scalability challenge in P2P FL

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)

</div>

<!-- RELATED:END -->
