---
title: >-
  [Paper Note] PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture
description: >-
  [NeurIPS 2025][AI Safety][Vertical Federated Learning] This paper proposes PubSub-VFL, an efficient two-party vertical federated learning framework based on a publisher/subscriber architecture. Through a hierarchical asy…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Vertical Federated Learning"
  - "Split Learning"
  - "Pub/Sub Architecture"
  - "Asynchronous Training"
  - "Resource Heterogeneity"
date: 2026-05-08
content_hash: 016f816a25797e28
---

# PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture

**Conference**: NeurIPS 2025
**arXiv**: [2510.12494](https://arxiv.org/abs/2510.12494)  
**Code**: Not available  
**Area**: AI Safety
**Keywords**: Vertical Federated Learning, Split Learning, Pub/Sub Architecture, Asynchronous Training, Resource Heterogeneity

## TL;DR

This paper proposes PubSub-VFL, an efficient two-party vertical federated learning framework based on a publisher/subscriber architecture. Through a hierarchical asynchronous mechanism and system-profiling-based hyperparameter optimization, it achieves 2–7× training speedup and up to 91% computational resource utilization while preserving privacy and model accuracy.

## Background & Motivation

Vertical Federated Learning (VFL) enables multiple parties holding disjoint features to collaboratively train models without exposing raw data, making it an important privacy-preserving paradigm for data collaboration. In VFL, each party trains a bottom model up to a split layer, securely transmits embeddings to the active party (which holds labels), and the active party completes training of the top model. However, existing VFL architectures face two major efficiency bottlenecks:

**System Coupling**: Even with parameter server (PS) architectures that improve data parallelism, VFL's synchronous training dependencies and ID alignment steps introduce waiting bottlenecks among workers across parties. Directly applying asynchronous mechanisms is constrained by VFL's unique ID alignment requirements.

**Resource and Data Heterogeneity**: Computational resources and feature dimensionalities differ significantly across parties. Existing methods typically focus on improving efficiency for a single party while neglecting global load balancing, resulting in insufficient overall resource utilization.

The root cause of both problems lies in the absence of a system design that decouples data alignment from training tasks, and the lack of a global resource optimization strategy that accounts for privacy constraints.

## Method

### Overall Architecture

PubSub-VFL addresses the above problems through a three-layer design: (1) a Pub/Sub architecture for **inter-party asynchrony**, decoupling ID alignment from training; (2) **intra-party semi-asynchrony** within the PS, adaptively adjusting synchronization intervals; and (3) **dynamic programming** based on system profiling to determine optimal hyperparameters.

### Key Designs

1. **Publisher/Subscriber Architecture and Channel Design**: Two types of communication channels are introduced: embedding channels and gradient channels. Each training batch is assigned a unique batch ID to label its channel, allowing workers to independently publish/subscribe intermediate results to the corresponding channel without waiting for the counterpart. For $n$ samples and batch size $B$, the system maintains $\lceil n/B \rceil$ channels. To prevent channel congestion, a **buffering mechanism** (FIFO eviction of stale data, with capacity of $p$ embeddings/$q$ gradients) and a **deadline mechanism** (if data is not received within timeout $T_{ddl}$, the current batch is discarded and reassigned) are designed.

2. **Intra-Party Semi-Asynchronous Mechanism**: An adaptive synchronization interval is further introduced within the PS framework, defined as:
    $\Delta T_t = \left\lceil \frac{\Delta T_0}{2} \cdot \tanh\left(\frac{2t}{\Delta T_0} - 2\right) + \frac{\Delta T_0}{2} \right\rceil$
   where $\Delta T_0$ is the initial interval and $t$ is the current training round. The interval is short in early training to ensure stable learning, and gradually increases as accuracy improves to reduce synchronization frequency. Together with the inter-party Pub/Sub asynchrony, this forms a hierarchical asynchronous mechanism.

3. **System Profiling and Dynamic Programming Optimization**: Computation and communication latencies for both parties are modeled as: forward propagation time $T_f^{(a)}(B) = \frac{\lambda_a B^{\gamma_a} w_a}{C_a}$, with backward propagation defined analogously. The optimization objective is to minimize the maximum per-iteration time across both parties:
    $\min \mathcal{O}(w_A, w_P, B) = \min_{w_a, w_p, B \leq B_{max}} \left\{ \max(T_A, T_P) \right\}$
   subject to the memory constraint $B_{max} = \min\left\{\left(\frac{\bar{M}_A - M_{A0}}{\rho_A}\right)^{1/\chi}, \left(\frac{\bar{M}_P - M_{P0}}{\rho_P}\right)^{1/\chi}\right\}$. Dynamic programming is used to search for the optimal configuration in the discrete space $(w_a, w_p, B)$.

### Privacy Protection

A Gaussian Differential Privacy (GDP) protocol is applied to perturb the embeddings sent by the passive party. The authors prove that PubSub-VFL maintains stable convergence after integrating GDP.

## Key Experimental Results

### Main Results — Accuracy Comparison

| Dataset | Metric | VFL | VFL-PS | AVFL | AVFL-PS | PubSub-VFL |
|--------|------|-----|--------|------|---------|------------|
| Energy | RMSE↓ | 84.58 | 84.44 | 85.41 | 85.39 | **85.64** |
| Blog | RMSE↓ | 23.20 | 23.12 | 23.38 | 23.45 | **22.34** |
| Bank | AUC↑ | 94.54 | 94.13 | 94.12 | 94.16 | **96.54** |
| Credit | AUC↑ | 81.90 | 81.34 | 80.83 | 80.34 | **82.34** |
| Synthetic | AUC↑ | 91.27 | 91.31 | 90.97 | 91.21 | **92.87** |

### Efficiency Comparison

| Method | Runtime | CPU Utilization | Waiting Time/Epoch | Communication Cost |
|------|---------|----------|--------------|---------|
| VFL | Baseline | ~50% | High | High |
| AVFL-PS | Second best | ~55% | Moderate | Moderate |
| PubSub-VFL | **7× speedup** | **+35% CPU** | **Lowest** | **Lowest** |

### Ablation Study

| Configuration | Energy | Bank | Credit | Synthetic | Note |
|------|--------|------|--------|-----------|------|
| PubSub-VFL (full) | 83.94 | **96.97** | **86.07** | **94.17** | Best |
| w/o deadline mechanism | 84.35 | 95.26 | 85.74 | 92.86 | AUC drops |
| w/o dynamic programming | 84.07 | 96.33 | 85.79 | 93.82 | Slight drop |
| w/o semi-async mechanism | 85.68 | 95.01 | 84.45 | 92.07 | Notable drop |
| w/o PubSub architecture | 83.98 | 95.17 | 85.93 | 93.52 | Efficiency loss |
| Vanilla VFL | 84.24 | 94.97 | 83.42 | 92.74 | Baseline |

### Hyperparameter Sensitivity

| # Workers | Accuracy (%) | Time (s) | CPU (%) | Waiting (s) |
|-----------|---------|---------|--------|---------|
| 4 | 92.13 | 712.78 | 67.52 | 1.47 |
| 8* | 92.06 | **668.11** | **88.04** | 1.53 |
| 20 | 92.00 | 1420.32 | 42.77 | 8.09 |

| Batch Size | Accuracy (%) | Time (s) | CPU (%) |
|-----------|---------|---------|--------|
| 32 | 92.06 | 668.11 | 88.04 |
| 256* | 92.67 | **92.54** | **91.07** |
| 1024 | 92.21 | 865.74 | 52.67 |

### Key Findings

- PubSub-VFL achieves 2–7× training speedup without sacrificing accuracy.
- Under severely heterogeneous resource conditions (CPU ratio 50:14), PubSub-VFL maintains 87.42% CPU utilization, compared to only 42.12% for AVFL-PS.
- The hierarchical asynchronous mechanism and deadline mechanism are critical for performance; removing them causes a 2.10% AUC drop on the Synthetic dataset.
- The differential privacy protocol has minimal impact on accuracy and CPU utilization, but increases communication cost.

## Highlights & Insights

- **Decoupling Design Philosophy**: Separating ID alignment from the training task is an elegant system design principle; the Pub/Sub architecture is a natural fit for such loosely coupled scenarios.
- **Complete Theoretical Guarantees**: Convergence proofs and differential privacy compatibility proofs are both provided.
- **End-to-End System Optimization**: The framework forms a complete solution spanning architectural design to automated hyperparameter selection.

## Limitations & Future Work

- The framework supports only two-party learning and has not been extended to multi-party VFL.
- System profiling parameters for dynamic programming require empirical experimentation, limiting automation.
- Experiments are conducted on CPUs rather than GPUs; performance under GPU heterogeneity remains unknown.
- The communication model adopts a simple additive formula; practical pipelining could further reduce latency.

## Related Work & Insights

- Compared to PS architectures in industrial frameworks such as FATE and PaddleFL, PubSub-VFL achieves better decoupling through an additional Pub/Sub layer.
- Relative to AVFL-style asynchronous methods, the hierarchical asynchronous design better balances convergence stability and efficiency.
- Insight: The application of message queue and event-driven architectures in distributed ML systems warrants further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing the Pub/Sub architecture into VFL is a novel idea, though the core techniques (asynchronous training, parameter servers) are combinations of existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five datasets, multiple heterogeneous scenarios, and complete ablation studies are provided, but GPU experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ System design is described clearly with rigorous mathematical modeling.
- **Value**: ⭐⭐⭐⭐ Practically meaningful for improving VFL system efficiency, though the two-party limitation reduces generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[AAAI 2026\] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](../../AAAI2026/ai_safety/matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](../../AAAI2026/ai_safety/healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)

</div>

<!-- RELATED:END -->
