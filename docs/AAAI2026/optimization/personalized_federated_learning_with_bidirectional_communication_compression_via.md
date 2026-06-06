---
title: >-
  [Paper Note] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching
description: >-
  [AAAI 2026][Optimization][Federated Learning] This paper proposes pFed1BS, a framework that achieves extreme bidirectional communication compression (>99% reduction) in federated learning via one-bit random sketching…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Federated Learning"
  - "Communication Efficiency"
  - "Personalization"
  - "one-bit compression"
  - "random sketching"
date: 2026-05-08
content_hash: 5ddb247c41bde799
---

# pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching

**Conference**: AAAI 2026
**arXiv**: [2511.13144](https://arxiv.org/abs/2511.13144)  
**Code**: To be confirmed  
**Area**: Optimization / Federated Learning
**Keywords**: Federated Learning, Communication Efficiency, Personalization, one-bit compression, random sketching

## TL;DR

This paper proposes pFed1BS, a framework that achieves extreme bidirectional communication compression (>99% reduction) in federated learning via one-bit random sketching, while introducing a sign-based regularizer for client model personalization. The framework simultaneously addresses communication bottlenecks and data heterogeneity in non-IID settings.

## Background & Motivation

Federated learning (FL) faces two fundamental challenges: (1) non-IID client data degrades the performance of a single global model; and (2) repeated transmission of high-dimensional model parameters between the server and clients incurs substantial communication overhead. Both issues are especially pronounced in bandwidth-constrained scenarios such as IoT, V2X, and remote sensing.

Existing work follows two independent lines:

- **Personalized Federated Learning (PFL)**: Methods such as pFedMe, Ditto, and FedRep achieve personalization via regularization or architectural decomposition, but still require transmitting full-precision high-dimensional parameters, resulting in high communication costs.
- **Communication-Efficient Federated Learning (CEFL)**: Methods such as OBDA (bidirectional one-bit quantization), OBCSAA (one-bit compressed sensing uplink), and zSignFed (sign-based compression) achieve extreme compression but train only a single global model and cannot handle data heterogeneity.

Through a systematic comparison in Table 1, the authors identify a clear research gap: **no existing framework simultaneously supports extreme bidirectional compression and native personalization**. pFed1BS is designed to fill this gap.

## Method

### Overall Architecture

pFed1BS formulates the federated learning problem as a bilevel optimization:

- **Client level**: Each client $k$ learns a personalized model $\bm{w}_k$ by minimizing an objective comprising a local loss, a sign-based regularization term, and an $\ell_2$ penalty.
- **Server level**: The server aggregates one-bit sketches from all clients to produce a global consensus vector $\bm{v}$.

### Key Designs

**1. One-Bit Random Sketch Communication**

Instead of transmitting high-dimensional model parameters $\bm{w}_k \in \mathbb{R}^n$, each client uploads a low-dimensional one-bit sketch $\text{sign}(\mathbf{\Phi}\bm{w}_k)$, where $\mathbf{\Phi} \in \mathbb{R}^{m \times n}$ ($m \ll n$) is a random projection matrix. The server likewise broadcasts only a one-bit consensus vector $\bm{v} \in \{\pm 1\}^m$, achieving extreme compression in both uplink and downlink directions.

**2. Sign-Based Regularizer**

A regularization term $g(\bm{v}, \mathbf{\Phi}\bm{w}_k) = \frac{1}{2}(\|\mathbf{\Phi}\bm{w}_k\|_1 - \langle\bm{v}, \mathbf{\Phi}\bm{w}_k\rangle)$ is introduced to encourage the sign of the local model projection to align with the global consensus while preserving local data characteristics. Since the $\ell_1$ norm is non-smooth, a differentiable approximation $h_\gamma(\bm{z}) = \frac{1}{\gamma}\sum_i \log(\cosh(\gamma z_i))$ is adopted, yielding the gradient $\nabla\tilde{g} = \mathbf{\Phi}^\top(\tanh(\gamma\mathbf{\Phi}\bm{w}_k) - \bm{v})$. As $\gamma \to \infty$, $\tanh$ approaches the $\text{sign}$ function.

**3. Fast Hadamard Transform for Efficient Projection**

Naive dense random matrix projection $\mathbf{\Phi}\bm{w}$ has complexity $\mathcal{O}(mn)$, which is infeasible for large models. The framework instead employs the Subsampled Randomized Hadamard Transform (SRHT): $\mathbf{\Phi}\bm{w} = \bm{S}'\bm{H}\bm{D}\tilde{\bm{w}}$, where $\bm{D}$ is a random sign-flipping matrix, $\bm{H}$ is the Walsh–Hadamard matrix, and $\bm{S}'$ is a subsampling matrix. This reduces computational complexity from $\mathcal{O}(mn)$ to $\mathcal{O}(n\log n)$ and eliminates the need to store a dense matrix.

**4. Server Aggregation**

Upon receiving client sketches, the server solves the discrete optimization $\min_{\bm{v}\in\{\pm 1\}^m} \sum_k p_k \tilde{g}(\bm{v}, \bm{z}_k)$, whose closed-form solution is weighted majority voting: $\bm{v}^* = \text{sign}(\sum_k p_k \bm{z}_k)$. This guarantees that the aggregation step is provably optimal rather than heuristic.

### Algorithm

Each communication round proceeds as follows: clients receive the global consensus $\bm{v}^t$ → perform $R$ steps of local SGD with the sign-based regularizer → compute and upload one-bit sketches → the server applies weighted majority voting to produce a new consensus $\bm{v}^{t+1}$ → broadcast.

### Loss & Training

Under standard assumptions ($L$-smoothness, bounded variance, etc.), pFed1BS is proven to converge to a stationary neighborhood of the global potential function at rate $\mathcal{O}(1/(RT))$. The convergence error is governed by three terms: stochastic noise $\mathcal{O}(\eta L_F \sigma^2)$, communication error $\mathcal{O}(\Delta_{\max}/(\eta R))$, and client sampling error $\mathcal{O}(\lambda E_S/(\eta R))$. The regularization coefficient must satisfy $\lambda = \mathcal{O}(1/n)$.

## Key Experimental Results

### Experimental Setup

- **Datasets**: MNIST, FMNIST, CIFAR-10, CIFAR-100, SVHN
- **Models**: Two-layer MLP for MNIST/FMNIST; VGG architecture for the remaining datasets
- **Non-IID Partition**: 20 clients with label-based data allocation
- **Baselines**: FedAvg, OBDA, OBCSAA, zSignFed, EDEN, FedBAT
- **Key Hyperparameters**: $\lambda=0.0005$, $\mu=0.00001$, $\gamma=10000$, compression ratio $m/n=0.1$
- **Hardware**: NVIDIA RTX 3090 Ti; results averaged over 10 independent runs

### Main Results (Table 2)

| Method | MNIST Acc(%) | CIFAR-10 Acc(%) | CIFAR-100 Acc(%) | Comm. Reduction |
|--------|-------------|-----------------|------------------|-----------------|
| FedAvg | 97.21 | 87.78 | 59.60 | - |
| OBDA | 92.54 | 73.26 | 42.47 | ↓96.88% |
| OBCSAA | 92.20 | 83.57 | 48.99 | ↓49.84% |
| zSignFed | 94.83 | 67.60 | 40.17 | ↓48.45% |
| EDEN | 96.50 | 84.91 | 47.55 | ↓60.88% |
| FedBAT | 96.42 | 81.20 | 46.89 | ↓61.75% |
| **pFed1BS** | **97.83** | **85.21** | **52.88** | **↓99.69%** |

pFed1BS achieves the highest or near-highest accuracy across all datasets while reducing communication costs by over 99%. On CIFAR-10, pFed1BS attains 85.21% accuracy with only 0.13 MB per round, whereas OBDA requires 1.34 MB yet achieves only 73.26%.

### Ablation Study

Under non-IID conditions on MNIST (Figures 3–4), the convergence curves of pFed1BS show faster convergence, higher final accuracy (97.83%), and a more rapid and stable decrease in training loss compared to all baselines. On more challenging datasets such as CIFAR-100, competing one-bit methods collapse in performance (e.g., zSignFed reaches only 40.17%), while pFed1BS maintains 52.88%, demonstrating that the personalization mechanism is critical for the viability of extreme compression.

## Highlights & Insights

- **First unified framework**: The first work to jointly formulate one-bit bidirectional communication and personalized federated learning as a rigorous joint optimization problem.
- **Extreme compression ratio**: Communication costs are reduced by over 99% while maintaining or exceeding the accuracy of full-precision methods.
- **Computational efficiency**: The Fast Hadamard Transform reduces projection complexity from $\mathcal{O}(mn)$ to $\mathcal{O}(n\log n)$.
- **Theoretical completeness**: A full convergence analysis is provided that accounts for the interactions among personalization, local updates, one-bit sketch error, and client sampling.
- **Closed-form optimal aggregation**: Server-side aggregation is not heuristic but is a provably optimal solution with rigorous guarantees.

## Limitations & Future Work

1. **Limited model scale**: Experiments are conducted only on MLP and VGG architectures; large-scale modern architectures such as ResNet and Transformer are not evaluated.
2. **Fixed compression ratio**: The compression ratio $m/n=0.1$ is fixed, with no adaptive adjustment mechanism.
3. **Single non-IID type**: Only label-based non-IID partitioning is tested; more complex scenarios such as feature distribution shift and quantity imbalance are not explored.
4. **Insufficient hyperparameter sensitivity analysis**: The interaction effects and robustness of the three hyperparameters $\lambda$, $\gamma$, and $\mu$ are not thoroughly investigated.
5. **Limited PFL baseline comparisons**: Direct comparisons with classical PFL methods such as pFedMe, Ditto, and Per-FedAvg are absent.

## Related Work & Insights

- **Personalized Federated Learning**: pFedMe (regularization-based), Ditto (dual-model), FedRep (shared representation + personalized head), DisPFL (personalized sparse masks).
- **Communication-Efficient Federated Learning**: Top-k sparsification, mixed-precision quantization (Chen & Vikalo 2024), compressed sensing (Li et al. 2021), OBDA/OBCSAA/zSignFed (one-bit methods).
- **Random Projection**: Subsampled Randomized Hadamard Transform (SRHT) for efficient dimensionality reduction.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

**Overall Rating: ⭐⭐⭐⭐** (4/5)

The paper fills a clear gap between communication compression and personalized federated learning with solid theoretical foundations, though the experimental scale is modest and direct comparisons with classical PFL methods are lacking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](../../NeurIPS2025/optimization/multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](../../NeurIPS2025/optimization/personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)
- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](../../NeurIPS2025/optimization/layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[ICML 2026\] Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation](../../ICML2026/optimization/delayed_momentum_aggregation_communication-efficient_byzantine-robust_federated_.md)
- [\[AAAI 2026\] PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing](peoat_personalization-guided_evolutionary_question_assembly_for_one-shot_adaptiv.md)

</div>

<!-- RELATED:END -->
