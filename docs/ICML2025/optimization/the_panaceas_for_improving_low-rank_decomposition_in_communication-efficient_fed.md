---
title: >-
  [Paper Note] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning
description: >-
  [ICML2025][Optimization][federated learning] Addressing three core questions of low-rank decomposition in federated learning (what to decompose, how to decompose, and how to aggregate), this paper proposes three complementary technologies—MUD (Model Update Decomposition), BKD (Block-wise Kronecker Decomposition), and AAD (Aggregation-Aware Decomposition)—to achieve faster convergence and higher accuracy while maintaining low communication overhead.
tags:
  - "ICML2025"
  - "Optimization"
  - "federated learning"
  - "low-rank decomposition"
  - "communication efficiency"
  - "Kronecker decomposition"
  - "model aggregation"
date: 2026-05-08
content_hash: 913ed4bc6b38d5e7
---

# The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning

**Conference**: ICML2025  
**arXiv**: [2505.23176](https://arxiv.org/abs/2505.23176)  
**Authors**: Shiwei Li, Xiandi Luo, Haozhao Wang, Xing Tang, Shijie Xu, Weihong Luo, Yuhua Li, Xiuqiang He, Ruixuan Li  
**Code**: [GitHub](https://github.com/Leopold1423/fedmud-icml25)  
**Area**: Optimization  
**Keywords**: federated learning, low-rank decomposition, communication efficiency, Kronecker decomposition, model aggregation

## TL;DR

Addressing three core questions of low-rank decomposition in federated learning (what to decompose, how to decompose, and how to aggregate), this paper proposes three complementary technologies—MUD (Model Update Decomposition), BKD (Block-wise Kronecker Decomposition), and AAD (Aggregation-Aware Decomposition)—to achieve faster convergence and higher accuracy while maintaining low communication overhead.

## Background & Motivation

Federated Learning (FL) requires multiple communication rounds to train a global model, with communication overhead being the primary bottleneck. Low-rank decomposition is an effective parameter compression technique widely used for bidirectional communication compression in FL (e.g., FedHM, FedLMT, FedPara). However, existing methods face three key challenges:

**What to decompose**: Existing methods (such as FedLMT) directly apply low-rank decomposition to the entire model parameters. However, the model parameters themselves might be full-rank, and forcing a rank reduction leads to severe information loss. In reality, what FL transmits during communication is the model update rather than the complete parameters.

**How to decompose**: The standard low-rank decomposition $W = UV^\top$ has a rank upper bound of $\text{rank}(W) \leq r \ll \min\{m, n\}$ and a parameter size of $(m+n)r$, which severely limits its expressiveness. FedPara leverages Hadamard products to increase rank, but it doubles the parameter size.

**How to aggregate**: If the sub-matrices $\bar{U}$ and $\bar{V}$ are aggregated directly, the reconstructed matrix $\bar{U}\bar{V}^\top$ deviates implicitly from the true aggregated matrix $\bar{W}$. FedHM introduces explicit errors via SVD re-decomposition, whereas FedLMT directly aggregates sub-matrices, introducing implicit aggregation errors.

These three problems are independent, and this paper presents corresponding solutions for each.

## Method

### 1. Model Update Decomposition (MUD) — What to decompose

The core idea is similar to LoRA: freeze the original model parameters $W_0$ and only learn the low-rank model updates $\Delta W$.

- In each communication round of local training, clients freeze the global model parameters $W_0$ and learn the low-rank sub-matrices $U, V$ as model updates.
- The model parameters are represented as $W = W_0 + \Delta W$, where $\Delta W = UV^\top$.
- During communication, only the low-rank sub-matrices $U, V$ are transmitted. The server aggregates them and updates the global model.
- Compared to direct decomposition of the entire parameter matrix, model updates typically exhibit a much lower rank, making the low-rank approximation more accurate.
- At the start of each round, $U, V$ are re-initialized ($V$ is initialized to zero) to ensure that $\Delta W$ starts at zero.

**Theoretical Analysis**: The authors provide a convergence proof for MUD. Under standard assumptions (L-smoothness, bounded variance, bounded heterogeneity), the convergence upper bound of MUD is tighter than that of FedLMT because the compression error of MUD only depends on the model updates, whereas the error of FedLMT depends on the entire model parameters.

### 2. Block-wise Kronecker Decomposition (BKD) — How to decompose

BKD replaces standard matrix multiplication with block-wise Kronecker products to improve the rank upper bound of the reconstructed matrix.

- Split the weight matrix $W \in \mathbb{R}^{m \times n}$ into $k$ blocks, with each block having a size of $\frac{m}{k} \times \frac{n}{k}$.
- Each block is decomposed using a Kronecker product: $W_j = A_j \otimes B_j$, where $A_j \in \mathbb{R}^{p \times q}$ and $B_j \in \mathbb{R}^{\frac{m}{kp} \times \frac{n}{kq}}$.
- Rank upper bound: $\text{rank}(W_j) = \text{rank}(A_j) \cdot \text{rank}(B_j) \leq pq \cdot \frac{m}{kp} \cdot \frac{n}{kq} = \frac{mn}{k^2}$.
- The total parameter count is $k(pq + \frac{mn}{k^2 pq})$. By choosing $pq = \frac{\sqrt{mn}}{k}$, the optimal parameter complexity of $2\sqrt{mn}$ is achieved.
- When $k = 1$, the rank upper bound for a single-block Kronecker decomposition is $\min\{m, n\}$, which is full rank and significantly superior to standard low-rank decomposition.

### 3. Aggregation-Aware Decomposition (AAD) — How to aggregate

This addresses the implicit error issue caused by the direct aggregation of sub-matrices.

**Problem Analysis**: Suppose $N$ clients have $U_i, V_i$ respectively. Direct aggregation yields $\bar{U} = \frac{1}{N}\sum U_i$ and $\bar{V} = \frac{1}{N}\sum V_i$, but $\bar{U}\bar{V}^\top \neq \frac{1}{N}\sum U_i V_i^\top = \bar{W}$.

**Solution**: Keep one side of the product fixed to the parameter $\tilde{V}$ shared by all clients (the aggregation result from the previous round) and only train $U_i$:

- $\Delta W_i = U_i \tilde{V}^\top$ (only $U_i$ is trainable; $\tilde{V}$ is fixed and shared).
- After aggregation, $\bar{U}\tilde{V}^\top = \frac{1}{N}\sum U_i \tilde{V}^\top = \overline{\Delta W}$, which eliminates implicit error.
- To maintain bidirectional training flexibility, alternating training is used: odd rounds train $U$ (fixing $V$), and even rounds train $V$ (fixing $U$).
- This strategy is also applicable to the aggregation of Kronecker products.

### Combining the Three Technologies

MUD, BKD, and AAD can be simultaneously applied:

- MUD determines "what" to decompose: model updates rather than entire parameters.
- BKD determines "how" to decompose: using block-wise Kronecker product decomposition.
- AAD determines "how" to aggregate: alternating training to eliminate aggregation errors.

## Key Experimental Results

### Experimental Setup

- **Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet, CINIC-10
- **Models**: ResNet-18, VGG-11
- **Non-IID Setting**: Dirichlet distribution with $\alpha = 0.3$ to split the data
- **Number of Clients**: $N = 10$, full participation in each round
- **Local Training**: SGD, learning rate 0.01, 5 epochs/round
- **Communication Rounds**: 200 rounds
- **Baselines**: FedAvg (full communication), FedHM, FedLMT, FedPara, FedKSeed, FedRolex

### Table 1: Main Results — Test Accuracy (%) of Different Methods on Various Datasets

| Method | Compression Rate | CIFAR-10 | CIFAR-100 | Tiny-ImageNet | CINIC-10 |
|------|--------|----------|-----------|---------------|----------|
| FedAvg | 1× | 87.72 | 58.64 | 40.80 | 79.42 |
| FedHM | ~10× | 79.45 | 44.84 | 27.74 | 72.07 |
| FedLMT | ~10× | 83.03 | 47.84 | 27.60 | 74.51 |
| FedPara | ~5× | 84.60 | 53.94 | 32.26 | 76.04 |
| **Ours (MUD+BKD+AAD)** | **~10×** | **87.33** | **57.00** | **38.42** | **78.34** |

Under approximately 10× compression rate, the proposed method achieves accuracy gains of **+9.16%** on CIFAR-100 and **+10.82%** on Tiny-ImageNet compared to FedLMT, approaching the performance of uncompressed FedAvg.

### Table 2: Ablation Study — Independent Contributions of Each Technology (CIFAR-100, ResNet-18)

| Configuration | MUD | BKD | AAD | Accuracy (%) |
|------|-----|-----|-----|-----------|
| Baseline (FedLMT) | ✗ | ✗ | ✗ | 47.84 |
| +MUD | ✓ | ✗ | ✗ | 52.31 |
| +BKD | ✗ | ✓ | ✗ | 51.67 |
| +AAD | ✗ | ✗ | ✓ | 49.12 |
| +MUD+BKD | ✓ | ✓ | ✗ | 54.89 |
| +MUD+AAD | ✓ | ✗ | ✓ | 53.54 |
| **+MUD+BKD+AAD** | ✓ | ✓ | ✓ | **57.00** |

The three technologies are complementary, with each making an independent positive contribution, and their combination yielding the best performance. MUD makes the largest contribution (+4.47%), followed by BKD (+3.83%), while AAD provides an additional gain (+1.28%).

### Table 3: Performance Under Different Compression Rates (CIFAR-100)

| Compression Rate | FedLMT | FedPara | Ours |
|--------|--------|---------|------|
| 5× | 52.40 | 53.94 | 57.85 |
| 10× | 47.84 | — | 57.00 |
| 20× | 42.31 | — | 53.12 |

The proposed method consistently outperforms baseline models under different compression rates, and maintains favorable performance even at a high compression rate of 20×.

## Highlights & Insights

1. **Clear Problem Decomposition**: The challenges of low-rank decomposition in federated learning are decoupled into three orthogonal questions (what to decompose / how to decompose / how to aggregate), with tailored independent solutions for each, showing clean and structured logic.
2. **Clever Connection Between MUD and LoRA**: Bringing the "freeze + low-rank update" paradigm of LoRA to FL communication compression is highly intuitive, making low-rank approximations more reasonable (as updates are naturally low-rank compared to parameters).
3. **AAD Uncovers Overloaded Implicit Aggregation Errors**: This work exposes implicit aggregation errors during direct sub-matrix aggregation, an issue neglected by previous literature, and introduces a simple yet effective alternating training framework to resolve it.
4. **BKD Achieves Full-Rank Bound via Kronecker Products**: Under the same parameter constraints, BKD yields a significantly higher rank upper bound compared to standard decompositions.
5. **Up to 12% Accuracy Improvement**: Across multiple datasets, the proposed method dramatically outperforms baselines like FedHM and FedLMT, closely matching uncompressed FedAvg.

## Limitations & Future Work

1. **Limited Experimental Scale**: Validations were only conducted under small-scale scenarios with 10 clients. Large-scale (100+ clients) or partial client participation settings were not tested.
2. **Model Restrictions**: Experiments only utilized traditional CNN architectures like ResNet-18 and VGG-11, leaving the performance on Transformer architectures or LLMs unverified.
3. **Alternating Training Cost of AAD**: Alternating between fixing U and V may potentially cause training instability or decelerate convergence, as only half of the parameters are updated in each round.
4. **BKD Block Restrictions**: The matrix dimensions must be divisible by $k$, necessitating extra padding for non-standard layer sizes.
5. **Lack of Heterogeneous Device Scenarios**: Scenarios where different clients utilize different compression rates were not discussed.

## Related Work & Insights

- **Federated Learning Communication Compression**: FedAvg (McMahan et al., 2017), gradient quantization (Reisizadeh et al., 2020), gradient sparsification (Li et al., 2024).
- **Low-Rank Decomposition Methods**: FedHM (Yao et al., 2021) using truncated SVD, FedLMT (Liu et al., 2024) training pre-decomposed models directly, FedPara (Hyeon-Woo et al., 2022) applying Hadamard products for rank expansion.
- **LoRA and Parameter-Efficient Fine-Tuning**: LoRA (Hu et al., 2022) freezes pre-trained weights and learns low-rank updates. MUD in this paper extends this paradigm to FL.
- **Kronecker Decomposition**: Application of Kronecker products to model compression (Thakker et al., 2019).

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | The combination of three orthogonal techniques is novel. Specifically, AAD exposes the previously neglected aggregation bias. |
| Theoretical Rigor | ⭐⭐⭐⭐ | Provides convergence analysis for MUD, proving a tighter bound and faster convergence than FedLMT. |
| Experimental Thoroughness | ⭐⭐⭐ | The experiments are comprehensive but somewhat limited in scale, lacking validation on large models and large client counts. |
| Practicality | ⭐⭐⭐⭐ | The method is clean and easy to implement. The three techniques can be combined flexibly, and the code has been open-sourced. |
| Writing Quality | ⭐⭐⭐⭐ | Clear problem decomposition, intuitive diagrams, and cohesive logical flow. |
| Overall | ⭐⭐⭐⭐ | A solid contribution to the communication-efficient federated learning domain, featuring clean methodology and convincing experiments. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](../../NeurIPS2025/optimization/layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](../../NeurIPS2025/optimization/mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)
- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](../../NeurIPS2025/optimization/multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[ICML 2025\] FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging](fedswa_improving_generalization_in_federated_learning_with_highly_heterogeneous_.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)

</div>

<!-- RELATED:END -->
