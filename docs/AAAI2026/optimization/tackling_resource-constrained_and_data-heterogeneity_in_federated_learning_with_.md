---
title: >-
  [Paper Note] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack
description: >-
  [AAAI 2026][Optimization][Federated Learning] This paper proposes FedCSPACK, a personalized federated learning method based on cosine-similarity-guided sparse parameter packing and double-weight aggregation. By performin…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Federated Learning"
  - "Data Heterogeneity"
  - "Resource Constraints"
  - "Sparse Communication"
  - "Personalized Federated Learning"
date: 2026-05-08
content_hash: 12e422ef4b699dd4
---

# Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack

**Conference**: AAAI 2026
**arXiv**: [2601.01840](https://arxiv.org/abs/2601.01840)
**Code**: [https://github.com/NigeloYang/FedCSPACK](https://github.com/NigeloYang/FedCSPACK)
**Area**: Optimization
**Keywords**: Federated Learning, Data Heterogeneity, Resource Constraints, Sparse Communication, Personalized Federated Learning

## TL;DR

This paper proposes FedCSPACK, a personalized federated learning method based on cosine-similarity-guided sparse parameter packing and double-weight aggregation. By performing parameter selection and sharing at the pack level, FedCSPACK simultaneously addresses data heterogeneity and client resource constraints, achieving 2–5× faster training, up to 96% reduction in communication overhead, and a 3.34% improvement in model accuracy.

## Background & Motivation

### The Dual Challenges of Federated Learning

Federated learning (FL) enables collaborative model training without exchanging raw data, yet faces two intertwined core challenges:

**Challenge 1: Data Heterogeneity (Non-IID)**
- Edge devices across different geographic regions generate data with significantly divergent distributions.
- A single global model struggles to adapt to the local data of all clients.
- This leads to slow convergence and poor inference performance.

**Challenge 2: System Resource Heterogeneity**
- Client devices differ in processor capability, memory, and bandwidth.
- Resource-constrained clients cannot keep pace with cooperative training on complex global models.
- This causes communication bottlenecks, computational delays, and participation imbalance.

### Limitations of Prior Work

Existing methods typically address only one of these challenges:

- **Addressing data heterogeneity**: FedProx (L2 regularization), MOON (contrastive learning), FedNTD (knowledge distillation), etc. — all neglect resource constraints.
- **Addressing resource constraints**: FedSPU (parameter sparsification), model splitting methods — none account for data heterogeneity.

**Key Insight**: In real-world deployments, data heterogeneity and resource constraints are not isolated phenomena but jointly acting core challenges that must be addressed simultaneously.

## Method

### Overall Architecture

FedCSPACK operates in four steps:

1. **Server**: Aggregates the global model $W^t$ and global mask $M^t$, then broadcasts them to all clients.
2. **Client training**: Updates the local model on local data.
3. **Parameter packing and selection**: Flattens model parameters into packs and uses cosine similarity to select the Top-K most contributive packs for sharing, while generating a mask with double weights.
4. **Server aggregation**: Performs weighted aggregation using the double-weight mask.

### Key Designs

#### 1. **Cosine Parameter Packing (Top-K Cosine-based)**: Resolving the Communication Bottleneck

**Problem**: Frequent full-model transmission imposes a heavy communication burden on resource-constrained clients. Traditional Top-K sparsification methods struggle to identify an appropriate parameter subset in dynamic training.

**Solution**: Parameter selection is performed at the *pack* level rather than the individual parameter level.

**Procedure**:
1. Flatten the local model and the global model into one-dimensional vectors $FW_i^t, FW^t$.
2. Compute the overall cosine similarity threshold:

$$\theta_a^t = \text{CosSim}(FW_i^t, FW^t) = \frac{FW_i^t \cdot FW^t}{\|FW_i^t\| \|FW^t\|}$$

3. Segment the flattened vector into parameter packs $PW_{i,j}^t$ of size $PACK$.
4. Compute the similarity $\theta_{i,j}^t$ for each pack.
5. Apply Top-K selection to identify the K packs satisfying $\theta_{i,j}^t < \theta_a^t$ as the shared parameter packs.

**Design Motivation**:
- Low cosine similarity indicates high inference loss; improving these packs is likely to yield better model performance.
- Pack-level operations are more computationally efficient than parameter-level ones.
- Unselected packs are retained as the client's unique local knowledge, preserving personalization.

#### 2. **Mask Double-Weight Aggregation**: Mitigating Misaligned Aggregation

**Problem**: Sparse parameter packs may cause misalignment or aggregation errors at the server, degrading global model performance. Moreover, cosine similarity alone cannot capture distance differences or magnitude shifts in parameter values.

**Solution**: A double-weight mask combining a directional weight (cosine similarity) and a distributional distance weight (KL divergence).

**Directional weight**: Provided by the cosine similarity $\theta_{i,k}^t$, reflecting the consistency of parameter update directions.

**Distributional distance weight**: Computed via KL divergence between parameter packs:

$$\beta_{i,j}^t = \sum PW_{i,j}^t \log \frac{PW_{i,j}^t}{PW_j^t}$$

**Final mask**:

$$M_{i,j}^t = \begin{cases} \theta_{i,k}^t + \beta_{i,k}^t & \text{shared pack positions} \\ 0 & \text{non-shared positions} \end{cases}$$

**Aggregation formula**:

$$PW_{i,j}^t = \begin{cases} \sum_{i=1}^{S_t} \frac{M_{i,j}^t}{M_j^{t+1}} PW_{i,k}^t & M_{i,j}^t \neq 0 \\ 0 & \text{otherwise} \end{cases}$$

**Design Motivation**:
- KL divergence supplements magnitude information that cosine similarity cannot capture.
- The double-weight mechanism enables the server to more comprehensively assess each pack's contribution.
- Directional alignment benefits are preserved while effectively mitigating the impact of distributional discrepancies.

#### 3. **Personalized Knowledge Preservation**: Local Characteristics in Unshared Packs

Clients share only K parameter packs; the remaining $PW_{i,j \setminus k}^t$ are retained as client-specific features, shielding them from interference by other clients' heterogeneous data. When a client receives a new global model, only positions flagged by the mask are updated by the global model, thereby preserving local personalized knowledge.

### Loss & Training

- Standard cross-entropy loss is used for local training.
- An SGD optimizer is used for local model updates: $W_i^t \leftarrow W_i^t - \eta \nabla f_i(W_i^t)$.
- Each round randomly samples a subset $S^t$ from $N$ clients to participate in training.
- Training proceeds for $T$ global epochs and $E$ local epochs.

## Key Experimental Results

### Main Results

**Top-1 accuracy on four datasets (Table 1, Dirichlet sampling)**:

| Method | FMNIST Dir(0.3) | CIFAR-10 Dir(0.3) | CIFAR-100 Dir(0.3) | EMNIST Dir(0.6) |
|--------|----------------|-------------------|-------------------|----------------|
| FedAvg | 84.39 | 69.71 | 39.15 | 84.03 |
| FedProx | 84.39 | 69.58 | 38.48 | 84.12 |
| MOON | 85.44 | 70.03 | 38.43 | 84.08 |
| FedNTD | 84.35 | 70.32 | 39.44 | 84.49 |
| FedSPU | 85.29 | 67.38 | 37.81 | 84.22 |
| **FedCSPACK** | **88.13** | **73.23** | **41.60** | **86.26** |

FedCSPACK achieves 78.71% on CIFAR-10 Dir(1.0) and 43.20% on CIFAR-100 Dir(1.0), both state-of-the-art results.

**Resource consumption (Table 2, T=100, Dir(0.3))**:

| Dataset/Model | Metric | FedAvg | FedSPU | FedCSPACK |
|--------------|--------|--------|--------|-----------|
| EMNIST/CNN | Communication (GB) | 18.18 | 4.29 | **0.73** |
| EMNIST/CNN | Time (h) | 11.16 | 13.92 | **11.32** |
| CIFAR-100/ResNet18 | Communication (GB) | 251.00 | 49.22 | **9.24** |
| CIFAR-100/ResNet18 | Time (h) | 0.81 | 1.01 | **0.88** |

Communication compression: 96.0% on EMNIST (18.18 GB → 0.73 GB) and 27× compression on CIFAR-100.

### Ablation Study

**Effect of double weights (Table 3, CIFAR-10 Dir(0.5))**:

| Weight Type | Round 10 | Round 50 | Round 100 |
|------------|---------|---------|----------|
| CS only (cosine) | 0.33 | 0.65 | 0.74 |
| KL only | 0.30 | 0.68 | 0.69 |
| **Double weight** | **0.33** | **0.71** | **0.79** |

**Effect of pack size**: Model accuracy remains largely stable as pack size increases, while training time gradually decreases, with more pronounced improvements in large-scale heterogeneous settings.

### Key Findings

1. **Extremely high communication efficiency**: 96% communication compression on EMNIST and 27× compression on CIFAR-100.
2. **Optimal accuracy–efficiency trade-off**: Communication overhead is drastically reduced while accuracy surpasses all competing methods.
3. **Double weights outperform single weights**: The combination of directional and distance weights is more effective than either alone.
4. **Robustness under low participation rates**: FedCSPACK maintains stable performance even when client participation rates are very low.
5. **Superior global model generalization**: Generalization performance on the worst-performing client (Client 3) is 20% higher than the best-performing prior SOTA.

## Highlights & Insights

1. **First pack-level personalized federated learning**: Elevates parameter packing to a first-class operation, moving beyond simple parameter-level sparsification.
2. **Simultaneous resolution of dual heterogeneity**: Balances data heterogeneity and resource constraints within a single framework, filling the gap left by prior methods.
3. **Robustness to pack size**: Model performance is insensitive to pack size, providing flexibility for practical deployment.
4. **Introduction of KL divergence as a distributional distance weight**: Supplements magnitude information that cosine similarity alone cannot capture.
5. **Negligible training time overhead**: Despite the additional packing and weight computation, wall-clock training time remains essentially unchanged.

## Limitations & Future Work

1. **Evaluated only on image classification**: More complex vision tasks such as object detection and semantic segmentation are not explored.
2. **Limited model architectures**: Only CNN and ResNet-18 are used; performance on large models such as ViT remains unknown.
3. **Fixed pack size**: The pack size is currently static; adaptive selection could yield further improvements.
4. **Theoretical justification for KL divergence on parameter packs**: Treating parameter values as probability distributions for KL divergence computation requires more rigorous theoretical analysis.
5. **Security considerations are absent**: Robustness in the presence of malicious clients is not discussed.

## Related Work & Insights

- **FedAvg** (McMahan et al. 2017): Standard federated averaging baseline.
- **FedProx** (Li et al. 2020): Constrains drift between local and global models via L2 regularization.
- **FedSPU** (Niu et al. 2025): Top-K model sparsification method focused solely on communication efficiency.
- **FedNTD** (Lee et al. 2022): Mitigates catastrophic forgetting through knowledge distillation.

**Insight**: In distributed systems, *selective sharing* is often more efficient than *full sharing*. Pack-level granularity control is more engineering-feasible than parameter-level control, and double-weight aggregation effectively compensates for information loss.

## Rating

- Novelty: ⭐⭐⭐⭐ (Pack-level operations and double-weight aggregation represent meaningful contributions, though the core technical components are relatively conventional.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 datasets, 10 baselines, multiple heterogeneity settings, comprehensive ablations.)
- Writing Quality: ⭐⭐⭐ (Overall structure is clear, but notation is occasionally inconsistent across sections.)
- Value: ⭐⭐⭐⭐ (Practically significant for deploying federated learning in resource-constrained environments.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](personalized_federated_learning_with_bidirectional_communication_compression_via.md)

</div>

<!-- RELATED:END -->
