---
title: >-
  [Paper Note] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients
description: >-
  [NeurIPS 2025][AI Safety][federated learning] This paper proposes DictPFL, a framework that decomposes model weights into a static dictionary and a trainable lookup table…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "federated learning"
  - "Homomorphic Encryption"
  - "Privacy-Preserving"
  - "Gradient Pruning"
  - "Dictionary Decomposition"
date: 2026-05-08
content_hash: 6bec89e9d057d7bd
---

# DictPFL: Efficient and Private Federated Learning on Encrypted Gradients

**Conference**: NeurIPS 2025
**arXiv**: [2510.21086](https://arxiv.org/abs/2510.21086)  
**Code**: [UCF-ML-Research/DictPFL](https://github.com/UCF-ML-Research/DictPFL)  
**Area**: AI Security
**Keywords**: federated learning, Homomorphic Encryption, Privacy-Preserving, Gradient Pruning, Dictionary Decomposition

## TL;DR

This paper proposes DictPFL, a framework that decomposes model weights into a static dictionary and a trainable lookup table, and combines this decomposition with encryption-aware pruning. DictPFL achieves full gradient protection via homomorphic encryption in federated learning while reducing communication overhead by 402–748× and training time by 28–65×, keeping total runtime within 2× of plaintext FL.

## Background & Motivation

Federated learning (FL) enables multiple parties to collaboratively train models without sharing raw data, yet shared gradients remain vulnerable to privacy leakage — gradient inversion attacks can reconstruct clients' original training data from shared gradients.

Homomorphic encryption (HE) is an ideal solution for protecting gradient privacy: clients encrypt gradients before uploading, and the server aggregates directly on ciphertexts without decryption. However, HE introduces prohibitive overhead:

- **Ciphertext expansion**: communication cost increases by 1–3 orders of magnitude
- **Computational cost**: encryption, decryption, and homomorphic aggregation are highly time-consuming
- In ViT training, HE-related operations (encryption, decryption, aggregation, communication) dominate total training time

The existing method FedML-HE adopts a selective encryption strategy, encrypting only the most sensitive 10% of gradients and transmitting the rest in plaintext. While this reduces overhead, unencrypted gradients still expose private information — experiments show that when 30% of gradients are unencrypted, an adversary can recover images with up to 23% similarity to the originals.

## Core Problem

How to **simultaneously achieve** in HE-based FL:

1. **Complete privacy protection**: all transmitted gradients must be fully encrypted, with no plaintext gradients exposed
2. **High efficiency**: reduce the communication and computation overhead of HE to near-plaintext FL levels

These two objectives were previously considered contradictory — full encryption implies high overhead, while low overhead requires sacrificing some privacy.

## Method

DictPFL consists of two core modules:

### 1. Decompose-for-Partial-Encrypt (DePE) — Dictionary Decomposition

**Core Idea**: Decompose a weight matrix $W \in \mathbb{R}^{n \times m}$ into a static dictionary $D \in \mathbb{R}^{n \times r}$ and a trainable lookup table $T \in \mathbb{R}^{r \times m}$, where $r \ll \min(n, m)$.

**Procedure**:

- Apply truncated SVD to the initial weights $W_0$: $W_0 \approx U_r \Sigma_r V_r^\top$
- Set dictionary $D = U_r \Sigma_r$ (frozen, identical across all clients, never transmitted)
- Initialize the lookup table as a zero matrix; the effective weights are constructed as $W = W_0 + D \cdot T$
- Only the gradients of $T$ are encrypted and transmitted for aggregation

**Key Design**: Retaining the original $W_0$ and initializing $T$ to zero (rather than directly using $V_r^\top$) avoids information loss from SVD truncation. At $r=4$, the number of trainable parameters is substantially reduced, directly decreasing the number of ciphertexts requiring encryption.

### 2. Prune-for-Minimum-Encrypt (PrME) — Encryption-Aware Pruning

PrME further reduces the number of parameters that need to be encrypted and transmitted, building upon DePE.

**Unique Challenges of Pruning under HE**:

- Independent pruning by each client leads to inconsistent sparsity patterns, whereas HE's SIMD batching mechanism requires aligned ciphertext slots across clients
- Non-linear comparison operations on encrypted indices cannot be performed on the server

**Temporal Inactivity Pruning (TIP)**:

- Uses the global gradient history from the preceding $\tau$ rounds (consistent across all clients) as a shared pruning criterion
- A parameter is pruned only when its gradient magnitude falls in the bottom $s\%$ for $\tau$ consecutive rounds
- Pruning mask: $M_{i,t}=0$ (pruned) when $\sum_{k=1}^{\tau} \mathbf{1}(|\delta w_{i,t-k}| < \theta_{s,t-k}) = \tau$

**Holistic Reactivation Correction (HRC)**:

- Addresses the permanent inactivation of pruned parameters in TIP
- Assigns a dynamic reactivation probability $p_i$ to each pruned parameter
- After reactivation, if the accumulated global gradient remains small, $p_i$ is decreased (multiplied by decay factor $\beta$); otherwise $p_i$ is increased
- Client-side reactivation consistency is ensured via a shared random seed
- Pruning masks are never sent to the server, preventing privacy inference through plaintext masks

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $r$ | 4 | Dictionary size |
| $s\%$ | 70% | Pruning ratio |
| $\tau$ | 3 | Pruning patience window |
| $\beta$ | 0.2 | Reactivation probability decay factor |

## Key Experimental Results

### Efficiency vs. Full Encryption (FedHE-Full)

- Communication overhead reduced by **402–748×**
- Training speed improved by **28–65×**
- Total runtime within **< 2×** of plaintext FL

### Efficiency vs. Selective Encryption (FedML-HE, 10% encrypted)

- Communication overhead reduced by **51–155×**
- Training speed improved by **4–19×**
- DictPFL simultaneously provides **complete** privacy protection (FedML-HE retains privacy leakage risk)

### Accuracy (ViT, 3-client homogeneous setting)

| Method | CIFAR-10 | GTSRB | Diabetic Retinopathy |
|--------|----------|-------|---------------------|
| FedHE-Full | baseline | baseline | 82.74% |
| FedHE-Top2 | — | 58.9% | — |
| DictPFL ($r$=4) | on par | 95.27% | 81.99% |

### Privacy Protection

- FedML-HE (30% unencrypted): adversary recovers images with up to 23% similarity
- DictPFL: all gradients encrypted, resistant to any gradient inversion attack

### Ablation Study

- **Dictionary size $r$**: $r=4$ achieves accuracy close to the full model; $r=2$ shows significant degradation
- **Pruning ratio**: 70% pruning with HRC reactivation matches the accuracy of 20% pruning while retaining the communication efficiency of 70% pruning
- **Pruning patience $\tau$**: $\tau=3$ sufficiently balances accuracy and communication efficiency

## Highlights & Insights

1. **First demonstration of practical HE-based FL**: runtime within 2× of plaintext FL, previously considered infeasible
2. **Zero-leakage design**: all transmitted gradients are encrypted; untransmitted parameters (the dictionary) remain local; pruning masks are never sent to the server
3. **Elegant DePE design**: retaining $W_0$ and zero-initializing $T$ avoids SVD truncation information loss; the dictionary $D$ is naturally consistent across clients without any communication
4. **HRC reactivation mechanism**: gracefully resolves the irreversibility of pruning under HE, balancing efficiency and convergence through dynamic probabilities
5. **Cross-task generality**: effective across image classification, text classification, and text generation tasks, covering ViT, BERT, and TinyLlama

## Limitations & Future Work

1. **Fixed dictionary**: the dictionary is constructed once before training and frozen, limiting adaptability to highly heterogeneous client data distributions; dynamic dictionaries could be explored
2. **Scenario coverage**: only the cross-silo setting (few clients) is evaluated; the cross-device setting (many resource-constrained devices) remains unverified
3. **Model families**: experiments focus on Transformer architectures; CNNs and other architectures are not covered
4. **SVD decomposition cost**: although a one-time operation, SVD on very large models may itself incur significant computation
5. **Selection of $r$**: accuracy is sensitive to dictionary size (a notable jump from $r=2$ to $r=4$); automatic selection of the optimal $r$ is not explored

## Related Work & Insights

| Method | Privacy Level | Communication | Training Speed | Accuracy |
|--------|--------------|---------------|---------------|----------|
| FedHE-Full | Fully encrypted | Very high (baseline) | Very slow | Highest |
| FedHE-Top2 | Fully encrypted (last layer only) | Moderate | Moderate | Lower |
| FedML-HE (10%) | Partially encrypted, leakage risk | High | Slow | High |
| DP-based FL | Noise-based protection | Low | Fast | Degraded |
| MPC-based FL | Aggregation-level protection | Varies | Varies | Lossless |
| **DictPFL** | **Fully encrypted** | **Very low** | **Near plaintext** | **High** |

The fundamental distinction from FedML-HE: FedML-HE reduces the volume of encrypted data (selective encryption), whereas DictPFL reduces the total volume of transmitted data (all encrypted but transmission volume is minimal). The design philosophy is entirely different; DictPFL eliminates privacy leakage at the source.

The dictionary decomposition idea generalizes broadly: decomposing a large parameter space into a shared static component and a personalized trainable component resembles the low-rank decomposition in LoRA, but freezes the left singular vectors of SVD rather than the right. This work also motivates future FL algorithm design to account for encryption scheme constraints (e.g., SIMD slot alignment) from the outset rather than as an afterthought. The pruning consistency problem is transferable to other distributed scenarios requiring consistency, such as sparse communication in distributed training. DictPFL is complementary to secure aggregation (SA): DictPFL protects client-to-server transmission, while SA protects individual contributions during aggregation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combined DePE+PrME design is novel; first to achieve practical HE-FL
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset, multi-model, multi-scenario ablations with complete privacy attack validation
- Writing Quality: ⭐⭐⭐⭐ — Clear figures, coherent motivation and methodology
- Value: ⭐⭐⭐⭐⭐ — Addresses the core practicality bottleneck of HE-FL with major implications for real-world deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping](enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)
- [\[NeurIPS 2025\] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning](design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[NeurIPS 2025\] FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)

</div>

<!-- RELATED:END -->
