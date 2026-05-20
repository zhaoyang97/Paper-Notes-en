---
title: >-
  [Paper Note] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing
description: >-
  [ICCV 2025][AI Safety][Federated Learning] This paper proposes the Client2Vec mechanism, which leverages a CLIP encoder and a Distribution Shifts Aware Index Generation Network (DSA-IGN) to generate…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Distribution Shifts"
  - "Client Indexing"
  - "CLIP"
  - "Non-IID Data"
date: 2026-05-08
content_hash: cfafd6622f0bdadf
---

# Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing

**Conference**: ICCV 2025
**arXiv**: [2405.16233](https://arxiv.org/abs/2405.16233)  
**Code**: [https://github.com/LINs-lab/client2vec](https://github.com/LINs-lab/client2vec)  
**Area**: AI Safety / Federated Learning
**Keywords**: Federated Learning, Distribution Shifts, Client Indexing, CLIP, Non-IID Data

## TL;DR

This paper proposes the Client2Vec mechanism, which leverages a CLIP encoder and a Distribution Shifts Aware Index Generation Network (DSA-IGN) to generate, prior to federated training, an index vector for each client that encodes both label and feature distribution information. The resulting indices are then used to improve three key stages of FL: client sampling, model aggregation, and local training.

## Background & Motivation

The core challenge of federated learning (FL) lies in data heterogeneity (non-IID) across clients. Existing methods primarily optimize **during training** by improving client sampling strategies, model aggregation weights, or local training objectives. Few works address this problem **before** training begins.

Prior pre-training-phase approaches (e.g., dataset distillation-based FedFed and synthetic pseudo-data-based VHL) suffer from high additional computational costs, limited applicability, and incompatibility with the training pipeline. Inspired by Word2Vec in NLP and domain indexing in domain generalization, the authors raise a key question: can a compact "identity vector" be generated for each client before training, encoding local data distribution information to assist throughout the entire training process?

Client2Vec offers three main advantages: (1) index generation is decoupled from FL training, reducing training burden; (2) each client requires only a single index vector, making the approach efficient and lightweight; (3) it can enhance all stages of FL training (sampling, aggregation, and local training).

## Method

### Overall Architecture

The framework consists of two phases: (1) **pre-training** — a DSA-IGN network generates an index vector $\boldsymbol{\beta}_i = [\boldsymbol{\beta}_i^f; \boldsymbol{\beta}_i^l]$ for each client, comprising a feature index and a label index; (2) **training** — the generated indices improve three application cases: client sampling, model aggregation, and local training.

### Key Designs

1. **CLIP Encoding and Index Definition**: A pretrained CLIP model encodes raw data $(x_{i,j}, y_{i,j})$ into image embeddings $\mathbf{D}_{i,j}$ (containing both label and client-specific information) and label embeddings $\mathbf{L}_{i,j}$ (containing label information only). The sample-level label index is directly set as $\mathbf{u}_{i,j}^l = \mathbf{L}_{i,j}$; the sample-level feature index $\mathbf{u}_{i,j}^f$ is obtained by disentangling client-specific, label-independent information from $\mathbf{D}_{i,j}$. The client index is the mean of all sample-level indices: $\boldsymbol{\beta}_i = \frac{1}{N_i}\sum_{j=1}^{N_i}\mathbf{u}_{i,j}$. The core design philosophy is that the feature index should encode client-specific distributional characteristics (e.g., style, background) rather than label-relevant classification information.

2. **Distribution Shifts Aware Index Generation Network (DSA-IGN)**: A three-layer Transformer encoder decomposes $\mathbf{D}_{i,j}$ into a data encoding $\mathbf{z}_{i,j}$ (label-related) and a feature index $\mathbf{u}_{i,j}^f$ (label-independent). Training involves four loss terms: (a) $\mathcal{L}_{\text{sim}}$ — aligns $\mathbf{z}_{i,j}$ with the label embedding to ensure label sensitivity; (b) $\mathcal{L}_{\text{orth}}$ — enforces orthogonality between $\mathbf{u}_{i,j}^f$ and $\mathbf{z}_{i,j}$; (c) $\mathcal{L}_{\text{recon}}$ — reconstructs $\mathbf{D}_{i,j}$ from the concatenation of $\mathbf{u}_{i,j}^f$ and $\mathbf{z}_{i,j}$ to preserve complete information; (d) $\mathcal{L}_{\text{div}}$ — a SimCLR-style negative pair loss that promotes diversity among $\mathbf{u}_{i,j}^f$ across samples to prevent training collapse. Two training strategies are supported: Global (uploading 128 samples to the server for centralized training) and Federated (distributed training via FedAvg).

3. **Three Application Cases**:

    - **Case 1 (Client Sampling)**: A greedy strategy that biases round-$t$ sampling toward clients similar to those selected in round $t-1$. The sampling probability is $p_i^t = \frac{\exp(S(\boldsymbol{\beta}_i, \mathcal{C}^{t-1})/\tau)}{\sum_j \exp(S(\boldsymbol{\beta}_j, \mathcal{C}^{t-1})/\tau)}$, where the similarity function $S$ jointly considers cosine similarities of both feature and label indices.
    - **Case 2 (Model Aggregation)**: Based on the Multiplicative Weights Update (MWU) algorithm, clients with higher similarity receive larger aggregation weights. Solving the optimization problem yields $p_{i,g}^t \propto q_i^t \exp(\frac{1}{\lambda_1}\sum_{\tau=1}^t \gamma^{t-\tau} S(\beta_i, \mathcal{C}^\tau))$, incorporating a profit term (similarity), an entropy regularization term, and a normalization constraint.
    - **Case 3 (Local Training)**: A projection layer maps local features to the same dimensionality as $\boldsymbol{\beta}_i^f$. An orthogonality loss $\mathcal{L}_{\text{orth}} = \|\mathbf{z}_P \mathbf{B}^f\|_1$ encourages local feature representations to be orthogonal to the client-specific index, supplemented by a distillation loss to preserve the information content of the original features.

### Loss & Training

The total loss for DSA-IGN is $\mathcal{L} = \mathcal{L}_{\text{div}} + \mathcal{L}_{\text{sim}} + \mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{recon}}$. During FL training, the local training loss is $\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{dist}}$, augmenting the standard classification loss with orthogonality constraints and knowledge distillation.

## Key Experimental Results

### Main Results

| Dataset (Model) | FL Algorithm | Baseline | +Sampling+Aggregation+Local Training (Global) | Max Gain |
|---|---|---|---|---|
| Shakespeare (LSTM) | FedAvg | 49.93 | **50.51** | +0.58 |
| CIFAR10 (ResNet18) | FedAvg | 42.24 | **59.29** | +17.05 |
| CIFAR10 (ResNet18) | FedAvgM | 42.56 | **69.37** | +26.81 |
| CIFAR10 (ResNet18) | FedDyn | 37.22 | **70.59** | +33.37 |
| DomainNet (MobileNetV2) | FedAvg | 46.31 | **57.43** | +11.12 |
| DomainNet (MobileNetV2) | Moon | 50.56 | **60.48** | +9.92 |

Gains are most pronounced on CIFAR10 (up to +33.37%) and exceed 10% on DomainNet, demonstrating Client2Vec's effectiveness against both label shift and feature shift.

### Ablation Study

| Configuration | CIFAR10 (FedAvg) | DomainNet (FedAvg) | Notes |
|---|---|---|---|
| Baseline | 42.24 | 46.31 | Original |
| +Sampling (i) | 44.60 | 50.78 | ~2–4% gain from sampling |
| +Sampling+Aggregation (i+ii) | 44.10 | 53.83 | Further gain from aggregation |
| +All (i+ii+iii) | **59.29** | **56.43** | Local training contributes most |

### Key Findings

- Improvements across the three cases are **cumulatively additive**, with local training (Case 3) contributing the most, underscoring the importance of eliminating client-specific information from local representations for model generalization.
- Visualizations on DomainNet show that index similarity among clients within the same feature domain approaches 1.0, while inter-domain distances are large, validating that the index vectors effectively encode distributional information.
- Inter-domain similarities align with human intuition: the Real domain is closer to Clipart, Painting, and Sketch, and farther from Infograph and Quickdraw.
- Both Global and Federated training strategies produce meaningful indices, with the Global strategy yielding clearer domain boundaries.

## Highlights & Insights

- The decoupling of "pre-training analysis" from "in-training optimization" is a noteworthy design principle: generate indices once, benefit throughout the entire pipeline.
- CLIP's cross-modal alignment capability elegantly resolves the challenge of mapping labels and images into a shared representation space.
- Orthogonality constraints appear consistently throughout the framework — disentangling feature indices from data encodings during index generation, and separating local features from client-specific feature indices during local training — reflecting a coherent and unified design philosophy.

## Limitations & Future Work

- The approach relies on CLIP pretrained models, and performance may degrade in domains poorly covered by CLIP (e.g., medical imaging).
- The Global strategy requires uploading partial data embeddings (CLIP features, not raw data) to the server; while privacy risk is reduced, it still warrants careful evaluation.
- Gains on NLP tasks (Shakespeare) are marginal (<1%), suggesting limited benefit in scenarios with mild distribution shifts.
- The index dimensionality $d_i$ and the number of DSA-IGN training epochs require task-specific tuning.

## Related Work & Insights

- This work extends the idea of Variational Domain Indexing (VDI) to the federated learning setting, addressing VDI's limitations in FL regarding communication cost, privacy, and neglect of label shift.
- Compared to data-sharing methods such as FedBR and VHL, Client2Vec incurs lower communication overhead (only index vectors need to be exchanged).
- The application of the MWU algorithm to model aggregation yields a theoretically elegant derivation of aggregation weights.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The pre-training index generation paradigm is novel, and the three application cases provide comprehensive coverage
- **Experimental Thoroughness**: ⭐⭐⭐⭐ A complete experimental matrix across three datasets, multiple baseline algorithms, and two training strategies
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and rigorous definitions
- **Value**: ⭐⭐⭐ Performance gains vary considerably across settings; benefits are limited in NLP scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](../../NeurIPS2025/ai_safety/flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)

</div>

<!-- RELATED:END -->
