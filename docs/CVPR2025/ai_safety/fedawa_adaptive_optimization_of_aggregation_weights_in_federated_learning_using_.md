---
title: >-
  [Paper Note] FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors
description: >-
  [CVPR 2025][AI Safety][Federated Learning] FedAWA is proposed, which is inspired by task arithmetic and uses client vectors (the difference between local parameters and global parameters) to adaptively optimize aggregation weights in federated learning. Clients whose updates align with the global optimization direction are assigned higher weights, consistently improving FedAvg by 1–4 percentage points in non-IID scenarios.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Aggregation Weight Optimization"
  - "Client Vectors"
  - "Task Arithmetic"
  - "Non-IID Data"
date: 2026-05-08
content_hash: 2d57199659c8eb60
---

# FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors

**Conference**: CVPR 2025  
**arXiv**: [2503.15842](https://arxiv.org/abs/2503.15842)  
**Code**: [https://github.com/ChanglongShi/FedAWA](https://github.com/ChanglongShi/FedAWA)  
**Area**: AI Security / Federated Learning  
**Keywords**: Federated Learning, Aggregation Weight Optimization, Client Vectors, Task Arithmetic, Non-IID Data

## TL;DR

FedAWA is proposed, which is inspired by task arithmetic and uses client vectors (the difference between local parameters and global parameters) to adaptively optimize aggregation weights in federated learning. Clients whose updates align with the global optimization direction are assigned higher weights, consistently improving FedAvg by 1–4 percentage points in non-IID scenarios.

## Background & Motivation

### Background

Federated learning (FL) trains a global model by aggregating model parameters across multiple clients. FedAvg aggregates weights based on dataset size, but in non-IID scenarios, the update directions of different clients may conflict, leading to unstable convergence of the global model.

### Limitations of Prior Work

Existing adaptive aggregation methods (e.g., FedLAW, L-DAWA) suffer from either high computational overhead (10+ seconds per round) or the requirement for auxiliary validation data. There is a lack of an aggregation weight optimization scheme that is both lightweight and effective.

### Key Challenge

Equal or data size-based weighted aggregation ignores the "quality" of client updates. Under non-IID settings, the update directions of some clients are detrimental (deviating from the global optimum) and their weights should be reduced.

### Key Insight

Task arithmetic theory in model merging suggests that the parameter difference vector (task vector) encapsulates task-specific knowledge. Applying this to FL, the client vector $\tau_k = \theta_k - \theta_g$ reflects the characteristics of local data and can be utilized to measure the "usefulness" of updates.

### Core Idea

Optimizing aggregation weights based on the alignment between client vectors and the global aggregation vector leads to a more consistent global update direction.

## Method

### Key Designs

1. **Client Vector-Driven Weight Optimization**:

    - **Function**: Adaptively assigns aggregation weights to each client.
    - **Mechanism**: Define the client vector as $\tau_k^t = \theta_k^t - \theta_g^t$ and the global aggregation vector as $\tau_g^t = \sum_k \lambda_k \tau_k^t$. The optimization objective is $\min_\lambda \sum_k \lambda_k \|\tau_k^t - \tau_g^t\|_2 + d(\sum_k \lambda_k \theta_k^t, \theta_g^t)$ subject to $\|\lambda\|_1 = 1$. The first term encourages selecting clients aligned with the global direction, while the second term constrains the aggregated model from deviating too far.
    - **Design Motivation**: Figure 2 verifies that the differences in client vectors indeed reflect dataset distribution discrepancies, and the global vector is closer to the "ideal" vector than any single client vector.

2. **Layer-wise Variant FedAWA-L**:

    - **Function**: Independently optimizes weights for each layer to achieve fine-grained control.
    - **Mechanism**: Solve for $\lambda_l^t$ independently for each layer $l$, allowing different layers to have distinct optimal weight combinations.
    - **Design Motivation**: Different layers learn features at different levels, and non-IID distributions affect shallow layers (local features) and deep layers (semantic features) differently.

### Loss & Training

Aggregation weights are computed via constrained optimization, using 1-cosine similarity as the distance function. Client-side local training employs standard SGD with cross-entropy. The aggregation time is only 0.82 seconds (vs. L-DAWA at 2.52 seconds and FedLAW at 10.11 seconds).

## Key Experimental Results

### Main Results

CIFAR-10 Top-1 Accuracy (%):

| Method | IID (α=100) | non-IID (α=0.5) |
|------|-------------|-----------------|
| FedAvg | 76.01 | 74.47 |
| FedProx | 76.47 | 73.85 |
| **FedAWA** | **80.10** | **75.65** |
| **FedAWA-L** | **79.70** | 74.90 |

### Ablation Study

| Configuration | Effect |
|------|------|
| Combination with FedDisco | Additional gains, demonstrating plug-and-play compatibility |
| K=10/30/50 clients | Consistent improvements, demonstrating parameter robustness |
| E=1/5/10 local epochs | Consistent improvements |

### Key Findings
- **Significant improvement even under IID**: 80.10 vs. 76.01, indicating that even with identical data distributions, equal-weight aggregation is not optimal.
- **Extremely low computational overhead**: 0.82 seconds per round, adding only a single optimization step compared to FedAvg.
- **Cross-architecture versatility**: Effective across CNN, ResNet, WRN, DenseNet, and ViT.

## Highlights & Insights
- **Federated learning adaptation of task arithmetic**—Client vectors serve as the FL equivalent of task vectors; this analogy is both simple and effective.
- **Plug-and-play design**—Can be integrated with any existing FL method without modifying the local training process.

## Limitations & Future Work
- Requires storage and transmission of client-level parameter vectors, increasing communication overhead.
- Aggregation time of FedAWA-L is 15.21 seconds compared to 0.82 seconds for the global version.
- Lack of theoretical convergence analysis.
- Client vectors may indirectly leak information regarding model updates.

## Rating
- Novelty: ⭐⭐⭐⭐ The transfer from task arithmetic to FL is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, architectures, and settings.
- Writing Quality: ⭐⭐⭐⭐ Clear and comprehensive.
- Value: ⭐⭐⭐ Moderate performance improvement but the method is lightweight and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [\[CVPR 2025\] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)
- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2025\] Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)

</div>

<!-- RELATED:END -->
