---
title: >-
  [Paper Note] FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation
description: >-
  [ICML 2025][Optimization][federated learning] This paper proposes FSL-SAGE, a federated split learning algorithm that estimates server-side gradient feedback via an auxiliary model. It significantly reduces communication overhead and client-side memory footprint while maintaining an $O(1/\sqrt{T})$ convergence rate comparable to FedAvg.
tags:
  - "ICML 2025"
  - "Optimization"
  - "federated learning"
  - "split learning"
  - "gradient estimation"
  - "auxiliary model"
  - "communication efficiency"
date: 2026-05-08
content_hash: b7b14d2fb03d6602
---

# FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation

**Conference**: ICML 2025  
**arXiv**: [2505.23182](https://arxiv.org/abs/2505.23182)  
**Code**: None  
**Area**: Optimization  
**Keywords**: federated learning, split learning, gradient estimation, auxiliary model, communication efficiency

## TL;DR
This paper proposes FSL-SAGE, a federated split learning algorithm that estimates server-side gradient feedback via an auxiliary model. It significantly reduces communication overhead and client-side memory footprint while maintaining an $O(1/\sqrt{T})$ convergence rate comparable to FedAvg.

## Background & Motivation

**Background**: Federated learning (FL) and split learning (SL) are two mainstream distributed privacy-preserving training paradigms. FL (such as FedAvg) requires clients to train the complete model, whereas SL splits the model into client and server components to reduce the client-side computational burden.

**Limitations of Prior Work**: FL requires clients to have sufficient memory to store and train the entire model, which is infeasible for large models (e.g., LLMs). While SL alleviates client-side burden, communication latency scales linearly with the number of clients because each client must interact with the server sequentially (forward-backward propagation). Existing federated split learning (FSL) methods attempt to combine the advantages of both, but either lack server-feedback (by substituting it with local loss) leading to accuracy degradation, or still suffer from communication bottlenecks.

**Key Challenge**: How to obtain high-quality gradient signals while keeping the client-side memory requirements low? Although using local loss avoids communication bottlenecks, the gradient quality is poor; while waiting for server feedback provides accurate gradients, it suffers from low communication efficiency.

**Goal**: Design an FSL algorithm that enables parallel training of multiple clients while providing high-quality gradient estimation.

**Key Insight**: Introduce a lightweight auxiliary model on the client side to simulate the behavior of the server-side model, periodically synchronizing the auxiliary model from the server.

**Core Idea**: Use the client's local auxiliary model to estimate the server-side gradient feedback (smashed activation gradient), with the auxiliary model tracking the evolution of the server model through periodic synchronization.

## Method

### Overall Architecture
The model is partitioned into two parts: the client model $f_c$ and the server model $f_s$. In standard SL, the client performs forward propagation to obtain the smashed activation $a = f_c(x)$, which is sent to the server. The server then completes forward and backward propagation and returns the gradient $\nabla_a \ell$. FSL-SAGE replaces this interaction with an auxiliary model $\tilde{f}_s$ to perform gradient estimation locally on the client.

Input: Local data distributed across $K$ clients → Client forward propagation → Gradient estimation via auxiliary model → Client update → Periodic synchronization of auxiliary models and aggregation

### Key Designs

1. **Auxiliary Model Gradient Estimation**:

    - Function: Substitute the server model $f_s$ with the auxiliary model $\tilde{f}_s$ locally on the client side to estimate the gradient of the smashed activation.
    - Mechanism: Use $\hat{g} = \nabla_a \ell(\tilde{f}_s(a), y)$ as an estimate of the true gradient $\nabla_a \ell(f_s(a), y)$.
    - Design Motivation: Avoid the sequential client-server communication bottleneck. The auxiliary model is merely an approximate copy of the server model; it incurs low overhead yet provides meaningful gradient directions.

2. **Periodic Auxiliary Model Adaptation**:

    - Function: Every $\tau$ communication rounds, the server distributes the latest model parameters to all clients to update their auxiliary models.
    - Mechanism: The synchronization period $\tau$ controls the deviation between the auxiliary model and the server model—a smaller $\tau$ leads to more accurate gradient estimation but higher communication overhead.
    - Design Motivation: The auxiliary model gradually becomes stale as training progresses; therefore, periodic synchronization is crucial to control estimation bias. A reasonable synchronization frequency is determined through theoretical analysis.

3. **Parallel Client Training**:

    - Function: Enable all clients to perform local training simultaneously without waiting for server-side feedback.
    - Mechanism: Because gradient estimation is completed entirely locally, there are no dependencies between clients.
    - Design Motivation: This is the core advantage of FSL-SAGE in communication efficiency compared to traditional SL.

### Loss & Training
Standard cross-entropy loss. The training strategy is as follows: clients perform multi-step local gradient descent (utilizing gradients estimated via the auxiliary model), and then transmit client model parameters to the server for aggregation (similar to FedAvg), while periodically synchronizing the auxiliary model.

## Key Experimental Results

### Main Results

| Dataset / Model | Metric (Top-1 Acc%) | FSL-SAGE | SplitFed | FedAvg | LocalLoss-FSL |
|------------|-------------------|----------|----------|--------|--------------|
| CIFAR-10 / ResNet-18 | Test Accuracy | **92.4%** | 90.1% | 93.1% | 88.7% |
| CIFAR-100 / ResNet-34 | Test Accuracy | **71.8%** | 67.3% | 73.2% | 63.5% |
| Tiny-ImageNet / VGG-16 | Test Accuracy | **58.6%** | 53.9% | 60.1% | 49.8% |

### Ablation Study

| Configuration | CIFAR-100 Accuracy | Comm. Volume (Relative to FedAvg) | Description |
|------|----------------|---------------------|------|
| FSL-SAGE ($\tau$=5) | **71.8%** | 0.3x | Optimal trade-off point |
| FSL-SAGE ($\tau$=1) | 72.5% | 0.8x | Frequent synchronization: slightly higher accuracy but increased communication |
| FSL-SAGE ($\tau$=20) | 68.2% | 0.15x | Infrequent synchronization: auxiliary model becomes stale |
| No auxiliary model (Local Loss) | 63.5% | 0.1x | Poor gradient quality |

### Key Findings
- FSL-SAGE achieves an $O(1/\sqrt{T})$ convergence rate, which is consistent with FedAvg.
- With only 30% of the communication volume of FedAvg, the accuracy gap is restricted to within 1-2%.
- The synchronization frequency $\tau$ of the auxiliary model is a key hyperparameter; synchronizing every 5-10 rounds is a reasonable choice.
- Compared to local loss-based methods, FSL-SAGE improves accuracy by 5-8 percentage points.

## Highlights & Insights
- **Strong integration of theory and practice**: Not only does it provide an $O(1/\sqrt{T})$ convergence proof, but the experiments also validate the theoretical predictions.
- **High practical value**: This approach holds practical significance for the distributed training of large models on resource-constrained devices (e.g., mobile devices, IoT).
- **The auxiliary model concept** can be generalized to other distributed optimization scenarios that require approximate feedback.

## Limitations & Future Work
- The auxiliary model itself incurs memory overhead, which may still present a bottleneck for extremely large models.
- The estimation bias of the auxiliary model may be larger under non-IID data distributions.
- Currently, validation is limited to CV tasks; applications in NLP/LLM scenarios remain to be explored.
- Adaptive adjustment strategies for the synchronization frequency of the auxiliary model are worth researching.

## Related Work & Insights
- FedAvg (McMahan et al., 2017): Federated learning baseline.
- SplitFed (Thapa et al., 2022): Federated split learning.
- The idea of using an auxiliary model to estimate gradients is conceptually similar to a reverse application of knowledge distillation.

## Personal Thoughts
- The auxiliary model approach is fundamentally about replacing remote precise computation with local approximations. Such trade-offs are ubiquitous in edge computing.
- The relationship between the synchronization frequency $\tau$ and estimation quality can be more precisely characterized using bias-variance decomposition.
- Adaptive synchronization strategies could be considered—triggering synchronization when the divergence between the auxiliary model and the server model exceeds a threshold.
- The combination of FSL-SAGE with model distillation is also an interesting direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Auxiliary model gradient estimation is a meaningful innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets and baselines are compared with thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and systematic description of methodology.
- Value: ⭐⭐⭐⭐ Practical value for resource-constrained federated learning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] A Unified View on Learning Unnormalized Distributions via Noise-Contrastive Estimation](a_unified_view_on_learning_unnormalized_distributions_via_noise-contrastive_esti.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[ICML 2025\] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](the_panaceas_for_improving_low-rank_decomposition_in_communication-efficient_fed.md)
- [\[ICLR 2026\] Binomial Gradient-Based Meta-Learning for Enhanced Meta-Gradient Estimation](../../ICLR2026/optimization/binomial_gradient-based_meta-learning_for_enhanced_meta-gradient_estimation.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](../../NeurIPS2025/optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)

</div>

<!-- RELATED:END -->
