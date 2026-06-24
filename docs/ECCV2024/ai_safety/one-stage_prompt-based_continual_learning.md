---
title: >-
  [Paper Note] One-stage Prompt-based Continual Learning
description: >-
  [ECCV 2024][AI Safety][Prompt-based Continual Learning] This paper proposes the OS-Prompt framework. By directly utilizing the token embeddings of ViT intermediate layers as prompt queries (rather than relying on an extra query ViT forward pass), it reduces the computational-cost of prompt-based continual learning by approximately 50%. It further compensates for the loss in representation capacity with a Query-Pool Regularization (QR) loss, outperforming CodaPrompt by about 1…
tags:
  - "ECCV 2024"
  - "AI Safety"
  - "Prompt-based Continual Learning"
  - "Vision Transformer"
  - "Computational Efficiency"
  - "Query-Pool Regularization"
  - "Class-Incremental Learning"
date: 2026-05-08
content_hash: a030fbc85da511c1
---

# One-stage Prompt-based Continual Learning

**Conference**: ECCV 2024  
**arXiv**: [2402.16189](https://arxiv.org/abs/2402.16189)  
**Code**: None  
**Area**: Continual Learning / Efficient Learning  
**Keywords**: Prompt-based Continual Learning, Vision Transformer, Computational Efficiency, Query-Pool Regularization, Class-Incremental Learning

## TL;DR

This paper proposes the OS-Prompt framework. By directly utilizing the token embeddings of ViT intermediate layers as prompt queries (rather than relying on an extra query ViT forward pass), it reduces the computational-cost of prompt-based continual learning by approximately 50%. It further compensates for the loss in representation capacity with a Query-Pool Regularization (QR) loss, outperforming CodaPrompt by about 1.4% on CIFAR-100, ImageNet-R, and DomainNet.

## Background & Motivation

Prompt-based Continual Learning (PCL) is the state-of-the-art (SOTA) paradigm in continual learning. It avoids catastrophic forgetting by training learnable prompt tokens on a pre-trained ViT without storing historical data (which is privacy-friendly and memory-efficient). However, existing PCL methods (such as L2P, DualPrompt, and CodaPrompt) require a **two-stage forward pass**: the first stage uses a frozen query ViT to generate prompt queries for selecting prompts from a prompt pool, while the second stage merges the selected prompts with image tokens in the backbone ViT for classification. This dual-ViT architecture doubles the logic and computation cost during both training and inference (around 35 GFLOPs for inference), severely limiting deployment on resource-constrained devices.

**Key Challenge**: The contradiction between the high accuracy of PCL and the high computational cost introduced by the dual-ViT architecture.

**Key Insight**: The authors observe that during the prompt continual learning process, the token embeddings in the early layers of ViTs change minimally (cosine distance $\le 0.1$). This implies that intermediate layer embeddings can be directly used as prompt queries, thereby eliminating the entire query ViT.

**Core Idea**: One-stage PCL. By replacing the extra query ViT with intermediate-layer [CLS] tokens, the inference GFLOPs are halved with an accuracy loss of $\le 1\%$.

## Method

### Overall Architecture

The OS-Prompt framework requires only **a single ViT forward pass**. After the image is input into the backbone ViT, the [CLS] token embedding of the current layer is directly used as the prompt query in each of the first five layers (layers 1-5). The similarity between the query and the keys in the prompt pool is computed to generate the prompt token for that layer, which is then injected into the self-attention mechanism via prefix tuning. Finally, the classification head outputs the prediction.

OS-Prompt++ additionally introduces a frozen reference ViT during training to extract the final-layer [CLS] token for QR loss regularization, while the reference ViT is not used during inference.

### Key Designs

1. **Intermediate-Layer Token Embedding as Prompt Query**:

    - **Function**: Directly use the [CLS] token $q_l = x_{l_{[CLS]}}$ of the $l$-th layer in the backbone ViT as the prompt query, replacing the query $q = Q(x)_{[CLS]}$ that previously required an extra query ViT forward pass.
    - **Mechanism**: Since prompts are only added to layers 1-5 and the backbone ViT weights are frozen, key token embeddings in early layers remain highly stable during continual learning.
    - **Design Motivation**: Through experiments in the CIFAR-100 10-task setup, the authors measured the change in cosine distance of token embeddings across different layers. The distance for layers 1-5 consistently remains $\le 0.1$, whereas that of the last layer is $\ge 0.1$ and continuously increases as tasks accumulate. This demonstrates that early-layer embeddings are stable enough to serve as queries.
    - **Difference from Prior Work**: Prior methods used an independent frozen ViT to ensure query consistency, whereas this work directly utilizes the internal embeddings of the backbone ViT, saving half the computation.

2. **Layer-wise Prompt Generation (CodaPrompt-style Weighted Summation)**:

    - **Function**: For the $l$-th layer, compute the cosine similarity $\gamma(\cdot)$ between query $q_l$ and prompt pool keys $\{k_l^1, ..., k_l^M\}$, and obtain the prompt $\phi_l = \sum_m \gamma(q_l, k_l^m) p_l^m$ via weighted summation.
    - **Mechanism**: Adopt the soft matching strategy of CodaPrompt to enable end-to-end training.
    - **Design Motivation**: Compared with the hard top-k selection in L2P, weighted summation allows gradients to flow through all prompt components.

3. **Query-Pool Regularization (QR) Loss**:

    - **Function**: During training, extract the final-layer [CLS] token $r$ using a reference ViT, constraining the similarity distribution between intermediate-layer queries and the prompt pool to approximate the reference distribution.
    - **Mechanism**: Define two softmax-normalized similarity vectors $A_{query}^l = \text{Softmax}(\frac{K_l q_l^T}{\|K_l\|_2 \|q_l\|_2})$ and $A_{ref}^l = \text{Softmax}(\frac{K_l r^T}{\|K_l\|_2 \|r\|_2})$. The QR loss is formulated as $\mathcal{L}_{QR} = \sum_l \|A_{query}^l - A_{ref}^l\|_2^2$.
    - **Design Motivation**: Since the representation capacity of intermediate-layer tokens is weaker than that of the final layer, direct application leads to an accuracy drop of approximately 1%. Inspired by knowledge distillation, the QR loss forces the prompt pool to learn representation relationships consistent with the final-layer query.
    - **Key Point**: The QR loss is used **only during training**. Since the reference ViT is omitted during inference, the inference cost remains at 50% of the baseline.

### Loss & Training

The total loss is the weighted sum of the cross-entropy classification loss and the QR loss:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{QR}$$

Where $\lambda$ is a hyperparameter (default is 1e-4), tuned using a 20% validation split of the training set. During training, only the keys and prompts in the prompt pool are updated, while the backbone ViT parameters remain frozen. Prefix-tuning is applied to split the prompt into $[\phi_k, \phi_v]$ and prepend them to the self-attention keys and values.

## Key Experimental Results

### Main Results

**ImageNet-R 10-task setup (class-incremental, averaged over 5 different random seeds):**

| Method | $A_N$ (↑) | $F_N$ (↓) | Inference GFLOPs |
|------|-----------|-----------|-------------|
| L2P | 69.29 | 2.03 | 35.1 (100%) |
| DualPrompt | 71.32 | 1.71 | 35.1 (100%) |
| CodaPrompt | 75.45 | 1.64 | 35.1 (100%) |
| **OS-Prompt** | 74.58 | 1.92 | **17.6 (50.1%)** |
| **OS-Prompt++** | **75.67** | **1.27** | **17.6 (50.1%)** |

**CIFAR-100 10-task setup:**

| Method | $A_N$ (↑) | $F_N$ (↓) |
|------|-----------|-----------|
| CodaPrompt | 86.25 ± 0.74 | 1.67 ± 0.26 |
| OS-Prompt | 86.42 ± 0.61 | 1.64 ± 0.14 |
| OS-Prompt++ | **86.68 ± 0.67** | **1.18 ± 0.21** |

**DomainNet 5-task setup:**

| Method | $A_N$ (↑) | $F_N$ (↓) |
|------|-----------|-----------|
| CodaPrompt | 73.24 ± 0.59 | 3.46 ± 0.09 |
| OS-Prompt++ | **73.32 ± 0.32** | **2.07 ± 0.06** |

### Ablation Study

**QR Loss Design Ablation (ImageNet-R 10-task):**

| Configuration | $A_N$ (↑) | $F_N$ (↓) | Description |
|------|-----------|-----------|------|
| No CosSim No Softmax | 75.00 | 1.68 | Baseline |
| CosSim Only | 75.47 | 1.38 | +0.47 |
| Softmax Only | 75.51 | 1.28 | +0.51 |
| CosSim + Softmax | **75.67** | **1.27** | Synergistic, best performance |

**Hyperparameter $\lambda$ Sensitivity (ImageNet-R 5/10/20-task):**

| $\lambda$ | Task-5 | Task-10 | Task-20 |
|-----------|--------|---------|---------|
| 1e-5 | 77.03 | 75.63 | 73.63 |
| 1e-4 | 77.07 | 75.67 | 73.77 |
| 5e-4 | 77.13 | 75.68 | 73.68 |

### Key Findings

- **QR Loss Contributes the Most**: OS-Prompt $\rightarrow$ OS-Prompt++ improves by 1.77% (72.00 $\rightarrow$ 73.77) on ImageNet-R 20-task, while the forgetting rate decreases from 1.09 to 0.79.
- **Insensitivity to Hyperparameter**: Performance variation remains negligible (< 0.1%) for $\lambda$ ranging from 1e-5 to 5e-4.
- **Number of Prompts**: OS-Prompt++ reaches a performance plateau with 50 prompts, whereas OS-Prompt requires more prompts to saturate.
- **Inference Latency**: Latency is reduced by approximately 50% across RTX 2080 Ti, RTX 3090, and A100 GPUs.
- **Compatibility with Different Prompt Formation Strategies**: When combined with L2P and DualPrompt strategies, the OS-Prompt framework still outperforms their original implementations.

## Highlights & Insights

- **Simple yet Effective Observation**: The stability of early-layer embeddings acts as the foundation of the proposed method. This observation is highly clean and generalizable, potentially holding true for any prompt tuning setup with a frozen backbone.
- **Decoupled Training-Inference Design**: The QR loss and the reference ViT are only utilized during training. Inference is entirely one-stage, achieving a win-win scenario for both accuracy and efficiency.
- **Transferable Concept**: The idea of "replacing extra forward passes with intermediate representations" can be extended to other dual-stage prompt learning methodologies.

## Limitations & Future Work

- **Training Cost is Not Reduced (OS-Prompt++ Version)**: While saving 50% on inference computation, the forward pass of the reference ViT is still required during training, yielding the same training GFLOPs as original methods.
- **Slight Degradation of CodaPrompt's Soft Matching under Intermediate Queries**: Experiments show that hard matching (e.g., top-k in L2P/Dual) is more robust to intermediate queries.
- **Evaluation Limited to ViT-B/16**: Validation was not conducted on larger models (e.g., ViT-L) or alternative architectures.
- **Accuracy Ceiling under Class-Incremental Setup**: A performance gap still exists compared to the Upper Bound (UB: 77.13%).

## Related Work & Insights

- **vs CodaPrompt**: CodaPrompt is a state-of-the-art method that trains an end-to-end prompt pool through weighted summation. OS-Prompt++ achieves higher accuracy on top of CodaPrompt while reducing inference costs by 50% through the one-stage framework.
- **vs L2P / DualPrompt**: These methods introduced the concept of the prompt pool but require a dual-ViT design. The OS-Prompt framework is compatible with, and outperforms, their prompt formation strategies.
- **vs DINO Pre-training**: OS-Prompt maintains its advantage even when evaluated under unsupervised pre-training weights (e.g., DINO), indicating that the method does not rely on a specific pre-training regime.

## Rating

- Novelty: ⭐⭐⭐⭐ The core observation (early-layer stability) is elegant and powerful, leading to a natural one-stage framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across three datasets, various task configurations, intensive ablations, GFLOPs/latency comparisons, different pre-trained weights, and various prompt formation strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rich tables/figures, and a smooth "motivation-observation-methodology" flow.
- Value: ⭐⭐⭐⭐ The 50% inference speedup is highly significant for the practical deployment of PCL, and the design pattern of the QR loss is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Noise-Assisted Prompt Learning for Image Forgery Detection and Localization](noise-assisted_prompt_learning_for_image_forgery_detection_and_localization.md)
- [\[ICML 2026\] Understanding Generalization and Forgetting in In-Context Continual Learning](../../ICML2026/ai_safety/understanding_generalization_and_forgetting_in_in-context_continual_learning.md)
- [\[ICML 2026\] Active Continual Learning with Metaplastic Binary Bayesian Neural Networks](../../ICML2026/ai_safety/active_continual_learning_with_metaplastic_binary_bayesian_neural_networks.md)
- [\[ICML 2026\] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](../../ICML2026/ai_safety/hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)
- [\[CVPR 2026\] COPYLENS: Towards Copyrighted Characters Infringement Detection via Copyright-Aware Prompt Learning](../../CVPR2026/ai_safety/copylens_towards_copyrighted_characters_infringement_detection_via_copyright-awa.md)

</div>

<!-- RELATED:END -->
