---
title: >-
  [Paper Note] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning
description: >-
  [ICCV 2025][Model Compression][Federated Unsupervised Learning] This paper proposes the Soft Separation and Distillation (SSD) framework, which addresses insufficient inter-client representation uniformity in federated unsupervised learning through two modules — Dimension Scaling Regularization (DSR) and Projector Distillation (PD) — significantly improving global representation quality without incurring additional communication overhead.
tags:
  - ICCV 2025
  - Model Compression
  - Federated Unsupervised Learning
  - Representation Uniformity
  - Dimension Scaling Regularization
  - Projector Distillation
  - Non-IID
date: 2026-05-08
content_hash: f730fbc5490fa1de
---

# Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning

**Conference**: ICCV 2025
**arXiv**: [2508.01251](https://arxiv.org/abs/2508.01251)
**Code**: [https://ssd-uniformity.github.io/](https://ssd-uniformity.github.io/)
**Area**: Model Compression / Federated Learning
**Keywords**: Federated Unsupervised Learning, Representation Uniformity, Dimension Scaling Regularization, Projector Distillation, Non-IID

## TL;DR

This paper proposes the Soft Separation and Distillation (SSD) framework, which addresses insufficient inter-client representation uniformity in federated unsupervised learning through two modules — Dimension Scaling Regularization (DSR) and Projector Distillation (PD) — significantly improving global representation quality without incurring additional communication overhead.

## Background & Motivation

Federated Unsupervised Learning (FUL) aims to learn expressive representations in distributed, label-free settings. In self-supervised representation learning, representation quality is governed by two metrics: **alignment** (measuring the distance between representations of semantically similar samples) and **uniformity** (measuring how uniformly representations are distributed on the unit hypersphere).

Existing FUL methods can achieve reasonable **intra-client** local uniformity, but struggle to maintain **inter-client** global uniformity after model aggregation. This problem stems from two core challenges:

**Non-IID data distribution**: Heterogeneous data distributions across clients cause representation directions to overlap in the embedding space.

**Decentralized nature of federated learning**: The server cannot access raw data or embeddings, precluding direct cross-client uniformity constraints.

Existing methods either focus on controlling global model consistency (e.g., FedX, L-DAWA) or address local dimensional collapse (e.g., FedDecorr, FedU2), but none explicitly resolve inter-client uniformity.

## Method

### Overall Architecture

SSD introduces two additional components into the standard federated learning pipeline. At initialization, the server assigns each client a unique dimension scaling vector. During local training, each client optimizes a joint objective consisting of four loss terms:

$$\mathcal{L}^k = \mathcal{L}_{\text{align}}^k + \beta \mathcal{L}_{\text{uniform}}^k + \gamma \mathcal{L}_{\text{DSR}}^k + \delta \mathcal{L}_{\text{distill}}^k$$

where $\beta, \gamma, \delta$ are set to 1.0, 1.0, and 0.1, respectively.

### Key Designs

1. **Dimension Scaling Regularization (DSR)**: A scaling vector $\mathbf{d}_k \in \mathbb{R}^d$ is defined for each client $k$, where a subset of dimensions is amplified by a scaling factor $\alpha$, and the scaled dimensions are non-overlapping across clients ($\mathcal{S}_i \cap \mathcal{S}_j = \emptyset$). The DSR loss pulls embeddings toward their scaled counterparts: $\mathcal{L}_{\text{DSR}}^k = \mathbb{E}[\|\mathbf{z} - \text{stopgrad}(\mathbf{z} \odot \mathbf{d}_k)\|_2^2]$. By encouraging representations from different clients to expand in distinct directions, DSR increases angular separation across clients, thereby improving global uniformity. This constitutes a "soft separation" strategy that preserves flexibility in the shared dimensions.

2. **Projector Distillation (PD)**: Empirical observation reveals that the optimization effect of DSR on the embedding space cannot be fully transferred to the representation space, as the projector $g(\cdot)$ absorbs most of the optimization signal. PD bridges this gap by minimizing the KL divergence between representations and embeddings: $\mathcal{L}_{\text{distill}}^k = \mathbb{E}[D_{\text{KL}}(\sigma(\mathbf{h}) \| \sigma(\mathbf{z}))]$. This enables the encoder to internalize beneficial structure learned in the embedding space while preserving the projector's critical role in preventing overfitting.

3. **Soft Separation vs. Hard Separation**: Hard separation (HSD) restricts each client to a completely independent subspace, achieving the highest uniformity but severely compromising alignment, which degrades downstream performance. SSD's soft separation achieves a favorable balance between uniformity and alignment.

### Loss & Training

- Four losses are jointly optimized: alignment loss + uniformity loss + DSR loss + PD distillation loss.
- Scaling factor $\alpha = 10.0$; each client is assigned approximately $\lfloor d/K \rfloor$ non-overlapping dimensions.
- Standard FedAvg aggregation is used; the only additional communication is the lightweight scaling vectors distributed at initialization.
- The encoder is ResNet-18; the projector is a two-layer linear network.

## Key Experimental Results

### Main Results

| Method | CIFAR10 Cross-Silo LP | CIFAR10 Cross-Silo FT 1% | CIFAR10 Cross-Silo FT 10% | CIFAR100 Cross-Silo LP | CIFAR100 Cross-Device LP |
|---|---|---|---|---|---|
| FedAlignUniform | 80.84 | 69.99 | 81.00 | 57.25 | 43.03 |
| FedX | 78.40 | 66.78 | 80.01 | 57.34 | 43.07 |
| FedDecorr | 80.13 | 69.09 | 80.33 | 57.25 | 44.74 |
| FedU2 | 81.01 | 69.62 | 81.01 | 57.40 | 42.90 |
| **SSD** | **81.32** | **70.74** | **81.67** | **57.38** | **45.21** |

### Ablation Study

| Configuration | LP | FT 1% | FT 10% | $-\mathcal{L}_{\text{uniform}} (\uparrow)$ |
|---|---|---|---|---|
| FedAlignUniform (baseline) | 80.84 | 69.99 | 81.00 | 3.79 |
| + PD | 80.74 | 69.78 | 80.71 | 3.80 |
| + DSR | 81.05 | 69.77 | 81.15 | 3.81 |
| + DSR + PD (SSD) | **81.32** | **70.74** | **81.67** | **3.84** |

### Key Findings

1. **PD alone yields negligible improvement**, while DSR alone yields marginal gains; however, their combination produces a significant synergistic effect — uniformity improves substantially, leading to the best overall performance.
2. **OOD generalization**: Under CIFAR100→CIFAR10 and TinyImageNet→CIFAR10 transfer settings, SSD achieves the highest LP accuracy (78.48% / 80.00%) and best effective rank (86.95 / 98.45).
3. **Robustness to scaling factor**: Performance remains stable across different values of $\alpha$ and is insensitive to the random selection of scaled dimensions.
4. **Removing the projector hurts performance**: Although DSR can notably improve uniformity without a projector (+0.05), retaining the projector still yields superior overall performance.

## Highlights & Insights

- **Precise problem formulation**: This work is the first to formally decompose uniformity in federated learning into intra-client and inter-client components, explicitly identifying that existing methods only optimize the former.
- **Simple yet effective design**: DSR requires only a lightweight scaling vector distributed at initialization, with no additional communication overhead and full compatibility with privacy constraints.
- **Insight behind Projector Distillation**: Experimental observations reveal a decoupling between embedding-space and representation-space norms, exposing the projector's role as a bottleneck for optimization transfer; PD is designed specifically to address this.

## Limitations & Future Work

- Validation is limited to CIFAR-10/100 and TinyImageNet; experiments on larger-scale datasets (e.g., ImageNet) and more diverse non-IID settings are absent.
- The dimension allocation strategy is simple (uniform partitioning); adaptive allocation based on client data characteristics warrants exploration.
- Only FedAvg aggregation is considered; integration with more advanced aggregation strategies remains unexplored.
- The behavior of DSR when the number of clients greatly exceeds the embedding dimensionality has not been verified.

## Related Work & Insights

- SSD is complementary to FedDecorr (local decorrelation) and FedU2 (local uniformity regularization): the latter two address intra-client issues, while SSD targets inter-client uniformity.
- The established importance of the projector in self-supervised learning (Chen et al., Gupta et al.) provides theoretical motivation for the PD module.
- The soft separation concept is generalizable to other federated learning scenarios requiring a trade-off between shared and client-specific representation spaces.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to formally identify and address inter-client uniformity; the DSR+PD design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive analysis covering ablations, robustness, soft/hard separation comparisons, and OOD generalization; dataset scale is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, derivations are rigorous, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Provides a new optimization perspective and a practical solution for federated unsupervised learning.

## Related Papers

- [\[ICCV 2025\] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)
- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[ECCV 2024\] Simple Unsupervised Knowledge Distillation With Space Similarity](../../ECCV2024/model_compression/simple_unsupervised_knowledge_distillation_with_space_similarity.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[ICCV 2025\] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)

</div>

<!-- RELATED:END -->
