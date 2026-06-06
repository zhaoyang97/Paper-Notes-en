---
title: >-
  [Paper Note] Class-Wise Federated Averaging for Efficient Personalization
description: >-
  [ICCV 2025][Optimization][Personalized Federated Learning] cwFedAvg extends FedAvg from client-level aggregation to class-level aggregation…
tags:
  - "ICCV 2025"
  - "Optimization"
  - "Personalized Federated Learning"
  - "Class-Wise Aggregation"
  - "Weight Distribution Regularization"
  - "Data Heterogeneity"
  - "Privacy Preservation"
date: 2026-05-08
content_hash: 62fe67dfec4ef494
---

# Class-Wise Federated Averaging for Efficient Personalization

**Conference**: ICCV 2025
**arXiv**: [2406.07800](https://arxiv.org/abs/2406.07800)  
**Code**: [github.com/regulationLee/cwFedAvg](https://github.com/regulationLee/cwFedAvg)  
**Area**: Federated Learning / Optimization
**Keywords**: Personalized Federated Learning, Class-Wise Aggregation, Weight Distribution Regularization, Data Heterogeneity, Privacy Preservation

## TL;DR

cwFedAvg extends FedAvg from client-level aggregation to class-level aggregation, constructing a dedicated global model per class and combining them into a personalized local model weighted by each client's class distribution. Coupled with Weight Distribution Regularization (WDR) to strengthen the alignment between class distribution and weight norms, the method achieves substantial personalization gains under non-IID settings while maintaining the same communication overhead as FedAvg.

## Background & Motivation

Federated Learning (FL) enables distributed collaborative training via model aggregation, yet FedAvg performs poorly under data heterogeneity (non-IID). The root cause lies in:

**Class-Specific Pathways**: Deep networks encode class information through weight pathways; the dominant pathways (composed of large-magnitude weights) exhibit distinct patterns across classes.

**Limitations of FedAvg**: Its aggregation weights are based solely on each client's total sample count, $p_i = n_i/n$, which fails to reflect class-specific pathway differences. A single global model cannot simultaneously capture the unique patterns of all clients.

Existing Personalized Federated Learning (PFL) approaches suffer from:
- FedFomo/FedAMP: Require downloading other clients' models or pairwise computations, incurring high communication/computation overhead;
- CFL/IFCA: Rely on clustering assumptions (clients must be partitioned into discrete groups);
- FedNH/FedUV: Regularization-based improvements remain limited.

## Method

### Overall Architecture

cwFedAvg employs a two-step aggregation process:
1. **Class-level local model aggregation**: Constructs a dedicated global model for each class;
2. **Class-level global model aggregation**: Combines $K$ class-specific global models into a personalized local model, weighted by each client's class distribution.

### Key Designs

1. **Class-Wise Aggregation**: For the $j$-th class in a $K$-class classification task, the class-level global model is obtained by weighted aggregation of local models:

$$\boldsymbol{w}_j^G = \sum_{i=1}^{M} q_{i,j} \boldsymbol{w}_i^L, \quad q_{i,j} = \frac{p_i \cdot p_{i,j}}{\sum_{i=1}^{M} p_i \cdot p_{i,j}} = \frac{n_{i,j}}{\sum_{i=1}^M n_{i,j}}$$

where $q_{i,j}$ represents client $i$'s proportional contribution to class $j$ across the system. This is equivalent to performing FedAvg separately for each class. The personalized local model is then obtained by combining class-level global models weighted by the class distribution: $\boldsymbol{w}_i^L = \sum_{j=1}^K p_{i,j} \boldsymbol{w}_j^G$.

2. **Weight Distribution Regularization (WDR)**: For cwFedAvg to function effectively, model weights must be strongly correlated with class distributions. Drawing on the theoretical finding of Anand et al.—that the $\ell_2$ norm of output-layer weights correlates positively with the corresponding class sample count—an approximate class distribution is defined as:

$$\tilde{p}_{i,j} = \frac{\|\mathbf{w}_{i,j}\|_2}{\sum_{k=1}^K \|\mathbf{w}_{i,k}\|_2}$$

WDR strengthens this alignment by minimizing the discrepancy between $\tilde{p}_{i,j}$ and the empirical class distribution $p_{i,j}$:

$$\mathcal{R}_i = \|\boldsymbol{p}_i - \tilde{\boldsymbol{p}}_i\|_2$$

The total loss is $\tilde{\mathcal{L}}_i = \mathcal{L}_i + \lambda \mathcal{R}_i$. This simultaneously addresses two issues: (a) enhancing weight–class distribution alignment to improve aggregation quality; (b) allowing the server to use $\tilde{p}_{i,j}$ as a proxy for the true $p_{i,j}$, thereby protecting privacy by avoiding direct exposure of $n_{i,j}$.

3. **Selective Layer Application**: Since lower layers of deep networks learn generic features while higher layers learn class-specific ones, cwFedAvg can apply class-level aggregation only to the output layer (or upper layers), while lower layers continue to use standard FedAvg. This reduces the memory cost of storing $K$ global models on the server. Experiments show that applying cwFedAvg only to the output layer already yields the majority of performance gains.

### Loss & Training

- Communication rounds: 1000;
- Local training: 1 epoch, learning rate 0.005, batch size 10;
- Regularization coefficient $\lambda$: 10 for MNIST/CIFAR-10, 1000 for CIFAR-100, 2000 for Tiny ImageNet;
- All class-level aggregation is performed server-side; communication cost is identical to FedAvg.

## Key Experimental Results

### Main Results

**Pathological setting (2 classes per client)**:

| Method | CIFAR-10 | CIFAR-100 | MNIST |
|--------|----------|-----------|-------|
| FedAvg | 60.68 | 28.22 | 98.70 |
| FedFomo | 90.76 | 63.12 | 99.13 |
| FedAMP | 88.82 | 63.29 | 99.26 |
| FedUV | 88.11 | 62.72 | 99.25 |
| **cwFedAvg (Output)** | **91.23** | **67.50** | **99.52** |

**Practical setting (Dirichlet distribution, α=0.1)**:

| Method | CIFAR-10 | CIFAR-100 | Tiny ImageNet | Tiny ImageNet* (ResNet-18) |
|--------|----------|-----------|---------------|---------------------------|
| FedAvg | 61.94 | 32.44 | 21.35 | 24.71 |
| FedAMP | 89.46 | 47.65 | 29.95 | 31.38 |
| CFL | 61.40 | 44.19 | 29.62 | 33.47 |
| **cwFedAvg (Output)** | **88.65** | **56.29** | **41.38** | **43.51** |

Gains are particularly pronounced on CIFAR-100 and Tiny ImageNet (+8.64 and +10.13, respectively).

### Ablation Study

| Configuration | CIFAR-100 (α=0.1) | Notes |
|---------------|-------------------|-------|
| FedAvg baseline | 32.44 | No personalization |
| cwFedAvg (w/o WDR) | ~45 | Weak weight–class alignment |
| cwFedAvg (all layers) | ~55 | Full-layer aggregation |
| cwFedAvg (output only) + WDR | **56.29** | Best efficiency–performance trade-off |

**Robustness across client counts and heterogeneity levels (CIFAR-100)**:

| Method | 50 clients | 100 clients | α=0.01 | α=0.5 | α=1.0 |
|--------|-----------|-------------|--------|-------|-------|
| FedAvg | 32.63 | 32.32 | 28.00 | 36.18 | 36.75 |
| FedAMP | 44.97 | 41.37 | 73.46 | 25.41 | 21.23 |
| cwFedAvg | See paper | See paper | Superior | Superior | Superior |

### Key Findings

- Under the IID limit, cwFedAvg provably reduces to FedAvg (theoretically demonstrated); under extreme non-IID conditions, it reduces to intra-class FedAvg (Eq. 9);
- Weight norm heatmaps visually demonstrate the personalization effect of cwFedAvg+WDR: the output-layer weight patterns of each client's model closely align with its data distribution;
- FedAMP exhibits a sharp performance drop as α increases (approaching IID), whereas cwFedAvg remains stable.

## Highlights & Insights

1. **Remarkably simple idea**: Effective personalization is achieved solely by modifying FedAvg's aggregation weight from $p_i$ to $p_i \cdot p_{i,j}$, with minimal algorithmic complexity;
2. **Zero additional communication overhead**: All class-level aggregation is performed server-side; clients still upload and download a single model, keeping communication volume identical to FedAvg;
3. **Privacy-preserving design**: WDR enables the server to infer class distributions from model weights ($\tilde{p}_{i,j}$), eliminating the need for clients to directly transmit sensitive $n_{i,j}$ values;
4. **Closed loop between theory and practice**: The method draws a complete logical chain from neural network pathway theory, through empirical observations on weight–class distribution alignment, to the practical design of WDR.

## Limitations & Future Work

- The server must store $K$ global models (mitigated by applying class-level aggregation only to the output layer);
- $\lambda$ requires manual tuning across datasets (ranging from 10 to 2000), with no adaptive strategy proposed;
- Experiments are limited to 4-layer CNNs and ResNet-18; effectiveness on larger models remains to be validated;
- The theoretical guarantee of WDR is restricted to output-layer weights; its effect on intermediate layers is an empirical assumption based on backpropagation cascade effects.

## Related Work & Insights

- Distinction from FedFomo: FedFomo requires clients to download other clients' models to learn combination weights ($O(M)$ communication cost), whereas cwFedAvg does not;
- The class distribution estimation via WDR can independently serve client selection in federated learning, offering utility beyond personalization;
- The gradient–class size relationship established by Anand et al. constitutes the theoretical foundation of this work.

## Rating

- **Novelty**: ⭐⭐⭐ The idea is clean but not deeply innovative; it is primarily a natural extension of FedAvg
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 4 datasets, multiple heterogeneity levels, varying client counts, pathway visualization
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical motivation is clear; visualizations are convincing
- **Value**: ⭐⭐⭐⭐ Highly practical—simple, efficient, and free of additional communication overhead

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](../../NeurIPS2025/optimization/layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[ICCV 2025\] Memory-Efficient 4-bit Preconditioned Stochastic Optimization](memory-efficient_4-bit_preconditioned_stochastic_optimization.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](federated_continual_instruction_tuning.md)

</div>

<!-- RELATED:END -->
