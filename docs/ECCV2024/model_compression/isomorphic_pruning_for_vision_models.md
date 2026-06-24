---
title: >-
  [Paper Note] Isomorphic Pruning for Vision Models
description: >-
  [ECCV 2024][Model Compression][Structured Pruning] This paper proposes Isomorphic Pruning, which models network sub-structures as graphs and groups them by graph isomorphism. By ranking and pruning independently within each isomorphic group, it addresses the problem of incomparable importance among heterogeneous sub-structures, outperforming specially designed pruning methods on both ViTs and CNNs.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Structured Pruning"
  - "Vision Transformer"
  - "CNN"
  - "Graph Isomorphism"
  - "Sub-structure Grouping"
date: 2026-05-08
content_hash: b77fee026f842368
---

# Isomorphic Pruning for Vision Models

**Conference**: ECCV 2024  
**arXiv**: [2407.04616](https://arxiv.org/abs/2407.04616)  
**Code**: Yes ([https://github.com/VainF/Isomorphic-Pruning](https://github.com/VainF/Isomorphic-Pruning))  
**Area**: Model Compression  
**Keywords**: Structured Pruning, Vision Transformer, CNN, Graph Isomorphism, Sub-structure Grouping

## TL;DR

This paper proposes Isomorphic Pruning, which models network sub-structures as graphs and groups them by graph isomorphism. By ranking and pruning independently within each isomorphic group, it addresses the problem of incomparable importance among heterogeneous sub-structures, outperforming specially designed pruning methods on both ViTs and CNNs.

## Background & Motivation

### Background

Structured pruning reduces the computational overhead of deep neural networks by removing redundant sub-structures. The standard pipeline is "rank-and-prune": evaluating the relative importance of different sub-structures using an importance criterion, removing the least important parts after ranking, and finally fine-tuning to recover performance.

### Core Problem: Incomparable Importance Among Heterogeneous Sub-structures

Classic networks (e.g., VGG) are composed of stacked similar convolutional layers, where global ranking reliably reveals relative importance. However, modern vision models (Vision Transformers, ConvNeXt) contain various heterogeneous structures such as self-attention, depth-wise convolutions, and residual connections:

- Even within the same multi-head attention layer, there exist two pruning strategies: pruning the entire head versus pruning the head dimension.
- The parameter sizes, weight distributions, and computational topologies of these heterogeneous sub-structures differ drastically.
- **Direct global ranking leads to biased pruning**: For example, in DeiT, the importance scores of MLP layers are concentrated at higher values, while those of attention layers are clustered near 0. A naive 50% global pruning would remove almost all attention parameters while retaining all embedding parameters.

### Key Observation

Under the same importance criterion, the importance distributions of heterogeneous sub-structures exhibit significant divergence, whereas isomorphic sub-structures (structures sharing the same computational topology) show similar importance distribution patterns. This inspires the idea of performing independent ranking within isomorphic structure groups.

## Method

### Overall Architecture

Isomorphic Pruning consists of three steps: (1) decomposing the network into a set of minimal independent sub-structures and modeling them as graphs; (2) clustering sub-structures into isomorphic groups via graph isomorphism detection; and (3) independently ranking and pruning within each isomorphic group.

### Key Designs

#### 1. Graph Modeling of Sub-structures

**Function**: Map the prunable sub-structures in the network into graphs, where nodes represent pruning operation triplets $(W, d, k)$ (parameter matrix, pruning axis, index), and edges represent dependency relationships.

**Mechanism**: Define a pruning function $\hat{w} = g(W, d, k)$, where $d \in \{0, 1\}$ represents pruning the output dimension or input dimension, and $k$ is the $k$-th row/column. For adjacent layers $W_1$ and $W_2$, the output dimension of $W_1$ is coupled with the input dimension of $W_2$, meaning they must be pruned simultaneously. Dependency modeling follows two rules:

- Coupling exists between the same dimension indices of adjacent layers.
- Different pruning operations on the same parameter matrix are coupled if they yield the same slices (e.g., normalization layers).

Starting from seed parameters, dependencies are recursively discovered to construct the complete sub-structure graph $G=(V, E)$.

**Design Motivation**: Graph representation naturally captures the coupling relationships and computational topological characteristics among the internal parameters of sub-structures, providing a formal foundation for subsequent isomorphism determination.

#### 2. Ranking with Graph Isomorphism

**Function**: Determine which sub-structure graphs are isomorphic (sharing the same computational topology) and cluster isomorphic sub-structures into groups.

**Mechanism**: The conditions for two graphs $G$ and $G'$ to be isomorphic (after edges are ordered by forward execution sequence) is defined as:

$$\text{Isomorphic}(G, G') = \mathbb{1}\{|G|=|G'| \land \text{label}(V_i)=\text{label}(V'_i); \forall i\}$$

where two node labels are identical if and only if they originate from the same layer type (Linear/Conv) and have the same pruning dimension. The index $k$ is allowed to be different, because different slices of the same parameter matrix along the same dimension are naturally isomorphic.

**Design Motivation**: Isomorphic sub-structures share the same architectural design, parameter scale, and computational graph, making importance comparisons within the same group meaningful and reliable.

#### 3. Grouped Ranking and Pruning

**Function**: Independently compute aggregated importance scores within each isomorphic group, rank them, and prune proportionally.

**Mechanism**: For an isomorphic group $\{G_1, G_2, \ldots, G_N\}$, the aggregated importance of each sub-structure is calculated as:

$$I^*(G(V,E)) = \sum_{(W,d,k) \in V} I(W, d, k)$$

where $I(W,d,k)$ can be any single-layer importance criterion (such as Magnitude Pruning based on L1 norm or gradient-based Taylor Pruning):

$$I(W,d,k) = \left\|\frac{\partial \mathcal{L}}{\partial g(W,d,k)} \cdot g(W,d,k)\right\|_2 \quad (\text{Taylor})$$

Within each isomorphic group, the $N$-dimensional importance vector is ranked, and the least important $p\%$ is removed.

**Design Motivation**: Parameter scales and computational patterns are consistent within the same group, allowing for fair comparison of aggregated importance scores; cross-group comparisons would be biased due to topological differences, which is avoided through isolated ranking.

### Loss & Training

- **Pruning Phase**: Utilizing the Taylor criterion, 100 mini-batches (64 images per batch) are randomly sampled, and gradients are accumulated to perform one-shot pruning without iterative pruning fine-tuning.
- **Fine-tuning Phase**: The original training protocol of each model is followed. DeiT is distilled using RegNetY for 300 epochs, utilizing the AdamW optimizer with a learning rate of 0.0005, cosine annealing, and a weight decay of 0.05.
- ConvNeXt/ResNet also conform to their original training settings.

## Key Experimental Results

### Main Results (Vision Transformer Pruning)

On ImageNet-1K, different-sized models are pruned from the pre-trained DeiT-Base:

| Method | Params(M) | MACs(G) | Top-1 Acc(%) |
|------|-----------|---------|-------------|
| DeiT-B (Original) | 87.34 | 17.69 | 83.32 |
| DeiT-S (Original) | 22.44 | 4.64 | 81.17 |
| NViT-S | 21.00 | 4.20 | 82.19 |
| SAViT | 25.40 | 5.30 | 81.66 |
| **DeiT-S (Ours)** | **20.69** | **4.16** | **82.41** |
| DeiT-T (Original) | 5.91 | 1.27 | 74.52 |
| NViT-T | 6.90 | 1.30 | 76.21 |
| X-Pruner | - | 2.40 | 78.93 |
| **DeiT-T (Ours)** | **5.74** | **1.21** | **77.50** |
| DeiT-0.6G (Ours) | 3.08 | 0.62 | 72.60 |

Pruning from DeiT-B to the DeiT-T scale improves the accuracy from the original 74.52% to **77.50%** (+2.98%) with lower MACs.

### Main Results (CNN Pruning)

| Method | Params(M) | MACs(G) | Base Acc | Final Acc | Δ Acc |
|------|-----------|---------|----------|-----------|-------|
| ConvNext-T (Original) | 28.59 | 4.47 | - | 82.06 | - |
| **ConvNext-T (Ours)** | **25.32** | **4.19** | 83.83 | **82.19** | -1.64 |
| ResNet-101 (Original) | 44.55 | 7.85 | - | 77.38 | - |
| **Res101-4.5G (Ours)** | **29.14** | **4.48** | 77.38 | **77.56** | **+0.16** |
| **Res101-3.8G (Ours)** | **24.87** | **3.85** | 77.38 | **77.43** | **+0.05** |
| ResNet-50 (Original) | 25.56 | 4.13 | - | 76.13 | - |
| Res50-2G (Ours) | 15.05 | 2.06 | 76.13 | 75.91 | -0.22 |

**Lossless compression** is achieved on ResNet-101 (where accuracy even increases under 4.5G and 3.8G settings).

### Ablation Study

Comparison of different importance criteria on DeiT-Base:

| Pruning Strategy | Criterion | Accuracy Trend | Note / Description |
|----------|------|----------|------|
| Global + Taylor | Taylor | Baseline | Traditional global ranking |
| **Isomorphic + Taylor** | Taylor | **Consistent Improvement** | Outperforms global at all pruning rates |
| Global + Hessian | Hessian | Baseline | Second-order information |
| **Isomorphic + Hessian** | Hessian | **Consistent Improvement** | Good generalizability of isomorphic grouping |
| Global + L1 | L1 Norm | Baseline | Simplest criterion |
| **Isomorphic + L1** | L1 Norm | **Consistent Improvement** | Even the simplest criterion benefits from this |

Isomorphic Pruning consistently outperforms the corresponding global/local pruning baselines across all pruning criteria and rates.

### Real Latency Analysis

| Model | GPU Latency(ms) | Speedup | CPU Latency(ms) | Speedup | Acc(%) |
|------|------------|--------|------------|--------|--------|
| DeiT-B | 802.97 | 1.00× | 197.73 | 1.00× | 83.32 |
| DeiT-S (Ours) | 230.84 | 3.48× | 44.10 | 4.48× | 82.41 |
| EfficientFormer-L3 | 249.05 | 3.22× | 167.16 | 1.18× | 82.40 |

The performance of pruned DeiT-S (82.41%) is comparable to EfficientFormer-L3 (82.40%), but its CPU latency is only 26% of the latter.

### Key Findings

1. **Severe divergence in heterogeneous sub-structures' importance distribution**: In DeiT, the Taylor score distributions of MLP and Attention parameters barely overlap, verifying the unreliability of direct global comparison.
2. **Bias of naive global pruning**: A 50% global threshold almost completely removes attention layers (scores clustered near 0) while entirely keeping embedding layers.
3. **Greater gain on ViTs than CNNs**: Due to the stronger internal heterogeneity in ViTs.
4. **Memory trade-offs of non-uniform architectures**: Non-uniform pruning achieves higher accuracy but slightly higher peak memory than uniform architectures.

## Highlights & Insights

1. **Extremely simple core idea**: Formalizing the intuition "do not compare the incomparable" via graph isomorphism. The method is simple to implement but highly effective.
2. **Strong architecture-agnosticism**: The same method can be applied to various architectures, including ViT, ConvNeXt, ResNet, and MobileNet, without any modification.
3. **Orthogonal to existing criteria**: As a ranking strategy, it can be combined with any importance criterion such as Taylor, Hessian, or L1.
4. **Significant real-world speedup**: Not only are the theoretical FLOPs reduced, but the measured GPU/CPU latency is also drastically decreased.

## Limitations & Future Work

1. **Fixed pruning granularity**: Each isomorphic group uses the same pruning ratio $p\%$, without exploring adaptive budget allocation across groups.
2. **Dependency on the fine-tuning phase**: It requires a long fine-tuning recovery period (e.g., DeiT 300 epochs); the accuracy drop is substantial without fine-tuning after one-shot pruning.
3. **Isomorphism determination may be too strict**: Strict graph isomorphism might produce too many groups, leading to very few samples in some groups, thus affecting ranking reliability.
4. **Dynamic/token pruning is not involved**: It only performs static structured pruning and lacks integration with dynamic token pruning.

## Related Work & Insights

- **DepGraph (CVPR 2023)**: The authors' prior work, which modeled dependency relationships but did not consider isomorphic grouping.
- **NViT (NeurIPS 2022)**: Learns target structures via sparse training, which is computationally expensive.
- **UPop (ICLR 2023)**: Unified pruning, but still relies on global ranking.
- Insight: The idea of graph isomorphism grouping can be generalized to the pruning of NLP Transformers and multimodal large models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing graph isomorphism into pruning provides a fresh perspective; the core idea is simple and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple architectures (ViT/CNN), with latency/memory analysis, transfer learning, and multi-criterion ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Formally rigorous, with clear illustrations (especially the importance distribution comparison in Fig.2 and Fig.4c).
- **Value**: ⭐⭐⭐⭐⭐ — The method is simple, general, and highly effective. With open-source code, it has high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Collaborative Multi-Mode Pruning for Vision-Language Models](../../CVPR2026/model_compression/collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](../../CVPR2025/model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[ECCV 2024\] Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning](token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin.md)
- [\[CVPR 2025\] MDP: Multidimensional Vision Model Pruning with Latency Constraint](../../CVPR2025/model_compression/mdp_multidimensional_vision_model_pruning_with_latency_constraint.md)
- [\[ECCV 2024\] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)

</div>

<!-- RELATED:END -->
