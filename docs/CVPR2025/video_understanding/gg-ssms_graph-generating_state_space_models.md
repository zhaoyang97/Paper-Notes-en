---
title: >-
  [Paper Note] GG-SSMs: Graph-Generating State Space Models
description: >-
  [CVPR 2025][Video Understanding][State Space Models] Proposes Graph-Generating State Space Models (GG-SSMs), which dynamically construct a Minimum Spanning Tree (MST) based on feature similarity to replace the fixed 1D scanning paths in traditional SSMs. This enables efficient modeling of complex non-local dependencies in high-dimensional data, achieving SOTA performance on 11 datasets.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "State Space Models"
  - "Dynamic Graph Construction"
  - "Minimum Spanning Tree"
  - "Visual SSM"
  - "Time Series Forecasting"
date: 2026-05-08
content_hash: cdfa02072d22d889
---

# GG-SSMs: Graph-Generating State Space Models

**Conference**: CVPR 2025  
**arXiv**: [2412.12423](https://arxiv.org/abs/2412.12423)  
**Code**: [https://github.com/uzh-rpg/gg_ssms](https://github.com/uzh-rpg/gg_ssms)  
**Area**: Video Understanding  
**Keywords**: State Space Models, Dynamic Graph Construction, Minimum Spanning Tree, Visual SSM, Time Series Forecasting

## TL;DR

Proposes Graph-Generating State Space Models (GG-SSMs), which dynamically construct a Minimum Spanning Tree (MST) based on feature similarity to replace the fixed 1D scanning paths in traditional SSMs. This enables efficient modeling of complex non-local dependencies in high-dimensional data, achieving SOTA performance on 11 datasets.

## Background & Motivation

Traditional SSMs (such as S4) perform exceptionally well in sequence data modeling but are limited by their fixed 1D sequence processing manner, failing to effectively capture non-local interactions in high-dimensional data (e.g., images, event streams). Although Mamba and VMamba introduce selective scanning and multi-directional scanning strategies, they still rely on predefined 1D scanning trajectories (such as unrolling along several directions on image grids) and cannot adaptively fit the intrinsic structure of the data.

The core challenge lies in: how to design a model that can adaptively capture complex non-local dependencies without introducing high computational costs? Graph models are naturally suited to represent complex relationships, but constructing and processing them on large-scale graphs typically faces high computational complexity. The key insight of GG-SSMs is: integrating **dynamic graph construction** into the SSM framework, using Chazelle's near-linear MST algorithm to maintain efficiency, while achieving adaptive scanning through graph structures driven by feature relationships.

## Method

### Overall Architecture

Given an input feature set $\{x_i\}_{i=1}^{L}$, GG-SSM first constructs a fully connected graph based on dissimilarity between features, then extracts a minimum spanning tree $\mathcal{T}$ using Chazelle's MST algorithm, and finally propagates SSM states along the edges of the spanning tree to obtain enhanced feature representations. The entire process transforms from "fixed scanning" to "data-driven adaptive scanning".

### Key Designs

1. **基于特征不相似度的图构建**:
    - Function: Model input features as a graph structure, where edge weights represent the relationship intensity between nodes
    - Mechanism: Define a fully connected undirected graph $G=(V,E)$, where each node $v_i$ corresponds to feature $x_i$. Edge weights are computed via cosine dissimilarity: $w_{ij} = \exp\left(-\frac{x_i^\top x_j}{\|x_i\|\|x_j\|}\right)$. Then, Chazelle's MST algorithm is used to extract a minimum spanning tree $\mathcal{T}$, where the MST keeps only the $L-1$ most critical edges
    - Design Motivation: The MST preserves the minimum-weight edge set connecting all nodes, capturing the core structural relationships of the data in the most compact way. Chazelle's algorithm has a complexity of $\mathcal{O}(E\alpha(E,V))$, where the inverse Ackermann function $\alpha$ grows extremely slowly, practically equivalent to linear time

2. **沿图的状态传播机制**:
    - Function: Aggregate information on the MST to compute the hidden state of each node
    - Mechanism: Define path weight $S_{ji} = \prod_{m=1}^{n} \bar{A}_{k_m}$, which is the product of all state transition matrices along the path from node $v_j$ to $v_i$. The hidden state of node $v_i$ is $h_i = \sum_{v_j \in V} S_{ji} \bar{B}_j x_j$, aggregating contributions from all other nodes (with decay strength modulated by path weights). The final output of the model is $y_i = C_i \text{Norm}(h_i) + D x_i$
    - Design Motivation: The tree structure of MST guarantees a unique path between any two nodes, avoiding redundant computation. Through the multiplicative decay of path weights, the contributions of distant nodes naturally decay, approximating an "attention" effect under a computational complexity of only $\mathcal{O}(L)$

3. **高效的前向/后向传播**:
    - Function: Guarantee linear computational complexity for training and inference
    - Mechanism: Forward propagation proceeds from leaf nodes to root nodes, with each node executing fixed operations (initializing state, aggregating child states, updating hidden states). Backward propagation proceeds from root to leaves, using dynamic programming to store intermediate results to avoid redundant calculations. The entire architecture relies on local neighborhood information, requiring no global operations
    - Design Motivation: The CUDA implementation of MST makes graph generation extremely fast, with single-pass forward propagation time close to Mamba. The linear complexity ensures scalability to large-scale datasets and high-resolution inputs

### Loss & Training

Training strategies vary by task: ImageNet classification adopts the standard training protocol of VMamba (200 epochs, AdamW, lr=$1\times10^{-3}$, cosine decay), along with data augmentation such as RandAugment, MixUp, and CutMix. Time series forecasting uses MSE/MAE loss, with a lookback length of 96 and prediction lengths of 96/192/336/720.

## Key Experimental Results

### Main Results

**ImageNet-1K Classification**:

| Model | Params (M) | FLOPs (G) | Top-1 Acc (%) |
|------|-----------|----------|-------------|
| DeiT-S | 22 | 4.6 | 79.8 |
| Swin-T | 28 | 4.5 | 81.3 |
| VMamba-T | - | - | 82.6 |
| **GG-SSM-T** | - | - | **83.6** |
| Swin-B | 88 | 15.4 | 83.5 |
| VMamba-B | - | - | 83.9 |
| **GG-SSM-B** | - | - | **84.9** |

**Event Camera Eye Tracking (LPW)**:

| Model | Params (M) | FLOPs (G) | p3 (%) | p5 (%) | p10 (%) |
|------|---------|----------|-------|-------|--------|
| VMamba+Mamba | 0.35 | 8.60 | 89.00 | 98.00 | 99.30 |
| **GG-SSM** | **0.22** | **8.01** | **89.33** | **98.89** | **99.50** |

### Ablation Study

**Time Series Forecasting (Weather Dataset, Average)**:

| Configuration | MSE | MAE | Description |
|------|-----|-----|------|
| GG-SSM | 0.2250 | 0.2623 | Optimal across all horizons |
| S-Mamba | 0.2510 | 0.2760 | Previous best SSM model |
| iTransformer | 0.2580 | 0.2780 | Transformer baseline |
| Crossformer | 0.2590 | 0.3150 | Cross-attention scheme |

### Key Findings

- GG-SSM achieves optimal or near-optimal performance across all 11 datasets, validating the universality of dynamic graph scanning.
- It outperforms VMamba by 1% on ImageNet, proving that data-driven scanning paths are superior to predefined ones.
- In the eye-tracking task, GG-SSM achieves higher accuracy with fewer parameters (0.22M vs 0.35M).
- The MST is implemented based on CUDA, with its forward propagation speed being comparable to Mamba.

## Highlights & Insights

- **Core Innovation**: Organically combines graph construction with SSMs, replacing "handcrafted scanning paths" with a "feature-similarity-driven MST", which is a paradigm shift from "data-independent" to "data-adaptive".
- **Theoretical Elegance**: MST mathematically guarantees connecting all nodes at minimum cost, making it naturally suited as a skeleton for information propagation.
- **Practicality & Efficiency**: Aided by the near-linear complexity of Chazelle's algorithm and its CUDA implementation, GG-SSM matches Mamba in efficiency.
- **Strong Universality**: The same framework is highly effective across vastly different tasks, such as image classification, optical flow estimation, event data processing, and time series forecasting.

## Limitations & Future Work

- Graph construction uses global features to compute dissimilarity, which may still pose scalability issues for high-resolution inputs.
- MST retains only $L-1$ edges, which may lose some useful redundant connections.
- Currently, the graph construction in the spatial dimension is primarily validated, while joint spatial-temporal graph construction for video is not yet fully explored.
- Although the paper claims to focus on "video understanding", the actual main experiments are mostly on image classification and time series, with relatively few video-related experiments (e.g., optical flow).

## Related Work & Insights

- **Relationship with Mamba/VMamba**: GG-SSM can be viewed as a fundamental improvement over VMamba's scanning strategy — transitioning from "handcrafted and fixed" to "data-driven".
- **Graph Neural Network Perspective**: GG-SSM extends the sequence processing of SSMs to graphs, sharing similarities with message passing in GNNs, but maintains sparsity and linear complexity through the MST.
- **Insight**: The approach of dynamic graph construction combined with efficient state propagation can be extended to other scenarios requiring the modeling of complex dependencies (e.g., point clouds, molecular graphs).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of combining dynamic graph construction with the SSM framework is very novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets spanning multiple domains like vision and time series fully validate its universality.
- Writing Quality: ⭐⭐⭐⭐ Overall clear, with rigorous mathematical derivations, though some details could be more compact.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for Visual SSMs, but validation in practical video understanding scenarios still needs to be strengthened.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](../../AAAI2026/video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](../../ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](../../ECCV2024/video_understanding/videomamba_spatio-temporal_selective_state_space_model.md)

</div>

<!-- RELATED:END -->
