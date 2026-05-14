---
title: >-
  [Paper Note] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models
description: >-
  [NeurIPS 2025][Model Compression][N:M sparsity] This paper proposes PermLLM, the first learnable channel permutation (LCP) framework for N:M sparse LLMs. By relaxing discrete permutation matrices into differentiable soft…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "N:M sparsity"
  - "channel permutation"
  - "model pruning"
  - "LLM compression"
  - "Sinkhorn normalization"
date: 2026-05-08
content_hash: b944f05b200302ef
---

# PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.10136](https://arxiv.org/abs/2510.10136)
**Code**: [GitHub](https://github.com/lanchengzou/PermLLM)
**Area**: Model Compression
**Keywords**: N:M sparsity, channel permutation, model pruning, LLM compression, Sinkhorn normalization

## TL;DR

This paper proposes PermLLM, the first learnable channel permutation (LCP) framework for N:M sparse LLMs. By relaxing discrete permutation matrices into differentiable soft permutation matrices via Sinkhorn normalization, PermLLM enables end-to-end optimization. Combined with a block-level permutation strategy that substantially reduces computational overhead, the framework effectively improves the performance of N:M sparse LLMs.

## Background & Motivation

The rapid growth in LLM scale poses severe challenges for efficient deployment. Semi-structured pruning (N:M sparsity)—retaining $M-N$ weights and zeroing out $N$ weights in every $M$ consecutive weights—is a highly promising compression paradigm that directly leverages NVIDIA GPU Sparse Tensor Cores to achieve approximately $2\times$ speedup, making it one of the most practical sparse formats available.

Channel permutation is a key technique for improving the accuracy of N:M sparse models: by reordering the input channels of a weight matrix, more important weights can be preserved. However, existing permutation methods suffer from a fundamental flaw:

**Reliance on hand-crafted quality metrics.** For example, RIA uses the sum of retained weight importance scores as a proxy for permutation quality, yet a significant gap exists between this proxy and the actual pruning error. The paper illustrates this issue with a concrete example—**a permutation that maximizes importance scores does not necessarily minimize output error, and may even increase it.** This is because hand-crafted metrics ignore complex interactions among weights and cross-layer dependencies.

A further challenge is the combinatorial explosion of permutations: $C_{in}$ channels admit $C_{in}!$ possible permutations. Even after simplification under the N:M constraint, the number of candidate solutions reaches approximately 2.6 million for $C_{in}=16, M=4$. Exhaustive search is entirely infeasible for the high-dimensional hidden layers in LLMs.

## Method

### Overall Architecture

PermLLM is a post-training pruning framework that integrates seamlessly with existing one-shot pruning methods (Wanda, RIA). The core pipeline is: (1) transform a learnable matrix into a soft permutation matrix via Sinkhorn normalization; (2) harden it into a strict permutation matrix using the Hungarian algorithm; (3) determine the pruning mask with a one-shot method based on the permuted weights; (4) optimize the permutation matrix end-to-end using a cosine similarity loss.

### Key Designs

1. **Soft permutation matrix relaxation**: Permutation matrices $\mathbf{P}$ are discrete binary matrices and are non-differentiable. PermLLM introduces a learnable matrix $\mathbf{W}_P$ and transforms it into a doubly stochastic matrix (rows and columns each sum to 1) via Sinkhorn normalization, yielding a soft permutation matrix $\hat{\mathbf{P}}$:

    $\hat{\mathbf{P}} = S^L(\mathbf{W}_P / \tau)$

   where $S^L$ denotes $L$ iterations of Sinkhorn normalization (alternating row/column normalization), and the temperature $\tau$ decays linearly from 1 to 0.1 to control the "hardness" of the matrix. During the forward pass, the Hungarian algorithm hardens $\hat{\mathbf{P}}$ into a strict permutation matrix $\mathbf{P}$ (by solving a linear assignment problem); during the backward pass, the straight-through estimator (STE) approximates the gradient as $\partial\mathbf{P}/\partial\hat{\mathbf{P}}=1$.

2. **Block-level channel permutation**: Full-matrix permutation requires $C_{in}^2$ parameters, which is impractical for LLMs. PermLLM partitions channels into $N_B$ blocks of size $B$, with each block learning an independent permutation, reducing the parameter count to $C_{in} \times B$ (a $B/C_{in}$ compression factor). The hardening time complexity is also reduced from $O(C_{in}^3)$ to $O(C_{in} \cdot B^2)$. The block permutation matrix takes the form of a block-diagonal matrix $\mathbf{P}_B = \text{diag}(\mathbf{P}_1, \mathbf{P}_2, \ldots, \mathbf{P}_{N_B})$.

3. **Pruning-aware permutation optimization**: Since permuting weights changes their order, the pruning mask changes accordingly. Mask determination uses softmax for a differentiable approximation (backward pass) and argmax for discrete selection (forward pass):

    $\hat{\mathbf{M}}_{i,kM:(k+1)M} = \text{Softmax}(\hat{\mathbf{S}}_{i,kM:(k+1)M})$

   The optimization objective directly minimizes the discrepancy between the outputs of the dense and sparse models:

    $\mathcal{L}_{cosine}(\mathbf{y}, \widetilde{\mathbf{y}}) = 1 - \frac{\mathbf{y} \cdot \widetilde{\mathbf{y}}}{\|\mathbf{y}\| \cdot \|\widetilde{\mathbf{y}}\|}$

   After training, weights are permuted and pruned as $\hat{\mathbf{W}}' = \mathbf{M}^* \odot (\mathbf{W}\mathbf{P}_B^*)$, and the output channels of the preceding layer are reordered accordingly to maintain computational consistency.

### Loss & Training

- Optimizer: AdamW, learning rate $\in$ {1e-3, 5e-3}
- Sinkhorn normalization iterations: 5
- Temperature annealing: linear decay from 1 to 0.1
- Default block size: 64
- Calibration data: 128 samples × 1024 tokens from the C4 dataset
- Training time: approximately 2.5 hours for 7B models (4×A100); approximately 5.5 hours for 13B models (8×A100)
- A custom CUDA kernel achieves an $84\times$ speedup for the channel permutation operation

## Key Experimental Results

### Main Results

**WikiText2 Perplexity (2:4 sparsity, ↓ lower is better)**

| Method | OPT-6.7B | LLaMA-7B | LLaMA-2 7B | LLaMA-3.1 8B | Qwen-2.5 7B |
|--------|----------|----------|------------|-------------|-------------|
| Dense | 10.86 | 5.68 | 5.47 | 6.24 | 7.74 |
| SparseGPT | 14.33 | 11.19 | 11.12 | 16.62 | 14.34 |
| Wanda | 16.29 | 11.59 | 12.16 | 23.42 | 24.44 |
| Wanda+CP | 15.28 | 11.07 | 11.00 | 21.09 | 18.76 |
| **PermLLM_Wanda** | **14.27** | **9.41** | **9.39** | **14.03** | **13.58** |
| RIA | 15.93 | 11.14 | 11.30 | 22.62 | 22.67 |
| RIA+CP | 15.13 | 10.99 | 10.26 | 19.80 | 17.58 |
| **PermLLM_RIA** | **14.23** | **9.95** | **9.60** | **15.79** | **15.93** |

**Zero-shot task average accuracy (2:4 sparsity)**

| Model | Method | HellaSwag | ARC_E | ARC_C | OBQA | RTE | Avg. |
|-------|--------|-----------|-------|-------|------|-----|------|
| LLaMA-2 7B | Dense | 57.13 | 76.30 | 43.26 | 31.60 | 62.45 | 54.15 |
| | Wanda | 41.59 | 61.74 | 30.20 | 24.00 | 53.07 | 42.12 |
| | Wanda+CP | 43.40 | 64.69 | 30.03 | 26.00 | 53.07 | 43.44 |
| | **PermLLM_Wanda** | **46.60** | **65.49** | **31.14** | **26.20** | **63.54** | **46.59** |
| Qwen-2.5 7B | Dense | 58.79 | 79.56 | 46.08 | 33.00 | 76.90 | 58.87 |
| | Wanda | 40.60 | 67.17 | 33.45 | 25.40 | 72.92 | 47.91 |
| | **PermLLM_Wanda** | **47.30** | **70.58** | **38.13** | **27.60** | **77.26** | **52.17** |

### Ablation Study

| Configuration | WikiText2 PPL | Avg. Zero-shot Acc. | Note |
|---------------|--------------|---------------------|------|
| Sinkhorn iters=0 (Qwen-2.5 7B) | 14.12 | 42.96 | Soft permutation deviates from doubly stochastic matrix |
| Sinkhorn iters=5 (Qwen-2.5 7B) | 14.03 | 43.33 | Doubly stochastic constraint aids learning |
| Sinkhorn iters=0 (LLaMA-3.1 8B) | 14.43 | 49.18 | — |
| Sinkhorn iters=5 (LLaMA-3.1 8B) | 13.58 | 52.17 | Significant improvement |

**Inference speedup (LLaMA-2 7B, 2048 tokens)**

| Component | Dense | 2:4 sparse + CP | Speedup |
|-----------|-------|-----------------|---------|
| Q/K/V/O_proj | 1.513ms | 0.927ms | 1.63× |
| Up/Gate_proj | 2.607ms | 1.526ms | 1.71× |
| Down_proj | 2.614ms | 1.535ms | 1.70× |
| Channel permutation overhead | — | 0.039ms | Negligible |

### Key Findings

- PermLLM consistently and significantly outperforms hand-crafted permutation methods (Wanda+CP, RIA+CP) across all evaluated models.
- The gains are especially pronounced on newer models (LLaMA-3.1, Qwen-2.5)—for example, Wanda+CP achieves a PPL of 21.09 on LLaMA-3.1 8B, whereas PermLLM reduces it to 14.03.
- The custom CUDA kernel reduces permutation overhead to just 0.039ms, preserving an overall speedup of approximately $1.67\times$.
- A block size of 64 achieves a favorable balance between accuracy and efficiency.

## Highlights & Insights

- The **core contribution** lies in reformulating the discrete permutation optimization problem as a continuous differentiable optimization problem—the combination of Sinkhorn normalization, the Hungarian algorithm, and STE is particularly elegant.
- The paper directly exposes the fundamental flaw of hand-crafted permutation metrics: maximizing importance scores $\neq$ minimizing output error.
- Block-level permutation is a clever engineering design that reduces parameter count and computational complexity from $O(C_{in}^2)$ and $O(C_{in}^3)$ to $O(C_{in} \cdot B)$ and $O(C_{in} \cdot B^2)$, respectively.
- The framework is designed as a plug-in module and can be combined with any one-shot pruning method.

## Limitations & Future Work

- The post-training optimization still requires several hours of GPU time, which may be insufficiently lightweight for extremely resource-constrained scenarios.
- Block-level permutation restricts cross-block channel reordering, potentially missing the global optimum.
- Only 2:4 and 4:8 sparsity ratios are evaluated; more flexible N:M configurations (e.g., 1:4, 3:8) remain unexplored.
- Whether the linear temperature annealing schedule is optimal has not been verified; alternative scheduling strategies may yield better results.
- The potential benefit of combining PermLLM with methods that jointly update weights (e.g., SparseGPT) warrants further investigation.

## Related Work & Insights

- RIA proposes a two-stage permutation strategy but relies on hand-crafted metrics; PermLLM directly addresses and improves upon this limitation.
- Wanda and SparseGPT serve as the mainstream one-shot pruning baselines.
- SR-STE applies STE to N:M sparsity mask learning; PermLLM adapts a similar idea to permutation learning.
- Insight: Sinkhorn normalization is widely used in other combinatorial optimization problems (e.g., optimal transport, graph matching); this paper demonstrates a novel application in model compression.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first learnable channel permutation framework; the Sinkhorn + Hungarian + STE combination for solving discrete optimization is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 7 models across multiple benchmarks with detailed ablations and inference speed analysis; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated and mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐⭐ — Channel permutation has broad practical utility; open-source code and an $84\times$ CUDA speedup make the contribution highly accessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions](layerif_estimating_layer_quality_for_large_language_models_using_influence_funct.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] A Simple Linear Patch Revives Layer-Pruned Large Language Models](a_simple_linear_patch_revives_layerpruned_large_language_mod.md)

</div>

<!-- RELATED:END -->
