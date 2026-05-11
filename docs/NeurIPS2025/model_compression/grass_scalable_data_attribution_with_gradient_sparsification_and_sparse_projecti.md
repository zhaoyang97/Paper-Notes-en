---
title: >-
  [Paper Note] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection
description: >-
  [NeurIPS 2025][Model Compression][Data Attribution] GraSS and FactGraSS are proposed as a two-stage gradient compression algorithm that exploits the inherent sparsity of per-sample gradients to achieve **sublinear** time…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Data Attribution"
  - "Gradient Compression"
  - "Sparse Projection"
  - "Influence Functions"
  - "Random Projection"
date: 2026-05-08
content_hash: 65425eac3f12e11e
---

# GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection

**Conference**: NeurIPS 2025
**arXiv**: [2505.18976](https://arxiv.org/abs/2505.18976)
**Code**: [GitHub](https://github.com/TRAIS-Lab/GraSS)
**Area**: Model Compression / Data Attribution
**Keywords**: Data Attribution, Gradient Compression, Sparse Projection, Influence Functions, Random Projection

## TL;DR

GraSS and FactGraSS are proposed as a two-stage gradient compression algorithm that exploits the inherent sparsity of per-sample gradients to achieve **sublinear** time and space complexity ($O(k')$), outperforming the SOTA baseline LoGra by **165%** in throughput on billion-parameter models while maintaining data attribution quality.

## Background & Motivation

Gradient-based data attribution methods (e.g., influence functions) require computing per-sample gradients $g_i = \nabla_\theta \ell(z_i; \hat{\theta})$ for each training sample, followed by inverse Fisher vector products (iFVP). For a model with $n$ training samples and $p$ parameters, the storage complexity is $O(np)$, posing a significant bottleneck for large models.

**Limitations of Prior Work**:
- **Dense Random Projection**: Provides Johnson-Lindenstrauss guarantees, but incurs projection cost $O(kp)$
- **FJLT** (used in Trak): $O((p+k)\log p)$, but does not exploit input sparsity
- **LoGra**: Exploits the Kronecker product structure of linear layer gradients, reducing complexity to $O(\sqrt{pk})$; current SOTA

**Core Observation**: Per-sample gradients are inherently highly sparse (due to activations such as ReLU), a property absent in mini-batch gradients, yet no existing method explicitly leverages this sparsity.

## Method

### Stage 1: Sparse Projection (SJLT)

The projection matrix $P$ is sparsified so that each column retains only $s = o_\epsilon(k)$ nonzero entries. When the input $g$ is itself sparse, the SJLT complexity reduces to:

$$O(s \cdot \text{nnz}(g))$$

where $\text{nnz}(g) = \|g\|_0$ denotes the number of nonzero elements. The authors set $s=1$ to maximize speed and develop a dedicated CUDA kernel to address thread contention and irregular memory access.

### Stage 2: Sparsification (Mask)

- **Random Mask (RM)**: Randomly selects $k$ coordinates; complexity $O(k)$
- **Selective Mask (SM)**: Selects informative coordinates via differentiable optimization:

$$S^* = \arg\max_{S \in \mathbb{R}^p} \mathbb{E}_{z_{\text{test}}} \left[\text{corr}\left((\langle g_i, g_{z_{\text{test}}}\rangle), (\langle \hat{g}_i, \hat{g}_{z_{\text{test}}}\rangle)\right)\right] - \lambda \|\sigma(S)\|_1$$

where $\hat{g}_i = \sigma(S) \odot g_i$ and $\ell_1$ regularization promotes a binary mask.

### GraSS: Two-Stage Combination

1. **Sparsification**: Reduces $p$-dimensional gradients to $k'$ dimensions ($k < k' \ll p$)
2. **Sparse Projection**: Projects the $k'$-dimensional vector to the target dimension $k$ via SJLT

Total complexity $O(k' + k') = O(k')$, sublinear in $p$.

### FactGraSS: Variant for Linear Layers

Directly combining GraSS with LoGra is problematic: LoGra decomposes the projection into two smaller subproblems, where SJLT is slower than dense projection at small scales. FactGraSS resolves this in three steps:

1. **Factored Sparsification**: Separately sparsify the input activations $z_{i,l}^{\text{in}}$ and output gradients $\mathcal{D}z_{i,l}^{\text{out}}$ to $k_l^{\text{in}'}$ and $k_l^{\text{out}'}$ dimensions
2. **Reconstruction**: Construct a $k_l' = k_l^{\text{in}'} \times k_l^{\text{out}'}$-dimensional "sparsified gradient" via the Kronecker product
3. **Sparse Projection**: Project the reconstructed result to $k_l$ dimensions via SJLT

**No materialization of the full gradient is required**, with total complexity $O(k_l')$. FactGraSS is theoretically faster than LoGra when $c \leq \sqrt{p_l/k_l}$ (where $k_l' = ck_l$).

| Method | Type | Complexity |
|--------|------|------------|
| Gauss | Baseline | $O(pk)$ |
| FJLT | Baseline | $O((p+k)\log p)$ |
| LoGra | Baseline (linear layers) | $O(\sqrt{p_l k_l})$ / layer |
| GraSS | Ours | $O(k')$ |
| FactGraSS | Ours (linear layers) | $O(k_l')$ / layer |

## Key Experimental Results

### Small-Scale Quantitative Evaluation (LDS)

**MLP + MNIST** (Trak framework, $k=4096$):

| Method | LDS | Compression Time (s) |
|--------|-----|----------------------|
| Gauss | 0.4253 | 8.74 |
| FJLT | 0.4359 | 4.33 |
| SJLT | **0.4280** | 0.52 |
| RM | 0.4054 | **0.15** |
| SM | 0.4163 | **0.13** |

**GPT2-small + WikiText** (Influence Functions, $k_l = 64 \times 64$):

| Method | LDS | Efficiency |
|--------|-----|------------|
| LoGra | 0.348 | Baseline |
| SJLT | 0.354 | Slower |
| Mask | 0.340 | Very fast |
| **FactGraSS** | **0.352** | **250% speedup over LoGra** |

### Large-Scale Efficiency Evaluation

On **Llama-2-7B** (7 billion parameters) + C4 dataset:
- FactGraSS achieves **165% higher** compression throughput than LoGra
- Significantly reduced memory footprint, supporting larger batch sizes

### Key Findings

1. Random Mask alone achieves non-trivial LDS; Selective Mask yields further improvements
2. SJLT outperforms dense projection in both speed and accuracy at large problem sizes, but requires FactGraSS to circumvent inefficiencies at small sizes
3. GraSS achieves the best efficiency–accuracy trade-off

## Highlights & Insights

- **Sparsity as a free lunch**: The natural sparsity of per-sample gradients has been overlooked by all prior methods; exploiting it yields order-of-magnitude speedups
- **Dedicated CUDA kernel**: Resolves race conditions and irregular memory access inherent to SJLT in PyTorch
- **Sublinear theoretical guarantee**: $O(k')$ complexity is independent of the model parameter count $p$, enabling theoretically unbounded scalability
- **FactGraSS elegantly avoids dual bottlenecks**: It neither materializes the full gradient ($O(p)$) nor suffers from SJLT inefficiency at small problem sizes

## Limitations & Future Work

1. The SJLT CUDA kernel is critical to the method's success; applicability to non-GPU hardware remains unknown
2. Selective Mask incurs a one-time optimization overhead (solving Eq. 1); scalability to large models requires further investigation
3. The sparsity assumption depends on activations such as ReLU; sparsity levels for GELU/SiLU (common in modern LLMs) may differ
4. FactGraSS applies only to linear layers; general GraSS must be used for nonlinear operations such as attention
5. Comparisons with data attribution frameworks beyond TRAK (e.g., DataInf) are not provided

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic exploitation of per-sample gradient sparsity for data attribution acceleration
- **Technical Depth**: ⭐⭐⭐⭐ — Elegant integration of SJLT, Mask, and Kronecker decomposition
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative and efficiency evaluation from MLP to 7B-parameter models
- **Practicality**: ⭐⭐⭐⭐⭐ — Open-source, general-purpose, and large-model-friendly
- **Overall**: ⭐⭐⭐⭐

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Ensemble++: Scalable Exploration via Ensemble](scalable_exploration_via_ensemble.md)
- [\[NeurIPS 2025\] SpecAttn: Speculating Sparse Attention](specattn_speculating_sparse_attention.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization](tighter_cmi-based_generalization_bounds_via_stochastic_projection_and_quantizati.md)
- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
