---
title: >-
  [Paper Note] ToMA: Token Merge with Attention for Diffusion Models
description: >-
  [ICML2025][Image Generation][token merging] This paper proposes ToMA, which reformulates token merging as a submodular optimization problem and implements merge/unmerge via attention-like linear transformations. This makes it compatible with GPU-optimized schemes like FlashAttention, achieving actual end-to-end speedups of 24% and 23% on SDXL and Flux, respectively, with negligible image quality degradation (DINO $\Delta < 0.07$).
tags:
  - "ICML2025"
  - "Image Generation"
  - "token merging"
  - "diffusion model acceleration"
  - "submodular optimization"
  - "GPU-aligned efficiency"
  - "training-free"
date: 2026-05-08
content_hash: cf33d1cd16894e38
---

# ToMA: Token Merge with Attention for Diffusion Models

**Conference**: ICML2025  
**arXiv**: [2509.10918](https://arxiv.org/abs/2509.10918)  
**Code**: [github.com/WenboLuu/ToMA](https://github.com/WenboLuu/ToMA)  
**Area**: Image Generation  
**Keywords**: token merging, diffusion model acceleration, submodular optimization, GPU-aligned efficiency, training-free

## TL;DR

This paper proposes ToMA, which reformulates token merging as a submodular optimization problem and implements merge/unmerge via attention-like linear transformations. This makes it compatible with GPU-optimized schemes like FlashAttention, achieving actual end-to-end speedups of 24% and 23% on SDXL and Flux, respectively, with negligible image quality degradation (DINO $\Delta < 0.07$).

## Background & Motivation

### Background

Diffusion Models have made breakthrough progress in the field of high-fidelity image generation. However, their core Transformer architecture faces an $O(N^2)$ quadratic complexity bottleneck due to the self-attention mechanism. As the number of tokens grows, inference latency is scaled up exponentially during the multi-step denoising process.

### Limitations of Prior Work

Existing acceleration methods can be divided into two main categories:

**Attention Optimization**: Methods such as FlashAttention and xFormers optimize attention computation through hardware-aware memory access patterns, which are already approaching hardware efficiency limits.

**Token Reduction**: Plug-and-play methods like ToMeSD and ToFu reduce FLOPs by merging redundant tokens. However, they suffer from a key limitation—their merge/unmerge operations rely on GPU-unfriendly primitives (such as sorting, scatter-writes, etc.). The introduced overhead **counteracts the theoretical speedup** when used in conjunction with highly efficient attention implementations like FlashAttention.

### Key Insight

When attention computation is optimized close to hardware limits by FlashAttention, the bottleneck shifts from attention computation to the token merging operation itself. At this stage, the merging overhead of prior methods (e.g., ToMeSD) becomes dominant, making actual acceleration unachievable. This constitutes a **gap between theoretical FLOPs reduction and actual wall-clock speedup**.

## Method

### Overall Architecture

ToMA (Token Merge with Attention) is a **training-free** plug-and-play framework that consists of three core stages:

1. **Facility Location Algorithm (Destination Token Selection)**: Selects a representative subset $D \subset N$ from all $N$ tokens via submodular optimization to maximize representation diversity.
2. **Attention-based Merge**: Constructs a low-rank attention matrix to transform $N \rightarrow D$ through linear operations, executing Self-Attention, Cross-Attention, and MLP within the reduced space.
3. **Inverse Unmerge**: Recovers full-resolution features by Transforming $D \rightarrow N$ via a pseudo-inverse operation.

The entire workflow is efficiently executed through localized processing (local windows) and parallel batch optimization.

### Key Designs

#### Design 1: Submodular Optimization-driven Token Selection

ToMA models token merging as a **Facility Location Problem**, which is a classic submodular maximization problem. The objective function satisfies the **property of diminishing returns**, and a greedy algorithm provides a guaranteed $(1 - 1/e)$ approximation ratio. This ensures that the selected set of destination tokens covers all token information in an approximately optimal manner. Compared to the heuristic matching of ToMeSD, this design provides **theoretical guarantees**.

A submodular function is defined on the subsets of a ground set $V$, satisfying: for any $A \subseteq B \subseteq V$ and element $v \in V \setminus B$, we have $f(v|A) \ge f(v|B)$, where $f(v|A) = f(A \cup \{v\}) - f(A)$ is the marginal gain.

#### Design 2: Attention-like Linear Transformation for Merge/Unmerge

- **Merge**: Constructs an attention weight matrix $M \in \mathbb{R}^{D \times N}$ to aggregate tokens via matrix multiplication, which is equivalent to the linear transformation of the attention mechanism.
- **Unmerge**: Recovers the original token dimension using the pseudo-inverse $M^+$ of $M$.

This design makes full use of **batched matrix multiplication (batched matmul)** on the GPU, avoiding GPU-unfriendly operations like sorting and scatter-writes, and is fully compatible with FlashAttention.

#### Design 3: Exploiting DMs' Inherent Properties to Reduce Overhead

ToMA leverages two inherent properties of diffusion models to further reduce computational overhead:

| Property | Description | Acceleration Strategy |
|------|------|----------|
| **Latent Space Locality** | Tokens in the latent space exhibit spatial local correlation | Parallelize merge operations within non-overlapping local windows (e.g., $8 \times 8$ patches) to reduce optimization scale |
| **Sequential Redundancy** | Merge patterns are highly similar across adjacent denoising steps and consecutive Transformer layers | Reuse merge patterns across time steps and layers to amortize optimization overhead |

These two strategies significantly decrease the invocation frequency of the Facility Location algorithm, which, combined with local window partitioning, keeps the token scale of each optimization controllable.

### Core Differences from Prior Work

| Dimension | ToMeSD | ToFu | **ToMA** |
|------|--------|------|----------|
| Token Selection | Heuristic greedy matching | Switch merge/prune based on linearity test | Submodular optimization (theoretical guarantee) |
| Merge Operation | Unweighted average + sorting | Same as ToMeSD | Attention-like matrix multiplication |
| Unmerge Operation | Copy destination embedding | Same as above | Pseudo-inverse linear transformation |
| GPU Friendliness | Low (sorting, scatter-writes) | Low | High (batched matmul) |
| Compatibility with FlashAttention | Overhead cancels speedup benefit | Similar | **Fully compatible, achieving actual acceleration** |
| Theoretical Guarantee | None | None | Submodular approximation ratio $(1-1/e)$ |

## Key Experimental Results

### Main Results

| Model | Method | Speedup | DINO $\Delta$ | Note |
|------|------|--------|--------|------|
| SDXL-base | Original | 1.00$\times$ | — | baseline |
| SDXL-base | +FlashAttention2 | — | — | Attention optimized |
| SDXL-base | +ToMeSD (on FA2) | $\approx$1.00$\times$ | — | Overhead cancels speedup, **no actual gain** |
| SDXL-base | **+ToMA (ratio=0.5, on FA2)** | **1.24$\times$** | **<0.07** | 24% actual end-to-end speedup |
| Flux.1-dev | **+ToMA** | **1.23$\times$** | **<0.07** | 23% actual speedup |

### Cross-GPU Architecture Validation

| GPU Architecture | Supported | Note |
|----------|----------|------|
| NVIDIA RTX 6000 | ✅ | Main experimental platform |
| NVIDIA V100 | ✅ | Evaluates cross-generational compatibility |
| NVIDIA RTX 8000 | ✅ | Evaluates cross-generational compatibility |

ToMA achieves SOTA speedup on a variety of GPU architectures, demonstrating that its GPU-aligned design possesses excellent hardware generalization.

### Key Findings

- **ToMeSD Failure Case**: When paired with FlashAttention2, the sorting and scatter-write overhead of ToMeSD dominates the computation time, resulting in actual speeds that can be even lower than the baseline FA2 without token reduction.
- **Quality Retention**: At a merge ratio of 0.5 (i.e., reducing 50% of tokens), ToMA changes the DINO score by $< 0.07$, yielding almost imperceptible degradation in image quality.
- **Actual vs. Theoretical Acceleration**: A key finding emphasized in this work is that theoretical FLOPs reduction does not equate to actual wall-clock speedup. The coordination between system-level optimization and algorithm design is of paramount importance.

## Highlights & Insights

1. **Profound Problem Insight**: The authors accurately identify the overlooked key issue of "theoretical FLOPs reduction $\ne$ actual speedup", pointing out that once attention is optimized by FlashAttention, the GPU-unfriendliness of merging operations becomes the new bottleneck.
2. **Algorithm-System Co-design**: This work tightly couples the theoretical guarantees of submodular optimization with GPU execution paradigms (batched matmul, local window parallelism), achieving a complete loop from theory to practice.
3. **Training-free, Plug-and-play**: It requires no retraining and can be directly embedded into existing diffusion model inference pipelines, demonstrating high practicality.
4. **Solid Theoretical Foundation**: Based on the $(1 - 1/e)$ approximation guarantee of submodular optimization, the method boasts a more solid theoretical foundation compared to heuristic methods.
5. **Dual Redundancy Exploitation**: The design simultaneously leverages both latent space locality and sequential redundancy—two inherent properties of diffusion models—to maximize the reduction of amortized merge overhead.

## Limitations & Future Work

1. **Limited Cached Content**: The full-text cache of the paper only covers up to the Preliminaries section. Detailed ablation studies and broader quantitative comparisons were unavailable, which might omit some experimental details.
2. **Fixed Merge Ratio**: The ratio of $0.5$ shown in the paper is static; whether it supports adaptive dynamic scaling is not fully discussed.
3. **Single Evaluation Metric**: It primarily relies on the DINO score to evaluate image quality, lacking more comprehensive generation quality evaluations such as FID or CLIP scores (which could be in the full paper but were missing from the cache).
4. **Local Window Partitioning**: It remains to be explored whether the fixed $8 \times 8$ window size is optimal, or if adaptive window size adjustment is needed under different resolutions.
5. **Video Diffusion Models**: The method is currently validated only on image diffusion models. Its extensibility to video diffusion models (e.g., Sora-like architectures) is worth exploring.
6. **Combination with Quantization/Distillation**: While the paper mentions that ToMA is compatible with orthogonal approaches, the combined effects with compression techniques like quantization and knowledge distillation are not deeply explored.

## Related Work & Insights

### Efficient Vision Transformers

- **Compact Architectures**: Architectures like Swin Transformer and PVT reduce complexity through structural design, but require retraining.
- **Pruning Strategies**: Methods like X-Pruner accelerate inference by removing redundant structures, but require post-training fine-tuning.
- **Knowledge Distillation**: Methods like DeiT transfer knowledge from large models to smaller ones.
- **Post-Training Quantization**: Lowers weight/activation precision to reduce computation.

### Token Reduction Methods

- **Learned Methods**: DynamicViT (generating pruning masks via MLP) and A-ViT (halting probability) require extra training.
- **Heuristic Methods**: ATS (reliant on class tokens, not applicable to generative tasks), ToDo (only downsampling KV, with limited effect), ToMeSD (GPU-unfriendly greedy matching), and ToFu (dynamically switching between merge and prune).

### Insights

The core insight of ToMA is that **the design of acceleration methods must align with the underlying hardware execution model**. If theoretical FLOPs reduction relies on GPU-unfriendly operations, it can actually backfire in engineering practice. This approach holds significant reference value for token reduction design in other domains, such as LLM inference acceleration and video model acceleration.

## Rating

- Novelty: ⭐⭐⭐⭐ — Reformulates token merging as a submodular optimization and implements it with attention-like transformations, presenting a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models and GPU architectures, though complete ablation data is missing in the cache.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, presenting a cohesive theory-system-experiment trinity.
- Value: ⭐⭐⭐⭐⭐ — Resolves the practical engineering pain point of incompatibility between token reduction and highly efficient attention, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](../../NeurIPS2025/image_generation/token_perturbation_guidance_for_diffusion_models.md)
- [\[CVPR 2025\] Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning](../../CVPR2025/image_generation/decouple-then-merge_finetune_diffusion_models_as_multi-task_learning.md)
- [\[CVPR 2026\] Guiding Token-Sparse Diffusion Models](../../CVPR2026/image_generation/guiding_token-sparse_diffusion_models.md)
- [\[ICML 2025\] Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots](hierarchical_masked_autoregressive_models_with_low-resolution_token_pivots.md)

</div>

<!-- RELATED:END -->
