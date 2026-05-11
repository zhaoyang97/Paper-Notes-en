---
title: >-
  [Paper Note] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer
description: >-
  [AAAI 2026][Interpretability][Kolmogorov-Arnold Network] This paper provides an in-depth analysis of the root cause behind KAT (Kolmogorov-Arnold Transformer) training being 123× slower than ViT. The bottleneck is identi…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Kolmogorov-Arnold Network"
  - "KAN"
  - "Transformer"
  - "GPU Optimization"
  - "Memory Bottleneck"
date: 2026-05-08
content_hash: aeb5b459b0ab0a62
---

# FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer

**Conference**: AAAI 2026
**arXiv**: [2505.13813](https://arxiv.org/abs/2505.13813)
**Code**: [github](https://github.com/OSU-STARLAB/FlashKAT)
**Area**: Interpretability
**Keywords**: Kolmogorov-Arnold Network, KAN, Transformer, GPU Optimization, Memory Bottleneck

## TL;DR

This paper provides an in-depth analysis of the root cause behind KAT (Kolmogorov-Arnold Transformer) training being 123× slower than ViT. The bottleneck is identified not as FLOPs but as **memory stalls caused by gradient accumulation during backpropagation** (global memory contention from atomic add operations). The proposed FlashKAT restructures GPU kernels to achieve an **86.5× training speedup** and reduces gradient rounding errors by nearly an order of magnitude.

## Background & Motivation

### The Rise and Struggles of KAN

The Kolmogorov-Arnold Network (KAN) has attracted attention as an alternative to MLPs. Its core idea is to learn flexible nonlinear functions on each edge (rather than using fixed weights with fixed activations), endowing it with greater expressivity and interpretability. However, KAN faces severe practical barriers:
- **Each edge requires up to 204 FLOPs** (compared to only 2 for MLP)
- B-spline-based implementations are **ill-suited for GPU optimization** (requiring recursive algorithms)
- **Training instability**

### KAT's Breakthrough and Residual Issues

KAT addresses three problems by introducing the Group-Rational KAN (GR-KAN):
1. Input grouping with shared coefficients substantially reduces FLOPs
2. The safe Padé Activation Unit (PAU) replaces B-splines, eliminating recursion
3. Variance-preserving initialization improves training stability

KAT achieves superior results over ViT on vision tasks (e.g., KAT-B 82.3% vs. ViT-B 79.1%). Nevertheless, **KAT training remains 123× slower than ViT despite near-identical FLOPs**. This large discrepancy suggests that FLOPs are not the true performance bottleneck.

### Tracing the Core Problem

This paper is the first to revisit KAT performance from a **memory-centric perspective**, an angle unexplored by prior work.

## Method

### Overall Architecture

The paper is structured in two parts:

**Part 1: Performance Bottleneck Diagnosis** (4 Insights) → Identifies memory stalls caused by atomic add operations during gradient accumulation in backpropagation

**Part 2: FlashKAT Solution** → Restructures GPU kernel grid layout to accumulate gradients within blocks before issuing atomic adds, reducing global memory accesses by a factor of $S_{block} \cdot d_g$

### Key Designs

#### 1. **Performance Bottleneck Diagnosis (4 Insights)**

**Insight 1: KAT Is Indeed Extremely Slow**

Empirical measurements on H200 GPU:
- KAT-T vs. ViT-T: 102× slower
- KAT-S vs. ViT-S: 123× slower
- KAT-B vs. ViT-B: 116× slower

**Insight 2: FLOPs Are Not the Bottleneck**

Artificially increasing FLOPs in GR-KAN by 8× (via nested loops) results in **no change in execution time or cycle count**. This is because the additional FLOPs term $(2m+2n+3) \times d_{in}$ in GR-KAN is negligible relative to $2 \times d_{in} \times d_{out}$.

**Insight 3: Backpropagation Dominates Overwhelmingly**

Forward pass takes only 4.96ms, while backpropagation requires 1.03s — a ratio of **207.7×**. Optimization efforts should therefore focus on backpropagation.

**Insight 4: Memory Stalls Are the Root Cause**

Warp state analysis via Nvidia Nsight Compute reveals:
- SM throughput is only 1.97%, L1 only 4.38%, HBM only 1.01%
- Each warp spends **412×** more time in "Stall Long Scoreboard" (waiting for global memory transfers) than in "Selected" (actual computation)
- The seemingly paradoxical observation (memory-bound yet low utilization) arises because **atomic add operations cause severe memory contention**, preventing the warp scheduler from effectively hiding latency

#### 2. **Analysis of the Atomic Add Problem in the Original KAT**

The gradient $\frac{\partial \mathcal{L}}{\partial a_{g,i}}$ in GR-KAN requires accumulating contributions from all batch, sequence, and intra-group elements (Equations 10–11).

In KAT's implementation (Algorithm 1), each thread independently computes the gradient contribution of one element and **immediately** performs an atomic add to $\mathbf{dA}$ and $\mathbf{dB}$ in global memory. This leads to:
- $3(m+n+1)$ global memory accesses per element (reading coefficients + reading accumulated values + writing back)
- Severe contention as multiple threads/blocks simultaneously write to the same locations
- Total global memory accesses = $3(m+n+2) \cdot B \cdot N \cdot d$

#### 3. **FlashKAT Kernel Restructuring (Algorithm 2)**

Core improvement: **Restructure the 1D grid into a 2D grid, accumulating gradients within blocks before issuing a single atomic add.**

2D grid design:
- First dimension: $T = \lceil B \cdot N / S_{block} \rceil$, handling the batch and sequence dimensions
- Second dimension: $n_g$ (number of groups), with each block processing all $d_g$ elements of one group

Key changes in workflow:
1. Each block loads the coefficients $\mathbf{A}_j, \mathbf{B}_j$ for its corresponding group **once** (rather than each thread loading them separately)
2. All $S_{block} \times d_g$ contributions are accumulated in shared memory/registers within the block
3. **Each block performs only a single** atomic add (rather than multiple atomic adds per thread)

Global memory access optimization:
$$\text{Original} = 3(m+n+2) \cdot B \cdot N \cdot d$$
$$\text{FlashKAT} = 3\left(\frac{m+n+1}{S_{block} \cdot d_g} + 1\right) \cdot B \cdot N \cdot d$$

Atomic adds and global memory accesses are reduced by a factor of $\frac{1}{S_{block} \cdot d_g}$. For a typical configuration of $S_{block}=1024, d_g=96$, this corresponds to a reduction of approximately **98,000×**.

### Loss & Training

- FlashKAT follows the same training curriculum as the original KAT (DeiT hyperparameters)
- Batch size 1024, AdamW optimizer
- Mimetic initialization for attention layers
- GR-KAN configuration: 8 groups, 6 numerator coefficients, 4 denominator coefficients
- First GR-KAN layer initialized as the identity function, second as Swish
- Data augmentation: RandAugment, Mixup, CutMix, Random Erasing, Label Smoothing, Stochastic Depth
- Custom GPU kernels implemented in Triton

## Key Experimental Results

### Main Results

ImageNet-1K training throughput and accuracy comparison:

| Model | Params | Top-1 Acc. | Training Throughput (img/s) | Speedup over KAT |
|-------|--------|-----------|----------------------------|------------------|
| ViT-T | 5.7M | 72.7% | 8954.97 | — |
| KAT-T | 5.7M | 74.6% | 87.73 | 1× |
| **FlashKAT-T** | 5.7M | **74.6%** | **6317.90** | **72.0×** |
| ViT-S | 22.1M | 78.8% | 5311.71 | — |
| KAT-S | 22.1M | 81.2% | 43.28 | 1× |
| **FlashKAT-S** | 22.1M | **81.4%** | **3741.91** | **86.5×** |
| ViT-B | 86.6M | 79.1% | 2457.15 | — |
| KAT-B | 86.6M | 82.3% | 21.24 | 1× |
| **FlashKAT-B** | 86.6M | **82.2%** | **1801.75** | **84.5×** |

Backpropagation kernel performance comparison:

| Model | Cycles | Time | SM Throughput | L1 Throughput | L2 Throughput | HBM Throughput |
|-------|--------|------|--------------|--------------|--------------|---------------|
| KAT | 2.4T | 1.03s | 1.97% | 4.38% | 5.24% | 1.01% |
| **FlashKAT** | **16.9M** | **7.33ms** | **32.24%** | **34.14%** | **44.76%** | **92.05%** |

### Ablation Study

Gradient rounding error comparison:

| Model | Gradient | Mean Absolute Error | Variance |
|-------|----------|-------------------|----------|
| KAT | $\mathbf{dA}$ | $8.84 \times 10^{-2}$ | $1.45 \times 10^{-2}$ |
| KAT | $\mathbf{dB}$ | $9.63 \times 10^{-2}$ | $8.11 \times 10^{-2}$ |
| **FlashKAT** | $\mathbf{dA}$ | $8.42 \times 10^{-4}$ | $1.35 \times 10^{-6}$ |
| **FlashKAT** | $\mathbf{dB}$ | $9.81 \times 10^{-4}$ | $1.11 \times 10^{-5}$ |

Warp state analysis comparison:

| State | KAT (Cycles) | FlashKAT (Cycles) |
|-------|-------------|------------------|
| Selected (Computation) | ~2.4 | Increased compute fraction |
| Stall Long Scoreboard | 981.51 | **2.31** |

### Key Findings

1. **Accuracy is fully preserved**: FlashKAT-T/S/B achieves the same accuracy as KAT-T/S/B (FlashKAT-S even gains +0.2%)
2. **Highly significant speedup**: 72.0–86.5×, reducing the training speed gap between KAT and ViT from 100+ × to approximately 25%
3. **Memory utilization improves from near-zero to reasonable levels**: HBM throughput from 1.01% → 92.05%; SM throughput from 1.97% → 32.24%
4. **Gradient accuracy improves by ~100×**: MAE drops from $10^{-2}$ to $10^{-4}$; variance decreases by 3–4 orders of magnitude
5. **Stall Long Scoreboard reduced by 425×**: from 981.51 cycles to 2.31 cycles
6. **Speedup is a pure system-level optimization**: no changes to model architecture, training outcomes, or numerical approximations

## Highlights & Insights

1. **A model of system-level analysis**: Rather than simply applying profiler outputs, the paper systematically eliminates hypotheses (FLOPs? forward pass?) to precisely pinpoint atomic add memory contention
2. **A counterintuitive finding**: "Memory-bound yet extremely low memory utilization" — this arises from the serialization effect of atomic add operations, causing warps to spend the vast majority of time waiting rather than transferring data or computing
3. **An elegant solution**: 2D grid restructuring combined with intra-block accumulation is a classical parallel computing optimization strategy, yet no one in the KAN/PAU community had recognized the need for it
4. **Improved gradient accuracy as a byproduct**: Intra-block reduction summation offers better numerical stability than large numbers of atomic adds
5. **Contribution to the broader learnable rational activation community**: FlashKAT's optimization is directly applicable to PAU and its variants

## Limitations & Future Work

1. **Still ~25% slower than ViT**: FlashKAT-B achieves 1801 img/s vs. ViT-B's 2457 img/s; the remaining gap is primarily attributable to the additional computation in the GR-KAN forward pass
2. **Only backpropagation is optimized**: While the forward pass is already much faster, applying analogous strategies there could yield further gains
3. **Only vision tasks are evaluated**: KAT's performance in NLP and other domains remains unvalidated
4. **Requires custom Triton kernels**: This introduces some barrier to end-user accessibility
5. **Mixed-precision training is unexplored**: Given the improvement in gradient accuracy, the advantages of FlashKAT under FP16/BF16 mixed-precision training may be even more pronounced

## Rating

- Novelty: ⭐⭐⭐⭐ — The memory-centric analytical perspective is novel; the solution, while based on classical strategies, is pioneering within this domain
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — In-depth profiling with Nsight Compute, multi-scale models, and comprehensive evaluation of accuracy, speed, and numerical error
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative arc from problem discovery to diagnosis to solution is exceptionally clear, serving as a model for systems optimization papers
- Value: ⭐⭐⭐⭐⭐ — Reducing the gap from 123× slower to only 25% slower directly makes KAT a practically viable competitive approach

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](../../ICLR2026/interpretability/initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)
- [\[ICLR 2026\] Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[NeurIPS 2025\] MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition](../../NeurIPS2025/interpretability/mopformer_motion-primitive_transformer_for_wearable-sensor_activity_recognition.md)
- [\[CVPR 2026\] Geometry-Guided Camera Motion Understanding in VideoLLMs](../../CVPR2026/interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)

</div>

<!-- RELATED:END -->
