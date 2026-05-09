---
title: >-
  [Paper Note] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][image_restoration] This paper proposes UCAN, a lightweight super-resolution network that unifies convolution and attention. By introducing Hedgehog Attention to overcome the low-rank bottleneck of linear attention, and combining Flash Attention for large-window modeling, a large-kernel distillation module, and cross-layer parameter sharing, UCAN achieves super-resolution performance comparable to much larger models under extremely low computational budgets.
tags:
  - CVPR 2026
  - Image Restoration
  - image_restoration
  - super_resolution
  - lightweight
  - attention
  - linear_attention
date: 2026-05-08
content_hash: f5c0365baff48cc6
---

# UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2603.11680](https://arxiv.org/abs/2603.11680)
**Code**: None
**Area**: Image Restoration
**Keywords**: image_restoration, super_resolution, lightweight, attention, linear_attention

## TL;DR

This paper proposes UCAN, a lightweight super-resolution network that unifies convolution and attention. By introducing Hedgehog Attention to overcome the low-rank bottleneck of linear attention, and combining Flash Attention for large-window modeling, a large-kernel distillation module, and cross-layer parameter sharing, UCAN achieves super-resolution performance comparable to much larger models under extremely low computational budgets.

## Background & Motivation

Lightweight super-resolution faces a central tension: **expanding the effective receptive field** is critical for capturing global context, yet enlarging attention windows or convolution kernels substantially increases computational cost.

Limitations of prior work:
- **CNN-based methods**: constrained to local receptive fields
- **Transformer-based methods** (SwinIR, ELAN): window attention limits global modeling
- **Linear attention**: although $O(N)$ complexity, insufficient feature diversity leads to **rank collapse**, limiting representational capacity
- **Mamba/SSM methods**: global modeling still suffers from efficiency issues

## Method

### Overall Architecture

UCAN adopts a hierarchical design: shallow feature extraction → Broad Effective Receptive Field Group (BERFG) for deep encoding → reconstruction module (convolution + PixelShuffle).

### 1. Hedgehog Attention — Overcoming the Low-Rank Bottleneck of Linear Attention

Standard linear attention replaces softmax with a feature map $\phi$, reducing complexity from $O(N^2)$ to $O(N)$:

$$\mathbf{o}_i^L = \frac{\phi(\mathbf{q}_i)^\top \left(\sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top\right)}{\phi(\mathbf{q}_i)^\top \left(\sum_j \phi(\mathbf{k}_j)\right)}$$

However, simple mappings such as ReLU/ELU result in severely rank-deficient attention matrices. UCAN adopts the Hedgehog Feature Map (HFM), which concatenates symmetric exponential pairs:

$$\phi_H(\mathbf{X}) = [\exp(\mathbf{W}^\top \mathbf{X} + \mathbf{b}_1), \ldots, \exp(-\mathbf{W}^\top \mathbf{X} - \mathbf{b}_m)]$$

where $\mathbf{W}$ is a shared projection and $\{\mathbf{b}_i\}$ are learnable biases. The symmetric pairing mechanism in HFM preserves information in both positive and negative directions; experiments show the rank is recovered to 46, far exceeding that of ReLU/ELU.

### 2. Semi-Sharing Hybrid Attention

BERFG consists of Sharing Blocks and Receiving Blocks:

- **Sharing Block**: fully computes the attention map $\mathbf{A}_{qk}$ for Shared Window MHA and the feature map $\mathbf{A}_{map}$ for Hedgehog Attention
- **Receiving Block**: directly reuses the attention computed by the Sharing Block, recomputing only the value projection

The Dual Fusion Layer integrates two branches:
- **Spatial branch** (Hedgehog Attention): $\mathbf{F}_{sb} = \phi(\mathbf{Q}) \phi(\mathbf{K})^\top \mathbf{V} + \mathbf{W}_d \mathbf{V}$
- **Channel branch** (Channel Attention): $\mathbf{F}_{cb} = \text{softmax}(\mathbf{Q}^\top \mathbf{K}) \mathbf{V}$

Total complexity is linear in spatial resolution:
$$\mathcal{O}_{\text{DFL}} = \underbrace{2C^2 HW}_{\text{channel}} + \underbrace{6HW \frac{C^2}{D} + 9HWC}_{\text{spatial}}$$

### 3. High Performance Attention (HPA)

A ConvMLP with 7×7 kernels captures local context, paired with **Flash Attention** to enable 32×32 large-window attention. At 128×128 resolution, Flash Attention is **13.4×** faster than naive attention.

### 4. Large Kernel Distillation (LKD)

Channels are split into fine-grained and coarse-grained subsets ($C_{fg} = \max(C/4, 16)$). Three-branch feature extraction (channel attention + small-kernel local + hierarchical large-kernel dilated convolution) is applied only to fine-grained channels, while coarse-grained channels are bypassed directly, expanding the receptive field at minimal overhead.

## Key Experimental Results

### Table 1: Quantitative Comparison for Lightweight SR ×4

| Method | Params | MACs | Set5 PSNR | Urban100 PSNR | Manga109 PSNR |
|--------|--------|------|-----------|---------------|---------------|
| OmniSR | 792K | 50.9G | 32.49 | 26.64 | 31.02 |
| ATD-light | 769K | 100.1G | 32.63 | 26.97 | 31.48 |
| MambaIRV2-lt | 790K | 75.6G | 32.51 | 26.82 | 31.24 |
| ESC | 968K | 149.2G | 32.68 | 27.07 | 31.54 |
| **UCAN** | **705K** | **38.1G** | **32.65** | 26.89 | 31.50 |
| **UCAN-L** | **902K** | **48.4G** | **32.68** | 27.06 | **31.63** |

### Table 2: Attention Mechanism Efficiency Comparison (128×128 Input)

| Module | Latency (ms) | Params |
|--------|-------------|--------|
| Naive Self-Attention | 2576.75 (1.0×) | 0.082M |
| Flash Attention | 191.80 (13.4×) | - |
| Dual Fusion Layer | 1294.83 (2.0×) | 0.014M |

UCAN-L achieves 31.63 dB on Manga109 ×4 with 48.4G MACs, surpassing MambaIRV2 by 0.39 dB while reducing computation by 36%.

## Highlights & Insights

- **Hedgehog Attention** fundamentally addresses the rank collapse problem in linear attention, significantly improving feature diversity over ReLU/ELU
- The **semi-sharing mechanism** reuses attention computations between Sharing and Receiving Blocks, approaching the efficiency of full parameter sharing while preserving representational diversity
- **Flash Attention with large windows** is applied for the first time in lightweight SR using 32×32 windows without increasing latency
- **Large Kernel Distillation** performs heavy computation on only 1/4 of the channels, efficiently expanding the receptive field
- Exceptional parameter efficiency: 705K parameters and 38.1G MACs are sufficient to compete with methods exceeding 950K parameters

## Limitations & Future Work

- Validated only on classical SR (bicubic degradation); real-world degradation scenarios are not evaluated
- Training overhead of the Hedgehog Feature Map (exponential operations) is not thoroughly discussed
- Ablation studies are conducted only on Set5 and Urban100, providing limited coverage
- No comparison with contemporaneous linear attention improvements (e.g., FLatten Transformer)
- UCAN underperforms several competing methods on BSDS100 (27.79 vs. 27.80), indicating limited advantage in edge cases

## Rating

⭐⭐⭐⭐ — Solid technical contributions; Hedgehog Attention and the semi-sharing mechanism are both novel and practically motivated. The method achieves an excellent accuracy–efficiency trade-off under extremely tight computational budgets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_with_rag-based_dataset_distillation_and_multi-ob.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)

</div>

<!-- RELATED:END -->
