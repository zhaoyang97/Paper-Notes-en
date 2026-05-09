---
title: >-
  [Paper Note] Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention
description: >-
  [ICLR2026][3D Vision][novel view synthesis] This paper proposes Efficient-LVSM, a dual-stream architecture that decouples input view encoding from target view generation, reducing the complexity of novel view synthesis from $O(N_{in}^2)$ to $O(N_{in})$. On RealEstate10K, the model achieves state-of-the-art performance (29.86 dB PSNR) using only 50% of LVSM's training time, with a 4.4× inference speedup.
tags:
  - ICLR2026
  - 3D Vision
  - novel view synthesis
  - Transformer
  - Dual-Stream Architecture
  - KV-Cache
  - Attention Decoupling
date: 2026-05-08
content_hash: 3cd14cfdffc1cc8b
---

# Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention

**Conference**: ICLR2026  
**arXiv**: [2602.06478](https://arxiv.org/abs/2602.06478)  
**Code**: [efficient-lvsm.github.io](https://efficient-lvsm.github.io/)  
**Area**: 3D Vision  
**Keywords**: novel view synthesis, Transformer, Dual-Stream Architecture, KV-Cache, Attention Decoupling

## TL;DR

This paper proposes Efficient-LVSM, a dual-stream architecture that decouples input view encoding from target view generation, reducing the complexity of novel view synthesis from $O(N_{in}^2)$ to $O(N_{in})$. On RealEstate10K, the model achieves state-of-the-art performance (29.86 dB PSNR) using only 50% of LVSM's training time, with a 4.4× inference speedup.

## Background & Motivation

Novel View Synthesis (NVS)—reconstructing 3D scenes from 2D images—is a central problem in computer vision. The field has evolved from per-scene optimization methods such as NeRF and 3DGS to feed-forward Transformer approaches like LVSM, which directly synthesize novel views from posed images without hand-crafted 3D priors.

However, LVSM's decoder-only design concatenates all input and target tokens before applying full self-attention, leading to two critical bottlenecks:

1. **Inefficiency**: Attention complexity scales quadratically with the number of input views as $O(N^2)$; input representations are recomputed redundantly when generating multiple target views.
2. **Performance limitations**: Heterogeneous tokens—content-rich input views and pose-only target queries—share the same attention parameters, hindering each stream from learning specialized representations.

## Core Problem

How can one decouple input encoding from target generation within an end-to-end feed-forward NVS framework, while simultaneously improving both efficiency and quality?

The encoder-decoder variant of LVSM avoids redundant computation but compresses all inputs into a single latent vector, causing significant information loss and degraded reconstruction quality. A new architecture is needed that preserves multi-level fine-grained features while supporting efficient inference.

## Method

### 1. Dual-Stream Architecture

The core idea is to fully decouple input view processing and target view generation into two independent streams:

- **Input Encoder**: Applies intra-view self-attention independently to each input view, with no cross-view interaction.
- **Target Decoder**: Target tokens first undergo self-attention, then query input features via cross-attention.

This design reduces complexity from $O(N^2 M)$ to $O(NM + N)$.

### 2. Intra-View Self-Attention

The input encoder applies self-attention exclusively among patches within each individual input view:

$$\mathbf{S_i}^l = \mathbf{S_i}^{l-1} + \text{Self-Attn}_{\text{input}}^l(\mathbf{S_i}^{l-1})$$

Each view is processed independently, naturally enabling zero-shot generalization to varying numbers of input views.

### 3. Self-then-Cross Attention (Target Decoder)

The target decoder alternates between self-attention and cross-attention:

$$\mathbf{T_j}^l = \mathbf{T_j}^{l-1} + \text{Self-Attn}_{\text{target}}^l(\mathbf{T_j}^{l-1})$$
$$\mathbf{T_j}^l = \mathbf{T_j}^l + \text{Cross-Attn}_{\text{target}}^l(\mathbf{T_j}^l, \mathbf{S_1}^l, ..., \mathbf{S_N}^l)$$

Self-attention enables target tokens to exchange scene-level information with one another, while cross-attention extracts the required content from input features. Ablation studies show that a 6+6 alternating structure outperforms 12 layers of pure cross-attention.

### 4. Co-Refinement Mechanism

Unlike conventional encoder-decoder designs that consume only the final encoder layer's features, co-refinement allows each decoder layer to query the corresponding encoder layer. This enables the decoder to leverage:

- Fine-grained texture details from early layers
- High-level semantic information from later layers

Visualizations confirm that co-refinement produces feature maps capturing significantly more target-view detail than a vanilla encoder-decoder baseline.

### 5. REPA Distillation

REPA is employed to distill visual features from a pretrained DINOv3 model into intermediate encoder layers by maximizing patch-level similarity:

$$\mathcal{L}_{REPA} = \frac{1}{N}\sum_{i=1}^{N}\text{sim}(f(\mathbf{I}), h_\phi(\mathbf{X_k}))$$

The pretrained encoder and projection layers are discarded at inference time, incurring no additional inference overhead. A key finding is that REPA yields substantially larger gains for Efficient-LVSM than for LVSM, since full self-attention in the latter entangles feature maps across different views.

### 6. KV-Cache and Incremental Inference

The decoupled design naturally supports KV-caching:

- Key/value pairs for input views are computed once and cached for reuse.
- Adding a new target view: cached representations are directly reused for rendering.
- Adding a new input view: only the new view is processed and appended to the cache.

The marginal cost of adding input or target views is nearly constant, making the model well-suited for interactive applications.

## Key Experimental Results

### Scene-Level (RealEstate10K, 2 Input Views)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| GS-LRM | 28.10 | 0.892 | 0.114 |
| LVSM Dec-Only (512) | 29.53 | 0.904 | 0.141 |
| **Efficient-LVSM (512)** | **29.86** | **0.905** | 0.147 |

### Object-Level (ABO / GSO, 512 Resolution)

| Method | ABO PSNR | GSO PSNR |
|------|----------|----------|
| GS-LRM | 29.09 | 30.52 |
| LVSM Dec-Only | 32.10 | 32.36 |
| **Efficient-LVSM** | **32.65** | **32.92** |

### Efficiency Comparison

- Training convergence: approximately **2×** faster than LVSM
- Inference speed: **4.4×** faster than LVSM Dec-Only (up to 14.9× in certain settings)
- Training resources: 64 A100 GPUs for 3 days, representing only **50%** of LVSM's training cost
- Incremental inference: latency and memory remain nearly constant as views are added

### Zero-Shot Generalization

Trained with 4 input views, the model generalizes at test time to varying numbers of input views, enabled by the per-view independent processing design.

## Highlights & Insights

1. **Rigorous problem analysis**: The limitations of LVSM's full self-attention are systematically analyzed from the perspectives of information heterogeneity and computational complexity, motivating the dual-stream decoupled design.
2. **Elegant co-refinement design**: Layer-wise alignment between encoder and decoder features fully exploits multi-scale information.
3. **Strong practical value**: KV-cache combined with incremental inference enables deployment in interactive 3D scene browsing applications.
4. **Dual gains in efficiency and quality**: Efficient-LVSM surpasses LVSM across all benchmarks while substantially reducing training and inference costs.
5. **Conditional finding on REPA distillation**: The work reveals a coupling between distillation effectiveness and attention architecture design.

## Limitations & Future Work

1. Intra-view attention in the input encoder contains no cross-view interaction; scene understanding relies entirely on the decoder's cross-attention, which may be insufficient for scenes with heavy occlusion.
2. Scaling behavior at larger model sizes or higher resolutions remains unexplored.
3. Evaluation is limited to static scenes; applicability to dynamic scenes is unknown.
4. LPIPS at 512 resolution is marginally worse than LVSM Dec-Only (0.147 vs. 0.141), indicating room for improvement in perceptual quality.

## Related Work & Insights

| Dimension | LVSM Dec-Only | LVSM Enc-Dec | Efficient-LVSM |
|------|--------------|--------------|----------------|
| Input complexity | $O(N^2)$ | $O(N^2)$ | $O(N)$ |
| Parameter sharing | Heterogeneous tokens shared | Separated but uses only last layer | Separated + layer-wise co-refinement |
| KV-Cache | Not supported | Not supported | Supported |
| Incremental inference | Not supported | Not supported | Supported |
| Variable-view generalization | Poor | Poor | Strong (zero-shot) |
| RealEstate10K PSNR | 29.53 | 28.55 | **29.86** |

Compared to Gaussian splatting-based methods such as pixelSplat and MVSplat, Efficient-LVSM does not rely on explicit 3D representations, is more end-to-end, and achieves higher quality, though at greater computational cost.

The following broader insights emerge from this work:

1. **Transferability of the dual-stream decoupling principle**: Any Transformer task involving heterogeneous inputs—such as multimodal understanding or robot perception—can benefit from decoupling the provider and query streams.
2. **Generality of co-refinement**: Layer-wise cross-querying can be applied to other encoder-decoder architectures beyond NVS.
3. **KV-Cache for 3D vision**: Importing the mature KV-cache technique from LLMs into 3D vision provides a new pathway toward real-time interactive rendering.
4. **Coupling between distillation and architecture**: The substantially different REPA gains across architectures suggest that distillation strategies should be co-designed with the model architecture.

## Rating
- Novelty: ⭐⭐⭐⭐ — The dual-stream co-refinement design represents a clear contribution to NVS, though individual components are not entirely novel in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablations covering scene-level and object-level benchmarks across efficiency, quality, and generalization dimensions.
- Writing Quality: ⭐⭐⭐⭐ — The progression from problem analysis to solution derivation is logically coherent and well-illustrated.
- Value: ⭐⭐⭐⭐⭐ — Simultaneous gains in efficiency and quality, strong practical utility of KV-cache support, and clear guidance for future work.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](../../ICCV2025/3d_vision/rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](../../ICCV2025/3d_vision/faster_and_better_3d_splatting_via_group_training.md)
- [\[CVPR 2026\] Scaling View Synthesis Transformers (SVSM)](../../CVPR2026/3d_vision/scaling_view_synthesis_transformers.md)
- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[CVPR 2026\] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](../../CVPR2026/3d_vision/physgm_large_physical_gaussian_model_for_feed-forward_4d_synthesis.md)

<!-- RELATED:END -->
