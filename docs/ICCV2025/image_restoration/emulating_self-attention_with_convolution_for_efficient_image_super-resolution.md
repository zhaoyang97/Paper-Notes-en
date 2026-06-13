---
title: >-
  [Paper Note] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution
description: >-
  [ICCV 2025][Image Restoration][Image Super-Resolution] Motivated by the observation that features and attention maps across adjacent self-attention layers exhibit high inter-layer similarity (89%/87%)…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Image Super-Resolution"
  - "Self-Attention Substitution"
  - "Large-Kernel Convolution"
  - "Flash Attention"
  - "Lightweight Network"
date: 2026-05-08
content_hash: 90c61e848cbda3f1
---

# Emulating Self-Attention with Convolution for Efficient Image Super-Resolution

**Conference**: ICCV 2025
**arXiv**: [2503.06671](https://arxiv.org/abs/2503.06671)  
**Code**: [GitHub](https://github.com/dslisleedh/ESC)  
**Area**: Image Restoration
**Keywords**: Image Super-Resolution, Self-Attention Substitution, Large-Kernel Convolution, Flash Attention, Lightweight Network

## TL;DR

Motivated by the observation that features and attention maps across adjacent self-attention layers exhibit high inter-layer similarity (89%/87%), this paper proposes ConvAttn — a module composed of a shared large-kernel convolution and a dynamic convolution kernel — to replace the majority of self-attention layers. Flash Attention is introduced into lightweight SR for the first time, extending the window size to $32 \times 32$, achieving state-of-the-art performance at minimal latency and memory cost.

## Background & Motivation

Transformers have demonstrated superior performance over CNNs in image super-resolution (SR), yet face serious practical deployment challenges:

**Memory Access Bottleneck**: Self-attention requires materializing the score matrix $S = QK^T$, along with memory-intensive operations such as tensor reshape and window masking, resulting in high latency and memory consumption. SwinIR-light is 4.7× slower and requires 2× more memory than a CNN with equivalent FLOPs.

**Misleading Efficiency Metrics**: Existing methods primarily focus on reducing FLOPs and parameter count while neglecting actual latency and memory footprint — the true bottlenecks in deployment.

**Key Finding**: Through empirical analysis of inter-layer similarity in SwinIR-light, this paper finds:
   - The CKA similarity of features $F$ extracted by self-attention between adjacent layers reaches **87%**
   - The cosine similarity of attention maps $A^{avg}$ between adjacent layers reaches **89%**

This indicates that self-attention extracts a substantial amount of **redundant features** across multiple layers.

Based on this, the paper proposes a bold strategy: retain only **one** self-attention layer per block (strengthened with a large window and Flash Attention), and replace all remaining layers with efficient convolutional modules.

## Method

### Overall Architecture

The ESC network consists of four main components:
- **Shallow Feature Extraction**: A $3 \times 3$ convolution maps the LR input to $C$-dimensional features
- **Deep Feature Extractor $H$**: Composed of $N$ ESCBlocks, all sharing a single $13 \times 13$ large kernel $LK$
- **Image-Level Skip Module $S$**: Processes the LR input in parallel
- **Upsampler $U$**: Fuses deep features and skip features to generate the SR image

### Key Designs

1. **ConvAttn Module (Convolutional Attention)**:

    - **Function**: Emulates two key advantages of self-attention — long-range dependency modeling and instance-dependent weighting — using convolution.
    - **Mechanism**: Channel features are split into two parts: $F^{att} \in \mathbb{R}^{H \times W \times 16}$ (first 16 channels) and $F^{idt} \in \mathbb{R}^{H \times W \times (C-16)}$ (remaining channels). Two convolutions are applied only to $F^{att}$:
        - **Shared large kernel $LK \in \mathbb{R}^{13 \times 13 \times 16 \times 16}$**: Shared across the entire network to capture long-range interactions (emulating long-range dependencies of attention)
        - **Dynamic kernel $DK \in \mathbb{R}^{3 \times 3 \times 1 \times 16}$**: Generated from the input via GAP + MLP for instance-dependent weighting (emulating the adaptive nature of attention)

    $$F^{res} = (F^{att} \circledast DK) + (F^{att} \circledast LK)$$

    The result is concatenated with $F^{idt}$ and fused via a $1 \times 1$ convolution.
    - **Design Motivation**: Given that inter-layer features are highly similar, long-range interaction patterns need not be recomputed at every layer. The shared $LK$ reduces parameter growth and optimization difficulty, while $DK$ provides per-layer input adaptability. Operating on only 16 channels substantially reduces memory access overhead.

2. **ESCBlock Structure**:

    - **Function**: Each block employs only one self-attention layer; the remaining $M$ layers use ConvAttn.
    - **Mechanism**:
    $$F_{i,0} = F_i^{in} + \text{SelfAttn}(\text{LN}(F_i^{in}))$$
    $$F_{i,j} = F_{i,j-1} + \text{ConvAttn}_j(\text{ConvFFN}_j(F_{i,j-1}), LK), \quad j=1,...,M$$
    Notably, ConvFFN is placed **before** self-attention, so that self-attention already incorporates local information when extracting features, eliminating the need for complex QKV projections. ConvAttn layers follow self-attention to model inter-window features (analogous to shifted windows).
    - **Design Motivation**: Since self-attention features are highly similar across layers, a single self-attention layer suffices to establish global relationships; subsequent layers can maintain these relationships with lightweight convolutions.

3. **Flash Attention Integration**:

    - **Function**: Flash Attention is introduced into lightweight SR for the first time, extending the window size to $32 \times 32$.
    - **Mechanism**: Flash Attention reduces memory consumption by avoiding materialization of the full score matrix. At a $32 \times 32$ window size, it achieves up to **16×** latency reduction and **12.2×** memory savings compared to standard implementations.
    - **Design Motivation**: Larger windows incur only marginal FLOPs overhead yet yield significant performance gains. The excessively large score matrix produced by large windows under standard implementations is resolved by Flash Attention. Since most self-attention layers have been replaced by ConvAttn, the overhead of using a large window in the remaining single self-attention layer remains manageable.

4. **ESC-FP Variant (FLOPs-Priority)**:

    - **Function**: For scenarios requiring reduced FLOPs and parameter count, the large kernel is decomposed using depthwise separable factorization.
    - **Mechanism**: $LK$ is decomposed into a pointwise kernel $LK^c \in \mathbb{R}^{1 \times 1 \times 16 \times 16}$ and a depthwise kernel $LK^s \in \mathbb{R}^{13 \times 13 \times 1 \times 16}$. The dynamic kernel can be merged with $LK^s$ via zero-padding:
    $$F^{res} = (F^{att} \circledast LK^c) \circledast (ZP(DK) + LK^s)$$

### Loss & Training

- $L_1$ loss computed on the Y channel
- Models trained and evaluated on both DIV2K and the large-scale DFLIP dataset
- Training patch size: $64 \times 64$
- Three model variants provided: ESC (latency-priority), ESC-light, and ESC-FP (FLOPs/parameter-priority)

## Key Experimental Results

### Main Results (Trained on DIV2K, ×4 SR)

| Method | Urban100 PSNR | Latency (ms) | Memory (MB) | FLOPs (G) | Params (K) |
|--------|--------------|-------------|------------|-----------|-----------|
| SwinIR-lt | 26.47 | 222.9 | 351 | 63.6 | 930 |
| ATD-lt | 26.97 | 189.7 | 753 | 100.1 | 769 |
| HiT-SRF | 26.80 | 82.1 | 1331 | 58.0 | 866 |
| MambaIRV2-lt | 26.82 | 153.4 | 748 | 75.6 | 790 |
| **ESC** | **27.07** | **21.9** | **215** | 149.2 | 968 |
| **ESC-FP** | 26.90 | 21.7 | 158 | 60.8 | 539 |

ESC surpasses ATD-light by 0.1 dB PSNR on Urban100 ×4 while being **8.7× faster**.

### Ablation Study

| Configuration | Set5×2 PSNR | Urban100×2 PSNR | Latency (ms) | Notes |
|--------------|------------|----------------|-------------|-------|
| SA Only | 38.27 | 33.23 | 128.2 | All SA, more layers |
| ConvAttn Only | 38.18 | 32.91 | 126.3 | No SA, notable performance drop |
| 9×9 LK | 38.33 | 33.42 | 117.1 | Kernel too small |
| 17×17 LK | 38.32 | 33.40 | 126.6 | Kernel too large, performance degrades |
| No LK Sharing + No DK | 38.32 | 33.37 | 119.7 | Baseline |
| No DK | 38.31 | 33.36 | 119.7 | DK contributes significantly |
| WS16 + More Layers | 38.24 | 33.05 | 118.0 | Small window inferior to large window + fewer layers |
| **ESC (13×13 LK)** | **38.35** | **33.46** | 120.9 | Optimal configuration |

### Key Findings

- Neither SA alone nor ConvAttn alone matches their combination; **complementary integration** is optimal
- $13 \times 13$ is the optimal kernel size; both larger and smaller kernels lead to performance degradation
- LAM visualization confirms that ESC's receptive field is comparable to or larger than that of pure Transformers (highest diffusion index)
- ESC's advantage is more pronounced under large-scale data training (DFLIP) — surpassing ATD-light by 0.27 dB on Urban100 ×4, demonstrating retained Transformer-level data scalability

## Highlights & Insights

- **Discovery of Inter-Layer Redundancy**: Quantitatively demonstrates 87–89% inter-layer similarity of self-attention, providing a solid empirical basis for reducing self-attention usage
- **Extreme Efficiency**: ESC achieves only 21.9 ms latency on HD images (vs. 189.7 ms for ATD-light), approaching CNN-level inference speed
- **Only 16 Channels Undergo Attention Operations**: Long-range modeling is boldly applied to only 16 out of 60 channels, with the remainder passed through directly, without sacrificing performance
- **Shared Large Kernel**: Sharing a single large kernel across the entire network is an elegant parameter-efficient design, analogous to a global "universal long-range interaction template"
- **First Application of Flash Attention in SR**: Overcomes the window size bottleneck in lightweight SR

## Limitations & Future Work

- Flash Attention currently requires CUDA optimization and is not friendly to CPU-only devices
- The $13 \times 13$ size of the shared large kernel may not be optimal across different tasks
- Dynamic kernel generation in ConvAttn relies on global average pooling (GAP), which may discard spatial information
- Validation is primarily conducted on SR; effectiveness on other low-level tasks (denoising, deblurring) requires further investigation
- Training patch size of only $64 \times 64$ may cause training instability due to padding with the large window ($32 \times 32$)

## Related Work & Insights

- Shares conceptual connections with ELAN's attention-sharing strategy, but ESC goes further by directly replacing attention with convolution rather than sharing attention maps
- Differs from large-kernel CNNs such as RepLKNet by incorporating dynamic kernels and a complementary self-attention design
- Provides a quantitative answer to the question of "when self-attention is necessary and when convolution suffices"

## Rating

- **Novelty**: ⭐⭐⭐⭐ The inter-layer redundancy observation is not entirely new, but converting it into concrete design choices (shared large kernel + dynamic kernel + minimal SA) is innovative
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation from DIV2K to DFLIP, ×2 to ×4, with full comparison of latency/memory/FLOPs, and in-depth LAM and feature visualization
- **Writing Quality**: ⭐⭐⭐⭐ Motivation analysis is clear and figures are informative, though some formulas contain dense subscript notation
- **Value**: ⭐⭐⭐⭐⭐ Addresses practical deployment bottlenecks in lightweight SR; both Flash Attention integration and ConvAttn design carry significant practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks](../../NeurIPS2025/image_restoration/spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] Efficient Concertormer for Image Deblurring and Beyond](efficient_concertormer_for_image_deblurring_and_beyond.md)
- [\[ICCV 2025\] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution](im-lut_interpolation_mixing_look-up_tables_for_image_super-resolution.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)

</div>

<!-- RELATED:END -->
