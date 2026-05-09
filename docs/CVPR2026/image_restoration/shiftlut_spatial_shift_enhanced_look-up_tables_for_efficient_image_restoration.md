---
title: >-
  [Paper Note] ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration
description: >-
  [CVPR 2026][Image Restoration][Look-Up Table] ShiftLUT is proposed to achieve the largest receptive field among LUT-based methods (65×65) via a Learnable Spatial Shift module (LSS), combined with an asymmetric dual-branch architecture and Error-bounded Adaptive Sampling (EAS). Under a storage budget of 104 KB and inference latency of 84 ms, ShiftLUT surpasses all existing LUT-based methods.
tags:
  - CVPR 2026
  - Image Restoration
  - Look-Up Table
  - Efficient Super-Resolution
  - Spatial Shift
  - Asymmetric Architecture
  - Adaptive Sampling
date: 2026-05-08
content_hash: c454e034fffc3b2f
---

# ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration

**Conference**: CVPR 2026
**arXiv**: [2603.00906](https://arxiv.org/abs/2603.00906)
**Code**: [GitHub](https://github.com/Sailor-t/ShiftLUT)
**Area**: Efficient Image Restoration
**Keywords**: Look-Up Table, Efficient Super-Resolution, Spatial Shift, Asymmetric Architecture, Adaptive Sampling

## TL;DR

ShiftLUT is proposed to achieve the largest receptive field among LUT-based methods (65×65) via a Learnable Spatial Shift module (LSS), combined with an asymmetric dual-branch architecture and Error-bounded Adaptive Sampling (EAS). Under a storage budget of 104 KB and inference latency of 84 ms, ShiftLUT surpasses all existing LUT-based methods.

## Background & Motivation

LUT-based methods replace convolutional operations with efficient memory lookups through a "space-for-time" strategy, making them suitable for deployment on edge devices such as smartphones. However, they face a fundamental tension:

**Limited Receptive Field**: LUT storage grows exponentially ($B^N$), making receptive field expansion prohibitively expensive.
   - MuLUT cascades multiple LUTs to enlarge the receptive field, but storage and latency increase linearly.
   - TinyLUT-F achieves a 33×33 receptive field at the cost of 171 KB and 146 ms.

**Inefficient Symmetric Dual-Branch Design**: SPLUT processes MSB (upper 6 bits) and LSB (lower 2 bits) branches in parallel, but the symmetric design wastes computation on the LSB branch—where the proportion of zero-valued activations in deep layers approaches 100%.

**Inflexible LUT Compression**: Existing methods apply a fixed sampling stride uniformly across all LUTs, despite different LUTs contributing unequally to the final output.

## Method

### Overall Architecture

ShiftLUT adopts a three-stage design:
1. **Feature Extraction**: The input image is split into MSB/LSB components; each passes through a 3×3 convolution for shallow feature extraction and is fused via element-wise addition.
2. **Deep Feature Processing**: Multiple Shift-Blocks are stacked (LSS → PwBlock → 3×3 DwConv).
3. **Final Restoration**: Channel refinement via PwBlock followed by PixelShuffle upsampling.

Three variants are defined: ShiftLUT-S (0 Shift-Blocks), ShiftLUT-M (1), and ShiftLUT-L (7).

During inference, all convolutions are converted to 1D LUTs via the SMS strategy, with rotation ensemble applied.

### Key Designs

1. **Learnable Spatial Shift Module (LSS)**: Expands the receptive field at minimal cost. The core idea is to learn an independent spatial offset per feature channel, enabling subsequent pointwise convolutions to aggregate information from diverse spatial locations.

    - A lightweight offset prediction network $\mathcal{O}(\cdot)$ (shallow convolution + MLP) predicts per-channel offsets:
    $\{(\Delta x_c, \Delta y_c)\}_{c=1}^{C} = \mathcal{O}(\mathbf{F})$
    - Each channel is spatially shifted according to its offset: $\mathbf{F}'_c(x,y) = \mathbf{F}_c(x - \Delta x_c, y - \Delta y_c)$
    - **Two-stage training**: Stage 1 jointly trains learnable floating-point offsets (with bilinear interpolation); Stage 2 removes the offset prediction network and replaces offsets with integer approximations of their training-time means, eliminating interpolation overhead at inference.
    - Key finding: offset variance is extremely low (< $10^{-3}$), indicating that LSS converges to fixed, channel-specific spatial sampling patterns.
    - Design motivation: Fixed-offset shift operations have been shown to be effective; learnable offsets enable the discovery of superior sampling configurations.

2. **Asymmetric Dual-Branch Architecture**: The LSB branch is simplified to a single 3×3 convolution, and the saved computation is reallocated to the information-dense MSB branch.

    - Empirical evidence: In the symmetric architecture, the proportion of zero-valued activations in deep layers of the LSB branch approaches 100%, demonstrating that applying a deep network to sparse signals is wasteful.
    - No performance degradation: The asymmetric design achieves PSNR of 28.19 versus 28.19 for the symmetric counterpart, while reducing latency from 164 ms to 84 ms.
    - Design motivation: MSB carries low-frequency structural information (spatially dense) while LSB carries high-frequency details (inherently sparse); the two should be treated differently.

3. **Error-bounded Adaptive Sampling (EAS)**: Automatically determines the optimal sampling stride for each LUT individually (rather than applying a uniform fixed stride), maximizing compression under a predefined error bound:
    $\max_{s \in \mathcal{S}} s \quad \text{s.t.} \quad \text{Error}(s) < \varepsilon$
    - Candidate set $\mathcal{S} = \{2^k | k=0,1,...,K_{\max}\}$ (powers of two for hardware-friendly bit-shift division).
    - The error metric incorporates a weighted penalty $\frac{s}{s-1}$ proportional to the storage savings ratio.
    - **Caching mechanism**: At inference time, interpolated LUT outputs are precomputed and cached in a shared buffer (only 64 bytes for 6-bit inputs), enabling direct per-pixel lookup without repeated interpolation.
    - Design motivation: Different LUTs contribute unequally to the output, making a uniform stride suboptimal.

### Loss & Training

- Dataset: DIV2K
- Optimizer: Adam, $\beta_1=0.9, \beta_2=0.999$, initial lr $5 \times 10^{-3}$, cosine annealing
- 200K iterations, batch size 32, patch size 48×48
- Two-stage training for LSS (Stage 1: learnable offsets → Stage 2: fixed integer offsets)
- EAS tolerance $\varepsilon = 0.4$

## Key Experimental Results

### Main Results

| Method | Storage | Latency | Receptive Field | Set5 PSNR | Urban100 | Manga109 | Avg. |
|--------|---------|---------|-----------------|-----------|----------|----------|------|
| SPLUT-L | 18432 KB | 265 ms | 5×5 | 30.52 | 24.46 | 27.70 | 27.42 |
| MuLUT | 4159 KB | 242 ms | 9×9 | 30.60 | 24.46 | 27.90 | 27.48 |
| TinyLUT-F | 171 KB | 146 ms | 33×33 | 31.18 | 24.92 | 28.83 | 28.01 |
| **ShiftLUT-S** | **24 KB** | **22 ms** | 9×9 | 30.50 | 24.39 | 27.65 | 27.39 |
| **ShiftLUT-M** | **38 KB** | **31 ms** | 17×17 | 30.77 | 24.62 | 28.18 | 27.66 |
| **ShiftLUT-L** | **104 KB** | **84 ms** | **65×65** | **31.33** | **25.12** | **29.16** | **28.19** |

Note: ShiftLUT-L achieves state-of-the-art performance on all benchmarks, with storage at only 61% of TinyLUT-F, 42% faster inference, and nearly twice the receptive field.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| w/o LSS vs. w/ LSS | PSNR +0.30+ dB | Consistently effective across all network configurations |
| Symmetric vs. Asymmetric | 28.19 vs. 28.19 PSNR; 164 ms vs. 84 ms | No performance drop; latency halved |
| EAS (ε=0.4) | Same PSNR as original LUT | Storage halved with negligible latency increase |
| EAS (ε=0.8) | PSNR drop < 0.03 dB | More aggressive compression while maintaining accuracy |
| Uniform sampling (step=4) | PSNR drop 0.04 dB; latency 211 ms | EAS is superior (84 ms + no PSNR loss) |

### Key Findings

- LSS consistently yields **> 0.30 dB improvement** across all network configurations (varying block counts and channel widths), demonstrating its generality.
- LAM (Local Attribution Map) analysis confirms that LSS enables the model to effectively utilize pixel information from a larger spatial extent.
- The extremely low offset variance confirms that LSS essentially learns optimal static sampling patterns.
- MSB 6-bit / LSB 2-bit is the optimal storage–performance trade-off.
- ShiftLUT also achieves state-of-the-art results in denoising (Set12: 32.43 vs. TinyLUT-F 32.22) and deblocking (Classic5: 29.12 vs. 28.74).

## Highlights & Insights

1. **Elegant LSS Design**: Learnable offsets discover optimal sampling configurations and are converted to zero-overhead fixed integer offsets after training, balancing flexibility and efficiency.
2. **Empirically Grounded Asymmetric Architecture**: The symmetric design's inefficiency is convincingly demonstrated through zero-valued activation analysis in the LSB branch.
3. **Adaptive Compression via EAS**: Per-LUT stride selection combined with a caching mechanism represents a substantive improvement over prior LUT compression strategies.
4. **Multi-task Validation**: Effectiveness is verified across three tasks: super-resolution, denoising, and deblocking.
5. **Pareto Frontier**: The ShiftLUT family establishes a new Pareto frontier in the storage–speed–quality space.

## Limitations & Future Work

1. The expressive capacity of LUT-based methods remains constrained by input quantization precision (6-bit MSB).
2. The two-stage training of LSS increases training complexity.
3. Only ×4 super-resolution is evaluated; other scale factors are not tested.
4. Performance under real-world degradations (non-bicubic) remains unknown.
5. Potential extensions include lightweight video restoration and larger LUT decomposition schemes.

## Related Work & Insights

- **TinyLUT**: 1D LUT with SMS (Separable Mapping Strategy); ShiftLUT builds upon this by introducing spatial shifts.
- **SPLUT**: Pioneered the dual-branch MSB/LSB architecture; ShiftLUT improves upon its symmetric design.
- **PCS/Group ShiftNet**: Fixed-offset shift operations for low-level vision; LSS extends these to learnable offsets.
- **ECLUT**: Expands coverage at the output end; LSS expands the receptive field at the feature level.
- Insight: The core bottleneck of LUT-based methods lies in the receptive field rather than computational cost; spatial shifting is the key to zero-overhead receptive field expansion.

## Rating

- Novelty: ⭐⭐⭐⭐ The two-stage learnable offset design of LSS is novel; the empirical justification of the asymmetric architecture is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks, 5+ datasets, and comprehensive ablations covering LSS, architecture, EAS, and bit allocation.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear; the motivation and interplay among the three contributions are presented fluently.
- Value: ⭐⭐⭐⭐⭐ Highly practical for edge deployment; the 24 KB–104 KB ultra-efficient models carry strong real-world deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution](../../ICCV2025/image_restoration/im-lut_interpolation_mixing_look-up_tables_for_image_super-resolution.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)

</div>

<!-- RELATED:END -->
