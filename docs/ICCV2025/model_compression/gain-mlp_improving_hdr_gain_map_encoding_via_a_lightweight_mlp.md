---
title: >-
  [Paper Note] Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP
description: >-
  [ICCV 2025][Model Compression][HDR Gain Map] This paper proposes replacing traditional JPEG/HEIC compression with a lightweight 10 KB MLP network for encoding HDR gain maps. The MLP takes SDR image color and spatial coor…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "HDR Gain Map"
  - "MLP"
  - "Implicit Neural Representation"
  - "Image Compression"
  - "Tone Mapping"
date: 2026-05-08
content_hash: a539c08477cb6276
---

# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP

**Conference**: ICCV 2025
**arXiv**: [2503.11883](https://arxiv.org/abs/2503.11883)
**Code**: None (authors state code will be released upon acceptance)
**Area**: Model Compression
**Keywords**: HDR Gain Map, MLP, Implicit Neural Representation, Image Compression, Tone Mapping

## TL;DR

This paper proposes replacing traditional JPEG/HEIC compression with a lightweight 10 KB MLP network for encoding HDR gain maps. The MLP takes SDR image color and spatial coordinates $(r,g,b,x,y)$ as input and incorporates exponential residual encoding (gamma map), outperforming existing methods and traditional compression techniques across multiple HDR reconstruction metrics.

## Background & Motivation

Modern displays (smartphones, tablets) broadly support high dynamic range (HDR) content, and modern cameras can natively capture HDR images. However, for compatibility with legacy SDR workflows and conventional displays, most images are still distributed in SDR formats (e.g., 8-bit JPEG/sRGB). To simultaneously support SDR and HDR devices, new encoding formats—Apple EDR, Android UltraHDR, Samsung SuperHDR, the Adobe gain map specification, and ISO 21496—are emerging. Their core idea is to embed a **pixel-wise gain map** as metadata within an SDR image, which, when applied to the SDR image on an HDR-capable display, recovers the HDR version.

Existing gain map encoding, however, suffers from inherent drawbacks: the gain map must be downsampled to 1/4 resolution, quantized to 8 bits, and then compressed with JPEG/HEIC/JPEG-XL. This pipeline inevitably introduces compression artifacts—**banding, haloing, blocking**, and loss of high-frequency detail. Increasing compression quality reduces artifacts but inflates file size.

The paper's core insight is twofold: (1) implicit neural representations (INRs), as continuous function approximators, naturally avoid the problems introduced by traditional quantization; and (2) an MLP need not encode an entire image from scratch—it only needs to encode a **spatially varying RGB transformation** (i.e., the gain map), for which the SDR image provides rich prior information. This allows the MLP to be extremely lightweight (10 KB) and to train very quickly (4 seconds per image).

## Method

### Overall Architecture

Given an SDR image $S$ and an HDR image $H$, traditional methods compute the gain map as $f(x,y) = (H+\epsilon)/(S+\epsilon)$, followed by log normalization, quantization, downsampling, and JPEG compression. Decoding reverses these steps to recover $H' = (S+\epsilon) \odot f'(x,y) - \epsilon$. This paper replaces the traditional compression step with an MLP that takes $(x,y,r,g,b)$ from the SDR image as input and directly predicts the corresponding gain map value.

### Key Designs

1. **Exponential Residual Encoding (Gamma Map)**:

    - Function: Replaces the multiplicative residual (gain map) with an exponential residual (gamma map).
    - Mechanism: Traditional gain maps use a multiplicative relationship: $f(x,y) = (H+\epsilon)/(S+\epsilon)$, decoded as $H' = (S+\epsilon) \odot f'(x,y) - \epsilon$. The exponential residual redefines this as $f(x,y) = \log(H+\epsilon)/\log(S+\epsilon)$, decoded as $H' = (S+\epsilon)^{f'(x,y)} - \epsilon$.
    - Design Motivation: The exponential residual more faithfully captures the essence of tone mapping operations, which are typically nonlinear power-function transforms. It constitutes a more accurate predictive coding approximation, reducing the complexity of the residual the MLP must learn. Experiments confirm that Gamma-MLP consistently outperforms Gain-MLP across all bit rates and with greater stability.

2. **Lightweight MLP Architecture**:

    - Function: Encodes the gain/gamma map using an extremely compact MLP.
    - Mechanism: A 5-dimensional input $(x,y,r,g,b)$ is mapped to a 24-dimensional sinusoidal embedding per dimension, yielding a 120-dimensional feature vector, which is passed through two ReLU MLP layers (16 neurons each) to produce a 3-channel output (RGB gain/gamma values). The resulting model size is only **10 KB**.
    - Training Configuration: A batch of 65,536 randomly sampled pixels, MSE loss, Adam optimizer (lr = $1\times10^{-2}$), 1,000 iterations, approximately 4 seconds per image (RTX 6000).
    - Design Motivation: Rather than predicting $(r,g,b)$ from $(x,y)$ alone (which is slow), the MLP leverages SDR image color as a strong prior input, since SDR color is highly correlated with the gain map. This enables extremely fast training and a very small model.

3. **Chromatic Noise Meta-Initialization**:

    - Function: Provides better weight initialization for the MLP.
    - Mechanism: The method of Daly et al. is used to generate 50 chromatic noise images that span the Rec. 2020 color gamut; their BT.709 SDR counterparts are produced using DaVinci Resolve's default tone mapping. The MLP is pre-trained on this synthetic data for 10,000 iterations to obtain meta-initialization weights.
    - Design Motivation: Unlike meta-initialization on natural images, chromatic noise images preserve natural image statistics (spatial and chromatic correlations) while avoiding content-specific bias. This accelerates convergence during per-image optimization and improves reconstruction quality.

### Loss & Training

- Loss: MSE loss, $\mathcal{L} = \|f'(x,y) - f(x,y)\|_2^2$, where $f$ is the ground-truth gain/gamma map and $f'$ is the MLP prediction.
- Following meta-initialization, the MLP is individually fine-tuned for each image (per-image optimization), converging within 1,000 iterations.
- At inference: the MLP is queried at all pixel coordinates to obtain the complete gain map, which is then applied to the SDR image to recover the HDR output.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | ΔE₀₀↓ | SSIM↑ | ΔE_IPT↓ | HDR-VDP3↑ | Size (KB) |
|--------|-------|--------|-------|---------|-----------|-----------|
| Gain-JPEG | 38.29 | 2.16 | 0.968 | 9.63 | 7.92 | 19.0 |
| Gamma-JPEG | 41.45 | 1.37 | 0.979 | 7.12 | 8.62 | 19.4 |
| Gain-HEIC | 39.20 | 1.98 | 0.972 | 8.71 | 8.14 | 18.4 |
| Gamma-HEIC | 42.21 | 1.27 | 0.982 | 6.57 | 8.75 | 18.2 |
| Direct-MLP [Le] | 46.30 | 0.96 | 0.988 | 4.66 | 9.06 | 10 |
| MLP-ITM [Liu] | 47.25 | 0.87 | 0.991 | 4.28 | **9.13** | 34 |
| **Gain-MLP (Ours)** | 47.60 | 1.02 | 0.992 | 4.27 | 8.98 | **10** |
| **Gamma-MLP (Ours)** | **48.53** | **0.78** | **0.993** | **3.91** | 9.11 | **10** |

### Ablation Study (Rate-Distortion Performance Across MLP Sizes and Encoding Variants)

| Configuration | PSNR Trend | Notes |
|---------------|-----------|-------|
| Gamma-MLP, 8 neurons | Still outperforms full-resolution JPEG/HEIC at the lowest bit rate | Exponential residual advantage is greatest at extremely low capacity |
| Gamma-MLP, 16 neurons (default) | Best across all bit rates | Optimal performance/size trade-off |
| Gamma-MLP, 64 neurons | Slight improvement with diminishing returns | Limited gain when network capacity is excessive |
| Gamma-MLP, 128 neurons | Near saturation | Training time remains 4 seconds |
| Gain-MLP vs. Gamma-MLP | Gamma leads at all bit rates | Exponential residual consistently outperforms multiplicative residual |
| Gain-MLP vs. Direct-MLP | Gain-MLP advantage larger at low bit rates | Base-residual paradigm benefits MLP encoding |
| JPEG 1/8 → full resolution | Approaches Gamma-MLP (16n) but remains inferior | Traditional methods require significantly more bits to match |

### Key Findings

- Gamma-MLP achieves **48.53 dB PSNR** at only **10 KB**, surpassing MLP-ITM (47.25 dB at 34 KB) while using only 29% of the storage.
- Exponential residual encoding (gamma map) consistently outperforms multiplicative residual (gain map) in both traditional compression and MLP-based encoding, validating its effectiveness as a predictive coding approximation.
- The primary advantage of MLP-based methods is **fixed memory overhead** (10 KB), independent of image resolution or encoding parameters.
- Qualitative analysis shows that banding, haloing, and blocking artifacts prevalent in traditional methods are substantially reduced by the MLP approach, with gamma map encoding achieving further improvement.

## Highlights & Insights

- **Well-targeted problem formulation**: Rather than pursuing general image compression (which requires large MLPs and long optimization), the method exploits the SDR image as a strong prior and encodes only the transformation residual, making the problem extremely lightweight.
- **Fixed 10 KB overhead**: Unlike traditional methods whose file size varies with image resolution and quality settings, the MLP size is entirely fixed, making it highly suitable for embedded metadata scenarios.
- **Theoretical insight of exponential residuals**: Reframing the gain map problem as encoding power-function parameters yields a more natural representation of the underlying transformation.
- **Elegance of chromatic noise initialization**: Using synthetic data that conforms to natural image statistics for meta-initialization avoids content-specific bias in a principled way.

## Limitations & Future Work

- MLP optimization still requires 4 seconds per image, which introduces latency for real-time applications.
- When SDR tone mapping involves heavy clipping, MLP reconstruction capability is limited, as the clipped information is irreversibly lost.
- Adaptive bit-rate allocation across images of different resolutions is not addressed—different resolutions may warrant MLPs of different sizes.
- Evaluation is conducted only on HD (1920×1080) and UHD (3840×2160) images; scalability to higher resolutions remains unexplored.
- Decoding requires MLP forward inference (though fast), introducing additional computation compared to purely decode-based approaches.

## Related Work & Insights

- **Le et al. (Direct-MLP)**: First to apply MLP with $(x,y,r,g,b)$ input for embedded gamut restoration, but outputs HDR RGB values directly rather than as a residual.
- **Liu et al. (MLP-ITM)**: Dual-network (spatial + color) with domain pre-training and hard example mining; at 34 KB it is larger yet slightly underperforms Gamma-MLP.
- **Canham et al.**: First to propose exponential residuals to improve gain map encoding in traditional compression; this paper demonstrates that the finding transfers equally to MLP-based encoding.
- **Insight**: In base-plus-residual coding paradigms, choosing the correct residual representation (linear vs. nonlinear) may matter more than choosing a better encoder.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] COMI: Coarse-to-fine Context Compression via Marginal Information Gain](../../ICLR2026/model_compression/comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)
- [\[ICCV 2025\] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [\[ICCV 2025\] MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion](motionfollower_editing_video_motion_via_score-guided_diffusion.md)

</div>

<!-- RELATED:END -->
