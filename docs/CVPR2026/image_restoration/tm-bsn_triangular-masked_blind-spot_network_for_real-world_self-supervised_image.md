---
title: >-
  [Paper Note] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising
description: >-
  [CVPR 2026][Image Restoration][blind-spot network] This paper proposes TM-BSN, a triangular-masked blind-spot network that designs the blind-spot region to precisely align with the diamond-shaped spatial correlation pattern of real-world sRGB noise, enabling self-supervised image denoising at full resolution without downsampling. Combined with knowledge distillation, TM-BSN achieves state-of-the-art self-supervised denoising performance on the SIDD and DND benchmarks.
tags:
  - CVPR 2026
  - Image Restoration
  - blind-spot network
  - self-supervised denoising
  - triangular-masked convolution
  - spatially correlated noise
  - knowledge distillation
date: 2026-05-08
content_hash: 2b10ef65b630f637
---

# TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising

**Conference**: CVPR 2026
**arXiv**: [2604.04484](https://arxiv.org/abs/2604.04484)
**Code**: [https://github.com/parkjun210/TM-BSN](https://github.com/parkjun210/TM-BSN)
**Area**: Image Restoration / Self-Supervised Denoising
**Keywords**: blind-spot network, self-supervised denoising, triangular-masked convolution, spatially correlated noise, knowledge distillation

## TL;DR

This paper proposes TM-BSN, a triangular-masked blind-spot network that designs the blind-spot region to precisely align with the diamond-shaped spatial correlation pattern of real-world sRGB noise, enabling self-supervised image denoising at full resolution without downsampling. Combined with knowledge distillation, TM-BSN achieves state-of-the-art self-supervised denoising performance on the SIDD and DND benchmarks.

## Background & Motivation

1. **State of the Field**: Blind-spot networks (BSNs) represent the mainstream approach to self-supervised image denoising. Their core idea is to prevent identity mapping by excluding the target pixel from its own receptive field, thereby estimating the clean signal without requiring clean image supervision.
2. **Limitations of Prior Work**: BSNs assume pixel-wise independent noise; however, in real-world sRGB images, the ISP pipeline—particularly the demosaicing stage—introduces strongly spatially correlated noise that violates this independence assumption, causing the network to degenerate into an identity mapping.
3. **Root Cause**: Existing solutions either apply pixel-shuffle downsampling (PD) to decorrelate noise, which alters the noise statistics and requires post-processing (e.g., AP-BSN), or enlarge the blind-spot region at full resolution (e.g., AT-BSN) using rectangular blind spots that mismatch the diamond-shaped correlation pattern of real noise, thereby excluding useful uncorrelated pixels.
4. **Paper Goals**: To design a self-supervised denoising network whose blind-spot shape precisely matches the spatial correlation geometry of real-world noise.
5. **Starting Point**: The authors observe that during demosaicing, each pixel is reconstructed from neighboring samples using spatially decaying weights, producing a diamond-shaped correlation pattern centered on the target pixel. The blind spot should precisely cover this diamond-shaped region.
6. **Core Idea**: Triangular-masked convolutions are used to construct a diamond-shaped blind spot that precisely matches the spatial correlation geometry of sRGB noise, maximizing the utilization of contextual information from uncorrelated pixels while excluding all correlated ones.

## Method

### Overall Architecture

The noisy input image is processed through four rotated branches (0°, 90°, 180°, 270°) for feature extraction. Each branch replaces standard 3×3 convolutions with triangular-masked convolutions (TMC) in the backbone and applies feature shifting to form the blind spot. The features from all four branches are inverse-rotated and concatenated along the channel dimension, followed by a 1×1 convolution to produce the denoised output. Optionally, knowledge distillation is applied to transfer complementary information from multiple blind-spot predictions into a lightweight U-Net student network.

### Key Designs

1. **Triangular-Masked Convolution (TMC)**:

    - **Function**: Restricts the convolutional kernel's receptive field to the upper-triangular region, serving as the foundation for constructing the diamond-shaped blind spot.
    - **Mechanism**: A binary mask $M_{ij} = 1 \text{ if } i \leq j$ is applied to the 3×3 convolutional kernel, zeroing out the lower-triangular elements. Stacking multiple TMC layers progressively expands the receptive field along the upper-triangular direction. Combined with a feature-shifting operation (shifting the feature map upward or rightward by $s$ pixels), the target pixel is excluded from its own receptive field, forming the blind spot.
    - **Design Motivation**: Conventional BSNs employ rectangular blind spots, which do not match the diamond-shaped spatial correlation region produced by demosaicing. TMC geometrically aligns the blind-spot shape with this diamond pattern, maximizing the use of uncorrelated contextual information.

2. **Four-Branch Rotational Aggregation**:

    - **Function**: Combines triangular receptive fields from four rotation directions to form a complete diamond-shaped blind spot.
    - **Mechanism**: The input image is rotated by 0°, 90°, 180°, and 270°, each processed by a separate TMC branch with feature shifting. Each branch's triangular receptive field covers only one directional quadrant; aggregating all four branches yields a complete diamond-shaped blind spot. Vertical and horizontal shifted features are concatenated to avoid the discontinuous coverage introduced by diagonal shifting.
    - **Design Motivation**: A single triangular mask can only form a unidirectional blind spot; multi-directional rotation is necessary to construct a symmetric diamond region while ensuring that all uncorrelated pixels outside the blind spot remain accessible.

3. **Knowledge Distillation (Recharged Distillation)**:

    - **Function**: Transfers complementary knowledge from multiple teacher predictions with varying blind-spot sizes into a lightweight student network, balancing accuracy and efficiency.
    - **Mechanism**: TM-BSN can efficiently generate multiple blind-spot predictions by varying the shift offset $s$, incurring only approximately 15% additional computation. The Recharged Distillation framework is adopted: a random subset of noisy pixels is injected into each teacher output, and a lightweight U-Net student network is trained with an L1 loss. The student network is not subject to blind-spot constraints and can directly access target pixel information.
    - **Design Motivation**: Different blind-spot sizes provide complementary restoration cues; distillation integrates these cues and enables the student to surpass the performance ceiling of any single blind-spot prediction.

### Loss & Training

- **TM-BSN Training**: Self-supervised L1 loss with shift offset $s=5$; Adam optimizer for 500k iterations.
- **Distillation Training**: Recharged Distillation loss $\mathcal{L}_{RD} = \sum_{s_i \in S} \| f_D(y) - \text{sg}[T_{s_i} \odot (1-M_i) + y \odot M_i] \|_1$, with inference offset set $S=\{2,3,4,5,6\}$; student U-Net (1.02M parameters) trained for 200k iterations.

## Key Experimental Results

### Main Results

| Dataset | Metric | TM-BSN (D) | APR (RD) | TBSN | AT-BSN (D) |
|--------|------|------------|----------|------|------------|
| SIDD Val | PSNR | **38.08** | 38.00 | 37.71 | 37.88 |
| SIDD Benchmark | PSNR | **38.31** | 38.26 | 38.02 | 38.14 |
| DND Benchmark | PSNR | **39.41** | 38.83 | 39.08 | 38.68 |
| DND (fully self-supervised) | PSNR | **38.96** | 38.57 | - | 38.29 |

### Ablation Study

| Configuration | SIDD Val PSNR | Note |
|------|---------------|------|
| Training $s=4$ | Severe degradation | Offset too small; correlated pixels not blocked; identity mapping occurs |
| Training $s=5$ | **37.31** | Optimal balance: avoids identity mapping while leveraging nearby information |
| Training $s=6$ | Suboptimal | Offset too large; useful context discarded |
| Training $s=7$ | Suboptimal | Excessive offset further reduces information |
| Distillation $S=\{1,2,3,4,5\}$ | Suboptimal | $s=1$ deviates too far from training offset; unstable teacher signals |
| Distillation $S=\{2,3,4,5,6\}$ | **Best** | Diverse yet reliable supervision targets |
| Distillation $S=\{3,4,5,6,7\}$ | Suboptimal | Large offsets limit information utilization |

### Key Findings

- Training offset $s=5$ is the optimal balance: $s=4$ is too small and causes identity mapping, while $s \geq 6$ is too large and discards useful context.
- TM-BSN (D) after distillation achieves +0.33 dB on DND over TBSN, with only 1.02M parameters and 3.21 ms inference time.
- In terms of efficiency, TM-BSN (D) requires only 26.74 GFLOPs and 3.21 ms inference, far outperforming TBSN (5463.9 GFLOPs, 1004.6 ms).

## Highlights & Insights

- **Diamond Blind-Spot Design**: This work is the first to precisely align the blind-spot shape with the spatial correlation geometry of noise, rather than relying on a simple rectangular blind spot. This approach underscores the importance of understanding the physical origin of noise (the interpolation pattern of demosaicing) for designing better denoising architectures.
- **Efficient Multi-Scale Prediction**: By sharing feature extraction and applying different shift offsets, the method generates multiple complementary predictions at only approximately 15% additional computation—a "extract once, use multiple times" design philosophy transferable to other tasks requiring multi-scale predictions.
- **Distillation Breaks the Blind-Spot Performance Ceiling**: The blind-spot constraint inherently limits information utilization; distilling into a student network free of blind-spot constraints overcomes this bottleneck.

## Limitations & Future Work

- The diamond blind-spot design assumes that the noise correlation pattern originates from standard Bayer CFA demosaicing; adjustments to the blind-spot shape may be required for non-standard CFAs or alternative ISP pipelines.
- The choice of training offset $s$ relies on ablation experiments, and an adaptive mechanism for determining the optimal offset is lacking.
- Validation is conducted only on SIDD and DND, without coverage of other self-supervised denoising scenarios such as medical imaging.
- Extending the diamond blind-spot concept to video denoising by designing three-dimensional blind spots that exploit spatiotemporal correlations is a promising direction.

## Related Work & Insights

- **vs. AP-BSN**: AP-BSN applies pixel-shuffle downsampling to decorrelate noise, requiring post-processing to remove checkerboard artifacts, and severely restricts the receptive field. TM-BSN operates at full resolution with no post-processing required.
- **vs. AT-BSN**: AT-BSN uses asymmetric operations to form rectangular blind spots at full resolution, but the rectangular shape mismatches the diamond correlation pattern, excluding uncorrelated corner pixels. TM-BSN's diamond blind spot provides a more precise fit.
- **vs. TBSN**: TBSN employs Transformer attention blocks with enormous computational overhead (5463 GFLOPs), whereas TM-BSN (D) achieves superior performance with only 26.7 GFLOPs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The diamond blind-spot design is physically motivated and the triangular-masked convolution implementation is elegant, though the overall framework remains an improvement within the BSN paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation on the standard SIDD and DND benchmarks with detailed ablation studies and complexity analysis, though the variety of datasets is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The derivation chain from the physical origin of noise to the architectural design is clear and complete, with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Achieves new state-of-the-art in self-supervised denoising with strong practical utility; the ideas offer inspiration for other tasks that exploit structural priors of noise.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](selfhvd_self-supervised_handheld_video_deblurring.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](../../ICCV2025/image_restoration/self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)

<!-- RELATED:END -->
