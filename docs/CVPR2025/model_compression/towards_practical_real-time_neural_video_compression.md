---
title: >-
  [Paper Note] Towards Practical Real-Time Neural Video Compression
description: >-
  [CVPR 2025][Model Compression][Neural Video Coding] This paper proposes DCVC-RT, the first neural video codec to achieve 1080p real-time encoding and decoding on consumer-grade hardware with compression efficiency surpassing H.266/VTM. The core finding is that operational complexity (rather than computational complexity) acts as the actual speed bottleneck. Based on this, implicit temporal modeling and a single-scale low-resolution latent representation are designed…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Neural Video Coding"
  - "Real-time Coding"
  - "Operational Complexity"
  - "Implicit Temporal Modeling"
  - "Model Integerization"
date: 2026-05-08
content_hash: e07d0843808c097b
---

# Towards Practical Real-Time Neural Video Compression

**Conference**: CVPR 2025  
**arXiv**: [2502.20762](https://arxiv.org/abs/2502.20762)  
**Code**: [https://github.com/microsoft/DCVC](https://github.com/microsoft/DCVC)  
**Area**: Model Compression/Video Compression  
**Keywords**: Neural Video Coding, Real-time Coding, Operational Complexity, Implicit Temporal Modeling, Model Integerization

## TL;DR

This paper proposes DCVC-RT, the first neural video codec to achieve 1080p real-time encoding and decoding on consumer-grade hardware with compression efficiency surpassing H.266/VTM. The core finding is that operational complexity (rather than computational complexity) acts as the actual speed bottleneck. Based on this, implicit temporal modeling and a single-scale low-resolution latent representation are designed, achieving encoding/decoding speeds of 125/113 fps on an A100 GPU while saving 21% bitrate.

## Background & Motivation

Neural video codecs (NVCs) have surpassed traditional codecs (H.265/HM, H.266/VTM) in compression ratio, but real-time encoding remains the biggest obstacle for practical deployment:

1. **Limitations of Prior Work**: MobileNVC achieves real-time decoding but has lower compression efficiency than x264; C3 achieves efficient decoding but requires time-consuming optimization for encoding; DHVC-2.0 requires a 4-GPU pipeline for real-time decoding, which is infeasible on a single GPU.
2. **Misguided Conventional Wisdom**: Existing works focus heavily on reducing computational complexity (MACs). However, the authors' key discovery indicates that the speedup gained from reducing channel dimensions is linear rather than the expected quadratic, proving that computational complexity is not the main bottleneck.
3. **Overlooked Operational Complexity**: Memory I/O overhead (influenced by latent representation size $P_{size}$) and function call overhead (influenced by the number of modules $P_{num}$) are the actual bottlenecks in execution speed.

This finding opens up a new avenue for acceleration: maintaining computing capacity while focusing heavily on reducing operational complexity.

## Method

### Overall Architecture

DCVC-RT adopts a conditional coding paradigm: the current frame is directly transformed into a 1/8 resolution latent space through patch embedding, then concatenated with the temporal context from the previous frame to be jointly processed by the encoder-decoder. All explicit motion estimation/compensation modules are discarded, greatly simplifying the pipeline.

### Key Designs

**Design 1: Single Low-Resolution Latent Representation Learning**

- **Function**: Eliminate the high memory I/O overhead of large latent representation sizes caused by progressive downsampling.
- **Mechanism**: Use patch embedding to directly transform the input frame into a single scale of 1/8 resolution, where all key modules (encoder, decoder, feature extractor, reconstruction network) operate.
- **Design Motivation**: In traditional NVCs with layer-by-layer half-resolution downsampling and doubling channel counts, the latent representation size $P_{size}$ in high-resolution layers is very large. At a single 1/8 scale, the latent size corresponding to $C=256$ is $4 \cdot H \cdot W$, which is sufficient to maintain representation capability, while providing a larger receptive field than progressive downsampling (beneficial for temporal modeling). The encoding speed is 3.6 times faster than progressive downsampling, while the BD-Rate degrades by only 0.3%.

**Design 2: Implicit Temporal Modeling**

- **Function**: Eliminate the high number of modules $P_{num}$ introduced by complex motion estimation/compensation modules.
- **Mechanism**: Extract temporal context from the reconstructed latent representation of the previous frame using a single simple feature extractor, concatenate it with the current frame's latent representation along the channel dimension, and jointly handle temporal redundancy via the encoder-decoder. The motion coding branch is completely removed.
- **Design Motivation**: Although the motion coding branch has low computational cost (only 1/13 of conditional coding), it contains up to 123 module layers (accounting for more than half of the 225 conditional coding layers). The high frequency of function calls poses a speed bottleneck. The implicit method redistributes computational capacity to frame coding modules, yielding a 3.4x improvement in encoding speed.

Comparisons under different motion content show: the BD-Rate is improved by 0.4% in low-motion scenes, degraded by 3.2% in high-motion scenes, and improved by 4.7% in scene changes.

**Design 3: Module Bank Rate Control + Model Integerization**

- **Function**: Support flexible rate control and cross-device consistency.
- **Mechanism**: (a) Rate control: introduce a module bank to learn distinct hyperprior modules for different quantization parameters (QP), accurately estimating the distribution of hyper information $z$ and saving about 3% bitrate. (b) Integerization: convert the floating-point model to int16 deterministic computation ($v_i = \text{round}(512 \cdot v_f)$), using precomputed lookup tables to process nonlinear Sigmoid functions.
- **Design Motivation**: In DCVC-RT, hyper information $z$ accounts for more than 10% of the total bitrate (due to the absence of motion bitstreams), meaning a single factorized prior is not precise enough. 16-bit integerization ensures completely identical coding outputs across different platforms.

### Loss & Training

A joint distortion loss consisting of YUV and RGB color spaces is adopted, paired with hierarchical quality $\lambda$ interpolation settings. QPs are randomly sampled in the range of 0-63 to achieve variable bitrate with a single model.

## Key Experimental Results

### Main Results: BD-Rate Comparison (YUV420, Anchor: VTM-17.0)

| Method | Average BD-Rate | Encoding fps | Decoding fps |
|------|------------|--------|--------|
| VTM-17.0 (H.266) | 0.0% | 0.01 | 23.6 |
| HM-16.25 (H.265) | +42.4% | 0.05 | 39.6 |
| ECM-11.0 | -20.5% | 0.002 | 3.4 |
| DCVC-FM | -22.1% | 3.4 | 4.2 |
| **DCVC-RT (fp16)** | **-21.0%** | **125.2** | **112.8** |

### Speed Comparison (1080p, Different Devices)

| Device | DCVC-FM Encoding/Decoding | DCVC-RT Encoding/Decoding |
|------|----------------|----------------|
| A100 | 5.0 / 5.9 fps | 125.2 / 112.8 fps |
| RTX 4090 | 4.2 / 4.1 fps | 98.2 / 96.5 fps |
| RTX 2080Ti | 2.3 / 2.4 fps | 40.3 / 34.3 fps |

### Ablation Study

| Component | BD-Rate | Encoding Time (ms) |
|------|---------|------------|
| Progressive Downsampling + Explicit Motion (DCVC-FM) | -22.1% | ~200 |
| 1/8 Single-scale + Explicit Motion | -21.8% | ~55 |
| 1/8 Single-scale + Implicit Temporal | -21.0% | ~8 |

### Key Findings

1. DCVC-RT is more than 18 times faster than DCVC-FM with only a 1.1% BD-Rate penalty, achieving an excellent rate-distortion-complexity trade-off.
2. Real-time 1080p coding with 40 fps encoding and 34 fps decoding is achieved for the first time on a consumer-grade GPU (RTX 2080Ti).
3. Operational complexity analysis reveals that halving the computational workload (reducing channel dimensions by half) only yields a ~1.5x speedup instead of the theoretical 4x, demonstrating that operational complexity is indeed the bottleneck.
4. MACs are reduced from 2642G to 385G (an 85% reduction), providing the computational foundation for speed improvements.

## Highlights & Insights

1. **Profound Insight on Operational vs. Computational Complexity**: This finding shifts the paradigm of NVC acceleration from blindly reducing channel dimensions/layers to minimizing the number of modules and latent representation sizes.
2. **Practical Value of Implicit Temporal Modeling**: Although it suffers a slight degradation in high-motion scenes, the 3.4x speed increase is significantly more important than a 3.2% BD-Rate penalty in practical applications.
3. **End-to-End Practicality**: It concurrently addresses three major actual deployment challenges: real-time coding, rate control, and cross-device consistency.

## Limitations & Future Work

1. Implicit temporal modeling leads to a 3.2% BD-Rate degradation in high-motion scenes, which might not be suitable for videos with extreme motion.
2. Model integerization (int16 to int32 accumulators) leads to a 2.7% BD-Rate penalty, and precision loss can be further optimized.
3. The evaluation is conducted under the configuration of intra-period=-1, whereas practical applications require features like random access.
4. Future work can explore combining implicit temporal modeling with lightweight motion hints to strike a better balance between speed and performance on high-motion scenes.

## Related Work & Insights

- **DCVC Series (DC -> FM -> RT)**: Microsoft's NVC evolution path, transitioning focus from rate-distortion performance to physical deployment.
- **MobileNVC**: The first consumer-grade real-time NVC, but suffering from insufficient compression ratio; DCVC-RT successfully resolves this conflict.
- **Neural Image Codecs**: Experiences in real-time execution of image codecs (such as ELIC) provide references for video coding.
- Inspiration: The analysis framework for operational complexity can be extended to other deep learning inference acceleration scenarios beyond video coding.

## Rating

⭐⭐⭐⭐⭐ — A milestone work. It achieves real-time NVC on consumer-grade hardware for the first time while outperforming H.266 in compression efficiency, addressing the core blocking point of NVC practical deployment. The insights on operational complexity carry widespread influence. A high-quality integration of engineering and research from MSRA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Real-Time Neural Video Compression with Unified Intra and Inter Coding](../../CVPR2026/model_compression/real-time_neural_video_compression_with_unified_intra_and_inter_coding.md)
- [\[CVPR 2025\] CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[CVPR 2026\] Ultra-Fast Neural Video Compression](../../CVPR2026/model_compression/ultra-fast_neural_video_compression.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)

</div>

<!-- RELATED:END -->
