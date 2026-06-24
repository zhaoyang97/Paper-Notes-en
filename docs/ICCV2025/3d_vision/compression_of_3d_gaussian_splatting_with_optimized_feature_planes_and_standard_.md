---
title: >-
  [Paper Note] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes CodecGS, which represents all Gaussian attributes via compact Tri-plane feature planes, combined with frequency-domain DCT entropy modeling and a channel-level bit allocation strategy, enabling efficient compression of feature planes using standard video codecs (HEVC). The method achieves storage sizes within ~10 MB while maintaining high rendering quality, yielding up to 146× compression over vanilla 3DGS.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "compression"
  - "feature planes"
  - "video codec"
  - "entropy modeling"
date: 2026-05-08
content_hash: c43db7bc7abc9aba
---

# Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs

**Conference**: ICCV 2025
**arXiv**: [2501.03399](https://arxiv.org/abs/2501.03399)  
**Code**: [https://fraunhoferhhi.github.io/CodecGS](https://fraunhoferhhi.github.io/CodecGS)  
**Area**: 3D Vision / 3DGS Compression
**Keywords**: 3D Gaussian Splatting, compression, feature planes, video codec, entropy modeling

## TL;DR

This paper proposes CodecGS, which represents all Gaussian attributes via compact Tri-plane feature planes, combined with frequency-domain DCT entropy modeling and a channel-level bit allocation strategy, enabling efficient compression of feature planes using standard video codecs (HEVC). The method achieves storage sizes within ~10 MB while maintaining high rendering quality, yielding up to 146× compression over vanilla 3DGS.

## Background & Motivation

- **3DGS** is renowned for high rendering quality and speed, but representing a 3D scene typically requires millions of Gaussians, consuming hundreds of MB to several GB of storage.
- This poses significant challenges for deployment on resource-constrained devices such as mobile phones and head-mounted displays.
- **Existing compression methods** primarily reduce Gaussian count and attribute sizes via point pruning and vector quantization (VQ), but tend to:
    - Focus on reducing rendering distortion while ignoring redundancy among Gaussian attributes
    - Fail to exploit spatial correlations among Gaussian attributes
    - Lack compatibility with mature video codec technology
- **Video codecs** (HEVC/VVC) are highly mature in rate-distortion optimization; prior work has applied them to NeRF compression, but the unstructured nature of 3DGS makes direct adaptation difficult.
- **Core Motivation**: Can 3DGS attributes be organized into structured 2D feature planes, thereby harnessing the efficient compression capabilities of standard video codecs?

## Method

### Overall Architecture

1. Train vanilla 3DGS for 15k iterations to complete point densification.
2. Predict all Gaussian attributes (color, scaling, rotation, opacity) using Tri-plane feature planes + MLP decoders.
3. Optimize feature planes for video codec compatibility via DCT-domain entropy modeling.
4. Apply channel importance scoring for bit allocation.
5. Compress feature planes using a standard HEVC encoder.

### Key Designs

1. **Tri-plane Feature Plane Representation**:

    - Adopts the static variant of k-planes (Tri-plane), decomposing 3D positions via Hadamard products across three planes (XY, XZ, YZ) to obtain compact features.
    - Each attribute (color, scaling, rotation, opacity) is decoded by an independent small MLP $g$.
    - Each plane has $512 \times 512$ resolution with 8 channels; 32 channels in total predict all attributes.
    - **Two-stage training**: vanilla 3DGS is first trained for 15k iterations to complete densification, then training switches to feature planes, avoiding conflicts between plane optimization and point densification.

2. **Progressive Training**:

    - Adopts a channel-level progressive masking strategy: at iteration stage $T_i$, only channels $[0, L_i]$ are updated.
    - $T_i = \{0, 5000, 10000, 15000\}$, $L_i = \{2, 4, 6, 8\}$.
    - Lower channels capture global/low-frequency information; higher channels capture high-frequency details.
    - This forms a multi-level representation that naturally supports subsequent channel-level bit allocation.

3. **Frequency-Domain DCT Entropy Modeling**:

    - Core observation: standard video codecs internally apply DCT for frequency-domain compression.
    - Rather than minimizing the entropy of spatial-domain parameters directly, the method applies block-level $N \times M$ DCT transform $\mathcal{F}$ to feature planes and minimizes the entropy of the transform coefficients: $I(\mathcal{F}(\mathcal{P}))$.
    - Block size $4 \times 4$ (consistent with the minimum Transform Unit in standard codecs).
    - Quantization step $Q_{\text{step}} = 2^8$ (paired with 16-bit scalar quantization).
    - Compared to $\mathcal{L}_1$ sparsification, DCT entropy modeling preserves signal in a block-wise approximation manner, more effectively maintaining original information.

4. **Channel Importance-based Bit Allocation**:

    - Channel importance scores $CI_c(\mathcal{P}) = \frac{1}{\sum P_i} \left| \frac{\partial E_i}{\partial \mathcal{P}_c} \right|$ measure each channel's sensitivity to visual quality.
    - Weights are defined as $w_c = CI_1 / CI_c$; higher channels (lower importance) receive larger weights and are thus compressed more aggressively.
    - Weighted entropy loss: $\mathcal{L}_{\text{ent}} = \sum_c w_c I(\mathcal{F}(\mathcal{P}_c))$.
    - Automatically determines per-channel bit allocation, avoiding exhaustive hyperparameter search.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{render}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}} + \lambda_1 \mathcal{L}_1$$

- $\mathcal{L}_{\text{render}}$: standard 3DGS rendering loss.
- $\mathcal{L}_{\text{ent}}$: DCT-domain weighted entropy loss, activated after 30k iterations.
- $\mathcal{L}_1$: $\ell_1$ regularization on feature planes to suppress noise in unoccupied regions.
- Channel importance is computed and fixed at 30k iterations.
- After training, feature planes are normalized to $[0,1]$ and scaled to 16-bit integers, then fed into HEVC in YUV400 format.

## Key Experimental Results

### Main Results (Table)

**Mip-NeRF360 / DeepBlending / Tank&Temples**:

| Method | PSNR↑ (360) | Size↓ (360) | PSNR↑ (DB) | Size↓ (DB) | PSNR↑ (T&T) | Size↓ (T&T) |
|--------|-------------|-------------|------------|------------|-------------|-------------|
| 3DGS | 27.49 | 745 MB | 29.42 | 664 MB | 23.69 | 431 MB |
| LightGaussian | 27.00 | 44.5 | 27.01 | 33.9 | 22.83 | 22.4 |
| C3DGS | 26.98 | 28.8 | 29.38 | 25.3 | 23.32 | 17.3 |
| CompGS | 27.26 | 16.5 | 29.69 | 8.77 | 23.71 | 9.61 |
| HAC | 27.53 | 15.3 | **30.19** | 7.46 | 23.70 | 8.44 |
| **Ours** | 27.30 | **9.78** | 29.82 | 8.62 | 23.63 | **7.46** |

> Achieves **76× compression** on Mip-NeRF360 with only 0.19 dB PSNR loss. All three datasets remain within ~10 MB.

### Ablation Study (Table)

| PC | $w_c$ | $\mathcal{L}_{\text{ent}}$ | PR | $\mathcal{L}_1$ | PSNR(dB) | Size(MB) |
|----|-------|----------|----|----|----------|----------|
| ✓ | | | | | 27.27 | 23.45 |
| ✓ | | | ✓ | | 27.40 | 22.81 |
| ✓ | | ✓ | ✓ | | 27.31 | **10.68** |
| ✓ | ✓ | ✓ | ✓ | | 27.29 | 9.96 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 27.30 | 9.78 |

> - $\mathcal{L}_{\text{ent}}$ is the most critical component: its introduction reduces size from 22.81 MB to 10.68 MB with only 0.09 dB PSNR degradation.
> - Channel bit allocation $w_c$ further reduces size to 9.96 MB.
> - Each component contributes to compression performance.

### Key Findings

- **DCT entropy modeling** significantly outperforms simple $\mathcal{L}_1$ sparsification: the latter causes notable quality degradation, whereas DCT preserves signal via block-wise approximation.
- **Progressive training** naturally forms a multi-level representation: lower channels concentrate energy, higher channels become sparse, providing a basis for differentiated bit allocation.
- **Comparison with HAC/CompGS**: these methods leverage anchor relationships based on Scaffold-GS, whereas the proposed method starts from vanilla 3DGS, making the two approaches complementary.
- **Generality of video codecs**: the method is compatible with both HM (reference implementation) and FFmpeg libx265 (hardware-accelerated), with the latter requiring only 25 s for encoding.
- **Piecewise-projective contraction** effectively maps unbounded 360° scenes to bounded planes, enhancing spatial correlations.

## Highlights & Insights

- **Standard video codec integration** is the primary highlight: it leverages decades of accumulated video coding technology with no custom decoder required and widely available hardware decoding.
- The design insight behind **DCT-domain entropy modeling** is elegant: since codecs internally use DCT, the training directly optimizes the distribution of DCT coefficients.
- **Channel importance-based automatic bit allocation** avoids exhaustive hyperparameter tuning, offering strong practical utility.
- The combination of progressive training and channel importance forms a complete "coarse-to-fine + differentiated compression" strategy.
- The method is independent of the densification process and fully compatible with the vanilla 3DGS rendering pipeline with no additional rendering overhead.

## Limitations & Future Work

- Training time is relatively long (~90 min/scene), primarily limited by the convergence speed of grid-based methods.
- Point pruning is not incorporated; the storage proportion attributed to point positions grows as compression ratio increases, leaving room for further optimization.
- SSIM/LPIPS on Tank&Temples are slightly below vanilla 3DGS, indicating that high-frequency detail fidelity can still be improved.
- Currently uses fixed QP=1 encoding; more flexible rate control strategies may further optimize rate-distortion performance.
- VVC (next-generation codec) could be explored as a replacement for HEVC to achieve better compression efficiency.

## Related Work & Insights

- **VideoRF / TeTriRF**: pioneered the use of standard video codecs for NeRF tensor decomposition compression, but are not applicable to unstructured 3DGS.
- **HAC / CompGS**: 3DGS compression methods based on Scaffold-GS that exploit anchor relationships; complementary to the proposed approach.
- **Self-Organizing Gaussian**: compresses Gaussian attributes using sorting algorithms combined with image codecs.
- **k-planes**: theoretical foundation for Tri-plane feature decomposition.
- Insight: converting unstructured 3D representations into structured 2D representations bridges mature 2D compression technologies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of DCT-domain entropy modeling and channel importance-based bit allocation is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three standard datasets, RD curves, detailed ablations, and codec comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear, figures and tables are informative, and ablation hierarchy is well-structured.
- **Value**: ⭐⭐⭐⭐ — Achieves seamless integration of 3DGS with standard video codecs with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CF³: Compact and Fast 3D Feature Fields](cf3_compact_and_fast_3d_feature_fields.md)
- [\[ICCV 2025\] Neural Compression for 3D Geometry Sets](neural_compression_for_3d_geometry_sets.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2025\] GIFStream: 4D Gaussian-based Immersive Video with Feature Stream](../../CVPR2025/3d_vision/gifstream_4d_gaussian-based_immersive_video_with_feature_stream.md)
- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)

</div>

<!-- RELATED:END -->
