---
title: >-
  [Paper Note] SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion
description: >-
  [ICLR 2026][Video Generation][Video diffusion models] SIGMark proposes the first blind watermarking framework for modern video diffusion models…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "Video diffusion models"
  - "watermarking"
  - "blind extraction"
  - "pseudorandom coding"
  - "causal 3D VAE"
  - "temporal robustness"
date: 2026-05-08
content_hash: cfb97413ff50457e
---

# SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion

**Conference**: ICLR 2026
**arXiv**: [2603.02882](https://arxiv.org/abs/2603.02882)
**Code**: [https://github.com/JeremyZhao1998/SIGMark-release](https://github.com/JeremyZhao1998/SIGMark-release)
**Area**: Video Generation
**Keywords**: Video diffusion models, watermarking, blind extraction, pseudorandom coding, causal 3D VAE, temporal robustness

## TL;DR
SIGMark proposes the first blind watermarking framework for modern video diffusion models, achieving scalable blind extraction with constant retrieval cost via Global Frame-level Pseudorandom Coding (GF-PRC), and addresses temporal perturbations under causal 3D VAE through a Segmented Group Ordering (SGO) module, attaining high bit accuracy and strong robustness on HunyuanVideo and Wan-2.2.

## Background & Motivation

**Background**: Video diffusion models (e.g., HunyuanVideo, Wan-2.2) are advancing rapidly, making invisible watermarking a critical technique for protecting copyright and tracing harmful AI-generated content. Existing approaches fall into two categories: post-processing watermarking (which degrades video quality) and in-generation watermarking (theoretically lossless but with notable limitations).

**Limitations of Prior Work**: Existing in-generation watermarking methods (e.g., VideoShield, VideoMark) are **non-blind** — extraction requires maintaining all message–key pairs and performing template matching, incurring costs that grow linearly with the number of users/requests and thus fail to scale to large platforms.

**Key Challenge**: Modern video diffusion models employ a causal 3D VAE that decodes a group of $d_t$ frames from a single temporal latent feature. Temporal perturbations (frame deletion, cropping) disrupt frame groupings, causing the VAE encoder to produce incorrect latent features and making watermark extraction **highly fragile to temporal distortions**.

**Goal**: (1) How to achieve blind watermark extraction with constant computational cost? (2) How to maintain temporal robustness under the causal 3D VAE?

**Key Insight**: Replace per-request key storage with globally shared frame-level PRC keys, and design an optical-flow segmentation plus sliding-window detection scheme to recover correct frame groupings.

**Core Idea**: Global frame-level pseudorandom coding enables blind extraction, while the SGO module recovers temporal ordering, reducing extraction complexity from linear to constant.

## Method

### Overall Architecture
SIGMark consists of an embedding stage and an extraction stage. During embedding, the watermark message is encoded into the initial latent noise via globally shared GF-PRC keys; the diffusion model then denoises this noise to produce a losslessly watermarked video. During extraction, the SGO module first recovers frame groupings from the (potentially perturbed) video, followed by diffusion inversion to obtain the latent noise, from which the message is decoded using the GF-PRC keys.

### Key Designs

1. **Global Frame-level Pseudorandom Coding (GF-PRC)**:

    - **Function**: Encodes the watermark message into the initial latent noise to enable blind extraction.
    - **Mechanism**: A set of global frame-level PRC keys $K[i]$ is maintained, one per temporal dimension of the latent space. During embedding: $\text{TP}[i] = \text{PRC.Encode}(m[i]; K[i])$, then mapped to noise via sign modulation: $z_0(m) = (\text{TP} \times 2 - 1) \times |z_0|$.
    - **Design Motivation**: PRC encoding maps the same message to different random template bits even under a shared global key, preserving noise diversity. Conventional stream ciphers (e.g., ChaCha20) cannot achieve this with fixed key material. Extraction requires only the global key, reducing complexity from $O(N)$ to $O(1)$.

2. **Segmented Group Ordering Module (SGO)**:

    - **Function**: Recovers causal frame groupings disrupted by temporal perturbations.
    - **Mechanism**: A two-step procedure — (1) **Optical flow segmentation**: bidirectional Farnebäck optical flow is computed between adjacent frames; temporal cut points are detected using median flow magnitude, forward-backward consistency, and motion-compensated residuals, partitioning the video into motion-consistent segments. (2) **Sliding window detection**: within each segment, a window padded with $d_t - 1$ frames slides across positions; for each window position $j$, the latent frame is obtained by inversion and its index is determined as $\hat{\text{Idx}[j]} = \text{argmax}(\text{PRC.Detect}(z_0'[j]; K[0,1,...,f_l]))$; detection halts when consecutive results are consistent.
    - **Design Motivation**: The causal 3D VAE requires correct frame groupings to produce consistent latent features. SGO exploits the frame-index detection capability of the global PRC keys to robustly recover correct groupings.

3. **Message Extraction**:

    - The regrouped frames are encoded by the causal 3D VAE and inverted to obtain $z_0'$.
    - The message is recovered via PRC decoding: $\hat{m[i]} = \text{PRC.Decode}(\frac{\text{Sgn}(z_0'[i])+1}{2}; K[i])$.
    - No original message storage or template matching is required.

### Loss & Training
SIGMark is a **training-free** method that embeds watermarks entirely at inference time. The mathematical transformation preserves the Gaussian distribution of the noise, $z_0(m) \sim \mathcal{N}(0, \mathbf{I})$, theoretically leaving generation quality unaffected. Inversion uses flow-matching Euler discrete inversion (for HunyuanVideo and Wan-2.2) conditioned on an empty prompt.

## Key Experimental Results

### Main Results (HunyuanVideo T2V, 512 bits)

| Method | Type | Bit Acc↑ | V-score↑ |
|--------|------|----------|----------|
| No-mark | – | – | 0.490 |
| DCT | Post-processing | 0.889 | 0.424 |
| VideoMark | Non-blind | 0.873 | 0.507 |
| VideoShield | Non-blind | 1.000 | 0.497 |
| **SIGMark** | **Blind** | **0.958** | **0.506** |

In high-capacity mode (512×16 bits), SIGMark achieves 0.885 bit accuracy, surpassing VideoMark (0.758).

### Robustness Results (HunyuanVideo I2V)

| Method | No distortion | Gaussian noise | Compression | Temporal drop | Temporal insert |
|--------|--------------|----------------|-------------|---------------|-----------------|
| VideoMark | 0.85 | 0.64↓0.21 | 0.63↓0.22 | 0.52↓0.19 | 0.51↓0.20 |
| VideoShield | 1.00 | 1.00↓0.00 | 0.99↓0.01 | 0.89↓0.10 | 0.84↓0.15 |
| **SIGMark** | **0.98** | **0.89↓0.09** | **0.84↓0.14** | **0.81↓0.10** | **0.87↓0.04** |

### Ablation Study

| Configuration | Bit Acc | Notes |
|---------------|---------|-------|
| Single PRC (non-GF) | 0.707 | Without global frame-level encoding |
| GF-PRC (full) | 0.905 | Significant gain from frame-level encoding |
| w/o SGO | 0.534 | Temporal robustness collapses |
| w/o OF-seg | 0.762 | Without optical flow segmentation |
| w/o SW-det | 0.823 | Without sliding window detection |
| SGO (full) | 0.869 | Both components are complementary |

### Key Findings
- Post-processing watermarks significantly degrade video quality (V-score drops from 0.490 to ~0.42), while in-generation methods are nearly lossless.
- GF-PRC not only enables blind extraction but also provides additional error correction through inter-frame redundancy.
- Both components of the SGO module (optical flow segmentation and sliding window detection) are indispensable.
- Extraction time for VideoShield grows linearly with the number of videos, whereas SIGMark remains constant.

## Highlights & Insights
- **Paradigm shift from non-blind to blind extraction**: SIGMark is the first video diffusion watermarking method to achieve true blind extraction, reducing extraction complexity from $O(N)$ to $O(1)$, which is essential for large-scale platform deployment. The key insight is that PRC's pseudorandom mapping is compatible with globally shared keys.
- **Systematic solution for temporal robustness**: The SGO module is specifically designed for the characteristics of causal 3D VAE and serves as a general frame-grouping recovery scheme transferable to other tasks requiring correct frame groupings.
- **Training-free plug-and-play**: The entire framework requires no fine-tuning of model parameters and can be directly applied to any video diffusion model.

## Limitations & Future Work
- Bit accuracy does not reach 100%, constrained by the error-correction capacity of PRC encoding and the precision of diffusion inversion.
- Accuracy under spatial perturbations (e.g., 0.89 under Gaussian noise) still lags behind VideoShield (1.00).
- The SGO module introduces additional computational overhead due to optical flow computation.
- Hybrid strategies combining in-generation and post-processing watermarking could be explored to further enhance robustness.

## Related Work & Insights
- **vs. VideoShield**: A non-blind method requiring storage of all message–key pairs with linear extraction cost. SIGMark achieves blind extraction at constant cost.
- **vs. VideoMark**: Also non-blind and PRC-based, but does not address global key sharing or temporal robustness under causal 3D VAE.
- **vs. Gaussian Shading**: An image watermarking method extended to video; does not account for the temporal characteristics of the causal 3D VAE.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First blind-extraction video diffusion watermarking framework; GF-PRC and SGO are elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two models, T2V/I2V settings, multiple perturbation types, comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated; method exposition is logically structured.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a practical deployment bottleneck in video watermarking; significant contribution to AI content security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](../../ICCV2025/video_generation/stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](target-aware_video_diffusion_models.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)

</div>

<!-- RELATED:END -->
