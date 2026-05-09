---
title: >-
  [Paper Note] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression
description: >-
  [CVPR 2026][Model Compression][multi-view image compression] This paper proposes the OmniParallax Attention Mechanism (OPAM) for Distributed Multi-view Image Compression (DMIC), which explicitly models inter-view correlations and aligned features between arbitrary view pairs via a two-stage parallax attention mechanism. The resulting ParaHydra framework is the first DMIC method to significantly outperform state-of-the-art MIC encoders while substantially reducing computational overhead.
tags:
  - CVPR 2026
  - Model Compression
  - multi-view image compression
  - distributed coding
  - parallax attention
  - feature fusion
  - entropy model
date: 2026-05-08
content_hash: 2cb815023714f43d
---

# Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression

**Conference**: CVPR 2026
**arXiv**: [2603.03615](https://arxiv.org/abs/2603.03615)
**Code**: N/A
**Area**: Model Compression
**Keywords**: multi-view image compression, distributed coding, parallax attention, feature fusion, entropy model

## TL;DR

This paper proposes the OmniParallax Attention Mechanism (OPAM) for Distributed Multi-view Image Compression (DMIC), which explicitly models inter-view correlations and aligned features between arbitrary view pairs via a two-stage parallax attention mechanism. The resulting ParaHydra framework is the first DMIC method to significantly outperform state-of-the-art MIC encoders while substantially reducing computational overhead.

## Background & Motivation

**Background**: Multi-view image compression (MIC) exploits inter-view redundancy to improve compression efficiency and is widely used in autonomous driving, VR, and related domains. Distributed MIC (DMIC) follows distributed source coding theory, where each view is encoded independently and decoded jointly, without requiring cross-view information at the encoder.

**Limitations of Prior Work**: LDMIC, the first end-to-end DMIC framework, fuses multi-view features via average pooling, treating all views equally. This ignores the varying degrees of correlation among views—when reconstructing a floor region, views where the floor is visible and unoccluded should be prioritized over views where it is blocked by pedestrians.

**Key Challenge**: How to accurately measure and exploit semantic correlations among multiple information sources to achieve adaptive feature fusion rather than naive averaging.

**Goal**: (1) Efficiently capture inter-view correlations over the full 2D spatial context; (2) adaptively fuse multi-view features according to their correlations; (3) jointly leverage cross-view information in both the joint decoder and the entropy model.

**Key Insight**: Drawing from parallax attention in stereo matching (PAM), which computes attention only along horizontal epipolar lines, the authors generalize it to the full 2D spatial domain via a two-stage horizontal-then-vertical process for complete 2D context modeling.

**Core Idea**: A two-stage horizontal + vertical parallax attention achieves full 2D cross-view feature alignment and correlation measurement at $O(N^3)$ complexity, serving both joint decoding and entropy modeling.

## Method

### Overall Architecture

ParaHydra pipeline: each view image $x_k$ is independently encoded into a latent representation $y_k$; after quantization, Para-JD (Parallax Joint Decoder) jointly decodes the representations to produce reconstructed images. Entropy coding uses Para-EM (Parallax Entropy Model), which incorporates three types of priors: channel context, local spatial context, and global spatial context. The entire framework is optimized end-to-end with an R-D loss.

### Key Designs

1. **OmniParallax Attention Mechanism (OPAM)**:

    - **Function**: Explicitly models correlations between any two information sources and generates aligned features.
    - **Mechanism**: Operates in two stages — first, Horizontal Parallax Attention (HPA) computes cross-correlation between each position in the main source and the corresponding row in the side source, producing horizontally aligned features $f_l^{hor}$; then, Vertical Parallax Attention (VPA) further aligns along the column dimension. Alignment reliability is measured via cycle consistency $C_l = C_l^{hor} \odot C_l^{ver}$. Each position can ultimately attend to the full 2D spatial domain of the side source.
    - **Design Motivation**: The original PAM is restricted to the epipolar direction and cannot capture full 2D context; full 2D self-attention incurs $O(N^4)$ complexity, which is prohibitive. OPAM maintains $O(N^3)$ through the two-stage decomposition.

2. **Parallax Multi-Information Fusion Module (PMIFM)**:

    - **Function**: Adaptively fuses information from multiple side sources based on semantic correlations provided by OPAM.
    - **Mechanism**: For each side source $f_k$, OPAM produces aligned features $f_k^t$ and consistency $C_k$. All $C_k$ are concatenated and passed through softmax to obtain normalized weights $W$; a weighted sum $f_i^t = \sum_{k \neq i} W_k \cdot f_k^t$ is computed, and the result is merged with the original features via a lightweight fusion network.
    - **Design Motivation**: Compared to LDMIC's average pooling, PMIFM assigns higher weights to views with richer information and less occlusion, effectively suppressing noise.

3. **Parallax Entropy Model (Para-EM)**:

    - **Function**: Aggregates channel context, local spatial context, and global spatial context for more accurate entropy estimation.
    - **Mechanism**: PMIFM is introduced into the Parallax Channel Context Module (PCCM) — the notion of "information source" is redefined as channel slices, and OPAM measures inter-slice correlations for selective aggregation. The global context module (PGCM) similarly uses PCCM to extract cross-slice global features before performing anchor–non-anchor attention.
    - **Design Motivation**: Methods such as MLIC treat all channel slices equally, allowing low-information slices to introduce noise. Para-EM uses OPAM to measure inter-channel correlations for selective aggregation.

### Loss & Training

R-D loss: $L = \lambda D + R = \lambda \sum_k d(x_k, \hat{x}_k) + \sum_k (R(\hat{y}_k) + R(\hat{z}_k))$. $\lambda$ is set to 1024/2048/4096/8192 (MSE) or 32/64/128/256 (MS-SSIM). Models are trained for 1400 epochs on multi-view datasets and 3000 epochs on stereo datasets, with a learning rate of $10^{-4}$ on a single A30 GPU.

## Key Experimental Results

### Main Results (BDBR relative to LDMIC; negative values indicate bitrate savings)

| Method | InStereo2K(2) | WildTrack(3) | WildTrack(6) | Mip-NeRF 360(3) |
|--------|---------------|--------------|--------------|-----------------|
| VVC | +48.68% | +49.47% | +25.16% | +7.14% |
| MV-HEVC | +84.84% | +31.84% | +10.01% | +41.15% |
| LDMIC | 0% | 0% | 0% | 0% |
| LMVIC | - | - | - | -14.30% |
| **ParaHydra** | **-6.92%** | **-19.72%** | **-24.18%** | **-18.20%** |

### Efficiency Comparison (InStereo2K 1024×832)

| Method | Encode (s) | Decode (s) | Params (M) | FLOPs (T) |
|--------|-----------|-----------|-----------|----------|
| LDMIC | 9.27 | 21.43 | 214.98 | 2.88 |
| **ParaHydra** | **0.27** | **0.33** | **105.25** | **1.78** |

### Key Findings

- ParaHydra is the first DMIC method to significantly outperform MIC encoders, saving 34.11% bitrate over LMVIC on Mip-NeRF 360(4).
- As the number of views increases from 3 to 6, the BDBR gain improves from −19.72% to −24.18%, demonstrating strong scalability.
- Encoding is accelerated by 34×, decoding by 65×, and parameter count is reduced by half.

## Highlights & Insights

- **Generality of the two-stage decomposition**: Decomposing 2D attention into horizontal + vertical 1D stages maintains $O(N^3)$ complexity and is transferable to multi-camera 3D reconstruction, multi-sensor fusion, and related tasks.
- **Flexible redefinition of "information source"**: OPAM is applied not only across views but also across channel slices for context modeling, demonstrating high mechanistic generality.

## Limitations & Future Work

- Each view uses an independent encoder, so parameter count scales linearly with the number of views.
- Cycle consistency may fail under severe occlusion or in regions with no overlap.
- Validation is limited to fixed-viewpoint multi-camera scenarios; dynamic viewpoints and large-baseline configurations remain untested.

## Related Work & Insights

- **vs. LDMIC**: Average pooling → OPAM + PMIFM adaptive fusion; BDBR savings of 19–24% and efficiency gains of 34–65×.
- **vs. LMVIC**: LMVIC relies on encoder-side 3D Gaussian priors (MIC paradigm); ParaHydra, as a DMIC method, surpasses LMVIC on Mip-NeRF 360.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The two-stage decomposition of OPAM generalizes PAM to full 2D with rigorous theoretical derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 4 datasets with varying view counts, including detailed efficiency and ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and architectural diagrams are clear.
- **Value**: ⭐⭐⭐⭐ The first DMIC method to surpass MIC; significant contribution to multi-view compression.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](../../AAAI2026/model_compression/hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)

<!-- RELATED:END -->
