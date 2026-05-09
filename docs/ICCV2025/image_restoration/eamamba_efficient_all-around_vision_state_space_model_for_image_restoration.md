---
title: >-
  [Paper Note] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration
description: >-
  [ICCV 2025][Image Restoration][Vision Mamba] This paper proposes EAMamba, a framework that introduces a Multi-Head Selective Scan Module (MHSSM) and an all-around scanning strategy to achieve multi-directional scanning without increasing computational complexity or parameter count. EAMamba addresses the computational overhead and local pixel forgetting issues of Vision Mamba in image restoration, achieving 31–89% FLOPs reduction while maintaining competitive performance across super-resolution, denoising, deblurring, and dehazing tasks.
tags:
  - ICCV 2025
  - Image Restoration
  - Vision Mamba
  - state space model
  - multi-head selective scan
  - all-around scanning
date: 2026-05-08
content_hash: 6c1049508eb5e272
---

# EAMamba: Efficient All-Around Vision State Space Model for Image Restoration

**Conference**: ICCV 2025
**arXiv**: [2506.22246](https://arxiv.org/abs/2506.22246)
**Code**: [https://github.com/daidaijr/EAMamba](https://github.com/daidaijr/EAMamba)
**Area**: Image Restoration
**Keywords**: Vision Mamba, state space model, image restoration, multi-head selective scan, all-around scanning

## TL;DR
This paper proposes EAMamba, a framework that introduces a Multi-Head Selective Scan Module (MHSSM) and an all-around scanning strategy to achieve multi-directional scanning without increasing computational complexity or parameter count. EAMamba addresses the computational overhead and local pixel forgetting issues of Vision Mamba in image restoration, achieving 31–89% FLOPs reduction while maintaining competitive performance across super-resolution, denoising, deblurring, and dehazing tasks.

## Background & Motivation
- **Background**: Image restoration has evolved from CNN → Vision Transformer → Vision Mamba, with Mamba emerging as a promising direction for modeling long-range dependencies at linear complexity.
- **Limitations of Prior Work**: Existing Vision Mamba methods (e.g., MambaIR) adopt 2D Selective Scan (2DSS), where each additional scanning direction incurs proportional increases in computation and parameters, limiting the scalability of scanning patterns.
- **Key Challenge**: (1) The computational complexity of 2DSS scales linearly with the number of scan sequences; (2) 2D scanning causes "local pixel forgetting"—spatially adjacent pixels may be far apart after flattening into a 1D sequence, losing local spatial relationships.
- **Goal**: Increase the number of scanning directions to capture more comprehensive spatial information while maintaining or reducing computational complexity.
- **Key Insight**: Drawing inspiration from multi-head attention, channels are grouped and scanned independently in different directions.
- **Core Idea**: Multi-Head Selective Scan (MHSS) via channel grouping eliminates the computational overhead of multi-directional scanning, making all-around scanning feasible.

## Method

### Overall Architecture
EAMamba adopts a UNet encoder-decoder architecture comprising a 4-level encoder (with [4, 6, 6, 7] MambaFormer blocks per level), a bottleneck module, a 4-level decoder, and a refinement module. Low-quality input images are passed through the encoder for multi-scale feature extraction; the decoder progressively reconstructs a residual image, which is added to the original input to produce the high-quality output. The core innovation resides in the MHSSM module within each MambaFormer block.

### Key Designs
1. **Multi-Head Selective Scan Module (MHSSM)**:

    - **Function**: Replaces standard 2DSS to efficiently process multi-directional 1D sequences.
    - **Mechanism**: The input feature $X \in \mathbb{R}^{H \times W \times C}$ is divided into $n$ groups along the channel dimension; each group independently performs scanning in a distinct direction, and the outputs are concatenated. The processing pipeline is: $Y = \text{LN}(\text{MHSS}(\text{SiLU}(\text{DWConv2D}(\text{Linear}(X)))))$, $X_{out} = \text{Linear}(Y \otimes \text{SiLU}(\text{Linear}(X)))$
    - **Design Motivation**: Standard 2DSS applies each scanning direction over all channels, causing complexity to grow linearly with the number of directions. MHSS distributes directions across channel groups, keeping total complexity comparable to single-direction scanning and breaking the scalability bottleneck.

2. **All-Around Scanning Strategy**:

    - **Function**: Augments horizontal and vertical scanning with diagonal direction scanning.
    - **Mechanism**: Combines 2D scanning (horizontal + vertical + their reverses) and diagonal scanning (main diagonal + flipped diagonal + their reverses), covering 8 directions in total to span the complete spatial neighborhood.
    - **Design Motivation**: Effective receptive field (ERF) visualization reveals that 2D-only scanning fails to capture diagonal directional information, leading to local pixel forgetting. All-around scanning resolves this through complementary multi-directional coverage.

3. **MambaFormer Block**:

    - **Function**: Serves as the fundamental building unit for both encoder and decoder.
    - **Mechanism**: $X' = X + \text{MHSSM}(\text{LN}(X))$, $X'' = X' + \text{Channel MLP}(\text{LN}(X'))$, analogous to a Transformer block but with the attention layer replaced by MHSSM.
    - **Design Motivation**: Retains the residual connection and normalization structure of Transformers for ease of composition and extension.

### Loss & Training
- L1 loss function
- Progressive training strategy: initial patch size 128×128 / batch size 64, gradually scaled up to 384×384 / batch size 8
- AdamW optimizer, initial learning rate $3 \times 10^{-4}$, cosine annealing to $1 \times 10^{-6}$
- Data augmentation: random horizontal/vertical flipping and 90° rotation
- Total of 450K iterations

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | EAMamba | MambaIR(-UNet) | FLOPs Reduction |
|---|---|---|---|---|
| SIDD (real denoising) | PSNR | 39.87 | 39.89 | 41% (137G vs. 230G) |
| CBSD68 σ=50 (synthetic denoising) | PSNR | 28.62 | 28.61 | 89% (137G vs. 1290G) |
| GoPro (deblurring) | PSNR | 33.58 | — | +0.31 dB over SFNet |
| SOTS-Indoor (dehazing) | PSNR | 43.19 | — | +3.14 dB over DehazeFormer-L |
| RealSR ×4 (super-resolution) | PSNR | 29.60 | 29.53 | 40% (137G vs. 230G) |

### Ablation Study

| Configuration | Param. (M) | FLOPs (G) | Urban100 σ=25 PSNR | Notes |
|---|---|---|---|---|
| Baseline (2DSSM) | 31.1 | 286 | 33.00 | MambaIR baseline |
| + MHSSM | 25.3 | 137 | 32.89 | FLOPs halved, PSNR drops only 0.1% |
| + all-around scan | 25.3 | 137 | 32.93 | All-around scanning partially recovers performance |

| Scan Combination | SIDD | RealSR ×4 | GoPro | SOTS-Indoor |
|---|---|---|---|---|
| 2D + Diagonal (default) | 39.87 | 29.60 | 33.58 | 43.19 |
| 2D + Z-order | 39.82 | 29.58 | 33.51 | 43.20 |
| 2D + Hilbert | 39.83 | 29.51 | 33.66 | 43.07 |

### Key Findings
- MHSSM reduces FLOPs by approximately 50% with only ~0.1 dB PSNR degradation, demonstrating an excellent efficiency–performance trade-off.
- All-around scanning consistently outperforms pure 2D scanning across all tasks; ERF visualization intuitively demonstrates more complete local spatial coverage.
- The 2D + Diagonal combination achieves the most balanced performance across most tasks.
- EAMamba surpasses the second-best method by 0.31 dB on deblurring (GoPro) and 3.14 dB on dehazing (SOTS-Indoor).

## Highlights & Insights
- Multi-head channel grouping with directional assignment is an elegant design that decouples the number of scanning directions from computational cost.
- ERF visualization effectively and intuitively demonstrates spatial coverage differences across scanning strategies, providing a clear basis for design choices.
- The all-around scanning strategy offers good scalability, allowing flexible combination of scanning patterns tailored to specific tasks.
- The substantial gain on dehazing (+3.14 dB) is particularly noteworthy.

## Limitations & Future Work
- Channel grouping may cause information isolation between groups, lacking cross-group interaction mechanisms.
- Although efficient, all-around scanning may yield uneven coverage for irregular image regions (e.g., boundaries, non-rectangular shapes).
- The method still relies on fixed combinations of scanning patterns and does not explore adaptive or learnable scanning strategies.
- The advantage over non-Mamba methods such as Restormer on real denoising and super-resolution is not yet decisive.

## Related Work & Insights
- **MambaIR**: The first work applying Mamba to image restoration, but its 2DSS incurs high computational overhead.
- **VMambaIR**: Proposes Omni Selective Scan with more directions, but at a correspondingly higher computational cost.
- **Restormer**: A representative Transformer-based method that remains competitive on certain tasks.
- **Insights**: The multi-head grouping concept can be generalized to other tasks requiring multi-directional or multi-scale processing; the channel grouping strategy shares conceptual similarities with group convolution and depthwise separable convolution.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Multi-head selective scanning is a clever design; all-around scanning effectively addresses local pixel forgetting.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers four task types (denoising / super-resolution / deblurring / dehazing) with multi-dataset validation and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured presentation; ERF visualizations and architecture diagrams are of high quality; experimental comparisons are thorough.
- **Value**: ⭐⭐⭐⭐ — Provides a general solution for improving the efficiency of Vision Mamba in low-level vision tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](../../AAAI2026/image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[ICCV 2025\] Efficient Concertormer for Image Deblurring and Beyond](efficient_concertormer_for_image_deblurring_and_beyond.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)

<!-- RELATED:END -->
