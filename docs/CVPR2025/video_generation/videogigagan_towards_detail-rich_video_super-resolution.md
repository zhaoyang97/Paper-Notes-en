---
title: >-
  [Paper Note] VideoGigaGAN: Towards Detail-rich Video Super-Resolution
description: >-
  [CVPR 2025][Video Generation][Video Super-Resolution] Ours proposes VideoGigaGAN, the first large-scale GAN-based video super-resolution model. By incorporating flow-guided feature propagation, an anti-aliasing module, and a high-frequency shuttle mechanism, it generates rich high-frequency details while maintaining temporal consistency, supporting $8\times$ super-resolution.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Super-Resolution"
  - "GAN"
  - "Temporal Consistency"
  - "High-Frequency Details"
  - "Anti-Aliasing"
date: 2026-05-08
content_hash: c193303e007b6e4d
---

# VideoGigaGAN: Towards Detail-rich Video Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2404.12388](https://arxiv.org/abs/2404.12388)  
**Code**: [Project Page](http://videogigagan.github.io)  
**Area**: Video Generation  
**Keywords**: Video Super-Resolution, GAN, Temporal Consistency, High-Frequency Details, Anti-Aliasing

## TL;DR

Ours proposes VideoGigaGAN, the first large-scale GAN-based video super-resolution model. By incorporating flow-guided feature propagation, an anti-aliasing module, and a high-frequency shuttle mechanism, it generates rich high-frequency details while maintaining temporal consistency, supporting $8\times$ super-resolution.

## Background & Motivation

Video super-resolution (VSR) faces two core challenges: **temporal consistency** and **high-frequency detail generation**. Existing methods like BasicVSR++ and TTVSR excel at temporal consistency, but due to regression training objectives, they tend to produce overly blurry results that lack high-frequency textures and details.

In the single-image super-resolution domain, GigaGAN generated rich details even at $8\times$ upscaling thanks to large-scale GAN training on billions of images. However, directly applying GigaGAN frame-by-frame to videos leads to severe temporal flickering and aliasing artifacts.

This paper reveals a fundamental contradiction in VSR—the **consistency-quality dilemma**: blurrier results naturally possess better temporal consistency, whereas the capacity of GANs to hallucinate high-frequency details inherently contradicts the temporal consistency objective. Previous VSR methods sacrificed high-frequency details via regression losses to trade for consistency, failing to truly resolve this dilemma.

The core key insight of VideoGigaGAN is to identify the crucial issues when applying GigaGAN to VSR (limited temporal receptive field, downsampling aliasing, and high-frequency flickering) and specifically design flow-guided feature propagation, anti-aliasing, and high-frequency shuttle mechanisms to simultaneously preserve both details and consistency.

## Method

### Overall Architecture

VideoGigaGAN is based on the asymmetrical U-Net architecture of the GigaGAN image upsampler (3 downsampling encoder blocks + $3+k$ upsampling decoder blocks). The overall pipeline is:

1. The low-resolution input video first passes through the **flow-guided feature propagation module** to obtain temporal-aware features.
2. These temporal features are fed into the **inflated GigaGAN** (a 3D version augmented with temporal modules).
3. The encoder employs **anti-aliasing modules** instead of strided convolutions to prevent aliasing.
4. High-frequency features are directly injected into the decoder via skip connections using the **high-frequency shuttle (HF shuttle)**.

### Key Designs

1. **Temporal Inflation**:
    - Function: To extend the 2D image GigaGAN into a 3D video model.
    - Mechanism: After the spatial self-attention in each decoder block, a 1D temporal convolution (kernel size=3, operating only along the temporal dimension) and temporal self-attention are added, both using residual connections. The discriminator is inflated in the same manner. All weights in the temporal layers are initialized to zero to ensure the behavior at the beginning of training matches the image upsampler.
    - Design Motivation: Directly utilizing 3D convolutions incurs excessive memory overhead; adding temporal modules solely on the decoder side is sufficient to improve consistency effectively.

2. **Flow-guided Feature Propagation**:
    - Function: To aggregate information across frames, handle large-motion scenarios, and ensure consistency between different clips.
    - Mechanism: Inspired by BasicVSR++, a bidirectional recurrent neural network (BiRNN) is introduced before the inflated GigaGAN. A lightweight flow estimator (SpyNet) is first used to predict bidirectional optical flow. Combined with the original frame pixels, the RNN learns temporal-aware features. Finally, optical-flow-guided backward warping aligns the features. During inference, features are first generated for the entire video, and then processed independently in non-overlapping segments.
    - Design Motivation: The spatial window of temporal attention is limited and cannot model large motions that exceed its receptive field. Optical flow propagation provides global temporal alignment capabilities.

3. **Anti-aliasing + HF Shuttle**:
    - Function: To eliminate aliasing flickering caused by downsampling while reserving high-frequency details.
    - Mechanism: (i) Replace all strided convolutions in the encoder with stride=1 convolutions followed by low-pass filtering (BlurPool) and subsampling. (ii) At each resolution level, decompose features into low-frequency (low-pass filtered) and high-frequency (residual) components, transferring the high-frequency components directly to the decoder via skip connections.
    - Design Motivation: GAN training encourages high-frequency hallucination, which makes aliasing issues more severe than in regression methods. BlurPool resolves aliasing but over-smoothes the results; the HF shuttle is key to resolving the contradiction of "removing aliasing without losing details."

### Loss & Training

- **GAN Loss**: Non-saturating GAN loss ($\mu_{\text{GAN}}=0.05$)
- **R1 Regularization**: Discriminator gradient penalty ($\mu_{\text{R1}}=0.2048$)
- **LPIPS Loss**: Perceptual similarity ($\mu_{\text{LPIPS}}=5$)
- **Charbonnier Loss**: Smooth L1 loss ($\mu_{\text{Char}}=10$)
- Training Configuration: 32 A100 GPUs, batch size = 32, random crop of $64\times64$ patches with 10 frames per sample, learning rate of $5\times10^{-5}$, and total iterations of 100K.

## Key Experimental Results

### Main Results

$4\times$ super-resolution on the REDS4 dataset (LPIPS↓/PSNR↑):

| Method | LPIPS↓ | PSNR↑ | Characteristics |
|------|--------|-------|------|
| BasicVSR | 0.2023 | 31.42 | Regression-based |
| BasicVSR++ | 0.1786 | 32.39 | Regression-based |
| RVRT | 0.1727 | 32.74 | Regression-based, highest PSNR |
| **Ours** | **0.1582** | 30.46 | GAN-based, lowest LPIPS |

Multi-dataset comparison: VideoGigaGAN achieves the best LPIPS across all 6 evaluation settings (REDS4: 0.1582, Vimeo-90K-T: 0.1120, Vid4: 0.1925, UDM10: 0.1060).

### Ablation Study

Progressive ablation on the REDS4 dataset:

| Configuration | LPIPS↓ | $E_{\text{warp}}^{\text{ref}}\downarrow(\times10^{-3})$ |
|------|--------|---------------------|
| GigaGAN (per-frame) | 0.2031 | 2.497 |
| + Temporal attention | 0.2029 | 2.462 |
| + Flow propagation | **0.1551** | 2.187 |
| + BlurPool | 0.1621 | **2.152** |
| + HF shuttle | 0.1582 | 2.177 |

### Key Findings

1. **Optical flow propagation contributes most**: LPIPS drops from 0.2029 to 0.1551, and $E_{\text{warp}}^{\text{ref}}$ drops from 2.462 to 2.187.
2. **Trade-off between anti-aliasing and details**: BlurPool improves consistency but blurs the results, while HF shuttle restores details with only a minor sacrifice in consistency.
3. **Flaws in traditional $E_{\text{warp}}$ metric**: Bicubic interpolation actually yields a lower $E_{\text{warp}}$ than the ground truth (GT), showing that $E_{\text{warp}}$ favors over-smoothed results. Hence, $E_{\text{warp}}^{\text{ref}}$ is proposed as a more robust reference.
4. PSNR does not accurately reflect human perception—VideoGigaGAN gets a lower PSNR but achieves superior visual quality and LPIPS.

## Highlights & Insights

1. **Explicit formalization of the consistency-quality dilemma**: The fundamental contradiction between "consistency and details" in VSR is systematically analyzed for the first time, along with concrete solutions.
2. **Elegant frequency separation design**: The integration of BlurPool and HF shuttle achieves "aliasing removal without loss of detail"—routing low-frequency components through the standard path to ensure consistency, while passing high-frequency details through a shortcut.
3. **Single forward inference**: Unlike diffusion-based approaches, VideoGigaGAN generates results in a single forward pass, providing significantly faster inference speeds.
4. **New evaluation metric $E_{\text{warp}}^{\text{ref}}$**: Exposes the bias issues present in classical $E_{\text{warp}}$.

## Limitations & Future Work

1. GAN-based methods lack generative diversity compared to diffusion models.
2. The training cost is high (utilizing 32 A100 GPUs).
3. The accuracy of the optical flow estimation heavily influences the final output quality.
4. Currently, only $4\times$ and $8\times$ super-resolution are demonstrated, and performance at higher ratios remains unexplored.
5. Future work could benefit from merging diffusion model strengths or exploring more lightweight architectures.

## Related Work & Insights

- **vs BasicVSR++**: BasicVSR++ uses second-order grid propagation and deformable alignment, yielding excellent temporal consistency but blurry details; VideoGigaGAN integrates GAN generation into its feature propagation logic.
- **vs Upscale-A-Video**: Concurrent work using diffusion-based video super-resolution requires iterative denoising, whereas VideoGigaGAN is faster with a single forward pass.
- **vs GigaGAN**: The image-based GigaGAN excels in detail generation but fails to maintain frame-by-frame consistency; VideoGigaGAN extends its capabilities to video via three key components.
- **vs LongVideoGAN**: LongVideoGAN uses a sliding window for video super-resolution but is restricted to low-diversity datasets; VideoGigaGAN can handle generic in-the-wild scenes.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce large-scale GANs to VSR, with an elegant frequency-decomposition anti-aliasing + HF shuttle design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets and metrics, including detailed ablation studies and introducing a new evaluation metric.
- Writing Quality: ⭐⭐⭐⭐ Strong problem analysis, with clear formulation of motivations and mechanics for each component.
- Value: ⭐⭐⭐⭐ Addresses the core challenge of VSR and offers practical solutions, serving as a highly valuable reference for the video generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2025\] BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatia.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[ECCV 2024\] Kalman-Inspired Feature Propagation for Video Face Super-Resolution](../../ECCV2024/video_generation/kalman-inspired_feature_propagation_for_video_face_super-resolution.md)
- [\[CVPR 2026\] Compressed-Domain-Aware Online Video Super-Resolution](../../CVPR2026/video_generation/compressed-domain-aware_online_video_super-resolution.md)

</div>

<!-- RELATED:END -->
