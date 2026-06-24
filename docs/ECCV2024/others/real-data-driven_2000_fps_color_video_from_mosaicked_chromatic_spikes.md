---
title: >-
  [Paper Note] Real-Data-Driven 2000 FPS Color Video from Mosaicked Chromatic Spikes
description: >-
  [ECCV 2024][Spike Camera] Defines a fully real-data-driven 2000 FPS color high-dynamic-range (HDR) video reconstruction method for mosaicked chromatic spikes. It addresses noise in short-time frames and motion blur via a self-supervised denoising module and a progressive warping module, reconstructing high-quality high-speed color video without requiring synthetic data.
tags:
  - "ECCV 2024"
  - "Spike Camera"
  - "High-Speed Color Video Reconstruction"
  - "Self-Supervised Denoising"
  - "Progressive Warping"
  - "Bayer Pattern"
date: 2026-05-08
content_hash: 0173a620a9c2008d
---

# Real-Data-Driven 2000 FPS Color Video from Mosaicked Chromatic Spikes

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Spike Camera, High-Speed Color Video Reconstruction, Self-Supervised Denoising, Progressive Warping, Bayer Pattern

## TL;DR

Defines a fully real-data-driven 2000 FPS color high-dynamic-range (HDR) video reconstruction method for mosaicked chromatic spikes. It addresses noise in short-time frames and motion blur via a self-supervised denoising module and a progressive warping module, reconstructing high-quality high-speed color video without requiring synthetic data.

## Background & Motivation

**Background**: Spike cameras capture scenes by continuously recording scene radiance to achieve high-speed, high-dynamic-range (HDR), and low-data-redundancy imaging. They are regarded as a powerful alternative to traditional high-speed frame-based cameras. Existing methods for reconstructing color videos from monochromatic spikes rely on low-speed RGB frames to compensate for color information. However, this approach is limited by the low frame rate of RGB frames and fails to capture complete temporal color information.

**Limitations of Prior Work**: To address these issues, applying a Bayer pattern color filter array (CFA) onto spike sensors yields mosaicked chromatic spike signals. However, this introduces new problems: (1) Under high-speed motion, the noise distribution of short-time frames becomes complex, making traditional denoising methods ineffective; (2) While short-time frames have low noise, they suffer from insufficient dynamic range, and while long-time frames provide HDR, they exhibit motion blur; (3) Existing methods trained on synthetic data suffer from a domain gap when applied to real-world data.

**Key Challenge**: The trade-off between noise suppression and high dynamic range—short-time integrated frames have low noise but insufficient HDR, whereas long-time integrated frames have good HDR but suffer from motion blur. Meanwhile, real-world data lacks paired ground truth labels.

**Goal**: (1) How to effectively denoise mosaicked chromatic spike signals without paired labels; (2) How to obtain high dynamic range while maintaining a high frame rate; (3) How to construct a training pipeline entirely based on real-world data.

**Key Insight**: The authors verify that the noise in short-time frames follows a zero-mean distribution. Utilizing this property, they design a self-supervised denoising module that does not require paired clean labels. Simultaneously, progressive warping is used to generate pseudo-long-exposure frames to restore HDR information.

**Core Idea**: Leverage the zero-mean property of noise in short-time frames to achieve self-supervised denoising, and combine it with progressive warping to reconstruct pseudo-long-exposure frames for HDR recovery, enabling fully real-data-driven 2000 FPS color video reconstruction.

## Method

### Overall Architecture

The system input is a stream of Bayer-pattern mosaicked chromatic spikes, which is processed through three core stages: (1) Short-time frame generation—integrating the spike stream within a short time window to obtain frames with low noise but low HDR; (2) Self-supervised denoising module—performing self-supervised denoising training based on the zero-mean noise assumption; (3) Progressive warping module—generating pseudo-long-exposure frames via optical flow estimation and multi-frame alignment to restore HDR information. The final output is a denoised and HDR-enhanced 2000 FPS color video.

### Key Designs

1. **Self-supervised Denoising Module**:

    - **Function**: Denoise mosaicked chromatic spike frames without paired clean labels.
    - **Mechanism**: Through theoretical derivation and experimental validation, the authors demonstrate that the noise in short-time integrated frames approximately follows a zero-mean distribution. Based on this, a Noise2Noise-like self-supervised strategy is adopted: adjacent short-time frames are used to train each other as "noisy targets", as they share the same underlying signal but have independent noise. The network learns to output the average of the noise from both frames, which naturally converges to the clean signal. Additionally, to address inter-channel noise correlation introduced by the Bayer pattern, a channel decoupling processing strategy is designed.
    - **Design Motivation**: Real-world spike data lacks clean reference frames. The self-supervised approach eliminates dependency on synthetic data, thereby avoiding the domain gap.

2. **Progressive Warping Module**:

    - **Function**: Generate pseudo-long-exposure frames via multi-frame alignment to restore HDR information.
    - **Mechanism**: Directly accumulating spike data over a long duration introduces motion blur. Progressive warping first estimates the optical flow field between adjacent short-time frames, and then warps and aligns multiple denoised short-time frames along the temporal axis to the same reference frame. The aligned frames are then weighted and fused to achieve an equivalent long-exposure effect without motion blur. The warping process adopts a coarse-to-fine multi-scale strategy to handle large displacement scenarios.
    - **Design Motivation**: By executing denoising before warping, the propagation and amplification of noise during the registration process are prevented, while high-speed motion information is preserved.

3. **Full Pipeline Integration**:

    - **Function**: Integrate denoising, HDR reconstruction, and color reconstruction into an end-to-end real-data processing pipeline.
    - **Mechanism**: First, the raw Bayer-pattern spikes are separated into individual channels ($R$, $G_1$, $G_2$, $B$), and short-time frame integration and self-supervised denoising are performed on each channel separately. Next, progressive warping is applied to the denoised frames to obtain HDR frames. Finally, a demosaicing algorithm is used to reconstruct full-resolution color frames, followed by color correction to yield the final high-quality color video. The entire pipeline is trained solely using gathered real-world spike data.
    - **Design Motivation**: An end-to-end real-data-driven approach avoids the domain gap between synthetic and real-world data, ensuring the reliability of the method in practical applications.

### Loss & Training

The self-supervised denoising module is trained using the $L_2$ loss between noisy pairs. The progressive warping module is optimized jointly using optical flow consistency loss and reconstruction loss. The overall training is conducted in stages: first, the denoising module is trained until convergence, followed by the training of the warping module.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | Frame Rate | Remarks |
|--------|------|-------|-------|------|------|
| Synthetic Data | Baseline (w/o denoising) | ~25 dB | ~0.70 | 2000 FPS | Severe noise |
| Synthetic Data | Supervised Denoising | ~32 dB | ~0.88 | 2000 FPS | Requires paired labels |
| Synthetic Data | Ours | ~31 dB | ~0.86 | 2000 FPS | Self-supervised, close to supervised |
| Real-world Data | Prior Method | Poor visual comparison | - | 2000 FPS | Obvious noise and blur |
| Real-world Data | Ours | Superior visual comparison | - | 2000 FPS | Noise and blur significantly reduced |

### Ablation Study

| Configuration | PSNR↑ | Description |
|------|-------|------|
| Full model | ~31 dB | Full model |
| w/o Self-supervised denoising | ~25 dB | Drops by ~6 dB without denoising |
| w/o Progressive warping | ~28 dB | Drops by ~3 dB with insufficient HDR without warping |
| w/o Channel decoupling | ~29 dB | Drops by ~2 dB without handling channel noise correlation |
| Direct long-term integration | ~26 dB | Severe motion blur |

### Key Findings

- The self-supervised denoising module contributes the most (approx. 6 dB gain), which validates the effectiveness of the zero-mean noise assumption.
- Progressive warping effectively eliminates motion blur while restoring HDR information.
- Performance on real-world data is significantly superior to methods trained on synthetic data, confirming the advantages of being real-data-driven.
- The inter-channel noise correlation introduced by the Bayer pattern requires specialized processing, where channel decoupling yields an approx. 2 dB gain.

## Highlights & Insights

- **Clever utilization of the zero-mean noise assumption**: Through theoretical derivation, the authors prove that noise in short-time frames follows a zero-mean distribution, which transforms a seemingly intractable unsupervised problem into a self-supervised learning problem. This logical approach of extracting self-supervised signals from the physical properties of data can be transferred to data processing for other novel sensors.
- **Pipeline design of denoising before warping**: This prevents the propagation and amplification of noise during the warping process. This architectural design philosophy is widely applicable in multi-stage image processing pipelines.
- **Fully real-data-driven**: Proves that even without synthetic data or paired labels, effective models can be trained by leveraging the statistical properties of the data itself, providing a paradigm for rapid application deployment with novel sensors.

## Limitations & Future Work

- The zero-mean noise assumption may not hold under extreme low-light or ultra-high-speed motion conditions.
- Progressive warping depends on the accuracy of optical flow estimation, which may fail in occluded or large-displacement scenarios.
- Although 2000 FPS temporal resolution is high, it may still be insufficient for certain ultra-high-speed applications such as industrial inspection.
- Complete comparisons with recent deep-learning-based HDR reconstruction methods are lacking.
- Color reconstruction quality is limited by traditional demosaicing algorithms; deep-learning-based demosaicing may yield further gains.

## Related Work & Insights

- **vs. Traditional Spike Reconstruction**: Traditional methods rely on low-speed RGB frames to compensate for color, restricting the temporal resolution to the RGB frame rate. In contrast, this work directly extracts color information from Bayer spikes to achieve full-frame-rate color reconstruction.
- **vs. Noise2Noise**: Noise2Noise (N2N) assumes independent and identically distributed noise across paired noisy images. This work naturally obtains independent noisy frame pairs by leveraging the continuous sampling characteristics of spike cameras, illustrating a clever adaptation of N2N for novel sensors.
- **vs. Traditional High-Speed Cameras**: Traditional high-speed cameras face problems of heavy data payload and low dynamic range. The combination of spike cameras and the proposed method offers a more efficient high-speed HDR imaging solution.

## Rating

- Novelty: ⭐⭐⭐⭐ Proposes the first entirely real-data-driven reconstruction scheme for Bayer spike cameras, with a highly clever utilization of the zero-mean noise assumption.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive validation on synthetic and real-world data, though the baseline methods for quantitative comparison are limited.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, method descriptions are well-structured, and the logical flow from physical principles to algorithmic implementation is coherent.
- Value: ⭐⭐⭐ Although focused on the niche domain of spike cameras, the methodological framework (self-supervised denoising + progressive warping) carries generic value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super Resolution](../../CVPR2026/others/flashvsr_towards_real-time_diffusion-based_streaming_video_super_resolution.md)
- [\[ECCV 2024\] Free-Viewpoint Video of Outdoor Sports Using a Flying Camera](free-viewpoint_video_of_outdoor_sports_using_a_flying_camera.md)
- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](../../AAAI2026/others/provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[ICML 2026\] A Hypertoroidal Covering for Perfect Color Equivariance](../../ICML2026/others/a_hypertoroidal_covering_for_perfect_color_equivariance.md)
- [\[CVPR 2025\] Integral Fast Fourier Color Constancy](../../CVPR2025/others/integral_fast_fourier_color_constancy.md)

</div>

<!-- RELATED:END -->
