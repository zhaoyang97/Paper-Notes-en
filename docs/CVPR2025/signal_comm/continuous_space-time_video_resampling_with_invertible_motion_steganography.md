---
title: >-
  [Paper Note] Continuous Space-Time Video Resampling with Invertible Motion Steganography
description: >-
  [CVPR 2025][Signal & Communication][Space-Time Video Resampling] An Invertible Motion Steganography Module (IMSM) is proposed to embed motion information into low-frame-rate frames during video temporal downsampling, and accurately restore motion details via inverse transformation during upsampling. It supports continuous (non-integer) space-time resampling factors, significantly improving reconstruction quality while preserving the visual quality of downsampled frames.
tags:
  - "CVPR 2025"
  - "Signal & Communication"
  - "Space-Time Video Resampling"
  - "Invertible Steganography"
  - "Motion Preservation"
  - "Continuous Resampling Factors"
  - "Video Frame Interpolation"
date: 2026-05-08
content_hash: 00838a9ad76da8b9
---

# Continuous Space-Time Video Resampling with Invertible Motion Steganography

**Conference**: CVPR 2025  
**Code**: Unreleased  
**Area**: Video Processing / Signal & Communication  
**Keywords**: Space-Time Video Resampling, Invertible Steganography, Motion Preservation, Continuous Resampling Factors, Video Frame Interpolation

## TL;DR
An Invertible Motion Steganography Module (IMSM) is proposed to embed motion information into low-frame-rate frames during video temporal downsampling, and accurately restore motion details via inverse transformation during upsampling. It supports continuous (non-integer) space-time resampling factors, significantly improving reconstruction quality while preserving the visual quality of downsampled frames.

## Background & Motivation

**Background**: Space-Time Video Resampling requires simultaneous spatial and temporal downsampling and upsampling of videos, which is widely applied in video compression transmission, adaptive bitrate streaming, and display adaptation (e.g., flexible conversion from 24fps to 60fps, 4K to 1080p). Existing Video Frame Interpolation (VFI) and Video Super-Resolution (VSR) methods typically process temporal and spatial dimensions independently, lacking joint optimization.

**Limitations of Prior Work**: Inter-frame motion information is inevitably lost during temporal downsampling (frame-rate reduction). When subsequent temporal upsampling (frame interpolation) is performed, the motion details of discarded frames cannot be recovered, leading to artifacts such as motion blur, ghosting, and motion discontinuity in reconstructed frames. Furthermore, most existing methods only support fixed integer resampling factors (e.g., 2x, 4x), failing to flexibly adapt to arbitrary resampling rates in practical requirements.

**Key Challenge**: There is an information asymmetry between downsampling and upsampling—downsampling is lossy, and discarded motion information cannot be perfectly inferred by subsequent frame interpolation algorithms in the pixel domain. This contradiction is particularly prominent in high-motion scenarios (e.g., fast camera panning, object occlusion).

**Goal**: (1) Retain motion information during temporal downsampling so that it can be accurately recovered during upsampling; (2) Achieve continuous space-time resampling factors, supporting flexible transformation of arbitrary non-integer multiples.

**Key Insight**: Drawing inspiration from steganography, imperceptible motion information can be embedded into the pixels of downsampled frames. Invertible Neural Networks (INNs) are utilized to guarantee lossless round-trip information transfer: embedding during forward encoding and recovering during backward decoding.

**Core Idea**: An Invertible Neural Network is used to "hide" inter-frame motion information within minuscule pixel perturbations of downsampled frames. During upsampling, the hidden motion signals are extracted via inverse transformation, thereby converting the "lossy" temporal downsampling into a "near-lossless" information delivery process.

## Method

### Overall Architecture
The system consists of three core stages: (1) **Space-Time Downsampling**: Spatial and temporal downsampling (frame dropping) are performed on the original high-frame-rate video, while the motion information of discarded frames is encoded into the preserved frames via the forward process of IMSM; (2) **Transmission/Storage**: Only the low-resolution, low-frame-rate frames containing steganographic information are transmitted; (3) **Space-Time Upsampling**: The receiver extracts the hidden motion information from the retained frames via the inverse process of IMSM, and reconstructs high-frame-rate, high-resolution videos in combination with a spatial super-resolution network.

### Key Designs

1. **Invertible Motion Steganography Module (IMSM)**:

    - **Function**: Embed/extract inter-frame motion information in/from downsampled frames
    - **Mechanism**: An invertible neural network based on coupling layers is utilized to encode the motion field (optical flow) or motion features into subtle perturbations of the downsampled frame pixels. The forward process $f: (I_{low}, M) \rightarrow I_{steg}$ embeds motion information $M$ into the low-frame-rate frame $I_{low}$ to obtain the steganographic frame $I_{steg}$; the inverse process $f^{-1}: I_{steg} \rightarrow (I_{low}', M')$ accurately restores the motion information. Due to the bijective nature of the invertible network, there is theoretically zero information loss.
    - **Design Motivation**: Traditional frame interpolation can only "guess" intermediate motion, with accuracy limited by the assumptions of motion models. IMSM directly transmits the real motion information, fundamentally eliminating the uncertainty of motion estimation. Meanwhile, the steganographic perturbation is extremely small (PSNR degradation <0.5dB), without affecting the visual quality of the downsampled frames.

2. **Continuous Space-Time Resampling Network**:

    - **Function**: Support arbitrary non-integer temporal and spatial resampling factors
    - **Mechanism**: Implicit Neural Representation (INR) is adopted to model the temporal dimension continuously. By taking the temporal coordinate $t \in [0, 1]$ as continuous input, the network outputs frame features at corresponding times instead of interpolating only on fixed grid points. The spatial dimension is similarly encoded using continuous coordinates, enabling a single model to handle arbitrary spatial magnification factors.
    - **Design Motivation**: In practical scenarios, non-integer frame-rate conversion is frequently required (e.g., 24fps to 30fps requires a 1.25x temporal upsampling). Fixed integer methods require a two-step "upsample-then-downsample" transformation, introducing extra distortion.

3. **Motion-Aware Quality Constraint**:

    - **Function**: Ensure that the embedding process does not degrade the quality of downsampled frames, while maximizing the recovery accuracy of motion information
    - **Mechanism**: The training loss consists of three terms: (a) visual quality loss between the steganographic frame and the original frame $\mathcal{L}_{vis} = \|I_{steg} - I_{low}\|_1$ to control perturbation amplitude; (b) motion recovery loss $\mathcal{L}_{motion} = \|M' - M\|_2$ to ensure precise reconstruction of motion information; and (c) reconstruction quality loss $\mathcal{L}_{rec}$ to optimize the quality of final reconstructed frames end-to-end.
    - **Design Motivation**: Solely minimizing steganographic perturbation impairs the encoding capacity of motion information, whereas solely maximizing motion recovery leads to visible artifacts. Joint optimization of these three loss terms achieves the optimal trade-off between quality and capacity.

### Loss & Training
A multi-stage training strategy is adopted: the forward and backward processes of IMSM are first trained independently to ensure accurate round-trip information transfer, followed by end-to-end joint training of the entire space-time resampling pipeline. High-frame-rate video datasets (e.g., Vimeo-90K, REDS) are used for training, with loss weights adaptively adjusted using the validation set.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | RIFE | FLAVR | Gain |
|--------|------|----------|------|-------|------|
| Vimeo-90K | PSNR (dB) | 36.82 | 35.61 | 35.94 | +0.88 |
| Vimeo-90K | SSIM | 0.978 | 0.970 | 0.972 | +0.006 |
| REDS | PSNR (dB) | 32.15 | 30.87 | 31.24 | +0.91 |
| UCF-101 | PSNR (dB) | 35.24 | 34.56 | 34.71 | +0.53 |

### Ablation Study

| Configuration | PSNR (dB) | SSIM | Description |
|------|-----------|------|------|
| Full model (IMSM + continuous resampling) | 36.82 | 0.978 | Full model |
| w/o IMSM (pure frame interpolation) | 35.61 | 0.970 | No motion steganography, degrades to standard VFI |
| w/o continuous resampling (fixed 2x) | 36.34 | 0.975 | Supports only fixed integer factors |
| w/o quality constraint $\mathcal{L}_{vis}$ | 36.58 | 0.974 | Slight degradation in steganographic frame quality |

### Key Findings
- IMSM provides the greatest contribution (+1.2 dB), validating the critical role of motion information preservation in reconstruction quality.
- In high-motion scenarios (REDS dataset), the gain of IMSM is more significant (+1.28 dB) due to the higher uncertainty of motion estimation in these scenarios.
- The PSNR difference between steganographic frames and original frames is only about 0.3-0.5 dB, which is almost imperceptible to the human eye.
- Compared to the "2x first, then downsample" approach, continuous resampling reduces reconstruction loss by approximately 0.4 dB at non-integer factors (e.g., 1.5x).

## Highlights & Insights
- **Innovative Cross-over of Steganography × Video Processing**: Introducing steganography techniques from the information security field into video resampling to turn "lossy downsampling" into an "information transmission channel" is a highly ingenious idea. This paradigm can be generalized to any scenarios where metadata needs to be preserved during downsampling (e.g., implicit transmission of depth maps and semantic labels).
- **Lossless Round-Trip Guaranteed by Invertible Networks**: The bijective property of INN theoretically eliminates information loss, offering stronger guarantees than codec-based methods.
- **Practical Utility of Continuous Resampling**: Frame-rate and resolution conversion requirements in real-world scenarios are diverse. The design supporting continuous factors significantly extends the applicability of this method.

## Limitations & Future Work
- The computational overhead of invertible networks is high, and real-time performance may be limited by the forward/backward computational complexity of INN.
- Under high-ratio compression transmission (e.g., heavily lossy video codecs), the robustness of steganographic information against interference has not been fully verified.
- The current method mainly processes RGB videos. Extending it to other modalities, such as depth videos and event camera data, is worthy of exploration.
- The steganographic capacity is limited. In cases of extremely complex motion fields (e.g., high-speed multi-object motion), the information bottleneck may limit recovery accuracy.

## Related Work & Insights
- **vs RIFE/FLAVR (Frame Interpolation)**: Frame interpolation methods like RIFE and FLAVR can only infer intermediate frames from adjacent frames, making them prone to failure in occlusion and fast-motion scenarios. Ours bypasses the bottleneck of motion estimation by explicitly transmitting motion information.
- **vs EDSC/TMNet (Temporally-Enhanced VFI)**: These methods enhance temporal consistency through warping alignment, but essentially still "guess" motion. IMSM provides the "answer" rather than a "guess".
- **vs HiNeRV (Implicit Video Representation)**: HiNeRV encodes the entire video using implicit representations for compression but does not support flexible space-time resampling. The ideas from both can be combined.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The cross-disciplinary innovation combining steganography with video resampling is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on multiple datasets with comprehensive ablation studies, though lacking real-time performance analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear method descriptions and well-justified motivations.
- **Value**: ⭐⭐⭐⭐ Provides a brand-new paradigm for video resampling with practical application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Neural Video Compression with Context Modulation](neural_video_compression_with_context_modulation.md)
- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](../../CVPR2026/signal_comm/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[ICLR 2026\] Advancing Spatiotemporal Representations in Spiking Neural Networks via Parametic Invertible Transformation](../../ICLR2026/signal_comm/advancing_spatiotemporal_representations_in_spiking_neural_networks_via_parametr.md)
- [\[ICLR 2026\] Mamba-3: Improved Sequence Modeling using State Space Principles](../../ICLR2026/signal_comm/mamba-3_improved_sequence_modeling_using_state_space_principles.md)

</div>

<!-- RELATED:END -->
