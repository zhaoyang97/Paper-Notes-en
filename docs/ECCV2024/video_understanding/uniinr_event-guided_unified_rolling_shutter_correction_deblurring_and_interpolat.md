---
title: >-
  [Paper Note] UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation
description: >-
  [ECCV 2024][Video Understanding][Event Camera] This paper proposes the UniINR framework, which leverages a unified spatio-temporal implicit neural representation (INR) to simultaneously perform rolling shutter correction, deblurring, and arbitrary frame-rate video frame interpolation from a single rolling shutter blurred frame and paired event streams in a single pass.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Event Camera"
  - "Rolling Shutter Correction"
  - "Deblurring"
  - "Video Frame Interpolation"
  - "Implicit Neural Representation"
date: 2026-05-08
content_hash: 3e251181b3fa922a
---

# UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation

**Conference**: ECCV 2024  
**arXiv**: [2305.15078](https://arxiv.org/abs/2305.15078)  
**Code**: [Yes](https://github.com/yunfanLu/UniINR)  
**Area**: Video Understanding  
**Keywords**: Event Camera, Rolling Shutter Correction, Deblurring, Video Frame Interpolation, Implicit Neural Representation

## TL;DR

This paper proposes the UniINR framework, which leverages a unified spatio-temporal implicit neural representation (INR) to simultaneously perform rolling shutter correction, deblurring, and arbitrary frame-rate video frame interpolation from a single rolling shutter blurred frame and paired event streams in a single pass.

## Background & Motivation

Consumer-grade cameras based on CMOS sensors commonly adopt the rolling shutter (RS) mechanism. In fast-moving scenes, captured frames often suffer from both RS distortion and motion blur simultaneously. Restoring high-frame-rate, sharp global shutter (GS) frames requires considering three tasks simultaneously:

**RS Correction**: Eliminates spatial distortion caused by row-by-row exposure.

**Deblurring**: Removes motion blur within the exposure time.

**Frame Interpolation**: Generates intermediate frames along the temporal dimension.

Traditional approaches decouple the three tasks into independent sub-problems and cascade existing networks, which accumulates errors and leads to significant artifacts. For example, cascading a frame interpolation network with an RS correction network yields degraded results. Armed with high temporal resolution, event cameras offer a potential solution to this problem. However, existing event-guided methods (e.g., EvUnRoll, TimeLens) can only handle a subset of these sub-tasks and fail to provide a unified solution.

## Method

### Overall Architecture

UniINR models the problem of "restoring arbitrary frame-rate sharp GS frames from an RS blurred frame and paired events" as a function $F(\mathbf{x}, t, \theta)$, where $\mathbf{x}=(x,y)$ represents the pixel coordinates, $t$ is the timestamp within the exposure time, and $\theta$ denotes the function parameters. The framework consists of three core components:

| Component | Function | Input | Output |
|------|------|------|------|
| STE (Spatio-Temporal Encoding) | Extracts spatio-temporal representations from the RS blurred frame and events | RS blurred frame + Event stream | Spatio-temporal representation $\theta \in \mathbb{R}^{H \times W \times C}$ |
| ETE (Exposure Time Embedding) | Encodes the exposure time information of the target frame into a temporal tensor | GS/RS timestamps | Temporal tensor $T \in \mathbb{R}^{H \times W \times C}$ |
| PPD (Pixel-by-Pixel Decoding) | Queries and decodes sharp frames from the STR | $\theta + T$ | Sharp GS/RS frame |

### Key Designs

**Spatio-Temporal Encoding (STE)**: Inspired by eSL-Net, a sparse learning-based backbone is utilized to extract the Spatio-Temporal Representation (STR) from the RS blurred frame and event stream. The STR stores motion information in the form of an $H \times W \times C$ 3D tensor, which directly maps time and coordinates to RGB values. This approach circumvents the high computational overhead associated with traditional optical flow estimation.

**Exposure Time Embedding (ETE)**: Since all pixels of a GS frame are exposed simultaneously while an RS frame is exposed row-by-row, timestamp maps need to be constructed separately for the two modes:
- GS Timestamp Map: $M_g[h][w] = t_g$ (identical for all pixels)
- RS Timestamp Map: $M_r[h][w] = t_s + (t_e - t_s) \times h / H$ (linearly varying with the row index)

A single-layer MLP is employed to project the $H \times W \times 1$ timestamp map up to $H \times W \times C$, aligning with the STR dimensions.

**Pixel-by-Pixel Decoding (PPD)**: Utilizing a 5-layer MLP decoder, the temporal tensor $T$ and STR $\theta$ are fused via element-wise addition, and decoded pixel-by-pixel to output the sharp frame: $I = f_{mlp}^{\circlearrowright^5}(T \oplus \theta)$. The core advantage is that the encoder only needs to be called once, whereas the decoder can be efficiently invoked $N$ times to generate $N$ frames.

### Loss & Training

The total loss consists of two parts:

$$\mathcal{L} = \lambda_b \mathcal{L}_b + \lambda_{re} \mathcal{L}_{re}$$

- **Blurred Frame-Guided Integration Loss** $\mathcal{L}_b$: Reconstructs the RS blurred frame by averaging a sequence of predicted sharp RS frames, and computes the Charbonnier loss with the input RS blurred frame.
- **Reconstruction Loss** $\mathcal{L}_{re}$: Directly supervises the Charbonnier loss between the predicted sharp GS frames and the ground truth (GT).

An Adam optimizer is used with a learning rate of $1 \times 10^{-4}$. The model is trained for 400 epochs on two NVIDIA RTX A5000 GPUs with a batch size of 2, utilizing mixed-precision training.

## Key Experimental Results

### Main Results

**RS Correction Performance Comparison (Fastec-Orig Dataset)**:

| Method | Input Frames | Event | Params (M) | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|---------|------|----------|-------|-------|--------|
| DSUN | 2 | ✗ | 3.91 | 26.52 | 0.79 | 0.122 |
| CVR | 2 | ✗ | 42.69 | 28.72 | 0.85 | 0.111 |
| EvUnroll | 1 | ✓ | 20.83 | 31.32 | 0.88 | 0.084 |
| EvShutter | 1 | ✓ | - | 32.41 | 0.91 | 0.061 |
| **UniINR** | **1** | **✓** | **0.38** | **33.91** | **0.92** | **0.049** |

### Ablation Study

**RS Correction + Deblurring Performance (Gev-Orig Dataset)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| DSUN | 23.10 | 0.70 | 0.166 |
| JCD | 24.90 | 0.82 | 0.105 |
| EvUnRoll | 30.14 | 0.91 | 0.061 |
| NIRE | 29.86 | 0.91 | - |
| **UniINR** | **31.47** | **0.93** | **0.038** |

### Key Findings

1. **Extremely Low Parameter Count**: UniINR has only 0.38M parameters, which is 55x smaller than EvUnroll (20.83M) and 112x smaller than CVR (42.69M).
2. **Highly Efficient Inference**: For 31x frame interpolation, processing takes only 2.8ms per frame, whereas the cascaded method EvUnRoll+TimeLens requires over 177ms.
3. **Consistently Outperforms Prior Methods**: Significantly exceeds existing methods across three scenarios: RS correction, RS + deblurring, and RS + deblurring + frame interpolation.
4. **Non-linear Relationship Between Efficiency and Interpolation Factors**: For 1x to 31x interpolation, the execution time only increases from 31ms to 86ms because the encoder is invoked only once.

## Highlights & Insights

- **Unified Modeling Concept**: Unifying three tightly coupled tasks into a continuous function via INR avoids cumulative errors from cascaded approaches, offering inspiration for other multi-degradation restoration problems.
- **Lightweight and Efficient**: The 0.38M parameter size and 2.8ms/frame inference speed demonstrate strong practical deployment potential.
- **Complementary Nature of Event Cameras**: Event streams provide high-temporal-resolution motion cues, perfectly compensating for the missing information in RS blurred frames.
- **Flexible Output Modes**: By modifying the timestamp map in ETE, the framework can output either RS or GS frames, rendering the design highly robust and flexible.

## Limitations & Future Work

1. Reliance on event camera data restricts applicability in conventional camera scenarios.
2. The current method is trained on simulated data; real-world data lacks ground truth (GT), hindering quantitative evaluation.
3. Specific architecture details of the sparse learning backbone require further investigation.
4. Robustness under extreme motion or complex occlusion scenarios requires further validation.
5. Scaling the framework to higher-resolution scenarios (e.g., 4K) remains to be explored.

## Related Work & Insights

- Unlike EvUnRoll (a two-stage system doing deblurring then RS correction) and TimeLens (event-guided frame interpolation), UniINR realizes true all-in-one processing.
- Applications of INR in low-level vision (e.g., LIIF for super-resolution, VideoINR for video processing) inspired this work.
- This may inspire similar unified implicit representation designs for other multi-degradation restoration tasks.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4.5 | First to unify RS correction, deblurring, and frame interpolation into a single INR formulation |
| Technical Depth | 4 | Rigorous mathematical modeling and clever design of spatio-temporal decomposition |
| Experimental Thoroughness | 4 | Comprehensive quantitative and qualitative comparisons across multiple datasets and tasks |
| Practicality | 4 | Highly practical due to extremely lightweight parameters and fast inference speed |
| Overall | 4 | Concisely elegant method with impressive performance, representing outstanding work in event-based video restoration |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/video_understanding/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)
- [\[ECCV 2024\] IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map](iam-vfi_interpolate_any_motion_for_video_frame_interpolation_with_motion_complex.md)
- [\[AAAI 2026\] VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation](../../AAAI2026/video_understanding/vtinker_guided_flow_upsampling_and_texture_mapping_for_high-resolution_video_fra.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](../../ICCV2025/video_understanding/emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[CVPR 2025\] BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions](../../CVPR2025/video_understanding/bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)

</div>

<!-- RELATED:END -->
