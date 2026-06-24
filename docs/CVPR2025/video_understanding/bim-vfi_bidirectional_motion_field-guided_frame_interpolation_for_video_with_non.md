---
title: >-
  [Paper Note] "BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions"
description: >-
  [CVPR2025][Video Understanding][Paper Note] Academic paper note for "BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions".
tags:
  - CVPR2025
  - Video Understanding
date: 2024-12-16
content_hash: 8dfe784060e9730c
---
# BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions

## Background & Motivation

Video Frame Interpolation (VFI) aims to synthesize intermediate frames between two given frames, which is widely applied in slow-motion generation, video coding, and frame rate up-conversion. Existing methods are mostly based on **optical flow estimation**: they first estimate forward and backward optical flows, and then synthesize the intermediate frames through warping.

However, motion fields in real-world videos are often **non-uniform**, meaning that motion velocities and directions vary significantly across different regions within the same frame. Typical scenarios include:

**High-speed foreground motion + static background**: such as athletes in sports events

**Multi-object motion with different speeds**: such as multiple vehicles in traffic scenes

**Mixed rotation and translation**: such as complex camera motion in handheld shooting

Traditional methods typically assume that bidirectional motions are independent, estimating the forward and backward optical flows separately. This overlooks a key piece of information: **there exists an inherent geometric constraint between bidirectional optical flows**. For the same 3D scene point, its projected displacements in the two frames satisfy specific mathematical relationships.

This paper proposes the **BiM (Bidirectional Motion) descriptor**, a compact representation that simultaneously encodes bidirectional motion relationships, along with a lightweight frame interpolation framework based on BiM.

## Method

### BiM Descriptor

The BiM descriptor $[R, \Phi]$ consists of two components:

#### Magnitude Ratio $R$

$$R = rac{\|\mathbf{f}_{0 \to 1}\|}{\|\mathbf{f}_{1 \to 0}\|}$$

Where $\mathbf{f}_{0 \to 1}$ and $\mathbf{f}_{1 \to 0}$ represent the forward and backward optical flows, respectively. $R$ captures the relative velocity information of the motion.

#### Angle Difference $\Phi$

$$\Phi = \angle(\mathbf{f}_{0 \to 1}) - \angle(\mathbf{f}_{1 \to 0}) - \pi$$

$\Phi$ measures the deviation between the directions of the forward and backward optical flows. For strictly linear motion, $\Phi = 0$; for non-linear motion (such as rotation or acceleration), $\Phi \neq 0$.

| Motion Type | $R$ | $\Phi$ | Description |
|---------|-----|--------|------|
| Uniform Translation | 1.0 | 0 | Same speed and opposite directions between frames |
| Accelerated Motion | >1.0 | 0 | Faster in the second half |
| Decelerated Motion | <1.0 | 0 | Faster in the first half |
| Curved Motion | ≈1.0 | ≠0 | Presence of directional deviation |
| Complex Non-linear | ≠1.0 | ≠0 | Both velocity and direction change |

### BiM-guided FlowNet

The BiM descriptor is injected into the optical flow estimation network as additional input channels:

$$\mathbf{f}_{t} = 	ext{FlowNet}(I_0, I_1, t, R, \Phi)$$

Unlike traditional methods that directly estimate the optical flow of intermediate frames, BiM-guided FlowNet leverages the global constraint information of bidirectional motions to significantly improve the optical flow accuracy in motion-non-uniform regions.

### Content-Aware Upsampling Network (CAUN)

Traditional frame interpolation uses bilinear interpolation or separable convolutions to upsample warped features. This paper proposes CAUN, a content-aware upsampling module:

- Input: low-resolution warped features, high-resolution original frames
- Core: adaptive sampling kernel generation based on local content
- Output: high-resolution synthesized frames

CAUN utilizes a finer sampling strategy in edge and texture regions and a larger receptive field in flat regions, achieving a balance between quality and efficiency.

### Knowledge Distillation (KDVCF)

To further compress the model, this paper designs the KDVCF (Knowledge Distillation for Video Content-aware Frame interpolation) strategy:

| Component | Teacher Model | Student Model |
|------|---------|---------|
| Backbone Network | ResNet-50 | MobileNetV3 |
| Parameters | 28.3M | **6.88M** |
| Distillation Loss | - | Feature Alignment + Output Matching |
| Inference Speed | 1× | **3.2×** |

The distillation strategy includes:
1. **Feature alignment distillation**: minimizing the L2 distance of intermediate-layer features
2. **Output matching distillation**: matching the perceptual loss of the final synthesized frame

## Experimental Results

### Main Results

| Method | Parameters | Vimeo90K PSNR↑ | SSIM↑ | UCF101 PSNR↑ | SNU-FILM Hard↑ |
|------|--------|---------------|-------|-------------|---------------|
| RIFE | 9.8M | 35.61 | 0.978 | 35.28 | 29.27 |
| IFRNet | 19.7M | 35.80 | 0.979 | 35.36 | 29.51 |
| AMT-S | 12.3M | 35.72 | 0.978 | 35.31 | 29.39 |
| EMA-VFI | 21.5M | 35.86 | 0.979 | 35.40 | 29.56 |
| **BiM-VFI** | **6.88M** | **36.01** | **0.980** | **35.52** | **29.72** |

BiM-VFI achieves the best results across all datasets with the fewest parameters (6.88M).

### Non-uniform Motion Scenarios

On the X-TEST and Xiph-4K datasets, which contain a substantial amount of non-linear motion, the advantages of BiM-VFI are even more pronounced:

| Method | X-TEST PSNR↑ | Xiph-4K PSNR↑ |
|------|-------------|--------------|
| RIFE | 28.93 | 31.42 |
| EMA-VFI | 29.34 | 31.89 |
| **BiM-VFI** | **30.12** | **32.47** |

### Ablation Study

| Configuration | Vimeo90K PSNR↑ | Parameters |
|------|---------------|--------|
| Full BiM-VFI | 36.01 | 6.88M |
| w/o BiM descriptor | 35.42 | 6.85M |
| w/o CAUN (Bilinear upsampling) | 35.67 | 5.91M |
| w/o KDVCF (Teacher model) | 36.23 | 28.3M |
| Only $R$ | 35.78 | 6.86M |
| Only $\Phi$ | 35.71 | 6.86M |

Both components of the BiM descriptor contribute to the performance, with the full BiM descriptor bringing a +0.59dB improvement.

## Conclusion & Future Work

By introducing the BiM descriptor $[R, \Phi]$ to explicitly model the internal relationship of bidirectional motion fields, and combining content-aware upsampling with knowledge distillation, BiM-VFI achieves state-of-the-art frame interpolation quality with only 6.88M parameters. This method is particularly suited for handling non-uniform motion scenarios, and its design philosophy—leveraging the constraint relationship of bidirectional optical flows—can be generalized to other video understanding tasks that require motion estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map](../../ECCV2024/video_understanding/iam-vfi_interpolate_any_motion_for_video_frame_interpolation_with_motion_complex.md)
- [\[CVPR 2025\] Anomize: Better Open Vocabulary Video Anomaly Detection](anomize_better_open_vocabulary_video_anomaly_detection.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/video_understanding/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)

</div>

<!-- RELATED:END -->
