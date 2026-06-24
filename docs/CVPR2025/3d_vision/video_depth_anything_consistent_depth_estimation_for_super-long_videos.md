---
title: >-
  [Paper Note] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos
description: >-
  [CVPR 2025][3D Vision][Video Depth Estimation] Video Depth Anything builds upon Depth Anything V2 by introducing a lightweight spatio-temporal head and a temporal gradient matching loss. Without requiring geometric priors or video generation priors, it generates temporally consistent, high-quality depth maps for videos of arbitrary length at a real-time speed of 30 FPS.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Video Depth Estimation"
  - "Temporal Consistency"
  - "Super-Long Videos"
  - "Temporal Gradient Matching"
  - "Keyframe Strategy"
date: 2026-05-08
content_hash: 974d2f7397e2d05b
---

# Video Depth Anything: Consistent Depth Estimation for Super-Long Videos

**Conference**: CVPR 2025  
**arXiv**: [2501.12375](https://arxiv.org/abs/2501.12375)  
**Code**: [Project Page](https://videodepthanything.github.io)  
**Area**: 3D Vision / Depth Estimation  
**Keywords**: Video Depth Estimation, Temporal Consistency, Super-Long Videos, Temporal Gradient Matching, Keyframe Strategy

## TL;DR

Video Depth Anything builds upon Depth Anything V2 by introducing a lightweight spatio-temporal head and a temporal gradient matching loss. Without requiring geometric priors or video generation priors, it generates temporally consistent, high-quality depth maps for videos of arbitrary length at a real-time speed of 30 FPS.

## Background & Motivation

Monocular depth estimation has made significant progress, but image-level models still suffer from flickering and motion blur in videos:
- **Test-time optimization methods** are highly inefficient and impractical for real-world applications.
- **Optical flow or pose-dependent methods** (such as NVDS, MAMo) are susceptible to errors in optical flow or pose estimation.
- **Video diffusion methods** (ChronoDepth, DepthCrafter) offer rich details but suffer from slow inference speed and can only process short videos within the training window (<10 seconds).
- Existing methods suffer from depth drift and flickering between windows in long videos (over several minutes).
- Core Problem: Can temporally stable depth estimation be achieved for videos of arbitrary length without sacrificing generalization ability, rich details, and computational efficiency?

## Method

### Overall Architecture

Based on the frozen Depth Anything V2 pretrained encoder, the DPT decoder head is replaced with a Spatio-Temporal Head (STH) by inserting temporal attention layers. The model is trained using 730k supervised video frames + 620k unlabeled images in self-training. During inference, a segment-wise processing strategy with keyframe reference and overlapping interpolation is adopted to support super-long videos.

### Key Design 1: Spatio-Temporal Head (STH)

**Function**: Introduces temporal cross-frame information interaction while keeping the spatial decoding capability of the DPT head.

**Mechanism**: Temporal layers are inserted into the DPT head, each comprising multi-head self-attention (SA) along the temporal dimension and a feed-forward network (FFN). In the input features $\mathbf{F}_i \in \mathbb{R}^{(B \times N) \times (H/p \times W/p) \times C_i}$, the temporal dimension $N$ is isolated, and self-attention is performed strictly along the temporal dimension. Absolute position encoding is utilized to encode spatial-temporal relationships. To reduce computational overhead, temporal layers are inserted only at low-resolution feature locations.

**Design Motivation**: Limiting temporal attention to the decoder head instead of the encoder prevents the limited video training data from degrading the strong representations learned by the pretrained encoder. The strategy of freezing the encoder and only training the head substantially reduces training costs.

### Key Design 2: Temporal Gradient Matching Loss (TGM)

**Function**: Constrains the temporal gradient of predicted depth to match that of the ground truth (GT), without using optical flow or geometric priors.

**Mechanism**: Traditional OPW losses assume constant depth for corresponding points in adjacent frames, which does not hold in dynamic scenes. TGM relaxes this assumption—instead of keeping depth constant, it forces the variations to be consistent: $\mathcal{L}_{\text{TGM}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \| |d_{i+1} - d_i| - |g_{i+1} - g_i| \|_1$, where $d_i, g_i$ are the predicted and GT depths, respectively. To simplify further, instead of using optical flow to find correspondences, the depths at the exact same coordinate positions across adjacent frames are directly used. It is computed only in regions with minor GT changes ($|g_{i+1} - g_i| < 0.05$) to avoid instabilities introduced by edges and dynamic objects.

**Design Motivation**: Eliminates dependency on optical flow (which incurs extra computations and errors), while offering a more reasonable constraint than simple temporal consistency—allowing natural depth changes but ensuring the change patterns match the GT.

### Key Design 3: Keyframe Inference Strategy for Super-Long Videos

**Function**: Supports arbitrary-length video inference without accumulating depth drift.

**Mechanism**: Each inference window consists of three parts: $N - T_o - T_k$ new frames + $T_o$ overlapping frames + $T_k$ keyframes. Keyframes are sampled from preceding frames at an interval of $\Delta_k$ frames. Setting $N=32, T_o=8, T_k=2, \Delta_k=12$ ensures that the first frame is always at the beginning of each window. Linear interpolation is used for joint depth prediction output: $\mathbf{D}_{o_i} = \mathbf{D}_{o_i}^{\text{pre}} \cdot w_i + \mathbf{D}_{o_i}^{\text{cur}} \cdot (1-w_i)$, where $w_i$ decays linearly from 1 to 0.

**Design Motivation**: Relying solely on overlapping frames accumulates scale drift. Keyframes inject global scale information into the current window, significantly reducing drift in long videos. Fixing the very first frame at the beginning of the window further enhances consistency.

### Loss & Training

$\mathcal{L}_{\text{all}} = \alpha\mathcal{L}_{\text{TGM}} + \beta\mathcal{L}_{\text{ssi}}$, where $\mathcal{L}_{\text{ssi}}$ is the scale-and-shift invariant loss from MiDaS. Unlabeled images are utilized for self-training using pseudo-labels generated by a teacher model.

## Key Experimental Results

### Main Results: Zero-Shot Video Depth Estimation

| Method | KITTI δ₁↑ | ScanNet δ₁↑ | Bonn δ₁↑ | NYUv2 δ₁↑ | Sintel δ₁↑ | TAE↓ |
|------|----------|-----------|---------|----------|----------|------|
| **VDA-L** | **0.944** | **0.926** | **0.959** | **0.971** | **0.644** | **0.570** |
| DepthCrafter | 0.753 | 0.730 | 0.803 | 0.822 | 0.695 | 0.639 |
| DAv2-L (Image) | 0.815 | 0.768 | 0.864 | 0.928 | 0.541 | 1.140 |
| ChronoDepth | 0.576 | 0.665 | 0.665 | 0.771 | 0.673 | 1.022 |

### Ablation Study: Loss Function Design

| Loss Function | Spatial Accuracy | Temporal Consistency |
|---------|---------|----------|
| TGM (Proposed) | **Best** | **Best** |
| OPW (Flow-based) | Poor | Medium |
| SE (Stability Error) | Medium | Good |
| No Temporal Loss | Best Spatial | Worst Temporal |

### Key Findings

- VDA-L achieves state-of-the-art (SOTA) spatial accuracy on 4 out of 5 datasets, and the best temporal consistency across all datasets.
- Compared to DepthCrafter, it significantly improves spatial accuracy while working dozens of times faster in terms of inference speed.
- The smallest model, VDA-S, achieves **30 FPS real-time performance**.
- Successfully processes super-long figure skating videos of 196 seconds (4690 frames) without obvious depth drift.
- Only negligible performance drop on a few image depth estimation datasets.

## Highlights & Insights

- **Simple yet Effective Temporal Consistency**: The TGM loss does not rely on optical flow or camera poses, achieving outstanding consistency purely by constraining temporal gradients.
- **Engineering-Research Integration**: The keyframe + overlapping frame inference strategy is simple yet highly practical, effectively addressing the drift issue in long videos.
- **Inheriting Foundation Model Capabilities**: The strategy of freezing the encoder and only training the decoder head successfully transfers the strong generalization capabilities of Depth Anything V2 to the video domain.

## Limitations & Future Work

- It outputs affine-invariant depth rather than metric depth, which limits downstream applications requiring absolute scale.
- It is still constrained by window size (32 frames); there is room for further improvement in the global consistency of extremely long videos.
- Rapid and large-scale motion scenarios may still present challenges.
- Future work can extend the framework to metric video depth estimation.

## Related Work & Insights

- Proves that high-quality, temporally consistent video depth can be obtained without using video diffusion models.
- The concept of "matching changes instead of absolute values" in the TGM loss can be extended to other temporal consistency tasks.
- The keyframe reference strategy can be adopted in any sliding window-based video processing model.

## Rating

⭐⭐⭐⭐⭐ — An work of high practical value. It losslessly extends Depth Anything V2 to videos, supports super-long videos, and achieves real-time inference, while setting new SOTA benchmarks in both spatial accuracy and temporal consistency. The design of the TGM loss is simple and elegant. Another masterpiece from ByteDance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)

</div>

<!-- RELATED:END -->
