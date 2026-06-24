---
title: >-
  [Paper Note] Gyro-based Neural Single Image Deblurring
description: >-
  [CVPR 2025][Image Restoration][gyro sensor] This paper proposes GyroDeblurNet, which represents complex hand shake through a novel camera motion field embedding. It features a gyro refinement module that utilizes image blur information to correct gyro errors, and a gyro deblurring module that removes blur using the corrected motion information. Combined with a curriculum learning strategy, GyroDeblurNet significantly outperforms existing methods on both synthetic and real-wor…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "gyro sensor"
  - "image deblurring"
  - "camera motion field"
  - "curriculum learning"
  - "deformable convolution"
date: 2026-05-08
content_hash: 32ab8034fc718266
---

# Gyro-based Neural Single Image Deblurring

**Conference**: CVPR 2025  
**arXiv**: [2404.00916](https://arxiv.org/abs/2404.00916)  
**Code**: To be confirmed  
**Area**: Image Restoration  
**Keywords**: gyro sensor, image deblurring, camera motion field, curriculum learning, deformable convolution

## TL;DR

This paper proposes GyroDeblurNet, which represents complex hand shake through a novel camera motion field embedding. It features a gyro refinement module that utilizes image blur information to correct gyro errors, and a gyro deblurring module that removes blur using the corrected motion information. Combined with a curriculum learning strategy, GyroDeblurNet significantly outperforms existing methods on both synthetic and real-world datasets.

## Background & Motivation

**Background**: Single-image deblurring remains highly challenging due to its severe ill-posed nature. Recent DNN-based methods still fail when dealing with large blur. Built-in gyroscopes in mobile phones provide valuable camera motion information, which can help alleviate this ill-posedness.

**Limitations of Prior Work**:
- **Inaccurate Gyroscope Data**: Real gyroscope signals contain noise, rotation center offsets, and lack translational motion information, causing the motion encoded by the gyroscope to be inconsistent with the actual image blur (gyro error).
- **Overly Simplified Motion Representation**: Existing methods (DeepGyro, EggNet, INformer) represent camera motion using only 1-2 vectors per pixel or a small number of homographies, failing to capture temporally complex hand-shake patterns.
- **Unrealistic Training Data**: Gyroscope data in existing datasets is derived from random sampling or Visual-Inertial datasets (continuous motion), which does not reflect the characteristics of hand shake during photo capture.

**Key Challenge**: How to effectively utilize the motion information provided by gyroscope data to assist deblurring even when the data contains large errors?

**Key Insight**: Instead of requiring accurate gyroscope data, this work designs a network architecture to actively handle errors—first refining the gyroscope data using image blur information, and then guiding the deblurring process with the refined data.

## Method

### Overall Architecture

GyroDeblurNet consists of two modules:
1. **Image Deblurring Module**: A U-Net architecture (using NAFBlock as the basic unit) with a Gyro Deblurring Block at the bottleneck.
2. **Gyroscope Module**: Convolutional layers that embed the motion field $\rightarrow$ multiple Gyro Refinement Blocks (refined using image features) $\rightarrow$ strided convolutions for downsampling.

Input: Blurred image $B$ + camera motion field $\mathcal{V}$, Output: Residual $R$, $D = B + R$.

### Key Designs

**1. Camera Motion Field Embedding**
- **Function**: Converts an arbitrary-length gyroscope data sequence $G = \{g_0, ..., g_{T-1}\}$ into a fixed-size tensor $\mathcal{V} \in \mathbb{R}^{W/s \times H/s \times 2M}$.
- **Mechanism**:
    - Cubic spline interpolation resamples the $T$ gyroscope samples into $M+1$ samples.
    - Integration yields $M+1$ camera orientations $\rightarrow$ computing $M+1$ homographies $H_m = KR(\theta_m)K^{-1}$.
    - For each pixel, displacement vectors between consecutive timestamps are computed and stacked to obtain $M$ 2D vectors ($2M$ channels).
    - Spatial smoothness of camera shake is leveraged to downsample by $s=2$.
- **Design Motivation**: $M=8$ vectors (significantly more than the 1-2 in existing works) can capture temporally complex hand shake while maintaining a fixed channel size compatible with CNNs; spatial downsampling reduces the memory footprint.

**2. Gyro Refinement Block**
- **Function**: Performs global correction of errors in gyro features using features from the image encoder.
- **Mechanism**:
    - Input gyro features encode multiple motion candidates with perturbations.
    - Concatenating image features and gyro features $\rightarrow$ global average pooling + 1x1 convolution $\rightarrow$ channel weights.
    - Channel weights are used to select motion candidate channels consistent with the image blur.
- **Design Motivation**: Gyroscope errors are global (rotation center offset, noise) and require global information for correction; the blur patterns in the image offer complementary cues.

**3. Gyro Deblurring Block**
- **Function**: Performs spatially adaptive deblurring using the refined gyro features.
- **Mechanism**: Two sub-blocks:
    - **Sub-block 1**: Concatenate image features and gyro features $\rightarrow$ predict deformable convolution offsets $\rightarrow$ perform spatially adaptive convolution on the image features.
    - **Sub-block 2**: Further refine deblurring results using spatial attention (convolution + Sigmoid).
- **Design Motivation**: The offsets of the deformable convolution are driven by both image and gyro information, allowing the image information to compensate even if there are residual errors in the gyroscope data.

### Loss & Training

**Curriculum Learning Strategy**:
- Initially train with error-free gyroscope data, and gradually introduce errors.
- Hybrid motion field: $\mathcal{V}_\alpha = (1-\alpha)\mathcal{V}_{clean} + \alpha \mathcal{V}_{noisy}$
- $\alpha$ increases gradually from 0 to 1.
- Noisy motion field: Randomly perturbing the rotation center + adding Gaussian noise measured from a real stationary phone.

Loss function: PSNR Loss (i.e., Charbonnier Loss).

Training configuration: 256x256 patch, batch size 16, Adam, 300 epochs, cosine annealing lr.

## Key Experimental Results

### Main Results (GyroBlur-Synth + GyroBlur-Real-S)

| Method | Category | PSNR↑ | SSIM↑ | NIQE↓ | TOPIQ↑ |
|---|---|---|---|---|---|
| NAFNet | Single Image | 25.06 | 0.709 | 5.27 | 0.409 |
| FFTformer | Single Image | 26.01 | 0.748 | 4.98 | 0.434 |
| Stripformer | Single Image | 25.93 | 0.740 | 4.71 | 0.456 |
| DeepGyro | Gyro | 23.78 | 0.665 | 5.64 | 0.381 |
| EggNet | Gyro | 25.49 | 0.727 | 5.18 | 0.413 |
| INformer | Gyro | 25.11 | 0.710 | 5.29 | 0.408 |
| Nan et al. | Non-blind | 22.22 | 0.531 | 5.81 | 0.348 |
| **Ours** | **Gyro** | **27.28** | **0.780** | **4.47** | **0.548** |

PSNR exceeds the best single-image method by 1.27 dB and the best gyro-based method by 1.79 dB.

### Ablation Study

| Configuration | PSNR | SSIM | Description |
|---|---|---|---|
| (a) No gyro data | 24.90 | 0.700 | No gyro data |
| (b) Train w/ error-free | 24.94 | 0.711 | Trained on error-free data, fails on real-world data |
| (c) No refinement | 25.47 | 0.713 | Equipped with gyro but without refinement |
| (d) Refine w/o image feat | 26.17 | 0.747 | Gyro refinement without using image features |
| (e) Deform conv only gyro | 26.32 | 0.754 | Deformable convolution using only gyro features |
| (f) Full w/o curriculum | 26.94 | 0.767 | Full model without curriculum learning |
| (g) **Full model** | **27.28** | **0.780** | Full model |

### Key Findings

1. **Training on error-free data is highly ineffective**: (b) only outperforms (a) by 0.04 dB, indicating that the model learns to discard gyroscope data when the errors are too large.
2. **Image feature-guided refinement is crucial**: (d) vs (c) achieves a 0.7 dB gain, and (f) vs (e) yields a 0.6 dB gain.
3. **Curriculum learning provides an additional 0.34 dB gain**: It helps the model progressively learn to handle errors.
4. **$M=8$ is the optimal temporal resolution**: $M=2 \rightarrow 25.71$, $M=8 \rightarrow 27.28$, $M=16 \rightarrow 27.32$ (witnessing diminishing returns).
5. **Cross-device generalization**: The model still outperforms other methods on Huawei P30 Pro (while trained on Samsung Galaxy S22).

## Highlights & Insights

- The design philosophy of "not requiring accurate sensors but teaching the network to handle errors" is both elegant and practical.
- The camera motion field embedding is a general gyro data representation scheme, with the parameter $M$ being highly flexible.
- The construction scheme of the GyroBlur-Synth dataset is scalable—requiring only a few minutes of gyro recording and simple calibration.
- The curriculum learning strategy is specifically designed for learning from noisy auxiliary signals, offering general reference value.

## Limitations & Future Work

- $M$ is fixed as a hyperparameter; long exposure times might require a larger $M$.
- Accelerometer data is not utilized, which might provide additional motion information.
- The image deblurring module architecture (NAFBlock U-Net) is relatively simple.
- Only mobile phone gyroscopes are validated, with other IMU devices remaining untested.
- Deblurring of moving objects is only handled indirectly through error robustness, without explicit modeling.

## Related Work & Insights

- DeepGyro is the first to apply gyroscope data to DNN-based deblurring, but it assumes accurate data.
- EggNet utilizes deformable convolutions to adapt gyroscope data but only uses 1-2 vectors.
- The blur synthesis pipeline of RSBlur is adopted in this work to generate realistic synthetic data.
- Insight: In sensor-assisted low-level vision tasks, handling sensor errors is more crucial than the sensor data itself.

## Rating

⭐⭐⭐⭐ — The problem definition is clear (handling gyro errors), the technical solution is highly targeted (a tri-coupling design of refinement, deblurring, and curriculum learning), the experimental setup is rigorous (synthetic + real-world + cross-device), and the PSNR margin is significant (+1.3 dB over the best single-image method).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Gyro-based Deep Video Deblurring](../../CVPR2026/image_restoration/gyro-based_deep_video_deblurring.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)
- [\[ICCV 2025\] Efficient Concertormer for Image Deblurring and Beyond](../../ICCV2025/image_restoration/efficient_concertormer_for_image_deblurring_and_beyond.md)

</div>

<!-- RELATED:END -->
