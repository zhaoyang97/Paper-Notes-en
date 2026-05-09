---
title: >-
  [Paper Note] AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception
description: >-
  [CVPR2026][Autonomous Driving][radar perception] AdaRadar is proposed — an online adaptive radar data compression framework based on DCT spectral pruning and zeroth-order surrogate gradients — achieving over 100× compression with only ~1 percentage point degradation in detection/segmentation performance, effectively alleviating the bandwidth bottleneck between radar sensors and computing units.
tags:
  - CVPR2026
  - Autonomous Driving
  - radar perception
  - adaptive compression
  - spectral pruning
  - zeroth-order gradient
  - rate control
  - quantization
  - DCT
  - object detection
  - semantic segmentation
date: 2026-05-08
content_hash: c574a7ecc0473561
---

# AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception

**Conference**: CVPR2026
**arXiv**: [2603.17979](https://arxiv.org/abs/2603.17979)
**Code**: [Project Page](https://jp4327.github.io/adaradar/)
**Area**: Autonomous Driving
**Keywords**: radar perception, adaptive compression, spectral pruning, zeroth-order gradient, rate control, quantization, DCT, object detection, semantic segmentation

## TL;DR

AdaRadar is proposed — an online adaptive radar data compression framework based on DCT spectral pruning and zeroth-order surrogate gradients — achieving over 100× compression with only ~1 percentage point degradation in detection/segmentation performance, effectively alleviating the bandwidth bottleneck between radar sensors and computing units.

## Background & Motivation

**Central role of radar in autonomous driving**: Compared to cameras and LiDAR, millimeter-wave radar offers robustness under all-weather and all-lighting conditions, directly measuring range and Doppler velocity, making it an indispensable modality for perception systems.

**Explosive raw radar data volume**: Modern MIMO radar data volume grows quadratically with the number of Tx/Rx antennas. For example, a 16×12 cascaded MIMO configuration produces approximately 100 MB per frame, with a throughput of several Gbps at 10 fps — far exceeding the capacity of low-bandwidth links such as CAN bus.

**Lack of dedicated radar codecs**: While mature JPEG and learned compression schemes exist for vision, no universal codec has been established for radar. Existing approaches typically compress data to sparse point clouds via CFAR, inevitably discarding rich tensor-level information.

**Existing methods lack adaptivity**: Energy-based index-value compression schemes such as RadarOcc operate at fixed compression ratios and cannot dynamically adjust to scene complexity at test time — wasting bandwidth in low-complexity scenes and degrading performance in high-complexity ones.

**Traditional backpropagation is infeasible here**: Pruning and quantization operations are non-differentiable; even if gradients could be estimated, their tensor size is comparable to the raw radar data, so transmitting gradients would negate the compression gain.

**Autoencoder-based approaches face deployment constraints**: Learned compression models typically have more parameters than downstream detection/segmentation models, making them impractical to deploy on the compute- and memory-constrained DSPs found at radar front-ends.

## Method

### Overall Architecture

AdaRadar connects the radar front-end (encoder) and the computing unit (decoder + inference network) via a feedback loop:

1. **Encoder**: Applies DCT to the raw range–Doppler tensor → adaptive spectral pruning → scaled quantization → transmits compressed coefficients and scale factors.
2. **Decoder**: Inverse quantization → IDCT reconstruction → feeds into downstream detection/segmentation network → outputs confidence scores or entropy.
3. **Feedback control**: Updates the pruning rate for the next frame online using zeroth-order surrogate gradients derived from detection confidence (or segmentation entropy), enabling adaptive rate control.

### Key Designs

**1) DCT Spectral Pruning**

- The radar feature map, formed by concatenating real and imaginary parts, is split into $M \times M$ blocks, each transformed with Type-II DCT.
- Observation: DCT coefficient energy in radar signals is highly concentrated in a small number of frequency components (e.g., high-frequency regions in the RADIal dataset), exhibiting strong sparsity.
- Given pruning rate $r$, the $k = \lfloor M^2 / r \rfloor$ coefficients with the largest magnitudes are retained per block, with the rest set to zero.
- Compared to JPEG's fixed quantization tables that attenuate high frequencies, this method adaptively retains coefficients ranked by magnitude, better suited to the diverse frequency structures of radar signals.

**2) Scaled Quantization**

- The peak magnitude $Q_{c,b}$ is computed per block and used as the reference for per-block uniform fixed-point quantization (e.g., 8-bit).
- The overhead for transmitting scale factors is only $s_{\text{FP}} / (s_{\text{FxP}} \cdot M^2)$ (merely 0.097% for 64×64 blocks with 8-bit quantization), preserving the high dynamic range of radar signals.

**3) Zeroth-Order Surrogate Gradient Adaptation**

- The maximum bounding-box confidence $p_{\max}$ serves as the surrogate objective (for segmentation tasks, average pixel-level entropy is used instead).
- A small negative perturbation $\epsilon$ is applied to the current compression rate, a second forward pass is performed to obtain $p^-$, and the gradient is estimated via finite differences: $\hat{\nabla}_r h \approx (p - p^-) / \epsilon$.
- Gradient descent updates the rate: $r_{t+1} = r_t - \eta \hat{\nabla}_{r_t} J$. Each step requires only **two forward passes**, with no backpropagation and no gradient tensor transmission over the bandwidth-limited link.

### Loss & Training

- Online optimization objective: $\max_{\{r_t\}} \mathbb{E}[J_t(r_t)]$, where $J_t = h(\mathbf{x}_t, r_t) - \lambda \cdot B(r_t)$, $\lambda$ controls the accuracy–bandwidth trade-off, and $B(\cdot)$ denotes instantaneous bit rate.
- Training phase: On-the-fly compression augmentation with randomly sampled pruning rates $r \sim U(r_{\min}, r_{\max})$ and quantization bit-widths, endowing downstream networks with robustness to varying compression rates.

## Key Experimental Results

### Main Results

**RADIal Dataset (Detection + Segmentation)**

| Method | Bits | Rate (bpp) | Compression | Precision | Recall | F1 | mIoU |
|--------|------|-----------|-------------|-----------|--------|-----|------|
| Uncompressed Baseline | 32 | 32 | 1× | 97.24 | 95.93 | 96.58 | 75.97 |
| Index-value [RadarOcc] | 32 | 2.67 | 12× | 97.55 | 62.12 | 75.91 | 49.86 |
| **AdaRadar (Ours)** | 4 | **0.32** | **101×** | 96.25 | 94.04 | **95.13** | **79.34** |

**CARRADA Dataset (Semantic Segmentation)**

| Method | Bits | Compression | mIoU | mDice |
|--------|------|-------------|------|-------|
| Uncompressed Baseline | 32 | 1× | 55.25 | 67.13 |
| Index-value | 32 | 29× | 38.96 | 46.90 |
| **AdaRadar (Ours)** | 8 | **117×** | **54.03** | **65.87** |

**Radatron Dataset (Object Detection)**

| Method | Bits | Compression | mAP | AP50 | AP75 |
|--------|------|-------------|-----|------|------|
| Baseline | 32 | 1× | 46.07 | 83.60 | 44.16 |
| Index-value | 32 | 7.5× | 45.72 | 80.44 | 47.54 |
| **AdaRadar (Ours)** | 8 | **30×** | **48.46** | **83.69** | **49.07** |

### Ablation Study

- **Spectral pruning vs. spatial index-value**: Spectral pruning incurs virtually no performance loss up to 5× compression, and its performance roll-off slope (23.1%/dec) is significantly better than index-value (46.6%/dec), demonstrating substantially greater robustness.
- **Rate–accuracy trade-off**: Accuracy remains nearly constant above 1 bpp, providing a principled basis for setting the lower bound $r_{\min}$ of the adaptive controller.
- **Compression as denoising**: On Radatron, AP75 improves by ~5% after compression, suggesting that spectral pruning acts as a noise and clutter filter.
- **Online adaptive control**: Adaptive control achieves an average of 0.279 bpp (115× compression) with AR of 93.91%; the bit rate fluctuates online with scene complexity, automatically reducing compression ratio in complex scenes to preserve performance.

## Highlights & Insights

- **Highly practical**: The encoder requires only DCT + sorting + rounding, and can run directly on existing embedded DSPs without any neural network deployment.
- **100×+ compression ratio**: Achieves 101× compression on RADIal with only ~1.5 pp F1 degradation; reaches 117× on CARRADA with only ~1 pp mIoU drop.
- **Zeroth-order gradients elegantly circumvent two key challenges**: non-differentiable operations and gradient transmission overhead, completing online rate adaptation with only two forward passes.
- **Strong generality**: Applicable to different FMCW radars, different tasks (detection/segmentation), and different datasets (RADIal/CARRADA/Radatron), and can be embedded into existing DNN pipelines without modification.

## Limitations & Future Work

- Joint compression with camera imagery is not considered; co-compression of radar–camera data in practical multimodal systems remains an open direction.
- Adaptive control relies on the assumption of temporal continuity between frames; rate tracking may lag when frame sequences are interrupted or scenes change abruptly.
- Additional lossless entropy coding (e.g., Huffman/RLE) could further improve compression ratio but was not adopted in this work to minimize latency.
- Evaluation is primarily conducted on 2D range-Doppler/range-azimuth slices; the compression effectiveness on full 4D radar tensors remains to be validated.

## Related Work & Insights

- **Radar perception**: FFTRadNet, T-FFTRadNet, and similar methods feed raw radar tensors directly into DNNs, outperforming point-cloud-based approaches but facing data volume challenges.
- **Radar compression**: The index-value scheme in RadarOcc achieves limited compression ratios with noticeable performance degradation; autoencoder-based approaches are too large to deploy at the sensor end.
- **Image compression**: JPEG's fixed quantization tables and end-to-end learned compression methods (e.g., Ballé et al.) are designed for visual perception and are ill-suited to the multi-channel, high-dynamic-range characteristics of radar signals.
- **Adaptive rate control**: Prior work focuses largely on offline optimization of rate-distortion trade-offs under fixed I/O constraints; this paper is the first to achieve task-feedback-driven online test-time rate adaptation in a radar context.

## Rating

- Novelty: ⭐⭐⭐⭐ — The zeroth-order surrogate gradient-driven online rate adaptation is conceptually novel; the systematic application of DCT spectral pruning to radar compression is a first
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, two task types, comprehensive quantitative and qualitative analysis, and thorough ablation studies
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, algorithmic description is rigorous, and figures and tables are carefully designed
- Value: ⭐⭐⭐⭐⭐ — Addresses a critical engineering bottleneck between radar sensors and computing units; the lightweight and deployable solution is of significant practical value for onboard radar systems
- Overall: ⭐⭐⭐⭐ — An important problem, a practical solution, and solid experiments; a landmark contribution to the radar data compression direction

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving](../../AAAI2026/autonomous_driving/radarmp_motion_perception_for_4d_mmwave_radar_in_autonomous_driving.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2026\] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule](perception_characteristics_distance_measuring_stability_and_robustness_of_percep.md)

</div>

<!-- RELATED:END -->
