---
title: >-
  [Paper Note] Recovering Parametric Scenes from Very Few Time-of-Flight Pixels
description: >-
  [ICCV 2025][Time-of-Flight sensor] This paper investigates the feasibility of recovering 3D parametric scene geometry using an extremely small number (as few as 15 pixels) of low-cost wide-field-of-view ToF sensors. An a…
tags:
  - "ICCV 2025"
  - "Time-of-Flight sensor"
  - "6D pose estimation"
  - "differentiable rendering"
  - "SPAD"
  - "parametric scene recovery"
date: 2026-05-08
content_hash: 79081d489d4133b3
---

# Recovering Parametric Scenes from Very Few Time-of-Flight Pixels

**Conference**: ICCV 2025
**arXiv**: [2509.16132](https://arxiv.org/abs/2509.16132)  
**Code**: [Project Page](https://cpsiff.github.io/recovering_parametric_scenes)  
**Area**: Other
**Keywords**: Time-of-Flight sensor, 6D pose estimation, differentiable rendering, SPAD, parametric scene recovery

## TL;DR

This paper investigates the feasibility of recovering 3D parametric scene geometry using an extremely small number (as few as 15 pixels) of low-cost wide-field-of-view ToF sensors. An analysis-by-synthesis framework combining feedforward prediction and differentiable rendering is proposed, demonstrating surprisingly strong performance on tasks such as 6D object pose estimation.

## Background & Motivation

Time-of-Flight (ToF) cameras are a key technology in modern 3D vision. Mainstream methods rely on high-resolution dense 3D data, reinforcing the assumption that dense depth measurements are a prerequisite for accurate 3D vision. Recently, however, a class of **ultra-low-cost** ToF sensors (<$3 each, <5mm in size) has emerged and been deployed in smartphones and wearable devices. These sensors are characterized by:

- **Extremely low spatial resolution**: as few as a single pixel
- **Wide field of view**: each pixel covers approximately 30° of viewing angle
- **Rich temporal information**: fine-grained time-of-flight data captured via transient histograms

Conventional approaches reduce these histograms to a single depth value via peak detection, discarding large amounts of information. The central hypothesis of this paper is that **even a small number of transient histograms encodes sufficient scene information to recover 3D structure under strong geometric priors**.

The paper asks: given a parametric shape model as prior, what is the **minimum number of depth measurements** required to recover a 3D scene?

## Method

### Overall Architecture

The method consists of two core components:
1. **Feedforward prediction network**: directly predicts scene parameters $\mathbf{P}_{\text{FF}}$ from sparse transient histograms $\{\mathbf{h}_i\}_{i=1}^n$
2. **Analysis-by-synthesis refiner**: iteratively optimizes scene parameters starting from $\mathbf{P}_{\text{FF}}$ using a differentiable renderer $\mathcal{R}$

### Key Designs

1. **SPAD transient imaging model**: Each sensor emits $N_{\text{emit}}$ photons; photons travel along direction $\boldsymbol{\omega}$ to a scene point $\mathbf{x}$ and reflect back to the sensor. The expected photon count in the $i$-th time bin is:

$$N[i] = N_{\text{emit}} \int_{\Omega} I(\boldsymbol{\omega}) \frac{\rho(\mathbf{x})}{\pi} \frac{\langle -\boldsymbol{\omega}, \hat{\mathbf{n}}(\mathbf{x}) \rangle}{\|\mathbf{x}\|^2} W\left(\frac{2\|\mathbf{x}\|}{c}, t_i\right) d\boldsymbol{\omega}$$

where $\rho(\mathbf{x})$ is albedo, $\hat{\mathbf{n}}(\mathbf{x})$ is the surface normal, and $W$ is the temporal binning function. Crucially, this model accounts for photon contributions from **all directions** within the wide field of view rather than a single peak depth. The final histogram is obtained by convolving with an empirical jitter kernel $\mathbf{s}$.

2. **Differentiable renderer**: The integral is discretized as a weighted sum over an $h \times w$ grid using Nvdiffrast for differentiable rasterization. The discontinuous binning function $W$ is approximated using a sigmoid:

$$W(t, t_i) = \sigma(k(t-t_i)) - \sigma(k(t-t_i-\Delta t))$$

The laser intensity distribution $I(\boldsymbol{\omega})$ is fit with a differentiable Gaussian kernel: $I(\boldsymbol{\omega}) = K_1 \exp(-K_2(\omega_x^2+\omega_y^2) - K_3(\omega_x^4+\omega_y^4))$. The entire rendering pipeline is fully differentiable with respect to scene parameters.

3. **Feedforward Transformer network**: Takes $n$ normalized histograms as input, embeds them via MLP with positional encoding, processes them through 4 Transformer blocks, and predicts scene parameters via an MLP over the concatenated output embeddings. A key challenge is the scarcity of real data; thus, **large-scale synthetic training data is generated using the renderer**, with sim-to-real transfer achieved through domain randomization (sensor position noise ±1.5cm, albedo randomization).

### Loss & Training

- **Feedforward network**: rotation loss + translation loss + point matching loss for asymmetric objects; ADD-S loss for symmetric objects
- **Refinement stage**: Adam optimizer minimizing $\sum_{i=1}^n \|\mathcal{R}(\mathbf{P})_i - \mathbf{h}_i\|$, with learning rates of 0.01 for rotation and 0.001 for translation, over 200 iterations
- Object and plane albedo parameters are jointly optimized
- Rotations are represented using the 6D continuous parameterization to avoid gimbal lock

## Key Experimental Results

### Main Results

**YCB Object 6D Pose Estimation (symmetric objects, AUC-ADD-S ↑)**:

| Method | Pixels | Crackers | Mustard | SPAM | Basketball | Tennis | Mean |
|--------|--------|----------|---------|------|-----------|--------|------|
| 1Px point cloud (sim) | 15 | 78.36 | 82.12 | 85.07 | 82.92 | 88.09 | 82.96 |
| **Ours: FF+Refiner (real)** | **15** | **90.04** | **90.07** | **90.00** | **95.76** | **96.06** | **92.20** |
| 16² point cloud (sim) | 3840 | 95.17 | 97.23 | 97.19 | 97.67 | 97.57 | 97.06 |
| Single-view RGB (real) | 407K | 60.71 | 87.93 | 58.95 | 65.46 | 77.68 | 66.18 |
| Single-view RGB-D (real) | 407K | 90.49 | 92.10 | 93.80 | 94.24 | 86.67 | 92.01 |

With only 15 ToF pixels, the proposed method approaches the performance of RGB-D methods using 400K pixels.

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| FF only vs. FF+Refiner (3D-printed objects) | AUC-ADD 70.96→80.71 | ~10-point gain from refiner |
| FF only vs. FF+Refiner (YCB) | AUC-ADD-S 91.83→92.20 | Smaller gain for symmetric objects |
| Sphere position/size recovery | Position error <1cm, diameter error <0.35cm | Effective despite 1.4cm temporal resolution |
| Hand pose (sim-only training) | PA-MPJPE 19.56mm | Large sim-to-real gap at close range |
| Hand pose (sim pretrain + real finetune) | PA-MPJPE 8.18mm | Transfer learning is effective |

### Key Findings

- Transient histograms contain substantially more information than single depth values; the proposed method consistently outperforms point cloud baselines across the 5–100 pixel range
- Point cloud baselines degrade severely under extreme sparsity due to insufficient coverage, while the wide-field-of-view nature of transient data mitigates this problem
- Sim-to-real transfer works well for 6D pose estimation but degrades for hand tracking at close range due to strong illumination effects
- The method surpasses single-view RGB by 26 percentage points (66.18→92.20) and nearly matches RGB-D (92.01 vs. 92.20)

## Highlights & Insights

- **Demonstration of extreme sparse sensing feasibility**: 15 sensors costing under $3 each suffice for 6D pose estimation, challenging the dependence on dense sensing
- **Maximizing information utilization**: rather than reducing histograms to peak depth values, the full temporal content of transient histograms is exploited
- **End-to-end differentiable pipeline**: gradients flow from imaging physics through to pose optimization
- **Hardware prototype validation**: results are verified not only in simulation but also using a real robotic arm with TMF8820 sensors across multiple objects

## Limitations & Future Work

- Lambertian surface and co-located sensor/illumination assumptions limit applicability to scenes with complex reflectance
- Sensor range is constrained (TMF8820 up to 1.5m), limiting evaluation to tabletop settings
- Fixed sensor configurations require retraining the network, reducing flexibility
- Large sim-to-real gap for hand tracking at close range (<15cm) due to unmodeled gating and pile-up effects
- Current work addresses single-object scenes; multi-object scenarios with occlusion remain unexplored

## Related Work & Insights

- Similar in spirit to Pixels2Pose (4×4 SPAD for human pose estimation) but more extreme (1 pixel/view), with the addition of a differentiable rendering refiner
- The analysis-by-synthesis paradigm with differentiable rendering is particularly effective under low-data regimes
- Directly applicable to wearable computing and low-power robotic perception
- Raises a broader question: for which vision tasks can high-resolution cameras be replaced by a small number of physical sensors?

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The problem setting of extreme sparse ToF sensing is highly original
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers simulation and real hardware across multiple scenarios, though limited to tabletop scale
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and the physical modeling is detailed
- **Value**: ⭐⭐⭐⭐ Opens a new direction for ultra-low-cost 3D sensing with significant practical potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes](adaptiveae_an_adaptive_exposure_strategy_for_hdr_capturing_i.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[NeurIPS 2025\] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems](../../NeurIPS2025/others/efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](../../NeurIPS2025/others/deep_learning_for_continuous-time_stochastic_control_with_jumps.md)

</div>

<!-- RELATED:END -->
