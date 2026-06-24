---
title: >-
  [Paper Note] Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions
description: >-
  [ECCV 2024][3D Vision][Event camera] This work proposes Deblur e-NeRF, which models the motion blur of event cameras using a physically accurate pixel bandwidth model, enabling the first direct and effective reconstruction of blur-free NeRFs from motion-blurred event streams.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Event camera"
  - "NeRF"
  - "motion blur"
  - "pixel bandwidth model"
  - "low light"
date: 2026-05-08
content_hash: e2b2fd3812598092
---

# Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions

**Conference**: ECCV 2024  
**arXiv**: [2409.17988](https://arxiv.org/abs/2409.17988)  
**Code**: [https://github.com/wengflow/deblur-e-nerf](https://github.com/wengflow/deblur-e-nerf)  
**Area**: 3D Vision  
**Keywords**: Event camera, NeRF, motion blur, pixel bandwidth model, low light

## TL;DR

This work proposes Deblur e-NeRF, which models the motion blur of event cameras using a physically accurate pixel bandwidth model, enabling the first direct and effective reconstruction of blur-free NeRFs from motion-blurred event streams.

## Background & Motivation

**Background**: Event cameras are advantageous in high-speed motion and low-light scenarios due to their characteristics such as high temporal resolution, high dynamic range, and low latency. Recently, significant progress has been made in reconstructing NeRF from event streams (e.g., Robust e-NeRF).

**Limitations of Prior Work**: Contrary to common belief, event cameras also suffer from motion blur under high-speed or low-light conditions. This is because the pixel bandwidth of the event sensor is limited, and bandwidth is approximately proportional to the incident light intensity. However, existing event NeRF reconstruction methods and event simulators do not account for the fully non-linear behavior of event motion blur.

**Key Challenge**: The fundamental cause of event motion blur is the limited bandwidth of the analog pixel circuitry—it restricts the minimum event detection latency, maximum detectable frequency of change, and maximum event generation rate. This limitation is particularly severe under high-speed or low-light conditions.

**Goal**: To directly reconstruct a blur-free NeRF from motion-blurred event streams under high-speed motion or low-light conditions.

**Key Insight**: Starting from the physical circuit analysis of event sensors, the authors construct a complete pixel bandwidth model to precisely describe the non-linear characteristics of event motion blur.

**Core Idea**: Outlining the event motion blur within an Analysis-by-Synthesis framework using a physically accurate pixel bandwidth model, thereby reconstructing clean NeRFs "through" the blur.

## Method

### Overall Architecture

Deblur e-NeRF is based on the Analysis-by-Synthesis framework of Robust e-NeRF, with the core extension of introducing a physically accurate pixel bandwidth model to describe event motion blur. During training, radiance values are rendered from NeRF, synthesized into motion-blurred effective log-radiance via the bandwidth model, and optimistically aligned with the actual event stream. After reconstruction, an improved translated-gamma correction is applied to resolve ambiguities.

### Key Designs

1. **Pixel Bandwidth Model**:

    - **Function**: To accurately model the frequency response characteristics of the event sensor pixel analog circuitry, outputting the motion-blurred effective log-radiance $\log L_{blur}$.
    - **Mechanism**: The pixel circuitry is modeled as a unity-gain 4th-order non-linear time-invariant (NLTI) low-pass filter, represented in state-space form:
    $$\dot{\mathbf{x}}(t) = A(u(t))\mathbf{x}(t) + B(u(t))u(t), \quad \mathbf{y}(t) = C\mathbf{x}(t)$$
      The filter consists of three cascaded stages: (a) a 2nd-order NLTI low-pass filter modeling the transient response of the logarithmic photoreceptor, where the damping ratio $\zeta$ and natural angular frequency $\omega_n$ are complex non-linear functions of the input, and the bandwidth is approximately proportional to $\exp(u)=L$; (b) a 1st-order LTI low-pass filter modeling the source follower buffer response with a constant cutoff frequency $\omega_{c,sf}$; (c) a 1st-order LTI low-pass filter modeling the differential amplifier response with a cutoff frequency $\omega_{c,diff}$.
    - **Design Motivation**: Event blur originates from the physical limitations of the pixel circuitry; thus, only by precisely modeling the circuit behavior can the blur be correctly handled.

2. **Motion-Blurred Log-Radiance Synthesis**:

    - **Function**: Given the $\log L_{sig}$ rendered by NeRF, this synthesizes the blurred signal $\log L_{blur}$ "seen" by the pixel.
    - **Mechanism**: First, the continuous-time non-linear model is linearized and discretized to obtain a discrete-time 4th-order linear time-varying (LTV) filter:
    $$\mathbf{x}[k+1] = A_d[k]\mathbf{x}[k] + B_d[k]u[k] + \tilde{B}_d[k]u[k+1]$$
      Then, the numerical solution of its transient response is solved, approximating $\mathbf{y}[k]$ as a weighted sum of past and current inputs: $\mathbf{y}[k] \approx \sum_{i=k_0}^{k} \hat{\mathbf{w}}[i] u[i]$, where $\hat{\mathbf{w}}[i]$ is the normalized weight.
    - **Importance Sampling Strategy**: An importance sampling strategy is proposed to sample input timestamps from a transformed exponential distribution $T_i \sim \text{Exp}(t_k - t_i; \omega_{c,dom,min})$ to approximate the weight function distribution, concentrating the samples where the weights are large, which is most effective under low-light conditions.

3. **Threshold-Normalized Total Variation Loss**:

    - **Function**: To replace the gradient loss of Robust e-NeRF, better regularizing large textureless regions.
    - **Mechanism**: Penalizing the variation of the predicted motion-blurred log-radiance within the event interval:
    $$\ell_{tv}(\mathbf{e}) = \left|\frac{\delta \log \hat{L}_{blur}}{\bar{C}}\right|$$
      where $\bar{C} = \frac{1}{2}(C_{-1} + C_{+1})$ is the average contrast threshold, and $\delta \log \hat{L}_{blur}$ is the change between two sampled time points.
    - **Design Motivation**: Compared with gradient loss, total variation loss imposes a stronger uniformity constraint on homogeneous regions and generalizes to arbitrary thresholds through threshold normalization.

4. **Translated-Gamma Correction**:

    - **Function**: To eliminate the unknown black level offset and gamma inaccuracy in the predicted effective radiance.
    - **Mechanism**: Translating-gamma correction is performed on the reconstructed $\hat{\mathbf{L}}$: $\hat{\mathbf{L}}_{sig,corr} = \mathbf{b} \odot \hat{\mathbf{L}}^a - \mathbf{c}$, optimizing parameters $a, \mathbf{b}, \mathbf{c}$ via Levenberg-Marquardt non-linear least squares.
    - **Design Motivation**: Event cameras only observe changes in radiance rather than absolute values, and the black level (dark current equivalent radiance) is unknown, necessitating additional correction.

### Loss & Training

- **Main reconstruction loss**: Threshold-normalized difference loss $\ell_{diff}$ as a variation of the Huber norm, with weight $\lambda_{diff}$.
- **Regularization loss**: Threshold-normalized total variation loss $\ell_{tv}$, with weight $\lambda_{tv}$.
- Supports joint optimization of pixel bandwidth model parameters $\Omega$ and NeRF parameters $\Theta$.
- Training batch size is only 1/8 of the baseline.

## Key Experimental Results

### Main Results (Synthetic Data - Different Camera Speeds)

| Condition | Metric | Deblur e-NeRF | Robust e-NeRF | Gain |
|------|------|---------------|---------------|------|
| Speed 0.125× | PSNR↑ | 28.71 | 28.31 | +0.40 |
| Speed 1× (Default) | PSNR↑ | 28.41 | 26.11 | +2.30 |
| Speed 4× (High Speed) | PSNR↑ | 27.48 | 22.18 | **+5.30** |
| Blur-free Upper Bound | PSNR↑ | 29.43 | 28.48 | +0.95 |

### Experimental Results (Real Sequences - EDS Dataset)

| Sequence | Metric | Deblur e-NeRF | Robust e-NeRF | E2VID+NeRF |
|------|------|---------------|---------------|------------|
| 08_peanuts_running | PSNR↑ | **18.27** | 18.00 | 14.85 |
| 08_peanuts_running | LPIPS↓ | **0.503** | 0.507 | 0.595 |
| 11_all_characters | PSNR↑ | **16.53** | 15.91 | 13.12 |
| 11_all_characters | LPIPS↓ | **0.511** | 0.552 | 0.627 |

### Ablation Study (Impact of Light Intensity)

| Illumination (lux) | Metric | Deblur e-NeRF | Robust e-NeRF | Gap |
|------------|------|---------------|---------------|------|
| 100,000 (Bright Light) | PSNR↑ | 28.73 | 27.62 | +1.11 |
| 1,000 (Normal) | PSNR↑ | 28.41 | 26.11 | +2.30 |
| 10 (Extremely Low Light) | PSNR↑ | 28.62 | 22.72 | **+5.90** |

### Key Findings

- The more severe the motion blur (high-speed / low-light), the larger the advantage of Deblur e-NeRF over existing methods, with PSNR exceeding Robust e-NeRF by 5.3 dB at 4× speed.
- Even under extremely low-light conditions (10 lux), the performance of the proposed method barely drops (28.62 vs. blur-free upper bound of 29.43), whereas Robust e-NeRF severely degrades.
- Even under ideal conditions without motion blur, the proposed total variation loss still outperforms the gradient loss by about 1 dB, proving its independent value.
- Supports joint optimization of pixel bandwidth model parameters without requiring precise priors.

## Highlights & Insights

- **First to reveal and resolve the impact of event camera motion blur in NeRF reconstruction**: This dispels the common misconception that "event cameras are free from motion blur".
- **Physically accurate modeling**: Modeling the bandwidth limitation from the circuit level rather than a simple linear approximation, offering strong physical interpretability.
- **Universal value of total variation loss**: It improves reconstruction quality even in blur-free scenarios, showing that the regularization improvement has independent significance.
- **Open-source improved event simulator and synthetic datasets**, providing infrastructure for future research.

## Limitations & Future Work

- When the black level $L_{dark}$ is unknown, the signal radiance and dark current cannot be separated, allowing only the reconstruction of effective radiance.
- The training batch size is constrained (only 1/8 of the baseline); improving computational efficiency may yield better results.
- Pixel bandwidth model parameters depend on specific sensor and bias settings, and cross-device generalization remains to be verified.
- Real-world experimental scenarios are limited, having been tested on only two sequences of the EDS dataset.

## Related Work & Insights

- **vs. Robust e-NeRF**: Both belong to event-based NeRF reconstruction, but Robust e-NeRF completely ignores event motion blur; this work extends its event generation model with a bandwidth model, achieving a PSNR improvement of over 5 dB in severely blurred scenarios.
- **vs. E2VID + NeRF**: A two-stage method that first reconstructs video frames from events and then applies NeRF, which performs significantly worse than end-to-end approaches.
- **vs. Image Deblurring NeRF**: Image blur can be simply modeled as an average over the exposure time (LTI low-pass filtering), but event blur is non-linear and light-intensity-dependent, requiring NLTI filter modeling.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to model event camera motion blur from a physical circuit perspective and apply it to NeRF reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and real-world experiments cover various speed/illumination conditions, though the real-world dataset scale is small.
- Writing Quality: ⭐⭐⭐⭐ The physical modeling is clearly explained, and the mathematical derivation is complete.
- Value: ⭐⭐⭐⭐ Challenges the common assumption that event cameras are free from blur, with open-source code and datasets driving the field forward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)
- [\[CVPR 2026\] Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events](../../CVPR2026/3d_vision/seeing_through_light_and_darkness_sensor-physics_grounded_deblurring_hdr_nerf_fr.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)
- [\[ECCV 2024\] Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction](deceptive-nerf3dgs_diffusion-generated_pseudo-observations_for_high-quality_spar.md)
- [\[ECCV 2024\] RoGUENeRF: A Robust Geometry-Consistent Universal Enhancer for NeRF](roguenerf_a_robust_geometry-consistent_universal_enhancer_for_nerf.md)

</div>

<!-- RELATED:END -->
