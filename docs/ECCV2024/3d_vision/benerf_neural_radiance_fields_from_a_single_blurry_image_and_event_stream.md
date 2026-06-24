---
title: >-
  [Paper Note] BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream
description: >-
  [ECCV 2024][3D Vision][NeRF] BeNeRF is proposed to jointly recover a neural radiance field and camera motion trajectory from only a **single blurry image** and its corresponding event stream. High-quality deblurring and novel view synthesis are achieved without requiring multi-view inputs or known poses.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "NeRF"
  - "Event Camera"
  - "Motion Deblurring"
  - "novel view synthesis"
  - "Camera Motion Estimation"
  - "B-Spline"
date: 2026-05-08
content_hash: 865748b060e592a6
---

# BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream

**Conference**: ECCV 2024  
**arXiv**: [2407.02174](https://arxiv.org/abs/2407.02174)  
**Code**: [wu-cvgl/BeNeRF](https://github.com/wu-cvgl/BeNeRF)  
**Area**: 3D Vision  
**Keywords**: NeRF, Event Camera, Motion Deblurring, novel view synthesis, Camera Motion Estimation, B-Spline

## TL;DR

BeNeRF is proposed to jointly recover a neural radiance field and camera motion trajectory from only a **single blurry image** and its corresponding event stream. High-quality deblurring and novel view synthesis are achieved without requiring multi-view inputs or known poses.

## Background & Motivation

**NeRF relies on multi-view sharp images**: Traditional NeRF requires multiple calibrated sharp RGB images to reconstruct 3D scenes, posing high requirements on both input quality and quantity.

**Blurry images contain rich information**: Although motion blur degrades image quality, it actually encodes the camera's motion trajectory during exposure as well as additional structural information, which can be effectively utilized.

**Complementary properties of event cameras**: Event cameras asynchronously capture pixel intensity changes with extremely low latency. Having an exceptionally high temporal resolution, they naturally complement the photon integration imaging process of frame-based cameras.

**Limitations of existing event-based deblurring methods**: Methods such as EDI and eSLNet can only restore sharp images but fail to extract camera motion trajectories or perform 3D reconstruction, limiting their application in 3D computer vision tasks.

**Deficiencies of multi-view event-based NeRF methods**: Though methods like E2NeRF incorporate event streams, they still require multi-view image inputs and employ a two-stage pipeline (reconstruction of sharp images using EDI followed by pose estimation via COLMAP), which introduces cumulative errors.

**Challenges of single-image NeRF**: Reconstructing NeRF from a single image is extremely ill-posed. Existing methods typically require pre-training on large-scale datasets to learn priors, resulting in limited generalization capability.

## Method

### Overall Architecture

Given a single blurry image and its corresponding event stream, BeNeRF jointly optimizes: (1) a neural radiance field representing the 3D scene using an MLP; and (2) the camera motion trajectory represented as a cubic B-spline curve in $\text{SE}(3)$ space. The entire system is trained by minimizing the discrepancy between the synthesized blurry and accumulated event images—generated through physical imaging models—and their real-world measurements.

### Key Designs

#### 1. SE(3) Cubic B-Spline Motion Modeling

- A group of learnable control points $\boldsymbol{T}_{c_i}^w \in \text{SE}(3)$ is used to define a cubic B-spline curve, representing the continuous trajectory of camera motion.
- Based on the matrix representation of the De Boor-Cox formula, the camera pose at any arbitrary time $t$ is interpolated from four neighboring control points.
- Due to the short exposure duration of a single image, only 4 control points are sufficient to represent the motion, which are initialized by random sampling around the identity pose.
- Compared with linear interpolation, the cubic B-spline can better model complex non-linear motion (yielding a PSNR improvement of approximately 3-5 dB).

#### 2. Physical Imaging Model for Blurry Images

- The blurry image is modeled as the average of $n$ virtual sharp images within the exposure time: $\mathbf{B}(\mathbf{x}) \approx \frac{1}{n}\sum_{i=0}^{n-1}\mathbf{I}_i(\mathbf{x})$.
- Each virtual sharp image is rendered from the NeRF using the corresponding pose interpolated from the B-spline.
- Ablation studies demonstrate that $n=19$ achieves the optimal balance between quality and efficiency.

#### 3. Event Stream Accumulation and Normalization

- Events within the time interval $\Delta t$ are accumulated into an event image $\mathbf{E}(\mathbf{x})$, which is normalized to eliminate the influence of the unknown contrast threshold $C$.
- Grayscale images at the start and end timestamps are rendered from NeRF to calculate the synthesized event image: $\hat{\mathbf{E}}(\mathbf{x})=\log(\mathbf{I}_{end})-\log(\mathbf{I}_{start})$.
- The event stream supplies abundant temporal constraints, effectively regularizing geometry ambiguities in single-image NeRF learning.

### Loss & Training

The total loss is a weighted sum of the photometric loss and event loss:

$$\mathcal{L}_{total} = \mathcal{L}_p + \beta \mathcal{L}_e$$

- $\mathcal{L}_p = \|\mathbf{B} - \hat{\mathbf{B}}\|^2$: MSE between the synthesized and real blurry images.
- $\mathcal{L}_e = \|\mathbf{E}_n - \hat{\mathbf{E}}_n\|^2$: MSE of the normalized event images.
- $\beta$ is set to 0.1 on synthetic data and 2 on real-world data. Two independent Adam optimizers are used to optimize the scene model and poses, respectively, with the learning rate exponentially decaying from $5\times10^{-4}$ over 80K training iterations.

## Key Experimental Results

### Table 1: Comparison with Single-Image Deblurring Methods on Synthetic Datasets (PSNR↑ / LPIPS↓)

| Method | Livingroom | Whiteroom | Pinkcastle | Tanabata | Outdoorpool | Average |
|---|---|---|---|---|---|---|
| SRN-Deblur | 30.86 / .253 | 27.59 / .250 | 23.12 / .325 | 19.89 / .426 | 27.79 / .359 | 25.85 / .323 |
| NAFNet | 29.92 / .227 | 28.16 / .199 | 22.41 / .306 | 18.96 / .391 | 26.75 / .328 | 25.24 / .290 |
| Restormer | 29.48 / .239 | 27.39 / .249 | 22.22 / .337 | 18.82 / .425 | 27.35 / .366 | 25.05 / .323 |
| **BeNeRF** | **37.11 / .063** | **32.95 / .079** | **29.68 / .076** | **32.14 / .052** | **36.38 / .068** | **33.65 / .068** |

BeNeRF outperforms the best baseline method by an average of **+7.8 dB** in PSNR, with a reduction of approximately **75%** in LPIPS.

### Table 2: Comparison with Multi-View NeRF Methods on the E2NeRF Synthetic Dataset (PSNR↑ / LPIPS↓)

| Method | Chair | Ficus | Hotdog | Lego | Materials | Mic | Average |
|---|---|---|---|---|---|---|---|
| NeRF | 24.29 / .125 | 22.98 / .104 | 27.75 / .116 | 21.95 / .210 | 19.99 / .151 | 20.50 / .158 | 22.91 / .144 |
| E2NeRF (Multi-view) | 31.28 / .061 | 30.00 / .036 | 34.34 / .066 | 28.11 / .108 | 27.27 / .092 | 27.60 / .072 | 29.77 / .073 |
| **BeNeRF (Single-image)** | 31.17 / **.050** | **30.81** / **.030** | 34.31 / **.054** | 28.09 / **.075** | **27.44** / **.071** | 26.13 / .074 | 29.66 / **.059** |

Using only a single blurry image achieves a PSNR level comparable to the multi-view E2NeRF method, with even superior LPIPS metrics.

### Real-world Dataset Results (BRISQUE↓)

| Method | Camera | Lego | Letter | Plant | Toys | Average |
|---|---|---|---|---|---|---|
| EDI | 29.74 | 29.35 | 28.74 | 31.09 | 37.09 | 31.20 |
| E2NeRF (Multi-view) | 33.40 | 33.85 | 37.41 | 32.02 | 43.00 | 35.94 |
| **BeNeRF (Single-image)** | **19.47** | **25.86** | **27.37** | **21.46** | **25.20** | **23.87** |

## Highlights & Insights

1. **Breakthrough in Extreme Settings**: The first to reconstruct NeRF from a single blurry image and event stream, representing the work with the most stringent input requirements in this field.
2. **No Pre-training or Priors Required**: As a test-time optimization method, it does not rely on pre-training on large-scale datasets, inherently avoiding generalization issues.
3. **Performance Comparable to Multi-View Methods**: Achieves comparable performance to E2NeRF (which requires multi-view inputs and long event streams) using only a single image, with even better LPIPS.
4. **Driven by Physical Models**: Both blur imaging and event generation are modeled based on physical processes, significantly outperforming learning-based methods on real-world data.
5. **Joint Pose Optimization**: Jointly recovers both the scene and motion from scratch, requiring no COLMAP or external pose estimation.

## Limitations & Future Work

1. **Processing Only a Single Image**: Unable to utilize temporal consistency in video sequences; extension to continuous frames would further improve reconstruction quality.
2. **High Training Overhead**: Optimizing each single image requires 80K iterations. Acceleration representations such as Instant-NGP are not integrated (noted as replaceable in the paper but not validated experimentally).
3. **Dependence on Event Cameras**: Requires extra event sensor hardware, which limits practical application scenarios.
4. **Limited Scene Types**: Validation is primarily conducted on small indoor scenes; performance on large-scale outdoor scenes or dynamic objects has not been evaluated.
5. **Sensitivity to the Number of Virtual Sharp Images**: The choice of $n=19$ is empirical, and different motion levels or patterns might require different configurations.

## Related Work & Insights

| Method | Input Requirements | Pose | Event Stream | 3D Reconstruction |
|---|---|---|---|---|
| BAD-NeRF | Multi-view blurry images | COLMAP Initialization | ✗ | ✓ |
| E2NeRF | Multi-view blurry + long event stream | COLMAP (EDI-assisted) | ✓ | ✓ |
| EDI | Single image + event stream | ✗ | ✓ | ✗ |
| **BeNeRF** | **Single image + event stream** | **Joint Optimization** | ✓ | ✓ |

BeNeRF simultaneously achieves camera motion estimation, image deblurring, and 3D scene reconstruction under the fewest input requirements.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to recover NeRF from a single blurry image and event stream; the problem setting is novel and highly challenging.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation on synthetic/real datasets, detailed ablation studies, and comparisons with multiple categories of baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear derivations of methods, rigorous physical modeling, and high-quality figures and tables.
- Value: ⭐⭐⭐⭐ — Achieves performance comparable to multi-view methods under extreme input conditions, demonstrating the great potential of event cameras in 3D vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](../../ICCV2025/3d_vision/evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)
- [\[ECCV 2024\] GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)
- [\[ECCV 2024\] G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)

</div>

<!-- RELATED:END -->
