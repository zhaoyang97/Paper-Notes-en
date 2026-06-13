---
title: >-
  [Paper Note] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes LocalDyGS, a framework that decomposes complex global dynamic scenes into local spaces defined by seed points and generates temporal Gaussians via static-…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "dynamic scene reconstruction"
  - "multi-view"
  - "local spatial modeling"
  - "static-dynamic decoupling"
  - "temporal Gaussians"
date: 2026-05-08
content_hash: 24cb1adc0774cf5d
---

# LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling

**Conference**: ICCV 2025
**arXiv**: [2507.02363](https://arxiv.org/abs/2507.02363)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, dynamic scene reconstruction, multi-view, local spatial modeling, static-dynamic decoupling, temporal Gaussians

## TL;DR

This paper proposes LocalDyGS, a framework that decomposes complex global dynamic scenes into local spaces defined by seed points and generates temporal Gaussians via static-dynamic feature decoupling to model local motions independently, achieving high-quality reconstruction of large-scale complex dynamic scenes for the first time.

## Background & Motivation

Multi-view dynamic scene reconstruction is a fundamental problem in 3D vision, with applications in free-viewpoint sports broadcasting, AR/VR, and gaming. Existing methods face the following challenges:

- **Deformation field methods** (4DGaussian): Map from canonical space to deformation fields; struggle with large-scale motion.
- **Trajectory tracking methods** (SpaceTimeGS): Represent trajectories via polynomial/Fourier series; suffer from blurring and flickering under large-scale motion.
- **Online streaming methods** (3DGStream): Model per-frame; limited capacity for large-range motion.
- **Storage issues**: 3DGStream requires 1230 MB for 300 frames; RealTimeGS exceeds 1000 MB.

Core problem: Explicitly tracking the long-term motion of each Gaussian point is infeasible for large-scale complex motion.

Core Idea: **Decompose the global dynamic scene into local spaces, where each seed point independently models short-range motion.** Moving objects are represented by multiple seeds—seeds activate when an object passes through and deactivate when it leaves.

## Method

### Overall Architecture

Two major components:
1. Global-to-local spatial decomposition: Fuse multi-frame SfM point clouds to initialize seeds.
2. Local spatial temporal Gaussian generation: Static-dynamic decoupling + MLP decoding.

### Key Designs

**1. Global Seed Initialization**

$N$ frames are sampled to extract and fuse SfM point clouds, initializing seed positions $\mu$. Each seed contains:
- Position $\mu$ (global parameter)
- Static feature $f_s \in \mathbb{R}^{64}$ (shared across time steps, initialized to 0)
- Local spatial scale $v \in \mathbb{R}^3$ (initialized to the average distance of the 3 nearest neighboring seeds)

**2. Spatio-temporal Field with Feature Decoupling**

- **Static feature** $f_s$: Independently optimized per local space, carries the majority of scene information.
- **Dynamic residual field** $F_d$: Multi-resolution 4D hash encoding (position + time) + shallow MLP.
    - Hash table size $2^{17}$; $L$ resolution levels concatenated and fused by MLP.
    - Input $(μ, t) \in \mathbb{R}^4$.
- **Weight field** $F_w$: Predicts $w_s, w_d$ to balance static and dynamic contributions.
- Weighted feature: $f_w = w_s \cdot f_s + w_d \cdot f_d$
- Key finding: Dynamic residuals approach zero; the majority of information is carried by static features.

**3. Temporal Gaussian Generation**

Each seed generates $k=10$ temporal Gaussians:
- Position: $\mu_t = \mu + v \cdot F_\mu(f_w)$
- Opacity: $\text{Sigmoid}(F_o(f_w, d))$, where $d$ is the camera viewing direction
- Rotation, scale, and color decoded via independent MLPs
- **Deactivation mechanism**: Deactivated when $\sigma < \tau_\alpha = 0.01$, reducing computational cost.

**4. Adaptive Seed Growing (ASG)**

- Records the maximum 2D projection gradient and 3D position of temporal Gaussians over $n$ iterations.
- New seeds are added when $\nabla_{\max} > \tau_g = 0.001$.
- Executed every 100 iterations between iterations 3000 and 15000.

### Loss & Training

$$\mathcal{L} = (1 - 0.2) \cdot \mathcal{L}_1 + 0.2 \cdot \mathcal{L}_{\text{SSIM}} + 0.001 \cdot \mathcal{L}_v$$

- $\mathcal{L}_v = \sum \prod(s_t^i)$: Volume regularization encouraging small Gaussian sizes to preserve locality.
- Adam optimizer, 30,000 iterations.

## Key Experimental Results

### N3DV Dataset (21 cameras, fine-grained motion)

| Method | PSNR↑ | LPIPS↓ | FPS↑ | Time↓ | Storage↓ |
|--------|-------|--------|------|-------|----------|
| 4DGaussian | 31.02 | 0.150 | 30 | 0.67h | 90MB |
| SpaceTimeGS | 32.05 | 0.044 | 140 | >5h | 200MB |
| 3DGStream | 31.67 | - | 215 | 1.0h | 1230MB |
| RealTimeGS | 32.01 | 0.055 | 114 | 9.0h | >1000MB |
| **LocalDyGS** | **32.28** | **0.043** | 105 | **0.58h** | **100MB** |

### MeetRoom Dataset (13 cameras, sparse views)

| Method | PSNR↑ | Time↓ | Storage↓ |
|--------|-------|-------|----------|
| 3DGS (per-frame) | 31.31 | 13h | 6330MB |
| 3DGStream | 30.79 | 0.6h | 1230MB |
| **LocalDyGS** | **32.45** | **0.36h** | **90MB** |

### VRU Basketball Court (34 cameras, large-scale complex motion)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| 4DGS | 28.32 | 0.930 | 0.186 |
| SpaceTimeGS | 27.42 | 0.926 | 0.193 |
| **LocalDyGS** | **30.58** | **0.944** | **0.173** |

Exceeds dynamic methods by more than 2 dB, approaching the static 3DGS upper bound.

### Ablation Study

| Component | Effect |
|-----------|--------|
| ASG seed growing | PSNR 31.81 → 33.02 (+1.21 dB) |
| Remove static feature | 29.46 vs. 31.40 (−1.94 dB) |
| Remove deactivation | FPS 89 → 105, PSNR nearly unchanged |
| $N=6$ vs. $N=30$ | Performance nearly identical (32.28 vs. 32.30) |
| $N=1$ | 31.84 (−0.44 dB), insufficient coverage |
| $k=5/10/20$ | $k=10$ achieves optimal balance |

### Key Findings

- First method to successfully handle large-scale dynamic datasets such as the VRU basketball court.
- Static-dynamic decoupling contributes the most (−1.94 dB if removed).
- Only 6 SfM frames are sufficient to achieve optimal performance; the method is insensitive to the initial point cloud.
- Training time: 35 minutes on a single RTX 3090.
- Storage: 100 MB / 300 frames vs. 1230 MB for 3DGStream.

## Highlights & Insights

1. **Core idea of local decomposition**: Reducing global long-range tracking to local short-range modeling is a key breakthrough for large-scale dynamic scenes.
2. **Effectiveness of static-dynamic decoupling**: Dynamic residuals approach zero, and decoupling substantially reduces modeling difficulty.
3. **Extreme efficiency**: Best quality + 35-minute training + 100 MB storage.
4. **First large-scale dynamic scene benchmark**: VRU basketball court validates practical applicability.
5. **Elegant dynamic extension of ScaffoldGS**: Inherits the anchor structure and incorporates the temporal dimension.
6. **Deactivation mechanism analogous to MoE**: Sparse activation reduces computational overhead.

## Limitations & Future Work

- Relies on multi-frame SfM initialization; fast motion or texture-less regions may lead to insufficient coverage.
- Rendering at 105 FPS is lower than 3DGStream at 215 FPS.
- Assumes camera parameters are known and accurately calibrated.
- Volume regularization may constrain the representation of large-scale objects.
- Memory of 4D hash encoding grows with resolution.

## Related Work & Insights

- **Static novel view synthesis**: 3DGS for real-time rendering; ScaffoldGS with anchor-based representation.
- **Deformation field methods**: 4DGaussian canonical-to-deformation mapping.
- **Trajectory tracking**: SpaceTimeGS with polynomial control.
- **Streaming methods**: 3DGStream for per-frame online reconstruction.
- **4DGS extensions**: Fit Gaussians in 4D space; high storage and computation cost.

## Rating

- **Novelty**: ★★★★☆ — Local spatial decomposition + feature decoupling introduces a new paradigm for dynamic scene modeling.
- **Technical Depth**: ★★★★★ — Three datasets at different motion scales; highly thorough ablation study.
- **Experimental Thoroughness**: ★★★★★ — Comprehensive quantitative and qualitative evaluation; first large-scale validation.
- **Practicality**: ★★★★★ — Fast training / low storage / high quality / first large-scale dynamic scene support.
- **Overall Score**: 9.0/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling](atlas_decoupling_skeletal_and_shape_parameters_for_expressive_parametric_human_m.md)
- [\[ICCV 2025\] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection](ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)
- [\[ICCV 2025\] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
