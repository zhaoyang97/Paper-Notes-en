---
title: >-
  [Paper Note] Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation
description: >-
  [NeurIPS 2025][3D Vision][4D Gaussian Splatting] This paper proposes 4D Gaussian Ray Tracing (4D-GRT), which integrates 4D Gaussian Splatting with physics-based ray tracing. After reconstructing dynamic scenes from multi-view videos, the method renders physically accurate video data with controllable camera effects including fisheye distortion, depth of field blur, and rolling shutter artifacts.
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "Ray Tracing"
  - "Camera Effects"
  - "Dynamic Scene Reconstruction"
  - "Data Generation"
date: 2026-05-08
content_hash: ea5e47a3eeca1ae6
---

# Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation

**Conference**: NeurIPS 2025
**arXiv**: [2509.10759](https://arxiv.org/abs/2509.10759)  
**Code**: [Project Page](https://shigon255.github.io/4DGRT-project-page)  
**Area**: 3D Vision
**Keywords**: 4D Gaussian Splatting, Ray Tracing, Camera Effects, Dynamic Scene Reconstruction, Data Generation

## TL;DR

This paper proposes 4D Gaussian Ray Tracing (4D-GRT), which integrates 4D Gaussian Splatting with physics-based ray tracing. After reconstructing dynamic scenes from multi-view videos, the method renders physically accurate video data with controllable camera effects including fisheye distortion, depth of field blur, and rolling shutter artifacts.

## Background & Motivation

**Real-world camera effects are ubiquitous**: Camera effects such as fisheye distortion, depth of field (DoF), and rolling shutter (RS) are pervasive in the real world, yet mainstream vision systems assume an ideal pinhole camera model and degrade significantly when encountering such effects.

**Severe shortage of training data**: High-quality dynamic scene data with accurate camera effect parameter annotations is lacking, preventing models from learning to handle diverse camera effects.

**High cost and domain gap of traditional synthesis methods**: While rendering engines such as Blender enable controllable parameter synthesis, scene modeling incurs high labor costs, and the sim-to-real gap limits model generalization.

**Video generative models lack understanding of camera parameters**: Current mainstream video generative models (world models) do not possess physical understanding of camera effect parameters; providing numerical parameters results in videos that severely violate physical laws, producing artifacts or incorrect effects.

**Dynamic Gaussian Splatting lacks light transport modeling**: Dynamic Gaussian Splatting methods excel at dynamic scene reconstruction but their rasterization-based rendering pipelines cannot simulate camera effects that require ray tracing.

**Dynamic NeRF is too slow**: Dynamic NeRF methods support ray tracing but render extremely slowly and achieve lower reconstruction quality than Gaussian Splatting, making them impractical for large-scale data generation.

## Method

### Overall Architecture

4D-GRT adopts a two-stage pipeline. In the **first stage**, a 4D Gaussian Splatting (4D-GS) representation is optimized from synchronized multi-view videos via differentiable ray tracing to reconstruct the dynamic scene. In the **second stage**, ray tracing is performed on the reconstructed scene using physical camera model parameters to render videos with controllable camera effects.

### Key Design 1: 4D Gaussian Deformation Field for Dynamic Scene Representation

- **Function**: The dynamic scene is represented as a set of canonical 3D Gaussians $G$ coupled with a deformation field network that predicts per-frame Gaussian attribute residuals $(\Delta x, \Delta r, \Delta s)$ via a spatiotemporal encoder and multi-head decoder.
- **Mechanism**: A 4D voxel plane $R_l$ extracts spatiotemporal features, which are fused by a lightweight MLP $\varphi$; separate heads $\varphi_x, \varphi_r, \varphi_s$ predict residuals for position, rotation, and scale, which are added to the canonical Gaussians to obtain the deformed $G_t$.
- **Design Motivation**: Compared to storing per-frame Gaussians directly, the deformation field parameterization is more compact and efficient. The spatiotemporal voxel plane encoding captures multi-resolution motion information, balancing expressiveness and memory overhead.

### Key Design 2: Differentiable Ray Tracing Rendering

- **Function**: During training, differentiable ray tracing replaces conventional 3D-GS rasterization to directly trace rays through 3D Gaussian primitives.
- **Mechanism**: The $k$-buffer hit-based marching scheme from 3DGRT is adopted, leveraging the NVIDIA OptiX hardware-accelerated interface so that rays traverse 3D Gaussian primitives to compute intersections and colors, supporting end-to-end differentiable optimization.
- **Design Motivation**: Rasterization-based rendering supports only pinhole projection and cannot simulate complex lens distortions or optics-related light transport effects. Ray tracing inherently supports arbitrary ray generation strategies for any camera model, providing a unified interface for subsequent camera effect simulation.

### Key Design 3: Physics-based Camera Effect Rendering Module

- **Function**: Three representative camera effects are simulated on the reconstructed scene: fisheye distortion, depth of field blur, and rolling shutter.
- **Mechanism**:
    - **Fisheye**: A 4th-order polynomial radial distortion model $\theta = k_0 + k_1 r + k_2 r^2 + k_3 r^3 + k_4 r^4$ converts pixel coordinates to physical sensor coordinates, from which the polar and azimuthal angles are computed to define spherical ray directions.
    - **Depth of Field**: Given a focus distance $f_z$ and aperture radius $r_a$, perturbed ray origins are uniformly sampled over a circular aperture and averaged across multiple samples to achieve physically accurate defocus blur.
    - **Rolling Shutter**: Each row of pixels traces rays through the deformed Gaussians $G_{t_r}$ at its corresponding exposure time $t_r$, with a row-chunking approximation strategy (chunk size $N_c$) employed for acceleration.
- **Design Motivation**: Directly embedding optical camera models within the ray tracing framework requires no complex Jacobian derivations or additional module modifications to Gaussian Splatting, preserving parameter controllability and physical accuracy.

### Key Design 4: Rolling Shutter Row-Chunking Acceleration Strategy

- **Function**: Image rows are divided into blocks of size $N_c$; all rows within a block share the same average exposure time, enabling parallel ray tracing.
- **Mechanism**: Assuming moderate scene motion and short shutter duration, exposure time differences among rows within a block are negligible. Larger blocks increase speed but may introduce block artifacts, while smaller blocks improve quality at the cost of speed.
- **Design Motivation**: Exact per-row deformation cannot be parallelized since each row requires an independently deformed set of Gaussians. The chunking approximation achieves a practical balance between speed and quality.

## Loss & Training

The training loss consists of two components: an L1 loss $\mathcal{L}_1(C_{v,t}, \hat{C}_{v,t})$ between the rendered image and the ground truth image, and a total variation regularization $\mathcal{L}_{TV}$ on the 4D voxel planes. At each iteration, a viewpoint $v$ and timestamp $t$ are randomly sampled; the deformed Gaussians are obtained and rendered via ray tracing for optimization. Training takes approximately 3 hours on an RTX 4090.

## Key Experimental Results

### Dataset & Setup

The authors construct 8 dynamic indoor scenes (basketball court, warehouse, living room, bathroom) using Blender 4.5. Each scene comprises 50 camera viewpoints, 50 frames, and 512×512 resolution, with paired data rendered under four camera effects (pinhole / fisheye / rolling shutter / DoF) as a benchmark.

### Main Results

**Table 1: Pinhole Rendering Quality Comparison**

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
|--------|-------------|--------|---------|-------|
| HexPlane | 23.11 | 0.7956 | 0.2942 | 0.20 |
| MSTH | 29.43 | 0.9023 | 0.1139 | 9.38 |
| **4D-GRT (Ours)** | **32.80** | 0.8898 | **0.1018** | **36.56** |

**Table 2: Depth of Field Effect Rendering Comparison**

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
|--------|-------------|--------|---------|-------|
| HexPlane | 18.37 | 0.7343 | 0.5056 | 0.01 |
| MSTH | 28.47 | 0.9009 | 0.1540 | 0.57 |
| **4D-GRT (Ours)** | **31.25** | **0.9124** | **0.1210** | **3.44** |

**Table 3: Rolling Shutter Effect Rendering (Varying Chunk Size)**

| Method | Chunk | PSNR (dB) ↑ | FPS ↑ |
|--------|-------|-------------|-------|
| HexPlane | N/A | 21.35 | 0.21 |
| MSTH | N/A | 28.70 | 9.35 |
| **4D-GRT** | 1 row | **31.61** | 0.76 |
| **4D-GRT** | 4 rows | **31.61** | 4.99 |
| **4D-GRT** | 16 rows | **31.61** | 13.54 |

### Key Findings

- 4D-GRT substantially outperforms both baselines in PSNR across pinhole, DoF, and rolling shutter rendering; fisheye also achieves the highest masked PSNR (28.89 vs. 26.79).
- Pinhole rendering reaches 36.56 FPS, approximately 4× faster than MSTH and ~180× faster than HexPlane.
- The row-chunking strategy is highly effective: at chunk size 16, FPS increases from 0.76 to 13.54 with negligible PSNR loss.
- Qualitative results on the real-world Neural 3D Video dataset demonstrate good generalization.

## Highlights & Insights

1. **The first work to perform ray tracing on a dynamic Gaussian scene representation**, bridging 4D-GS reconstruction and physics-based camera effect simulation.
2. The two-stage design is elegant and clean — reconstruction and effect rendering are decoupled, allowing any combination of camera effects to be rendered from a single reconstructed scene.
3. The work reveals the inability of current video generative models (world models) to interpret camera effect parameters, which has practical engineering significance.
4. The paper provides the first multi-view dynamic scene paired benchmark encompassing four camera effects, filling a critical data gap.
5. The rolling shutter row-chunking acceleration strategy is simple yet effective, offering flexible trade-offs between speed and accuracy.

## Limitations & Future Work

1. **Dependency on multi-view video input**: High-quality reconstruction requires sufficient, synchronized multi-view video; sparse-view or monocular input scenarios are not supported.
2. **Quantitative evaluation limited to synthetic data**: Quantitative results on real-world scenes are absent; real-world evaluation is limited to qualitative demonstration.
3. **Long training time**: While 3 hours is much faster than HexPlane (12 h), it is significantly slower than MSTH (8 min), potentially becoming a bottleneck for large-scale data generation.
4. **Rolling shutter currently limited to static cameras**: Simulation of rolling shutter under moving cameras has not been implemented, limiting the scope of application.
5. **Unobserved region quality is uncontrollable for fisheye rendering**: Reconstruction quality beyond the training field of view is not guaranteed, necessitating masked evaluation.

## Related Work & Insights

- **3DGRT**: Performs ray tracing on static 3D Gaussians; this paper directly extends that approach to dynamic scenes.
- **3DGUT**: Approximates Gaussian projection via the unscented transform to support diverse camera models, but is likewise restricted to static scenes.
- **4D-GS (Wu et al.)**: The dynamic Gaussian representation framework directly adopted in this work, which encodes the deformation field using spatiotemporal voxel planes.
- **HexPlane / MSTH**: Plane/grid-based 4D NeRF methods that support ray tracing but suffer from slow speed and lower quality.
- **Curved Diffusion / AKiRa**: Diffusion-model-based camera effect synthesis methods that lack physical constraints and precise parameter control.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce ray tracing into 4D Gaussian representations for dynamic scene camera effect generation
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison across four effects with a synthetic benchmark and qualitative real-world validation, though downstream task evaluation is absent
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete methodology, and detailed experiments
- Value: ⭐⭐⭐⭐ — Provides a practical data generation solution for improving camera effect robustness in vision models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](../../CVPR2026/3d_vision/4c4d_4_camera_4d_gaussian_splatting.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](../../ICCV2025/3d_vision/radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[NeurIPS 2025\] VisualSync: Multi-Camera Synchronization via Cross-View Object Motion](visual_sync_multi-camera_synchronization_via_cross-view_object_motion.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](../../CVPR2025/3d_vision/irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[CVPR 2026\] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting](../../CVPR2026/3d_vision/stochastic_ray_tracing_for_the_reconstruction_of_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
