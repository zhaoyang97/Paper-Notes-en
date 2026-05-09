---
title: >-
  [Paper Note] From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images
description: >-
  [CVPR 2026][3D Vision][Urban Reconstruction] A two-stage pipeline for reconstructing city-scale 3D models from sparse satellite images: Z-Monotonic SDF for geometry to ensure structural integrity of buildings, followed by a fine-tuned FLUX diffusion model for "deterministic inpainting" that synthesizes photorealistic textures from degraded maps, enabling view extrapolation of nearly 90° from orbit to ground level.
tags:
  - CVPR 2026
  - 3D Vision
  - Urban Reconstruction
  - Satellite Imagery
  - 2.5D SDF
  - Texture Restoration
  - View Extrapolation
date: 2026-05-08
content_hash: c411cb1e5f7f7cc8
---

# From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images

**Conference**: CVPR 2026
**arXiv**: [2512.07527](https://arxiv.org/abs/2512.07527)
**Code**: [Project Page](https://pku-vcl-geometry.github.io/Orbit2Ground/)
**Area**: 3D Vision
**Keywords**: Urban Reconstruction, Satellite Imagery, 2.5D SDF, Texture Restoration, View Extrapolation

## TL;DR

A two-stage pipeline for reconstructing city-scale 3D models from sparse satellite images: Z-Monotonic SDF for geometry to ensure structural integrity of buildings, followed by a fine-tuned FLUX diffusion model for "deterministic inpainting" that synthesizes photorealistic textures from degraded maps, enabling view extrapolation of nearly 90° from orbit to ground level.

## Background & Motivation

**Background**: NeRF/3DGS have achieved success in object- and street-level reconstruction, but city-scale reconstruction faces data acquisition challenges — ground/UAV capture is costly and has limited coverage. Satellite imagery provides cheap city-scale coverage.

**Limitations of Prior Work**: Satellite imagery poses extreme challenges — a nearly 90° viewpoint gap exists between the source (nadir) and target (ground-level) views; building facades suffer from severe perspective foreshortening; atmospheric distortion and sensor limitations cause texture degradation. NeRF/3DGS rely on dense parallax and clear photometric signals, and fail completely in this setting.

**Key Challenge**: Satellite imagery provides almost no parallax information for vertical structures (MVS can only recover rooftop and ground point clouds), yet the goal is to synthesize ground-level photorealistic renderings from such data.

**Key Insight**: Decouple the problem into geometry and appearance subproblems — geometry is constrained by city-specific 2.5D priors (buildings are predominantly vertical extrusions), while appearance quality is compensated by generative model priors.

**Core Idea**: Z-Monotonic SDF ensures geometrically correct structures → a deterministic image restoration network compensates for texture quality.

## Method

### Overall Architecture

A two-stage design:
- **Stage 1 (Geometry)**: Optimize Z-Monotonic SDF → differentiable isosurface extraction → high-fidelity watertight mesh
- **Stage 2 (Appearance)**: Base texture back-projection → FLUX fine-tuned restoration network → iterative texture refinement

### Key Designs

1. **Z-Monotonic SDF**:

    - Core constraint: SDF values are monotonically non-decreasing along the Z-axis $\frac{\partial s(x,y,z)}{\partial z} \geq 0$
    - **Physical meaning**: Continuous surfaces (rooftops/ground) → $s=0$ has a unique solution defining height; building edges → $s=0$ forms a vertical plateau over $[z_{ground}, z_{roof}]$, automatically generating vertical facades
    - Implementation: 256×256 2D grid with learnable parameters $h_j$ per cell, monotonic curves constructed via $\tanh$ activation + spatial interpolation
    - Optimization objective: $\mathcal{L}_{geo} = \sum_{p} \|p_z - m^*(p)_z\|_1 + \lambda_{Lap}\mathcal{L}_{Lap} + \lambda_{Nrm}\mathcal{L}_{Nrm}$
    - **Design Motivation**: Why 2.5D instead of full 3D? Satellite data is inherently top-down, and buildings are predominantly vertical extrusions. The ability to model non-monotonic structures (e.g., bridges) is sacrificed in exchange for decisive robustness against geometric ambiguity.

2. **Deterministic Image Restoration Network**:

    - Fine-tuned from FLUX-Schnell, trained on 100K pairs of aerial images
    - Key design: learns a **deterministic mapping** from degraded renderings to clean targets (rather than averaging multiple stochastic samples)
    - Loss: $\mathcal{L}_{restorer} = \mathcal{L}_{LPIPS}(\hat{I}, I_{high}) + \lambda_{CHAR}\mathcal{L}_{CHAR}(\hat{I}, I_{high})$
    - **Design Motivation**: Why deterministic restoration instead of SDS distillation? The stochasticity of generative models produces cross-view inconsistent details; deterministic restoration guarantees global consistency and is more efficient (single forward pass vs. multi-sample averaging).

3. **Iterative Texture Refinement**:

    - Loop: render degraded views → restoration network enhancement → use as supervision to optimize texture → next iteration benefits from higher-quality inputs
    - Camera trajectory: uniform 150m grid sampling, altitude 450m, pitch 45°, four cardinal directions
    - Only 2 iterations are sufficient

### Loss & Training

- Geometry and appearance are optimized separately, not jointly
- Geometry optimization uses Adam (lr=0.01); texture uses MSE + SSIM loss
- Restoration network trained for 10K steps, batch size 96
- **Overall efficiency**: Single A6000 GPU, 1.5 hours for 1 km²

## Key Experimental Results

### Main Results

| Method | MatrixCity F1↑ | MatrixCity CD↓ | DFC PSNR↑ | GoogleEarth PSNR↑ |
|------|---------------|---------------|-----------|-------------------|
| Mip-Splatting | 0.377 | 0.161 | 10.289 | 12.214 |
| 2DGS | 0.556 | 0.073 | 7.366 | 11.022 |
| CityGS-X | 0.189 | 0.227 | FAIL | 12.674 |
| Skyfall-GS | 0.296 | 0.359 | 12.460 | 12.456 |
| **Ours** | **0.643** | **0.036** | **13.059** | **12.770** |

F1 improves by 0.09, Chamfer Distance is reduced by 50%, with particularly significant gains on the DFC dataset.

### Ablation Study

| Configuration | F1↑ | CD↓ | PSNR↑ |
|------|-----|-----|-------|
| Naive MC 128 | 0.279 | 0.0749 | 16.800 |
| Naive MC 256 | 0.412 | 0.0757 | 17.002 |
| w/o Regularization | 0.637 | 0.0364 | 17.115 |
| **Full Model** | **0.643** | **0.0357** | **17.153** |
| w/o Image Restoration | - | - | 17.038 |

### Key Findings

- Z-Monotonic SDF significantly outperforms the naive Marching Cubes baseline (F1: 0.643 vs. 0.279/0.412)
- The restoration network contributes non-negligibly (PSNR: 17.153 vs. 17.038), with substantial qualitative differences
- Competing methods produce fragmented, floater geometry, or blurry textures in the satellite setting; the proposed method is robust and stable

## Highlights & Insights

- **Clear problem formulation**: Framing satellite-to-ground as "extreme view extrapolation" precisely captures the core challenge
- **Elegant 2.5D representation**: Domain priors of urban structure are leveraged to convert an ill-posed problem into constrained optimization
- **Deterministic restoration vs. generative distillation**: A more practical, consistent, and efficient design choice
- **Large-scale validation**: Reconstruction over a 4 km² real urban area demonstrates practical applicability

## Limitations & Future Work

- The 2.5D assumption cannot handle non-monotonic structures such as bridges and overpasses
- The restoration network requires large amounts of paired aerial training data
- Texture quality is ultimately bounded by the hallucination capacity of the restoration model and may be inconsistent with true appearance
- Resolution is limited by the 256×256 grid

## Related Work & Insights

- **GaussianShading / Skyfall-GS**: Recent work on satellite-based reconstruction
- **FLUX**: The backbone diffusion model underlying the restoration network
- **DreamFusion / SDS**: An alternative route for generative priors (the paper argues its inapplicability in this setting)

## Rating

- Novelty: ⭐⭐⭐⭐ The Z-Monotonic SDF design is elegant; the two-stage decoupling is well-motivated
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation on synthetic, real, and large-scale data
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation and method description are exceptionally clear
- Value: ⭐⭐⭐⭐ High practical value for satellite reconstruction with clear applications in urban planning and simulation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[CVPR 2026\] Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion](yocity_personalized_and_boundless_3d_realistic_city_scene_generation_via_self-cr.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Affostruction: 3D Affordance Grounding with Generative Reconstruction](affostruction_3d_affordance_grounding_with_generative_reconstruction.md)
- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
