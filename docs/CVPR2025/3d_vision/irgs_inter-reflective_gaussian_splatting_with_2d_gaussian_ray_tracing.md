---
title: >-
  [Paper Note] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing
description: >-
  [CVPR 2025][3D Vision][Inverse Rendering] This paper proposes the IRGS framework, which integrates the complete rendering equation (without simplification) into Gaussian Splatting for the first time. By leveraging the proposed differentiable 2D Gaussian ray tracing, it calculates incoming light visibility and indirect radiance in real-time, achieving significantly superior relighting and material estimation results compared to prior methods on multiple inverse rendering bench…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Inverse Rendering"
  - "Gaussian Splatting"
  - "Ray Tracing"
  - "Indirect Illumination"
  - "Material Estimation"
date: 2026-05-08
content_hash: 5edb2b8ec6171dc4
---

# IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing

**Conference**: CVPR 2025  
**arXiv**: [2412.15867](https://arxiv.org/abs/2412.15867)  
**Code**: [https://fudan-zvg.github.io/IRGS](https://fudan-zvg.github.io/IRGS)  
**Area**: 3D Computer Vision  
**Keywords**: Inverse Rendering, Gaussian Splatting, Ray Tracing, Indirect Illumination, Material Estimation

## TL;DR
This paper proposes the IRGS framework, which integrates the complete rendering equation (without simplification) into Gaussian Splatting for the first time. By leveraging the proposed differentiable 2D Gaussian ray tracing, it calculates incoming light visibility and indirect radiance in real-time, achieving significantly superior relighting and material estimation results compared to prior methods on multiple inverse rendering benchmarks.

## Background & Motivation

1. **Background**: Inverse rendering aims to reconstruct geometry and estimate material and illumination from a set of posed images. 3D Gaussian Splatting (3DGS) has emerged as a promising 3D scene representation due to its exceptional rendering quality and efficiency.

2. **Limitations of Prior Work**: Due to the lack of an efficient Gaussian ray tracer, existing 3DGS-based inverse rendering methods either adopt simplified rendering equations (e.g., GS-IR uses a split-sum approximation) or approximate indirect radiance using learnable parameters (e.g., R3DG), leading to inaccurate material and illumination estimation.

3. **Key Challenge**: The rendering equation requires accurate computation of visibility and indirect radiance for each incoming direction. However, because 3DGS is based on rasterization and cannot perform ray tracing, various compromises have to be made.

4. **Goal**: How to fully implement the rendering equation without any simplifications in an efficient Gaussian splatting framework to accurately capture inter-reflection effects.

5. **Key Insight**: Inspired by 3DGRT (3D Gaussian Ray Tracing), but noting that directly applying 3DGRT to pre-trained 3DGS models leads to significant quality degradation, the authors opt to perform ray tracing on 2D Gaussian primitives—as 2D Gaussian discs have well-defined ray-plane intersections.

6. **Core Idea**: Construct differentiable ray tracing on 2D Gaussian splatting to provide real-time query capability of visibility and indirect radiance for the complete evaluation of the rendering equation.

## Method

### Overall Architecture
IRGS adopts a two-stage training framework. The input is a set of posed RGB images, and the output consists of decomposed geometry, material (albedo and roughness), and illumination (environment cubemap). The first stage pre-trains a standard 2DGS model to obtain reliable geometry. Based on this, the second stage introduces material properties and the complete rendering equation, utilizing 2D Gaussian ray tracing to calculate the visibility and indirect radiance of incoming light.

### Key Designs

1. **2D Gaussian Ray Tracing (2DGRT)**:

    - **Function**: Performs efficient and accurate ray tracing on 2D Gaussian primitives to calculate the accumulated opacity and color of arbitrary rays.
    - **Mechanism**: An adaptive icosahedron mesh is used as a bounding proxy for each 2D Gaussian, and an OptiX hardware-accelerated BVH is constructed using ray-triangle intersections. For each 2D Gaussian, the ray-plane intersection point $\boldsymbol{p} = \boldsymbol{r}_o + \tau\boldsymbol{r}_d$ is analytically computed, where $\tau = \frac{\boldsymbol{n}^\top(\boldsymbol{\mu}-\boldsymbol{r}_o)}{\boldsymbol{n}^\top\boldsymbol{r}_d}$. The final result is then obtained via k-buffer sorting and alpha blending.
    - **Design Motivation**: When 3DGRT performs ray tracing on 3D Gaussians, the ray-Gaussian intersection definition is inconsistent (3DGRT takes the maximum response point along the ray, which differs from the 2D projection calculation in splatting), leading to severe quality degradation when ray tracing is directly applied to pre-trained 3DGS. In contrast, the normal vector of a 2D Gaussian disc clearly defines the ray-plane intersection, ensuring consistency from splatting to ray tracing, with minimal quality degradation.

2. **Integration of the Complete Rendering Equation**:

    - **Function**: Applies the complete rendering equation at the pixel level after rasterization without any simplification.
    - **Mechanism**: The incoming radiance is decomposed into direct radiance (environment cubemap) and indirect radiance. Monte Carlo stratified sampling is used with $N_r=256$ incoming directions, and the visibility $V$ and indirect radiance $L_{ind}$ for each direction are queried via 2DGRT: $(L_{ind}(\omega_i, x), 1-V(\omega_i, x)) \leftarrow \text{Trace}(x, \omega_i)$. A simplified Disney BRDF model (only albedo and roughness) is adopted, splitting the BRDF into a diffuse term $f_d = a/\pi$ and a specular term $f_s$.
    - **Design Motivation**: Previous methods either simplified the rendering equation using the split-sum approximation (GS-IR) or approximated indirect radiance with learnable SH parameters (R3DG), failing to accurately model realistic inter-reflections. This method fully retains the rendering equation, and differentiable ray tracing allows gradients to propagate back to optimize indirect light.

3. **Efficient Optimization Scheme and Relighting Strategy**:

    - **Function**: Manages the computational demands of Monte Carlo sampling and queries indirect radiance during relighting.
    - **Mechanism**: Each training iteration sets a maximum of $N_{rays}=2^{18}$ rays and evaluates the rendering equation only for $\lfloor N_{rays}/N_r \rfloor$ random pixels, enabling the use of a large number of sampled rays ($N_r=256$) to improve estimation quality. During relighting, the albedo, roughness, and normal values for each incoming ray direction are obtained through ray tracing alpha blending, and then the indirect radiance is efficiently queried using both the split-sum approximation and a pre-filtered environment map.
    - **Design Motivation**: Full evaluation of the rendering equation for dense pixels is computationally expensive (256 rays per pixel). The pixel sub-sampling strategy maintains a high sampling count per pixel while controlling the overall computation. During relighting, changes in illumination make the optimized SH colors no longer applicable, and the split-sum approximation avoids recursive evaluations.

### Loss & Training
The first-stage loss: $\mathcal{L}^1 = \mathcal{L}_c + \lambda_n\mathcal{L}_n + \lambda_d\mathcal{L}_d + \lambda_{s,n}\mathcal{L}_{s,n} + \lambda_o\mathcal{L}_o$, which includes RGB reconstruction loss, normal consistency loss, depth distortion loss, edge-aware smoothness loss, and binary cross-entropy mask loss. The second stage adds a PBR color L1 loss $\mathcal{L}_1^{pbr}$, a white light prior regularization $\mathcal{L}_{light}$, and edge-aware smoothness regularization for albedo and roughness. Training takes about 40 minutes (on an RTX 3090), with 40K iterations / 15 mins for the first stage and 20K iterations / 25 mins for the second stage.

## Key Experimental Results

### Main Results

| Dataset | Metric | IRGS | R3DG | GS-IR | TensoIR |
|--------|------|------|------|-------|---------|
| Synthetic4Relight | Relight PSNR↑ | **34.90** | 31.00 | 25.40 | 29.69 |
| Synthetic4Relight | Albedo PSNR↑ | **30.81** | 28.31 | 19.48 | 30.58 |
| Synthetic4Relight | Roughness MSE↓ | **0.008** | 0.013 | 0.011 | 0.015 |
| TensoIR | Relight PSNR↑ | **29.907** | 27.367 | 24.374 | 28.580 |
| TensoIR | Albedo PSNR↑ | **33.796** | 26.199 | 30.286 | 29.275 |
| TensoIR | Normal MAE↓ | 4.112 | 5.927 | 4.948 | **4.100** |

### Ablation Study

| Configuration | NVS PSNR | Albedo PSNR | Relight PSNR | Description |
|------|----------|-------------|--------------|------|
| Full model | 35.48 | 30.81 | 34.68 | Full model |
| Detach indirect | 34.21 | 30.29 | 34.22 | Do not backpropagate indirect light gradients |
| w/o indirect (train) | 34.09 | 30.10 | 33.93 | Remove indirect light term during training |
| w/o indirect (relight) | - | - | 33.84 | No indirect light during relighting |
| $N_r=16$ | 34.01 | 30.21 | 29.46 | Too few sampled rays, relighting drops by 5.2dB |
| $N_r=64$ | 34.98 | 30.63 | 33.11 | 64 rays, relighting is still 1.6dB worse |

### Key Findings
- The number of sampled rays is crucial for relighting quality: increasing $N_r$ from 16 to 256 improves relighting PSNR by over 5dB.
- Differentiable indirect light is key: detaching the indirect light gradient leads to unrealistic indirect radiance estimation.
- The training time of IRGS (0.7h) is comparable to R3DG (0.9h) and significantly faster than NeRF-based methods (3-48h).
- R3DG performs better on NVS (36.80 vs 35.48), but this is because it performs shading on Gaussians, sacrificing relighting performance.

## Highlights & Insights
- **2D vs 3D Gaussian Ray Tracing**: 2D Gaussians have a well-defined ray-plane intersection, allowing direct ray tracing on pre-trained 2DGS with minimal quality degradation. This insight demonstrates the importance of representation consistency for hybrid rendering pipelines.
- **Trade-off between Pixel-Level Sub-Sampling and High Sample Counts**: By evaluating the rendering equation only on a random subset of pixels, high-quality estimation is achieved using 256 rays rather than using a small number of rays across all pixels. This strategy can be transferred to other computationally intensive rendering tasks.
- **Necessity of the Complete Rendering Equation**: Experiments demonstrate that simplifying assumptions (split-sum, learnable SH) indeed limit the accuracy of material and illumination estimation. Although the full rendering equation is computationally heavy, it can be resolved through engineering optimization.

## Limitations & Future Work
- Real-time rendering is not supported (approximately 1 second per frame), limiting practical applications.
- The authors mention that acceleration could be achieved by baking indirect radiance or pre-computing radiation transfer, but this has not been implemented.
- The material is assumed to be dielectric, which does not support accurate modeling of metallic materials.
- The method has only been validated on synthetic and simple real-world scenes; its applicability to complex real indoor scenes remains uncertain.

## Related Work & Insights
- **vs GS-IR**: GS-IR uses split-sum approximation + baked volume to store occlusion, whereas this work directly queries via ray tracing. IRGS significantly outperforms in relighting (34.90 vs 25.40 PSNR), demonstrating the advantage of the complete rendering equation.
- **vs R3DG**: R3DG shading is performed independently on each Gaussian and parameterizes indirect radiance with SH, while this work performs pixel-level shading and traces indirect light in real-time. R3DG achieves better NVS results but shows a significant gap in relighting.
- **vs 3DGRT**: 3DGRT proposes a Gaussian ray tracing framework but suffers from ray-plane intersection inconsistency on 3D Gaussians. This work addresses this critical issue by using 2D Gaussians instead.

## Rating
- Novelty: ⭐⭐⭐⭐ Integrating the complete rendering equation into Gaussian Splatting without simplification for the first time; 2DGRT is a significant technical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation studies across three datasets, though lacking validation on large-scale real-world scenes.
- Writing Quality: ⭐⭐⭐⭐ Clear equations and a logically structured motivation.
- Value: ⭐⭐⭐⭐ Provides a correct technical route for inverse rendering in Gaussian Splatting (complete rendering equation + ray tracing), which is expected to drive future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)
- [\[CVPR 2026\] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting](../../CVPR2026/3d_vision/stochastic_ray_tracing_for_the_reconstruction_of_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2026\] Geometric-Photometric Event-based 3D Gaussian Ray Tracing](../../CVPR2026/3d_vision/geometric-photometric_event-based_3d_gaussian_ray_tracing.md)
- [\[CVPR 2025\] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)

</div>

<!-- RELATED:END -->
