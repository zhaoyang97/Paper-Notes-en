---
title: >-
  [Paper Note] Hardware-Rasterized Ray-Based Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper presents VKRayGS, the first hardware-rasterized Ray-based 3D Gaussian Splatting (RayGS) rendering scheme. Through rigorous mathematical derivations, it constructs a minimum bounding quad in 3D space, achieving approximately a 40x rendering speedup while maintaining the high-quality rendering of RayGS, and additionally proposes a MIP anti-aliasing scheme for RayGS.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Hardware Rasterization"
  - "Ray Tracing"
  - "Novel View Synthesis"
  - "VR Rendering"
date: 2026-05-08
content_hash: ee3833c64a380b64
---

# Hardware-Rasterized Ray-Based Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.18682](https://arxiv.org/abs/2503.18682)  
**Code**: [https://github.com/facebookresearch/vkraygs](https://github.com/facebookresearch/vkraygs)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Hardware Rasterization, Ray Tracing, Novel View Synthesis, VR Rendering  

## TL;DR
This paper presents VKRayGS, the first hardware-rasterized Ray-based 3D Gaussian Splatting (RayGS) rendering scheme. Through rigorous mathematical derivations, it constructs a minimum bounding quad in 3D space, achieving approximately a 40x rendering speedup while maintaining the high-quality rendering of RayGS, and additionally proposes a MIP anti-aliasing scheme for RayGS.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3DGS) renders scenes via splatting, which is fast but suffers from projection approximation errors. RayGS eliminates this error through exact ray-Gaussian intersection calculation, achieving higher quality rendering.

2. **Limitations of Prior Work**: Although RayGS provides better quality, its computational overhead increases significantly. Current mainstream implementations (such as GOF) are based on CUDA software rasterization, resulting in framerates too low to meet the requirements of high-framerate applications like VR.

3. **Key Challenge**: The support of standard 3DGS is an ellipse in the image plane, which is easy to rasterize with hardware. However, the support of RayGS can be a half-hyperbola, making it impossible to wrap simply with a quad in the image plane.

4. **Goal**: How to map RayGS to the standard hardware rasterization pipeline (vertex shader + fragment shader) while maintaining rendering quality?

5. **Key Insight**: Bypass solving for the bounding quad in the image plane, and instead seek a quad in 3D space that bounds the RayGS primitive's support. It is proved that the set of maximum density points on the support boundary forms a 2D ellipse in 3D space, which is isomorphic to a unit circle.

6. **Core Idea**: By proving that the support boundary of a RayGS primitive forms an ellipse in 3D space, an isomorphic mapping to a unit circle is established. This allows for the efficient computation of the minimum bounding quad in 3D space, which is then mapped to the hardware rasterization pipeline.

## Method

### Overall Architecture
The input is a set of trained RayGS Gaussian primitives, and the output is the rendered image for a given viewpoint. The rendering pipeline reuses the standard Vulkan hardware rasterization architecture, with core modifications in the vertex shader (computing the 3D bounding quad vertices for each Gaussian primitive) and the fragment shader (computing the RayGS ray divergence/opacity for each pixel). The backend alpha blending logic is identical to standard GS hardware implementations.

### Key Designs

1. **3D Space Ellipse-to-Unit Circle Isomorphic Mapping (Vertex Shader Core)**:

    - **Function**: Computes the minimum-area bounding quad in 3D space for each Gaussian primitive.
    - **Mechanism**: For a given Gaussian primitive (center $\boldsymbol{\mu}$, covariance $\Sigma$), the points of maximum density along each ray on the RayGS support boundary form a set $\mathcal{E}$. An isomorphic mapping between $\mathcal{E}$ and the unit circle $\mathbb{S}_1$ is proved via three transformations: first applying $\mathbf{S}^{-1}\mathbf{R}_p^\top$ to normalize the ellipsoid to a unit sphere, then rotating to align the normalized center with the z-axis, and finally extracting and normalizing the x-y sub-vector to obtain points on the unit circle. The inverse mapping $\Phi^{-1}$ allows any 2D quad bounding the unit circle to be mapped back to 3D space. To minimize the area, the eigenvectors of the $2 \times 2$ matrix $\mathbf{B} = \mathbf{Q}_{0:2}^\top \mathbf{Q}_{0:2}$ are solved to find the primary packaging axes of the ellipse, constructing an axis-aligned, tightest bounding rectangle.
    - **Design Motivation**: Choosing the one with the minimum area out of infinitely many valid 3D quads minimizes the number of invalid pixels processed by the fragment shader. The eigendecomposition of a $2 \times 2$ matrix has a closed-form solution, making its computational overhead extremely small.

2. **RayGS Ray Divergence Calculation in Fragment Shader**:

    - **Function**: Efficiently computes the RayGS rendering opacity for each pixel utilizing hardware interpolation capabilities.
    - **Mechanism**: For any ray $\boldsymbol{x}$ passing through the quad, there exist interpolation coefficients $\boldsymbol{\alpha}$ such that the RayGS ray divergence can be represented as $\mathcal{D}_{\text{ray}} = \{c^{-2} + \|\mathbf{Z}_{\text{ray}}\boldsymbol{\alpha}\|^{-2}\}^{-1}$. Therefore, one only needs to specify $\mathbf{Z}_{\text{ray}}$ at the vertices, let the hardware perform automatic interpolation, and then perform a dot product and a simple scalar operation in the fragment shader. Compared to the fragment shader of standard GS (which only requires a dot product), the extra overhead is merely one inversion and one sum.
    - **Design Motivation**: By deriving a closed-form relationship, the complex RayGS ray divergence calculation is reduced to a linear interpolation automatically handled by the hardware plus a simple post-processing operation, maximizing GPU parallel performance.

3. **MIP Anti-Aliasing Scheme for RayGS**:

    - **Function**: Solves aliasing artifacts when training and testing resolutions are inconsistent.
    - **Mechanism**: For normalized rays, the 3D Gaussian distribution is marginalized along the plane orthogonal to the ray direction, yielding a 2D Gaussian. A smoothing step is performed using an isotropic 2D Gaussian corresponding to the pixel size, approximating the integration over the pixel area. In the final opacity calculation, the dilated covariance $\hat{\Sigma} = \Sigma + \sigma_x^2 \tau^2(\mathbf{x})\mathbf{I}$ is utilized, and an opacity modulation factor $\sqrt{|\Sigma|c^2 / (|\hat{\Sigma}|\hat{c}^2)}$ is introduced. To ensure efficient implementation, the pixel-dependent terms are approximated as constants.
    - **Design Motivation**: In VR applications, users can move freely, and the discrepancy between training and testing resolutions can be large. Not handling MIP leads to severe aliasing and flickering.

### Loss & Training
This is an inference-time (renderer) work that does not introduce new training losses. It directly renders utilizing pre-trained RayGS models from methods like GOF. The MIP scheme can also be used during training by modifying the opacity.

## Key Experimental Results

### Main Results

| Scene | Metric | VKRayGS | GOF (CUDA) | Speedup |
|------|------|---------|------------|--------|
| MipNeRF360 (9-scene avg) | FPS↑ | ~232 | ~5.2 | ~40× |
| bicycle | FPS↑ | 177 | 4 | 44× |
| bonsai | FPS↑ | 341 | 6 | 57× |
| MipNeRF360 Avg | PSNR↑ | ~27.2 | ~27.3 | -0.4% |
| MipNeRF360 Avg | LPIPS↓ | ~0.223 | ~0.237 | +5.9% |

### Ablation Study

| Configuration | FPS | PSNR | Description |
|---------|-----|------|------|
| GOF (CUDA RayGS) | ~5 | 27.3 | Baseline RayGS renderer |
| VKRayGS (Ours) | ~232 | 27.2 | 40× speedup with virtually no quality loss |
| VKGS (Standard GS Hardware) | Faster | Slightly Lower | Standard GS quality is inferior to RayGS |
| GS CUDA (INRIA) | Medium | Slightly Lower | CUDA vs Vulkan: Vulkan is approx. 2× faster |

### Key Findings
- An average 40x speedup is a massive breakthrough, directly elevating RayGS from 'unusable' to VR-grade real-time rendering.
- The quality degradation is minute (PSNR drops by approx. 0.1dB), and the LPIPS is even better in some scenes, indicating that the discrepancy stems from implementation details rather than the methodology itself.
- Real-time RayGS rendering can be achieved on a mid-range GPU like the RTX 2080 (>170 FPS in most scenes).
- The near-clipping plane needs to be disabled; otherwise, the clipped quads will yield visible discontinuities.

## Highlights & Insights
- **Elegance of Mathematical Derivation**: The derivation of the isomorphic mapping from 3D ellipses to the unit circle is rigorous with clear geometric intuition, serving as a textbook example of simplifying a complex geometric problem into a $2 \times 2$ matrix eigendecomposition.
- **Exceptional Engineering Value**: Achieving a 40x speedup while preserving quality directly unlocks the usage of RayGS in VR/MR scenarios, demonstrating outstanding practicality.
- **Theoretical Rigorousness of the MIP Scheme**: Compared to the heuristic approach of MIP-Splatting, this work derives its solution starting from the marginalization of 3D Gaussian distributions, which is theoretically more rigorous.

## Limitations & Future Work
- The current scheme requires disabling the near-clipping plane, which might cause rendering issues in certain extreme scenarios.
- It depends on the Vulkan API. Although theoretically portable to OpenGL, cross-platform compatibility still needs verification.
- In the MIP scheme, pixel-dependent terms are approximated as constants, which may degrade accuracy for extremely thin but long Gaussian primitives.
- Currently, it only handles the inference stage. Extending hardware-rasterized rendering to differentiable rendering for training would yield even greater value.

## Related Work & Insights
- **vs GOF (CUDA)**: Under the same RayGS model, the rendering speed is boosted by 40x, demonstrating the massive advantage of hardware rasterization over CUDA software rasterization.
- **vs VKGS**: VKGS is a hardware-rasterized implementation of standard GS. VKRayGS incorporates RayGS support on top of it, achieving superior quality at a similar speed.
- **vs Original 3DGS**: RayGS eliminates projection approximation errors, and combined with the fast renderer proposed in this work, achieves a win-win in both quality and speed.

## Rating
- Novelty: ⭐⭐⭐⭐ First hardware rasterization solution for RayGS with novel mathematical derivation
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of speed and quality on standard benchmarks
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and clear mathematical derivation with intuitive illustrations
- Value: ⭐⭐⭐⭐⭐ 40x speedup directly unlocks VR application scenarios, possessing extremely high engineering value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[CVPR 2026\] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting](../../CVPR2026/3d_vision/stochastic_ray_tracing_for_the_reconstruction_of_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Geometric-Photometric Event-based 3D Gaussian Ray Tracing](../../CVPR2026/3d_vision/geometric-photometric_event-based_3d_gaussian_ray_tracing.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)

</div>

<!-- RELATED:END -->
