---
title: >-
  [Paper Note] 3D-HGS: 3D Half-Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][Half-Gaussian] This work proposes the 3D Half-Gaussian (3D-HGS) reconstruction kernel, which splits a 3D Gaussian into two halves using a cutting plane, each having independent opacity. Acting as a **plug-and-play** reconstruction kernel to replace standard Gaussian kernels, it significantly enhances rendering quality at shape and color discontinuities without sacrificing rendering speed, outperforming all SOTA methods on Mip-NeRF360, Tanks & Temples…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Half-Gaussian"
  - "Reconstruction Kernel"
  - "plug-and-play"
  - "Discontinuity"
  - "novel view synthesis"
date: 2026-05-08
content_hash: 53836e3e338dfd56
---

# 3D-HGS: 3D Half-Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2406.02720](https://arxiv.org/abs/2406.02720)  
**Code**: [https://lihaolin88.github.io/CVPR-2025-3DHGS](https://lihaolin88.github.io/CVPR-2025-3DHGS)  
**Area**: 3D Vision / Novel View Synthesis / 3D Gaussian Splatting  
**Keywords**: Half-Gaussian, Reconstruction Kernel, plug-and-play, Discontinuity, novel view synthesis

## TL;DR
This work proposes the 3D Half-Gaussian (3D-HGS) reconstruction kernel, which splits a 3D Gaussian into two halves using a cutting plane, each having independent opacity. Acting as a **plug-and-play** reconstruction kernel to replace standard Gaussian kernels, it significantly enhances rendering quality at shape and color discontinuities without sacrificing rendering speed, outperforming all SOTA methods on Mip-NeRF360, Tanks & Temples, and Deep Blending.

## Background & Motivation
3D-GS uses 3D Gaussian kernels as primitive representatives for scene reconstruction. Standard Gaussian functions are smooth low-pass filters, making them inherently deficient in modeling discontinuous functions (such as object boundaries or texture edges)—fitting step functions leads to Gibbs oscillations (overshoot and undershoot) and blurred edges. This manifests as artifacts at object boundaries and texture-rich regions. From a frequency domain perspective, standard Gaussians have limited bandwidth and lack high-frequency components, making it difficult to capture sharp transitions accurately.

## Core Problem
Due to the use of symmetric Gaussian kernels, 3D-GS cannot efficiently model the shape and color discontinuities common in real-world scenes. How can a stronger reconstruction kernel be designed to capture these discontinuities while maintaining the simplicity and rendering speed of 3D-GS, enabling plug-and-play enhancement for all existing 3D-GS variants?

## Method

### Overall Architecture
The core idea is extremely simple: a 3D Gaussian is split into two halves by a plane passing through its center. Each half has its own independent opacity parameter $\alpha_1, \alpha_2$. The newly added parameters are only the plane normal vector $\mathbf{n}$ (3 parameters, leveraging the unused normal field in 3D-GS) and 1 extra opacity value. In total, only 1 additional parameter is introduced.

### Key Designs
1. **3D Half-Gaussian Kernel Definition**:
$$HG_\Sigma(\mathbf{x}-\mu) = \begin{cases} e^{-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)} & \mathbf{n}^T(\mathbf{x}-\mu) \geq 0 \\ 0 & \mathbf{n}^T(\mathbf{x}-\mu) < 0 \end{cases}$$
A pair of complementary half-Gaussians share the same mean, rotation, scaling, and color, but each has independent opacity. When $\alpha_1=\alpha_2$, it degenerates to a standard Gaussian. Key property: Half-Gaussians have higher bandwidth in the frequency domain than full Gaussians, allowing them to capture high-frequency components better.

2. **Closed-form Integration Projection**: Integrating the 3D half-Gaussian along the $z$-axis yields a closed-form solution: $\int HG \, dz = \frac{1}{2} I(x,y) \hat{G}_{\hat\Sigma}(\hat{\mathbf{x}}-\hat\mu)$, where $I(x,y)$ is a scaling factor related to the normal vector given by the complementary error function (erfc). This ensures differentiable rendering and efficient computation.

3. **Efficient Half-Gaussian Rasterizer**: Approximately 75% of the half-Gaussian kernels have one half completely transparent (with an opacity difference $> 0.5$). Naively processing both halves would waste computation. Therefore, an independent rasterization scheme is designed to compute only the valid region of each half—determining the inner boundary via the projected cutting plane and the outer boundary of the Gaussian to establish a bounding box, splatting only the valid region. Consequently, the FPS is slightly better than standard 3D-GS.

4. **Plug-and-Play**: Only the forward and backward propagation of the rasterizer need to be modified, with no changes required for the training pipeline, loss functions, or densification strategies. It has been validated as a plug-and-play solution on four methods: 3D-GS, Scaffold-GS, Mip-Splatting, and GS-MCMC.

### Loss & Training
- Same loss function as the original 3D-GS (L1 + SSIM)
- Learning rate of 0.003 for normal vectors (0.3 for Deep Blending)
- Learning rate of opacity and normal vectors decays by $1.4\times$ every 5000 steps
- Opacity threshold increased to 0.01 to remove noise

## Key Experimental Results

### Mip-NeRF 360 Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3D-GS | 28.04 | 0.844 | 0.214 |
| **3D-HGS** | **28.82** (+0.78) | 0.860 | 0.197 |
| Scaffold-GS | 28.88 | 0.866 | 0.186 |
| **Scaffold-HGS** | **29.18** (+0.30) | 0.873 | 0.175 |
| Mip-Splatting | 28.34 | 0.852 | 0.199 |
| **Mip-HGS** | **28.83** (+0.49) | 0.863 | 0.185 |
| GS-MCMC | 28.98 | 0.868 | 0.180 |
| **HGS-MCMC** | **29.22** (+0.24) | 0.874 | 0.170 |

### Tanks & Temples

| Method | PSNR↑ |
|------|-------|
| 3D-GS | 23.60 |
| **3D-HGS** | **24.45** (+0.85) |
| **HGS-MCMC** | **25.21** (SOTA) |

### Rendering Speed and Memory (Mip-NeRF 360)

| Method | FPS↑ | Memory (MB)↓ |
|------|------|----------|
| 3D-GS | 115 | 762 |
| 3D-HGS | **125** | **694** |

Due to the efficient rasterizer and fewer kernels, FPS actually increases by around 8%, and memory is reduced by 9%.

### Kernel Comparison (Same training settings, based on 3D-GS)

| Kernel Function | Average PSNR |
|--------|---------|
| 3D Gaussian | 28.04 |
| 2D Gaussian (2D-GS) | 28.06 |
| GES | 27.62 |
| **Half-Gaussian** | **28.82** |

The half-Gaussian is the only kernel function that outperforms the standard Gaussian across all 11 scenes.

### Ablation Study
- **Efficient Rasterizer**: Without optimization, the FPS is only 76. With optimization, the FPS reaches 125, while PSNR is almost unaffected.
- **Cutting Plane Visualization**: Normal vectors naturally align with the scene's surface normals, confirming that half-Gaussians model object surface boundaries.
- **Opacity Distribution**: Over 75% of the kernels exhibit a normalized opacity difference of $>0.5$, indicating that most kernels indeed utilize the unique properties of the two halves.

## Highlights & Insights
- **Elegant Mathematical Intuition**: Explains why half-Gaussians are better suited for modeling discontinuities than full Gaussians from the perspective of Gibbs phenomenon and frequency-domain bandwidth—half-Gaussians have higher bandwidth in the frequency domain.
- **Minimal Parameter Increase**: Only 1 extra parameter (additional opacity) is added per kernel. The 3 normal vector parameters utilize the existing but unused `normal` field in 3D-GS.
- **True Plug-and-Play**: Consistently improves PSNR by 0.13-0.85 dB across 4 different 3D-GS methods without changing training strategies.
- **Faster and Smaller**: Thanks to efficient rasterization and a smaller number of kernels, both FPS and memory usage are superior to the original 3D-GS.

## Limitations & Future Work
- The cutting plane is assumed to pass through the center and be flat—curved surfaces might require non-planar division.
- Only opacity differs between the two halves; allowing colors to differ as well might yield further improvements.
- The interaction with learning-based densification strategies has not been explored.
- Current evaluation is limited to static scenes; the effectiveness on dynamic scenes remains unknown.

## Related Work & Insights
- **2D-GS**: Uses 2D Gaussians for surface rendering. While it improves performance in some scenes, it drops in others. 3D-HGS consistently improves across all scenes.
- **GES**: Uses generalized exponential kernels without modifying the rasterizer, leading to poor performance in complex scenes. 3D-HGS modifies the rasterizer and provides a closed-form solution.
- **FreGS / Mip-Splatting / Scaffold-GS**: These methods focus on anti-aliasing or structured representation and are complementary to 3D-HGS—3D-HGS kernels can be directly plugged into these methods to yield additional gains.

## Related Work & Insights
- "Changing the kernel function" is a new route for 3D-GS improvement—no network changes, no training pipeline changes, no loss function changes, only modifying the reconstruction primitive.
- The idea of half-Gaussians can be generalized to other kernel-based methods (such as kernel density estimation, SPH fluid simulation).
- The cutting plane normal naturally aligns with the surface normal, which can be useful for normal estimation or surface reconstruction.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Extremely elegant idea—using half-Gaussians to solve discontinuity modeling, supported by rigorous mathematical derivation. The concept of Gaussian splitting is highly intuitive yet previously unexplored.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 11 scenes + 4 baselines + kernel function comparison + speed analysis + visualization. The main limitation is the lack of other datasets.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation (Gibbs/frequency domain), rigorous mathematical derivation, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Promotes plug-and-play performance improvements across all 3D-GS methods, offering extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
