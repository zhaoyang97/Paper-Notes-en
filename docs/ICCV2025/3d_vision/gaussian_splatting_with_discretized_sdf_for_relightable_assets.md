---
title: >-
  [Paper Note] Gaussian Splatting with Discretized SDF for Relightable Assets
description: >-
  [ICCV 2025][3D Vision][Inverse Rendering] This paper proposes encoding a continuous SDF as an additional per-Gaussian attribute via an SDF-to-opacity transformation that unifies Gaussian splatting and SDF representations. Combined with a projection-based consistency loss and spherical initialization, the method achieves relighting quality surpassing existing Gaussian-based inverse rendering approaches using only 4 GB of GPU memory.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Inverse Rendering"
  - "3D Gaussian Splatting"
  - "Discretized SDF"
  - "Geometry Regularization"
  - "Relighting"
date: 2026-05-08
content_hash: 67abf5e0bf0e24bd
---

# Gaussian Splatting with Discretized SDF for Relightable Assets

**Conference**: ICCV 2025
**arXiv**: [2507.15629](https://arxiv.org/abs/2507.15629)  
**Code**: [https://github.com/NK-CS-ZZL/DiscretizedSDF](https://github.com/NK-CS-ZZL/DiscretizedSDF)  
**Area**: 3D Vision
**Keywords**: Inverse Rendering, 3D Gaussian Splatting, Discretized SDF, Geometry Regularization, Relighting

## TL;DR

This paper proposes encoding a continuous SDF as an additional per-Gaussian attribute via an SDF-to-opacity transformation that unifies Gaussian splatting and SDF representations. Combined with a projection-based consistency loss and spherical initialization, the method achieves relighting quality surpassing existing Gaussian-based inverse rendering approaches using only 4 GB of GPU memory.

## Background & Motivation

Decomposing geometry, material, and illumination from multi-view RGB images (inverse rendering) is a classical and important problem with significant downstream applications in augmented and virtual reality. Geometry quality is a prerequisite for material and illumination estimation — only sufficiently accurate normals and surfaces enable reasonable BRDF decomposition.

**Strengths of SDF and its success in the NeRF era**: SDF-based geometry priors have demonstrated effectiveness in NeRF frameworks such as NeuS. Methods like TensoSDF and NeRO enable robust geometry–material decomposition for diverse materials, including highly reflective objects, at the cost of expensive ray marching and slow training and rendering speeds.

**Challenges of integrating 3DGS into inverse rendering**: 3D Gaussian Splatting (3DGS) has become a dominant representation due to real-time rendering and fast training. However, applying inverse rendering within the GS framework suffers from insufficient geometric constraints. Early methods such as GShader and GS-IR rely only on weak constraints like normal alignment, resulting in limited decomposition quality.

**Costs of existing solutions**: To compensate for weak geometric constraints, methods such as GS-ROR and GSDF introduce additional continuous SDF networks jointly optimized with Gaussians. While this improves geometry quality, it introduces three problems: (1) substantially increased memory usage (GS-ROR requires 22 GB vs. 4 GB for the proposed method); (2) complex optimization strategies (warm-up, multi-stage training) to balance the two representations; and (3) increased model complexity.

**Core insight of this paper**: Can the SDF be encoded directly into the Gaussian primitives themselves, eliminating the need for an additional network and achieving a unified representation that combines GS flexibility with SDF robustness? Concretely, each Gaussian is assigned an SDF value attribute; an SDF-to-opacity transformation links the SDF to the Gaussian opacity; and the SDF is rendered via splatting.

## Method

### Overall Architecture

The method builds a deferred shading inverse rendering pipeline on top of 2DGS (2D Gaussian Splatting). Each 2D Gaussian disk encodes standard PBR attributes (albedo, roughness, etc.), scale, and rotation, along with an additional SDF value $s_i$. Opacity is not learned directly but is derived via the SDF-to-opacity transformation. A Principled BRDF with split-sum approximation is adopted as the shading model.

### Key Designs

1. **Discretized SDF and SDF-to-Opacity Transformation**:

    - Function: Encodes an SDF sample value on each Gaussian primitive and maps it to opacity via a transformation function.
    - Mechanism: The SDF-to-opacity transformation takes the form of a bell-shaped function:
    $o_i = \mathcal{T}_\gamma(s_i) = \frac{4 \cdot e^{-\gamma s_i}}{(1 + e^{-\gamma s_i})^2}$
   where $s_i$ is the SDF value of the $i$-th Gaussian and $\gamma$ is a globally learnable parameter controlling the variance of the transformation. Opacity is maximal (= 1) when the SDF value is zero and decreases as the Gaussian moves away from the surface.
    - Design Motivation: This transformation seamlessly connects SDF and Gaussian rendering — the SDF defines surface position (zero level set), and the resulting opacity distribution naturally encourages Gaussians to cluster near the surface. No additional network is required; SDF information is encoded entirely within the Gaussian primitives.

2. **Median Loss**:

    - Function: Addresses convergence difficulties in jointly optimizing SDF values $s_i$ and the transformation parameter $\gamma$.
    - Mechanism: The median of the unsigned distances $|s|_m$ across all Gaussians is identified as a reliable convergence indicator. When the opacity at the median distance is too high, the transformation must be narrowed. A reference value $\gamma_m$ is derived such that $\mathcal{T}_{\gamma_m}(|s|_m) = 0.5$:
    $\gamma_m = -\frac{\log(3 - 2\sqrt{2})}{|s|_m}$
   The median loss then guides $\gamma$ to quickly approach $\gamma_m$: $\mathcal{L}_\gamma = \max(\gamma_m - \gamma, 0)$
    - Design Motivation: Without this loss, the joint search space for $\gamma$ and $s_i$ is too large and convergence is extremely slow. The median provides global statistical information to anchor the transformation parameter.

3. **Projection-based Consistency Loss**:

    - Function: Regularizes the discretized SDF in lieu of the inapplicable Eikonal loss.
    - Core Problem: Continuous SDFs can be regularized by the Eikonal loss ($|\nabla f|=1$), but a discretized SDF only provides gradient direction (i.e., normals) without gradient magnitude, rendering the classical Eikonal loss inapplicable.
    - Mechanism: Each Gaussian is projected onto the zero level set of the SDF to obtain a projection point $\mu_{proj} = \mu_i - s_i \frac{\nabla f_i}{|\nabla f_i|}$, and the depth of this projection point is required to be consistent with the alpha-blended surface depth:
    $\mathcal{L}_p = \frac{1}{N}\sum_{i \in N} \begin{cases} \epsilon_{proj}^i, & \epsilon_{proj}^i \leq \varepsilon \\ 0, & \epsilon_{proj}^i > \varepsilon \end{cases}$
   A threshold $\varepsilon$ filters anomalous Gaussians caused by occluded surfaces or self-intersections.
    - Design Motivation: The authors prove that this loss approximates the Eikonal condition, providing effective regularization for the discretized SDF. It is activated after 1K iterations, once a coarse geometry has stabilized.

4. **Spherical Initialization**:

    - Function: Initializes foreground Gaussians with points uniformly sampled on a unit sphere.
    - Mechanism: Draws on the established success of spherical initialization in volumetric SDF rendering methods such as NeuS.
    - Design Motivation: Prevents incorrect early-stage geometry from becoming trapped in local optima. Standard initialization frequently leads to broken surfaces in early training that are difficult to recover from.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_c + \lambda_n\mathcal{L}_n + \lambda_d\mathcal{L}_d + \lambda_\gamma\mathcal{L}_\gamma + \lambda_p\mathcal{L}_p + \lambda_{sm}\mathcal{L}_{sm} + \lambda_m\mathcal{L}_m$

Individual terms: $\mathcal{L}_c$ (rendering loss), $\mathcal{L}_n$ (normal consistency), and $\mathcal{L}_d$ (distortion) are inherited from 2DGS; $\mathcal{L}_\gamma$ (median loss) and $\mathcal{L}_p$ (projection-based consistency loss) are proposed in this work; $\mathcal{L}_{sm}$ (PBR attribute smoothness regularization) and $\mathcal{L}_m$ (optional mask loss). The model is trained for 30K iterations using the Adam optimizer.

## Key Experimental Results

### Main Results (Glossy Blender Relighting)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Training Time | Memory |
|--------|-------|-------|--------|---------------|--------|
| GShader | 16.95 | 0.8430 | 0.2365 | 0.5h | 4G |
| GS-IR | 16.26 | 0.6494 | 0.1361 | 0.5h | 8G |
| R3DG | 17.09 | 0.8258 | 0.1260 | 1h | 20G |
| GS-ROR | 23.39 | 0.9140 | 0.0769 | 1.5h | 22G |
| **Ours** | **24.52** | **0.9229** | **0.0762** | **1h** | **4G** |

### Ablation Study (Glossy Blender, 3 Scenes)

| Configuration | Angel PSNR/SSIM | Horse PSNR/SSIM | Teapot PSNR/SSIM |
|---------------|----------------|-----------------|------------------|
| Baseline (no SDF) | 20.23/0.8533 | 20.72/0.8998 | 19.22/0.8674 |
| + Discretized SDF + Median Loss | 21.09/0.8815 | 22.13/0.9120 | 23.26/0.9223 |
| + Spherical Init. (alone) | 21.43/0.8881 | 21.96/0.9102 | 22.97/0.9176 |
| + SDF + Spherical Init. | 21.59/0.8910 | 22.87/0.9284 | 23.45/0.9222 |
| + SDF + Projection Loss | 21.65/0.8917 | 23.21/0.9395 | 24.05/0.9289 |
| + SDF + Projection + Spherical (Full) | **22.03/0.8919** | **24.01/0.9481** | **24.19/0.9293** |

### Key Findings

- PSNR on Glossy Blender exceeds GS-ROR by 1.13 dB (24.52 vs. 23.39) while requiring only 4 GB of memory (vs. 22 GB for GS-ROR, a 5× reduction).
- The proposed method also achieves state-of-the-art results on the TensoIR synthetic dataset: average PSNR of 27.78 vs. 27.07 for GS-ROR.
- Chamfer Distance on Glossy Blender averages 0.0107, substantially outperforming GS-ROR (0.0140) and R3DG (0.0315).
- Normal MAE on Shiny Blender averages 6.48°, better than GS-ROR (7.23°).
- Incrementally adding each of the three components (discretized SDF, projection loss, spherical initialization) yields consistent improvements, validating the necessity of each design.

## Highlights & Insights

- **Elegance of unified representation**: Without introducing any additional network or representation, the SDF is encoded directly as a Gaussian attribute, achieving GS rendering efficiency and SDF geometric robustness within a single representation.
- **Theoretical contribution**: The projection-based consistency loss is proven to approximate the Eikonal condition, providing a theoretical foundation for regularizing discretized SDFs.
- **Exceptional efficiency**: 4 GB memory and 1 hour of training suffice to outperform GS-ROR, which requires 22 GB and 1.5 hours — a practically significant result.
- **Clever use of median statistics**: Global statistical information (the median) is used to guide a local parameter (transformation width), resolving the anchoring problem in joint optimization.

## Limitations & Future Work

- Only direct illumination is considered; the method may fail in scenes with complex occlusion and indirect lighting, necessitating the addition of an indirect illumination term.
- In scenes with complex inter-reflections (e.g., Toaster, Luyu), performance still falls short of GS-ROR, which incorporates a full SDF network.
- The current focus is on object-level relighting; extension to unbounded scenes has not been explored.
- Mesh extraction and BRDF parameter export are not addressed.

## Related Work & Insights

- **NeuS / 3DGSR**: Pioneering works incorporating SDF into volumetric rendering and Gaussian frameworks; the SDF-to-opacity transformation in this paper draws inspiration from these approaches.
- **Neural-Pull**: Shares a theoretical basis with the projection-based consistency loss through the concept of projecting onto the zero level set.
- **GS-ROR**: The current state-of-the-art in Gaussian-based inverse rendering, introducing an additional SDF network; the primary baseline for this work.
- Insight: The paradigm of replacing "additional networks" with "attribute encoding" is generalizable to other GS methods that require supplementary geometric priors.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Encoding discretized SDF as a Gaussian attribute is a highly original idea)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 synthetic datasets + 1 real dataset + extensive ablations + multiple quantitative metrics)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous theoretical derivations and polished figures)
- Value: ⭐⭐⭐⭐⭐ (5× memory reduction with improved performance; strong practical significance)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](../../CVPR2025/3d_vision/hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[CVPR 2025\] RNG: Relightable Neural Gaussians](../../CVPR2025/3d_vision/rng_relightable_neural_gaussians.md)
- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](../../CVPR2025/3d_vision/svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)

</div>

<!-- RELATED:END -->
