---
title: >-
  [Paper Note] Flash Cache: Reducing Bias in Radiance Cache Based Inverse Rendering
description: >-
  [ECCV 2024][3D Vision][Inverse Rendering] Proposes an unbiased radiance cache-based inverse rendering method. By utilizing occlusion-aware vMF importance sampling and quick cache control variates, it eliminates rendering bias present in existing methods while maintaining computational efficiency, thereby improving the quality of material and illumination decomposition.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Inverse Rendering"
  - "Radiance Cache"
  - "Variance Reduction"
  - "Control Variates"
  - "Importance Sampling"
date: 2026-05-08
content_hash: e4776be85611fac8
---

# Flash Cache: Reducing Bias in Radiance Cache Based Inverse Rendering

**Conference**: ECCV 2024  
**arXiv**: [2409.05867](https://arxiv.org/abs/2409.05867)  
**Code**: [https://benattal.github.io/flash-cache/](https://benattal.github.io/flash-cache/) (project page)  
**Area**: 3D Vision  
**Keywords**: Inverse Rendering, Radiance Cache, Variance Reduction, Control Variates, Importance Sampling

## TL;DR

Proposes an unbiased radiance cache-based inverse rendering method. By utilizing occlusion-aware vMF importance sampling and quick cache control variates, it eliminates rendering bias present in existing methods while maintaining computational efficiency, thereby improving the quality of material and illumination decomposition.

## Background & Motivation

**Background**: Volume rendering (NeRF/3D-GS) excels in novel view synthesis and 3D reconstruction, but faces significant computational challenges when applied to inverse rendering (decomposing geometry, material, and illumination) — physically accurate global illumination rendering requires recursive path tracing.

**Limitations of Prior Work**: Existing radiance cache-based inverse rendering methods are divided into two categories, both of which introduce bias:
   - **NeRF cache-based** (e.g., TensoIR): Uses NeRF volume rendering to estimate incident light, which is of high quality but expensive, and only computes the reflection integral at the expected ray termination point $\rightarrow$ introduces volume rendering bias;
   - **Lightweight MLP cache-based** (e.g., InvRender/NeRO): Fast for single queries but has low capacity, failing to capture high-frequency and near-field illumination $\rightarrow$ introduces reflection integral bias.

**Key Challenge**: The conflict between computational efficiency and rendering unbiasedness — accurate global illumination requires massive recursive sampling, but biased gradients during optimization lead to degraded quality in material and illumination decomposition.

**Goal**: To eliminate bias in the rendering process as much as possible while maintaining a reasonable computational overhead.

**Key Insight**: Rather than striving to eliminate all sources of bias, this work leverages variance reduction techniques (importance sampling + control variate) to make an unbiased estimator feasible within a practical computational budget.

**Core Idea**: Uses a fast cache as a control variate for the NeRF cache and an occlusion-aware vMF distribution for importance sampling, efficiently estimating the rendering equation without introducing bias.

## Method

### Overall Architecture

The system is based on volume rendering combined with physical Monte Carlo rendering. For each camera ray: (1) an unbiased estimator of the volume rendering integral (sampling $K$ points based on a categorical distribution instead of summing over $N$ points) is used to select a few sample points; (2) at each sample point, importance sampling of incident light directions is performed using an occlusion-aware mix of vMF distributions; (3) a fast cache is used to evaluate a large number of cheap incident radiance samples, and a NeRF cache is used to evaluate a small number of accurate residual correction samples, which are combined via a control variate scheme to obtain an unbiased estimate of outgoing radiance.

### Key Designs

1. **Dual Cache + Control Variate**:

    - **Function**: Designs a fast but approximate radiance cache and uses it as a control variate for the high-quality NeRF cache, achieving unbiased and efficient incident light estimation.
    - **Mechanism**: The fast cache outputs $S = 8$ sample distances and weights via a sampling function $g_{\text{sample}}(\mathbf{x}, \boldsymbol{\omega}_i)$, queries low-resolution NGP features at these locations, and decodes the incident radiance via an MLP: $\hat{L}_i^{fast} = \text{MLP}_{\text{color}}(\sum_{k=1}^{S} w'_k \mathbf{f}_k)$. A control variate scheme is employed to combine the two caches:
    $$\hat{L}_o = \hat{L}_o^{fast} + \Delta\hat{L}_o$$
   where $\hat{L}_o^{fast}$ evaluates the fast cache with $M'$ samples, and $\Delta\hat{L}_o$ evaluates the difference between the NeRF cache and the fast cache with $M \ll M'$ samples. This requires only $M$ expensive NeRF cache queries to achieve a quality close to using $M'$ queries.
    - **Design Motivation**: If the fast cache is sufficiently accurate (close to the NeRF cache), the variance of the residual term will be very small, allowing accurate estimation with only a few samples. This is superior to directly using a low-capacity MLP cache, which introduces bias.

2. **Occlusion-Aware vMF Importance Sampler**:

    - **Function**: Learns a spatially varying incident light distribution to efficiently sample incident directions, reducing the variance of Monte Carlo estimation.
    - **Mechanism**: Uses an NGP to map each spatial point $\mathbf{x}$ to the mixture parameters of $L$ vMF distributions:
    $$q(\boldsymbol{\omega}; \mathbf{x}) = \frac{1}{Z} \sum_{\ell=1}^{L} \lambda_\ell(\mathbf{x}) \text{vMF}(\boldsymbol{\omega}; \boldsymbol{\mu}_\ell(\mathbf{x}), \kappa_\ell(\mathbf{x}))$$
   where the mean direction is parameterized as $\boldsymbol{\mu}_\ell(\mathbf{x}) = (\boldsymbol{\mu}'_\ell(\mathbf{x}) - \mathbf{x}) / \|\boldsymbol{\mu}'_\ell(\mathbf{x}) - \mathbf{x}\|$, meaning each vMF lobe points in the projected direction of a 3D light source location. It is optimized via a loss function $\mathcal{L}_{\text{vMF}}$ to align the distribution with the true incident radiance distribution (including occlusion).
    - **Design Motivation**: Unlike only modeling 3D locations of light sources (such as the concurrent work by Ling et al.), the spatially varying parameters of this method can adaptively "turn off" corresponding light source lobes at occluded positions, thereby significantly reducing sampling variance in occluded regions. It uses 128 vMF lobes.

3. **Efficient Volume Rendering Estimator**:

    - **Function**: Reduces the number of points along the camera ray that require physical rendering evaluation, from $N$ to $K$ (where $K=1$ is sufficient).
    - **Mechanism**: Samples $K$ indices from the rendering weight distribution $\{w_k\}$ according to a categorical distribution: $j_1, \ldots, j_K \sim \text{Cat}(w_1, \ldots, w_N)$, and then only evaluates the physical appearance at these $K$ points: $\hat{L}_i(\mathbf{o}, \boldsymbol{\omega}_o) = \frac{1}{K} \sum_{k=1}^{K} L_o(\mathbf{x}(t_{j_k}), \boldsymbol{\omega}_o)$. This is an unbiased estimator.
    - **Design Motivation**: When the weight distribution has a single peak (which is common in practice), the variance is very low even with $K=1$. While $N$ points are still needed to evaluate density, only $K$ points are required to evaluate the expensive physical appearance model.

4. **Material Representation**:

    - **Function**: Represents spatially varying material properties.
    - **Mechanism**: Uses the Disney-GGX BRDF model, outputting metallic $m$, roughness $r$, and albedo $\mathbf{a}$ via an NGP network $g_{\text{material}}(\mathbf{x})$. Normals are a combination of the negative gradient of the density field (analytical normals) and prediction from an NGP.
    - **Design Motivation**: A standard physical BRDF model, consistent with Ref-NeRF.

### Loss & Training

Total regularization loss:
$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{normals}} + \mathcal{L}_{\text{BRDF}} + \mathcal{L}_{\text{consistency}} + \mathcal{L}_{\text{interlevel}} + \mathcal{L}_{\text{density}}$$

- **Two-stage optimization**: (1) First optimize the NeRF cache and initial geometry to match the input images; (2) Optimize the geometry, material, fast cache, and importance sampler.
- The second stage uses $L_2$ loss combined with a gradient trick to ensure unbiased inverse rendering gradients.
- $\mathcal{L}_{\text{consistency}}$: Encourages the specular/diffuse colors of the NeRF cache to align with specular/diffuse lobes of physical rendering.
- $\mathcal{L}_{\text{BRDF}}$: Smoothness regularization for BRDF parameters ($L_1$ variant).
- Employs the same secondary rays to simultaneously supervise both the fast cache and the importance sampler, avoiding wasted computation.

## Key Experimental Results

### Main Results: TensoIR-Synthetic Dataset

| Method | Normal MAE ↓ | Albedo PSNR ↑ | NVS PSNR ↑ | Relight PSNR ↑ |
|------|------------|-------------|-----------|--------------|
| NeRFactor | 6.314 | 25.125 | 24.679 | 23.383 |
| InvRender | 5.074 | 27.341 | 27.367 | 23.973 |
| TensoIR | 4.100 | 29.275 | 35.088 | 28.580 |
| **Ours** | **3.355** | **30.274** | 34.908 | **29.724** |

### Ablation Study

| Configuration | NVS PSNR ↑ | Albedo PSNR ↑ | MAE ↓ | Description |
|------|----------|-------------|------|------|
| Full Method | 34.915 | 30.345 | 3.355 | All components |
| W/o Control Variates | 34.629 | 30.329 | 3.352 | Slight drop only in NVS |
| W/o Control Variates + w/o vMF | 34.653 | 30.237 | 3.352 | Albedo degraded |
| W/o Control Variates + w/o vMF + w/o Estimator | 34.249 | 29.838 | 3.491 | Normals significantly degraded |

### Real Data: Open Illumination

| Method | NVS PSNR (Diffuse) | NVS PSNR (Specular) | Relight (Diffuse) | Relight (Specular) |
|------|----------------|----------------|--------------|--------------|
| TensoIR | 32.043 | 27.115 | 30.932 | 26.955 |
| **Ours** | 31.781 | **27.180** | 29.679 | **27.810** |

### Key Findings

- Normal accuracy is optimal (MAE 3.355 vs. TensoIR 4.100), significantly outperforming all baselines.
- Albedo quality is optimal (PSNR 30.274 vs. 29.275), performing better in separating indirect illumination from albedo (e.g., the edge of the plate in the Hotdog scene).
- Relighting quality is optimal (29.724 vs. 28.580), validating more accurate material decomposition.
- Demonstrates a clear advantage over TensoIR on specular/glossy materials, while TensoIR is slightly better on diffuse materials.
- Visual comparison of 16spp rendering clearly demonstrates the respective contributions of vMF sampling and the fast cache to noise reduction.

## Highlights & Insights

- **The application of control variates** is highly elegant — using a cheap fast cache to handle most of the incident light estimation while the NeRF cache is only responsible for correcting the residual, mathematically guaranteeing unbiasedness. This represents an excellent application of classical Monte Carlo variance reduction techniques in neural rendering.
- **Occlusion-aware importance sampling** addresses a neglected problem — light source occlusion varies across different surface points, and the spatially varying vMF distribution can adaptively accommodate this variation.
- The design of the fast cache is clever — directly outputting sampling distances and weights (bypassing the proposal sampling pipeline), requiring only 8 sample points to capture high-frequency near-field illumination.
- Categorical distribution sampling of the volume rendering integral is a simple and effective technique ($K=1$ is sufficient); although previously proposed by Gupta et al., it is well integrated here.
- The entire system is a systematic "bias elimination" pipeline, with each component addressing a specific source of bias.

## Limitations & Future Work

- The quadrature approximation of the volume rendering integral (Eq. 3) remains a source of bias, which could be combined with unbiased volume rendering methods (e.g., differential ratio tracking).
- The fast cache sometimes struggles to capture extremely fine structures of near-field illumination; although the control variate prevents bias, it increases variance.
- Does not claim to be more efficient than existing methods — rather, it provides tools to make an unbiased system runnable on commercial GPUs.
- Can be integrated with denoisers to further reduce the number of secondary ray samples.
- Shows no significant advantage over TensoIR in diffuse-only scenes, as the benefits of bias elimination are mainly reflected in specular and complex light transport scenarios.

## Related Work & Insights

- **vs TensoIR**: TensoIR uses NeRF as a radiance cache but only evaluates it at a single point (the expected ray termination point) $\rightarrow$ volume rendering bias. Flash Cache maintains unbiasedness through categorical distribution sampling, and its quality is significantly better on specular scenes.
- **vs InvRender/NeRO**: These methods use a low-capacity MLP cache and cannot capture high-frequency lighting $\rightarrow$ reflection integral bias. Although Flash Cache's fast cache is also approximate, it avoids bias through the control variates scheme.
- **vs Ling et al. (Concurrent Work)**: Also uses importance sampling, but their Gaussian Mixture Model does not consider occlusion. Flash Cache's spatially varying vMF distribution is occlusion-aware, which is key to variance reduction.
- **Insight**: The variance reduction paradigm of combining control variates and importance sampling can be generalized to other neural rendering tasks requiring Monte Carlo estimation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Systematically integrates classical Monte Carlo variance reduction techniques into neural field inverse rendering, with clear theoretical support for each technical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on both synthetic and real datasets with comprehensive ablations, though the number of real scenes is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear, with a seamless unbiasedness analysis from volume rendering to Monte Carlo rendering, supported by intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides a theoretically more rigorous framework for the inverse rendering community, showing practical improvements in specular material decomposition, though its practicality is constrained by the overall computational overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)
- [\[CVPR 2026\] STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction](../../CVPR2026/3d_vision/stac_plug-and-play_spatio-temporal_aware_cache_compression_for_streaming_3d_reco.md)
- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](../../CVPR2025/3d_vision/pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[ICLR 2026\] RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering](../../ICLR2026/3d_vision/radiogs_radiometric_gaussian_surfels.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](../../ICCV2025/3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)

</div>

<!-- RELATED:END -->
