---
title: >-
  [Paper Note] MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference
description: >-
  [NeurIPS 2025][3D Vision][Gaussian splatting] MaterialRefGS is proposed to achieve high-fidelity novel view synthesis and accurate illumination decomposition for reflective surfaces…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Gaussian splatting"
  - "reflection modeling"
  - "multi-view consistency"
  - "PBR"
  - "illumination decomposition"
date: 2026-05-08
content_hash: 37d2463a48dccf6b
---

# MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference

**Conference**: NeurIPS 2025
**arXiv**: [2510.11387](https://arxiv.org/abs/2510.11387)  
**Code**: [Project Page](https://wen-yuan-zhang.github.io/MaterialRefGS)  
**Area**: 3D Vision
**Keywords**: Gaussian splatting, reflection modeling, multi-view consistency, PBR, illumination decomposition

## TL;DR

MaterialRefGS is proposed to achieve high-fidelity novel view synthesis and accurate illumination decomposition for reflective surfaces, via multi-view consistent material inference constraints and a 2DGS ray-tracing-based environment modeling strategy.

## Background & Motivation

**Reflective surfaces** remain a core challenge in novel view synthesis with 3D Gaussian Splatting. The original 3DGS encodes view-dependent color using spherical harmonics, which cannot capture high-frequency specular reflections. Recent methods attach reflection-related material properties (metalness, roughness, etc.) to Gaussian primitives and synthesize the final color via PBR deferred shading.

**Two major limitations of existing PBR methods**:

**Ill-posedness of material inference**: All material parameters are optimized solely through photometric loss after light transport, and many combinations of lighting and material can explain the same pixel, leading to suboptimal illumination decomposition. More critically, **the view-dependent nature of the Gaussian representation contradicts the goal of learning view-independent material properties**: the same physical attribute yields inconsistent appearances across views via alpha blending, preventing BRDF from inferring accurate reflections from such ambiguous observations.

**Absence of indirect illumination**: When modeling specular lighting with the split-sum approximation, radiance queried from an environment map is incorrect if the incident direction is occluded by other objects. Existing methods either ignore occlusion or handle it coarsely with offline binary visibility indicators.

**Core insight of this paper**: Multi-view consistent material inference is the key to learning accurate reflections. The paper addresses the above issues through a three-pronged approach: (1) constraining material maps across views to be geometrically consistent, (2) leveraging multi-view photometric variance as a prior for reflection strength, and (3) modeling indirect illumination via differentiable ray tracing.

## Method

### Overall Architecture

2DGS serves as the base representation, with each Gaussian carrying material attributes: diffuse color $c_d \in \mathbb{R}^3$, albedo $a$, metalness $m$, and roughness $r$. Rendering follows deferred shading with PBR: material attributes are rasterized into screen-space material maps, and the final color is computed using a simplified Disney BRDF as $c(\omega_o) = (1-m)c_d + L_s(\omega_o, a, m, r, n)$. Diffuse radiance is predicted directly by Gaussians; specular radiance is evaluated via the BRDF.

### Key Designs

1. **Multi-view Material Consistency**: For a surface point $p$ visible in both views $v_i$ and $v_j$, its projections onto the material maps of both views should be identical. Concretely, a $7 \times 7$ pixel patch is sampled at reference view $v_i$ and warped to source view $v_j$ via homography $H_{ij}$:
    $$H_{ij} = K_j\left(R_{ij} - \frac{T_{ij}n_i^T}{d_i}\right)K_i^{-1}$$
   MSE loss is then computed on the diffuse, roughness, and metalness maps: $\mathcal{L}_{mv} = \|\Psi_i[P(\pi_i(p))] - \Psi_j[P'(\pi_j(p))]\|_2$. **Design Motivation**: This constrains Gaussians to produce consistent material outputs across views, suppressing view-specific overfitting and enabling the BRDF to infer globally consistent physical reflections from unambiguous material observations.

2. **Reflection Strength Prior**: Highly reflective surfaces exhibit significantly different appearances across viewpoints. Based on this observation, photometric variation along the camera trajectory is tracked over object surfaces to quantify a reflection score:

    - For the reference view, $3 \times 3$ patches are sampled and warped to $M$ neighboring views.
    - Cross-view standard deviation is computed as the reflection score $\text{refscore}$ (Eq. 6).
    - A spatial reflection fusion module (ball query + top-$K$ averaging) aggregates this into a multi-view consistent reflection strength prior $w_{ref}$.
    - $w_{ref}$ supervises the metalness map: high-reflection regions are encouraged to have higher metalness values.

   The loss is: when predicted metalness $\Psi^M(u,v) < M_0$, $\mathcal{L}_{ref} = w_{ref} \cdot \Gamma(u,v) \cdot |M_0 - \Psi^M|$. SAM2 segments the foreground into semantic regions, with region-wise gating $\Gamma$.

3. **2DGS Ray-Tracing-Based Environment Modeling**: Incident radiance is decomposed into direct and indirect components: $L_i(\omega_i) = L_{\text{indirect}}(\omega_i) + (1-O(\omega_i)) \cdot L_{\text{direct}}(\omega_i)$. Direct light is queried from a learnable environment cubemap. Indirect light is obtained via BVH ray tracing—rays are cast along reflection directions, intersecting Gaussians are sorted by depth, and splatting is applied to compute indirect color and occlusion probability. The entire process is fully differentiable, allowing Gaussians to participate in environment modeling via joint optimization. Compared to the offline binary visibility indicator in Ref-Gaussian, this approach is physically more accurate and supports end-to-end training.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_c + \lambda_{n-d}\mathcal{L}_{n-d} + \lambda_n\mathcal{L}_n + \lambda_{mv}\mathcal{L}_{mv} + \lambda_{ref}\mathcal{L}_{ref}$$

- $\mathcal{L}_c = 0.8 \cdot \mathcal{L}_{rgb} + 0.2 \cdot \mathcal{L}_{D-SSIM}$: photometric loss
- $\mathcal{L}_{n-d}$: depth-normal consistency loss from 2DGS
- $\mathcal{L}_n = |1 - N^T\hat{N}|$: normal prior loss (supervised by StableNormal)

**Training schedule** (30k iterations total):
- 0–3k iter: train 2DGS + normal prior only
- 3k–10k iter: introduce PBR and environment modeling
- 10k+ iter: remove normal prior (to avoid bias from inaccurate predictions); introduce multi-view regularization

## Key Experimental Results

### Main Results (Novel View Synthesis)

| Method | ShinyBlender PSNR↑ | ShinyBlender MAE↓° | GlossySynthetic PSNR↑ | Ref-Real PSNR↑ | MipNeRF360 LPIPS↓ |
|--------|--------------------|--------------------|----------------------|----------------|-------------------|
| 3DGS | 30.37 | - | 26.01 | 23.85 | 0.214 |
| 3DGS-DR | 33.94 | 2.62 | 29.49 | 23.99 | 0.249 |
| Ref-Gaussian | 35.04 | 4.59 | 30.08 | 24.61 | 0.272 |
| EnvGS | 33.83 | 6.36 | 28.17 | 24.85 | 0.222 |
| **MaterialRefGS** | **35.57** | **2.04** | **30.83** | **25.04** | **0.181** |

On ShinyBlender, the proposed method outperforms 3DGS-DR by 1.6 dB in PSNR, reducing normal MAE from 2.62° to 2.04°. On the complex real-world MipNeRF360 benchmark, it achieves the lowest LPIPS among all compared methods.

### Ablation Study

| Configuration | ShinyBlender PSNR↑ | Ref-Real PSNR↑ | Ref-Real LPIPS↓ | Note |
|---------------|--------------------|----------------|-----------------|------|
| w/o $\mathcal{L}_{mv}$, w/o $\mathcal{L}_{ref}$ | 34.87 | 24.24 | 0.260 | No multi-view constraints |
| w/ $\mathcal{L}_{mv}$, w/o $\mathcal{L}_{ref}$ | 35.21 | 24.47 | 0.242 | Consistency only |
| w/o $\mathcal{L}_n$ | 35.37 | 24.39 | 0.229 | No normal prior |
| w/o Environment | 34.69 | 24.76 | 0.199 | No ray-tracing environment |
| **Full Model** | **35.57** | **25.04** | **0.185** | All components |

All three components—normal prior, material consistency, and environment modeling—contribute to geometric accuracy (normal MAE table):

| Configuration | MAE↓° | CD↓ |
|---------------|-------|-----|
| w/o $\mathcal{L}_n$, w/o Reg, w/o Env | 3.47 | 0.94 |
| w/o $\mathcal{L}_n$ | 2.59 | 0.68 |
| Full Model | 2.04 | 0.60 |

### Key Findings

- Multi-view material consistency is the key to improving illumination decomposition quality: adding this constraint transforms material maps from cross-view inconsistent to spatially uniform.
- The reflection strength prior quantifies reflectivity from photometric variation, providing a self-supervised signal for metalness without additional labels.
- Indirect illumination modeling via environment tracing is critical for inter-reflection scenes with occlusion—removing it causes noticeable blurring in reflective regions.
- Existing reflection methods often degrade on complex real-world scenes such as MipNeRF360; MaterialRefGS maintains competitive performance.
- The normal prior guides geometry initialization in early training but should be removed in later stages to avoid bias—a training design that reflects awareness of optimization dynamics.

## Highlights & Insights

- The idea of using **multi-view consistency as regularization for material inference** is intuitive yet previously overlooked—the assumption that the same physical property should be consistent across views is a fundamental premise of PBR.
- Quantifying reflection strength via multi-view photometric variation and using it as a metalness prior converts a routine observation into an effective supervision signal.
- Assigning diffuse radiance to Gaussians and specular radiance to the BRDF reduces optimization difficulty, making the approach more suitable for complex scenes than full-BRDF methods.
- Differentiable ray-tracing-based indirect illumination modeling allows Gaussians themselves to participate in joint optimization of environment modeling.

## Limitations & Future Work

- The method does not constitute a full inverse rendering pipeline and is not suited for quantitative evaluation of material decomposition or relighting.
- The SAM2 segmentation and ball query operations for the reflection strength prior add preprocessing complexity.
- Multi-view constraints may introduce unnecessary computational overhead for simple diffuse scenes.
- The efficiency of indirect illumination ray tracing is affected by the number of Gaussians and may become a bottleneck in very large scenes.

## Related Work & Insights

- **2DGS**: The base representation of this work, which flattens 3D Gaussians into 2D disks to yield better surface normals.
- **3DGS-DR / Ref-Gaussian**: Prior deferred shading PBR methods that lack multi-view material consistency constraints.
- **EnvGS**: Also employs ray tracing for environment modeling but does not address multi-view material consistency.
- Takeaway: Introducing cross-view constraints into material and geometry optimization for 3DGS is a general and effective regularization strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Multi-view material consistency constraints and the reflection strength prior represent significant innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on 4 datasets (synthetic + real), thorough ablation including normal and geometry accuracy metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation is clearly articulated with high-quality illustrations.
- **Value**: ⭐⭐⭐⭐ Provides practical improvements for 3DGS rendering of reflective surfaces; the multi-view consistency paradigm is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PanSt3R: Multi-view Consistent Panoptic Segmentation](../../ICCV2025/3d_vision/panst3r_multi-view_consistent_panoptic_segmentation.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](../../ICCV2025/3d_vision/materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](../../ICCV2025/3d_vision/supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[NeurIPS 2025\] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](../../ICCV2025/3d_vision/auto-regressively_generating_multi-view_consistent_images.md)

</div>

<!-- RELATED:END -->
