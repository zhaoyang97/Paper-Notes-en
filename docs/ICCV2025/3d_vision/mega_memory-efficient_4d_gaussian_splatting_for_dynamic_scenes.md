---
title: >-
  [Paper Note] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes
description: >-
  [ICCV 2025][3D Vision][4D Gaussian Splatting] This paper proposes MEGA, a memory-efficient framework for 4D Gaussian Splatting that eliminates redundant spherical harmonic coefficients via DC-AC color decomposition (8× c…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "memory-efficient"
  - "dynamic scenes"
  - "color compression"
  - "entropy-constrained deformation"
date: 2026-05-08
content_hash: cc17212e596a145b
---

# MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes

**Conference**: ICCV 2025
**arXiv**: [2410.13613](https://arxiv.org/abs/2410.13613)  
**Code**: [Xinjie-Q/MEGA](https://github.com/Xinjie-Q/MEGA)  
**Area**: 3D Vision
**Keywords**: 4D Gaussian Splatting, memory-efficient, dynamic scenes, color compression, entropy-constrained deformation

## TL;DR

This paper proposes MEGA, a memory-efficient framework for 4D Gaussian Splatting that eliminates redundant spherical harmonic coefficients via DC-AC color decomposition (8× compression) and reduces the total number of Gaussians through entropy-constrained Gaussian deformation. MEGA achieves approximately 190× and 125× storage compression on the Technicolor and Neural 3D Video datasets, respectively, while maintaining comparable rendering quality and real-time speed.

## Background & Motivation

**4D Gaussian Splatting (4DGS)** extends 3DGS to dynamic scenes by representing scene motion with 4D spacetime Gaussian hypercylinders, obtaining per-frame 3D Gaussians via temporal slicing for real-time rendering. However, 4DGS faces a **severe storage bottleneck**:

**Massive redundancy in spherical harmonic coefficients**: Among the 161 parameters per 4D Gaussian, 144 are 4D spherical harmonic (SH) coefficients, accounting for 89% of the total. These SH coefficients encode view- and time-dependent color variations but contain substantial redundancy.

**Explosion in the number of Gaussians**: Rendering the Birthday scene requires up to 13 million Gaussians, incurring approximately 7.79 GB of storage. This arises from two primary causes:
   - 4DGS assumes that each sliced Gaussian undergoes only linear motion with constant covariance, so complex nonlinear motions must be represented by superposing multiple Gaussians.
   - The temporally decaying opacity $\sigma(t) = e^{-\frac{(t-\mu_t)^2}{2\mathbf{W}}}$ causes each Gaussian to be visible only near its temporal center, such that **only approximately 6% of Gaussians participate in rendering at any given time**.

**Inapplicability of existing 3DGS compression methods**: Techniques such as pruning, SH distillation, and vector quantization in 3DGS are designed for static scenes and do not account for the temporal and multi-view factors in 4DGS.

**Mechanism**: Compression is pursued along two dimensions: (1) reducing the number of parameters per Gaussian by eliminating SH coefficients, and (2) reducing the total number of Gaussians by enlarging the spatiotemporal influence of each Gaussian.

## Method

### Overall Architecture

MEGA comprises three core components:

1. **Memory-efficient 4D Gaussian representation**: SH coefficients are replaced with DC-AC color representation.
2. **Entropy-constrained Gaussian deformation**: A deformation predictor combined with an opacity entropy loss.
3. **Storage compression**: FP16 precision followed by zip delta compression.

The rendering pipeline consists of four steps: per-Gaussian transformation → temporal slicing → projection → differentiable rasterization.

### Key Design 1: DC-AC Color (DAC) Representation

Inspired by the concepts of direct current (DC) and alternating current (AC) in electrical engineering, the color attribute is decoupled into:

- **DC component**: A per-Gaussian DC color $\mathbf{c}_{dc} \in \mathbb{R}^3$ with only 3 parameters, encoding the inherent steady-state color information of the scene.
- **AC predictor**: A shared lightweight MLP $\mathcal{F}_\phi$ that predicts color variations conditioned on time and viewpoint.

The final color is computed via a residual connection:
$$\mathbf{c}_{t,v} = \text{sigmoid}(\mathbf{c}_{dc} + \mathcal{F}_\phi(\text{sg}(\boldsymbol{\mu}_{3D}), \text{sg}(\mathbf{d}_v), t, \mathbf{c}_{dc}))$$

where $\text{sg}(\cdot)$ denotes the stop-gradient operation and $\mathbf{d}_v$ is the normalized viewing direction. The AC predictor consists of three linear layers with inputs including 3D position, viewing direction, time, and DC color.

**Effect**: Each Gaussian stores only 3 color parameters (versus 144 SH coefficients), achieving approximately **8× parameter compression**. The key insight is that the DC component retains the core color information while the AC predictor supplements the missing spatiotemporal variation.

### Key Design 2: Entropy-Constrained Gaussian Deformation

**Deformation predictor**: For each 4D Gaussian, time- and view-dependent geometric deformations are predicted. The 4D center $\boldsymbol{\mu}_{4D}$, viewing direction $\mathbf{d}_v$, and time $t$ are mapped to a high-dimensional space via frequency positional encoding $\gamma$, and a lightweight MLP $\mathcal{F}_\theta$ predicts deformation increments:

$$(m_{\mu_{4D}}^{t,v}, m_{s_{4D}}^{t,v}, m_{q_l}^{t,v}, m_{q_r}^{t,v}) = \mathcal{F}_\theta(\gamma(\text{sg}(\boldsymbol{\mu}_{4D})), \gamma(\text{sg}(\mathbf{d}_v)), \gamma(t))$$

Deformations are applied to the original parameters via **multiplicative** modulation (rather than additive), enabling each Gaussian to represent nonlinear motion and shape changes:

$$\boldsymbol{\mu}_{4D}^{t,v} = \boldsymbol{\mu}_{4D} \times m_{\mu_{4D}}^{t,v}, \quad \mathbf{s}_{4D}^{t,v} = \mathbf{s}_{4D} \times m_{s_{4D}}^{t,v}$$

**Opacity entropy loss**: This loss encourages the spatial opacity $o$ to approach binary values (0 or 1), facilitating the identification and pruning of useless Gaussians:

$$\mathcal{L}_{opa} = \frac{1}{N} \sum_{j=1}^N (-o_j \log(o_j))$$

Gaussians with near-zero opacity are pruned every $K$ iterations. Combined with the deformation predictor, the fraction of Gaussians participating in rendering increases from below 50% to approximately 75%.

### Loss & Training

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{ssim} + \kappa \mathcal{L}_{opa}$$

where $\lambda = 0.2$ and $\kappa = 0.0005$.

### Storage Compression

Training is performed in half precision (FP16). All learnable parameters are stored in FP16 format, followed by zip delta compression, which provides an additional reduction of approximately 10% in storage.

## Key Experimental Results

### Main Results 1: Technicolor Dataset

| Method | PSNR↑ | DSSIM1↓ | LPIPS↓ | FPS↑ | Storage↓ |
|--------|-------|---------|--------|------|----------|
| DyNeRF | 31.80 | - | 0.1400 | 0.02 | 30.00MB |
| HyperReel | 32.70 | 0.0470 | 0.1090 | 4.00 | 60.00MB |
| Deformable 3DGS | 30.95 | 0.0696 | 0.1553 | 76.09 | 61.36MB |
| STG | 33.35 | 0.0404 | 0.0846 | 141.73 | 51.35MB |
| 4DGS | 32.07 | 0.0535 | 0.1189 | 55.26 | **6107.07MB** |
| **MEGA** | **33.57** | **0.0442** | **0.1014** | 83.14 | **32.45MB** |

Compared to 4DGS, MEGA achieves **190× storage compression** (6107→32 MB), a 1.5 dB improvement in PSNR, and a 50% increase in FPS. Relative to the previous SOTA method STG, MEGA achieves 0.22 dB higher PSNR with smaller storage.

### Main Results 2: Neural 3D Video Dataset

| Method | PSNR↑ | DSSIM2↓ | FPS↑ | Storage↓ |
|--------|-------|---------|------|----------|
| MixVoxels-X | 31.73 | 0.0150 | 4.60 | 500.00MB |
| Dynamic 3DGS | 30.46 | 0.0190 | 460.00 | 2772.00MB |
| STG | 32.04 | 0.0145 | 273.47 | 175.35MB |
| 4DGS | 31.57 | 0.0164 | 96.69 | 3128.00MB |
| **MEGA** | **31.49** | **0.0165** | 77.42 | **25.05MB** |

On Neu3DV, MEGA achieves **125× storage compression** (3128→25 MB) with visual quality comparable to 4DGS.

### Ablation Study: Contribution of Each Component (Birthday / Fabien / Flame Steak / Sear Steak)

| Variant | PSNR (Birthday) | # Gaussians | # Parameters |
|---------|-----------------|-------------|--------------|
| 4DGS baseline | 31.00 | 13.00M | 2094M |
| w/ grid replacing SH | 30.49 | 16.33M | 293M |
| w/ DAC | 31.60 | 15.43M | 309M |
| w/ DAC + deformation | 31.35 | 15.75M | 315M |
| w/ DAC + $\mathcal{L}_{opa}$ | 31.46 | 9.15M | 183M |
| **w/ DAC + deformation + $\mathcal{L}_{opa}$** | **32.02** | **0.91M** | **18M** |

- The grid-based approach (directly replacing SH) results in a notable quality drop (−0.51 dB), whereas DAC maintains or improves quality (+0.60 dB).
- Using deformation alone increases the number of Gaussians; using $\mathcal{L}_{opa}$ alone limits the expressive capacity of the Gaussians.
- Their combination reduces the number of Gaussians from 13M to 0.91M (**14× reduction**) while achieving a 1.02 dB improvement in PSNR.

## Highlights & Insights

1. **Intuitive elegance of DC-AC decomposition**: Treating color analogously to an electrical signal—DC retains the base color while AC encodes variations—provides an effective and elegant replacement for 144-dimensional SH coefficients.
2. **Synergistic effect of deformation and entropy constraint**: Deformation enlarges the effective spatiotemporal range of each Gaussian, making it more valuable, while the entropy loss removes redundant Gaussians. Only their combination (not either alone) simultaneously reduces Gaussian count and preserves quality.
3. **Remarkable compression ratios**: 190× (Technicolor) and 125× (Neu3DV) storage compression with virtually no quality loss.
4. **Significant practical implications for AR/VR on-device deployment**, compressing GB-scale models to tens of megabytes.

## Limitations & Future Work

1. Both the AC predictor and the deformation predictor are shared MLPs, which may become capacity bottlenecks for extremely complex scenes.
2. PSNR on Neu3DV is marginally lower than 4DGS (31.49 vs. 31.57), indicating a slight quality trade-off under aggressive compression.
3. Evaluation is conducted on only two datasets, lacking outdoor dynamic scenes and more diverse test conditions.
4. The multiplicative deformation formulation may limit the expressibility of certain deformation patterns.

## Related Work & Insights

- **Static 3D representation compression**: Compact-3DGS (redundancy pruning + SH distillation), LightGaussian (vector quantization).
- **Dynamic scene NeRF methods**: DyNeRF, HyperReel, HexPlane, K-Planes.
- **Dynamic scene Gaussian methods**: Deformable 3DGS (regularization + deformation field), 4DGS (4D spacetime Gaussians), STG (spacetime Gaussians), E-D3DGS.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 5 |
| Overall | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 4D Gaussian Splatting SLAM](4d_gaussian_splatting_slam.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](../../CVPR2026/3d_vision/motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](../../ICLR2026/3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
