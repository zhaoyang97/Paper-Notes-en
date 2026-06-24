---
title: >-
  [Paper Note] Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis
description: >-
  [ECCV 2024][3D Vision][Thermal Infrared Imaging] This paper proposes Thermal3D-GS, which models atmospheric transmission effects and thermal conduction physical processes using neural networks and introduces temperature consistency constraints, achieving high-quality novel-view synthesis of thermal infrared images, and establishing the first large-scale thermal infrared novel-view synthesis dataset, TI-NSD.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Thermal Infrared Imaging"
  - "Novel-view Synthesis"
  - "3D Gaussian Splatting"
  - "Physical Modeling"
  - "Atmospheric Transmission"
date: 2026-05-08
content_hash: d36947bec09e40e5
---

# Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2409.08042](https://arxiv.org/abs/2409.08042)  
**Code**: [Project Page](https://github.com/Thermal3DGS)  
**Area**: 3D Vision  
**Keywords**: Thermal Infrared Imaging, Novel-view Synthesis, 3D Gaussian Splatting, Physical Modeling, Atmospheric Transmission

## TL;DR

This paper proposes Thermal3D-GS, which models atmospheric transmission effects and thermal conduction physical processes using neural networks and introduces temperature consistency constraints, achieving high-quality novel-view synthesis of thermal infrared images, and establishing the first large-scale thermal infrared novel-view synthesis dataset, TI-NSD.

## Background & Motivation

Thermal infrared imaging possesses all-weather imaging capabilities and strong penetration, presenting significant advantages in nighttime and harsh weather scenarios. However, directly applying visible-light novel-view synthesis methods to thermal infrared images leads to two specific issues: (1) **Floaters**—atmospheric transmission effects cause varying radiation attenuation of the same object under different views, prompting 3D-GS to learn incorrect compensating Gaussians; (2) **Blurry edges**—heat conduction between objects alters the boundary temperature gradients, and averaging across multiple frames leads to the loss of edge information.

## Method

### Overall Architecture

Based on the 3D-GS framework, two physics-driven modules are added: (1) Atmospheric Transmission Field (ATF)—modeling atmospheric attenuation; (2) Thermal Conduction Module (TCM)—modeling the impact of thermal conduction on edges. Additionally, a temperature discontinuity loss constraint is introduced.

### Key Designs

**Atmospheric Transmission Field (ATF)**: Models radiation attenuation based on the Bouguer-Lambert-Beer law $I = I_0 e^{\mu(\lambda)d}$. An MLP network (depth 8, hidden dimension 256) takes the position-encoded Gaussian location $\gamma(x)$ and capture time $\gamma(t)$ as inputs to predict the absorption coefficient $\mu_{abs}$, scattering coefficient $\mu_{sca}$, and propagation distance $d$:

$$SH = SH_0 \cdot e^{(\mu_{abs} + \mu_{sca})d}$$

This decouples attenuation effects from the geometry, allowing 3D-GS to independently learn attenuation-free geometric structures.

**Thermal Conduction Module (TCM)**: Based on the 2D temperature field heat conduction equation $\frac{\partial u}{\partial t} = \alpha \Delta u$, where $\Delta$ is the 2D Laplacian operator, and $\alpha = k/(c\rho)$ is the thermal diffusivity. Since $\alpha$ is non-uniform across pixels, a 3-layer convolutional network is employed to fuse the input image with its second-order gradient features to simulate pixel-level $\alpha$, repairing the heat loss caused by thermal conduction through a residual addition mechanism.

**Temperature Discontinuity Loss**: Real temperature fields are typically smooth and continuous; corner points denote learning anomalies. The response function of Harris corner detection is used as a weight to emphasize the L1 loss on abnormal regions:

$$\mathcal{L}_{dis} = \frac{R}{R_{max}} \max(1 - \frac{i}{iter_t}, 0) \mathcal{L}_1$$

where $iter_t = 5000$ serves as the decay threshold.

### Loss & Training

$$\mathcal{L}_{total} = \lambda_{dis}\mathcal{L}_{dis} + \lambda\mathcal{L}_{D-SSIM} + (1-\lambda_{dis}-\lambda)\mathcal{L}_1$$

where $\lambda_{dis} = \lambda = 0.2$.

## Key Experimental Results

### Quantitative Comparison on TI-NSD Dataset

Average results across 20 scenes:

| Method | Indoor PSNR↑ | Outdoor PSNR↑ | UAV PSNR↑ | Mean PSNR↑ | Mean SSIM↑ | Mean LPIPS↓ |
|------|-----------|-----------|-------------|-----------|-----------|-----------|
| Plenoxels | 22.13 | 22.15 | 25.56 | 23.28 | 0.805 | 0.390 |
| INGP-Base | 26.99 | 26.00 | 20.86 | 24.62 | 0.811 | 0.332 |
| INGP-Big | 27.46 | 26.45 | 20.82 | 24.91 | 0.812 | 0.323 |
| 3D-GS (30k) | 32.98 | 28.89 | 34.51 | 32.01 | 0.936 | 0.206 |
| **Ours (30k)** | **36.01** | **32.60** | **36.74** | **35.04** | **0.955** | **0.187** |

### Ablation Study

| Method | Indoor PSNR↑ | Outdoor PSNR↑ | UAV PSNR↑ | Mean PSNR↑ |
|------|-----------|-----------|-------------|-----------|
| 3D-GS | 32.98 | 28.89 | 34.51 | 32.01 |
| 3D-GS + ATF | 35.12 | 31.53 | 36.65 | - |
| **3D-GS + ATF + TCM + Loss** | **36.01** | **32.60** | **36.74** | **35.04** |

### Key Findings

- Compared to the 3D-GS baseline, the average improvement is 3.03 dB PSNR (35.04 vs 32.01), representing a significant enhancement.
- The ATF module contributes the most (~2 dB), effectively eliminating the floater issue.
- TCM shows obvious performance improvements in boundary regions between high-and-low-temperature objects, recovering sharp edges blurred by thermal conduction.
- In UAV scenes, fast flight speeds cause motion blur, yet our method still maintains excellent reconstruction quality.

## Highlights & Insights

1. **The first method dedicated to thermal infrared novel-view synthesis**, incorporating physical imaging processes (atmospheric transmission, thermal conduction) into the 3D-GS framework.
2. The TI-NSD dataset (20 scenes, 6664 frames, covering indoor/outdoor/UAV) fills the dataset gap in this field.
3. Physics-formulation-driven network design—instead of directly solving physical equations (which is an underdetermined problem), physical equations are used to guide the network structural design.

## Limitations & Future Work

- It assumes that each Gaussian shares a uniform attenuation coefficient, which may not apply to scenarios with drastically changing attenuation.
- TCM only operates in the 2D image space, without modeling thermal conduction in 3D space.
- The temperature discontinuity loss relies on the accuracy of Harris corner detection.

## Related Work & Insights

This paper pioneeringly combines thermal infrared physical characteristics with 3D reconstruction. The concept of physics-driven degradation modeling (decoupling imaging effects from geometry) can be extended to novel-view synthesis in other specialized imaging modalities (e.g., SAR, medical imaging).

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[CVPR 2026\] Thermal is Always Wild: Characterizing and Addressing Challenges in Thermal-Only Novel View Synthesis](../../CVPR2026/3d_vision/thermal_is_always_wild_characterizing_and_addressing_challenges_in_thermal-only_.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)
- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)

</div>

<!-- RELATED:END -->
