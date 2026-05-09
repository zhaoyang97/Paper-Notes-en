---
title: >-
  [Paper Note] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This work presents the first density-guided poisoning attack against 3D Gaussian Splatting (3DGS). By injecting illusion Gaussians into low-density regions and introducing adaptive noise to disrupt multi-view consistency, the method achieves attacks that are clearly visible from target viewpoints while remaining imperceptible from all others.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - data poisoning attack
  - adversarial security
  - kernel density estimation
  - multi-view consistency
date: 2026-05-08
content_hash: 02e52ce6339fc9c1
---

# StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions

**Conference**: ICCV 2025
**arXiv**: [2510.02314](https://arxiv.org/abs/2510.02314)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, data poisoning attack, adversarial security, kernel density estimation, multi-view consistency

## TL;DR

This work presents the first density-guided poisoning attack against 3D Gaussian Splatting (3DGS). By injecting illusion Gaussians into low-density regions and introducing adaptive noise to disrupt multi-view consistency, the method achieves attacks that are clearly visible from target viewpoints while remaining imperceptible from all others.

## Background & Motivation

As 3DGS is increasingly deployed in safety-critical applications such as autonomous driving and AR/VR, its security vulnerabilities warrant careful examination. Prior work IPA-NeRF demonstrated view-dependent illusion injection for NeRF, but direct adaptation to 3DGS proves largely ineffective. The authors identify two root causes:

**Inherent multi-view consistency defense**: 3DGS employs an explicit point cloud representation whose intrinsic multi-view consistency treats inconsistent illusion content as noise and eliminates it during training.

**Explicit geometric constraints**: Back-projecting illusion objects into the Gaussian point cloud is hindered by depth ambiguity and potential occlusion by existing geometry.

Naïvely adapting IPA-NeRF to 3DGS (IPA-Splat) or directly overlaying illusion content onto training images both fail, demonstrating the need for attack strategies specifically tailored to 3DGS properties.

## Method

### Overall Architecture

The method consists of two complementary strategies: (1) density-guided point cloud attack, which places illusion Gaussians in low-density regions; and (2) view-consistency disruption attack, which corrupts 3DGS multi-view consistency via adaptive noise scheduling.

The attack objective is formulated as:
$$\min_{\tilde{G}} \| \tilde{I}_{\text{ILL}} - I_{\text{ILL}} \|_2^2 + \sum_{v_k \neq v_p} \| \mathfrak{R}(\tilde{G}, v_k) - \mathfrak{R}(G, v_k) \|_2^2$$
That is, maximize illusion visibility at the target viewpoint while minimizing impact on benign viewpoints.

### Key Design 1: Density-Guided Point Cloud Attack

**Scene space analysis**: The reconstructed Gaussian point cloud $G$ is first voxelized into a uniform grid $\mathcal{S}$ within an axis-aligned bounding box (AABB). Each Gaussian's opacity $\alpha(g)$ is estimated via volume rendering, and per-voxel density is aggregated as $\rho(s) = \sum_{g \in s} \alpha(g)$.

**Continuous density estimation**: Kernel density estimation (KDE) smooths the discrete density into a continuous function:
$$f(x) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} K_h(x - c(s)) \cdot \rho(s)$$

**Optimal placement**: Rays are cast from the target camera center $C$ through each pixel of the illusion object, and the minimum-density point along each ray is selected:
$$x_{\min} = \arg\min_{x \in C+t \cdot d, t \in [t_{\min}, t_{\max}]} f(x)$$
where $t_{\min}=0.3$ avoids near-camera floating artifacts. A new Gaussian is inserted at $x_{\min}$ with color sampled from the corresponding illusion pixel.

The core rationale is that low-density regions are either not covered by other viewpoints or occluded by existing geometry, minimizing interference with benign views.

### Key Design 2: View-Consistency Disruption Attack

In scenes with high viewpoint overlap, point cloud injection alone is insufficient. The method selectively adds Gaussian noise to training images of benign viewpoints while keeping the target viewpoint image clean:
$$I_k' = \mathbf{1}_{v_k = v_p} \cdot I_k + \mathbf{1}_{v_k \neq v_p} \cdot \text{clip}(I_k + \eta)$$

Noise magnitude $\sigma_t$ decays over training iterations via three schedules:
- **Linear decay**: $\sigma_{\text{linear}}(t) = \sigma_0 \cdot (1 - t/T)$
- **Cosine decay**: $\sigma_{\text{cosine}}(t) = \sigma_0 \cdot \cos(\pi t / 2T)$
- **Square-root decay**: $\sigma_{\text{sqrt}}(t) = \sigma_0 \cdot \sqrt{1 - t/T}$

Strong early noise disrupts the multi-view consistency defense, while noise reduction in later stages preserves rendering quality on benign viewpoints.

### Loss & Training

The attack objective is the two-term optimization described above: the first term drives illusion rendering at the target viewpoint toward the desired appearance; the second constrains unaffected viewpoints. The 3DGS model itself is trained with the standard $\ell_1 + \ell_{\text{ssim}}$ reconstruction loss.

## Key Experimental Results

### Main Results: Single-Viewpoint Attack (Table 1, Mip-NeRF 360 Dataset)

| Method | V-illusory PSNR↑ | V-illusory SSIM↑ | V-test PSNR↑ |
|------|:-:|:-:|:-:|
| IPA-NeRF (Nerfacto) | 16.00 | 0.582 | 21.94 |
| IPA-NeRF (Instant-NGP) | 17.60 | 0.618 | 20.00 |
| IPA-Splat | 13.23 | 0.518 | 27.39 |
| **Ours** | **27.04** | **0.813** | **27.76** |

The proposed method achieves a V-illusory PSNR of 27.04 (far exceeding the best baseline of 17.60) while preserving benign viewpoint quality (27.76 vs. original 29.45, a drop of only 1.69).

### Multi-Viewpoint Attack (Table 3, 2/3/4 Poisoned Viewpoints)

When simultaneously attacking four viewpoints, the method still achieves 26.95 PSNR on illusion viewpoints, with benign viewpoints degrading only to 27.59.

### Ablation Study (Table 6 — Attack Component Combinations)

| Configuration | V-illusory PSNR | ASR |
|------|:-:|:-:|
| Poisoned view images only | 13.22 | 0/7 |
| Images + density-guided | 26.01 | 6/7 |
| Consistency disruption only | 13.31 | 0/7 |
| **Full combination** | **27.04** | **7/7** |

Both the density-guided point cloud attack and the view-consistency disruption are indispensable; the full combination achieves a 100% attack success rate.

### Key Findings

- KDE bandwidth $h=7.5$ yields the optimal trade-off (Table 4).
- Initial noise $\sigma_0=100$ with linear decay performs best (Table 5).
- The spatial influence of the attack is highly localized: the illusion is visible within ±5° of the target viewpoint and has negligible effect beyond that range (Figure 10).
- KDE-based evaluation confirms a negative correlation between scene density and attack difficulty (Table 2).

## Highlights & Insights

1. **Pioneering work**: The first illusion-injection poisoning attack targeting 3DGS, revealing a security blind spot inherent to explicit 3D representations.
2. **Elegant KDE-driven design**: Kernel density estimation bridges "geometric visibility" and "attack feasibility," reformulating the 3D attack as a density optimization problem.
3. **Attack-defense symmetry insight**: Multi-view consistency in 3DGS is both a rendering quality asset and a defensive barrier; this work cleverly combines noise injection and low-density exploitation to bypass this defense.
4. **Systematic evaluation protocol**: A KDE-based attack difficulty assessment scheme is proposed, providing a reproducible benchmark for future security research.

## Limitations & Future Work

- Attack effectiveness degrades in scenes with high viewpoint overlap (Hard setting PSNR drops to 17.53).
- Illusion objects must be designed in advance; adaptive generation is not supported.
- Defensive countermeasures are not explored — future work should investigate detection and mitigation of such attacks.

## Related Work & Insights

- **IPA-NeRF**: The pioneering poisoning attack on NeRF, injecting view-dependent illusions via bi-level optimization; not transferable to 3DGS.
- **Poison-Splat**: A resource exhaustion attack on 3DGS targeting memory overhead, distinct from the visual illusion injection focus of this work.
- **GaussianMarker / 3D-GSW**: Watermarking methods for 3DGS that embed binary information, representing the complementary side of 3DGS security research.
- **FGSM / PGD**: Classical adversarial attack methods; this work extends the adversarial concept to 3D scene representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First attack on 3DGS; novel problem formulation; original KDE-guided strategy.
- **Technical Depth**: ⭐⭐⭐⭐ — Elegant component design, though modules are relatively independent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, multi-viewpoint settings, difficulty stratification, and comprehensive ablations.
- **Value**: ⭐⭐⭐⭐ — Exposes security vulnerabilities with important implications for defensive research.
- **Overall Recommendation**: ⭐⭐⭐⭐☆ — Pioneering security research; the practical threat posed by the attack scenario warrants further substantiation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[NeurIPS 2025\] DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/dc4gs_directional_consistency-driven_adaptive_density_control_for_3d_gaussian_sp.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)

</div>

<!-- RELATED:END -->
