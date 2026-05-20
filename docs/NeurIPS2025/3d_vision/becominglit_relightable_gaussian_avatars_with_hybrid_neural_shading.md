---
title: >-
  [Paper Note] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading
description: >-
  [NeurIPS 2025][3D Vision][relightable avatar] This paper proposes BecomingLit, a method that reconstructs high-fidelity, relightable, and real-time renderable head avatars from low-cost light stage multi-view sequences u…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "relightable avatar"
  - "3D Gaussian Splatting"
  - "hybrid neural shading"
  - "light stage"
  - "BRDF"
  - "face reconstruction"
date: 2026-05-08
content_hash: 5c06305e172a8f1d
---

# BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading

**Conference**: NeurIPS 2025
**arXiv**: [2506.06271](https://arxiv.org/abs/2506.06271)  
**Code**: [jonathsch.github.io/becominglit](https://jonathsch.github.io/becominglit)  
**Area**: 3D Vision
**Keywords**: relightable avatar, 3D Gaussian Splatting, hybrid neural shading, light stage, BRDF, face reconstruction

## TL;DR

This paper proposes BecomingLit, a method that reconstructs high-fidelity, relightable, and real-time renderable head avatars from low-cost light stage multi-view sequences using 3D Gaussian primitives and hybrid neural shading (neural diffuse BRDF + analytic Cook-Torrance specular). A new publicly available OLAT facial dataset is also released.

## Background & Motivation

**Strong industrial demand**: The need for relightable, photorealistic head avatars in VR/metaverse applications is rapidly growing, yet most existing methods bake the training illumination into the appearance, making re-lighting impossible.

**High cost of traditional light stages**: Existing methods (e.g., RGCA, Deep Appearance Models) rely on room-scale setups with hundreds of lights and cameras, affordable only by a few institutions.

**Scarcity of public datasets**: Controlled multi-view datasets for facial appearance modeling are extremely limited (Goliath covers only 4 subjects at relatively low resolution), hindering broad academic research.

**Analytic BRDFs are insufficient for skin**: Skin exhibits significant subsurface scattering and other global illumination effects that purely analytic models (e.g., Lambertian + Cook-Torrance) cannot accurately reproduce, particularly in the diffuse component.

**Poor generalization of existing neural methods**: RGCA learns precomputed radiance transfer (PRT), which generalizes poorly to unseen illumination; it also relies on per-identity VAE expression spaces, preventing cross-identity reenactment.

**Goal**: Using a low-cost light stage consisting of 16 cameras and 40 LEDs, combined with the FLAME parametric model and a hybrid neural BRDF, the paper aims to match state-of-the-art quality at approximately one-tenth the cost while supporting monocular video-driven animation.

## Method

### Overall Architecture

Input FLAME expression/pose parameters → geometry module $\mathcal{F}_g$ predicts 3D Gaussian attributes → hybrid neural shading (diffuse $\mathcal{F}_d$ + analytic specular) → Gaussian Splatting rendering → joint optimization via L1 + SSIM photometric loss + regularization.

### Key Designs

**1. Expression-Dependent Geometry Module $\mathcal{F}_g$ (UV-Space CNN)**

- A fixed number of anisotropic Gaussian primitives are defined on the FLAME UV map (512² UV → ~202k primitives).
- $\mathcal{F}_g$ is a transposed convolutional network that takes FLAME expression coefficients (109-dim) as input and outputs per-texel position offsets $\delta\mu$, rotation $q$, scale $s$, opacity $\sigma$, and expression features $f^{expr}$ (32-dim).
- Gaussian centers = FLAME mesh interpolated positions + TBN-rotated static local offsets + dynamic offsets $\delta\mu$; static offsets carry most of the displacement, while dynamic offsets are regularized to remain small, improving generalization to novel expressions.

**2. Hybrid Neural Shading**

- **Diffuse $\mathcal{F}_d$**: A 3-layer MLP (hidden size 64) that takes 6th-order spherical harmonic coefficients (encoding incident light) and $f^{expr}$ as input, and outputs a scalar reflectance multiplied by a statically learned albedo $a_k$ to produce the diffuse color. Parameterized as a monochromatic BRDF, it is trained under white light and evaluated per RGB channel at inference to support colored illumination. Subsurface scattering and self-occlusion are learned implicitly.
- **Specular $f_s$**: Based on the Cook-Torrance model. The NDF uses a 2-lobe Blinn-Phong mixture (roughness $r$ interpolated linearly); the Fresnel term uses the Schlick approximation; the masking-shadowing term is derived from the NDF. A small CNN $\mathcal{F}_v$ conditioned on $(f^{expr}, \omega_o)$ predicts specular intensity $k_s$ and normal perturbation $\delta n$; the shading normal is computed by normalizing the sum of the mesh normal and $\delta n$.
- Environment map relighting uses the split-sum approximation (pre-integrated mipmap + 2D BRDF LUT).

**3. Low-Cost OLAT Dataset**

- 16 industrial cameras (2200×3208, 72 fps, PTP sub-microsecond synchronization) + 40 high-CRI LEDs covering the front hemisphere.
- 10 subjects, each with ~150s of multi-sequence data (predefined expressions + reading + free expressions); OLAT frames and full-illumination tracking frames alternate at a 2:1 ratio.
- Setup cost is approximately one-tenth that of Goliath (144 views / 460 lights).

### Loss & Training

$$\mathcal{L} = \underbrace{\lambda_{l1}\mathcal{L}_{l1} + \lambda_{SSIM}\mathcal{L}_{SSIM}}_{\text{photometric}} + \lambda_{normal}\|\delta n\| + \lambda_{alpha}\mathcal{L}_{alpha} + \lambda_{scale}\mathcal{L}_{scale} + \lambda_{pos}\|\delta\mu\|$$

- $\lambda_{l1}=1.0, \lambda_{SSIM}=0.2$; $\lambda_{alpha}=\lambda_{scale}=2\text{e-}2, \lambda_{pos}=1\text{e-}5$.
- The alpha loss (L2 between rendered alpha and foreground mask) is a critical regularizer that prevents transparency artifacts during environment map relighting — enabling front-hemisphere-only training to avoid back-side artifacts.

## Key Experimental Results

### Main Results (Table 2, 4 subjects, 15 training views / 1 test view, 4 hold-out illuminations)

| Method | Relighting PSNR↑ | SSIM↑ | LPIPS↓ | Relight+Reenact PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|
| RGCA | 29.21 | 0.8462 | 0.1659 | 26.31 | 0.8206 | 0.1917 |
| RGCA_FLAME | 29.78 | 0.8464 | 0.1444 | 26.91 | 0.8282 | 0.1667 |
| **BecomingLit** | **31.38** | **0.8956** | **0.1040** | **28.08** | **0.8730** | **0.1317** |

- Outperforms RGCA by approximately **+2.2 dB** in relighting PSNR, with a **37%** reduction in LPIPS.
- Replacing RGCA's per-identity VAE with a FLAME expression space (RGCA_FLAME) consistently improves performance, confirming that a shared expression space is superior for reenactment.

### Ablation Study (Table 3)

| Variant | Relight PSNR | Reenact PSNR | Key Findings |
|---|---|---|---|
| w/ PBR shading | 29.42 | 26.31 | Purely analytic BRDF cannot model subsurface scattering; produces plastic appearance |
| w/ PRT diffuse | 29.23 | 25.47 | PRT generalizes worst to unseen illuminations (−2.6 dB vs. full model) |
| w/ SG specular | 31.55 | 28.09 | Slightly better under point lights but lacks pore-level specular detail |
| w/o alpha loss | 31.34 | 28.07 | Transparency artifacts appear during environment map relighting |
| w/o expr features | 31.23 | 28.13 | Degraded recovery of pore details and specular highlights |
| **Full model** | **31.38** | **28.08** | Best overall performance |

### Runtime (Table 4, RTX A6000, 202k primitives, 1100×1604)

| Method | CNN | Diffuse | Specular | Splatting | Total |
|---|---|---|---|---|---|
| RGCA | 9ms | 1ms | 1ms | 9ms | 20ms |
| **Ours** | 4ms | 3ms | 1ms | 9ms | **17ms** (~59 FPS) |

### Key Findings

1. The core advantage of hybrid neural shading lies in the diffuse component: implicit learning of subsurface scattering substantially outperforms PRT and analytic PBR models in generalization.
2. Analytic Cook-Torrance specular produces more natural pore-level reflections under environment lighting than Spherical Gaussians.
3. The alpha regularizer enables front-hemisphere-only setups to support full-environment relighting, roughly halving setup complexity, cost, and training computation.
4. Expression-dependent features $f^{expr}$ jointly influence diffuse and specular quality, serving as a bridge between geometry and appearance.

## Highlights & Insights

- **Novel hybrid shading paradigm**: The diffuse term is learned implicitly by a neural network while the specular term uses a classical analytic model, achieving both physical interpretability and expressive power.
- **Practical low-cost setup**: 16 cameras + 40 LEDs suffice to reach state-of-the-art quality, at roughly one order of magnitude lower cost than Goliath (144 cameras + 460 lights).
- **First public high-resolution OLAT facial dataset**: 10 subjects, 72 fps, 2200×3208, filling a significant gap in community resources.
- **End-to-end pipeline**: FLAME-driven → 3DGS geometry → hybrid shading → splatting rendering, with 17ms real-time inference.
- **Monocular video drivable**: After training, only FLAME parameters are required for animation, making the method naturally compatible with monocular trackers such as VHAP.

## Limitations & Future Work

1. Training still requires thousands of light stage frames with diverse expressions; reconstruction from casual smartphone footage remains out of reach, leaving a gap to consumer-level applications.
2. Geometry is constrained by FLAME: the oral interior is not modeled, the method is sensitive to gaze direction tracking, and fine-grained geometric detail is limited.
3. The diffuse network $\mathcal{F}_d$ is trained from scratch without leveraging facial appearance priors, which may lead to instability under limited data.
4. The dataset covers only 10 subjects with limited diversity in ethnicity and age.
5. Ethical risk: photorealistic avatars drivable from monocular video pose a deepfake threat, despite requiring an initial capture session.

## Related Work & Insights

| Direction | Representative Methods | Relation to This Work |
|---|---|---|
| 3DGS head modeling | GaussianAvatars, NPGA, RGCA | This work adds relightability on top of the 3DGS foundation |
| Light stage facial capture | Debevec 2000, Guo 2019, Goliath | This work proposes a lower-cost setup and a new dataset |
| Precomputed radiance transfer | RGCA (PRT SH coefficients) | This work replaces PRT with neural diffuse, improving generalization |
| Neural shading | LitNeRF, ReNeRF, Neural BRDF | This work is the first to combine neural diffuse + analytic specular on dynamic 3DGS avatars |
| Drivable avatars | FLAME, VHAP, Codec Avatars | This work reuses the shared FLAME expression space, eliminating per-identity encoders |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The hybrid neural shading paradigm (neural diffuse + analytic specular), the low-cost light stage, and the public dataset are all novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative comparison on 4 subjects, 5 ablation variants, runtime analysis, and demonstrations of environment relighting and monocular reenactment; subject count is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, complete method figures, rigorous mathematical derivations, and detailed appendix.
- **Value**: ⭐⭐⭐⭐ — The public dataset and low-cost pipeline are likely to stimulate follow-up work; real-time performance and flexible driving make the method suitable for VR/AR deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid Physical-Neural Simulator for Fast Cosmological Hydrodynamics](hybrid_physical-neural_simulator_for_fast_cosmological_hydrodynamics.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](../../ICCV2025/3d_vision/gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[NeurIPS 2025\] ROGR: Relightable 3D Objects using Generative Relighting](rogr_relightable_3d_objects_using_generative_relighting.md)
- [\[ICCV 2025\] Sequential Gaussian Avatars with Hierarchical Motion Context](../../ICCV2025/3d_vision/sequential_gaussian_avatars_with_hierarchical_motion_context.md)

</div>

<!-- RELATED:END -->
