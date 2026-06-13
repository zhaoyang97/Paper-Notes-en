---
title: >-
  [Paper Note] MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting
description: >-
  [NeurIPS 2025][3D Vision][3D relighting] MetaGS is proposed to achieve high-quality 3D scene relighting under out-of-distribution (OOD) lighting conditions by embedding a differentiable Blinn-Phong reflectance model into…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D relighting"
  - "out-of-distribution generalization"
  - "meta-learning"
  - "3D Gaussian splatting"
  - "Blinn-Phong model"
date: 2026-05-08
content_hash: 0396ebe952379fbb
---

# MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting

**Conference**: NeurIPS 2025
**arXiv**: [2405.20791](https://arxiv.org/abs/2405.20791)  
**Code**: Unavailable  
**Area**: 3D Vision
**Keywords**: 3D relighting, out-of-distribution generalization, meta-learning, 3D Gaussian splatting, Blinn-Phong model

## TL;DR

MetaGS is proposed to achieve high-quality 3D scene relighting under out-of-distribution (OOD) lighting conditions by embedding a differentiable Blinn-Phong reflectance model into 3D Gaussian splatting and adopting a bilevel meta-learning training strategy.

## Background & Motivation

3D scene relighting requires a model to alter lighting effects while preserving scene geometry. OLAT (One Light At a Time) is a practical data acquisition setup in which each exposure is illuminated by a single point light source.

**Root Cause — Out-of-Distribution Relighting**:

Existing OLAT methods assume consistent light source distributions between training and testing. In real-world scenarios, however, light positions are arbitrary, and test lights may fall in regions not covered during training. As shown in Figure 1, when test lighting originates from the hemisphere opposite to the training lights, state-of-the-art methods such as NRHints and GS3 degrade severely — producing chaotic specular highlights, shadows, and even color shifts. The underlying causes are:

**Overfitting to specific lighting patterns**: Standard 3DGS/NeRF loss functions optimize reconstruction accuracy only on training samples, without encouraging generalization across lighting conditions.

**Non-extrapolatable implicit lighting representations**: Encoding lighting effects implicitly via SH coefficients or MLPs cannot generalize to unseen lighting directions.

**Geometry–lighting entanglement**: Under the OLAT setting, per-frame colors vary drastically with illumination, increasing ambiguity in learning true geometry and reflectance properties.

**Key Insight**: OOD relighting is addressed along two dimensions — (1) physically-based priors (Phong model) are used to explicitly disentangle diffuse, specular, and ambient components, enabling the model to understand the physics of light interaction; (2) a meta-learning strategy simulates OOD test conditions during training, forcing the model to learn lighting-agnostic geometry and reflectance properties.

## Method

### Overall Architecture

MetaGS extends standard 3DGS by associating each Gaussian point with an additional normal $\mathbf{n}$, a 3-channel diffuse color $k_d$, and a 1-channel specular coefficient $k_s$. Training proceeds in three stages: base Gaussian attributes are first trained, normal optimization is then introduced, and finally diffuse/specular components are incorporated with joint meta-learning training.

### Key Designs

1. **Differentiable Phong Reflectance Model**: The color of each Gaussian point is explicitly decomposed into three components:

    - **Ambient** $L_a$: constant ambient illumination represented by the zeroth-order spherical harmonic coefficient $f_0$
    - **Diffuse** $L_d = k_d I_d$, where $I_d = \frac{I}{r^2}\max(0, \mathbf{n} \cdot \mathbf{l})$ (Lambert's law)
    - **Specular** $L_s = k_s I_s$, where $I_s = \frac{I}{r^2}\max(0, \mathbf{n} \cdot \mathbf{h})^p$ (Blinn-Phong model)

   $\mathbf{l}$ denotes the point-to-light direction, and $\mathbf{h}$ is the halfway vector between the viewing direction $\mathbf{v}$ and the light direction $\mathbf{l}$. The total color is $L_p = L_a + T_i^{\text{light}} \sum(k_d I_d + k_s I_s)$, where $T_i^{\text{light}}$ is a shadow visibility factor computed via BVH ray tracing. **Design Motivation**: The explicit physical formulation enables the model to compute specular highlight positions from normals and light directions, rather than memorizing per-view colors, thereby generalizing to unseen directions.

2. **Meta-Learning Bilevel Optimization**: Rendering under different lighting conditions is treated as independent tasks. The core intuition is to simulate OOD test conditions at each gradient update step:

    - **Inner loop**: Independent training on each support set sample (specific lighting), producing $m$ sub-model hypotheses $\theta_i' \leftarrow \theta - \alpha\nabla_\theta\mathcal{L}(\theta; \mathcal{D}_i^{\text{sup}})$
    - **Outer loop**: Evaluation of these sub-models on the query set (different lighting), aggregating losses to update global parameters $\theta \leftarrow \theta - \beta\sum_{i=1}^m\nabla_\theta\mathcal{L}(\theta_i'; \mathcal{D}_i^{\text{query}})$

   The resulting second-order gradients explicitly encourage the lighting attributes ($f_0, k_d, k_s$) and geometric attributes ($\mathbf{x}, \mathbf{n}, R, S, \alpha$) of each Gaussian to converge consistently across different lighting conditions, rather than overfitting to specific training samples.

3. **BVH Shadow Computation**: For each Gaussian point, a ray is traced from its center toward the light source, and the accumulated transmittance $T_i^{\text{light}}$ across all Gaussians along the path is computed as the shadow visibility factor. Compared to implicit shadow modeling, this explicit physical approach is more interpretable and generalizable.

### Loss & Training

**Three-Stage Training**:
- **Stage 1** (4k iter): Training of base Gaussian attributes (position, rotation, scale, opacity, SH coefficients) to obtain coarse geometry and averaged color.
- **Stage 2** (4k iter): Introduction of normal attribute optimization with progressive alignment to depth-derived pseudo-normals.
- **Stage 3** (5k iter): Incorporation of diffuse/specular components with joint meta-learning training of all parameters.

**Objective Function**: RGB loss (L1 + D-SSIM) + sparsity loss (encouraging opacity toward 0 or 1) + normal alignment loss.

Total training time is approximately 1 hour on a single RTX 3090.

## Key Experimental Results

### Main Results (OOD Relighting PSNR)

| Method | Ball | PlaCup | RubCup | Cat | CatSmall | CupFabric | Fish | FurScene | Pikachu | Pixiu |
|--------|------|--------|--------|-----|----------|-----------|------|----------|---------|-------|
| NRHints | 17.25 | 23.92 | 27.44 | 18.04 | 24.63 | 24.65 | 22.57 | 21.55 | 24.00 | 23.03 |
| GS3 | 18.84 | 20.30 | 24.37 | 17.66 | 23.34 | 25.04 | 21.12 | 17.34 | 24.11 | 19.63 |
| **MetaGS** | **26.76** | **27.54** | **27.95** | **26.45** | **26.44** | **27.29** | **24.68** | **24.82** | **25.54** | **25.65** |

MetaGS achieves an average PSNR approximately 3–9 dB higher than NRHints and 2–8 dB higher than GS3 on synthetic data.

| Setting | Ball | PlaCup | RubCup |
|---------|------|--------|--------|
| IRON (colocated) | 26.99 | 34.43 | 36.22 |
| **MetaGS** (colocated) | **38.72** | **36.90** | **38.89** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Notes |
|---------------|-------|-------|--------|-------|
| **Full model** | **27.42** | **0.9546** | **0.0505** | All components |
| w/o Meta-learning | 19.14 | 0.8781 | 0.0892 | PSNR drops by 8 dB |
| w/o Shadow | 21.53 | 0.9105 | 0.0735 | Shadow computation is critical |

### Key Findings

- Meta-learning is the most critical component; removing it causes a PSNR drop exceeding 8 dB, reducing the model to overfitting the training lighting patterns.
- Under the camera-light-colocated setting, MetaGS substantially outperforms IRON (Ball: 38.72 vs. 26.99).
- Trained exclusively under the OLAT setting, MetaGS successfully generalizes to relighting with unseen environment light maps.
- The explicit Phong decomposition yields interpretable diffuse and specular components — ablation visualizations show that without meta-learning, component estimates are entirely incorrect.
- The method remains effective on real-world data, where baselines exhibit color shifts and floating artifacts, while MetaGS produces physically plausible highlights and shadows.

## Highlights & Insights

- **First integration of meta-learning with 3DGS**: Bilevel optimization is introduced into Gaussian splatting training, pioneering OOD generalization in volumetric rendering.
- The Phong model, though simple, provides the correct physical prior — explicitly modeling normal–light interactions generalizes better than implicit SH representations.
- Treating different lighting conditions as tasks in a multi-task learning framework and using cross-task validation after inner-loop adaptation to constrain generalization is a meta-learning paradigm transferable to other 3D scene understanding tasks.
- BVH ray tracing for shadows is differentiable yet physically correct, balancing gradient-based training with physical fidelity.

## Limitations & Future Work

- Only direct illumination is modeled; indirect illumination and global light transport are not addressed.
- The Phong model has limited capacity to represent materials with strong subsurface scattering or anisotropic reflectance.
- Meta-learning introduces second-order gradients, increasing training computational cost.
- Experiments are conducted primarily on desktop-scale objects; scalability to large scenes remains unvalidated.

## Related Work & Insights

- **NRHints**: A NeRF-based OLAT relighting method with implicit lighting modeling that degrades severely under OOD conditions.
- **GS3**: A 3DGS-based OLAT relighting method that similarly underperforms in OOD settings.
- **MAML**: The classic meta-learning method that inspires MetaGS's bilevel optimization.
- **Insight**: The combination of physical priors and meta-learning may serve as a general paradigm for addressing distribution shift in various 3D vision tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First introduction of meta-learning into 3DGS relighting; the formulation of the OOD relighting problem itself constitutes a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across OOD, colocated, and environment map settings with clear ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated; algorithmic framework is described rigorously.
- **Value**: ⭐⭐⭐⭐ Practically significant for robustness in 3D relighting; the meta-learning paradigm has broad generalization potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[ICCV 2025\] A Unified Interpretation of Training-Time Out-of-Distribution Detection](../../ICCV2025/3d_vision/a_unified_interpretation_of_training-time_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] ROGR: Relightable 3D Objects using Generative Relighting](rogr_relightable_3d_objects_using_generative_relighting.md)
- [\[NeurIPS 2025\] From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries](from_programs_to_poses_factored_real-world_scene_generation_via_learned_program_.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)

</div>

<!-- RELATED:END -->
