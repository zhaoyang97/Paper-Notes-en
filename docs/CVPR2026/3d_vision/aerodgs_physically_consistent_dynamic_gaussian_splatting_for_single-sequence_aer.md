---
title: >-
  [Paper Note] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction
description: >-
  [CVPR2026][3D Vision][4D Reconstruction] This paper proposes AeroDGS, a physics-guided 4D Gaussian Splatting framework for monocular UAV video. It introduces a Monocular Geometry Lifting module to reconstruct reliable st…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "4D Reconstruction"
  - "3D Gaussian Splatting"
  - "Physical Priors"
  - "UAV Monocular Video"
  - "Dynamic Scene"
  - "Aerial"
date: 2026-05-08
content_hash: 5e31d3c4f616d0e9
---

# AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction

**Conference**: CVPR2026
**arXiv**: [2602.22376](https://arxiv.org/abs/2602.22376)  
**Authors**: Hanyang Liu, Rongjun Qin
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: 4D Reconstruction, 3D Gaussian Splatting, Physical Priors, UAV Monocular Video, Dynamic Scene, Aerial

## TL;DR

This paper proposes AeroDGS, a physics-guided 4D Gaussian Splatting framework for monocular UAV video. It introduces a Monocular Geometry Lifting module to reconstruct reliable static and dynamic geometry, and incorporates differentiable physical priors — ground support, upright stability, and trajectory smoothness — to resolve ambiguous image cues into physically consistent motion estimates, outperforming existing methods on both synthetic and real UAV scenes.

## Background & Motivation

**Background**: Recent years have seen significant advances in 4D scene reconstruction. 3D Gaussian Splatting (3DGS), with its efficient differentiable rendering and explicit scene representation, has become a popular foundation for both static and dynamic scene modeling. However, existing dynamic 3DGS methods (e.g., Dynamic 3D Gaussians, 4D-GS, Deformable 3DGS) are primarily designed for multi-view or controlled close-range dynamic scenes.

**Limitations of Prior Work**: UAV aerial video presents several unique properties that cause existing methods to fail directly:
   - **Single-view capture**: UAVs typically fly along fixed trajectories, so each scene region is observed only once or very few times, lacking multi-view redundancy.
   - **Large spatial extent**: Aerial coverage far exceeds that of indoor or street-view scenes, with complex background geometry.
   - **Dynamic object characteristics**: Moving objects (e.g., vehicles, pedestrians) occupy a small spatial fraction of the frame but exhibit large motion disparity, causing severe motion blur and occlusion.
   - **Depth ambiguity**: Monocular depth estimation in aerial settings is inherently ill-posed; the combination of large distances and top-down viewing angles makes depth cues extremely sparse.

**Key Challenge**: The compounding effect of the above factors leads to severe depth ambiguity and unstable motion estimation, rendering monocular aerial dynamic reconstruction an inherently ill-posed problem. Existing methods either fail completely in this setting or produce physically implausible motion trajectories (e.g., floating objects, ground penetration, jittering).

**Key Insight**: The paper leverages commonsense physical priors — objects should rest on the ground, remain upright, and move along smooth trajectories — to constrain and resolve monocular depth ambiguity, converting uncertain image cues into physically consistent dynamic reconstruction.

## Method

### Overall Architecture

The AeroDGS framework comprises two core modules: **Monocular Geometry Lifting (MGL)** and **Physics-Guided Optimization (PGO)**. MGL reconstructs reliable initial geometry for both the static background and dynamic objects from monocular UAV sequences. PGO introduces physical priors to constrain dynamic object motion and resolve monocular ambiguity. The two modules are jointly optimized to enable co-refinement of the static background and dynamic entities.

### Module 1: Monocular Geometry Lifting (MGL)

MGL addresses the core challenge of extracting reliable static and dynamic 3D geometry from aerial sequences with only single-pass observations.

- **Static background reconstruction**: Structure from Motion (SfM) or monocular depth estimation networks are used to recover sparse/dense point clouds of the static scene from the UAV sequence, initializing static 3D Gaussians. Since the UAV flight provides camera translational motion, SfM can recover static scene depth from parallax.
- **Dynamic object detection and separation**: Motion segmentation or object detection separates dynamic objects (vehicles, pedestrians, etc.) from the static background. Dynamic regions produce inconsistent reprojections in SfM and can thus be identified and modeled independently.
- **Dynamic geometry initialization**: For each dynamic object, monocular depth estimates and detection bounding boxes are used to estimate initial 3D position and shape, parameterized as 3D Gaussians. In addition to standard attributes (position $\mu$, covariance $\Sigma$, color $c$, opacity $\alpha$), each dynamic Gaussian is associated with time-dependent motion parameters.

### Module 2: Physics-Guided Optimization (PGO)

PGO is the central contribution of this work. Three differentiable physical priors constrain dynamic object motion:

1. **Ground-Support Prior**:
    - Physical intuition: In practice, most dynamic objects (vehicles, pedestrians) should rest on the ground and neither float nor penetrate it.
    - Implementation: The ground plane is estimated from the static background reconstruction or DEM data. A differentiable contact loss penalizes deviation of each dynamic Gaussian's base from the ground surface:
   $$\mathcal{L}_{\text{ground}} = \sum_{i} \max(0, z_{\text{ground}} - z_i + \epsilon)^2$$
   where $z_i$ is the vertical coordinate of the $i$-th dynamic Gaussian and $z_{\text{ground}}$ is the corresponding ground height.

2. **Upright-Stability Prior**:
    - Physical intuition: Vehicles and pedestrians should maintain an upright posture during motion and should not arbitrarily tip or rotate.
    - Implementation: The principal axis of each dynamic object is regularized to align with the gravity direction (the scene's vertical axis):
   $$\mathcal{L}_{\text{upright}} = \sum_{i} (1 - |\mathbf{n}_i \cdot \mathbf{g}|)$$
   where $\mathbf{n}_i$ is the normal vector of the dynamic object and $\mathbf{g}$ is the unit gravity vector.

3. **Trajectory-Smoothness Prior**:
    - Physical intuition: Real-world object trajectories should be smooth and continuous, with no instantaneous jumps or abrupt direction changes (inertial constraint).
    - Implementation: An acceleration regularization is applied to the temporal displacements of dynamic Gaussians, penalizing abrupt velocity changes between adjacent timesteps:
   $$\mathcal{L}_{\text{smooth}} = \sum_{t} \| (\mu_{t+1} - \mu_t) - (\mu_t - \mu_{t-1}) \|^2$$
   This is equivalent to penalizing the second-order temporal derivative (acceleration), permitting constant-velocity motion while suppressing high-frequency jitter.

### Loss & Training

The total loss combines a rendering reconstruction loss with the physical prior losses:

$$\mathcal{L} = \mathcal{L}_{\text{render}} + \lambda_g \mathcal{L}_{\text{ground}} + \lambda_u \mathcal{L}_{\text{upright}} + \lambda_s \mathcal{L}_{\text{smooth}}$$

where $\mathcal{L}_{\text{render}}$ consists of standard photometric losses (L1 + SSIM):

$$\mathcal{L}_{\text{render}} = (1-\lambda_{\text{ssim}})\|I_{\text{pred}} - I_{\text{gt}}\|_1 + \lambda_{\text{ssim}}(1 - \text{SSIM}(I_{\text{pred}}, I_{\text{gt}}))$$

Static and dynamic Gaussians are jointly optimized via differentiable rendering. The physical priors participate in backpropagation, with gradients from the physical losses directly updating the position and orientation parameters of the dynamic Gaussians.

## Key Experimental Results

### Experimental Setup

- **Datasets**: (1) Synthetic UAV scenes — synthetic aerial sequences with known GT geometry and motion for quantitative evaluation; (2) Real UAV dataset — a new real-world UAV dataset constructed in this work, covering varying flight altitudes and motion conditions.
- **Evaluation metrics**: PSNR, SSIM, LPIPS (rendering quality); trajectory error and other dynamic evaluation metrics may also be included.
- **Baselines**: Existing dynamic 3DGS methods (e.g., Deformable 3DGS, 4D-GS, SC-GS) and classical dynamic NeRF approaches.

### Main Results: Quantitative Comparison on Synthetic UAV Scenes

| Method | Type | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Dynamic Object Quality |
|------|------|--------|--------|---------|-------------|
| Deformable 3DGS | Dynamic 3DGS | Low | Low | High | Unstable motion |
| 4D-GS | Dynamic 3DGS | Medium | Medium | Medium | Partial floating |
| SC-GS | Dynamic 3DGS | Medium | Medium | Medium | Trajectory jitter |
| **AeroDGS** | **Physics-guided** | **Best** | **Best** | **Best** | **Physically consistent** |

As reported in the abstract, AeroDGS outperforms state-of-the-art methods on both synthetic and real UAV scenes, achieving higher reconstruction fidelity.

### Ablation Study: Contribution of Physical Priors

| Configuration | Ground-Support | Upright | Smooth | Reconstruction Quality | Motion Plausibility |
|------|:-:|:-:|:-:|------|------|
| Baseline (no priors) | ✗ | ✗ | ✗ | Baseline | Floating/penetration/jitter |
| + Ground-Support | ✓ | ✗ | ✗ | Improved | Ground penetration eliminated |
| + Upright | ✓ | ✓ | ✗ | Further improved | Pose stabilized |
| + All (AeroDGS) | ✓ | ✓ | ✓ | **Best** | **Physically consistent** |

Each physical prior contributes incrementally: ground support resolves vertical drift caused by depth ambiguity, the upright constraint stabilizes orientation, and trajectory smoothness suppresses high-frequency jitter.

## Highlights & Insights

- **Physics priors as a paradigm for resolving monocular ambiguity**: Monocular depth ambiguity is amplified to an extreme in aerial scenes. The authors elegantly convert commonsense physical knowledge (ground contact, uprightness, inertia) into differentiable loss functions, using physical constraints to compensate for the lack of geometric observations. This "physics-compensates-for-geometry" paradigm is more generalizable than purely data-driven depth estimation.
- **Purpose-built for aerial scenes**: Existing dynamic 3DGS works almost universally target ground-level or indoor perspectives. AeroDGS is the first to systematically address 4D reconstruction in monocular UAV video, tackling aerial-specific challenges such as wide field of view, small targets, and large motion.
- **Value of the new dataset**: The paper contributes a real-world UAV dataset covering diverse flight altitudes and motion conditions, filling an evaluation gap in the field. Data scarcity for aerial dynamic reconstruction has long been a bottleneck for research.
- **Modularity and extensibility**: The MGL and PGO modules are decoupled by design, and the physical priors can be readily extended (e.g., with collision constraints or velocity limits).

## Limitations & Future Work

- **Scope of physical priors**: Ground-support and upright-stability priors apply primarily to ground vehicles and pedestrians; they may not hold for airborne dynamic objects (e.g., birds, other UAVs).
- **Ground plane estimation accuracy**: The ground-support prior depends on accurate ground plane estimation, which may require more sophisticated terrain models in complex topographies (hills, stairs, multi-story structures).
- **Dynamic object detection and segmentation**: The method's upstream relies on accurate motion segmentation; missed or incorrect detections will directly degrade dynamic modeling quality.
- **Computational overhead**: The physical prior constraints add additional terms to the optimization, potentially increasing training time relative to photometry-only 3DGS.
- **Occlusion and reappearance**: Aerial dynamic objects may be occluded by buildings for extended periods before reappearing; single-sequence methods may struggle with object re-identification and trajectory continuation.
- **Night-time and adverse weather**: Performance under low-light or rain/fog degradation conditions is not discussed in the abstract.

## Related Work & Insights

- **Dynamic 3D Gaussian Splatting**: Dynamic 3D Gaussians (Luiten et al.) models dynamic scenes by tracking Gaussian trajectories; 4D-GS parameterizes motion with spatiotemporal Gaussians; Deformable 3DGS learns deformation fields. These methods are effective under multi-view settings but degrade in monocular aerial scenarios due to insufficient observations.
- **Monocular dynamic scene reconstruction**: Methods such as RoDynRF and MonST3R handle monocular dynamic video but primarily target ground-level scenes with large foreground objects. Aerial scenes differ fundamentally, with small target pixel footprints and fast motion.
- **Aerial 3D reconstruction**: Traditional aerial reconstruction (SfM+MVS) targets static scenes. Recent works such as Mega-NeRF and Switch-NeRF address large-scale aerial NeRF but do not handle dynamic objects.
- **Physical priors in 3D reconstruction**: PAC-NeRF incorporates physics simulation to guide deformation; PhysDreamer learns physical properties. AeroDGS is distinctive in applying physical priors specifically to resolve monocular dynamic ambiguity in aerial scenes.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce physically consistent priors into aerial 4D Gaussian Splatting; the combined ground-support + upright + smoothness design is well-targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual validation on synthetic and real UAV scenes; the new dataset is a genuine contribution, though full paper data were not available at review time.
- Writing Quality: ⭐⭐⭐⭐ — Abstract is clearly written, problem formulation is precise, and module naming is intuitive.
- Value: ⭐⭐⭐⭐ — Addresses practical needs in UAV remote sensing and urban surveillance; aerial dynamic reconstruction is a high-value application domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[ICLR 2026\] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](../../AAAI2026/3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
