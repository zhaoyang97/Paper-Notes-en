---
title: >-
  [Paper Note] Sequential Gaussian Avatars with Hierarchical Motion Context
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes SeqAvatar, which leverages explicit 3DGS representations combined with hierarchical motion context (coarse-grained skeletal motion + fine-grained per-poin…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Human Avatars"
  - "Non-Rigid Deformation"
  - "Motion Context"
  - "SMPL"
date: 2026-05-08
content_hash: af73b261f63125a3
---

# Sequential Gaussian Avatars with Hierarchical Motion Context

**Conference**: ICCV 2025
**arXiv**: [2411.16768](https://arxiv.org/abs/2411.16768)  
**Code**: [Project Page](https://zezeaaa.github.io/projects/SeqAvatar/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Human Avatars, Non-Rigid Deformation, Motion Context, SMPL

## TL;DR

This paper proposes SeqAvatar, which leverages explicit 3DGS representations combined with hierarchical motion context (coarse-grained skeletal motion + fine-grained per-point velocity) to model motion-correlated appearance changes in human avatars. Spatio-temporal multi-scale sampling further enhances the robustness of motion conditioning. SeqAvatar achieves state-of-the-art rendering quality across multiple datasets while maintaining real-time rendering speed.

## Background & Motivation

3DGS-based animatable human avatar reconstruction has made significant progress in recent years, yet a core challenge persists: **insufficient modeling of non-rigid deformation**, manifesting in the following aspects:

**Many-to-one pose-to-appearance mapping**: The same body pose can correspond to different appearances under varying motion states (e.g., inertial swinging of a skirt). Existing methods rely solely on the spatial pose of the current frame and cannot disambiguate such cases.

**Lack of local detail**: Current methods primarily rely on global skeletal information to predict deformation, failing to model fine-grained motion in regions far from bones (e.g., flowing garments, hair).

**Limitations of existing sequence-based modeling**: NeRF-based methods such as Dyco attempt to model motion sequences using body pose residuals, but the global nature of pose sequences limits their ability to capture finer-grained motion details, and they cannot fully exploit the explicit point representation of 3DGS.

Core insight: **The explicit point representation of 3DGS enables per-point motion modeling** — independent velocity vectors can be computed for each Gaussian primitive, capturing local detail variations beyond skeletal motion.

## Method

### Overall Architecture

SeqAvatar introduces hierarchical motion context conditioning on top of the standard SMPL+LBS+3DGS pipeline. The workflow proceeds as follows: (1) initialize canonical-space Gaussians from SMPL template vertices → (2) construct coarse skeletal motion condition $f_{\Delta\mathcal{P}}$ and fine per-point velocity condition $f_\mathcal{V}$ → (3) predict non-rigid deformation via MLP → (4) apply LBS rigid transformation to observation space → (5) render via Gaussian splatting.

### Key Designs

1. **Coarse Skeleton Motion**:

    - For target frame $t$, uniformly-spaced historical frames are sampled: $\mathcal{T} = \{t-s, t-2s, ..., t-Ls\}$
    - Pose differences between adjacent frames are computed in axis-angle form: $\Delta\mathcal{P} = \{\Delta P^t = \delta(P^t, P^{t-s}) | t \in \mathcal{T}\}$, where $P \in \mathbb{R}^{K \times 3}$ denotes body pose
    - These differences are encoded into fixed-dimensional embeddings via MLP: $f_{\Delta\mathcal{P}} = \mathcal{E}_{\Delta\mathcal{P}}(\Delta\mathcal{P}) \in \mathbb{R}^{32}$
    - **Design Motivation**: Compared to using the current-frame pose directly, pose difference sequences capture the temporal dynamics of motion, enabling disambiguation between different appearance states under identical poses

2. **Fine Vertex Motion**:

    - Per-point velocity for each Gaussian primitive cannot be computed directly (Gaussian positions change continuously during optimization, and non-rigid transformations introduce circular dependencies)
    - Solution: A **motion template field** $\mathcal{F}_\mathbf{V} = \{\mathbf{V}_i\}_{i=1}^{N}$ is constructed to store velocity for each SMPL template vertex
    - SMPL vertex velocity is computed by first transforming template vertices $\mathbf{T}$ to observation space via standard LBS: $\mathbf{T_o}^t = \mathbf{LBS}(\mathbf{T}, \mathbf{B}^t, \mathbf{W})$, then computing $\mathbf{V}^t = (\mathbf{T_o}^t - \mathbf{T_o}^{t-s}) / s$
    - Each Gaussian primitive retrieves its velocity from the motion template field via KNN sampling
    - **Key Advantage**: By exploiting the explicit point representation of 3DGS, independent local motion information is provided to each point, capturing motion in regions not covered by skeletal kinematics (e.g., flowing skirts)

3. **Spatio-Temporal Multi-Scale Sampling (STMS)**:

   **Spatial dimension**: For each Gaussian primitive $\mathcal{G}_i$, velocities of $\tau$ nearest-neighbor template vertices are sampled as input, learning a motion embedding for the local region:
    $e_i^t = \mathcal{E}_{knn}(\{\mathbf{V}_j^t\}), \quad j \in \mathbf{KNN}(\mathbf{T}, \mathbf{x}_i)$

   **Temporal dimension**: Multi-scale sequences with increasing intervals are used to simultaneously capture overall motion trends and inter-frame details:
    $\mathcal{S} = \{s = s_0 + i\Delta s\}_{i=0}^{i=m}$

   Multi-scale skeletal and per-point motion conditions are concatenated and fed into their respective encoders:
    $f_{\Delta\mathcal{P}} = \mathcal{E}_{\Delta\mathcal{P}}(\{\Delta\mathcal{P}_s\}), \quad f_\mathcal{V} = \mathcal{E}_\mathcal{V}(\{\mathcal{V}_s\}), \quad s \in \mathcal{S}$

   **Design Motivation**: Small intervals capture fine-grained inter-frame changes while large intervals capture overall motion trends; the two are complementary and improve generalization

4. **Non-Rigid Deformation Prediction**: All motion conditions are combined, and an MLP predicts position, scale, and rotation offsets for each Gaussian:
    $\delta\mathbf{x}, \delta\mathbf{s}, \delta\mathbf{r} = \mathcal{E}_{non-rigid}(\mathbf{x}, P, f_{\Delta\mathcal{P}}, f_\mathcal{V})$

   Canonical-space Gaussians are then updated as: $\mathbf{x'} = \mathbf{x} + \delta\mathbf{x}$, $\mathbf{s'} = \mathbf{s} + \delta\mathbf{s}$, $\mathbf{r'} = \mathbf{r} \cdot \delta\mathbf{r}$

### Loss & Training

The composite loss function is:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{color} + \lambda_2 \mathcal{L}_{ssim} + \lambda_3 \mathcal{L}_{lpips} + \mathcal{L}_{mask}$$

- $\mathcal{L}_{color}$: L1 color loss
- $\mathcal{L}_{ssim}$: SSIM structural similarity loss
- $\mathcal{L}_{lpips}$: LPIPS perceptual loss
- $\mathcal{L}_{mask}$: L2 loss between rendered alpha and body mask

Additional regularization terms $\mathcal{L}_{isopos}$ and $\mathcal{L}_{isocov}$ constrain the position and covariance of Gaussian primitives. A pose refinement MLP $\mathcal{E}_{pose}$ is also employed to improve SMPL pose estimation.

LBS weights are updated via learned offsets: $\omega_k(\mathbf{x}) = \omega_k^{SMPL}(\mathbf{x}) + \mathcal{E}_{lbs}(\mathbf{x})$

## Key Experimental Results

### Main Results

DNA-Rendering dataset (average over 6 scenes):

| Method | PSNR↑ | SSIM↑ | LPIPS*↓ |
|--------|-------|-------|---------|
| 3DGS-Avatar | 28.63 | 0.9565 | 41.43 |
| GART | 28.99 | 0.9597 | 44.55 |
| GauHuman | 29.55 | 0.9600 | 40.96 |
| **SeqAvatar** | **32.05** | **0.9711** | **30.91** |

I3D-Human dataset (Novel View, average over 4 scenes):

| Method | PSNR↑ | SSIM↑ | LPIPS*↓ | FPS |
|--------|-------|-------|---------|-----|
| 3DGS-Avatar | 30.86 | 0.9608 | 34.07 | Real-time |
| Dyco (NeRF) | 31.06 | 0.9607 | 30.71 | ~0.7 |
| GauHuman | 30.13 | 0.9562 | 45.37 | Real-time |
| **SeqAvatar** | **32.24** | **0.9664** | **29.78** | ~45 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS*↓ | Note |
|---------------|-------|-------|---------|------|
| (a) Baseline (no non-rigid deformation) | 29.76 | 0.9569 | 38.35 | LBS only |
| (b) + Standard non-rigid MLP + current pose | 31.05 | 0.9617 | 34.35 | + pose conditioning |
| (c) + $\Delta\mathcal{P}$ skeletal motion | 31.89 | 0.9645 | 32.17 | + coarse temporal |
| (d) + $\mathcal{V}$ per-point velocity | 32.01 | 0.9651 | 31.23 | + fine-grained motion |
| **(e) + STMS (full)** | **32.24** | **0.9664** | **29.78** | Full method |

### Key Findings

- The skeletal motion condition $\Delta\mathcal{P}$ yields the largest performance jump (31.05→31.89 PSNR), demonstrating that temporal motion information is critical for non-rigid deformation modeling
- The per-point velocity condition $\mathcal{V}$ further improves local region detail (31.89→32.01), particularly in regions far from bones such as flowing garments
- STMS multi-scale sampling provides an additional 0.23 dB PSNR gain and enhances generalization
- SeqAvatar maintains real-time rendering (~45 FPS on I3D-Human), approximately 60× faster than NeRF-based Dyco (~0.7 FPS)
- SeqAvatar outperforms the best 3DGS baseline GauHuman by ~2.5 dB PSNR on DNA-Rendering
- Out-of-distribution pose animation (trained on one sequence, rendered on unseen poses from another) also performs well

## Highlights & Insights

- **Fully exploits the advantages of 3DGS explicit representation**: Per-point velocity is a capability unique to 3DGS; NeRF cannot naturally achieve this due to its implicit representation
- **Elegant design of the motion template field**: Gaussian primitive velocities are indirectly provided via SMPL template vertex velocities, avoiding circular dependencies and optimization instability
- **Multi-scale temporal sampling**: Analogous to multi-scale receptive fields in convolution, this captures motion information at different temporal frequencies
- **Excellent performance-speed trade-off**: Rendering quality surpasses Dyco (NeRF) while maintaining a 60× speed advantage

## Limitations & Future Work

- The Gaussian representation may introduce slight blurring in rendering; NeRF's ray integration tends to produce sharper results
- Local velocity cues are derived from the coarse SMPL model rather than dense surface tracking, potentially limiting the precision of fine-grained garment deformation
- The method depends on the accuracy of SMPL initialization and pose estimation
- Validation under monocular video input has not been performed (all experiments use multi-view inputs)
- Training efficiency and memory consumption for long sequences warrant further optimization

## Related Work & Insights

- Core distinction from Dyco (NeRF + pose sequences): SeqAvatar additionally utilizes per-point velocity beyond pose residuals, and achieves real-time rendering via 3DGS
- Unlike 3DGS-based methods such as 3DGS-Avatar and GauHuman, SeqAvatar incorporates temporal motion information
- The motion template field concept is generalizable to other 3DGS-based dynamic scene reconstruction tasks requiring per-point motion modeling
- The multi-scale temporal sampling strategy can serve as a general module for other generative models conditioned on motion sequences

## Rating

- Novelty: ⭐⭐⭐⭐ The hierarchical motion context design is novel; the motion template field elegantly resolves circular dependencies
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on three datasets (DNA-Rendering / I3D-Human / ZJU-MoCap) covering both novel view and novel pose settings
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, figures are intuitive, and mathematical derivations are complete
- Value: ⭐⭐⭐⭐ Provides an effective motion conditioning enhancement scheme for 3DGS-based human avatar modeling

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/3d_vision/motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[ICCV 2025\] Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion](global_motion_corresponder_for_3d_point-based_scene_interpolation_under_large_mo.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)

</div>

<!-- RELATED:END -->
