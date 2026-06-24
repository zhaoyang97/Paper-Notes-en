---
title: >-
  [Paper Note] GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments
description: >-
  [ICCV 2025][3D Vision][Continual Learning] This paper presents GaussianUpdate, the first method to integrate 3D Gaussian representations with continual learning. It achieves real-time rendering and change visualization in temporally varying scenes through a three-stage update strategy (appearance update → geometric layout update → joint refinement) and visibility-aware generative replay.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Continual Learning"
  - "3D Gaussian Splatting"
  - "Scene Update"
  - "Change Detection"
  - "Generative Replay"
date: 2026-05-08
content_hash: 27eab535403bc5d1
---

# GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments

**Conference**: ICCV 2025  
**arXiv**: [2508.08867](https://arxiv.org/abs/2508.08867)  
**Code**: [Project Page](https://zju3dv.github.io/GaussianUpdate)  
**Area**: 3D Vision  
**Keywords**: Continual Learning, 3D Gaussian Splatting, Scene Update, Change Detection, Generative Replay

## TL;DR

This paper presents GaussianUpdate, the first method to integrate 3D Gaussian representations with continual learning. It achieves real-time rendering and change visualization in temporally varying scenes through a three-stage update strategy (appearance update → geometric layout update → joint refinement) and visibility-aware generative replay.

## Background & Motivation

Real-world scenes change over time (object addition/removal, illumination variation), yet existing methods suffer from fundamental limitations:

**High retraining cost**: Retraining from scratch upon every change causes storage to grow unboundedly over time.

**Catastrophic forgetting**: Directly fine-tuning on new data causes the model to lose information about historical scenes.

**Inapplicability of dynamic modeling**: 4D methods assume continuous dynamic change and cannot handle discrete events such as sudden object appearance or disappearance.

**Inefficiency of NeRF-based methods**: Methods such as CLNeRF cannot localize changed regions and suffer from low rendering efficiency.

The key advantage of GaussianUpdate lies in the explicit 3D Gaussian representation, which naturally supports precise localization and management of changed Gaussian primitives.

## Method

### Global Appearance Model

A 4D hash grid $\mathbf{H}$ and a compact MLP $\mathbf{F}$ are used to model illumination changes:

$$\{\Delta s_i^t, \Delta Y_i^t\} = F(H\{\mu_i, t\})$$

Time-dependent incremental scaling and spherical harmonics attributes are inferred for each Gaussian.

### Three-Stage Update Strategy

**Stage 1: Global Appearance Update**

Illumination changes are learned in layout-stable regions:
1. The previous-timestep image $I_f^{t-1}$ is rendered and compared with the new image $I_f^t$.
2. SAM is applied to obtain instance segmentation masks; IoU is computed to identify layout-stable regions $M_f$.
3. The hash encoding is optimized exclusively within stable regions:
$$\mathcal{L}_{st} = (1-\lambda)\mathcal{L}_1 \cdot \mathcal{M}_f + \lambda\mathcal{L}_{D-SSIM} \cdot \mathcal{M}_f$$

**Stage 2: Geometric Layout Update**

With the appearance model fixed, object addition and removal are handled:
- **New objects**: New Gaussian primitives $G_{add}$ are initialized from COLMAP.
- **Disappeared objects**: A learned removal factor $m$ is applied via a steep sigmoid activation:
$$\psi(m) = \frac{1}{1 + e^{-1000m}}$$

- A regularization term drives $\psi(m)$ toward binary values:
$$\mathcal{L}_{reg} = \lambda_2(1-\psi(m))\psi(m) + \lambda_3 BCE(\psi(m), 1)$$

- DBSCAN is applied to filter sparse outliers.

**Stage 3: Joint Refinement**

After pruning, the appearance model and new Gaussian parameters are jointly optimized. Importance-based pruning reduces memory overhead:
$$v_i = \max_{n \in N_1}(\alpha_i^n T_i^n)$$

### Visibility-Aware Continual Learning

- **Visibility pool** $P_v$: Stores the state (active/inactive) of all Gaussian primitives across the entire temporal sequence.
- **Generative replay**: The efficiency of Gaussian rendering is exploited by re-rendering past scenes using historical camera poses as training data.
- No additional image storage is required; only camera extrinsics need to be recorded.

## Key Experimental Results

### Comparison on Benchmark Datasets

| Method | Breville PSNR↑ | Kitchen PSNR↑ | Living Room PSNR↑ |
|--------|---------------|---------------|-------------------|
| Baseline | 20.66 | 16.44 | 17.28 |
| CLNeRF | 28.02 | 28.40 | 24.58 |
| 4DGS | 28.92 | 27.03 | 24.41 |
| **Ours** | **30.11** | **28.02** | **26.10** |
| Upper Bound | 30.36 | 27.99 | 26.22 |

GaussianUpdate surpasses both CLNeRF and 4DGS across all scenes and approaches the upper bound (independent training per timestep).

| Method | Community PSNR↑ | Community SSIM↑ |
|--------|-----------------|-----------------|
| CLNeRF | 22.88 | 0.629 |
| 4DGS | 22.99 | 0.711 |
| **Ours** | **23.88** | **0.764** |

### Key Findings

- **Real-time rendering**: The explicit 3DGS representation guarantees real-time rendering capability.
- **Change visualization**: The visibility pool enables scene rendering at arbitrary timesteps and explicit change visualization.
- Proximity to the upper bound demonstrates effective information retention.

## Highlights & Insights

1. **Three-stage decoupled strategy**: Avoids ambiguity arising from the entanglement of appearance and geometric changes.
2. **Explicit change management**: Three change types (illumination, removal, addition) are handled separately with dedicated modules.
3. **Storage-free generative replay**: Historical scenes are replayed by leveraging Gaussian rendering efficiency; only camera extrinsics need to be stored.
4. **Visibility pool design**: A natural advantage of explicit representation, enabling primitive-level scene management.

## Limitations & Future Work

- The COLMAP-based addition strategy is sensitive to the quality of feature matching.
- SAM-based detection of layout-stable regions may be imprecise in certain cases.
- Only discrete timestep updates are supported; continuous dynamics are not addressed.
- The hash grid incurs growing storage overhead as the number of timesteps increases.

## Related Work & Insights

- CLNeRF: Continual learning for NeRF.
- 4DGS: Dynamic 3D Gaussian Splatting.
- NeRF-w: Appearance embedding for illumination variation handling.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First integration of 3DGS with continual learning)
- Technical Depth: ⭐⭐⭐⭐⭐ (Three-stage strategy + visibility pool + generative replay)
- Experimental Thoroughness: ⭐⭐⭐⭐ (10 scenes with multiple baselines)
- Value: ⭐⭐⭐⭐⭐ (High practical value: real-time rendering + change visualization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[ICCV 2025\] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update](2d_gaussian_splattingbased_sparseview_transparent_object_dep.md)
- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](../../CVPR2025/3d_vision/wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)

</div>

<!-- RELATED:END -->
