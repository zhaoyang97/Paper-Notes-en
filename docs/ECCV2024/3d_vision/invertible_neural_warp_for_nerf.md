---
title: >-
  [Paper Note] Invertible Neural Warp for NeRF
description: >-
  [ECCV 2024][3D Vision][NeRF] This paper proposes using Invertible Neural Networks (INNs) to over-parameterize the rigid transformation function of camera poses, significantly improving pose estimation accuracy and reconstruction quality in joint NeRF optimization. It demonstrates that invertibility is a critical constraint when using MLPs to model rigid warps.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "NeRF"
  - "Pose Estimation"
  - "Invertible Neural Networks"
  - "Over-parameterization"
  - "Joint Optimization"
date: 2026-05-08
content_hash: 4a36e83122b1e9d6
---

# Invertible Neural Warp for NeRF

**Conference**: ECCV 2024  
**arXiv**: [2407.12354](https://arxiv.org/abs/2407.12354)  
**Code**: [Project Page](https://sfchng.github.io/ineurowarping-github.io/)  
**Area**: 3D Vision  
**Keywords**: NeRF, Pose Estimation, Invertible Neural Networks, Over-parameterization, Joint Optimization

## TL;DR

This paper proposes using Invertible Neural Networks (INNs) to over-parameterize the rigid transformation function of camera poses, significantly improving pose estimation accuracy and reconstruction quality in joint NeRF optimization. It demonstrates that invertibility is a critical constraint when using MLPs to model rigid warps.

## Background & Motivation

**Background**: NeRF achieves high-quality novel view synthesis via volume rendering but requires precisely known camera poses. Methods like BARF, NeRFmm, and GARF enable joint optimization of poses and NeRF, usually employing a compact 6D $SE(3)$ representation for camera poses.

**Limitations of Prior Work**: Compact $SE(3)$ parameterization suffers from narrow basins of convergence during joint optimization with NeRF, making it easy to fall into local minima and leading to inaccurate pose estimation.

**Key Challenge**: The deep learning community has demonstrated that over-parameterization can improve optimization convergence. However, simply using an MLP to over-parameterize poses (the naive approach) completely fails in practice—with a 0% convergence success rate. The reason is that standard MLPs do not guarantee the invertibility (bijectivity) of the transformation.

**Goal**: How to correctly over-parameterize camera poses in NeRF to exploit the optimization benefits of over-parameterization.

**Key Insight**: Modeling camera poses as rigid warp functions of rays rather than global rotation and translation, and employing an INN to structurally guarantee the invertibility of the warp.

**Core Idea**: Invertibility is a necessary constraint for MLP-based over-parameterized rigid warps—INNs naturally guarantee bijectivity at the architectural level without requiring auxiliary networks.

## Method

### Overall Architecture

Traditional BARF-like methods represent the pose of each frame as $P = (\mathbf{R}_t, \mathbf{t}_t) \in SE(3)$, which is used to transform rays from the camera coordinate system to the world coordinate system. This work replaces this with a globally shared INN $h_{\mathbf{\Theta}_\mathcal{W}}$ integrated with a per-frame learnable latent code $\Phi_t \in \mathbb{R}^D$:

$$h(\mathbf{r}^{(C)}; \mathbf{\Theta}_\mathcal{W}, \Phi_t): \mathbb{R}^{3+D} \to \mathbb{R}^3$$

The INN takes the pixel position in the camera coordinate system $\mathbf{x}_{i,t}^{(C)}$ and the frame-specific latent code $\Phi_t$ as input, and outputs the corresponding position in the world coordinate system $\mathbf{x}_{i,t}^{(W)}$.

The final optimization problem is defined as:

$$\min_{\Phi_t, \mathbf{\Theta}_\mathcal{W}, \mathbf{\Theta}_{rgb}} \sum_{t=1}^{T} \sum_{\mathbf{u}} \|\hat{\mathcal{I}}(h(\mathbf{r}^{(c)}; \mathbf{\Theta}_\mathcal{W}, \Phi_t); \mathbf{\Theta}_{rgb}) - \mathcal{I}_i(\mathbf{u})\|_2^2 + \lambda \mathcal{L}_{rigid}$$

### Key Designs

1. **Invertible Neural Network (INN) Pose Representation**: Compared to the traditional 6-parameter $SE(3)$ representation, the INN treats each pixel as an independent ray, learning the mapping from camera coordinates to world coordinates. The INN achieves bijective mapping by combining affine coupling blocks, where each block splits the input into two parts—the first part remains unchanged and parameterizes the transformation of the second part.

Key Advantages:
- Architectural guarantee of invertibility, bypassing the need for auxiliary networks.
- The same INN is shared across all frames, ensuring parameter efficiency.
- The INN predicts homeomorphisms, which are more flexible than rigid transformations and provide smoother optimization trajectories.

Design Motivation: While $SE(3)$ parameterization is precise, its basin of convergence is small. MLP over-parameterization offers a better optimization landscape but must guarantee bijectivity—otherwise, a world point could map to multiple camera points, leading to optimization collapse.

2. **Rigidity Prior Constraint**: Since each pixel is processed independently, the INN outputs do not naturally satisfy global rigid motion. To address this, a rigidity regularization is introduced, solving for a closed-form rigid registration using the known camera-world correspondences:

$$\mathcal{L}_{rigid} = \min_{T^*} \sum_{i=1}^{L} \|\mathbf{x}_{i,t}^{(C)} - T^* \circ \mathbf{x}_{i,t}^{(W)}\|_2^2$$

Direct Linear Transform (DLT) is used to solve for the homography in 2D experiments, and the Umeyama algorithm is used to solve for $SE(3)$ in 3D experiments. This constraint pulls the output toward a globally consistent rigid transformation without destroying the flexibility of the INN.

3. **Implicit vs. Explicit Invertibility Comparison**: The paper systematically compares three over-parameterization strategies:

    - **Naive MLP**: Uses only a forward network $h_{fwd}$ with no invertibility guarantee — 0% success rate.
    - **Implicit-Invertible MLP**: Dual networks $h_{fwd} + h_{bwd}$ approximating invertibility via a consistency loss $\|x^{(C)} - \hat{x}^{(C)}\|_2^2$ — 65% success rate, but doubles the computational cost.
    - **Explicit-Invertible INN** (Ours): Invertibility guaranteed by architecture — 75% success rate, with no extra computational overhead.

### Loss & Training

Total Loss = NeRF Photometric Loss + Rigidity Prior:

$$\mathcal{L} = \sum_{t=1}^{T} \sum_{\mathbf{u}} \|\hat{\mathcal{I}} - \mathcal{I}_i\|_2^2 + \lambda \mathcal{L}_{rigid}$$

- Using the Adam optimizer, the learning rate for $\mathbf{\Theta}_{rgb}$ decays from $1\times10^{-3}$ to $3\times10^{-4}$.
- The learning rate for $\mathbf{\Theta}_\mathcal{W}$ decays from $5\times10^{-4}$ to $1\times10^{-6}$.
- 2048 rays are sampled per step, training for 200K iterations.
- Conducive coarse-to-fine positional encoding scheduling from BARF is adopted.
- The INN architecture employs NDR-INN with a latent code dimension of $D = 16$.

## Key Experimental Results

### Main Results — 2D Planar Alignment (Statistics over 20 homographies)

| Method | Corner Error(px)↓ | Patch PSNR↑ | Success Rate↑ |
|------|-------------------|-------------|---------|
| BARF | 29.63 ± 28.18 | 28.94 ± 4.38 | 0.30 |
| Naive MLP | 85.59 ± 30.31 | 25.86 ± 2.07 | **0.00** |
| Implicit-Invertible | 13.92 ± 22.93 | 33.70 ± 3.93 | 0.65 |
| **INN (Ours)** | **4.70 ± 6.47** | **34.71 ± 2.37** | **0.75** |

### Main Results — LLFF Real Forward-Facing Scenes (Average over 8 scenes)

| Method | Rotation(°)↓ | Translation(×100)↓ | PSNR (before)↑ | PSNR (after)↑ |
|------|-------------|---------------------|-----------|-----------|
| BARF | 0.90 | 0.40 | 17.00 | 23.82 |
| L2G | 0.48 | 0.30 | 17.99 | 24.35 |
| **INN (Ours)** | **0.31** | **0.24** | **19.31** | 24.28 |

### Main Results — DTU 360° Scenes (Average over 14 scenes, 15° initial error)

| Method | Rotation(°)↓ | Translation(×100)↓ | Depth Error↓ | Chamfer↓ |
|------|-------------|---------------------|-------------|----------|
| BARF | 2.52 | 7.07 | 0.20 | 6.35 |
| L2G | 4.08 | 11.67 | 0.22 | 6.53 |
| **INN (Ours)** | **1.17** | **3.07** | **0.13** | **4.89** |

### Ablation Study — Key Role of Invertibility

| Method | Invertibility Guarantee | Success Rate (2D) | Extra Computation |
|------|-----------|-----------|---------|
| Naive MLP | None | 0% | None |
| Implicit-Invertible | Approximate (Dual Networks) | 65% | ×2 |
| Explicit-Invertible INN | Architectural Guarantee | **75%** | None |

### Key Findings

- Over-parameterization alone is insufficient—invertibility is a necessary requirement for learning rigid warps.
- Compared to BARF, the INN improves rotation accuracy by ~65% on LLFF and ~53% on DTU.
- Globally sharing a single INN with per-frame latent codes performs better than independent per-frame INNs, as gradient sharing yields extra benefits.
- The homeomorphisms predicted by the INN exhibit non-rigid deformations during intermediate optimization stages, providing a more flexible optimization trajectory to bypass local minima.
- In DTU 360° scenes, L2G performs worse than BARF, whereas the proposed INN consistently achieves the best results.

## Highlights & Insights

- The key role of invertibility is clearly demonstrated through systematic comparative experiments spanning Naive $\to$ Implicit $\to$ Explicit designs.
- The analysis from the perspective of homeomorphisms is highly insightful: the INN can "temporarily deform" during optimization to bypass local minima.
- The design featuring global sharing and per-frame latent codes balances parameter efficiency and expressive capability.
- The rigidity prior is incorporated as a soft constraint, preserving the flexibility of the INN.

## Limitations & Future Work

- Currently, the method is only applied to vanilla NeRF and has not been extended to newer representations such as 3D Gaussian Splatting.
- The representation capacity of the INN is constrained by the specific architecture (NDR-INN); stronger invertible architectures can be explored.
- Initialization issues under large baselines or large rotation angles are not addressed.
- The full INN network must be retained during inference, meaning the final poses cannot be compactly stored as 6 parameters like standard $SE(3)$ (though they can be extracted via Eq. 5).

## Related Work & Insights

- **BARF**: The baseline method; its coarse-to-fine positional encoding scheduling is adopted in this work.
- **L2G**: Also utilizes an over-parameterization strategy but uses an MLP to predict $SE(3)$; this work demonstrates that learning the warp function directly with an INN is superior.
- **NoPe-NeRF**: Constrains poses using monocular depth priors; this is orthogonal to the proposed method and can be combined.
- **Deformation Fields in Dynamic NeRF**: The use of INNs to represent deformations has been validated in the temporal dimension; this work extends it to the camera pose space.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Utilizing INNs for pose over-parameterization is highly novel, and the verification of invertibility is thorough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multilevel validation on 2D/LLFF/DTU is provided with comprehensive baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The progressive narrative from Naive to INN is highly logical and clear.
- **Value**: ⭐⭐⭐⭐ — Outlines a new paradigm for NeRF pose optimization; the insights on invertibility have generalized significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions](deblur_e-nerf_nerf_from_motion-blurred_events_under_high-speed_or_low-light_cond.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] The NeRFect Match: Exploring NeRF Features for Visual Localization](the_nerfect_match_exploring_nerf_features_for_visual_localization.md)
- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)

</div>

<!-- RELATED:END -->
