---
title: >-
  [Paper Note] Motion-Aware Animatable Gaussian Avatars Deblurring
description: >-
  [CVPR 2026][3D Vision][3D human reconstruction] This paper proposes the first method for directly reconstructing sharp, animatable 3D Gaussian human avatars from blurry video…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D human reconstruction"
  - "motion blur"
  - "3D Gaussian splatting"
  - "SMPL"
  - "deblurring"
date: 2026-05-08
content_hash: 2e647947301dfc19
---

# Motion-Aware Animatable Gaussian Avatars Deblurring

**Conference**: CVPR 2026
**arXiv**: [2411.16758](https://arxiv.org/abs/2411.16758)
**Code**: [GitHub](https://github.com/MyNiuuu/MAD-Avatar)
**Area**: 3D Vision
**Keywords**: 3D human reconstruction, motion blur, 3D Gaussian splatting, SMPL, deblurring

## TL;DR

This paper proposes the first method for directly reconstructing sharp, animatable 3D Gaussian human avatars from blurry video, leveraging a 3D-aware physical blur formation model and an SMPL-based human motion model to jointly optimize the avatar representation and motion parameters.

## Background & Motivation

Creating 3D human avatars from multi-view video is an important task in computer vision. Existing methods (e.g., GauHuman) rely on high-quality sharp image inputs; however, motion blur is unavoidable in real-world scenarios due to variations in the speed and intensity of human movement. Blur introduces two problems: (1) the 3DGS model learns a distorted 3D representation owing to the inherent ambiguity induced by motion blur; and (2) even when cameras are calibrated, blurry frames lead to erroneous SMPL parameter estimation. A straightforward two-stage pipeline (2D deblurring followed by reconstruction) ignores 3D scene information and causes multi-view inconsistencies.

## Method

### Overall Architecture

The 3DGS avatar reconstruction problem is decomposed into two sub-tasks: optimizing sub-frame motion representations and constructing a sharp 3DGS avatar model in canonical space. Blurry frames are synthesized by averaging a series of "virtual" sharp rendered images, and a loss is computed against the observed blurry frames.

### Key Designs

1. **3D Blur Formation Model**: The physical 2D blur process is extended to 3D human avatar modeling. A blurry image is represented as the average of $T$ rendered images sampled during the exposure period:
    $\mathbf{I}^B = \frac{1}{T}\sum_{t=0}^{T-1}\mathcal{R}(\mathcal{W}(\{G_k(\mathbf{x})\}_{k=0}^{K-1}, \mathcal{S}_t), \mathbf{R}, \mathbf{K})$
   where $\mathcal{W}$ deforms the canonical-space 3D Gaussians to observation space according to SMPL parameters $\mathcal{S}_t$, and $\mathcal{R}$ denotes the rasterization process. This naturally embeds the deblurring problem into the 3D reconstruction framework.

2. **Sub-frame Rigid Sequence Pose Model (B-spline Interpolation)**: Using the 24 joints of the SMPL model, $P$ control parameters $\tilde{\Theta}^j \in \mathbb{R}^{P \times 3}$ are defined per joint, and intermediate poses are interpolated via the De Boor–Cox B-spline formula:
    $\hat{\Theta}_t^j = \mathbf{B}(t) \cdot \mathcal{M}^P \cdot \tilde{\Theta}^j$
   where $\mathbf{B}(t)$ is the temporal basis and $\mathcal{M}^P$ is the interpolation matrix. B-splines guarantee continuity of joint motion; the control parameters are initialized from coarse estimates and optimized during training.

3. **Pose Deformation Model**: B-splines can only capture basic pose trajectories and are insufficient for modeling non-rigid high-frequency pose variations. A CNN $G_{disp}$ is introduced to predict a residual displacement for each joint at each time step:
    $\Theta_t^j = \hat{\Theta}_t^j + G_{disp}(\hat{\Theta}_t^j; \theta_{disp})$
   This enables the model to more accurately capture complex pose dynamics.

4. **Inter-frame Motion Regularization**: Motion blur is subject to directional ambiguity (motions in opposite directions can produce similar blur patterns). Regularization is applied by measuring the geodesic distance between pose parameters at the end of one exposure period and the beginning of the next:
    $\mathcal{L}_{reg} = \frac{1}{24 \cdot (N_e - 1)}\sum_{n=0}^{N_e-2}\sum_{j=0}^{23}|\hat{\Theta}_{n,T-1}^j - \hat{\Theta}_{n+1,0}^j|_G$
   This enforces temporal consistency across frames.

### Loss & Training

The total loss consists of an L1 reconstruction loss between the synthesized and observed blurry frames plus the inter-frame regularization term:

$$\mathcal{L} = \|\hat{\mathbf{I}}^B - \mathbf{I}^B\|_1 + \mathcal{L}_{reg}$$

The Adam optimizer ($\beta_1=0.9, \beta_2=0.999$) is used, with learning rate and decay schedules following the original 3DGS. Input resolutions are $512 \times 512$ for synthetic data and $612 \times 512$ for real data; training is performed on a single RTX 4090.

## Key Experimental Results

### Main Results

| Method | Syn. PSNR↑ | Syn. SSIM↑ | Syn. LPIPS↓ | Real PSNR↑ | Real SSIM↑ | Real LPIPS↓ |
|--------|-----------|-----------|------------|-----------|-----------|------------|
| GauHuman | 23.080 | 0.7660 | 0.2277 | 25.602 | 0.8044 | 0.2380 |
| BSST+GauHuman | 23.081 | 0.7698 | 0.2212 | 25.568 | 0.8068 | 0.2342 |
| **Ours** | **25.546** | **0.8290** | **0.1476** | **27.010** | **0.8271** | **0.1668** |

### Ablation Study

| Configuration | Syn. PSNR↑ | Syn. LPIPS↓ | Real PSNR↑ | Note |
|---------------|-----------|------------|-----------|------|
| w/o interp. | 24.009 | 0.1620 | 25.825 | No motion interpolation; largest degradation |
| w/o pose deform | 25.301 | 0.1545 | 26.426 | Missing high-frequency pose details |
| w/o LBS opt. | 25.394 | 0.1486 | 26.821 | Fixed skinning weights |
| Full model | 25.546 | 0.1476 | 27.010 | All components included |

### Key Findings

- Two-stage baselines (2D deblurring followed by reconstruction) yield limited improvement, as 2D deblurring cannot guarantee multi-view consistency.
- The inter-frame regularization $\mathcal{L}_{reg}$ is critical for rendering quality at non-middle timesteps (non-middle timestep PSNR improves from 24.421 to 25.417).
- Among three trajectory representations—B-spline, Slerp, and Linear—B-spline performs best, though the margin is modest.

## Highlights & Insights

- This is the first work to address the reconstruction of sharp, animatable 3D human avatars from blurry video, filling a notable gap in the field.
- The approach of seamlessly integrating deblurring with 3D reconstruction is elegant: rather than deblurring prior to reconstruction, the blur formation process is modeled directly in 3D space.
- Two benchmark datasets are constructed: a synthetic dataset based on ZJU-MoCap and a real-world dataset captured with a 360-degree hybrid-exposure camera system.

## Limitations & Future Work

- The method relies on coarse SMPL parameter estimates for initialization; poor initialization quality may hinder convergence.
- Only human motion blur is addressed; joint handling of camera motion blur is not considered.
- Extension to multi-person scenes and more complex occlusion scenarios remains unexplored.
- The current method supports single-person avatar reconstruction only; mutual occlusion and contact region handling in multi-person interaction scenarios await further investigation.

## Related Work & Insights

- **vs. NeRF/3DGS deblurring methods** (e.g., DeblurNeRF, BAD-NeRF): These methods primarily address camera motion blur or defocus blur in static scenes. This paper focuses on motion blur for animatable humans, requiring additional modeling of human joint dynamics and leveraging SMPL priors to constrain the motion space.
- **vs. sharp-input methods such as GauHuman**: GauHuman assumes sharp input frames; the proposed method can serve as a blur-aware front-end to improve the robustness of such methods under low-quality inputs.
- The B-spline motion modeling strategy is generalizable to blurry reconstruction of other dynamic objects (e.g., animals, hands, deformable objects).
- The physics-driven 3D blur formation model is the key bridge connecting deblurring and 3D reconstruction—rather than deblurring first and then reconstructing, blur is used directly as a supervision signal within 3D optimization.
- The 360-degree hybrid-exposure camera system (4 blurry + 8 sharp synchronized cameras) provides a valuable real-world benchmark for blur-aware 3D reconstruction.
- A DIY demo using an iPhone 16 Pro demonstrates the practical potential of the method on consumer-grade devices.

## Rating

- **Novelty**: ★★★★☆ — First to address blur-aware avatar reconstruction; the problem formulation is clear and valuable.
- **Technical Depth**: ★★★★☆ — The combination of physical blur modeling, B-spline interpolation, pose deformation CNN, and inter-frame regularization is elegantly designed.
- **Experimental Thoroughness**: ★★★★★ — Synthetic and real datasets, comprehensive ablation studies, and a DIY iPhone 16 Pro demonstration.
- **Value**: ★★★★☆ — Motion blur is common in real-world scenarios; the method fills an important gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars](progressiveavatars_progressive_animatable_3d_gaussian_avatars.md)
- [\[CVPR 2026\] Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image](zero-shot_reconstruction_of_animatable_3d_avatars_with_cloth_dynamics_from_a_sin.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](../../ICCV2025/3d_vision/comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] Sequential Gaussian Avatars with Hierarchical Motion Context](../../ICCV2025/3d_vision/sequential_gaussian_avatars_with_hierarchical_motion_context.md)

</div>

<!-- RELATED:END -->
