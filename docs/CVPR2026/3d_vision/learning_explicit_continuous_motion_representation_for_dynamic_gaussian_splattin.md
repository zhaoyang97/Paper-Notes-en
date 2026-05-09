---
title: >-
  [Paper Note] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos
description: >-
  [CVPR 2026][3D Vision][Dynamic Gaussian Splatting] This paper proposes to explicitly model the continuous positional and rotational deformation trajectories of dynamic Gaussians via adaptive SE(3) B-spline motion bases, combined with a soft segment reconstruction strategy and multi-view diffusion model priors, achieving high-quality novel view synthesis of dynamic scenes from monocular video. The method surpasses existing approaches on both the iPhone and NVIDIA datasets.
tags:
  - CVPR 2026
  - 3D Vision
  - Dynamic Gaussian Splatting
  - Monocular Video
  - SE(3) B-Spline
  - Motion Representation
  - Novel View Synthesis
date: 2026-05-08
content_hash: 1770996c8695b14b
---

# Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos

**Conference**: CVPR 2026
**arXiv**: [2603.25058](https://arxiv.org/abs/2603.25058)
**Code**: [https://github.com/hhhddddddd/se3bsplinegs](https://github.com/hhhddddddd/se3bsplinegs)
**Area**: 3D Vision
**Keywords**: Dynamic Gaussian Splatting, Monocular Video, SE(3) B-Spline, Motion Representation, Novel View Synthesis

## TL;DR

This paper proposes to explicitly model the continuous positional and rotational deformation trajectories of dynamic Gaussians via adaptive SE(3) B-spline motion bases, combined with a soft segment reconstruction strategy and multi-view diffusion model priors, achieving high-quality novel view synthesis of dynamic scenes from monocular video. The method surpasses existing approaches on both the iPhone and NVIDIA datasets.

## Background & Motivation

Reconstructing dynamic scenes from monocular video is a core problem in computer vision, with broad applications in VR/AR and film production. Existing 3D Gaussian Splatting-based methods exhibit notable deficiencies when handling dynamic scenes:

1. **Implicit methods (e.g., D3DGS, 4DGS)** learn the transformation from canonical to observation space via MLPs or k-planes, and cannot guarantee continuity of deformation trajectories.
2. **Explicit methods (e.g., SplineGS)** use cubic Hermite splines to model continuous positional deformation trajectories, but neglect the continuous variation of Gaussian orientations.
3. **Motion-basis-based methods (e.g., SoM, MoSca)** model deformation through learned affine transformations or motion scaffolds, but do not handle the continuity of both position and orientation in a unified manner.

**Key Challenge**: When the orientation of dynamic Gaussians changes discontinuously, severe artifacts appear in rendered images, particularly in regions with complex motion. The authors' **Key Insight** is to leverage SE(3) cumulative B-spline functions, which mathematically guarantee simultaneous continuity of both position and orientation, providing a unified solution to this problem.

## Method

### Overall Architecture

Given a monocular video as input, static and dynamic Gaussians are first initialized via depth reprojection. The motion of dynamic Gaussians is governed by learnable SE(3) B-spline motion bases. During training, an adaptive control mechanism dynamically adjusts the number of motion bases and control points. A soft segment reconstruction strategy fuses dynamic Gaussians from different reference timestamps into the observation timestamp, with additional viewpoint supervision provided by a multi-view diffusion model. The final output is a dynamic Gaussian field suitable for novel view rendering.

### Key Designs

1. **SE(3) B-Spline Motion Bases**:
    - **Function**: Explicitly model the continuous positional and rotational deformation trajectories of dynamic Gaussians.
    - **Mechanism**: SE(3) cumulative B-spline functions are used to construct motion trajectories from a small number of learnable control points. Pose states $Q = [R, t]$ are first obtained from 3D tracking points to initialize control points. Relative pose transformations between adjacent tracking points are computed as $\Delta Q = Q_i^{-1} Q_{i+1}$, then mapped to the Lie algebra space via the logarithmic map $\xi = \log(\Delta Q)$. B-spline basis functions $\Omega_i(t)$ then interpolate to yield the continuous transformation at arbitrary time $t$: $T(t) = (\prod_{i=0}^{N_c-1} \exp(\Omega_i(t)\xi_i)) T_0$.
    - **Design Motivation**: SE(3) B-splines mathematically guarantee simultaneous continuity of translational position and rotational orientation, whereas Hermite splines handle only position. This fundamentally eliminates rendering artifacts caused by orientation discontinuities.

2. **Adaptive Motion Basis Control Mechanism**:
    - **Function**: Dynamically adjust the number of motion bases and control points to balance expressive capacity and computational efficiency.
    - **Mechanism**: Comprises pruning and densification operations. *Pruning*: every $N_{prune}=500$ iterations, the algorithm attempts to remove one control point by selecting the point whose removal minimizes trajectory change; if the resulting error falls below a threshold $\epsilon_{prune}=5.0$, the pruning is executed. *Densification*: every $N_{densify}=500$ iterations, regions of complex motion are identified via the intersection of rendering error and dynamic region masks, and the corresponding motion bases are densified by duplicating control points with added random perturbations.
    - **Design Motivation**: Motion complexity varies greatly across different scene regions. Excessive control points increase computational cost and risk overfitting, while too few limit expressiveness. The adaptive mechanism concentrates resources where they are needed.

3. **Soft Segment Reconstruction Strategy**:
    - **Function**: Mitigate the interference of long-interval motion deformations on scene reconstruction.
    - **Mechanism**: When transforming dynamic Gaussians from all reference timestamps to the observation timestamp, Gaussian opacity is modulated according to temporal distance: $o' = \text{sigmoid}(\text{scale} \cdot (1 - |t_{ref} - t_{obs}|)) \cdot o$ (scale=5.0). Reference Gaussians farther in time receive lower opacity.
    - **Design Motivation**: For long-duration videos, transformation accuracy decreases for reference timestamps far from the observation timestamp. Opacity attenuation allows nearby reference Gaussians to dominate reconstruction, reducing uncertainty from inaccurate long-range transformations.

### Loss & Training

The total loss comprises six terms: reconstruction loss $\mathcal{L}_{rec}$ (L1 + SSIM, $\beta=0.2$), geometric depth loss $\mathcal{L}_{geo}$ ($\lambda=0.075$), multi-view SDS loss $\mathcal{L}_{sds}$ ($\lambda=0.01$, using a multi-view diffusion model to provide priors for unseen regions), ARAP rigidity constraint loss $\mathcal{L}_{arap}$, optical flow tracking loss $\mathcal{L}_{track}$, and camera smoothness loss $\mathcal{L}_{smo}$ ($\lambda=0.01$, constraining smooth variation of camera extrinsics between adjacent frames). Camera extrinsics are jointly optimized as learnable parameters. Training runs for 8,000 iterations.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | MoSca | SplineGS | SoM | Gain (vs MoSca) |
|---------|--------|------|-------|----------|-----|-----------------|
| iPhone | mPSNR↑ | **20.17** | 19.33 | 15.52 | 17.13 | +0.84 |
| iPhone | mSSIM↑ | **0.729** | 0.718 | 0.483 | 0.674 | +0.011 |
| iPhone | mLPIPS↓ | **0.274** | 0.274 | 0.371 | 0.279 | on par |
| NVIDIA | PSNR↑ | **27.81** | 26.76 | 27.12 | 24.58 | +1.05 |
| NVIDIA | SSIM↑ | **0.871** | 0.854 | 0.872 | 0.651 | +0.017 |
| NVIDIA | LPIPS↓ | **0.049** | 0.070 | 0.052 | 0.124 | −0.021 |

Training requires only 30 minutes on a single RTX 4090, with inference at 45.124 FPS, achieving a favorable balance between efficiency and quality.

### Ablation Study

| Configuration | iPhone mPSNR | iPhone mLPIPS | NVIDIA PSNR | NVIDIA LPIPS |
|---------------|-------------|---------------|-------------|--------------|
| Full model | **20.17** | **0.274** | **27.81** | **0.049** |
| w/o Adaptive Control | 18.84 | 0.350 | 26.87 | 0.128 |
| w/o Soft Segment | 19.02 | 0.328 | 27.06 | 0.085 |
| w/o $\mathcal{L}_{sds}$ | 19.39 | 0.288 | 27.13 | 0.074 |
| w/o $\mathcal{L}_{smo}$ | 19.18 | 0.295 | 27.15 | 0.076 |

Motion representation replacement ablation (iPhone mPSNR): SoM Pose transformation → 18.17; MoSca Motion Scaffold → 19.26; proposed SE(3) B-spline → **20.17**.

### Key Findings

- The adaptive control mechanism contributes the most (removing it drops iPhone mPSNR from 20.17 to 18.84, a decrease of 1.33), demonstrating that motion basis density should adapt to scene complexity.
- The soft segment reconstruction strategy yields more pronounced improvements on the iPhone dataset (longer sequences, more complex motion) than on the NVIDIA dataset, consistent with its design motivation.
- The SE(3) B-spline motion representation improves mPSNR by 2.0 and 0.91 over Pose transformation and Motion Scaffold, respectively, validating the importance of jointly modeling positional and orientational continuity.
- The method exhibits robustness to errors in 2D tracking priors; adding random noise in the range $[-15, 15]$ results in only marginal performance degradation (mPSNR: 20.17 → 20.11).

## Highlights & Insights

- **Introducing SE(3) cumulative B-splines into dynamic Gaussian Splatting** is the key contribution. While B-splines are widely used in robotics and SLAM, their application for jointly modeling continuous positional and rotational deformation in dynamic 3DGS is novel. This idea is transferable to any setting requiring continuous rigid-body motion modeling.
- The **adaptive pruning + densification** strategy is highly practical—simple motion regions use few control points while complex regions are automatically densified, reducing computation while improving quality.
- A **30-minute training time** is highly competitive among comparable methods (vs. 13 hours for MarbleGS), representing a clear efficiency advantage.

## Limitations & Future Work

- As acknowledged in Figure 7 of the paper, the method struggles with large-scale non-rigid motion (e.g., clothing dynamics during human dancing), since SE(3) B-splines are fundamentally rigid-body motion models.
- The SDS loss introduces an additional dependency on diffusion models, and its generalization across diverse scenes has not been thoroughly validated.
- Evaluation is limited to the iPhone (5 scenes) and NVIDIA (7 scenes) datasets, with limited scene diversity.
- Future work could explore combining SE(3) B-splines with non-rigid deformation models (e.g., blend shapes or SMPL) to handle more complex dynamic motions.

## Related Work & Insights

- **vs. SplineGS**: Cubic Hermite splines modeling position only vs. the proposed SE(3) B-splines modeling both position and orientation — the latter achieves 4.65 higher mPSNR on the iPhone dataset.
- **vs. MoSca**: Constructs 4D motion scaffolds using 2D foundation models; Motion Scaffolds offer high degrees of freedom but do not guarantee continuity. This work replaces them with mathematically continuous B-splines.
- **vs. SoM (Shape-of-Motion)**: Models motion as a linear combination of SE(3) motion bases without continuous parameterization; this work provides a continuous parameterization via B-splines.
- The adaptive control mechanism may inspire adaptive resolution strategies in point cloud processing, NeRF, and related fields.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing the well-established SE(3) B-spline into dynamic 3DGS is a meaningful engineering innovation, though not theoretically groundbreaking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablations are comprehensive and include motion representation replacement comparisons; tracking robustness analysis is a notable feature, though the number of evaluated datasets is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, mathematical derivations are complete, and visualizations are abundant.
- **Value**: ⭐⭐⭐⭐ High practical value — fast training, strong quality, and publicly available code.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[ICLR 2026\] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](../../AAAI2026/3d_vision/mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)
- [\[CVPR 2026\] Dark3R: Learning Structure from Motion in the Dark](dark3r_learning_structure_from_motion_in_the_dark.md)

<!-- RELATED:END -->
