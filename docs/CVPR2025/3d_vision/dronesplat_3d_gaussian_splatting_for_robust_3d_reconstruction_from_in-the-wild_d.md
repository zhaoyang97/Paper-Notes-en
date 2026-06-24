---
title: >-
  [Paper Note] DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery
description: >-
  [CVPR 2025][3D Vision][drone 3D reconstruction] DroneSplat proposes a robust 3DGS framework for in-the-wild drone imagery, mitigating dynamic distractors via an adaptive local-global masking strategy, addressing reconstruction quality under limited views using Multi-View Stereo (MVS)-based geometry-aware point sampling and voxel-guided optimization, and presenting a dataset of 24 drone reconstruction scenes.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "drone 3D reconstruction"
  - "3D Gaussian Splatting"
  - "dynamic distractor removal"
  - "sparse-view reconstruction"
  - "adaptive masking"
date: 2026-05-08
content_hash: 2acf4447c7db6236
---

# DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery

**Conference**: CVPR 2025  
**arXiv**: [2503.16964](https://arxiv.org/abs/2503.16964)  
**Code**: [https://bityia.github.io/DroneSplat/](https://bityia.github.io/DroneSplat/)  
**Area**: 3D Vision  
**Keywords**: drone 3D reconstruction, 3D Gaussian Splatting, dynamic distractor removal, sparse-view reconstruction, adaptive masking

## TL;DR
DroneSplat proposes a robust 3DGS framework for in-the-wild drone imagery, mitigating dynamic distractors via an adaptive local-global masking strategy, addressing reconstruction quality under limited views using Multi-View Stereo (MVS)-based geometry-aware point sampling and voxel-guided optimization, and presenting a dataset of 24 drone reconstruction scenes.

## Background & Motivation

1. **Background**: Drones have become critical tools for in-the-wild scene reconstruction due to their high mobility. Radiance fields like NeRF and 3DGS have demonstrated exceptional potential in 3D representation and novel view synthesis.
2. **Limitations of Prior Work**: Applying radiance field methods to in-the-wild drone imagery faces two major challenges—(a) **Scene Dynamics**: Drone images often contain moving objects (cars, pedestrians) that violate the multi-view consistency assumption; (b) **Sparse Views**: The view coverage over certain areas during a single flight is limited, causing radiance fields to overfit to input views.
3. **Key Challenge**: Existing dynamic distractor elimination methods either rely on predefined categories (failing to distinguish static/dynamic objects of the same class) or use hard thresholds (failing to adapt to varying scenes and training phases). Sparse-view methods introduce geometric priors but lack dedicated 3DGS optimization strategies.
4. **Goal**: Simulataneously address dynamic distractor identification/removal and high-quality geometric reconstruction under limited views for in-the-wild drone imagery.
5. **Key Insight**: The authors observe that (a) dynamic object residuals change significantly across different training phases, making fixed thresholds impractical; (b) dense point clouds predicted by DUSt3R provide rich priors but require complementary voxel-level optimization constraints to be fully utilized.
6. **Core Idea**: Combine adaptive local-global masking for robust dynamic distractor removal with geometry-aware point sampling and voxel-guided optimization to leverage multi-view stereo priors for high-quality reconstruction under sparse views.

## Method

### Overall Architecture
Given a set of posed drone images, the pipeline of DroneSplat is as follows: (1) use DUSt3R to predict dense point clouds and obtain initialized points through geometry-aware sampling; (2) during 3DGS training, identify and eliminate dynamic distractors via adaptive local-global masking (ALGM); (3) constrain the Gaussian optimization process using a voxel-guided optimization strategy, ultimately achieving robust static scene reconstruction.

### Key Designs

1. **Adaptive Local Masking**:

    - **Function**: Adaptively adjust thresholds based on real-time residuals and statistical methods to identify dynamic distractors in each frame.
    - **Mechanism**: First, class-agnostic segmentation is performed on each image using SAM v2 to obtain object-level masks, and the object-level mean of normalized residuals $\mathcal{R}_i^j(t)$ is calculated. The adaptive threshold $\mathcal{T}_i^L(t) = \mathbb{E}[R_i](t) + \text{Var}[R_i](t)(1 + \lambda_L \frac{T_{max}-t}{T_{max}})$ is dynamically adjusted based on the mean and variance of residuals in the current frame, loosening the upper bound during early training to accommodate different convergence rates of objects. Objects with residuals exceeding the threshold are marked as dynamic distractors.
    - **Design Motivation**: Fixed thresholds perform differently across scenes and training phases—too high might miss large dynamic objects, while too low might misidentify static objects. The adaptive threshold automatically adjusts based on the actual residual distribution without manual parameter tuning.

2. **Complement Global Masking**:

    - **Function**: Track dynamic distractors across frames to handle objects that are temporarily static in some frames but dynamic overall (such as vehicles waiting at red lights).
    - **Mechanism**: Set a higher global threshold $\mathcal{T}_i^G = \mathbb{E}[R_i](t) + \lambda_G \text{Var}[R_i](t)$ ($\lambda_G > 1 + \lambda_L$), marking objects exceeding the threshold as tracking candidates. Five prompt points are selected from the candidate mask and fed into the video segmentation feature of SAM v2 to track the object across all frames. The global mask set accumulates over training iterations: $\mathcal{M}_i^G(t) = \mathcal{M}_i^G(t-1) \cup \hat{m}_i^j$. The final mask is the union of local and global masks.
    - **Design Motivation**: Local masking can only identify objects with high residuals in the current frame, failing to handle dynamic objects that are temporarily stationary. Global masking fills this gap through cross-frame tracking, with both working synergistically to achieve comprehensive dynamic object removal.

3. **Voxel-guided Gaussian Splatting**:

    - **Function**: Leverage multi-view stereo priors to constrain 3DGS optimization, addressing the overfitting problem under limited views.
    - **Mechanism**: (a) **Geometry-aware point sampling**: Use DUSt3R to predict dense point clouds, extract FPFH descriptors to measure geometric features, and compute a comprehensive score $\text{Score}(p) = \text{Conf}(p) \cdot \tilde{\text{FPFH}}(p)$ combined with confidence. High-scoring top-k points within adaptively partitioned voxels are preserved to initialize Gaussians. (b) **Voxel-guided optimization**: Impose boundary constraints on Gaussians within each voxel (the center and scaling must not exceed $\tau$ times the voxel length), with exponential decay applied to the gradients of Gaussians exceeding the boundaries. If accumulated gradients reach a threshold and point towards an empty voxel, they split/clone into that empty voxel. Low-quality voxels (with insufficient Gaussians or excessively low average opacity) are removed.
    - **Design Motivation**: Although InstantSplat utilizes DUSt3R to initialize Gaussians, it lacks optimization constraints, leading to suboptimal performance. The voxel-guided strategy extends the constraints of geometric priors from initialization to the entire optimization process, effectively preventing Gaussians from drifting away from reasonable positions.

### Loss & Training
- Use standard 3DGS loss (L1 + D-SSIM), excluding loss contributions from dynamic areas using masks.
- Adaptive masking starts after 500 iterations, with $\lambda_L = 0.4$ and $\lambda_G = 2.8$.
- Total training iterations: 7000, running on an NVIDIA A100.
- DUSt3R uses a resolution of 512, with voxel parameters $N = 80$ and $k = 3$.

## Key Experimental Results

### Main Results

| Dataset/Scene | Metric | DroneSplat | 3DGS | WildGaussians | Best Baseline |
|------------|------|-----------|------|---------------|-------------|
| DroneSplat (Low Dynamic) | PSNR↑ | **24.56** | 22.43 | 22.41 | 23.29 (GS-W) |
| DroneSplat (Medium Dynamic) | PSNR↑ | **17.89** | 17.04 | 16.96 | 17.44 (GS-W) |
| DroneSplat (High Dynamic) | PSNR↑ | **19.51** | 17.11 | 17.15 | 17.09 (GS-W) |
| On-the-go Dataset | PSNR↑ | Competitive | baseline | Second Best | NeRF-HuGS |

### Ablation Study

| Configuration | Description |
|------|------|
| DroneSplat (full) | Full framework, optimal performance |
| Ours (COLMAP) | Use COLMAP initialization instead of DUSt3R; performance is close but slightly lower |
| w/o Global Masking | Remove global masking; temporarily stationary dynamic objects cannot be eliminated |
| w/o Voxel-guided | Remove voxel guidance; reconstruction quality drops significantly under limited views |

### Key Findings
- **The higher the dynamic level, the greater the advantage of DroneSplat**: In high dynamic scenes, DroneSplat's PSNR improves by up to 2.4 dB compared to 3DGS, indicating that the ALGM strategy is particularly effective in complex dynamic environments.
- **Significant synergy between local and global masking**: Global masking captures objects that are temporarily stationary in certain frames but dynamic overall (such as cars waiting at traffic lights), which would otherwise be missed if only local masking were used.
- **Voxel guidance is crucial for sparse views**: In scenes with only 6 input views, voxel-guided optimization provides more stable geometric constraints compared to vanilla 3DGS.
- **Self-collected dataset fills the gap**: The provided dataset of 24 drone scenes covers various dynamic intensities and static scenes, offering a standardized evaluation platform for this field.

## Highlights & Insights
- **Exquisite adaptive threshold design**: The local threshold automatically tightens as training progresses (loose early on to allow for differing convergence speeds, and strict later on for precise identification) without requiring manual parameter tuning. This training-aware thresholding strategy can be transferred to any radiance field task requiring anomaly detection.
- **Local-global masking collaborative mechanism**: Local masking addresses prominent dynamic objects in current frames, while global masking handles cross-frame consistent dynamic objects via SAM v2 video tracking. This hierarchical detection approach is highly suitable for handling complex non-static scenes.
- **Voxel-guided optimization extends prior constraints from initialization to the entire process**: This addresses InstantSplat's issue of "good initialization but scattered optimization." The expansion mechanism for empty voxels is also cleverly designed.

## Limitations & Future Work
- **Computational overhead of SAM v2**: Global mask tracking relies on video segmentation from SAM v2, which can become a bottleneck for efficiency in large-scale scenes (thousands of images).
- **Limited adaptability of voxel size**: Currently, the voxel size is determined by partitioning the shortest side of the scene equally, which may not be flexible enough for highly irregular scenes.
- **Unhandeld lighting variations**: In-the-wild drone imagery often undergoes illumination changes, which are currently not considered on exposure inconsistency.
- **Limited to 3DGS**: Whether the method can extend to variants like 2DGS or Mip-Splatting remains unverified.

## Related Work & Insights
- **vs RobustNeRF**: RobustNeRF filters high-residual pixels using a fixed threshold that remains unchanged throughout training. DroneSplat's adaptive threshold dynamically adjusts based on the residual distribution, which is more robust.
- **vs NeRF-HuGS**: NeRF-HuGS requires manual tuning for each scene, with the threshold remaining static during training. DroneSplat is fully automated and adaptively adjusts the threshold over training.
- **vs WildGaussians**: WildGaussians predicts pixel-level uncertainty using DINOv2 features, but low-resolution features lead to blurry boundaries and missed small targets. DroneSplat addresses this issue with object-level masks.
- **vs InstantSplat**: InstantSplat initializes with DUSt3R but lacks constraints during the subsequent optimization. DroneSplat's voxel-guided strategy fills this gap.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the adaptive local-global masking and voxel-guided optimization are sophisticated designs targeting specific challenges.
- Experimental Thoroughness: ⭐⭐⭐⭐ Self-collected dataset + two public datasets, evaluating performance on both distractor removal and sparse-view reconstruction.
- Writing Quality: ⭐⭐⭐⭐ Clear problem analysis, logical methodological exposition, and high-quality illustrations.
- Value: ⭐⭐⭐⭐ A practical framework tailored for real-world drone application scenarios, accompanied by a valuable dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] 3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)
- [\[CVPR 2025\] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)

</div>

<!-- RELATED:END -->
