---
title: >-
  [Paper Note] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction
description: >-
  [CVPR 2025][3D Vision][Novel View Synthesis] This paper proposes a depth-guided bundle (GDB) sampling strategy that groups adjacent rays into bundles for joint processing via sphere-cone sampling. Concurrently, it adaptively allocates the number of sampling points based on depth confidence. When applied to ENeRF and MVSGaussian, it achieves a 1.27dB PSNR improvement and a 47% speedup in FPS on the DTU dataset.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Generalizable NeRF"
  - "Bundle Sampling"
  - "Depth Guidance"
  - "Efficient Rendering"
date: 2026-05-08
content_hash: d3fec13c3eea9ce9
---

# Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2505.19793](https://arxiv.org/abs/2505.19793)  
**Code**: [https://github.com/KLMAV-CUC/GDB-NeRF](https://github.com/KLMAV-CUC/GDB-NeRF)  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, Generalizable NeRF, Bundle Sampling, Depth Guidance, Efficient Rendering

## TL;DR

This paper proposes a depth-guided bundle (GDB) sampling strategy that groups adjacent rays into bundles for joint processing via sphere-cone sampling. Concurrently, it adaptively allocates the number of sampling points based on depth confidence. When applied to ENeRF and MVSGaussian, it achieves a 1.27dB PSNR improvement and a 47% speedup in FPS on the DTU dataset.

## Background & Motivation

**Background**: Generalizable NeRF methods can synthesize novel views from multi-view images without requiring scene-specific optimization. Representative methods such as ENeRF and MVSGaussian have achieved decent quality. However, rendering high-resolution images still requires dense pixel-by-pixel sampling of all rays, which remains computationally heavy.

**Limitations of Prior Work**: Although existing generalizable NeRF methods reduce the number of sampling points along each ray through depth guidance (e.g., ENeRF only samples 2 points per ray), they still cast one ray independently for each pixel, neglecting spatial correlation among adjacent pixels. This implies that for a massive number of pixels in smooth regions, sampling is largely redundant.

**Key Challenge**: Plenoptic sampling theory indicates that natural scenes are typically piecewise smooth, with high-frequency information concentrated only at edges and depth discontinuities. However, existing methods employ the same sampling density for all pixels, wasting significant computation on smooth regions while potentially undersampling complex ones.

**Goal**: To design a sampling strategy that exploits spatial locality to reduce sampling in smooth regions and increase sampling in complex regions, thereby significantly improving rendering efficiency without sacrificing quality.

**Key Insight**: Guided by plenoptic sampling theory, we observe that adjacent rays can be grouped into "bundles" and jointly sampled using a single cone instead of multiple independent rays. Meanwhile, depth confidence can serve as a proxy indicator for scene complexity to adaptively adjust sampling density.

**Core Idea**: Sample using bundles instead of single rays, obtaining joint representations and ray-specific detailed representations through sphere encoding, combined with depth-guided adaptive sampling to simultaneously reduce both the number of rays and the sampling points per bundle, achieving a dual improvement in efficiency and quality.

## Method

### Overall Architecture

Given multi-view source images and their camera parameters, the depth range is first estimated using multi-scale feature extraction and cost volume construction. Then, the pixels of the target view are grouped into bundles of size $K \times K$, with each bundle modeled as a cone. Inscribed spheres inside the cone are used for sampling, with the number of sampling points adjustively allocated based on depth predictions. For each spherical sample, both joint bundle representations and ray-specific representations are extracted. After aggregation through volume rendering, they are decoded by a neural renderer to output the final image.

### Key Designs

1. **Sphere-based Cone Sampling**:

    - **Function**: Groups $K \times K$ adjacent rays into a bundle and performs unified sampling using a cone model.
    - **Mechanism**: Divides the target view image into $H/K \times W/K$ bundles, where each bundle corresponds to $K \times K$ pixels. Starting from the camera projection center, a cone is projected along the average direction of all ray directions, whose intersection at the image plane is a disk with a radius of $r_{tar} = K \cdot r_p$. Inside the cone, search-based inscribed spheres $\mathcal{S}(\dot{x}, \dot{r})$ are sampled, where the sphere center is the centroid of corresponding ray intersections and the radius is determined by the cone geometry. Consequently, the number of sampled points is reduced from $O(HWN)$ to approximately $O(HWN/K^2)$.
    - **Design Motivation**: Adjacent pixels usually share similar scene contents, making independent sampling redundant. Cone sampling covers multiple pixels at once, dramatically reducing the overall number of samples.

2. **Multi-view Image-based Sphere Encoding**:

    - **Function**: Extracts both a "joint bundle representation" (low-frequency) and a "ray-specific representation" (high-frequency) for each spherical sample.
    - **Mechanism**: The joint bundle representation leverages a **mipmap hierarchy**—building a mipmap pyramid of source view feature maps. Each sphere is projected onto the source view, and its appropriate mipmap level $l = \log_2(r_{src}/r_p)$ is determined based on its footprint area, extracting pre-filtered features via trilinear interpolation. The ray-specific representation projects the 3D points inside the sphere corresponding to the $K \times K$ rays onto the source view to extract pixel-aligned colors, preserving high-frequency details. Concatenating both representations yields the complete sampled feature.
    - **Design Motivation**: Bundle sampling inevitably loses high-frequency details. Mipmaps provide low-frequency features suitable for the sampling scale, while ray-specific colors compensate for high-frequency details, striking a balance between efficiency and detail preservation.

3. **Depth-Guided Adaptive Sampling**:

    - **Function**: Dynamically adjusts the number of sampling points for each bundle based on depth confidence.
    - **Mechanism**: Utilizes the depth estimation module to predict the depth value and confidence interval $R$ for each bundle. The number of sampling points is calculated via $N_{\mathcal{C}} = \max(\lceil 2R/\delta_s \rceil, N_{max})$, where $\delta_s$ is the minimum sampling interval. Bundles with narrow depth ranges (smooth regions, high depth confidence) require only 1-2 sampling points, while bundles with wide depth ranges (edges, occlusions) are allocated more points. This aligns with plenoptic sampling theory, which suggests that each sample should cover a narrow disparity range.
    - **Design Motivation**: Unlike previous methods like ENeRF that use a fixed number of samples for all rays, adaptive sampling reallocates computational resources to where they are truly needed, yielding an FPS increase of over 50%.

### Loss & Training

The same loss function as MVSGaussian is adopted. Training operates in two stages: first pre-training for 100 epochs using uniform sampling (a fixed $N_{max}$ spheres for each bundle) to ensure stable model initialization, followed by switching to the depth-guided adaptive sampling stage for subsequent training. $N_{max}$ is set to 6, and $\delta_s$ is set to 1/64 of the scene depth range.

## Key Experimental Results

### Main Results (DTU Dataset, 3-view, 512×640)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Samples/Ray | FPS↑ |
|------|-------|-------|--------|-----------|------|
| ENeRF | 27.61 | 0.957 | 0.089 | 2 | 19.5 |
| MVSGaussian | 28.21 | 0.963 | 0.076 | 1 | 21.5 |
| MuRF | 28.76 | 0.961 | 0.077 | 80 | 0.934 |
| **ENeRF+Ours (2×2)** | **28.86** | **0.964** | **0.073** | 0.42 | **28.6** |
| ENeRF+Ours (4×4) | 28.21 | 0.957 | 0.088 | 0.10 | 43.6 |
| MVSGaussian+Ours | 28.40 | 0.962 | 0.076 | 1 | 23.4 |

### Ablation Study (DTU, 2×2 bundle)

| Configuration | PSNR | FPS | Description |
|------|------|-----|------|
| Full model (2×2) | 28.86 | 28.6 | Full model |
| w/o Sphere sampling | 27.66 | 29.2 | Replaced with sphere-center point sampling, PSNR drops by 1.2dB |
| w/o Adaptive sampling | 28.85 | 17.0 | Fixed sampling, FPS drops by 40% |
| w/o Ray-specific repr. | 28.47 | 29.4 | Loss of high-frequency details, PSNR drops by 0.39dB |
| w/o Joint bundle repr. | 27.83 | 33.7 | Only using ray-specific, PSNR drops by 1.03dB |

### Key Findings

- **Bundle sampling yields a win-win for quality and speed**: Compared to original ENeRF, ENeRF+Ours (2×2) not only speeds up by 47% (19.5 -> 28.6 FPS) but also boosts PSNR by 1.27dB, which is attributed to the anti-aliasing effects of mipmap pre-filtered features.
- **4×4 bundles show an even greater advantage in speed**: Reaching 43.6 FPS (2.2x that of original ENeRF) with only a marginal drop in PSNR, making it highly suitable for real-time applications.
- **Adaptive sampling contributes the most to the speedup**: Disabling adaptive sampling drops the FPS from 28.6 to 17.0, illustrating that reducing sampling in smooth regions based on depth confidence is crucial for efficiency gains.
- **Joint bundle representation is more critical than ray-specific representation**: Removing the joint bundle representation drops the PSNR by 1.03dB, whereas removing the ray-specific representation drops it by only 0.39dB (with 2x2 bundles), indicating that pre-filtered low-frequency features contribute more to the overall quality.
- **Robust cross-dataset generalization**: Achieving comparable or superior results on Real Forward-facing and NeRF Synthetic datasets.

## Highlights & Insights

- **Plenoptic sampling theory guides neural network design**: This is a rare attempt to incorporate classical light field sampling theory (Chai et al., 2000) into deep learning frameworks. The theoretical analysis provides clear guidance for the design of the sampling strategy, making it more interpretable than purely data-driven methods.
- **Ingenious design of the dual representation strategy**: The combination of joint bundle representation and ray-specific representation mirrors the concept of low-frequency base plus high-frequency residuals in image processing, which can be extended to various rendering tasks requiring a balance between efficiency and details.
- **Generality of the method**: The proposed strategy does not depend on a specific network architecture and can be integrated plug-and-play to accelerate various backbones such as ENeRF and MVSGaussian.

## Limitations & Future Work

- **Depth estimation accuracy acts as a bottleneck**: Adaptive sampling relies heavily on the confidence of depth predictions; incorrect depth estimations will lead to under- or over-sampling.
- **Limited improvement for MVSGaussian+Ours**: Because MVSGaussian itself already samples only 1 point per ray, the advantage of bundle sampling is less pronounced compared to its application on ENeRF.
- **Training overhead from the pre-training stage**: Requiring an additional 100 epochs of uniform-sampling pre-training increases the overall training budget.
- **Future Directions**: Exploring more flexible bundle division strategies (such as non-uniform bundle sizes) that utilize smaller bundles in complex regions and larger bundles in smooth regions.

## Related Work & Insights

- **vs ENeRF**: ENeRF reduces the sampling points per ray to 2 using depth guidance, but still casts rays pixel-by-pixel. Building upon this, the proposed method packs rays into bundles, simultaneously reducing both the number of rays and the number of samples.
- **vs Mip-NeRF**: Mip-NeRF uses cone sampling instead of ray sampling to achieve anti-aliasing, but still casts independent cones for each pixel. This work merges cones from multiple pixels into a single larger cone.
- **vs MVSGaussian**: MVSGaussian combines MVS and 3DGS to achieve real-time generalizable rendering, but its single-point sampling limits further optimization. The proposed strategy can also accelerate 3DGS pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐ The joint bundle sampling strategy combined with plenoptic theory is a novel idea; the dual-representation strategy is ingeniously designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, diverse baseline comparisons, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and detailed experimental analysis.
- Value: ⭐⭐⭐⭐ High generality, providing valuable practical references for accelerating generalizable NeRFs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](nerfprior_learning_neural_radiance_field_as_a_prior_for_indoor_scene_reconstruct.md)
- [\[CVPR 2026\] MU-GeNeRF: Multi-view Uncertainty-guided Generalizable Neural Radiance Fields for Distractor-aware Scene](../../CVPR2026/3d_vision/mu-generf_multi-view_uncertainty-guided_generalizable_neural_radiance_fields_for.md)
- [\[CVPR 2025\] ProbeSDF: Light Field Probes for Neural Surface Reconstruction](probesdf_light_field_probes_for_neural_surface_reconstruction.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[ECCV 2024\] Efficient Depth-Guided Urban View Synthesis (EDUS)](../../ECCV2024/3d_vision/efficient_depth-guided_urban_view_synthesis.md)

</div>

<!-- RELATED:END -->
