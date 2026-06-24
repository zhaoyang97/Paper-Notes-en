---
title: >-
  [Paper Note] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation
description: >-
  [ICCV 2025][3D Vision][3D human pose estimation] This paper proposes the PersPose framework, which addresses the inaccurate depth estimation caused by existing methods neglecting field-of-view (FOV) information. It encodes cropped camera intrinsics as a 2D map via Perspective Encoding (PE) and centers the subject through Perspective Rotation (PR) to eliminate perspective distortion.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D human pose estimation"
  - "perspective encoding"
  - "perspective rotation"
  - "camera intrinsics"
  - "monocular"
date: 2026-05-08
content_hash: 13f9dcb6855c6897
---

# PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation

**Conference**: ICCV 2025
**arXiv**: [2508.17239](https://arxiv.org/abs/2508.17239)  
**Code**: [GitHub](https://github.com/KenAdamsJoseph/PersPose)  
**Area**: 3D Vision
**Keywords**: 3D human pose estimation, perspective encoding, perspective rotation, camera intrinsics, monocular

## TL;DR

This paper proposes the PersPose framework, which addresses the inaccurate depth estimation caused by existing methods neglecting field-of-view (FOV) information. It encodes cropped camera intrinsics as a 2D map via Perspective Encoding (PE) and centers the subject through Perspective Rotation (PR) to eliminate perspective distortion.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Existing 3D HPE methods that use cropped images as input suffer from two commonly overlooked issues:

**Depth information loss due to cropping**: Identical cropped images may correspond to different relative depths (Figure 2: subjects a and b share the same crop but differ in depth); different crops may correspond to the same depth (subjects a and c).

**FOV is critical for depth estimation**: Even when using full images, the absence of FOV information leads to incorrect depth estimates (Figure 3: two cameras with different FOVs capture the same subject, yielding visually distinct images with identical depth labels).

Core insight: The cropping operation is equivalent to modifying the camera intrinsics. The post-crop intrinsics $K^{\text{crop}} = AK$ encapsulate both cropping and FOV information.

## Method

### Perspective Encoding

The cropped intrinsics $K^{\text{crop}}$ are encoded into a 2D PE map $M^{xy}$ by projecting each pixel coordinate $(u_i, v_i)$ onto the $z=1$ plane:

$$(K^{\text{crop}})^{-1} \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

The projected region on the $z=1$ plane geometrically encodes a unique view frustum. Different focal lengths correspond to regions of different sizes, and an off-axis principal point corresponds to a shifted region.

The PE map and the cropped image are processed through separate convolutional layers and then combined via element-wise addition.

### Perspective Rotation

Since the subject can appear at arbitrary positions in the image, the principal point $(c_x^{\text{crop}}, c_y^{\text{crop}})$ varies substantially, increasing the difficulty of fitting. PR addresses this by rotating the scene to center the subject:

1. Project the bounding box center onto the $z=1$ plane to obtain $(x_c, y_c, 1)$.
2. Compute the rotation axis and angle:
$$\mathbf{n} = \frac{(x_c, y_c, 1) \times (0,0,d)^\top}{\|(x_c, y_c, 1) \times (0,0,d)^\top\|}$$
$$\phi = \arccos\frac{(x_c, y_c, 1) \cdot (0,0,d)}{\|(x_c, y_c, 1)\| \cdot \|(0,0,d)\|}$$
3. Derive the rotation matrix $R$ via the Rodrigues formula; the perspective transformation matrix is then $M = KRK^{-1}$.

After PR, the mapping function is simplified from 4 inputs to 2 inputs:
$$f_\theta: (I^{\text{crop}}, f^{\text{crop}}, c_x^{\text{crop}}, c_y^{\text{crop}}) \rightarrow P_{\text{XYZ}}$$
reduced to:
$$\tilde{f}_\theta: (I^{\text{crop}}, f^{\text{crop}}) \rightarrow P_{\text{XYZ}}$$

### Inference Pipeline

1. Apply PR to the original image to obtain a centered image $I'$, then crop from the center.
2. Compute $K^{\text{crop}}$ and encode it as a PE map.
3. The network predicts 2D joint coordinates with relative depth $P_{\text{UVD}}$ and a scale factor $\hat{s}$.
4. Combined with the intrinsics, these are converted to the rotated 3D pose $P'_{\text{XYZ}}$.
5. Apply the inverse rotation: $P_{\text{XYZ}} = R^\top P'_{\text{XYZ}}$.

## Key Experimental Results

### Main Results — 3DPW Dataset

| Method | PA-MPJPE↓ | MPJPE↓ |
|--------|-----------|--------|
| HMR | 81.3 | 130.0 |
| SPIN | 59.2 | 96.9 |
| CLIFF | 43.0 | 69.0 |
| Prev. SOTA | — | 65.0 |
| **PersPose** | **38.7** | **60.1** |

PersPose achieves a 7.54% reduction in MPJPE on the in-the-wild dataset 3DPW, establishing a new state of the art.

### Multi-Dataset Comparison

| Dataset | Metric | PersPose |
|---------|--------|----------|
| 3DPW | MPJPE↓ | **60.1** |
| Human3.6M | MPJPE↓ | Competitive |
| MPI-INF-3DHP | PCK↑ | SOTA |

PersPose achieves consistent state-of-the-art or competitive results across multiple benchmarks.

## Highlights & Insights

1. **Strong conceptual insight**: The paper clearly demonstrates that cropping is equivalent to modifying camera intrinsics, revealing a long-overlooked issue in the field.
2. **Elegant solution**: The PE and PR modules are individually simple yet highly targeted.
3. **Grounded in physical principles**: The design is derived from camera imaging geometry rather than relying on black-box approaches.
4. **Plug-and-play**: Both PE and PR can be integrated into any existing HPE framework.

## Limitations & Future Work

- Requires known or accessible camera focal length information.
- The PR operation introduces additional image transformation computation.
- Nonlinear distortion from extreme wide-angle lenses is not modeled.
- Temporal consistency across video sequences remains unexplored.

## Related Work & Insights

- CLIFF: Corrects global rotation using bounding box information.
- SPEC: Estimates focal length from images.
- Ray3D: Leverages intrinsics to lift 2D keypoints into 3D rays.

## Rating

- Novelty: ⭐⭐⭐⭐ (PE and PR are cleverly designed)
- Technical Depth: ⭐⭐⭐⭐ (geometric derivations are complete and clear)
- Experimental Thoroughness: ⭐⭐⭐⭐ (three datasets + ablation study)
- Practical Value: ⭐⭐⭐⭐⭐ (a fundamental improvement for real-world HPE)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2025\] Pano360: Perspective to Panoramic Vision with Geometric Consistency](../../CVPR2025/3d_vision/pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)
- [\[ICLR 2026\] SpatialHand: Generative Object Manipulation from 3D Perspective](../../ICLR2026/3d_vision/spatialhand_generative_object_manipulation_from_3d_prespective.md)
- [\[CVPR 2026\] Affine Perspective-Three-Point Problem](../../CVPR2026/3d_vision/affine_perspective-three-point_problem.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](../../CVPR2025/3d_vision/extreme_rotation_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
