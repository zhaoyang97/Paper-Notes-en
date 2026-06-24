---
title: >-
  [Paper Note] Any6D: Model-free 6D Pose Estimation of Novel Objects
description: >-
  [CVPR 2025][Human Understanding][6D pose estimation] This paper proposes the Any6D framework to estimate the 6D pose and size of novel objects from a single RGB-D anchor image. By combining InstantMesh 3D reconstruction, oriented bounding box coarse alignment, and joint size-pose refinement, Any6D achieves an ADD-S of 98.7% on HO3D, significantly outperforming GEDI's 71.9%.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "6D pose estimation"
  - "model-free"
  - "single anchor"
  - "InstantMesh"
  - "FoundationPose"
  - "render-and-compare"
date: 2026-05-08
content_hash: 5614944a69a33335
---

# Any6D: Model-free 6D Pose Estimation of Novel Objects

**Conference**: CVPR 2025  
**arXiv**: [2503.18673](https://arxiv.org/abs/2503.18673)  
**Code**: [Project Page](https://taeyeop.com/any6d)  
**Area**: Human Understanding  
**Keywords**: 6D pose estimation, model-free, single anchor, InstantMesh, FoundationPose, render-and-compare

## TL;DR
This paper proposes the Any6D framework to estimate the 6D pose and size of novel objects from a single RGB-D anchor image. By combining InstantMesh 3D reconstruction, oriented bounding box coarse alignment, and joint size-pose refinement, Any6D achieves an ADD-S of 98.7% on HO3D, significantly outperforming GEDI's 71.9%.

## Background & Motivation

**Background**: 6D object pose estimation is crucial in robotic manipulation and augmented reality. Existing methods can be categorized into instance-level (requiring precise CAD models), category-level (requiring category priors), and category-agnostic methods.

**Limitations of Prior Work**:
   - CAD model-based methods require precise textured 3D models, which are expensive to acquire.
   - Multi-view methods (Gen6D, OnePose, FoundationPose model-free mode) require multiple reference images or video sequences.
   - Single-view matching methods (Oryon, LoFTR) suffer from drastic performance drops under occlusion or non-overlapping viewpoints.

**Key Challenge**: In practical robotic scenarios, robots facing novel objects in new environments cannot acquire CAD models or multi-view images. None of the existing methods can handle this effectively.

**Key Insight**: Leveraging image-to-3D generative models (InstantMesh) to reconstruct the complete 3D shape from a single image, combined with depth information to estimate the metric scale, thereby enabling a robust full-to-partial matching.

**Core Idea**: Single RGB-D $\to$ InstantMesh normalized 3D reconstruction $\to$ Oriented bounding box coarse alignment $\to$ FoundationPose joint size-pose refinement $\to$ Render-and-compare to select the optimal hypothesis.

## Method

### Overall Architecture
Given an anchor image $I_A$ (RGB-D) and a query image $I_Q$ (RGB-D), the goal is to estimate the relative pose $\mathbf{T}_{A \to Q} \in SE(3)$.
The method consists of two steps:
1. Reconstruct a normalized shape $O_N$ from the anchor image, estimate the metric-scale shape $O_M$ and the anchor pose $T_{O_M \to A}$ via Object Alignment.
2. Estimate $T_{O_M \to Q}$ using the metric-scale shape and the query image, with the final relative pose obtained as $\mathbf{T}_{A \to Q} = (T_{O_M \to A})^{-1} \cdot T_{O_M \to Q}$.

### Key Designs

1. **3D Shape Reconstruction (InstantMesh)**

    - **Function**: Generates a normalized 3D mesh $O_N$ (range $[-1, 1]$) from the RGB of the anchor image.
    - **Core Limitation**: The generated shape lacks a metric scale and cannot be directly used for pose estimation.
    - **Advantages**: Compared to NeRF or partial-view reconstruction, it generates a complete shape, supporting full-to-partial matching.

2. **Coarse Object Alignment**

    - **Function**: Estimates the initial object size $s \in \mathbb{R}^3$ and the coarse pose.
    - **Mechanism**: Uses the **Oriented Bounding Box (OBB)** to determine the object center.
    - **Why not use other center estimation methods**:
        - Point cloud mean: Unreliable when partially visible.
        - Axis-Aligned Bounding Box (AABB): Center shifts under partial occlusion.
    - **Workflow**: Samples different rotation angles, calculates the bounding box IoU between $I_A$ and $O_N$, and selects the rotation + scale combination with the highest IoU.

3. **Fine Object Alignment**

    - **Function**: Jointly refines size and pose.
    - **Extensions based on FoundationPose**:
        - Original FoundationPose only samples pose hypotheses in $SO(3)$.
        - Any6D additionally samples sizes $\Delta s \in [0.6, 1.4]$.
    - **Alternate iteration of three modules**: pose estimation $\to$ size estimation $\to$ axis alignment.
    - **Render-and-Compare to select the optimal**: Pose ordering network + self-attention global scoring.

4. **Pose Selection**

    - **Two-stage strategy**: First, a pose ordering network compares the rendered images with the cropped observations, and then a self-attention module fuses all hypothesis embeddings to output the final score.

### Loss & Training
- **No additional training required**: Leverages pre-trained InstantMesh and FoundationPose.
- **Online inference**: Conducts optimization-based alignment during online inference.

## Key Experimental Results

### Main Results (HO3D Dataset)

| Method | Input Modality | ADD-S↑ | ADD↑ | AR↑ |
|------|----------|--------|------|------|
| Oryon | RGB-D+Language | 23.0 | 0.0 | 1.0 |
| LoFTR | RGB-D | 29.5 | 2.3 | 3.2 |
| GEDI | Depth | 71.9 | 9.7 | 7.4 |
| **Any6D (Ours)** | **RGB-D** | **98.7** | **40.4** | **38.3** |

### Other Datasets

| Dataset | ADD-S↑ | ADD↑ | AR↑ |
|--------|--------|------|------|
| YCBINEOAT | 89.3 | 45.6 | 37.5 |
| Toyota-Light (ADD(-S)) | 32.2 | AR: 43.3 | MSSD: 55.8 |
| REAL275 (ADD(-S)) | 53.5 | AR: 51.0 | MSPD: 65.3 |
| LM-O (vs GigaPose) | AR: 28.6 | MSPD: 36.1 | VSD: 17.6 |

### Ablation Study (HO3D Dataset)

| Configuration | ADD-S↑ | ADD↑ | AR↑ | CD↓ |
|------|--------|------|------|------|
| Baseline (NeRF partial view) | 28.6 | 0.0 | 0.2 | 1.02 |
| (1) Without any alignment | 0.0 | 0.0 | 0.0 | 1.47 |
| (2) W/o coarse size, w/ refinement + axis alignment| 98.0 | 25.5 | 26.8 | 0.53 |
| (3) W/ coarse size, w/o refinement | 83.7 | 26.6 | 22.5 | 0.92 |
| (4) W/ coarse size + refinement, w/o axis alignment | 92.3 | 23.6 | 24.9 | 0.66 |
| **Full (Ours)** | **98.7** | **40.4** | **38.3** | **0.49** |

### Key Findings
- Coarse size estimation is the foundation; omitting it leads to complete failure (Configuration 1).
- Axis alignment brings a significant improvement to ADD and AR (+14.9 AR).
- Size refinement prevents XYZ aspect ratio distortion.

## Highlights & Insights
- **Single RGB-D is sufficient**: No CAD models, multi-view images, or video sequences are required.
- **Oriented Bounding Box (OBB)** center estimation is simple and effective, addressing partial visibility issues.
- **Full-to-partial matching**: Complete reconstruction eliminates the ambiguity of partial matching.
- Significantly outperforms baseline methods in hand-occlusion (HO3D) and robotic-grasping (YCBINEOAT) scenarios.

## Limitations & Future Work
- Relies heavily on the reconstruction quality of InstantMesh; performance drops when the initial 3D shape is inaccurate.
- Currently lacks a shape update/refinement step.
- Inference speed is bottlenecked by InstantMesh.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Jointly estimates size and pose with InstantMesh + FoundationPose.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 datasets + detailed ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and complete framework.
- **Value**: ⭐⭐⭐⭐⭐ Holds significant practical value for robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](../../AAAI2026/human_understanding/coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
