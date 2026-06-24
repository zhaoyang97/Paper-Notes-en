---
title: >-
  [Paper Note] One2Any: One-Reference 6D Pose Estimation for Any Object
description: >-
  [CVPR 2025][Human Understanding][Single-Reference] This paper proposes One2Any, which estimates the 6D pose of any novel object using only a single reference image. It encodes the reference pose using Reference Object Coordinates (ROC, based on the reference camera frame rather than canonical coordinates), conditionally generates dense ROC maps via VQVAE+U-Net, and restores the pose using the Umeyama algorithm. It achieves 93.7% ADD-S AUC on YCB-Video with an inference time o…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Single-Reference"
  - "6D Pose"
  - "Conditional Generation"
  - "Reference Object Coordinates"
  - "VQVAE"
date: 2026-05-08
content_hash: 95eba7a55277f50a
---

# One2Any: One-Reference 6D Pose Estimation for Any Object

**Conference**: CVPR 2025  
**arXiv**: [2505.04109](https://arxiv.org/abs/2505.04109)  
**Code**: [https://github.com/lmy1001/One2Any](https://github.com/lmy1001/One2Any)  
**Area**: Human Understanding / 6D Pose Estimation  
**Keywords**: Single-Reference, 6D Pose, Conditional Generation, Reference Object Coordinates, VQVAE

## TL;DR

This paper proposes One2Any, which estimates the 6D pose of any novel object using only a single reference image. It encodes the reference pose using Reference Object Coordinates (ROC, based on the reference camera frame rather than canonical coordinates), conditionally generates dense ROC maps via VQVAE+U-Net, and restores the pose using the Umeyama algorithm. It achieves 93.7% ADD-S AUC on YCB-Video with an inference time of only 0.09 seconds.

## Background & Motivation

**Background**: 6D object pose estimation is crucial for robotic grasping and AR. Traditional methods require precise CAD models or multi-view reference images. Recent "CAD-free / few-reference" methods (such as FoundationPose and Oryon) still require multi-view geometry or costly online inference.

**Limitations of Prior Work**: (1) FoundationPose requires 1 second/frame for inference (11× slower than One2Any); (2) Oryon requires video sequences rather than a single reference; (3) NOCS (Normalized Object Coordinate Space) requires a canonical coordinate definition for the object, which is infeasible for novel objects.

**Key Challenge**: The information in a single reference image is extremely limited (only one view), but 6D pose estimation requires understanding the complete geometry of the object.

**Key Insight**: Abandon the canonical coordinate assumption of NOCS and instead use the reference camera frame as the coordinate system. ROC can be defined solely from the reference image itself, without requiring any prior knowledge of the object's geometry.

**Core Idea**: Substituting NOCS with Reference Object Coordinates (ROC) + conditional generation of dense coordinate maps = single-reference 6D pose estimation for any object.

## Method

### Key Designs

1. **Reference Object Coordinates (ROC)**:

    - **Function**: Defines object surface coordinates using the reference camera frame as the coordinate system.
    - **Mechanism**: Generates a 3D point cloud of the object from the depth map and mask of the reference image, directly using the coordinates in the reference camera coordinate system as the ROC. This requires no CAD model or canonical coordinates of the object.
    - **Design Motivation**: Ablation shows that leveraging ROC is 6.5% higher than directly predicting rotation/translation (91.2% vs 84.7% ADD-S).

2. **ROPE Encoder + OPD Decoder**:

    - **Function**: Generates the dense ROC map of the query image from the reference image.
    - **Mechanism**: The ROPE encoder encodes the RGB+ROC+mask of the reference image into an object representation. The OPD decoder, based on a pre-trained VQVAE + U-Net, generates the ROC map of the query image via cross-attention conditioned on the ROPE features.
    - **Design Motivation**: Conditional generation is better suited for handling large viewpoint variations than feature matching, as it can "imagine" unseen object surfaces.

3. **Umeyama Pose Recovery**:

    - **Function**: Recovers the 6D pose from the predicted ROC map and the query depth map.
    - **Mechanism**: Aligns the predicted ROC 3D points with the actual 3D points of the query image using the Umeyama algorithm.
    - **Design Motivation**: A classic geometric method that is robust and efficient.

### Loss & Training

Smooth L1 loss: $\mathcal{L} = \frac{1}{N}\sum_{i,j} Q_M(i,j) E(i,j)$, with $\beta=0.1$. Inference takes 0.09s/frame.

## Key Experimental Results

### Main Results

| Dataset | One2Any | Oryon | FoundationPose |
|--------|---------|-------|---------------|
| YCB ADD-S AUC | **93.7%** | 13.3% | 92.7% |
| Real275 AR | **54.9%** | 46.5% | - |
| Inference Time | **0.09s** | 0.95s | 1.0s |

### Ablation Study

| Configuration | ADD-S AUC |
|------|-----------|
| Direct Rotation/Translation Prediction | 84.7% |
| **ROC Representation** | **91.2%** |
| RGB+Depth Input | 90.0% |
| **RGB+ROC+Mask Input** | **91.2%** |

### Key Findings
- **Substituting NOCS with ROC is key**: No canonical coordinates are needed; the reference frame is sufficient for definition.
- **Extremely fast inference**: 0.09 seconds, suitable for real-time robotic applications.
- **Oryon collapses under a single reference**: 13.3% vs 93.7%—methods specifically designed for multi-view fail to transfer.

## Highlights & Insights
- **The simplicity and elegance of ROC**—no prior knowledge of the object is required; the reference image itself defines the coordinate system.
- **Generative vs. Discriminative**—utilizing conditional generation rather than feature matching to handle large viewpoint variations, which is more robust.

## Limitations & Future Work
- Requires ground-truth (GT) depth and masks.
- Performs poorly on textureless objects (only 33.1% on LINEMOD ape).
- The quality of the reference viewpoint affects performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of substituting NOCS with ROC is simple and powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets including Real275/YCB/LINEMOD/Toyota.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ A practical single-reference 6D pose solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image](unopose_unseen_object_pose_estimation_with_an_unposed_rgb-d_reference_image.md)
- [\[CVPR 2025\] Any6D: Model-free 6D Pose Estimation of Novel Objects](any6d_model-free_6d_pose_estimation_of_novel_objects.md)
- [\[AAAI 2026\] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](../../AAAI2026/human_understanding/coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
