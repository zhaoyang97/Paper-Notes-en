---
title: >-
  [Paper Note] Cubify Anything: Scaling Indoor 3D Object Detection
description: >-
  [CVPR 2025][Autonomous Driving][Indoor 3D Object Detection] This paper proposes the Cubify Anything 1M (CA-1M) dataset—the first large-scale indoor 3D detection dataset with exhaustive annotations of all objects on laser scans (440K objects / 1K scenes / 3.5K captures / 13M frames / pixel-perfect projection), and introduces CuTR, a pure Transformer detector, demonstrating that without 3D inductive biases (point clouds/voxels), 3D detection can outperform point-cloud-based met…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Indoor 3D Object Detection"
  - "Large-scale Dataset"
  - "Transformer"
  - "Pixel-level 3D Annotation"
  - "Category-agnostic"
date: 2026-05-08
content_hash: 79653a5ea5047e8b
---

# Cubify Anything: Scaling Indoor 3D Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2412.04458](https://arxiv.org/abs/2412.04458)  
**Code**: Dataset and models will be released  
**Area**: Autonomous Driving  
**Keywords**: Indoor 3D Object Detection, Large-scale Dataset, Transformer, Pixel-level 3D Annotation, Category-agnostic

## TL;DR
This paper proposes the Cubify Anything 1M (CA-1M) dataset—the first large-scale indoor 3D detection dataset with exhaustive annotations of all objects on laser scans (440K objects / 1K scenes / 3.5K captures / 13M frames / pixel-perfect projection), and introduces CuTR, a pure Transformer detector, demonstrating that without 3D inductive biases (point clouds/voxels), 3D detection can outperform point-cloud-based methods when data is abundant.

## Background & Motivation

**Background**: Indoor 3D object detection is primarily conducted on datasets like SUN RGB-D and ScanNet, where models typically operate on 3D point clouds or voxels. Mainstream methods rely on specialized operations such as sparse 3D convolutions and require 3D inductive biases to compensate for the insufficient dataset scale.

**Limitations of Prior Work**: (1) Existing datasets lack exhaustive annotations, focusing primarily on large room-scale objects (chairs, beds, tables) while ignoring daily small objects; (2) The accuracy of annotation sources is limited, as they are based on noisy 3D reconstructions from consumer-grade depth sensors, leading to obvious misalignments when projected back to images; (3) Point cloud methods couple model design with annotation biases, limiting scalability; (4) 3D inductive biases (sparse convolutions, KNN) are unfriendly to hardware accelerators other than GPUs.

**Key Challenge**: The existing "small data + strong inductive bias" paradigm is effective on small-scale datasets but restricts scaling to more object categories and larger scales.

**Goal**: To construct a large-scale, high-precision, exhaustively annotated dataset and verify whether a model with minimal inductive bias can outperform complex 3D methods under data abundance.

**Key Insight**: Leverage high-precision FARO laser scans from ARKitScenes (rather than consumer-grade depth) for annotation, and utilize precise registration between the laser scan and hand-held captures to achieve pixel-perfect frame-level projection.

**Core Idea**: Exhaustively annotate all objects (category-agnostic) on laser scans, precisely project them to each image frame to obtain pixel-level 3D bounding box annotations, and then use a pure Transformer to directly predict 3D boxes from 2D features.

## Method

### Overall Architecture
Divided into two parts, the dataset and the model: (1) CA-1M dataset: Annotate 9-DOF 3D bounding boxes on FARO laser scans $\rightarrow$ project to each hand-held capture frame using registration matrices $\rightarrow$ render considering view frustum and occlusion $\rightarrow$ obtain 2D+3D box annotations for each frame; (2) CuTR model: ViT backbone extracts 2D features $\rightarrow$ single-stage/single-scale detection head $\rightarrow$ directly predicts 2D and 3D boxes from RGB(-D), without any 3D spatial operations.

### Key Designs

1. **CA-1M Dataset Construction**:

    - **Function**: Provides the first high-precision, exhaustively annotated, pixel-perfect large-scale indoor 3D detection dataset.
    - **Mechanism**: Annotate 9-DOF 3D bounding boxes (category-agnostic) for all visible objects, including small ones, in each scene on FARO laser scans (sub-centimeter accuracy). Then, project the 3D bounding boxes onto each frame using the precise registration between laser scans and iPad Pro RGB-D captures provided by ARKitScenes. Handle view frustum truncation and occlusion relationships during projection to ensure pixel-level consistency between 2D/3D boxes and frame content. 1000+ scenes, 3500+ captures, 440K unique objects, 13M training frames.
    - **Design Motivation**: Address three major limitations of existing datasets: non-exhaustive (only annotating large objects), imprecise (noisy depth annotations), and inconsistent (misalignment when projecting 3D boxes back to images).

2. **CuTR (Cubify Transformer)**:

    - **Function**: A pure Transformer 3D object detector that directly predicts 3D bounding boxes from 2D features.
    - **Mechanism**: Use a pre-trained ViT backbone to process RGB images and optional depth maps, extracting 2D feature maps. A single-stage detection head (similar to DETR) is attached to simultaneously output 2D boxes and 3D boxes (including 3D centers, sizes, and orientations). The entire process involves no 3D spatial operations (no voxelization, no sparse convolutions, no KNN).
    - **Design Motivation**: Validate the hypothesis that "large data can substitute 3D inductive biases." If the data is of high-enough quality and scale, a simple Transformer can learn to reason 3D directly from 2D.

3. **Rendering Pipeline from 3D Boxes to Frame-Level Annotations**:

    - **Function**: Convert scene-level 3D bounding box annotations into 2D+3D bounding box annotations for each image frame.
    - **Mechanism**: For each frame's camera pose, project the 3D boxes into the camera coordinate system and check if they lie within the view frustum; calculate occlusion relationships (whether other objects block the current object); generate a list of visible objects and their 2D/3D boxes for each frame. This process guarantees pixel-level alignment between annotations and image content.
    - **Design Motivation**: Scene-level annotations cannot be directly used for frame-level training; a precise rendering pipeline is crucial for high-quality frame-level annotations.

### Loss & Training
Standard detection loss: classification loss + 2D box regression loss + 3D box regression loss (center/size/orientation). Training only requires pre-trained weights for the ViT backbone, with no need for complex 3D infrastructure.

## Key Experimental Results

### Main Results

| Method | CA-1M 3D Recall@62% | SUN RGB-D (After Pre-training) | Input |
|------|---------------------|---------------------|------|
| CuTR | **62%+** | **Outperforms point cloud methods** | RGB-D |
| VoteNet (Point Cloud) | Low | Competitive | Point cloud |
| ImVoteNet (Hybrid) | Medium | Competitive | RGB-D+Point cloud |

### Ablation Study

| Configuration | Key Effect |
|------|---------|
| CuTR RGB-Only (No Depth) | Promising performance, demonstrating the feasibility of reasoning 3D from pure images |
| CuTR RGB-D | Best performance, depth provides critical geometric cues |
| Point cloud methods on CA-1M | Underperforms CuTR; 3D inductive biases become a limitation under large-scale data |
| CuTR pre-trained on CA-1M $\rightarrow$ SUN RGB-D | Outperforms point cloud methods, proving pre-training transferability |

### Key Findings
- At the scale of CA-1M, CuTR without 3D inductive biases outperforms point-cloud-based methods—proving that "data > inductive bias" also holds true in 3D detection.
- Noisy consumer-grade LiDAR depth affects point cloud methods more severely, whereas CuTR shows greater robustness in handling noisy depth.
- RGB-only CuTR also exhibits promising performance, suggesting that depth input might not be strictly necessary for 3D detection.
- CA-1M pre-training significantly boosts performance on smaller datasets (demonstrating the transfer learning value).

## Highlights & Insights
- The "data decoupling hypothesis" is inspiring—decoupling the spatial precision of data (annotated on laser scans) and frame-level perfection (pixel-level projection) allows the same annotation to serve both scene-level and frame-level stages.
- The category-agnostic exhaustive annotation strategy establishes the data foundation for the "detect anything" trend.
- The minimalist design of CuTR (pure Transformer without 3D operations) is hardware-friendly and can run on various accelerators.

## Limitations & Future Work
- The dataset is based on 1000+ scenes from ARKitScenes, so diversity is still limited (mostly indoor residential/office settings).
- Category-agnostic annotations lack semantic information, making them difficult to directly apply to downstream tasks requiring category labels.
- Frame-level predictions lack temporal consistency; video-level detection and tracking are natural extensions.
- Can be combined with MLLMs to achieve natural language-based spatial understanding.

## Related Work & Insights
- **vs SUN RGB-D**: Small-scale (10K frames), only large objects annotated, limited annotation accuracy. CA-1M is over 10x larger and pixel-perfect.
- **vs ScanNet**: Lacks explicit 3D bounding box annotations, containing only instance segmentations. CA-1M provides precise 9-DOF boxes.
- **vs ARKitScenes**: Annotated on noisy handheld reconstructions and restricted to only 21 categories of large objects. CA-1M is exhaustively annotated on high-precision laser scans.
- **vs Omni3D**: Focuses on cross-dataset generalization. CA-1M pursues exhaustiveness and precision within a single dataset.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The dataset construction paradigm represents a significant shift (from coarse 3D to pixel-perfect), and CuTR validates an important hypothesis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage of CA-1M and SUN RGB-D, including transfer learning validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-argued motivation, detailed description of dataset design.
- **Value**: ⭐⭐⭐⭐⭐ The dataset and model present significant infrastructure value for indoor 3D understanding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PAP: A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[ECCV 2024\] Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene](../../ECCV2024/autonomous_driving/approaching_outside_scaling_unsupervised_3d_object_detection_from_2d_scene.md)
- [\[CVPR 2025\] Segment Anything, Even Occluded](segment_anything_even_occluded.md)
- [\[CVPR 2025\] EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)

</div>

<!-- RELATED:END -->
