---
title: >-
  [Paper Note] TCC-Det: Temporarily Consistent Cues for Weakly-Supervised 3D Detection
description: >-
  [ECCV 2024][3D Vision][Weakly-Supervised 3D Detection] This paper proposes TCC-Det, a weakly-supervised 3D object detection method that requires absolutely no manual 3D annotations. By leveraging an off-the-shelf 2D detector (Mask-RCNN) and multi-frame temporal consistency cues, it generates high-quality pseudo 3D labels to train a 3D point cloud detector (Voxel-RCNN). It outperforms all prior weakly-supervised methods on KITTI and Waymo, significantly narrowing the gap to fu…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Weakly-Supervised 3D Detection"
  - "Pseudo-label Generation"
  - "Temporal Consistency"
  - "Point Cloud Object Detection"
  - "LiDAR"
date: 2026-05-08
content_hash: 678f4434bb86c51d
---

# TCC-Det: Temporarily Consistent Cues for Weakly-Supervised 3D Detection

**Conference**: ECCV 2024  
**Paper Link**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3732_ECCV_2024_paper.php)  
**Code**: [GitHub](https://github.com/jskvrna/TCC-Det)  
**Area**: 3D Vision / Autonomous Driving / Weakly-Supervised Learning  
**Keywords**: Weakly-Supervised 3D Detection, Pseudo-label Generation, Temporal Consistency, Point Cloud Object Detection, LiDAR

## TL;DR

This paper proposes TCC-Det, a weakly-supervised 3D object detection method that requires absolutely no manual 3D annotations. By leveraging an off-the-shelf 2D detector (Mask-RCNN) and multi-frame temporal consistency cues, it generates high-quality pseudo 3D labels to train a 3D point cloud detector (Voxel-RCNN). It outperforms all prior weakly-supervised methods on KITTI and Waymo, significantly narrowing the gap to fully-supervised methods.

## Background & Motivation

**Background**: 3D object detection in LiDAR point clouds is a core capability for autonomous driving and robotics. Current mainstream approaches rely on a large volume of manually annotated 3D bounding boxes to train detectors. Although fully-supervised methods like Voxel-RCNN and PointPillars achieve excellent performance, their heavy reliance on annotated data severely limits their scalability.

**Limitations of Prior Work**: 3D annotation is extremely expensive and time-consuming, requiring annotators to precisely box the 3D position, dimensions, and orientation of each object in 3D point clouds. This leads to two issues: first, the volume of available annotated data is limited; second, it is difficult to cover edge cases and rare scenarios in the datasets, as the probability of these occurrences is naturally low in small datasets.

**Key Challenge**: Fully-supervised methods require massive amounts of precise 3D annotations to guarantee performance, but the cost of acquiring these annotations is extremely high, creating a trilemma of "annotation cost vs. data scale vs. detection performance". While existing weakly-supervised methods attempt to replace 3D annotations with 2D annotations or other weak signals, the generated pseudo-labels suffer from poor quality, and single-frame information is insufficient to accurately reconstruct 3D geometry.

**Goal**: (1) How to generate high-quality 3D pseudo-labels entirely without manual 3D annotations? (2) How to utilize multi-frame temporal consistency to compensate for information loss when transitioning from single-frame 2D detection to 3D estimation? (3) How to train a 3D detector on pseudo-labels to achieve performance close to fully-supervised baselines?

**Key Insight**: The authors observe that the real world exhibits temporal consistency—the geometric shape and position of the same object are coherent across consecutive frames. By aggregating multi-frame 2D detection results and point cloud data, more complete 3D geometric information can be recovered compared to single frames. This observation, combined with off-the-shelf high-quality 2D detectors (e.g., Mask-RCNN), provides a path to obtain reliable 3D supervision signals without any 3D annotations.

**Core Idea**: Utilizing the temporal consistency of multi-frame RGB and LiDAR data, high-quality 3D pseudo-labels are automatically generated through an off-the-shelf 2D detector, frame aggregation, and an optimization pipeline, followed by fine-tuning the 3D detector with newly designed loss functions.

## Method

### Overall Architecture

The training of TCC-Det consists of two stages: (1) Pseudo-label generation stage: The inputs are multi-frame raw sensor data (including RGB images and LiDAR point clouds). Through precise inter-frame ego-motion estimation, 2D detection and tracking, multi-frame aggregation, and optimization, the pipeline outputs 3D pseudo bounding boxes for each target. (2) Detector training stage: Standard 3D detectors (Voxel-RCNN) are trained using the generated pseudo-labels and then fine-tuned in a second round with additionally designed TFL and MAL losses.

### Key Designs

1. **Multi-frame Pseudo-label Generation Pipeline**:

    - **Function**: Automatically generates high-quality 3D bounding box pseudo-labels from unlabeled sensor data.
    - **Mechanism**: The pipeline is divided into four steps. First, precise frame-to-frame transformation matrices are obtained via inter-frame ego-motion estimation (utilizing LiDAR point cloud registration or vehicle IMU/GPS data) to align all frames into a unified coordinate system. Second, Mask-RCNN is run on each RGB frame to obtain 2D instance segmentation masks, and tracking is applied to establish cross-frame correspondences. Third, the LiDAR point clouds of the same object across multiple frames are aggregated based on the transformation matrices, creating a more complete 3D point cloud representation. Finally, an optimization process (incorporating shape prior assumptions) fits the aggregated point cloud to precise 3D bounding boxes.
    - **Design Motivation**: Point clouds from a single frame are highly sparse for distant objects, often consisting of only a few points, making accurate 3D box estimation impossible. Through multi-frame aggregation, the accumulated points for the same object increase dramatically, providing a more complete geometric layout for highly accurate 3D box estimation.

2. **Temporal Fitting Loss (TFL)**:

    - **Function**: Incorporates temporal consistency constraints during the detector fine-tuning stage to improve pseudo-label utilization.
    - **Mechanism**: TFL exploits the coherence of object positions and orientations across adjacent frames. For detection results of the same object in consecutive frames, TFL constrains them to be consistent after applying frame transformations. Specifically, the detection box of frame $t$ is mapped to frame $t+1$ using the transformation matrix, and the deviation from the detection result in frame $t+1$ is calculated as the loss. This ensures that even if pseudo-labels for certain frames are inaccurate, the temporal constraint can correct or alleviate the errors.
    - **Design Motivation**: Pseudo-labels inevitably contain noise; training solely on standard detection losses might cause the model to fit this noise. TFL offers an implicit label denoising mechanism through cross-frame constraints.

3. **Multi-frame Alignment Loss (MAL)**:

    - **Function**: Further leverages multi-frame information to enhance the detector's accuracy in estimating 3D shapes and positions.
    - **Mechanism**: MAL utilizes multi-frame alignment clues at the feature level. For feature representations of the same object in different frames, MAL encourages them to remain consistent after coordinate transformation. This scales the temporal consistency prior from the label level down to the feature level, facilitating more robust representations.
    - **Design Motivation**: While pseudo-labels can only constrain the output layer, MAL enforces temporal consistency from the perspective of feature learning, providing a deeper layer of supervision that aids the model in learning better 3D spatial representations.

### Loss & Training

Training is divided into two steps: First, Voxel-RCNN is trained with standard detection losses using pseudo-labels (50 epochs, batch size 25). Second, leveraging the model trained in the first step, fine-tuning is performed with TFL and MAL (10 epochs, batch size 2). This two-stage strategy ensures that the model first learns basic detection capabilities from pseudo-labels, and then further refines accuracy via temporal consistency losses.

## Key Experimental Results

### Main Results

| Dataset | Difficulty | Metric (AP) | TCC-Det | Prev. SOTA | Fully Supervised (Voxel-RCNN) |
|--------|------|----------|---------|---------------|-------------------|
| KITTI (Car) | Easy | AP 3D | ~82% | ~72% | 92.38% |
| KITTI (Car) | Moderate | AP 3D | ~72% | ~60% | 85.29% |
| KITTI (Car) | Hard | AP 3D | ~68% | ~56% | 82.86% |
| Waymo (Vehicle) | Overall | AP 3D | Significant gain | - | Fully supervised baseline |

### Ablation Study

| Configuration | KITTI Mod. AP | Description |
|------|--------------|------|
| Full model (TCC-Det) | ~72% | Full model (pseudo-labels + TFL + MAL) |
| w/o TFL | ~69% | Dropping temporal fitting loss decreases performance by ~3% |
| w/o MAL | ~70% | Dropping multi-frame alignment loss decreases performance by ~2% |
| Single-frame pseudo-labels | ~62% | Standard single-frame approach without multi-frame aggregation drops performance significantly |
| w/o optimization step | ~66% | Performance drops significantly without the 3D box optimization |

### Key Findings
- Multi-frame aggregation is the largest contributor to performance improvement. Without multi-frame aggregation, performance drops by approximately 10 percentage points, demonstrating that temporally consistent multi-frame information is crucial for supplementing sparse single-frame point clouds.
- TFL and MAL each contribute a 2-3% improvement in the fine-tuning stage, complementing each other.
- On KITTI, TCC-Det narrows the gap between weakly-supervised and fully-supervised methods from approximately 25% to around 13%, representing a significant advancement.

## Highlights & Insights
- **Zero-Annotation Training**: The entire method requires no human 3D annotations, utilizing only off-the-shelf 2D detectors and raw sensor data to train 3D detectors. This implies that massive amounts of training data can be acquired cheaply, potentially even using daily fleet driving data, completely breaking the annotation bottleneck.
- **Temporal Consistency as a Free Supervision Signal**: The brilliance lies in exploiting the intrinsic physical structure of the world—the fact that objects remain consistent across consecutive frames—as a free supervision signal. This paradigm can be extended to other tasks requiring 3D annotations (e.g., 3D semantic segmentation, 3D tracking).
- **Sound Engineering Pipeline Design**: Although the method involves multiple steps, each step utilizes mature, off-the-shelf tools (Mask-RCNN, point cloud registration, etc.), keeping the actual deployment barrier low.

## Limitations & Future Work
- The evaluation is currently restricted to vehicle detection; the effectiveness on small objects like pedestrians and cyclists is unknown, as their LiDAR points are sparser, potentially limiting the impact of multi-frame aggregation.
- The pseudo-label generation pipeline is long (motion estimation $\rightarrow$ 2D detection $\rightarrow$ tracking $\rightarrow$ aggregation $\rightarrow$ optimization), presenting high computational overhead, making it less suitable for online real-time scenarios.
- It relies heavily on precise ego-motion estimation. If IMU/GPS signals are degraded or LiDAR point cloud registration fails, pseudo-label quality will significantly deteriorate.
- It might perform better on static objects. Fast-moving objects may register motion blur/ghosting effects during multi-frame aggregation, necessitating more precise motion compensation.

## Related Work & Insights
- **vs VS3D**: Methods like VS3D also perform weakly-supervised 3D detection, but rely heavily on single-frame 2D-to-3D mapping, lacking temporal dimension information. TCC-Det outperforms them significantly by obtaining a more complete 3D geometry via multi-frame aggregation.
- **vs WS3D**: WS3D requires center-point annotations as weak supervision signals, whereas TCC-Det requires absolutely no 3D-level annotations, resulting in much lower overhead.
- **vs Self-supervised Pre-training Methods**: Some methods (e.g., BEV-MAE) leverage self-supervised pre-training to reduce reliance on annotations, but still require high-quality fine-tuning annotations. TCC-Det directly replaces manual labels with pseudo-labels.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The integration of multi-frame temporal consistency for weakly-supervised 3D detection is clear; the designs of TFL and MAL are reasonable but not exceptionally novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two major benchmarks, KITTI and Waymo, with complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clearly described, and the pipeline logic is sound.
- **Value**: ⭐⭐⭐⭐ High practical value; zero-annotation training of 3D detectors is of great significance for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](../../CVPR2026/3d_vision/rewis3d_reconstruction_improves_weaklysupervised_s.md)
- [\[ECCV 2024\] TC-Stereo: Temporally Consistent Stereo Matching](temporally_consistent_stereo_matching.md)
- [\[ECCV 2024\] Interactive 3D Object Detection with Prompts](interactive_3d_object_detection_with_prompts.md)
- [\[AAAI 2026\] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization](../../AAAI2026/3d_vision/enhancing_generalization_of_depth_estimation_foundation_model_via_weakly-supervi.md)
- [\[ECCV 2024\] Formula-Supervised Visual-Geometric Pre-training (FSVGP)](formula-supervised_visual-geometric_pre-training.md)

</div>

<!-- RELATED:END -->
