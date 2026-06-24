---
title: >-
  [Paper Note] HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos
description: >-
  [CVPR 2025][3D Vision][Hand-Object Interaction] Meta releases HOT3D, the first large-scale egocentric multi-view hand-object interaction dataset based on real wearable devices (Project Aria + Quest 3). It contains 833 minutes of recordings with over 3.7 million images, capturing 19 subjects interacting with 33 objects. The paper demonstrates through experiments that multi-view methods significantly outperform single-view methods in tasks like 3D hand tracking and 6DoF object…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Hand-Object Interaction"
  - "Egocentric Multi-view"
  - "3D Hand Tracking"
  - "6DoF Object Pose"
  - "Dataset Benchmark"
date: 2026-05-08
content_hash: 2240dd2495c44cc4
---

# HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos

**Conference**: CVPR 2025  
**arXiv**: [2411.19167](https://arxiv.org/abs/2411.19167)  
**Code**: [https://facebookresearch.github.io/hot3d](https://facebookresearch.github.io/hot3d)  
**Area**: 3D Vision  
**Keywords**: Hand-Object Interaction, Egocentric Multi-view, 3D Hand Tracking, 6DoF Object Pose, Dataset Benchmark

## TL;DR

Meta releases HOT3D, the first large-scale egocentric multi-view hand-object interaction dataset based on real wearable devices (Project Aria + Quest 3). It contains 833 minutes of recordings with over 3.7 million images, capturing 19 subjects interacting with 33 objects. The paper demonstrates through experiments that multi-view methods significantly outperform single-view methods in tasks like 3D hand tracking and 6DoF object pose estimation.

## Background & Motivation

1. **Background**: Automatic understanding of hand-object interactions by vision systems has extensive application value in scenarios such as AR/VR, robotic skill transfer, and contextual AI assistants. Existing datasets are mainly divided into three categories: hand-only datasets (InterHand2.6M, FreiHAND, etc.), object-only datasets (BOP benchmarks such as YCB-V, T-LESS, etc.), and joint hand-object datasets (HO-3D, DexYCB, ARCTIC, HOI4D, etc.).

2. **Limitations of Prior Work**: (1) Almost no datasets provide hardware-synchronized multi-view egocentric video streams recorded with real wearable devices. Existing datasets are either exocentric (relying on external cameras), single-view, or use rudimentary head-mounted simulators (e.g., ARCTIC with cameras mounted on a helmet). (2) Real AR/VR devices (such as Quest 3) are naturally equipped with multiple cameras, but the potential of multi-view methods is under-explored. (3) Most existing hand-object interaction scenarios are simple grasp/release actions, lacking complex dynamic manipulation scenarios in daily life.

3. **Key Challenge**: Multi-view egocentric vision is a standard capability of current AR/VR devices, but the academic community lacks corresponding large-scale, high-quality datasets to drive research in this important direction.

4. **Goal**: To release a dataset specifically designed for multi-view egocentric hand-object interaction research and to demonstrate the significant advantages of multi-view methods compared to single-view methods.

5. **Key Insight**: Leverage Meta's two real head-mounted displays (Aria and Quest 3) to record high-quality annotated data in an optical motion capture lab, covering both simple and complex daily interaction scenarios.

6. **Core Idea**: Build the first large-scale multi-view egocentric hand-object interaction benchmark using real AR/VR head-mounted displays and an optical motion capture system, and demonstrate the significant advantages of multi-view methods.

## Method

### Overall Architecture

The construction of HOT3D includes three parts: data recording, annotation, and benchmark design. Recording is performed using Project Aria (1 RGB + 2 monochrome cameras) and Quest 3 (2 monochrome cameras) in a laboratory equipped with an optical motion capture system. Annotations of high-precision 3D poses for hands and objects are obtained via the motion capture system. Benchmark experiments focus on three tasks: 3D hand tracking, model-based 6DoF object pose estimation, and 3D reconstruction of handheld unknown objects.

### Key Designs

1. **Dataset Design and Construction**:

    - Function: Provide multi-view, hardware-synchronized, and high-quality annotated egocentric hand-object interaction data.
    - Mechanism: 833 minutes of recording $\rightarrow$ 1.5M+ multi-view frames (3.7M+ images), with all image streams synchronized via hardware triggers (30 fps). 19 subjects interact with 33 objects in 4 daily scenarios (inspection, kitchen, office, living room). 3D models of objects are acquired via an internal scanning pipeline, featuring high-resolution geometry and PBR materials. Hand annotations are provided in both UmeTrack and MANO formats. Aria additionally provides SLAM point clouds and eye-tracking signals. The dataset is split into a training set (13 subjects, 1M frames) and a test set (6 subjects, 0.5M frames), with 1.16M frames passing absolute annotation quality review. An additional 3,832 curated clips (150 frames / 5 seconds each) are provided for convenient benchmark evaluation.
    - Design Motivation: Utilizing real head-mounted displays ensures the utility and representativeness of the data. Hardware synchronization addresses multi-view temporal alignment issues. Diverse scenarios and objects guarantee the generalization capability of the dataset. Compared to RGB-D optimization approaches, optical motion capture provides higher annotation precision.

2. **Multi-View UmeTrack Hand Tracking Benchmark**:

    - Function: Evaluate performance differences between multi-view and single-view 3D hand tracking.
    - Mechanism: The UmeTrack hand tracker is trained on three data combinations: UmeTrack dataset only (Quest 2), HOT3D-Quest3 only, and a combination of both. During training, one of the views is randomly masked out to improve robustness. The evaluation metric is the Mean Keypoint Position Error (MKPE). Results show that after joint training, the binocular mode achieves an MKPE of 9.5/10.9 mm, representing 41% and 29% improvements over the monocular mode (13.4/15.4 mm), respectively.
    - Design Motivation: Most hand tracking methods are evaluated only on single views, whereas AR/VR devices are naturally equipped with multiple cameras, necessitating a quantitative evaluation of multi-view benefits.

3. **Multi-View FoundPose 6DoF Object Pose Estimation**:

    - Function: Evaluate the performance gain of multi-view settings for training-free object pose estimation.
    - Mechanism: Extend the FoundPose method to multi-view setups: during inference, object images are cropped from all available views, DINOv2 feature matching is performed across all views to establish multi-view 2D-3D correspondences, and the pose is jointly solved via a generalized PnP problem. At a 5cm/5° threshold, Aria's 3-view recall increases from 25.2% to 33.8% (+34%), and Quest 3's 2-view recall increases from 28.9% to 36.9% (+28%).
    - Design Motivation: Multi-view settings not only provide stronger geometric constraints but also reveal object regions heavily occluded in single views. Utilizing DINOv2 as a backbone enables the method to generalize across different sensor types (RGB/monochrome).

### Loss & Training

As a dataset paper, the focus of this work lies in dataset construction and benchmarking rather than loss function design. Hand tracking employs the original training pipeline of UmeTrack. Object pose estimation is based on FoundPose's training-free pipeline (DINOv2 feature matching + PnP-RANSAC). 3D lifting utilizes DINOv2 stereo matching to predict handheld object point cloud depth.

## Key Experimental Results

### Main Results

**3D Hand Tracking (Table 2, MKPE↓, mm)**:

| Training Data | No. of Views | UmeTrack Test Set | HOT3D Test Set |
|---|---|---|---|
| UmeTrack | 1 | 13.6 | 24.2 |
| UmeTrack | 2 | 9.7 | 25.6 |
| UmeTrack + HOT3D | 1 | 13.4 | 15.4 |
| **UmeTrack + HOT3D** | **2** | **9.5** | **10.9** |

**6DoF Object Pose Estimation (Table 3, Recall↑, %)**:

| Test Data | No. of Views | 5cm/5° | 10cm/10° | 20cm/20° |
|---|---|---|---|---|
| HOT3D-Aria | 1 | 25.2 | 41.7 | 54.5 |
| **HOT3D-Aria** | **3** | **33.8** | **52.9** | **66.2** |
| HOT3D-Quest3 | 1 | 28.9 | 46.6 | 58.9 |
| **HOT3D-Quest3** | **2** | **36.9** | **55.9** | **66.4** |

### Ablation Study

**Cross-Dataset Domain Gap Analysis (Table 2)**:

| Training Data | View | UmeTrack MKPE | HOT3D MKPE | Description |
|---|---|---|---|---|
| UmeTrack only | 1 | 13.6 | 24.2 | Domain gap on hand-object interaction scenarios |
| HOT3D only | 1 | 23.7 | 18.0 | Domain gap on hand-hand interaction scenarios |
| Joint Training | 1 | 13.4 | 15.4 | Domain gap is effectively bridged |
| Joint Training | 2 | 9.5 | 10.9 | Multi-view further improves performance by 41% |

### Key Findings

- **Multi-view improvement is significant and consistent**: Binocular hand tracking improves by 29-41% compared to monocular tracking, and multi-view object pose estimation yields a 13-34% improvement. This holds great significance for low-power egocentric vision system design—multiple (low-cost) cameras are more suitable for AR glasses than active depth sensors (high power consumption).
- **Joint training effectively bridges domain gaps**: A model trained solely on UmeTrack performs poorly on HOT3D (24.2 mm) and vice versa (23.7 mm). Joint training dramatically improves performance on both sides, showing that hand-object and hand-hand interaction data are highly complementary.
- **FoundPose works well on monochrome images**: Thanks to the strong generalization capabilities of DINOv2, reasonable pose estimation accuracy is achieved even with Quest 3's monochrome-only cameras.
- The statistical analysis of objects' "travel distance" (Figure 4) serves as an interesting dataset characteristic analysis, reflecting the differences in frequency and manner with which different objects are manipulated.

## Highlights & Insights

- **First multi-view egocentric dataset with real head-mounted displays**: Unlike datasets using RGB-D cameras or helmet simulators, HOT3D utilizes consumer VR headsets (Quest 3) and research-grade AR glass prototypes (Aria) that are actually shipped, making the data distribution closer to real-world application scenarios.
- **"The potential of multi-view methods is underestimated"**: This finding has a direct impact on AR/VR device design—multi-camera setups are not only inexpensive and energy-efficient, but the performance gains they provide may render active depth sensors unnecessary.
- **3D object models with PBR materials**: Supports physically-based rendering, which can be used to synthesize training data, serving as a unique advantage of this dataset.
- **Object onboarding sequences**: Facilitates benchmarking of model-free tracking methods, including both static and dynamic setups, reflecting thoughtful experimental design.

## Limitations & Future Work

- The dataset is limited to rigid objects, lacking support for deformable/articulated objects (which ARCTIC supports).
- All recordings are conducted in a single laboratory, resulting in limited background diversity (partially mitigated by randomizing furniture/lighting).
- Annotation relies on optical markers; markers attached to hands and objects may affect natural interactions and visual appearance.
- Test set annotations are not publicly released, requiring submissions through a dedicated evaluation server, which increases the barrier to entry.
- Highly complex mechanics, such as bimanual collaborative manipulation of heavy objects, are not included.
- Quest 3 only features monochrome cameras; the lack of RGB images might limit the applicability of certain methods.

## Related Work & Insights

- **vs ARCTIC**: ARCTIC contains more articulated objects but has only one simulated egocentric view (helmet-mounted camera) and does not support true consumer hardware. HOT3D uses real head-mounted displays and features native multi-view streams.
- **vs HOI4D**: HOI4D features 800 objects and diverse environments but uses single-view egocentric RGB-D data, and the annotations are derived from RGB-D optimization (yielding lower precision). HOT3D provides higher annotation precision via optical motion capture.
- **vs DexYCB**: DexYCB only contains exocentric views and near-static grasping, lacking dynamic manipulation scenarios. HOT3D covers complex manipulation across kitchen, office, etc.
- **vs HO-Cap**: HO-Cap is the only other hand-object dataset using a real head-mounted display (HoloLens), but its annotation precision relies on RGB-D optimization, which is far lower than HOT3D's optical motion capture precision.

## Rating

- Novelty: ⭐⭐⭐⭐ First multi-view egocentric hand-object dataset on real head-mounted displays, filling a major gap in the field; the methodology (multi-view extension) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Clear comparisons of multi-view vs. single-view across three tasks, and convincing cross-dataset generalization experiments, although the object pose baseline is simple.
- Writing Quality: ⭐⭐⭐⭐⭐ Exemplary writing for a dataset paper, with comprehensive comparison tables and detailed statistical analysis.
- Value: ⭐⭐⭐⭐⭐ Direct and significant impact on the AR/VR field, demonstrating the value of multi-view methods; the dataset will be widely used.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)
- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [\[ICCV 2025\] Multi-View 3D Point Tracking](../../ICCV2025/3d_vision/multi-view_3d_point_tracking.md)
- [\[CVPR 2026\] ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos](../../CVPR2026/3d_vision/forehoi_feed-forward_3d_object_reconstruction_from_daily_hand-object_interaction.md)

</div>

<!-- RELATED:END -->
