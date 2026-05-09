---
title: >-
  [Paper Note] UAVScenes: A Multi-Modal Dataset for UAVs
description: >-
  [ICCV 2025][Autonomous Driving][UAV perception] UAVScenes is the first large-scale multi-modal UAV dataset that simultaneously provides per-frame semantic annotations for both images and LiDAR point clouds along with accurate 6-DoF poses. It contains over 120,000 annotated frames and supports six perception tasks including semantic segmentation, depth estimation, localization, scene recognition, and novel view synthesis.
tags:
  - ICCV 2025
  - Autonomous Driving
  - UAV perception
  - multi-modal dataset
  - semantic segmentation
  - depth estimation
  - LiDAR point cloud
date: 2026-05-08
content_hash: 2b8e76298fa32259
---

# UAVScenes: A Multi-Modal Dataset for UAVs

**Conference**: ICCV 2025
**arXiv**: [2507.22412](https://arxiv.org/abs/2507.22412)
**Code**: [https://github.com/sijieaaa/UAVScenes](https://github.com/sijieaaa/UAVScenes)
**Area**: Autonomous Driving
**Keywords**: UAV perception, multi-modal dataset, semantic segmentation, depth estimation, LiDAR point cloud

## TL;DR

UAVScenes is the first large-scale multi-modal UAV dataset that simultaneously provides per-frame semantic annotations for both images and LiDAR point clouds along with accurate 6-DoF poses. It contains over 120,000 annotated frames and supports six perception tasks including semantic segmentation, depth estimation, localization, scene recognition, and novel view synthesis.

## Background & Motivation

### The Demand for UAV Perception

With the rapid growth of the low-altitude economy, UAVs have been widely deployed in aerial taxi, low-altitude logistics, agriculture, inspection, and emergency response. Unlike ground vehicles, UAVs operate above ground constraints and require high-quality training datasets to achieve reliable perception capabilities.

### Systematic Deficiencies in Existing UAV Datasets

The authors conducted a systematic survey of existing UAV datasets and identified three levels of problems:

**Level 1: Single-Modality Only**
- Many datasets (UAVDT, VisDrone, UAVid, FloodNet, etc.) contain only camera images without 3D LiDAR data.
- 3D scene understanding and high-accuracy multi-modal fusion are therefore infeasible.

**Level 2: Multi-Modal but Lacking Per-Frame Annotations**
- NTU VIRAL, GrAco, FIReStereo, and MUN-FRL provide camera+LiDAR data but are primarily designed for SLAM or 3D reconstruction.
- UrbanScene3D and Hessigheim 3D annotate only reconstructed 3D maps rather than individual frames.
- GauU-Scene uses encrypted DJI-L1 point clouds, making per-frame LiDAR data inaccessible.

**Level 3: The Core Gap**
- **No existing multi-modal UAV dataset provides both per-frame image annotations and per-frame LiDAR point cloud annotations simultaneously.**
- This directly impedes research on advanced perception tasks such as per-frame semantic segmentation, depth estimation, and precise localization.

### Positioning and Contributions

UAVScenes extends the MARS-LVIG dataset (originally a multi-modal UAV dataset for SLAM only) through three major contributions:
1. Adding 19-class semantic annotations (16 static + 2 dynamic + 1 background) for per-frame images.
2. Adding semantic annotations for per-frame LiDAR point clouds.
3. Reconstructing accurate 6-DoF poses (the original dataset provides only 4-DoF RTK poses).

## Method

### Overall Architecture

The construction pipeline of UAVScenes consists of three stages: 3D reconstruction for 6-DoF pose estimation → image semantic annotation → LiDAR point cloud semantic annotation. Each stage includes rigorous quality control and manual review.

### Key Designs

#### 1. 6-DoF Pose Reconstruction

- **Function**: Upgrades MARS-LVIG's 4-DoF RTK poses to complete 6-DoF poses.
- **Mechanism**:
    - LVI-SLAM methods (FAST-LIVO, R3LIVE) were first attempted but yielded poor reconstruction quality due to LiDAR degeneracy caused by the downward-facing flight orientation.
    - Structure-from-Motion (SfM) was adopted instead; COLMAP, RealityCapture, Metashape, and DJI Terra were evaluated.
    - DJI Terra was ultimately selected as it accepts GNSS coordinates for initialization and is specifically designed for UAV scenes, yielding the best reconstruction quality.
    - The entire MARS-LVIG dataset was divided into 8 splits based on environment and lighting conditions; SfM reconstruction was performed independently for each split (3–10 hours per split).
- **Design Motivation**: Accurate 6-DoF poses are the foundation for all downstream tasks, particularly novel view synthesis and precise localization. 4-DoF poses encode only 3D position and yaw angle, which is insufficient for fine-grained evaluation.

#### 2. Image Semantic Annotation Pipeline

- **Function**: Provides 19-class pixel-level semantic annotations for 120,000+ frames.
- **Mechanism** (two-step process):
    - **Static class annotation** (16 classes): Manual semantic labeling is performed on the reconstructed 3D point cloud map and then rendered back to the corresponding camera views to obtain 2D semantic masks. 3D consistency ensures cross-frame annotation coherence.
    - **Dynamic class annotation** (2 classes: cars and trucks): Instance-level manual annotation is performed on individual frames. The tracking functionality of X-AnyLabeling partially accelerates this process, but tracking instability necessitates extensive manual verification and correction. Over 280,000 dynamic instances were annotated in total.
    - Static and dynamic annotations are merged into complete per-frame labels.
- **Design Motivation**: Annotating static classes on the 3D map guarantees cross-frame consistency (i.e., the same building receives consistent labels across frames), which is difficult to ensure with conventional frame-by-frame annotation.

#### 3. LiDAR Point Cloud Semantic Annotation

- **Function**: Provides semantic annotations for per-frame Livox-Avia LiDAR point clouds.
- **Mechanism**:
    - Camera–LiDAR hardware synchronization and calibration are leveraged to project image semantic annotations onto the corresponding LiDAR point clouds.
    - Automatic projection is followed by manual consistency checking and correction.
    - Only the open Livox-Avia point clouds are used, as the DJI-L1 output is encrypted and inaccessible.
- **Design Motivation**: Image-to-point-cloud projection efficiently produces initial annotations, which are then refined through manual correction to ensure quality.

### Loss & Training

As a dataset paper, no specific model training is involved. Benchmark experiments follow standard training protocols for each respective task.

## Key Experimental Results

### Main Results (Image Semantic Segmentation)

| Params | Architecture | Model | mIoU ↑ |
|--------|-------------|-------|--------|
| 22M | Transformer | DeiT3-s | 67.6 |
| 38M | Transformer | DeiT3-m | **68.3** |
| 22M | Transformer | ViT-s | 63.9 |
| 5M | Transformer | ViT-t | 62.8 |
| 25M | CNN | ResNet-50 | 61.3 |
| 44M | CNN | ResNet-101 | 60.7 |
| 21M | CNN | ResNet-34 | 59.9 |
| 28M | CNN | ConvNext-t | 55.3 |
| 48M | CNN | MambaOut-s | 51.8 |

All models use UperNet as the segmentation head. Transformer-based models consistently outperform CNN-based models, with DeiT3-m achieving the best mIoU of 68.3%.

### Ablation Study (LiDAR Semantic Segmentation)

| Params | Model | mIoU ↑ | Note |
|--------|-------|--------|------|
| 38M | MinkUNet | 32.7 | Voxelized point cloud method |
| 39M | SPUNet | 34.4 | Sparse convolution method |
| 11M | PTv2 | 33.2 | Point Transformer method |

LiDAR segmentation mIoU is substantially lower than image segmentation (~33% vs. ~68%), indicating that semantic segmentation of aerial LiDAR point clouds is highly challenging.

### Key Findings

- **Transformers outperform CNNs**: Even with fewer parameters (e.g., ViT-t at only 5M), Transformer architectures surpass large-parameter CNNs on UAV semantic segmentation.
- **LiDAR segmentation is much harder than image segmentation**: The mIoU gap is approximately 35 percentage points, likely attributable to low point cloud density and domain shift from ground-level patterns.
- **Challenges of dynamic object annotation**: IoU scores for cars and trucks are notably lower than for static categories, reflecting the difficulty of dynamic object detection in UAV scenes.
- **Small objects such as Solar Panels and Umbrellas**: IoU values are extremely low (<10%), representing a primary direction for future improvement.
- **Multi-traversal coverage**: The dataset includes multiple passes over the same scenes, enabling temporal tasks such as scene change detection.

## Highlights & Insights

- **Uniqueness**: The first real-world UAV dataset globally to simultaneously provide 6-DoF poses, per-frame image annotations, and per-frame LiDAR annotations.
- **Annotation pipeline innovation**: Annotating static classes on the 3D map and rendering back to 2D ensures cross-frame consistency while reducing annotation cost.
- **Comprehensive task coverage**: A single dataset supports six distinct tasks, providing a unified evaluation platform for UAV perception research.
- **Scale advantage**: 120,000+ annotated frames substantially exceed most existing UAV datasets.
- **Open-source and reproducible**: Both data and code are publicly available.

## Limitations & Future Work

- **Limited environmental diversity**: Based on MARS-LVIG, the dataset covers only town, valley, airport, and island scenes, lacking extreme weather, nighttime, and other challenging conditions.
- **DJI-L1 encryption restriction**: High-quality DJI-L1 LiDAR data cannot be utilized; only Livox-Avia data are available.
- **Limited dynamic object categories**: Only cars and trucks are annotated; pedestrians, bicycles, and other dynamic objects are absent.
- **SfM reconstruction accuracy**: Reliance on the commercial DJI Terra tool may yield lower reconstruction accuracy than SLAM under ideal conditions.
- **Annotation scalability**: Despite the semi-automatic pipeline, labeling 280,000+ dynamic instances requires substantial human effort, posing a cost challenge for scaling to larger datasets.

## Related Work & Insights

- **SemanticKITTI**: The benchmark dataset for LiDAR semantic segmentation in ground-level autonomous driving; its annotation methodology directly inspired this work.
- **nuScenes / Waymo**: Representative multi-modal autonomous driving datasets, both limited to ground-level perspectives.
- **MARS-LVIG**: The foundation of this dataset, originally designed solely for SLAM.
- Key insight: The dataset construction methodology from ground-level autonomous driving is transferred to the UAV domain, leveraging 3D-to-2D annotation rendering to ensure cross-frame consistency.

## Rating

- Novelty: ⭐⭐⭐⭐ — Fills a clear gap in the dataset landscape.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task benchmarks are comprehensive, though individual task experiments are relatively small in scale.
- Writing Quality: ⭐⭐⭐⭐ — Comparison tables are clear and related work is thoroughly surveyed.
- Value: ⭐⭐⭐⭐⭐ — As an infrastructure-level contribution, it holds high value for UAV perception research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[ICCV 2025\] 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views](3drealcar_an_in-the-wild_rgb-d_car_dataset_with_360-degree_views.md)
- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](../../NeurIPS2025/autonomous_driving/urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)

<!-- RELATED:END -->
