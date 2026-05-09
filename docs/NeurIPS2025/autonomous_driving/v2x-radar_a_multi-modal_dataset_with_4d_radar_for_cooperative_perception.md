---
title: >-
  [Paper Note] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception
description: >-
  [NeurIPS 2025][Autonomous Driving][cooperative perception] This paper presents V2X-Radar, the first large-scale real-world multi-modal vehicle-to-everything (V2X) cooperative perception dataset incorporating 4D radar, LiDAR, and multi-view camera data. The dataset covers diverse weather and lighting conditions, providing 20K LiDAR frames, 40K camera images, 20K 4D radar scans, and 350K annotated bounding boxes, along with comprehensive benchmarks across three sub-datasets.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - cooperative perception
  - 4D radar
  - V2X
  - multi-modal dataset
  - 3D object detection
date: 2026-05-08
content_hash: 4713b7ae97bda12a
---

# V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception

**Conference**: NeurIPS 2025
**arXiv**: [2411.10962](https://arxiv.org/abs/2411.10962)
**Code**: [GitHub](https://github.com/yanglei18/V2X-Radar)
**Area**: Autonomous Driving
**Keywords**: cooperative perception, 4D radar, V2X, multi-modal dataset, 3D object detection

## TL;DR

This paper presents V2X-Radar, the first large-scale real-world multi-modal vehicle-to-everything (V2X) cooperative perception dataset incorporating 4D radar, LiDAR, and multi-view camera data. The dataset covers diverse weather and lighting conditions, providing 20K LiDAR frames, 40K camera images, 20K 4D radar scans, and 350K annotated bounding boxes, along with comprehensive benchmarks across three sub-datasets.

## Background & Motivation

Single-vehicle perception faces critical safety challenges due to occlusion and limited sensing range. V2X cooperative perception extends the perceptual range through information sharing; however, a significant gap remains:

**Existing V2X datasets lack 4D radar**: Datasets such as OPV2V, DAIR-V2X, and V2V4Real include only cameras and LiDAR, overlooking the robustness advantages of 4D radar under adverse weather conditions.

**Single-vehicle datasets already incorporate 4D radar**: Works such as K-Radar and Dual-Radar have demonstrated the strong adaptability of 4D radar to extreme weather including rain, snow, and fog, yet no such dataset exists for cooperative perception.

**Insufficient scene diversity**: Existing real-world V2X datasets are typically collected under specific time periods or weather conditions, lacking comprehensive coverage of day/night, clear/rain/snow/fog scenarios.

## Method

### Overall Architecture

The V2X-Radar dataset comprises three subsets:
- **V2X-Radar-C**: Vehicle-infrastructure cooperative perception (40 sequences)
- **V2X-Radar-I**: Infrastructure-side perception (10 sequences)
- **V2X-Radar-V**: Single-vehicle perception (10 sequences)

### Key Designs

1. **Sensor Platform Configuration**:

    - **Vehicle side**: RoboSense RS-Ruby-80 LiDAR (80-beam) + Basler camera (1920×1080) + Arbe Phoenix 4D radar (77 GHz, horizontal ±50°) + GPS/IMU + C-V2X communication unit
    - **Infrastructure side**: RoboSense RS-Ruby-80 LiDAR + 3 Basler multi-view cameras (1536×864) + OCULI EAGLE 4D radar (79 GHz, horizontal ±56°) + GPS/IMU + C-V2X communication unit
    - The 4D radar provides sparse point clouds with Doppler information, offering inherent robustness to adverse weather.

2. **Data Synchronization and Calibration**:

    - **Temporal synchronization**: All computer clocks are first aligned to GPS time, followed by hardware-triggered synchronization via PTP + PPS signals; the temporal offset between vehicle-side and infrastructure-side sensors is controlled within 20 ms.
    - **Spatial calibration**: Camera intrinsics are calibrated using a checkerboard; LiDAR–camera extrinsics are estimated by minimizing reprojection error over 100 2D–3D point correspondences; LiDAR–4D radar calibration uses high-intensity points on corner reflectors.
    - **Vehicle-infrastructure registration**: Initial registration is based on RTK localization, refined through the CBM algorithm and manual adjustment.

3. **Data Annotation and Quality Control**:

    - A **semi-automatic annotation pipeline** combines automatic labeling with human refinement through multiple rounds of quality checks.
    - Five object categories: pedestrian, cyclist, car, bus, and truck.
    - Cooperative annotation: vehicle-side and infrastructure-side annotations are generated in their respective LiDAR coordinate systems, then unified into the infrastructure coordinate system and deduplicated via IoU matching.
    - **Privacy protection**: Model-based detection combined with manual frame-by-frame inspection anonymizes all road names, localization data, license plates, and faces.

### Loss & Training

As a dataset paper, V2X-Radar does not introduce novel training strategies. Instead, it provides benchmark experiments under three fusion paradigms:
- **Early Fusion**: Raw point clouds from all agents are merged prior to detection.
- **Late Fusion**: Each agent performs independent detection, followed by NMS-based merging.
- **Intermediate Fusion**: Each agent extracts intermediate features for transmission and fusion (F-Cooper, V2X-ViT, CoAlign, HEAL).

## Key Experimental Results

### Main Results

Infrastructure-side 3D object detection (V2X-Radar-I):

| Method | Modality | Vehicle AP@0.7 (Easy) | Pedestrian AP@0.5 (Easy) | Cyclist AP@0.5 (Easy) |
|--------|----------|-----------------------|--------------------------|----------------------|
| PV-RCNN | LiDAR | 88.83 | 77.13 | 91.82 |
| CenterPoint | LiDAR | 86.44 | 67.90 | 90.26 |
| BEVHeight++ | Camera | 48.48 | 33.05 | 45.19 |
| RPFA-Net | 4D Radar | 64.79 | 51.64 | 45.86 |

Single-vehicle 3D object detection (V2X-Radar-V):

| Method | Modality | Vehicle AP@0.7 (Easy) | Pedestrian AP@0.5 (Easy) | Cyclist AP@0.5 (Easy) |
|--------|----------|-----------------------|--------------------------|----------------------|
| PV-RCNN | LiDAR | 88.27 | 67.04 | 89.48 |
| BEVHeight++ | Camera | 17.47 | 10.43 | 12.99 |
| RPFA-Net | 4D Radar | 42.77 | 11.51 | 17.03 |

### Ablation Study

Cooperative perception vs. single-vehicle perception (vehicle category, AP@0.5):

| Configuration | Overall | 0–30m | 30–50m | 50–100m | Note |
|---------------|---------|-------|--------|---------|------|
| No Fusion (Camera) | 6.76 | 9.41 | 5.16 | 2.27 | Single camera extremely weak |
| Late Fusion (Camera) | 32.88 | 40.58 | 30.37 | 20.00 | Cooperation yields large gains |
| V2X-ViT (LiDAR, Sync) | ~85+ | ~90+ | ~80+ | ~60+ | Intermediate fusion best |
| V2X-ViT (LiDAR, Async) | degraded | - | - | - | Communication latency significantly hurts performance |

### Key Findings

- **LiDAR remains strongest; 4D radar outperforms camera**: In infrastructure-side detection, 4D radar (RPFA-Net, 64.79%) is substantially lower than LiDAR (PV-RCNN, 88.83%) but significantly surpasses camera (BEVHeight++, 48.48%).
- **Vehicle-side camera performance drops sharply**: Single-view vehicle camera (17.47%) is far inferior to multi-view infrastructure camera (48.48%), underscoring the importance of viewpoint coverage.
- **Cooperative perception yields substantial improvements**: All cooperative methods significantly outperform their single-vehicle counterparts.
- **Communication latency is a critical challenge**: Under asynchronous settings (100 ms delay), all methods exhibit notable performance degradation, particularly under the strict IoU=0.7 threshold.
- **Annotations per sample reach up to 90**: This far exceeds single-vehicle datasets such as KITTI/nuScenes (approximately 10–30), reflecting the enhanced environmental understanding enabled by cooperative perception.

## Highlights & Insights

- The dataset fills a critical gap in the V2X domain by introducing 4D radar data, which is essential for safe driving under adverse weather conditions.
- Data collection spanning 9 months covers clear/rain/fog/snow conditions and day/dusk/night scenarios, offering superior scene diversity compared to existing datasets.
- Rigorous temporal synchronization (<20 ms) and spatial calibration pipelines ensure high data quality.
- The three-subset design supports multiple research directions, enhancing the dataset's general applicability.

## Limitations & Future Work

- The dataset scale is relatively limited (20K frames), still behind nuScenes (40K) or Waymo (200K+).
- Only V2I scenarios involving a single vehicle and a single roadside unit are covered; V2V or multi-infrastructure cooperative settings are not included.
- 4D radar point clouds are extremely sparse (typically only dozens of points per vehicle), limiting the direct applicability of existing methods.
- Annotations are restricted to 3D bounding boxes, without richer labels such as semantic segmentation or tracking IDs.
- The vehicle side is equipped with only a single front-facing camera, limiting the applicability of surround-view perception algorithms.

## Related Work & Insights

- **Relation to DAIR-V2X**: DAIR-V2X is a real-world V2I dataset but lacks 4D radar; V2X-Radar augments this setting with the 4D radar modality.
- **Relation to K-Radar/Dual-Radar**: These are single-vehicle 4D radar datasets; V2X-Radar extends 4D radar into the cooperative perception domain.
- **Insight**: The potential of 4D radar in cooperative perception remains largely unexplored, with broad research opportunities in multi-modal fusion (radar + LiDAR + camera) and adverse-weather robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ First V2X dataset with 4D radar, filling an important gap
- Experimental Thoroughness: ⭐⭐⭐⭐ Three subsets × multiple detection methods × synchronous/asynchronous settings
- Writing Quality: ⭐⭐⭐⭐ Follows standard dataset paper conventions with thorough detail
- Value: ⭐⭐⭐⭐ Significant contribution to advancing 4D radar research in cooperative perception

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[AAAI 2026\] RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving](../../AAAI2026/autonomous_driving/radarmp_motion_perception_for_4d_mmwave_radar_in_autonomous_driving.md)
- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](../../ICCV2025/autonomous_driving/cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)

<!-- RELATED:END -->
