---
title: >-
  [Paper Note] H-V2X: A Large Scale Highway Dataset for BEV Perception
description: >-
  [ECCV 2024][Autonomous Driving][V2X] Introduces H-V2X, the first large-scale real-world highway V2X BEV perception dataset covering over 100 km of highway segments with over 1.9 million fine-grained annotated samples. It establishes three benchmark tasks (BEV detection, tracking, and trajectory prediction) and proposes an innovative baseline method integrating vector maps.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "V2X"
  - "Highway"
  - "BEV Perception"
  - "Object Detection"
  - "Trajectory Prediction"
date: 2026-05-08
content_hash: 51ac609f8c890d20
---

# H-V2X: A Large Scale Highway Dataset for BEV Perception

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Autonomous Driving / BEV Perception Dataset  
**Keywords**: V2X, Highway, BEV Perception, Object Detection, Trajectory Prediction

## TL;DR

Introduces H-V2X, the first large-scale real-world highway V2X BEV perception dataset covering over 100 km of highway segments with over 1.9 million fine-grained annotated samples. It establishes three benchmark tasks (BEV detection, tracking, and trajectory prediction) and proposes an innovative baseline method integrating vector maps.

## Background & Motivation

**Background**: Vehicle-to-Everything (V2X) perception is a crucial direction in autonomous driving, utilizing roadside infrastructure sensors to cover vehicle-side perception blind spots. Existing V2X perception datasets, such as DAIR-V2X, V2X-Sim, and RCooper, primarily focus on urban intersection scenarios, utilizing roadside cameras and LiDAR for 3D object detection.

**Limitations of Prior Work**: Highway scenarios significantly differ from urban intersections, featuring higher vehicle speeds ($100+\text{ km/h}$), more regular lanes but more dangerous lane-changing behaviors, different occlusion patterns (large trucks occluding smaller vehicles), and frequent tailgating. However, existing datasets rarely cover highway scenarios. Furthermore, most infrastructure-side datasets constrain perception tasks to monocular 3D detection due to the scarcity of synchronized multi-sensor data, failing to support joint perception in the BEV space.

**Key Challenge**: While highways represent one of the earliest commercialization scenarios for autonomous driving (e.g., highway pilot, electronic toll collection), academic research lacks high-quality highway datasets to support the development and evaluation of BEV perception algorithms. The distributions of urban intersection datasets (low speed, dense pedestrians, complex traffic signals) cannot directly generalize to highway scenarios.

**Goal**: (1) Fill the gap in highway V2X BEV perception datasets; (2) Provide a large-scale dataset with synchronized multi-camera data and high-precision annotations; (3) Construct highway-specific perception benchmark tasks and baseline methods.

**Key Insight**: Multiple synchronized camera systems are deployed along actual highway sections, using joint 2D-3D calibration to ensure projection accuracy in the BEV space, with manual quality checks to guarantee annotation quality. High-definition vector maps are provided alongside the dataset to supply road structure priors for downstream tasks.

**Core Idea**: Constructing H-V2X, the first large-scale BEV perception dataset for highways, and proposing baseline methods integrating vector map information for BEV detection, tracking, and trajectory prediction.

## Method

### Overall Architecture

The construction of the H-V2X dataset involves four stages: (1) Data Collection – deploying multiple synchronized cameras along highway segments covering over 100 km; (2) Sensor Calibration – joint 2D-3D calibration ensuring projection accuracy into the BEV space; (3) Data Annotation – combining automatic labeling and manual quality verification for fine-grained vehicle classification inside the BEV space; (4) Benchmark Construction – designing three perception tasks and baseline methods.

### Key Designs

1. **Multi-Camera Synchronized Capture**:

    - Function: Ensures spatial and temporal alignment of multi-view images to support accurate projection into the BEV space.
    - Mechanism: Camera poles are deployed at regular intervals along the highway, with each pole mounting multiple cameras to cover different viewpoints. All cameras are time-synchronized using the NTP protocol (error $<10\text{ ms}$). Joint 2D-3D calibration utilizes calibration boards placed on the road to simultaneously optimize intrinsic and extrinsic parameters, ensuring consistency in projecting multi-camera images into the BEV space. Annotators label directly and consistently in the BEV space, preventing inconsistency across multiple viewpoints.
    - Design Motivation: Due to the high speed of highway vehicles, temporal desynchronization would cause significant target position offsets across views. Joint calibration is more effective at reducing cumulative errors compared to independent calibrations.

2. **Fine-Grained Annotation**:

    - Function: Provides fine-grained target classification specific to highway scenarios.
    - Mechanism: Vehicles are categorized into multiple classes: small vehicles (sedans, SUVs), medium vehicles (vans, pickups), large vehicles (trucks, semi-trailers), and special-purpose vehicles (construction vehicles, emergency vehicles). In addition to 3D bounding boxes, annotations include driving directions and lane assignments. The dataset contains over 1.9 million annotated samples, covering diverse conditions like sunny, rainy, foggy, daytime, and nighttime scenes.
    - Design Motivation: Since motion patterns differ heavily between vehicle sizes on highways (e.g., trucks accelerate slowly and require wider lane-changing envelopes), fine-grained classification is critical for safety-oriented decision-making.

3. **Vector Map-enhanced Baselines**:

    - Function: Encodes and integrates HD vector map information into BEV perception models to boost detection and prediction accuracy.
    - Mechanism: Vector maps (lane lines, road boundaries, ramps) are rasterized into multi-channel BEV feature maps or passed through a polygon encoder to generate structured map embeddings. In BEV detection, map features are fused with image-derived BEV features using channel concatenation or attention mechanisms. In trajectory prediction, map information acts as a constraint to restrict predicted trajectories inside navigable regions. Utilizing the vector map significantly reduces the Final Displacement Error (FDE) of trajectory prediction.
    - Design Motivation: Given the highly regular highway road structures (straight roads, curves, ramps), vector maps offer powerful structural priors. Unlike urban scenes, highway vehicle trajectories are tightly coupled with lane layout, leading to greater performance gains from map fusion.

### Loss & Training

The BEV detection task adopts the CenterPoint framework, utilizing heatmap focal loss and L1 regression loss. The tracking task is based on Hungarian matching association over detection results. Trajectory prediction utilizes a multi-modal prediction loss (best-of-K strategy) integrated with a feasible-region constraint loss derived from the vector map.

## Key Experimental Results

### Main Results

| Task | Method | Key Metric | Without Map | With Map | Gain |
|------|------|---------|--------|--------|------|
| BEV Detection | CenterPoint | mAP | 38.2 | 41.7 | +3.5 |
| BEV Detection | BEVFormer | mAP | 42.1 | 45.6 | +3.5 |
| Tracking | CTracker | MOTA | 45.3 | 48.1 | +2.8 |
| Trajectory Prediction | HiVT | minADE/minFDE | 1.82/3.94 | 1.56/3.21 | 14.3%/18.5% |

### Ablation Study

| Configuration | mAP | Description |
|------|-----|------|
| Day only | 43.8 | Daytime scenes |
| Night only | 35.2 | Nighttime accuracy drops significantly (-8.6) |
| Rain/Fog | 37.5 | Adverse weather has a clear impact |
| Small vehicles | 48.3 | Optimal detection on common categories |
| Large vehicles | 52.1 | Large targets are easier to detect |
| Special-purpose vehicles | 28.6 | Rare categories are difficult to detect |

### Key Findings
- Vector map fusion provides the most significant performance gain in the trajectory prediction task (FDE is reduced by 18.5%), owing to the tight coupling between highway trajectories and lane layouts.
- Detection accuracy in nighttime scenes drops significantly (-8.6 mAP), likely caused by insufficient illumination for roadside cameras.
- Large vehicles are easy to detect but difficult to associate during tracking (due to severe occlusions), while small vehicles are difficult to detect but tend to yield stable tracking.
- Trajectory prediction in ramp areas is the most challenging due to lane merging and branching behaviors.

## Highlights & Insights

- **Fills a critical gap in highway V2X datasets**: Provides the first large-scale BEV dataset for one of the most active commercialization scenarios in autonomous driving, offering high practical value.
- **Inspiring design paradigm for vector map fusion**: The regular road structures of highways serve as natural prior constraints. Encoding these geometric priors into perception models significantly enhances performance, a paradigm transferable to other structured environments such as airport runways and railway lines.
- The multi-task benchmark design covers the entire pipeline from detection to tracking and prediction, enabling system-level evaluations.

## Limitations & Future Work

- The dataset is collected from a single highway region; road structures and driving behaviors may exhibit regional bias.
- Only camera sensors are provided, lacking LiDAR data, which limits research into multi-modal fusion.
- The annotation framework does not include traffic events (such as accidents, congestion, or broken-down vehicles), restricting research into safety-critical tasks.
- Communication latency and bandwidth limitations in cooperative vehicle-infrastructure systems are not considered, which are critical issues in actual deployment.
- Cross-scenario transfer learning can be introduced, such as pre-training on urban intersection datasets and fine-tuning on highway datasets.

## Related Work & Insights

- **vs DAIR-V2X**: DAIR-V2X focuses on urban intersections and includes dual-modality LiDAR and Camera data, but lacks highway scenarios. H-V2X complements it in scenario coverage and data scale.
- **vs V2X-Sim**: V2X-Sim is a simulation dataset; though it offers diverse scenarios, it suffers from a sim-to-real gap. H-V2X contains real-world data and offers higher practical value.
- **vs nuScenes/Waymo**: These are vehicle-side datasets, whereas H-V2X utilizes an infrastructure-side perspective. While BEV perception challenges are similar, the infrastructure-side features fixed viewpoints and larger coverage areas.

## Rating

- Novelty: ⭐⭐⭐⭐ The first highway V2X BEV dataset, filling a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Benchmark experiments across three tasks are fairly comprehensive, with ablation analysis on map fusion.
- Writing Quality: ⭐⭐⭐⭐ The dataset is clearly described with rich statistical information.
- Value: ⭐⭐⭐⭐⭐ Highly influential for promoting highway autonomous driving perception research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Navigation Instruction Generation with BEV Perception and Large Language Models](navigation_instruction_generation_with_bev_perception_and_large_language_models.md)
- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](../../NeurIPS2025/autonomous_driving/urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [\[CVPR 2026\] V2U4Real: A Real-world Large-scale Dataset for Vehicle-to-UAV Cooperative Perception](../../CVPR2026/autonomous_driving/v2u4real_a_real-world_large-scale_dataset_for_vehicle-to-uav_cooperative_percept.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[ECCV 2024\] Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](accelerating_online_mapping_and_behavior_prediction_via_dire.md)

</div>

<!-- RELATED:END -->
