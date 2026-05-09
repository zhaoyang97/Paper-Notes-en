---
title: >-
  [Paper Note] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration
description: >-
  [ICCV 2025][Autonomous Driving][V2X Collaborative Perception] Mixed Signals is the first real-world V2X dataset featuring heterogeneous LiDAR configurations (varying mounting heights and tilt angles), collected by 3 autonomous vehicles and one roadside unit. It provides 45,100 point cloud frames and 240,600 annotated bounding boxes, and is also the first V2X dataset collected in a left-hand traffic country (Australia).
tags:
  - ICCV 2025
  - Autonomous Driving
  - V2X Collaborative Perception
  - Point Cloud Dataset
  - Heterogeneous Sensors
  - 3D Object Detection
  - Vulnerable Road Users
date: 2026-05-08
content_hash: 87c740b36e4e971b
---

# Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration

**Conference**: ICCV 2025
**arXiv**: [2502.14156](https://arxiv.org/abs/2502.14156)
**Code**: [https://mixedsignalsdataset.cs.cornell.edu/](https://mixedsignalsdataset.cs.cornell.edu/)
**Area**: Autonomous Driving
**Keywords**: V2X Collaborative Perception, Point Cloud Dataset, Heterogeneous Sensors, 3D Object Detection, Vulnerable Road Users

## TL;DR

Mixed Signals is the first real-world V2X dataset featuring heterogeneous LiDAR configurations (varying mounting heights and tilt angles), collected by 3 autonomous vehicles and one roadside unit. It provides 45,100 point cloud frames and 240,600 annotated bounding boxes, and is also the first V2X dataset collected in a left-hand traffic country (Australia).

## Background & Motivation

### The Necessity of V2X Collaborative Perception

Single-vehicle autonomous driving systems continue to face significant challenges in complex scenarios: critical traffic participants may be occluded, and sensors may unexpectedly fail. As autonomous vehicle deployment scales up, vehicle-to-everything (V2X) communication—between multiple vehicles and between vehicles and infrastructure—offers new possibilities for addressing these issues, as each vehicle can leverage shared perception information to detect road users missed by its own sensors.

### Three Key Limitations of Existing V2X Datasets

**1. Insufficient Diversity**
- All existing datasets use identical or near-identical LiDAR configurations across connected autonomous vehicles (CAVs). In real-world collaborative perception deployments, however, LiDAR mounting positions, angles, and models will likely vary across manufacturers and vehicle types.
- Existing datasets cover only 3 right-hand traffic countries (China, the United States, and Germany), whereas approximately one-third of countries worldwide use left-hand traffic, representing distinct traffic dynamics.

**2. Inadequate Coverage of Vulnerable Road Users (VRUs)**
- Pedestrians, cyclists, and other VRUs are ubiquitous in urban environments.
- V2V4Real and 3 synthetic datasets contain no VRU annotations whatsoever.
- DAIR-V2X annotates 4 VRU categories but does not release category-level statistics.
- TUMTrafV2X has VRUs comprising only 24.6% of annotations, which are ignored during evaluation.

**3. Inconsistent Data Quality**
- Data collection and alignment in multi-agent, multi-sensor settings are extremely challenging.
- Imprecise pose estimation and localization system failures result in poor point cloud alignment quality in existing datasets.
- Annotation inconsistencies across time steps and viewpoints degrade training effectiveness.

### Positioning of Core Contributions

Mixed Signals fills these gaps through three firsts:

**First heterogeneous CAV LiDAR configurations**: 3 vehicles employ 2 distinct sensor mounting setups (differing in height and tilt angle).

**First dataset from a left-hand traffic country**: Collected in Sydney, Australia.

**Most comprehensive VRU coverage**: VRU bounding boxes account for 50.3% of all annotations, far exceeding other datasets.

## Method

### Overall Architecture

The data collection system consists of 3 CAVs and 1 roadside unit (RSU), equipped with a total of 5 LiDAR sensors. Data were collected at 10 Hz over 2 hours at a busy intersection in Sydney, with 37 curated 30-second segments containing rich vehicle, pedestrian, and cyclist interaction scenarios.

### Key Designs

#### 1. Heterogeneous Sensor Configuration

- **Function**: Creates sensor domain gaps by using the same LiDAR model under different mounting configurations.
- **Specific Setup**:
    - **Electric Vehicles (EV) ×2**: OS1-128 beam LiDAR, mounting height 1.63 m, **tilted downward 15°**
    - **Urban Vehicle (Laser) ×1**: OS1-128 beam LiDAR, mounting height 1.9 m, **horizontally mounted**
    - **Roadside Unit (RSU)**: OS-Dome 128 beam (long-range detection) + OS1-64 beam (short-range detection), mounting height 2.5 m
- **Design Motivation**: Although the same LiDAR model is used, the different mounting heights and tilt angles create a realistic domain gap. The tilted installation on EVs results in unobservable rear regions and missing intensity information—a practical challenge completely overlooked by prior datasets.

#### 2. Precise Synchronization and Localization

- **Synchronization**: GPS timestamps are used to temporally align all LiDARs at 10 Hz, with cross-sensor timestamp error kept within 50 ms.
- **Localization**:
    - Rather than relying on GNSS, which is unreliable in urban environments, the system uses a **high-density precise point cloud map** as a reference.
    - **Scan matching** is employed to estimate vehicle poses, achieving localization accuracy of 15 cm in position and 0.4° in heading.
    - All vehicles and the RSU are localized within a unified `map_frame` coordinate system.
- **Design Motivation**: Precise alignment is the lifeblood of a V2X dataset. Comparative visualizations against existing datasets (Figure 4) demonstrate that Mixed Signals achieves substantially better point cloud alignment in both lateral and longitudinal directions.

#### 3. High-Quality Annotation

- **Annotation Pipeline**:
    - Point clouds from all agents are aggregated into the RSU TOP sensor coordinate frame.
    - Professional annotation company FlipSideAI performs 3D bounding box annotation using the SegmentsAI tool.
    - Ten fine-grained categories are annotated: Car, Truck, Pedestrian, Bus, Electric Vehicle, Trailer, Motorcycle/Bike, Bicycle, Portable Personal Mobility, and Emergency Vehicle.
    - Keyframes are sampled at 1 Hz for manual annotation; non-keyframes are obtained via linear interpolation between the nearest preceding and following keyframes.
    - Multiple rounds of monitoring, review, and correction cycles are applied.
- **Annotation Quality Validation**: By back-projecting bounding boxes of the same object from different sensors into a unified coordinate frame (Figure 5), Mixed Signals demonstrates substantially better cross-sensor and cross-temporal annotation consistency compared to existing datasets.
- **Design Motivation**: Pioneer datasets are often annotated by non-professional annotators, resulting in poor consistency. This work invests substantial annotation resources to ensure data reliability.

#### 4. Detection Category Design

The 10 fine-grained annotation categories are merged into 3 detection classes:
- **Vehicle**: car, truck, emergency vehicle, bus, electric vehicle, trailer
- **Bike**: motorbike, bicycle, portable personal mobility
- **Pedestrian**: pedestrian

VRUs (Bike + Pedestrian) account for 50.3% of all annotated bounding boxes—the highest proportion among all V2X datasets.

### Loss & Training

As a dataset paper, the benchmark experiments follow a standard 3D object detection setup: the evaluation region is $[-51.2, 51.2]$ m², the visibility threshold is 5 points, and BEV IoU matching is used with thresholds of 0.3/0.5/0.7.

## Key Experimental Results

### Main Results (Collaborative Object Detection)

| Method | Vehicle AP@0.5 | Bike AP@0.5 | Pedestrian AP@0.5 | Bandwidth (MB) |
|--------|---------------|-------------|-------------------|----------------|
| No Fusion | 0.42 | 0.19 | 0.47 | 0.00 |
| Late Fusion | 0.43 | 0.56 | 0.57 | 0.11 |
| Laly Fusion | 0.61 | 0.68 | 0.69 | 0.11 |
| Early Fusion | 0.65 | 0.65 | 0.74 | 7.79 |
| V2V-Net | 0.72 | 0.69 | 0.42 | 4.19 |
| F-Cooper | 0.75 | 0.68 | 0.72 | 15.31 |
| where2comm | 0.77 | 0.74 | 0.31 | 16.78 |
| V2V-AM | 0.83 | 0.79 | 0.69 | 16.78 |
| **V2X-ViT** | **0.84** | **0.71** | **0.77** | 19.36 |
| **Attentive Fusion** | **0.82** | **0.71** | **0.74** | 5.26 |

### Ablation Study (RSU-Enhanced Single-Vehicle Detection)

| Vehicle + RSU | Method | Vehicle AP@0.5 | Bike AP@0.5 | Ped AP@0.5 |
|--------------|--------|---------------|-------------|------------|
| EV-1 | No Fusion (EV-1) | 0.33 | 0.28 | 0.37 |
| EV-1 | Attentive Fusion | **0.53** | **0.60** | **0.57** |
| EV-2 | No Fusion (EV-2) | 0.33 | 0.16 | 0.08 |
| EV-2 | Attentive Fusion | **0.56** | **0.56** | **0.40** |
| Laser | No Fusion (Laser) | 0.30 | 0.32 | 0.46 |
| Laser | Attentive Fusion | **0.71** | **0.66** | **0.58** |

The Laser vehicle achieves the best performance due to its horizontally mounted LiDAR providing full 360° coverage; the tilted LiDAR on EVs introduces rear blind spots and missing intensity information.

### Key Findings

- **Collaborative perception consistently outperforms single-vehicle perception**: All fusion methods surpass the No Fusion baseline across all categories, validating the value of V2X.
- **Performance–bandwidth trade-off**: Early/intermediate fusion achieves higher accuracy but incurs large bandwidth costs, while late/Laly fusion consumes minimal bandwidth at the cost of limited accuracy. Laly Fusion achieves competitive results at extremely low bandwidth, making it suitable for practical deployment.
- **Domain gap from heterogeneous sensors is real**: The performance gap between EV and Laser vehicles is pronounced—particularly for EV-2, whose pedestrian detection AP is only 0.08 without fusion—demonstrating that sensor configuration differences have non-negligible performance implications.
- **Right-hand to left-hand traffic domain shift**: Directly transferring a model trained on the right-hand traffic dataset V2V4Real to Mixed Signals causes severe performance degradation, with predicted heading directions reversed, indicating that the model has learned right-hand traffic priors.
- **VRU detection remains a major open challenge**: Even the best-performing methods show substantially lower AP for Bike and Pedestrian compared to Vehicle, indicating that dedicated algorithmic designs are needed for VRU detection.
- **RSU communication is universally beneficial**: In the V2I setting, all vehicles achieve significant detection performance gains through communication with the RSU.

## Highlights & Insights

- **Precise problem formulation**: Heterogeneous sensor configurations represent a genuine challenge in real-world V2X deployment that has been entirely overlooked by prior work.
- **Data quality benchmark**: New standards are established for synchronization accuracy (50 ms) and localization accuracy (15 cm / 0.4°).
- **Significance of VRU coverage**: A 50.3% VRU annotation ratio elevates vulnerable road user detection to a first-class task rather than a secondary evaluation.
- **Contribution of left-hand traffic**: The dataset reveals that traffic direction is an important source of domain gap, a factor completely ignored in prior research.
- **Novel task formulation**: The RSU-enhanced single-vehicle detection setup—with a fixed RSU model and a trainable vehicle-side model—more closely reflects real-world deployment scenarios.

## Limitations & Future Work

- **Single collection site**: Data were collected at only one intersection in Sydney, limiting scene diversity.
- **LiDAR-only annotation**: Camera image annotations and multimodal fusion annotations are absent.
- **Limited number of vehicles**: At most 3 CAVs, which is smaller in scale compared to synthetic datasets (up to 7 vehicles).
- **Limited weather conditions**: Only daytime clear-weather data during peak hours; adverse conditions such as rain, fog, and nighttime are not covered.
- **Detection-centric**: Although track IDs are provided, benchmarks for tracking and prediction tasks are not yet fully developed.
- **VRU class imbalance**: Despite a high overall VRU proportion, fine-grained categories such as portable personal mobility may still have insufficient sample sizes.

## Related Work & Insights

- **V2V4Real / V2X-Real**: Real-world V2X datasets collected in the United States; this work demonstrates their non-transferability to left-hand traffic scenarios.
- **DAIR-V2X**: A large-scale V2I dataset collected in China, but subject to geographic access restrictions.
- **V2X-Sim / OPV2V**: Synthetic V2X datasets that are large in scale but lack real-world heterogeneity.
- **Laly Fusion**: A low-bandwidth, high-efficiency fusion method that performs well on this dataset and warrants further investigation.
- **Takeaway**: V2X dataset construction must give greater consideration to deployment diversity—including sensor heterogeneity, traffic regulation differences, and environmental variety.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First heterogeneous-sensor + first left-hand traffic V2X dataset
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive baselines but limited to a single scene
- Writing Quality: ⭐⭐⭐⭐ — Clear comparative analysis with rich visualizations
- Value: ⭐⭐⭐⭐⭐ — Substantial infrastructure contribution to the V2X community

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[CVPR 2026\] BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds](../../CVPR2026/autonomous_driving/buildanypoint_3d_building_structured_abstraction_from_diverse_point_clouds.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)
- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](../../NeurIPS2025/autonomous_driving/urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)

<!-- RELATED:END -->
