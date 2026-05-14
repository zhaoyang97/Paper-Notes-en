---
title: >-
  [Paper Note] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception
description: >-
  [NeurIPS 2025][Autonomous Driving][Vehicle-to-Everything] UrbanIng-V2X is the first real-world cooperative perception dataset spanning multiple vehicles, multiple infrastructure sensors…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Vehicle-to-Everything"
  - "V2X Dataset"
  - "Cooperative Perception"
  - "Multiple Intersections"
  - "3D Object Detection"
date: 2026-05-08
content_hash: c9c8cd9d62a80886
---

# UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception

**Conference**: NeurIPS 2025
**arXiv**: [2510.23478](https://arxiv.org/abs/2510.23478)
**Code**: [https://github.com/thi-ad/UrbanIng-V2X](https://github.com/thi-ad/UrbanIng-V2X)
**Area**: Autonomous Driving / Cooperative Perception
**Keywords**: Vehicle-to-Everything, V2X Dataset, Cooperative Perception, Multiple Intersections, 3D Object Detection

## TL;DR

UrbanIng-V2X is the first real-world cooperative perception dataset spanning multiple vehicles, multiple infrastructure sensors, and multiple urban intersections. It provides 712K annotated instances across 13 categories in 34 scenes, and through a cross-intersection evaluation strategy (SIS) quantitatively reveals a substantial generalization gap of 14 mAP exhibited by existing cooperative perception methods on unseen intersections.

## Background & Motivation

**Background**: Cooperative Perception leverages V2X communication to enable vehicles and infrastructure to share sensor data, mitigating single-vehicle occlusion and limited field of view. Several real-world datasets have advanced the field: V2V4Real (V2V only), DAIR-V2X-C (V2I, 28 intersections but single vehicle), TUMTraf-V2X (V2I, single intersection), and V2X-Real (V2V+V2I, single intersection).

**Limitations of Prior Work**: (1) No existing dataset simultaneously combines multi-vehicle, multi-infrastructure, and multi-intersection coverage—a combination critical for evaluating the scalability of cooperative perception systems in realistic urban environments. (2) Training and testing on a single intersection may yield inflated performance—models may learn intersection-specific geometric patterns and traffic behaviors rather than generalizable cooperative perception capabilities.

**Key Challenge**: Evaluating generalization requires cross-intersection testing, yet collecting cross-intersection data poses substantial engineering challenges in hardware deployment, spatiotemporal synchronization, and annotation consistency across heterogeneous sources.

**Goal**: (1) Construct the first real-world V2X dataset with multi-vehicle, multi-infrastructure, and multi-intersection coverage; (2) Design an evaluation strategy to quantify the generalization gap of models on unseen intersections; (3) Provide a complete toolchain (development kit, HD maps, digital twin) to empower community research.

**Key Insight**: Three urban intersections with distinct geometric layouts are selected at a high-definition test site in Ingolstadt, Germany. Two connected vehicles and seven sensor poles are deployed, a rigorous spatiotemporal synchronization scheme is designed, approximately 8 hours of data are recorded, and 34 representative scenes are carefully curated.

**Core Idea**: The first real-world V2X dataset covering multiple intersections, multiple vehicles, and multiple infrastructure nodes; a Separate Intersection Split (SIS) is introduced to expose the generalization bottleneck of cooperative perception methods.

## Method

### Overall Architecture

The construction of UrbanIng-V2X encompasses five core components: (1) large-scale multimodal sensor deployment; (2) precise spatiotemporal synchronization and calibration; (3) LiDAR motion compensation and multi-source fusion; (4) scene selection and 3D annotation; and (5) cross-intersection evaluation strategy design. Accompanying tools include OpenCOOD/nuScenes format converters, Lanelet2 HD maps, and a CARLA digital twin.

### Key Designs

1. **Most Comprehensive V2X Sensor Deployment in the Field**

    - **Function**: Achieves unprecedented multimodal, multi-viewpoint sensor coverage.
    - **Mechanism**: Each vehicle is equipped with 6 FHD RGB cameras (360°), one 128-line LiDAR, and a high-precision IMU (RTK, 1 cm positioning). Across 3 intersections, 7 sensor poles are deployed, each carrying 1–3 VGA thermal cameras alongside a LiDAR combination (64-line mid-range + 32-line close-range). Maximum per-scene totals: 12 vehicle-mounted RGB cameras, 2 vehicle-mounted LiDARs, 17 infrastructure thermal cameras, and 12 infrastructure LiDARs.
    - **Design Motivation**: Thermal cameras are introduced into a V2X dataset for the first time, enabling research under nighttime and adverse illumination conditions. The mid-range + close-range LiDAR combination extends spatial coverage. The RGB + thermal + LiDAR trimodal combination is the richest offered by any existing dataset.

2. **Precise Spatiotemporal Synchronization and Calibration Scheme**

    - **Function**: Ensures accurate alignment of heterogeneous multi-source sensors.
    - **Mechanism**: Temporal synchronization uses UTC as a unified reference—vehicle sensors are synchronized via a GPS→IMU→PTP chain, LiDARs are phase-locked to align rotational periods, and cameras are hardware-triggered when the LiDAR beam sweeps across their field of view. Infrastructure sensors are synchronized via PTP/NTP servers; the maximum offset between thermal cameras (30 FPS) and LiDARs (20 FPS) is 16.6 ms. Spatial calibration employs conical retroreflective targets with RTK GPS (2 cm accuracy), optimized by minimizing reprojection error. Per-point LiDAR motion compensation addresses vehicle motion during rotational scanning.
    - **Design Motivation**: Multi-source alignment precision is the core challenge in cooperative perception. The phase-locking + hardware-triggering + per-point compensation scheme substantially outperforms simple timestamp matching. The estimated maximum spatial error under 50 km/h conditions is 0.7 m—unavoidable given that targets are also in motion.

3. **Cross-Intersection Evaluation Strategies (EIS vs. SIS)**

    - **Function**: Separately assess in-domain performance on known environments and out-of-domain generalization on unseen intersections.
    - **Mechanism**: **EIS** (Equal Intersection Split)—sequence-level partitioning where each split contains sequences from all intersections (21 train / 6 val / 7 test), evaluating performance on known intersections. **SIS** (Separate Intersection Split)—intersection-level partitioning where all data from a given intersection appears exclusively in either training or testing (leave-one-out scheme), evaluating generalization. Three configurations are designed: $\text{SIS}_{1/2\text{vs}3}$, $\text{SIS}_{1/3\text{vs}2}$, and $\text{SIS}_{2/3\text{vs}1}$.
    - **Design Motivation**: Frame-level splits carry severe data leakage risk due to high similarity between temporally adjacent frames; sequence-level splits may still overestimate performance due to geometric similarity across intersections. Only intersection-level splits can genuinely evaluate generalization. The observed 14 mAP gap validates the necessity of this evaluation design.

### Loss & Training

All baseline experiments use a PointPillars backbone and evaluate No Fusion, Early Fusion, Late Fusion, and five Intermediate Fusion methods (F-Cooper, AttFuse, V2X-ViT, Where2Comm, CoBEVT). Four super-categories are used: Vehicle, Two-Wheelers, Heavy Vehicle, and Pedestrian. Evaluation metrics are mAP@0.3 and mAP@0.5.

## Key Experimental Results

### Main Results ($\text{SIS}_{1/2\text{vs}3}$ Split)

| Method | AP_Veh@0.5 | AP_HVeh@0.5 | AP_Ped@0.5 | AP_TWheel@0.5 | mAP@0.5 |
|--------|-----------|-------------|-----------|--------------|---------|
| No Fusion | 40.9 | 17.6 | 0.7 | 13.8 | 18.3 |
| Early Fusion | 41.1 | 24.8 | 3.5 | 21.6 | 22.8 |
| Late Fusion | 24.6 | 6.9 | 0.8 | 12.1 | 11.1 |
| F-Cooper | 46.7 | 24.0 | 3.1 | 23.2 | 24.2 |
| AttFuse | **47.6** | **27.8** | **4.6** | 22.1 | **25.5** |
| V2X-ViT | 46.2 | 22.2 | 3.5 | 18.0 | 22.5 |
| CoBEVT | 46.0 | 29.6 | 3.3 | 20.5 | 24.9 |

### Ablation Study (Generalization Gap)

| Split | mAP@0.5 | Description |
|-------|---------|-------------|
| EIS avg (seen intersections) | 38.2 | Testing on intersections seen during training |
| SIS avg (unseen intersections) | 24.2 | Testing on completely unseen intersections |
| **Generalization Gap** | **−14.0** | **Existing methods severely overfit to specific intersections** |

| SIS Configuration | Test Intersection | mAP@0.5 | Characteristics |
|-------------------|-------------------|---------|-----------------|
| $\text{SIS}_{1/2\text{vs}3}$ | Intersection 3 | 24.6 | Medium difficulty |
| $\text{SIS}_{1/3\text{vs}2}$ | Intersection 2 | 19.1 | Hardest (densest and most dynamic) |
| $\text{SIS}_{2/3\text{vs}1}$ | Intersection 1 | 28.9 | Relatively simple |

### Key Findings

- **The 14 mAP generalization gap is the most important finding of this paper**: EIS (38.2) → SIS (24.2), demonstrating that existing cooperative perception methods severely overfit to the geometric and traffic patterns of specific intersections—an important warning for the community.
- Intermediate Fusion consistently outperforms other strategies overall; AttFuse leads with 25.5 mAP. Late Fusion performs worst (11.1 mAP), indicating that multi-source object list association remains the primary bottleneck.
- Pedestrian detection is universally challenging (best result only 4.6 AP@0.5), attributable to small object size and sparse LiDAR returns.
- Substantial inter-intersection variability is observed: Intersection 2 is the most difficult (dense pedestrian and vehicle traffic, complex dynamics), while Intersection 1 is the easiest (highest point cloud visibility).
- Average annotated instances per frame range from 78 to 129, far exceeding those of DAIR-V2X-C and TUMTraf-V2X.

## Highlights & Insights

- **Exposing the illusion of inflated performance**: High mAP on a single intersection may be meaningless—models may have learned that "the right-turn lane at this intersection has this specific geometry" rather than general cooperative perception capabilities. The 14 mAP gap strongly advocates for multi-intersection evaluation becoming the standard practice.
- **Introduction of thermal cameras**: Infrared thermal cameras are incorporated into a V2X dataset for the first time (17 cameras, 38.8K images), opening a new dimension for cooperative perception research under nighttime and adverse illumination conditions.
- **Engineering reference value**: The synchronization scheme combining LiDAR phase-locking, camera hardware triggering, and per-point motion compensation achieves industrial-grade precision and serves as a direct reference for other multi-source sensor system designs.
- **Added value of the CARLA digital twin**: The georeferenced digital twin supports synthetic data generation and sim-to-real research, providing a scalable pathway when real-world data collection is costly.

## Limitations & Future Work

- Only 3 intersections, all located within the single city of Ingolstadt, leaving geographic diversity limited—cross-city and cross-country evaluation is a natural next step.
- The 34 scenes of approximately 20 seconds each (~680 seconds of effective data) represent a modest total scale by industrial standards, though annotation quality is high.
- Current baselines evaluate only LiDAR-only methods, leaving the multimodal fusion potential of RGB and thermal camera data largely unexplored—this gap requires follow-up from the research community.
- The maximum spatial fusion error of 0.7 m has a pronounced impact on small targets such as pedestrians (approximately 0.4 × 0.6 m in size).
- No adverse weather data is included (only illumination variation); evaluation of cooperative perception under rain, snow, and fog conditions is absent.

## Related Work & Insights

- **vs. DAIR-V2X-C**: Covers 28 intersections but is limited to single-vehicle V2I and a China-specific region. UrbanIng-V2X covers fewer intersections but provides multi-vehicle + multi-infrastructure coverage, global availability, HD maps, and a digital twin.
- **vs. V2X-Real**: Includes dual-vehicle and infrastructure sensors but is confined to a single intersection and lacks HD maps. UrbanIng-V2X extends to multiple intersections and completes the toolchain.
- **vs. TUMTraf-V2X**: Also a German single-intersection V2I dataset. UrbanIng-V2X is its natural extension—multiple intersections, multiple vehicles, and thermal cameras.
- **vs. V2V4Real**: Pure V2V with no infrastructure. UrbanIng-V2X completes the full V2V+V2I scenario.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First V2X dataset to satisfy the combined requirements of multi-vehicle + multi-infrastructure + multi-intersection + thermal cameras, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers a variety of fusion strategies but is limited to LiDAR-only evaluation; multimodal fusion and a broader range of detection methods remain unevaluated.
- **Writing Quality**: ⭐⭐⭐⭐ — Dataset construction details are thorough, the synchronization scheme is described with engineering clarity, and statistical analysis is comprehensive.
- **Value**: ⭐⭐⭐⭐ — The 14 mAP generalization gap delivers an important warning to the cooperative perception community; the SIS evaluation strategy deserves adoption as standard practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[NeurIPS 2025\] L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery](l2rsi_cross-view_lidar-based_place_recognition_for_large-scale_urban_scenes_via_.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
