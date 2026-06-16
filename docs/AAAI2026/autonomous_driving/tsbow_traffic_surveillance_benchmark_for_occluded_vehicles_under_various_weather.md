---
title: >-
  [Paper Note] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions
description: >-
  [AAAI2026][Autonomous Driving][Object Detection] This paper presents TSBOW — a large-scale CCTV-based traffic surveillance dataset comprising 198 videos, over 32 hours of real-world traffic data, and 3.2 million frames…
tags:
  - "AAAI2026"
  - "Autonomous Driving"
  - "Object Detection"
  - "Traffic Surveillance"
  - "Benchmark Dataset"
  - "Occlusion Detection"
  - "Adverse Weather"
  - "CCTV"
date: 2026-05-08
content_hash: 8d0379530749605b
---

# TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions

**Conference**: AAAI2026
**arXiv**: [2602.05414](https://arxiv.org/abs/2602.05414)  
**Code**: [SKKUAutoLab/TSBOW](https://github.com/SKKUAutoLab/TSBOW)  
**Authors**: Ngoc Doan-Minh Huynh, Duong Nguyen-Ngoc Tran, Long Hoang Pham et al. (Sungkyunkwan University)
**Area**: Autonomous Driving
**Keywords**: Object Detection, Traffic Surveillance, Benchmark Dataset, Occlusion Detection, Adverse Weather, CCTV

## TL;DR

This paper presents TSBOW — a large-scale CCTV-based traffic surveillance dataset comprising 198 videos, over 32 hours of real-world traffic data, and 3.2 million frames, covering all four seasons (clear/haze/rain/snow including extreme disaster scenarios), spanning 8 categories of traffic participants, with a focus on addressing the challenge of occluded vehicle detection under adverse weather conditions.

## Background & Motivation

Global warming has intensified the frequency and severity of extreme weather events, which degrade CCTV signal and video quality, disrupt traffic flow, and increase accident rates. Existing traffic surveillance datasets such as UAVDT and UA-DETRAC primarily cover clear and lightly rainy conditions, with severely insufficient coverage of extreme weather scenarios (heavy snowfall, dense fog, etc.). Although the AAURainSnow dataset includes rain and snow, it contains only 1.83 hours of data and does not provide bounding box annotations.

From the detection perspective, object detection models face multiple challenges under adverse weather: puddle reflections distort bounding box dimensions; camera shake caused by strong winds and unstable connections introduces motion blur; snow-covered vehicles blend into white backgrounds and become difficult to distinguish. The scale and diversity of existing datasets are insufficient to train models that operate robustly in complex real-world scenarios.

Furthermore, existing benchmark datasets typically contain only 3–5 annotated categories, lacking important traffic participants such as micromobility devices and pedestrians. In dense urban areas, occlusion is a severe problem (vehicles heavily occlude one another at intersections, construction sites, etc.), and models frequently fail to detect heavily occluded instances. Therefore, there is an urgent need to construct a large-scale, high-quality traffic surveillance benchmark covering diverse weather conditions, road types, and occlusion levels.

## Core Problem

How to construct a large-scale traffic surveillance benchmark covering diverse weather conditions throughout the year (particularly extreme disaster weather), multiple road types, and varying degrees of occlusion, so as to advance research on occluded vehicle detection under adverse weather?

## Method

### Data Collection Design

TSBOW data is sourced from a fixed-route CCTV system in Suwon, South Korea, and is systematically categorized along four dimensions:

- **Scene Type**: Road, Intersection, Special Case (shared lanes / overpasses / road construction), Disaster (extreme weather severely degrading video quality)
- **Weather Condition**: Normal (clear/cloudy), Haze/Fog, Rain (visible raindrops required), Snow (visible snowflakes required)
- **Road Type**: Urban (two-lane), Standard (four-lane), Boulevard (six or more lanes)
- **Scale**: Fine, Medium, Coarse — determined by camera height and distance

### Semi-Automatic Annotation Pipeline

Annotation follows a five-stage pipeline: (1) video preprocessing, defining ROI regions and extracting keyframes; (2) manual annotation using the X-Anylabeling tool; (3) semi-automatic annotation using a YOLOv12x model fine-tuned on Korean vehicle characteristics; (4) annotation verification and quality control; (5) post-processing to generate the final dataset. A total of 48,061 frames were manually annotated, with over 3.2 million frames annotated semi-automatically.

### Annotation Category System

The dataset contains 8 categories: car, bus, truck, small truck, micromobility, pedestrian, unidentified, and others. Occlusion levels are divided into three tiers based on IoU: no occlusion (<15%), partial occlusion (15–40%), and heavy occlusion ($\geq$40%), with 721,684, 266,420, and 143,051 instances respectively.

## Key Experimental Results

### Baseline Model Performance Comparison (Test Set, imgsz=1280, 100 epochs)

| Model | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| YOLOv8x | 0.783 | 0.705 | 0.733 | 0.609 |
| YOLO11x | 0.786 | 0.696 | 0.734 | 0.614 |
| YOLOv12x | **0.806** | 0.662 | **0.744** | **0.615** |
| RT-DETR-x | 0.731 | **0.740** | 0.718 | 0.552 |

### YOLOv12x Per-Category Performance

| Category | Instances | Precision | Recall | mAP50 | mAP50-95 |
|----------|-----------|-----------|--------|-------|----------|
| Car | 479,560 | 0.959 | 0.932 | 0.959 | 0.849 |
| Bus | 21,037 | 0.917 | 0.929 | 0.951 | 0.876 |
| Small Truck | 36,152 | 0.878 | 0.830 | 0.870 | 0.750 |
| Truck | 6,690 | 0.824 | 0.575 | 0.720 | 0.643 |
| Pedestrian | 32,779 | 0.833 | 0.605 | 0.715 | 0.447 |
| Micromobility | 18,490 | 0.793 | 0.574 | 0.726 | 0.519 |
| Unidentified | 4,855 | 0.475 | 0.223 | 0.317 | 0.221 |

### Cross-Scene/Weather/Scale Performance Analysis (YOLOv12x)

| Group | Condition | mAP50 | mAP50-95 |
|-------|-----------|-------|----------|
| Scene | Intersection | 0.759 | 0.633 |
| Scene | Disaster | 0.656 | 0.510 |
| Weather | Rain | 0.789 | 0.664 |
| Weather | Snow | 0.723 | 0.597 |
| Scale | Fine | 0.686 | 0.559 |
| Scale | Coarse | 0.733 | 0.581 |

### Cross-Dataset Generalization (YOLOv12x, Car Category Only)

| Training Data | Precision | Recall | mAP50 | mAP50-95 |
|---------------|-----------|--------|-------|----------|
| UAVDT | 0.647 | 0.141 | 0.383 | 0.328 |
| UA-DETRAC | 0.820 | 0.295 | 0.558 | 0.459 |
| **TSBOW** | **0.743** | **0.869** | **0.846** | **0.792** |

## Highlights & Insights

- **Comprehensive Extreme Weather Coverage**: The first large-scale CCTV traffic surveillance dataset containing disaster-level weather scenarios such as heavy snowstorms and dense fog, with 198 videos spanning all four seasons (52 clear / 15 haze / 46 rain / 85 snow).
- **Notable Scale**: Over 32 hours, 3.2 million frames, and 71 million+ annotated instances, far exceeding UAVDT (10 hours / 80K frames) and UA-DETRAC (10 hours / 140K frames).
- **Fine-Grained 8-Category Annotations**: Covers car, bus, truck, small truck, micromobility, pedestrian, and others, with a relatively balanced class distribution (car accounts for 69%, compared to over 83% in other benchmarks).
- **Systematic Occlusion Analysis**: Three-tier occlusion classification; mAP50 drops to 0.656 in disaster scenarios, quantifying the challenge posed by extreme conditions to detection models.
- **Cross-Dataset Generalization Advantage**: Models trained on TSBOW achieve a Car mAP50 of 0.846, substantially outperforming models trained on other datasets.

## Limitations & Future Work

- **Daytime-Only Coverage**: The current version contains no nighttime data; detection under adverse weather at night poses even greater challenges, and the authors plan to address this in future releases.
- **Geographic Homogeneity**: All data is collected in Suwon, South Korea; vehicle types and traffic patterns may not generalize globally.
- **Limited Manual Annotation Ratio**: The 48,061 manually annotated frames account for only approximately 1.5% of total frames; the remainder is semi-automatically annotated by YOLOv12x, which may introduce systematic bias.
- **Absence of Tracking and Segmentation Annotations**: The current release provides only bounding box annotations, without tracking IDs or instance segmentation masks.

## Related Work & Insights

| Dataset | Duration | Total Frames | Categories | Resolution | Weather Coverage |
|---------|----------|--------------|------------|------------|-----------------|
| UAVDT | 10h | 80K | 3 | 1080×540 | Clear/Rain |
| UA-DETRAC | 10h | 140K | 5 | 960×540 | Clear/Cloudy/Rain/Night |
| AAURainSnow | 1.83h | 132K | — | 640×480 | Rain/Snow/Haze/Fog |
| **TSBOW** | **32.36h** | **3.2M** | **8** | **1280×720** | **Clear/Haze/Rain/Snow+Disaster** |

TSBOW comprehensively surpasses existing benchmarks in scale, resolution, number of categories, and weather diversity. UAVDT and UA-DETRAC lack snow scenarios; AAURainSnow provides no bounding box annotations; TSBOW is the only traffic surveillance dataset including disaster-level extreme weather.

This dataset provides a valuable validation platform for robust object detection under adverse weather. The mAP50 of only 0.656 in disaster scenarios indicates that current state-of-the-art detectors still have substantial room for improvement under extreme conditions. The dataset is further applicable to domain adaptation research (weather domain transfer), image restoration-assisted detection, and multi-scale dense object detection.

## Rating

- Novelty: ⭐⭐⭐⭐ — First large-scale CCTV traffic surveillance benchmark covering disaster-level extreme weather, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Provides baselines for 4 SOTA detectors with detailed cross-scene/weather/scale analysis and cross-dataset generalization evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Dataset construction pipeline is clearly described with thorough statistical analysis.
- Value: ⭐⭐⭐⭐ — Dataset is publicly available and holds high practical value for intelligent transportation systems research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation](../../CVPR2026/autonomous_driving/hg-lane_high-fidelity_generation_of_lane_scenes_under_adverse_weather_and_lighti.md)
- [\[AAAI 2026\] Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions](walking_further_semantic-aware_multimodal_gait_recognition_under_long-range_cond.md)
- [\[ICCV 2025\] TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes](../../ICCV2025/autonomous_driving/trafficloc_localizing_traffic_surveillance_cameras_in_3d_scenes.md)
- [\[AAAI 2026\] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework](when_person_re-identification_meets_event_camera_a_benchmark_dataset_and_an_attr.md)
- [\[CVPR 2026\] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule](../../CVPR2026/autonomous_driving/perception_characteristics_distance_measuring_stability_and_robustness_of_percep.md)

</div>

<!-- RELATED:END -->
