---
title: >-
  [Paper Note] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection
description: >-
  [ICLR 2026][Object Detection][person detection] ForestPersons is the first large-scale benchmark dataset specifically designed for under-canopy missing person detection in forest environments (96,482 images + 204,078 annotations). By simulating the low-altitude flight perspective of micro aerial vehicles (MAVs) at 1.5–2.0 meters, the dataset covers multi-season, multi-weather, multi-pose, and multi-occlusion-level conditions representative of real search-and-rescue (SAR) scenarios, providing a solid foundation for training and evaluating under-canopy person detection models.
tags:
  - ICLR 2026
  - Object Detection
  - person detection
  - forest search and rescue
  - UAV
  - occlusion-awareness
  - dataset
date: 2026-05-08
content_hash: 980dffa1b55e5275
---

# ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection

**Conference**: ICLR 2026
**arXiv**: [2603.02541](https://arxiv.org/abs/2603.02541)
**Code**: [https://huggingface.co/datasets/etri/ForestPersons](https://huggingface.co/datasets/etri/ForestPersons)
**Area**: Object Detection
**Keywords**: person detection, forest search and rescue, UAV, occlusion-awareness, dataset

## TL;DR

ForestPersons is the first large-scale benchmark dataset specifically designed for under-canopy missing person detection in forest environments (96,482 images + 204,078 annotations). By simulating the low-altitude flight perspective of micro aerial vehicles (MAVs) at 1.5–2.0 meters, the dataset covers multi-season, multi-weather, multi-pose, and multi-occlusion-level conditions representative of real search-and-rescue (SAR) scenarios, providing a solid foundation for training and evaluating under-canopy person detection models.

## Background & Motivation

**Background**: UAVs have been widely deployed in SAR operations, enabling rapid coverage of large open areas. With advances in hardware miniaturization and SLAM technology, MAVs are now capable of safe navigation and exploration in GPS-denied forest environments.

**Limitations of Prior Work**:

1. **Viewpoint Limitation**: Existing SAR datasets (HERIDAL, WiSARD, SARD, VTSaR) are collected from high-altitude nadir or oblique perspectives, where dense forest canopies cause persons to occupy only a few pixels in the image, making detection extremely difficult.
2. **Scene Bias**: Ground-level person detection datasets (COCO, CrowdHuman, CityPersons) primarily cover standing or walking individuals in urban environments, which differs substantially from forest SAR scenarios — cases such as lying down, sitting, and vegetation occlusion are rarely represented.
3. **Missing Annotations**: No existing dataset simultaneously provides occlusion-level and pose annotations, preventing systematic evaluation of detection capability under varying difficulty conditions.

**Key Challenge**: The missing persons most critical to detect in forest SAR operations are precisely those in scenarios least covered by existing datasets — beneath the forest canopy, occluded by vegetation, and in non-standing poses.

**Goal**: Construct ForestPersons, the first large-scale benchmark dataset focused on under-canopy person detection, simulating the low-altitude MAV perspective and supplemented with semantic annotations for pose and visibility, to support the development of detection models suited to real SAR scenarios.

## Method

### Overall Architecture

The construction pipeline of ForestPersons proceeds as follows: forest environment video capture → frame sampling → bounding box annotation → pose and visibility attribute annotation → face anonymization → difficulty-aware data splitting. The entire pipeline is designed around the core principle of faithfully reproducing real SAR conditions.

### Key Design 1: Multi-Dimensional Data Collection Strategy

Data collection aims to replicate the complexity of real SAR scenarios as closely as possible:

- **Viewpoint Simulation**: Handheld or tripod-mounted cameras capture footage at 1.5–2.0 meters height, simulating the low-altitude ground-level perspective of MAVs flying beneath the forest canopy.
- **Pose Diversity**: Volunteers enact states of fatigue or disorientation, adopting three poses — standing, sitting, and lying on the ground — while subject to natural occlusion by vegetation, branches, and terrain.
- **Environmental Coverage**: Covers four seasons (dense summer canopy vs. leafless winter with snow), multiple weather conditions (sunny/cloudy/light rain), and different times of day (afternoon/dusk).
- **Total Scale**: 96,482 images and 204,078 annotated instances sampled from 377 video clips.

### Key Design 2: Three-Dimensional Semantic Annotation Scheme

In addition to bounding boxes, each person instance is annotated with two SAR-relevant semantic attributes:

**Pose Categories** (3 classes):

| Class | Description | SAR Significance |
|-------|-------------|-----------------|
| Standing | Upright | Mild condition / conscious |
| Sitting | Seated | Fatigued / waiting |
| Lying | Prone/supine | Injured / unconscious |

**Visibility Levels** (4 levels):

| Level | Description | Occlusion |
|-------|-------------|-----------|
| 100 | Fully visible | No occlusion |
| 70 | Slightly occluded | Most of body clearly visible |
| 40 | Partially occluded | Person identifiable but notably occluded |
| 20 | Heavily occluded | Barely identifiable |

When pose is difficult to determine due to occlusion, annotators refer to adjacent video frames for decision-making.

### Key Design 3: Model-Driven Difficulty-Aware Data Splitting

Data is split at the video-sequence level to prevent temporal leakage between adjacent frames across splits. The splitting strategy is based on model-driven difficulty estimation:

1. A COCO-pretrained Faster R-CNN computes $AP_{50}$ on each video sequence.
2. The difficulty score is defined as $1 - AP_{50}$.
3. Sequences are grouped into three difficulty tiers: easy ($< 0.45$), medium ($0.45 \le \text{score} < 0.75$), and hard ($\ge 0.75$).
4. Sequences are proportionally distributed across train/validation/test splits.

Final split: training set — 67,686 images + 145,816 annotations; validation set — 18,243 images + 37,395 annotations; test set — 10,553 images + 20,867 annotations.

## Key Experimental Results

### Main Results: Transfer Performance of Existing Datasets to Under-Canopy Scenarios

Faster R-CNN models trained on different datasets are evaluated on the ForestPersons test set to quantify the inadequacy of existing data:

| Training Data | Type | Own Test AP | ForestPersons AP | ForestPersons AP₅₀ |
|---------------|------|:-----------:|:----------------:|:------------------:|
| SARD | SAR/nadir | 58.6 | 3.0 | 7.8 |
| HERIDAL | SAR/nadir | 35.0 | 0.2 | 0.3 |
| WiSARD | SAR/oblique | 18.5 | 11.3 | 29.0 |
| COCO-Person | ground/urban | 54.0 | 40.8 | 66.9 |
| CrowdHuman | ground/urban | 39.4 | 31.9 | 58.8 |
| CityPersons | ground/urban | 38.7 | 5.9 | 15.1 |

All SAR datasets achieve AP below 12% on ForestPersons, confirming that high-altitude perspective data cannot generalize to under-canopy scenarios. Among ground-level datasets, COCO performs best (AP = 40.8), yet still exhibits significant performance degradation, underscoring the domain gap between urban and forest environments.

### Baseline Results: Multi-Detector Performance on ForestPersons

| Detection Model | Backbone | AP | AP₅₀ | AP₇₅ | AR |
|----------------|----------|----|------|------|----|
| SSD | MobileNetV2 | 45.0 | 83.6 | 43.1 | 53.7 |
| YOLOv3 | YOLO | 50.2 | 86.5 | 53.9 | 58.6 |
| YOLOX | YOLO | 51.0 | 89.0 | 54.4 | 58.2 |
| DETR | Transformer | 53.9 | 88.7 | 59.4 | 67.9 |
| RetinaNet | ResNet-50 | 64.2 | 93.9 | 74.4 | 70.9 |
| Faster R-CNN | ResNet-50 | 64.4 | 92.7 | 75.4 | 70.0 |
| DINO | Transformer | 65.3 | 94.0 | 76.2 | **77.7** |
| YOLOv11 | YOLO | 65.6 | 93.4 | 75.6 | 71.7 |
| CZ Det | Cascade-Zoom | 65.6 | **96.1** | **77.9** | 71.6 |
| Deformable R-CNN | ResNet-50 | **66.3** | 93.4 | 77.5 | 71.3 |

Deformable R-CNN achieves the highest overall AP (66.3), though the best-performing model varies by metric: DINO leads in AR (77.7, which is more critical in SAR contexts emphasizing recall), while CZ Det achieves the highest AP₅₀ and AP₇₅.

### Ablation Study: Impact of Attributes on Detection Performance

| Training → Test Attribute | Standing AP | Sitting AP | Lying AP |
|--------------------------|:-----------:|:----------:|:--------:|
| Standing only | 45.3–60.1 | 30.0–44.5 | 31.7–46.0 |
| All poses | 49.3–65.5 | 50.6–65.7 | 47.5–65.1 |

Training exclusively on Standing data leads to severe performance degradation for Sitting and Lying detection (approximately −20 AP). Training with all pose categories yields substantial improvements across all three pose classes, confirming the necessity of multi-pose data coverage.

**Correlation between Visibility Level and Detection Performance**: Detection accuracy increases consistently with visibility level (from level 20 to level 100), validating that the difficulty gradient design in ForestPersons aligns with realistic SAR conditions.

## Highlights & Insights

### Strengths

1. **Filling a Critical Gap**: ForestPersons is the first large-scale person detection dataset focused on the under-canopy perspective; its 96K+ image scale exceeds the largest prior SAR dataset (WiSARD, 44K) by more than twofold.
2. **Comprehensive Annotation**: The three-dimensional annotation scheme — bounding boxes, pose, and visibility — provides a unique foundation for systematically studying occlusion robustness.
3. **Thorough Experiments**: Beyond multi-detector benchmarking, cross-dataset transfer experiments quantitatively demonstrate the necessity of the proposed dataset.

### Limitations & Future Work

1. Data collection relies on volunteers simulating SAR scenarios, which may introduce distributional bias relative to the actual appearance and pose of real missing persons.
2. Only RGB data is provided (a thermal infrared variant, ForestPersonsIR, is briefly mentioned in the appendix), leaving the potential of multimodal fusion largely unexplored.
3. The best detector achieves only 66.3% AP, indicating substantial room for improvement, yet the paper proposes no targeted detection method to address this challenge.

### Rating

⭐⭐⭐⭐ — As a dataset paper, ForestPersons excels in problem definition clarity, data scale, and annotation quality, opening an important research direction for computer vision in forest search-and-rescue operations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InfoDet: A Dataset for Infographic Element Detection](infodet_a_dataset_for_infographic_element_detection.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](../../ICCV2025/object_detection/revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](../../CVPR2026/object_detection/evaluating_fewshot_pill_recognition_under_visual_d.md)

</div>

<!-- RELATED:END -->
