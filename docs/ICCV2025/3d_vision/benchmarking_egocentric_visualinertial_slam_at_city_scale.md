---
title: >-
  [Paper Note] Benchmarking Egocentric Visual-Inertial SLAM at City Scale
description: >-
  [ICCV 2025][3D Vision][Visual-Inertial SLAM] This paper introduces LaMAria — the first city-scale egocentric multi-sensor VIO/SLAM benchmark dataset — providing centimeter-accurate ground truth via surveying-grade contro…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Visual-Inertial SLAM"
  - "egocentric vision"
  - "benchmark"
  - "city-scale localization"
  - "wearable devices"
date: 2026-05-08
content_hash: ffcc576ed79e351f
---

# Benchmarking Egocentric Visual-Inertial SLAM at City Scale

**Conference**: ICCV 2025
**arXiv**: [2509.26639](https://arxiv.org/abs/2509.26639)
**Code**: [lamaria.ethz.ch](https://www.lamaria.ethz.ch/) (dataset & evaluation platform)
**Area**: 3D Vision / SLAM
**Keywords**: Visual-Inertial SLAM, egocentric vision, benchmark, city-scale localization, wearable devices

## TL;DR

This paper introduces LaMAria — the first city-scale egocentric multi-sensor VIO/SLAM benchmark dataset — providing centimeter-accurate ground truth via surveying-grade control points. It systematically evaluates mainstream academic SLAM methods on real egocentric data, revealing a substantial performance gap between academic systems and commercial solutions.

## Background & Motivation

With the proliferation of wearable devices such as AR glasses, accurate 6-DoF localization in egocentric scenarios has become critical. However, existing VIO/SLAM datasets are almost exclusively collected from robotic or vehicular platforms, featuring controlled motion patterns, limited scale, and ground truth derived from indoor motion-capture systems (limited coverage) or GNSS (insufficient accuracy). Egocentric data presents unique challenges: highly diverse and uncontrolled motion (natural head movements, riding vehicles), camera calibration drift over long recordings, low-light conditions at night, exposure changes during indoor-outdoor transitions, and dynamic scenes (pedestrians, self-occlusion by the body). These challenges are largely absent from existing academic benchmarks.

## Core Problem

1. **Absence of a suitable benchmark**: No existing dataset simultaneously satisfies "egocentric capture + city scale + centimeter-level ground truth + multi-sensor + diverse challenging scenarios."
2. **Unknown robustness of existing SLAM systems**: Top academic methods perform well on controlled datasets (e.g., EuRoC), but their behavior on real egocentric data — and the magnitude of the performance gap — remains unknown.

## Method

### Overall Architecture

This is a **dataset/benchmark paper**. The core contribution is the design of the LaMAria dataset and a reliable evaluation methodology. The overall approach proceeds in three steps: (1) recording large-scale egocentric multi-sensor data in Zurich using Project Aria glasses; (2) obtaining centimeter-accurate pose annotations via the surveying concept of "control points"; and (3) systematically evaluating 8+ mainstream VIO/SLAM methods on this dataset.

### Key Designs

1. **Data collection — Project Aria device**: The paper uses Meta's Project Aria glasses-form-factor device, equipped with dual grayscale global-shutter cameras (640×480, 20 FPS), one RGB rolling-shutter camera (1408×1408, 10 FPS), dual IMUs (1 kHz / 800 Hz), magnetometer, barometer, GNSS, and WiFi/Bluetooth. Over six months, 63 sequences were recorded in central Zurich, covering approximately 1.5 km² and 50 m of elevation change. Each sequence averages 1.5 km / 26 minutes, with the longest reaching 2.87 km / 48 minutes, totaling over 22 hours and 70+ km of trajectories.

2. **Centimeter-accurate sparse ground truth — control point alignment**: This is the paper's most critical technical innovation. The authors leverage "Control Points (CPs)" from the surveying domain — 483 ground markers with known precise coordinates (horizontal accuracy ~1 cm) maintained by the City of Zurich. AprilTag fiducials are placed at each control point and automatically detected as the Aria device passes by. During evaluation, detected markers in SLAM output trajectories are triangulated and aligned to the control points via a $\text{Sim}(3)$ transformation; the alignment error directly reflects trajectory accuracy. Unlike dense motion-capture ground truth (e.g., EuRoC), this approach is sparse but highly accurate (cross-validation shows errors 70× smaller than the uncertainty budget) and scales to city-scale environments.

3. **Dense pseudo ground truth — multi-sensor joint optimization**: To support fine-grained analysis, the authors generate dense pseudo-GT poses by jointly optimizing visual feature reprojection errors, IMU pre-integration constraints, control-point triangulation errors, and control-point alignment errors, initialized from the Aria SLAM trajectory. The median positional uncertainty of the resulting pseudo-GT is approximately 20 cm.

4. **Tiered experimental splits**: Four difficulty levels are designed — Level I (controlled motion on a platform) to Level IV (authentic head-worn egocentric motion) — bridging the gap between standard academic datasets and fully challenging real-world data, enabling precise identification of system failure points.

### Loss & Training

*(This is a benchmark dataset paper; no training procedure applies.)*

**Scoring function design**: For the alignment error $e$ at each control point, a piecewise linear scoring function $s(e)$ is defined: error ≤5 cm scores 100, 50 cm scores 75, 1 m scores 60, 2 m scores 40, 5 m scores 20, and ≥10 m scores 0. Additional metrics include CP@1m (recall of control points within 1 m) and R@5m (positional recall within 5 m against dense pseudo-GT).

## Key Experimental Results

### Main Results (multi-camera + IMU configuration, 2D score)

| Category | Metric | OpenVINS+Maplab | OKVIS2 | ORB-SLAM3 (mono-I) | Aria SLAM |
|----------|--------|-----------------|--------|---------------------|-----------|
| Short (18 seq) | score | 26.0 | 24.2 | 28.3 | 90.7 |
| Medium (10 seq) | score | 21.3 | 13.6 | 20.3 | 78.5 |
| Long (16 seq) | score | 12.6 | 3.6 | 14.2 | 70.8 |
| Low-light (9 seq) | score | 16.5 | 15.4 | 6.2 | 84.2 |
| Moving platform (10 seq) | score | 13.0 | 4.2 | 15.7 | 53.6 |

**Core finding**: The best academic method achieves only ~28 points (out of 100) on short sequences, while Aria's commercial SLAM scores 90.7. The gap widens further on long sequences and challenging scenarios. Even Aria SLAM scores only 53.6 on the moving-platform category, confirming that the benchmark is far from saturated.

### Tiered Experimental Split (ATE RMSE, meters)

All methods perform adequately at Level I (controlled motion), e.g., ORB-SLAM3 achieves only 0.03–0.43 m. However, most methods degrade severely or fail at Level IV (egocentric motion), with academic methods exhibiting ATE values of 10–50 m.

### Ablation Study

- **Sensor configuration**: Multi-camera + IMU substantially outperforms monocular + IMU; monocular-only configurations fall far behind.
- **Online calibration**: A key differentiator between Aria SLAM and academic methods is support for **online optimization of time-varying intrinsics**. In long sequences, focal length variations reach 0.11%, and fixing factory calibration leads to measurable accuracy degradation.
- **Scale and gravity errors**: OpenVINS exhibits 6.38% scale error and 3.79° gravity direction error on short sequences, while Aria SLAM achieves only 0.15% / 0.18°.
- **Variance analysis**: ORB-SLAM3 exhibits the largest variance across runs, indicating insufficient robustness.

## Highlights & Insights

1. **Elegant surveying-grade GT acquisition**: The combination of city-maintained control points and AprilTag fiducials achieves a perfect balance of "non-intrusive natural motion + centimeter accuracy + city-scale coverage" — far more scalable than motion-capture systems and far more accurate than GNSS.
2. **The progressive Level I–IV experimental split** is a particularly clever design, enabling precise identification of which level of motion complexity causes a given system to fail.
3. **Comprehensive and fair evaluation**: Each method underwent hyperparameter tuning (in some cases in collaboration with the original authors), with 3 runs averaged and standard deviations reported — a level of rigor rarely seen in SLAM benchmark papers.
4. **Quantifies the true gap between industry and academia**: Aria SLAM scores 3–4× higher than the best academic method, providing clear directions for improvement (online calibration, robust loop closure, moving-platform handling).

## Limitations & Future Work

1. **Single-city data**: All recordings were collected in Zurich, limiting diversity in urban structure and texture distribution.
2. **Device constraints**: Only one device type (Project Aria) is used; the grayscale SLAM cameras have a relatively low resolution of 640×480, and the minimal overlap between the two SLAM camera fields of view precludes conventional stereo configurations.
3. **Inherent limitations of the control point approach**: Sparse evaluation cannot capture local trajectory errors between control points; accuracy in moving-platform segments is limited, precluding reliable dense GT in those regions.
4. **Closed-source Aria SLAM**: The best-performing system in the benchmark is not reproducible, limiting the research community's ability to learn from its implementation details.
5. **Incomplete evaluation of learning-based SLAM**: DPVO/DPV-SLAM were evaluated in vision-only mode only, as they do not support IMU integration.

## Related Work & Insights

| Dataset | Sensor Platform | Scale | GT Accuracy | GT Source | Egocentric | Multi-challenge |
|---------|----------------|-------|-------------|-----------|------------|-----------------|
| EuRoC | UAV | Small (indoor) | cm | Motion capture | ✗ | Partial |
| TUM-VI | Handheld | Medium | cm | Motion capture | ✗ | ✗ |
| 4Seasons | Vehicle | Large | dm | VI+GNSS | ✗ | Partial |
| LaMAR | Head-worn+handheld | Medium | dm | V-SLAM+LiDAR | ✓ | ✓ |
| **LaMAria** | **Head-worn** | **Large (city)** | **cm** | **Surveying CPs** | **✓** | **✓** |

LaMAria is the first VIO/SLAM benchmark to simultaneously cover large scale, egocentric perspective, centimeter-level accuracy, and diverse challenging scenarios.

The findings carry broader implications: for SLAM researchers, online calibration, robust loop closure, and moving-platform handling are the three most pressing research directions, with the industry-academia gap highlighting the importance of engineering-level optimization. For wearable AR products, the dataset directly reflects localization challenges in real user scenarios. Methodologically, the "surveying control points + visual fiducials" GT acquisition paradigm is transferable to the construction of other large-scale localization benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ (the control-point GT scheme is highly original, though benchmark papers are inherently constrained in methodological novelty)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8+ systems, multiple sensor configurations, tiered splits, variance analysis, short-segment analysis — comprehensive coverage)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear structure; the integration of SLAM and surveying domain knowledge is well-presented)
- Value: ⭐⭐⭐⭐⭐ (fills an important benchmark gap; dataset is publicly released; long-term impact on the SLAM community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[ICCV 2025\] Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps](outdoor_monocular_slam_with_global_scale-consistent_3d_gaussian_pointmaps.md)
- [\[ICCV 2025\] Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](egom2p_egocentric_multimodal_multitask_pretraining.md)

</div>

<!-- RELATED:END -->
