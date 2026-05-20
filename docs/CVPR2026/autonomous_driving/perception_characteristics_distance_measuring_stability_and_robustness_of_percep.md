---
title: >-
  [Paper Note] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule
description: >-
  [CVPR2026][Autonomous Driving][Perception evaluation metrics] This paper proposes the Perception Characteristics Distance (PCD), a novel metric that quantifies the maximum reliable detection range of a perception system…
tags:
  - "CVPR2026"
  - "Autonomous Driving"
  - "Perception evaluation metrics"
  - "detection range reliability"
  - "uncertainty modeling"
  - "variance change-point detection"
  - "autonomous driving safety"
date: 2026-05-08
content_hash: e8953a4fcd985b0f
---

# Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule

**Conference**: CVPR2026
**arXiv**: [2506.09217](https://arxiv.org/abs/2506.09217)  
**Code**: [datadrivenwheels/PCD_Python](https://github.com/datadrivenwheels/PCD_Python)  
**Area**: Autonomous Driving / Perception Evaluation
**Keywords**: Perception evaluation metrics, detection range reliability, uncertainty modeling, variance change-point detection, autonomous driving safety

## TL;DR

This paper proposes the Perception Characteristics Distance (PCD), a novel metric that quantifies the maximum reliable detection range of a perception system by statistically modeling how the mean and variance of detection confidence evolve with distance. Given a detection quality threshold $y^{thres}$ and a probability threshold $p^{thres}$, PCD identifies the furthest distance at which reliability requirements are satisfied, addressing the inability of conventional static metrics such as AP and IoU to capture distance-dependent behavior and stochastic variation.

## Background & Motivation

1. **Limitations of traditional metrics**: Classical evaluation metrics such as AP, IoU, and F1 are based on static, per-frame assessment and ignore the temporal and spatial continuity of real driving scenarios, making them insensitive to stability differences across detection ranges.
2. **Instability at long range**: Detectors such as YOLOX produce stable confidence scores (≥0.90) at short range (<30 m), but exhibit severe fluctuations at long range (≥70 m, potentially as low as 0.24), making fixed-threshold classification prone to significant errors.
3. **Fragility of threshold-based decisions**: Control logic in ADAS/ADS typically applies a confidence threshold to produce binary detection decisions, a paradigm that fails to capture the stochasticity and distance-dependent variability inherent in perception outputs.
4. **Safety requirements**: Autonomous driving safety depends on accurate estimation of the maximum reliable detection range; decision systems must know beyond what distance perception outputs can no longer be trusted.
5. **Lack of controlled benchmark datasets**: Existing driving datasets (nuScenes, KITTI, BDD100K) are collected in naturalistic environments and lack the controlled conditions necessary for systematic evaluation of perception robustness.
6. **Insensitivity of existing metrics to condition variation**: Conventional metrics such as AP show limited sensitivity to changes in weather or illumination, and thus cannot effectively characterize perception degradation under adverse conditions.

## Method

### Overall Architecture

The core mechanism of PCD is to model the perception output (IoU × Confidence) as a function of distance $x$, statistically estimate its mean and variance, and identify the maximum distance at which reliability requirements are met under a given detection quality threshold $y^{thres}$ and probability threshold $p^{thres}$.

**Rationale for IoU × Confidence**: Confidence alone reflects only model certainty, while IoU captures only localization accuracy. Their product simultaneously encodes detection quality and certainty, making it more suitable for evaluating perceptual stability.

### Key Designs

**Variance Change-Point Detection**:

1. Penalized B-spline regression (Penalized B-spline, $K=10$, cubic order) is used to fit the mean function $f(x)$ of IoU×Confidence over distance.
2. A test statistic based on the Schwarz Information Criterion (SIC) is constructed to detect significant change points in residual variance.
3. A sequential hypothesis testing strategy is adopted: the first change point $x_{\tau_1}$ is detected on the full dataset, followed by recursive detection of subsequent change points on each partitioned subset.
4. Change points divide the distance range into segments within which variance is approximately constant.

**PCD Computation**:

- Within each segment, IoU×Confidence follows a normal distribution $y_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$.
- PCD is defined as the maximum distance $x_i$ satisfying $P_Y(y_i > y^{thres}) > p^{thres}$.
- The aggregate metric aPCD averages over multiple combinations of $(p^{thres}, y^{thres})$, summarizing overall perceptual capability in a manner analogous to AUC.

### Loss & Training

This paper presents an evaluation metric rather than a learned model; no training loss is involved. The computation of PCD relies on:

- Penalized spline regression with regularization: $\sum_{i=1}^n [y_i - \sum_j \beta_j B_j(x_i)]^2 + \lambda \sum_{j=3}^K (\Delta^2 \beta_j)^2$, where $\lambda=0.6$
- Change-point hypothesis testing based on log-likelihood ratios and the SIC criterion.

## Key Experimental Results

### SensorRainFall Dataset

- Collected at the Virginia Smart Road facility under controlled rainfall intensity of 64 mm/h.
- Four environmental conditions: sunny daytime, rainy daytime, rainy nighttime, rainy nighttime with streetlights.
- 1,231 front-view images at 1920×1080, covering distances from 4 m to 250 m.
- Two target types: a red sedan and a pedestrian mannequin, with ground-truth bounding boxes, segmentation masks, and precise distance annotations.

### Main Results (representative results from 16 experimental groups)

**Instance Segmentation — Vehicle — Sunny Daytime**:

| Model | aPCD (m) | AP50:95 | AP50 | AR | F1_50 |
|-------|----------|---------|------|----|-------|
| Mask2Former | **107.1** | **0.423** | **0.633** | **0.427** | **0.778** |
| Mask R-CNN | 89.8 | 0.376 | 0.579 | 0.381 | 0.736 |
| ConvNeXt-V2 | 89.5 | 0.395 | 0.553 | 0.399 | 0.715 |
| RTMDet | 43.5 | 0.349 | 0.593 | 0.353 | 0.747 |
| SOLOv2 | 36.6 | 0.233 | 0.276 | 0.237 | 0.438 |

**Object Detection — Vehicle — Rainy Nighttime** (a representative case where aPCD and traditional metric rankings diverge):

| Model | aPCD (m) | AP50:95 | AP50 | AR | F1_50 |
|-------|----------|---------|------|----|-------|
| GLIP | **37.3** | 0.133 | 0.288 | 0.136 | 0.451 |
| Grounding DINO | 29.6 | 0.125 | 0.297 | 0.128 | 0.461 |
| YOLOX | 23.8 | 0.106 | 0.212 | 0.109 | 0.353 |
| DyHead | 21.5 | **0.144** | **0.362** | **0.146** | **0.534** |
| Deformable DETR | 3.8 | 0.056 | 0.133 | 0.058 | 0.239 |

### Ablation Study

- **Effect of sample size**: Change-point detection achieves good accuracy and stability when the number of detected change points is fewer than 4; larger sample sizes yield more precise variance change-point estimation.
- **Effect of effect size**: Under a 50-50 sample split, a variance ratio of approximately 3× is sufficient for reliable detection.
- **Threshold sensitivity**: PCD varies smoothly with threshold under sunny and rainy daytime conditions; under rainy nighttime conditions, PCD exhibits sharp fluctuations, indicating greater threshold sensitivity.

## Highlights & Insights

1. **Filling a metric gap**: This is the first probabilistic, distance-aware perception evaluation metric that directly links detection reliability to physical distance.
2. **Revealing blind spots of traditional metrics**: In the rainy nighttime scenario, GLIP achieves the highest aPCD despite not ranking first in AP, demonstrating that AP ordering fails to reflect distance-dimensional stability (DyHead exhibits larger variance at long range).
3. **Safety envelope definition**: PCD can be directly used to define the safety operational envelope of ADS, informing decision-making distances under varying environmental conditions.
4. **Controlled dataset**: SensorRainFall is the only publicly available perception evaluation dataset collected in a highly controlled environment, eliminating confounding variables.
5. **Methodological rigor**: The combination of penalized splines and sequential variance change-point detection provides theoretically grounded heteroscedastic modeling.

## Limitations & Future Work

1. **Limited dataset scale**: SensorRainFall contains only 1,231 images and 2 object categories, limiting scene diversity.
2. **Validation confined to proprietary dataset**: PCD has not been validated on mainstream benchmarks such as nuScenes or KITTI.
3. **Narrow task coverage**: Evaluation is restricted to object detection and instance segmentation, without extension to 3D detection, depth estimation, or semantic segmentation.
4. **Normality assumption**: The assumption that IoU×Confidence follows a normal distribution within each segment may not hold under extreme conditions.
5. **Static target evaluation**: Target objects are stationary; evaluation of moving targets with varying speeds and poses is absent.
6. **Single-sensor scope**: Experiments are based solely on camera imagery, without PCD evaluation for LiDAR, Radar, or multi-sensor fusion systems.

## Related Work & Insights

| Method | Characteristics | Limitations |
|--------|----------------|-------------|
| AP / mAP | Precision-recall summary based on IoU thresholds | Ignores distance dimension and detection stability |
| PDQ (Hall et al.) | Joint spatial and semantic uncertainty | Does not account for distance dependence |
| LRP (Oksuz et al.) | Simultaneously considers localization, FP, and FN | Remains a static frame-level metric |
| AD (Mao et al.) | Introduces temporal delay evaluation | Does not model the distance-reliability relationship |
| GIoU (Rezatofighi et al.) | Addresses IoU for non-overlapping boxes | Unrelated to distance and stability |
| **PCD (Ours)** | **Distance-dependent + uncertainty-aware + dual adjustable thresholds** | **Limited dataset scope and task coverage** |

## Rating

- Novelty: ⭐⭐⭐⭐ — Defines perception evaluation from a novel distance-uncertainty perspective with a distinctive viewpoint.
- Experimental Thoroughness: ⭐⭐⭐ — Systematic evaluation across multiple models and conditions, but confined to the authors' own dataset with no validation on mainstream benchmarks.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical formulations are clear, examples are intuitive, and figures are well designed.
- Value: ⭐⭐⭐⭐ — Practically meaningful for ADS safety assessment and complementary to existing evaluation frameworks, though broader empirical validation is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks](mind_the_hitch_dynamic_calibration_and_articulated_perception_for_autonomous_tru.md)
- [\[AAAI 2026\] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System](../../AAAI2026/autonomous_driving/roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)
- [\[CVPR 2026\] AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](adaradar_rate_adaptive_spectral_compression_for_radar-based_perception.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](../../AAAI2026/autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[CVPR 2026\] A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)

</div>

<!-- RELATED:END -->
