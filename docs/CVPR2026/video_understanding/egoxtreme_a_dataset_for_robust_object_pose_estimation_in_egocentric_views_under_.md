---
title: >-
  [Paper Note] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions
description: >-
  [CVPR 2026][Video Understanding][egocentric view] This paper introduces EgoXtreme, the first large-scale benchmark for 6D object pose estimation in egocentric views under extreme conditions, encompassing three real-world challenges — severe motion blur, dynamic illumination, and smoke occlusion — and reveals critical failures of current state-of-the-art pose estimators under these conditions.
tags:
  - CVPR 2026
  - Video Understanding
  - egocentric view
  - 6D pose estimation
  - extreme conditions
  - smart glasses
  - benchmark dataset
date: 2026-05-08
content_hash: d04a5035cb3ccbee
---

# EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions

**Conference**: CVPR 2026
**arXiv**: [2603.25135](https://arxiv.org/abs/2603.25135)
**Code**: [https://taegyoun88.github.io/EgoXtreme/](https://taegyoun88.github.io/EgoXtreme/)
**Area**: Video Understanding / 6D Pose Estimation
**Keywords**: egocentric view, 6D pose estimation, extreme conditions, smart glasses, benchmark dataset

## TL;DR

This paper introduces EgoXtreme, the first large-scale benchmark for 6D object pose estimation in egocentric views under extreme conditions, encompassing three real-world challenges — severe motion blur, dynamic illumination, and smoke occlusion — and reveals critical failures of current state-of-the-art pose estimators under these conditions.

## Background & Motivation

Smart glasses are emerging as important wearable devices, providing rich contextual understanding in "hands-busy, eyes-focused" scenarios. 6D object pose estimation from egocentric views is essential for understanding the wearer's activity environment.

However, existing 6D pose estimation benchmarks suffer from a significant **reality gap**:

- **YCB-Video, LineMod, T-LESS** and other standard datasets are collected under controlled laboratory conditions with stable illumination and slow motion.
- **H2O and HOT3D** introduce egocentric viewpoints but feature mild lighting conditions and lack testing under truly extreme environments.
- Real-world smart glasses usage poses unique challenges:
    - The camera is mounted on a moving head, producing rapid and frequent motion.
    - Close-range viewpoints cause severe truncation.
    - Dynamic/extreme illumination (headlamps, flashlights, emergency lights, exit signs).
    - Visual occlusion from smoke and haze.

Core problem: Bridging the gap through simulation alone (synthetic data, frame-averaging to simulate blur, etc.) is insufficient — **a real-world extreme-condition benchmark is needed to drive the development of robust models**.

## Method

### Overall Architecture

EgoXtreme is a dataset contribution, consisting of: (1) design and data collection across three challenging scenarios; (2) a high-precision 6D annotation pipeline based on an OptiTrack motion capture system; and (3) comprehensive benchmark evaluation of state-of-the-art methods.

### Key Designs

1. **Three Challenge Scenarios**: Covering different types of extreme conditions → each scenario targets a distinct perceptual difficulty → ensuring multi-dimensional evaluation.

    - **Industrial Maintenance**: Fine-grained manipulation with tools such as hammers, drills, and saws; 6 lighting conditions + smoke; tests robustness to precise motion and illumination variation.
    - **Sports Scenarios**: Rapid swinging of table tennis paddles, baseball bats, golf clubs, etc. (object speed up to 1.37 m/s); 4 lighting conditions; tests robustness to extreme motion blur.
    - **Emergency Rescue**: Searching for emergency items such as first-aid kits and fire extinguishers; 6 lighting conditions + smoke; tests detection under severe camera shake and visual occlusion.
    - **Design Motivation**: The three scenarios challenge pose estimation under fine motion, high-speed motion, and emergency conditions respectively, comprehensively exposing model weaknesses.

2. **8 Lighting Conditions**: Full coverage from standard to extreme → systematic analysis of illumination effects.

    - Standard: normal brightness, medium brightness, high brightness.
    - Extreme: low light, dynamic headlamp lighting, flashlight, rotating warning light, green emergency exit light.
    - Smoke is additionally introduced in selected scenarios to simulate fire or industrial environments.

3. **High-Precision 6D Annotation Pipeline**: OptiTrack + Aria glasses → sub-millimeter accuracy → ensuring reliable ground truth.

    - Aria glasses capture 30 fps RGB images with synchronized 1000 fps SLAM data.
    - OptiTrack motion capture system tracks reflective markers at 120 fps.
    - The Umeyama method is used for coordinate system alignment, and Kalman filtering corrects SLAM drift.
    - Manual temporal offset calibration compensates for exposure delay.

### Loss & Training

As a dataset contribution, no specific loss function is defined. Evaluation follows standard BOP metrics:
- ADD(-S) recall @ thresholds 0.1d, 0.2d, 0.3d (where $d$ is the object diameter)
- MSSD (Maximum Symmetry-aware Surface Distance) ↓
- MSPD (Maximum Symmetry-aware Projection Distance) ↓

## Key Experimental Results

### Main Results

Performance of three state-of-the-art zero-shot 6D pose estimators on EgoXtreme:

| Scenario + Condition | FoundPose @0.3d | GigaPose @0.3d | PicoPose @0.3d |
|---|---|---|---|
| Sports-Standard | 4.72 | 24.64 | 24.61 |
| Sports-Extreme | 2.42 (-49%) | 19.04 (-23%) | 17.86 (-27%) |
| Maintenance-Standard | 37.61 | 62.77 | 76.84 |
| Maintenance-Extreme | 30.03 (-20%) | 45.52 (-27%) | 64.09 (-17%) |
| Maintenance-Standard+Smoke | 30.00 | 52.86 | 59.87 |
| Maintenance-Extreme+Smoke | 25.63 | 45.11 | 52.30 |
| Rescue-Standard | 12.88 | 46.34 | 67.83 |
| Rescue-Extreme | 0.56 (-96%) | 21.30 (-54%) | 36.23 (-47%) |
| Rescue-Extreme+Smoke | 0.76 | 21.54 | 31.54 |

- **Sports scenarios are the most challenging**: recall does not exceed 25% for any model; FoundPose falls below 5%.
- **Performance collapses under extreme rescue conditions**: PicoPose drops from 67.83% to 36.23% (−31.6 pp).
- FoundPose nearly completely fails under extreme conditions — its sparse feature matching cannot obtain sufficient correspondences under truncation and blur.

### Ablation Study

Effect of image preprocessing (deblurring / dehazing / low-light enhancement) on PicoPose:

| Scenario | No Preprocessing @0.3d | +Deblur | +Dehaze | +Low-light | +Deblur+Dehaze |
|---|---|---|---|---|---|
| Maintenance | 63.32 | 57.55 (-5.8) | 57.71 (-5.6) | 58.08 (-5.2) | 53.28 (-10.0) |
| Rescue | 51.70 | 45.51 (-6.2) | 37.74 (-14.0) | 46.75 (-5.0) | 43.65 (-8.1) |
| Sports | 23.02 | 22.24 (-0.8) | 22.05 (-1.0) | — | 20.99 (-2.0) |

**Image restoration not only fails to help, but actively degrades pose estimation accuracy.** Dehazing is particularly detrimental in rescue scenarios (@0.1d: only 4.74%), as non-uniform smoke causes restoration to introduce severe artifacts.

Effect of tracking strategies on sports scenarios (GigaPose):

| Strategy | Baseball Bat @0.3d | Tennis Racket @0.3d | Golf Club @0.3d |
|---|---|---|---|
| Per-frame | 60.55 | 50.91 | 8.36 |
| Direct temporal | 14.29 (-76%) | 22.77 (-55%) | 3.61 (-57%) |
| Fusion temporal | 60.59 | 49.56 | 13.98 |
| **Hybrid temporal** | **64.46 (+6%)** | **50.55** | **14.35 (+72%)** |

### Key Findings

1. **Across-the-board performance collapse under extreme conditions**: All state-of-the-art models exhibit 20–96% performance degradation under extreme lighting and motion blur.
2. **Image restoration is a trap**: Deblurring, dehazing, and low-light enhancement may appear visually beneficial but consistently degrade downstream pose estimation — challenging the intuition of "restore first, then estimate."
3. **Temporal information is critical**: Naive inter-frame propagation (Direct temporal) worsens performance under high-speed motion, but a confidence-based hybrid strategy (Hybrid temporal) selectively leverages temporal priors and substantially improves performance in dynamic scenarios.
4. **FoundPose is the most fragile**: Its sparse feature matching fails readily under truncation and blur; insufficient correspondences cause PnP-RANSAC to produce no output.
5. **Different degradation types are orthogonal**: Illumination variation causes distribution shift, while motion blur causes feature loss — distinct strategies are required to address each.

## Highlights & Insights

1. **Fills a critical evaluation gap**: Existing datasets cannot represent the extreme conditions of real-world smart glasses usage; EgoXtreme is the first large-scale benchmark to systematically cover these challenges.
2. **The finding that image restoration is unhelpful is highly valuable**: It challenges the common assumption of "enhance first, then recognize" and reveals that artifacts introduced by restoration methods are harmful to downstream tasks.
3. **Data scale and quality are both achieved**: 1.3 million frames, 775.5 minutes, 15 participants, with sub-millimeter precision annotations via OptiTrack.
4. **Clever experimental design**: Ground-truth bounding boxes are used to decouple detection error, focusing evaluation on the robustness of pose estimation itself.
5. **Importance of temporal modeling**: The success of the Hybrid temporal strategy suggests that future work should develop video-based pose estimation methods.

## Limitations & Future Work

1. **Indoor only**: Reliance on the OptiTrack motion capture system prevents outdoor data collection.
2. **No hand annotations**: Hand annotation under extreme motion blur is difficult, yet hand-object interaction is important for understanding egocentric manipulation.
3. **13 objects**: Fewer object categories compared to YCB-V (21) or HOT3D (33).
4. **Limited scenarios**: Only three scenarios; other common smart glasses applications such as cooking and medical assistance are absent.
5. **Depth modality unexplored**: Only RGB-only methods are evaluated; although smart glasses typically lack depth sensors, RGB-D methods would still be meaningful as an upper-bound reference.

## Related Work & Insights

- **HOT3D**: The largest egocentric object tracking benchmark, but with mild lighting and slow motion — complementary to EgoXtreme.
- **H2O**: An egocentric hand-object interaction dataset, also lacking extreme conditions.
- **GigaPose / FoundPose / PicoPose**: Three representative zero-shot pose estimators, covering coarse-to-fine two-stage and end-to-end approaches.
- **DarkIR**: A recent method jointly addressing low light, noise, and blur, yet still ineffective for downstream pose estimation.
- Takeaway: Robustness in pose estimation cannot rely on pixel-level enhancement; fundamental improvements at the levels of feature representation and temporal modeling are required.

## Rating

- Novelty: ⭐⭐⭐⭐ — Primarily a dataset contribution; the scenario design is creative but methodological contributions are limited.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three SOTA models, multiple condition combinations, preprocessing analysis, and tracking strategy comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with detailed data descriptions.
- Value: ⭐⭐⭐⭐ — Significant for advancing robust pose estimation research, though follow-up method development by the community is needed.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[AAAI 2026\] Lifelong Domain Adaptive 3D Human Pose Estimation](../../AAAI2026/video_understanding/lifelong_domain_adaptive_3d_human_pose_estimation.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)
- [\[ICCV 2025\] UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions](../../ICCV2025/video_understanding/umdatrack_unified_multi-domain_adaptive_tracking_under_adverse_weather_condition.md)

<!-- RELATED:END -->
