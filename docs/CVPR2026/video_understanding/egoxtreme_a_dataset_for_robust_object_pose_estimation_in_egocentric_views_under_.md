---
title: >-
  [Paper Note] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions
description: >-
  [CVPR 2026][Video Understanding][Egocentric View] Ours proposes EgoXtreme, the first large-scale 6D object pose estimation benchmark for egocentric views under extreme conditions. It covers three real-world challenges—severe motion blur, dynamic lighting, and smoke occlusion—revealing significant failures of current SOTA pose estimators in these environments.
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Egocentric View"
  - "6D Pose Estimation"
  - "Extreme Conditions"
  - "Smart Glasses"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: e1d7f18ca5c38d5b
---

# EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions

**Conference**: CVPR 2026  
**arXiv**: [2603.25135](https://arxiv.org/abs/2603.25135)  
**Code**: [https://taegyoun88.github.io/EgoXtreme/](https://taegyoun88.github.io/EgoXtreme/)  
**Area**: Video Understanding / 6D Pose Estimation  
**Keywords**: Egocentric View, 6D Pose Estimation, Extreme Conditions, Smart Glasses, Benchmark Dataset

## TL;DR

Ours proposes EgoXtreme, the first large-scale 6D object pose estimation benchmark for egocentric views under extreme conditions. It covers three real-world challenges—severe motion blur, dynamic lighting, and smoke occlusion—revealing significant failures of current SOTA pose estimators in these environments.

## Background & Motivation

Smart glasses are becoming essential wearable devices, providing rich contextual understanding in "hands-busy, eyes-on" scenarios. To understand the wearer's environment, 6D object pose estimation in egocentric views is critical.

However, existing 6D pose estimation benchmarks suffer from a significant **reality gap**: standard datasets like YCB-Video, LineMod, and T-LESS are collected in controlled laboratories with stable lighting and slow movements. While H2O and HOT3D introduce egocentric views, they feature mild lighting and lack testing in true extreme environments. Real-world smart glasses usage faces unique challenges: cameras fixed on a constantly moving head produce rapid and frequent motion; close-up views lead to severe truncation; lighting is dynamic and extreme (switching between headlamps, flashes, emergency lights, and exit signs); and visual occluders like smoke or haze may be present.

**Core Problem**: Bridging the gap solely through simulation (synthetic data, frame-averaging for blur) is insufficient—**a real-world extreme condition benchmark is required to drive the development of robust models**.

## Method

### Overall Architecture

EgoXtreme is a dataset-driven work organized around three pillars: (1) **Three Challenge Scenarios**—decomposing "extreme" into industrial maintenance, sports, and emergency response tasks, each targeting a specific perceptual degradation; (2) **8 Systematic Lighting Conditions**—spreading lighting across a spectrum from normal to extreme to allow for isolated variable analysis; (3) **High-Precision 6D Annotation Pipeline**—utilizing a dual-system setup with Aria glasses and OptiTrack motion capture to ensure ground truth (GT) accuracy even in extreme imagery. A comprehensive benchmark evaluation of current SOTA zero-shot pose estimators is conducted atop this dataset to reveal their failure modes.

### Key Designs

**1. Three Challenge Scenarios: Grounding "Extreme" into Real-World Tasks**

Instead of a generic definition of "extreme," EgoXtreme identifies three specific scenarios targeting typical degradations. The industrial maintenance scenario involves fine-grained operations with tools (hammer, drill, saw) under 6 lighting types plus smoke, testing the model's ability to track objects during precise slow motion and rapid lighting shifts. The sports scenario features rapid swings of rackets and bats (object speeds up to 1.37 m/s) under 4 lighting types, specifically creating extreme motion blur. The emergency response scenario involves searching for first-aid kits and fire extinguishers amidst violent camera shakes and smoke under 6 lighting types, approximating real rescue scenes. Collectively, these scenarios expose the weaknesses of pose estimators across fine motion, high-speed motion, and emergency occlusions.

**2. 8 Lighting Conditions: Systematic Spectrum for Variable Decomposition**

EgoXtreme provides 8 distinct lighting levels to identify exactly which conditions cause model failure. These include three standard levels (normal, middle, high) and five extreme levels: low light (low), dynamic headlamp (head), flashlight (flash), rotating warning lights (warning), and green safety exit lights (green). These extreme settings correspond to specific light sources encountered by smart glasses—headlamps moving with the head, periodic emergency flashes, and strong color shifts from exit signs. Some scenes also overlap with smoke to simulate fire or industrial environments. This allows for direct comparison of object performance while isolating lighting as the sole variable.

**3. High-Precision 6D Annotation Pipeline: OptiTrack and Aria Dual-System Calibration**

Extreme conditions make manual 6D annotation nearly impossible due to blur and smoke. EgoXtreme employs motion capture for GT generation: Aria glasses capture RGB at 30fps and sync 1000fps SLAM data, while the OptiTrack system tracks retroreflective markers at 120fps to obtain sub-millimeter object trajectories. The two coordinate systems are aligned using the Umeyama method, with SLAM drift corrected via Kalman filtering and camera exposure latency compensated through manual time-offset calibration. Since the GT is derived from motion capture rather than the image itself, annotation accuracy remains unaffected even when RGB frames are severely blurred.

### Loss & Training

As a dataset work, there is no specific loss function. Evaluation uses standard BOP metrics:
- ADD(-S) recall @ thresholds 0.1d, 0.2d, 0.3d (d is object diameter)
- MSSD (Maximum Symmetric Surface Distance) ↓
- MSPD (Maximum Symmetric Projection Distance) ↓

## Key Experimental Results

### Main Results

Performance of three SOTA zero-shot 6D pose estimators on EgoXtreme:

| Scenario + Condition | FoundPose @0.3d | GigaPose @0.3d | PicoPose @0.3d |
|:---|:---:|:---:|:---:|
| Sport - Standard | 4.72 | 24.64 | 24.61 |
| Sport - Extreme | 2.42 (-49%) | 19.04 (-23%) | 17.86 (-27%) |
| Maintenance - Standard | 37.61 | 62.77 | 76.84 |
| Maintenance - Extreme | 30.03 (-20%) | 45.52 (-27%) | 64.09 (-17%) |
| Maintenance - Standard + Smoke | 30.00 | 52.86 | 59.87 |
| Maintenance - Extreme + Smoke | 25.63 | 45.11 | 52.30 |
| Emergency - Standard | 12.88 | 46.34 | 67.83 |
| Emergency - Extreme | 0.56 (-96%) | 21.30 (-54%) | 36.23 (-47%) |
| Emergency - Extreme + Smoke | 0.76 | 21.54 | 31.54 |

- **Sport scenario is most challenging**: Recursive recall for all models does not exceed 25%, with FoundPose dropping below 5%.
- **Performance plunges under extreme emergency conditions**: PicoPose drops from 67.83% to 36.23% (a 31.6%p decrease).
- FoundPose fails almost completely under extreme conditions—its sparse feature matching method cannot obtain sufficient correspondences under truncation and blur.

### Ablation Study

Impact of image pre-processing (deblur/dehaze/low-light enhancement) on PicoPose:

| Scenario | No Pre-processing @0.3d | +Deblur | +Dehaze | +Enhance | +Deblur+Dehaze |
|:---|:---:|:---:|:---:|:---:|:---:|
| Maintenance | 63.32 | 57.55 (-5.8) | 57.71 (-5.6) | 58.08 (-5.2) | 53.28 (-10.0) |
| Emergency | 51.70 | 45.51 (-6.2) | 37.74 (-14.0) | 46.75 (-5.0) | 43.65 (-8.1) |
| Sport | 23.02 | 22.24 (-0.8) | 22.05 (-1.0) | — | 20.99 (-2.0) |

**Image restoration is not only ineffective but actually hurts pose estimation accuracy!** Dehazing is particularly detrimental in emergency scenarios (only 4.74% @0.1d) because it introduces severe noise under non-uniform smoke conditions.

Impact of tracking strategies on the Sport scenario (GigaPose):

| Strategy | Baseball Bat @0.3d | Tennis Racket @0.3d | Golf Club @0.3d |
|:---|:---:|:---:|:---:|
| Per-frame | 60.55 | 50.91 | 8.36 |
| Direct temporal | 14.29 (-76%) | 22.77 (-55%) | 3.61 (-57%) |
| Fusion temporal | 60.59 | 49.56 | 13.98 |
| **Hybrid temporal** | **64.46 (+6%)** | **50.55** | **14.35 (+72%)** |

### Key Findings

1. **Catastrophic failure under extreme conditions**: All SOTA models show a 20-96% performance drop under extreme lighting or motion blur.
2. **Image restoration is a trap**: Deblurring, dehazing, and low-light enhancement may look better visually, but they degrade downstream pose estimation—challenging the intuition of "restore then estimate."
3. **Temporal information is key**: Simple frame propagation (Direct temporal) worsens under high-speed motion, but confidence-based Hybrid temporal strategies can selectively use temporal priors to significantly improve dynamic performance.
4. **FoundPose is highly fragile**: Sparse feature matching easily fails under truncation and blur; insufficient correspondence points lead to the PnP-RANSAC producing no output.
5. **Degradation types are orthogonal**: Lighting changes cause distribution shifts, while motion blur causes feature loss—requiring different mitigation strategies.

## Highlights & Insights

1. **Fills a critical evaluation gap**: Existing datasets do not represent the extreme conditions of real smart glasses usage. EgoXtreme is the first large-scale benchmark to systematically cover these challenges.
2. **Valuable discovery on image restoration**: Challenges the common "enhance then recognize" assumption by revealing that artifacts introduced by restoration methods are harmful to downstream tasks.
3. **Scale and quality**: 1.3 million frames, 775.5 minutes, 15 participants, and OptiTrack sub-millimeter precision labels.
4. **Clever experimental design**: Uses GT bounding boxes to decouple detection errors, focusing on the robustness of pose estimation itself.
5. **Importance of temporal modeling**: The success of the Hybrid temporal strategy suggests that future research should focus on video-based pose estimation methods.

## Limitations & Future Work

1. **Indoor only**: Reliance on the OptiTrack system prevents data collection in outdoor environments.
2. **Lack of hand labels**: Hand annotation is difficult under extreme motion blur, yet hand-object interaction is vital for understanding egocentric operations.
3. **13 objects**: Fewer object categories compared to YCB-V (21) or HOT3D (33).
4. **Scenario limitations**: Only three scenarios; lacks other common smart glasses applications like cooking or medical assistance.
5. **Unexplored depth modality**: Evaluates RGB-only methods; though smart glasses often lack depth sensors, RGB-D methods as an upper bound would be meaningful.

## Related Work & Insights

- **HOT3D**: The largest egocentric object tracking benchmark, but features mild lighting and slow motion, making it complementary to EgoXtreme.
- **H2O**: An egocentric hand-object interaction dataset, also lacking extreme conditions.
- **GigaPose / FoundPose / PicoPose**: Representative zero-shot pose estimators utilizing coarse-to-fine or end-to-end approaches.
- **DarkIR**: A recent method for joint low-light, noise, and blur handling, which remains ineffective for downstream pose estimation.
- **Insight**: Pose estimation robustness shouldn't rely on pixel-level enhancement but must be addressed fundamentally through feature representation and temporal modeling.

## Rating

- Novelty: ⭐⭐⭐⭐ — Primarily a dataset contribution; scenario design is innovative, though methodological contribution is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three SOTA models, multiple condition combinations, pre-processing analysis, and tracking strategy comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with detailed data descriptions.
- Value: ⭐⭐⭐⭐ — Significant for driving research in robust pose estimation, contingent on community adoption of the benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DarkShake-DVS: Event-based Human Action Recognition under Low-light and Shaking Camera Conditions](darkshake-dvs_event-based_human_action_recognition_under_low-light_and_shaking_c.md)
- [\[CVPR 2026\] Robust Promptable Video Object Segmentation](robust_promptable_video_object_segmentation.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)
- [\[ECCV 2024\] EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere](../../ECCV2024/video_understanding/egoposer_robust_real-time_egocentric_pose_estimation_from_sparse_and_intermitten.md)
- [\[CVPR 2026\] Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation](beyond_static_frames_temporal_aggregate-and-restore_vision_transformer_for_human.md)

</div>

<!-- RELATED:END -->
