---
title: >-
  [Paper Note] SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations
description: >-
  [CVPR 2026][3D Vision][Video Dataset] This paper presents SceneScribe-1M — a large-scale multimodal video dataset comprising one million in-the-wild videos spanning over 4,000 hours…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Video Dataset"
  - "Geometric Annotation"
  - "Semantic Annotation"
  - "World Foundation Model"
  - "Depth Estimation"
date: 2026-05-08
content_hash: b77e65e2ef48b796
---

# SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations

**Conference**: CVPR 2026
**arXiv**: [2604.07990](https://arxiv.org/abs/2604.07990)
**Code**: [https://wangyunnan.github.io/SceneScribe-1M](https://wangyunnan.github.io/SceneScribe-1M)
**Area**: 3D Vision / Video Understanding
**Keywords**: Video Dataset, Geometric Annotation, Semantic Annotation, World Foundation Model, Depth Estimation

## TL;DR

This paper presents SceneScribe-1M — a large-scale multimodal video dataset comprising one million in-the-wild videos spanning over 4,000 hours, with comprehensive annotations including structured text descriptions, accurate camera parameters, temporally consistent depth maps, and 3D point trajectories. The dataset serves as a unified resource for 3D geometric perception and video generation tasks.

## Background & Motivation

1. **Background**: The integration of 3D geometric perception and video synthesis is central to building World Foundation Models (WFMs). Existing datasets focus either on 3D understanding (e.g., RE10K, CO3Dv2) or video generation (e.g., Panda-70M, Koala-36M), leaving a gap for a unified resource supporting both directions.
2. **Limitations of Prior Work**: (A) 3D perception datasets: synthetic data suffers from domain gaps, while real-data annotations are limited by computational cost and the constraints of SfM/SLAM, with dynamic scene annotations remaining small-scale; (B) Video generation datasets: provide rich semantic annotations but lack geometric annotations; (C) Concurrent works such as Sekai (~400 hours) and SpatialVID (lacking 3D point trajectories) are insufficient in either scale or annotation completeness.
3. **Key Challenge**: WFMs require simultaneous 3D geometric understanding and video generation capabilities, yet the two types of tasks demand data that differ substantially in scale and annotation type.
4. **Goal**: To construct a sufficiently large and comprehensively annotated video dataset that jointly supports 3D tasks — including depth estimation, scene reconstruction, and dynamic point tracking — as well as text/pose-conditioned video generation.
5. **Key Insight**: Leveraging powerful proprietary models (Qwen2.5-VL-72B for semantics, MegaSaM for geometry, TAPIP3D for point trajectories) to perform large-scale parallel annotation on 1,000+ GPUs.
6. **Core Idea**: A carefully designed filtering and multi-model annotation pipeline that simultaneously acquires structured text descriptions, camera poses, temporally consistent depth maps, dynamic masks, and 3D point trajectories from one million open-domain videos.

## Method

### Overall Architecture

The data pipeline consists of three stages: (1) **Collection** — aggregating large-scale video sources from HD-VILA-100M, Panda-70M, Koala-36M, and Pexels; (2) **Preprocessing** — quality filtering (resolution >1080p, FPS ≥10, duration 5s–1min) + content screening (evaluating six dimensions with Qwen2.5-VL-72B) + temporal segmentation via TransNetV2; (3) **Annotation** — three specialized models independently annotate text descriptions, geometric information, and 3D point trajectories. The final output is one million fully annotated video clips, along with a static subset, SceneScribe-MVS, filtered via multi-view reprojection.

### Key Designs

1. **Multi-Dimensional Quality Filtering and Content Screening**:

    - Function: Ensure diversity and motion richness of video content.
    - Mechanism: Hard-parameter filtering (resolution, frame rate, duration) is followed by Qwen2.5-VL-72B serving as an automated evaluator. Six-dimensional QA templates are designed to assess video quality — clips are excluded if they exhibit unknown motion intensity, watermarks, lens distortion, or strong lighting interference. Non-contiguous videos are segmented at shot boundaries via TransNetV2, and the resulting clips are filtered again.
    - Design Motivation: Hard parameters alone cannot guarantee content quality. Using MLLMs for content screening is substantially more efficient than manual annotation and provides broader coverage.

2. **Tri-Model Joint Geometry and Semantics Annotation Pipeline**:

    - Function: Simultaneously generate text descriptions, camera parameters, depth maps, dynamic masks, and 3D point trajectories for each video.
    - Mechanism: (A) Qwen2.5-VL-72B generates structured scene descriptions (scene setting, subjects, actions); (B) MegaSaM jointly estimates optical flow and uncertainty to obtain motion probability maps, performs camera tracking via an improved DROID-SLAM with monocular depth priors, and optimizes temporally consistent high-resolution depth maps; (C) TAPIP3D projects 2D features into 3D world space using MegaSaM's depth and poses to produce robust long-term 3D point trajectories. The pipeline runs as parallel inference on 1,000+ H20 GPUs, consuming approximately 150k GPU hours in total.
    - Design Motivation: No single model can complete all annotations simultaneously. MegaSaM outperforms DROID-SLAM and VGGT under dynamic scenes and limited parallax; TAPIP3D supplements the dynamic point tracking capability absent in MegaSaM.

3. **Motion-Decoupled Sampling for the SceneScribe-MVS Subset**:

    - Function: Construct a subset suited for multi-view tasks (which favor static objects) while preserving camera motion diversity.
    - Mechanism: Multi-view reprojection (Algorithm 1) computes geometric and photometric consistency errors $e_{2d}, e_{3d}, e_{rgb}$, from which a motion mask $M_{motion}$ is derived. Two object motion scores are defined: (1) $s_1$, aggregated from the motion mask; (2) $s_2$, the average motion distance based on point trajectories. Thresholds $\tau_4, \tau_5$ are applied to select static scenes. Crucially, this approach decouples camera motion from object motion — statistics confirm that the camera motion distribution of the MVS subset closely mirrors that of the full dataset.
    - Design Motivation: Multi-view 3D reconstruction requires static scenes but should not constrain camera motion. Filtering by overall motion magnitude would inadvertently exclude clips with rich camera motion.

### Loss & Training

This is a dataset paper and does not involve novel model training. Downstream validation experiments adopt the default training configurations of the respective task models.

## Key Experimental Results

### Main Results

Monocular Depth Estimation (MoGe model, averaged over 8 benchmarks):

| Setting | Rel ↓ | δ₁ ↑ |
|---|---|---|
| MoGe (w/o SceneScribe) - Scale-inv | 6.17 | 93.8 |
| MoGe (w SceneScribe) - Scale-inv | **6.14** | **94.0** |
| MoGe (w/o SceneScribe) - Affine-inv | 4.72 | 95.8 |
| MoGe (w SceneScribe) - Affine-inv | **4.68** | **95.9** |

Scene Reconstruction — VGGT (CO3Dv2 + ETH3D):

| Method | Pose AUC30 ↑ | Pose AUC15 ↑ |
|---|---|---|
| VGGT (w/o SceneScribe) | 89.5 | 83.4 |
| VGGT (w SceneScribe) | **89.9** | **83.8** |

4D Reconstruction — MonST3R (Sintel):

| Method | ATE ↓ | RPE trans ↓ | RPE rot ↓ |
|---|---|---|---|
| MonST3R (w/o SceneScribe) | 0.108 | 0.042 | 0.732 |
| MonST3R (w SceneScribe) | **0.099** | **0.038** | **0.685** |

Video Generation — AC3D (RealEstate10K):

| Method | TransErr ↓ | RotErr ↓ | FID ↓ | FVD ↓ | CLIP ↑ |
|---|---|---|---|---|---|
| AC3D (w/o SceneScribe) | 0.374 | 0.039 | 1.27 | 38.20 | 28.62 |
| AC3D (w SceneScribe) | **0.318** | **0.026** | **1.19** | **35.15** | **29.98** |

### Ablation Study

2D/3D Point Tracking:

| Task | Method | Key Metric | Gain |
|---|---|---|---|
| 2D (CoTracker3) | w/ SceneScribe | TAP-Vid δ_avg^vis avg. 77.4 | +0.8 |
| 3D (SpatialTrackerV2) | w/ SceneScribe | TAPVid-3D AJ avg. 23.5 | +0.25 |

### Key Findings

- SceneScribe-1M yields consistent performance improvements across all downstream tasks (depth estimation, scene reconstruction, 4D reconstruction, point tracking, and video generation), validating annotation quality.
- Video generation benefits the most (TransErr reduced from 0.374 to 0.318, a 15% decrease), indicating that accurate camera parameters are particularly critical for controllable video generation.
- MonST3R shows substantial ATE improvement (0.108→0.099), demonstrating that large-scale real dynamic scene data effectively bridges the domain gap introduced by synthetic training data.
- Gains on MoGe are modest — its original training set TartanAir already provides precise annotations — yet SceneScribe's real-world data still offers complementary value.
- Motion-decoupled sampling is effective: the camera motion distribution of SceneScribe-MVS closely matches that of the full dataset, while dynamic objects are significantly reduced.

## Highlights & Insights

- **Annotation completeness as the core differentiator**: Simultaneously providing text descriptions, camera poses, depth maps, dynamic masks, and 3D point trajectories is unique among comparable datasets, enabling a single dataset to serve both 3D perception and video generation.
- **Industrial-scale annotation pipeline**: Parallel annotation on 1,000+ GPUs consuming 150k GPU hours demonstrates a mature methodology for large-scale AI data engineering. The engineering contribution of modifying the official MegaSaM codebase to enable multi-machine parallel inference is noteworthy.
- **Motion decoupling**: The approach of distinguishing camera motion from object motion via depth reprojection consistency is elegant and practical, applicable to any scenario requiring separation of static and dynamic components from mixed motion signals.
- At 4,000+ hours, the dataset is approximately seven times larger than the concurrent work Sekai (600+ hours) and includes 3D point trajectories absent in the latter.

## Limitations & Future Work

- Annotation quality is bounded by the capabilities of the employed models — MegaSaM still degrades in sparse feature regions, and TAPIP3D has limited handling of long-term occlusion.
- Depth annotations are in relative scale and lack metric depth, restricting applications that require absolute depth.
- Video sources are predominantly web videos, with limited coverage of domain-specific scenarios such as autonomous driving and robotics.
- The absence of instance-level or panoptic segmentation annotations constrains object-level understanding tasks.
- Potential improvements include: integrating metric depth estimation models (e.g., UniDepth) to provide absolute depth; adding semantic segmentation annotations; and extending data collection to domain-specific videos (autonomous driving, embodied AI).

## Related Work & Insights

- **vs. SpatialVID**: SpatialVID contains two million videos but lacks 3D point trajectories; SceneScribe-1M provides more complete annotations (depth + poses + 3D trajectories + descriptions) across one million videos.
- **vs. Sekai**: Sekai focuses on structured descriptions, depth, and poses at approximately 400 hours; SceneScribe-1M is roughly seven times larger and additionally provides 3D point trajectories.
- **vs. PointOdyssey**: PointOdyssey is a synthetic dataset (159 scenes) with ground-truth depth and trajectories but suffers from domain gaps; SceneScribe-1M uses real videos and, while annotations are not ground truth, far surpasses it in scale and diversity.
- The dataset can serve as a general-purpose pretraining resource across multiple research directions, acting as a catalyst for the development of World Foundation Models.

## Rating

- Novelty: ⭐⭐⭐ Innovation in dataset work lies primarily in annotation completeness and scale; methodological novelty is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across six downstream tasks, though each task is validated with only one model.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, thorough tabular comparisons, and detailed statistical analysis.
- Value: ⭐⭐⭐⭐⭐ Fills the gap for large-scale jointly annotated geometric and semantic video datasets, with significant implications for WFM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)
- [\[CVPR 2026\] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control](versecrafter_dynamic_realistic_video_world_model_with_4d_geometric_control.md)
- [\[CVPR 2026\] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)

</div>

<!-- RELATED:END -->
