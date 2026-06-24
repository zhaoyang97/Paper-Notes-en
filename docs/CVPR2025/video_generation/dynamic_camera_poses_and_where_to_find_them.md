---
title: >-
  [Paper Note] Dynamic Camera Poses and Where to Find Them
description: >-
  [CVPR 2025][Video Generation][Dynamic camera poses] Proposes DynPose-100K—a large-scale dataset containing 100K dynamic internet videos and their camera pose annotations, achieved through a video filtering pipeline combining specialist models with a VLM, and a pose estimation pipeline integrating state-of-the-art point tracking, dynamic masking, and global BA.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Dynamic camera poses"
  - "large-scale dataset"
  - "video filtering"
  - "structure from motion"
  - "point tracking"
date: 2026-05-08
content_hash: abce1e616addc3de
---

# Dynamic Camera Poses and Where to Find Them

**Conference**: CVPR 2025  
**arXiv**: [2504.17788](https://arxiv.org/abs/2504.17788)  
**Code**: [https://research.nvidia.com/labs/dir/dynpose-100k](https://research.nvidia.com/labs/dir/dynpose-100k)  
**Area**: Video Generation  
**Keywords**: Dynamic camera poses, large-scale dataset, video filtering, structure from motion, point tracking

## TL;DR

Proposes DynPose-100K—a large-scale dataset containing 100K dynamic internet videos and their camera pose annotations, achieved through a video filtering pipeline combining specialist models with a VLM, and a pose estimation pipeline integrating state-of-the-art point tracking, dynamic masking, and global BA.

## Background & Motivation

Large-scale annotation of camera poses on dynamic internet videos is crucial for fields such as video generation, view synthesis, and robotics, but faces two major challenges:

1. **The vast majority of internet videos are unsuitable for pose estimation** — In 1,000 randomly selected Panda-70M videos, only 9% meet the requirements for pose estimation. Reasons include cartoon/synthetic content, heavy post-processing, lack of clear reference frames, static scenes, and blurry backgrounds.
2. **Pose estimation for dynamic videos is highly challenging** — Moving objects can occlude static scenes, causing correspondences in traditional SfM to fail; variations in scene appearance increase the difficulty of matching.

Existing datasets are either synthetic (small-scale, e.g., < 500 videos) or restricted to specific domains (such as autonomous driving, kitchen scenes, or pet-centric captures), lacking large-scale, diverse, real-world dynamic video datasets.

## Method

### Overall Architecture

The construction of DynPose-100K consists of two main stages: (1) **Candidate video filtering** — Screening approximately 137K videos suitable for pose estimation from 3.2 million Panda-70M videos; (2) **Dynamic pose estimation** — Estimating high-quality camera poses for the filtered videos, ultimately retaining 100K videos (those with >80% registered frames).

### Key Designs

1. **Hybrid Video Filtering Pipeline (Specialist + VLM Filtering)**:
    - **Function**: Automatically filters dynamic videos suitable for camera pose estimation from massive internet video collections.
    - **Mechanism**: Defines three categories of filtering criteria — C1 (real-world + high-quality), C2 (feasible for pose estimation), and C3 (dynamic camera + scene). Six specialist models are used to handle common issues: ① Hands23 classifier to remove cartoon/static scenes; ② distortion detection model to remove non-perspective distortions; ③ focal length projection to remove zoom/telephoto videos; ④ videos with excessively large dynamic masks (insufficient static points); ⑤ optical flow detection for shot cuts and static videos; ⑥ point tracking to detect track deaths or overly stable tracking. Then, GPT-4o mini acts as a general VLM to answer 8 questions covering all criteria, handling long-tail issues that specialist models cannot capture (e.g., post-edited text overlays).
    - **Design Motivation**: A single filter is insufficient to cover all issue types. Specialist models precisely handle high-frequency issues, while the VLM flexibly handles long-tail issues. Combining them significantly outperforms using either in isolation.

2. **Dynamic Pose Estimation Pipeline**:
    - **Function**: Estimates accurate camera poses (intrinsics + extrinsics) in dynamic scenes.
    - **Mechanism**: A three-step workflow — ① **Dynamic masking**: Integrates four complementary methods (OneFormer semantic segmentation, Hands23 hand-object interaction segmentation, RoDynRF motion segmentation based on Sampson error, and SAM2 mask propagation); ② **Point tracking**: Uses BootsTAP to track a dense point grid within a sliding window, providing long-term dense correspondences; ③ **Global Bundle Adjustment**: Uses Theia-SfM for global BA, with inputs being static trajectory correspondences excluding the dynamic masked regions.
    - **Design Motivation**: Compared with ParticleSfM, this pipeline upgrades the masking method (incorporating more complementary components) and correspondence estimation (upgrading from optical flow propagation to long-term point tracking), significantly reducing errors on dynamic internet videos.

3. **Evaluation Framework**:
    - **Function**: Evaluates pose quality on internet videos lacking ground-truth poses.
    - **Mechanism**: Dual evaluation — ① Designs the Lightspeed synthetic benchmark (a ray-traced RC car scene) with ground-truth poses for direct comparison; ② Annotates 10K precise correspondence points on Panda-Test to indirectly evaluate reprojection error via Sampson error.
    - **Design Motivation**: Dynamic internet videos do not have ground-truth poses, which necessitates a carefully planned evaluation protocol; the Lightspeed scene provides a combination of dynamics, diversity, and ground-truth camera poses.

### Loss & Training

DynPose-100K itself is a dataset construction project and does not involve training. However, the authors demonstrate the training value of the dataset by fine-tuning DUSt3R with 2K videos from DynPose-100K, achieving lower average error on Panda-Test than MonST3R trained on synthetic data.

## Key Experimental Results

### Main Results (Pose Estimation Quality)

| Method | Lightspeed ATE↓ | Lightspeed RPE Rot↓ | Panda-Test <5px↑ | Panda-Test Mean↓ |
|------|------|----------|------|------|
| COLMAP | 0.388m | 2.03° | 51.1% | 27.5px |
| COLMAP+Mask | 0.323m | 1.64° | 47.8% | 30.1px |
| ParticleSfM | 0.185m | 2.99° | 70.0% | 12.5px |
| DROID-SLAM | 0.198m | 1.75° | 57.8% | 11.0px |
| MonST3R | 0.149m | 1.21° | 55.6% | 9.86px |
| **Ours** | **0.072m** | **1.31°** | **72.2%** | **5.76px** |

### Filtering Performance (Panda-Test)

| Filtering Method | Precision at DynPose-100K Threshold |
|------|---------|
| CamCo (reconstructed points) | ~0.35 |
| GPT-4o mini (binary) | ~0.20 |
| GPT-4o mini (score) | ~0.25 |
| Hands23 alone | ~0.15 |
| **Ours (all combined)** | **0.78** |

### Key Findings

- Every component in the filtering pipeline contributes: incrementally adding Flow→Tracking→Masking→Focal→Distort→VLM to Hands23 continuously improves the PR curve.
- On Lightspeed, the proposed method reduces trajectory error by 50% (on all videos) and 90% (on the subset of videos where all methods succeeded) compared to all other methods.
- The dataset video lengths are primarily concentrated between 4 and 10 seconds, a range that offers sufficient camera motion and rich dynamic content.
- Fine-tuning DUSt3R with only 2K videos/140K frames achieves better performance than MonST3R (trained on 1.3 million frames of synthetic data), proving the efficiency advantage of real-world data.

## Highlights & Insights

- **A model of systems engineering**: Deconstructs dataset construction into two independent sub-problems—video filtering and pose estimation—systematically combining state-of-the-art methods for each.
- **The combination of specialist models + general VLM is highly practical**: This paradigm can be widely applied to various data cleaning/filtering scenarios.
- Scale (100K videos) and diversity (covering humans, vehicles, animals, indoor/outdoor scenes, etc.) far exceed existing dynamic pose datasets.
- The dataset is fully open-source, whereas competitors CamCo and B-Timer are not publicly available.

## Limitations & Future Work

- Video segments are relatively short (4-10 seconds); longer videos (e.g., minute-level) may require different processing strategies.
- The filtering pipeline requires deploying multiple specialist models, resulting in high engineering costs.
- Pose estimation is still based on classic SfM; future work could explore end-to-end learning methods.
- Scene-level 3D reconstruction quality evaluation is not yet provided.

## Related Work & Insights

- Improvement relationship with ParticleSfM: Upgrades tracking (BootsTAP replaces optical flow) and masking (four complementary masks replace a single method).
- Comparison with MonST3R/DROID-SLAM: Although learning-based methods can register all frames, their accuracy is inferior to classic SfM pipelines.
- Insight: In dynamic scene understanding, the efficiency of "filter first, then process" is significantly higher than attempting to "process everything."

## Rating
- Novelty: ⭐⭐⭐ Mostly system-level engineering innovation; individual components are existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, including filtering evaluation, synthetic benchmark comparison, real-world video evaluation, and downstream application.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-justified design choices.
- Value: ⭐⭐⭐⭐⭐ Fills the gap for large-scale dynamic video pose datasets, posing a major impact on downstream tasks like video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](../../ICCV2025/video_generation/free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] ReCapture: Generative Video Camera Controls for User-Provided Videos Using Masked Video Fine-Tuning](recapture_generative_video_camera_controls_for_user-provided_videos_using_masked.md)
- [\[CVPR 2025\] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control](gen3c_3d-informed_world-consistent_video_generation_with_precise_camera_control.md)
- [\[ICML 2026\] Physics in 2-Steps: Locking Motion Priors Before Visual Refinement Erases Them](../../ICML2026/video_generation/physics_in_2-steps_locking_motion_priors_before_visual_refinement_erases_them.md)

</div>

<!-- RELATED:END -->
