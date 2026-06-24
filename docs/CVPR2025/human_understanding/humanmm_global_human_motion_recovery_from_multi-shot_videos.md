---
title: >-
  [Paper Note] HumanMM: Global Human Motion Recovery from Multi-shot Videos
description: >-
  [CVPR 2025][Human Understanding][Human Motion Recovery] HumanMM proposes the first framework to recover 3D human motion in the global coordinate system from multi-shot videos. By integrating a shot transition detector, enhanced SLAM, calibration-based orientation alignment, and a motion integrator, it achieves continuous motion reconstruction across shot transitions.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Human Motion Recovery"
  - "Multi-shot Videos"
  - "Global Coordinate System"
  - "Shot Transition Alignment"
  - "SLAM"
date: 2026-05-08
content_hash: 6a9690f8284a2b72
---

# HumanMM: Global Human Motion Recovery from Multi-shot Videos

**Conference**: CVPR 2025  
**arXiv**: [2503.07597](https://arxiv.org/abs/2503.07597)  
**Code**: Yes (provided on project page)  
**Area**: Human Understanding  
**Keywords**: Human Motion Recovery, Multi-shot Videos, Global Coordinate System, Shot Transition Alignment, SLAM

## TL;DR
HumanMM proposes the first framework to recover 3D human motion in the global coordinate system from multi-shot videos. By integrating a shot transition detector, enhanced SLAM, calibration-based orientation alignment, and a motion integrator, it achieves continuous motion reconstruction across shot transitions.

## Background & Motivation

**Background**: Significant progress has been made in 3D human motion recovery (HMR), with methods like HMR2.0 performing exceptionally in the camera coordinate system. Methods like WHAM and GVHMR further realize global motion recovery by integrating SLAM to estimate camera parameters. However, these methods all focus on single-shot videos.

**Limitations of Prior Work**: A large number of online videos (sports broadcasts, talk shows, concerts, etc.) are shot with multiple cameras, containing frequent shot transitions. Segmenting multi-shot videos into single shots drastically reduces sequence lengths (the longest segments in current datasets are less than 20 seconds), which is highly detrimental to tasks requiring long sequences (such as long-term motion generation). Directly processing multi-shot videos faces two fundamental challenges.

**Key Challenge**: While human motion in multi-shot videos is physically continuous, the abrupt viewpoint changes caused by shot transitions make motion estimation in the camera coordinate system discontinuous at transition boundaries. It is crucial to address the contradiction between "motion continuity" and "viewpoint discontinuity."

**Goal**: (1) How to align human orientation and motion in the global coordinate system during shot transitions? (2) How to reconstruct accurate human motion in the global coordinate system (resolving issues like foot sliding and temporal inconsistency)?

**Key Insight**: The authors build on a key observation: human motion in multi-shot videos is typically continuous during shot transitions, with only the camera viewpoint changing. Therefore, by estimating the relative camera rotation between transition frames, motion from different shots can be aligned into a unified global coordinate system.

**Core Idea**: Achieve global motion recovery from multi-shot videos using shot detection, enhanced SLAM, 2D keypoint-based camera calibration for orientation alignment, and a cross-shot Transformer for pose smoothing.

## Method

### Overall Architecture
The input is a long video $\mathbf{V}=\{I_t\}_{t=1}^T$ containing multiple shot transitions, and the output is the SMPL motion parameters in the global coordinate system. The pipeline consists of five steps: (1) extracting motion features and detecting shot transition frames; (2) estimating camera poses for each single-shot segment using Masked LEAP-VO, and recovering initial motion using GVHMR; (3) aligning cross-shot human orientation via camera calibration; (4) smoothing cross-shot human poses using the ms-HMR Transformer; (5) recovering trajectories and eliminating foot sliding via a motion integrator (BiLSTM + trajectory optimizer).

### Key Designs

1. **Shot Transition Detector**:

    - **Function**: Accurately recognize shot transition frames in the video.
    - **Mechanism**: Three complementary modules are used in series: (1) SceneDetect to detect obvious background changes; (2) bounding box tracking to detect sudden changes in person scale (calculating the IoU of bounding boxes in adjacent frames, identifying a transition if below a threshold); (3) human keypoint tracking to detect fine-grained pose/orientation changes (calculating the IoU of corresponding keypoints in adjacent frames). The three modules work together serially to cover various types of shot transitions from coarse to fine.
    - **Design Motivation**: A single detector cannot cover all types of transitions—SceneDetect cannot handle viewpoint switches with similar backgrounds, and bounding boxes cannot capture pose changes when the scale remains constant. Therefore, multiple complementary modules are essential.

2. **Masked LEAP-VO + Orientation Alignment**:

    - **Function**: Accurately estimate camera trajectories for each shot and align human orientation across shots.
    - **Mechanism**: Improved based on LEAP-VO: first, SAM is used to generate human masks, and feature points inside the masks are set as invisible to exclude the interference of dynamic humans on bundle adjustment ("Masked LEAP-VO"). For cross-shot orientation alignment, an Orientation Alignment Module (OAM) is proposed: based on **Assumption 1** (human orientation and displacement are continuous in the global coordinate system during shot transitions), the orientation alignment problem is formulated as estimating the relative camera rotation $\mathbf{R}_{\delta_{cam}}$. Specifically, 2D keypoints on both sides of the transition frame are extracted, matching points are filtered using RANSAC, and the relative rotation is solved via the SVD of the essential matrix $\mathbf{E}=[\mathbf{T}]_\times \mathbf{R}$.
    - **Design Motivation**: Methods like DROID-SLAM become inaccurate when human occlusion is severe because too few feature points remain after masking. LEAP-VO utilizes CoTracker for long-range feature tracking, retaining sufficient information even after masking. For orientation alignment, instead of directly masking the human, human keypoints are used as explicit feature matches because the human is the only reliable correspondence across shots at that moment.

3. **ms-HMR + Motion Integrator**:

    - **Function**: Smooth human poses across shots, recover trajectories, and eliminate foot sliding.
    - **Mechanism**: ms-HMR is a Transformer encoder whose input is the initial motion parameters $\{\theta_t\}_{t=1}^T$ across all shots (including shot index positional encodings) and whose output is the refined motion parameters $\{\phi_t\}_{t=1}^T$. During training, random rotational noise (0-1 rad) is added to the root pose to simulate inaccuracies in shot transitions. The motion integrator uses a bidirectional LSTM to predict foot-ground contact probabilities and root velocities, and then uses a trajectory optimizer (extended from WHAM) to eliminate foot sliding.
    - **Design Motivation**: Shot transitions cause partial occlusions, and the visible body parts in different shots are complementary. The global attention mechanism of the Transformer can exploit this complementary information across shots. The training strategy of adding noise makes the model robust to inaccuracies introduced by shot transitions.

### Loss & Training
The ms-HMR, trajectory predictor, and foot-sliding optimizer are trained for 80 epochs on the AMASS, 3DPW, Human3.6M, and BEDLAM datasets. Contact probability and velocity are supervised using MSE loss. Random rotational noise and body pose noise are added during training to simulate shot transition errors.

## Key Experimental Results

### Main Results (ms-Motion Dataset, 2-shot Setting)

| Dataset | Method | PA-MPJPE↓ | WA-MPJPE↓ | RTE↓ | ROE↓ | Foot Sliding↓ |
|--------|------|-----------|-----------|------|------|--------------|
| ms-AIST | GVHMR | 60.72 | 231.36 | 6.20 | 96.58 | 7.65 |
| ms-AIST | WHAM | 65.34 | 336.82 | 4.39 | 84.48 | 2.75 |
| ms-AIST | **Ours** | **36.82** | **121.35** | **2.56** | **69.23** | **2.66** |
| ms-H3.6M | GVHMR | 64.63 | 254.30 | 6.94 | 81.93 | 8.80 |
| ms-H3.6M | **Ours** | **40.52** | **132.13** | **3.65** | **53.39** | **4.17** |

### Performance Changes under Different Shot Numbers

| Setting | PA-MPJPE↓ (ms-AIST) | WA-MPJPE↓ | ROE↓ |
|------|---------------------|-----------|------|
| 2-shot | 36.82 | 121.35 | 69.23 |
| 3-shot | 38.52 | 141.38 | 67.71 |
| 4-shot | 39.63 | 161.52 | 70.31 |

### Key Findings
- HumanMM reduces PA-MPJPE by 39% (on ms-AIST) and ROE by 28% compared to GVHMR, demonstrating the significant effectiveness of cross-shot alignment and pose smoothing.
- As the number of shots increases (2→3→4), PA-MPJPE only increases by 7.6%, showing the good scalability of the method to larger shot numbers.
- RTE (trajectory error) is significantly better than all baselines, validating the effectiveness of Masked LEAP-VO and the trajectory optimizer.
- SLAHMR performs the worst across all metrics, indicating that directly applying existing single-shot methods to multi-shot scenarios is completely unfeasible.
- The foot-sliding metric is close to WHAM, proving the motion integrator's effectiveness in eliminating sliding.

## Highlights & Insights
- **Pioneering Problem Definition**: This work is the first to formally define and address the global coordinate system HMR problem from multi-shot videos, filling an important gap. A large portion of online videos are multi-shot, and solving this problem can significantly expand the usable scale of motion datasets.
- **Clever Orientation Alignment Design**: Formulating cross-shot orientation alignment as relative camera rotation estimation, it leverages human 2D keypoints as cross-shot correspondences for camera calibration. While humans are dynamic objects to be excluded in normal SLAM, they serve as the only reliable correspondence for cross-shot alignment here.
- **ms-Motion Dataset Construction**: Utilizing multi-view data from AIST and H3.6M to construct a multi-shot evaluation set, this approach is ingenious and provides a standardized benchmark for future study.

## Limitations & Future Work
- The assumption of continuous human orientation across shot transitions may violate in some fast-action scenarios.
- The ms-Motion dataset consists of synthetic multi-shot videos (concatenated from multi-view data), which still has a distribution gap compared to real-world online multi-shot videos.
- Shot transition detection relies on manually tuned IoU thresholds, offering limited robustness.
- Future Directions: Integrating pre-trained vision models to enhance shot transition detection, supporting multi-person multi-shot scenarios, and directly training and evaluating on real-world online videos.

## Related Work & Insights
- **vs GVHMR**: GVHMR represents the current state-of-the-art for single-shot global-coordinate HMR. However, when applied directly to multi-shot videos, it causes orientation jumps and trajectory fragmentation due to not handling shot transitions. HumanMM introduces alignment and smoothing modules on top of it.
- **vs WHAM**: WHAM uses DROID-SLAM for camera estimation, which suffers from insufficient features when human occlusion is severe; HumanMM utilizes Masked LEAP-VO to preserve more features through long-range tracking.
- **vs Pavlakos et al.**: This is the only previous multi-shot work, but it solely addresses close-up to far-shot transitions in the camera coordinate system, failing to handle the global coordinate system or arbitrary shot transitions.
- It can serve as a data acquisition tool for motion generation tasks, automatically extracting long-sequence motion data from massive online videos.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally defines and solves the global HMR problem from multi-shot videos for the first time, which is highly significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively evaluated on a self-built benchmark, though qualitative results on real-world online videos are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and complete methodology pipeline, though the system is relatively complex.
- Value: ⭐⭐⭐⭐⭐ Fills an important gap, directly expands the scale of motion data, and strongly drives downstream tasks such as motion generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)
- [\[CVPR 2025\] RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance](reperformer_immersive_human-centric_volumetric_videos_from_playback_to_photoreal.md)
- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[CVPR 2026\] Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining](../../CVPR2026/human_understanding/mocap-2-to-3_multi-view_lifting_for_monocular_motion_recovery_with_2d_pretrainin.md)
- [\[CVPR 2025\] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation](sonic_shifting_focus_to_global_audio_perception_in_portrait_animation.md)

</div>

<!-- RELATED:END -->
