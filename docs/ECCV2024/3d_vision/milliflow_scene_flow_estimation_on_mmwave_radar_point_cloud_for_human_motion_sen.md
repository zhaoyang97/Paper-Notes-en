---
title: >-
  [Paper Note] milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing
description: >-
  [ECCV2024][3D Vision][mmWave Radar] This work proposes milliFlow, the first scene flow estimation method for mmWave radar point clouds. By leveraging multi-scale feature extraction, global aggregation, GRU temporal propagation, and constrained regression, it reduces EPE3D from the sub-optimal 0.107m to 0.046m (centimeter-level accuracy) on a self-collected dataset. It also demonstrates the enhancement effects of scene flow features on downstream tasks…
tags:
  - "ECCV2024"
  - "3D Vision"
  - "mmWave Radar"
  - "Scene Flow"
  - "Human Motion Sensing"
  - "Point Cloud"
  - "Auto-labeling"
date: 2026-05-08
content_hash: be7b4dde5fb30e5f
---

# milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing

**Conference**: ECCV2024  
**arXiv**: [2306.17010](https://arxiv.org/abs/2306.17010)  
**Code**: [https://github.com/Toytiny/milliFlow](https://github.com/Toytiny/milliFlow)  
**Area**: 3D Vision  
**Keywords**: mmWave Radar, Scene Flow, Human Motion Sensing, Point Cloud, Auto-labeling

## TL;DR
This work proposes milliFlow, the first scene flow estimation method for mmWave radar point clouds. By leveraging multi-scale feature extraction, global aggregation, GRU temporal propagation, and constrained regression, it reduces EPE3D from the sub-optimal 0.107m to 0.046m (centimeter-level accuracy) on a self-collected dataset. It also demonstrates the enhancement effects of scene flow features on downstream tasks, including human activity recognition (+7.9%), human body parsing (+3.6%), and human tracking.

## Background & Motivation

**Background**: mmWave radar has become a research hotspot for human sensing due to its privacy preservation (no image capture) and all-weather, all-light operation. Existing works use radar point clouds for activity recognition, gesture recognition, etc., but lack low-level motion representation.

**Limitations of Prior Work**: Radar point clouds are extremely sparse (~100 points representing the entire human body), highly noisy (multipath reflections create ghost points), and lack texture information. Directly transferring LiDAR scene flow methods (such as FlowNet3D, PV-RAFT) yields poor results because they are designed for dense point clouds.

**Key Challenge**: The motion information provided by radar (radial Doppler velocity) has low resolution and lacks tangential components, making it unusable directly as scene flow. Meanwhile, existing autonomous driving scene flow methods assume dense point clouds and rich geometric structures.

**Goal**: (1) How to estimate high-quality scene flow on extremely sparse and highly noisy radar point clouds? (2) How to automatically generate training labels (avoiding manual labeling)? (3) Whether scene flow can enhance downstream human sensing tasks.

**Key Insight**: Leveraging co-located RGB-D cameras for auto-labeling—using OpenPose to extract 2D keypoints + lifting depth to 3D $\rightarrow$ constructing skeletons $\rightarrow$ assigning the motion of the nearest skeleton segment to each radar point.

**Core Idea**: Designing a five-module scene flow network tailored for radar characteristics (multi-scale local features + global aggregation + temporal GRU + constrained regression), and constructing the first radar scene flow dataset using an RGB-D camera auto-labeling scheme.

## Method

### Overall Architecture
Given two consecutive frames of radar point clouds $\mathcal{X}, \mathcal{Y}$ as input, the model outputs a 3D displacement vector $\mathcal{F} = \{f_i\}_{i=1}^N$ for each point. Five cascaded modules are involved: local feature extraction $\rightarrow$ global aggregation $\rightarrow$ flow embedding generation $\rightarrow$ temporal information propagation $\rightarrow$ constrained scene flow regression.

### Key Designs

1. **Multi-scale Local Feature Extraction**:

    - **Function**: Four parallel Set Abstraction (SA) layers with grouping radii of [0.05, 0.1, 0.2, 0.4m] respectively.
    - **Mechanism**: Different radii capture multi-scale local geometry ranging from fine limb movements to coarse-grained torso motions.
    - **Design Motivation**: Since radar point clouds are extremely sparse, single-scale features easily miss the correlations of distant points, whereas multi-scale features ensure information completeness.

2. **Global Feature Aggregation + GRU Temporal Propagation**:

    - **Function**: Aggregating local features into a global feature vector using MLP attention weights; the GRU propagates the global state across frames as $h^t = \text{GRU}(h^{t-1}, g^{\mathcal{B}})$.
    - **Mechanism**: Global aggregation enables each point to "know" the state of the entire body (e.g., the torso remains stationary while arms swing), and the GRU leverages temporal continuity to smooth motion estimation.
    - **Design Motivation**: Radar suffer from severe point loss (some body parts disappear between consecutive frames), and temporal information can compensate for the missing information within single frames.

3. **Constrained Scene Flow Regression**:

    - **Function**: After the MLP regresses the 3D displacement of each point, clamping constraints are applied on each axis such that $f \in [-\epsilon, \epsilon]$, with $\epsilon = 0.1$m.
    - **Mechanism**: There is an upper limit on human motion displacement between consecutive frames (at a frame rate of ~13Hz, the maximum speed of the human body corresponds to approximately 10cm/frame).
    - **Design Motivation**: Constraints prevent anomalously large displacement predictions (caused by noise or ghost points), and ablation studies demonstrate that this is the single component with the largest contribution.

4. **RGB-D Auto-labeling Scheme**:

    - **Function**: Using a co-located RGB-D camera to obtain skeleton motions as pseudo-labels.
    - **Mechanism**: OpenPose extracts 2D keypoints (14 keypoints, confidence >0.5), which are lifted to 3D using depth values to construct 13 skeleton segments. Each radar point is assigned to the nearest skeleton segment, and its pseudo-label is set as the motion of that segment's endpoints.
    - **Key Filtering**: Discarding keypoints with confidence <0.5 and abnormal labels with displacements >0.5m.

### Loss & Training
A weighted L2 loss is used to distinguish between large and small motions: $\mathcal{L} = \alpha_l \mathcal{L}_{large} + \alpha_s \mathcal{L}_{small}$, where $\alpha_l=2, \alpha_s=1$, with a threshold $\zeta=0.1$m. An validity mask $\mathcal{M}$ filters noisy labels.

## Key Experimental Results

### Main Results

| Method | EPE3D All(m)↓ | Moving(m) | Static(m) | Acc3D Strict↑ | Acc3D Relax↑ |
|------|-------------|-----------|-----------|--------------|-------------|
| PV-RAFT | 0.161 | 0.170 | 0.107 | 0.179 | 0.292 |
| RaFlow | 0.107 | 0.115 | 0.094 | 0.271 | 0.427 |
| **milliFlow** | **0.046** | **0.051** | **0.009** | **0.406** | **0.703** |

EPE3D is reduced from the second-best 0.107m to 0.046m (a 57% reduction), and Acc3D Relax is improved from 42.7% to 70.3%.

### Ablation Study

| Configuration | EPE3D(m)↓ | Acc3D Strict↑ | Acc3D Relax↑ |
|------|-----------|--------------|-------------|
| Full Model | 0.046 | 0.406 | 0.703 |
| w/o Constrained Regression | 0.083 | 0.286 | 0.490 |
| w/o Context Features | 0.071 | 0.315 | 0.536 |
| w/o Global Aggregation | 0.061 | 0.361 | 0.628 |
| w/o Temporal Propagation | 0.053 | 0.382 | 0.676 |

### Downstream Tasks

| Task | Baseline | + Scene Flow Features | Gain |
|------|------|-----------|------|
| Activity Recognition (Mean HAR) | 48.14% | 56.01% | **+7.87%** |
| Human Parsing (mIoU) | 49.09% | 52.72% | **+3.63%** |
| Body Part Tracking (mJE 1 Frame) | — | 0.025m | — |

### Key Findings
- **Constrained regression contributes the most**: Removing it increases EPE3D from 0.046 to 0.083 (+80%), showing that noise suppression is critical for radar.
- **Generalization to unseen activities**: On sitting/squatting/nodding which are not included in the training set, the EPE3D is only 0.034m (even better than on the training set), indicating that general motion patterns are learned.
- **Real-time inference**: 74ms/frame, meeting the 13.5Hz radar frame rate requirement, with a GPU memory footprint of only 134MB.

## Highlights & Insights
- **First to introduce scene flow to mmWave radar human sensing**: This fills an important gap—previously, radar-based human sensing either used point clouds directly for classification or utilized Doppler for coarse-grained analysis. Scene flow provides a fine-grained, per-point motion representation.
- **Ingenious and practical auto-labeling scheme**: A co-located RGB-D camera is used to provide labels during the training phase, while only radar is required at inference time. This paradigm of "camera-assisted training, radar-only deployment" solves the core bottleneck of the lack of annotations in radar data.
- **Simplicity and effectiveness of constrained regression**: Despite being a simple clamp operation, its effect on noisy point clouds is extremely significant (yielding the largest drop in the ablation study) and can be generalized to motion estimation for any sparse sensors.

## Limitations & Future Work
- **Limitation to single person, frontal view, and indoors**: It does not address multi-person scenarios, non-frontal orientations, or outdoor environments (where sunlight interferes with depth measurements).
- **Stationary platform assumption**: The radar is mounted statically, without ego-motion compensation; scenarios involving mobile robots/UAVs would require additional processing.
- **Skeleton assumption constraints**: Modeling human motion as rigid skeleton movements prevents handling non-rigid deformations such as cloth fluttering.
- **Opportunities for improvement**: (1) Replacing 3D radar with 4D radar (with elevation information) to increase point cloud density; (2) introducing Transformers instead of GRU for longer-term temporal modeling.

## Related Work & Insights
- **vs RaFlow**: The only other existing radar scene flow method, which also uses RGB-D-assisted labeling. milliFlow achieves a 57% improvement in EPE3D, because RaFlow lacks multi-scale features and temporal modeling.
- **vs LiDAR Scene Flow Methods (PV-RAFT, etc.)**: These methods perform poorly when applied directly to sparse radar point clouds (EPE3D > 0.15m) as they are designed for dense LiDAR data. milliFlow is specifically optimized for radar sparsity.
- **vs Pure Doppler Methods**: Doppler only provides radial velocity with low resolution. In contrast, milliFlow estimates complete 3D displacements, significantly increasing the amount of information.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The first work on radar scene flow, presenting valuable problem definition and labeling schemes, although the network design is a combination of existing modules.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Scene flow + 3 downstream tasks, generalization tests, detailed ablation studies, and execution time analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem motivation and sound experimental design.
- **Value**: ⭐⭐⭐⭐ Pioneered a new direction in radar scene flow, with the dataset and annotation scheme holding lasting value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](../../NeurIPS2025/3d_vision/rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[CVPR 2026\] mmWaveFlow: Unified Enhancement and Generation of mmWave Human Point Clouds](../../CVPR2026/3d_vision/mmwaveflow_unified_enhancement_and_generation_of_mmwave_human_point_clouds.md)
- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](../../CVPR2025/3d_vision/zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2026\] ARES: Unifying Asymmetric RGB-Event Stereo for Probabilistic Scene Flow Estimation](../../CVPR2026/3d_vision/ares_unifying_asymmetric_rgb-event_stereo_for_probabilistic_scene_flow_estimatio.md)
- [\[AAAI 2026\] Class-Partitioned VQ-VAE and Latent Flow Matching for Point Cloud Scene Generation](../../AAAI2026/3d_vision/class-partitioned_vq-vae_and_latent_flow_matching_for_point_cloud_scene_generati.md)

</div>

<!-- RELATED:END -->
