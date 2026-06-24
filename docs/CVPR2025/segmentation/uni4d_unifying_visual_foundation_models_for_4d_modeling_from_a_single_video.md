---
title: >-
  [Paper Note] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video
description: >-
  [CVPR 2025][Segmentation][4D Modeling] Uni4D proposes a multi-stage optimization framework that unifies multiple pre-trained visual foundation models (depth estimation, point tracking, segmentation, etc.) into an energy minimization problem. Without requiring retraining or fine-tuning, it jointly recovers camera poses, static/dynamic 3D geometry, and dense 3D motion trajectories from casual monocular videos, achieving state-of-the-art performance on multiple dynamic scene dat…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "4D Modeling"
  - "Visual Foundation Models"
  - "Dynamic Scene Reconstruction"
  - "Camera Pose Estimation"
  - "Structure-from-Motion"
date: 2026-05-08
content_hash: d4bc0c92bb754380
---

# Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video

**Conference**: CVPR 2025  
**arXiv**: [2503.21761](https://arxiv.org/abs/2503.21761)  
**Code**: [https://davidyao99.github.io/uni4d](https://davidyao99.github.io/uni4d)  
**Area**: 4D Scene Understanding / 3D Vision  
**Keywords**: 4D Modeling, Visual Foundation Models, Dynamic Scene Reconstruction, Camera Pose Estimation, Structure-from-Motion

## TL;DR

Uni4D proposes a multi-stage optimization framework that unifies multiple pre-trained visual foundation models (depth estimation, point tracking, segmentation, etc.) into an energy minimization problem. Without requiring retraining or fine-tuning, it jointly recovers camera poses, static/dynamic 3D geometry, and dense 3D motion trajectories from casual monocular videos, achieving state-of-the-art performance on multiple dynamic scene datasets.

## Background & Motivation

**Background**: In recent years, a large number of visual foundation models have emerged, performing exceptionally well in single tasks such as depth estimation (UniDepth, Metric3D), segmentation (SAM, Grounding-SAM), and motion tracking (CoTracker). However, integrating these capabilities into 4D (time + geometry) modeling remains an open problem. Traditional SfM/SLAM methods rely on rigid-body assumptions and cannot handle dynamic scenes.

**Limitations of Prior Work**: Existing 4D understanding methods face two core difficulties: (1) High-quality 4D ground truth data is extremely scarce, making end-to-end training difficult; (2) 4D understanding involves multiple interrelated subtasks such as camera pose estimation, 3D reconstruction, and dynamic tracking, where their respective data-driven cues are noisy and lack an effective unified framework. Existing methods, such as CasualSAM which requires fine-tuning network weights, and MonST3R which requires additional training, both limit the generalization ability and scalability of the models.

**Key Challenge**: The outputs of individual foundation models (e.g., depth, tracking, segmentation) are essentially different 2D projections of the 4D world, and inconsistencies exist among them. Directly using these noisy cues cannot yield a coherent 4D scene representation.

**Goal**: Design a training-free framework to integrate multiple pre-trained foundation models in a modular fashion, recovering a consistent 4D scene representation through a structured optimization strategy.

**Key Insight**: The authors observe that each visual cue (segmentation, depth, tracking) is a certain 2D projection of the 4D world. The key is to find a 4D scene representation that makes all projections consistent. Therefore, the problem is modeled as energy minimization.

**Core Idea**: Construct a unified energy function by using the outputs of multiple foundation models as observational cues combined with motion and geometric priors, and progressively solve for camera parameters, static geometry, and dynamic geometry through a three-stage divide-and-conquer optimization strategy.

## Method

### Overall Architecture

The input is a casually captured monocular video. First, three types of visual cues are extracted using pre-trained foundation models: dynamic segmentation (RAM + GPT-4o + Grounding-SAM + DEVA), dense motion trajectories (CoTrackerV3), and monocular depth (UniDepthV2). Then, through a three-stage optimization, the following are sequentially solved: (1) initial camera parameters; (2) bundle adjustment for static regions; (3) non-rigid bundle adjustment for dynamic regions. Finally, dense 4D point clouds are obtained through depth fusion.

### Key Designs

1. **Three-stage Divide-and-Conquer Optimization**:

    - **Function**: Decompose the highly non-linear, joint optimization problem with millions of degrees of freedom into three manageable subproblems.
    - **Mechanism**: Stage 1 utilizes depth and tracking to establish 2D-3D correspondences, initializing camera parameters via reprojection errors within a sliding window. Stage 2 fixes dynamic regions and jointly optimizes camera poses and static 3D points (classical bundle adjustment + camera motion smoothness prior). Stage 3 freezes camera parameters and only optimizes dynamic point clouds (non-rigid BA + motion prior) to avoid unstable dynamic cues from interfering with pose estimation.
    - **Design Motivation**: Directly optimizing all variables jointly is highly prone to falling into local optima. Introducing variables progressively from easy to difficult across stages, with a good initialization in each stage, significantly improves convergence.

2. **Adaptive Camera Motion Prior**:

    - **Function**: Constrain the temporal smoothness of camera trajectories.
    - **Mechanism**: Penalize the rate of change of relative motion (rotation and translation) between adjacent frames, but adaptively adjust the weight based on the magnitude of motion—relaxing the constraint during large motions and tightening it during small motions. Specifically, the normalized rate of change is used: $E_{\text{rot}} = 2\|\text{rad}(R_{t\to t+1}) - \text{rad}(R_{t-1\to t})\| / (\|\text{rad}(R_{t-1\to t})\| + \|\text{rad}(R_{t\to t+1})\|)$.
    - **Design Motivation**: A smoothness prior with fixed weight over-constrains during fast motion and under-constrains during slow motion. Adaptive weights make the prior effective across various motion patterns.

3. **ARAP + Temporal Smoothness Dynamic Motion Prior**:

    - **Function**: Regularize dynamic point clouds to reduce ambiguities in non-rigid reconstruction.
    - **Mechanism**: The ARAP (As-Rigid-As-Possible) prior finds neighbors for each dynamic control point via KNN, penalizing sudden changes in distance between adjacent pairs to maintain local rigidity. The temporal smoothness prior directly penalizes the displacements of dynamic points in adjacent frames. The combination of both allows reasonable non-rigid deformation while preventing excessive distortion.
    - **Design Motivation**: Non-rigid SfM is highly ill-posed, and a degenerate solution will be obtained without priors. However, the authors deliberately avoid using category-specific priors (such as rigid motion, articulated motion) to maintain the generality of the method.

### Loss & Training

The overall energy function is $E_{BA} + E_{NR} + E_{\text{motion}} + E_{\text{cam}}$, where $E_{BA}$ and $E_{NR}$ represent the static and dynamic reprojection errors respectively, $E_{\text{motion}}$ includes the ARAP and temporal smoothness terms, and $E_{\text{cam}}$ is the camera motion prior. The Adam optimizer is used, with 600 iterations per sliding window in Stage 1, 2000 iterations in Stage 2, and 1000 iterations in Stage 3. The learning rate decays from 1e-3/1e-2 to 1e-4, using ReduceLROnPlateau and EarlyStopping. The entire framework takes about 5 minutes on a 50-frame video (on an RTX A6000).

## Key Experimental Results

### Main Results

| Dataset | Metric | Uni4D | MonST3R | CasualSAM | Gain |
|--------|------|-------|---------|-----------|------|
| Sintel | ATE ↓ | 0.110 | 0.108 | 0.137 | Comparable |
| Sintel | RPE rot ↓ | 0.338 | 0.729 | 0.630 | -53.6% |
| TUM-dyn | ATE ↓ | 0.012 | 0.108 | 0.036 | -88.9% |
| TUM-dyn | RPE rot ↓ | 0.335 | 1.371 | 0.745 | -75.6% |
| Bonn | ATE ↓ | 0.017 | 0.023 | 0.024 | -26.1% |
| Sintel (depth) | Abs Rel ↓ | 0.216 | 0.358 | 0.292 | -26.0% |
| Bonn (depth) | Abs Rel ↓ | 0.038 | 0.060 | 0.069 | -36.7% |

### Ablation Study

| Setting | ATE ↓ | RPE trans ↓ | RPE rot ↓ |
|------|-------|------------|-----------|
| Stage 1 only | 0.150 | 0.051 | 0.551 |
| Stage 2 only | 0.587 | 0.193 | 4.12 |
| Full (Stage 1+2+3) | 0.110 | 0.032 | 0.338 |

Depth consistency ablation (Sintel):

| Method | SC ↓ | δ_SC<0.01 ↑ |
|------|------|-------------|
| UniDepth | 0.109 | 31.8 |
| Uni4D | 0.043 | 69.3 |

### Key Findings

- The initialization provided by Stage 1 is crucial; running Stage 2 alone yields very poor results (ATE 0.587), indicating that joint optimization requires a strong initialization.
- The poses produced by Stage 1 exhibit drift, which is effectively corrected by the bundle adjustment in Stage 2.
- Uni4D improves the depth consistency metric of UniDepth from 0.109 to 0.043, eliminating the flickering caused by direct fusion.
- The advantage in real-world scenes (TUM, Bonn) is significantly larger than in synthetic data (Sintel), demonstrating that the combination of foundation models generalizes better in the real world.

## Highlights & Insights

- **Training-free Modular Design**: No retraining or fine-tuning of any model is required, and better foundation models can be substituted at any time. This "plug-and-play" approach is more flexible than end-to-end methods.
- **Ingenious Divide-and-Conquer Optimization Strategy**: The three-stage strategy of progressively introducing variables from easy to hard is highly practical, especially the decision to freeze camera parameters in Stage 3—which, though counter-intuitive, prevents geometry/trajectory estimation from dynamic noise contamination.
- **Adaptive Camera Motion Prior**: The approach of adjusting regularization strength based on the magnitude of motion is simple yet effective, and can be transferred to other tasks involving trajectory estimation.

## Limitations & Future Work

- It relies heavily on the quality of the foundation models; if a model fails in a specific scene, the overall performance will degrade.
- It depends on dense point tracking to establish correspondences, which may fail in texture-sparse regions.
- It lacks rendering capabilities (unlike NeRF/3DGS-based methods) and focuses solely on geometric recovery.
- Future work can introduce more foundation models (e.g., normal estimation, material decomposition) to further enrich the dimensions of 4D understanding.

## Related Work & Insights

- **vs MonST3R**: MonST3R fine-tunes DUSt3R for 4D reconstruction, requiring training data and exhibiting significant geometric noise in distant areas. Uni4D requires no training, yielding cleaner static and dynamic geometry.
- **vs CasualSAM**: CasualSAM fine-tunes depth network weights and introduces uncertainty modeling, but often exhibits distorted geometry and poor dynamic segmentation. Uni4D achieves better static-dynamic separation through the segmentation foundation model.
- This paper demonstrates the paradigm advantage of "combining foundation models": it does not require 4D data training, and unifies the outputs of multiple models solely through ingenious optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ The core is combining existing models rather than designing a new architecture, but the unified framework and three-stage optimization design are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates pose, depth, and reconstruction quality across multiple datasets, with clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is logically clear, with well-explained energy functions and optimization pipelines.
- Value: ⭐⭐⭐⭐ The idea of "unifying foundation models for 4D" has strong practical significance and scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene](learning_4d_panoptic_scene_graph_generation_from_rich_2d_visual_scene.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](../../ICCV2025/segmentation/tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[CVPR 2025\] SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models](sketchfusion_learning_universal_sketch_features_through_fusing_foundation_models.md)
- [\[CVPR 2025\] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)
- [\[NeurIPS 2025\] Seg-VAR: Image Segmentation with Visual Autoregressive Modeling](../../NeurIPS2025/segmentation/seg-var_image_segmentation_with_visual_autoregressive_modeling.md)

</div>

<!-- RELATED:END -->
