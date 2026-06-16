---
title: >-
  [Paper Note] TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels
description: >-
  [NeurIPS 2025][3D tracking] This paper presents TrackingWorld, a pipeline for dense 3D tracking of almost all pixels from monocular video. It lifts sparse 2D trajectories to dense ones via a tracking upsampler…
tags:
  - "NeurIPS 2025"
  - "3D tracking"
  - "monocular video"
  - "world coordinate system"
  - "dense tracking"
  - "camera pose estimation"
date: 2026-05-08
content_hash: 4660758895a84992
---

# TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels

**Conference**: NeurIPS 2025
**arXiv**: [2512.08358](https://arxiv.org/abs/2512.08358)  
**Code**: [Project Page](https://igl-hkust.github.io/TrackingWorld.github.io/)  
**Area**: Video Understanding
**Keywords**: 3D tracking, monocular video, world coordinate system, dense tracking, camera pose estimation

## TL;DR

This paper presents TrackingWorld, a pipeline for dense 3D tracking of almost all pixels from monocular video. It lifts sparse 2D trajectories to dense ones via a tracking upsampler, iteratively tracks newly appearing objects across all frames, and employs an optimization-based framework to lift 2D trajectories into world-coordinate 3D space with explicit decoupling of camera motion and object motion.

## Background & Motivation

### Two Fundamental Limitations of Existing 3D Tracking Methods

**Limitation 1: Inability to Disentangle Camera Motion from Object Motion**

Methods such as OmniMotion, SpatialTracker, and DELTA assume a static camera and model 3D flow in the camera coordinate system. However, downstream tasks (e.g., motion analysis, novel view synthesis) generally require distinguishing camera motion from dynamic object motion. Recent work MotionGS also demonstrates that explicitly accounting for camera poses improves 3D tracking quality. Although ST4RTrack and TAPIP3D attempt world-coordinate tracking, the former suffers from long-term drift while the latter is limited to sparse tracking and cannot recover camera motion.

**Limitation 2: Tracking Only Pixels from the First Frame**

Existing methods are restricted to tracking sparse pixels from the first video frame and cannot follow dynamic targets that appear in subsequent frames. Although DELTA proposes an upsampler to produce dense 3D trajectories, it remains confined to the first frame. Estimating dense 3D trajectories for all pixels across all frames remains an open problem.

### Clarification on "Almost All Pixels"

The qualifier "almost all" refers to the intentional filtering of noisy and outlier trajectories in the final output to ensure robustness. This represents a deliberate engineering trade-off rather than a methodological limitation.

## Method

### Overall Architecture

TrackingWorld comprises two main stages:
1. **Dense 2D Tracking**: lifting sparse 2D trajectories to dense ones covering all frames.
2. **2D-to-3D Lifting**: estimating camera poses and converting 2D trajectories into world-coordinate 3D trajectories via a three-stage optimization framework.

The inputs are a monocular video and preprocessed outputs from foundation models (sparse trajectories, depth maps, dynamic masks); the outputs are dense 3D trajectories and per-frame camera poses.

### Key Designs

#### 1. Sparse-to-Dense 2D Trajectory Upsampling

**Core Finding**: The upsampling module from DELTA generalizes to arbitrary 2D trajectories, not only those produced by DELTA itself.

Given sparse 2D trajectories $\mathbf{P}_{\text{sparse}} \in \mathbb{R}^{(\frac{H}{s} \times \frac{W}{s}) \times T \times 2}$, the upsampler predicts a weight matrix $\mathbf{W}$ from features:

$$\mathbf{P}_{\text{dense}} = \mathbf{W}^T \mathbf{P}_{\text{sparse}}$$

In practice, each dense point is associated only with its spatially neighboring sparse trajectories, ensuring computational efficiency.

**Per-Frame Tracking and Deduplication**: 2D tracking and upsampling are performed on all frames, but most regions have already been tracked in earlier frames. Accordingly, pixels lying in the vicinity of any existing visible 2D trajectory are discarded, and isolated regions smaller than a threshold $\tau=50$ are removed via connected-component analysis. Experiments confirm that this filtering strategy consistently improves accuracy.

#### 2. Initial Camera Pose Estimation (Stage 1)

Static-region 2D trajectories $\mathbf{P}_{\text{static}}$ are selected using a foreground dynamic mask. After back-projecting them into 3D space via monocular depth, a reprojection loss is defined:

$$\mathcal{L}_{\text{proj}} = \sum_i^{N_{\text{inliers}}} \sum_{t_1}^{T} \sum_{t_2}^{T} \|\pi_{t_2} \pi_{t_1}^{-1}(\mathbf{P}_{\text{static}}(i,t_1), \mathbf{D}_{\text{static}}(i,t_1)) - \mathbf{P}_{\text{static}}(i,t_2)\|_2^2$$

For computational efficiency, the video is divided into $C$ clips; intra-clip poses are estimated in parallel and then merged into global poses by estimating inter-clip poses.

#### 3. Dynamic Background Refinement (Stage 2)

**Design Motivation**: Foreground dynamic masks are often imprecise; the background may still contain moving objects (e.g., a rolling apple) that interfere with bundle adjustment.

An **As-Static-As-Possible (ASAP)** constraint is introduced: each point in the "static" region is additionally modeled with a time-varying offset $\mathbf{O}_{\text{static}}$:

$$\mathbf{T}'_{\text{static}}(i,t) = \mathbf{T}_{\text{static}}(i) + \mathbf{O}_{\text{static}}(i,t)$$

Camera poses and static 3D coordinates are jointly optimized using a bundle adjustment loss plus an ASAP regularizer:

$$\mathcal{L}_{\text{asap}} = \sum_{i,t} \|\mathbf{O}_{\text{static}}(i,t)\|_1$$

The L1 norm encourages most offsets to be zero (truly static points), while non-zero offsets identify dynamic background points. The joint objective is:

$$\mathcal{L}_{\text{static}} = \lambda_{\text{ba}} \mathcal{L}_{\text{ba}} + \lambda_{\text{dc}} \mathcal{L}_{\text{dc}} + \lambda_{\text{asap}} \mathcal{L}_{\text{asap}}$$

where $\lambda_{\text{ba}}=1, \lambda_{\text{dc}}=1, \lambda_{\text{asap}}=5$. $\mathcal{L}_{\text{dc}}$ is a depth consistency loss constraining projected depths to be consistent with monocular depth estimates.

#### 4. Dynamic Object Tracking (Stage 3)

Background points satisfying $\|\mathbf{O}_{\text{static}}(i,\cdot)\|_2 \geq \varepsilon$ are also reclassified as dynamic. Dynamic 3D trajectories $\mathbf{T}_{\text{dynamic}} \in \mathbb{R}^{N_{\text{dynamic}} \times T \times 3}$ are directly optimized; the training objective includes a reprojection loss, a depth consistency loss, an as-rigid-as-possible constraint $\mathcal{L}_{\text{arap}}$, and a temporal smoothness constraint $\mathcal{L}_{\text{ts}}$:

$$\mathcal{L}_{\text{dyn}} = \lambda_{\text{ba}} \mathcal{L}_{\text{ba}} + \lambda_{\text{dc}} \mathcal{L}_{\text{dc}} + \lambda_{\text{arap}} \mathcal{L}_{\text{arap}} + \lambda_{\text{ts}} \mathcal{L}_{\text{ts}}$$

where $\lambda_{\text{arap}}=100, \lambda_{\text{ts}}=10$.

### Loss & Training

The overall framework is optimization-based rather than learning-based. Processing a 30-frame video on an RTX 4090 takes approximately 20 minutes. Clip-level parallelism and static-point downsampling (downsampling factor $\varpi$) are used to accelerate optimization, reducing runtime from 60 minutes to 8 minutes with negligible accuracy loss.

## Key Experimental Results

### Camera Pose Estimation

| Method | Sintel ATE↓ | Sintel RTE↓ | Bonn ATE↓ | TUM-D ATE↓ |
|--------|-------------|-------------|-----------|------------|
| MonST3R | 0.111 | 0.044 | 0.029 | 0.063 |
| Align3R | 0.128 | 0.042 | 0.023 | 0.027 |
| Uni4D* | 0.116 | 0.046 | 0.017 | 0.039 |
| **Ours (DELTA)** | **0.088** | **0.035** | **0.016** | **0.016** |

### Dense 3D Tracking Depth Accuracy

| Method | Sintel Abs Rel↓ | Sintel δ<1.25↑ | Bonn Abs Rel↓ | TUM-D Abs Rel↓ |
|--------|----------------|----------------|---------------|----------------|
| DELTA+UniDepth (no optimization) | 0.636 | 63.1 | 0.153 | 0.178 |
| **Ours (DELTA)** | **0.218** | **73.3** | **0.058** | **0.084** |

### Ablation Study

| Configuration | ATE↓ | RTE↓ | RRE↓ | Abs Rel↓ | δ<1.25↑ |
|---------------|------|------|------|----------|---------|
| w/o per-frame tracking | 0.171 | 0.047 | 0.748 | / | / |
| w/o initial poses | 0.659 | 0.153 | 1.382 | 0.230 | 72.4 |
| w/o dynamic object tracking | 0.088 | 0.035 | 0.410 | 0.468 | 73.0 |
| w/o $\mathbf{O}_{\text{static}}$ | 0.092 | 0.036 | 0.459 | 0.224 | 72.6 |
| w/o depth consistency loss | 0.093 | 0.036 | 0.441 | 0.234 | 71.2 |
| **Full model** | **0.088** | **0.035** | **0.410** | **0.218** | **73.3** |

### Key Findings

1. **Per-frame tracking is critical**: removing it degrades ATE from 0.088 to 0.171 (+94%), as crucial pose estimation cues from subsequent frames are lost.
2. **Initial pose estimation is indispensable**: without good initialization, camera poses are almost unrecoverable (ATE 0.659), as joint optimization fails to converge simultaneously.
3. **The ASAP constraint is effective**: removing $\mathbf{O}_{\text{static}}$ degrades RRE from 0.410 to 0.459; visualizations show dynamic background objects (e.g., apples) being incorrectly projected.
4. **Strong generalization of the 2D upsampler**: applying it to CoTrackerV3 trajectories reduces EPE (1.45→1.24) while cutting runtime from 3.00 to 0.25 minutes (12× speedup).
5. **Robustness across depth estimators**: consistent improvements are observed with ZoeDepth, Depth Pro, and UniDepth, demonstrating tolerance to varying depth prior quality.

## Highlights & Insights

1. **Explicit world-coordinate modeling**: unlike DELTA and SpatialTracker operating in the camera coordinate system, explicitly decoupling camera motion from object motion yields significant quality gains.
2. **Elegant ASAP constraint design**: L1 sparse regularization automatically identifies dynamic background regions without relying on perfect segmentation masks.
3. **Modular pipeline with foundation models**: the framework flexibly integrates different 2D trackers, depth estimators, and dynamic segmenters, forming an extensible pipeline.
4. **Physically consistent 3D tracking**: bundle adjustment enforces geometric consistency, improving depth accuracy approximately threefold over raw monocular estimates.
5. **By-product**: the method directly outputs temporally consistent video depth sequences, surpassing existing video depth methods on multiple benchmarks.

## Limitations & Future Work

- Dependence on multiple auxiliary models (2D tracker, depth estimator, dynamic mask predictor) introduces additional computational overhead and requirements on component quality.
- The optimization-based approach requires approximately 20 minutes (8 minutes with acceleration) for 30-frame videos, remaining far from real-time.
- Feed-forward solutions (e.g., jointly processing all frames to directly predict states, inspired by VGGT) may offer a more efficient future direction.
- While ST4RTrack suffers from pairwise matching drift, its feed-forward design philosophy merits further exploration.
- Robustness under extreme occlusion or large viewpoint changes has not been thoroughly validated.

## Related Work & Insights

- **2D Point Tracking**: CoTrackerV3, TAPIR, LocoTrack, TAP-Net
- **3D Point Tracking**: SpatialTracker, DELTA, OmniMotion, ST4RTrack, TAPIP3D
- **4D Reconstruction**: Uni4D, MonST3R, Align3R, MegaSaM
- **Depth Estimation**: UniDepth, Depth Pro, DepthCrafter
- **Insights**: The ASAP constraint is generalizable to any scene understanding task requiring static-dynamic separation; the strong generalization of the tracking upsampler highlights the advantages of modular design.

## Rating
- Novelty: ⭐⭐⭐⭐☆ — World-coordinate dense 3D tracking with per-frame extension represents a meaningful advance, though individual components largely build on prior techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across four dimensions (camera pose, depth accuracy, sparse 3D tracking, dense 2D tracking) with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐☆ — The pipeline is described clearly with precise problem formulation.
- Value: ⭐⭐⭐⭐⭐ — Establishes a new performance benchmark for dense 3D tracking and serves as a foundation module for multiple downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Ultrametric Cluster Hierarchies: I Want 'em All!](ultrametric_cluster_hierarchies_i_want_em_all.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](../../ICLR2026/others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[NeurIPS 2025\] MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation](metafind_scene-aware_3d_asset_retrieval_for_coherent_metaverse_scene_generation.md)
- [\[NeurIPS 2025\] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [\[ICCV 2025\] Recovering Parametric Scenes from Very Few Time-of-Flight Pixels](../../ICCV2025/others/recovering_parametric_scenes_from_very_few_time-of-flight_pixels.md)

</div>

<!-- RELATED:END -->
