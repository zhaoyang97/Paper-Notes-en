---
title: >-
  [Paper Note] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation
description: >-
  [ICCV 2025][Video Understanding][Event Camera] This paper proposes EMoTive, an event camera-based 3D motion estimation framework that encodes fine-grained temporal evolution via Event Kymograph and models spatiotemporal…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Event Camera"
  - "3D Motion Estimation"
  - "Non-uniform Parametric Curves"
  - "Optical Flow"
  - "Motion in Depth"
date: 2026-05-08
content_hash: 351393f457276e05
---

# EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation

**Conference**: ICCV 2025
**arXiv**: [2503.11371](https://arxiv.org/abs/2503.11371)
**Code**: None
**Area**: Video Understanding
**Keywords**: Event Camera, 3D Motion Estimation, Non-uniform Parametric Curves, Optical Flow, Motion in Depth

## TL;DR
This paper proposes EMoTive, an event camera-based 3D motion estimation framework that encodes fine-grained temporal evolution via Event Kymograph and models spatiotemporal trajectories using event-density-guided non-uniform NURBS parametric curves. Optical flow and motion-in-depth fields are derived from these trajectories, achieving state-of-the-art performance on the newly constructed CarlaEvent3D dataset and real-world benchmarks.

## Background & Motivation
Visual 3D motion estimation—inferring the motion of 2D pixels in 3D space—is a core capability for spatial intelligence. However, existing methods face a fundamental contradiction:

**Core Pain Point: Spatiotemporal Motion Inconsistency Caused by Depth Variation.** When an object moves along the depth axis, its projection in image space undergoes local deformation (scaling), which violates the **local spatial motion smoothness** or **temporal motion invariance** assumptions underlying conventional optical flow methods. For example, a vehicle approaching the camera exhibits heterogeneous motion directions and speeds at different image locations.

**Limitations of Prior Work**:
- Frame-correspondence methods (e.g., RAFT) rely on local smoothness assumptions and cannot handle motion heterogeneity induced by depth variation.
- Dual-space decoupling methods (e.g., ScaleFlow) attempt to estimate planar motion and depth motion separately in feature and scale spaces, but since both spaces originate from the same pixel domain, the fundamental contradiction persists.
- Conventional frame cameras are constrained by fixed sampling rates, yielding insufficient temporal observations.

**Opportunity from Event Cameras**: Event cameras asynchronously report per-pixel brightness changes at microsecond-level temporal resolution, providing unprecedented fine-grained temporal observations. The key insight is that event streams can be projected along decoupled x-t and y-t planes to form **Event Kymographs**, capturing temporal evolution at microsecond precision and enabling non-stationary modeling of heterogeneous spatiotemporal motion.

**Core Idea: Model spatiotemporal trajectories using event-guided non-uniform parametric curves**, deriving optical flow via multi-time trajectory sampling and motion in depth via trajectory temporal gradients.

## Method

### Overall Architecture
The EMoTive pipeline proceeds as follows: (1) project event streams into Event Voxels (spatial features) and Event Kymographs (temporal features); (2) construct spatiotemporal dual cost volumes for motion representation; (3) fuse spatiotemporal features via a density-aware adaptive mechanism to update NURBS curve control points; (4) sample trajectories at multiple time steps to obtain optical flow, and derive motion-in-depth fields via temporal gradients.

### Key Designs
1. **Event Kymograph**:

    - Function: Projects event streams onto decoupled x-t and y-t planes to encode fine-grained temporal evolution.
    - Mechanism: Unlike Event Voxels, which perform coarse temporal quantization via triangular kernels, Kymographs apply a continuous Gaussian temporal projection kernel:
    $K_x = \sum_i p_i k(x - x_i) g(t - t_i | \sigma), \quad g(a|\sigma) = \exp(-(a/\sigma)^2)$
      where $\sigma$ controls the temporal smoothing scale, preserving temporal precision at the 10 μs level.
    - Design Motivation: Conventional Event Voxels inevitably discard fine temporal cues through time quantization. Spatial-axis decoupling enables independent analysis of x- and y-directional motion, while the continuous Gaussian kernel preserves fine temporal resolution.

2. **Spatiotemporal Dual Cost Volumes**:

    - Function: Construct cost pyramids independently along spatial and temporal dimensions to provide multi-scale motion matching information.
    - Mechanism:
        - **Spatial Cost Volume**: Extracts spatial features $f_{hw}$ from Event Voxels and builds a multi-resolution inner-product pyramid.
        - **Temporal Cost Volume**: Partitions Kymographs into blocks by temporal anchor points, extracts temporal features $f_{ht}, f_{wt}$ via 1D convolution, performs cross-block cross-correlation, and fuses via tensor product: $C_t^m(n,i,k,j,l) \doteq C_{ht}^m(n,i,j) \otimes C_{wt}^m(n,k,l)$
    - Design Motivation: The spatial cost volume captures inter-frame displacement information, while the temporal cost volume leverages the fine temporal resolution of Kymographs to capture motion dynamics. The two are complementary—the former addresses *where*, the latter addresses *how motion evolves*.

3. **Density-aware Adaptive NURBS Trajectory**:

    - Function: Guides the knot distribution and weights of non-uniform B-spline curves via event density to adaptively model heterogeneous motion.
    - Mechanism: The trajectory is defined as a NURBS curve:
    $\mathcal{T}(t,x,y) = \frac{\sum_i^n N_{i,p}(t) w_i \mathbf{P}_i(x,y)}{\sum_i^n N_{i,p}(t) w_i}$
      Density adaptation proceeds in three stages:
        - Compute the spatiotemporal density distribution $D_s$ from Kymographs.
        - Extract the top-$n$ temporal indices as key parameters.
        - Compute adjustable knots via sliding-window averaging and derive weights via density normalization.
    - Design Motivation: Uniform B-splines assume temporally uniform motion distribution and cannot represent heterogeneous dynamics such as acceleration or deceleration. Event density naturally reflects motion intensity—high-density regions correspond to intense motion and should be assigned more knots and higher weights to enhance curve expressiveness.

4. **From Trajectory to 3D Motion**:

    - Function: Jointly obtains optical flow and motion in depth through multi-time trajectory sampling and temporal gradient analysis.
    - Mechanism:
        - Optical Flow: $\mathcal{O}(x,y) = \mathcal{T}(\tau, x, y)$, querying the trajectory displacement at the target time.
        - Motion in Depth: $\mathcal{M} = \frac{v_0 \Delta t + \Delta x}{v_1 \Delta t + \Delta x}$, estimating instantaneous velocity from trajectory gradients to derive the depth change ratio.
        - Multi-view Fusion: $\mathcal{M}_k = \frac{1}{k} \sum_i \frac{t_k}{t_i}(\mathcal{M}_i - 1) + 1$, aggregating observations across multiple time steps for improved robustness.
    - Design Motivation: The trajectory is a unified representation of motion—optical flow is a discrete temporal sample of the trajectory, and motion in depth is a differential property of the trajectory. This analytical framework naturally unifies 2D and 3D motion estimation.

### Loss & Training
- Multi-task loss: $L = L_{\text{flow}} + L_{\text{depth}} + \lambda L_t$, with $\lambda = 10^{-7}$.
- Optical flow loss: exponentially weighted L1 loss $L_{\text{flow}} = \sum_k \gamma^{N-k} (|\mathcal{O}_x^{(k)}|_1 + |\mathcal{O}_y^{(k)}|_1)$.
- Motion-in-depth loss: similarly exponentially weighted L1 loss.
- Temporal gradient regularization: $L_t = \sum_i |\mathcal{T}'(t_{i+1}) - \mathcal{T}'(t_i)|_1$, preventing high-order trajectory distortion.
- AdamW optimizer with OneCycle learning rate schedule; trained for 60,000 iterations.

## Key Experimental Results

### Main Results
Evaluation on the CarlaEvent3D dataset (dense labels):

| Model | Flow EPE↓ | Flow F1↓ | Mid log-mid↓ | Params (M) | Inference Time (s) |
|-------|----------|---------|-------------|-----------|-------------------|
| E-RAFT | 2.781 | 24.604 | - | 5.04 | 0.049 |
| Expansion | 7.821 | 57.653 | 171.237 | 12.13 | 0.300 |
| ScaleFlow | 4.518 | 42.885 | 268.050 | 10.70 | 0.090 |
| Scale++ | 5.242 | 40.081 | 260.165 | 42.96 | 0.119 |
| EMoTive (Uniform) | 2.669 | 24.607 | 122.023 | 5.67 | 0.040 |
| **EMoTive** | **2.547** | **22.866** | **113.593** | 5.61 | 0.040 |

### Ablation Study

| Configuration | Flow EPE↓ | Mid log-mid↓ | Note |
|--------------|----------|-------------|------|
| EMoTive (Uniform B-spline) | 2.669 | 122.023 | Uniform baseline |
| EMoTive (NURBS, no density adaptation) | ~2.6 | ~118 | Non-uniform without event guidance |
| EMoTive (Full) | **2.547** | **113.593** | Density-adaptive NURBS |
| w/o Event Kymograph | ~3.0+ | ~140+ | Voxel only |
| w/o Multi-view Fusion | ~2.55 | ~130+ | Single-time depth estimation only |

### Key Findings
- **Event-guided non-uniform trajectory design is critical**: compared to uniform B-splines, non-uniform NURBS improves optical flow EPE by 4.6% and motion-in-depth log-mid by 6.9%.
- Methods relying on spatial correlation (Expansion, ScaleFlow, etc.) degrade significantly on event data (EPE increases by 1.9–5.3×), as event data has low spatial redundancy.
- EMoTive has only 5.61M parameters (47.6% fewer than ScaleFlow) and achieves 40 ms inference per 100 ms of event data, which is 52.9% faster than ScaleFlow.
- Temporal gradient regularization is important for preventing spiral artifacts; $\lambda = 10^{-7}$ is the optimal value.

## Highlights & Insights
- **Elegant problem decomposition**: optical flow and motion-in-depth estimation are unified under a trajectory parametric curve framework with clear physical interpretation.
- **Event Kymograph is a key innovation**: spatial-axis decoupling combined with a continuous Gaussian temporal kernel fully exploits the microsecond-level temporal resolution of event cameras.
- **Density-adaptive mechanism naturally matches event camera characteristics**: event density reflects motion intensity, making it a physically consistent signal for guiding curve parameter allocation.
- **The newly constructed dataset addresses a gap in the field**: CarlaEvent3D provides complete 3D motion annotations across diverse dynamic scenes and weather conditions.

## Limitations & Future Work
- CarlaEvent3D is a synthetic dataset; the simulation-to-real domain gap may affect generalization to real-world scenes.
- The order and number of control points of the NURBS curves are fixed and may not adapt to extremely complex motion patterns.
- The rigid-body constant-velocity assumption may not hold for non-rigid or variable-speed motion scenarios.
- Insufficient comparison with recent frame-event fusion methods (e.g., the BlinkVision series).
- Although computationally more efficient than baselines, further optimization is needed for real-time applications.

## Related Work & Insights
- **Trajectory parameterization is a powerful tool for motion estimation**: from point cloud trajectories in CamLiFlow to event trajectories in EMoTive, parametric curve representations are receiving increasing attention.
- **The "temporal microscope" property of event cameras** remains underexploited; Kymograph-style temporal projection offers a promising new representation.
- **Density as a motion prior** is a signal unique to event cameras and can potentially be generalized to other event-driven tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Event-guided non-uniform trajectory parameterization is a meaningful contribution; the Kymograph representation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both a newly constructed dataset and real-world benchmarks; however, most baselines are adapted methods rather than native event-based approaches.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and mathematical derivations are rigorous, though notation is occasionally dense.
- Value: ⭐⭐⭐⭐ Makes an important contribution to the field of event-driven 3D motion estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[ICCV 2025\] Towards Efficient General Feature Prediction in Masked Skeleton Modeling](towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)

</div>

<!-- RELATED:END -->
