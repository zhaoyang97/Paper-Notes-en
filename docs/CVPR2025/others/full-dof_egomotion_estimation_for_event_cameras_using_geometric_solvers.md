---
title: >-
  [Paper Note] Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers
description: >-
  [CVPR 2025][Event camera] Proposes the first geometric solver method to estimate full 6-DoF egomotion (angular and linear velocities) solely using event streams. By establishing line-segment geometric constraints on the eventail manifold—specifically incidence and novel coplanarity relations—a sparse solver requiring as few as 8 events is designed, enabling decoupled rotation and translation estimation without requiring an IMU.
tags:
  - "CVPR 2025"
  - "Event camera"
  - "egomotion estimation"
  - "geometric solver"
  - "6-DoF pose"
  - "coplanarity constraint"
date: 2026-05-08
content_hash: 89295e70568b8e08
---

# Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers

**Conference**: CVPR 2025  
**arXiv**: [2503.03307](https://arxiv.org/abs/2503.03307)  
**Code**: [https://github.com/jizhaox/relpose-event](https://github.com/jizhaox/relpose-event)  
**Area**: Others  
**Keywords**: Event camera, egomotion estimation, geometric solver, 6-DoF pose, coplanarity constraint

## TL;DR
Proposes the first geometric solver method to estimate full 6-DoF egomotion (angular and linear velocities) solely using event streams. By establishing line-segment geometric constraints on the eventail manifold—specifically incidence and novel coplanarity relations—a sparse solver requiring as few as 8 events is designed, enabling decoupled rotation and translation estimation without requiring an IMU.

## Background & Motivation

### Background

**Background**: Event cameras are widely utilized in robotic navigation due to their high temporal resolution and high dynamic range. Most existing event-camera motion estimation methods assume that rotational displacement is known (provided by an IMU) and only estimate translation, or they estimate rotation only.

**Limitations of Prior Work**: (1) Relying on IMU-provided rotational priors limits the independence and lightweight nature of the system; (2) estimating translation only or rotation only fails to meet the demand for full 6-DoF motion estimation; (3) existing methods lack a theoretical foundation to recover full motion solely from event streams.

**Key Challenge**: Event streams (asynchronous pixel-level brightness changes) differ from traditional frames, making it impossible to directly apply classical epipolar geometry methods. How to extract 6-DoF motion signals from the spatiotemporal structure of event streams remains an open problem.

**Goal**: To recover full 6-DoF egomotion (3 angular velocity and 3 linear velocity components) solely from event streams, without requiring an IMU or other external sensors.

**Key Insight**: Leverage line-segment geometry on the eventail manifold: each event forms a ray in spatiotemporal space, and a set of collinear events forms a line segment. Motion constraint equations are established using the incidence relations (shared points) and coplanarity relations (shared plane normal vectors) among these line segments.

**Core Idea**: Model the event stream as a collection of spatiotemporal line segments, construct systems of equations using incidence and coplanarity relations among the segments, and recover full 6-DoF egomotion using a geometric solver requiring as few as 8 events.

## Method

### Overall Architecture
Given an input event stream $\{(x_i, y_i, t_i, p_i)\}$, line segments on the eventail manifold are constructed in spatiotemporal space. Systems of equations regarding the motion parameters $(\omega_x, \omega_y, \omega_z, v_x, v_y, v_z)$ are constructed via two types of geometric constraints: (1) linear equations from incidence relations (point-on-line constraints); (2) bilinear equations from coplanarity relations (normal vector constraints). An Adam optimizer and a first-order rotation approximation are used for efficient solving. Pure rotational degeneracy is handled as a special case.

### Key Designs

1. **Geometric Modeling of the Eventail Manifold**:

    - **Function**: Convert the event stream into usable geometric structures.
    - **Mechanism**: Each event $(x, y, t)$ defines a ray in 3D spatiotemporal space, and continuous events on the same edge form a line segment. The directions of these line segments are proportional to pixel velocities, which are determined by scene depth and egomotion. Reliable line-segment direction estimation is achieved by aggregating multiple events.
    - **Design Motivation**: A single event contains too little information (only a single brightness change trigger), but the structure of line segments in spatiotemporal space encodes motion information.

2. **Incidence Relation Solver**:

    - **Function**: Establish linear equations through point-on-line constraints.
    - **Mechanism**: If an event point lies on a line segment, its coordinates must satisfy the line equation. Substituting the motion parameters into the line-segment direction expression yields linear constraints on $(\omega, v)$. A minimum of 8 constraints (8 event-line pairs) are required to form the system of linear equations $Ax=0$.
    - **Design Motivation**: Solving linear equations is fast and stable, making it the preferred choice for a minimal solver.

3. **Coplanarity Relation Solver**:

    - **Function**: Provide additional constraints through normal vector constraints.
    - **Mechanism**: If two line segments are coplanar, their direction vectors and connecting vector must satisfy a zero triple product constraint. This yields a bilinear equation $n_1^T \cdot d_{12} = 0$, where $n_1$ is the normal vector and $d_{12}$ is the connection between the two line segments. This constraint does not require shared points, making it applicable to disjoint line segments in space.
    - **Design Motivation**: The incidence relation requires a clear association between line segments and points, whereas the coplanarity relation is more flexible—any two line segments can generate a constraint.

### Loss & Training
As a non-learning method, the Adam optimizer is utilized to minimize geometric residuals. A first-order rotation approximation $R \approx I + [\omega]_\times \Delta t$ simplifies the non-linear equations. Dedicated theoretical analysis and handling are provided for pure rotational degeneracy.

## Key Experimental Results

### Main Results

Evaluated on real event camera sequences from the VECtor dataset, split into 0.3-second non-overlapping intervals:

| Sequence | IncBat $\varepsilon_{ang}$ | IncBat $\varepsilon_{lin}$(°) | CopBat $\varepsilon_{ang}$ | CopBat $\varepsilon_{lin}$(°) |
|------|-----------|-----------|-----------|-----------|
| desk-normal | 0.232 | 23.0 | 0.236 | 25.1 |
| mountain-normal | 0.195 | 17.5 | 0.216 | 18.7 |
| sofa-normal | 0.229 | 21.1 | 0.221 | 20.6 |

Where $\varepsilon_{ang} \in [0,1]$ (lower is better), and $\varepsilon_{lin}$ is the angular error (°).

### Ablation Study

Runtime and numerical stability ($M=5$ line segments, $N=100$ events/segment, noise-free synthetic data, 1000 scenes):

| Solver | Rotation Parameterization | SR1 (threshold 0.01) | SR2 (threshold 0.05) | Median Runtime |
|--------|-----------|---------------|---------------|-------------|
| IncBat | +cascad | 97.3% | — | 17.1 ms |
| CopBat | +cascad | — | — | 16.7~48.7 ms |
| IncBat | +exact | Lower | — | Slower |
| IncBat | +approx | Medium | — | Faster |

Key variable analysis ($M=10$, $N=8\sim1000$, pixel noise $0.5$ pix, timestamp jitter $0.5$ ms):
- IncBat outperforms CopBat when the number of events is $<100$; as the number of events increases, both converge towards similar performance.
- The error drops significantly as the number of line segments increases from 1 to 50; all methods fail at $M=1$ due to rotational-translational ambiguity.
- The error increases monotonically with noise; in the absence of noise, the solver achieves nearly 100% success rate.

### Key Findings
- **Cascade rotation parameterization** performs best: utilizing a first-order approximation for rapid initialization, followed by fine-tuning with exact parameterization to balance efficiency and accuracy.
- **Coplanarity relations** provide key complementary constraints in scenarios with non-intersecting line segments, presenting a major theoretical innovation.
- Pure rotation is a degenerate case (translation is unestimable); practical deployment requires motion classification and detection.
- The actual error levels ($\varepsilon_{ang} \approx 0.2$, $\varepsilon_{lin} \approx 20°$) are sufficient for integration into VIO/SLAM pipelines.

## Highlights & Insights
- **Outstanding Theoretical Contribution**: It is proven for the first time that full 6-DoF motion can be recovered solely from event streams, and a theoretical lower bound for the minimum number of events (8) is established.
- **Novelty of Coplanarity Relations**: Traditional line-segment geometry mostly utilizes incidence relations; coplanarity relations do not require explicit intersections of line segments, significantly expanding the number of usable constraints.
- **No External Sensors Required**: Removing the reliance on IMUs enables motion estimation using only event cameras.

## Limitations & Future Work
- The first-order rotation approximation is only valid under small motions; fast-rotation scenarios require higher-order expansions.
- Noise in practical event streams affects the accuracy of line-segment direction estimation.
- There is a lack of quantitative comparison against learning-based event-camera motion estimation methods.
- The analysis of computational efficiency is not detailed enough.

## Related Work & Insights
- **vs. CMax methods**: The CMax series only estimates rotation, whereas ours simultaneously estimates both rotation and translation.
- **vs. Traditional frame-based methods (e.g., 5-point algorithm)**: Frame-based methods require feature matching; ours extracts motion directly from the spatiotemporal structure of events.
- **vs. Event + IMU fusion**: Removing IMU dependency makes the system more lightweight.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first pure event-based 6-DoF solver, with a major theoretical contribution from coplanarity relations.
- **Experimental Thoroughness**: ⭐⭐⭐ Theoretically rigorous, but the experimental scale is relatively small, lacking large-scale quantitative comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear geometric derivation and a complete theoretical framework.
- **Value**: ⭐⭐⭐⭐ Provides fundamental contributions to event-camera SLAM and autonomous navigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](event_ellipsometer_event-based_mueller-matrix_video_imaging.md)
- [\[CVPR 2025\] EBS-EKF: Accurate and High Frequency Event-based Star Tracking](ebs-ekf_accurate_and_high_frequency_event-based_star_tracking.md)
- [\[CVPR 2025\] Order-One Rolling Shutter Cameras](order-one_rolling_shutter_cameras.md)
- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)
- [\[AAAI 2026\] Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](../../AAAI2026/others/spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)

</div>

<!-- RELATED:END -->
