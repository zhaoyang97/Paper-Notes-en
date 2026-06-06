---
title: >-
  [Paper Note] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks
description: >-
  [ICCV 2025][Motion Estimation] This paper proposes a unified linear N-point solver that recovers camera linear velocity and 3D point structure from 2D point correspondences with arbitrary timestamps…
tags:
  - "ICCV 2025"
  - "Motion Estimation"
  - "Asynchronous Feature Tracks"
  - "Linear Solver"
  - "Event Camera"
  - "Rolling Shutter"
date: 2026-05-08
content_hash: 075b04c3e11c9ae7
---

# A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks

**Conference**: ICCV 2025
**arXiv**: [2507.22733](https://arxiv.org/abs/2507.22733)  
**Code**: [GitHub](https://github.com/suhang99/AsyncTrack-Motion-Solver)  
**Area**: Geometric Computer Vision
**Keywords**: Motion Estimation, Asynchronous Feature Tracks, Linear Solver, Event Camera, Rolling Shutter

## TL;DR
This paper proposes a unified linear N-point solver that recovers camera linear velocity and 3D point structure from 2D point correspondences with arbitrary timestamps, supporting global shutter, rolling shutter, and event camera sensor modalities.

## Background & Motivation
Recovering camera motion and scene structure from point correspondences is a core problem in geometric computer vision. Classical algorithms such as the 5-point and 8-point methods are well-established, yet they rely on a fundamental assumption: correspondences are drawn from a pair of synchronously captured views, each representing an instantaneous snapshot of the scene.

With advances in sensor technology, however, this synchrony assumption is increasingly difficult to satisfy. Rolling shutter cameras capture images row by row, assigning different timestamps to different rows; event cameras are fully asynchronous sensors in which each pixel independently returns a stream of brightness-change events with microsecond temporal resolution. Existing methods either force asynchronous data into synchronous frames—discarding the core advantage of event cameras—or rely on line-feature-based solvers that require prominent linear structures in the scene.

The root cause of the difficulty is that existing geometric solvers cannot natively handle feature tracks from asynchronous sensors, while naively synchronizing asynchronous data sacrifices temporal resolution. The paper's starting point is to exploit a constant-velocity motion model and first-order kinematic formulation to derive a point incidence relation that is linear in both 3D point positions and velocity, enabling efficient solution via a linear system.

**Core Idea**: Under a constant-velocity motion model, observations of a point at arbitrary timestamps form a linear constraint system over the unknowns (3D point positions and linear velocity), solvable efficiently via the Schur complement.

## Method

### Overall Architecture
Given a set of timestamped 2D point tracks $\{(\mathbf{x}_{ij}, t_{ij})\}$, known angular velocity $\boldsymbol{\omega}$ (from an IMU or other estimator), and camera intrinsics $\mathbf{K}$, the solver recovers the normalized linear velocity $\hat{\mathbf{v}}$ and 3D points $\hat{\mathbf{P}}_i$. Observations are first converted into rotation-compensated bearing vectors; a linear constraint system is then assembled and solved efficiently via the Schur complement and SVD.

### Key Designs
1. **Point Incidence Relation**:

    - **Function**: Establishes geometric constraints between 2D observations, 3D point positions, and camera motion.
    - **Mechanism**: The bearing vector $\mathbf{f}_{ij}'$ of 3D point $\mathbf{P}_i$ projected onto the image plane at time $t_{ij}$ must be parallel to the 3D point in the camera frame. Applying the zero cross-product condition yields: $[\mathbf{f}_{ij}']_\times \mathbf{P}_i - t_{ij}' [\mathbf{f}_{ij}']_\times \mathbf{v} = \mathbf{0}$
    - **Design Motivation**: Unlike the classical epipolar constraint, this incidence relation operates directly on kinematic parameters (velocity) and 3D points without assuming timestamp synchronization. It subsumes the classical 5-point algorithm and recent line-feature-based solvers as special cases.

2. **Schur Complement Efficient Solver**:

    - **Function**: Exploits the sparse block-diagonal structure of the system matrix $\mathbf{A}$ to avoid SVD on the full large matrix.
    - **Mechanism**: The normal equations $\mathbf{A}^\top \mathbf{A} \mathbf{x} = \mathbf{0}$ are written in block form; the Schur complement reduces the $3M+3$-dimensional problem to SVD of a $3 \times 3$ matrix $\mathbf{B}$: $\mathbf{B} = \mathbf{M}_D - \mathbf{M}_B^\top \mathbf{M}_A^{-1} \mathbf{M}_B$. Since $\mathbf{M}_A$ is block-diagonal with $3 \times 3$ blocks, its inversion costs $O(M)$ rather than $O(M^3)$.
    - **Design Motivation**: Directly computing SVD of $\mathbf{A} \in \mathbb{R}^{3N \times (3M+3)}$ is expensive at large observation counts. The Schur complement reduces the computational bottleneck to a $3 \times 3$ SVD, yielding a minimum-case solve time of only 63 μs.

3. **Degeneracy and Solution Uniqueness Analysis**:

    - **Function**: Systematically characterizes conditions under which the solver produces degenerate solutions and analyzes solution multiplicity.
    - **Mechanism**: SVD yields two candidates $\hat{\mathbf{v}}$ and $-\hat{\mathbf{v}}$; the positive depth constraint $(\hat{\mathbf{P}}_i)_z > 0$ selects the correct solution. Degeneracy conditions require at least 2 observations at distinct timestamps per track so that $\mathbf{F}_i$ is full rank, and the matrix $\mathbf{B}$ must have rank at least 2.
    - **Design Motivation**: Explicitly characterizing the solver's applicability conditions and minimal sample requirements provides theoretical guidance for RANSAC sampling strategies.

### Loss & Training
The method requires no training and is a purely geometric solver. At deployment it is embedded in a RANSAC loop: each iteration samples $M=4$ tracks with $N_i=5$ observations each, generates a velocity hypothesis, and classifies inliers by angular residual $\bar{\theta}_i$ on bearing vectors (threshold 5°), with early termination when the inlier ratio exceeds 0.9. The final velocity estimate is refined using all inliers.

## Key Experimental Results

### Main Results

| Sensor Type | Sequence | eventail (baseline) | Ours | Ours (high conf.) |
|-------------|----------|---------------------|------|-------------------|
| Global Shutter | desk-normal | 22.7° / 23.4° | 15.1° / 8.5° | 10.2° / 7.3° |
| Global Shutter | shapes_trans | 31.8° / 32.7° | 17.1° / 7.2° | 9.9° / 6.2° |
| Rolling Shutter | Seq 4 | 43.8° / 40.8° | 27.5° / 20.1° | 22.6° / 17.4° |
| Rolling Shutter | Seq 5 | 45.5° / 44.8° | 24.7° / 17.0° | 19.3° / 13.8° |
| Event Camera | mountain-normal | 25.2° / 21.4° | 17.1° / 16.1° | 16.9° / 15.8° |
| Event + Global | shapes_trans | — | 14.4° / 7.5° | 7.0° / 6.7° |

Metric: angular error in velocity direction (mean / median, degrees); lower is better.

### Ablation Study

| Configuration | Key Effect | Notes |
|---------------|------------|-------|
| Track count $M$: 3→30 | Error decreases substantially | More spatial samples consistently help; diminishing returns beyond 30 |
| Observations per track $n$: 2→50 | Limited improvement | Temporal density contributes little to noise robustness |
| Time window: 0.05→0.4 s | Error steadily decreases | Longer tracks better average high-frequency noise |
| With vs. without rolling shutter correction | ~2° difference | Validates the importance of correct timestamp association |
| Events only vs. events + global shutter | Significant improvement | Multi-sensor fusion is a unique advantage of this method |

### Key Findings
- Point tracks are easier to extract than line features, with particular advantages in natural scenes lacking prominent linear structures.
- Event camera and global shutter camera tracks can be seamlessly fused, with complementary benefits: images provide high spatial resolution while events provide high temporal resolution.
- Full motion direction can be recovered from as few as 1 3D point and 3 temporal observations—a theoretically surprising minimal-case result.

## Highlights & Insights
- **Unified Theoretical Framework**: Across the continuum from global shutter to event cameras, this is the first genuinely assumption-free asynchronous point solver.
- **Elegant Use of Schur Complement**: Exploiting sparse structure reduces a large-scale linear system to a $3 \times 3$ SVD, enabling very fast practical execution.
- **Transferable Perspective Shift**: Reformulating motion estimation from "recovering relative pose" to "recovering first-order dynamics" offers broader inspiration for other visual odometry problems.

## Limitations & Future Work
- The method depends on known angular velocity $\boldsymbol{\omega}$ (typically from an IMU), limiting its use as a standalone solution in purely visual settings.
- The constant-velocity assumption limits applicability to non-uniform motions such as sharp accelerations or abrupt turns.
- Higher-order derivative estimation (e.g., acceleration) is highly sensitive to noise on real data; the paper acknowledges this as an open problem.
- Validation is confined to relatively small-scale datasets; integration into large-scale SLAM systems has not been tested.

## Related Work & Insights
- **vs. Gao et al. (eventail)**: eventail is a velocity solver based on line features; this paper extends the formulation to point features, achieving greater generality in natural scenes.
- **vs. Classical 5-point/8-point**: Classical methods assume synchronous capture; this paper relaxes that assumption via first-order kinematic modeling, constituting a natural generalization of classical theory.
- **vs. Saurer et al.**: A similar point incidence relation is used for absolute pose estimation from known 2D–3D correspondences; the present work requires only 2D observations and simultaneously recovers motion and structure.
- **vs. Contrast Maximization (CM)**: CM estimates motion by iteratively optimizing parametric image warping of events, which is computationally expensive and restricted to homographic warp scenarios. The proposed method is a closed-form linear solver with substantially higher efficiency.

## Additional Notes
- Minimum-case solver runtime is only 63 μs (Intel Xeon CPU), suitable for real-time RANSAC-embedded applications.
- The appendix derives a general formulation for arbitrary-order Taylor expansions; the first-order (linear velocity) case is the primary focus of the paper.
- Acceleration estimation is evaluated in simulation but found to be excessively noise-sensitive on real data, left as future work.
- Constraint analysis shows that a single 3D point with 3 temporal observations suffices to recover the full motion direction—a theoretically significant minimal configuration.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First unified linear solver for asynchronous point tracks; outstanding theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated on simulated and real data across three sensor modalities and multiple sequences; large-scale application is missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear; theoretical analysis is complete; experimental logic is coherent.
- **Value**: ⭐⭐⭐⭐ — Establishes an important theoretical foundation for geometric vision in the event-camera era with lasting impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)
- [\[ICCV 2025\] From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)
- [\[ICCV 2025\] DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding](despite_exploring_contrastive_deep_skeletonpointcloudimutext.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](../../NeurIPS2025/others/infrequent_exploration_in_linear_bandits.md)

</div>

<!-- RELATED:END -->
