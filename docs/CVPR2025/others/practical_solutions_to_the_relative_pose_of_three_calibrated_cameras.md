---
title: >-
  [Paper Note] Practical Solutions to the Relative Pose of Three Calibrated Cameras
description: >-
  [CVPR 2025][Three-view relative pose] This paper addresses the classic challenge of relative pose estimation for three calibrated cameras from four point correspondences in three views (4p3v). It proposes practical solutions based on approximate geometry—using affine camera approximations or mean point correspondence approximations to estimate the relative pose of the first two cameras, and then registering the third camera via P3P. Combined with local optimization…
tags:
  - "CVPR 2025"
  - "Three-view relative pose"
  - "Minimal solver"
  - "Mean point correspondence"
  - "RANSAC"
  - "Multi-view geometry"
date: 2026-05-08
content_hash: e5d9427c50f26ad3
---

# Practical Solutions to the Relative Pose of Three Calibrated Cameras

**Conference**: CVPR 2025  
**arXiv**: [2303.16078](https://arxiv.org/abs/2303.16078)  
**Code**: [https://github.com/kocurvik/threeview](https://github.com/kocurvik/threeview)  
**Area**: LLM Evaluation  
**Keywords**: Three-view relative pose, Minimal solver, Mean point correspondence, RANSAC, Multi-view geometry

## TL;DR

This paper addresses the classic challenge of relative pose estimation for three calibrated cameras from four point correspondences in three views (4p3v). It proposes practical solutions based on approximate geometry—using affine camera approximations or mean point correspondence approximations to estimate the relative pose of the first two cameras, and then registering the third camera via P3P. Combined with local optimization, this approach achieves SOTA accuracy on real-world data.

## Background & Motivation

Camera geometry estimation is a core problem in many computer vision applications (visual navigation, SfM, augmented reality, autonomous driving, and visual localization). Within the RANSAC framework, estimating the model using as few correspondence points as possible is crucial, since the number of RANSAC iterations grows exponentially with the size of the required minimal sample.

The relative pose estimation of three calibrated cameras (particularly under the 4p3v configuration—4 point correspondences visible in all 3 views) is a fundamental yet highly challenging problem. This problem has 272 theoretical solutions, and the corresponding algebraic equations are extremely complex. The existing SOTA method (Hruby's homotopy continuation method) has a success rate of only 26.3% on noise-free data and is complex to implement; another method (Nister's epipole search method) has no publicly available implementation.

In contrast, the 5pt+P3P solver adopts a simple strategy of "estimating the two-view pose first, then registering the third view", which works well but requires 5 corresponding points. The core idea of this paper is: **Can we estimate the geometric relationship between the first two cameras using an approximate method when only 4 correspondences are available, and then utilize P3P to register the third camera?**

## Method

### Overall Architecture

The method decomposes the 4p3v problem into two steps: (1) using 4 point correspondences to estimate the approximate relative pose of the first two cameras; (2) using 3 triangulated 3D points to register the third camera via a P3P solver. Two types of approximation schemes are proposed: affine approximation (4p3v(A)) and mean point approximation (4p3v(M)), and multiple enhancement strategies are designed to improve accuracy.

### Key Designs

1. **Affine Approximation Solver 4p3v(A)**:
    - - Function: Approximates the geometry of the first two views using an affine camera model.
    - - Mechanism: Uses 4 point correspondences to estimate the affine fundamental matrix $\mathbf{F}_A$ (solved linearly), triangulates 3 points, and then registers the third camera using P3P.
    - - Design Motivation: The affine fundamental matrix can be solved linearly from 4 points, making the computation extremely fast. Although the affine approximation can be coarse, prior work has shown that it can still achieve good accuracy when combined with RANSAC and local optimization.

2. **Mean Point Approximation Solver 4p3v(M)**:
    - - Function: Converts the 4p3v problem into a 5pt+P3P style problem by generating an approximate 5th point correspondence.
    - - Mechanism: Under the parallel perspective projection assumption, the mean of three 3D points projects to the mean points in both images. Therefore, the image mean points of 3 correspondences in both views are taken as the 5th approximate correspondence $\mathbf{m}^1 \leftrightarrow \mathbf{m}^2$, and then the standard 5pt solver + P3P are called.
    - - Design Motivation: The mean point approximation can be viewed as a first-order homography approximation, with guaranteed error upper bounds—the epipolar line must pass through the interior of the corresponding triangle, meaning the maximum distance from the mean point to the epipolar line is constrained by the triangle size. This approximation is much more accurate than the affine approximation and does not require complex additional implementation.

3. **Improvement Strategies (Making Approximate Solvers Practical)**:
    - - Function: Boosts the robustness and accuracy of the approximate solvers within RANSAC.
    - - Mechanism: 
        - **4p3v(M±δ)**: Generates two additional offset points near the mean point and calls the 5pt solver 3 times to increase the diversity of candidate solutions.
        - **Early Non-Minimal Refitting (ENM)**: After inliers are detected using the approximate geometry, a non-minimal 5pt solver is used to re-estimate a more precise two-view geometry.
        - **Fourth Point Filtering (+F)**: Filters unreasonable solutions using the 4th point in the third view.
        - **Fourth Point Refinement (+R)**: Minimizes epipolar errors on the 4p3v configuration using Levenberg-Marquardt (LM) optimization. Just 2 iterations can significantly improve accuracy.
    - - Design Motivation: Pure approximation can lead to error propagation into triangulation and the third camera registration. These improvement strategies can significantly improve accuracy without substantially increasing computational overhead.

### Loss & Training

This work is a geometric method rather than a learning-based method. The evaluation metrics are pose errors (the maximum of rotation and translation angular errors) and AUC values at different thresholds. Evaluation is conducted within RANSAC frameworks (PoseLib and GC-RANSAC) using SuperPoint features and the LightGlue matcher to extract correspondences.

## Key Experimental Results

### Main Results

PhotoTourism dataset (Average of 12 scenes, PoseLib RANSAC), AUC@10°:

| Method | AUC@10° | AVG Error(°) | Median Error(°) | Runtime (ms) |
|------|---------|-----------|-----------|------------|
| 4p3v(HC) [Hruby] | 73.53 | 16.37 | 11.62 | 164.10 |
| 5pt+P3P (Baseline) | 74.30 | 15.40 | 11.60 | 133.77 |
| 4p3v(M+δ)+R+F+ENM | **75.90** | **15.10** | **11.50** | 138.12 |

Cambridge Landmarks dataset:

| Method | AUC@10° | AVG Error(°) | Runtime (ms) |
|------|---------|-----------|------------|
| 4p3v(HC) | 64.58 | 18.73 | 60.11 |
| 5pt+P3P | 65.33 | 17.46 | 24.04 |
| 4p3v(M+δ)+R+F+ENM | **66.83** | **16.98** | 39.39 |

### Ablation Study

Two-view accuracy (Median rotation error on PhotoTourism, in degrees):

| Configuration | Median Error | Description |
|------|---------|------|
| 5pt (Standard) | ~15° | Baseline |
| 4p(A) (Affine, without ENM) | ~70° | Affine approximation is too coarse |
| 4p(M) (Mean point, without ENM) | ~20° | Mean point approximation is significantly better than affine |
| 4p(M±δ) (Mean point + offset) | ~16° | Close to 5pt accuracy |
| 4p(M±δ)+ENM | ~15° | Almost identical to 5pt+ENM |

Ablation on the choice of the offset $\delta$: $\delta = 0.08 \times \text{(longest triangle side)}$ performs best, though performance is insensitive to this parameter.

### Key Findings

- The mean point approximation (M-series) significantly outperforms the affine approximation (A-series) across all scenes, being more robust to scene geometry and RANSAC thresholds.
- Enhanced with ENM refitting and +R refinement, the 4p3v(M) series of solvers outperform both the 5pt+P3P baseline and the HC solver in most scenes.
- The optimal location for the mean point correspondence is indeed close to the triangular barycentric center (mean barycentric coordinates of (0.33, 0.33)), which experimentally validates the theoretical analysis.
- While the performance of 4p3v(A) can approach that of 4p3v(M) after LO-RANSAC, 4p3v(M) remains more stable.
- In large-scale experiments involving 90,000 camera triplets, the proposed method consistently outperforms existing SOTA approaches.

## Highlights & Insights

- **Design Philosophy of Simplifying Complexity**: Translates a complex algebraic problem with 272 potential solutions into an approximation problem using existing, mature solvers (5pt+P3P). The concept is elegant and engineering-friendly.
- **Theoretical Guarantees for Mean Point Approximation**: Proves that the epipolar line must cross the corresponding triangle, yielding an upper bound on the mean point error, thereby providing a mathematical foundation for the reliability of the approximation.
- **Modular Improvement Strategies**: Enhancements like ENM, +F, and +R are independent and orthogonal, allowing flexible combination and high practicality.
- **First Large-Scale Real-Data Evaluation**: Systematically evaluates the 4p3v problem across 18 real-world scenes and 90,000 triplets, filling a gap in the literature.
- **Ease of Implementation**: The entire pipeline relies on existing highly-efficient solvers without requiring the training of MLP classifiers or complex homotopy continuation.

## Limitations & Future Work

- The accuracy of 4p3v(M) depends on scene geometry (such as depth distribution and inter-camera angles); under certain extreme configurations, the approximation error can be large.
- It remains an approximate solution rather than an exact minimal solver for the (4,4,4) configuration.
- The error of the affine approximation increases significantly when there are large angles between the camera optical axes.
- Future work could explore better strategies for generating the 5th point correspondence (e.g., methods based on local homographies).
- This idea can be extended to more complex configurations involving unknown focal lengths or generalized cameras in the future.

## Related Work & Insights

- **vs 4p3v(HC) [Hruby 2022]**: The HC method requires training an MLP classifier to choose initial solutions, suffering from a low success rate (26.3%) and a complicated implementation. In contrast, the proposed method is simpler, more robust, and achieves higher accuracy on real-world data.
- **vs Nister's Epipole Search**: Nister's method requires searching 1000 points on a degree-10 curve and lacks a public implementation. This method uses the mean point approximation, which comes with guaranteed error upper bounds.
- **vs 5pt+P3P Baseline**: The 5pt+P3P solver requires a (5,5,3) configuration instead of (4,4,4). By generating an approximate correspondence, this work converts the (4,4,4) configuration into (5,5,3), yielding higher accuracy when combined with optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using mean point approximation to generate the 5th correspondence is ingenious and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The systematic evaluation on 3 datasets, 18 scenes, and 90,000 triplets is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly stated, and the improvement strategies are presented in a progressive manner.
- Value: ⭐⭐⭐⭐ Provides the first truly practical solution for the classic 4p3v problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Order-One Rolling Shutter Cameras](order-one_rolling_shutter_cameras.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](scene-agnostic_pose_regression_for_visual_localization.md)
- [\[CVPR 2025\] Three-View Focal Length Recovery From Homographies](three-view_focal_length_recovery_from_homographies.md)
- [\[CVPR 2025\] Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](full-dof_egomotion_estimation_for_event_cameras_using_geometric_solvers.md)
- [\[CVPR 2025\] Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)

</div>

<!-- RELATED:END -->
