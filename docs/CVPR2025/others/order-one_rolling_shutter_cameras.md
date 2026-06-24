---
title: >-
  [Paper Note] Order-One Rolling Shutter Cameras
description: >-
  [CVPR 2025][Rolling shutter] A unified theory of Order-One Rolling Shutter (RS1) cameras is proposed, proving the mathematical characterization of rolling shutter camera classes that map a spatial point to exactly one image point, constructing explicit parametrizations, and providing a complete classification of the 31 relative pose minimal problems for linear RS1 cameras.
tags:
  - "CVPR 2025"
  - "Rolling shutter"
  - "multi-view geometry"
  - "camera model"
  - "minimal problems"
  - "algebraic geometry"
date: 2026-05-08
content_hash: c0e847c09d66c289
---

# Order-One Rolling Shutter Cameras

**Conference**: CVPR 2025  
**arXiv**: [2403.11295](https://arxiv.org/abs/2403.11295)  
**Code**: None  
**Area**: Computer Vision / Camera Geometry  
**Keywords**: Rolling shutter, multi-view geometry, camera model, minimal problems, algebraic geometry

## TL;DR
A unified theory of Order-One Rolling Shutter (RS1) cameras is proposed, proving the mathematical characterization of rolling shutter camera classes that map a spatial point to exactly one image point, constructing explicit parametrizations, and providing a complete classification of the 31 relative pose minimal problems for linear RS1 cameras.

## Background & Motivation

**Background**: Rolling shutter (RS) cameras dominate consumer and smartphone markets, where row-by-row scanning leads to image distortion. Over the past 20 years, various absolute pose calculation methods for RS cameras have emerged, but relative pose problems remain largely unresolved. Existing multi-view geometry theories are based on global shutter (GS) cameras.

**Limitations of Prior Work**: (1) General RS cameras project a single point in space into multiple image points, making standard perspective projection multi-view geometry inapplicable; (2) existing RS pose estimation methods are fragmented and tailored to specific motion models, lacking a unified theoretical framework; (3) minimal problems for RS relative pose have not been systematically classified.

**Key Challenge**: RS cameras are ubiquitous (mobile phones, cars, drones), but their multi-view geometry theory is significantly less mature than that of GS cameras—directly applying GS theory in moving scenes leads to severe errors.

**Goal**: Establish a systematic mathematical theory for a class of important RS cameras (RS1 cameras, where a spatial point projects to exactly one image point).

**Key Insight**: Utilize algebraic geometry tools to formalize the projection process of RS cameras as a rational map, defining and studying RS1 cameras in terms of the degree of the mapping.

**Core Idea**: Formalize the back-projection of the RS camera as a parameterized mapping $\Lambda$, where the inverse mapping $\Phi = \Lambda^{-1}$ exists and is rational if and only if the camera is RS1 $\rightarrow$ this derives the geometric characterization that all rolling shutter planes of an RS1 camera intersect at a spatial line $K$, from which all minimal problems are further classified.

## Method

### Overall Architecture
This is a theoretical work divided into four parts: (1) back-projection models for RS cameras $\rightarrow$ (2) mathematical characterization of RS1 cameras $\rightarrow$ (3) explicit parameterization of RS1 cameras $\rightarrow$ (4) complete classification of minimal problems.

### Key Designs

1. **Back-Projection RS Camera Model**:

    - Function: Unify the description of ray-to-image correspondences in RS cameras.
    - Mechanism: Define a ray mapping $\Lambda$ that maps each image point to the spatial ray projecting to that point, and a rolling shutter plane mapping $\Sigma$ that maps each scanline to its corresponding camera plane. The combination of $\Lambda$ and $\Sigma$ provides a complete geometric description of the RS camera.
    - Design Motivation: Existing RS models are scattered across different papers using different conventions; a unified back-projection framework makes theoretical analysis possible.

2. **RS1 Camera Characterization (Theorem 4)**:

    - Function: Provide the necessary and sufficient geometric conditions for an RS1 camera.
    - Mechanism: Prove that an RS camera is RS1 if and only if all its rolling shutter planes intersect at a single spatial line $K$. Furthermore, $\Sigma$ is birational, the camera center moves along a curve $\mathcal{C}$ that is either equal to $K$ or intersects $K$ at $\deg(\mathcal{C})-1$ points.
    - Design Motivation: Geometric characterization provides a theoretical foundation for constructing RS1 camera parameterizations and determining whether practical scenarios satisfy the RS1 condition.

3. **Complete Classification of Minimal Problems**:

    - Function: Enumerate all minimal problems of relative pose for linear RS1 cameras.
    - Mechanism: Systematically enumerate all minimal problems under point/line correspondences (including incidence constraints) for 2 to 5 linear RS1 cameras, yielding exactly 31 problems. Calculate the degree (number of solutions) for each problem. Several practical problems are discovered: (a) two cameras with 7/9 point correspondences, with degrees of 140/364, solvable via homotopy continuation; (b) 3 point + 2 line correspondences, with a degree of only 28, suitable for symbolic-numeric minimal solvers.
    - Design Motivation: Minimal problems are at the core of robust estimation frameworks like RANSAC—knowing all minimal problems and their degrees allows for selecting the optimal problem configuration.

### Loss & Training
Purely theoretical work, no training involved.

## Key Experimental Results

### Main Results

| Number of Cameras | Number of Minimal Problems | Lowest Degree | Example Practical Problem |
|---|---|---|---|
| 2 | Multiple | 28 | 3 points + 2 lines (degree 28), 9 points (degree 364) |
| 3 | Multiple | 160 | Specific point-line configurations |
| 4-5 | Multiple | Large | To be further decomposed and simplified |
| 1 / >5 | **0** | - | No minimal problems exist |

A total of 31 minimal problems were discovered, all of which are novel.

### Ablation Study

| Special Case | Description |
|---|---|
| Constant-rotation RS1 | Corresponds to Straight-Cayley cameras |
| Pure translation with constant velocity | Corresponds to linear RS cameras (Dai et al.) |
| General RS1 | Encompasses both + more scenarios |

### Key Findings
- The intersection of the rolling shutter planes of an RS1 camera at a single line is an elegant geometric invariant that can be used to quickly determine whether a practical scenario conforms to the RS1 assumption.
- The image of a spatial line under RS1 is a rational curve whose degree equals the degree of the projection mapping $\Phi$, and this curve passes through a special point at infinity $\deg(\Phi)-1$ times—this can be used to simplify relative pose estimation.
- No minimal problems exist for a single RS1 camera, meaning a single RS image must rely on other constraints.

## Highlights & Insights
- **Elegant Application of Algebraic Geometry to Practical Vision Problems**: Systems of RS camera theories are formalized using the language and tools of projective algebraic geometry, which not only explains prior work but also uncovers new minimal problems.
- RS1 cameras cover several practical scenarios (moving vehicles/drones with constant linear motion, flatbed scanners, push-broom satellite imaging), giving the theory immediate application prospects.
- The complete classification of 31 minimal problems provides a "menu" for future research in RS-SLAM.

## Limitations & Future Work
- RS1 only covers situations where a spatial point projects to a single image point; general RS motion (such as rapid rotation) does not satisfy this condition.
- Solver implementation for the minimal problems is not yet complete (the paper leans heavily towards theory).
- The degree of minimal problems for 4-5 cameras is very large; whether they can be decomposed into simpler sub-problems remains an open question.
- How to quickly determine if the RS1 condition is satisfied in practical applications requires further work.

## Related Work & Insights
- **vs Dai et al. (2016)**: Defined linear RS cameras (pure translation with constant velocity), which RS1 generalizes.
- **vs Albl et al. (2019)**: Studied degenerate cases of RS cameras, which the RS1 theory covers and explains under a unified framework.
- **vs GS Multi-View Geometry**: RS1 is a natural generalization of GS perspective cameras to RS—the perspective camera is a special case of RS1.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Established a systematic theory of RS1 cameras, filling a major gap in RS multi-view geometry.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical contribution, lacking numerical validation and real-world application experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorously derived mathematical proofs, with a clear theoretical structure.
- Value: ⭐⭐⭐⭐ Provides a long-overdue theoretical foundation for SLAM/SfM with RS cameras.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)
- [\[CVPR 2025\] Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](full-dof_egomotion_estimation_for_event_cameras_using_geometric_solvers.md)
- [\[AAAI 2026\] Higher-Order Responsibility](../../AAAI2026/others/higher-order_responsibility.md)
- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](../../NeurIPS2025/others/influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[NeurIPS 2025\] DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets](../../NeurIPS2025/others/dpa_a_one-stop_metric_to_measure_bias_amplification_in_classification_datasets.md)

</div>

<!-- RELATED:END -->
