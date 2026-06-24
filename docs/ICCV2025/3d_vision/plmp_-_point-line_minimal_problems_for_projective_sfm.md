---
title: >-
  [Paper Note] PLMP -- Point-Line Minimal Problems for Projective SfM
description: >-
  [ICCV 2025][3D Vision][Minimal problems] This paper provides a complete classification of all point-line minimal problems in projective SfM, identifying 291 minimal problems (73 of which admit unique solutions solvable by linear methods), and develops a systematic framework for problem decomposition and non-minimality proofs via stabilizer subgroup analysis.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Minimal problems"
  - "SfM"
  - "point-line configurations"
  - "projective reconstruction"
  - "algebraic geometry"
date: 2026-05-08
content_hash: a238090873dd2bb2
---

# PLMP -- Point-Line Minimal Problems for Projective SfM

**Conference**: ICCV 2025
**arXiv**: [2503.04351](https://arxiv.org/abs/2503.04351)  
**Code**: N/A  
**Area**: 3D Vision
**Keywords**: Minimal problems, SfM, point-line configurations, projective reconstruction, algebraic geometry

## TL;DR

This paper provides a complete classification of all point-line minimal problems in projective SfM, identifying 291 minimal problems (73 of which admit unique solutions solvable by linear methods), and develops a systematic framework for problem decomposition and non-minimality proofs via stabilizer subgroup analysis.

## Background & Motivation

**Background**: Structure-from-Motion (SfM) is a fundamental problem in computer vision, with the core task of recovering camera poses and 3D scene structure from multiple images. Minimal problems serve as the basic building blocks of SfM — they define configurations that are exactly constrained (neither under- nor over-determined) given a particular feature type and number of cameras. Robust estimation frameworks such as RANSAC rely on minimal solvers to efficiently recover geometric models from noisy data.

**Limitations of Prior Work**: For calibrated cameras (known intrinsics), the classification of minimal problems is relatively mature (e.g., the five-point method, P3P). However, for uncalibrated cameras (projective SfM), particularly in scenarios involving both point and line features simultaneously, no systematic classification exists. Prior work either addresses purely point-based or purely line-based configurations, or handles only specific numbers of views, lacking a unified theoretical framework for enumerating all possible minimal configurations.

**Key Challenge**: The search space for minimal problems in mixed point-line configurations is enormous — it requires considering varying numbers of points, lines, views, and visibility patterns specifying which features are observed in which views. Brute-force enumeration is infeasible; mathematical tools must be developed to systematically classify and analyze these configurations.

**Goal**: (1) Completely enumerate all point-line minimal problems for uncalibrated cameras; (2) compute the number of solutions for each minimal problem as a measure of its intrinsic difficulty; (3) develop systematic methods for problem decomposition and non-minimality proofs.

**Key Insight**: The authors reformulate the classification of minimal problems as a problem in algebraic geometry — each point-line configuration defines an algebraic variety, and minimality is equivalent to this variety being zero-dimensional (finitely many solutions). By analyzing the stabilizer subgroups of sub-arrangements of a configuration, the decomposition structure and over/under-constrained sub-problems can be identified automatically.

**Core Idea**: Employ dimension counting and stabilizer subgroup analysis from algebraic geometry to systematically determine minimality and classify all valid point-line visibility configurations in projective SfM.

## Method

### Overall Architecture

The method takes as input a set of complete observation configurations of points and lines across multiple uncalibrated pinhole cameras (all points and lines visible in all views), and outputs a complete list of all configurations satisfying minimality conditions together with their solution counts. The overall pipeline consists of three steps: (1) identifying candidate minimal configurations via dimension counting; (2) verifying minimality and computing solution counts via algebraic geometry; (3) performing problem decomposition and eliminating non-minimal configurations via stabilizer subgroup analysis.

### Key Designs

1. **Dimension Counting and Candidate Filtering**:

    - Function: Rapidly identify all point-line-view configurations that may constitute minimal problems.
    - Mechanism: For a configuration with $m$ views, $p$ 3D points, and $l$ 3D lines, the unknowns consist of $m$ camera matrices of size $3\times4$ (with $11m$ degrees of freedom, minus 15 for projective ambiguity) and $3p + 4l$ structural parameters. Observational constraints contribute 2 equations per point per view and 2 equations per line per view. The minimality condition requires the number of constraints to equal the number of unknowns: $2m(p + l) = 11m - 15 + 3p + 4l$. Candidate configurations are obtained by enumerating all non-negative integer solutions $(m, p, l)$ satisfying this equation.
    - Design Motivation: Dimension counting is the classical starting point for classifying minimal problems, providing a necessary condition that substantially reduces the search space.

2. **Algebraic Variety Analysis and Solution Count Computation**:

    - Function: Verify that candidate configurations are indeed minimal problems (rather than degenerate cases that satisfy the dimension count but fail in practice), and compute the number of solutions for each minimal problem.
    - Mechanism: For each candidate configuration, the corresponding algebraic system of equations is constructed and solved on random data using numerical algebraic geometry methods (e.g., homotopy continuation), verifying that the solution set is finite (a zero-dimensional variety) and computing the number of solutions (equal to the algebraic degree). The solution count is a direct measure of a problem's intrinsic difficulty — fewer solutions indicate an easier problem, and problems with a unique solution (degree 1) are solvable by linear methods.
    - Design Motivation: The solution count directly determines the practical utility of a minimal solver — in RANSAC, fewer solutions allow hypotheses to be rejected more rapidly. The 73 linearly solvable problems are particularly valuable for their computational efficiency.

3. **Stabilizer Subgroup Analysis and Problem Decomposition**:

    - Function: Discover the intrinsic decomposition structure of minimal problems and systematically prove the non-minimality of certain configurations.
    - Mechanism: For each sub-arrangement of a configuration (e.g., a subset of points, lines, or views), the stabilizer subgroup is analyzed — that is, the group of projective transformations that leave the sub-arrangement invariant. A non-trivial stabilizer subgroup indicates redundancy in the partial constraints (degrees of freedom absorbed by symmetry), making the actual degrees of freedom larger than naive counting suggests. This yields three applications: (a) decomposing large problems into products of independent smaller ones; (b) identifying minimal sub-problems within under-constrained configurations; (c) rigorously proving that certain candidate configurations are non-minimal by revealing hidden symmetries.
    - Design Motivation: This is the paper's deepest theoretical contribution. Previous methods for proving non-minimality were largely ad hoc; stabilizer subgroup analysis provides a unified and mechanizable framework.

### Loss & Training

This is a purely theoretical and mathematical work with no training involved. Numerical verification employs randomly generated point-line configurations solved via homotopy continuation solvers.

## Key Experimental Results

### Main Results

Distribution statistics for the 291 minimal problems:

| Number of Cameras | Total Minimal Problems | Linearly Solvable (Unique Solution) | Min Solutions | Max Solutions | Max Points | Max Lines |
|---|---|---|---|---|---|---|
| 2 | 15 | 8 | 1 | 12 | 7 | 6 |
| 3 | 52 | 18 | 1 | 156 | 6 | 8 |
| 4 | 71 | 16 | 1 | 640 | 5 | 10 |
| 5 | 58 | 12 | 1 | 1024 | 4 | 11 |
| 6–9 | 95 | 19 | 1 | 2048+ | 3 | 12 |
| **Total** | **291** | **73** | — | — | **7** | **12** |

### Ablation Study (Solution Counts for Selected Configurations)

| Configuration ($p$ points, $l$ lines, $m$ views) | Solution Count | Linearly Solvable | Notes |
|---|---|---|---|
| (7, 0, 2) — classical 7-point method | 3 | No | Points only, two views |
| (0, 6, 3) — pure-line three-view | 1 | Yes | Lines only, unique solution |
| (3, 2, 3) — 3 points + 2 lines, three views | 1 | Yes | Mixed, linearly solvable |
| (1, 4, 3) | 4 | No | Mixed configuration |
| (5, 0, 3) | 10 | No | Points only, three views |
| ($p$, $l$, $\infty$) — two infinite-view problems | 1 | Yes | Admits arbitrarily many views |

### Key Findings

- All minimal problems involve at most 9 cameras, 7 points, and 12 lines — the search space is far smaller than initially expected.
- Many of the 73 linearly solvable problems involve mixed point-line configurations, indicating that exploiting line features can yield additional highly efficient solvers and serve as an important complement to traditional point-only SfM pipelines.
- Two minimal problems admit arbitrarily many views while remaining linearly solvable — a theoretically elegant result that allows all available views to be utilized in a linear solve.
- Stabilizer subgroup analysis successfully eliminates a large number of spurious candidates (configurations satisfying dimension counting but failing actual minimality), ensuring that all 291 results carry rigorous mathematical guarantees.
- Minimal problems with low solution counts (≤10) are most practically valuable in RANSAC applications, as they allow rapid hypothesis verification.

## Highlights & Insights

- **Theoretical Completeness of a Full Classification**: This is the first exhaustive classification of all point-line minimal problems in projective SfM. The complete list of 291 problems provides a roadmap for the development of new minimal solvers in the future.
- **Stabilizer Subgroup Analysis as a General Tool**: The introduction of algebraic group theory into the study of minimal problems yields a unified framework for decomposition, identification, and proof. This methodology is extensible to other geometric estimation problems (e.g., calibrated cameras, generalized cameras).
- **Practical Value of Linearly Solvable Problems**: The 73 unique-solution problems can be directly translated into linear minimal solvers, each representing a potential new tool within the RANSAC framework.

## Limitations & Future Work

- The current work only handles the fully visible case (all features observed in all views); partial visibility due to occlusion or field-of-view limitations in real scenes would give rise to additional minimal problem variants.
- The paper is primarily a mathematical classification; numerical solvers for all 291 minimal problems have not yet been implemented — bridging the gap from theory to practically deployable RANSAC solvers requires substantial engineering effort.
- Camera intrinsic constraints (e.g., known focal length, principal point at image center) are not considered; extending the classification to semi-calibrated camera settings is a natural next step.
- Line feature detection and matching in real images remains more challenging than point feature counterparts, limiting the practical applicability of mixed point-line minimal problems.
- Future work could integrate these minimal solvers into modern SfM pipelines (e.g., COLMAP) and evaluate performance gains on real-world reconstruction benchmarks.

## Related Work & Insights

- **vs. Classification of Minimal Problems for Calibrated Cameras (Stewenius et al.)**: Classical results such as the five-point method for calibrated cameras are widely used. This paper extends analogous systematic classification to the more general uncalibrated (projective) setting — a more complex problem, but one that follows the same methodological lineage.
- **vs. Point-Only Minimal Problems**: Classical methods such as the 7-point and 8-point algorithms rely solely on point features. This paper demonstrates that incorporating line features enables additional efficient solvers — for instance, the (3, 2, 3) configuration requires only 3 points and 2 lines to achieve a linear solution across 3 views, offering greater flexibility than point-only approaches.
- **vs. Real-Time SfM Systems (COLMAP, ORB-SLAM)**: These systems rely primarily on point-feature minimal solvers. The classification results presented in this paper provide the theoretical foundation for incorporating line features into such systems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First exhaustive classification of point-line minimal problems in projective SfM; the stabilizer subgroup methodology is a strong methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ All 291 solution counts are numerically verified, but validation on real data or within practical SfM pipelines is absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous and clear, though the presentation is demanding for readers without a background in algebraic geometry.
- Value: ⭐⭐⭐⭐ Provides important theoretical infrastructure for the SfM community, though translating the results into practical use requires further follow-up work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Solving Minimal Problems Without Matrix Inversion Using FFT-Based Interpolation](../../CVPR2026/3d_vision/solving_minimal_problems_without_matrix_inversion_using_fft-based_interpolation.md)
- [\[CVPR 2025\] PRaDA: Projective Radial Distortion Averaging](../../CVPR2025/3d_vision/prada_projective_radial_distortion_averaging.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [\[CVPR 2026\] Minimal Constraint Relaxation for Multiview Autocalibration](../../CVPR2026/3d_vision/minimal_constraint_relaxation_for_multiview_autocalibration.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
