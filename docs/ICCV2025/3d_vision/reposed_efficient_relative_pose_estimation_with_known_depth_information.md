---
title: >-
  [Paper Note] RePoseD: Efficient Relative Pose Estimation with Known Depth Information
description: >-
  [ICCV 2025][3D Vision][Relative pose estimation] This paper proposes a set of efficient minimal solvers for relative pose estimation that jointly estimate the scale and affine parameters of monocular depth estimation (MD…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Relative pose estimation"
  - "monocular depth estimation"
  - "minimal solvers"
  - "RANSAC"
  - "multi-view geometry"
date: 2026-05-08
content_hash: cae6b3df32db045c
---

# RePoseD: Efficient Relative Pose Estimation with Known Depth Information

**Conference**: ICCV 2025
**arXiv**: [2501.07742](https://arxiv.org/abs/2501.07742)  
**Code**: Coming soon  
**Area**: 3D Vision
**Keywords**: Relative pose estimation, monocular depth estimation, minimal solvers, RANSAC, multi-view geometry

## TL;DR
This paper proposes a set of efficient minimal solvers for relative pose estimation that jointly estimate the scale and affine parameters of monocular depth estimation (MDE) alongside the relative pose. The proposed solvers outperform state-of-the-art depth-aware solvers across three camera configurations (calibrated / shared focal length / unknown individual focal lengths), and large-scale experiments provide a definitive answer to the question of whether MDE depth actually benefits relative pose estimation.

## Background & Motivation

Relative pose estimation is fundamental to core tasks such as SfM, visual localization, and autonomous navigation. Classical methods rely on 2D-2D point correspondences and epipolar geometry constraints—requiring 5 correspondences for the calibrated case, 6 for shared focal length, and 7–8 for distinct focal lengths. Within the RANSAC framework, the number of required correspondences directly determines the number of iterations, making the reduction of this count a critical optimization target.

Recent MDE methods (e.g., Depth Anything v2, MoGe, UniDepth) have achieved significant accuracy improvements, creating new opportunities to leverage depth information for pose estimation. However, existing depth-aware solvers face three core challenges:

**Large depth noise**: Learned depth predictions carry far greater noise than 2D image measurements.

**Scale/affine ambiguity**: MDE depth is typically defined only up to an unknown scale factor, or even an unknown scale-plus-shift transformation.

**Parameter inconsistency**: The scale/shift parameters may differ between images or even across regions within a single image.

Prior work (e.g., Rel3PT, Madpose) either exploits only relative depth information without fully leveraging depth maps, fails to model the unknown shift, or produces solvers that are computationally inefficient. The **core idea** of this paper is therefore: **to fully exploit the complete information from both depth maps (rather than relative depth alone), jointly estimate the relative pose together with the scale/affine parameters of depth, and design smaller, faster solvers**.

## Method

### Overall Architecture

Given 2D point correspondences and monocular depth estimates for an image pair, the paper establishes a unified parameterization in which the true depth is expressed as an affine transformation of the estimated depth: $\eta_i = s_1(\alpha_i + u)$, $\lambda_i = s_2(\beta_i + v)$, where $s_1, s_2$ are scale parameters and $u, v$ are shift parameters. Substituting into the projection equations yields a unified constraint with unknowns: relative scale $s = s_2/s_1$, shifts $u, v$, rotation $\mathbf{R}$, and translation $\mathbf{t}$.

### Key Designs

1. **3PTsuv Solver (Calibrated Case)**

    - **Function**: Estimates the 9-DOF problem ($s, u, v, \mathbf{R}, \mathbf{t}$) from 3 3D–3D point correspondences.
    - **Mechanism**: Pairwise subtraction eliminates translation; the norm-preserving property of rotation then eliminates $\mathbf{R}$, yielding 3 equations in $s^2, u, v$ only. Substituting $c = s^2$ reduces the degree, and Gauss–Jordan elimination produces a quartic in $u$ with a closed-form solution.
    - **Design Motivation**: Compared with Madpose's $12\times16$ GJ elimination followed by a $4\times4$ eigendecomposition, this solver requires only a $3\times6$ GJ elimination with a closed-form solution, achieving approximately $3\times$ speedup (1.46 μs vs. 4.45 μs).

2. **4PTfsuv Solver (Shared Focal Length Case)**

    - **Function**: Jointly estimates $s, u, v, f$ (10-DOF) from 4 depth-augmented point correspondences.
    - **Mechanism**: Four 3D–3D correspondences yield 6 equations in 4 unknowns; 4 equations are selected and solved via the Gröbner basis method.
    - **Design Motivation**: The GJ elimination matrix is $24\times32$ (vs. Madpose's $36\times44$), yielding approximately $2\times$ speedup (12.5 μs vs. 23.6 μs).

3. **4PTf1,2suv Solver (Distinct Focal Lengths Case)**

    - **Function**: Estimates $s, u, v, f_1, f_2$ (11-DOF) from 4 3D–3D correspondences.
    - **Mechanism**: From the over-determined system of 6 equations in 5 unknowns, 5 equations are selected. The substitution $cf_2 = \tilde{f}_2$ simplifies the polynomials, leading to a $20\times24$ GJ elimination with at most 4 solutions.
    - **Design Motivation**: The matrix size is roughly half that of Madpose ($40\times44$), achieving approximately $3\times$ speedup (6.45 μs vs. 20.2 μs).

4. **Scale-Only (Zero-Shift) Solvers**

    - **Function**: When the depth shift is zero (as with certain MDE methods), simpler solvers such as P3P or the new 3PTfs00 are employed.
    - **Mechanism**: Shift parameters are not modeled, reducing the degrees of freedom of the problem.
    - **Design Motivation**: Experiments show that for networks such as MoGe and UniDepth, omitting the shift model actually yields better results—a finding that contradicts the conclusions reported in the Madpose paper.

### Loss & Training

The paper employs LO-RANSAC (via PoseLib) with Sampson error for scoring and local optimization, using a 2-pixel threshold and a fixed budget of 1000 iterations. Comparisons are also made against Madpose's Hybrid RANSAC scheme, which uses both Sampson error and reprojection error (thresholds of 2 px and 16 px, respectively), at substantially increased computational cost. A key finding is: **even for 3D points, it is advisable to construct the essential/fundamental matrix and measure Sampson error, as this is generally more robust than directly using reprojection error**.

## Key Experimental Results

### Main Results

| Dataset / Depth / Matcher | Solver | Median Error ε(°) ↓ | mAA ↑ | Runtime (ms) ↓ |
|---------------------------|--------|----------------------|-------|----------------|
| ETH3D / MoGe / SP+LG | 5PT | 0.91 | 87.67 | 48.14 |
| ETH3D / MoGe / SP+LG | P3P | 0.91 | 87.67 | 25.72 |
| ETH3D / MoGe / SP+LG | 3PTsuv (M) | 0.89 | 87.71 | 33.45 |
| ETH3D / MoGe / SP+LG | **3PTsuv (ours)** | **0.89** | 87.67 | **22.41** |
| ETH3D / MoGe / SP+LG | 3PTsuv (ours)+H | **0.85** | **88.24** | 554.79 |
| ETH3D / Real / SP+LG | **3PTsuv (ours)+H** | **0.52** | **91.42** | 543.48 |
| ETH3D / UniDepth / RoMA | 3PTsuv (ours) | **0.55** | 91.01 | **83.72** |

### Ablation Study

| Configuration | Key Metric | Remark |
|---------------|-----------|--------|
| With shift (suv) vs. without (s00) | MoGe: s00 is better | High-quality MDE does not require shift modeling |
| Sampson vs. reprojection error | Sampson generally superior | More robust under large depth noise |
| PoseLib vs. Hybrid RANSAC | Hybrid more accurate but 10–30× slower | Accuracy–speed trade-off |
| Comparison of MDE networks | MoGe / UniDepth best | Outperform MiDaS / DA v2 |
| SP+LG vs. RoMA vs. MASt3R | RoMA best overall | Dense matching provides better correspondences |

### Key Findings

- With high-quality depth estimation (MoGe / UniDepth), depth-aware solvers significantly outperform 5PT.
- For metric depth (UniDepth), zero-shift solvers (P3P and 3PTfs00) without shift modeling perform better.
- For scale/affine-invariant depth (MiDaS / DA v2), modeling the shift improves accuracy.
- MASt3R, despite employing expensive non-linear optimization, underperforms RANSAC-based approaches when high-quality depth or correspondences are available.

## Highlights & Insights

1. **Unified framework covering six depth parameterization cases**: The paper systematically analyzes all combinations of known/unknown/shared scale and shift, providing a complete catalog of viable solvers.
2. **Smaller and faster solvers**: Through algebraic simplification, solver matrix sizes are substantially reduced (up to approximately half of Madpose's), with practical speedups of 2–3×.
3. **Challenging existing conclusions**: By evaluating zero-shift solvers not covered by Madpose, the paper demonstrates that "modeling shift is always beneficial" does not universally hold.
4. **Comprehensive experimental coverage**: Five MDE methods × three matchers × three datasets × two RANSAC frameworks, yielding practical guidelines for real-world use.

## Limitations & Future Work

- The paper assumes spatially uniform depth scale within an image, whereas some MDE methods may exhibit spatially varying scale.
- Hybrid RANSAC incurs a significant computational overhead (10–30×), motivating more efficient implementations.
- The work addresses only the two-view case and does not extend to multi-view joint optimization.
- Incorporating P3P into the Hybrid RANSAC scheme may yield further performance gains.

## Related Work & Insights

- Madpose [Yu et al.] is the closest concurrent work; the proposed solvers achieve substantially better efficiency.
- DUSt3R / MASt3R represents the end-to-end 3D reconstruction paradigm, yet their accuracy falls short of classical RANSAC-based solver approaches in certain scenarios.
- Takeaway: Monocular depth estimation is now accurate enough to genuinely improve geometric estimation in practice, but correct solver configuration must be chosen accordingly.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Technically rigorous solver derivations, though the core idea of jointly estimating pose and depth parameters is shared with a concurrent work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Exceptionally broad coverage (5 MDE methods × 3 matchers × 3 datasets × 2 RANSAC frameworks); a benchmark-setting experimental design.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical structure with detailed mathematical derivations.
- **Value**: ⭐⭐⭐⭐ — Provides a practical guide to the question of whether MDE depth benefits pose estimation, with clear engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICCV 2025\] FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation](fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation](boxdreamer_dreaming_box_corners_for_generalizable_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
