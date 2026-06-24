---
title: >-
  [Paper Note] Three-View Focal Length Recovery From Homographies
description: >-
  [CVPR 2025][Focal length recovery] An efficient solver is proposed to recover focal lengths from three-view homographies. By leveraging normal consistency constraints to derive new explicit constraints, the problem is formulated as solving univariate or bivariate polynomials, achieving speedups of 80x-270x over existing methods.
tags:
  - "CVPR 2025"
  - "Focal length recovery"
  - "three-view homography"
  - "self-calibration"
  - "polynomial solvers"
  - "planar scenes"
date: 2026-05-08
content_hash: b38dc5c60d8631ed
---

# Three-View Focal Length Recovery From Homographies

**Conference**: CVPR 2025  
**arXiv**: [2501.07499](https://arxiv.org/abs/2501.07499)  
**Code**: [GitHub](https://github.com/kocurvik/hf)  
**Area**: Other  
**Keywords**: Focal length recovery, three-view homography, self-calibration, polynomial solvers, planar scenes

## TL;DR

An efficient solver is proposed to recover focal lengths from three-view homographies. By leveraging normal consistency constraints to derive new explicit constraints, the problem is formulated as solving univariate or bivariate polynomials, achieving speedups of 80x-270x over existing methods.

## Background & Motivation

Recovering camera intrinsic parameters from multiple views is a classic problem in computer vision. In planar scenes (commonly found in man-made environments such as floors, walls, and doors), homography is a fundamental tool for estimating relative camera poses:
- **Two-view homography** cannot provide additional focal length constraints—any choice of intrinsic parameters can achieve an arbitrary homography by appropriately placing the views and the plane.
- **Three-view homography** can theoretically recover focal lengths, but existing methods are highly inefficient.
- Heikkilä's method requires computing eigenvalues of an $82 \times 82$ (equal focal lengths) or a $176 \times 176$ (unequal focal lengths) matrix, with execution times of 1404μs and 5486μs, respectively.
- The three-view focal length problem for general scenes is more complex (up to 668 solutions), and solvers using homotopy continuation take 16.7-154ms, making them unsuitable for practical applications.
- The key observation of this paper: the two homographies defined by three views share the same planar normal vector $\mathbf{n}$, which provides additional constraints.

## Method

### Overall Architecture

Given four coplanar points on the same plane observed by three cameras, two 2D homographies $\mathbf{G}_2$ and $\mathbf{G}_3$ are obtained. The core idea is to leverage the consistency of the normal vector $\mathbf{n}$ in the Euclidean homography $\mathbf{H}_j = \mathbf{R}_j + \frac{\mathbf{t}_j}{d}\mathbf{n}^\top$. By using elimination methods, polynomial constraints depending solely on the focal length parameters are derived, which are then solved efficiently using Sturm sequences or hidden variable techniques.

### Key Design 1: Normal Consistency Constraints

**Function**: Extracting focal length constraints from the shared normal vector of two homographies.

**Core Idea**: Consider the transposed Euclidean homography $\mathbf{H}_j^\top = \mathbf{R}_j^\top + \frac{\mathbf{n}}{d}\mathbf{t}_j^\top$, where the corresponding essential matrix $\tilde{\mathbf{E}}_j = [\mathbf{n}]_\times \mathbf{R}_j^\top$ is related to the same normal vector $\mathbf{n}$. Applying the orthogonality of rotation matrices yields the key identity $[\mathbf{n}]_\times \mathbf{Q}_j [\mathbf{n}]_\times^\top = s_j [\mathbf{n}]_\times [\mathbf{n}]_\times^\top$, where $\mathbf{Q}_j = (\mathbf{K}_j^{-1}\mathbf{G}_j\mathbf{K}_1)^\top(\mathbf{K}_j^{-1}\mathbf{G}_j\mathbf{K}_1)$, resulting in 12 equations. By using Macaulay2 computer algebra to eliminate the unknown normal vector $(n_x, n_y)$ and scale factors $(s_2, s_3)$, 7 polynomial constraints of degree 6 depending only on the elements of $\mathbf{Q}_j$ are obtained.

**Design Motivation**: Compared to directly eliminating variables from the trace constraints (which involve 18 elements of $\mathbf{H}_j$), leveraging the constraints on the 12 elements of the symmetric matrix $\mathbf{Q}_j$ is more compact and efficient.

### Key Design 2: Sturm Sequence Solver for Single Unknown Cases

**Function**: Efficiently solving for cases with three cameras sharing the same focal length (Case I) and equal focal lengths with a known reference focal length (Case II).

**Core Idea**: Substituting $\mathbf{K}_j$ into the generators of the elimination ideal: Case I yields a 9th-degree univariate polynomial in $\alpha = f^2$ (with 7 overdetermined equations, taking 1 is sufficient), and Case II yields a 6th-degree univariate polynomial. Sturm sequences are applied to locate all real roots efficiently, with a time complexity significantly lower than that of matrix eigenvalue decomposition.

**Design Motivation**: To avoid the high computational cost of the $82 \times 82$ matrix eigenvalue decomposition in Heikkilä's method. Sturm sequences are extremely efficient for root-finding of univariate polynomials (taking approximately 17-19μs).

### Key Design 3: Hidden Variable Technique for Two Unknowns Cases

**Function**: Solving Case III with unequal focal lengths ($f_1=f, f_2=f_3=\rho$) and Case IV (where $f_1$ is known, and $f_2, f_3$ are different).

**Core Idea**: Taking Case III as an example, the 7 polynomials contain two unknowns $\alpha=f^2, \beta=\rho^2$, with a maximum degree of $\alpha^3\beta^6$. Choosing $\alpha$ as the hidden variable, the system is rewritten as $\mathbf{C}(\alpha)\tilde{\mathbf{v}} = \mathbf{0}$, where $\mathbf{C}(\alpha) = \alpha^3\mathbf{C}_3 + \alpha^2\mathbf{C}_2 + \alpha\mathbf{C}_1 + \mathbf{C}_0$. Utilizing the property that $\mathbf{C}_3$ is of rank 4, transposed operations and the variable substitution $\gamma=1/\alpha$ are applied to eliminate zero eigenvalues, reformulating the problem as an eigenvalue problem of an $18 \times 18$ matrix.

**Design Motivation**: Compared to Heikkilä's $176 \times 176$ matrix, the $18 \times 18$ eigenvalue problem is 27 times faster, making practical application feasible.

### Loss & Training

Since this paper presents algebraic solvers, it does not involve neural network training losses. Symmetric transfer error is utilized as the inlier threshold within the RANSAC framework.

## Key Experimental Results

### Main Results: Solver Efficiency Comparison

| Case | Solver | No. of Solutions | Matrix Size | Time (μs) |
|------|--------|---------|---------|---------|
| I (fff) | Ours | 9 | Sturm | **17.3** |
| I (fff) | Heikkilä | 70 | 82×82 | 1404 |
| III (fρρ) | Ours | 17 | 18×18 | **200** |
| III (fρρ) | Heikkilä | 152 | 176×176 | 5486 |

### Ablation Study: Synthetic Data Accuracy

| Method | Median Error (deg) in Case I | Median Error (deg) in Case III |
|------|-------------------|---------------------|
| Ours $\mathbf{H}_{fff}$ | **Best** | - |
| Heikkilä $\mathbf{H}_{fff}$ | Slightly worse | - |
| 6pt Two-View Solver | Significantly worse | - |
| Ours $\mathbf{H}_{f\rho\rho}$ | - | **Best** |

### Key Findings

- The proposed solver is **81 times faster** than Heikkilä's in Case I (17.3μs vs. 1404μs), and **27 times faster** in Case III (200μs vs. 5486μs).
- Under high noise levels, the accuracy of the proposed three-view solvers is significantly superior to the two-view baselines.
- The effectiveness of the method is also validated on a newly proposed real-world dataset comprising 1870 images and 14 different cameras.
- Case I yields only 9 unique solutions (compared to Heikkilä's 70), and fewer solutions make selecting the correct solution easier.

## Highlights & Insights

- **Elegant Mathematical Derivation**: Complex multi-view geometry problems are simplified into low-dimensional polynomial root-finding problems using elimination ideal techniques.
- **High Practicality**: The 17μs execution time allows the solver to be embedded into RANSAC loops for real-time applications.
- **First Systematic Evaluation**: The authors present the first comprehensive evaluation of three-view focal length self-calibration in planar scenes across extensive synthetic and real-world scenarios.

## Limitations & Future Work

- The scene is required to contain a sufficiently large coplanar point set, rendering it inapplicable to non-planar scenes.
- At least 4 coplanar point correspondences are required, which imposes demands on the quality of feature matching.
- The execution times of Case III and IV (106-200μs), although greatly improved, are still higher than those of Case I/II.
- Future work could integrate the DEGENSAC framework to handle partially planar scenes.

## Related Work & Insights

- By integrating classical algebraic geometry methods like Sturm sequences and the hidden variable technique, the work demonstrates the value of traditional mathematical tools in modern computer vision problems.
- The application of the elimination ideal technique in solving minimal problems is highly noteworthy.
- The construction of the new dataset provides a standardized benchmark for evaluating focal length self-calibration methods.

## Rating

⭐⭐⭐ — Mathematically elegant and practical solver design, but the problem itself is relatively niche (planar scene focal length self-calibration), leading to a somewhat limited impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Focal Split: Untethered Snapshot Depth from Differential Defocus](focal_split_untethered_snapshot_depth_from_differential_defocus.md)
- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)
- [\[CVPR 2025\] Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)
- [\[ICLR 2026\] Aligning Collaborative View Recovery and Tensorial Subspace Learning via Latent Representation for Incomplete Multi-View Clustering](../../ICLR2026/others/aligning_collaborative_view_recovery_and_tensorial_subspace_learning_via_latent_.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)

</div>

<!-- RELATED:END -->
