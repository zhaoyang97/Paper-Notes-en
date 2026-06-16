---
title: >-
  [Paper Note] Linear Fundamental Matrix Estimation from 7 or 5 Points
description: >-
  [CVPR 2026][3D Vision][RANSAC] This paper provides an **elementary geometric explanation** for the phenomenon that "7-point fundamental matrix estimation is uniquely solvable under a specific point-line configuration (V-Umlaut: 5 points falling on two lines)" and proposes the first **linear solver** for this minimal problem. By treating it as a 5-po
tags:
  - CVPR 2026
  - 3D Vision
  - RANSAC
date: 2026-05-08
content_hash: 5d239b9411b5d6f2
---
# Linear Fundamental Matrix Estimation from 7 or 5 Points

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kucukpinar_Linear_Fundamental_Matrix_Estimation_from_7_or_5_Points_CVPR_2026_paper.html)  
**Code**: https://github.com/CIVA-Lab/v-umlaut  
**Area**: 3D Vision / Two-view Geometry  
**Keywords**: Fundamental Matrix, Minimal Solver, Two-view Geometry, RANSAC, Virtual Correspondences

## TL;DR
This paper provides an **elementary geometric explanation** for the phenomenon that "7-point fundamental matrix estimation is uniquely solvable under a specific point-line configuration (V-Umlaut: 5 points falling on two lines)" and proposes the first **linear solver** for this minimal problem. By treating it as a 5-point solver (supplemented by two virtual midpoints) and integrating it with Early Non-Minimal Refitting, the method achieves accuracy comparable to state-of-the-art (SOTA) 5-point methods while being several times faster in RANSAC.

## Background & Motivation

**Background**: Two-view geometry is the cornerstone of 3D reconstruction. The fundamental matrix $F$ is a $3\times3$ matrix of rank 2, defined up to scale, encoding the epipolar geometry between two perspective cameras. Classically, linear solutions require at least 8 point correspondences (8-point algorithm), while non-linear minimal solvers require only 7 points in the uncalibrated case (7-point algorithm, solving a cubic equation) or 5 points in the calibrated case (5-point algorithm, solving a decic equation).

**Limitations of Prior Work**: Minimal solvers are repeatedly called within robust frameworks like RANSAC, making the **speed and stability of a single solve** crucial. The 7-point method requires solving a cubic equation and returns up to 3 candidate solutions, which is slow and requires subsequent filtering. The 5-point method is highly accurate but involves a more computationally expensive decic equation. The question arises whether a solver can be both **linear (avoiding high-degree equations for stability/speed) and minimal (using the fewest points possible)**.

**Key Challenge**: The attributes of being linear and minimal are traditionally mutually exclusive—reducing the point count to the minimal set usually necessitates non-linearity. Recently, work [29] categorizing n-view minimal problems **unexpectedly discovered** that the 7-point problem has a **unique solution** under very specific configurations. However, the proof in [29] relies on specific camera parameterization and lacks a practical solver.

**Goal**: ① Theoretical—Clarify how this "unexpected uniqueness" manifests within the standard 7-point algorithm, proving it is a genuine geometric result rather than a parameterization artifact. ② Practical—Construct a linear minimal solver for this specific configuration and make it work in real-world RANSAC pipelines.

**Key Insight**: The focus is on the most general unique configuration in [29]—5 independent points plus 2 dependent points lying on two lines passing through $p_1$, termed **V-Umlaut** (resembling the letter V with two dots). Borrowing the "virtual correspondence" idea from [47], the two dependent points do not need to correspond to real scene features; instead, the **2D midpoints** of two line segments can serve as virtual correspondences.

**Core Idea**: First, use elementary linear algebra to simplify the 7-point V-Umlaut problem into 5 linear equations with 5 unknowns (yielding a unique solution), creating the first linear minimal solver. Second, by replacing the two dependent points with "virtual midpoints of the existing 5 correspondences," it is downgraded into a practical **5-point heuristic solver** that can be directly integrated into RANSAC.

## Method

### Overall Architecture
The core of the solver is the **dimensionality reduction of the non-linear 7-point problem into a linear system** under the V-Umlaut configuration. Given 7 correspondences $(p_i, q_i)$ satisfying collinearity constraints, the algorithm proceeds in four steps: ① Use two per-view homographies $H_1, H_2$ to map the first 4 points of each view to the standard basis (normalization), causing the fundamental matrix to take a fixed form $F_{\text{coords}}$ with zeros on the diagonal and $f_{32}=1$; ② Extract parameters $(s, x, y)$ (barycentric coordinates of dependent points + coordinates of the 5th point) from the normalized points; ③ Compute the 5 unknown elements directly through an explicit rational mapping $\Phi(s,x,y)=F_{\text{coords}}$; ④ De-normalize $F = H_2^\top F_{\text{coords}} H_1$. The entire pipeline involves no high-degree equations and is linear, stable, and fast. In practical deployment, the requirement for two dependent points is replaced by virtual midpoints, causing it to degenerate into a general 5-point solver.

Since the mechanism is based on algebraic derivation rather than a multi-module pipeline, no flowchart is provided; formulas describe the process more clearly.

### Key Designs

**1. V-Umlaut Configuration: Making the 7-Point Problem a "Unique Solution" Special Case**

The limitation is that the general 7-point problem has 3 solutions. V-Umlaut imposes constraints on point positions to gain uniqueness: among the 7 correspondences, the 6th and 7th are **dependent points**—they lie on lines passing through $p_1$, i.e., $p_{5+i} \in \text{span}(p_1, p_{1+i})$ and $q_{5+i} \in \text{span}(q_1, q_{1+i})$ for $i\in\{1,2\}$. Using barycentric coordinates, scalars $\tilde s_{ij}$ exist such that $p_6 = \tilde s_{11}p_1 + (1-\tilde s_{11})p_2$, etc. The goal is to recover $F$ satisfying $\text{rank}\,F=2$ and $q_i^\top F p_i = 0$. Under four non-degeneracy assumptions ($X_1,X_2,X_3$ are not collinear; dependent points do not coincide with known points; no traditional degeneracies; and $\{C_P,C_Q,X_1,X_2\}$ as well as $\{C_P,C_Q,X_1,X_3\}$ are not coplanar), the fundamental matrix for this configuration is unique.

**2. Per-view Projective Normalization: Compressing the Matrix into 5 Unknowns**

To keep the equations clean, homographies map each view to a standard coordinate frame. In the first view, let $A=[p_1\ p_2\ p_3]$, and use three $3\times3$ determinants $\Delta_{423},\Delta_{143},\Delta_{124}$ to form $D=\text{diag}(\cdot)$. Setting $H_1 = D^{-1}A^{-1}$ results in $z_i = H_1 p_i$ satisfying $z_1\sim e_1, z_2=e_2, z_3=e_3, z_4=e_1+e_2+e_3$ (standard basis). $H_2$ is constructed similarly for the second view. The first three normalized correspondences $z_i\sim w_i\sim e_i$ **automatically** force $F_{\text{coords}}$ into a fixed form (zero diagonal, $f_{32}=1$), while the fourth correspondence forces the sum of elements to be zero. Thus, the 8-dimensional $F$ is compressed to 5 free elements $f_{12},f_{13},f_{21},f_{23},f_{31}$, which can be read from image quantities via simple formulas like $s_{ij} = \frac{1}{1 + z_{62}/z_{61}}$.

**3. Spurious Solution Analysis: Proving Uniqueness is Not an Artifact**

Like the standard 7-point method, $\det(F_{\text{coords}})=0$ plus $w_i^\top F_{\text{coords}} z_i = 0$ ($i=4,5,6,7$) yields 3 solutions. This paper proves **two are degenerate for V-Umlaut**: they satisfy $f_{12}=f_{21}=0$ or $f_{13}=f_{31}=0$, respectively. For $i=6$, $w_6^\top F_{\text{coords}} z_6 = w_{62}z_{61}f_{21} + w_{61}z_{62}f_{12}=0$. If $f_{12}=0$, then $f_{21}=0$, leading to a degenerate affine fundamental matrix where epipoles lie on $e_1e_2$, implying $C_P,C_Q,X_1,X_2$ are coplanar—contradicting the non-degeneracy assumption. The same applies to $i=7$. Thus, any non-degenerate solution must satisfy $f_{12}f_{13}\neq 0$. The solver actively avoids these spurious solutions, confirming that the uniqueness in [29] is a geometric fact and not a byproduct of camera parameterization.

**4. Simplification from Non-linear to Linear: Reducing the Determinant Constraint**

Using equations for $i=6$ and $i=7$ to express $f_{21}$ and $f_{31}$ linearly in terms of $f_{12}$ and $f_{13}$, the determinant constraint $\det(F_{\text{coords}})=0$ simplifies to $f_{12}f_{13}\,(c_1(s)f_{31} + c_2(s)) = 0$, where $c_1,c_2$ are simple rational functions of input parameters $s$. Given $f_{12}f_{13}\neq 0$, the remaining part $c_1(s)f_{31} + c_2(s)=0$ is a **linear** equation. Combined with the original 4 linear equations, this provides 5 linear equations for 5 unknowns, yielding a unique solution. Cramer's rule allows each element to be written as an explicit rational mapping $F_{\text{coords}} = \Phi(s,x,y)$ ($\Phi:\mathbb{R}^{10}\dashrightarrow\mathbb{R}^5$). This is the source of the "linear and minimal" property.

**5. Virtual Midpoint Heuristic: Downgrading to a Practical 5-Point Solver**

In real scenes, it is rare to find dependent points falling exactly on two lines. Borrowing from [47], the authors set the two dependent correspondences $(p_6,q_6)$ and $(p_7,q_7)$ as the **2D midpoints** of two line segments formed by the existing 5 correspondences. These virtual midpoints do not need to correspond to physical features. This is partially justified by affine approximation: under an affine camera, a midpoint projects to a midpoint. Consequently, only 5 real correspondences are needed to call the V-Umlaut solver (denoted as 5pF-V-Umlaut), which returns a **unique** solution—significantly lighter than the 10 candidates of the 5-point method or 3 of the 7-point method. To recover accuracy, Early Non-Minimal Refitting (ENM) from [47] is applied: after each RANSAC round, a non-minimal solve is performed using $n\ge5$ inliers. A **Fast ENM** variant is also proposed, keeping only the second-stage pose closest to the initial V-Umlaut estimate to save scoring overhead.

### Loss & Training
This is a geometric solver with no learning-based training. Synthetic experiments use rejection sampling to ensure points are in front of both cameras and within the image, comparing two noise models (independent point noise vs. dependent point noise along lines). Real-world experiments use the PoseLib LO-RANSAC framework with a 1-pixel epipolar threshold.

## Key Experimental Results

### Main Results
Real data: 13 scenes from PhotoTourism + Aachen Day-Night v1.1, generating 5000 image pairs per scene. Metrics: mAA@10° (mean average accuracy for pose error $\max(\theta_R,\theta_t) < 10°$), median pose error (MED), and average runtime (ms). RANSAC is fixed at 1000 iterations.

| Method | PT MED(°)↓ | PT mAA@10↑ | PT Runtime (ms)↓ | Aachen mAA@10↑ | Aachen Runtime (ms)↓ |
|------|------|------|------|------|------|
| 5pE (SOTA 5-pt) | 1.26 | 79.58 | 24.68 | 64.24 | 23.47 |
| 4pE(M) (Virtual [47]) | 1.31 | 78.95 | 23.66 | 62.20 | 21.66 |
| 7pF (7-pt) | 5.53 | 46.44 | 8.07 | 24.69 | 7.99 |
| **5pF-V-Umlaut (Ours)** | 4.26 | 49.79 | **3.52** | 24.43 | **3.16** |
| 5pE + ENM | 1.26 | 79.65 | 116.17 | 64.16 | 98.99 |
| 7pF + ENM | 1.26 | 79.58 | 61.44 | 61.41 | 55.21 |
| **5pF-V-Umlaut + ENM (Ours)** | **1.25** | 79.62 | 44.12 | **64.07** | 36.93 |
| **5pF-V-Umlaut + Fast ENM (Ours)** | 1.26 | 79.44 | 36.37 | 64.05 | **31.28** |

- Without ENM, 5pF-V-Umlaut is the **fastest** (PT 3.52 ms, Aachen 3.16 ms) and significantly more accurate than the uncalibrated 7pF (PT mAA 49.79 vs 46.44), though it trails the analytically superior 5pE.
- With ENM, 5pF-V-Umlaut + ENM **matches SOTA** (PT mAA 79.62 vs 79.58 for 5pE) while the runtime is a fraction of other ENM methods (PT 44.12 ms vs 116.17 ms for 5pE+ENM).
- Fast ENM further reduces time (Aachen 31.28 ms) with negligible accuracy loss, offering the best balance.

### Synthetic / Robustness Experiments
Using two camera intrinsics and scene geometries (Standard and WAMI), 10,000 trials were performed without RANSAC to analyze noise sensitivity.

| Noise Model | Observation |
|------|------|
| 5 independent points + 2D Gaussian noise | 5pF-V-Umlaut is more sensitive than 5pE/7pF but **degrades gracefully** |
| 1D noise along lines for 2nd view dependent points | Errors stem primarily from **dependent point correspondence noise**—suggesting better matching could improve results |

### Key Findings
- **Uniqueness is Verifiable**: In numerical examples where 7pt returns 3 solutions, V-Umlaut returns 1; geometrically, the 3 solutions signify point sets on ruled quadrics, and the two spurious solutions correspond to the same quadric.
- **Bottleneck in Dependent Points**: Synthetic experiments identify the main error source as the noise in virtual/dependent correspondences, pointing toward improved matching as a future direction.
- **ENM is Key for Accuracy**: While increasing RANSAC samples helps, ENM is necessary to match SOTA accuracy; fortunately, it maintains a significant speed advantage.

## Highlights & Insights
- **"Linear and Minimal"**: Breaking the conventional belief that minimal sets require high-degree equations, this work uses a specific configuration to reduce the 7-point problem to linear equations, an elegant algebraic construction for real-time RANSAC.
- **Elementary Proof vs. Heavy Machinery**: Unlike [29], which relied on complex parameterization, this paper uses elementary geometry (coplanar epipoles) to explain uniqueness, proving it isn't an artifact.
- **Virtual Midpoint "Cheat" is Effective**: Using segment midpoints as virtual points—justified by affine approximation—works well on real data, a transferable idea for other constrained minimal problems.
- **Fast ENM Engineering Trade-off**: Retaining only the second-stage solution closest to the initial estimate saves significant RANSAC scoring time with almost no loss in precision.

## Limitations & Future Work
- **Higher Noise Sensitivity**: The lone 5pF-V-Umlaut is more sensitive to pixel noise than 5pE/7pt, requiring more RANSAC iterations or ENM to reach top-tier accuracy.
- **Reliance on Virtual Approximation**: The virtual midpoint is only strictly valid under affine projection; perspective distortion introduces errors.
- **Speed Advantage Partially Offset by ENM**: To match SOTA accuracy, ENM is required; though still faster than other ENM methods, it is no longer as extremely fast as the "pure minimal" solve.
- **Potential for Expansion**: Future work could explore replacing the Nistér 5-point method in ENM with iterative refinement initialized by V-Umlaut or extending V-Umlaut to triple virtual correspondences (e.g., room corner calibration).

## Related Work & Insights
- **vs. 5pE ([32])**: 5pE solves a decic equation with 10 candidates; it is more accurate but slower. Ours is linear with a unique solution, matching accuracy with ENM while being ~3x faster.
- **vs. 7pF**: Both handle uncalibrated $F$; 7pt uses a cubic with 3 solutions. Ours is linear, faster, more stable, and more accurate in RANSAC.
- **vs. 4pE(M) ([47])**: Ours reuses their virtual point and ENM strategy but migrates it from a 3-view 4-point setting to 2-view linear $F$ estimation with a complete theoretical explanation.
- **vs. [29]**: [29] proposed the uniqueness but lacked a practical solver. This paper provides an elementary proof and the first stable linear solver, turning a theoretical phenomenon into a practical tool.

## Rating
- Novelty: ⭐⭐⭐⭐ Unearths a "linear yet minimal" solver in a mature field and clarifies a theoretical phenomenon.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid synthetic analysis and benchmarking across datasets, though could be tested on more diverse scenes.
- Writing Quality: ⭐⭐⭐⭐ Clear step-by-step derivations and spurious solution analysis.
- Value: ⭐⭐⭐⭐ Practical for real-time SfM/pose estimation due to the significant speedup over SOTA in RANSAC.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners](featurising_pixels_from_dynamic_3d_scenes_with_linear_in-context_learners.md)
- [\[CVPR 2026\] DepthFocus: Controllable Depth Estimation for See-Through Scenes](depthfocus_controllable_depth_estimation_for_see-through_scenes.md)
- [\[CVPR 2026\] Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation](depth_any_panoramas_a_foundation_model_for_panoramic_depth_estimation.md)
- [\[CVPR 2026\] AIMDepth: Asymmetric Image-Event Mamba for Monocular Depth Estimation](aimdepth_asymmetric_image-event_mamba_for_monocular_depth_estimation.md)
- [\[CVPR 2026\] MD2E: Modeling Depth-to-Edge Cues for Monocular Metric Depth Estimation](md2e_modeling_depth-to-edge_cues_for_monocular_metric_depth_estimation.md)

</div>

<!-- RELATED:END -->
