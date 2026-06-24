---
title: >-
  [Paper Note] Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World
description: >-
  [CVPR 2025 (Best Paper Award Candidate & Oral)][Optimization][Vanishing Point Estimation] GlobustVP introduces convex relaxation techniques to the Manhattan World vanishing point estimation problem for the first time. By formulating the joint estimation of vanishing point locations and line-to-VP associations as a QCQP and relaxing it into an SDP, it achieves a globally optimal and highly efficient solver (~50ms/image) robust to up to 70% outliers.
tags:
  - "CVPR 2025 (Best Paper Award Candidate & Oral)"
  - "Optimization"
  - "Vanishing Point Estimation"
  - "Convex Relaxation"
  - "Manhattan World"
  - "Semidefinite Programming"
  - "Global Optimality"
date: 2026-05-08
content_hash: 0cdec22543e3dbfa
---

# Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World

**Conference**: CVPR 2025 (Best Paper Award Candidate & Oral)  
**arXiv**: [2505.04788](https://arxiv.org/abs/2505.04788)  
**Code**: [https://github.com/WU-CVGL/GlobustVP](https://github.com/WU-CVGL/GlobustVP)  
**Area**: Optimization / 3D Vision  
**Keywords**: Vanishing Point Estimation, Convex Relaxation, Manhattan World, Semidefinite Programming, Global Optimality

## TL;DR

GlobustVP introduces convex relaxation techniques to the Manhattan World vanishing point estimation problem for the first time. By formulating the joint estimation of vanishing point locations and line-to-VP associations as a QCQP and relaxing it into an SDP, it achieves a globally optimal and highly efficient solver (~50ms/image) robust to up to 70% outliers.

## Background & Motivation

**Background**: Vanishing point (VP) estimation is a fundamental task in 3D computer vision, indispensable for applications such as camera calibration, scene understanding, SLAM, and autonomous driving. Under the Manhattan World assumption, a scene contains three mutually orthogonal dominant directions, corresponding to three vanishing points, with each line segment associated with one of them.

**Limitations of Prior Work**: VP estimation requires solving two coupled subproblems simultaneously: (1) determining which VP each line segment belongs to (line-to-VP association); (2) determining the position of each VP. These two subproblems are highly coupled, leading to a challenging combinatorial optimization problem. Existing methods are either sub-optimal solvers (e.g., J-Linkage, RANSAC variants) that do not guarantee global optimality and easily fall into local optima, or pursue global optimality at an extremely high computational cost (e.g., exhaustive search), failing to meet real-time requirements.

**Key Challenge**: The fundamental trade-off between global optimality and computational efficiency—methods guaranteed to find the global optimum typically have extremely high computational complexity, while efficient methods often yield only approximate solutions. Moreover, the massive presence of outliers (line segments not belonging to any dominant direction) in real-world scenes further increases the difficulty of the problem.

**Goal**: Design a VP estimator that can approximate the global optimum while exhibiting high robustness and efficiency.

**Key Insight**: Convex relaxation is a classic tool for solving combinatorial optimization problems—relaxing a non-convex problem into a convex one allows it to be solved in polynomial time, and the tightness of the relaxation theoretically guarantees solution quality. The authors observe that the VP estimation problem can be elegantly formulated as a QCQP and relaxed into an SDP.

**Core Idea**: Utilize a "soft" association scheme with truncated multi-selection error to jointly estimate VP locations and line-VP associations. The original problem is formulated as a QCQP and relaxed into an SDP, followed by an efficient implementation via an iterative solving strategy (searching for one VP and its associated line segments independently at a time).

## Method

### Overall Architecture

The input to GlobustVP is a set of detected 2D line segments, and the output consists of the image coordinates of the three vanishing points and the association of each line segment to a vanishing point (or labeled as an outlier). The method is divided into three steps: (1) back-projecting each line segment into a normal vector passing through the camera center and estimating the uncertainty of each line segment; (2) estimating each VP independently through an iterative SDP solver; (3) enforcing the orthogonality constraints among the three VPs through local optimization.

### Key Designs

1. **Soft Association Scheme with Truncated Multi-Selection Error**:

    - **Function**: Jointly optimize VP positions and associations without pre-determining line-VP associations.
    - **Mechanism**: Traditional methods typically allocate associations before optimizing positions (or vice versa); such alternating optimization easily falls into local optima. GlobustVP adopts a truncated error function, where the cost of each line segment is defined as its geometric distance to the closest VP (truncated to handle outliers). Specifically, for line segment $l_i$ and three vanishing points $\{v_1, v_2, v_3\}$, the cost is $\min(\min_j d(l_i, v_j)^2, c^2)$, where $c$ is the truncation threshold. This "min-of-min" structure naturally achieves a soft association, where the optimization process automatically finds the most appropriate VP for each line segment.
    - **Design Motivation**: The truncated error restricts the influence of outliers to the constant $c^2$, preventing outliers from excessively corrupting the estimation. Meanwhile, joint optimization avoids the local optimum issues inherent in alternating methods.

2. **Convex Relaxation from QCQP to SDP**:

    - **Function**: Transform the original non-convex optimization problem into a convex problem solvable in polynomial time.
    - **Mechanism**: In each iteration, other VPs are fixed, and one VP along with its associated line segments is estimated independently. The original problem can then be formulated as a Quadratically Constrained Quadratic Program (QCQP) with respect to a single VP position: the objective function is quadratic (sum of squared geometric distances) and the constraints are also quadratic (unit norm constraint on the VP direction vector). By introducing a matrix variable $X = xx^T$ and relaxing the rank constraint to $X \succeq xx^T$, the QCQP is relaxed into Semidefinite Programming (SDP), which can be efficiently solved using standard convex optimization solvers (SCS/MOSEK).
    - **Design Motivation**: SDP relaxation is one of the tightest convex relaxations, proven to have tiny or even zero relaxation gaps in many combinatorial optimization problems. For geometric problems like VP estimation, SDP relaxation is particularly effective.

3. **Iterative Solving and Orthogonality Constraints**:

    - **Function**: Ensure the final three VPs satisfy the orthogonality requirements of the Manhattan World.
    - **Mechanism**: The core loop of GlobustVP is: independently estimate each VP in each iteration (treating other line segments as outliers), and then enforce pairwise orthogonality constraints among the three VPs through a projection operation. Specifically, after solving three independent SDPs, the orthogonality constraint is satisfied by performing Singular Value Decomposition (SVD) on the VP direction matrix and projecting it to the nearest orthogonal matrix. This "independent solving followed by joint constraints" strategy significantly reduces the computational complexity of each step while maintaining global optimality approximation.
    - **Design Motivation**: Directly encoding the orthogonality constraints of the three VPs into the SDP would drastically increase the problem size. Decoupled solving followed by projection is a highly efficient approximation strategy.

### Uncertainty Modeling

GlobustVP computes geometric uncertainty weights for each line segment based on its length and position. Shorter line segments and those near image boundaries receive lower weights, while long and stable line segments receive higher weights. This enhances robustness against detection noise.

## Key Experimental Results

### Main Results (YUD Real Dataset)

| Method | AUC@2° | AUC@5° | AUC@10° | Globally Optimal | Execution Time |
|------|--------|--------|---------|---------|---------|
| J-Linkage | 57.7 | 69.3 | 80.5 | ✗ | ~100ms |
| Quasi-VP | 57.8 | 72.5 | 84.3 | ✗ | ~80ms |
| NeurVPS | 52.2 | 64.2 | 78.1 | ✗ | ~200ms |
| **GlobustVP** | **67.6** | **87.3** | **96.1** | ✓ | **~50ms** |

### Ablation Study (Synthetic Data Outlier Robustness)

| Outlier Ratio | GlobustVP F1 | J-Linkage F1 | Quasi-VP F1 | Description |
|---------|-------------|-------------|------------|------|
| 0% | ~0.99 | ~0.97 | ~0.98 | All methods perform similarly without outliers |
| 30% | ~0.97 | ~0.85 | ~0.90 | GlobustVP starts to show its advantage |
| 50% | ~0.94 | ~0.65 | ~0.72 | Other methods degrade significantly under half outliers |
| 70% | ~0.88 | ~0.35 | ~0.45 | GlobustVP remains robust under high outlier ratios |

### Key Findings

- GlobustVP achieves an AUC@2° of 67.6% on YUD, significantly outperforming the second-best, Quasi-VP (57.8%), by nearly 10 percentage points.
- On synthetic data, it demonstrates robustness with up to a 70% outlier ratio, far exceeding all compared methods.
- The execution time is ~50ms/image, which is faster than deep learning methods like NeurVPS (~200ms including GPU inference) and does not require training.
- The SDP relaxation shows an extremely small relaxation gap in experiments, verifying the tightness of the convex relaxation.
- Uncertainty weighting contributes significantly to the results; removing it drops the AUC@2° by approximately 3-5 percentage points.

## Highlights & Insights

- **First to introduce convex relaxation to VP estimation**: This is a theoretically elegant and practically effective contribution. The concept of converting a combinatorial optimization problem into an SDP can also be transferred to other geometric estimation problems (such as essential matrix estimation, pose graph optimization).
- **Global optimality, high efficiency, and robustness achieved together**: Typically, these three goals are difficult to satisfy simultaneously. GlobustVP strikes an excellent balance through an iterative decoupled solving strategy. In particular, the execution speed of ~50ms makes it directly applicable to real-time systems.
- **Purely geometric method without deep learning**: In an era dominated by deep learning, this work demonstrates the strong competitiveness of classical optimization methods on specific problems, showing that mathematical modeling combined with problem structure remains a highly effective research path.

## Limitations & Future Work

- The Manhattan World assumption does not hold in many real-world scenarios (e.g., natural scenes, curved architecture), limiting the applicability of the method.
- Although SDP solvers (SCS/MOSEK) are fast, they may still encounter performance bottlenecks when the number of line segments is very large (>1000).
- The current method relies heavily on the quality of external line segment detectors such as LSD; errors from detectors directly affect the VP estimation results.
- Non-Manhattan vanishing points (such as slanted directions) are not handled, though these directions are also important in practical scenarios.
- As a CVPR 2025 Best Paper Award Candidate & Oral, its follow-up extension work deserves close attention.

## Related Work & Insights

- **vs J-Linkage**: J-Linkage associates line segments and VPs via a clustering voting scheme; it is a random sampling method without global optimality guarantees. The deterministic convex relaxation of GlobustVP substantially outperforms it in both accuracy and stability.
- **vs NeurVPS**: NeurVPS uses deep learning to predict VPs, which requires training data and may degrade when generalizing to unseen scenes. GlobustVP, as a purely geometric method, requires no training and performs better on YUD.
- **vs T-Linkage / Quasi-VP**: These methods improve random sampling strategies to increase robustness, but still lack global optimality guarantees. GlobustVP possesses a stronger theoretical foundation.
- The convex relaxation strategy can be explored for application to other robust estimation problems in 3D vision (e.g., pose estimation, multi-view geometry).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to bring convex relaxation into VP estimation, with a solid theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations on both synthetic and real data, extensive outlier robustness analysis, and open-source code.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation, with rigorous derivation logic from QCQP to SDP.
- Value: ⭐⭐⭐⭐⭐ Best Paper Candidate status is well-deserved, offering broad inspiration for robust optimization in 3D vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification](../../ICML2025/optimization/interior-point_vanishing_problem_in_semidefinite_relaxations_for_neural_network_.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](../../NeurIPS2025/optimization/robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[AAAI 2026\] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](../../AAAI2026/optimization/convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/optimization/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ICLR 2026\] Enhancing Learning with Noisy Labels via Rockafellian Relaxation](../../ICLR2026/optimization/enhancing_learning_with_noisy_labels_via_rockafellian_relaxation.md)

</div>

<!-- RELATED:END -->
