---
title: >-
  [Paper Note] Hyperion: A Fast, Versatile Symbolic Gaussian Belief Propagation Framework for Continuous-Time SLAM
description: >-
  [ECCV 2024][3D Vision][Continuous-Time SLAM] This paper presents Hyperion, a continuous-time Gaussian Belief Propagation (GBP) SLAM framework that automatically generates ultra-efficient B/Z-spline implementations based on the SymForce symbolic computation framework. It achieves comparable accuracy to traditional NLLS solvers (Ceres) in motion tracking and localization scenarios, while naturally supporting distributed multi-agent inference.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Continuous-Time SLAM"
  - "Gaussian Belief Propagation"
  - "B-spline"
  - "Distributed Optimization"
  - "Symbolic Computation"
date: 2026-05-08
content_hash: 68ef860d87dd01bd
---

# Hyperion: A Fast, Versatile Symbolic Gaussian Belief Propagation Framework for Continuous-Time SLAM

**Conference**: ECCV 2024  
**arXiv**: [2407.07074](https://arxiv.org/abs/2407.07074)  
**Code**: [https://github.com/VIS4ROB-lab/hyperion](https://github.com/VIS4ROB-lab/hyperion)  
**Area**: 3D Vision / Robotics  
**Keywords**: Continuous-Time SLAM, Gaussian Belief Propagation, B-spline, Distributed Optimization, Symbolic Computation

## TL;DR

This paper presents Hyperion, a continuous-time Gaussian Belief Propagation (GBP) SLAM framework that automatically generates ultra-efficient B/Z-spline implementations based on the SymForce symbolic computation framework. It achieves comparable accuracy to traditional NLLS solvers (Ceres) in motion tracking and localization scenarios, while naturally supporting distributed multi-agent inference.

## Background & Motivation

**Background**: Simultaneous Localization and Mapping (SLAM) is a core problem in robotic perception. Traditional discrete-time SLAM estimates discrete poses and requires strict synchronization of sensor timestamps. Continuous-time SLAM (CTSLAM) uses continuous-time motion parameterization (typically B-splines) to interpolate poses at arbitrary timestamps, naturally supporting asynchronous multi-sensor fusion from rolling shutter cameras, event cameras, and IMUs.

**Limitations of Prior Work**: (1) The computational complexity of CTSLAM is extremely high—continuous-time parameterization involves complex spline evaluation and analytical Jacobian calculations, and existing hand-crafted implementations lack efficiency; (2) Almost all SLAM systems (both discrete and continuous-time) employ centralized non-linear least squares (NLLS) optimization, which strictly limits them to single-agent scenarios; (3) Hand-deriving and implementing spline-related cost functions and derivatives is error-prone and time-consuming.

**Key Challenge**: The advantages of CTSLAM (asynchronous fusion, continuous motion estimation) are bottlenecked by its high computational complexity and centralized architecture, hindering its deployment in practical multi-agent scenarios. There is an urgent need for: (1) much faster implementations of continuous-time motion parameterizations; (2) a naturally distributed optimization paradigm to replace centralized NLLS.

**Goal**: (1) How to significantly accelerate the computation of B/Z-splines? (2) How to construct a distributed, asynchronous continuous-time SLAM optimization framework?

**Key Insight**: Leverage the SymForce symbolic computation framework to automatically generate ultra-efficient C++ spline code (eliminating manual derivation), and introduce Gaussian Belief Propagation (GBP) as a distributed probabilistic inference paradigm—where GBP achieves distributed, asynchronous inference via message passing on factor graphs, naturally suited for multi-agent SLAM.

**Core Idea**: Combine SymForce automatic code generation for the fastest B/Z-spline computations with the distributed inference paradigm of Gaussian Belief Propagation to construct the first continuous-time GBP SLAM framework.

## Method

### Overall Architecture

The core of Hyperion is a continuous-time factor graph GBP solver. In the factor graph, variable nodes represent the control points (basis) of B/Z-splines, while factor nodes represent sensor measurement constraints (absolute poses or visual reprojection). GBP iteratively solves for optimal motion estimation by alternating message passing between nodes and factors. The framework supports both batch and sliding-window optimization, as well as multi-threaded parallelization and dropout update strategies.

### Key Designs

1. **Ultra-Efficient Spline Implementation via SymForce**:
    - **Function**: Automatically generate analytical, highly optimized B/Z-spline evaluation and Jacobian computation code.
    - **Mechanism**: Combine the recursive spline formulation of Sommer et al. with the SymForce symbolic code generation framework. Define the evaluation of poses, velocities, and accelerations at the symbolic level, where SymForce automatically simplifies expressions, eliminates common subexpressions, and generates optimal C++ implementations. Analytical Jacobians are generated automatically instead of relying on automatic differentiation.
    - **Design Motivation**: Hand-crafted B-spline implementations (e.g., Sommer et al.) rely on automatic differentiation for Jacobian calculations, which is highly inefficient. The implementations generated automatically by SymForce accelerate pose/velocity/acceleration evaluations by $2.43\text{x}$ to $110.31\text{x}$.

2. **Continuous-Time GBP Solver**:
    - **Function**: Perform distributed probabilistic inference via message passing on factor graphs.
    - **Mechanism**: GBP alternates between two main steps: node updates and factor updates. Nodes update their belief $B(n_j) = \mathcal{N}^{-1}(\eta_{n_j}, \Lambda_{n_j})$ by aggregating all incoming factor-to-node messages from adjacent factors. Factors compute and send factor-to-node messages by collecting node-to-factor messages from neighboring nodes combined with the linearized information of the residual. For Lie groups (e.g., rotations), a Mixture of Gaussians Representation (MGR) is introduced to handle tangent space coordinate transformations.
    - **Design Motivation**: GBP is inherently distributed and asynchronous—each node and factor can be updated independently without a global solver. This makes multi-agent SLAM feasible, where each agent only needs to process its local factor graph and exchange messages with neighbors.

3. **Robustness and Flexibility Extensions**:
    - **Function**: Handle outliers and support flexible update strategies.
    - **Mechanism**: Introduce robust loss functions $\rho$ to modify residuals and Jacobians; implement synchronous updates (all nodes/factors update every round to simulate NLLS) and dropout updates (nodes/factors update randomly with probabilities $d_n$, $d_f$); introduce a step size parameter $\alpha$ to control convergence speed; mark non-optimized parameters as constant nodes to reduce marginalized computation.
    - **Design Motivation**: The convergence of GBP in loopy graphs requires dropout strategies to guarantee stability; optimization with constant nodes reduces the $\mathcal{O}(n^3)$ computational cost of Cholesky decomposition.

### Loss & Training

The optimization objective of GBP is equivalent to minimizing the weighted sum of squared residuals:

$$\Theta^* = \arg\min_\Theta \sum_s \sum_t \frac{1}{2}\|\bar{r}(t, \theta_s)\|^2$$

This formulation is completely identical to traditional NLLS. The step size is set to $\alpha_{n_j} = \alpha_{f_i} = 0.7$, and the spline basis interval is 0.1 seconds. Robust kernel functions such as Huber are supported. Synchronous update strategies are employed in experiments for fair comparison.

## Key Experimental Results

### Main Results

**Error comparison under absolute pose settings (RMSE, different perturbation levels):**

| Solver | ±0.01 (R/t) | ±0.05 | ±0.10 | ±0.50 | ±1.00 |
|--------|-------------|-------|-------|-------|-------|
| Hyperion R[rad] | 5.2e-6 | 5.2e-6 | 5.2e-6 | 5.9e-6 | 5.3e-6 |
| Ceres R[rad] | 5.2e-6 | 5.2e-6 | 5.2e-6 | 5.2e-6 | 5.2e-6 |
| Hyperion t[m] | 5.8e-6 | 5.8e-6 | 5.9e-6 | 6.0e-6 | 1.3e-5 |
| Ceres t[m] | 5.9e-6 | 5.9e-6 | 5.9e-6 | 5.9e-6 | 5.9e-6 |

**B-spline implementation speed comparison (vs Sommer et al.):**

| Operation | Order | With Jacobian | Speedup |
|------|------|-----------|--------|
| SE(3) B-Spline Evaluation | 6 | No | **2.43x** |
| SE(3) B-Spline Evaluation | 6 | Yes | **22.71x** |
| SU(2) Z-Spline Evaluation | 6 | No | **4.47x** |
| SU(2) Z-Spline Evaluation | 6 | Yes | **110.31x** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Dropout 0% | Converges in ~8 iterations | Baseline (synchronous update) |
| Dropout 10% | ~10 iterations | Slight slowdown |
| Dropout 30% | ~14 iterations | Moderate slowdown |
| Dropout 60% | ~20 iterations | Still converges to the same solution |
| B-Spline vs Z-Spline | Z-Spline converges faster | Interpolant nature of Z-spline yields better initial constraints |
| GBP vs Ceres Iteration Count | GBP requires 2-4 more iterations | Inherent cost of distributed inference |

### Key Findings

- Hyperion and Ceres converge to the identical solution across all perturbation and noise levels—validating the correctness of GBP.
- SymForce-generated code yields the most significant speedup (22-110x) in scenarios evaluated with Jacobians, as it eliminates automatic differentiation.
- Z-splines generally converge faster than B-splines, because the base points of the interpolating splines lie directly on the motion trajectory.
- Although dropout updates increase the iteration count, they preserve the consistency of the final solution, making it suitable for unreliable communication scenarios.
- In localization setups (with landmark reprojection), visual factors introduce more loops, rendering GBP less sensitive to dropout.
- On a single core, GBP is approximately 6-7.5x slower than Ceres, but its inherent parallelism and scaling to distributed setups can bridge this gap.

## Highlights & Insights

- **Huge engineering value in SymForce automation**: It eliminates the most tedious manual derivation and implementation steps in CTSLAM, while simultaneously achieving performance that exceeds human-crafted optimization.
- **GBP opens a new paradigm for multi-agent SLAM**: Inherently distributed, asynchronous, and capable of handling imperfect communication.
- **Equivalency to Ceres solutions**: Verifies the correctness and practicality of GBP on non-trivial SLAM problems.
- **Open-source contribution**: Provides a factor library that seamlessly integrates with Ceres and SymForce.

## Limitations & Future Work

- Single-core performance is still inferior to highly optimized Ceres (6-7.5x slower), requiring further engineering optimizations.
- Numerical instability in loopy graphs (such as full SLAM systems) may lead to convergence challenges.
- Currently only validated in simulations and simple real-world scenarios, and has not yet been deployed in a full multi-agent SLAM system.
- Covariance estimation adds extra computational overhead (Ceres does not explicitly estimate covariance).
- Covariance information can be utilized to implement adaptive node updates (only updating nodes with high uncertainty) to further accelerate execution.

## Related Work & Insights

- **Ceres Solver**: The benchmark of traditional centralized NLLS optimizers and the standard for comparison against Hyperion.
- **Murai et al. (Robot Web)**: Pioneering work of GBP in distributed SLAM, introducing MGR to handle Lie groups.
- **SymForce**: The symbolic code generation framework, which serves as key infrastructure in this work.
- **Sommer et al.**: Hand-optimized B-spline implementation, serving as the baseline for comparison in this work.
- **Insight**: The methodology of combining symbolic computation with automatic code generation can be generalized to other robotics tasks requiring highly efficient analytical derivatives.

## Rating

- Novelty: ⭐⭐⭐⭐ Combines GBP with continuous-time SLAM for the first time; SymForce acceleration is a valuable engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐ Primarily focused on simulation experiments, with limited real-world validation.
- Writing Quality: ⭐⭐⭐⭐ The mathematical derivation is rigorous and thorough, though highly dense.
- Value: ⭐⭐⭐⭐ Lays the theoretical and engineering foundation for distributed multi-agent CTSLAM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [\[ECCV 2024\] GaussReg: Fast 3D Registration with Gaussian Splatting](gaussreg_fast_3d_registration_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
