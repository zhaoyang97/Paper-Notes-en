---
title: >-
  [Paper Note] Towards Stable and Storage-efficient Dataset Distillation: Matching Convexified Trajectory
description: >-
  [CVPR 2025][Optimization][Dataset Distillation] This paper proposes the MCT (Matching Convexified Trajectory) method. By replacing SGD expert trajectories with a linear convex combination trajectory from random initialization to the optimal point, MCT simultaneously addresses the three major challenges of the traditional MTT method: trajectory instability, slow convergence, and high storage consumption.
tags:
  - "CVPR 2025"
  - "Optimization"
  - "Dataset Distillation"
  - "Trajectory Matching"
  - "Convexified Trajectory"
  - "Neural Tangent Kernel"
  - "Storage-efficient"
date: 2026-05-08
content_hash: 52d202c74905e8f9
---

# Towards Stable and Storage-efficient Dataset Distillation: Matching Convexified Trajectory

**Conference**: CVPR 2025  
**arXiv**: [2406.19827](https://arxiv.org/abs/2406.19827)  
**Code**: None  
**Area**: Optimization / Dataset Distillation  
**Keywords**: Dataset Distillation, Trajectory Matching, Convexified Trajectory, Neural Tangent Kernel, Storage-efficient

## TL;DR

This paper proposes the MCT (Matching Convexified Trajectory) method. By replacing SGD expert trajectories with a linear convex combination trajectory from random initialization to the optimal point, MCT simultaneously addresses the three major challenges of the traditional MTT method: trajectory instability, slow convergence, and high storage consumption.

## Background & Motivation

Dataset Distillation (DD) compresses large-scale real datasets into small synthetic datasets, enabling models trained on synthetic data to achieve performance close to those trained on real datasets. Among DD methods, Multi-step Trajectory Matching (MTT) is an important branch that aligns the training trajectory of the student network on synthetic data with that of the expert network on real data.

The authors identify three severe limitations of the MTT method: (1) **Unstable expert trajectories**: The expert trajectories trained by SGD exhibit severe oscillations with fluctuating validation accuracy, leading to unstable training dynamics learned by the student network. (2) **Slow distillation convergence**: A large number of iterations are required to generate effective synthetic datasets. (3) **High storage overhead**: It requires saving model parameters at all checkpoints (around 50 models) along the trajectory, causing massive storage costs.

The authors present a new perspective to understand the essence of DD and MTT: by reformulating the MTT loss function, the optimization objective of DD can be viewed as obtaining a set of parameters (the synthetic dataset) that accurately predict the direction and magnitude of the next update step at any point in the parameter space. Based on this, the solution is to find an expert trajectory that is stable, easy to fit, and storage-efficient.

## Method

### Overall Architecture

The pipeline of MCT is as follows: (1) Train an expert network on real data to obtain the standard SGD trajectory $\tau_{mtt}$; (2) Construct the convexified trajectory $\tau_{conv}$ using the starting point $\theta_\mathcal{T}^{(0)}$ and the ending point $\theta_\mathcal{T}^{(K)}$ of the trajectory; (3) Perform distillation using a continuous sampling strategy along the convexified trajectory; (4) Recover the entire trajectory by storing only 2 models and a set of constants.

### Key Designs

1. **Convexified Expert Trajectory**:
    - **Function**: Construct stable, monotonically increasing expert trajectories to replace oscillating SGD trajectories.
    - **Mechanism**: Inspired by NTK linearized dynamics $f_\theta(x) \approx f_{\theta_0}(x) + (\theta - \theta_0)^\mathsf{T} \nabla_\theta f_{\theta_0}(x)$, the trajectory is constructed as a convex combination (linear interpolation) of the starting and ending points: $\hat{\theta}_\mathcal{T}^{(t)} = (1 - \lambda_t) \theta_\mathcal{T}^{(0)} + \lambda_t \theta_\mathcal{T}^{(K)}$, where $\lambda_t$ is the interpolation coefficient. The update direction $\vec{V}_\mathcal{T}^{(t)}$ at each point along this trajectory always points to the optimal point, and the validation accuracy increases monotonically.
    - **Design Motivation**: SGD trajectory oscillation makes the sampled update directions unstable; a linear trajectory ensures that the update direction always points to the final converged point, greatly simplifying the patterns that the synthetic data needs to fit.

2. **Continuous Sampling Strategy**:
    - **Function**: Continuously sample training points from the convexified trajectory to ensure comprehensive learning.
    - **Mechanism**: Since the convexified trajectory is continuous (no longer a discrete set of waypoints), model parameters at any point in time can be obtained as sampling starters by continuously selecting $\lambda_t \in [0, 1]$. This greatly enriches the training "dataset" (i.e., sampled points in the parameter space), allowing the synthetic data to learn the update rules more comprehensively.
    - **Design Motivation**: The original MTT can only sample discretely from a limited set of pre-stored waypoints, missing the information between trajectories.

3. **Loss Function Reformulation under a New Perspective**:
    - **Function**: Provide a new framework to understand the essence of DD.
    - **Mechanism**: Rewrite the MTT loss function $\mathcal{L}(\mathcal{S},\mathcal{T}) = \frac{\|\theta_\mathcal{S}^{(t+N)} - \theta_\mathcal{T}^{(t+M)}\|_2^2}{\|\theta_\mathcal{T}^{(t)} - \theta_\mathcal{T}^{(t+M)}\|_2^2}$ as $\frac{\|\vec{V}_\mathcal{S} - \vec{V}_\mathcal{T}\|_2^2}{\|\vec{V}_\mathcal{T}\|_2^2}$ by viewing the expert trajectory waypoints as "training data" $\{(\theta_\mathcal{T}^{(t)}, \vec{V}_\mathcal{T}^{(t)})\}$. The core of DD is to learn a function that predicts update vectors in the parameter space.
    - **Design Motivation**: This perspective directly reveals the root causes of the three issues: unstable $V$, a small training set that is difficult to fit, and big storage requirements for large training sets.

### Loss & Training

- **Loss Function**: Same normalized L2 trajectory matching loss as MTT.
- **Storage Optimization**: Only requires storing two models $\theta_\mathcal{T}^{(0)}$ and $\theta_\mathcal{T}^{(K)}$ plus interpolation coefficients, achieving massive savings compared to storing ~50 models in MTT.
- **Distillation Acceleration**: The stability of the convexified trajectory significantly reduces the number of iterations required for convergence.

## Key Experimental Results

### Main Results

CIFAR-10 dataset distillation performance (IPC = Images Per Class):

| Method | IPC=1 | IPC=10 | IPC=50 | Storage | Convergence Iterations↓ |
|------|-------|--------|--------|------|---------|
| MTT | — | — | — | ~50 models | High |
| **MCT** | **Higher** | **Higher** | **Higher** | **2 models** | **Significantly Reduced** |

### Ablation Study

| Configuration | Description |
|------|------|
| MTT original trajectory | Oscillations cause instability and slow convergence |
| Convexified trajectory (w/o continuous sampling) | Improved stability, but limited sampling points |
| Convexified trajectory + continuous sampling | Best performance, learning the trajectory comprehensively |
| Different $\lambda_t$ distributions | Uniform distribution is the most effective |

### Key Findings

- The model validation accuracy along the convexified trajectory of MCT increases monotonically, completely eliminating the oscillations of the SGD trajectory.
- Storage is reduced from ~50 models to 2 models, improving storage efficiency by approximately 25x.
- Convergence is significantly accelerated—under the same accuracy threshold, MCT requires far fewer distillation iterations than MTT.
- The continuous sampling strategy provides richer training points, which is particularly crucial for low IPC settings.
- Outperforms conventional MTT on CIFAR-10, CIFAR-100, and Tiny ImageNet datasets.

## Highlights & Insights

- **The perspective of re-understanding the essence of DD is highly inspiring**: Viewing distillation as "learning to predict optimal update vectors in the parameter space" naturally unifies three seemingly independent issues under the same explanatory framework.
- **The idea of replacing SGD trajectories with linear interpolation is surprisingly simple yet highly effective**: Inspired by NTK theory, linear interpolation in parameter space approximates linearized dynamics, which both ensures correct directionality and significantly simplifies trajectory structure.

## Limitations & Future Work

- The NTK linearization assumption may not hold for networks with insufficient depth or width.
- The convexified trajectory assumes a near-linear path between the start and end points; overly complex loss landscapes might violate this assumption.
- Its performance on larger-scale models (e.g., ResNet-101) and larger datasets (e.g., ImageNet-1K) has not been validated.
- The optimal distribution for the continuous sampling strategy requires further research.

## Related Work & Insights

- **vs MTT**: MTT is the direct baseline that this method improves upon; MCT maintains the same overall distillation framework but replaces the expert trajectory, simultaneously resolving stability, speed, and storage issues.
- **vs Distribution Matching (DM)**: DM performs distillation via distribution matching instead of trajectory matching, avoiding trajectory storage issues but generally underperforming compared to trajectory matching methods.
- **vs FRePo/TESLA**: These methods optimize the computational efficiency and loss functions of MTT, but do not address the issue of trajectory oscillation.

## Rating

- Novelty: ⭐⭐⭐⭐ NTK-inspired convexified trajectory is simple yet effective, and the new perspective is valuable for understanding DD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three datasets under multiple IPC settings, comparing convergence, stability, and storage dimensions.
- Writing Quality: ⭐⭐⭐⭐ Thorough analysis of issues, with a natural derivation from motivation to methodology.
- Value: ⭐⭐⭐⭐ 25x storage savings, faster convergence, and better performance, solving the practical pain points of MTT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Algorithms for Incremental Metric Bipartite Matching](../../ICLR2026/optimization/efficient_algorithms_for_incremental_metric_bipartite_matching.md)
- [\[ACL 2025\] AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](../../ACL2025/optimization/ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)
- [\[NeurIPS 2025\] Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](../../NeurIPS2025/optimization/conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)
- [\[CVPR 2025\] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)

</div>

<!-- RELATED:END -->
