---
title: >-
  [Paper Note] Continuous Simplicial Neural Networks
description: >-
  [NEURIPS2025][Autonomous Driving][Simplicial Neural Networks] This paper proposes COSIMO, the first continuous simplicial neural network based on partial differential equations (PDEs)…
tags:
  - "NEURIPS2025"
  - "Autonomous Driving"
  - "Simplicial Neural Networks"
  - "PDE"
  - "Over-smoothing"
  - "Hodge Laplacian"
  - "Topological Deep Learning"
date: 2026-05-08
content_hash: fa2a16d672e25872
---

# Continuous Simplicial Neural Networks

**Conference**: NEURIPS2025
**arXiv**: [2503.12919](https://arxiv.org/abs/2503.12919)  
**Code**: [ArefEinizade2/COSIMO](https://github.com/ArefEinizade2/COSIMO)  
**Area**: Autonomous Driving
**Keywords**: Simplicial Neural Networks, PDE, Over-smoothing, Hodge Laplacian, Topological Deep Learning

## TL;DR

This paper proposes COSIMO, the first continuous simplicial neural network based on partial differential equations (PDEs), which realizes continuous information flow by defining heat diffusion dynamics on the Hodge Laplacian. COSIMO demonstrates superior stability and over-smoothing control compared to discrete SNNs.

## Background & Motivation

- Graph neural networks (GNNs) model only pairwise interactions between nodes and cannot capture higher-order relationships (e.g., triangles, tetrahedra, and other multi-body interactions).
- Simplicial complexes extend the expressive power of graphs by introducing $k$-simplices and the Hodge Laplacian; however, existing simplicial neural networks (SNNs) primarily rely on **discrete filtering** (matrix polynomials), which suffers from two core limitations:
    - **Manual tuning of filter order**: The discrete polynomial orders $T_d, T_u$ are hyperparameters that incur high tuning costs.
    - **Difficulty in controlling over-smoothing**: Features tend to converge as the number of layers increases, and discrete SNNs can only mitigate this by modifying the topological structure, which is impractical.
- In the GNN literature, continuous models (e.g., Neural ODEs and graph diffusion equations) have been shown to better control over-smoothing and improve robustness to structural perturbations; however, **continuous SNNs remain unexplored**.

## Core Problem

How to design a continuous simplicial neural network that: (1) possesses a dynamically learnable receptive field rather than a fixed polynomial order; (2) remains stable under topological perturbations; and (3) can effectively control the rate of over-smoothing?

## Method

### Simplicial Complex Preliminaries

- **$k$-simplices**: 0-simplices = nodes, 1-simplices = edges, 2-simplices = triangles.
- **Incidence matrix** $\mathbf{B}_k$: describes the incidence relationship between $(k-1)$-simplices and $k$-simplices.
- **Hodge Laplacian**: $\mathbf{L}_k = \mathbf{B}_k^\top \mathbf{B}_k + \mathbf{B}_{k+1}\mathbf{B}_{k+1}^\top$, decomposed into the lower Laplacian $\mathbf{L}_{k,d}$ and upper Laplacian $\mathbf{L}_{k,u}$.
- **Dirichlet energy**: $E(\mathbf{x}_k) = \mathbf{x}_k^\top \mathbf{L}_k \mathbf{x}_k$, measuring the smoothness of a signal.

### PDE Framework of COSIMO

The core idea is to define heat diffusion PDEs on the decoupled upper and lower Hodge Laplacians to enable continuous-time information propagation:

1. **Independent lower diffusion**: $\frac{\partial \mathbf{x}_{k,d}(t_d)}{\partial t_d} = -\mathbf{L}_{k,d} \mathbf{x}_{k,d}(t_d)$
2. **Independent upper diffusion**: $\frac{\partial \mathbf{x}_{k,u}(t_u)}{\partial t_u} = -\mathbf{L}_{k,u} \mathbf{x}_{k,u}(t_u)$
3. **Joint diffusion**: coupled interaction dynamics between upper and lower spaces.
4. **Integral output**: combination of the solutions from the independent and joint dynamics.

### COSIMO Layer Definition

The closed-form solution of the PDE is a matrix exponential, and the $l$-th layer is defined as:

$$\mathbf{X}_k^l = \sigma\left(e^{-t_d \mathbf{L}_{k,d}} \mathbf{X}_{k,d}^{l-1} \boldsymbol{\Theta}_{k,d}^l + e^{-t_u \mathbf{L}_{k,u}} \mathbf{X}_{k,u}^{l-1} \boldsymbol{\Theta}_{k,u}^l + e^{-t_d \mathbf{L}_{k,d}} \mathbf{X}_k^{l-1} \boldsymbol{\Psi}_{k,d}^l + e^{-t_u \mathbf{L}_{k,u}} \mathbf{X}_k^{l-1} \boldsymbol{\Psi}_{k,u}^l\right)$$

- $t_d, t_u$ are **learnable continuous receptive field parameters** (a key advantage), replacing the manually tuned orders in discrete filters.
- Multi-branch aggregation ($M$ branches) is supported to enhance expressive capacity.

### Efficient Implementation

By leveraging the eigenvalue decomposition (EVD) of the Hodge Laplacian and retaining the top $K$ dominant eigenpairs to approximate the matrix exponential, the complexity is reduced from $\mathcal{O}(|\mathcal{X}_k|^3)$ to $\mathcal{O}(|\mathcal{X}_k|^2 (K_k^{(d)} + K_k^{(u)} + K_k))$.

### Stability Analysis

Under additive perturbations $\tilde{\mathbf{B}}_k = \mathbf{B}_k + \mathbf{E}_k$ to the incidence matrix, the output error of COSIMO is bounded:

$$\delta_{\mathbf{X}_k} \leq t_d \delta_{k,d} e^{t_d \delta_{k,d}}(\|\mathbf{x}_{k,d}(0)\| + \|\mathbf{x}_k(0,0)\|) + t_u \delta_{k,u} e^{t_u \delta_{k,u}}(\|\mathbf{x}_{k,u}(0)\| + \|\mathbf{x}_k(0,0)\|)$$

When perturbations are sufficiently small, $\delta_{\mathbf{X}_k} = \mathcal{O}(\epsilon_k) + \mathcal{O}(\epsilon_{k+1})$, generalizing the stability results of continuous GNNs to higher-order structures.

### Over-Smoothing Analysis

- **Discrete SNN**: The upper bound on Dirichlet energy is governed solely by the topological structure $\tilde{\lambda}_{\max}$, requiring topological modifications to mitigate over-smoothing.
- **COSIMO**: An $e^{-2\varphi}$ decay factor is introduced into the upper bound (where $\varphi = \min_k\{t_d \lambda_{\min}(\mathbf{L}_{k,d}), t_u \lambda_{\min}(\mathbf{L}_{k,u})\}$), allowing the over-smoothing rate to be reduced by decreasing the receptive field parameters $t$, without any topological modification.

## Key Experimental Results

| Task | Dataset | COSIMO | Strongest Baseline |
|------|---------|--------|--------------------|
| Trajectory prediction | ocean-drifts | **0.550** | SCCNN 0.545 |
| Mesh regression | Shrec-16 (small) | **0.010** MSE | SCCNN 0.020 |
| Mesh regression | Shrec-16 (full) | **0.027** MSE | SCCNN 0.063 |
| Node classification | high-school | **0.90** | SCCNN/GSAN 0.88 |
| Node classification | senate-bills | **0.69** | GCN 0.67 |
| Graph classification | proteins | **0.79** | SaNN/GSAN 0.77 |

- Over-smoothing experiment: at $t=10^{-2}$, COSIMO exhibits a slower over-smoothing rate than discrete SNNs; increasing $t$ accelerates the rate.
- Stability experiment: model performance degrades gracefully as SNR varies from -5 dB to 20 dB, validating the theoretical bounds.

## Highlights & Insights

- **Pioneering contribution**: COSIMO is the first neural network to define continuous PDE dynamics on simplicial complexes, filling the gap in topological deep learning with respect to continuous models.
- **Rigorous theory**: Formal analyses of both the stability bound and the over-smoothing convergence rate are provided and empirically validated.
- **Learnable receptive field**: $t_d, t_u$ serve as learnable parameters, eliminating the need for hyperparameter search over polynomial orders in discrete methods.
- **Significant advantage on Shrec-16**: MSE is reduced by more than 50% compared to SCCNN, demonstrating the potential of continuous models for mesh processing.

## Limitations & Future Work

- **EVD overhead**: Although truncated eigendecomposition reduces complexity, preprocessing for large-scale simplicial complexes still requires $\mathcal{O}(K N^2)$.
- **Future directions suggested by the authors**: Exploring non-negative matrix factorization, Cholesky decomposition, or implicit Euler methods as alternatives to EVD.
- **Limited gains in trajectory prediction**: Performance is inferior to SCNN on synthetic datasets and only marginally better than SCCNN on ocean-drifts.
- **Interpretability of continuous time parameters**: The physical meaning of $t_d, t_u$ and their optimal value ranges lack thorough discussion.
- **Static topology assumption**: The current framework assumes a fixed topological structure; extension to time-varying simplicial complexes remains unexplored.

## Related Work & Insights

| Method | Type | Receptive Field | Over-Smoothing Control | Stability Analysis |
|--------|------|-----------------|------------------------|--------------------|
| SNN/SCNN | Discrete filtering | Fixed order | Difficult (requires topological modification) | None |
| SCCNN | Discrete Hodge-aware | Fixed order | Partial analysis | Yes (discrete) |
| Continuous GNN | PDE on graphs | Learnable time | Controllable | Yes |
| **COSIMO** | **PDE on simplicial complexes** | **Learnable $t_d, t_u$** | **Controllable (via $t$)** | **Yes (generalized to higher order)** |

- COSIMO is a natural generalization of continuous GNNs from graphs to simplicial complexes; the core contribution lies in the decoupled PDE formulation for the upper and lower Hodge Laplacians.
- COSIMO is most directly comparable to SCCNN: it replaces the matrix polynomial $\sum \alpha_i \mathbf{L}^i$ in SCCNN with the matrix exponential $e^{-t\mathbf{L}}$.

### Further Insights

- The continuous PDE paradigm is broadly applicable and can potentially be extended to other higher-order structures such as cell complexes and hypergraphs.
- The over-smoothing control mechanism (via the receptive field parameter $t$) provides practical guidance for designing deep topological models.
- In autonomous driving, simplicial modeling for trajectory prediction—where nodes represent locations, edges represent paths, and triangles represent regions—constitutes a promising application direction.

## Rating
- Novelty: ⭐⭐⭐⭐ — First continuous SNN with a complete theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple tasks and theory-validation experiments, though trajectory prediction gains are limited.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical exposition is clear; theory and experiments are well organized.
- Value: ⭐⭐⭐⭐ — Establishes a foundation for the continuization of topological deep learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception](../../ICCV2025/autonomous_driving/unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)
- [\[AAAI 2026\] I-INR: Iterative Implicit Neural Representations](../../AAAI2026/autonomous_driving/i-inr_iterative_implicit_neural_representations.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](../../CVPR2026/autonomous_driving/neural_distribution_prior_for_lidar_ood_detection.md)
- [\[CVPR 2026\] CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation](../../CVPR2026/autonomous_driving/cyclebev_regularizing_view_transformation_networks_via_view_cycle_consistency_fo.md)
- [\[ICLR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../ICLR2026/autonomous_driving/spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)

</div>

<!-- RELATED:END -->
