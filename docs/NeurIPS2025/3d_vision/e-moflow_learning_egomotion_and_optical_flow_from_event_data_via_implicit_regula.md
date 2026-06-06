---
title: >-
  [Paper Note] E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization
description: >-
  [NeurIPS 2025][3D Vision][Event Camera] This paper proposes E-MoFlow, which models optical flow as an implicit neural representation and egomotion as a continuous spline…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Event Camera"
  - "Optical Flow Estimation"
  - "Egomotion Estimation"
  - "Implicit Regularization"
  - "Unsupervised Learning"
  - "Neural ODE"
date: 2026-05-08
content_hash: b34769888d16f0b1
---

# E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization

**Conference**: NeurIPS 2025
**arXiv**: [2510.12753](https://arxiv.org/abs/2510.12753)  
**Code**: [Project Page](https://akawincent.github.io/EMoFlow/)  
**Area**: 3D Vision
**Keywords**: Event Camera, Optical Flow Estimation, Egomotion Estimation, Implicit Regularization, Unsupervised Learning, Neural ODE

## TL;DR

This paper proposes E-MoFlow, which models optical flow as an implicit neural representation and egomotion as a continuous spline, jointly optimizing both via differential geometric constraints under an unsupervised paradigm to achieve 6-DoF egomotion and dense optical flow estimation from event data.

## Background & Motivation

Event cameras asynchronously sense per-pixel brightness changes, offering advantages such as high dynamic range and low latency. However, their data characteristics pose unique challenges for motion estimation:

1. **Aperture problem in optical flow**: The locality of event data means estimated flow is inherently normal flow, making full motion field recovery infeasible.
2. **Ill-posed 6-DoF egomotion**: Without depth priors, estimating complete 6-DoF motion from event data is theoretically underdetermined.
3. **Lack of reliable data association**: Event data lacks the long-term feature matching capability of conventional frame-based images.

Limitations of existing solutions:
- **Explicit spatial-temporal regularization** (e.g., smoothness losses): Introduces bias and increases computational overhead.
- **Explicit structure-from-motion parameterization** (joint estimation of flow + depth + motion): Increases degrees of freedom, making optimization more prone to local minima.

Core innovation: replacing explicit regularization with **implicit regularization**—naturally imposing constraints through the inductive biases of the representation itself (network smoothness, spline continuity) and differential geometric constraints (bypassing explicit depth estimation).

## Core Problem

How to jointly estimate optical flow and 6-DoF egomotion from event data in an unsupervised framework, while avoiding the bias of explicit regularization and the local minima introduced by explicit depth estimation?

## Method

### Continuous Optical Flow Representation

Optical flow is modeled as an implicit neural representation. Given time $t$ and normalized pixel coordinates $\mathbf{x}$, the network outputs the flow vector (velocity field):

$$\mathbf{u}_\theta(t, \mathbf{x}) = \text{NN}_\theta(t, \mathbf{x})$$

The warp trajectory of events is modeled as a Neural ODE:

$$\frac{d\mathbf{x}_k(t)}{dt} = \text{NN}_\theta(t, \mathbf{x}_k(t)), \quad \mathbf{x}_k(t_k) = \mathbf{x}_k$$

Integral solution: $\mathbf{x}_k(t) = \mathbf{x}_k + \int_{t_k}^t \text{NN}_\theta(s, \mathbf{x}_k(s)) ds$

Gradients are efficiently computed via the adjoint ODE, avoiding gradient explosion and memory issues from direct backpropagation:

$$\frac{d\mathbf{a}_k(t)}{dt} = -\mathbf{a}_k(t)^\top \frac{\partial \text{NN}_\theta(t, \mathbf{x}_k(t))}{\partial \mathbf{x}_k(t)}$$

Key distinction: the method directly models the velocity field rather than displacement, making it suitable for more aggressive motion scenarios.

### Continuous Egomotion Representation

Cubic B-splines represent the angular velocity and linear velocity of the camera:

$$[\boldsymbol{\omega}_\beta(t); \boldsymbol{\nu}_\beta(t)] = \sum_{i=0}^n \mathbf{B}_{i,3}(t) \beta_i$$

where $\beta_i \in \mathbb{R}^6$ are control points and $\mathbf{B}_{i,3}$ are cubic basis functions. The continuity of the spline naturally encodes a temporal smoothness prior.

### Differential Flow Loss

Based on contrast maximization: events are warped to a reference time $t_{\text{ref}}$ and accumulated into an Image of Warped Events (IWE), with the contrast of the IWE maximized:

$$L_{\text{flow}} = -\frac{1}{HW} \sum_{i,j} (I_{ij} - \mu_I)^2$$

### Differential Geometry Loss

Based on differential epipolar constraints, optical flow and egomotion are jointly constrained in homogeneous coordinates **without explicit depth estimation**:

$$L_{\text{geometry}} = \left\| \hat{\mathbf{u}}_\theta(t, \mathbf{x})^\top [\boldsymbol{\nu}_\beta(t)]_\times \hat{\mathbf{x}} - \hat{\mathbf{x}}^\top \mathbf{s}_\beta(t) \hat{\mathbf{x}} \right\|_2^2$$

where $\mathbf{s}_\beta(t) = \frac{1}{2}([\boldsymbol{\nu}_\beta(t)]_\times [\boldsymbol{\omega}_\beta(t)]_\times + [\boldsymbol{\omega}_\beta(t)]_\times [\boldsymbol{\nu}_\beta(t)]_\times)$.

Joint optimization of the two losses prevents the flow estimation from collapsing to degenerate solutions (e.g., in pure translation scenarios).

### Training Procedure

$$\min_{\theta, \beta} \mathbb{E}_{t_{\text{ref}}} [L_{\text{flow}}(\mathcal{E}_{\text{neigh}}(t_{\text{ref}}), \theta)] + \mathbb{E}_{\{\mathbf{x}, t\} \sim \mathcal{E}} [L_{\text{geometry}}(t, \mathbf{x}, \theta, \beta)]$$

Optimization is performed from scratch for each event sequence (test-time optimization), requiring no pretraining.

## Key Experimental Results

### MVSEC Optical Flow Estimation (dt=1)

| Method | Type | Mean EPE ↓ | Mean %Out ↓ |
|--------|------|-----------|------------|
| ADM-Flow | Supervised | 0.533 | 0.340 |
| MultiCM-V2 | Model-agnostic | 0.348 | 0.055 |
| MultiCM | Model-agnostic | 0.455 | 0.270 |
| EV-MGRFlowNet | Unsupervised | 0.495 | 0.958 |
| **E-MoFlow** | **Unsupervised** | **0.450** | **0.328** |

Best among unsupervised methods; competitive with supervised method ADM-Flow.

### MVSEC Optical Flow Estimation (dt=4, Large Motion)

| Method | Type | Mean EPE ↓ |
|--------|------|-----------|
| MultiCM-V2 | Model-agnostic | 1.108 |
| EV-MGRFlowNet | Unsupervised | 1.763 |
| **E-MoFlow** | **Unsupervised** | **1.773** |

Comparable to other unsupervised methods under large-motion conditions; slightly below MultiCM-V2, which also jointly estimates depth and motion.

### DSEC Optical Flow Estimation

E-MoFlow achieves the best or second-best results among unsupervised methods on the DSEC dataset, confirming generalizability.

### Egomotion Estimation

E-MoFlow achieves state-of-the-art unsupervised 6-DoF egomotion estimation on MVSEC, with both angular and linear velocity errors outperforming existing methods.

## Highlights & Insights

1. **Elegant implicit regularization design**: Spatial-temporal priors are imposed through the network smoothness of the Neural ODE and the continuity of B-splines, without explicit loss terms.
2. **Bypassing depth estimation**: Differential geometric constraints directly couple optical flow and egomotion, avoiding local minima caused by increased optimization degrees of freedom.
3. **Fully unsupervised**: Requires no annotations, depth, or grayscale image supervision.
4. Joint estimation of optical flow and egomotion is of significant practical importance in the event camera domain.

## Limitations & Future Work

- Per-sequence test-time optimization results in low inference efficiency.
- Performance under large-motion conditions (dt=4) falls short of MultiCM-V2; large displacement remains a challenge.
- Evaluation is limited to MVSEC and DSEC; validation on a broader range of real-world scenarios is lacking.
- No comparison with recent supervised large-model methods.

## Related Work & Insights

- **vs MultiCM/MultiCM-V2**: The MultiCM series requires explicit depth estimation and motion field parameterization; E-MoFlow bypasses depth via differential geometric constraints.
- **vs USL-EV-FlowNet**: USL-EV-FlowNet jointly predicts optical flow and egomotion via reprojection loss but relies on discretized volumetric representations; E-MoFlow employs continuous implicit representations.
- **vs EvLinearSolver**: Linear solvers require prior knowledge (e.g., known angular velocity) and cannot fully recover 6-DoF; E-MoFlow is fully autonomous.

The concept of implicit regularization is broadly applicable: selecting representations with appropriate inductive biases to replace explicit regularization terms. Modeling event trajectories with Neural ODEs can generalize to motion estimation for other asynchronous data. The technique of replacing motion field equations with differential geometric constraints to avoid depth estimation is worth extending to other settings.

## Rating

- ⭐ Novelty: 9/10 — The framework combining implicit regularization and differential geometric constraints for joint optimization is highly elegant.
- ⭐ Experimental Thoroughness: 7/10 — Two datasets with comparisons against multiple methods, but broader real-world evaluation and efficiency analysis are lacking.
- ⭐ Writing Quality: 8/10 — Mathematical derivations are rigorous and motivation is clearly articulated.
- ⭐ Value: 8/10 — A significant contribution to the event camera domain; the joint estimation framework holds practical importance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Linearly Constrained Diffusion Implicit Models](linearly_constrained_diffusion_implicit_models.md)
- [\[NeurIPS 2025\] Flux4D: Flow-based Unsupervised 4D Reconstruction](flux4d_flow-based_unsupervised_4d_reconstruction.md)
- [\[CVPR 2026\] EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](../../CVPR2026/3d_vision/eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)
- [\[NeurIPS 2025\] Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis](shallow_flow_matching_for_coarse-to-fine_text-to-speech_synthesis.md)

</div>

<!-- RELATED:END -->
