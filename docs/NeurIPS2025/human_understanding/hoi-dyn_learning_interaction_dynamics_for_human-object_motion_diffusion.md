---
title: >-
  [Paper Note] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion
description: >-
  [NeurIPS 2025][Human Understanding][Human-Object Interaction] This paper models human-object interaction (HOI) generation as a Driver-Responder system…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "Human-Object Interaction"
  - "Motion Diffusion"
  - "Interaction Dynamics"
  - "Driver-Responder"
  - "Transformer"
date: 2026-05-08
content_hash: 730cc0244574177b
---

# HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2507.01737](https://arxiv.org/abs/2507.01737)
**Code**: [Available](https://wulin97.github.io/hoi-dyn)
**Area**: Human Understanding
**Keywords**: Human-Object Interaction, Motion Diffusion, Interaction Dynamics, Driver-Responder, Transformer

## TL;DR

This paper models human-object interaction (HOI) generation as a Driver-Responder system, employing a lightweight Transformer-based interaction dynamics model to explicitly predict how objects respond to human actions. A residual dynamics loss is introduced during training to enforce causal consistency, while inference efficiency is preserved.

## Background & Motivation

Generating realistic 3D human-object interactions (HOI) is a critical problem in VR/AR, computer animation, and robotics. Prior work exhibits the following limitations:

**Independent modeling**: Most methods treat human and object motion separately, resulting in physically implausible and causally inconsistent behaviors.

**Contact modeling difficulty**: Some approaches focus on object affordance or contact point prediction, but accurately modeling contact regions is inherently challenging.

**Lack of causal structure**: Existing diffusion models can generate globally plausible sequences but fail to capture fine-grained dynamics governing how objects respond to human actions.

The authors identify a key observation: HOI is fundamentally an **asymmetric system** — human motion follows internally driven dynamics (autonomous motion), whereas object motion is externally driven (cannot occur spontaneously). This asymmetry naturally motivates the Driver-Responder modeling paradigm.

## Method

### Overall Architecture

HOI-Dyn consists of two core components:

1. **Conditional Motion Diffusion**: A Transformer-based conditional diffusion model that jointly encodes human, object, and interaction context.
2. **Interaction Dynamics**: An auxiliary supervision module that enforces fine-grained causal consistency during training.

The central design principle is the **Driver-Responder formulation**:

$$\text{Driver (Human)}: h^{(t+1)} = h^{(t)} + \Delta t \cdot F_h(h^{(t)})$$
$$\text{Responder (Object)}: o^{(t+1)} = o^{(t)} + \Delta t \cdot F_o(o^{(t)}, s^{(t)}, u^{(t)})$$

where $u^{(t)}$ is the control signal (error feedback based on human intent and object behavior), and $s^{(t)}$ is the interaction context (contact state, object geometry, etc.).

### Key Designs

#### 1. Interaction Dynamics Model

Core idea: the relative motion of an object can be predicted from the relative motion of the human body:

$$\Delta o^{(t)} \approx \mathcal{D}(s^{(t)}, o^{(t)}, \Delta h^{(t)}; \theta_\mathcal{D})$$

To improve sensitivity to interactions of varying magnitude, the prediction horizon is extended from 1 to $k$, where $k$ is sampled uniformly from $[1, K]$:

$$\Delta o_{t \to t+k}^* \approx \mathcal{D}(s^{(t)}, o^{(t)}, \Delta h_{t \to t+k}; \theta_\mathcal{D})$$

Predictions are parameterized as rigid-body transformations (rotation $\hat{\mathcal{R}} \in SO(3)$ and translation $\hat{\mathcal{T}} \in \mathbb{R}^3$), with SVD projection applied to ensure valid rotation matrices.

#### 2. Object Dynamics Cost Function

Based on keypoint transformation error:

$$\Phi(\Delta o, \Delta o^*) = \|\mathcal{P}^{(t+k)} - \hat{\mathcal{P}}^{(t+k)}\|_1$$

#### 3. Implicit Contact Handling

An elegant design choice: **explicit contact modeling is not required**. In the absence of contact, no response is generated; when contact occurs, the object response is naturally determined by the interaction dynamics.

#### 4. Network Architecture

The interaction dynamics model employs a lightweight Transformer (only 0.5M parameters), taking as input the current object state, interaction context, and accumulated human motion, and outputting rigid-body transformations for object motion. A coupled design (joint prediction of rotation and translation) outperforms a decoupled design.

### Loss & Training

**Two-stage training**:

1. **Stage 1**: Pre-train the interaction dynamics model $\mathcal{D}$ using the dynamics loss:
$$\mathcal{L} = \mathbb{E}_{t, k \sim \mathcal{U}(1,K)} \left[\frac{1}{k} \cdot \Phi(\Delta o_{t \to t+k}, \Delta o^*_{t \to t+k})\right]$$

2. **Stage 2**: Train the diffusion model with total loss $= \mathcal{L}_{\text{hoi}} + \mathcal{L}_{\text{dyn}} + \mathcal{L}_{\text{obj}}$

**Residual dynamics loss** (core contribution):

$$\mathcal{L}_{\text{dyn}} = \mathbb{E}_t \left[\|\Phi(\Delta\hat{o}_t^*, \Delta\hat{o}_t) - \Phi(\Delta o_t^*, \Delta o_t)\|_1\right]$$

The key insight is that even when $\mathcal{D}$ is imperfect, taking the difference between residuals computed on generated and ground-truth sequences cancels systematic biases, allowing supervision to focus on genuine generation inconsistencies. This relies on the assumption that $\mathcal{D}$ is locally smooth and temporally homogeneous.

The dynamics model is **not used at inference**, preserving runtime efficiency.

## Key Experimental Results

### Main Results

| Method | FID ↓ | $C_{F1}$ ↑ | C% ↑ | MPJPE ↓ | $T_{\text{obj}}$ ↓ | $R_{\text{obj}}$ ↓ |
|------|-------|---------|------|---------|---------|---------|
| InterDiff | 208.0 | 0.33 | 0.27 | 25.91 | 88.35 | 1.65 |
| MDM | 6.16 | 0.53 | 0.43 | 17.86 | 24.46 | 1.85 |
| CHOIS | 0.87 | 0.66 | 0.54 | 16.01 | 14.29 | 0.99 |
| **HOI-Dyn** | **0.48** | **0.71** | **0.60** | **15.60** | **12.47** | **0.90** |

| Method (3D-FUTURE) | FID ↓ | C% ↑ | FS ↓ |
|------|-------|------|------|
| CHOIS | 1.67 | 0.47 | 0.42 |
| **HOI-Dyn** | **1.62** | **0.54** | **0.37** |

### Ablation Study

| Architecture | K | Params | K=2 Loss |
|------|---|--------|----------|
| Coupled (K=2) D4-F64-H8 | 2 | 0.483M | 0.462 |
| Coupled (K=1) D4-F64-H8 | 1 | 0.483M | 0.514 |
| Decoupled (K=2) (D1-F64-H8)×2 | 2 | 0.463M | 0.532 |
| Coupled (K=2) D8-F128-H8 | 2 | 0.994M | 0.845 |

### Key Findings

1. **Prediction horizon K**: $K=2$ or $K=3$ yields optimal performance. Too small ($K=1$) fails to capture large-magnitude motions; too large ($K=10$) weakens modeling of subtle interactions.
2. **Coupled vs. decoupled**: The coupled design for joint rotation and translation prediction significantly outperforms the decoupled counterpart at comparable parameter counts, validating the intrinsic coupling between rotation and translation in HOI.
3. **Model scale**: A lightweight 0.5M-parameter model is sufficient; scaling to 1M parameters leads to performance degradation due to overfitting.
4. **Qualitative comparison**: CHOIS produces premature motion artifacts (objects spontaneously moving toward contact points before human action); HOI-Dyn eliminates such artifacts.

## Highlights & Insights

1. **Novel Driver-Responder perspective**: Framing HOI from a control-theoretic standpoint naturally resolves the contact modeling challenge.
2. **Elegant residual loss design**: Differencing residuals cancels systematic biases in the dynamics model, enabling effective supervision from an imperfect auxiliary model.
3. **Training-inference decoupling**: The dynamics model is used only during training, introducing no additional inference overhead.
4. **Lightweight and efficient**: 0.5M-parameter dynamics model with approximately 10 hours of training on a single A4500 GPU.

## Limitations & Future Work

1. Training is limited to the FullBodyManipulation dataset, restricting scene diversity.
2. $K$ requires manual selection; adaptive prediction horizon strategies merit exploration.
3. The local smoothness and temporal homogeneity assumptions of the dynamics model may not hold for fast or highly dynamic interactions.
4. The rigid-body transformation assumption limits applicability to deformable object interactions.

## Related Work & Insights

- **CHOIS** (SOTA baseline): Sparse waypoint-guided diffusion without explicit causal interaction modeling.
- **OMOMO**: Generates human poses conditioned on full object trajectories, but lacks bidirectional causal modeling.
- **CG-HOI**: Uses contact fields on the human mesh as priors, but contact field prediction is itself challenging.
- **Insight**: The residual loss concept generalizes to other settings involving imperfect auxiliary models.

## Rating

- Novelty: ⭐⭐⭐⭐ — Driver-Responder formalization and residual dynamics loss are conceptually novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive quantitative and qualitative evaluation with thorough ablations, though limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and motivation is well-articulated.
- Value: ⭐⭐⭐⭐ — Significant contribution to HOI generation with broadly applicable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](../../CVPR2026/human_understanding/remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](../../ICCV2025/human_understanding/dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](mospa_human_motion_generation_driven_by_spatial_audio.md)
- [\[CVPR 2026\] HandX: Scaling Bimanual Motion and Interaction Generation](../../CVPR2026/human_understanding/handx_scaling_bimanual_motion_and_interaction_generation.md)

</div>

<!-- RELATED:END -->
