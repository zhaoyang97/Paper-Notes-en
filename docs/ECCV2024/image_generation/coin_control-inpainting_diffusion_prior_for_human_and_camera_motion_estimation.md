---
title: >-
  [Paper Note] COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation
description: >-
  [ECCV 2024][Image Generation][Global Human Motion Estimation] This paper proposes the COIN method, which simultaneously estimates high-quality global human motion and camera motion from monocular dynamic camera video through an improved version of Score Distillation Sampling via Control-Inpainting, combined with joint human-scene relationship losses.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Global Human Motion Estimation"
  - "Camera Motion Estimation"
  - "Score Distillation Sampling"
  - "Motion Diffusion Prior"
  - "Control-Inpainting"
date: 2026-05-08
content_hash: 825b489fe80243da
---

# COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation

**Conference**: ECCV 2024  
**arXiv**: [2408.16426](https://arxiv.org/abs/2408.16426)  
**Code**: [Yes](https://nvlabs.github.io/COIN/)  
**Area**: Human Motion Estimation / Diffusion Models  
**Keywords**: Global Human Motion Estimation, Camera Motion Estimation, Score Distillation Sampling, Motion Diffusion Prior, Control-Inpainting

## TL;DR

This paper proposes the COIN method, which simultaneously estimates high-quality global human motion and camera motion from monocular dynamic camera video through an improved version of Score Distillation Sampling via Control-Inpainting, combined with joint human-scene relationship losses.

## Background & Motivation

Recovering global human motion and camera motion from monocular moving camera video is an important yet highly challenging problem. The core difficulty lies in the **entanglement** of human motion and camera motion: the motion observed in the camera coordinate system is a superposition of both.

Main limitations of prior work:

**Regression methods** (e.g., GLAMR): Ignore camera motion or assume simple scenes, leading to poor generalization.

**Motion prior + SLAM methods** (e.g., SLAHMR, PACE): Use VAE motion priors to constrain human motion in a low-dimensional space, leading to over-smoothing; camera optimization relies entirely on global human motion, causing catastrophic failures when initial human motion is inaccurate.

**Diffusion models as natural candidates for priors**: Although they theoretically encode rich motion priors, directly applying SDS produces inconsistent and over-smoothed motion.

The authors demonstrate a skateboarding scene in Fig. 1 where PACE and WHAM fail completely, whereas COIN successfully recovers the motion—thanks to its improved diffusion prior and human-scene relationship constraints.

## Method

### Overall Architecture

COIN is an **iterative optimization framework** that jointly optimizes global human motion $\mathbf{H}$, camera motion $\mathcal{C}$, camera scale $s$, initial-frame camera pose (height $h_0$ and orientation $R_0$), and human shape $\beta$.

Overall optimization objective: $\min_{\mathbf{H}, \mathcal{C}, s, h_0, R_0, \beta} \mathcal{L}_{\text{body}} + \mathcal{L}_{\text{COIN-SDS}} + \mathcal{L}_{\text{HSR}}$

Where:
- $\mathcal{L}_{\text{body}}$: Standard human body reconstruction loss (2D reprojection + 3D joints + shape regularization + temporal smoothness + foot contact)
- $\mathcal{L}_{\text{COIN-SDS}}$: Improved Score Distillation Sampling loss (core contribution)
- $\mathcal{L}_{\text{HSR}}$: Human-scene relationship loss (resolves scale ambiguity)

Initialization pipeline: HybrIK extracts frame-by-frame local SMPL parameters $\rightarrow$ DROID-SLAM obtains camera-to-world transformations $\rightarrow$ local motion is transformed to world coordinates (which drifts due to the unknown SLAM scale).

### Key Designs

**1. Multi-Step Denoising instead of One-Step SDS**

Standard SDS utilizes a diffusion model with one-step denoising to obtain the pseudo ground-truth $\hat{\mathbf{H}}_0^t$, but minor perturbations in the input cause drastic output variations (inconsistency). COIN instead adopts a 10-step DDIM denoising process:

$$\tilde{\mathbf{H}}_{t-\Delta t} = \sqrt{\bar{\alpha}_{t-\Delta t}} \cdot \hat{\mathbf{H}}_0^t + \sqrt{1-\bar{\alpha}_{t-\Delta t}} \cdot \boldsymbol{\epsilon}_\phi^t$$

Multi-step denoising yields higher quality and more consistent pseudo-ground-truth motion.

**2. Dynamic Controlled Sampling**

A ControlNet-style control branch $\phi_c$ is attached to the pre-trained diffusion model to guide the generation using partially observed human motion as control signals:

$$\tilde{\mathbf{H}}_0^t = \mathcal{D}_{\phi, \phi_c}(\tilde{\mathbf{H}}_t, t, \mathbf{c} \odot \mathbf{M})$$

The key innovation is **dynamic control**: instead of using a fixed initial estimate as the control signal, each iteration updates the control signal with the optimized result from the previous step $\mathbf{c} = \mathbf{H}$, forming a self-evolving control. This avoids performance degradation caused by inaccurate initial estimates.

The control branch employs a ControlNet architecture: it copies four encoder blocks of the pre-trained model, followed by zero convolutions, while freezing the pre-trained model and only training the control branch.

**3. Soft Inpainting**

During the denoising process, "known regions" and "unknown regions" are distinguished, where the known regions preserve observations and the unknown regions are sampled by the diffusion model. The key is using a **continuous weight mask** rather than a binary mask:

$$\tilde{\mathbf{M}} = w(t) \cdot \mathbf{S} \odot \mathbf{M}$$

where $w(t) = \max(0, \frac{t-0.5}{0.5})$ decreases over denoising timesteps, and $\mathbf{S}$ represents observation confidence scores. This allows the diffusion model to slightly correct high-confidence observations while largely reconstructing low-confidence regions.

**4. Human-Scene Relationship Loss (HSR Loss)**

Utilizing scene point clouds reconstructed by SLAM to constrain camera scale. The core idea is that scene points projected onto visible vertices of the human mesh should be occluded by the human body (i.e., have greater depth).

$$\mathcal{L}_{\text{HSR}} = -\frac{1}{|\mathcal{P}|} \sum_{i=1}^T \sum_{p \in \mathcal{P}^*} \min(0, \mathcal{T}^{(i)}(p)_z - j^{(i)}(p)_z) \cdot \mathbb{1}(\text{invisible})$$

This leverages the human-scene depth relationship to provide complementary constraint information to the motion prior, decoupling the camera scale from the dependence on human motion.

### Loss & Training

- COIN-SDS Loss: $\mathcal{L}_{\text{COIN-SDS}} = \frac{\omega(t)\sqrt{\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}} \|\mathbf{H} - \tilde{\mathbf{H}}_0\|_2^2$
- Body Loss: $\mathcal{L}_{\text{body}} = \mathcal{L}_{\text{2D}} + \mathcal{L}_{\text{3D}} + \mathcal{L}_\beta + \mathcal{L}_{\text{smooth}} + \mathcal{L}_{\text{contact}}$
- The motion diffusion model is trained on the AMASS dataset, and the control branch is efficiently fine-tuned based on the frozen pre-trained model.

## Key Experimental Results

### Main Results

**Table 1: Global human motion estimation on the RICH dataset**

| Method | PA-MPJPE↓ | W-MPJPE↓ | WA-MPJPE↓ | ACCEL↓ |
|------|-----------|----------|-----------|--------|
| GLAMR | 79.9 | 653.7 | 365.1 | 107.7 |
| SLAHMR | 52.5 | 571.6 | 323.7 | 9.4 |
| WHAM | 46.2 | 497.6 | 272.7 | 6.7 |
| PACE | 49.3 | 380.0 | 197.2 | 8.8 |
| **COIN** | **42.9** | **254.5** | **169.5** | **7.5** |

COIN improves W-MPJPE by 33% compared to PACE (380.0 $\rightarrow$ 254.5) and by 49% compared to WHAM.

**Table 2: Global human motion estimation on the EMDB dataset**

| Method | PA-MPJPE↓ | W-MPJPE100↓ | RTE↓ | ROE↓ |
|------|-----------|-------------|------|------|
| WHAM | 41.9 | 439.2 | 8.4 | 36.3 |
| **COIN** | **32.7** | **407.3** | **3.5** | **34.1** |

### Ablation Study

**Ablation study on the RICH dataset (W-MPJPE)**

| Variant | W-MPJPE↓ |
|------|----------|
| Vanilla SDS | 1453.5 |
| COIN w/o Controlled Sampling | 825.0 |
| COIN w/o Dynamic Control | 293.8 |
| COIN w/o Soft Inpainting | 325.8 |
| COIN w/o $\mathcal{L}_{\text{HSR}}$ | 273.0 |
| **COIN (Full)** | **254.5** |

Each component contributes significantly, with Controlled Sampling having the greatest impact (removing it degrades W-MPJPE from 254.5 to 825.0).

### Key Findings

1. Vanilla SDS fails completely in motion estimation (W-MPJPE=1453.5), confirming the inconsistency issue of directly applying SDS.
2. Dynamic control is the most critical component; static control (w/o Dynamic Control) still performs decently but shows a significant gap.
3. The HSR loss primarily helps tackle the scale problem, with limited impact on local motion (PA-MPJPE).
4. COIN improves not only global motion but also local motion quality (achieving the best PA-MPJPE across three datasets).

## Highlights & Insights

- **Generalizability of the improved SDS scheme**: The concept of control-inpainting can be transferred to other tasks requiring prior distillation from diffusion models (e.g., 3D generation, motion synthesis).
- **Self-evolving control signals**: Dynamically updating control conditions to form a positive feedback loop is an elegant solution to address inaccurate initial estimations.
- **Human-scene depth consistency**: Calibrating camera scale using occlusion relationships in scene point clouds serves as an ingenious bridge connecting SLAM and human reconstruction.

## Limitations & Future Work

1. **Computational cost**: Performing 10-step DDIM denoising at every optimization iteration makes the overall optimization slow.
2. **Dependence on SLAM quality**: If DROID-SLAM fails in complex scenes, the entire pipeline is affected.
3. **Single-person assumption**: The current framework targets single-person scenes, requiring extensions for multi-person interactive scenarios.
4. **Training data limitations of motion diffusion models**: The AMASS dataset has limited coverage of motion types; extreme motions (e.g., gymnastics) may fail.

## Related Work & Insights

- The control-inpainting SDS scheme in COIN provides a new template for improving SDS in other diffusion prior applications (e.g., text-to-3D, motion generation).
- The concept of human-scene relationship loss can be extended to human-human relation constraints in multi-person scenarios.
- Complementary to regression methods like WHAM, allowing COIN's optimization results to serve as training data for regression methods.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

## Related Work & Insights

| Method | Mechanism | Motion Prior | Camera Estimation | Limitations of Prior Work |
|------|---------|---------|---------|---------|
| GLAMR | Regressing global motion | None | Does not estimate | Ignores camera motion, poor generalization |
| SLAHMR | SLAM + VAE prior optimization | HuMoR (VAE) | SLAM initialization + optimization | Over-smoothing, poor 2D alignment |
| PACE | SLAM + conditional VAE prior | LEMO (cVAE) | Relies on human motion optimization | Catastrophic camera failure when human motion initialization is poor |
| WHAM | Regression + 2D keypoint lifting | Implicit learning | Angular velocity input | Does not explicitly recover camera motion |
| RoHM | Motion diffusion model recovery | MDM (diffusion) | Does not estimate | Restores only local motion, does not handle camera |
| **COIN** | **Control-Inpainting SDS + HSR** | **Improved motion diffusion**| **Joint optimization + HSR constraints** | **High computational cost, relies on SLAM** |

The core differences between COIN and SLAHMR/PACE: replacing the VAE prior with a diffusion model resolves the over-smoothing caused by VAE constraining motion in low-dimensional spaces; using COIN-SDS instead of Vanilla SDS addresses denoising inconsistency; incorporating the HSR loss to supplement the motion prior prevents cascading failures when human motion initialization is inaccurate.

## Insights & Connections

- **A general paradigm for improved SDS**: The control-inpainting strategy (dynamic control signal + soft inpainting + multi-step denoising) proposed by COIN represents a systematic improvement over standard SDS, which can be extended to downstream tasks requiring prior distillation from diffusion models such as text-to-3D and video generation.
- **Self-evolving optimization strategy**: Designing the control signal of the next round using the optimization results of the previous round is akin to the EM algorithm or bootstrapping, which is well-suited for iterative optimization scenarios with unreliable initial estimates.
- **Cross-modal consistency constraints**: The HSR loss utilizes scene geometry and human depth occlusion relationships to calibrate camera scale. This notion of cross-modal (motion vs. scene) consistency constraints can be expanded to more complex settings such as human-object interaction and multi-person scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->
