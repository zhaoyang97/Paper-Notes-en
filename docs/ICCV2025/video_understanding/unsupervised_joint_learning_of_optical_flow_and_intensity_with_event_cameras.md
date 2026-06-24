---
title: >-
  [Paper Note] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras
description: >-
  [ICCV 2025][Video Understanding][event cameras] This paper proposes the first unsupervised learning framework based on a single network for jointly estimating optical flow and image intensity from event camera data. The core contribution is a complementary loss formulation combining a newly derived Event-based Photometric Error (PhE) with Contrast Maximization (CMax).
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "event cameras"
  - "optical flow estimation"
  - "image intensity reconstruction"
  - "unsupervised learning"
  - "joint estimation"
date: 2026-05-08
content_hash: e2c1ce1a32381247
---

# Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras

**Conference**: ICCV 2025  
**arXiv**: [2503.17262](https://arxiv.org/abs/2503.17262)  
**Code**: [GitHub](https://github.com/tub-rip/E2FAI)  
**Area**: Video Understanding  
**Keywords**: event cameras, optical flow estimation, image intensity reconstruction, unsupervised learning, joint estimation

## TL;DR

This paper proposes the first unsupervised learning framework based on a single network for jointly estimating optical flow and image intensity from event camera data. The core contribution is a complementary loss formulation combining a newly derived Event-based Photometric Error (PhE) with Contrast Maximization (CMax).

## Background & Motivation

Event cameras are novel bio-inspired vision sensors offering high temporal resolution, high dynamic range (HDR), low power consumption, and minimal motion blur. Their output is an asynchronous stream of per-pixel brightness change events rather than conventional frame images, necessitating fundamentally new algorithms.

**Core Observation**: Under constant illumination, **motion and appearance are naturally coupled** in event cameras — events are triggered by moving brightness patterns. Consequently, the two fundamental visual quantities (optical flow = motion, intensity = appearance) are intrinsically co-determined: they either co-exist and are jointly recorded, or neither exists.

Yet existing methods treat these two tasks as **almost entirely separate**:
- Optical flow estimation: EV-FlowNet, E-RAFT, etc., trained independently.
- Intensity reconstruction: E2VID, etc., trained independently.
- The few joint methods are either restricted to pure rotational motion or require two separate networks cascaded together.

This leads to two problems: (1) the inherent synergy between motion and appearance is left unexploited, and (2) cascading two independent models results in slow inference and error accumulation.

**Goal**: Design a unified unsupervised framework in which a single network simultaneously outputs optical flow and intensity images, fully exploiting their synergy through newly derived loss functions.

## Method

### Overall Architecture

The model adopts a classic U-Net architecture, taking a 15-channel event voxel grid as input and producing 3 output channels (2-channel optical flow + 1-channel intensity). During training, **two consecutive event data samples** are fed at each step; the network predicts optical flow and intensity for each, and their relationship is enforced via a temporal consistency loss. At inference, a single event voxel grid suffices to simultaneously produce both optical flow and intensity.

### Key Designs

1. **Event-based Photometric Error (PhE)**

   Starting from the Event Generation Model (EGM): $\Delta L = L(\mathbf{x}_k, t_k) - L(\mathbf{x}_k, t_k - \Delta t_k) = p_k C$

   After warping event $e_k$ and its predecessor event to reference time $t_{\text{ref}}$, the per-event photometric error is defined as:

   $\epsilon_k = (L(\mathbf{x}'_k) - L(\mathbf{x}'_{k-1})) - p_k C$

   A key property: each PhE term simultaneously constrains approximately 8 intensity pixels and 1 optical flow pixel, enabling joint estimation. The total PhE loss is the mean absolute residual over all events:

   $\mathcal{L}_{\text{PhE}}(L, F) = \frac{1}{N_e} \sum_{k=1}^{N_e} |\epsilon_k|$

   PhE does not suffer from event collapse and places greater emphasis on appearance (intensity) constraints.

2. **Contrast Maximization (CMax)**

   Based on the gradient sharpness of the Image of Warped Events (IWE):

   $\mathcal{L}_{\text{CMax}}(F) = 1 \Big/ \left(\frac{1}{|\Omega|}\int_\Omega \|\nabla \text{IWE}(\mathbf{x})\|_1 \, d\mathbf{x}\right)$

   The sole optimizable variable in CMax is optical flow, making it more focused on motion estimation. PhE and CMax are thus complementary: the former emphasizes appearance, the latter emphasizes motion.

3. **Temporal Consistency (TC)**

   A core advantage of joint estimation: the predicted optical flow $F_{i \to i+1}$ is used to warp intensity $L_i$ to time $t_{i+1}$, which is then compared against the directly predicted $L_{i+1}$ from the second sample:

   $\mathcal{L}_{\text{TC}} = \frac{1}{|\Omega|}\int_\Omega |L_{i+1}(\mathbf{x}) - \mathcal{W}(\mathbf{x}; L_i, F_{i \to i+1})| \, d\mathbf{x}$

   The TC loss jointly constrains the temporal coherence of both optical flow and intensity, and is the primary source of advantage that joint estimation holds over independent estimation.

### Loss & Training

The total loss is a weighted sum of five terms:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{PhE}} + \lambda_2 \mathcal{L}_{\text{CMax}} + \lambda_3 \mathcal{L}_{\text{FTV}} + \lambda_4 \mathcal{L}_{\text{ITV}} + \lambda_5 \mathcal{L}_{\text{TC}}$$

where $\mathcal{L}_{\text{FTV}}$ and $\mathcal{L}_{\text{ITV}}$ denote total variation (TV) regularization on optical flow and intensity, respectively. Weights are set to $\lambda_1=30, \lambda_2=1, \lambda_3=10, \lambda_4=0.001, \lambda_5=1$.

Training details: the model is trained exclusively on the DSEC training set for 130 epochs using the AdamW optimizer, on 4 RTX A6000 GPUs with a batch size of 24. The CMax reference time is randomized to reduce event collapse risk; the PhE reference time is fixed to the end of each sample.

## Key Experimental Results

### Main Results

**Optical Flow — DSEC Test Set (Overall)**

| Method | Type | EPE↓ | AE↓ | %Out↓ | Inference Time (ms) |
|--------|------|------|-----|-------|---------------------|
| E-RAFT | Supervised | 0.788 | 10.56 | 2.684 | 46.3 |
| IDNet | Supervised | 0.719 | 2.723 | 2.036 | - |
| MotionPriorCM | Unsupervised | 3.2 | 8.53 | 15.21 | 17.9 |
| BTEB | Unsupervised | 3.86 | - | 31.45 | - |
| EV-FlowNet | Unsupervised | 3.86 | - | 31.45 | - |
| **Ours** | **Unsupervised** | **1.781** | **6.439** | **11.241** | **15.1** |

Among unsupervised methods, the proposed approach reduces EPE by approximately **20%** and AE by approximately **25%**, while achieving the shortest inference time.

### Ablation Study

| Configuration | EPE (Flow) | SSIM (Intensity) | Notes |
|---------------|-----------|------------------|-------|
| Full model | 1.781 | Best | PhE + CMax + TV + TC |
| w/o PhE | Degraded | Severely degraded | PhE provides critical intensity constraints |
| w/o CMax | Significantly degraded | — | CMax provides core motion constraints |
| w/o TC | Temporal inconsistency | SSIM notably drops | TC is the key advantage of joint estimation |
| w/o TV regularization | Increased flow noise | Blurred boundaries | Smoothness regularization is critical in event-free regions |

### Key Findings

- PhE and CMax are complementary: PhE emphasizes intensity constraints, CMax emphasizes motion constraints, and neither is dispensable.
- The TC loss contributes most significantly to intensity quality improvement — this is the central advantage of joint over independent estimation.
- The model is trained solely on DSEC yet generalizes well to different cameras and scenes (BS-ERGB, HDR, ECD), demonstrating strong cross-domain generalization.
- In HDR scenes, the proposed method's intensity reconstruction surpasses certain supervised methods (e.g., E2VID), as unsupervised training avoids the sim-to-real gap introduced by synthetic data.

## Highlights & Insights

- The observation that **"motion and appearance are naturally coupled"** serves as the conceptual foundation of the paper. The derivation of PhE from the Event Generation Model mathematically establishes a joint constraint on optical flow and intensity in an elegant manner.
- **Single network, dual outputs** — a single forward pass at inference simultaneously yields both optical flow and intensity, with inference time shorter than any single-task state-of-the-art method.
- PhE is free from the event collapse problem, making it an important complement to the CMax framework.
- The TC loss design is ingenious: the predicted optical flow is used to warp the intensity at the previous timestep, which is compared against the directly predicted intensity at the next timestep, forming a self-supervised signal.

## Limitations & Future Work

- Intensity reconstruction still lags behind state-of-the-art supervised methods (e.g., E2VID, HyperE2VID) on full-reference metrics (MSE, SSIM).
- The contrast threshold $C$ is fixed at 0.2, whereas the actual value varies across event camera devices.
- The U-Net architecture is relatively simple; adopting more advanced backbones (e.g., Transformer-based) may further improve performance.
- The work is currently limited to 2D optical flow and does not extend to scene flow or depth estimation.

## Related Work & Insights

- Bardow et al. (2016)'s SOFIE is among the earliest joint methods but is restricted to rotational motion; BTEB (2021) requires two separate cascaded networks.
- The CMax framework (Gallego et al.) is a central paradigm in event-based vision; combining it with PhE represents an important advance.
- Insight: The principle of "motion–appearance coupling" in event camera data may generalize to other sensor fusion problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First single-network unsupervised joint optical flow + intensity estimation method; elegant PhE derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation of both optical flow and intensity across multiple datasets; thorough ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear; experimental presentation is well-organized.
- **Value**: ⭐⭐⭐⭐ Introduces a new joint estimation paradigm for the event-based vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](../../CVPR2026/video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)

</div>

<!-- RELATED:END -->
