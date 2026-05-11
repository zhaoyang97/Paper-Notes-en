---
title: >-
  [Paper Note] GAS: Generative Avatar Synthesis from a Single Image
description: >-
  [ICCV 2025][3D Vision][Human Avatar Generation] GAS is a framework that unifies novel view synthesis and novel pose synthesis into a video generation task by combining dense appearance cues from a generalizable NeRF with…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Human Avatar Generation"
  - "Single Image"
  - "Video Diffusion"
  - "NeRF"
  - "Multi-view Consistency"
date: 2026-05-08
content_hash: 787e926ce9ed24e8
---

# GAS: Generative Avatar Synthesis from a Single Image

**Conference**: ICCV 2025
**arXiv**: [2502.06957](https://arxiv.org/abs/2502.06957)
**Code**: [Project Page](https://humansensinglab.github.io/GAS/)
**Area**: 3D Vision
**Keywords**: Human Avatar Generation, Single Image, Video Diffusion, NeRF, Multi-view Consistency

## TL;DR

GAS is a framework that unifies novel view synthesis and novel pose synthesis into a video generation task by combining dense appearance cues from a generalizable NeRF with a video diffusion model. A modality switcher decouples the two tasks, enabling view-consistent and temporally coherent human avatar generation from a single image.

## Background & Motivation

Human avatar generation has broad applications in gaming, film, sports, and telepresence, yet existing techniques typically require expensive capture setups.

**Two major camps of prior methods and their limitations**:

**Regression-based generalizable methods** (GHNeRF, GPS-Gaussian, etc.):
   - Incorporate 3D human priors (SMPL) to support sparse or even single-view input
   - However, the regression nature causes **many-to-one mapping averaging**, resulting in blurry outputs
   - Limited to rigid deformation and unable to model clothing dynamics

**Generative diffusion-based methods** (Animate Anyone, Champ, Human4DiT):
   - Conditioned on sparse human templates (depth/normal maps) to produce high-quality results
   - However, the **gap** between sparse conditioning signals and real appearance leads to multi-view flickering and temporal inconsistency

**Core insight of GAS**: Using **dense appearance cues** reconstructed by NeRF as diffusion model conditions — providing richer structural information than sparse normal maps and bridging the gap between conditioning signals and real appearance.

## Method

### Overall Architecture

Two-stage training:
1. **Train a generalizable human NeRF**: Trained on multi-view datasets to support rendering arbitrary views/poses from a single image
2. **Train a video diffusion model**: Conditioned on NeRF renderings and SMPL normal maps to learn the distribution shift

### Stage 1: Generalizable Human NeRF

Based on a single-view generalizable NeRF, target-space points are transformed into the SMPL canonical space via inverse LBS for feature querying:

$$\boldsymbol{\sigma}(\boldsymbol{x}), \boldsymbol{c}(\boldsymbol{x}) = \mathcal{F}(\boldsymbol{x}, \boldsymbol{p}, \gamma_d(\boldsymbol{d}))$$

Although NeRF renderings are blurry (due to mean regression), they provide dense, 3D-consistent appearance conditioning.

### Stage 2: Video Diffusion Model

Built on Stable Video Diffusion (SVD), three conditioning inputs are fused:

1. **NeRF rendering** $C_{\text{nerf}}$: VAE encoding + small CNN → dense appearance cues
2. **SMPL normal map** $C_{\text{smpl}}$: 2D convolution → geometric structure cues
3. **Reference image** $C_{\text{vae}}$: VAE encoding → appearance preservation

$C_{\text{nerf}}$ and $C_{\text{smpl}}$ are fused via element-wise addition and injected into the first convolutional layer of the UNet.

### Unified View-Pose Synthesis

**Core innovation**: Novel view synthesis and novel pose synthesis are unified as a video generation task:
- Novel view: Fixed pose, varying camera trajectory $\{P_1, ..., P_T\}$
- Novel pose: Fixed camera, varying SMPL pose $\{\boldsymbol{\theta}_1, ..., \boldsymbol{\theta}_T\}$

Both tasks share model parameters; in-the-wild dynamic videos used for pose synthesis training naturally transfer to view synthesis, improving generalization.

### Modality Switcher

Naive joint training causes dynamic motion to corrupt view consistency. A one-hot switcher $\boldsymbol{s}$ is concatenated with the timestep embedding and injected into the UNet:

$$\mathcal{L}_{\mathcal{U}_\theta} = \mathbb{E}[\|\epsilon - \mathcal{U}_\theta(Z_t, t, \boldsymbol{h}_{\text{clip}}, C_{\text{vae}}, C_{\text{nerf}}, C_{\text{smpl}}, \boldsymbol{s})\|]$$

The switcher enables the network to distinguish: view synthesis → prioritize view consistency; pose synthesis → prioritize realistic deformation.

### Training Details

- NeRF: Trained on MVHumanNet
- Diffusion model: Initialized from SVD 1.1, trained on 8× A100 GPUs for 3 days, 150k iterations
- Inference CFG: Triangular CFG for view synthesis (front 1 → back 2 → front 1); fixed CFG=2 for pose synthesis

## Key Experimental Results

### Novel View Synthesis (Main Results)

| Method | THuman PSNR↑ | 2K2K PSNR↑ | THuman LPIPS↓ | 2K2K FVD↓ |
|:---|:---:|:---:|:---:|:---:|
| Animate Anyone | 22.48 | 18.48 | 0.061 | 1422.1 |
| Champ | 20.96 | 22.14 | 0.074 | 480.3 |
| Animate Anyone* | 25.20 | 26.22 | 0.046 | 286.4 |
| Champ* | 23.89 | 25.66 | 0.054 | 279.3 |
| **GAS (Ours)** | **26.77** | **28.82** | **0.041** | **191.3** |

*denotes versions fine-tuned on the 3D scan dataset in this paper. GAS outperforms all baselines by a substantial margin across all metrics.

### Novel Pose Synthesis

| Method | TikTok PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|:---|:---:|:---:|:---:|:---:|
| Animate Anyone | 17.21 | 0.762 | 0.225 | 1274.1 |
| Champ | 18.48 | 0.806 | 0.182 | 585.0 |
| Champ* | 18.57 | 0.797 | 0.187 | 893.7 |
| **GAS (Ours)** | **19.11** | **0.833** | **0.176** | **362.0** |

### Ablation Study

| Ablation | Key Findings |
|:---|:---|
| w/o NeRF conditioning | Using only SMPL normal maps as conditions leads to appearance inconsistency (multi-view flickering) |
| w/o modality switcher | Joint training impairs view consistency (dynamic motion interference) |
| 3D scan training only | Insufficient generalization; poor quality on in-the-wild data |
| + Internet videos | Parameter sharing improves view synthesis generalization (gains on both in-domain and out-of-domain) |

### Key Findings

1. **Critical role of NeRF conditioning**: Dense appearance cues are foundational to multi-view consistency; sparse normal maps are insufficient
2. **Cross-task transfer via parameter sharing**: The diversity from pose synthesis training naturally improves view synthesis quality
3. **Necessity of the modality switcher**: Different tasks impose different notions of "consistency" and require explicit decoupling
4. **Triangular CFG**: Linearly increasing CFG from front to back views for view synthesis effectively balances fidelity and generation quality

## Highlights & Insights

1. **Dense conditioning over sparse conditioning**: Using NeRF renderings instead of normal maps as diffusion conditions is a practically elegant and simple idea
2. **Dual-task unification**: A single model handles both view and pose synthesis, achieving high parameter efficiency and mutual reinforcement
3. **Practical training data strategy**: Mixed training with 3D scans (small-scale, precise) and Internet videos (large-scale, diverse)

## Limitations & Future Work

1. Relies on the accuracy of SMPL fitting for NeRF; inaccurate SMPL leads to rendering artifacts that propagate into the diffusion model
2. Currently limited to 20-frame video generation; long sequences require a sliding window approach
3. NeRF rendering quality is limited in heavily occluded regions (e.g., fully rear-facing views)

## Related Work & Insights

- **Generalizable human NeRF**: GHNeRF, GPS-Gaussian, EVA3D
- **Generative human animation**: Animate Anyone, Champ, Human4DiT
- **Video diffusion models**: SVD, Sora

## Rating

- Novelty: ⭐⭐⭐⭐ — Dense NeRF conditioning combined with a unified view/pose video generation framework
- Technical Depth: ⭐⭐⭐⭐ — Targeted design of the modality switcher and CFG strategy
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple datasets (3D scans + multi-view video + TikTok), comprehensive comparisons and ablations
- Value: ⭐⭐⭐⭐ — Single-image input with view and pose controllability; broad application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[ICCV 2025\] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior](driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)
- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)

</div>

<!-- RELATED:END -->
