---
title: >-
  [Paper Note] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] MoBGS proposes an end-to-end dynamic deblurring 3D Gaussian Splatting framework. Through two core modules, Blur-adaptive Latent Camera Estimation (BLCE) and Latent Camera-induced Exposure Estimation (LCEE), it reconstructs sharp spatiotemporal novel views from blurry monocular videos, significantly outperforming existing SOTA methods on the Stereo Blur dataset.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "motion deblurring"
  - "dynamic novel view synthesis"
  - "Neural ODE"
  - "monocular video"
date: 2026-05-08
content_hash: dbe7d8baa708a402
---

# MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video

**Conference**: AAAI 2026  
**arXiv**: [2504.15122](https://arxiv.org/abs/2504.15122)  
**Code**: TBD  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, motion deblurring, dynamic novel view synthesis, Neural ODE, monocular video  

## TL;DR

MoBGS proposes an end-to-end dynamic deblurring 3D Gaussian Splatting framework. Through two core modules, Blur-adaptive Latent Camera Estimation (BLCE) and Latent Camera-induced Exposure Estimation (LCEE), it reconstructs sharp spatiotemporal novel views from blurry monocular videos, significantly outperforming existing SOTA methods on the Stereo Blur dataset.

## Background & Motivation

Novel View Synthesis (NVS) has made significant progress in recent years and is widely applied in fields such as VR, AR, and video production. However, existing dynamic NVS methods (e.g., D3DGS, 4DGS, SplineGS) heavily rely on the quality of input 2D observations. In everyday hand-held videos, motion blur frequently occurs due to fast-moving objects or camera shake, causing severe degradation in the rendering quality of NVS methods. This is because NVS relies on precise scene geometry and appearance reconstruction, and the lack of sharp details caused by blur is "baked" into the 3D representation, resulting in blurry or ringing artifacts.

Some deblurring NVS methods have been proposed previously, such as DeblurNeRF, BAD-NeRF, and BAD-GS, which model the blurring process by estimating the camera trajectory during exposure. However, these methods mainly focus on deblurring **static scenes**, lacking specialized modeling of dynamic object motion. A few methods targeting dynamic scenes (DyBluRF, MoBluRF, Deblur4DGS) have started to handle the joint blur problem of global camera motion and local object motion, yet they still suffer from two key limitations: (1) latent camera pose estimation lacks guidance from the input blur level; (2) exposure time estimation does not account for the synchronization relationship between global camera motion and local object motion blur.

## Core Problem

How can global camera motion blur and local dynamic object motion blur be simultaneously and accurately modeled from a blurry monocular video to achieve end-to-end sharp dynamic novel view synthesis?

## Method

### Overall Architecture

MoBGS is built upon SplineGS, representing the scene as static 3D Gaussians $\{G^{\text{st}}_i\}_{i=1}^{n^{\text{st}}}$ and dynamic 3D Gaussians $\{G^{\text{dy}}_i\}_{i=1}^{n^{\text{dy}}}$, the latter of which models smooth motion trajectories via spline curves. Given $N_f$ frames of blurry monocular videos $\{\bm{B}_t\}_{t=1}^{N_f}$ and their camera poses $\{\bm{\mathcal{P}}_t\}$, MoBGS decouples global camera motion and local object motion blur through two core modules: (1) BLCE estimates latent camera poses; (2) LCEE estimates latent exposure time. Ultimately, $N_l$ latent sharp frames are averaged to approximate the blurry frame:

$$\hat{\bm{B}}_t = \frac{1}{N_l} \sum_{k=1}^{N_l} \hat{\bm{C}}_{\hat{\tau}_t^{(k)}, \hat{\bm{\mathcal{P}}}_t^{(k)}}$$

### Key Designs

#### Key Design 1: Blur-adaptive Latent Camera Estimation (BLCE)

The core idea of BLCE is to utilize the blur intensity of the input frame as a prior to guide the estimation of latent camera poses.

**Blur Score Calculation**: Utilizing the observation that blurry frames have a higher proportion of low-frequency components in the frequency domain, a 2D DFT is performed on the input frame $\bm{B}_t$, defining the blur score as:

$$\beta_t = \frac{\sum_{\xi \in \Lambda} M_t(\xi)}{\sum_{\xi} M_t(\xi)}, \quad M_t = |\tilde{\mathcal{F}}(\bm{B}_t)|$$

where $\Lambda$ is a center-cropped square region with a side length of $s$ (experimentally, $s=20$ in the experiments).

**Blur Feature Extraction**: Extract the blur feature through a shallow MLP $F_\theta$ and positional encoding $\phi(\cdot)$: $\bm{\Phi}_t = F_\theta(\phi(\beta_t))$.

**Blur-adaptive Neural ODE**: Unlike the standard Neural ODE used in existing methods (SMURF, CRiM-GS), MoBGS injects the blur feature $\bm{\Phi}_t$ into the ODE solver, allowing it to adaptively adjust according to the blur intensity of each frame. The initial latent feature is encoded from the camera pose $\textbf{z}_t(u_0) = F_{\theta_{\text{enc}}}(\bm{\mathcal{P}}_t)$, and then integrated to obtain a series of latent vectors:

$$\textbf{z}_t(u_k) = \textbf{z}_t(u_0) + \int_{u_0}^{u_k} f(\textbf{z}_t(u), u, \bm{\Phi}_t; \psi) \, du$$

Each $\textbf{z}_t(u_k)$ is decoded to predict the screw axis $(\bm{\omega}_t^{(k)}; \bm{v}_t^{(k)}) \in \mathbb{R}^6$, and finally the latent camera pose is obtained via residual transformation: $\hat{\bm{\mathcal{P}}}_t^{(k)} = \bm{\mathcal{P}}_t \Psi(\bm{\omega}_t^{(k)}, \bm{v}_t^{(k)})$.

#### Key Design 2: Latent Camera-induced Exposure Estimation (LCEE)

The core insight of LCEE is that the blur of global camera motion and local object motion occurs within the **same exposure time interval**. Therefore, one can leverage the latent camera poses estimated by BLCE to infer the exposure duration.

Specifically, LCEE estimates the latent exposure time $\hat{\mathcal{T}}_t$ by comparing the ratio of 2D displacements of static 3D Gaussian means $\bm{\mu}_i^{\text{st}}$ on the image plane across two intervals:

$$\hat{\mathcal{T}}_t = \frac{2}{n^{\text{st}}} \sum_{i=1}^{n^{\text{st}}} \frac{D(\hat{\bm{\mathcal{P}}}_t^{(1)}, \hat{\bm{\mathcal{P}}}_t^{(N_l)}, \bm{\mu}_i^{\text{st}}) + \epsilon}{D(\bm{\mathcal{P}}_{t-1}, \bm{\mathcal{P}}_{t+1}, \bm{\mu}_i^{\text{st}}) + \epsilon}$$

where $D(\cdot)$ calculates the projected 2D displacement of the static Gaussian under two camera poses. This method directly relates the estimation of exposure time to the scale of camera motion without requiring additional learnable parameters or manual tuning.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rgb}} \mathcal{L}_{\text{rgb}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}$$

where $\mathcal{L}_{\text{rgb}}$ is the L1 loss between the input blurry frame and the rendered blurry frame, and $\mathcal{L}_{\text{depth}}$ is the L1 loss between the rendered depth and the GT depth. $\lambda_{\text{rgb}}=1.0$, $\lambda_{\text{depth}}=0.2$.

## Key Experimental Results

### Main Results

#### Dynamic Deblurring NVS Results on the Stereo Blur Dataset (Full Image)

| Method | LPIPS↓ | MUSIQ↑ | tOF↓ | PSNR↑ | FPS |
|------|--------|--------|------|-------|-----|
| SplineGS | 0.141 | 42.88 | 1.409 | 26.41 | 300 |
| GShiftNet + SplineGS | 0.074 | 55.29 | 0.748 | 26.54 | 329 |
| DyBluRF | 0.079 | 50.82 | 0.889 | 25.62 | 0.2 |
| MoBluRF | 0.078 | 51.84 | 0.816 | 25.69 | 0.1 |
| **MoBGS (Ours)** | **0.050** | **57.64** | **0.507** | **28.80** | **480** |

#### Dynamic Region Results

| Method | LPIPS↓ | tOF↓ | PSNR↑ |
|------|--------|------|-------|
| SplineGS | 0.168 | 1.417 | 22.31 |
| DyBluRF | 0.158 | 1.367 | 19.41 |
| MoBluRF | 0.155 | 1.456 | 20.63 |
| **MoBGS (Ours)** | **0.096** | **1.093** | **23.41** |

### Ablation Study

#### LCEE Ablation Study (Dynamic Region)

| Exposure Time Setting | LPIPS↓ | tOF↓ | PSNR↑ |
|------------|--------|------|-------|
| Fixed $\hat{\mathcal{T}}_t=0.0$ | 0.120 | 1.237 | 23.20 |
| Fixed $\hat{\mathcal{T}}_t=0.5$ | 0.117 | 1.276 | 23.12 |
| Learnable $\hat{\mathcal{T}}_t$ | 0.128 | 1.261 | 23.24 |
| **LCEE (Ours)** | **0.096** | **1.093** | **23.41** |

#### $N_l$ Ablation

| $N_l$ | LPIPS↓ | MUSIQ↑ | tOF↓ | PSNR↑ | Training Time |
|-------|--------|--------|------|-------|---------|
| 3 | 0.069 | 53.66 | 0.594 | 28.79 | 0.8h |
| 5 | 0.055 | 56.48 | 0.526 | 28.78 | 1.0h |
| **9 (Ours)** | **0.050** | **57.64** | **0.507** | **28.80** | **1.5h** |

## Highlights & Insights

- **Blur-adaptive Neural ODE**: Injecting the blur score/feature into the ODE solver enables adaptive adjustment of latent camera pose estimation based on the per-frame blur level, which is a key improvement over existing Neural ODE methods.
- **Elegant Exposure Time Estimation**: The LCEE method utilizes the physical prior that global camera motion and local object motion share the same exposure interval. It estimates exposure duration via 2D projection displacement ratios, eliminating the need for additional learnable parameters.
- **Comprehensive Performance Improvements**: On the Stereo Blur dataset, MoBGS achieves 0.050 in LPIPS (a 36% gain over MoBluRF's 0.078), 28.80dB in PSNR (a 3.1dB gain), while reaching a rendering speed of ~480 FPS (4800× faster than MoBluRF).

## Limitations & Future Work

- **Per-scene Optimization Paradigm**: Training from scratch is required for each new scene, preventing generalization to unseen scenes. The authors point out that future work could integrate the deblurring module into feed-forward generalizable 3DGS methods.
- **Dependency on Depth Supervision**: Relies on a pretrained monocular depth estimation model (UniDepth) to provide ground truth depth, where depth quality may affect reconstruction accuracy.
- **Limited Dataset Scale**: The Stereo Blur dataset only contains 6 scenes, and the DAVIS dataset lacks ground truth for NVS, requiring a more comprehensive future evaluation.

## Related Work & Insights

Compared to static deblurring NVS methods (BAD-GS, Deblurring 3DGS), MoBGS significantly outperforms them because the latter cannot handle object motion blur in dynamic scenes. Compared to cascade methods (performing 2D deblurring network preprocessing followed by NVS), MoBGS's end-to-end paradigm shows a 1.3-2dB advantage in PSNR, demonstrating that joint optimization of 3D reconstruction and deblurring outperforms separate processing. Compared to recent dynamic deblurring NVS methods: DyBluRF uses a fixed exposure time and is based on NeRF, making rendering extremely slow (0.2 FPS); MoBluRF decomposes global/local blur but neglects exposure time estimation, and is also NeRF-based (0.1 FPS); Deblur4DGS is based on 3DGS, but its learnable exposure time lacks constraints, leading to inconsistent deblurring. MoBGS significantly leads in all metrics while achieving real-time rendering speeds.

## Related Work & Insights

- **Utilization of Frequency Domain Priors**: The design of the blur score leverages the frequency domain characteristic that blurry images have a higher proportion of low-frequency components; this concept can be generalized to other image degradation perception tasks.
- **Conditioning ODE Solvers**: The idea of injecting external priors into Neural ODEs (Blur-adaptive Neural ODE) is highly generalizable and can be applied to other scenarios requiring conditional continuous dynamical modeling.
- **Global-Local Motion Consistency Constraints**: The constraint optimization idea in LCEE, where global camera motion and local object motion share the same exposure interval, can inspire other tasks requiring cross-scale motion consistency modeling (e.g., video stabilization, motion segmentation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — BLCE and LCEE are cleverly designed; both the blur-adaptive Neural ODE and the exposure estimation based on projected displacement ratios represent meaningful innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablation studies (BLCE, LCEE, $N_l$, $s$) are provided, along with several baseline comparisons and qualitative visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured, with logical motivations and methodological derivations, accompanied by intuitive figures.
- **Value**: ⭐⭐⭐⭐ — End-to-end dynamic deblurring NVS is an important problem in practical applications; the simultaneous improvements in performance and speed offer high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MSCD-GS: Motion-Separated Cooperative Deblurring Dynamic Reconstruction via Gaussian Splatting](../../CVPR2026/3d_vision/mscd-gs_motion-separated_cooperative_deblurring_dynamic_reconstruction_via_gauss.md)
- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/3d_vision/motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](../../CVPR2025/3d_vision/splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](../../CVPR2025/3d_vision/diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)

</div>

<!-- RELATED:END -->
