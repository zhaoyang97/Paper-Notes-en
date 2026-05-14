---
title: >-
  [Paper Note] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] MoBGS proposes an end-to-end dynamic deblurring 3D Gaussian Splatting framework that reconstructs sharp spatiotemporal novel views from blurry monocular video via two core mo…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "motion deblurring"
  - "dynamic novel view synthesis"
  - "Neural ODE"
  - "monocular video"
date: 2026-05-08
content_hash: 3e4bf29c2470b14e
---

# MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video

**Conference**: AAAI 2026
**arXiv**: [2504.15122](https://arxiv.org/abs/2504.15122)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, motion deblurring, dynamic novel view synthesis, Neural ODE, monocular video

## TL;DR

MoBGS proposes an end-to-end dynamic deblurring 3D Gaussian Splatting framework that reconstructs sharp spatiotemporal novel views from blurry monocular video via two core modules — Blur-adaptive Latent Camera Estimation (BLCE) and Latent Camera-induced Exposure Estimation (LCEE) — achieving substantial improvements over existing state-of-the-art methods on the Stereo Blur dataset.

## Background & Motivation

Novel View Synthesis (NVS) has made remarkable progress in recent years and is widely applied in VR, AR, and film production. However, existing dynamic NVS methods (e.g., D3DGS, 4DGS, SplineGS) are highly dependent on the quality of the input 2D observations. In casually captured everyday videos, motion blur frequently occurs due to fast-moving objects or camera shake, severely degrading the rendering quality of NVS methods. This is because NVS relies on accurate reconstruction of scene geometry and appearance, and the loss of sharp details caused by blur is "baked" into the 3D representation, producing blurry or ringing artifacts.

Several deblurring NVS methods have been proposed, including DeblurNeRF, BAD-NeRF, and BAD-GS, which model the blur process by estimating camera trajectories during the exposure period. However, these methods primarily address **static scene** deblurring and lack dedicated modeling of dynamic object motion. A small number of methods targeting dynamic scenes (DyBluRF, MoBluRF, Deblur4DGS) have begun to tackle the joint blur problem arising from global camera motion and local object motion, but two key limitations remain: (1) latent camera pose estimation lacks guidance from the degree of blur in the input; and (2) exposure time estimation does not account for the co-occurrence of global camera motion blur and local object motion blur.

## Core Problem

How to simultaneously and accurately model global camera motion blur and local dynamic object motion blur from blurry monocular video, enabling end-to-end sharp dynamic novel view synthesis?

## Method

### Overall Architecture

MoBGS is built upon SplineGS, representing the scene as static 3D Gaussians $\{G^{\text{st}}_i\}_{i=1}^{n^{\text{st}}}$ and dynamic 3D Gaussians $\{G^{\text{dy}}_i\}_{i=1}^{n^{\text{dy}}}$, where the latter model smooth motion trajectories via spline curves. Given $N_f$ frames of blurry monocular video $\{\bm{B}_t\}_{t=1}^{N_f}$ and their camera poses $\{\bm{\mathcal{P}}_t\}$, MoBGS disentangles global camera motion blur and local object motion blur through two core modules: (1) BLCE for estimating latent camera poses, and (2) LCEE for estimating latent exposure time. The final rendered blurry frame is approximated by averaging $N_l$ latent sharp frames:

$$\hat{\bm{B}}_t = \frac{1}{N_l} \sum_{k=1}^{N_l} \hat{\bm{C}}_{\hat{\tau}_t^{(k)}, \hat{\bm{\mathcal{P}}}_t^{(k)}}$$

### Key Design 1: Blur-adaptive Latent Camera Estimation (BLCE)

The core idea of BLCE is to leverage the blur intensity of input frames as a prior to guide latent camera pose estimation.

**Blur Score Computation**: Exploiting the observation that blurry frames exhibit a higher proportion of low-frequency components in the frequency domain, a 2D DFT is applied to input frame $\bm{B}_t$, and the blur score is defined as:

$$\beta_t = \frac{\sum_{\xi \in \Lambda} M_t(\xi)}{\sum_{\xi} M_t(\xi)}, \quad M_t = |\tilde{\mathcal{F}}(\bm{B}_t)|$$

where $\Lambda$ is a center-cropped square region of side length $s$ (set to $s=20$ in experiments).

**Blur Feature Extraction**: A blur feature is extracted via a shallow MLP $F_\theta$ and positional encoding $\phi(\cdot)$: $\bm{\Phi}_t = F_\theta(\phi(\beta_t))$.

**Blur-adaptive Neural ODE**: Unlike the standard Neural ODEs used in prior methods (SMURF, CRiM-GS), MoBGS injects the blur feature $\bm{\Phi}_t$ into the ODE solver, enabling it to adaptively adjust according to the blur intensity of each frame. The initial latent feature is encoded from the camera pose as $\textbf{z}_t(u_0) = F_{\theta_{\text{enc}}}(\bm{\mathcal{P}}_t)$, and a series of latent vectors is obtained through integration:

$$\textbf{z}_t(u_k) = \textbf{z}_t(u_0) + \int_{u_0}^{u_k} f(\textbf{z}_t(u), u, \bm{\Phi}_t; \psi) \, du$$

Each $\textbf{z}_t(u_k)$ is decoded to predict a screw axis $(\bm{\omega}_t^{(k)}; \bm{v}_t^{(k)}) \in \mathbb{R}^6$, and the latent camera pose is obtained via residual transformation: $\hat{\bm{\mathcal{P}}}_t^{(k)} = \bm{\mathcal{P}}_t \Psi(\bm{\omega}_t^{(k)}, \bm{v}_t^{(k)})$.

### Key Design 2: Latent Camera-induced Exposure Estimation (LCEE)

The core insight of LCEE is that global camera motion blur and local object motion blur occur **within the same exposure time interval**. Therefore, the latent camera poses estimated by BLCE can be used to infer the exposure duration.

Specifically, LCEE estimates the latent exposure time $\hat{\mathcal{T}}_t$ by comparing the ratio of 2D displacements of static 3D Gaussian means $\bm{\mu}_i^{\text{st}}$ projected onto the image plane over two intervals:

$$\hat{\mathcal{T}}_t = \frac{2}{n^{\text{st}}} \sum_{i=1}^{n^{\text{st}}} \frac{D(\hat{\bm{\mathcal{P}}}_t^{(1)}, \hat{\bm{\mathcal{P}}}_t^{(N_l)}, \bm{\mu}_i^{\text{st}}) + \epsilon}{D(\bm{\mathcal{P}}_{t-1}, \bm{\mathcal{P}}_{t+1}, \bm{\mu}_i^{\text{st}}) + \epsilon}$$

where $D(\cdot)$ computes the projected 2D displacement of a static Gaussian under two camera poses. This approach directly ties exposure time estimation to camera motion magnitude, requiring no additional learnable parameters or manual tuning.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rgb}} \mathcal{L}_{\text{rgb}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}$$

where $\mathcal{L}_{\text{rgb}}$ is the L1 loss between the input blurry frame and the rendered blurry frame, and $\mathcal{L}_{\text{depth}}$ is the L1 loss between the rendered depth and the GT depth. $\lambda_{\text{rgb}}=1.0$, $\lambda_{\text{depth}}=0.2$.

## Key Experimental Results

### Dynamic Deblurring NVS Results on Stereo Blur Dataset (Full Image)

| Method | LPIPS↓ | MUSIQ↑ | tOF↓ | PSNR↑ | FPS |
|--------|--------|--------|------|-------|-----|
| SplineGS | 0.141 | 42.88 | 1.409 | 26.41 | 300 |
| GShiftNet + SplineGS | 0.074 | 55.29 | 0.748 | 26.54 | 329 |
| DyBluRF | 0.079 | 50.82 | 0.889 | 25.62 | 0.2 |
| MoBluRF | 0.078 | 51.84 | 0.816 | 25.69 | 0.1 |
| **MoBGS (Ours)** | **0.050** | **57.64** | **0.507** | **28.80** | **480** |

### Dynamic Region Results

| Method | LPIPS↓ | tOF↓ | PSNR↑ |
|--------|--------|------|-------|
| SplineGS | 0.168 | 1.417 | 22.31 |
| DyBluRF | 0.158 | 1.367 | 19.41 |
| MoBluRF | 0.155 | 1.456 | 20.63 |
| **MoBGS (Ours)** | **0.096** | **1.093** | **23.41** |

### Ablation Study

#### LCEE Ablation (Dynamic Region)

| Exposure Time Setting | LPIPS↓ | tOF↓ | PSNR↑ |
|-----------------------|--------|------|-------|
| Fixed $\hat{\mathcal{T}}_t=0.0$ | 0.120 | 1.237 | 23.20 |
| Fixed $\hat{\mathcal{T}}_t=0.5$ | 0.117 | 1.276 | 23.12 |
| Learnable $\hat{\mathcal{T}}_t$ | 0.128 | 1.261 | 23.24 |
| **LCEE (Ours)** | **0.096** | **1.093** | **23.41** |

#### $N_l$ Ablation

| $N_l$ | LPIPS↓ | MUSIQ↑ | tOF↓ | PSNR↑ | Training Time |
|-------|--------|--------|------|-------|---------------|
| 3 | 0.069 | 53.66 | 0.594 | 28.79 | 0.8h |
| 5 | 0.055 | 56.48 | 0.526 | 28.78 | 1.0h |
| **9 (Ours)** | **0.050** | **57.64** | **0.507** | **28.80** | **1.5h** |

## Highlights & Insights

- **Blur-adaptive Neural ODE**: Injecting blur scores/features into the ODE solver allows latent camera pose estimation to adapt to the blur level of each individual frame, representing a key improvement over existing Neural ODE-based approaches.
- **Elegant Exposure Time Estimation**: LCEE leverages the physical prior that global camera motion and local object motion share the same exposure interval, estimating exposure duration via 2D projected displacement ratios without any additional learnable parameters.
- **Comprehensive Performance Gains**: On the Stereo Blur dataset, MoBGS achieves an LPIPS of 0.050 (a 36% improvement over MoBluRF's 0.078) and a PSNR of 28.80 dB (a 3.1 dB gain), while reaching a rendering speed of ~480 FPS (4800× faster than MoBluRF).

## Limitations & Future Work

- **Per-scene Optimization Paradigm**: Each new scene requires training from scratch, with no generalization to unseen scenes. The authors suggest that future work could integrate the deblurring module into feed-forward generalizable 3DGS frameworks.
- **Dependence on Depth Supervision**: The method relies on a pretrained monocular depth estimation model (UniDepth) for depth ground truth, and reconstruction accuracy may be affected by depth quality.
- **Limited Dataset Scale**: The Stereo Blur dataset contains only 6 scenes, and the DAVIS dataset lacks NVS ground truth, leaving room for more comprehensive evaluation.

## Related Work & Insights

Compared to static deblurring NVS methods (BAD-GS, Deblurring 3DGS), MoBGS substantially outperforms them, as these methods cannot handle object motion blur in dynamic scenes. Compared to cascade approaches (preprocessing with a 2D deblurring network before NVS), MoBGS's end-to-end formulation yields a 1.3–2 dB PSNR advantage, demonstrating that jointly optimizing 3D reconstruction and deblurring is superior to decoupled processing. Among recent dynamic deblurring NVS methods: DyBluRF uses a fixed exposure time and is NeRF-based with extremely slow rendering (0.2 FPS); MoBluRF decomposes global/local blur but neglects exposure time estimation and is also NeRF-based (0.1 FPS); Deblur4DGS, though 3DGS-based, uses an unconstrained learnable exposure time that leads to inconsistent deblurring. MoBGS achieves significant improvements across all metrics while attaining real-time rendering speed.

The following insights emerge for broader research:
- **Frequency-domain Priors**: The blur score design exploits the frequency-domain property that blurry images exhibit a higher proportion of low-frequency components, a principle generalizable to other image degradation-aware tasks.
- **Conditioning Neural ODE Solvers**: The idea of injecting external priors into a Neural ODE (Blur-adaptive Neural ODE) is broadly applicable to other scenarios requiring conditionally governed continuous dynamical modeling.
- **Global–Local Motion Consistency Constraints**: The LCEE constraint that global camera motion and local object motion share the same exposure interval can inspire other tasks requiring cross-scale motion consistency modeling, such as video stabilization and motion segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Both BLCE and LCEE are cleverly designed; the blur-adaptive Neural ODE and projection-displacement-ratio-based exposure estimation represent meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies are provided (BLCE, LCEE, $N_l$, $s$), with multiple baselines and qualitative visualizations.
- Writing Quality: ⭐⭐⭐⭐ — The paper is clearly structured with well-motivated problem formulation, coherent methodological derivation, and intuitive figures and tables.
- Value: ⭐⭐⭐⭐ — End-to-end dynamic deblurring NVS addresses an important practical problem, and the dual gains in performance and rendering speed are of high applied value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/3d_vision/motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2026/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](../../ICCV2025/3d_vision/evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)

</div>

<!-- RELATED:END -->
