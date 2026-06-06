---
title: >-
  [Paper Note] Image Restoration via Diffusion Models with Dynamic Resolution
description: >-
  [ICML 2026][Image Generation][Dynamic resolution diffusion] SubDAPS / SubDAPS++ integrates pixel-space diffusion restoration methods such as DPS and DAPS into a "dynamic resolution diffusion model" framework—sampling in…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Dynamic resolution diffusion"
  - "DAPS"
  - "Conjugate Gradient"
  - "predictor-corrector"
  - "ISR"
date: 2026-05-08
content_hash: 39595c06efd03f8e
---

# Image Restoration via Diffusion Models with Dynamic Resolution

**Conference**: ICML 2026  
**arXiv**: [2605.14267](https://arxiv.org/abs/2605.14267)  
**Code**: https://github.com/StarNextDay/SubDAPS (Available)  
**Area**: Diffusion Models / Image Restoration / Accelerated Inference  
**Keywords**: Dynamic resolution diffusion, DAPS, Conjugate Gradient, predictor-corrector, ISR

## TL;DR
SubDAPS / SubDAPS++ integrates pixel-space diffusion restoration methods such as DPS and DAPS into a "dynamic resolution diffusion model" framework—sampling in $64^2 / 128^2$ subspaces during early stages and returning to $256^2$ full resolution later. By replacing Langevin dynamics with Conjugate Gradient, employing threshold-based switching between stochastic and deterministic sampling, and appending a corrector step that requires no additional network evaluations, it outpaces pixel and latent diffusion methods across four linear and two non-linear restoration tasks in both speed and most performance metrics.

## Background & Motivation

**Background**: Diffusion models have demonstrated strong performance in image restoration. Pixel-space methods (DPS, DDRM, DDNM, DiffPIR, DAPS, AdaPS) perform repeated sampling directly on $256^2 \times 3$ grids, offering high inversion quality but at slow speeds. Latent-space methods (PSLD, ReSample, LatentDAPS, SILO) sample in VAE latent spaces; while theoretically cheaper, the requirement for VAE encoding/decoding at each step often makes them slower than pixel-space counterparts.

**Limitations of Prior Work**: (a) Pixel-space methods compute entirely at high dimensions, where a significant portion of computation in early stages is redundantly spent on "sketching global structure." (b) Latent-space methods reduce latent dimensions but incur the overhead of repeated encoding/decoding, and the VAE itself limits the achievable reconstruction quality.

**Key Challenge**: There is a need to "save computation early" while "drawing details late." Neither pixel nor latent approaches are optimal at both ends; a "dimensionality-on-demand" diffusion process is required.

**Goal**: (a) Migrate dynamic resolution diffusion (Subspace Diffusion / UDPM / DVDP / DiMR / Fresco) from pure generation to general image restoration. (b) Enable pixel-space algorithms like DPS and DAPS to maintain measurement consistency within a dynamic resolution framework. (c) Further optimize noise injection, measurement updates, and trajectory correction to push both quality and speed.

**Key Insight**: The authors leverage the insight from Jing et al. (2022) that early timesteps mainly deal with low-frequency components which can be handled in a low-resolution subspace. This naturally aligns with ISR tasks that "restore global structure before filling in high-frequency details."

**Core Idea**: First, a pretrained pixel DM is finetuned with shared weights across three resolutions ($64^2 / 128^2 / 256^2$). DPS and DAPS are adapted into SubDPS and SubDAPS as baselines. Finally, SubDAPS++ is synthesized by introducing three improvements to SubDAPS: CG for measurement solving, deterministic switching, and a predictor-corrector pass.

## Method

### Overall Architecture
Inference proceeds backward along time $0 = t_0 < t_1 < \dots < t_N = T$, where each timestep is associated with a dimension $d_i$ such that $d = d_0 \geq d_1 \geq \dots \geq d_N$ (using $256^2 \to 128^2 \to 64^2$ in the paper). Each step involves three actions: (1) using $\bm{x}_\theta(\bm{x}_{t_i}, t_i)$ to obtain an unconditional prediction $\hat{\bm{x}}_0$; (2) using the measurement to correct $\hat{\bm{x}}_0$ into a consistent $\tilde{\bm{x}}_0$; (3) if the step transitions from $d_i$ to $d_{i-1}$, the state is projected up and noise is injected to match the diffusion prior; otherwise, a convergence criterion determines whether to continue with stochastic noise or switch to a deterministic update. SubDAPS++ adds a predictor-corrector pass after the main loop to refine the trajectory without additional network evaluations.

### Key Designs

1. **SubDPS / SubDAPS: Implementing Measurement Consistency in Dynamic Resolution**:

    - **Function**: Enables classical DPS / DAPS to perform corrections using observations within subspaces and naturally handle resolution transitions.
    - **Mechanism**: For DPS, when $d_{i-1} = d_i$, the likelihood gradient is rewritten as $\nabla_{\bm{x}_{t_i}} \log p_{t_i}(\bm{y} | \bm{x}_{t_i}) \approx -\zeta_{t_i} \nabla_{\bm{x}_{t_i}} \|\bm{y} - \mathcal{A}(\bm{U}_i \bm{x}_\theta(\bm{x}_{t_i}, t_i))\|^2$, using an upsampling matrix $\bm{U}_i$ to project subspace predictions back to the original image domain for measurement calculation. At dimension switches where $d_{i-1} \neq d_i$, inspired by the idea in DAPS that early stochastic noise can correct accumulated errors, the authors omit specialized correction and set $\bm{x}_{t_{i-1}} = \alpha_{t_{i-1}} \dot{\bm{U}}_i \bm{x}_\theta(\bm{x}_{t_i}, t_i) + \sigma_{t_{i-1}} \bm{\epsilon}_i$. For DAPS, the optimization problem $\hat{\bm{x}}_0^{t_i} = \arg\min_{\bar{\bm{x}}_0} \big( r_{t_i} \|\bar{\bm{x}}_0 - \tilde{\bm{x}}_0^{t_i}\|^2 + \|\bm{y} - \mathcal{A}(\bm{U}_i \bar{\bm{x}}_0)\|^2 \big)$ is solved before performing similar stochastic sampling.
    - **Design Motivation**: Since the gradient trick of DPS and the decoupled trajectory of DAPS were designed for pixel-space, the authors introduce the $\bm{U}_i$ upsampler before the measurement operator to make them self-consistent in subspaces and at transition points—an engineered yet elegant "operator modification."

2. **SubDAPS++ Deterministic Switching + CG Measurement**:

    - **Function**: Simultaneously reduces artifacts at low timesteps and lowers iteration costs in SubDAPS.
    - **Mechanism**: (a) Deterministic Switching. Defining $h = \min\{i: d_{i-1} \neq d_i\}$ as the index of the final dimension change; when $i < h$ (stable at full resolution) and $\|\bm{x}_\theta(\bm{x}_{t_i}, t_i) - \hat{\bm{x}}_0^{t_i}\|^2 \leq \tau$, a deterministic update $\bm{x}_{t_{i-1}} = \alpha_{t_{i-1}} \hat{\bm{x}}_0^{t_i} + \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}}(\bm{x}_{t_i} - \alpha_{t_i} \hat{\bm{x}}_0^{t_i})$ is used; otherwise, noise injection continues. (b) Replacing Langevin with Conjugate Gradient. SubDAPS uses Langevin dynamics to solve measurement updates, which is slow and limited to differentiable operators. SubDAPS++ adopts Fletcher-Reeves CG: each step linearizes $\mathcal{A}(\bm{U}_i(\bar{\bm{x}}_0^{(j)} + \alpha \bm{d}_j))$ via a first-order Taylor expansion to get a closed-form line search $\alpha_j = (\bm{g}_j^\top \bm{d}_j) / (r_{t_i} \bm{d}_j^\top \bm{d}_j + \bm{\omega}_j^\top \bm{\omega}_j)$, updating the search direction with $\bm{d}_{j+1} = \bm{g}_{j+1} + \frac{\bm{g}_{j+1}^\top \bm{g}_{j+1}}{\bm{g}_j^\top \bm{g}_j} \bm{d}_j$.
    - **Design Motivation**: Stochastic noise at low timesteps can damage the diffusion prior and create artifacts. The authors use "stable dimension + converged prediction" to characterize the timing for switching to deterministic updates. The benefit of CG is that the closed-form line search applies to both linear and non-linear measurements, proving faster and requiring fewer hyperparameters than Langevin.

3. **Predictor-Corrector Trajectory Refinement without Extra Network Evaluations**:

    - **Function**: Re-evaluates the trajectory after the main loop to remove deviations introduced by the stochastic sampling.
    - **Mechanism**: Drawing from the second-order corrector form of UniPC: $\bm{x}_{t_{i-1}}^c = \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}} \dot{\bm{U}}_i \bm{x}_{t_i}^c - \left(\sigma_{t_{i-1}} \frac{\alpha_{t_i}}{\sigma_{t_i}} - \alpha_{t_{i-1}}\right) \hat{\bm{x}}_0^{t_{i-1}} - \sigma_{t_{i-1}} \mathcal{I}_i \frac{\hat{\bm{x}}_0^{t_{i-1}} - \dot{\bm{U}}_i \hat{\bm{x}}_0^{t_i}}{\lambda_{t_{i-1}} - \lambda_{t_i}}$, where $\lambda_t = \log(\alpha_t/\sigma_t)$ is the half log-SNR. This step completely reuses $\hat{\bm{x}}_0^{t_i}$ values cached during the main loop without invoking the neural network.
    - **Design Motivation**: Noise is added in the main loop to prevent divergence but biases the trajectory. The UniPC-style corrector uses an analytical formula to pull the state toward a "standard" diffusion trajectory, yielding performance gains almost for free.

### Loss & Training
- **Training**: Finetuned on the Dhariwal-Nichol pretrained pixel DM. The objective jointly denoises across $\bm{x}_0$, $\tilde{\bm{U}}^\top \bm{x}_0$, and $\hat{\bm{U}}^\top \bm{x}_0$ resolutions, allowing a single network to handle $256/128/64$ dimensions. This is finetuned once and shared across all downstream tasks.
- **Inference**: Differs from SubDAPS by simplifying the multi-step ODE solver for $\tilde{\bm{x}}_0$ into a single evaluation $\tilde{\bm{x}}_0 = \bm{x}_\theta(\bm{x}_{t_i}, t_i)$; measurement consistency uses $J$ CG steps; the switching threshold $\tau$, upsampling index $h$, noise level $\sigma$, and iteration count $N$ are hyperparameters.

## Key Experimental Results

### Main Results

| Task (256² FFHQ) | Type | DiffPIR | MGPS | DAPS | AdaPS | LatentDAPS | **SubDAPS++** |
|---|---|---|---|---|---|---|---|
| Inpainting 70% rand, PSNR ↑ | pixel/latent/dynamic | 32.16 | 31.41 | 30.68 | **32.34** | 31.17 | 32.21 |
| Inpainting 70% rand, LPIPS ↓ | | 0.052 | 0.050 | 0.073 | 0.057 | 0.090 | **0.056** |
| SR ×4, PSNR ↑ | | 27.64 | 27.58 | 28.88 | 27.34 | 28.56 | **29.34** |
| SR ×4, LPIPS ↓ | | 0.116 | 0.110 | 0.162 | **0.090** | 0.174 | 0.157 |
| Gaussian Deblur (FFHQ), PSNR ↑ | | 28.07 | 27.78 | 28.91 | 27.02 | 28.50 | — (≈ DAPS) |
| Motion Deblur (FFHQ), PSNR ↑ | | 26.95 | 26.82 | 28.27 | 27.06 | 27.58 | **28.28** |

| Task (256² ImageNet) | Type | DPS | DAPS | LatentDAPS | **SubDAPS++** |
|---|---|---|---|---|---|
| Inpainting 70%, PSNR ↑ | | 25.33 | 27.63 | 27.33 | **28.61** |
| Inpainting 70%, FID ↓ | | 141.99 | 56.73 | 85.24 | **49.15** |
| SR ×4, PSNR ↑ | | 21.68 | 25.54 | 25.43 | **25.79** |
| SR ×4, LPIPS ↓ | | 0.432 | 0.354 | 0.377 | 0.358 |

### Ablation Study

| Configuration | Description |
|---|---|
| SubDPS | Naive migration of DPS to dynamic resolution; performance matches DPS (weakest), validating framework feasibility. |
| SubDAPS | Yields results comparable to or better than DAPS, with speedups due to subspace processing. |
| SubDAPS + CG replacement | Faster measurement updates and compatibility with non-linear operators. |
| SubDAPS + deterministic switching | Reduces artifacts at low timesteps; increases PSNR and decreases LPIPS. |
| SubDAPS + corrector | NN-free second-order refinement; provides a free performance boost. |
| Full Stack = SubDAPS++ | Ranks first or second in most metrics across 6 tasks. |

### Key Findings
- Dynamic resolution is better suited for restoration than latent routes: It avoids the overhead of repeated VAE encoding/decoding and lacks VAE reconstruction bottlenecks, making SubDAPS++ faster and more accurate than LatentDAPS across nearly all datasets.
- The "low-resolution structure first, full-resolution detail later" approach is highly effective for tasks like ISR and inpainting that naturally proceed from global to local; global degradation scenarios like motion deblur benefit significantly.
- Controlling the "stochastic to deterministic" switch based on both prediction convergence and dimension stability is more robust than fixed timestep switching—it allows the current trajectory to determine when to use SDE vs ODE.
- The CG update with first-order Taylor closed-form line search allows measurement updates for any differentiable $\mathcal{A}$ (not just linear), offering better extensibility than methods like DDRM/DDNM.

## Highlights & Insights
- Migrating dynamic resolution diffusion from generation to restoration is the most insightful contribution—the same concept takes on different utility in a restoration context, where "coarse-to-fine" is a natural inductive bias.
- The synthesis of three "patches" (CG, switching, corrector) from different domains (numerical optimization, deterministic ODE switching, UniPC predictor-corrector) is a model of effective engineering composition.
- Sharing a single finetuned multi-resolution DM across all tasks and resolutions eliminates the need to train new models for specific resolutions, making it industry-friendly.
- Using CG + first-order Taylor for closed-form line search in measurement updates is a refined detail, providing fast convergence and general applicability to differentiable operators.

## Limitations & Future Work
- Dynamic resolution is limited to three scales; the need for more layers or complex upsampling for higher resolutions (e.g., above $1024^2$) remains unexplored.
- The switching threshold $\tau$ is a fixed hyperparameter; the authors suggest individual tuning for certain tasks, indicating a need for adaptive $\tau$.
- The corrector is applied post-hoc; it does not correct errors mid-generation. Severe hallucinations occurring early cannot be reversed later.
- Comparisons with the latest LDM restoration (e.g., SD-based methods) are limited, particularly in natural image super-resolution where SD-based approaches often excel in perceptual metrics.

## Related Work & Insights
- **vs DPS / DAPS**: This work serves as a "dynamic resolution rewrite" of DPS/DAPS—retaining their measurement correction logic while moving each step to the appropriate dimension for speed and quality.
- **vs PSLD / ReSample / LatentDAPS**: Latent routes rely on VAE for dimensionality reduction at the cost of repeated encoder/decoder calls; dynamic resolution bypasses VAE, achieving a better balance of speed and quality.
- **vs Subspace DM / UDPM / DiMR / Fresco**: These are generation-focused; this work presents the first systematic restoration version, solving the challenge of maintaining measurement consistency across dimension switches.
- **vs UniPC**: While UniPC's corrector is for deterministic ODEs, this work adapts it as a "post-hoc free correction" after a stochastic main loop, demonstrating clever cross-paradigm application.
- **Inspiration**: The dynamic resolution concept could be applied inversely—e.g., switching spatial resolution over time segments in video generation or LoD in 3D reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of dynamic resolution DM to general image restoration with DPS/DAPS adaptation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across 4 linear + 2 non-linear tasks on FFHQ/ImageNet against three baseline classes.
- Writing Quality: ⭐⭐⭐⭐ Clear algorithms, derivations, and logic for switching conditions.
- Value: ⭐⭐⭐⭐ Simultaneously faster and more accurate without VAE reliance, offering a practical acceleration scheme for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Eliminating VAE for Fast and High-Resolution Generative Detail Restoration](../../ICLR2026/image_generation/eliminating_vae_for_fast_and_high-resolution_generative_detail_restoration.md)
- [\[ICML 2026\] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)
- [\[ICML 2026\] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)
- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](../../ICLR2026/image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)

</div>

<!-- RELATED:END -->
