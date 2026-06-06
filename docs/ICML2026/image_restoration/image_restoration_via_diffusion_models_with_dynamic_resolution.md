---
title: >-
  [Paper Note] Image Restoration via Diffusion Models with Dynamic Resolution
description: >-
  [ICML 2026][Image Restoration][Dynamic Resolution Diffusion] SubDAPS / SubDAPS++ adapts pixel-space diffusion restoration methods like DPS and DAPS into a "dynamic resolution diffusion model" framework—sampling in $64^2…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Dynamic Resolution Diffusion"
  - "DAPS"
  - "Conjugate Gradient"
  - "predictor-corrector"
  - "ISR"
date: 2026-05-08
content_hash: f13c907fccd0885f
---

# Image Restoration via Diffusion Models with Dynamic Resolution

**Conference**: ICML 2026  
**arXiv**: [2605.14267](https://arxiv.org/abs/2605.14267)  
**Code**: https://github.com/StarNextDay/SubDAPS (available)  
**Area**: Diffusion Models / Image Restoration / Accelerated Inference  
**Keywords**: Dynamic Resolution Diffusion, DAPS, Conjugate Gradient, predictor-corrector, ISR

## TL;DR
SubDAPS / SubDAPS++ adapts pixel-space diffusion restoration methods like DPS and DAPS into a "dynamic resolution diffusion model" framework—sampling in $64^2 / 128^2$ subspaces in early stages and returning to $256^2$ full resolution later. It replaces Langevin with conjugate gradient, switches between stochastic/deterministic sampling via thresholding, and adds a corrector step that requires no extra network evaluation. On four linear and two nonlinear restoration tasks, it outperforms both pixel and latent diffusion methods on most metrics while being faster at inference.

## Background & Motivation

**Background**: Diffusion models are highly effective for image restoration. Pixel-space methods (DPS, DDRM, DDNM, DiffPIR, DAPS, AdaPS) sample repeatedly on $256^2 \times 3$, achieving high inversion quality but at a slow speed. Latent-space methods (PSLD, ReSample, LatentDAPS, SILO) sample in the VAE latent space, theoretically cheaper, but require VAE encoder/decoder at every step, often making them slower than pixel methods overall.

**Limitations of Prior Work**: (a) Pixel methods compute in high dimensions throughout, with most computation in early stages spent on "drawing global structure," leading to redundancy. (b) Latent methods save on latent space dimensions but pay the price of repeated encoding/decoding, and the VAE itself limits reconstruction quality.

**Key Challenge**: The desire is to "save computation early" and "draw details late"; pixel/latent methods each only address one side. A dimensionality-on-demand diffusion process is needed.

**Goal**: (a) Transfer dynamic resolution diffusion (Subspace Diffusion / UDPM / DVDP / DiMR / Fresco) from pure generation to general image restoration; (b) Enable pixel-space algorithms like DPS/DAPS to maintain measurement consistency under dynamic resolution; (c) Further optimize noise injection, measurement update, and trajectory correction submodules to simultaneously improve quality and speed.

**Key Insight**: The authors leverage the insight from Jing et al. (2022) that "early timesteps are mainly low-frequency and can be done in low-resolution subspaces," which naturally fits ISR tasks that "first recover global structure, then add high-frequency details."

**Core Idea**: First, finetune a pretrained pixel DM for three shared-resolution levels ($64^2 / 128^2 / 256^2$); adapt DPS/DAPS into SubDPS/SubDAPS as baselines; then propose three improvements for SubDAPS (CG for measurement, deterministic switching, predictor-corrector) to form SubDAPS++.

## Method

### Overall Architecture
Inference proceeds backward along time $0 = t_0 < t_1 < \dots < t_N = T$, with each step associated with a dimension $d_i$, where $d = d_0 \geq d_1 \geq \dots \geq d_N$ (the paper uses three levels: $256^2 \to 128^2 \to 64^2$). Each step involves: (1) using $\bm{x}_\theta(\bm{x}_{t_i}, t_i)$ to obtain the unconditional prediction $\hat{\bm{x}}_0$; (2) correcting $\hat{\bm{x}}_0$ with the measurement to obtain $\tilde{\bm{x}}_0$; (3) if sampling from $d_i$ to $d_{i-1}$, project the state back to the higher dimension and inject noise to match the diffusion prior; otherwise, decide whether to continue injecting random noise or switch to deterministic updates based on convergence criteria. SubDAPS++ adds a predictor-corrector pass after the main loop to correct the entire trajectory without further network evaluation.

### Key Designs

1. **SubDPS / SubDAPS: Bringing Measurement Consistency into Dynamic Resolution**:

    - **Function**: Enables classic DPS/DAPS to use observation-based correction in subspaces and naturally handle resolution switching.
    - **Mechanism**: For DPS, when $d_{i-1} = d_i$, rewrite the likelihood gradient as $\nabla_{\bm{x}_{t_i}} \log p_{t_i}(\bm{y} | \bm{x}_{t_i}) \approx -\zeta_{t_i} \nabla_{\bm{x}_{t_i}} \|\bm{y} - \mathcal{A}(\bm{U}_i \bm{x}_\theta(\bm{x}_{t_i}, t_i))\|^2$ (using upsampling matrix $\bm{U}_i$ to project subspace predictions back to the original image domain for measurement). At dimension switches $d_{i-1} \neq d_i$, inspired by DAPS's "early random noise can correct accumulated errors," the authors simply skip explicit correction: $\bm{x}_{t_{i-1}} = \alpha_{t_{i-1}} \dot{\bm{U}}_i \bm{x}_\theta(\bm{x}_{t_i}, t_i) + \sigma_{t_{i-1}} \bm{\epsilon}_i$. For DAPS, first solve the optimization $\hat{\bm{x}}_0^{t_i} = \arg\min_{\bar{\bm{x}}_0} \big( r_{t_i} \|\bar{\bm{x}}_0 - \tilde{\bm{x}}_0^{t_i}\|^2 + \|\bm{y} - \mathcal{A}(\bm{U}_i \bar{\bm{x}}_0)\|^2 \big)$, then use similar stochastic sampling for the next step.
    - **Design Motivation**: The gradient trick in DPS and decoupled trajectory in DAPS were designed for pixel-space; by simply adding an upsampler $\bm{U}_i$ before the measurement operator, the authors make them consistent in both subspace and at switching points—an engineering yet elegant "operator adaptation."

2. **SubDAPS++ Deterministic Switching + CG Measurement**:

    - **Function**: Simultaneously eliminates artifacts and iteration cost of SubDAPS at low timesteps.
    - **Mechanism**: (a) Deterministic switching. Define $h = \min\{i: d_{i-1} \neq d_i\}$ (the index of the last dimension change); when $i < h$ (already stabilized at full resolution) and $\|\bm{x}_\theta(\bm{x}_{t_i}, t_i) - \hat{\bm{x}}_0^{t_i}\|^2 \leq \tau$, use $\bm{x}_{t_{i-1}} = \alpha_{t_{i-1}} \hat{\bm{x}}_0^{t_i} + \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}}(\bm{x}_{t_i} - \alpha_{t_i} \hat{\bm{x}}_0^{t_i})$ for deterministic update; otherwise, continue injecting noise. (b) Conjugate gradient replaces Langevin. SubDAPS uses Langevin for measurement update in Eq. (16), which is slow and only suitable for differentiable operators; SubDAPS++ switches to Fletcher-Reeves CG: each step linearizes $\mathcal{A}(\bm{U}_i(\bar{\bm{x}}_0^{(j)} + \alpha \bm{d}_j))$ via first-order Taylor expansion, yielding closed-form line search $\alpha_j = (\bm{g}_j^\top \bm{d}_j) / (r_{t_i} \bm{d}_j^\top \bm{d}_j + \bm{\omega}_j^\top \bm{\omega}_j)$, with search direction updated as $\bm{d}_{j+1} = \bm{g}_{j+1} + \frac{\bm{g}_{j+1}^\top \bm{g}_{j+1}}{\bm{g}_j^\top \bm{g}_j} \bm{d}_j$.
    - **Design Motivation**: Random noise at low timesteps can destroy the diffusion prior and cause artifacts; the authors use "dimension stabilized + prediction converged" as conditions for deterministic update switching. CG's closed-form line search works for both linear and nonlinear measurements, making it faster and less hyperparameter-sensitive than Langevin.

3. **Predictor-Corrector Trajectory Correction without Extra Network Evaluation**:

    - **Function**: After the main loop, traverse the trajectory again to correct bias introduced by the stochastic main loop.
    - **Mechanism**: Inspired by UniPC's second-order corrector: $\bm{x}_{t_{i-1}}^c = \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}} \dot{\bm{U}}_i \bm{x}_{t_i}^c - \left(\sigma_{t_{i-1}} \frac{\alpha_{t_i}}{\sigma_{t_i}} - \alpha_{t_{i-1}}\right) \hat{\bm{x}}_0^{t_{i-1}} - \sigma_{t_{i-1}} \mathcal{I}_i \frac{\hat{\bm{x}}_0^{t_{i-1}} - \dot{\bm{U}}_i \hat{\bm{x}}_0^{t_i}}{\lambda_{t_{i-1}} - \lambda_{t_i}}$, where $\lambda_t = \log(\alpha_t/\sigma_t)$ is the half log-SNR. This step fully reuses cached $\hat{\bm{x}}_0^{t_i}$ from the main loop, without calling the neural network.
    - **Design Motivation**: Noise is added in the main loop to prevent divergence, but it widens trajectory bias; the UniPC-style corrector analytically pulls the state back to a more "standard" diffusion trajectory, providing almost free performance gains.

### Loss & Training

- **Training**: Finetune a Dhariwal-Nichol pretrained pixel DM, with targets for $\bm{x}_0$, $\tilde{\bm{U}}^\top \bm{x}_0$, and $\hat{\bm{U}}^\top \bm{x}_0$ at three resolutions, enabling a single network to handle $256/128/64$ resolutions; only one finetuning is needed, shared across all downstream tasks.
- **Inference**: The only difference from SubDAPS is that the multi-step ODE solver for estimating $\tilde{\bm{x}}_0$ is reduced to a single network evaluation $\tilde{\bm{x}}_0 = \bm{x}_\theta(\bm{x}_{t_i}, t_i)$; measurement consistency uses $J$ steps of CG; switching threshold $\tau$, upsampling index $h$, noise level $\sigma$, and iteration number $N$ are all hyperparameters.

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
| SubDPS | Naive port of DPS to dynamic resolution, matches DPS performance (weakest), mainly to validate framework feasibility |
| SubDAPS | Achieves results comparable to or slightly better than DAPS, with speedup from subspace |
| SubDAPS + CG replaces Langevin | Faster measurement update, compatible with nonlinear operators |
| SubDAPS + deterministic switching | Reduces low-timestep artifacts, increases PSNR, decreases LPIPS |
| SubDAPS + corrector | Second-order correction without network calls, almost free extra gain |
| All combined = SubDAPS++ | Ranks first or second on most metrics across six tasks |

### Key Findings
- Dynamic resolution is more suitable for restoration tasks than latent approaches: it avoids the overhead of repeated VAE encoding/decoding and lacks the VAE reconstruction bottleneck, so SubDAPS++ is both faster and more accurate than LatentDAPS on nearly all datasets.
- "Low-resolution for structure early, full-resolution for details late" fits ISR/inpainting tasks that inherently require global-to-local processing; motion deblurring, a globally degraded scenario, benefits especially.
- Controlling "stochastic→deterministic" switching with both prediction convergence and dimension stability is more robust than switching by timestep—letting the current trajectory decide whether to use SDE or ODE based on remaining noise.
- CG with first-order Taylor closed-form line search enables measurement updates for any differentiable $\mathcal{A}$ (not just linear), making it more extensible than DDRM/DDNM, which are designed for linear operators.

## Highlights & Insights
- The cross-domain transfer of dynamic resolution diffusion from generation to restoration is the most insightful contribution—while the idea is the same, its significance is entirely different for restoration, which naturally fits the "coarse-to-fine" structure.
- The three patches (CG, switching, corrector) each originate from different neighboring fields (numerical optimization/samplers, deterministic ODE switching, UniPC predictor-corrector), but the authors combine them seamlessly, exemplifying engineering integration.
- Sharing a single finetuned multi-resolution DM for all tasks and resolutions eliminates the need to train separate models for each resolution, making deployment industry-friendly.
- Using CG with first-order Taylor closed-form step size for measurement is elegant—retaining the fast convergence of conjugate gradient while making the code applicable to any differentiable operator, enhancing transferability.

## Limitations & Future Work
- Dynamic resolution is limited to three levels; the paper does not discuss whether more layers and more complex upsampling matrices are needed for higher resolutions (above $1024^2$).
- The switching threshold $\tau$ is a fixed hyperparameter; the authors acknowledge that it needs to be tuned for some tasks. Adaptive $\tau$ should be a future direction.
- The corrector is applied post hoc in batch, not for error correction during generation; if severe hallucination occurs, it may be irrecoverable in later stages.
- Comparisons with the latest LDM restoration methods (e.g., SD-based) are limited; especially for natural image super-resolution, SD-based methods often have perceptual metric advantages.

## Related Work & Insights
- **vs DPS / DAPS**: This work is essentially a "dynamic resolution rewrite of DPS/DAPS"—retaining their measurement correction approach but moving each step to the appropriate dimension, accelerating without loss of quality.
- **vs PSLD / ReSample / LatentDAPS**: Latent approaches rely on VAE for dimensionality reduction, paying the price of repeated encoder/decoder calls; dynamic resolution does not require VAE, thus balancing speed and quality.
- **vs Subspace DM / UDPM / DiMR / Fresco**: Those are "generation" versions of dynamic resolution; this is the first systematic "restoration" version, solving the challenge of measurement consistency at dimension switches.
- **vs UniPC**: UniPC's corrector is for deterministic ODEs; here, it is adapted to the end of the stochastic main loop as a "post hoc free correction," a clever paradigm transfer.
- **Insights**: The dynamic resolution idea can be used in reverse—for example, in video generation by varying spatial resolution over time, or in 3D reconstruction by varying voxel resolution by level of detail. The essence is always a "coarse-to-fine" inductive bias.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of dynamic resolution DM to general image restoration, with joint adaptation of DPS/DAPS.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four linear + two nonlinear tasks × FFHQ/ImageNet datasets × three baselines (pixel/latent/dynamic) for comprehensive comparison.
- Writing Quality: ⭐⭐⭐⭐ Algorithm pseudocode, formula derivations, and switching conditions are clearly explained; figures are somewhat limited.
- Value: ⭐⭐⭐⭐ Fast and accurate inference without relying on VAE, making it a truly practical acceleration solution for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[CVPR 2026\] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](../../CVPR2026/image_restoration/blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](../../CVPR2026/image_restoration/disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[NeurIPS 2025\] DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](../../NeurIPS2025/image_restoration/dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)
- [\[ICLR 2026\] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](../../ICLR2026/image_restoration/horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)

</div>

<!-- RELATED:END -->
