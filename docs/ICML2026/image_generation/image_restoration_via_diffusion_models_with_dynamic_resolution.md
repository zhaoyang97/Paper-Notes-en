---
title: >-
  [Paper Note] Image Restoration via Diffusion Models with Dynamic Resolution
description: >-
  [ICML 2026][Image Generation][DAPS] SubDAPS / SubDAPS++ integrates pixel-space diffusion restoration methods (such as DPS and DAPS) into a "dynamic resolution diffusion" framework. The process initiates sampling in $64^2 / 128^2$ subspaces and returns to the $256^2$ full resolution only in later stages. By replacing Langevin dynamics with Conjugate Gradi
tags:
  - ICML 2026
  - Image Generation
  - DAPS
  - predictor-corrector
  - ISR
date: 2026-05-08
content_hash: d57e35e60cdda730
---
# Image Restoration via Diffusion Models with Dynamic Resolution

**Conference**: ICML 2026  
**arXiv**: [2605.14267](https://arxiv.org/abs/2605.14267)  
**Code**: https://github.com/StarNextDay/SubDAPS (Available)  
**Area**: Diffusion Models / Image Restoration / Accelerated Inference  
**Keywords**: Dynamic resolution diffusion, DAPS, Conjugate Gradient, predictor-corrector, ISR

## TL;DR
SubDAPS / SubDAPS++ integrates pixel-space diffusion restoration methods (such as DPS and DAPS) into a "dynamic resolution diffusion" framework. The process initiates sampling in $64^2 / 128^2$ subspaces and returns to the $256^2$ full resolution only in later stages. By replacing Langevin dynamics with Conjugate Gradient (CG), implementing threshold-based switching between stochastic and deterministic sampling, and appending a corrector step that requires no extra network evaluations, it outperforms most pixel and latent diffusion methods across four linear and two nonlinear restoration tasks with faster inference speeds.

## Background & Motivation

**Background**: Diffusion models have demonstrated strong performance in image restoration. Pixel-space methods (e.g., DPS, DDRM, DDNM, DiffPIR, DAPS, AdaPS) perform repeated sampling directly on $256^2 \times 3$, which ensures high inversion quality but remains slow. Latent-space methods (e.g., PSLD, ReSample, LatentDAPS, SILO) perform sampling in the VAE latent space, which is theoretically cheaper; however, the requirement for VAE encoding/decoding at every step often makes them slower than pixel-space methods.

**Limitations of Prior Work**: (a) Pixel-space methods compute everything at high dimensions, where early stages spend excessive computation on "sketching" global structures, leading to redundancy. (b) Latent methods save on latent space dimensions but incur the cost of repeated encoding/decoding, while the VAE itself limits the maximum attainable reconstruction quality.

**Key Challenge**: There is a desire to "save costs early" while "refined details late." Neither pixel nor latent extremes are optimal; a dimensionality-on-demand diffusion process is required.

**Goal**: (a) Migrate dynamic resolution diffusion (e.g., Subspace Diffusion, UDPM, DVDP, DiMR, Fresco) from pure generation to general image restoration. (b) Adapt pixel-space algorithms like DPS and DAPS to maintain measurement consistency within a dynamic resolution framework. (c) Further optimize noise injection, measurement updates, and trajectory correction to enhance both quality and speed.

**Key Insight**: The authors leverage the insight from Jing et al. (2022) that early timesteps mainly recover low-frequency information, which can be processed in low-resolution subspaces. This naturally aligns with ISR tasks that restore global structures before filling in high-frequency details.

**Core Idea**: A pretrained pixel DM is finetuned with shared weights across three resolutions ($64^2 / 128^2 / 256^2$). SubDPS/SubDAPS are constructed as baselines. Subsequently, three enhancements—CG for measurement updates, deterministic switching, and a predictor-corrector step—are proposed to synthesize SubDAPS++.

## Method

### Overall Architecture
Inference proceeds backward from $0 = t_0 < t_1 < \dots < t_N = T$, where each time step is associated with a dimension $d_i$ such that $d = d_0 \geq d_1 \geq \dots \geq d_N$ (using $256^2 \to 128^2 \to 64^2$ in this work). At each step, three actions are performed: (1) Obtain an unconditional prediction $\hat{\bm{x}}_0$ using $\bm{x}_\theta(\bm{x}_{t_i}, t_i)$. (2) Refine $\hat{\bm{x}}_0$ into a measurement-consistent $\tilde{\bm{x}}_0$. (3) If upsampling from $d_i$ to $d_{i-1}$, project the state back and inject noise to match the diffusion prior; otherwise, determine whether to continue stochastic noise injection or switch to deterministic updates based on a convergence criterion. SubDAPS++ concludes with a predictor-corrector pass to refine the entire trajectory without additional network evaluations.

```mermaid
mermaid
flowchart TD
    A["Multi-res DM: 64²→128²→256² shared weights, one-time finetune"] --> B["Reverse step t_i: Network predicts unconditional x̂₀"]
    B --> C["SubDPS / SubDAPS: Upsampler U_i projects back to original resolution for measurement consistency → x̃₀"]
    C --> D["Deterministic switching + Conjugate Gradient measurement<br/>Inject noise at switch; use ODE if dims stable & prediction converges, else SDE"]
    D -->|"Remaining steps exist"| B
    D ==>|"Main loop ends"| E["Predictor-corrector trajectory refinement<br/>Reuses cached x̂₀, zero extra network evaluations"]
    E --> F["Restoration Result"]
```

### Key Designs

**1. SubDPS / SubDAPS: Incorporating an upsampler for measurement consistency in subspaces**

The gradient tricks in DPS and decoupled trajectories in DAPS are designed for pixel-space. When sampling in $64^2/128^2$ subspaces, the observation $\bm{y}$ remains in the original image domain, causing a mismatch. The solution is to prepend an upsampling matrix $\bm{U}_i$ before the measurement operator to project subspace predictions back to the original resolution. For SubDPS, at steps where dimensions remain constant ($d_{i-1} = d_i$), the likelihood gradient is redefined as $\nabla_{\bm{x}_{t_i}} \log p_{t_i}(\bm{y} | \bm{x}_{t_i}) \approx -\zeta_{t_i} \nabla_{\bm{x}_{t_i}} \|\bm{y} - \mathcal{A}(\bm{U}_i \bm{x}_\theta(\bm{x}_{t_i}, t_i))\|^2$. For SubDAPS, the optimization problem $\hat{\bm{x}}_0^{t_i} = \arg\min_{\bar{\bm{x}}_0} \big( r_{t_i} \|\bar{\bm{x}}_0 - \tilde{\bm{x}}_0^{t_i}\|^2 + \|\bm{y} - \mathcal{A}(\bm{U}_i \bar{\bm{x}}_0)\|^2 \big)$ is solved before sampling.

At resolution switches ($d_{i-1} \neq d_i$), upsampling introduces errors. Leveraging the observation from DAPS that early stochastic noise injection can correct accumulated errors, the authors omit specific corrections at switch steps and simply use $\bm{x}_{t_{i-1}} = \alpha_{t_{i-1}} \dot{\bm{U}}_i \bm{x}_\theta(\bm{x}_{t_i}, t_i) + \sigma_{t_{i-1}} \bm{\epsilon}_i$, allowing subsequent noise to smooth out switching artifacts.

**2. SubDAPS++ Deterministic Switching + Conjugate Gradient: Reducing artifacts and iteration costs**

SubDAPS uses stochastic noise throughout, but at low timesteps where the resolution is full, excess noise can disrupt the diffusion prior and leave artifacts. A switching criterion is introduced: let $h = \min\{i: d_{i-1} \neq d_i\}$ denote the index of the last dimension change. If $i < h$ (dimensions stabilized at $256^2$) and the prediction converges $\|\bm{x}_\theta(\bm{x}_{t_i}, t_i) - \hat{\bm{x}}_0^{t_i}\|^2 \leq \tau$, the process switches to a deterministic update $\bm{x}_{t_{i-1}} = \alpha_{t_{i-1}} \hat{\bm{x}}_0^{t_i} + \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}}(\bm{x}_{t_i} - \alpha_{t_i} \hat{\bm{x}}_0^{t_i})$. This allows the trajectory to decide between SDE and ODE based on convergence rather than a fixed timestep.

Additionally, Langevin dynamics in the measurement update are replaced with Fletcher-Reeves Conjugate Gradient. CG linearizes $\mathcal{A}(\bm{U}_i(\bar{\bm{x}}_0^{(j)} + \alpha \bm{d}_j))$ via first-order Taylor expansion to derive a closed-form step size $\alpha_j = (\bm{g}_j^\top \bm{d}_j) / (r_{t_i} \bm{d}_j^\top \bm{d}_j + \bm{\omega}_j^\top \bm{\omega}_j)$, with the search direction updated via $\bm{d}_{j+1} = \bm{g}_{j+1} + \frac{\bm{g}_{j+1}^\top \bm{g}_{j+1}}{\bm{g}_j^\top \bm{g}_j} \bm{d}_j$. This closed-form line search is significantly faster than Langevin and handles both linear and nonlinear measurements.

**3. Predictor-Corrector Refinement: Correcting trajectory bias for free**

Stochastic noise in the main loop prevents divergence but broadens trajectory deviation. SubDAPS++ applies a second-order corrector adapted from UniPC after the main loop:

$$\bm{x}_{t_{i-1}}^c = \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}} \dot{\bm{U}}_i \bm{x}_{t_i}^c - \left(\sigma_{t_{i-1}} \frac{\alpha_{t_i}}{\sigma_{t_i}} - \alpha_{t_{i-1}}\right) \hat{\bm{x}}_0^{t_{i-1}} - \sigma_{t_{i-1}} \mathcal{I}_i \frac{\hat{\bm{x}}_0^{t_{i-1}} - \dot{\bm{U}}_i \hat{\bm{x}}_0^{t_i}}{\lambda_{t_{i-1}} - \lambda_{t_i}}$$

where $\lambda_t = \log(\alpha_t/\sigma_t)$ is the half log-SNR. Crucially, this step reuses the cached $\hat{\bm{x}}_0^{t_i}$ from the main loop, requiring zero additional network calls to refine the state.

### Loss & Training
- **Training**: Finetuned on the Dhariwal-Nichol pretrained pixel DM. The objective jointly denoises $\bm{x}_0$, $\tilde{\bm{U}}^\top \bm{x}_0$, and $\hat{\bm{U}}^\top \bm{x}_0$ across three resolutions, enabling a single network to handle $256/128/64$ dimensions. This finetuning is performed once and shared across all tasks.
- **Inference**: Unlike DAPS, multi-step ODE solvers for estimating $\tilde{\bm{x}}_0$ are reduced to a single network evaluation $\tilde{\bm{x}}_0 = \bm{x}_\theta(\bm{x}_{t_i}, t_i)$. Measurement consistency is enforced with $J$ steps of CG.

## Key Experimental Results

### Main Results

| Task (256² FFHQ) | Type | DiffPIR | MGPS | DAPS | AdaPS | LatentDAPS | **SubDAPS++ (Ours)** |
|---|---|---|---|---|---|---|---|
| Inpainting 70% rand, PSNR ↑ | dyn/px/lat | 32.16 | 31.41 | 30.68 | **32.34** | 31.17 | 32.21 |
| Inpainting 70% rand, LPIPS ↓ | | 0.052 | 0.050 | 0.073 | 0.057 | 0.090 | **0.056** |
| SR ×4, PSNR ↑ | | 27.64 | 27.58 | 28.88 | 27.34 | 28.56 | **29.34** |
| SR ×4, LPIPS ↓ | | 0.116 | 0.110 | 0.162 | **0.090** | 0.174 | 0.157 |
| Motion Deblur, PSNR ↑ | | 26.95 | 26.82 | 28.27 | 27.06 | 27.58 | **28.28** |

| Task (256² ImageNet) | Type | DPS | DAPS | LatentDAPS | **SubDAPS++ (Ours)** |
|---|---|---|---|---|---|
| Inpainting 70%, PSNR ↑ | | 25.33 | 27.63 | 27.33 | **28.61** |
| Inpainting 70%, FID ↓ | | 141.99 | 56.73 | 85.24 | **49.15** |
| SR ×4, PSNR ↑ | | 21.68 | 25.54 | 25.43 | **25.79** |

### Ablation Study

| Configuration | Description |
|---|---|
| SubDPS | Naive port of DPS to dynamic resolution; performance similar to DPS (weakest baseline), essentially validates the framework. |
| SubDAPS | Achieves results comparable to or better than DAPS while gaining speed from subspace computation. |
| SubDAPS + CG | Measurement updates are faster and compatible with nonlinear operators. |
| SubDAPS + deterministic switch | Reduces artifacts at low timesteps; PSNR increases, LPIPS decreases. |
| SubDAPS + corrector | Zero-cost second-order refinement providing a significant final gain. |

### Key Findings
- Dynamic resolution is more suited for restoration than latent-space routes: it avoids VAE overhead and reconstruction bottlenecks, making SubDAPS++ faster and more accurate than LatentDAPS.
- Generating structure at low resolution and details at full resolution is highly effective for global-to-local tasks like ISR and inpainting.
- Using convergence to control the stochastic-to-deterministic switch is more robust than fixed timestep schedules.
- The CG + Taylor closed-form line search provides better extensibility than DDRM/DDNM, as it applies to any differentiable operator.

## Highlights & Insights
- The migration of dynamic resolution from generation to restoration is insightful; restoration naturally benefits from the "coarse-to-fine" inductive bias.
- The combination of CG, deterministic switching, and UniPC correctors is a prime example of engineering synthesis where disparate techniques work in harmony.
- A single finetuned multi-resolution DM serves all tasks, making it deployment-friendly.
- The analytical closed-form step size for CG is elegant—it maintains the convergence speed of conjugate gradients while remaining universal for any differentiable measurement operator $\mathcal{A}$.

## Limitations & Future Work
- Only three resolution stages are used; higher resolutions ($1024^2$+) might require more layers.
- The threshold $\tau$ is a fixed hyperparameter and may require manual tuning for specific tasks.
- The corrector is applied post-hoc; errors leading to severe hallucinations cannot be recovered mid-generation.
- Comparisons with the latest SD-based LDM restoration methods are limited, particularly regarding perceptual metrics in natural image super-resolution.

## Related Work & Insights
- **vs DPS / DAPS**: Effectively a "dynamic resolution rewrite" of these methods—preserving their logic while executing each step at the optimal dimension.
- **vs PSLD / LatentDAPS**: Latent routes pay a heavy price in VAE calls. Dynamic resolution maintains a better balance of speed and reconstruction quality.
- **vs UniPC**: While UniPC correctors are designed for ODEs, their application here as a post-hoc stochastic correction is a clever cross-paradigm adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VibeToken: Scaling 1D Image Tokenizers and Autoregressive Models for Dynamic Resolution Generations](../../CVPR2026/image_generation/vibetoken_scaling_1d_image_tokenizers_and_autoregressive_models_for_dynamic_reso.md)
- [\[ICLR 2026\] Eliminating VAE for Fast and High-Resolution Generative Detail Restoration](../../ICLR2026/image_generation/eliminating_vae_for_fast_and_high-resolution_generative_detail_restoration.md)
- [\[ICML 2026\] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)
- [\[CVPR 2026\] Training-free, Perceptually Consistent Low-Resolution Previews with High-Resolution Image for Efficient Workflows of Diffusion Models](../../CVPR2026/image_generation/training-free_perceptually_consistent_low-resolution_previews.md)
- [\[ICML 2026\] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)

</div>

<!-- RELATED:END -->
