---
title: >-
  [Paper Note] ReNoise: Real Image Inversion Through Iterative Noising
description: >-
  [ECCV 2024][Image Generation] Proposed the ReNoise iterative renoising method to improve the image inversion quality of diffusion models. By applying the UNet multiple times at each inversion timestep and averaging the predictions, it improves trajectory estimation accuracy, which is particularly effective for few-step diffusion models (such as SDXL Turbo and LCM).
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: aa88bad2683cf332
---

# ReNoise: Real Image Inversion Through Iterative Noising

**Conference**: ECCV 2024  
**arXiv**: [2403.14602](https://arxiv.org/abs/2403.14602)  
**Area**: Image Generation

## TL;DR

Proposed the ReNoise iterative renoising method to improve the image inversion quality of diffusion models. By applying the UNet multiple times at each inversion timestep and averaging the predictions, it improves trajectory estimation accuracy, which is particularly effective for few-step diffusion models (such as SDXL Turbo and LCM).

## Background & Motivation

- Text-guided diffusion model editing requires first inverting real images into the diffusion model domain.
- The core challenge of the inversion problem: the denoising steps are irreversible—the model is trained to map $z_t \to z_{t-1}$, but inversion requires $z_{t-1} \to z_t$.
- DDIM Inversion approximates the inversion direction through a linearity assumption, but this assumption leads to significant errors when the step size is large.
- **New Challenge**: Recent few-step diffusion models (e.g., SDXL Turbo requiring only 1-4 steps) involve extremely large step sizes, causing conventional inversion methods to fail completely.
- Existing improvement methods (such as Null-Text Inversion) require time-consuming optimization (3 minutes vs. 13 seconds), making them unsuitable for interactive editing.

## Method

### Overall Architecture

ReNoise is a meta-algorithm that can be combined with any sampler (such as DDIM, Ancestral-Euler, or LCM sampler):

1. At each inversion timestep $t$, $z_t$ is estimated based on $z_{t-1}$.
2. An iterative renoising process generates a sequence of estimates for $z_t$, denoted as $\{z_t^{(k)}\}_{k=1}^{\mathcal{K}+1}$.
3. A weighted average of the final few estimates is computed to obtain a more precise $z_t$.
4. Optionally, editability enhancement and noise correction can be incorporated.

### Key Designs

**1. Iterative Renoising**

Original DDIM Inversion approximates $\epsilon_\theta(z_t, t)$ using $\epsilon_\theta(z_{t-1}, t)$, but $z_{t-1}$ and $z_t$ can be far apart.

ReNoise improvements:
- The first approximation yields $z_t^{(1)}$ (equivalent to DDIM Inversion).
- The $k$-th iteration: use $z_t^{(k)}$, which is closer to the true $z_t$, as the UNet input to obtain a more accurate direction estimate.
- $z_t^{(k+1)} = \text{InverseStep}(z_{t-1}, \epsilon_\theta(z_t^{(k)}, t))$

**Key Intuition**: Each iteration starts from $z_{t-1}$ but utilizes a more accurate direction, progressively approaching the true $z_t$.

**2. Prediction Averaging**

Since fixed-point iteration may converge non-monotonically, a weighted average of the final few estimates is computed:

$$z_t^{(\text{avg})} = \sum_{k=1}^{\mathcal{K}} w_k \cdot z_t^{(k)}$$

The averaging strategy effectively suppresses errors caused by non-monotonic oscillations.

**3. Editability Enhancement**

The inverted noise prediction may deviate from Gaussian white noise statistics, which impairs editability. This is improved using two regularization losses:

- $\mathcal{L}_{\text{patch-KL}}$: encourages the KL divergence between the predicted noise and random noise to be minimized at the patch level.
- $\mathcal{L}_{\text{pair}}$: penalizes the correlation between pixel pairs.

**4. Noise Correction for Non-Deterministic Samplers**

For samplers with $\rho_t > 0$ (such as DDPM and Ancestral-Euler), the gap between the inversion and denoising trajectories is bridged by optimizing the external noise $\epsilon_t$, while preserving the noise distribution characteristics.

### Loss & Training

The generalized formulation of sampler inversion:

$$z_t = \frac{z_{t-1} - \psi_t \epsilon_\theta(z_t, t, c) - \rho_t \epsilon_t}{\phi_t}$$

where $\phi_t$, $\psi_t$, and $\rho_t$ are sampler-specific parameters. The editability loss is formulated as $\mathcal{L}_{\text{edit}} = \mathcal{L}_{\text{patch-KL}} + \mathcal{L}_{\text{pair}}$.

### Convergence Analysis

The convergence condition of the iteration is analyzed using Taylor expansion:

$$\|\Delta^{(k+1)}\| \leq \frac{\psi_t}{\phi_t} \cdot \|\frac{\partial \epsilon_\theta}{\partial z}\|_{z_t^{(k-1)}} \cdot \|\Delta^{(k)}\| + O(\|\Delta^{(k)}\|^2)$$

Experimental validation confirms that the scaled Jacobian norm is consistently $<1$, verifying practical convergence of the algorithm. The distance between successive estimates decreases exponentially.

## Key Experimental Results

### Main Results

Comparison of image reconstruction under a fixed budget of 100 UNet operations (SDXL):

| Inversion Steps | Inference Steps | ReNoise Steps | L2↓ | PSNR↑ | LPIPS↓ |
|---------|---------|------------|-----|-------|--------|
| 50 | 50 | 0 | 0.00364 | 26.023 | 0.06273 |
| 75 | 25 | 0 | 0.00382 | 25.466 | 0.06605 |
| 80 | 20 | 0 | 0.00408 | 25.045 | 0.07099 |
| 90 | 10 | 0 | 0.01023 | 20.249 | 0.10305 |
| 25 | 25 | 2 | 0.00182 | **29.569** | 0.03637 |
| 20 | 20 | 3 | **0.00167** | **29.884** | **0.03633** |
| 10 | 10 | 8 | 0.00230 | 28.156 | 0.04678 |

### Ablation Study

Incremental effects of various components on SDXL Turbo:

| Configuration | L2↓ | PSNR↑ | LPIPS↓ |
|------|-----|-------|--------|
| Euler Inversion | 0.0700 | 11.784 | 0.20337 |
| + 1 ReNoise | 0.0552 | 12.796 | 0.20254 |
| + 4 ReNoise | 0.0249 | 16.521 | 0.14821 |
| + 9 ReNoise | 0.0126 | 19.702 | 0.10850 |
| + Averaging | **0.0087** | 21.491 | **0.08832** |
| + Edit Losses | 0.0276 | 18.432 | 0.12616 |
| + Noise Correction (Full) | 0.0196 | **22.077** | 0.08469 |

### Key Findings

1. **ReNoise outperforms increasing inversion steps**: Given the same budget of UNet operations, 20 inversion steps with 3 ReNoise steps (PSNR=29.884) significantly outperforms 80 inversion steps with 0 ReNoise steps (PSNR=25.045).
2. On SDXL Turbo (a 4-step model), ReNoise improves the PSNR from 11.784 to 22.077 (+87%).
3. The averaging strategy is a critical component, boosting the PSNR from 19.702 to 21.491.
4. Editability-enhancing losses slightly degrade reconstruction quality (PSNR 21.491 → 18.432) but ensure editability.
5. Noise correction effectively compensates for the reconstruction degradation caused by editability losses.
6. Inversion takes only 13 seconds, which is vastly faster than the 3 minutes required by Null-Text Inversion.
7. This method is compatible with a wide range of models (SD, SDXL, SDXL Turbo, LCM) and samplers.

## Highlights & Insights

- **Compute-efficiency-first design philosophy**: Rather than increasing the total number of operations, the operations are redistributed (reducing timesteps while increasing iterations per step) to achieve highly efficient inversion.
- **High versatility**: Compatible with both deterministic and non-deterministic samplers, as well as standard and few-step models, serving as a true meta-algorithm.
- **Perfect combination of theory and practice**: From ODE backward Euler solving to fixed-point iteration, theoretical analysis aligns consistently with experimental validation.
- **Practical inversion solution for few-step models**: For the first time, inversion and editing on SDXL Turbo (4-step) are made feasible.

## Limitations & Future Work

- There is an inherent trade-off between reconstruction and editability; latent codes that yield perfect reconstruction are often less editable.
- Although convergence conditions are theoretically analyzed, the actual convergence rate in practice depends on the model and the data.
- Sensitivity to prompts: different prompts can result in varying inversion quality.
- Hyperparameters for editability enhancement (such as which weights $w_k > 0$ in the iterations at each step) require tuning.
- Although editing results on few-step models far surpass previous methods, they still fall short of the editing quality achieved by standard 50-step models.

## Rating

- Novelty: ⭐⭐⭐⭐ — The iterative renoising idea is intuitive and elegant.
- Practicality: ⭐⭐⭐⭐⭐ — Universal, fast, and adaptive to few-step models.
- Performance: ⭐⭐⭐⭐ — Substantial improvement in reconstruction quality.
- Overall Rating: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)

</div>

<!-- RELATED:END -->
