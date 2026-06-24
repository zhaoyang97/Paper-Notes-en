---
title: >-
  [Paper Note] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems
description: >-
  [ICCV 2025][Image Generation][Flow Matching] FlowDPS derives a Tweedie formula for flow models to decompose the Flow ODE into a clean image estimation component and a noise estimation component. Likelihood gradients are injected into the clean image component while stochastic noise is introduced into the noise component, enabling principled posterior sampling for inverse problems under the flow matching framework. FlowDPS surpasses all prior methods on four linear inverse pro…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Flow Matching"
  - "Posterior Sampling"
  - "Inverse Problems"
  - "Tweedie Formula"
  - "Stable Diffusion 3.0"
date: 2026-05-08
content_hash: af90392b5d8f77ab
---

# FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems

**Conference**: ICCV 2025
**arXiv**: [2503.08136](https://arxiv.org/abs/2503.08136)  
**Code**: [GitHub](https://github.com/FlowDPS-Inverse/FlowDPS)  
**Area**: Diffusion Models / Inverse Problem Solving
**Keywords**: Flow Matching, Posterior Sampling, Inverse Problems, Tweedie Formula, Stable Diffusion 3.0

## TL;DR
FlowDPS derives a Tweedie formula for flow models to decompose the Flow ODE into a clean image estimation component and a noise estimation component. Likelihood gradients are injected into the clean image component while stochastic noise is introduced into the noise component, enabling principled posterior sampling for inverse problems under the flow matching framework. FlowDPS surpasses all prior methods on four linear inverse problems using SD3.0.

## Background & Motivation

Inverse problems (e.g., super-resolution, deblurring) aim to recover an original signal from degraded observations $\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n}$. The problem is inherently ill-posed and requires prior knowledge to constrain the solution space.

**Success of diffusion-based approaches**: Methods such as DPS and DDRM leverage image priors learned by diffusion models to guide sampling trajectories for posterior sampling. Within the LDM framework, variants including PSLD, ReSample, and DAPS have also been proposed.

**Rise of flow models and associated challenges**: Flow Matching has become a dominant paradigm in generative modeling (e.g., SD3.0 is flow-based), yet rigorous theoretical analysis and effective methods for inverse problem solving under this framework remain lacking. Existing attempts such as FlowChef and PnP-Flow are heuristic and lack theoretical connections to posterior sampling.

The root cause lies in the structural difference between the ODE formulation of flow models and the SDE formulation of diffusion models, which prevents direct transfer of existing diffusion-based inverse problem methods to the flow framework.

The core idea of this paper is: **derive a flow-model analogue of the Tweedie formula, decompose the Flow ODE into denoising and noise-adding components, and thereby enable posterior sampling to arise naturally within the flow framework**.

## Method

### Overall Architecture

During the reverse sampling process of a flow model, FlowDPS:
1. Applies the Tweedie formula to estimate the clean image $\hat{\mathbf{x}}_{0|t}$ and the noise $\hat{\mathbf{x}}_{1|t}$ from the velocity field.
2. Injects a data consistency gradient into the clean image estimate → $\tilde{\mathbf{x}}_{0|t}$.
3. Mixes stochastic noise into the noise estimate → $\tilde{\mathbf{x}}_{1|t}$.
4. Combines the two components via learned coefficients to obtain the next sampling point.

### Key Designs

1. **Flow-Model Tweedie Formula**:

    - Function: Derives conditional expectation estimates of the clean image and noise from the learned velocity field $v_t(\mathbf{x}_t)$.
    - Mechanism: For affine conditional flows $\mathbf{x}_t = a_t \mathbf{x}_0 + b_t \mathbf{x}_1$, the marginal velocity field takes the form:
    $v_t(\mathbf{x}) = \dot{a}_t \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t] + \dot{b}_t \mathbb{E}[\mathbf{x}_1|\mathbf{x}_t]$
      yielding the Tweedie formula:
    $\hat{\mathbf{x}}_{0|t} = \left[a_t - \dot{a}_t \frac{b_t}{\dot{b}_t}\right]^{-1}\left(\mathbf{x}_t - \frac{b_t}{\dot{b}_t} v_t(\mathbf{x}_t)\right)$
    - Design Motivation: This decomposition reveals the intrinsic structural similarity between the Flow ODE and diffusion models at a geometric level, establishing a theoretical foundation for posterior sampling.

2. **Posterior Velocity Field Derivation**:

    - Function: Incorporates observational constraints into the Flow ODE sampling process.
    - Mechanism: Applying Bayes' rule, the posterior velocity field is:
    $v_t(\mathbf{x}_t|\mathbf{y}) = v_t(\mathbf{x}_t) - \zeta_t \nabla_{\mathbf{x}_t} \log p_t(\mathbf{y}|\mathbf{x}_t)$
      Combined with a manifold projection assumption to simplify Jacobian computation, the final sampling update is:
    $\mathbf{x}_{t+dt} = C_1(t)\tilde{\mathbf{x}}_{0|t} + C_2(t)\tilde{\mathbf{x}}_{1|t}$
      where $\tilde{\mathbf{x}}_{0|t} = \hat{\mathbf{x}}_{0|t} - \beta_t \nabla_{\hat{\mathbf{x}}_{0|t}} \log p(\mathbf{y}|\hat{\mathbf{x}}_{0|t})$ enforces data consistency.
    - Design Motivation: Injecting the likelihood gradient precisely into the clean image component (rather than into $\mathbf{x}_t$ directly) yields more accurate guidance with an adaptive step size $\beta_t$.

3. **Stochastic Noise Injection and Latent FlowDPS**:

    - Function: Blends deterministic estimates with stochastic noise in the noise component, and extends the framework to latent-space flow models.
    - Mechanism:
    $\tilde{\mathbf{x}}_{1|t} = \sqrt{1-\eta_t}\hat{\mathbf{x}}_{1|t} + \sqrt{\eta_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$
      In the latent space, multi-step conjugate gradient optimization is applied to $\hat{\mathbf{z}}_{0|t}(\mathbf{y})$, with trajectory interpolation to maintain stability.
    - Design Motivation: Stochastic noise injection generalizes the DDIM→DDPM transition, enhancing diversity and robustness; latent-space operations enable high-resolution inference.

### Loss & Training

FlowDPS is a **zero-shot method** requiring no additional training. It directly utilizes a pretrained flow model (SD3.0) as the prior. Key hyperparameters include:
- Stochastic noise ratio $\eta_t$, controlling the determinism/stochasticity balance.
- Step size $\beta_t = \frac{\zeta_t}{a_t}\frac{dt}{C_1(t)}$, which is adaptive (large early, approaching zero late).
- Interpolation parameter $\gamma$, controlling the strength of data consistency updates.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | FlowDPS | FlowChef | ReSample | Gain |
|--------|------|------|----------|------|------|
| AFHQ 768² SR×12 Avg | FID↓ | **16.85** | 21.14 | 41.17 | −4.29 vs FlowChef |
| AFHQ 768² SR×12 Avg | LPIPS↓ | **0.198** | 0.249 | 0.300 | −0.051 |
| AFHQ 768² SR×12 Bic | FID↓ | **15.71** | 21.31 | 39.94 | −5.60 |
| AFHQ 768² Deblur Gauss | FID↓ | **23.00** | 36.46 | 44.22 | −13.46 |
| FFHQ 768² SR×12 Avg | FID↓ | **33.78** | 41.50 | 102.7 | −7.72 |
| FFHQ 768² SR×12 Bic | FID↓ | **33.75** | 39.75 | 102.4 | −5.99 |
| FFHQ 768² Deblur Motion | FID↓ | **38.14** | 104.7 | 95.16 | −66.56 |

### Ablation Study

| Configuration | PSNR | FID | Notes |
|------|---------|------|------|
| FlowDPS (ODE, no stochastic noise) | Baseline | Baseline | Purely deterministic posterior sampling |
| FlowDPS (SDE, with stochastic noise) | Higher | **Lower** | Stochastic noise improves FID |
| FlowDPS w/ multi-step CG | **Highest** | **Lowest** | Multi-step gradients improve data consistency |
| FlowDPS w/ interpolation | Best balance | Best | Trajectory interpolation prevents deviation |
| FlowChef (direct guidance on $\mathbf{x}_t$) | Lower | Higher | Lacks decomposition and adaptive step size |

### Key Findings

- FlowDPS achieves substantial improvements across all four linear inverse problems (SR-Avgpool, SR-Bicubic, Gaussian deblurring, and motion deblurring).
- Compared to FlowChef, decomposed gradient injection (modifying only $\hat{\mathbf{x}}_{0|t}$) substantially outperforms direct modification of $\mathbf{x}_t$.
- The adaptive behavior of $\beta_t$ — emphasizing data consistency early and reverting to the generative prior late — is critical for high-quality reconstruction.
- The substantial improvement from PSLD (FID > 140) to FlowDPS (FID < 40) demonstrates that the flow matching framework is markedly more effective than conventional LDM-based inverse problem methods.

## Highlights & Insights

1. **Solid theoretical contribution**: The flow-model Tweedie formula is derived rigorously, revealing deep geometric connections between the Flow ODE and diffusion models.
2. **Zero-shot plug-and-play**: No training is required; the method directly leverages SD3.0, offering high practical utility.
3. **Elegance of the decomposition**: Posterior sampling is cleanly decomposed into "clean estimation + data consistency" and "noise estimation + stochastic injection," yielding a concise and effective framework.
4. **Adaptive step size $\beta_t$** emerges naturally from the theoretical derivation, eliminating the need for manual tuning.

## Limitations & Future Work

- Validation is currently limited to linear inverse problems ($\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n}$); extension to nonlinear inverse problems remains to be explored.
- Latent-space operations require gradient computation through the decoder, increasing GPU memory overhead.
- The interpolation coefficient $\gamma$ and the number of CG steps still require task-specific tuning.
- Multi-step CG optimization increases inference time.

## Related Work & Insights

- This paper establishes a theoretical bridge between flow models and diffusion models for inverse problem solving, paving the way for a unified framework in future work.
- The flow-model Tweedie formula can be extended to other flow-based conditional generation tasks, such as image editing and inpainting.
- The decomposed posterior sampling paradigm can be adapted to broader inverse problem settings including video and 3D reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The flow-model Tweedie formula and ODE decomposition represent significant theoretical contributions to the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons across three datasets and four tasks, though experiments on nonlinear inverse problems are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous; the narrative from background to method to experiments is logically coherent, with intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Provides a principled framework for solving inverse problems with flow models, with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling](ld-rps_zero-shot_unified_image_restoration_via_latent_diffusion_recurrent_poster.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](../../ICML2026/image_generation/saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICLR 2026\] Efficient Approximate Posterior Sampling with Annealed Langevin Monte Carlo](../../ICLR2026/image_generation/efficient_approximate_posterior_sampling_with_annealed_langevin_monte_carlo.md)

</div>

<!-- RELATED:END -->
