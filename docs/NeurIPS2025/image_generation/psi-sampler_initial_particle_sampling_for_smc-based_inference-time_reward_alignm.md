---
title: >-
  [Paper Note] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models
description: >-
  [NeurIPS 2025][Image Generation][Inference-time alignment] This paper proposes the Ψ-Sampler framework, which introduces initial particle sampling based on the preconditioned Crank-Nicolson Langevin (pCNL) algorithm into SMC-based inference-time reward alignment. By initializing particles from a reward-aware posterior distribution, the framework substantially improves alignment performance on layout-guided generation, quantity-aware generation, and aesthetic preference generation.
tags:
  - NeurIPS 2025
  - Image Generation
  - Inference-time alignment
  - Sequential Monte Carlo
  - Reward alignment
  - MCMC
  - Preconditioned Crank-Nicolson
date: 2026-05-08
content_hash: 6e4b5a19294820ce
---

# Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.01320](https://arxiv.org/abs/2506.01320)
**Code**: [Project Page](https://psi-sampler.github.io/)
**Area**: Diffusion Models / Image Generation
**Keywords**: Inference-time alignment, Sequential Monte Carlo, Reward alignment, MCMC, Preconditioned Crank-Nicolson

## TL;DR

This paper proposes the Ψ-Sampler framework, which introduces initial particle sampling based on the preconditioned Crank-Nicolson Langevin (pCNL) algorithm into SMC-based inference-time reward alignment. By initializing particles from a reward-aware posterior distribution, the framework substantially improves alignment performance on layout-guided generation, quantity-aware generation, and aesthetic preference generation.

## Background & Motivation

**Paradigm shift in inference-time alignment**: Analogous to the paradigm shift in LLMs from pre-training to post-training (e.g., GPT-o3, DeepSeek's "Aha moment"), diffusion models are increasingly emphasizing alignment optimization at the inference stage. SMC methods treat the denoising process as sequential sampling, guiding generation by maintaining multiple particles and resampling according to a reward function.

**Core deficiency of existing SMC methods**: All existing SMC methods (TDS, DAS, FPS, etc.) initialize particles from a standard Gaussian prior, entirely ignoring reward information. This leads to two critical problems:

**Diffusion coefficient decay**: In the late denoising stage, $g(t)^2 \to 0$, causing the reward gradient's influence to vanish as it is scaled by $g^2(t)$, making it increasingly difficult to guide particles toward high-reward regions.

**Multimodal reward functions**: When the reward function is highly non-convex, intermediate distributions in the late stages become highly concentrated around specific modes, reducing inter-mode connectivity and making it difficult for particles to escape local optima.

**Core Idea**: Rather than relying on particle exploration in the late denoising stages, the paper proposes sampling high-quality particles from a **reward-aware posterior distribution** at the initial stage. The optimal initial distribution admits an analytic form: $\tilde{p}_1^*(\mathbf{x}_1) = \frac{1}{Z_1} p_1(\mathbf{x}_1) \exp\left(\frac{r(\mathbf{x}_{0|1})}{\alpha}\right)$.

**Challenges of high-dimensional MCMC**: The latent space of FLUX models reaches dimension 65536. The acceptance rate of conventional MALA degrades sharply as dimensionality increases (step size must shrink as $O(d^{-1/3})$), resulting in extremely slow mixing.

## Method

### Overall Architecture

Ψ-Sampler = pCNL initial particle sampling + standard SMC denoising. High-quality initial particles are first sampled via pCNL from the posterior distribution, then fed into the standard SMC pipeline. The total NFE budget is split equally between initial particle sampling and SMC.

### Key Designs

1. **Reward-aware posterior distribution**:
   The optimal initial distribution is $\tilde{p}_1^*(\mathbf{x}_1) \propto p_1(\mathbf{x}_1) \exp(r(\mathbf{x}_{0|1})/\alpha)$, where $\mathbf{x}_{0|1}$ is the clean image estimated from $\mathbf{x}_1$ via the Tweedie formula. This is an unnormalized density of known form, amenable to MCMC sampling.
   - Recent distillation techniques (e.g., rectified flow) straighten trajectories and enable earlier accurate Tweedie estimation, creating favorable conditions for effective reward evaluation from the fully noisy state.
   - The regularization parameter $\alpha$ controls the trade-off between reward maximization and prior preservation.

2. **Preconditioned Crank-Nicolson Langevin (pCNL) algorithm**:
   The pCN algorithm is designed for infinite-dimensional / function-space settings, with a semi-implicit Euler discretization at its core:
   $$\mathbf{x}' = \rho \mathbf{x} + \sqrt{1-\rho^2}\left(\mathbf{z} + \frac{\sqrt{\epsilon}}{2} \nabla \frac{r(\mathbf{x}_{0|1})}{\alpha}\right), \quad \rho = \frac{1-\epsilon/4}{1+\epsilon/4}$$

   **Key differences from MALA**:
   - The pCNL proposal distribution preserves the Gaussian prior (prior-preserving), so the acceptance rate does not degrade with dimensionality.
   - In a 65536-dimensional space, MALA's acceptance rate drops to nearly zero at step sizes above 0.05, whereas pCNL maintains reasonable acceptance rates at step size 2.0.
   - Larger pCNL step sizes yield faster mixing and more efficient exploration.
   - pCNL also employs Metropolis-Hastings correction to guarantee convergence to the correct distribution.

3. **Initial particle sampling procedure**:
   - Sample the initial state from the prior.
   - Run the pCNL chain and discard the burn-in period.
   - Draw $K$ particles at fixed intervals (thinning).
   - Use a fixed step size for simplicity and effectiveness.
   - The sampled particles are directly used as initial particles for SMC.

### Loss & Training

This is a purely inference-time method requiring no training. The core computational cost consists of:
- Computing $\nabla r(\mathbf{x}_{0|1})$ (the gradient of the reward with respect to the Tweedie estimate) at each pCNL step.
- Computing weighted denoising steps at each SMC step.
- Allocating the total NFE budget between pCNL and SMC.

## Key Experimental Results

### Main Results — Quantitative Comparison on Three Tasks (FLUX Model)

| Task | Metric | DPS | FreeDoM | TDS | DAS | Top-K | ULA | MALA | **Ψ-Sampler** |
|------|--------|-----|---------|-----|-----|-------|-----|------|---------------|
| Layout generation | GroundingDINO↑ | 0.166 | 0.177 | 0.417 | 0.363 | 0.425 | 0.370 | 0.401 | **0.467** |
| Layout generation | mIoU (held-out)↑ | 0.215 | 0.229 | 0.402 | 0.342 | 0.427 | 0.374 | 0.401 | **0.471** |
| Quantity-aware | T2I-Count↓ | 14.19 | 15.21 | 1.804 | 1.151 | 1.077 | 3.035 | 1.601 | **0.850** |
| Quantity-aware | MAE (held-out)↓ | 15.7 | 15.68 | 5.3 | 4.18 | 3.68 | 4.83 | 3.58 | **2.93** |
| Aesthetic preference | Aesthetic↑ | 6.139 | 6.310 | 6.853 | 6.935 | 6.879 | 6.869 | 6.909 | **7.012** |

### Ablation Study — pCNL vs. MALA Step Size

| Step Size | MALA Acceptance | pCNL Acceptance | MALA mIoU | pCNL mIoU |
|-----------|----------------|----------------|-----------|-----------|
| 0.05 | ~25% | ~80% | ~0.40 | ~0.43 |
| 0.5 | ~0% | ~50% | Degraded | **~0.47** |
| 2.0 | ~0% | ~25% | Degraded | ~0.45 |

### Key Findings

- **Posterior initialization vs. prior initialization**: All posterior-initialized methods outperform prior-initialized SMC, confirming the importance of initial particle quality.
- **pCNL vs. ULA/MALA**: ULA lacks MH correction and introduces bias; MALA suffers from drastically reduced acceptance rates in high dimensions; pCNL addresses both issues.
- **Severe inadequacy of single-particle methods**: DPS and FreeDoM are substantially inferior to SMC-based methods across all tasks.
- **Generalization on held-out rewards**: Ψ-Sampler achieves the best performance not only on training rewards but also on held-out rewards, indicating that the improvement stems from genuinely higher-quality samples.
- **2D toy experiment validation**: On a 6-mode Gaussian mixture, MALA+SMC still exhibits mode dropping, whereas Ψ-Sampler achieves complete coverage of the target distribution.

## Highlights & Insights

- **First application of pCN to generative modeling**: Transfers high-dimensional MCMC techniques from Bayesian inverse problems in PDE-constrained settings to inference-time alignment in generative models.
- **Accurate diagnosis**: Clearly identifies diffusion coefficient decay as the cause of reward guidance failure in late denoising stages, motivating the front-loading of computational budget to the initialization phase.
- **Synergy with distillation trends**: The trend toward straighter trajectories and improved early Tweedie estimation in modern models directly benefits the proposed posterior initialization approach.
- **Consistent theory-experiment correspondence**: The optimal initial distribution derived from the SOC framework is empirically validated.

## Limitations & Future Work

- Assumes differentiable reward models — not applicable to non-differentiable rewards.
- Relies on the accuracy of the Tweedie approximation — early-step estimates may be insufficiently accurate for non-distilled models.
- pCNL requires a Gaussian prior, restricting its application to $t=1$.
- Uses a fixed step size; adaptive step size strategies may further improve efficiency.
- May be misused to generate misleading or harmful photorealistic fake images — responsible deployment is necessary.

## Related Work & Insights

- Complementary to SMC methods such as TDS (Twisted Diffusion Sampler) and DAS: those methods improve intermediate SMC steps, while Ψ-Sampler improves initialization.
- The pCN algorithm originates from the field of PDE-constrained Bayesian inverse problems; this cross-domain transfer yields substantial gains.
- Consistent with the inference-time scaling philosophy in LLMs: improving alignment through search rather than fine-tuning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First introduction of the pCN algorithm to generative modeling; the idea of initial posterior sampling is clear and compelling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three substantially different tasks, multi-dimensional metrics, step size analysis, and toy experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and theoretical derivations are complete, though the MCMC background raises the entry barrier.
- **Value**: ⭐⭐⭐⭐⭐ — Introduces a new dimension for inference-time scaling (initialization quality); the method is general and orthogonal to existing SMC approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](../../ICLR2026/image_generation/glass_flows_reward_alignment_diffusion.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)

</div>

<!-- RELATED:END -->
