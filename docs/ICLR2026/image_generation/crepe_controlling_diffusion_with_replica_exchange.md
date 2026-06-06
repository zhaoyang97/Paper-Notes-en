---
title: >-
  [Paper Note] CREPE: Controlling Diffusion with Replica Exchange
description: >-
  [ICLR 2026][Image Generation][replica exchange] This paper proposes CREPE, an inference-time control method for diffusion models based on Replica Exchange (Parallel Tempering)…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "replica exchange"
  - "parallel tempering"
  - "inference-time control"
  - "SMC alternative"
  - "reward tilting"
  - "CFG debiasing"
date: 2026-05-08
content_hash: eb1fce11f9608b87
---

# CREPE: Controlling Diffusion with Replica Exchange

**Conference**: ICLR 2026
**arXiv**: [2509.23265](https://arxiv.org/abs/2509.23265)  
**Code**: Available (GitHub)  
**Area**: Diffusion Models / Inference-Time Control
**Keywords**: replica exchange, parallel tempering, inference-time control, SMC alternative, reward tilting, CFG debiasing

## TL;DR
This paper proposes CREPE, an inference-time control method for diffusion models based on Replica Exchange (Parallel Tempering), serving as the computational dual of SMC — it operates in parallel across denoising steps while generating samples serially. CREPE offers high sample diversity, supports online refinement, and handles a variety of tasks including temperature annealing, reward tilting, model composition, and CFG debiasing.

## Background & Motivation

**Background**: Inference-time control of diffusion models — satisfying new constraints without retraining — is an active research direction. The dominant approach is Sequential Monte Carlo (SMC), which corrects biases introduced by heuristic guidance by maintaining a set of weighted particles along the denoising trajectory.

**Limitations of Prior Work**: SMC has three key limitations: (a) it requires maintaining a large number of particles simultaneously throughout the entire denoising trajectory, incurring high memory overhead; (b) sample diversity is poor, particularly when the particle count is small, as resampling leads to particle collapse; (c) samples cannot be refined after generation — if results are unsatisfactory or new constraints are introduced, generation must restart from scratch.

**Key Challenge**: The "parallel particles × serial timesteps" paradigm of SMC inherently creates bottlenecks in diversity and flexibility, motivating a computationally dual alternative.

**Goal**: To propose an alternative to SMC that (a) generates particles one at a time rather than in batches, (b) maintains high diversity after burn-in, (c) supports online refinement and early stopping, and (d) covers diverse tasks including tempering, reward tilting, model composition, and CFG debiasing.

**Key Insight**: Replica Exchange / Parallel Tempering is precisely the computational dual of SMC — it runs chains in parallel across different denoising steps while generating samples serially. This MCMC framework is adapted to the diffusion model setting.

**Core Idea**: Adapt the swap moves of Parallel Tempering to the path space of diffusion models, leveraging the Radon-Nikodym Estimator to compute acceptance probabilities, thereby enabling inference-time control without access to explicit target densities.

## Method

### Overall Architecture

CREPE maintains $M+1$ particles, each residing at a distinct diffusion timestep $t_0 < t_1 < \cdots < t_M$ (spanning from the data distribution to pure noise). Each iteration consists of:
1. **Communication step**: Adjacent particles exchange positions via APT swap moves — forward and backward proposal paths are generated, and acceptance probabilities determine whether swaps occur.
2. **Local exploration step**: Each particle performs a local MCMC update at its respective timestep.
3. Both steps can be parallelized.

### Key Designs

1. **Accelerated PT Swap Move in Diffusion Path Space**:

    - **Function**: Enables particles $(x, x')$ residing at timesteps $t$ and $t'$ to exchange positions via forward/backward diffusion paths.
    - **Mechanism**: Starting from $x$, the forward diffusion process is applied to reach $t'$; starting from $x'$, the reverse process is applied to reach $t$. A Metropolis-Hastings acceptance probability $\alpha_{t,t'}$ determines whether the swap is accepted. This probability is computed via the Radon-Nikodym Estimator (RNE), which exploits the ratio of forward and backward transition probabilities from the pretrained diffusion model.
    - **Design Motivation**: Standard PT requires access to the unnormalized target density, which is unavailable during inference-time control. The RNE relation $p_{t'}(x_{t'})/p_t(x_t) = R_{t,t'}^{-1}$ circumvents the need to evaluate densities directly.

2. **Annealing Path Design**:

    - **Function**: Defines sequences of intermediate distributions for different control tasks.
    - **Mechanism**:
        - Tempering: $\pi_t(x) \propto p_t^j(x)^\beta$
        - Reward tilting: $\pi_t(x) \propto p_t^j(x) \exp(r_t(x))$
        - Model composition: $\pi_t(x) \propto \prod_j p_t^j(x)$
        - CFG debiasing: $\pi_t(x) \propto p_t(x)^{1-w} p_t(x|c)^w$
    - **Design Motivation**: All target distributions can be expressed in terms of pretrained model density ratios, making the acceptance probabilities computable via the RNE.

3. **Online Refinement**:

    - **Function**: Dynamically adds or modifies constraints while the MCMC chain is running.
    - **Mechanism**: The MCMC chain can run indefinitely; introducing a new reward term at any point only requires modifying the annealing path, and PT naturally adapts.
    - **Design Motivation**: SMC is a one-shot procedure and cannot be modified post hoc; CREPE, as an MCMC method, naturally supports iterative refinement.

4. **Support for Both Continuous and Discrete Diffusion**:

    - **Function**: Derives swap rates for both Gaussian diffusion (SDE) and discrete masked diffusion (CTMC).
    - **Design Motivation**: Provides coverage of image generation (continuous) and text/discrete data (discrete masked diffusion, e.g., MDLM).

### Loss & Training

- No training is required; CREPE operates entirely at inference time.
- Requires only the forward and reverse processes of a pretrained diffusion model.
- Computational cost is comparable to SMC but distributed differently — PT incurs a burn-in cost, after which the per-sample cost remains constant.

## Key Experimental Results

### Main Results

**Molecular Temperature Annealing (Alanine Dipeptide / Tetrapeptide / Hexapeptide)**

| Method | Energy TVD ↓ | TICA MMD ↓ | Notes |
|--------|-------------|-----------|-------|
| FKC (SMC) | 0.345 | 0.116 | SMC baseline |
| CREPE (Ours) | **0.224** | **0.096** | Dipeptide |
| CREPE | **0.122** | **0.035** | Tetrapeptide |

**CFG Debiasing (ImageNet-64)**

| Method | #Samples | IR ↑ | CLIP ↑ | FID ↓ |
|--------|----------|------|--------|-------|
| FKC (SMC) | 8 | **-0.29** | **24.17** | **1.85** |
| CREPE | 8 | -0.30 | 24.10 | 1.92 |
| FKC | 512 | -0.08 | 24.31 | 1.96 |
| CREPE | 512 | **0.09** | 24.28 | **1.79** |

### Key Findings
- SMC outperforms CREPE with few samples (due to burn-in), but CREPE surpasses SMC as the sample count increases, with FID improving continuously.
- CREPE's core advantage lies in **diversity** — SMC's resampling causes particle collapse (visually similar outputs within a batch), whereas CREPE's MCMC chain naturally explores a broader distribution.
- In online refinement experiments, CREPE satisfies newly introduced constraints within only 1k iterations, demonstrating strong flexibility.
- CREPE is also effective on discrete diffusion (MNIST MDLM), confirming the generality of the approach.

## Highlights & Insights
- The **computational dual perspective on SMC** is exceptionally elegant — flipping "parallel particles × serial timesteps" to "serial particles × parallel timesteps" captures the core contribution in a single sentence. This duality (Syed et al., 2024) reflects deep connections in sampling theory.
- **Online refinement** is entirely beyond the reach of SMC and is highly valuable for practical applications such as interactive generation and iterative design.
- The **unified framework** covers tempering, reward tilting, model composition, and CFG debiasing, and these objectives can be freely combined. The methodology is broadly applicable.

## Limitations & Future Work
- Sample quality during burn-in is poor; CREPE is inferior to SMC in low-sample regimes.
- Each swap move requires simulating both forward and backward diffusion paths, imposing non-trivial computational overhead.
- For high-resolution images (ImageNet-512), only qualitative results for reward tilting are presented; quantitative comparisons are lacking.
- Acceptance rates may decrease with dimensionality, requiring finer annealing schedules.
- Combinations with guidance methods (e.g., DPS, FreeDoM) remain unexplored.

## Related Work & Insights
- **vs. FKC (SMC)**: Computational dual relationship. SMC is superior with few samples; CREPE is superior with many. CREPE offers better diversity.
- **vs. Twisted SMC / DDRM**: Both are debiasing methods for inference-time control, but CREPE is MCMC-based rather than importance-sampling-based.
- **vs. APT (Zhang et al., 2025)**: CREPE extends APT from the setting with known unnormalized densities to the setting where only a pretrained diffusion model is available.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First adaptation of Parallel Tempering to inference-time control of diffusion models; the SMC dual perspective is remarkably elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple modalities including molecular, image, trajectory, and discrete data, but quantitative evaluations on high-resolution images are limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretically rigorous but notation-dense; a strong background in stochastic processes is recommended.
- **Value**: ⭐⭐⭐⭐ — Introduces a new paradigm for inference-time control of diffusion models, with unique advantages in diversity and online refinement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICLR 2026\] Diffusion Alignment as Variational Expectation-Maximization](diffusion_alignment_as_variational_expectation-maximization.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](../../ICML2026/image_generation/conditional_diffusion_sampling.md)
- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)

</div>

<!-- RELATED:END -->
