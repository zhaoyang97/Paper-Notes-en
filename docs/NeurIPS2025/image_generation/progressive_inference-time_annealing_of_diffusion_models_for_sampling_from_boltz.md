---
title: >-
  [Paper Note] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities
description: >-
  [NeurIPS 2025 Spotlight][Image Generation][Boltzmann Sampling] This paper proposes PITA (Progressive Inference-Time Annealing), a framework that combines temperature annealing and diffusion smoothing as two complementary interpolation strategies. PITA trains an initial diffusion model at high temperature, then applies a novel Feynman-Kac PDE with SMC resampling to progressively anneal toward lower temperatures at inference time, training a sequence of diffusion models up to t…
tags:
  - "NeurIPS 2025 Spotlight"
  - "Image Generation"
  - "Boltzmann Sampling"
  - "Temperature Annealing"
  - "Feynman-Kac"
  - "Diffusion Models"
  - "Molecular Conformation"
date: 2026-05-08
content_hash: 9e36fddc0843fe27
---

# Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.16471](https://arxiv.org/abs/2506.16471)  
**Code**: [GitHub](https://github.com/taraak/pita)  
**Area**: Diffusion Models / Sampling / Molecular Simulation
**Keywords**: Boltzmann Sampling, Temperature Annealing, Feynman-Kac, Diffusion Models, Molecular Conformation

## TL;DR

This paper proposes PITA (Progressive Inference-Time Annealing), a framework that combines temperature annealing and diffusion smoothing as two complementary interpolation strategies. PITA trains an initial diffusion model at high temperature, then applies a novel Feynman-Kac PDE with SMC resampling to progressively anneal toward lower temperatures at inference time, training a sequence of diffusion models up to the target temperature. This approach achieves equilibrium sampling of alanine dipeptide and tripeptide in Cartesian coordinates for the first time.

## Background & Motivation

**Background**: Efficient sampling from unnormalized Boltzmann distributions is a central challenge in computational biology, chemistry, and physics. Classical approaches include MCMC (often combined with parallel tempering or annealed importance sampling) and molecular dynamics (MD). MCMC annealing suffers from the "mass transport" problem—mode weights are influenced by mode width—while MD requires extremely fine time steps (femtosecond scale), incurring prohibitive computational cost.

**Limitations of Prior Work**: Recently proposed diffusion-based samplers are theoretically attractive for mode mixing, but face three major difficulties in realistic molecular systems: (1) the absence of training data makes accurate Stein score learning difficult; (2) training objectives such as reverse KL are prone to mode collapse; and (3) energy function evaluations are prohibitively numerous. Carefully tuned MCMC with parallel tempering can even outperform state-of-the-art diffusion samplers when normalized by the number of energy evaluations.

**Key Insight**: Temperature annealing and diffusion paths represent two complementary simplification strategies—annealing eliminates high-energy barriers via heating to facilitate mode mixing, while diffusion avoids mass transport through noise injection. PITA combines both: it first collects data via simple MCMC at high temperature to train a diffusion model, then progressively lowers the temperature through inference-time annealing.

## Method

### Overall Architecture

PITA trains a sequence of diffusion models $\{M_{\beta_i}\}$, progressing from high temperature $\beta_0$ (low $\beta$ = high temperature) to the target temperature $\beta_K$ (where $\beta = 1/T$). At each step:
1. A diffusion model (score model + EBM) is trained at temperature $1/\beta_i$ using available samples.
2. The trained model generates samples at temperature $1/\beta_{i+1}$ via inference-time annealing.
3. The new samples are used to train the next diffusion model.

### Key Designs

1. **Inference-Time Annealing (Proposition 1, Core Innovation)**: Given a trained score model $s_t(x;\theta) \approx \nabla\log p_t(x)$ and energy model $U_t(x;\eta) \approx -\log p_t(x)$, the annealed marginal distribution is defined as $q_t(x) \propto \exp(-\gamma U_t(x;\eta))$ with $\gamma = \beta_{i+1}/\beta_i > 1$. A Feynman-Kac PDE governing the time evolution of $q_t$ is derived, yielding the following sampling SDE:
$$dx_t = \left(-a_t x_t + \frac{\zeta_t^2}{2}(s_t(x_t) - \gamma\xi_t \nabla U_t(x_t;\eta))\right)dt + \zeta_t\sqrt{\xi_t}dW_t$$
with weight update:
$$d\log w_t = \left[\frac{\zeta_t^2}{2}\langle\nabla, s_t\rangle - \gamma\langle\nabla U_t, -a_t x + \frac{\zeta_t^2}{2}s_t\rangle - \gamma\frac{\partial U_t}{\partial t}\right]dt$$
**Design Motivation**: When $\gamma=1$ (no annealing) and the model is perfect, the weight variance is exactly zero (Proposition 2), recovering standard diffusion sampling. This guarantees that importance weights remain concentrated for small annealing steps, ensuring high sampling efficiency.

2. **Training Phase (Algorithm 1)**: Four loss functions are jointly optimized:
    - **Denoising Score Matching**: trains the score model $D_t(x_t;\theta)$.
    - **Target Score Matching**: directly supervises the score using $\nabla_x \log \pi(x)$ from the target distribution at high noise levels (activated only when $t \geq t_{\text{thresh}}$), compensating for the high variance of DSM near the data distribution.
    - **EBM Distillation**: distills the score model into an energy-based model.
    - **Energy Pinning**: supervises the endpoint energy model $U_{t=1}(x;\eta)$ with the target energy $\beta_{i+1}\log\pi(x)$, fixing the gauge (translation invariance) of the energy.

3. **Geometric Annealing Variant (Proposition 3 / Appendix A.2)**: For unbounded supports (e.g., $\text{supp}(\pi) = \mathbb{R}^d$), direct annealing may cause numerical instability. A geometric average $\mathcal{N}(0,\mathbb{1})^{(1-\beta)}\pi(x)^\beta$ is instead used, ensuring normalizability at any temperature.

### Network Architecture & Training Details

- LJ-13: EGNN backbone; a single diffusion model conditioned on $\beta$.
- Alanine dipeptide/tripeptide: DiT backbone with a fine-tuning strategy where each temperature step is trained exclusively on samples from the current temperature.
- EBM parameterization follows Thornton et al. (2025); preconditioning follows Karras et al. (2022).

## Key Experimental Results

### Main Results 1: LJ-13 Particle System ($T_L=4 \to T_S=1$)

| Method | Distance-$\mathcal{W}_2$ ↓ | Energy-$\mathcal{W}_2$ ↓ | Geometric-$\mathcal{W}_2$ ↓ |
|------|---------------------------|--------------------------|------------------------------|
| iDEM | 0.127 | 30.78 ± 24.46 | 1.61 ± 0.01 |
| Adjoint Sampling | - | 2.40 ± 1.25 | 1.67 ± 0.01 |
| TA-BG (TarFlow) | 1.21 ± 0.02 | 61.47 ± 0.12 | 4.16 ± 0.01 |
| **PITA** | **0.04 ± 0.00** | **2.26 ± 0.21** | **1.65 ± 0.00** |

PITA achieves a Distance-$\mathcal{W}_2$ that is 3× lower than iDEM and 30× lower than TA-BG.

### Main Results 2: Alanine Dipeptide ALDP ($T_L=1200K \to T_S=300K$)

| Method | Rama-KL | Tica-$\mathcal{W}_1$ ↓ | Energy-$\mathcal{W}_1$ ↓ | Energy-$\mathcal{W}_2$ ↓ | $\mathbb{T}$-$\mathcal{W}_2$ |
|------|---------|------------------------|-------------------------|--------------------------|-----|
| **PITA** | **4.773** | **0.112** | **1.530** | **1.615** | 0.270 |
| MD-Diff | 1.308 | 0.113 | 3.627 | 3.704 | 0.310 |
| TA-BG | 14.993 | 0.219 | 83.457 | 86.176 | 0.979 |
| Score Scaling | 4.588 | 0.183 | 10.282 | 10.460 | 0.550 |

PITA substantially outperforms baselines on energy metrics: Energy-$\mathcal{W}_1$ is **58%** lower than MD-Diff and **98%** lower than TA-BG.

### Main Results 3: Alanine Tripeptide AL3 ($T_L=1200K \to T_S=300K$)

| Method | Rama-KL | Tica-$\mathcal{W}_2$ ↓ | Energy-$\mathcal{W}_1$ ↓ | Energy-$\mathcal{W}_2$ ↓ | Energy Evaluations |
|------|---------|------------------------|-------------------------|--------------------------|------------|
| **PITA** | **1.209** | 0.952 | **2.567** | **2.592** | $8\times10^7$ |
| MD-Diff | 9.662 | 0.426 | 7.416 | 7.599 | $8\times10^7$ |
| TA-BG | 2.078 | 0.454 | 4.782 | 4.863 | $8\times10^7$ |

PITA is the first diffusion-based method to achieve equilibrium sampling of alanine tripeptide in Cartesian coordinates.

### Ablation Study

| Configuration | Energy-$\mathcal{W}_1$ (ALDP) | Description |
|------|------------------------------|------|
| PITA (full) | 1.530 | With MD relaxation |
| PITA (no relaxation) | 86.270 | Without MD relaxation; dramatic performance drop |
| FKC (Skreta 2025) | 11.281 | Resampling only at the final step |
| Score Scaling | 10.282 | Naive score scaling |

### Key Findings

- PITA achieves state-of-the-art performance while maintaining a comparable energy evaluation budget of $5$–$8\times10^7$ evaluations, far fewer than what MD requires for equivalent quality.
- TA-BG performs reasonably at high temperature but degrades sharply as temperature decreases, as importance sampling variance explodes under large temperature gaps.
- TICA plots confirm that PITA successfully recovers slow dynamical modes of the molecular system.
- Brief MD relaxation near the target temperature substantially improves the physical plausibility of generated configurations.

## Highlights & Insights

- The combination of temperature annealing and diffusion paths is elegant: annealing handles mode mixing while diffusion avoids mass transport.
- The Feynman-Kac PDE derivation provides a unified mathematical framework for annealed diffusion, subsuming several existing methods as special cases.
- Proposition 2 guarantees zero weight variance in the absence of annealing, ensuring numerical stability for small annealing steps.
- The progressive fine-tuning strategy—sequentially training from high to low temperature—proves highly effective in practice.

## Limitations & Future Work

- Simultaneous training of a score model and an EBM is required, and EBM training is itself a challenging problem.
- Automatic determination of an optimal temperature schedule (number of steps and step sizes) remains an open question.
- Tica-$\mathcal{W}$ metrics suggest that PITA may underperform certain baselines in recovering mode weights, potentially requiring more refined resampling strategies.
- Validation is currently limited to small molecular systems (13-atom LJ clusters, dipeptide and tripeptide); extension to larger protein systems remains unclear.

## Related Work & Insights

- **Relation to Boltzmann Generators (Noé et al. 2019)**: PITA avoids direct importance sampling, instead performing annealed importance sampling along the diffusion time axis via Feynman-Kac.
- **Complementarity with training-free diffusion samplers** (iDEM, Adjoint Sampling, etc.): PITA leverages high-temperature MCMC data for initialization.
- The temperature scheduling strategy can draw inspiration from Annealed Importance Sampling and Parallel Tempering.
- Integration with transferable sampling (Klein & Noé 2024) may enable generalization across molecular systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The Feynman-Kac PDE derivation is mathematically elegant, and the progressive training framework combining temperature annealing with diffusion models is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers LJ-13, ALDP, and AL3, though limited to small molecular systems due to domain constraints.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous, though the notation and auxiliary results are dense.
- **Value**: ⭐⭐⭐⭐⭐ Achieving equilibrium sampling of peptides in Cartesian coordinates for the first time marks an important milestone for diffusion-based samplers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)
- [\[ICML 2025\] Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions](../../ICML2025/image_generation/annealing_flow_generative_models_towards_sampling_high-dimensional_and_multi-mod.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[ICLR 2026\] Inference-Time Scaling of Diffusion Models Through Classical Search](../../ICLR2026/image_generation/inference-time_scaling_of_diffusion_models_through_classical_search.md)
- [\[NeurIPS 2025\] BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants](boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)

</div>

<!-- RELATED:END -->
