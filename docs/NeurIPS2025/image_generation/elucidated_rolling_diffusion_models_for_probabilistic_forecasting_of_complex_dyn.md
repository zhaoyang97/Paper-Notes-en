---
title: >-
  [Paper Note] Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics
description: >-
  [NeurIPS 2025][Image Generation][rolling diffusion] This paper proposes ERDM, the first framework to successfully unify the Rolling Diffusion paradigm with the principled design choices of EDM (noise schedule…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "rolling diffusion"
  - "EDM"
  - "probabilistic forecasting"
  - "weather prediction"
  - "Navier-Stokes"
date: 2026-05-08
content_hash: abf3c60cf0cdcde4
---

# Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2506.20024](https://arxiv.org/abs/2506.20024)  
**Code**: [NVlabs/ERDM](https://github.com/NVlabs/ERDM)  
**Area**: Image Generation
**Keywords**: rolling diffusion, EDM, probabilistic forecasting, weather prediction, Navier-Stokes

## TL;DR

This paper proposes ERDM, the first framework to successfully unify the Rolling Diffusion paradigm with the principled design choices of EDM (noise schedule, preconditioning, Heun sampler). By employing a progressive noise schedule that explicitly models growing uncertainty, ERDM significantly outperforms autoregressive EDM baselines on Navier-Stokes and ERA5 weather forecasting benchmarks.

## Background & Motivation

Probabilistic forecasting of complex dynamical systems (e.g., medium-range weather forecasting ≤15 days) faces two core challenges:

1. **Uncertainty grows over time**: The chaotic nature of the atmosphere causes far-future predictions to carry substantially greater uncertainty than near-future ones. Standard diffusion models apply uniform noise across all time steps and thus fail to capture this progressive growth.
2. **Autoregressive generation is inefficient**: Prevailing methods (e.g., GenCast) generate individual snapshots sequentially, requiring a full reverse diffusion process per step, which is computationally expensive.

Rolling Sequence Diffusion Models (RSDM) mitigate these issues by assigning increasing noise levels to different lead times, but existing RSDMs are built on DDPM and therefore do not exploit the superior network preconditioning, loss weighting, and second-order samplers offered by the EDM framework.

## Core Problem

How to systematically transfer EDM's principled design choices (noise schedule, preconditioning, loss weighting, sampler) to the progressive-noise setting of rolling diffusion, while resolving technical challenges such as temporal loss weighting and first-window initialization?

## Method

### 1. Rolling EDM Noise Schedule

ERDM operates on sequences of window size $W$, assigning increasing noise levels to each snapshot $w$. The core noise schedule is defined as:

$$\bar{\sigma}_w(t) = \left(\sigma_{\max}^{1/\rho} + t_{w,t}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\right)^\rho$$

where $t_{w,t} = 1 - \frac{w-t}{W}$ is the local diffusion time. The key parameter is the curvature $\rho$: the EDM default of $\rho=7$ performs poorly in ERDM, and the authors find $\rho=-10$ to be superior — this keeps all snapshots operating at lower noise levels, providing more informative signals for denoising.

The schedule satisfies $\sigma_{\min} = \bar{\sigma}_1(1) < \bar{\sigma}_1(0) = \bar{\sigma}_2(1) < \cdots < \bar{\sigma}_W(0) = \sigma_{\max}$, ensuring that noise levels connect naturally as the window advances.

### 2. Probability Flow ODE and Sampling

For a windowed sequence $\bar{\bm{x}}_{1:W}$ with progressive noise, the probability flow ODE is:

$$\mathrm{d}\bar{\bm{x}} = -\text{diag}(\bar{\sigma}_1(t)\dot{\sigma}_1(t)\mathbf{I}_D, \ldots, \bar{\sigma}_W(t)\dot{\sigma}_W(t)\mathbf{I}_D) \nabla_{\bar{\bm{x}}} \log p(\bar{\bm{x}}; \bar{\bm{\sigma}}(t)) \mathrm{d}t$$

During sampling, after each complete ODE integration ($t: 0 \to 1$), the first snapshot is fully denoised and output; the remaining snapshots shift one position to the left, and a fresh pure-noise snapshot is appended at the end. This process repeats, reducing NFE by **5×** compared to autoregressive EDM.

### 3. Uncertainty-Aware Loss Weighting

EDM's original loss weighting $\lambda(\sigma)$ ensures that the target network $F_\theta$ receives inputs and produces outputs with unit variance. ERDM augments this with a log-normal PDF weighting:

$$\text{Effective weight} = \lambda(\bar{\sigma}_w) \cdot f(\bar{\sigma}_w; P_{\text{mean}}, P_{\text{std}})$$

$f(\bar{\sigma}_w)$ emphasizes snapshots at intermediate noise levels — the critical transition zone from determinism to stochasticity, where learning is most informative. Ablations show that $P_{\text{mean}} > 0$ (vs. the EDM default of $-1.2$) is essential for ERDM.

### 4. First-Window Initialization Strategy

An external predictor (e.g., a pretrained EDM) generates $\hat{\bm{y}}_{1:W}$, to which noise is then added: $\bar{\bm{x}}_w \sim \mathcal{N}(\hat{\bm{y}}_w, \bar{\sigma}_w^2(0)\mathbf{I}_D)$. Compared to initializing from pure noise, this approach prevents the denoiser from wasting learning capacity on the initialization regime.

### 5. Hybrid Spatiotemporal Architecture

Causal temporal attention layers are inserted into a 2D ADM U-Net (prior to each downsampling/upsampling block), with the temporal layers also conditioned on noise levels. This hybrid 3D architecture outperforms the naive approach of stacking the temporal dimension into the channel dimension by **4×**.

## Key Experimental Results

### Navier-Stokes Fluid Dynamics (64-step trajectories, 50-member ensemble)

| Method | Final CRPS | Calibration |
|--------|-----------|-------------|
| EDM W=1 (autoregressive) | Baseline | Noticeably underdispersed |
| EDM W=4 | Second best | Underdispersed |
| DYffusion | Comparable to EDM W=4 | Moderate |
| **ERDM W=6** | **~50% better than EDM W=4** | **Best** |

ERDM maintains a consistent ~50% CRPS advantage beyond step 15, demonstrating substantially stronger long-range forecasting capability.

### ERA5 Weather Forecasting (1.5° resolution, 15 days, 10-member ensemble)

| Comparison | Relative CRPS Gain | Training Cost |
|------------|-------------------|---------------|
| EDM baseline | **ERDM improves ~10%** | 4 H200 × 5 days |
| IFS ENS (numerical model) | Competitive at medium-to-long range | Computationally intensive |
| NeuralGCM ENS | Comparable at medium-to-long range | 128 v5 TPUs × 10 days |
| GenCast | Similar architecture, different resolution | 32 v5 TPUs × 3.5 days |

ERDM's power spectrum is on par with the physical model IFS ENS — a level of physical fidelity that is exceptionally rare among ML weather models. NeuralGCM, by contrast, underestimates energy at mid-to-high frequencies.

### Inference Efficiency (15 days, 5 members, A100)

| Model | NFE | Inference Time (s) | GPU Memory (GB) |
|-------|-----|--------------------|-----------------|
| EDM | 600 | 237 | 21 |
| **ERDM** | **120** | **209** | 49 |

ERDM reduces NFE by 5×, achieving slightly faster total inference time, at the cost of doubled memory usage.

### Key Ablation Findings

1. **Noise schedule $\rho=-10$** vs. $\rho=7$ (EDM default): 2× CRPS degradation.
2. **Fixed vs. stochastic noise schedule**: Stochastic training degrades performance by ~2×.
3. **Log-normal loss weighting**: Removing it degrades performance by >2×.
4. **Spatiotemporal architecture vs. channel stacking**: Improper architecture causes 4× performance degradation.

## Highlights & Insights

1. **First framework to successfully unify EDM with rolling diffusion**, systematically resolving the adaptation of noise schedule, preconditioning, and sampler.
2. Identification of **three critical design choices** — $\rho=-10$ curvature, log-normal loss weighting, and hybrid spatiotemporal architecture — each of which is individually indispensable.
3. Training requires only **4 GPUs × 5 days**, far less than NeuralGCM (128 TPUs × 10 days), while achieving comparable medium-to-long-range forecasting performance.
4. Physical fidelity (power spectrum) reaches the level of the numerical model IFS ENS — an extremely rare achievement for ML-based weather models.

## Limitations & Future Work

1. The 3D denoiser architecture **doubles GPU memory requirements** (49 GB vs. 21 GB), limiting scalability to higher resolutions.
2. Short-range forecasting (<2 days) underperforms IFS ENS, owing to the EDM initialization strategy and an architecture not specifically optimized for weather.
3. The system depends on an external model to initialize the first window, increasing overall system complexity.
4. The explicit loss weighting may be suboptimal relative to importance sampling approaches, leaving room for further improvement.

## Related Work & Insights

| Dimension | RSDM / FIFO-Diffusion | GenCast | ERDM |
|-----------|----------------------|---------|------|
| Diffusion backbone | DDPM | EDM | **EDM** |
| Progressive noise | ✓ | ✗ | **✓** |
| Preconditioning | None | EDM standard | **Vectorized EDM** |
| Loss weighting | Simple | EDM standard | **Uncertainty-aware** |
| Spatiotemporal architecture | Basic | Graph network | **Hybrid 3D U-Net** |

**Progressive noise as natural uncertainty modeling**: In any sequential forecasting task with increasing uncertainty, a progressive noise schedule is a more principled choice than uniform noise.

**Transferability of EDM design principles**: Preconditioning and loss weighting are not limited to image generation; they can be systematically transferred to diffusion-based modeling in scientific computing.

**Rolling window mechanism**: By advancing the window rather than performing step-by-step autoregression, ERDM realizes a form of "parallelized autoregression" that achieves a better trade-off between information propagation efficiency and computational cost.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First successful unification of EDM and rolling diffusion; all three key contributions represent substantive innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual benchmarks (fluid dynamics + weather), multiple baselines, comprehensive ablations, power spectrum analysis, and calibration evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous mathematical derivations, clear algorithmic descriptions, and intuitive figures.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a powerful and general framework for diffusion-based probabilistic forecasting of scientific dynamical systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models](cross-fluctuation_phase_transitions_reveal_sampling_dynamics_in_diffusion_models.md)
- [\[NeurIPS 2025\] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)

</div>

<!-- RELATED:END -->
