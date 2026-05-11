---
title: >-
  [Paper Note] Compositional amortized inference for large-scale hierarchical Bayesian models
description: >-
  [ICLR 2026][Image Generation][amortized Bayesian inference] This paper extends compositional score matching (CSM) to hierarchical Bayesian models…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "amortized Bayesian inference"
  - "hierarchical model"
  - "compositional score matching"
  - "diffusion model"
  - "scalability"
date: 2026-05-08
content_hash: 88644ac3ed1f3263
---

# Compositional amortized inference for large-scale hierarchical Bayesian models

**Conference**: ICLR 2026
**arXiv**: [2505.14429](https://arxiv.org/abs/2505.14429)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: amortized Bayesian inference, hierarchical model, compositional score matching, diffusion model, scalability

## TL;DR
This paper extends compositional score matching (CSM) to hierarchical Bayesian models, addressing numerical instability under large numbers of data groups via a novel error-damping estimator and mini-batch strategy. It achieves, for the first time, amortized inference over hierarchical models exceeding 750,000 parameters (250,000+ data groups), validated on a real-world fluorescence lifetime imaging application.

## Background & Motivation

**Background**: Amortized Bayesian inference (ABI) uses neural networks to learn a universal posterior function, enabling zero-latency sampling after training. However, scaling ABI to hierarchical models remains a major challenge, as training requires simulating complete "dataset-of-datasets" for each batch, incurring substantial computational cost.

**Limitations of Prior Work**: (a) Direct ABI requires simulating $J$ groups of data (where $J$ can reach hundreds of thousands), demanding tens of thousands of simulations per batch; (b) existing CSM methods become numerically unstable when the number of groups $J > 100$—the cumulative composition of scores compounds approximation errors, causing divergent sampling; (c) MCMC methods (e.g., NUTS) are also infeasible for large-scale hierarchical models.

**Key Challenge**: The divide-and-conquer strategy of CSM makes training efficient (single-group simulation), but inference requires composing $J$ score estimates—a process that becomes numerically explosive as $J$ grows. More data groups → more accumulated score terms → more severe compounding of approximation errors.

**Goal**: Enable CSM to remain numerically stable at scales of hundreds of thousands of data groups, realizing amortized inference for large-scale hierarchical models.

**Key Insight**: Introduce an error-damping bridging density that attenuates the influence of composed scores in high-noise regimes and recovers full accumulation in low-noise regimes, combined with a mini-batch estimator to address memory constraints.

**Core Idea**: Modulate the accumulation rate of composed scores along the diffusion trajectory via a time-varying damping function $d(t)$—damping in high-noise regions (preventing divergence) and recovering in low-noise regions (preserving correctness).

## Method

### Overall Architecture
The hierarchical model is defined as: $\mathbf{Y}_j \sim p(\mathbf{Y}_j | \boldsymbol{\theta}_j, \boldsymbol{\eta})$, $\boldsymbol{\theta}_j \sim p(\boldsymbol{\theta} | \boldsymbol{\eta})$, $\boldsymbol{\eta} \sim p(\boldsymbol{\eta})$. Two score networks are trained: a local network $s^{\text{local}}(\boldsymbol{\theta}_t, \boldsymbol{\eta}, \mathbf{Y}_j)$ and a global network $s^{\text{global}}(\boldsymbol{\eta}_t, \mathbf{Y}_j)$. Training requires only single-group simulations. At inference, the error-damping CSM composes global scores to sample $\boldsymbol{\eta}$, followed by conditional sampling of each $\boldsymbol{\theta}_j$.

### Key Designs

1. **SDE Sampler Replacing Langevin Sampling**:

    - **Function**: Replaces fixed-step annealed Langevin sampling with an adaptive-step SDE solver.
    - **Mechanism**: Applies the reverse SDE formulation with an adaptive solver for automatic step-size adjustment.
    - **Design Motivation**: Langevin sampling requires many steps and is sensitive to step size; adaptive solvers automatically reduce step size in high-noise regions and increase it in low-noise regions.

2. **Error-Damping Bridging Density**:

    - **Function**: Introduces a time-varying damping function $d(t)$ to modulate the composed scores.
    - **Formula**: $p_t(\boldsymbol{\eta}_t | \mathbf{Y}_{1:J}) \propto p(\boldsymbol{\eta}_t)^{(1-J)(1-t)d(t)} \prod_j p_t(\boldsymbol{\eta}_t | \mathbf{Y}_j)^{d(t)}$
    - **Constraints**: $d(0)=1$ (recovers the true posterior at low noise); $d(1) \leq 1$ (attenuates contribution at high noise).
    - **Damping Schedule**: Exponential decay $d(t) = \exp(-\ln(1/d_1) \cdot t)$, where $d_1$ is a tunable hyperparameter.
    - **Design Motivation**: Adaptive solvers require extremely small step sizes in high-noise regions, where score approximation errors are most severe; damping the composed score contribution in this regime mitigates divergence.

3. **Mini-batch Composed Score Estimator**:

    - **Function**: Approximates the full accumulation over $J$ groups using a random subset.
    - **Formula**: $\hat{s}(\boldsymbol{\eta}_t) = (1-J)(1-t)\nabla \log p(\boldsymbol{\eta}_t) + \frac{J}{M}\sum_{i=1}^M s(\boldsymbol{\eta}_t, \mathbf{Y}_{j_i})$
    - **Property**: Unbiased estimator (proven in Proposition 3.1).
    - **Design Motivation**: Full accumulation over $J > 10{,}000$ groups is computationally and memory infeasible. Mini-batching introduces variance, which is controlled in conjunction with the damping strategy.

4. **Noise Schedule Adjustment**:

    - **Function**: Uses a different noise schedule at inference than at training, compressing the high-noise interval.
    - **Mechanism**: Increases the shift parameter $s$ of the cosine schedule to reduce the number of sampling steps spent in high-noise regions.
    - **Design Motivation**: High-noise regions are where error accumulation is most severe; minimizing dwell time in this interval reduces overall instability.

### Loss & Training
Global and local score models are jointly trained (Eq. 11) using denoising score matching with likelihood weighting. Training requires only single-group simulations, making simulation efficiency extremely high and keeping training cost independent of $J$.

## Key Experimental Results

### Main Results (Convergence Across Methods)

| Method | N=10 | N=100 | N=10K | N=100K |
|--------|------|-------|-------|--------|
| Annealed Langevin | ✓ | ✗ | ✗ | ✗ |
| Euler-Maruyama | ✓ | ✗ | ✗ | ✗ |
| Probability ODE | ✓ | ✓ | ✗ | ✗ |
| GAUSS | ✓ | ✓ | ✗ | ✗ |
| **Ours (damping)** | **✓** | **✓** | **✓** | **✓** |

### Ablation Study (Hierarchical AR Model)

| Configuration | Global Param. RMSE | Local Param. RMSE | Notes |
|---------------|--------------------|-------------------|-------|
| Direct ABI (small scale) | Best | Best | Requires full simulation |
| CSM (no damping) | Diverges | — | Fails for $J > 100$ |
| CSM + damping | Near direct ABI | Near direct ABI | Simulation cost << one full simulation |

### Key Findings
- **All existing CSM methods fail for $N > 100$**: Langevin, Euler-Maruyama, ODE, and GAUSS all fail beyond 10K data points.
- **Error damping is the key to large-scale scalability**: With damping, stable convergence is achieved at 100K data points.
- **Validated on a real application**: Fluorescence lifetime imaging with $J > 250{,}000$ groups and $> 750{,}000$ parameters—the first amortized hierarchical inference at such scale.
- **High training efficiency**: No full hierarchical dataset simulation is required; single-group simulation suffices, so training cost does not grow with $J$.

## Highlights & Insights
- **Divide-and-conquer Bayesian inference**: Decomposition into single groups during training and composition at inference avoids the exponential simulation cost of "dataset-of-datasets."
- **Elegant time-varying damping design**: The constraints $d(0)=1$ and $d(1)\leq 1$ ensure posterior correctness at low noise and numerical stability at high noise—a principled balance between unbiasedness and stability.
- **Unbiasedness of mini-batch estimation**: The proof of unbiasedness for the random subset estimator is concise, and variance remains controllable in conjunction with damping.
- **First breakthrough of the CSM scalability bottleneck**: From 100 data points to 100,000—a three-order-of-magnitude improvement.

## Limitations & Future Work
- **Damping parameter $d_1$ requires tuning**: Although it can be adjusted at inference time, the optimal value depends on the problem scale. Adaptive $d_1$ selection is a natural direction for improvement.
- **Score networks trained independently per group**: Information sharing across groups is limited; inter-group information propagation could further improve efficiency.
- **Only two-level hierarchical models are validated**: Deeper hierarchical structures (3+ levels) would require recursive composition, and stability in such settings remains unverified.
- **Mini-batching introduces variance**: Although the estimator is unbiased, variance grows with $J/M$. Adaptive mini-batch sizing may be beneficial.

## Related Work & Insights
- **vs. Geffner et al. (2023) CSM**: The original CSM uses annealed Langevin sampling and fails beyond $N > 10$. This work employs SDE solvers with damping, scaling to $100K+$.
- **vs. Linhart et al. (GAUSS)**: Uses a second-order Gaussian approximation, limited to 100 observations. This work achieves a three-order-of-magnitude extension via damping and mini-batching.
- **vs. Direct ABI (Habermann/Heinrich)**: Direct ABI requires full hierarchical simulation; this work requires only single-group simulation, offering a substantial computational advantage at large $J$.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The error-damping bridging density and mini-batch composed estimator are novel and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three-level validation (Gaussian toy → hierarchical AR → real fluorescence imaging), though comparison with more baselines would strengthen the evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous; the narrative from instability analysis to proposed solution is clear.
- **Value**: ⭐⭐⭐⭐⭐ First to scale amortized hierarchical Bayesian inference to real scientific application size (750,000 parameters).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[AAAI 2026\] HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models](../../AAAI2026/image_generation/hierarchicalprune_position-aware_compression_for_large-scale_diffusion_models.md)
- [\[ICLR 2026\] Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/image_generation/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](../../NeurIPS2025/image_generation/large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)

</div>

<!-- RELATED:END -->
