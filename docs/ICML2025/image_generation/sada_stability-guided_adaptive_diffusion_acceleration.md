---
title: >-
  [Paper Note] SADA: Stability-guided Adaptive Diffusion Acceleration
description: >-
  [ICML2025][Image Generation][Diffusion model acceleration] Proposes a Stability Criterion based on the second-order difference of ODE trajectories to uniformly control step-wise and token-wise sparsity decisions, achieving $\ge 1.8\times$ acceleration with $\text{LPIPS} \le 0.10$ and $\text{FID} \le 4.5$ on SD-2/SDXL/Flux, which significantly outperforms DeepCache and AdaptiveDiffusion.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "ODE solvers"
  - "stability criterion"
  - "adaptive sparsity"
  - "training-free acceleration"
  - "token pruning"
  - "feature caching"
date: 2026-05-08
content_hash: cf04e0a35f1a734f
---

# SADA: Stability-guided Adaptive Diffusion Acceleration

**Conference**: ICML2025  
**arXiv**: [2507.17135](https://arxiv.org/abs/2507.17135)  
**Code**: [GitHub](https://github.com/Ting-Justin-Jiang/sada-icml)  
**Area**: Diffusion Acceleration  
**Keywords**: Diffusion model acceleration, ODE solvers, stability criterion, adaptive sparsity, training-free acceleration, token pruning, feature caching

## TL;DR

Proposes a Stability Criterion based on the second-order difference of ODE trajectories to uniformly control step-wise and token-wise sparsity decisions, achieving $\ge 1.8\times$ acceleration with $\text{LPIPS} \le 0.10$ and $\text{FID} \le 4.5$ on SD-2/SDXL/Flux, which significantly outperforms DeepCache and AdaptiveDiffusion.

## Background & Motivation

Diffusion models have achieved remarkable success in image/video/audio generation, but their inference efficiency is constrained by two major bottlenecks:

**Iterative denoising process**: Tens of sampling steps are required, with each step demanding a full forward pass.

**Quadratic complexity of attention computation**: The computational cost of self-attention is prohibitive at high resolutions.

Existing training-free acceleration methods mainly fall into two categories:

- **Reducing inference steps**: High-order ODE solvers such as DDIM and DPM-Solver.
- **Reducing single-step computation**: DeepCache (step-wise caching) and Token Merging/Pruning (token-wise).

However, these two categories of methods exhibit a prominent **fidelity gap** because:

- **(a)** Fixed or pre-searched sparsity patterns cannot adapt to the distinct denoising trajectories of different prompts.
- **(b)** These methods fail to leverage information from the underlying ODE formulation and its numerical solvers.

SADA is proposed to address these two challenges.

## Method

### Core Idea: Unified Control via Stability Criterion

SADA models the diffusion acceleration problem as a **stability prediction problem**, where the core is leveraging exact gradient information along the ODE trajectory $y_t = \frac{dx_t}{dt}$ to measure the local dynamic stability of the denoising process.

### 1. Stability Criterion

The second-order difference of the ODE trajectory $\Delta^{(2)} y_t$ is defined as the stability metric. At time step $t$, if the following condition is met:

$$
(x_{t-1} - \hat{x}_{t-1}) \cdot \Delta^{(2)} y_t < 0
$$

then this step is determined to be **stable** and can be accelerated. Here, $\hat{x}_{t-1}$ is the third-order extrapolation estimate.

- When the criterion returns **True** (stable) $\rightarrow$ Execute **step-wise / multistep-wise cache-assisted pruning**.
- When the criterion returns **False** (unstable) $\rightarrow$ Execute **token-wise cache-assisted pruning**.

### 2. Step-wise Cache-Assisted Pruning

Two approximation schemes are provided:

**(a) Step-wise Approximation (Adams-Moulton Method)**: Extrapolate along the ODE trajectory using the third-order Adams-Moulton method:

$$
\hat{x}_{t-1} = x_t - \frac{5\Delta t}{6} y_t - \frac{5\Delta t}{6} y_{t+1} + \frac{2\Delta t}{3} y_{t+2}
$$

The local truncation error is $\mathcal{O}(\Delta t^2)$, which has a lower mean error and smaller standard deviation compared to simple finite differences.

**(b) Multistep-wise Approximation (Lagrange Interpolation)**: When the trajectory enters the stable region, uniform step-wise pruning and Lagrange interpolation are adopted. By caching $x_0^t$ every $k$ steps, the skipped steps are reconstructed using interpolation:

$$
\hat{x}_0^t = \sum_{i \in I} \left( \prod_{j \in I \setminus \{i\}} \frac{t - t_j}{t_i - t_j} \right) x_0^{t_i}
$$

The interpolation error is $\mathcal{O}(h^{k+1})$.

### 3. Token-wise Cache-Assisted Pruning

When the step-wise stability criterion returns False, stability is evaluated at a finer token level:

- Tokens are divided into an **unstable group** $\mathcal{I}_{\text{fix}}$ (requiring full computation) and a **stable group** $\mathcal{I}_{\text{reduce}}$ (approximated via caching).
- Attention computation is only performed on tokens in $\mathcal{I}_{\text{fix}}$.
- Tokens in $\mathcal{I}_{\text{reduce}}$ are replaced by the cached representation $\mathcal{C}_l$ from the previous step.
- The cache is incrementally updated after each computation.

### 4. Theoretical Guarantees

- **Theorem 3.1**: Lagrange extrapolation error $R_k(h) = \mathcal{O}(h^k)$
- **Theorem 3.6**: Reconstruction error $\|\hat{x}_0^t - x_0^t\| = \mathcal{O}(\Delta t) + \mathcal{O}(\Delta x_t)$
- **Theorem 3.2-3.3**: Proofs for the continuity of sampling trajectories and the consistency of denoisers.

## Key Experimental Results

### Main Results: MS-COCO 2017 Quantitative Results

| Model | Solver | Method | PSNR↑ | LPIPS↓ | FID↓ | Speedup |
|------|--------|------|-------|--------|------|--------|
| SD-2 | DPM++ | DeepCache | 17.70 | 0.271 | 7.83 | 1.43× |
| SD-2 | DPM++ | AdaptiveDiffusion | 24.30 | 0.100 | 4.35 | 1.45× |
| SD-2 | DPM++ | **SADA** | **26.34** | **0.094** | **4.02** | **1.80×** |
| SDXL | DPM++ | DeepCache | 21.30 | 0.255 | 8.48 | 1.74× |
| SDXL | DPM++ | AdaptiveDiffusion | 26.10 | 0.125 | 4.59 | 1.65× |
| SDXL | DPM++ | **SADA** | **29.36** | **0.084** | **3.51** | **1.86×** |
| Flux | Flow | TeaCache | 19.14 | 0.216 | 4.89 | 2.00× |
| Flux | Flow | **SADA** | **29.44** | **0.060** | **1.95** | **2.02×** |

### Ablation Study: Few-Step Sampling

| Model | Solver | Steps | PSNR↑ | LPIPS↓ | FID↓ | Speedup |
|------|--------|------|-------|--------|------|--------|
| SD-2 | DPM++ | 50 | 26.34 | 0.094 | 4.02 | 1.80× |
| SD-2 | DPM++ | 25 | 28.15 | 0.073 | 3.13 | 1.48× |
| SD-2 | DPM++ | 15 | 29.84 | 0.072 | 3.05 | 1.24× |
| SDXL | DPM++ | 50 | 29.36 | 0.084 | 3.51 | 1.86× |
| SDXL | DPM++ | 25 | 30.84 | 0.073 | 2.80 | 1.52× |

Fidelity unexpectedly improves as the number of steps decreases (less error accumulation), while still providing an additional $\sim 1.25\text{-}1.5\times$ speedup.

### Cross-Modal / Cross-Task

- **MusicLDM Audio Generation**: $1.81\times$ speedup, with spectral LPIPS around only $0.01$.
- **ControlNet Controllable Generation**: $1.41\times$ speedup, plug-and-play without any modifications.

## Highlights & Insights

1. **Theoretical Innovation**: For the first time, directly bridges numerical ODE solvers with sparsity-aware architecture optimization, unifying step-wise and token-wise acceleration decisions via a stability criterion.
2. **Adaptive Allocation**: Different prompts automatically acquire distinct sparsity patterns without manual hyperparameter tuning or pre-searching.
3. **Principled Approximation**: Utilizes the Adams-Moulton method and Lagrange interpolation to provide approximation schemes with bounded errors, rather than simply reusing noise.
4. **Broad Compatibility**: Cross-architecture (UNet/DiT), cross-solver (Euler/DPM++), cross-modal (image/audio), and cross-task (ControlNet).
5. **Plug-and-Play**: Training-free with no extra hyperparameter tuning, directly serving as a plugin for the sampling process.

## Limitations & Future Work

1. **Unverified on Video Generation**: The paper does not validate performance on video diffusion models (e.g., Sora-like architectures).
2. **Limited Acceleration in Extremely Few-Step Scenarios**: At 15 steps, the speedup drops to $\sim 1.25\times$, showing diminishing marginal returns.
3. **Stability Criterion Relies on Historic Cache**: The initial steps require full computation to accumulate gradient history, introducing a cold-start overhead.
4. **Token Pruning vs. Token Merging**: The paper chooses token pruning over merging (with appendix analyses showing merging acts as a low-pass filter), though merging might be superior in certain scenarios.
5. **Single-GPU Evaluation**: Acceleration performance in multi-GPU distributed settings remains unreported.

## Related Work & Insights

- **DeepCache** (Ma et al., 2024): Caches intermediate features of UNet and reuses them at fixed intervals; SADA's adaptive strategy significantly outperforms its fixed patterns.
- **AdaptiveDiffusion** (Ye et al., 2024): Uses third-order differences to determine whether to skip steps but directly reuses noise without correction; SADA introduces ODE gradients for principled corrections.
- **TeaCache** (Liu et al., 2025): Introduces an accumulated error threshold for caching decisions; SADA reduces its FID on Flux from 4.89 to 1.95.
- **DPM-Solver Series** (Lu et al., 2022): High-order ODE solvers, to which SADA is orthogonal and complementary.
- **Token Merging** (Bolya & Hoffman, 2023): Merges similar tokens to reduce attention computation, whereas SADA opts for a pruning + cache scheme.

## Rating

- Novelty: ⭐⭐⭐⭐ — Unifying step-wise and token-wise decisions via a stability criterion is a brand-new paradigm, and the bridge between ODE solvers and architectural sparsity has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three mainstream models $\times$ two solvers $\times$ multi-step ablation + cross-modal validation, which is relatively comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous theoretical derivation, clear notation system, and intuitive figures.
- Value: ⭐⭐⭐⭐⭐ — A practical plug-and-play acceleration scheme, offering $1.8\text{-}2\times$ speedup with high fidelity, possessing direct value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models](morse_dual-sampling_for_lossless_acceleration_of_diffusion_models.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[CVPR 2025\] RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories](../../CVPR2025/image_generation/rayflow_instance-aware_diffusion_acceleration_via_adaptive_flow_trajectories.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[ICML 2025\] Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion](multidimensional_adaptive_coefficient_for_inference_trajectory_optimization_in_f.md)

</div>

<!-- RELATED:END -->
