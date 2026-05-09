---
title: >-
  [Paper Note] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration
description: >-
  [Image Generation] This paper proposes TAP, a framework that uses a first-layer probe to adaptively select the optimal predictor (from a Taylor expansion family) for each token at each step, enabling training-free diffusion model acceleration with a 6.24× speedup on FLUX.1-dev without perceptible quality degradation.
tags:
  - Image Generation
date: 2026-05-08
content_hash: e91bd6c3bcb1499b
---

# TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration

## Basic Information

- **Conference**: CVPR 2026
- **arXiv**: [2603.03792](https://arxiv.org/abs/2603.03792)
- **Code**: Not released
- **Area**: Image Generation / Diffusion Model Acceleration
- **Keywords**: Diffusion Acceleration, Token-Adaptive, Training-Free, Feature Caching, Taylor Predictor

## TL;DR

This paper proposes TAP, a framework that uses a first-layer probe to adaptively select the optimal predictor (from a Taylor expansion family) for each token at each step, enabling training-free diffusion model acceleration with a 6.24× speedup on FLUX.1-dev without perceptible quality degradation.

## Background & Motivation

Diffusion models have achieved state-of-the-art results in image and video generation, but slow inference remains a core bottleneck—each denoising step requires a full forward pass. Existing acceleration methods fall into two categories:

**Reducing sampling steps**: High-order ODE solvers such as DDIM and DPM-Solver reduce step counts, but quality degrades significantly under aggressive acceleration.

**Reducing per-step computation**: Feature caching methods (DeepCache, Δ-DiT, TeaCache, ToCa) exploit temporal redundancy to reuse intermediate features; prediction methods (TaylorSeer, FreqCa, SpeCa) forecast future features.

**Key limitations of prior work**: All previous methods apply a **single global prediction strategy** uniformly across all tokens and all timesteps, overlooking the fact that different tokens exhibit vastly different temporal evolution characteristics:

- **Smooth background tokens**: Evolve slowly; low-order predictors suffice.
- **Edge/moving object tokens**: Exhibit rapid changes; high-order or alternative predictors are required.
- A globally uniform predictor leads to error accumulation and severe quality degradation under aggressive acceleration ratios.

Furthermore, existing adaptive methods (TeaCache, SpeCa) rely on **manually tuned thresholds**, limiting robustness.

## Method

### Overall Architecture

The core idea of TAP (Token-Adaptive Predictor) is to **independently select** the optimal predictor for each token at every sampling step. The overall pipeline consists of three stages:

**Stage 1: Compute and Cache**

The first step of every $N$-step window executes a full model forward pass and caches two key quantities:

$$\mathbf{h}_t = \text{Modulate}(\text{Norm}_1(\mathbf{x}_t), \mathbf{s}_t, \mathbf{g}_t)$$

$$\mathbf{r}_t = f_\theta(\mathbf{x}_t, t) - \mathbf{x}_t$$

Here $\mathbf{h}_t$ is the modulated first-layer input (used for subsequent probe evaluation), and $\mathbf{r}_t$ is the residual between model input and output (used for prediction). Critically, only the **first-layer input** and the **global residual** are cached, requiring $O(1)$ storage independent of model depth—whereas methods such as TaylorSeer and ToCa require $O(L)$ layer-wise caching.

**Stage 2: Taylor Predictor Family**

A compact set of candidate predictors is constructed by varying two dimensions:

- **Taylor expansion order** $m \in [O_l, O_r]$: Lower-order predictors are more robust to abrupt dynamics; higher-order predictors are more accurate for smooth dynamics.
- **Prediction distance** $k_p \in [k - \lambda, k]$: Different tokens have different Taylor convergence radii; high-order expansions beyond the convergence radius diverge.

The prediction formula is:

$$\mathcal{F}_{\text{pred}}(\mathbf{x}_{t-k}; m, k_p) = \sum_{i=0}^{m} \frac{\Delta^i \mathcal{F}(\mathbf{x}_t)}{i! \cdot N^i} (-k_p)^i$$

where $\Delta^i \mathcal{F}$ denotes the $i$-th order finite difference. With default settings $M=3$ (orders 0, 1, 2), $\lambda=4$, and $\delta=1$, a total of $\lfloor(4+1)/1\rfloor \times 3 = 15$ candidate predictors are generated.

**Stage 3: Probe-then-Select**

The core innovation: the **first-layer modulated input** is used as a proxy indicator of predictor quality. At each step, only a single first-layer forward pass (negligible cost) is required, after which all candidate predictors are evaluated in parallel.

For each token $(b, n)$, the proxy loss for each predictor $p$ is computed as:

$$\mathcal{L}_p^{b,n} = d(\widehat{\mathbf{h}}_{t,p}^{b,n}, \mathbf{h}_t^{b,n})$$

where $d(\cdot, \cdot)$ denotes cosine distance. The predictor minimizing the proxy loss is selected:

$$p^{\star, b, n} = \arg\min_{p \in \mathcal{P}} \mathcal{L}_p^{b,n}$$

The final output is then assembled using the residual predicted by the selected predictor:

$$\widehat{f}_\theta(\mathbf{x}_t, t) = \mathbf{x}_t + \widehat{\mathbf{r}}_t$$

### Key Designs

1. **Threshold-free design**: Selection is based on the **relative** proxy error across predictors, eliminating any need for manual hyperparameter tuning.
2. **Validity of the probe mechanism**: Input perturbations are highly correlated with output errors, so first-layer evaluation accurately reflects downstream prediction quality.
3. **Minimal overhead**: On FLUX.1-dev, TAP adds only 0.1 GB of GPU memory (~0.3% of the base model) and incurs only 0.015% extra FLOPs.
4. **Framework generality**: The candidate set is not restricted to Taylor expansions; other predictors such as FoCa and FreqCa can be integrated.

### Complexity Analysis

Standard diffusion sampling requires $T$ full forward passes. TAP performs only 1 complete computation per $N$ steps, replacing the remaining $N-1$ steps with predictions. Prediction and evaluation involve only element-wise operations and small polynomial computations. Additional memory is $O(1)$ (only the residual and first-layer input are stored), independent of model depth $L$.

## Key Experimental Results

### Main Results: Text-to-Image Generation (FLUX.1-dev)

| Method | FLOPs Speedup | ImageReward ↑ | CLIP ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------------|---------------|--------|--------|--------|---------|
| 50 steps (baseline) | 1.00× | 0.95 | 30.63 | - | - | - |
| FORA (N=7) | 6.24× | 0.80 | 30.42 | 13.43 | 0.60 | 0.55 |
| TeaCache (l=2.0) | 6.17× | 0.66 | 30.07 | 14.23 | 0.60 | 0.58 |
| TaylorSeer (N=8,O=2) | 6.24× | 0.91 | 30.62 | 14.72 | 0.61 | 0.50 |
| **TAP (N=8)** | **6.24×** | **0.99** | **31.19** | **16.11** | **0.64** | **0.44** |

At a high acceleration ratio of 6.24×, TAP achieves an ImageReward of 0.99, **surpassing the unaccelerated baseline (0.95)**, while achieving a PSNR approximately 1.4 dB higher than TaylorSeer. TeaCache degrades severely at this acceleration ratio (ImageReward of only 0.66).

### Main Results: Qwen-Image Text-to-Image

| Method | FLOPs Speedup | ImageReward ↑ | CLIP ↑ | PSNR ↑ | LPIPS ↓ |
|--------|--------------|---------------|--------|--------|---------|
| 50 steps (baseline) | 1.00× | 1.23 | 33.74 | - | - |
| FORA (N=3) | 2.94× | 0.92 | 32.25 | 14.12 | 0.50 |
| TeaCache (l=0.8) | 3.57× | 1.18 | 33.52 | 19.07 | 0.27 |
| TaylorSeer (N=4,O=2) | 3.57× | 1.18 | 33.44 | 18.02 | 0.30 |
| **TAP (N=4)** | **3.57×** | **1.23** | **33.80** | **20.13** | **0.22** |

On Qwen-Image, TAP likewise maintains baseline-level ImageReward (1.23), with a PSNR improvement of approximately 2 dB.

### Video Generation (HunyuanVideo + VBench)

| Method | FLOPs Speedup | VBench (%) ↑ |
|--------|--------------|-------------|
| 50 steps | 1.00× | 66.61 |
| FORA (N=5) | 4.98× | 63.87 |
| TeaCache (l=0.4) | 4.55× | 65.13 |
| TaylorSeer (N=6,O=2) | 4.98× | 64.89 |
| **TAP (N=6)** | **4.98×** | **65.46** |

TAP achieves the best quality–efficiency trade-off in video generation, with VBench declining by only 1.7%.

### Ablation Study

**Effect of Taylor predictor family order and distance:**

| Configuration | ImageReward ↑ |
|---------------|---------------|
| O=2 only (single global predictor) | ~0.89 |
| O=0,1,2 (multi-order) | ~0.95 |
| O=0,1,2 + λ=4 (multi-distance) | ~0.99 |
| δ=0.1 (finer granularity) | ~0.995 (marginal gain) |

Key findings from ablations:

1. **Zero-order predictor (order 0) is critical**: It is more robust to abrupt, discontinuous token dynamics and yields greater improvement than using only high-order predictors.
2. **Left-shifting the prediction window is effective**: Using an earlier expansion point avoids exceeding the Taylor convergence radius; right-shifting yields no benefit.
3. **Global predictor comparison**: ImageReward values across 30 candidate predictors range from 0.86 to 0.92—no single predictor is universally optimal—whereas TAP consistently surpasses any individual predictor through adaptive selection.
4. **Robustness to hyperparameters**: The default setting of $\delta=1$ is sufficient; finer granularity yields negligible gains.

## Highlights & Insights

- **Token-level adaptivity**: TAP is the first diffusion acceleration method to achieve per-token, per-step predictor selection, substantially outperforming globally uniform strategies.
- **Threshold-free design**: Selection via relative proxy error completely eliminates the need for manual hyperparameter tuning.
- **Negligible overhead**: Only 0.3% additional memory and 0.015% extra FLOPs, with virtually zero impact on practical inference latency.
- **Strong architectural generality**: Compatible with diverse architectures including FLUX.1, Qwen-Image, and HunyuanVideo, as well as distilled models.
- **Quality can exceed baseline**: Under certain settings, ImageReward and CLIP scores are higher than those of the unaccelerated model.
- **$O(1)$ storage**: Only the residual and first-layer input are cached, independent of model depth.

## Limitations & Future Work

- The predictor family is primarily based on Taylor expansions, offering limited coverage of highly nonlinear token dynamics.
- The probe relies on first-layer outputs; it may fail when the correlation between first-layer features and deeper-layer features weakens.
- No experiments combining TAP with orthogonal acceleration techniques such as knowledge distillation or model pruning are presented.
- Video generation experiments are validated only on HunyuanVideo.
- Code has not been released, making reproducibility currently unverifiable.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The problem motivation is clear (token-level heterogeneity), the method is elegantly designed (probe-then-select), and the approach requires no training with minimal overhead. Experiments cover multiple models and tasks with thorough ablations. The core contribution—per-token adaptive predictor selection—represents a meaningful advance in diffusion acceleration. One star is deducted because the candidate predictors remain confined to the Taylor family and the approach lacks theoretical guarantees (i.e., why should the first-layer probe necessarily serve as a reliable proxy?).

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](../../ICCV2025/image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](../../ICCV2025/image_generation/panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](../../ICCV2025/image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)

<!-- RELATED:END -->
