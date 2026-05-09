---
title: >-
  [Paper Note] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache
description: >-
  [CVPR 2026][Image Generation][Diffusion Model Acceleration] Diffusion model sampling acceleration is formulated as a global path planning problem. By constructing a Path-Aware Cost Tensor (PACT) to quantify the path dependency of skipping errors and using dynamic programming to select the optimal sequence of key steps, DPCache achieves a 4.87× speedup on FLUX, surpassing the full-step baseline by +0.028 ImageReward.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Model Acceleration
  - Caching
  - Dynamic Programming
  - Path Planning
  - Training-Free
date: 2026-05-08
content_hash: fa59f61f1f1f5d8a
---

# Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache

**Conference**: CVPR 2026  
**arXiv**: [2602.22654](https://arxiv.org/abs/2602.22654)  
**Code**: [GitHub](https://github.com/argsss/DPCache)  
**Area**: Diffusion Model Acceleration  
**Keywords**: Diffusion Model Acceleration, Caching, Dynamic Programming, Path Planning, Training-Free  

## TL;DR

Diffusion model sampling acceleration is formulated as a global path planning problem. By constructing a Path-Aware Cost Tensor (PACT) to quantify the path dependency of skipping errors and using dynamic programming to select the optimal sequence of key steps, DPCache achieves a 4.87× speedup on FLUX, surpassing the full-step baseline by +0.028 ImageReward.

## Background & Motivation

Inference overhead for diffusion models (especially DiT architectures) is high. Training-free caching methods are favored but suffer from fundamental flaws:
- **Fixed Scheduling** (e.g., DeepCache): Uniform skipping patterns ignore local feature dynamics, leading to large deviations in critical transition regions.
- **Local Adaptive Scheduling** (e.g., TeaCache/AdaCache): Greedy, short-sighted decisions may skip crucial steps, causing irreversible drift and error accumulation.
- Both strategies overlook the global structure of the denoising trajectory.

**Root Cause**: Optimal acceleration requires a global perspective—considering the entire denoising trajectory and planning sampling steps holistically.

## Method

### Overall Architecture

DPCache consists of three stages: (1) Calibration phase: Run the full denoising process on a few samples to construct a 3D Path-Aware Cost Tensor; (2) Optimal schedule selection: Use dynamic programming to select the optimal key step sequence given a step budget $K$; (3) Inference phase: Execute full computation only at key steps and use cached features to predict remaining steps.

### Key Designs

1. **Path-Aware Cost Tensor (PACT)**: Traditional 2D cost matrices only encode the skipping error between two steps, but cached prediction features depend on the previously cached state (path dependency). PACT is a 3D tensor $\mathcal{C} \in \mathbb{R}^{(T+1)\times(T+1)\times(T+1)}$, where each element $\mathcal{C}[i,j,k]$ ($i>j>k$) represents the cumulative prediction error from step $j$ to step $k$, conditioned on the predecessor key step being $i$:
$$\mathcal{C}[i,j,k] = \sum_{\tau=k}^{j-1} \|\mathbf{h}_\tau^L - \mathbf{h}_{pred,\tau}^L(i,j)\|_1$$
Using cumulative error instead of single-point error prevents large strides that seem locally optimal but remain globally unstable. Only the last layer features are used for construction to save memory. A sentinel step $t=0$ is introduced for unified boundary handling.

2. **Dynamic Programming for Optimal Schedule Selection**: Given the cost tensor and target step count $K$, the total path cost is minimized:
$$\arg\min_{\mathcal{T}} \sum_{m=2}^{K} \mathcal{C}[t_{m-1}, t_m, t_{m+1}]$$
A DP table $D[m,k]$ (minimum cumulative cost to reach step $k$ using $m$ key steps) and a path table $P[m,k]$ (for backtracking) are maintained. The recurrence relation is:
$$D[m,k] = \min_{j>k} D[m-1,j] + \mathcal{C}[P[m-1,j], j, k]$$
The first $M=3$ steps are fixed as mandatory to preserve early denoising dynamics. The time complexity is $O(KT^2)$, which is negligible when $K<T=50$.

3. **Generalization via Calibration Set**: Denoising trajectory shapes depend primarily on the diffusion model rather than the generated content. Thus, a universal optimal schedule can be obtained using only ~10 calibration samples, eliminating the need for per-input computation. During inference, key steps execute full forward passes and cache features, while other steps are predicted via Taylor expansion:
$$\mathcal{F}_{pred,m}^l(\mathbf{x}_{t-k}) = \mathbf{h}_t^l + \sum_{i=1}^m \frac{\Delta^i \mathbf{h}_t^l}{i! \cdot \mathcal{N}^i}(-k)^i$$

### Loss & Training

- Completely training-free method; no fine-tuning or distillation required.
- The calibration phase uses ~10 samples, and the prediction order is set to 2nd-order Taylor expansion.
- Decoupled from specific predictors—seamlessly integrates with future improved prediction methods.

## Key Experimental Results

### Main Results (FLUX.1-dev, DrawBench)

| Method | Gain | ImageReward↑ | CLIP Score↑ | PSNR↑ | SSIM↑ |
|------|--------|-------------|-------------|-------|-------|
| Full 50-step Baseline | 1.00× | 0.979 | 17.40 | — | — |
| TeaCache | 3.42× | 0.934 | 17.17 | 16.31 | 0.6812 |
| TaylorSeer | 3.51× | 0.939 | 17.31 | 16.95 | 0.6922 |
| SpeCa | 3.62× | 0.975 | 17.27 | 18.35 | 0.6773 |
| **Ours (K=13)** | **3.54×** | **1.007** | **17.34** | **21.65** | **0.8106** |
| **Ours (K=9)** | **4.87×** | **0.958** | **17.33** | **18.77** | **0.7117** |

Ours (K=13) **exceeds the full-step baseline** by +0.028 ImageReward at a 3.54× speedup!

### Ablation Study (HunyuanVideo, VBench)

| Method | Gain | VBench↑ | PSNR↑ | SSIM↑ |
|------|--------|---------|-------|-------|
| Full 50-step | 1.00× | 80.93 | — | — |
| TaylorSeer | 3.87× | 80.33 | 18.53 | 0.6025 |
| SpeCa | 4.05× | 80.26 | 20.09 | 0.6370 |
| **Ours (K=11)** | **4.05×** | **80.35** | **23.11** | **0.7462** |

The method significantly outperforms others in video generation, with particularly notable improvements in PSNR.

### Key Findings

- The core advantage of global optimal scheduling lies in **trajectory fidelity** (PSNR/SSIM/LPIPS leading by a large margin), indicating that path planning keeps the sampling trajectory closer to the original denoising path.
- Advantages become more pronounced at higher acceleration ratios—surpassing the baseline at 3.54× and remaining far ahead of others at 4.87×.
- Modeling path dependency (3D PACT vs. 2D cost matrix) is critical; 2D matrices fail to capture the influence of predecessor steps on current skipping errors.
- 10 calibration samples are sufficient, confirming that denoising trajectory structures are content-independent.

## Highlights & Insights

- Modeling sampling acceleration as a path planning problem is highly novel—dynamic programming is perfectly suited for this scenario.
- The 3D PACT design accurately captures path dependency, representing a fundamental advancement over existing 2D analyses.
- Resulting in **better-than-baseline performance** on FLUX after acceleration suggests that skipping certain steps may not only preserve quality but potentially enhance it by preventing cumulative error.
- The design's decoupling from the predictor makes it a universal acceleration framework.

## Limitations & Future Work

- The calibration phase requires a full diffusion run once, adding to preprocessing time.
- Memory overhead for PACT construction is $O(T^3)$, which may become a bottleneck for large step counts $T$.
- Currently assumes all samples share the same optimal schedule, which may be suboptimal for out-of-distribution samples.
- Exploration of integration with adaptive scheduling is possible—PACT providing the global framework while local fine-tuning adapts to specific samples.

## Related Work & Insights

- DeepCache discovered cross-step similarity in high-level features; DPCache optimizes when to cache versus compute based on this.
- TaylorSeer’s Taylor expansion predictor is complementary to DPCache’s path planning framework.
- Unlike step-reduction methods (DDIM/DPM-Solver/Distillation), DPCache belongs to the per-step computation optimization category.
- The path planning perspective could inspire computational optimization in other sequential decision-making problems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of path planning, PACT, and DP is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers DiT, FLUX, and HunyuanVideo across image and video domains.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous formalization, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Training-free, universal, and high-performance, offering immense practical value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)

<!-- RELATED:END -->
