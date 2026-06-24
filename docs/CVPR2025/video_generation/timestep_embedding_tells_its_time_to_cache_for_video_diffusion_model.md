---
title: >-
  [Paper Note] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model
description: >-
  [CVPR 2025][Video Generation][Video Diffusion Models] This paper proposes TeaCache, a training-free caching acceleration method for video diffusion models. It estimates the output differences of the model between adjacent timesteps by leveraging the timestep-embedding-modulated noise inputs, calibrated via polynomial fitting to adaptively decide when to cache or reuse outputs. It achieves a $4.41\times$ speedup on Open-Sora-Plan with virtually lossless visual quality (VBench…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Diffusion Models"
  - "Cache Acceleration"
  - "Timestep Embedding"
  - "Training-free Acceleration"
  - "DiT"
date: 2026-05-08
content_hash: 302ef65e18cbd712
---

# Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2411.19108](https://arxiv.org/abs/2411.19108)  
**Code**: [https://liewfeng.github.io/TeaCache](https://liewfeng.github.io/TeaCache)  
**Area**: Image/Video Generation / Diffusion Model Acceleration  
**Keywords**: Video Diffusion Models, Cache Acceleration, Timestep Embedding, Training-free Acceleration, DiT

## TL;DR

This paper proposes TeaCache, a training-free caching acceleration method for video diffusion models. It estimates the output differences of the model between adjacent timesteps by leveraging the timestep-embedding-modulated noise inputs, calibrated via polynomial fitting to adaptively decide when to cache or reuse outputs. It achieves a $4.41\times$ speedup on Open-Sora-Plan with virtually lossless visual quality (VBench drops by only 0.07%).

## Background & Motivation

**Background**: Diffusion Transformers (DiTs) have become the core backbone network for video generation, but slow inference speed remains the primary bottleneck hindering widespread application. The sequential nature of the denoising process limits parallelization capability, and this issue worsens as model parameter scale, video resolution, and duration increase.

**Limitations of Prior Work**: Existing acceleration methods fall into two categories: (1) distillation/post-training—requiring significant additional training costs; (2) caching mechanisms (e.g., PAB, FORA)—which are training-free but employ a uniform caching strategy, caching and reusing model outputs at equally spaced timesteps. However, output differences between adjacent timesteps are not uniformly distributed (with large changes at some timesteps and negligible ones at others). Consequently, uniform caching strategies lack flexibility and fail to maximize caching efficiency.

**Key Challenge**: To determine whether the output at a certain timestep can reuse the cache, one must know the difference between the current output and the cached output—but this difference is unknown until the output is actually computed. This presents a "chicken-and-egg" problem.

**Goal**: To predict the magnitude of output differences without computing the model outputs, thereby intelligently selecting the optimal caching timing.

**Key Insight**: There is a strong correlation between model inputs and outputs. If input differences can be used to estimate output differences, caching decisions can be made proactively at virtually zero cost. The key observation is that among the three inputs of diffusion models, text embeddings remain constant, noise inputs are insensitive to timesteps, and timestep embeddings change but are independent of the input content. Only the "timestep-embedding-modulated noise input" contains both timestep and content information simultaneously, showing the strongest correlation with the output.

**Core Idea**: Use the difference in timestep-embedding-modulated noise inputs as a proxy estimator for model output differences, and then calibrate the scaling bias via polynomial fitting to achieve an adaptive, non-uniform caching strategy.

## Method

### Overall Architecture

TeaCache operates during the inference phase of DiT diffusion models. For each denoising timestep: (1) compute the relative L1 difference of the timestep-embedding-modulated noise input between the current and the previous step; (2) map this input difference to an estimate of the output difference using a pre-calibrated polynomial function; (3) when the accumulated difference exceeds a threshold $\delta$, compute a new model output and cache it, otherwise reuse the cached output. The entire process requires no additional training and serves as a plug-and-play acceleration scheme.

### Key Designs

1. **Timestep-Embedding-Modulated Input Difference Estimation**:

    - **Function**: Estimate expensive output differences using virtually zero-cost input differences.
    - **Mechanism**: In each Transformer block of a diffusion model, the timestep embedding modulates the input and output magnitudes of the self-attention layer and the FFN via AdaLN. Therefore, the "timestep-embedding-modulated noise input" (i.e., the features at the input stage of the Transformer) contains both noise content and timestep information, exhibiting the strongest correlation with the model output. The relative L1 distance is used to measure the input difference: $L1_{rel}(\mathbf{F}, t) = \|\mathbf{F}_t - \mathbf{F}_{t+1}\|_1 / \|\mathbf{F}_{t+1}\|_1$
    - **Design Motivation**: Constant text embeddings cannot reflect changes, and standalone timestep embeddings or noise inputs are individually incomplete. The modulated input shows the strongest correlation with output differences in experiments (validated across Open-Sora, Latte, and OpenSora-Plan).

2. **Polynomial Fitting for Scaling Calibration**:

    - **Function**: Bridge the scaling bias between input and output differences.
    - **Mechanism**: Although input and output differences exhibit consistent trends, there are magnitude discrepancies. A simple polynomial fitting $y = f(x) = a_0 + a_1x + a_2x^2 + \cdots + a_nx^n$ is used to map the input difference $x$ to the estimated output difference $y$. The fitting data is collected by running full inference once on 70 prompts (a one-time offline cost), and solved easily using numpy's `poly1d` function.
    - **Design Motivation**: Directly using raw input differences for decisions leads to sub-optimal timestep selection. Polynomial fitting is simple yet highly effective: a first-order fitting already improves VBench by 0.24%, and benefits saturate beyond the fourth order.

3. **Accumulated-Difference Adaptive Caching Strategy**:

    - **Function**: Adaptively decide when to refresh the cache.
    - **Mechanism**: Accumulate the calibrated difference $\sum_{t=t_a}^{t_b-1} f(L1_{rel}(\mathbf{F}, t))$ starting from the last cached timestep $t_a$. When the accumulated value exceeds the threshold $\delta$, a new output is computed at $t_b$ and cached; otherwise, the cached output is reused. A critical detail is that only residual signals (output minus input) are cached, ensuring that even when the cache is reused, the model output still updates progressively with the input.
    - **Design Motivation**: Unlike uniform caching, the adaptive strategy skips considerable computation during periods of flat output variation (e.g., the middle segment of a U-shaped curve) while preserving full computation during intense changes, significantly boosting caching efficiency.

### Loss & Training

TeaCache is entirely training-free. The polynomial fitting coefficients are calibrated once offline using 70 sampled prompts for each base model. The threshold $\delta$ controls the speed-quality trade-off: slow=0.1, fast=0.2.

## Key Experimental Results

### Main Results

**Comparison of Speedup and Quality across Three Base Models**:

| Model | Method | FLOPs (P) ↓ | Speedup ↑ | VBench ↑ | LPIPS ↓ | PSNR ↑ |
|------|------|------------|---------|---------|---------|--------|
| Latte | PAB-fast | 2.52 | 1.34× | 73.13% | 0.3903 | 17.16 |
| Latte | **TeaCache-slow** | **1.86** | **1.86×** | **77.40%** | **0.1901** | **22.09** |
| Latte | **TeaCache-fast** | **1.12** | **3.28×** | 76.69% | 0.3133 | 18.62 |
| Open-Sora | PAB-fast | 2.50 | 1.40× | 76.95% | 0.1743 | 23.58 |
| Open-Sora | **TeaCache-slow** | **2.40** | **1.55×** | **79.28%** | **0.1316** | **23.62** |
| OSP | PAB-fast | 8.35 | 1.56× | 71.81% | 0.5499 | 15.47 |
| OSP | **TeaCache-slow** | **3.13** | **4.41×** | **80.32%** | **0.2145** | **21.02** |
| OSP | **TeaCache-fast** | **2.06** | **6.83×** | 79.72% | 0.3155 | 18.95 |

### Ablation Study

| Caching Metric | Model | VBench ↑ | Description |
|---------|------|---------|------|
| Timestep embedding | Open-Sora | Lower | Does not change with content |
| **Modulated noise input** | Open-Sora | **Higher** | Contains both timestep and content information |

| Polynomial Order | VBench ↑ | LPIPS ↓ | Description |
|-----------|---------|---------|------|
| No fitting | Baseline | Baseline | Directly use input differences |
| 1st order | +0.24% | Improved | Simple linear correction |
| 4th order | Saturated | Saturated | High-order yields no extra gains |

### Key Findings

- Achieved a dramatic $4.41\times$ (slow) and $6.83\times$ (fast) speedup on Open-Sora-Plan, suggesting significant redundancy during its 150-step sampling process.
- Output difference curves vary greatly across different models: Open-Sora exhibits a U-shape, whereas Latte and OSP present an inverted L-shape, justifying the necessity of an adaptive strategy over a uniform one.
- Caching Mechanism vs. Reducing Timesteps: Reducing timesteps coarsens the $\alpha_t$ scheduling parameters, degrading generation quality, whereas caching retains the complete scheduling parameters.
- TeaCache-slow achieves identical VBench scores (77.40%) to the original overall model on Latte, realizing a lossless $1.86\times$ speedup.
- Excellent Multi-GPU Scalability: TeaCache consistently outperforms PAB as the number of GPUs increases.

## Highlights & Insights

- **Highly ingenious approach to estimating output differences from inputs**: The virtually zero-cost proxy metric bypasses the paradox of "needing the output before deciding whether to compute it." This methodology is transferable to any iterative computation system.
- **Modulation effect of timestep embeddings**: Deeply analyzes how timestep embeddings modulate the magnitude of each layer's input and output through AdaLN layers. This observation serves as the critical foundation of the method's design.
- **Minimalist yet highly effective**: The polynomial fitting requires only a single line of Numpy code and is calibrated once using 70 prompts. The entire method introduces zero additional network parameters or training, showcasing textbook simplicity.

## Limitations & Future Work

- Polynomial fitting coefficients need to be calibrated separately for each base model, requiring re-fitting when changing models.
- The caching strategy assumes residual signals are a good approximation, which may introduce artifacts under extreme scenarios (e.g., sudden scene cuts).
- The threshold $\delta$ is a global constant, ignoring the influence of complex differences across different prompts on the optimal threshold.
- Layer-wise caching strategies could be explored, as output redundancy may vary across different layers.
- Synergy with CFG-aware caching (e.g., FasterCache) may further boost the acceleration ratio.

## Related Work & Insights

- **vs. PAB**: PAB sets different uniform caching intervals based on attention block types (spatial/temporal/cross), but remains a static strategy. The adaptive strategy of TeaCache comprehensively outperforms PAB across all models.
- **vs. $\Delta$-DiT**: $\Delta$-DiT caches residuals between attention layers, yielding limited speedup (only $1.02\times$). TeaCache achieves speedups that are 1–2 orders of magnitude higher.
- **vs. DeepCache/FORA**: These methods are designed for UNets or specific layers, whereas TeaCache operates directly at the entire DiT model level, offering superior versatility.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The approach of estimating output differences from model inputs is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Fully validated across three base models, multiple resolutions/frame settings, and diverse ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-rationalized motivation with comprehensive analysis and visualizations.
- Value: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, massive speedup, with exceptionally high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](../../ICML2025/video_generation/ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)
- [\[CVPR 2025\] Improved Video VAE for Latent Video Diffusion Model](improved_video_vae_for_latent_video_diffusion_model.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)
- [\[CVPR 2025\] One-Minute Video Generation with Test-Time Training](one-minute_video_generation_with_test-time_training.md)

</div>

<!-- RELATED:END -->
