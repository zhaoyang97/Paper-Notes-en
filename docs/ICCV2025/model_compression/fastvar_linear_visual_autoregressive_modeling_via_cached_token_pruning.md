---
title: >-
  [Paper Note] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning
description: >-
  [ICCV 2025][Model Compression][Visual Autoregressive] FastVAR proposes a training-free post-hoc acceleration method for VAR models. Grounded in the observation that large-scale steps primarily model high-frequency textur…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Visual Autoregressive"
  - "next-scale prediction"
  - "token pruning"
  - "high-resolution generation"
  - "training-free acceleration"
  - "cached token restoration"
date: 2026-05-08
content_hash: b56a5aba2f353e90
---

# FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning

**Conference**: ICCV 2025
**arXiv**: [2503.23367](https://arxiv.org/abs/2503.23367)  
**Code**: [https://github.com/csguoh/FastVAR](https://github.com/csguoh/FastVAR)  
**Area**: Model Acceleration / Visual Autoregressive / Token Pruning
**Keywords**: Visual Autoregressive, next-scale prediction, token pruning, high-resolution generation, training-free acceleration, cached token restoration

## TL;DR
FastVAR proposes a training-free post-hoc acceleration method for VAR models. Grounded in the observation that large-scale steps primarily model high-frequency textures and are robust to pruning, it selects pivotal tokens via frequency-guided scoring (PTS) to retain only high-frequency tokens during the forward pass, and restores pruned positions using cached token maps from earlier scales (CTR). Built on top of FlashAttention, FastVAR achieves an additional 2.7× speedup with less than 1% performance degradation, and for the first time enables 2K image generation in 1.5 seconds on a single RTX 3090 GPU.

## Background & Motivation

Visual Autoregressive (VAR) models reformulate the conventional next-token prediction paradigm as next-scale prediction, substantially reducing the number of generation steps. However, VAR suffers from a severe resolution scalability problem:

**Computational complexity grows rapidly with resolution**: Unlike next-token prediction, which processes a single token per step, VAR must process an entire token map at each step. The number of tokens grows as $O(n^2)$ (where $n$ is image resolution), and the attention layer reaches $O(n^4)$ complexity.

**Large-scale steps are the latency bottleneck**: Even with FlashAttention enabled, inference latency grows super-linearly. The final two scale steps account for 60% of total runtime.

**Scaling to high resolution is infeasible**: Existing VAR models run out of memory (OOM) when generating 2K-resolution images, hindering practical deployment.

**Existing acceleration methods are inapplicable**: Diffusion model acceleration techniques (DeepCache, ToMeSD) cannot be directly applied to VAR; parallel decoding strategies for AR models (speculative decoding) are also incompatible with the next-scale paradigm.

## Method

### Three Key Observations

**Observation 1: Large-scale steps are the bottleneck yet robust to pruning**
- Runtime profiling shows the final two scale steps account for 60% of total inference time.
- Pruning sensitivity experiments show that large-scale steps suffer far less performance degradation under the same pruning ratio compared to small-scale steps.
- Conclusion: Token pruning should be concentrated at large-scale steps.

**Observation 2: Large-scale steps primarily model high-frequency content**
- Visualization of intermediate predictions $\tilde{r}_k$ shows that small-scale steps generate the overall structure and contours ("structure-building phase"), while large-scale steps add texture details ("texture-filling phase").
- Spectral analysis reveals that low-frequency components largely converge during small-scale steps, whereas high-frequency components still exhibit significant variation at large-scale steps.
- Conclusion: Redundant low-frequency tokens can be pruned, retaining only high-frequency tokens.

**Observation 3: Tokens across scales exhibit strong cross-scale correlation**
- Attention map analysis reveals that tokens at the current scale attend not only to neighboring tokens at the same scale, but also exhibit strong sparse diagonal correlation with corresponding token positions from the previous scale.
- Conclusion: Token maps from the previous scale can be used to approximate the outputs at pruned positions, compensating for information loss.

### Key Designs

**Pivotal Token Selection (PTS)**

To address the challenge of frequency-domain token selection (FFT operates in the frequency domain and cannot easily localize the frequency characteristics of individual spatial tokens), PTS proposes the following approximation:

1. Estimate the low-frequency component: apply global average pooling to the input $x_k$ to obtain the DC component $\bar{x}_k = \text{global\_avg\_pool}(x_k)$.
2. Compute the high-frequency component: $\text{high-freq} = x_k - \bar{x}_k$.
3. Compute the pivotal score: $s_k = \|x_k - \bar{x}_k\|_2$ (L2 norm).
4. Token selection: retain the Top-$K$ tokens with the highest scores as pivotal tokens.

An additional benefit of PTS is that reducing the number of input tokens also reduces KV-Cache size, optimizing GPU memory usage and subsequent cross-scale attention computation.

**Cached Token Restoration (CTR)**

To restore the complete 2D image structure after pruning:

1. Cache the output token map of each layer at the final step of the structure-building phase (step $K-N$).
2. Upsample the cached token map via interpolation to match the current scale resolution: $y_k^{\text{cache}} = \text{interpolate}(y_{K-N}, (h_k, w_k))$.
3. Fill the interpolated cached values into the pruned positions using index set $\mathcal{I}$, restoring the complete token map.

This design exploits the strong diagonal attention correlation across scales—tokens at corresponding positions from the previous scale serve as reliable approximations of the outputs at pruned positions.

**Progressive Pruning Rate Schedule**

Larger scale steps are more robust to pruning and are therefore assigned higher pruning ratios:
- Infinity model: $\{40\%, 50\%, 100\%, 100\%\}$ (the final two steps are completely skipped and replaced by interpolation)
- HART model: $\{50\%, 75\%\}$

### Implementation Characteristics
- **Training-free**: Applied plug-and-play to pretrained VAR models; backbone-agnostic.
- **FlashAttention-compatible**: Stacked on top of FlashAttention for an additional 2.7× speedup.
- **Zero-shot high-resolution extension**: The cached restoration mechanism enables zero-shot generation of 2K images beyond the training resolution.

## Key Experimental Results

### GenEval Benchmark (1024×1024)

| Method | Type | Latency | Speedup | GenEval Overall |
|--------|------|--------:|--------:|----------------:|
| SDXL | Diffusion | 4.3s | — | 0.55 |
| SD3-medium | Diffusion | 4.4s | — | 0.62 |
| LlamaGen | AR | 37.7s | — | 0.32 |
| Show-o | AR | 50.3s | — | 0.68 |
| HART | VAR | 0.95s | 1.0× | 0.51 |
| **HART + FastVAR** | VAR | **0.63s** | **1.5×** | **0.51** |
| Infinity | VAR | 2.61s | 1.0× | 0.73 |
| **Infinity + FastVAR** | VAR | **0.95s** | **2.7×** | **0.72** |

- HART + FastVAR: 1.5× speedup with no change in GenEval score.
- Infinity + FastVAR: 2.7× speedup with only 0.01 drop in GenEval.
- Compared to LlamaGen, Infinity + FastVAR achieves 39.7× speedup alongside a 125% performance improvement.

### MJHQ30K Benchmark (FID)

| Method | Speedup | landscape FID | people FID |
|--------|--------:|--------------:|-----------:|
| HART | 1.0× | 25.43 | 30.61 |
| HART + FastVAR | 1.5× | 22.52 | 28.19 |
| Infinity | 1.0× | 24.68 | 30.27 |
| Infinity + FastVAR | 2.7× | 24.68 | 30.55 |

HART + FastVAR even reduces FID on the people category by 2.42, indicating a quality improvement.

### vs. Token Merging (ToMe)

| Method | Speedup | FID↓ | GenEval↑ |
|--------|--------:|-----:|---------:|
| ToMe (1.19×) | 1.19× | 29.07 | 0.48 |
| ToMe (1.36×) | 1.36× | 35.22 | 0.46 |
| **FastVAR (1.51×)** | **1.51×** | **28.19** | **0.51** |
| **FastVAR (1.70×)** | **1.70×** | **28.97** | **0.50** |

FastVAR maintains superior FID and GenEval scores at higher speedup ratios, demonstrating that cached token restoration outperforms token merging strategies.

### Zero-Shot 2K Resolution Generation
- Single NVIDIA RTX 3090 GPU (24 GB)
- 15 GB memory footprint
- 1.5 seconds per 2K image
- The original baseline fails with OOM

### Memory Reduction
- FlashAttention baseline: 18.9 GB
- FastVAR: 14.7 GB (22.2% reduction)

## Highlights & Insights

- **Thorough problem analysis**: The computational characteristics of VAR are systematically revealed through three complementary lenses—latency profiling, spectral analysis, and attention map analysis—each directly motivating a specific design choice.
- **Simple yet elegant method**: PTS requires only a global average pooling followed by L2 norm ranking; CTR requires only a single interpolation step and index-based filling, keeping implementation complexity minimal.
- **Bold 100% pruning rate design**: The final two scale steps can be entirely skipped and replaced by interpolation, indicating that the redundancy of large-scale steps in VAR far exceeds expectations.
- **Training-free and FlashAttention-compatible**: Completely orthogonal to model training and existing acceleration techniques, enabling stacked application.
- **First consumer-GPU 2K generation**: Advances the state from "infeasible" to "15 GB / 1.5 s", delivering significant engineering and practical value.
- **Frequency-domain token importance scoring**: Approximating low-frequency components via the DC component is simple yet effective, avoiding the additional overhead of FFT.

## Limitations & Future Work

- The applicability of a 100% pruning rate (completely skipping the final two steps) depends on the specific backbone and does not generalize to all VAR models.
- Global average pooling as a low-frequency approximation in PTS is a coarse estimate and may be insufficient for scenes with complex textures.
- The progressive pruning rate schedule is currently set manually without adaptation; different content types or resolutions may require individual tuning.
- Validation is primarily conducted on class-conditional and text-to-image generation; video VAR and other modalities remain unexplored.
- Although zero-shot 2K generation is feasible, its quality may fall short of models explicitly trained at high resolution; a direct comparison with such models is absent.
- The rationale for selecting the cache step $K-N$ is insufficiently motivated, and more optimal caching strategies may exist.

## Related Work & Insights

- **vs. CoDe**: CoDe employs model ensembling (a large model for small scales and a small model for large scales), relying on the availability of models of different sizes. FastVAR requires no additional models and is more generally applicable.
- **vs. ToMe/ToMeSD**: Token Merging merges multiple tokens into one, but struggles in VAR to compress the entire token map to a limited token count, leading to rapid performance degradation. FastVAR replaces token merging with cached restoration, leveraging the unique cross-scale structure of VAR.
- **vs. DeepCache**: DeepCache reuses low-resolution U-Net layer features; FastVAR adopts an analogous idea by caching early-scale outputs. The key distinction is that VAR's multi-scale autoregressive structure naturally provides a hierarchical caching granularity.
- **Impact on the VAR ecosystem**: FastVAR unlocks high-resolution VAR generation on consumer-grade hardware, potentially facilitating the practical deployment of VAR models in real-world applications.

## Rating
- Novelty: ⭐⭐⭐⭐ The method design logic driven by three core observations is clear and well-motivated; the cached restoration strategy fully exploits the unique structure of VAR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks (GenEval + MJHQ30K), two VAR backbones, and detailed ablations covering scale sensitivity, pruning rates, and comparison with ToMe.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent paper structure with a well-executed observe-then-design narrative that builds progressively; figures are abundant and intuitive.
- Value: ⭐⭐⭐⭐⭐ A training-free, plug-and-play acceleration scheme that enables 2K generation on consumer-grade GPUs for the first time, offering substantial practical value to the VAR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/model_compression/linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](representation_shift_unifying_token_compression_with_flashattention.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)

</div>

<!-- RELATED:END -->
