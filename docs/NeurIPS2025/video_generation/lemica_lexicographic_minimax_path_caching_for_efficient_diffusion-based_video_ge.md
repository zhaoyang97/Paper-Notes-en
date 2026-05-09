---
title: >-
  [Paper Note] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation
description: >-
  [NeurIPS 2025][Video Generation][diffusion acceleration] This paper proposes LeMiCa, a training-free acceleration framework for diffusion-based video generation that formulates cache scheduling as a lexicographic minimax path optimization problem on a directed acyclic graph (DAG), achieving simultaneous gains in speed and quality (2.9× speedup on Latte; LPIPS as low as 0.05 on Open-Sora) via global error control.
tags:
  - NeurIPS 2025
  - Video Generation
  - diffusion acceleration
  - caching
  - DAG optimization
  - lexicographic minimax
date: 2026-05-08
content_hash: e1854e1307aa4302
---

# LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation

**Conference**: NeurIPS 2025  
**arXiv**: [2511.00090](https://arxiv.org/abs/2511.00090)  
**Code**: [GitHub](https://github.com/UnicomAI/LeMiCa)  
**Area**: Video Generation  
**Keywords**: diffusion acceleration, video generation, caching, DAG optimization, lexicographic minimax

## TL;DR

This paper proposes LeMiCa, a training-free acceleration framework for diffusion-based video generation that formulates cache scheduling as a lexicographic minimax path optimization problem on a directed acyclic graph (DAG), achieving simultaneous gains in speed and quality (2.9× speedup on Latte; LPIPS as low as 0.05 on Open-Sora) via global error control.

## Background & Motivation

- **High inference cost of diffusion models**: Although DiT architectures substantially improve video quality, multi-step denoising incurs heavy computation and high latency, constraining interactive applications.
- **Limitations of prior Work — local greedy strategies**: State-of-the-art methods such as TeaCache decide whether to cache based on local relative differences between adjacent timestep outputs ($L1_{rel}$) and a fixed threshold. However, the diffusion denoising process exhibits **temporal heterogeneity** — early steps shape global structure while later steps refine fine details — causing semantic inconsistency under a uniform threshold.
- **Global degradation from accumulated local errors**: Minimizing adjacent-step differences (local-greedy error) ignores the cumulative effect of small errors along the denoising trajectory, ultimately degrading both video quality and content consistency.
- **Prior acceleration methods require training or engineering effort**: Distillation, pruning, and quantization demand large datasets and retraining; caching methods hold a natural advantage as training-free alternatives.

## Method

### Revisiting Caching in Diffusion Sampling

**Local-greedy error** (prior methods): Decisions are made based on the relative L1 distance between adjacent timestep outputs:

$$L1_{rel}(O, t) = \frac{\|O_t - O_{t+1}\|_1}{\|O_{t+1}\|_1}$$

**Global result-aware error** (proposed): Measures the impact of each cache segment on the final output:

$$L1_{glob}(i \to j) = \frac{1}{N} \|x_0^{\text{cache}(i \to j)} - x_0^{\text{original}}\|_1$$

Key findings: ① Global error propagation is non-uniform and time-dependent — downstream errors caused by early caching are amplified exponentially; ② The temporal position of a cache segment matters more than its length.

### Lexicographic Minimax Path Caching (LeMiCa)

**Graph construction**: A DAG is constructed where each edge represents a candidate cache segment with an edge weight equal to the global result-aware error. A maximum skip length is imposed to control complexity. The static graph is built offline using multiple prompts and random seeds (70 prompts × 10 seeds, averaged).

**Graph optimization**: Given a fixed inference budget $B$ (number of model forward passes), the method seeks the optimal path from source to sink under the lexicographic minimax criterion:

$$\min_{P \in \mathcal{P}_{s \to t}^{(B)}} \text{LexMax}\big(\text{sort\_desc}(\{w(e) \mid e \in P_{\text{cache}}\})\big)$$

That is, among all feasible paths, the approach first minimizes the maximum cache error; ties are broken by the second-largest error, and so on. This is more robust than shortest-path strategies because cache errors do not accumulate independently — extreme errors cause disproportionate damage to the final output.

**Why not shortest path**: Early caching errors are exponentially amplified during denoising; conventional shortest-path minimizes total accumulated cost without bounding per-step peak errors. Lexicographic minimax explicitly constrains worst-case degradation.

### Key Properties

- **Training-free**: Cache schedules are precomputed offline with no runtime overhead.
- **Model-agnostic**: Applicable to diverse architectures including Open-Sora, Latte, and CogVideoX.
- **Robust static cache schedule**: Only 1–20 samples are needed to construct a high-quality DAG.

## Key Experimental Results

### Table 1: Inference Efficiency vs. Visual Quality (Single GPU)

| Method | Model | Speedup↑ | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| TeaCache-slow | Open-Sora | 1.50× | 79.20% | 0.134 | 0.837 | 23.50 |
| **LeMiCa-slow** | Open-Sora | **1.52×** | **79.26%** | **0.050** | **0.923** | **31.32** |
| TeaCache-fast | Open-Sora | 2.10× | 78.24% | 0.252 | 0.743 | 19.03 |
| **LeMiCa-fast** | Open-Sora | **2.44×** | **78.34%** | **0.187** | **0.798** | **21.76** |
| TeaCache-slow | Latte | 1.65× | 77.40% | 0.195 | 0.775 | 21.52 |
| **LeMiCa-slow** | Latte | **1.69×** | **77.45%** | **0.091** | **0.865** | **27.65** |
| TeaCache-fast | Latte | 2.60× | 76.09% | 0.318 | 0.674 | 18.04 |
| **LeMiCa-fast** | Latte | **2.93×** | **76.75%** | **0.273** | **0.700** | **19.43** |
| TeaCache-slow | CogVideoX | 1.70× | 76.79% | 0.053 | 0.928 | 31.07 |
| **LeMiCa-slow** | CogVideoX | **1.72×** | **76.89%** | **0.023** | **0.958** | **35.93** |

LeMiCa consistently outperforms TeaCache across all models. LeMiCa-slow reduces LPIPS from 0.134 to 0.050 on Open-Sora (2.7× improvement) and achieves LPIPS of only 0.023 on CogVideoX. LeMiCa-fast reaches 2.93× speedup on Latte (up from TeaCache's 2.60×).

### Table 2: Number of Samples Required for DAG Construction

| # Samples | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 78.58 | 0.164 | 0.838 | 24.51 |
| 5 | 78.70 | 0.161 | 0.843 | 24.57 |
| 10 | 78.95 | 0.158 | 0.844 | 24.56 |
| 350 | 79.27 | 0.143 | 0.851 | 24.67 |

Strong results are obtained with as few as 1 sample (PSNR 24.51 vs. 24.67 with 350 samples), demonstrating the robustness of the static cache schedule.

### Table 3: Ablation on Path Strategy

| Path Strategy | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|:---|:---:|:---:|:---:|:---:|
| Shortest Path | 76.04 | 0.203 | 0.809 | 22.90 |
| **Minimax Path** | **79.27** | **0.143** | **0.851** | **24.67** |

The minimax path comprehensively outperforms the shortest path, validating the core argument that controlling peak error is more important than minimizing accumulated error.

### Trajectory Robustness

Varying the scale parameter of the sampling schedule (0.5–1.5), LeMiCa consistently achieves lower LPIPS than TeaCache across all trajectories, demonstrating effectiveness under diverse denoising paths.

## Highlights & Insights

- **Novel theoretical perspective**: Formalizing cache scheduling as a lexicographic minimax path problem on a DAG introduces a new optimization paradigm for diffusion acceleration.
- **Global error control**: The global result-aware error eliminates the influence of temporal heterogeneity, while lexicographic minimax explicitly constrains worst-case degradation.
- **Minimal perceptual degradation**: LPIPS of 0.05 on Open-Sora and 0.023 on CogVideoX represent near-lossless acceleration.
- **Zero additional runtime overhead**: Cache schedules are fully precomputed offline and applied via direct lookup at inference time.

## Limitations & Future Work

- **Offline DAG construction cost**: Building the graph requires full denoising trajectories evaluated across multiple prompts, incurring a non-trivial one-time computational cost.
- **Fixed sampling path assumption**: The assumption that denoising trajectories are stable for a trained model may not hold under certain adaptive schedulers.
- **Evaluated only on text-to-video**: The method has not been validated on image-to-video generation, video editing, or other related tasks.
- **Maximum skip length requires manual tuning**: The setting is heuristic-based, and no automatic determination procedure is provided.

## Related Work & Insights

- **Diffusion acceleration** (DDIM [Song+ 2020], consistency distillation [Song+ 2023], adversarial distillation [Sauer+ 2024]): These methods require training; LeMiCa is training-free.
- **Diffusion caching** (DeepCache [Ma+ 2023], PAB [Zhao+ 2024], TeaCache [Liu+ 2024]): Based on local greedy strategies; LeMiCa proposes a global optimization alternative.
- **Δ-DiT [Chen+ 2024], T-GATE [Zhang+ 2024]**: Image generation acceleration methods with limited effectiveness when extended to video.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Lexicographic minimax path optimization represents a genuinely new perspective in the diffusion caching literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three models, multiple ablations, sample efficiency analysis, and trajectory robustness evaluation.
- Writing Quality: ⭐⭐⭐⭐ — The analysis of local vs. global error is thorough and the figures are clear.
- Value: ⭐⭐⭐⭐⭐ — Provides a strong training-free acceleration baseline for diffusion-based video generation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[ICLR 2026\] PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](../../ICLR2026/video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)
- [\[NeurIPS 2025\] Training-Free Efficient Video Generation via Dynamic Token Carving](training-free_efficient_video_generation_via_dynamic_token_carving.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)

<!-- RELATED:END -->
