---
title: >-
  [Paper Note] PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation
description: >-
  [ICLR 2026][Video Generation][Feature caching] This paper proposes PreciseCache — a plug-and-play acceleration framework that precisely detects and skips genuinely redundant computations in video generation. It consists…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "Feature caching"
  - "video diffusion"
  - "low-frequency difference"
  - "step-level caching"
  - "block-level caching"
date: 2026-05-08
content_hash: 974f8a1b60356f7d
---

# PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation

**Conference**: ICLR 2026
**arXiv**: [2603.00976](https://arxiv.org/abs/2603.00976)  
**Code**: None  
**Area**: Video Generation / Inference Acceleration
**Keywords**: Feature caching, video diffusion, low-frequency difference, step-level caching, block-level caching

## TL;DR

This paper proposes PreciseCache — a plug-and-play acceleration framework that precisely detects and skips genuinely redundant computations in video generation. It consists of LFCache (step-level, based on a Low-Frequency Difference (LFD) metric) and BlockCache (block-level, based on an input-output difference metric), achieving an average 2.6× speedup with negligible quality degradation on mainstream models such as Wan2.1-14B.

## Background & Motivation

**Background**: Video diffusion models (e.g., Sora, HunyuanVideo, CogVideoX, Wan2.1) continue to improve in generation quality, but inference remains extremely slow — Wan2.1-14B requires approximately 907 seconds to generate a single 720P video on 4 A800 GPUs. Feature caching is currently the dominant training-free acceleration method, which skips network inference for certain steps by reusing cached features from previous denoising steps.

**Limitations of Prior Work**:

- **Uniform caching strategies (e.g., PAB)**: Cache every $n$ steps, ignoring the varying contributions of different denoising steps to final generation quality — high-noise steps establish the structural and content information of the video (non-skippable), while low-noise steps refine high-frequency details (safely skippable).
- **Existing adaptive caching methods**: Require complex additional fitting or extensive hyperparameter tuning, and the caching criteria remain insufficiently precise.
- **Using adjacent-step prediction differences as caching indicators directly** (e.g., TeaCache): Such indicators exhibit weak correlation with final generation quality, leading to suboptimal caching strategies.

**Key Challenge**: How to design a runtime-adaptive caching criterion that can precisely distinguish between "genuinely redundant computation" and "computation critical to generation quality," so as to maximize speedup while preserving video quality?

**Goal**: This paper proposes Low-Frequency Difference (LFD) as a precise metric for step-level redundancy — grounded in the key insight that the diffusion process models low-frequency structural information during high-noise stages (important) and refines high-frequency details during low-noise stages (cacheable). LFD is shown to be highly consistent with the impact of caching on final video quality.

## Method

### Overall Architecture

PreciseCache consists of two complementary caching mechanisms:

1. **LFCache (step-level caching)**: At each denoising step, the low-frequency difference (LFD) between the current step and the last cached step is computed as the caching decision metric. If LFD is below a threshold, the step is skipped (cached features reused); otherwise, full inference is executed.
2. **BlockCache (block-level caching)**: Within steps not skipped by LFCache, the redundancy of each Transformer block is further analyzed. Only pivotal blocks — those that significantly modify input features — are retained; non-pivotal blocks are skipped.

The two mechanisms are applied in cascade: LFCache eliminates step-level redundancy, and BlockCache eliminates block-level redundancy within retained steps, achieving two-level acceleration.

### Key Designs 1: Low-Frequency Difference (LFD) Metric and Its Efficient Computation

**Definition of LFD**: The network prediction $\bm{F}_i$ is decomposed into low-frequency components $\bm{F}_i^{LF}$ and high-frequency components $\bm{F}_i^{HF}$ via Fast Fourier Transform (FFT), and the low-frequency difference between adjacent steps is defined as:

$$\Delta_i^{LF} = \| \bm{F}_i^{LF} - \bm{F}_{i+1}^{LF} \|_2$$

The low-frequency region is defined as a circular mask with radius equal to $\frac{1}{5}$ of the minimum spatial dimension. The paper empirically verifies that $\Delta_i^{LF}$ is highly consistent with the impact of cache reuse at that step on final video quality: high-noise steps exhibit large LFD (significant structural changes, non-cacheable), while low-noise steps exhibit small LFD (only high-frequency detail changes, safely cacheable).

**Efficient Estimation**: Directly computing LFD requires full inference at the current step (defeating the purpose of acceleration). A key observation is that LFD is insensitive to latent resolution. Therefore, the latent is first downsampled and a fast "trial" inference is performed to estimate LFD:

$$\widetilde{\bm{Z}}_i = \text{Downsample}(\bm{Z}_i), \quad \widetilde{\bm{F}}_i = \epsilon_\theta(\widetilde{\bm{Z}}_i, t_i)$$

The downsampling ratio is 2× in the temporal dimension and 4×4 in the spatial dimensions, making the additional overhead of trial inference negligible.

**Cumulative Error Strategy**: The cumulative LFD $\sum_{i=a}^{b} \widetilde{\Delta}_i^{LF}$ is used as the final indicator. Full inference is executed when this exceeds threshold $\delta$; otherwise, cached features are reused. The threshold is set via a relative factor: $\delta = \widetilde{\Delta}_{max}^{LF} \times \alpha$, where $\alpha = 0.5$ (Base configuration) or $0.7$ (Turbo configuration).

### Key Designs 2: BlockCache (Block-level Caching)

For timesteps not skipped by LFCache, further acceleration is achieved by analyzing the redundancy of individual Transformer blocks within the DiT. The input-output difference of each block is computed as:

$$\bm{D}_{k_i}^j = \bm{F}_{k_i}^j - \bm{F}_{k_i}^{j-1}$$

The top $c\%$ blocks with the largest differences are designated as pivotal blocks; the rest are non-pivotal. In the subsequent $L$ non-skipped steps, non-pivotal blocks estimate their output directly using the cached difference $\bm{D}_{k_i}^j$:

$$\bm{F}_{k_{i-l}}^j = \begin{cases} \mathcal{B}^j(\bm{F}_{k_{i-l}}^{j-1}, t_{k_{i-l}}), & j \in \mathcal{I}_i \text{ (pivotal blocks)} \\ \bm{F}_{k_{i-l}}^{j-1} + \bm{D}_{k_i}^j, & j \notin \mathcal{I}_i \text{ (non-pivotal blocks)} \end{cases}$$

In the Flash configuration, the cache rate is set to 40% (i.e., 60% of blocks are skipped), with $L = 3$.

### Key Designs 3: Plug-and-Play and Cross-Architecture Adaptability

PreciseCache does not modify any parameters or architecture of the base model:
- LFD relies only on FFT (a standard operation) and downsampling.
- BlockCache only requires access to the input and output of each block (via standard hooks).
- The only hyperparameters are the relative factor $\alpha$ and block cache rate $c\%$, which require minimal adjustment across models.

## Key Experimental Results

### Main Results: Efficiency and Quality Comparison on 4 Mainstream Models (4 A800 GPUs)

| Method | Model | MACs (P) ↓ | Speedup ↑ | VBench ↑ | LPIPS ↓ | PSNR ↑ |
|:-----|:-----|:----------|:---------|:---------|:--------|:-------|
| Baseline | Wan2.1-14B | 329.2 | 1× | 83.62% | - | - |
| PAB | Wan2.1-14B | 233.5 | 1.38× | 82.91% | 0.1853 | 26.18 |
| TeaCache | Wan2.1-14B | 166.3 | 1.94× | 83.24% | 0.1012 | 27.22 |
| FasterCache | Wan2.1-14B | 183.9 | 1.73× | 83.47% | 0.0741 | 28.45 |
| **Ours-base** | Wan2.1-14B | 204.5 | **1.59×** | **83.56%** | **0.0451** | **29.12** |
| **Ours-turbo** | Wan2.1-14B | 151.0 | **2.15×** | **83.52%** | **0.0633** | **28.98** |
| **Ours-flash** | Wan2.1-14B | 122.4 | **2.63×** | **83.43%** | 0.0812 | 28.76 |
| Baseline | HunyuanVideo | 14.92 | 1× | 80.66% | - | - |
| TeaCache | HunyuanVideo | 8.93 | 1.64× | 80.51% | 0.0911 | 28.15 |
| **Ours-turbo** | HunyuanVideo | 7.49 | **1.95×** | **80.49%** | 0.0884 | 29.06 |
| **Ours-flash** | HunyuanVideo | 6.04 | **2.44×** | 80.02% | 0.0902 | 28.64 |

Key findings: PreciseCache-flash achieves 2.63× speedup on Wan2.1-14B with only a 0.19% drop in VBench (83.62% → 83.43%), while PAB already incurs a 0.71% VBench drop at a mere 1.38× speedup. On LPIPS/PSNR metrics, PreciseCache-base consistently achieves the best results (LPIPS 0.0451 vs. the best competitor at 0.0741).

### Ablation Study: Effect of Downsampling Rate and Number of GPUs

| Downsampling Factor (T×H×W) | Latency (s) | VBench ↑ | LPIPS ↓ |
|:-------------------|:---------|:---------|:--------|
| Baseline (no caching) | 907 (1×) | 83.62% | - |
| 1×2×2 | 918 (0.98×) | 83.57% | 0.0797 |
| 1×4×4 | 525 (1.73×) | 83.49% | 0.0801 |
| **2×4×4 (default)** | **416 (2.18×)** | **83.52%** | **0.0793** |
| 1×8×8 | 401 (2.26×) | 83.18% | 0.1946 |
| 4×4×4 | 403 (2.25×) | 83.02% | 0.1875 |

2×4×4 achieves the best trade-off: an insufficient downsampling ratio fails to accelerate effectively (1×2×2 yields only 0.98×), while excessive downsampling degrades LFD estimation accuracy and reduces quality (4×4×4 drops VBench to 83.02%).

| No. of GPUs | Wan2.1 Baseline | + PreciseCache | Speedup |
|:---------|:-----------|:--------------|:-------|
| 1 | 3326s | 1330s | 2.50× |
| 2 | 1732s | 753s | 2.30× |
| 4 | 907s | 416s | 2.18× |
| 8 | 459s | 229s | 2.00× |

PreciseCache remains effective across different GPU counts, with the highest speedup ratio achieved at single-GPU settings (2.50×), and is orthogonally complementary to DSP parallelism strategies.

## Rating

**Rating**: ⭐⭐⭐⭐

**Highlights & Insights**:

- The design of the LFD metric is elegant and physically intuitive — the low-to-high frequency generation order in the diffusion process inherently determines the distribution of step-level redundancy.
- The downsampled trial inference cleverly resolves the chicken-and-egg problem of needing to perform inference in order to compute the caching indicator, with negligible practical overhead.
- The two-level caching architecture (step-level + block-level) is complementary, with acceleration effects compounding.
- Experiments cover 4 mainstream video generation models, multiple resolutions, and GPU configurations, demonstrating strong generalizability.
- Plug-and-play with no training required; hyperparameters are minimal and stable across models.

**Limitations & Future Work**:

- The low/high frequency partitioning ratio for LFD (radius of $\frac{1}{5}$) lacks theoretical justification and relies on empirical tuning.
- The Flash configuration exhibits non-trivial quality degradation on certain metrics (e.g., LPIPS), indicating the boundary of aggressive acceleration.
- Comparison with distillation-based acceleration methods (e.g., consistency distillation) is absent — the two families of methods are complementary but not discussed.
- The selection of pivotal blocks in BlockCache is static (based on the last full inference), whereas the importance of blocks may change dynamically during the denoising process.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](../../CVPR2026/video_generation/disca_accelerating_video_diffusion_transformers_wi.md)
- [\[NeurIPS 2025\] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](../../NeurIPS2025/video_generation/lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[CVPR 2026\] LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](../../CVPR2026/video_generation/linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](streaming_autoregressive_video_generation_via_diagonal_distillation.md)

</div>

<!-- RELATED:END -->
