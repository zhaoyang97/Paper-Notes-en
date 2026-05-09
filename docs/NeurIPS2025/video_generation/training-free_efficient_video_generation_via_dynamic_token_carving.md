---
title: >-
  [Paper Note] Training-Free Efficient Video Generation via Dynamic Token Carving
description: >-
  [NeurIPS 2025][Video Generation][Training-free acceleration] This paper proposes Jenga, a training-free inference acceleration framework for video DiTs that achieves 8.83× speedup on HunyuanVideo with only a 0.01% drop in VBench score. The framework combines dynamic block attention carving (sparse KV block selection after token reordering via 3D space-filling curves) and a progressive resolution strategy (coarse-to-fine denoising), which operate orthogonally.
tags:
  - NeurIPS 2025
  - Video Generation
  - Training-free acceleration
  - sparse attention
  - 3D space-filling curves
  - progressive resolution
  - video diffusion Transformer
date: 2026-05-08
content_hash: 1c199b064272e7c2
---

# Training-Free Efficient Video Generation via Dynamic Token Carving

**Conference**: NeurIPS 2025
**arXiv**: [2505.16864](https://arxiv.org/abs/2505.16864)
**Code**: [https://github.com/dvlab-research/Jenga](https://github.com/dvlab-research/Jenga)
**Area**: Diffusion Models / Video Generation Acceleration
**Keywords**: Training-free acceleration, sparse attention, 3D space-filling curves, progressive resolution, video diffusion Transformer

## TL;DR

This paper proposes Jenga, a training-free inference acceleration framework for video DiTs that achieves 8.83× speedup on HunyuanVideo with only a 0.01% drop in VBench score. The framework combines dynamic block attention carving (sparse KV block selection after token reordering via 3D space-filling curves) and a progressive resolution strategy (coarse-to-fine denoising), which operate orthogonally.

## Background & Motivation

**Background**: Video diffusion Transformers (DiTs) such as HunyuanVideo and Wan2.1 can generate high-quality videos, but inference is prohibitively slow — HunyuanVideo requires approximately 27 minutes to generate a 5-second 720P video on a single H800 GPU, severely limiting practical deployment.

**Limitations of Prior Work**: The inference bottleneck stems from two orthogonal factors: (1) the $O(N^2)$ complexity of self-attention — 720P video yields ~115K tokens, with attention accounting for 77.8% of total computation; and (2) multi-step diffusion sampling — 50 denoising steps introduce a 50× computational overhead. Existing acceleration methods either address only one factor (e.g., STA/CLEAR apply sparse attention but achieve only 1.5–2× speedup; TeaCache skips steps but does not reduce per-step computation), or require additional training (step distillation degrades quality and is costly to train).

**Key Challenge**: Existing sparse attention methods rely on fixed spatial-temporal locality patterns, ignoring the variation in attention distributions across different inputs, layers, and heads, which limits how aggressively they can be applied. Furthermore, reducing the total token count (lower resolution) and reducing KV interactions (sparse attention) represent two independent acceleration dimensions that should be exploited jointly.

**Goal**: To design a training-free, plug-and-play inference pipeline that simultaneously reduces per-step token interactions and total step count, achieving 5–10× speedup while preserving generation quality.

**Key Insight**: Two key observations motivate the design: (1) diffusion denoising proceeds from low to high frequency — early steps do not require high-resolution latents; (2) later steps do not require dense full attention — video latents exhibit substantial redundancy, and extreme sparsity (retaining only 1% of KV blocks) can still preserve fine details.

**Core Idea**: Analogous to the physical Jenga game, Jenga maximally removes redundant blocks while maintaining structural stability — ProRes reduces total token count, AttenCarve reduces token interactions, and their orthogonal combination yields multiplicative speedup.

## Method

### Overall Architecture

Jenga partitions the original $T$-step denoising process into $S$ stages. The first stage generates content structure at low resolution, and subsequent stages progressively increase resolution to refine details. Within each stage, the video latent is reordered into locally coherent blocks using a 3D space-filling curve, and only the most important KV block pairs are selected via dynamic top-K selection, skipping redundant attention computations. The entire pipeline requires no training and can be directly applied to any video DiT.

### Key Designs

1. **Attention Carving (AttenCarve)**:

    - **Function**: Reduces full attention from $O(N^2)$ to $O(N'N)$, where $N'$ is the average number of selected tokens.
    - **Mechanism**: Video latent tokens are first reordered from $z_{thw}$ to $z_{blk} = \mathcal{G}(z_{thw})$ using a generalized Hilbert curve (3D SFC), such that tokens adjacent in 1D are also adjacent in 3D space. The reordered tokens are evenly divided into $M$ blocks of size $m=128$ tokens each. A union of three block-wise masks is constructed: (a) Importance Mask $\mathbf{B}_{top}$ — block-averaged attention probabilities $\mathbf{R} = \text{softmax}(\hat{Q}\hat{K}^T/\sqrt{d_k})$ are used to retain the top-$k$ KV blocks per query block, with an additional probability threshold $p$ to prevent loss of global information; (b) Condition Mask — all attention interactions related to text conditioning are fully retained; (c) Adjacency Mask — blocks within the 3D 26-neighborhood are retained to eliminate boundary artifacts.
    - **Design Motivation**: Unlike fixed local windows (CLEAR/SVG), dynamic top-K selection adapts to the heterogeneous attention patterns across heads — shallow layers tend to be local, deeper layers encode semantics, and certain heads aggregate globally. The probability threshold specifically protects these globally-aggregating heads. SFC reordering preserves locality more effectively than linear partitioning, reducing the number of blocks required.

2. **Progressive Resolution (ProRes)**:

    - **Function**: Reduces the total token count in early denoising steps, compressing pipeline-level computation.
    - **Mechanism**: The $T$-step denoising process is divided into $S$ stages, progressing from low resolution $R_1$ to the target resolution $R_S$. At the end of each stage, the clean latent $\hat{x}_0^s$ is predicted, upsampled to the next stage's resolution via 3D area interpolation, and re-noised to continue denoising: $x_{t-1} = (1-\sigma_t) \times \mathcal{U}(\hat{x}_0^s) + \sigma_t \tilde{\epsilon}$. A text-attention amplifier is introduced: during low-resolution stages, a bias $\beta = -\rho \log(\text{numel}(R_s)/\text{numel}(R_S))$ is added to visual-text attention, amplifying text conditioning to prevent field-of-view (FOV) degradation caused by the model over-focusing on local regions at low resolution. A fixed 23-step timestep schedule is also employed, achieving comparable performance to TeaCache-fast with no additional computational overhead.
    - **Design Motivation**: The coarse-to-fine nature of diffusion denoising — establishing content structure early and refining details later — naturally justifies starting at low resolution. The text-attention amplifier elegantly resolves the low-resolution → narrow FOV issue by reinforcing global text conditioning, effectively encouraging the model to generate as if operating at full resolution.

### Loss & Training

Jenga is entirely training-free; all components are plug-and-play. AttenCarve employs a custom sparse attention kernel implemented in Triton. Multi-GPU parallelism is supported via xDiT, with 8 GPUs providing an additional 6.28× speedup.

## Key Experimental Results

### Main Results

| Method | NFE | VBench↑ | VBench-Q↑ | VBench-S↑ | DiT Time | Speedup |
|---|---|---|---|---|---|---|
| HunyuanVideo Baseline | 50 | 82.74% | 85.21% | 72.84% | 1625s | 1.00× |
| CLEAR (r=32) | 50 | 82.68% | 86.06% | 69.17% | 1848s | 0.89× |
| MInference | 50 | 83.36% | 85.41% | 75.16% | 815s | 1.99× |
| SVG | 50 | 83.11% | 85.87% | 72.07% | 988s | 1.64× |
| AttenCarve (attention only) | 50 | 83.42% | 85.31% | 75.85% | 748s | 2.17× |
| Jenga-Base (1 stage) | 23 | **83.34%** | 85.19% | 75.92% | 347s | 4.68× |
| Jenga-Turbo (2 stages) | 24 | 83.07% | 84.47% | **77.48%** | 225s | 7.22× |
| Jenga-Flash (2 stages, high sparsity) | 24 | 82.73% | 84.01% | 77.58% | 184s | **8.83×** |

| Model / Setting | VBench | Latency | Speedup |
|---|---|---|---|
| HunyuanVideo-I2V Baseline | 87.49% | 1499s | 1.00× |
| + Jenga | 87.75% | 338s | 4.43× |
| Wan2.1-1.3B Baseline | 83.28% | 115s | 1.00× |
| + Jenga | 82.68% | 24s | 4.79× |
| AccVideo (distilled model) | 83.82% | 161s | 1.00× |
| + Jenga | 83.39% | 76s | 2.12× |
| HunyuanVideo 8GPU | 82.74% | 225s | 1.00× |
| + Jenga-Flash 8GPU | 82.73% | 39s | **5.77×** |

### Ablation Study

| Configuration | VBench | Latency | Notes |
|---|---|---|---|
| Linear hwt partitioning | 82.82% | 229s | Shift artifacts; requires more blocks |
| SFC partitioning | 83.07% | 225s | Better locality; fewer blocks needed |
| w/o adjacency mask | 81.82% | 221s | Grid artifacts at block boundaries |
| w/o condition mask | 82.42% | 222s | Degraded text semantics |
| 2-stage ProRes | 83.07% | 225s | Best quality-speed trade-off |
| 3-stage ProRes | 80.53% | 157s | 10.35× speedup but quality degrades |
| Text amplifier $\rho$=0.0 | 82.40% | - | FOV degradation at low resolution |
| Text amplifier $\rho$=0.5 | 83.07% | - | Optimal FOV preservation |

### Key Findings

- Jenga-Base (attention carving + step skipping only) **surpasses** the baseline VBench score (83.34% vs. 82.74%), primarily due to a substantial improvement in semantic score (75.92% vs. 72.84%) — sparse attention forces the model to focus on salient information.
- Dynamic block selection (AttenCarve) outperforms fixed-pattern methods (CLEAR/SVG) by 1.3–2.4× in speed while achieving superior quality.
- The text-attention amplifier effectively mitigates FOV degradation in low-resolution generation.
- A 2.12× speedup is still achieved on a distilled model (AccVideo, 5 steps), confirming orthogonality with step distillation.
- User studies indicate that the perceptual quality of Jenga outputs is indistinguishable from the baseline.
- Block selection introduces only 2.8% additional computational overhead and 3.7% additional memory usage (71.84 → 74.49 GiB).

## Highlights & Insights

- The framework design is highly elegant: attention acceleration and pipeline acceleration are decoupled into two independent orthogonal dimensions that can be flexibly combined. AttenCarve accelerates individual steps; ProRes reduces step count and token count; their product yields superlinear overall speedup.
- SFC reordering combined with dynamic top-K selection reflects a deep understanding of video attention sparsity: different layers and heads exhibit distinct patterns (local, positional, semantic, global), which fixed patterns cannot simultaneously accommodate, yet the overhead of dynamic selection is minimal.
- The training-free property is a significant advantage — Jenga can be directly applied to HunyuanVideo, Wan2.1, AccVideo, and other models without any fine-tuning.

## Limitations & Future Work

- Latent-space resizing in ProRes occasionally produces boundary artifacts, particularly in static scenes or at sharp edges. Detailed prompts can mitigate this issue, but a fundamental fix would require pixel-domain resizing, incurring an additional ~50s overhead.
- The current SFC partitioning is static and does not leverage semantic information for token importance estimation. Learnable attention carving remains an avenue for future exploration.
- Three-stage ProRes exhibits noticeable quality degradation (80.53%), and latent alignment across stages remains a challenge.
- This work focuses on inference acceleration and is orthogonal to training-side optimizations (step distillation, architectural improvements), though joint exploration has not been pursued.

## Related Work & Insights

- **vs. STA/CLEAR/SVG**: These methods rely on fixed local windows or spatial-temporal sparse patterns; CLEAR is even slower than the baseline (0.89×). Jenga's dynamic selection achieves 2.17× speedup with superior quality.
- **vs. TeaCache**: TeaCache achieves 2.31× speedup by caching and skipping features at the step level, which is orthogonal to Jenga's ProRes and complementary — ProRes reduces token count at the step level. Combined use yields further gains.
- **vs. Bottleneck Sampling**: BottleneckSampling also employs a variable-resolution strategy but retains full resolution in the first stage. ProRes is more aggressive in starting from low resolution and compensates for FOV degradation through the text-attention amplifier.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The orthogonal combination of dynamic block attention carving and progressive resolution is an elegant design; the FOV correction via the text-attention amplifier is particularly clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluation across four model variants (T2V, I2V, distilled model, Wan2.1), detailed ablations, user studies, multi-GPU deployment, and 16-dimensional VBench breakdown.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive method presentation, and an exceptionally detailed appendix (algorithm pseudocode, parameter tables, implementation details).
- **Value**: ⭐⭐⭐⭐⭐ — Training-free 8.83× speedup with negligible quality loss and plug-and-play applicability confer very high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation](s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[CVPR 2026\] Training-free Motion Factorization for Compositional Video Generation](../../CVPR2026/video_generation/training-free_motion_factorization_for_compositional_video_generation.md)
- [\[NeurIPS 2025\] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)
- [\[CVPR 2026\] LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](../../CVPR2026/video_generation/linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)

</div>

<!-- RELATED:END -->
