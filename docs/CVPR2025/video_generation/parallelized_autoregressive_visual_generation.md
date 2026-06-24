---
title: >-
  [Paper Note] Parallelized Autoregressive Visual Generation
description: >-
  [CVPR 2025][Video Generation][Autoregressive Generation] PAR (Parallelized Autoregressive) is proposed to analyze visual token dependency, generating weakly dependent tokens that are spatially distant in parallel while maintaining sequential generation for locally, strongly dependent tokens, achieving 3.6-9.5x speedup with almost no loss in quality.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Autoregressive Generation"
  - "Parallel Decoding"
  - "Visual Token Dependency"
  - "Image Generation"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: bcac1bbc88a5f3e2
---

# Parallelized Autoregressive Visual Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.15119](https://arxiv.org/abs/2412.15119)  
**Code**: [Project Page](https://yuqingwang1029.github.io/PAR-project)  
**Area**: Video Generation  
**Keywords**: Autoregressive Generation, Parallel Decoding, Visual Token Dependency, Image Generation, Inference Acceleration

## TL;DR

PAR (Parallelized Autoregressive) is proposed to analyze visual token dependency, generating weakly dependent tokens that are spatially distant in parallel while maintaining sequential generation for locally, strongly dependent tokens, achieving 3.6-9.5x speedup with almost no loss in quality.

## Background & Motivation

Autoregressive models have demonstrated powerful capabilities in visual generation, but the inference speed of token-by-token sequential prediction is extremely slow—requiring 576 steps for an image with $24 \times 24 = 576$ tokens, which severely limits practical applications.

Existing acceleration schemes have their own limitations: (1) speculative decoding requires an additional draft model; (2) MaskGIT adopts a non-autoregressive paradigm, which changes the modeling approach; (3) VAR requires a specialized multi-scale tokenizer and longer token sequences.

**Core Problem**: Can parallel generation be achieved while maintaining the simplicity and flexibility of standard autoregressive models?

**Key Insight**: **The feasibility of parallel generation depends on token dependency**. Locally adjacent tokens exhibit strong dependency—independently sampling multiple strongly dependent tokens leads to inconsistencies (e.g., distorted tiger faces, broken zebra stripes); whereas spatially distant tokens exhibit weak dependency and can be safely generated in parallel. However, the **initial tokens** of each region, even if distant, are crucial as they jointly determine the global structure, and thus must be sequentially generated.

## Method

### Overall Architecture

PAR is built upon a standard autoregressive Transformer and achieves parallelization through a token re-ordering mechanism. The image token grid is partitioned into $M \times M$ regions, which are generated in two stages: (1) sequentially generating the initial token of each region to establish the global structure; (2) parallelly generating tokens at the corresponding positions across different regions. 2D RoPE is utilized to preserve spatial position information, and intra-group bidirectional attention is leveraged to enrich local context.

### Key Design 1: Non-Local Parallel Generation Strategy

**Function**: Significantly reducing the number of generation steps while preserving quality

**Mechanism**: The $H \times W$ token grid is divided into $M \times M$ regions (e.g., $M=2$ yields 4 regions), and then tokens are grouped across regions according to their corresponding positions:

$$\{[v_1^{(1)}, \cdots, v_1^{(M^2)}], [v_2^{(1)}, \cdots, v_2^{(M^2)}], \cdots, [v_k^{(1)}, \cdots, v_k^{(M^2)}]\}\}$$

Stage 1: Sequentially generating the initial token of each region $v_1^{(i)} \sim \mathbb{P}(v_1^{(i)} | v_1^{(<i)})$

Stage 2: Parallelly generating tokens at the $j$-th position of all regions $\{v_j^{(r)}\}_{r=1}^{M^2} \sim \mathbb{P}(\{v_j^{(r)}\} | v_{<j})$

Taking $M=2$ as an example, the total number of steps is reduced from 576 to $4 + \frac{576-4}{4} = 147$.

**Design Motivation**: Directly predicting adjacent tokens in parallel causes severe quality degradation because the joint distribution cannot be factorized into independent distributions. The weak correlation between spatially distant tokens minimizes the impact of independent sampling. Sequentially generating the initial tokens first avoids conflicts in global structure.

### Key Design 2: Intra-Group Bidirectional Attention + Global Autoregression

**Function**: Enriching the visible context for each token during parallel generation

**Mechanism**: Under a naive causal mask, tokens in a parallelly generated group (e.g., $6b$) can only attend to tokens in the previous group up to the corresponding position (up to $5b$). This is modified to intra-group bidirectional attention + inter-group causal attention: when predicting the current group $[6a, 6b, 6c, 6d]$, each token can access the entire previous group $[5a, 5b, 5c, 5d]$ as context.

**Design Motivation**: Naive causal masking restricts the context window, resulting in insufficient information acquisition for each token. Intra-group bidirectional attention enriches local context without violating the global autoregressive property, while remaining compatible with KV-cache optimization.

### Key Design 3: Learnable Transition Tokens

**Function**: Helping the model smoothly transition from sequential generation to parallel generation mode

**Mechanism**: Intersperse $n-1$ learnable tokens $[M_1, M_2, M_3]$ between the initial sequential tokens $[1, 2, \cdots, n]$ and the parallel groups. These tokens have the same dimension as regular tokens, participate in training but do not predict physical visual content, serving solely as a "signal" for mode switching.

**Design Motivation**: The model needs to adapt to the transition from single-token prediction to multi-token prediction. Learnable transition tokens provide a gentle transition mechanism, preventing abrupt behavioral shifts in the model during mode switching.

### Loss & Training

Standard autoregressive cross-entropy loss is computed on the re-ordered token sequence. During training, token predictions at all stages are supervised simultaneously.

## Key Experimental Results

### Main Results: ImageNet 256×256 Class-Conditional Image Generation

| Method | Params | FID↓ | IS↑ | Steps | Time (s)↓ |
|------|------|------|------|------|----------|
| LlamaGen-XXL | 1.4B | 2.34 | 253.9 | 576 | 12.41 |
| MaskGIT | 227M | 6.18 | 182.1 | 8 | 0.13 |
| VAR-d30 | 2B | 1.97 | 334.7 | 10 | 0.27 |
| **PAR-XXL (4×)** | **1.4B** | **2.29** | **271.4** | **147** | **3.46** |
| **PAR-XXL (16×)** | **1.4B** | **3.02** | **247.6** | **40** | **1.31** |

### Ablation Study: Comparison of Parallel Strategies

| Strategy | FID↓ | IS↑ |
|------|------|-----|
| Standard AR (Baseline) | 2.34 | 253.9 |
| **PAR-4× (Non-Local)** | **2.29** | **271.4** |
| Naive Parallel (Adjacent Tokens) | 10.05 | 150.2 |
| Without Sequential Initialization | 5.87 | 195.3 |

### Key Findings

- PAR-4× achieves a **$3.6\times$ physical speedup**, with the FID decreasing from 2.34 to **2.29** (which is actually a slight improvement!).
- PAR-16× achieves a **$9.5\times$ speedup**, with the FID only increasing from 2.34 to 3.02 (representing minimal quality degradation).
- Directly parallelizing adjacent tokens leads to catastrophic quality degradation (FID of 10.05), validating the correctness of the token dependency analysis.
- Sequential initialization is critical: generating initial tokens non-sequentially increases the FID from 2.29 to 5.87.
- The method is compatible with both VQGAN and MAGVIT-v2 tokenizers, and extending it to video generation (UCF-101) only requires switching to 3D positional encodings.

## Highlights & Insights

1. **Dependency Analysis-Driven Design**: Instead of blind parallelization, the decision of which tokens to parallelize is based on an in-depth analysis of token dependency intensity.
2. **Minimalist and Versatile**: It requires no modifications to the architecture or tokenizers. Achieved solely through token re-ordering and a small number of learnable tokens, it can be seamlessly integrated into any standard AR model.
3. **4x Speedup and Ironically Improved Quality**: Non-local parallelization actually enhances the information exchange across different regions.

## Limitations & Future Work

- Parallelization in the temporal dimension performs poorly due to strong temporal dependencies in videos; thus, parallelization is restricted to the spatial dimension.
- Small but visible quality degradation exists under high parallelization factors ($16\times$), leaving room for improvement in extreme acceleration scenarios.
- The number of sequential generation steps for initial tokens is fixed to $M^2$; adaptive selection strategies are worth exploring.
- Future work can combine this method with speculative decoding to further accelerate inference.

## Related Work & Insights

- **LlamaGen**: Standard AR visual generation baseline, which PAR accelerates.
- **VAR**: An acceleration scheme using multi-scale prediction, but requires a specialized tokenizer; PAR is more general.
- **MaskGIT**: Non-autoregressive parallel generation, which alters the modeling paradigm; PAR maintains the autoregressive property.

## Rating

⭐⭐⭐⭐⭐ — The method design is exquisite and its principles are clear. The non-local parallel strategy based on token dependency achieves both theoretical rationality and practical effectiveness. The results of $3.6\times$ speedup with no degradation (and even slight improvement) in quality are highly impressive. The generality and simplicity of the method endow it with broad impact. It provides an elegant solution to the efficiency problem of autoregressive visual generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Taming Teacher Forcing for Masked Autoregressive Video Generation](taming_teacher_forcing_for_masked_autoregressive_video_generation.md)
- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)
- [\[CVPR 2025\] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](from_slow_bidirectional_to_fast_autoregressive_video_diffusion_models.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)

</div>

<!-- RELATED:END -->
