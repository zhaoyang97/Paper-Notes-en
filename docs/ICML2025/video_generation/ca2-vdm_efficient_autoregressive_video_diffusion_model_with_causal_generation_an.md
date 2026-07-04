---
title: >-
  [Paper Note] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing
description: >-
  [ICML 2025][Video Generation][Video Diffusion Models] Ca2-VDM is proposed, which eliminates redundant calculations of conditional frames in autoregressive video diffusion models through two key designs: Causal Generation and Cache Sharing. It reduces computational complexity from quadratic to linear, generating 80-frame videos 2.5 times faster than the baseline while maintaining state-of-the-art generation quality.
tags:
  - "ICML 2025"
  - "Video Generation"
  - "Video Diffusion Models"
  - "Autoregressive Generation"
  - "KV-Cache"
  - "Causal Attention"
  - "Long Video Generation"
date: 2026-05-08
content_hash: d7c2400c28fcb377
---

# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

**Conference**: ICML 2025  
**arXiv**: [2411.16375](https://arxiv.org/abs/2411.16375)  
**Code**: [https://github.com/Dawn-LX/CausalCache-VDM](https://github.com/Dawn-LX/CausalCache-VDM)  
**Area**: Video Generation  
**Keywords**: Video Diffusion Models, Autoregressive Generation, KV-Cache, Causal Attention, Long Video Generation

## TL;DR

Ca2-VDM is proposed, which eliminates redundant calculations of conditional frames in autoregressive video diffusion models through two key designs: Causal Generation and Cache Sharing. It reduces computational complexity from quadratic to linear, generating 80-frame videos 2.5 times faster than the baseline while maintaining state-of-the-art generation quality.

## Background & Motivation

Existing video diffusion models (VDMs) typically generate long videos autoregressively: each step generates a short clip, conditioned on the last few frames of the previous clip. This paradigm faces two core efficiency bottlenecks:

**Redundant Computation**: Overlapping conditional frames exist between adjacent clips, forcing the model to recompute the features of these frames at each autoregressive step. As conditional frames accumulate over autoregressive steps to provide long-term context, the computational requirement exhibits **quadratic growth**.

**Cache Storage**: Since existing VDMs use the same timestep embedding for both conditional frames and noisy frames, the KV features vary across different denoising steps. Consequently, caching KV separately for each denoising step consumes an enormous amount of GPU memory.

Key Challenge: Long-term context requires scaling up conditional frames, but scaling these frames causes a dramatic surge in computational and storage overhead. The authors argue that the crux lies in the **Bidirectional Attention** mechanism of existing VDMs, which makes the KV features of conditional frames dependent on the current noisy frames, rendering them impossible to precompute and reuse.

## Method

### Overall Architecture

Ca2-VDM is based on a Spatial-Temporal Transformer (initialized with Open-Sora v1.0), with core modifications focused on the attention mechanism. The overall pipeline is divided into training and inference stages:

- **Training Stage**: The input video sequence is partially noised—the first $P$ frames are kept clean as a clean prefix (condition), and the remaining $L-P$ frames are noised as the denoising target. $P$ is randomly sampled during training. Crucially, the clean prefix uses $\text{tEmb}(0)$, while the denoising target uses $\text{tEmb}(t)$, ensuring that the two parts have distinct timestep embeddings.
- **Inference Stage**: Each autoregressive (AR) step consists of two sub-stages: (1) **Denoising Stage**—denoising to generate a new chunk using the pre-stored KV-cache; (2) **Cache Writing Stage**—performing a single forward pass on the denoised result to compute and store the new KV-cache for the next step.

### Key Designs

#### 1. Causal Temporal Attention

Replacing standard bidirectional temporal attention with causal (unidirectional) attention, which ensures each frame only attends to its preceding frames via a lower-triangular mask:

$$\text{CausalAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{C'}} + \mathbf{M}\right)\mathbf{V}$$

where $\mathbf{M}$ is a lower-triangular mask matrix ($M_{i,j} = -\infty$ if $i < j$). The key significance of this design is that the KV features of conditional frames no longer depend on subsequent noisy frames, allowing them to be precomputed and cached in the previous AR step and reused directly in all subsequent AR steps.

#### 2. Cache Sharing

Causal generation ensures that the KV of the clean prefix is solely determined by the clean frames themselves, independent of the denoising timestep $t$. During training, $\text{tEmb}(0)$ is fixed for the clean prefix, and the same applies during inference—this enables **the same KV-cache to be shared across all denoising steps**.

In contrast to methods like Live2diff, which require storing separate KV-caches for each denoising step (where storage is proportional to the number of denoising steps $T$), the KV-cache storage of Ca2-VDM is independent of $T$, saving $T$ times GPU memory.

#### 3. Prefix-Enhanced Spatial Attention

To enhance the spatial guidance of the conditional frames on the generated frames, the spatial features of the most recent $P'$ frames in the clean prefix are injected into the spatial attention of each frame via spatial concatenation:

- For denoising target frames $i \geq P$: Concat the spatial features of the last $P'$ prefix frames onto the K/V of frame $i$.
- For clean prefix frames $i < P$: Achieve alignment via self-repetition.

The attention map size is $(HW) \times ((P'+1)HW)$. $P'$ is typically set to 3 (a small value), keeping the computational overhead manageable.

#### 4. Temporal KV-Cache Queue and Cyclic-TPEs

**KV-Cache Queue**: As the autoregressive process proceeds, the number of conditional frames $P_k$ grows continually. When $P_k$ reaches the preset maximum $P_{\max}$, the earliest KV-cache is dequeued to maintain a constant queue length. This ensures that the computational and storage costs do not grow infinitely while still leveraging long-term context.

**Cyclic-TPEs**: When the cumulative generation length exceeds the training length, the temporal position encodings (TPE) will be exhausted. Since early TPEs are already bound to the stored KV-cache, they cannot simply be reset. The authors designed a cyclic shift mechanism where the denoising targets are assigned to TPE indices starting over from the beginning. During training, this behavior is aligned by using cyclic shifted TPE sequences with random offsets.

#### 5. Spatial KV-Cache

Since $P' < l$, prefix-enhancement for the current chunk only requires the spatial KV-cache of the most recent chunk. Thus:
- Only **one chunk** of spatial KV-cache is stored.
- It is overwritten at each AR step, requiring no queue structure.

### Loss & Training

The **loss function** adopts an improved diffusion loss:

$$\widetilde{\mathcal{L}}_{\text{simple}}(\theta) = \mathbb{E}_{\mathbf{z}, \boldsymbol{\epsilon}, t}\left[\|(\boldsymbol{\epsilon}_\theta([\mathbf{z}_0^{0:P}, \mathbf{z}_t^{P:L}], \mathbf{t}) - \boldsymbol{\epsilon}) \odot \mathbf{m}\|_2^2\right]$$

- $\mathbf{m}$ is a loss mask that computes loss only on the denoising target part ($i \geq P$).
- $\mathbf{t}$ is the timestep vector: 0 for the prefix part, and $t$ for the target part.
- During actual training, $\mathcal{L}_{\text{vlb}}$ (with learnable covariance) is additionally optimized.

**Training Strategy**:
- Clean prefix length $P$ is randomly sampled: $P \in \{1, 1+l, \ldots, 1+nl\}$ (where $l$ is a multiple of the chunk length).
- Training videos of varying lengths are used: $L_{\text{train}} = P + l$.
- TPE sequences are randomly cyclically shifted during training to support Cyclic-TPEs at inference.
- Initialized based on Open-Sora v1.0, utilizing the T5 text encoder and StableDiffusion VAE.

## Key Experimental Results

### Main Results

**Zero-Shot Text-to-Video FVD Evaluation** (at 16x256x256 resolution):

| Method | Condition | MSR-VTT FVD | UCF101 FVD |
|------|------|-------------|------------|
| ModelScope | T | 550 | 410 |
| VideoComposer | T | 580 | - |
| Make-A-Video | T | - | 367.2 |
| PixelDance | T+I | 381 | 242.8 |
| SEINE | T+I | 181 | - |
| **Ca2-VDM** | **T+I** | **181** | **277.7** |

**UCF-101 Fine-tuning FVD** (at 256x256 resolution):

| Method | Resolution | FVD |
|------|--------|-----|
| MCVD | 64² | 1143 |
| VDT | 64² | 225.7 |
| VideoFusion | 128² | 220 |
| Latte | 256² | 333.6 |
| PVDM | 256² | 343.6 |
| **Ca2-VDM** | **256²** | **184.5** |

**80-Frame Generation Speed Comparison** (256x256, single A100):

| Method | Scalable Condition | Time (s) | Relative Speedup |
|------|-----------|----------|---------|
| StreamT2V | ✗ | 150 | 1× |
| OS-Ext | ✓ | 130.1 | 1.15× |
| OS-Fix | ✗ | 77.5 | 1.94× |
| **Ca2-VDM** | **✓** | **52.1** | **2.88×** |

### Ablation Study

**Ablation of Condition Length $P_{\max}$ and Prefix-Enhancement (PE)** (SkyTimelapse, 48 frames / 6 AR steps):

| $P_{\max}$ | PE | Chunk 1 FVD | Chunk 2 FVD | Chunk 3 FVD | Description |
|-----------|-----|-------------|-------------|-------------|-------------|
| 25 | ✗ | 274.8 | 244.5 | 275.1 | Short condition, no enhancement |
| 25 | ✓ | 257.4 | 216.5 | 238.5 | +PE brings significant improvement |
| 41 | ✗ | 187.3 | 209.3 | 263.2 | Long condition yields good results |
| 41 | ✓ | **185.0** | **202.9** | **240.5** | Combining both is optimal |

**GPU Memory Comparison** (256x256, 50 denoising steps):

| Method | KV-cache Memory | Total Memory |
|------|-------------|--------|
| Live2diff (T=50) | 17.70 GB | 29.46 GB |
| Ca2-VDM w/ PE | 0.86 GB | 4.79 GB |
| Ca2-VDM w/o PE | 0.77 GB | 3.95 GB |

### Key Findings

1. **Linear vs. Quadratic Complexity**: The time cost of OS-Ext grows quadratically with AR steps, while Ca2-VDM grows only linearly.
2. **Prefix-Enhancement is Effective**: PE brings FVD improvements under all configurations (improving Chunk 2 by approximately 10%).
3. **Longer Conditional Frames Improve Late-Stage Quality**: Increasing $P_{\max}$ from 25 to 41 reduces Chunk 1 FVD from 274.8 to 187.3.
4. **Cache Sharing Saves Memory**: Compared to Live2diff's 17.7 GB KV-cache, Ca2-VDM only requires 0.86 GB (a 20x reduction).
5. **FLOPs Impact on Different Attention Layers**: When expanding conditional frames, the FLOPs of all three attention layers in OS-Ext grow, whereas for Ca2-VDM, only temporal attention slightly increases, while spatial and text cross-attention remain unchanged.

## Highlights & Insights

1. **Migrating KV-cache from LLMs to VDMs is non-trivial**: LLMs generate one token per step and compute KV cache synchronously. In contrast, VDMs require iterative model calls per AR step (with varying $t$), and each temporal token corresponds to $HW$ visual grid elements, resulting in a storage overhead significantly larger than text. This paper cleverly addresses these two unique challenges through causal attention and cache sharing.
2. **Distinct timestep embedding is the key to cache sharing**: The simple modification of using different timestep embeddings for conditional frames and denoising targets decouples the KV of conditional frames from $t$, serving as the foundation of the entire approach.
3. **Cyclic-TPEs elegantly handle position encoding in ultra-long inference**: It avoids the problem where TPEs cannot be reset after being bound to the KV-cache.

## Limitations & Future Work

1. **Information Loss in Causal Attention**: Unidirectional attention naturally loses backward information flow compared to bidirectional attention. Although PE provides some compensation, whether the generation quality degrades in more complex scenarios warrants further research.
2. **Resolution Restrictions**: All experiments were conducted at 256x256 resolution. The scalability to higher-resolution video generation (e.g., 512+) remains unverified.
3. **Simple Dequeuing Strategy for KV-cache Queue**: Evicting the earliest frames first may discard crucial global information. A wiser selective eviction strategy could potentially improve quality further.
4. **Potential Safety Risks**: Efficient real-time video generation could be misused for deepfake content generation, requiring compliance with security measures such as watermarking.

## Related Work & Insights

- **Open-Sora v1.0**: The backbone foundation of Ca2-VDM, demonstrating the effectiveness of Spatial-Temporal Transformers in video generation.
- **StreamT2V / GenLV**: Representative autoregressive VDMs with fixed conditional frame lengths, which suffer from frame-to-frame consistency issues (abrupt transitions).
- **Live2diff**: A concurrent work that also utilizes KV-cache but lacks cache sharing, resulting in high GPU memory overhead.
- **Insights**: This technical approach can be extended to other diffusion model scenarios that require autoregressive generation (such as audio or 3D generation). The paradigm of causal attention combined with cache sharing could become a general acceleration solution.

## Rating

- Novelty: ⭐⭐⭐⭐ — Causal attention itself is not new, but combining it with cache sharing, Cyclic-TPE, etc., to form a complete solution in VDMs shows strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across multiple datasets, thorough ablation, and exhaustive efficiency analysis, though resolution and dataset diversity are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem definition, excellent figures, and a complete logical chain.
- Value: ⭐⭐⭐⭐ — Effectively resolves the core efficiency bottleneck of autoregressive VDMs, offering a practical 2.5-3x speedup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](../../CVPR2025/video_generation/timestep_embedding_tells_its_time_to_cache_for_video_diffusion_model.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](../../AAAI2026/video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)
- [\[CVPR 2026\] Accelerating Autoregressive Video Diffusion via History-Guided Cache and Residual Correction](../../CVPR2026/video_generation/accelerating_autoregressive_video_diffusion_via_history-guided_cache_and_residua.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)

</div>

<!-- RELATED:END -->
