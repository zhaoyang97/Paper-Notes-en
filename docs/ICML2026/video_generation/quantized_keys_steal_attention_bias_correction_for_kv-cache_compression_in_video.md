---
title: >-
  [Paper Note] Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation
description: >-
  [ICML 2026][Video Generation][KV Cache Quantization] This paper discovers that KV cache quantization in chunked autoregressive video diffusion models causes a **systematic shift in attention weights** ("Quantized Keys St…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "KV Cache Quantization"
  - "Attention Bias"
  - "Diffusion Models"
  - "Jensen's Inequality"
date: 2026-05-08
content_hash: a93a2c8933b9cfaa
---

# Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation

**Conference**: ICML 2026  
**arXiv**: [2605.26266](https://arxiv.org/abs/2605.26266)  
**Code**: To be confirmed  
**Area**: Video Generation / Model Compression  
**Keywords**: KV Cache Quantization, Attention Bias, Diffusion Models, Jensen's Inequality

## TL;DR
This paper discovers that KV cache quantization in chunked autoregressive video diffusion models causes a **systematic shift in attention weights** ("Quantized Keys Steal Attention"). By deriving a per-score correction term based on Jensen's Inequality, it restores near-BF16 video quality under aggressive INT2 quantization (VBench 78.02 vs. 78.27) while saving 50% memory.

## Background & Motivation

**Background**: Chunked autoregressive video diffusion models (e.g., MAGI-1, SkyReels-V2) avoid redundant computation by maintaining a KV cache for previously generated video chunks. To save memory, industry practices employ quantization techniques to compress cache keys and values to low bit-widths (INT2, INT4).

**Limitations of Prior Work**: Aggressive KV cache quantization (especially INT2) severely degrades video quality, leading to the destruction of subject and scene structures. Existing quantization methods (QuaRot, RTN, etc.) focus on reducing the quantization noise itself but fail to fully resolve deeper issues introduced by quantization.

**Key Challenge**: Integer quantization introduces **zero-mean noise** at the attention score level, which theoretically should not alter the expected behavior of attention. However, the **convexity** of the exponential function in softmax breaks this symmetry—positive deviations are amplified more than negative deviations are suppressed. This leads to a systematic overestimation of the quantized cache keys' contribution to the partition sum.

**Goal**:
- Identify and quantify the impact of this systematic bias (Jensen bias) on attention weight distribution.
- Derive theoretically sound correction terms to recover the original attention distribution without requiring retraining.
- Implement a correction scheme with extremely low overhead.

**Key Insight**: Starting from Jensen's Inequality in probability theory—in softmax, $\mathbb{E}[e^{s_i + \delta_i}] = e^{s_i} \cdot \mathbb{E}[e^{\delta_i}] > e^{s_i}$ (when $\delta_i$ is zero-mean quantization noise), which causes the contribution of cache keys to be systematically inflated.

**Core Idea**: Offset the Jensen bias by subtracting a correction term $b_i = \log \mathbb{E}[e^{\delta_i}]$ from the cached attention scores, ensuring the expected contribution of the corrected score matches the unquantized case.

## Method

### Overall Architecture
In chunked autoregressive video diffusion, the current block's query attends to two sets of keys: the current block's keys (high-precision BF16) and the preceding blocks' keys (low-bit quantized cache). Standard softmax attention calculates partition sums for these two sets separately and then normalizes them. Quantization causes the partition sum on the cache key side to be systematically exaggerated, thereby "stealing" attention mass that should have been allocated to the current block. This paper derives a **per-token correction term** to be subtracted from cache key scores before the softmax operation to restore the original attention balance.

### Key Designs

1.  **Theoretical Derivation of Jensen Bias**:
    - **Function**: Accurately describes how quantization disrupts the attention weight balance through the convexity of softmax.
    - **Mechanism**: Let the quantized score of cache key $i$ be $\hat{s}_i = s_i + \delta_i$, where $\delta_i = \frac{q^\top \epsilon_i}{\sqrt{d}}$ is the quantization noise projection, and $\epsilon_i \sim \mathcal{U}(-\Delta_i/2, +\Delta_i/2)$. At the expected level, the cache-side partition sum is $\mathbb{E}[\hat{Z}_\mathcal{S}] = \sum_{i \in \mathcal{S}} e^{s_i} \cdot \mathbb{E}[e^{\delta_i}]$. By Jensen's Inequality, $\mathbb{E}[e^{\delta_i}] \geq e^{1 \cdot \mathbb{E}[\delta_i]} = 1$ (since $\delta_i$ is zero-mean), thus $\mathbb{E}[\hat{Z}_\mathcal{S}] \geq Z_\mathcal{S}$. The gap in this inequality is the Jensen bias—leading to cache keys stealing attention mass.
    - **Design Motivation**: This is the fundamental cause of quantization-induced degradation; rather than improving the quantization scheme, it is better to directly correct the resulting bias.

2.  **Derivation of Per-Score Correction Terms**:
    - **Function**: Computes a correction value $b_i$ for each cache token to restore the expectation of the corrected contribution to its original value.
    - **Mechanism**: Requiring $e^{s_i - b_i} \cdot \mathbb{E}[e^{\delta_i}] = e^{s_i}$ yields $b_i = \log \mathbb{E}[e^{\delta_i}]$. As quantization noise components are independent across channels, the expectation decomposes. For uniform quantization noise, the exact form is $b_i = \sum_{c=1}^d \log\left(\frac{\sinh(q_c \Delta_{i, c} / (2 \sqrt{d}))}{q_c \Delta_{i, c} / (2 \sqrt{d})}\right)$. Using a second-order Taylor expansion $\log(\sinh(\alpha) / \alpha) \approx \alpha^2 / 6$ yields a concise approximation: $b_i \approx \frac{1}{24 d} \sum_{c=1}^d q_c^2 \Delta_{i, c}^2$.
    - **Design Motivation**: Theoretically derived from unbiasedness, with Taylor approximation used in practice to greatly simplify computation; this approximation can also be extended to other formats (FP, MXFP, etc.).

3.  **Inference-time Application + Complexity Control**:
    - **Function**: Applies the correction during inference with minimal overhead.
    - **Mechanism**: The correction term depends only on existing quantization parameters (step size $\Delta_{i, c}$) and the query norm $\|q\|$, requiring no additional storage. For grouped quantization (group size $g = 32$), the extra computational complexity is $O(QK \cdot d / g)$, which is only a $1/g$ overhead compared to the $O(QK \cdot d)$ of standard $QK^\top$.
    - **Design Motivation**: Ensures the correction is a practical solution; its implementation in FlexAttention adds only about 5% end-to-end latency.

### Loss & Training
This method is a **training-free inference-stage correction**. The original model parameters and training objectives remain unchanged; only score calibration is applied before the softmax.

## Key Experimental Results

### Main Results

| Model | Quantization Scheme | Precision | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VBench ↑ | Description |
|-------|---------------------|-----------|--------|--------|---------|----------|-------------|
| MAGI-1 | None | BF16 | — | — | — | 78.27 | Baseline |
| MAGI-1 | QuaRot+RTN | INT2 ✗ | 17.10 | 0.630 | 0.453 | 70.24 | W/o correction |
| MAGI-1 | QuaRot+RTN | INT2 ✓ | 22.97 | 0.801 | 0.165 | **78.02** | W/ correction |
| MAGI-1 | QVG+Correction | INT2 | 25.29 | 0.856 | 0.107 | 78.23 | Optimal combination |
| SkyReels-V2 | QuaRot+RTN | INT2 ✗ | 19.20 | 0.708 | 0.319 | 71.44 | W/o correction |
| SkyReels-V2 | QuaRot+RTN | INT2 ✓ | 20.42 | 0.784 | 0.202 | **78.58** | W/ correction |

After correction, INT2 almost entirely recovers BF16 quality; the correction is orthogonal and combinable with upstream compression methods like QVG; it achieves 50% memory savings at the same quality level.

### Ablation Study

| Configuration | Attention Mass Shift $\Delta P_\mathcal{S}$ | Median | Description |
|---------------|------------------------------------------|--------|-------------|
| BF16 Baseline | — | 0 | No shift |
| INT2 W/o correction | Significant positive shift | +0.15 | Cache keys steal attention |
| INT2 W/ correction | Near zero after correction | ~0 | Bias offset |

### Key Findings
- Attention mass shift directly corresponds to the degree of video quality degradation.
- Correction improves PSNR across all group sizes, preserving the memory-quality tradeoff curve but shifting it toward higher quality.
- Performs best under aggressive quantization (INT2 is superior to INT4).
- Cross-domain applicability: Preliminary LLM experiments show the same bias mechanism appears in chunked prefill scenarios.

## Highlights & Insights
- **Theoretical Simplicity**: Distills complex quantization issues into a single root cause (Jensen bias). The correction formula requires only query norms and step sizes, without needing complex statistics or retraining—an elegant information-theoretic perspective.
- **Cross-domain Applicability**: Although the paper focuses on video diffusion, the same bias mechanism exists in LLM chunked prefill, indicating the discovery's universality.
- **Plug-and-play**: The method is orthogonal to any upstream quantization schemes (QuaRot, RTN, QVG, etc.) and can be seamlessly combined, providing high engineering value.

## Limitations & Future Work
- Noise model assumptions: The derivation is based on uniform zero-mean noise from integer quantization; it may fail for non-uniform or biased noise. Floating-point formats like FP and MXFP require re-derivation.
- Validity at the expectation level: Correction is unbiased in the expected sense, but when attention is highly concentrated on a few cache tokens, noise from a single sample might dominate the effect, reducing the gains from correction.
- Limitations in single-token decoding: The method performs optimally in multi-token current block scenarios. In standard LLM single-token decoding, where one query attends to multiple cache tokens, competition between cache tokens is weaker, limiting the room for correction.

## Related Work & Insights
- **vs KIVI / KVQuant / QuaRot / QuantVideoGen**: These methods reduce noise during the quantization phase by redistributing step sizes or using rotations; this work complementarily removes residual bias during the decoding phase through bias correction. The two can be combined.
- **vs KVLinC**: KVLinC uses trained linear adapters to correct quantization errors; this work's correction is analytically derived and training-free, offering better generalization.
- **vs General studies on diffusion model quantization**: Previous research focused on the quantization of the softmax calculation itself; this paper uniquely identifies structural bias introduced by KV cache quantization through the convexity of softmax.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Reveals the fundamental problem of KV quantization from the perspective of Jensen's Inequality; the theoretical view is completely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Three video models $\times$ two quantization schemes + detailed ablations (attention shift, storage-quality tradeoff, cross-domain LLM).
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logical chain, interlocking from phenomenon $\to$ root cause $\to$ solution $\to$ verification.
- Value: ⭐⭐⭐⭐⭐  A plug-and-play solution with strong industrial usability; directly benefits all models using KV cache quantization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)
- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)

</div>

<!-- RELATED:END -->
