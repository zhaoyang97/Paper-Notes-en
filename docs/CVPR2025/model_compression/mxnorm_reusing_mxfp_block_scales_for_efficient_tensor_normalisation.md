---
title: >-
  [Paper Note] MXNorm: Reusing MXFP block scales for efficient tensor normalisation
description: >-
  [CVPR2025][Model Compression][RMSNorm] MXNorm proposes to reuse the block absmax already computed during MXFP quantization to approximate RMS, fusing normalization and MX quantization into a single statistics gathering operation. This achieves a drop-in replacement for RMSNorm, obtaining up to a 2.4× kernel speedup while maintaining training accuracy in Llama 3 8B pre-training.
tags:
  - "CVPR2025"
  - "Model Compression"
  - "RMSNorm"
  - "MXFP8"
  - "Low-precision Training"
  - "Quantization-aware Training"
  - "LLM Pre-training"
  - "Normalization Acceleration"
date: 2026-05-08
content_hash: 04e831d7f0324192
---

# MXNorm: Reusing MXFP block scales for efficient tensor normalisation

**Conference**: CVPR2025  
**arXiv**: [2603.13180](https://arxiv.org/abs/2603.13180)  
**Code**: No public code (Graphcore internal implementation, PyTorch pseudocode in appendix)  
**Area**: Efficient Training / Quantization  
**Keywords**: RMSNorm, MXFP8, Low-precision Training, Quantization-aware Training, LLM Pre-training, Normalization Acceleration

## TL;DR
MXNorm proposes to reuse the block absmax already computed during MXFP quantization to approximate RMS, fusing normalization and MX quantization into a single statistics gathering operation. This achieves a drop-in replacement for RMSNorm, obtaining up to a 2.4× kernel speedup while maintaining training accuracy in Llama 3 8B pre-training.

## Background & Motivation
**Background**: Over the past 8 years, GPU matrix multiplication acceleration has increased by 80× (V100 $\rightarrow$ GB200), but memory bandwidth has only increased by 8.9×, and CUDA cores by only 5.1×. With the popularization of low-precision matrix multiplications such as MXFP8/FP4, non-matrix multiplication operations (normalization, element-wise computation, reduction) are becoming the new throughput bottlenecks.

**Key Challenge**: Pre-Norm Transformers (such as the Llama family) perform RMSNorm before each QKV projection and FFN projection. RMSNorm requires reduction over the entire hidden dimension (computing the mean square), followed immediately by MX quantization on the same tensor (computing block absmax) — these are two independent statistics gathering operations.

**Key Insight**: Both RMSNorm and MXCast gather statistics along the hidden dimension to scale elements. When a probability distribution is linearly scaled, its expected absmax is also scaled proportionally — therefore, the generalized power mean of block absmax can be used to estimate RMS.

**Goal**: Fuse normalization and quantization into a single operation, reducing the reduction size by 32× (from $D$ dimensions down to $K=D/B$ block absmax).

## Method

### Core Idea
For input tensor $X \in \mathbb{R}^{T \times K \times B}$ ($T$ tokens, $D=KB$ dimensions divided into $K$ blocks of size $B$):
- **RMSNorm** requires computing the mean square of all $KB$ elements: $\rho_t = (\frac{1}{KB}\sum_{kb} X_{tkb}^2)^{-1/2}$
- **MXCast** requires computing the absmax for each block: $m_{tk} = \max_b |X_{tkb}|$
- **MXNorm** reuses block absmax, estimating the RMS using its generalized $p$-mean: $\tilde{\rho}_t = c^{(p,B)} \cdot (\frac{1}{K}\sum_k m_{tk}^p)^{-1/p}$

Where $c^{(p,B)}$ is a correction constant depending on $p$, $B$, and the normalized distribution, estimated through Monte Carlo sampling from a Gaussian distribution.

### Theoretical Guarantee (Theorem 1)
When the input originates from a scale-family distribution (such as a Gaussian distribution), the ratio of the generalized $p$-mean of block absmax to RMS converges almost surely to a constant $c^{(p,B)}$. This implies that only a multiplicative correction factor is needed to convert block absmax statistics into an approximation of RMS.

### MXNormLinear
- The gain parameter $\gamma$ of normalization cannot be directly applied to tensors in MX format.
- Utilizing the associative property of linear operations, $\gamma$ is incorporated into the weights of the subsequent linear layer: $H = \text{MXNorm}(X) \cdot \text{MXCast}(W\gamma)^\top$
- Backpropagation uses the gradient calculation of RMSNorm as a straight-through estimator.

### Choice of $p$-value
- **p=1** (arithmetic mean): The reduction is simpler, but the upper bound of the output is $O(K)$, making it insufficiently robust to outliers.
- **p=2** (RMS mean): The upper bound of the output is $O(\sqrt{K})$, which is of the same order as RMSNorm's $O(\sqrt{D})$, providing tighter range constraints.

### Output Upper Bound Analysis
- RMSNorm output upper bound: $\|x\|_\infty \leq \sqrt{D}$
- MXNorm(p=2) output upper bound: $\|x\|_\infty \leq \sqrt{K}/c$
- MXNorm(p=1) output upper bound: $\|x\|_\infty \leq K/c$
- Tighter upper bound constraints are crucial for training stability.

## Key Experimental Results

### Llama 3 Pre-training Comparison (SlimPajama Dataset)

| Model Size | RMSNorm Loss | MXNorm(p=1) Loss | MXNorm(p=2) Loss |
|---------|-------------|-----------------|-----------------|
| 125M | 3.090±0.004 | 3.113±0.012 | 3.116±0.010 |
| 1B | 2.692±0.011 | 2.684±0.009 | 2.691±0.007 |
| 8B (300B tokens) | 2.132 | 2.175 (❌ significantly worse) | **2.126** (✅ matched) |

### Zero-Shot Downstream Tasks (OLMES, 8B Model)
- MXNorm(p=2) leads RMSNorm on 5/10 tasks, and RMSNorm leads on 5/10 tasks.
- MXNorm(p=1) falls behind significantly on most tasks (ARC-C: 39.1 vs 45.3, BoolQ: 56.6 vs 73.2).

### Kernel Speedup (GB200, torch.compile)
- Maximum isolated kernel speedup: **2.4×**
- MXFP8 (B=32) average speedup: 41.7%
- NVFP4 (B=16) average speedup: 31.2%
- Llama 3 8B Transformer layer end-to-end speedup: MXFP8 **1.3%**, NVFP4 **2.6%**

### Training Stability Findings
- MXNorm(p=1) exhibits loss spikes during 8B scale training, triggered by outlier features.
- The root cause is not inaccurate RMS estimation (it is highly accurate prior to the spike), but rather loose output range constraints ($O(K)$ vs $O(\sqrt{K})$). This leads to excessively large weight updates triggered by outlier features, which compound and accumulate.

### Approximation Quality Verification
- For inputs sampled from Gaussian distributions, the MX scale and value distributions generated by MXNorm are almost identical to those of RMSNorm+MXCast.
- The dequantized $r^2$ goodness-of-fit asymptotically approaches 1.0 as the number of blocks increases.
- Outstanding approximation quality is achieved with only 1024 elements (32 blocks of size 32).

### Comparison with Existing Accelerated Normalization Methods
- **nGPT**: Eliminates normalization by constraining weights to a hypersphere, but introduces extra overhead in optimization steps.
- **FlashNorm**: Computes RMS asynchronously and multiplies weights by raw inputs, but suffers from potential accumulator overflow risks.
- **Partial Element RMS**: Estimates RMS using only the first $k$ elements, which easily misses outlier values.
- **Advantages of MXNorm**: Does not modify the training recipe, carries no overflow risks, and leverages pre-existing computations rather than introducing approximate sampling.

## Highlights & Insights
- **Model of Theoretical-Engineering Synergy**: Complete chain of logic from mathematical theorems (Theorem 1 proving that the power mean of absmax can estimate RMS) to practical kernel implementations.
- **Precise Problem Definition**: Identifies the statistical gathering redundancy between normalization and quantization, and proposes a fusion scheme to reduce reduction scale by 32×.
- **In-Depth Analysis of $p=1$ vs $p=2$**: Not only demonstrates that $p=2$ is superior but also explains *why* $p=1$ fails through output upper-bound theory and loss spike analyses. This provides universal value for understanding normalisation layer stability mechanisms.
- **No Extra Hyperparameters**: MXNorm is a drop-in replacement for RMSNorm, with the correction constant $c$ automatically estimated from the distribution.
- **Forward-Looking Perspective**: Clearly points out that as lower precision formats like NVFP4/INT2 become more pervasive, non-matmul bottlenecks will intensify, amplifying the value of MXNorm.

## Limitations & Future Work
- Limited end-to-end speedup: Only 1.3% to 2.6% at the Llama 3 8B layer level, as normalization itself accounts for a minor portion of the total computation.
- Validated only on the Llama 3 architecture and the SlimPajama dataset, without testing other architectures (e.g., MoE, Vision Transformer).
- Theory relies on the assumption of a Gaussian distribution, and scenarios where activation distributions deviate from Gaussian in the later stages of training are not fully discussed.
- Backpropagation uses the straight-through estimator of RMSNorm gradients, introducing approximation errors.
- Currently supports only the Norm $\rightarrow$ Linear pattern in Pre-Norm architectures.
- High computational resource requirements: The 8B experiment required training on 64 H100 GPUs for 8 days.
- The impact and utility of MXNorm on the inference phase (non-training) are not discussed.

## Rating
- Novelty: ⭐⭐⭐⭐ Unique perspective in observing that the statistical collection of normalization and quantization can be reused.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation from 125M to 8B, including learning rate sweeps, stability analyses, and kernel benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and clear theoretical derivations; the diff-comparison design of Algorithm 1/2 is elegant.
- Value: ⭐⭐⭐⭐ The value of the interface will continue to grow alongside the adoption of low-precision formats, but current end-to-end gains remain limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Beyond Text Compression: Evaluating Tokenizers Across Scales](../../ACL2025/model_compression/beyond_text_compression_tokenizers.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ICLR 2026\] Entropy-Based Block Pruning for Efficient Large Language Models](../../ICLR2026/model_compression/entropy-based_block_pruning_for_efficient_large_language_models.md)
- [\[ICML 2025\] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference](../../ICML2025/model_compression/blockdialect_block-wise_fine-grained_mixed_format_quantization_for_energy-effici.md)
- [\[ICLR 2026\] TRAC: Tensor-Train Based Across-Layer Compression for Parameter-Efficient Fine-Tuning](../../ICLR2026/model_compression/trac_tensor-train_based_across-layer_compression_for_parameter-efficient_fine-tu.md)

</div>

<!-- RELATED:END -->
