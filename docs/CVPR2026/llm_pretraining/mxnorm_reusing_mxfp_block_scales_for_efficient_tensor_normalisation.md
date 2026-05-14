---
title: >-
  [Paper Note] MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation
description: >-
  [CVPR 2026][LLM Pretraining][RMSNorm] GPU matrix multiplication throughput has improved 80x (V100 to GB200) while reduction/elementwise operations improved only 5-9x…
tags:
  - "CVPR 2026"
  - "LLM Pretraining"
  - "RMSNorm"
  - "MXFP8"
  - "Block Scales"
  - "Normalization"
  - "Low-Precision Training"
  - "LLM"
  - "Llama"
  - "Tensor Quantization"
  - "Kernel Optimization"
date: 2026-05-08
content_hash: d198f52ff726b0e3
---

# MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation

**Conference**: CVPR 2026  
**arXiv**: [2603.13180](https://arxiv.org/abs/2603.13180)  
**Code**: TBD  
**Area**: Efficient Training / Low-Precision Computation / Normalization  
**Keywords**: RMSNorm, MXFP8, Block Scales, Normalization, Low-Precision Training, LLM, Llama, Tensor Quantization, Kernel Optimization

## TL;DR

GPU matrix multiplication throughput has improved 80x (V100 to GB200) while reduction/elementwise operations improved only 5-9x, making RMSNorm a new bottleneck in low-precision training. MXNorm directly reuses the block scales already computed during MXFP8 quantization to estimate RMS, achieving a 32x reduction size decrease. Theorem 1 proves that the generalized $p$-mean of block absmax converges to a constant multiple of RMS. Llama 3 pretraining (125M/1B/8B) validates that MXNorm($p=2$) matches RMSNorm with minimal accuracy difference, with torch.compile benchmarks showing up to 2.4x isolated kernel speedup and +1.3%/+2.6% Llama 3 8B layer acceleration for MXFP8/NVFP4. Drop-in replacement with zero additional hyperparameters.

## Background & Motivation

The past 8 years have seen highly uneven GPU AI accelerator performance improvements:

| GPU | Low-Precision MatMul (TFLOPS) | CUDA Core (TFLOPS) | Memory BW (TB/s) |
|-----|-------------------------------|---------------------|-------------------|
| V100 | 125 (FP16) | 15.7 | 0.9 |
| A100 | 312 [2.5x] | 19.5 [1.2x] | 2.0 [2.2x] |
| H100 | 1979 [15.8x] | 67.0 [4.3x] | 3.4 [3.7x] |
| GB200 | 10000 [**80x**] | 80.0 [**5.1x**] | 8.0 [**8.9x**] |
| Rubin* | 35000 [280x] | 130.0 [8.3x] | 22.0 [24.4x] |

Matrix multiplication throughput has improved 80x in 8 years, while reduction/elementwise operations (limited by CUDA cores and memory bandwidth) improved only 5-9x. **This gap is accelerating** (Rubin generation: 280x vs 8.3x/24.4x).

In frontier LLMs like Llama, RMSNorm is widely used (Pre-Norm architecture) before every attention and FFN layer. RMSNorm's core operation is a reduction along the hidden dimension to compute root mean square—precisely a reduction operation that cannot be accelerated by matrix multiplication units. As matmul gets faster, non-matmul operations like RMSNorm become the new bottleneck.

**Key observation**: In Pre-Norm transformers, RMSNorm immediately precedes MXFP quantization. Both need to gather statistics along the hidden dimension to rescale elements—RMSNorm computes RMS, MXFP computes per-block absmax. **Can the block scales already computed by MXFP approximate RMS, avoiding redundant reduction?**

## Core Problem

How to fuse RMSNorm's reduction operation with MXFP8 quantization's block scale computation, using existing block-level statistics to approximate the global RMS, reducing normalization overhead with negligible training accuracy loss?

## Method

### Background: RMSNorm

Given tensor $X \in \mathbb{R}^{T \times D}$, RMSNorm computes per-row inverse RMS:

$$\rho_t = \left(\frac{1}{D}\sum_{d=1}^{D} X_{td}^2\right)^{-1/2}$$

Then normalizes and multiplies by learnable gain $\gamma$: $Y_{td} = \rho_t \cdot X_{td} \cdot \gamma_d$

**Bottleneck**: Computing $\rho_t$ requires reduction over $D$ elements (sum of squares), where $D$ is typically 4096-8192.

### Background: MXFP8 Quantization (MXCast)

The $D$ columns are divided into $K$ blocks of size $B$ ($B=32$, $K=D/B$). Per-block absmax: $m_{tk} = \max_b |Y_{tkb}|$, then power-of-2 scale: $S_{tk} = \text{cast}(m_{tk}/256; E8M0)$, and quantized values: $V_{tkb} = \text{cast}(Y_{tkb}/S_{tk}; E4M3)$.

**Key**: MXCast already computes $K$ block absmax values—these block-level statistics contain tensor scale information.

### MXNorm Core Idea

**Theorem 1** (Core theoretical guarantee): Let $X_i$ be $D=KB$ i.i.d. samples with block absmax $m_k$. Define the generalized $p$-mean:

$$G_K^{(p)} = \left(\frac{1}{K}\sum_{k=1}^{K} m_k^p\right)^{1/p}$$

Then as $K \to \infty$:

$$\frac{G_K^{(p)}}{\text{RMS}(X)} \to c(p, B)$$

i.e., the generalized $p$-mean of block absmax converges to a constant multiple of RMS that depends only on $p$, $B$, and the distribution shape.

**Intuition**: If the entire tensor is scaled by $\sigma$, both RMS and block absmax power mean are scaled by $\sigma$ → their ratio is constant.

### MXNorm Implementation

Estimate inverse RMS using the generalized $p$-mean of block absmax:

$$\tilde{m}_{tk} = \max_b |X_{tkb}|$$
$$\tilde{\rho}_t = \tilde{c}(p,B) \cdot \left(\frac{1}{K}\sum_{k=1}^{K} \tilde{m}_{tk}^p\right)^{-1/p} + \epsilon$$

where $\tilde{c}(p,B)$ is precomputed via Gaussian Monte Carlo sampling.

**Reduction size from $D$ to $K=D/B$**: With $B=32$, reduction size shrinks by 32x. For $D=4096$, RMSNorm reduces over 4096 elements; MXNorm reduces over only 128 block absmax values.

### MXNormLinear (Gain Parameter Handling)

MXNorm's output is in MXFP format (block scales + quantized values), making elementwise gain $\gamma$ multiplication inconvenient.

**Solution**: Leveraging associativity of linear operations, absorb $\gamma$ into the subsequent Linear layer's weight matrix:

$$H = \text{MXNorm}(X) \cdot \text{MXCast}(W \cdot \gamma)^\top$$

Backward pass uses RMSNorm's gradient as a straight-through estimator. Caches $X$ and $\tilde{\rho}$ for backward (same memory overhead as standard RMSNorm + MXCast).

### Choice of p: p=1 vs p=2

- **p=1** (arithmetic mean): Insensitive to outlier features; output upper bound is $O(K)$—too large
- **p=2** (RMS/quadratic mean): Output upper bound is $O(\sqrt{K})$, matching RMSNorm's $O(\sqrt{D})$ magnitude

Output upper bounds affect training stability: tighter bounds limit extreme values' impact on weight updates. MXNorm($p=1$) exhibits loss spikes at 8B scale, while MXNorm($p=2$) remains stable.

## Key Experimental Results

### Pretraining Stability (Learning Rate Sensitivity)

**125M & 1B models**: At optimal learning rates, all three approaches show minimal training loss differences:
- RMSNorm 125M: 3.090±0.004, 1B: 2.692±0.011
- MXNorm($p=1$) 125M: 3.113±0.012, 1B: 2.684±0.009
- MXNorm($p=2$) 125M: 3.116±0.010, 1B: 2.691±0.007

### 8B Model Pretraining (300B tokens on SlimPajama)

| Approach | Final Loss |
|----------|-----------|
| RMSNorm | 2.132 |
| MXNorm($p=1$) | 2.175 (significantly worse, with loss spikes) |
| MXNorm($p=2$) | **2.126** (nearly identical to RMSNorm) |

MXNorm($p=2$) slightly outperforms RMSNorm (2.126 vs 2.132). MXNorm($p=1$) falls behind due to loss spikes caused by excessive output upper bounds allowing outlier features.

### Zero-Shot Downstream Evaluation (OLMES, 10 NLP Tasks)

| Approach | Tasks Won |
|----------|-----------|
| RMSNorm | 5/10 |
| MXNorm($p=2$) | **5/10** |
| MXNorm($p=1$) | 0/10 |

MXNorm($p=2$) ties with RMSNorm on downstream zero-shot performance (each winning 5 benchmarks), indicating fully comparable training quality.

### Kernel Speedup (torch.compile, GB200)

**Isolated kernel benchmark** (MXNorm vs RMSNorm+MXCast fused):
- MXFP8 (B=32, E4M3): Average speedup **41.7%**, maximum **2.4x**
- NVFP4 (B=16, E2M1): Average speedup **31.2%**
- Speedup increases with token count and stabilizes with hidden dimension

**Full transformer layer benchmark** (Llama 3 8B, 8 transformer layers on GB200):
- MXFP8: Geometric mean speedup **+1.3%**
- NVFP4: Geometric mean speedup **+2.6%**

As precision decreases (NVFP4 < MXFP8), matmul gets faster and normalization accounts for a larger share → MXNorm's speedup advantage grows.

### Approximation Quality

MXFP dequantized tensor's $r^2$ goodness-of-fit asymptotically approaches 1 with increasing block count. At hidden dim ≥ 1024 (32 blocks), approximation quality is already excellent. Scale and value tensor distributions are nearly identical to RMSNorm+MXCast.

## Highlights & Insights

- **Elegant "free lunch" design**: MXNorm reuses MXFP8 quantization's existing block absmax computation without introducing any new statistical collection—fusing two previously independent reductions into one, a classic compute reuse approach
- **Rigorous theoretical guarantee**: Theorem 1 provides a strict convergence proof (via strong law of large numbers + continuous mapping theorem), establishing a theoretical foundation beyond mere empirical effectiveness
- **Deep $p=2 > p=1$ insight**: Beyond approximation accuracy, the output upper bound ($O(\sqrt{K})$ vs $O(K)$) determines training stability. This analysis offers a new perspective for normalization layer design: bounds matter more than approximation quality
- **True drop-in replacement**: No new hyperparameters, no training pipeline modifications, backward uses RMSNorm gradient as STE, gain parameter absorbed into weight matrix → minimal engineering integration cost
- **Forward-looking**: As the matmul vs non-matmul performance gap continues widening across GPU generations (Rubin: 280x vs 8.3x), MXNorm's value will keep growing

## Limitations & Future Work

1. **Limited end-to-end gains**: Llama 3 8B full layer speedup is only +1.3% (MXFP8)/+2.6% (NVFP4)—since normalization currently accounts for a small fraction of matmul-dominated architectures. This fraction will grow with hardware generations
2. **Only validated on Pre-Norm architecture**: RMSNorm preceding Linear is MXNorm's prerequisite; Post-Norm or other normalization placements are not applicable
3. **Only applicable to block quantization formats**: MXNorm relies on MXFP block scales; per-tensor or per-channel quantization (FP8) lacks block absmax
4. **Gaussian distribution assumption**: $\tilde{c}(p,B)$ is precomputed via Gaussian Monte Carlo; actual activation distributions may deviate from Gaussian (especially with outlier features), though experiments show sufficient robustness
5. **MoE architectures not covered**: Normalization placement and behavior in Mixture-of-Experts models may differ and requires additional validation
6. **Other non-matmul bottlenecks unaddressed**: RoPE, gated linear units, and similar operations are also constrained by non-matmul performance, left for future work

## Related Work & Insights

- **FlashNorm**: Asynchronously computes RMS then uses raw input for matmul → risks accumulator swamping; MXNorm avoids this via block scales
- **Partial RMS**: Uses only the first $k$ elements to estimate RMS → easily misses outliers; MXNorm's block absmax naturally covers all elements
- **Training without normalization** (e.g., nGPT constraining weights to hypersphere, tanh replacing normalization): Introduces other overhead (slower optimizer steps, elementwise special functions); MXNorm preserves normalization at lower cost
- MXNorm's unique advantage is "piggybacking" on existing quantization computation—theoretically zero additional overhead
- **Broader inspiration**: The "reuse intermediate results from one operation as input to another" approach can generalize: e.g., attention's softmax also requires max/sum reduction—could it be fused with quantization?
- The relationship between normalization output upper bounds and training stability deserves further investigation, potentially inspiring new adaptive normalization designs

## Rating

| Dimension | Score (1-5) | Note |
|-----------|-------------|------|
| Novelty | 4.0 | Core idea is elegantly simple (reuse block scales); engineering-oriented incremental innovation |
| Practicality | 4.5 | True drop-in replacement with zero extra hyperparameters; measurable speedup that grows with hardware updates |
| Experimental Thoroughness | 4.5 | Three scales (125M→1B→8B) + LR sweep + stability analysis + kernel benchmark + OLMES, very comprehensive |
| Writing Quality | 4.5 | GPU performance table motivation is compelling; theory and experiments tightly connected; detailed appendix with PyTorch implementation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Block-Sample MAC-Bayes Generalization Bounds](../../ICLR2026/llm_pretraining/block-sample_mac-bayes_generalization_bounds.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](../../ICCV2025/llm_pretraining/make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](../../NeurIPS2025/llm_pretraining/efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)

</div>

<!-- RELATED:END -->
