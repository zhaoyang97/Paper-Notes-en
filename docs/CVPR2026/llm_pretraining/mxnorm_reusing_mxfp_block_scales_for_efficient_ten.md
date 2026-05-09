---
title: >-
  [Paper Note] MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation
description: >-
  [CVPR 2026][LLM Pretraining][RMSNorm] MXNorm fuses RMSNorm with MXFP quantization by reusing the block absmax values already computed during MXFP8 quantization to approximate the RMS value, eliminating the separate normalization reduction operation. It maintains training accuracy on Llama 3 up to 8B parameters while achieving up to 2.4x kernel speedup on GB200.
tags:
  - CVPR 2026
  - LLM Pretraining
  - RMSNorm
  - MXFP Quantization
  - Low-Precision Training
  - Normalization Fusion
date: 2026-05-08
content_hash: 6dd08c82bebb787f
---

# MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation

**Conference**: CVPR 2026  
**arXiv**: [2603.13180](https://arxiv.org/abs/2603.13180)  
**Code**: N/A (Graphcore internal implementation, based on TorchTitan + TorchAO)  
**Area**: Pretraining  
**Keywords**: RMSNorm, MXFP Quantization, Low-Precision Training, Normalization Fusion, LLM Pretraining

## TL;DR
MXNorm fuses RMSNorm with MXFP quantization by reusing the block absmax values already computed during MXFP8 quantization to approximate the RMS value, eliminating the separate normalization reduction operation. It maintains training accuracy on Llama 3 up to 8B parameters while achieving up to 2.4x kernel speedup on GB200.

## Background & Motivation
Over recent years, AI accelerator performance improvements have been highly uneven: low-precision matrix multiplication throughput has improved 80x from V100 to GB200, while reduction/elementwise operation bandwidth has only improved 5-9x. This gap causes operations that were previously non-bottlenecks (such as normalization layers) to become new performance bottlenecks. RMSNorm is the standard normalization layer in mainstream LLMs like the Llama series, requiring a reduction across the entire hidden dimension to compute the RMS value. In Pre-Norm transformers, RMSNorm is immediately followed by MXFP quantization (MXCast), and MXCast itself also computes block absmax—both perform similar statistical aggregation. This computational redundancy presents an optimization opportunity.

## Core Problem
Matrix multiplication acceleration far exceeds the speedup of other operations, making the reduction operation in RMSNorm a new bottleneck for low-precision transformer training. Can the block-level statistics already computed during MXFP quantization replace RMSNorm's full-dimension reduction, thereby fusing normalization and quantization?

## Method

### Overall Architecture
In the typical "RMSNorm → Linear" pattern of Pre-Norm transformers, MXNorm fuses RMSNorm and MXCast into a single operation. The input is a BF16 activation tensor $X \in \mathbb{R}^{T \times D}$; the output is an MXFP8 format $(S, V)$ tuple (block scales + quantized values). The key idea is approximating RMS using a generalized $p$-mean of block absmax values, eliminating the full $D$-dimension reduction step and requiring reduction over only $K = D/B$ block absmax values ($B=32$ gives a 32x reduction).

### Key Designs
1. **Block Absmax Approximates RMS (Theorem 1)**: The paper rigorously proves that for i.i.d. elements from a scale-family distribution, the ratio of the generalized $p$-mean of block absmax to RMS converges almost surely to a constant $c(p,B)$ that depends only on $p$, $B$, and the distribution shape as $K \to \infty$. A correction coefficient $\tilde{c}(p,B)$ (estimated via Gaussian Monte Carlo) can linearly scale the block absmax $p$-mean to approximate RMS.

2. **MXNorm Forward Computation**: After blocking the input tensor, compute block absmax $\tilde{m}_{tk}$, then estimate inverse RMS: $\tilde{\rho}_t = \tilde{c}(p,B) \cdot (\frac{1}{K}\sum_k \tilde{m}_{tk}^p)^{-1/p}$. Use $\tilde{\rho}_t$ to directly adjust block scales and values, completing normalization + quantization in one step. This reduces the reduction size from $D$ (full hidden dimension) to $K = D/B$ (number of blocks), a $B$-fold decrease (typically $B=32$).

3. **Stability Difference between p=2 vs p=1**: $p=1$ (arithmetic mean) and $p=2$ (RMS mean) perform similarly on small models, but at 8B scale, $p=1$ exhibits loss spikes causing final performance degradation. The analysis reveals that MXNorm($p=1$) has an output upper bound of $O(K)$, while $p=2$ has $O(\sqrt{K})$—the looser bound allows outlier features to trigger training instability. This finding also reveals the critical role of normalization's "truncation" effect on training stability.

4. **MXNormLinear (Gain Parameter Handling)**: RMSNorm has a learnable gain $\gamma$, but elementwise broadcasting on MX-quantized tensors is inconvenient. Leveraging the associativity of linear operations, $\gamma$ is absorbed into the subsequent Linear layer's weights: $H = \text{MXNorm}(X) \cdot \text{MXCast}(W \gamma)^\top$. This can be pre-fused at inference; during training, gradients are handled separately.

### Loss & Training
- Backward pass uses RMSNorm's gradient as a straight-through estimator (STE), ensuring gradient smoothness
- Forward: MXNorm; backward: RMSNorm gradient—no additional hyperparameters introduced
- Backward requires caching input $X$ and inverse RMS estimate $\tilde{\rho}$, but this matches the caching requirements of standard RMSNorm + MXCast

## Key Experimental Results

| Model Scale | Metric | RMSNorm | MXNorm(p=1) | MXNorm(p=2) |
|------------|--------|---------|-------------|-------------|
| 125M | Training Loss | 3.090±0.004 | 3.113±0.012 | 3.116±0.010 |
| 1B | Training Loss | 2.692±0.011 | 2.684±0.009 | 2.691±0.007 |
| 8B (300B tokens) | Training Loss | 2.132 | 2.175 | 2.126 |

| Setting | Metric | MXNorm vs RMSNorm Speedup |
|---------|--------|--------------------------|
| Kernel (MXFP8, B=32) | Geometric Mean Speedup | 41.7% |
| Kernel (NVFP4, B=16) | Geometric Mean Speedup | 31.2% |
| Kernel (max) | Single Speedup | 2.4× |
| Llama 3 8B Layer (MXFP8) | End-to-End Layer Speedup | 1.3% |
| Llama 3 8B Layer (NVFP4) | End-to-End Layer Speedup | 2.6% |

8B OLMES zero-shot evaluation: MXNorm(p=2) wins 5 out of 10 tasks, RMSNorm wins 5—overall tied.

### Ablation Study
- $p=1$ exhibits loss spikes at 8B scale; final loss is significantly worse than $p=2$ and RMSNorm
- $p=2$ matches the RMSNorm baseline at all model scales
- Post-Round MXNorm (using E8M0-rounded scales to estimate RMS) is severely unstable at 8B scale, showing that rounding-induced quantization noise destroys approximation quality
- Approximation quality improves asymptotically with increasing block count; 1024 elements (32 blocks) already yield good $r^2$
- Output upper bound is the key indicator for training stability: $O(\sqrt{K})$ for $p=2$ significantly outperforms $O(K)$ for $p=1$

## Highlights & Insights
- **Minimalist design**: Zero additional hyperparameters as a drop-in replacement; only the normalization layer implementation changes, not the architecture
- **Theory-practice loop**: From rigorous mathematical proof (Theorem 1 + upper bound analysis) to actual kernel speedup, every step is theoretically supported
- **Deep analysis of training stability**: The $p=1$ vs $p=2$ comparison reveals the importance of normalization's "truncation effect" in suppressing outlier features—an insight that transcends MXNorm itself
- **Applicable to lower precision formats**: MXNorm's advantage grows with decreasing precision (FP4, INT2, ternary), making the direction very forward-looking

## Limitations & Future Work
- End-to-end speedup is relatively limited (1.3% for MXFP8 on Llama 3 8B layers), as normalization accounts for a small fraction of matmul-dominated architectures
- Only validated on Pre-Norm transformer "Norm → Linear" patterns; Post-Norm or other normalization placements are unexplored
- Only validated on language models; vision transformers and VLM scenarios are not covered
- Joint fusion optimization with other non-matmul operations (RoPE, gated linear units, etc.) is not explored
- Requires MX format hardware support (currently limited to Blackwell and later GPUs)

## Related Work & Insights
- **FlashNorm** (asynchronous RMS computation + matmul with unnormalized input): risks accumulator overflow; MXNorm is safer via block scales
- **Partial RMS** (using only the first $k$ elements to estimate RMS): easily misses outliers; MXNorm uses block absmax covering all elements
- **Training without normalization** (e.g., nGPT constraining weights to hypersphere, tanh replacing normalization): introduces additional overhead; MXNorm maintains normalization functionality at lower cost
- MXNorm's unique advantage is "piggybacking" on existing quantization computation—theoretically zero additional overhead

## Rating
- Novelty: ⭐⭐⭐⭐ — Core idea is elegantly simple (reuse block scales); engineering-oriented incremental innovation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three scales (125M/1B/8B) + LR sweep + stability analysis + kernel benchmark + OLMES evaluation, very comprehensive
- Writing Quality: ⭐⭐⭐⭐⭐ — GPU performance table motivating the problem is compelling; theory and experiments are tightly connected
- Value: ⭐⭐⭐⭐ — Practical contribution to the low-precision training ecosystem, though limited by MX format hardware adoption and end-to-end speedup magnitude

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)
- [\[CVPR 2026\] Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](evidential_transformation_network_post_hoc_uncertainty_estimation.md)
- [\[CVPR 2026\] Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)
- [\[CVPR 2026\] Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)
- [\[CVPR 2026\] Linking Modality Isolation in Heterogeneous Collaborative Perception](linking_modality_isolation_in_heterogeneous_collaborative_perception.md)

</div>

<!-- RELATED:END -->
