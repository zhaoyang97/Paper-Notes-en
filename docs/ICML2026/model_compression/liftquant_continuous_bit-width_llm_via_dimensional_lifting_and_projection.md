---
title: >-
  [Paper Note] LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection
description: >-
  [ICML 2026][Model Compression][Continuous bit-width] LiftQuant decouples LLM quantization bit-width from discrete integers (2/3/4 bit) into continuous fractions (e.g.…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Continuous bit-width"
  - "lift-then-project"
  - "high-dimensional projection"
  - "1-bit lattice"
  - "Pareto-optimal deployment"
date: 2026-05-08
content_hash: 2f8f37db8be4a060
---

# LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection

**Conference**: ICML 2026  
**arXiv**: [2606.04050](https://arxiv.org/abs/2606.04050)  
**Code**: https://github.com/Heliulu/LiftQuant  
**Area**: Model Compression / LLM Quantization / Deployment Optimization  
**Keywords**: Continuous bit-width, lift-then-project, high-dimensional projection, 1-bit lattice, Pareto-optimal deployment

## TL;DR
LiftQuant decouples LLM quantization bit-width from discrete integers (2/3/4 bit) into continuous fractions (e.g., 2.4-bit) through a "high-dimensional 1-bit lattice $\rightarrow$ low-dimensional weight space projection" (lift-then-project) mechanism. This allows a 70B model to fit precisely into a 24GB GPU with PPL significantly better than the 2-bit baseline. The entire decoding path utilize only linear transformations and 1-bit uniform quantizers, making it hardware-friendly.

## Background & Motivation

**Background**: Weight-only quantization is essential for LLM deployment. Two main schools exist: Uniform Quantization (UQ) (including AWQ, QLoRA, QuIP#, QuaRot, SpinQuant, etc., which use INT2/3/4 after preprocessing) and Vector Quantization (VQ) (including AQLM, VPTQ, QTIP, which use learned codebooks for higher accuracy but require slower LUT inference).

**Limitations of Prior Work**: (1) All conventional methods are locked into integer bit-widths—for instance, Llama-3-70B cannot fit on a 24GB card at 3-bit, while 2-bit leads to catastrophic PPL drops, wasting the memory headroom between 2-3 bits. (2) UQ allows coarse adjustments via group size (e.g., EfficientQAT moving from 128 to 64), but these are discrete "steps" rather than continuous values. (3) Non-power-of-two codebooks (e.g., ternary 1.58-bit) require specialized kernels. (4) Q-Palette achieves fractional bits by mixing multiple quantizers, but maintaining a library of heterogeneous kernels is engineering-heavy.

**Key Challenge**: Hardware budgets are continuous (24GB, 12GB, etc.), whereas model bit-widths are discrete (2, 3, 4), causing memory utilization to be permanently suboptimal. Simultaneously, VQ offers high accuracy but slow LUTs, while UQ is fast but less accurate; this "accuracy vs. speed" trade-off becomes even more difficult at fractional bit-widths.

**Goal**: (1) Transition bit-width from integers to continuous fractions (2.0, 2.4, 2.5, ...) to precisely match hardware budgets. (2) Maintain VQ-level accuracy while enjoying UQ-level hardware friendliness. (3) Use a unified operator for all bit-widths to avoid per-bit kernel development.

**Key Insight**: It is observed that by projecting a high-dimensional ($\mathbb{R}^D$) 1-bit lattice ($\{\pm 1\}^D$, 1 bit each) onto a lower-dimensional weight space ($\mathbb{R}^d$, $d < D$) via a matrix $\bm M$, the effective bit-width is the ratio $D/d$. Since $D$ and $d$ are flexible structural parameters, this ratio can be any fraction. According to the Central Limit Theorem (CLT), projections of high-dimensional lattices naturally form Gaussian-like dense codebooks—gaining VQ expressivity while using hardware-friendly 1-bit operators for decoding.

**Core Idea**: Lift-then-project — weights are represented as $\bm w \simeq \bm M \bm w_q$, where $\bm M$ is a learned global projection matrix and $\bm w_q \in \{\pm 1\}^D$ is a 1-bit quantized vector; the bit-width = $D/d$ is continuously adjustable.

## Method

### Overall Architecture

Three steps:
1. **Learn Projection Matrix $\bm M$**: Defines global mapping structure and fractional bit-width.
2. **Layer-wise Whitening Transform $\bm T$**: Reshapes weights into i.i.d. Gaussian distributions to satisfy projection assumptions.
3. **Quantization + Decoding Pipeline**: Fused as $\bm o = \text{diag}(\bm s) \bm W (\bm M \bm T \bm a)$, utilizing only linear transforms and a 1-bit quantizer.

Notation: LQ-$D/d$ (e.g., LQ-24/10 = 2.4-bit).

### Key Designs

1. **Lift-then-Project Quantization Mechanism**:
    - **Function**: Transitions bit-width from discrete integers to continuous fractions $D/d$.
    - **Mechanism**: Weight $\bm w \in \mathbb{R}^d$ is represented as $\bm w \simeq \bm M \bm w_q$, with $\bm w_q \in \{\pm 1\}^D$ (1-bit lattice) and $\bm M \in \mathbb{R}^{d \times D}$ (projection matrix). Each $w_i = \sum_j \bm M_{ij} \bm y_j$ is a sum of independent random variables; CLT ensures the projection naturally forms a Gaussian-like dense codebook. The effective bit-width = $D/d$ is continuously adjustable.
    - **Design Motivation**: Traditional VQ learns codebooks directly on $\mathbb{R}^d$, limited by codebook size. UQ with scalar quantization is less flexible. This method decouples "codebook design" from "bit-width"—the former is controlled by the structure of $\bm M$, and the latter by the ratio $D/d$.

2. **Optimizing $\bm M$ + Accelerating Nearest Neighbor Search**:
    - **Function**: Optimizes $\bm M$ for Gaussian weights and ensures nearest neighbor search (NNS) completes within reasonable time.
    - **Mechanism**: $\bm M^* = \arg\min_{\bm M} \mathbb{E}_{\bm w \sim \mathcal{N}}[\min_{\bm w_q \in \{\pm 1\}^{d_s \cdot b}} \|\bm w - \bm M \bm w_q\|]$. $\bm M$ is initialized as an orthogonal matrix to encourage uncorrelated projections. Differentiable optimization is achieved via soft-argmin with temperature 10. While NNS has exponential complexity $2^D$, this work reduces the search space to $2^{D-d}$ via pseudo-inverse projection and auxiliary vector padding.
    - **Design Motivation**: CLT guarantees asymptotic Gaussianity, but explicit optimization of $\bm M$ is necessary for finite $D$. Without acceleration, NNS for $D \geq 24$ is impractical; pseudo-inverse initialization provides a high-quality starting point making local searches of $2^{D-d}$ viable.

3. **Layer-wise Whitening Transform $\bm T$ + Fused Inference**:
    - **Function**: Reshapes actual weights into i.i.d. Gaussian to meet projection assumptions; integrates dequantization into GEMM.
    - **Mechanism**: A lightweight whitening matrix $\bm T$ transforms weights into approximate Gaussian distributions per layer. During inference, $\bm o = \text{diag}(\bm s) \bm W (\bm M \bm T \bm a)$, where $\bm M$ and $\bm T$ are small, globally shared matrices (calculated as $\bm M \bm T \bm a$), and $\bm W$ is the GEMM of 1-bit quantized matrices.
    - **Design Motivation**: LLM weights are not perfectly Gaussian; layer-wise whitening ensures the lift-then-project premise holds. Fused inference makes dequantization virtually overhead-free—requiring only one extra small matrix multiplication and scaling.

## Key Experimental Results

### Llama-2-7B Wikitext-2 PPL (Standard Gaussian Source)

| Encoding | bits | MSE | Info | PPL | Search Time (1M params) |
|------|------|-----|------|-----|----|
| LQ-32/20 | 1.60 | 0.146 | 1.39 | 7.71 | 0.3s |
| **LQ-16/8** | **2.00** | 0.089 | 1.75 | **6.60** | ≪0.1s |
| LQ-32/16 | 2.00 | 0.082 | 1.79 | 6.53 | 4s |
| **LQ-30/14** | **2.14** | **0.070** | 1.92 | **6.30** | 4s |
| **LQ-24/10** | **2.40** | **0.053** | 2.12 | **6.10** | 1s |
| Int2 | 2.00 | 0.119 | 1.53 | 7.62 | – |
| E8 (QuIP#) | 2.00 | 0.089 | 1.75 | 6.60 | – |
| TCQ (QTIP) | 2.00 | 0.073 | 1.89 | 6.28 | – |

At exactly 2.00 bit, LQ is slightly weaker than QTIP (TCQ is more efficient at 64 dimensions); however, with a small increase to 2.14 bit, LQ surpasses QTIP. At 2.4 bit, the PPL of 6.10 significantly outperforms all 2-bit baselines.

### Pareto Deployment of 70B Model on 24GB GPU

| Method | bits | Memory (GB) | WikiText-2 PPL | C4 PPL |
|------|------|--------|--------------|--------|
| QTIP 2-bit | 2.00 | 17.5 | 5.21 | 6.94 |
| EfficientQAT 2-bit | 2.00 | 18.0 | 5.45 | 7.18 |
| QTIP 3-bit | 3.00 | 26.3 | – | – | (OOM) |
| **LQ-24/10 (2.4-bit)** | **2.40** | **23.6** | **4.92** | **6.51** |

LQ precisely compresses the 70B model to fit 24GB, achieving significantly lower PPL than 2-bit baselines, whereas 3-bit results in Out-of-Memory (OOM).

### A similar scenario for a 32B model on 12GB GPU shows LQ-20/8 (2.5-bit) perfectly filling the budget with better PPL than 2-bit baselines.

### Key Findings
- **Continuous bit-width unlocks the Pareto frontier**: Increasing 2-bit to 2.4-bit (extra 0.4 bit) yields substantial PPL improvements, which is impossible with integer bit-widths.
- **Both CLT guarantees and explicit $\bm M$ optimization are necessary**: CLT provides the direction, but explicit optimization of $\bm M$ is required for finite $D$ to match QTIP's performance.
- **2-3 bits is the "sweet spot" for LiftQuant**: Above 4-bit, quantization is nearly lossless, making fractional adjustments less beneficial. This work focuses on the 2-3 bit gap where hardware budget misalignment is greatest.
- **Search complexity is manageable**: Searches can be completed in seconds when $D-d \leq 20$, practical for real-world use with pseudo-inverse initialization.

## Highlights & Insights
- **Decoupling bit-width from coding format is a true paradigm breakthrough**: Previous quantization methods tied these together (codebook size determined bit-width). This work separates them via dimensional lifting and projection—a concept extendable to all codebook design problems.
- **CLT serves as the bridge between 1-bit lattices and Gaussian codebooks**: Projecting a high-dimensional lattice naturally produces a Gaussian-like distribution, perfectly matching the Gaussian nature of LLM weights—an elegant alignment of theory and engineering.
- **The cost of hardware friendliness is 0.1 bit**: Compared to the complex Trellis Codes of QTIP, LQ uses only linear transforms and 1-bit operators, making it engineering-simple. The trade-off is a slight performance drop at the same bit-width, which is overcome by adding just 0.1 bit. This "0.1 bit for engineering simplicity" trade-off is highly practical.
- **70B deployment on 24GB is an industrial killer-app**: The demand for running 70B models on consumer-grade GPUs is urgent. This paper provides a ready-to-use solution that can be immediately applied.

## Limitations & Future Work
- Nearest neighbor search is still $2^{D-d}$, only practical for $D-d \leq 20$. Achieving higher coding gains (e.g., $D \geq 64$ like QTIP) requires more efficient searching.
- In scenarios above 4-bit where $d \leq 6$, high-dimensional inter-channel correlation is lost—a common constraint for all VQ methods.
- Whitening matrix $\bm T$ is learned layer-wise; significant distribution shifts might require recalibration via a calibration set.
- Weight-only only; joint activation quantization (e.g., W4A4, W2A4) has not been addressed.
- $\bm M$ is shared globally; whether grouping by hidden dimensions or layer families would improve results remains unexplored.

## Related Work & Insights
- **vs. Uniform Quantization (AWQ / QuIP#)**: UQ is limited to discrete INT2/3/4; LiftQuant is continuously adjustable.
- **vs. Vector Quantization (AQLM / VPTQ / QTIP)**: VQ is accurate but slow due to LUT; LiftQuant approaches QTIP accuracy while remaining hardware-friendly via linear operators.
- **vs. Q-Palette (mixed quantizers for fractional bits)**: Q-Palette requires heterogeneous kernel libraries; LiftQuant uses a single unified operator.
- **Insight**: The lift-then-project approach can be generalized to KV-cache, activation, and optimizer state quantization—any scenario requiring continuous compression ratios. CLT-based codebook generation is a universal methodology.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to implement continuous bit-width LLM quantization; the lift-then-project mechanism is truly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across Gaussian source theory, Llama 7B/13B/70B PPL, and 24GB/12GB GPU Pareto deployment.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear CLT motivation; Figure 1 Pareto curves directly address pain points; Figure 2 codebook visualization is intuitive.
- Value: ⭐⭐⭐⭐⭐ Running 70B models on 24GB GPUs is an industrial necessity; fractional bit concepts will influence future LLM quantization design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[NeurIPS 2025\] Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment](../../NeurIPS2025/model_compression/q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](../../CVPR2026/model_compression/generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
