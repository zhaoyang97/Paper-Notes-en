---
title: >-
  [Paper Note] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance
description: >-
  [ICCV 2025][Image Restoration][Translation Equivariance] This paper systematically investigates the impact of Translation Equivariance (TE) on the convergence speed and generalization ability of image restoration networks. It proposes Sliding Key-Value Self-Attention (SkvSA), its adaptive variant (ASkvSA), and Downsampled Self-Attention (DSA), and constructs TEAFormer, which achieves state-of-the-art performance on super-resolution, deblurring, denoising…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Translation Equivariance"
  - "Transformer"
  - "Adaptive Attention"
  - "Super-Resolution"
date: 2026-05-08
content_hash: 5ce15e25320053f2
---

# Enhancing Image Restoration Transformer via Adaptive Translation Equivariance

**Conference**: ICCV 2025
**arXiv**: [2506.18520](https://arxiv.org/abs/2506.18520)  
**Code**: Unavailable  
**Area**: Image Restoration
**Keywords**: Translation Equivariance, Image Restoration, Transformer, Adaptive Attention, Super-Resolution

## TL;DR

This paper systematically investigates the impact of Translation Equivariance (TE) on the convergence speed and generalization ability of image restoration networks. It proposes Sliding Key-Value Self-Attention (SkvSA), its adaptive variant (ASkvSA), and Downsampled Self-Attention (DSA), and constructs TEAFormer, which achieves state-of-the-art performance on super-resolution, deblurring, denoising, and other tasks while maintaining linear complexity.

## Background & Motivation

### State of the Field

A fundamental property of image restoration is *fidelity*—the restoration result over a target region should be equivariant under geometric transformations of the input. In the CNN era, the sliding nature of convolution naturally endows networks with translation equivariance. However, the introduction of Transformers has disrupted this property: the global indexing and positional encodings of Self-Attention (SA), as well as the feature-shifting operations in Window Attention (WA), both break translation equivariance.

### Limitations of Prior Work

**The dilemma between SA's computational bottleneck and WA's fixed receptive field**: SA provides a global receptive field but incurs $O(N^2)$ complexity; WA achieves linear complexity but has a fixed receptive field—neither satisfies TE.

**Overlooked consequences of missing TE**: Empirical evidence shows that the absence of TE leads to slow training convergence (NTK condition number 7.5× higher) and poor generalization (SRGA value 12% higher).

**Existing restoration Transformers lack systematic TE-aware design**: The attention modules in SwinIR, HAT, DAT, and similar methods do not satisfy TE.

### Root Cause

Starting from two fundamental theorems—**sliding indexing** (Theorem 3.2: if each output position depends only on inputs within a fixed local neighborhood, TE is satisfied) and **component composition** (Theorem 3.3: sequential and parallel combinations of TE modules remain TE-compliant)—this paper redesigns the attention mechanism to simultaneously achieve a global receptive field and linear complexity while preserving TE.

## Method

### Overall Architecture

TEAFormer adopts a classic residual-in-residual structure. The input image $I \in \mathbb{R}^{H \times W \times 3}$ is first processed by a convolutional layer to extract shallow features $F_0$, which are then passed through $N_g$ Translation Equivariance Groups (TEGs) to extract deep features $F_1$. The residual sum $F_r = F_0 + F_1$ is fed into a reconstruction module to produce the high-quality output. Each TEG contains $N_b$ Translation Equivariance Blocks (TEBs), each comprising a TEA attention module and a feed-forward network.

### Key Designs

#### 1. **Sliding Key-Value Self-Attention (SkvSA)**
- **Function**: Replaces the global KV indexing of SA with local KV indexing based on a sliding window.
- **Mechanism**: For the $i$-th query $Q_i$, attention is computed using only KV pairs drawn from the sliding window $[i - \frac{w \cdot s}{2}, i + \frac{w \cdot s}{2}]$, where $w$ is the window size and $s$ is the stride.
- **Boundary handling**: A blocking strategy is adopted—queries at the boundary use KV pairs from a predefined blocking window.
- **Design Motivation**: By Theorem 3.2, attention computed within a sliding window naturally satisfies TE while maintaining $O(N)$ complexity.

#### 2. **Adaptive Sliding Key-Value Self-Attention (ASkvSA)**
- **Function**: Adaptively determines the optimal KV index positions for each query.
- **Mechanism**: K and V are reshaped from 1D to 2D, and depthwise separable convolutions (kernel size $k$) are applied to generate adaptive offsets $\mathcal{F} \in \mathbb{R}^{H \times W \times 2}$, which shuffle KV positions within the fixed sliding window.
- **TE guarantee**: Since convolution satisfies TE, composing ASkvSA with SkvSA still satisfies TE by Theorem 3.3.
- **Design Motivation**: A fixed window constrains the flexibility of KV selection; adaptive indexing allows the model to identify the most relevant KV pairs for each query.

#### 3. **Downsampled Self-Attention (DSA) + Adaptive Fusion**
- **Function**: Provides coarse-grained global KV indexing via downsampling, compensating for ASkvSA's inability to capture long-range dependencies.
- **Mechanism**: Average pooling downsamples K' and V' to $N_d$ tokens, which are then used to compute global attention with Q.
- **Fusion**: $\text{TEA}_i(X) = \alpha_s \cdot \text{ASkvSA}(X) + \alpha_d \cdot \text{DSA}(X)$, where $\alpha_s, \alpha_d$ are learnable parameters.
- **Design Motivation**: The adaptive indexing in ASkvSA shuffles only within a local window, so distant but relevant pixels may still be missed. DSA supplements this with global information at low resolution.

### Computational Complexity

The total FLOPs of TEA is $3ND^2 + 2NDk^2 + 2Nw^2D + 2NN_dD$. With hyperparameters $w=15, k=3, N_d=16$, the computational cost is comparable to WA with window size 16, and both scale as $O(N)$.

### Loss & Training

- Super-resolution: $L_1$ loss, Adam optimizer, learning rate $2 \times 10^{-4}$, cosine scheduler, DF2K dataset.
- Deblurring/denoising: $L_1$ loss, AdamW optimizer, progressive training strategy.

## Key Experimental Results

### Main Results (4× Super-Resolution)

| Method | Params | FLOPs | Urban100 PSNR | Urban100 SSIM | Manga109 PSNR |
|--------|--------|-------|---------------|---------------|---------------|
| SwinIR | 11.8M | 1.848T | 27.45 | 0.8254 | 32.03 |
| HAT | 20.6M | 3.662T | 27.97 | 0.8368 | 32.48 |
| DAT | 14.7M | 2.155T | 27.87 | 0.8343 | 32.51 |
| IPG | 16.8M | 4.732T | 28.13 | 0.8392 | 32.53 |
| **TEAFormer** | **21.8M** | **1.035T** | **28.67** | **0.8489** | **32.99** |

### Ablation Study (4× SR, Urban100)

| Model | SkvSA | ASkvSA | DSA | PSNR/SSIM | NTK Condition↓ | SRGA Value↓ | Latency (ms) |
|-------|-------|--------|-----|-----------|----------------|-------------|--------------|
| SwinIR (w=8) | - | - | - | 27.45/0.8254 | 1746.49 | 3.655 | 130.0 |
| SwinIR-Large (w=16) | - | - | - | 27.94/0.8362 | 1554.65 | 3.610 | 214.5 |
| TEAFormer | ✔ | - | - | 28.31/0.8444 | 243.75 | 3.206 | 230.1 |
| TEAFormer | - | ✔ | - | 28.47/0.8457 | 283.99 | 3.298 | 284.4 |
| TEAFormer | ✔ | - | ✔ | 28.49/0.8470 | 203.06 | 3.261 | 340.9 |
| **TEAFormer** | - | **✔** | **✔** | **28.67/0.8489** | **236.78** | **3.275** | 386.7 |

### Defocus Deblurring (DPDD Dataset, Single-Image Input)

| Method | Params | Indoor PSNR | Outdoor PSNR | Combined PSNR |
|--------|--------|-------------|--------------|---------------|
| Restormer | 26.1M | 28.87 | 23.24 | 25.98 |
| GRL-B | 19.9M | 29.06 | 23.45 | 26.18 |
| **TEAFormer** | **15.4M** | **29.50** | **23.55** | **26.45** |

### Key Findings

- **TE has a substantive impact on convergence and generalization**: Introducing SkvSA alone (the simplest TE scheme) reduces the NTK condition number from 1746 to 244 (7.2× reduction) and the SRGA value from 3.655 to 3.206.
- **ASkvSA + DSA is the optimal combination**: The parallel combination of adaptive local indexing and coarse-grained global information achieves the best balance between performance and efficiency.
- **TEAFormer shows a pronounced advantage on Urban100**: It achieves 28.67 dB at 4× SR, surpassing HAT by 0.7 dB (at comparable parameter counts) and IPG by 0.54 dB (with lower FLOPs and 2× faster inference).
- **The lightweight TEAFormer-L (829K parameters) also performs competitively**: It outperforms SwinIR-L by 0.57 dB on Urban100.

## Highlights & Insights

1. **Theory-driven design paradigm**: Starting from the mathematical definition of TE, two fundamental theorems are derived, which then systematically guide the design of attention modules. This "theorem → design" paradigm is relatively rare in the vision community.
2. **NTK and SRGA as analytical tools**: Using the NTK condition number to measure convergence speed and SRGA to measure generalization ability provides new evaluation dimensions for attention mechanism analysis.
3. **Breaking the SA vs. WA dichotomy**: By combining adaptive sliding indexing with downsampled global attention in parallel, the method simultaneously achieves flexible local indexing and a global receptive field under $O(N)$ complexity.
4. **Strong cross-task generalization**: The same architecture achieves state-of-the-art results across super-resolution, defocus deblurring, and denoising.

## Limitations & Future Work

1. **High inference latency**: The full TEAFormer has a latency of 386.7ms, approximately 3× slower than SwinIR's 130ms.
2. **Imperfect boundary handling**: The blocking strategy at boundaries only approximately satisfies TE; strict TE at boundaries would require more refined design.
3. **Average pooling in DSA**: While efficient, average pooling only approximately satisfies TE; the paper notes that learnable polyphase filters could serve as a replacement.
4. **Multiple hyperparameters**: The four hyperparameters $w, s, k, N_d$ require careful tuning.
5. **Absence of perceptual quality evaluation**: Only PSNR/SSIM are reported; perceptual metrics such as LPIPS are not evaluated.

## Related Work & Insights

- The ViTAE series on inductive bias points out that ViT suffers from training difficulties due to the lack of inductive bias; this paper provides a more concrete analysis from the TE perspective.
- SwinIR/HAT's window attention reduces complexity at the cost of TE and flexibility—TEAFormer's sliding design fills this gap.
- The dynamic window selection in BiFormer and DAT bears conceptual similarities to ASkvSA's adaptive indexing, but TEAFormer provides a more rigorous theoretical foundation from the perspective of TE guarantees.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic study of TE's impact on restoration Transformers, with solid theoretical derivations and design choices.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers SR/deblurring/denoising tasks with comprehensive ablation studies and convergence/generalization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The theoretical sections are clear, but the dense notation and scattered symbol definitions may impede readability.
- **Value**: ⭐⭐⭐⭐⭐ — Offers a new theoretical perspective and practical solution for restoration Transformer design, with broad reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration](devil_is_in_the_uniformity_exploring_diverse_learners_within_transformer_for_ima.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[ICCV 2025\] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions](metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](../../ECCV2024/image_restoration/seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)

</div>

<!-- RELATED:END -->
