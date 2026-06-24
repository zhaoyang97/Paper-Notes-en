---
title: >-
  [Paper Note] Memory-Efficient 4-bit Preconditioned Stochastic Optimization
description: >-
  [ICCV 2025][Optimization][Shampoo] This paper proposes a 4-bit quantization scheme based on Cholesky decomposition and error feedback, compressing the preconditioner matrices of the Shampoo optimizer to 4-bit precision. The approach substantially reduces GPU memory consumption while preserving training performance close to 32-bit Shampoo, with convergence guarantees provided for both smooth and non-smooth settings.
tags:
  - "ICCV 2025"
  - "Optimization"
  - "Shampoo"
  - "quantization"
  - "Cholesky decomposition"
  - "error feedback"
  - "low-precision optimizer"
  - "memory-efficient training"
date: 2026-05-08
content_hash: dc7cad86839f4429
---

# Memory-Efficient 4-bit Preconditioned Stochastic Optimization

**Conference**: ICCV 2025
**arXiv**: [2412.10663](https://arxiv.org/abs/2412.10663)  
**Code**: To be confirmed  
**Area**: Optimization / Preconditioned Stochastic Optimization
**Keywords**: Shampoo, quantization, Cholesky decomposition, error feedback, low-precision optimizer, memory-efficient training

## TL;DR

This paper proposes a 4-bit quantization scheme based on Cholesky decomposition and error feedback, compressing the preconditioner matrices of the Shampoo optimizer to 4-bit precision. The approach substantially reduces GPU memory consumption while preserving training performance close to 32-bit Shampoo, with convergence guarantees provided for both smooth and non-smooth settings.

## Background & Motivation

Preconditioned stochastic optimization algorithms (e.g., Shampoo) exploit non-diagonal preconditioner matrices to capture inter-parameter correlations, achieving superior convergence rates and empirical performance over first-order optimizers (SGD, Adam, etc.). However, Shampoo requires storing four preconditioner matrices $(L_k, R_k, L_k^{-1/4}, R_k^{-1/4})$, incurring substantial GPU memory overhead that severely limits its scalability in large-scale model training.

A straightforward solution is to quantize the preconditioner matrices from 32-bit to 4-bit, but naive quantization leads to significant information loss and performance degradation. For example, on ViT-Small/CIFAR-100, 32-bit Shampoo achieves 77.95% test accuracy, whereas direct 4-bit quantization yields only 74.56%, a gap of 3.39%. The core challenge is therefore: **how to effectively compress preconditioner matrices at 4-bit ultra-low precision while preserving their spectral properties and avoiding optimization quality degradation?**

## Method

### Overall Architecture

The proposed 4-bit Shampoo comprises three progressively layered techniques:

1. **Baseline: Direct Quantization (VQ)** — block-wise 4-bit quantization applied directly to preconditioner matrices
2. **Core Contribution 1: Cholesky Quantization (CQ)** — Cholesky decomposition performed prior to quantizing the lower-triangular factor
3. **Core Contribution 2: Compensated Cholesky Quantization (CQ+EF)** — error feedback introduced on top of Cholesky quantization

### Key Design 1: Cholesky Quantization (CQ)

Rather than directly quantizing the symmetric positive definite preconditioner matrices $L_k$ and $R_k$, the method first applies Cholesky decomposition:

$$C_k^L = \text{Cholesky}(L_k + \epsilon I), \quad C_k^R = \text{Cholesky}(R_k + \epsilon I)$$

The lower-triangular Cholesky factor is then quantized as $\bar{C}_k^L = \mathcal{Q}(C_k^L)$. During recovery, the preconditioner is reconstructed via $L_k = \mathcal{D}(\bar{C}_k^L)\mathcal{D}(\bar{C}_k^L)^T$.

**Two key advantages:**

- **Memory reduction by half**: The Cholesky factor is lower-triangular, requiring approximately half the storage of a full matrix.
- **Spectral preservation**: Matrices recovered from quantized Cholesky factors automatically remain symmetric positive definite (SPD), with inverse 1/4-roots closer to those of the 32-bit counterpart. Quantitative validation on synthetic PD matrices shows NRE reduced from 46.14 (direct quantization) to 9.19 (Cholesky quantization), and AE from 27.19 to 9.20.

**Quantization strategy**: Diagonal elements are retained at 32-bit precision for numerical stability; off-diagonal elements are quantized to 4-bit using block-wise linear-2 mapping.

### Key Design 2: Error Feedback (EF)

Inspired by error feedback in gradient compression for distributed training, the method adapts this mechanism to preconditioner compression:

- A 4-bit error state $\bar{E}_k^L$ is maintained to record the quantization error of the Cholesky factor.
- Prior to each quantization step, the previous error is used to compensate: $\bar{C}_k^L = \mathcal{Q}(C_k^L + E_{k-1}^L)$
- The error state is updated via exponential moving average: $E_k^L = \beta_e E_{k-1}^L + (1-\beta_e)(C_k^L + E_{k-1}^L - \mathcal{D}(\bar{C}_k^L))$

**Key storage trick**: The Cholesky factor occupies the lower-triangular portion and the error state (strictly lower-triangular, with zero diagonal) occupies the upper-triangular portion of the same matrix, incurring no additional memory overhead.

### Loss & Training

Model updates follow the standard Shampoo procedure, using quantized inverse 1/4-roots to precondition the gradients:

$$W_{k+1} = W_k - \eta_k \mathcal{D}(\hat{L}_k) G_k \mathcal{D}(\hat{R}_k)$$

The learning rate is scaled via the grafting trick. Both SGD and AdamW are supported as base optimizers.

### Convergence Theory

- **Smooth non-convex setting**: 4-bit Shampoo achieves the optimal convergence rate $\mathcal{O}(1/\sqrt{T})$.
- **Non-smooth non-convex setting (e.g., ReLU networks)**: Global convergence to a stationary point set is proven — the **first convergence guarantee** for preconditioned gradient descent in the non-smooth regime.

## Key Experimental Results

### Main Results: CIFAR-100 Test Accuracy (%) / Peak Memory (MB)

| Model | Optimizer | Accuracy | Memory |
|-------|-----------|----------|--------|
| VGG-19 | SGDM | 74.43 | 597.3 |
| VGG-19 | SGDM + 32-bit Shampoo | 75.02 | 1065.2 |
| VGG-19 | SGDM + 4-bit VQ | 74.36 | 662.2 |
| VGG-19 | SGDM + 4-bit CQ+EF | **75.21** | 662.2 |
| ViT-Small | AdamW | 73.00 | 2930.0 |
| ViT-Small | AdamW + 32-bit Shampoo | 77.95 | 3448.9 |
| ViT-Small | AdamW + 4-bit VQ | 74.56 | 3001.7 |
| ViT-Small | AdamW + 4-bit CQ+EF | **77.74** | 3001.7 |

### ImageNet Test Accuracy (%) / Peak Memory (MB)

| Model | Optimizer | Accuracy | Memory |
|-------|-----------|----------|--------|
| ResNet-50 | AdamW Base | 77.56 | 11356 |
| ResNet-50 | 32-bit Shampoo | 78.06 | 11986 |
| ResNet-50 | 4-bit CQ+EF | **78.00** | 11445 |
| ViT-Base | AdamW Base | 72.59 | 11840 |
| ViT-Base | 32-bit Shampoo | 75.47 | 13319 |
| ViT-Base | 4-bit CQ+EF | **75.01** | 12052 |

### LLM Pre-training: LLaMA on C4 Perplexity (PPL, lower is better)

| Model | Optimizer | PPL | Memory (GB) |
|-------|-----------|-----|-------------|
| LLaMA-130M | AdamW Base | 27.32 | 45.9 |
| LLaMA-130M | 32-bit Shampoo | 26.93 | 48.2 |
| LLaMA-130M | 4-bit CQ+EF | **26.98** | 46.2 |

### Ablation Study: Incremental Component Analysis

| Method | ResNet-34/CIFAR-100 | ViT-Small/CIFAR-100 |
|--------|---------------------|---------------------|
| 4-bit VQ | 79.45 | 74.56 |
| 4-bit CQ | 80.27 (+0.82) | 77.34 (+2.78) |
| 4-bit CQ+EF | **80.52** (+1.07) | **77.74** (+3.18) |

### Spectral Preservation Metrics (NRE / AE, lower is better)

| Method | Synthetic NRE | Synthetic AE | Epoch100 NRE | Epoch100 AE |
|--------|---------------|--------------|--------------|-------------|
| Direct Quantization VQ | 46.14 | 27.19 | 25.71 | 18.51 |
| Cholesky CQ | **9.19** | **9.20** | **4.85** | **4.85** |

### Key Findings

- CQ+EF reduces the 4-bit accuracy gap on ViT-Small/CIFAR-100 from 3.39% to 0.21%.
- On ImageNet/ViT-Base, 4-bit CQ+EF is only 0.46% below 32-bit Shampoo while saving 1267 MB of memory.
- The computational overhead of Cholesky decomposition is negligible: ResNet-50/ImageNet training takes 1899 min vs. 1883 min for VQ.

## Highlights & Insights

1. **Novel quantization target**: Rather than directly quantizing preconditioner matrices, the method quantizes their Cholesky factors — the first such attempt. The triangular structure and SPD recovery property jointly address both memory reduction and information preservation.
2. **Elegant combination of error feedback and triangular structure**: The error state can be stored in the upper-triangular portion of the same matrix at zero additional memory cost.
3. **First convergence proof in the non-smooth setting**: Theoretical guarantees are provided for preconditioned gradient methods with non-smooth activations such as ReLU.
4. **Comprehensive validation**: Experiments span diverse architectures including VGG, ResNet, Swin, ViT, and LLaMA.

## Limitations & Future Work

1. Cholesky decomposition requires strictly positive definite preconditioner matrices, necessitating $\epsilon I$ regularization.
2. Diagonal elements are retained at 32-bit, resulting in an overall compression ratio below pure 4-bit.
3. Validation is limited to image classification and LLM pre-training; extension to detection and segmentation tasks has not been explored.
4. Error feedback introduces an additional hyperparameter $\beta_e$, with insufficient sensitivity analysis provided.

## Related Work & Insights

- **Preconditioned optimizers**: Shampoo, K-FAC, K-BFGS, AdaBK
- **Optimizer quantization**: 4-bit Adam and related methods, which target only first-order optimizer states
- **Error feedback**: Originally proposed for distributed gradient compression; this work is the first to apply it to preconditioner matrix quantization compensation

## Rating

| Dimension | Score (1–10) |
|-----------|--------------|
| Novelty | 8 |
| Theoretical Depth | 8 |
| Experimental Thoroughness | 8 |
| Value | 8 |
| Writing Quality | 7 |
| **Overall** | **8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](../../NeurIPS2025/optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [\[ICML 2026\] Memory-Efficient LLM Pretraining via Minimalist Optimizer Design](../../ICML2026/optimization/memory-efficient_llm_pretraining_via_minimalist_optimizer_design.md)
- [\[ICLR 2026\] Arbitrary-Order Block SignSGD for Memory-Efficient LLM Fine-Tuning](../../ICLR2026/optimization/arbitrary-order_block_signsgd_for_memory-efficient_llm_fine-tuning.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)
- [\[ICLR 2026\] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems](../../ICLR2026/optimization/a_memory-efficient_hierarchical_algorithm_for_large-scale_optimal_transport_prob.md)

</div>

<!-- RELATED:END -->
