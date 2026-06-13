---
title: >-
  [Paper Note] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks
description: >-
  [ICML 2026][Model Compression][BNN] SURGE attaches a "full-precision auxiliary branch" in parallel to each binarization layer. While the forward output remains unchanged…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "BNN"
  - "STE"
  - "gradient mismatch"
  - "dual-path compensation"
  - "adaptive gradient scaling"
date: 2026-05-08
content_hash: e76eb9b0c2270157
---

# SURGE: Surrogate Gradient Adaptation in Binary Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.10989](https://arxiv.org/abs/2605.10989)  
**Code**: Not yet public  
**Area**: Model Compression / Binary Neural Networks / Quantization-Aware Training  
**Keywords**: BNN, STE, gradient mismatch, dual-path compensation, adaptive gradient scaling

## TL;DR
SURGE attaches a "full-precision auxiliary branch" in parallel to each binarization layer. While the forward output remains unchanged, the backward pass propagates additional "non-STE truncated" high-order gradients from the auxiliary branch. These contributions are dynamically balanced using AGS based on the gradient norm ratio, allowing a BNN to achieve 62.0% top-1 on ResNet-18/ImageNet, outperforming ReCU by 1.0% and IR-Net by 3.9%.

## Background & Motivation

**Background**: Binary Neural Networks (BNNs) quantize weights and activations to $\{-1,+1\}$, theoretically offering $32\times$ memory compression and $58\times$ inference acceleration, making them the most aggressive quantization scheme for edge deployment. During training, almost all BNNs rely on the Straight-Through Estimator (STE): using $\text{sign}(\cdot)$ in the forward pass while treating $\frac{\partial\mathbf{B}_W}{\partial W}\approx 1$ and $\frac{\partial\mathbf{B}_x}{\partial x}\approx\mathbb{1}_{\{|x|\le 1\}}$ as surrogate gradients in the backward pass.

**Limitations of Prior Work**: STE suffers from two fundamental issues. First, the true gradient of the sign function is zero almost everywhere; using an identity function as a substitute introduces systemic deviation, widely recognized as "gradient mismatch." Second, activation gradients are zeroed out when they fall outside $[-1, 1]$, discarding a significant amount of information. Existing works (e.g., sigmoid approximation in DSQ, asymptotic sign in IR-Net, feature distribution alignment in ReCU) mostly rely on handcrafted approximation functions, which do not guarantee optimality.

**Key Challenge**: In BNN training, there is a core contradiction between "strict binarization in the forward pass (to ensure inference acceleration)" and "sufficiently rich gradients in the backward pass (to ensure learnability)." As long as the forward pass uses the sign function, the backward pass is typically forced to make do with first-order identity surrogates.

**Goal**: 1) Supplement the main branch with "non-STE, low-bias" gradients from an external source without altering the forward output; 2) Prevent magnitude imbalances in the compensation gradients from disrupting the convergence of the main branch; 3) Discard the auxiliary branch entirely during the inference stage for zero additional overhead.

**Key Insight**: Since STE is a first-order approximation of the sign function, one can parallelize a "full-precision replica" for each layer and use its true gradients to compensate for the high-order terms missing from the STE. Furthermore, since the magnitudes of the two gradient paths are unknown, norm-ratio adaptive scaling is employed for dynamic balancing.

**Core Idea**: A "forward self-cancellation, backward pass-through" detach trick is used to ensure the full-precision auxiliary branch only participates in the backward pass. AGS then adaptively scales the contributions based on $\frac{\|g_b\|_2}{\|g_a\|_2+\epsilon}$, refining the first-order STE surrogate into a hybrid estimation that closer resembles the true gradient.

## Method

### Overall Architecture
For every binary linear operator (conv, linear, attention projection), SURGE attaches a full-precision replica (auxiliary branch) of the exact same dimensions in parallel. During the forward pass, a $\text{detach}$ trick is utilized to make the two occurrences of the auxiliary branch cancel each other out, ensuring the final output is strictly identical to that of a pure BNN. During the backward pass, the auxiliary branch propagates gradients normally while the main branch uses STE; the two merge at the input. AGS dynamically calculates a scaling factor $\lambda_{\text{AGS}}$ to balance the contributions. After training, the auxiliary branch is discarded, leaving a standard BNN for inference.

### Key Designs

1.  **Dual-Path Gradient Compensator (DPGC)**:

    - **Function**: Provides each binarization layer with additional "non-STE truncated" high-order gradient information without modifying the forward output.
    - **Mechanism**: Define binary forward as $f_b(x;W_b)=Q_W(W_b)^\top Q_x(x)$ and full-precision forward as $f_a(x;W_a)=W_a^\top x$, with scaling $f_{ao}(x)=\lambda f_a(x)$. The output is formulated as $\text{output}=f_b(x;W_b)-f_{ao}(x;W_a)\downarrow+f_{ao}(x;W_a)$, where $\downarrow$ denotes the stop-gradient operator. In the forward pass, the second and third terms cancel out, resulting in $\text{output} = f_b$. In the backward pass, the gradient of the detached term is truncated, while $f_b$ follows STE and $f_{ao}$ follows full-precision propagation. Thus, $\frac{\partial\mathcal{L}}{\partial x}=g_b+\lambda g_a$, where $g_b$ is the first-order STE approximation and $g_a$ provides high-order compensation from the true gradients of the auxiliary branch.
    - **Design Motivation**: Traditional STE improvements (piecewise polynomial, SignSwish, etc.) are essentially just "switching to a different approximation," which does not resolve the root issue. The DPGC detach trick allows the contradictions of "strict binary output" and "full-precision backward signals" to coexist. It is a clever engineering construct that yields zero extra overhead during inference.

2.  **Adaptive Gradient Scaler (AGS)**:

    - **Function**: Dynamically balances the magnitudes of $g_b$ and $g_a$ to prevent the auxiliary path gradients from overwhelming the main branch training.
    - **Mechanism**: The scaling factor is defined as $\lambda_{\text{AGS}}=\eta\frac{\|g_b\|_2}{\|g_a\|_2+\epsilon}$, where $\eta$ is a base scaling coefficient and $\epsilon=10^{-8}$ prevents division by zero. Starting from a second-moment model, the paper proves the optimal $\lambda^*=\frac{\langle\delta_b,\mu_a\rangle}{\|\mu_a\|_2^2+\text{tr}(\text{Var}(g_a))}$ (where $\delta_b$ is the STE bias vector). Under the assumption that alignment $\cos\theta$, relative bias ratio $\beta$, and noise ratio $\rho$ are approximately stable, it derives $\lambda^*\approx\eta\frac{\|\mu_b\|

## Related Papers

- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/model_compression/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[AAAI 2026\] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](../../AAAI2026/model_compression/bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)
- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](../../ICML2025/model_compression/an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/model_compression/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[AAAI 2026\] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](../../AAAI2026/model_compression/bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)
- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->
