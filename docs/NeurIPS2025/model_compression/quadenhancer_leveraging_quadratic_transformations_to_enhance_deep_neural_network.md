---
title: >-
  [Paper Note] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks
description: >-
  [NeurIPS 2025][Model Compression][quadratic transformation] This paper proposes a lightweight quadratic enhancer (QuadEnhancer) that introduces sparsified quadratic interaction terms into each linear layer…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "quadratic transformation"
  - "nonlinearity enhancement"
  - "lightweight module"
  - "LoRA fine-tuning"
  - "weight sharing"
date: 2026-05-08
content_hash: f15bdb185472fdfa
---

# QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2510.03276](https://arxiv.org/abs/2510.03276)  
**Code**: [GitHub](https://github.com/chitar/QuadEnhancer)  
**Area**: Model Compression
**Keywords**: quadratic transformation, nonlinearity enhancement, lightweight module, LoRA fine-tuning, weight sharing

## TL;DR

This paper proposes a lightweight quadratic enhancer (QuadEnhancer) that introduces sparsified quadratic interaction terms into each linear layer, achieving significant performance improvements over existing neural network architectures with negligible additional parameters and computational overhead.

## Background & Motivation

The core building block of modern deep neural networks is the combination of a linear transformation and a nonlinear activation function. Although this paradigm has achieved remarkable success in approximating complex functions, its nonlinear expressiveness still has room for improvement. Three main directions exist for enhancing nonlinearity:

**More complex activation functions** (e.g., Swish, GELU, Mish): These approaches focus on element-wise transformations and cannot capture inter-neuron interactions.

**Nonlinear network modules** (e.g., LSTM gating, attention mechanisms): These are typically task-specific and lack generality.

**Polynomial transformations as replacements for linear operations** (e.g., polynomial networks, QuadraNet): These offer stronger theoretical expressiveness but are limited in practice due to the dramatic growth in parameter count and computational cost.

The core limitation is that standard quadratic transformations require $O(dn^2)$ additional parameters, which is unacceptable for practical deployment. The motivation of this paper is to find a method that introduces quadratic terms to enhance expressiveness while keeping the additional parameter and computational overhead negligible.

## Method

### Overall Architecture

QuadEnhancer is a plug-and-play module that can be appended to any linear layer. For a standard linear transformation $\tilde{\mathbf{y}} = \mathbf{W}\mathbf{x}$, the enhanced output becomes:

$$\mathbf{z} = (\mathbf{\Lambda}\tilde{\mathbf{y}}) \odot \tilde{\mathbf{y}} + \tilde{\mathbf{y}} + \mathbf{b}$$

where $\mathbf{\Lambda}$ is a sparse banded matrix and $\odot$ denotes the Hadamard element-wise product.

### Key Designs

1. **Rank-1 Decomposition for Parameter Reduction**: In the original quadratic transformation, each output dimension corresponds to an $n \times n$ matrix $\mathbf{V}_i$, resulting in $O(dn^2)$ parameters. Constraining each $\mathbf{V}_i$ to a rank-1 matrix $\mathbf{V}_i = \mathbf{p}_i\mathbf{q}_i^\top$ reduces the parameter count to $O(2dn)$. The quadratic transformation then becomes $\mathbf{z} = (\mathbf{P}\mathbf{x}) \odot (\mathbf{Q}\mathbf{x}) + \mathbf{W}\mathbf{x} + \mathbf{b}$.

2. **Weight Sharing**: Setting $\mathbf{P} = \mathbf{\Lambda}\mathbf{W}$ and $\mathbf{Q} = \mathbf{W}$ reuses the weight matrix $\mathbf{W}$ of the linear layer. This yields two benefits: (a) the parameter count is reduced from $3dn$ to $dn + d^2$; and (b) the linear response $\tilde{\mathbf{y}} = \mathbf{W}\mathbf{x}$ needs to be computed only once, reducing computational redundancy.

3. **Sparsification of $\mathbf{\Lambda}$**: $\mathbf{\Lambda} \in \mathbb{R}^{d \times d}$ still has $O(d^2)$ parameters. By restricting it to a banded matrix of bandwidth $k$ (augmented with small triangular regions at the lower-left and upper-right corners to form a circulant structure), the parameter count is reduced to $k \times d$. When $k=1$, the additional parameters relative to the original linear layer are only $O(1/n)$. Computationally, $\mathbf{\Lambda}\tilde{\mathbf{y}} = \sum_{r \in \mathcal{K}} \boldsymbol{\lambda}_r \odot \text{Roll}(\tilde{\mathbf{y}}, r)$, where Roll denotes the cyclic shift operation.

### Loss & Training

- The shift $r=0$ (i.e., the pure quadratic term $\tilde{y}_i^2$) is excluded, as squared terms are more prone to numerical instability compared to cross terms $\tilde{y}_i \tilde{y}_j$ (variance of 2 vs. 1, with large values occurring orders of magnitude more frequently).
- Experiments fix $\mathcal{K} = \{1\}$, introducing quadratic interactions only between adjacent neurons.
- No special loss function is required; standard task losses (e.g., cross-entropy) are used for end-to-end training.

## Key Experimental Results

### Main Results — Image Classification

| Model | Params | ImageNet | Caltech | CIFAR-10 | CIFAR-100 | Pets | Avg |
|-------|--------|----------|---------|----------|-----------|------|-----|
| ViT-M | 2.45M | 63.70 | 87.77 | 96.35 | 80.25 | 91.03 | 82.44 |
| ViT-M+QE | 2.47M | 65.30 | 90.32 | 97.09 | 82.59 | 91.88 | 83.91 |
| ViT-XT | 2.82M | 66.04 | 90.25 | 96.51 | 81.24 | 91.03 | 84.99 |
| ViT-XT+QE | 2.83M | 67.34 | 90.77 | 96.78 | 82.64 | 97.97 | 86.82 |
| ViT-T | 5.37M | 73.96 | 93.07 | 97.97 | 86.13 | 93.87 | 88.57 |
| ViT-T+QE | 5.40M | 75.15 | 94.03 | 98.03 | 86.88 | 94.95 | 89.27 |

### Main Results — LLM Fine-tuning (Commonsense Reasoning)

| Model | Method | Params | BoolQ | HellaSwag | ARC-e | ARC-c | Avg |
|-------|--------|--------|-------|-----------|-------|-------|-----|
| LLaMA-7B | LoRA/32 | 53.5M | 68.90 | 78.10 | 77.80 | 61.30 | 74.73 |
| LLaMA-7B | LoRA/16+QE | 27.6M | 69.69 | 87.11 | 79.41 | 63.99 | 77.85 |
| LLaMA3-8B | LoRA/32 | 54.0M | 70.80 | 91.70 | 84.20 | 71.20 | 80.79 |
| LLaMA3-8B | LoRA/32+QE | 54.7M | 74.92 | 95.02 | 89.85 | 79.60 | 85.46 |

### Ablation Study

| Configuration | Params | ImageNet | CIFAR-10 | CIFAR-100 | Avg | Note |
|---------------|--------|----------|----------|-----------|-----|------|
| ViT-M+QuadraNet | 2.53M | 61.17 | 95.81 | 79.08 | 79.41 | Three independent weight matrices |
| ViT-M+SwiGLU | 2.58M | 63.25 | 96.76 | 80.58 | 81.13 | Gated linear unit |
| ViT-M+QuadEnhancer | 2.47M | 65.30 | 97.09 | 82.59 | 82.40 | Fewest parameters, best performance |

### Key Findings

- QE with **half the parameters** (LoRA/16) already surpasses full-rank LoRA/32 (a gain of 2.64% on LLaMA2-7B), demonstrating the substantial expressiveness contributed by quadratic interactions.
- As model and data scale increase, the gain from QE grows consistently from 0.07% to 1.19%, indicating favorable scaling properties.
- On the Pets dataset, ViT-XT+QE achieves a remarkable improvement of 6.94%, suggesting that the method is particularly effective for fine-grained classification tasks.

## Highlights & Insights

- **Extremely lightweight**: At $k=1$, the additional parameters consist of only $d$ scalars, and the extra FLOPs amount to merely $4d$, constituting near-zero overhead.
- **Plug-and-play**: The module can be seamlessly applied to diverse architectures including ViT, GPT-2, and LLaMA, and is compatible with parameter-efficient fine-tuning methods such as LoRA.
- **Numerically stable by design**: By excluding squared terms and retaining only cross terms, the method avoids overflow issues under FP16 precision.

## Limitations & Future Work

- The experimental scale is relatively small (ViT-M/T are very small models, and WikiText-2 is a small corpus); validation at larger scales is needed.
- Only the $k=1$ setting is explored; the effect of varying bandwidth $k$ is not systematically studied.
- Theoretical analysis explaining why and under what conditions QuadEnhancer is most effective is absent.
- Training time comparisons show non-trivial overhead in early stages, and latency analysis for practical deployment is insufficient.

## Related Work & Insights

- Compared to QuadraNet and SwiGLU, QuadEnhancer achieves superior performance with fewer parameters through weight sharing and sparsification.
- This approach can be generalized to any neural network layer based on linear transformations, including convolutional layers (by treating convolution as matrix multiplication).
- The enhancement of PEFT methods such as LoRA is particularly valuable, suggesting that quadratic interactions may serve as a general means to improve fine-tuning effectiveness.

## Rating

- Novelty: ⭐⭐⭐⭐ Quadratic transformations are not a new concept, but the parameter reduction strategy via weight sharing and banded sparsification is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three tasks (CV, NLP, LLM fine-tuning) with ablation and scaling analyses
- Writing Quality: ⭐⭐⭐⭐⭐ Derivations are clear and the step-by-step simplification is easy to follow
- Value: ⭐⭐⭐⭐ The plug-and-play enhancement scheme is practically useful, but requires validation at larger scale

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks](global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/model_compression/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[NeurIPS 2025\] DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration](denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)

</div>

<!-- RELATED:END -->
