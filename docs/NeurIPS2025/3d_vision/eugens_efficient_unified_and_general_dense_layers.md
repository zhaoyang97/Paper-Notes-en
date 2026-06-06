---
title: >-
  [Paper Note] EUGens: Efficient, Unified, and General Dense Layers
description: >-
  [NeurIPS 2025][3D Vision][efficient neural network] EUGens introduces a new family of efficient dense layers that leverage Random Features to reduce the inference complexity of fully connected feedforward layers (FFLs) f…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "efficient neural network"
  - "random features"
  - "feedforward layer"
  - "Transformer"
  - "NeRF"
date: 2026-05-08
content_hash: 7eaadd1e292ea809
---

# EUGens: Efficient, Unified, and General Dense Layers

**Conference**: NeurIPS 2025
**arXiv**: [2410.09771](https://arxiv.org/abs/2410.09771)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: efficient neural network, random features, feedforward layer, Transformer, NeRF

## TL;DR

EUGens introduces a new family of efficient dense layers that leverage Random Features to reduce the inference complexity of fully connected feedforward layers (FFLs) from quadratic to linear. The framework unifies existing efficient FFL extensions and achieves up to 27% inference speedup and 30% parameter compression across LLM pre-training, ViT image classification, and NeRF/iSDF 3D reconstruction tasks, while supporting layer-wise knowledge distillation without backpropagation.

## Background & Motivation

- Fully connected feedforward layers (FFLs) are core components of Transformers and implicit neural representations (NeRF, iSDF, etc.), consuming a large share of parameters and computation.
- FFL inference complexity is $O(d^2 + dl)$, scaling quadratically with the hidden dimension.
- Existing acceleration methods (pruning, quantization, knowledge distillation, structured matrices) each have limitations and lack a general FFL replacement strategy.
- Random Features methods can decouple weight and input processing, but prior work (URF/SNNK) requires activation functions to have Fourier transforms, precluding unbounded activations such as ReLU.

## Core Problem

How to design a general-purpose efficient dense layer that can approximate a standard FFL with arbitrary polynomial activation functions in linear complexity, while preserving expressive capacity and remaining compatible with pre-trained models?

## Method

### 1. EUGen Layer Definition

A $k$-th order EUGen layer decouples the processing of weights $\mathbf{w}$ and inputs $\mathbf{x}$:

$$\text{EUGen}^k(\mathbf{w}, \mathbf{x}) = g(\mathbf{w})^\top f(\mathbf{x})$$

where $\mathbf{x}^+ = [\mathbf{x};\, \|\mathbf{x}\|_2]$ introduces direct dependence on the input norm, and $G_j^i$ denotes random projection matrices used to construct feature maps of different orders via Hadamard products and concatenation.

### 2. Theoretical Guarantees

**Theorem 3.1**: For any polynomial activation function $f$, EUGens can construct an unbiased estimator approximating the FFL output. This is the first unbiased approximation result for arbitrary polynomial activations.

**Theorem 3.2–3.3**: Variance formulas and exponentially small probability concentration inequalities. The failure probability decreases exponentially in the number of random features $m$.

**Theorem 3.4**: Via polynomial approximation, the framework extends to general continuous activation functions (e.g., ReLU, GeLU, Softplus).

### 3. Inference Complexity

The weight side can be precomputed. Inference complexity becomes $O(mdk^2 + ml)$. In practice, using $k \leq 3$, when $m \ll \min(d, l)$ the complexity reduces from $O(d^2)$ to $O(d)$.

### 4. QMC Improvement

Gaussian orthogonal matrices (GOMs) replace standard Gaussian projections to reduce estimation variance.

### 5. Layer-wise Knowledge Distillation

By storing the inputs and outputs of target layers, EUGen layer parameters can be optimized via MSE minimization. When $G_j^i$ is sampled from a fixed distribution, a closed-form solution exists, eliminating the need for backpropagation.

## Key Experimental Results

### LLM Pre-training (GPT-2, 124M, OpenWebText)

Replacing FFLs with EUGen layers yields validation loss close to vanilla GPT-2 while substantially reducing inference parameters. Replacing 6 layers reduces parameters by approximately 30%.

### ViT Image Classification (ViT-Base)

| Setting | ImageNet Acc | Inference Parameter Ratio |
|---------|-------------|--------------------------|
| Vanilla ViT | Baseline | 100% |
| EUGen (6 layers replaced) | Near baseline | ~70% |
| Low-Rank (same parameter budget) | Significant drop | ~70% |

EUGens significantly outperforms the Low-Rank baseline under the same parameter budget.

### NeRF 3D Reconstruction

| Method | PSNR | Inference Speedup | Model Compression |
|--------|------|------------------|-------------------|
| NeRF | Baseline | 1× | 1× |
| **EUGen-NeRF** | Near baseline | **24% speedup** | **30% compression** |
| Mip-NeRF 360 | Baseline | 1× | 1× |
| **EUGen-Mip-NeRF** | Near baseline | **27% speedup** | — |

### iSDF Real-time SDF Reconstruction

EUGen-iSDF achieves **22.6%** inference speedup and 5% training speedup with comparable reconstruction quality.

### Knowledge Distillation

NeRF distillation achieves up to **26%** inference speedup without retraining.

## Highlights & Insights

- Solid theoretical foundation: the first unbiased FFL approximation for arbitrary polynomial activation functions.
- Truly general-purpose component: the same layer design integrates seamlessly into LLM, ViT, NeRF, and iSDF architectures.
- Closed-form knowledge distillation eliminates retraining, offering significant plug-and-play value for pre-trained models.
- Direct dependence on input norm expands the expressive space beyond standard FFLs.

## Limitations & Future Work

- Approximation errors accumulate as more layers are replaced, potentially degrading performance at large replacement scales.
- Experiments are limited to $k \leq 2$; the practical utility of higher-order variants remains insufficiently validated.
- The combination with other compression techniques (e.g., orthogonalization, quantization) is unexplored.
- Validation on very large models (e.g., 70B LLMs) is absent.

## Related Work & Insights

- vs **SNNK/URF**: EUGens subsumes these as special cases; SNNK requires activation functions to have Fourier transforms, making it inapplicable to ReLU.
- vs **Low-Rank**: EUGens achieves significantly higher accuracy under the same parameter budget.
- vs **Pruning/Quantization**: Orthogonal and complementary; can be combined.
- vs **Instant-NGP/3DGS**: Different levels of NeRF acceleration (hash encoding/splatting vs. FFL replacement).

EUGens demonstrates that random feature methods are underexplored for deep learning acceleration. The weight–input decoupling paradigm may inspire analogous speedups for attention layers. Closed-form distillation has direct application value for large-model deployment on edge devices and real-time inference.

## Rating

- ⭐ **Novelty**: 4/5 — Outstanding theoretical contribution (unbiased polynomial approximation) with elegant architectural design.
- ⭐ **Experimental Thoroughness**: 4.5/5 — Covers four task categories across NLP, CV, and 3D; comprehensive ablation study.
- ⭐ **Writing Quality**: 4/5 — Well-balanced between theory and experiments; clear structure.
- ⭐ **Value**: 4/5 — General-purpose efficient layer replacement with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](../../ICLR2026/3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [\[NeurIPS 2025\] Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction](locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)
- [\[ICLR 2026\] Einstein Fields: A Neural Perspective To Computational General Relativity](../../ICLR2026/3d_vision/einstein_fields_a_neural_perspective_to_computational_general_relativity.md)

</div>

<!-- RELATED:END -->
