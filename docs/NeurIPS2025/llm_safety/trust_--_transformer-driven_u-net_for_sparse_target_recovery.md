---
title: >-
  [Paper Note] TRUST -- Transformer-Driven U-Net for Sparse Target Recovery
description: >-
  [NeurIPS 2025][LLM Safety][Sparse Recovery] This paper proposes the TRUST architecture, which integrates the Transformer attention mechanism with a U-Net decoder to jointly learn the sensing operator and reconstruct spar…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Sparse Recovery"
  - "Transformer"
  - "U-Net"
  - "Inverse Problems"
  - "Sensing Matrix Learning"
date: 2026-05-08
content_hash: 1f21e7c94f9ff403
---

# TRUST -- Transformer-Driven U-Net for Sparse Target Recovery

**Conference**: NeurIPS 2025
**arXiv**: [2506.01112](https://arxiv.org/abs/2506.01112)
**Code**: None
**Area**: Signal Processing / Image Reconstruction
**Keywords**: Sparse Recovery, Transformer, U-Net, Inverse Problems, Sensing Matrix Learning

## TL;DR

This paper proposes the TRUST architecture, which integrates the Transformer attention mechanism with a U-Net decoder to jointly learn the sensing operator and reconstruct sparse signals under unknown sensing matrices, achieving significant improvements over conventional methods in SSIM and PSNR.

## Background & Motivation

In the inverse problem $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{w}$, sparse recovery solves underdetermined systems by exploiting signal sparsity. However, existing methods face the following challenges:

**Unknown Sensing Matrix**: Most traditional methods assume $\mathbf{A}$ is known, yet it is often unavailable in practice.

**Limited Training Data**: Only a small number of observation–target pairs $\{(\mathbf{x}, \mathbf{y})\}$ are available.

**Hallucination Artifacts**: Deep learning methods tend to produce hallucinations inconsistent with the true signal.

**Local–Global Features**: Pure U-Net lacks a global receptive field, while pure Transformer lacks multi-scale local features.

## Method

### Overall Architecture

TRUST adopts a hybrid encoder–decoder architecture:
- **Encoder**: A Transformer branch that captures long-range dependencies and estimates the sparse support set.
- **Decoder**: A U-Net-style decoder that refines reconstruction through multi-scale feature fusion.
- **Skip Connections**: Skip connections between Transformer layers and the decoder.

### Key Designs

1. **Transformer Encoding Branch**:

    - Multi-head self-attention layers capture global dependencies in the input observations.
    - Features at different levels of abstraction are extracted layer by layer.
    - The sparse support of the signal (i.e., which positions are nonzero) is estimated.

2. **U-Net Decoding Path**:

    - Multi-scale transposed convolutions progressively recover spatial resolution.
    - Features from each Transformer layer serve as guidance information.
    - Local detail recovery is refined.

3. **Skip Connection Design**:

    - Unlike symmetric skip connections in conventional U-Net.
    - Asymmetric connections from Transformer layers to decoder layers.
    - The decoder is granted access to image features at different levels of abstraction.

### Loss & Training

$$\mathcal{L} = \|\hat{\mathbf{x}} - \mathbf{x}\|_2^2 + \lambda_1 \|\hat{\mathbf{x}}\|_1 + \lambda_2 \mathcal{L}_{\text{SSIM}}$$

The three terms correspond to reconstruction error, sparsity regularization, and structural similarity constraint, respectively.

## Key Experimental Results

### Main Results (Sparse Signal Recovery)

| Method | PSNR (dB) ↑ | SSIM ↑ | Reconstruction Time (ms) ↓ | Hallucination Rate (%) ↓ |
|--------|------------|--------|--------------------------|--------------------------|
| ISTA | 24.3 | 0.712 | 850 | 12.5 |
| LISTA | 27.8 | 0.781 | 120 | 8.3 |
| U-Net | 30.2 | 0.845 | 45 | 15.7 |
| SwinIR | 31.5 | 0.868 | 78 | 9.2 |
| Transformer-only | 31.8 | 0.872 | 92 | 7.8 |
| **TRUST** | **34.6** | **0.921** | **52** | **3.1** |

### Results at Different Compression Ratios

| Compression Ratio (M/N) | U-Net PSNR | Transformer PSNR | TRUST PSNR | TRUST SSIM |
|------------------------|-----------|-----------------|-----------|-----------|
| 0.1 | 22.1 | 23.5 | **26.8** | 0.785 |
| 0.2 | 26.3 | 27.1 | **30.2** | 0.856 |
| 0.3 | 29.5 | 30.2 | **33.1** | 0.905 |
| 0.5 | 32.8 | 33.5 | **36.2** | 0.945 |

### Ablation Study

| Configuration | PSNR ↑ | SSIM ↑ |
|---------------|--------|--------|
| Full TRUST | 34.6 | 0.921 |
| w/o Skip Connections | 31.8 | 0.875 |
| w/o Transformer Encoder | 30.2 | 0.845 |
| w/o U-Net Decoder | 31.5 | 0.868 |
| w/o Sparsity Regularization | 33.2 | 0.902 |

### Key Findings

1. TRUST demonstrates the most pronounced advantage at low compression ratios (M/N=0.1), indicating that global context is critical under severe undersampling.
2. Skip connections contribute approximately 2.8 dB PSNR improvement, making them the most critical component.
3. The hallucination rate drops from 15.7% (U-Net) to 3.1%, confirming the robustness of the hybrid architecture.
4. Inference speed is close to that of pure U-Net and far faster than traditional iterative methods.

## Highlights & Insights

- **Hybrid Architecture Design**: The Transformer handles global reasoning while the U-Net handles multi-scale reconstruction, with clearly delineated roles.
- **Hallucination Suppression**: Reconstruction is guided by sparse support estimation, effectively mitigating the hallucination problem common in deep learning methods.
- **Blind Sparse Recovery**: The sensing matrix $\mathbf{A}$ need not be known; the model is learned end-to-end, offering strong practical utility.

## Limitations & Future Work

1. Validation is currently limited to 2D signals; extension to 3D volumetric data requires further investigation.
2. The computational overhead of the Transformer encoder remains substantial for high-resolution signals.
3. Setting the sparsity hyperparameter requires domain knowledge.
4. No comparison is made against recent diffusion-based reconstruction methods.

## Related Work & Insights

- **LISTA (Gregor & LeCun, 2010)**: Unrolls ISTA into a learnable network.
- **SwinIR**: Image restoration based on the Swin Transformer.
- **TransUNet**: A Transformer–U-Net hybrid architecture for medical image segmentation, from which this work draws similar inspiration.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 3 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Practical Value | 4 |
| Overall Recommendation | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[AAAI 2026\] Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models](../../AAAI2026/llm_safety/hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m.md)
- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)

</div>

<!-- RELATED:END -->
