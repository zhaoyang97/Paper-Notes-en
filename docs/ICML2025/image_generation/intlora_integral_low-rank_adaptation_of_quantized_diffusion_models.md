---
title: >-
  [Paper Note] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models
description: >-
  [ICML2025][Image Generation][LoRA] IntLoRA is proposed to fine-tune quantized diffusion models using integer low-rank parameters. After merging weights, quantized inference weights are directly obtained without additional PTQ, balancing both training and inference efficiency.
tags:
  - "ICML2025"
  - "Image Generation"
  - "LoRA"
  - "network quantization"
  - "diffusion models"
  - "low-rank adaptation"
  - "integer arithmetic"
  - "inference acceleration"
date: 2026-05-08
content_hash: 10307e65b9dd522a
---

# IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models

**Conference**: ICML2025  
**arXiv**: [2410.21759](https://arxiv.org/abs/2410.21759)  
**Code**: [csguoh/IntLoRA](https://github.com/csguoh/IntLoRA)  
**Area**: Image Generation  
**Keywords**: LoRA, network quantization, diffusion models, low-rank adaptation, integer arithmetic, inference acceleration

## TL;DR

IntLoRA is proposed to fine-tune quantized diffusion models using integer low-rank parameters. After merging weights, quantized inference weights are directly obtained without additional PTQ, balancing both training and inference efficiency.

## Background & Motivation

Large-scale text-to-image diffusion models (e.g., Stable Diffusion, SDXL, FLUX) perform exceptionally well in personalized generation tasks. However, full-parameter fine-tuning is limited by memory on consumer GPUs. The combination of LoRA and quantization techniques (such as QLoRA) allows direct fine-tuning on quantized weights, lowering training costs.

**Core Problem**: Existing methods use FP16 low-rank parameters during training. During merging, quantized pre-trained weights must be converted back to FP16, and another PTQ step is required during deployment. This workflow has two major drawbacks:

**Process Redundancy**: The training $\rightarrow$ dequantization $\rightarrow$ merging $\rightarrow$ re-quantization pipeline introduces extra PTQ steps, increasing deployment complexity.

**Severe Performance Degradation**: PTQ under low-bit (e.g., 4-bit) conditions leads to significant quality degradation (DINO drops from 0.48 to 0.21).

The root cause is the **arithmetic type mismatch** between pre-trained weights (INT) and adapter weights (FP16), forcing the merged weights back to the floating-point domain.

## Method

The core idea of IntLoRA is to make low-rank adaptation parameters operate in the integer domain, so that the merged weights are naturally in a quantized format. This consists of three key techniques:

### 1. Adapter-Quantization Separation (AQS)

Original LoRA initializes $\mathbf{AB}$ to zero to keep the fine-tuning starting point aligned with the pre-trained weights. However, an all-zero distribution is unfriendly to quantization (scale factor $s=0$ leads to division by zero). AQS introduces an auxiliary matrix $\mathbf{R}$ to decouple gradient from quantization:

$$\mathbf{W'} = \mathcal{Q}[\mathbf{W} - \text{sg}(\mathbf{R})] + \text{sg}(\mathbf{R}) + \mathbf{AB}$$

where $\text{sg}(\cdot)$ denotes the stop-gradient operation. $\mathbf{AB}$ is still initialized to zero to preserve the original LoRA gradient, while $\text{sg}(\mathbf{R}) + \mathbf{AB}$ provides a non-zero distribution to facilitate quantization. $\mathbf{R}$ can be generated online via distribution statistics and fixed random seeds without extra storage.

### 2. Multiplicative Low-rank Adaptation (MLA)

Original LoRA adopts an additive form $\mathbf{W} + \mathbf{AB}$. When both are quantized independently, they cannot be directly merged (they would have to share a quantizer, restricting the parameter space). MLA rewrites the addition into an equivalent multiplicative form:

$$\mathbf{W'} = \underbrace{\left[s \cdot \mathbf{I} + \frac{1}{\mathbf{W}_{\text{round}} - z} \odot (\mathbf{R} + \mathbf{AB})\right]}_{\text{Adapter term (trainable)}} \odot \underbrace{(\mathbf{W}_{\text{round}} - z)}_{\text{Pre-trained term (integer)}}$$

The adapter term and pre-trained term can use **independent quantizers**, eliminating the constraint of sharing quantization parameters.

### 3. Variance Matching Control (VMC)

The variance $\sigma_\mathbf{R}$ of the auxiliary matrix $\mathbf{R}$ faces a dilemma: too large and the original $\mathbf{W}$ cannot be reconstructed after quantization; too small and the adapter term distribution is not concentrated enough around zero. VMC aligns the variance ratio:

$$\mathbf{R}^* = r^\alpha \cdot \mathbf{R}, \quad r = \frac{\sigma_\mathbf{W}}{\sigma_\mathbf{R}}$$

The scalar $\alpha$ acts as a fine-tuning exponent to balance quantization difficulty with information preservation.

### Two Implementation Versions

**IntLoRA_MUL** (Integer Multiplication): Applies uniform affine quantization to the adapter term, and merging is completed via integer Hadamard product:

$$\mathbf{W'} = \bar{s} \cdot (\mathbf{U}_{\text{round}} - \bar{z}) \odot (\mathbf{W}_{\text{round}} - z)$$

**IntLoRA_SHIFT** (Shift): Applies $\log_2$ quantization to the adapter term, completing adaptation via shift operations:

$$\mathbf{W'} = \text{sign}(\mathbf{V}) \odot [(\mathbf{W}_{\text{round}} - z) \gg \text{shift}]$$

During training, STE (Straight-Through Estimator) is used to backpropagate quantized gradients.

## Key Experimental Results

### Subject-Driven Generation (DreamBooth, SD v1.5)

| Method | Bit-width | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|------|-------|---------|---------|
| LoRA (FP16) | W16A16 | 0.4828 | 0.6968 | 0.2954 |
| QLoRA | W8A8 | 0.4153 | 0.6661 | 0.2824 |
| IR-QLoRA | W8A8 | 0.4070 | 0.6630 | 0.2841 |
| **IntLoRA_MUL** | **W8A8** | **0.4498** | **0.6882** | **0.2858** |
| QLoRA | W4A8 | 0.2136 | 0.6134 | 0.2510 |
| QA-LoRA | W4A8 | 0.4127 | 0.6897 | 0.2700 |
| **IntLoRA_MUL** | **W4A8** | **0.4242** | **0.6913** | **0.2710** |

### Controllable Generation FID↓ (ControlNet)

| Method | 8-bit S2I | 8-bit L2F | 4-bit S2I | 4-bit L2F |
|------|-----------|-----------|-----------|-----------|
| LoRA (FP16) | 31.39 | 37.50 | 31.39 | 37.50 |
| QLoRA | 31.09 | 38.88 | 71.75 | 117.37 |
| IR-QLoRA | 31.81 | 36.30 | 35.83 | 39.63 |
| **IntLoRA_MUL** | **31.08** | **37.52** | **30.87** | **33.62** |

### Training and Inference Efficiency (SD v1.5, RTX 3090)

| Method | Bit-width | Training Speed | Model Size | Requires PTQ |
|------|------|----------|----------|---------|
| LoRA | W32A32 | 0.68s/img | 7700MB | ✔ |
| QLoRA | W8A8 | 0.85s/img | 1925MB | ✔ |
| IntLoRA_MUL | W8A8 | 0.87s/img | 1925MB | **✘** |
| QLoRA | W4A8 | 0.85s/img | 963MB | ✔ |

IntLoRA achieves a training speed comparable to QLoRA, but **eliminates the PTQ step** during inference, directly yielding quantized weights.

## Highlights & Insights

1. **Elimination of Inference PTQ**: The core contribution is converting adaptation parameters into integer operations, making the merged weights naturally quantized and enabling end-to-end, PTQ-free deployment.
2. **Mathematical Equivalence of MLA**: The reconstruction from addition to multiplication preserves mathematical equivalence while decoupling the quantizer constraints between pre-training and adaptation.
3. **AQS Gradient-Quantization Decoupling**: Intelligently utilizes stop-gradient to resolve the conflict between the "learning requires zero initialization" and "quantization requires non-zero distribution" demands.
4. **Solid Theoretical VMC Analysis**: Derives the optimal auxiliary matrix from the variance-correlation coefficient trade-off with theoretical rigor.
5. **Zero Storage Overhead for Auxiliary Matrix**: $\mathbf{R}$ is generated online using fixed seeds, avoiding any increase in model storage.
6. **Significant Advantage in 4-bit Scenarios**: While QLoRA's DINO score plummets to 0.21 under W4A8, IntLoRA maintains 0.42, showing a massive performance gap.

## Limitations & Future Work

1. **Evaluation Limited to Diffusion Models**: The method has not been verified on LLMs; though technically transferable, generalizability remains to be studied.
2. **No Training Acceleration**: STE and auxiliary matrices introduce additional computations during training, so the training speed does not exceed QLoRA.
3. **Hyperparameter Search for $\alpha$**: The exponent $\alpha$ in VMC requires searching for different tasks, lacking automation.
4. **Activation Quantization Still Uses PTQ**: The paper focuses on weight quantization, while activation quantization still relies on traditional schemes without a unified solution.
5. **IntLoRA_SHIFT Performs Worse than IntLoRA_MUL**: The accuracy of $\log_2$ quantization is limited, meaning the practical advantages mainly stem from the MUL version.

## Related Work & Insights

- **QLoRA / IR-QLoRA**: Applies LoRA on quantized weights but still requires PTQ, which is the direct target for improvement in this work.
- **QA-LoRA**: Shares parameters through group quantization, sacrificing adaptation capacity.
- **EfficientDM**: Uses LoRA for diffusion model QAT, but incurs heavy training overhead.
- **Insights**: The concept of integer-domain adaptation can be extended to other PEFT methods (such as Adapters) and hybrid-precision strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-part design (addition-to-multiplication reconstruction, AQS decoupling, and VMC adjustment) is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three tasks: subject generation, controllable generation, and style customization, along with ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, intuitive diagrams, and well-defined problem formulations.
- Value: ⭐⭐⭐⭐ Addresses real-world pain points when deploying quantized LoRA, showing outstanding advantages in 4-bit scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)
- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](../../NeurIPS2025/image_generation/gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[CVPR 2026\] Low-Rank Residual Diffusion Models](../../CVPR2026/image_generation/low-rank_residual_diffusion_models.md)

</div>

<!-- RELATED:END -->
