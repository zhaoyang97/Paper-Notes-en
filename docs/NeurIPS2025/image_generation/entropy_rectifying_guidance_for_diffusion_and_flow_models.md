---
title: >-
  [Paper Note] Entropy Rectifying Guidance for Diffusion and Flow Models
description: >-
  [NeurIPS 2025][Image Generation][diffusion models] This paper proposes Entropy Rectifying Guidance (ERG), which manipulates the Hopfield energy landscape of attention layers (via temperature scaling and step-size adjustm…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "diffusion models"
  - "guidance mechanism"
  - "attention energy"
  - "classifier-free guidance"
  - "flow matching"
date: 2026-05-08
content_hash: 6b27460116c17dc7
---

# Entropy Rectifying Guidance for Diffusion and Flow Models

**Conference**: NeurIPS 2025
**arXiv**: [2504.13987](https://arxiv.org/abs/2504.13987)  
**Code**: None  
**Area**: Image Generation
**Keywords**: diffusion models, guidance mechanism, attention energy, classifier-free guidance, flow matching

## TL;DR

This paper proposes Entropy Rectifying Guidance (ERG), which manipulates the Hopfield energy landscape of attention layers (via temperature scaling and step-size adjustment) to obtain a weak prediction signal as a substitute for the unconditional prediction in conventional CFG, simultaneously improving quality, diversity, and consistency in text-to-image, class-conditional, and unconditional generation.

## Background & Motivation

- **Background**: Diffusion models and flow matching models represent the current SOTA in image generation. Classifier-Free Guidance (CFG) is the most widely adopted guidance technique, enhancing generation quality and consistency by combining conditional and unconditional predictions.
- **Limitations of Prior Work**: CFG suffers from an inherent quality–diversity–consistency trilemma:
    - **Diversity collapse**: Higher guidance scales cause generated samples to converge to a narrow distribution.
    - **Oversaturation**: Excessively strong guidance leads to oversaturated colors.
    - **Unconditional training overhead**: Training resources must be allocated to unconditional generation.
    - **Inapplicability to unconditional sampling**: CFG relies on conditional/unconditional contrast and cannot be applied to purely unconditional generation.
- **Key Challenge**: Existing approaches such as AutoGuidance require an additional weak model (increasing memory usage), while SEG/SAG are designed for U-Net architectures and are difficult to transfer to DiT.
- **Goal**: To simultaneously improve all three dimensions of performance within a single model, without any additional training.

## Method

### Overall Architecture

The core idea of ERG is to exploit the Hopfield energy interpretation of attention layers: by modifying the attention mechanism, a "weaker" prediction signal is obtained and then contrasted with the normal prediction to perform guidance. The method consists of two components:

1. **I-ERG (Image ERG)**: Modifies the energy landscape of attention layers in the denoising model.
2. **C-ERG (Condition ERG)**: Modifies the energy landscape of attention layers in the text encoder.

The final guidance update combines the normal denoising prediction $D$ with the modified denoising prediction $D_\xi$ via a weighted sum, where $w$ denotes the guidance strength. $D_\xi$ is computed using the denoising model with modified attention layers, and $\phi_{c_\tau}$ is the weakened condition embedding obtained by modifying the text encoder attention.

### Key Designs

**Attention Manipulation from a Hopfield Energy Perspective**: Standard attention can be interpreted as a CCCP update of the Hopfield energy function. ERG introduces inference-time hyperparameters into this energy function to modify the energy landscape:

- **Temperature parameter $\tau$**: Controls the sharpness of softmax attention. $\tau < 1$ yields more uniform (smoothed) attention; $\tau > 1$ yields more concentrated (sharper) attention.
- **Pattern matching weight $\alpha$**: Controls the relative importance of the state–pattern matching term over the state–pattern norm term.
- **Step size $\gamma$**: Gradient descent step size controlling the magnitude of energy optimization.
- **Iteration count $K$**: Number of gradient descent steps for energy landscape optimization.

**Multi-Step Gradient Descent Update** (Algorithm 2): Within each attention layer, $K$ gradient descent steps are performed to update the query: $Q = Q - \gamma \cdot (Q - \alpha \cdot \text{softmax}(\tau \cdot \beta \cdot QK^T) \cdot V)$. When $\alpha = \gamma = \tau = K = 1$, this reduces to standard attention.

**Text Encoder Manipulation (C-ERG)**: Temperature scaling is applied to the self-attention of each layer in the text encoder (e.g., Llama3-8B, Flan-T5-XL), reducing the determinism of key-query matching and yielding a "blurred" condition embedding. C-ERG is applied throughout the entire denoising process.

**Denoising Model Manipulation (I-ERG)**: Modifications are applied only to specific layers and time steps (after a kickoff threshold $\kappa$), avoiding excessive penalization of negative components in the early sampling stage.

**Combination with Other Methods**: ERG can be seamlessly combined with CADS (Conditional Annealed Diffusion Sampler) and APG (Adaptive Projected Guidance) for further performance gains.

### Loss & Training

ERG is a purely inference-time method requiring no training modifications or additional model training. All hyperparameters are set at inference time only. The base model is trained with rectified flow-matching; during training, each of the two text encoders is independently disabled with probability $\sqrt{0.1}$ (approximately 10% probability of both being disabled simultaneously).

## Key Experimental Results

### Main Results

**Text-to-Image Generation** (COCO'14, 512 resolution, 1.9B parameter model):

| Method | FID | Density | Coverage | CLIPScore | VQAScore | NFE |
|--------|-----|---------|----------|-----------|----------|-----|
| CFG | 12.81 | 98.24 | 71.12 | 26.45 | 70.15 | 2 |
| APG | 11.88 | 104.07 | 73.06 | 26.54 | 72.47 | 2 |
| SAG* | 11.68 | 103.58 | 72.74 | 26.81 | 72.16 | 2 |
| **ERG** | 13.62 | **120.25** | **73.21** | **26.86** | **73.96** | 2 |
| ERG+APG | **11.37** | 115.08 | **80.50** | 26.74 | 73.55 | 2 |
| ERG+CADS | 12.87 | **128.54** | 76.23 | 26.75 | 73.45 | 2 |

**Unconditional Generation** (T2I model, empty prompt):

| Method | FID | Density | Coverage |
|--------|-----|---------|----------|
| No guidance | 101.50 | 8.99 | 3.63 |
| SEG* | 37.75 | 55.56 | 34.79 |
| **ERG** | **36.25** | **55.84** | **51.59** |

**Class-Conditional Generation** (ImageNet, DiT-XL/2, 512 resolution):

| Method | FID | Density | Coverage |
|--------|-----|---------|----------|
| CFG | 5.65 | 146.97 | **86.70** |
| **ERG** | **4.56** | **163.63** | 86.13 |

### Ablation Study

**Contribution of Each Component**:

| C-ERG | I-ERG | $\gamma$ | FID | Density | Coverage | CLIP | VQA |
|-------|-------|----------|-----|---------|----------|------|-----|
| ✗ | ✗ | ✗ | 12.81 | 98.24 | 71.12 | 26.45 | 70.15 |
| ✓ | ✗ | ✗ | 13.06 | 109.52 | 72.06 | 26.73 | 73.10 |
| ✓ | ✓ | ✗ | 13.62 | 120.25 | 73.21 | 26.86 | 73.96 |
| ✓ | ✓ | ✓ | 13.62 | 123.65 | 74.07 | 26.81 | 74.67 |

- C-ERG primarily improves CLIPScore and VQAScore (consistency).
- I-ERG primarily improves Density and Coverage (quality and diversity).
- Multi-step gradient descent ($K > 1$) yields no significant gain; $K = \gamma = 1$ is used by default.

### Key Findings

1. Compared to CFG, ERG improves Density by +22 points and VQAScore by +3.8 points while also improving Coverage.
2. ERG + APG achieves Pareto-frontier improvements across all three dimensions.
3. In unconditional generation, Coverage increases from 34.79 (SEG*) to 51.59, a gain of approximately 48%.
4. $\tau_c < 1$ improves CLIPScore while $\tau_c > 1$ improves Coverage, allowing the temperature parameter to flexibly control the diversity–consistency trade-off.

## Highlights & Insights

- **Theoretical Elegance**: By connecting attention to Hopfield energy, ERG provides a principled energy landscape perspective for guidance.
- **Strong Generality**: Applicable to conditional, unconditional, and class-conditional generation without architectural constraints (compatible with both U-Net and DiT).
- **Zero Additional Training**: A purely inference-time method requiring no weak model training, with the same memory overhead as CFG.
- **Composability**: Orthogonal to and composable with methods such as APG and CADS, yielding additional gains when combined.
- **No Increase in NFE**: Requires only 2 function evaluations per step, identical to CFG.

## Limitations & Future Work

- ERG alone is slightly inferior to SAG* on the FID metric; combining with APG is necessary to achieve optimal results.
- The hyperparameter space is large ($\alpha$, $\gamma$, $\tau$, $K$, and kickoff threshold $\kappa$), necessitating grid search.
- Integration with non-constant weighting schedules (e.g., time-dependent CFG schedules) has not been explored.
- Experiments are conducted solely on flow matching architectures; traditional DDPM/DDIM samplers have not been evaluated.
- The selection of which layers to apply I-ERG to must be determined empirically.

## Related Work & Insights

- **SEG (Hong, 2024)**: Achieves energy-smoothed guidance via Gaussian-blurred attention, but is limited to U-Net and requires 3 NFE.
- **AutoGuidance (Karras et al., 2024)**: Uses a weak model for contrast, but requires an additional model and increased memory.
- **APG (Sadat et al., 2025)**: Addresses oversaturation via projected guidance; complementary to ERG.
- **CADS (Sadat et al., 2024)**: Increases diversity through condition noise injection; can be stacked with ERG.
- **Insight**: The energy interpretation of attention mechanisms may serve as a theoretical tool for a broader range of inference-time intervention methods.

## Rating

- **Novelty**: 4/5 — Attention manipulation from a Hopfield energy perspective offers a theoretically novel approach to guidance.
- **Value**: 5/5 — Purely inference-time, zero training overhead, plug-and-play.
- **Experimental Thoroughness**: 4/5 — Multi-task and multi-ablation evaluation, though limited to the authors' own model.
- **Writing Quality**: 4/5 — Well-structured with complete theoretical derivations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Entropy](neural_entropy.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] EVODiff: Entropy-aware Variance Optimized Diffusion Inference](evodiff_entropy-aware_variance_optimized_diffusion_inference.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[NeurIPS 2025\] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)

</div>

<!-- RELATED:END -->
