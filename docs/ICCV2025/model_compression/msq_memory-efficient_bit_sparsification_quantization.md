---
title: >-
  [Paper Note] MSQ: Memory-Efficient Bit Sparsification Quantization
description: >-
  [ICCV 2025][Model Compression][mixed-precision quantization] MSQ is proposed to achieve mixed-precision quantization discovery by computing the least significant bit (LSB) directly from weights via a RoundClamp quantizer…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "mixed-precision quantization"
  - "bit-level sparsity"
  - "quantization-aware training"
  - "memory-efficient"
  - "Hessian"
date: 2026-05-08
content_hash: 2c1eaccafd9d727b
---

# MSQ: Memory-Efficient Bit Sparsification Quantization

**Conference**: ICCV 2025
**arXiv**: [2507.22349](https://arxiv.org/abs/2507.22349)  
**Institution**: Sungkyunkwan University, University of Arizona
**Area**: Model Compression / Quantization / Mixed-Precision Quantization
**Keywords**: mixed-precision quantization, bit-level sparsity, quantization-aware training, memory-efficient, Hessian, model compression

## TL;DR
MSQ is proposed to achieve mixed-precision quantization discovery by computing the least significant bit (LSB) directly from weights via a RoundClamp quantizer and imposing L1 regularization to induce sparsity, without explicitly creating bit-level trainable parameters. This reduces trainable parameters by 8× and training time by 86% while maintaining competitive accuracy–compression trade-offs.

## Background & Motivation
Deploying DNNs on mobile/edge devices requires quantization. Uniform quantization suffers from accumulated noise in sensitive layers, while mixed-precision quantization yields better results but entails an enormous search space.

Limitations of existing methods:
1. **Search-based methods (HAQ)**: High computational cost; do not account for sensitivity changes during training.
2. **Sensitivity analysis (HAWQ)**: Analyzes only pre-trained models, ignoring dynamic sensitivity changes during training.
3. **Bit-level methods (BSQ/CSQ)**: Treat each bit as an independent trainable variable — multiplying trainable parameters by $n$, causing GPU memory and training time to surge.

## Core Problem
How to achieve effective bit-level sparsity without incurring the overhead of bit-level trainable parameters?

## Method

### Core Idea
Key observation: it is unnecessary to train each bit independently. The LSB is computed directly from the floating-point weights $W$, and L1 regularization is applied to drive the LSB toward zero; once zeroed, the bit can be safely pruned, reducing precision.

### Key Designs

1. **RoundClamp Quantizer**:

    - Standard STE quantization: $W_n = \text{Round}[(2^n - 1) \cdot W] / (2^n - 1)$
    - MSQ bipartite bit-slicing: decomposes an $n$-bit value into an MSB part and an LSB part.
    - RoundClamp: first rounds to $n$-bit, then clamps to the $(n{-}1)$-bit range.
    - The difference between Round and Clamp equals the contribution of the LSB.
    - Gradients of the LSB are back-propagated to the floating-point weights $W$ via STE.

2. **LSB Sparsification Regularization**:

    - $\mathcal{L}_{\text{reg}} = \lambda \cdot \sum |\text{LSB}|$
    - L1 regularization drives the LSB toward zero.
    - When the LSBs of all weights in a given layer approach zero, the layer can safely be reduced from $n$-bit to $(n{-}1)$-bit.
    - No independent bit variables need to be created.

3. **Hessian-Aware Aggressive Pruning**:

    - Hessian trace is used to estimate per-layer sensitivity.
    - Insensitive layers are assigned faster bit-pruning rates.
    - Multiple LSBs can be pruned simultaneously (e.g., directly from 4-bit to 2-bit).
    - This substantially accelerates training.

4. **Complete Training Pipeline**:

    - Initialization: all layers start from high precision.
    - Forward pass: RoundClamp quantization is applied.
    - Loss: task loss + $\lambda \cdot \mathcal{L}_1(\text{LSB})$.
    - Periodic detection of LSB sparsity: the lowest bit is pruned when the sparsity exceeds a threshold.
    - Hessian guidance enables aggressive pruning of insensitive layers.

### Core Differences from BSQ/CSQ

| | BSQ/CSQ | MSQ |
|---|---------|-----|
| Trainable Parameters | Independent variable per bit | Original floating-point weights only |
| Parameter Count | $n\times$ original | $1\times$ original |
| Multi-bit Pruning | 1 bit per step | Hessian-guided multi-bit pruning |

## Key Experimental Results

### Training Efficiency

| Method | Trainable Parameters | Training Time |
|--------|----------------------|---------------|
| BSQ | $8\times$ original | Baseline |
| **MSQ** | **$1\times$ original** | **$-86\%$** |

### ResNet Accuracy–Compression Trade-off

| Method | Model | Top-1 (%) | Compression Ratio |
|--------|-------|-----------|-------------------|
| BSQ | ResNet-18 | 69.2 | 8.0× |
| CSQ | ResNet-18 | 69.4 | 8.0× |
| **MSQ** | ResNet-18 | **69.1** | **8.0×** |

- Accuracy is on par with BSQ/CSQ while training cost is substantially reduced.

### Scalability
- First extension of bit-level quantization to ViT architectures (ViT-S/B).
- Extended to heterogeneous CNNs such as MobileNetV3.

### Ablation Study
- RoundClamp vs. standard STE: RoundClamp provides more accurate gradient direction for the LSB.
- Hessian-guided aggressive pruning: reduces training epochs by 30–40% under equivalent compression.
- L1 regularization strength: too large leads to accuracy degradation; too small slows convergence.

## Highlights & Insights
- **Eliminating bit-level parameter overhead is the core contribution**: fundamentally resolves the critical limitation of BSQ (i.e., $n$-fold parameter inflation).
- **Elegant RoundClamp quantizer design**: the LSB is constructed from the difference between Round and Clamp, which is mathematically natural and elegant.
- **Hessian-guided multi-bit pruning**: accelerates training while leveraging per-layer sensitivity information.
- **Strong scalability**: first validation of bit-level quantization on ViT and MobileNetV3.
- **Practical engineering value**: the substantial improvement in training efficiency makes mixed-precision quantization genuinely viable in resource-constrained scenarios.

## Limitations & Future Work
- Accuracy is marginally lower than BSQ/CSQ (0.1–0.3%), indicating that fine-grained control via bit-level parameters retains some value.
- L1 may not be the optimal sparsification objective; Group Lasso warrants exploration.
- Hessian computation incurs overhead, albeit far smaller than the time saved.
- Validation is limited to ImageNet classification; performance on detection and segmentation tasks remains unknown.
- Only weight quantization is addressed; activation quantization is not considered.

## Related Work & Insights
- **vs. HAQ**: Search-based with high computational cost and static analysis; MSQ dynamically discovers the precision assignment.
- **vs. BSQ**: Pioneering bit-level approach but suffers from severe parameter inflation; MSQ preserves the advantages while eliminating the overhead.
- **vs. CSQ**: Smooths BSQ but does not address the parameter inflation problem.

The insight that "each bit need not be trained independently" suggests that many problems seemingly requiring fine-grained parameters admit more efficient indirect solutions. The RoundClamp construction (difference between two quantization operations as a useful signal) offers inspiration for other quantization methods. The work has direct engineering value for on-device quantization-aware training.

## Rating
- Novelty: ⭐⭐⭐⭐ RoundClamp and LSB regularization are novel, though the overall approach is a natural extension of BSQ.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough efficiency comparisons and multi-architecture validation, but downstream task evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐ Comparison with BSQ/CSQ is clear; Figures 1 and 2 are informative.
- Value: ⭐⭐⭐⭐⭐ An 8× parameter reduction and 86% training time reduction make bit-level quantization genuinely practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](../../NeurIPS2025/model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[ICCV 2025\] Scheduling Weight Transitions for Quantization-Aware Training](scheduling_weight_transitions_for_quantization-aware_training.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](../../ICML2026/model_compression/nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)
- [\[NeurIPS 2025\] Memory-Efficient Training with In-Place FFT Implementation](../../NeurIPS2025/model_compression/memory-efficient_training_with_in-place_fft_implementation.md)

</div>

<!-- RELATED:END -->
