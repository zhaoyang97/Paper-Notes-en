---
title: >-
  [Paper Note] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization
description: >-
  [ICLR 2026][Model Compression][Zeroth-order optimization] This paper proposes QZO, a method that estimates gradients via zeroth-order perturbations applied to quantization scaling factors (rather than discrete weights)…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Zeroth-order optimization"
  - "quantized model fine-tuning"
  - "memory-efficient training"
  - "quantization scaling factors"
  - "gradient variance"
date: 2026-05-08
content_hash: daa54e36dd377bb0
---

# Fine-tuning Quantized Neural Networks with Zeroth-order Optimization

**Conference**: ICLR 2026
**arXiv**: [2505.13430](https://arxiv.org/abs/2505.13430)
**Code**: [GitHub](https://github.com/maifoundations/QZO)
**Area**: Model Compression / Efficient Fine-Tuning
**Keywords**: Zeroth-order optimization, quantized model fine-tuning, memory-efficient training, quantization scaling factors, gradient variance

## TL;DR
This paper proposes QZO, a method that estimates gradients via zeroth-order perturbations applied to quantization scaling factors (rather than discrete weights), and stabilizes training with directional derivative clipping (DDC). QZO enables memory-efficient fine-tuning of 4-bit/2-bit LLMs with over 18× total memory reduction.

## Background & Motivation

**Background**: Fine-tuning LLMs requires storing weights, gradients, optimizer states, and activations — a typical 7B model demands 56 GB. Existing approaches compress individual components: LoRA reduces trainable parameters, GaLore compresses optimizer states, and MeZO eliminates gradient storage via zeroth-order optimization.

**Limitations of Prior Work**: These methods address only part of the memory problem. Weights themselves remain a significant bottleneck — a 7B model in bfloat16 requires 14 GB — and even with MeZO eliminating gradients, 14 GB must still be allocated for weights. The most direct remedy is weight quantization (e.g., int4 requires only ~3.5 GB), but quantized weights are discrete and cannot be directly perturbed in zeroth-order schemes.

**Key Challenge**: Zeroth-order optimization requires perturbations in a continuous space, yet quantized weights are discrete; the estimated gradients are continuous and thus cannot directly update discrete weights without a dequantize–requantize cycle.

**Goal**: Enable zeroth-order optimization on quantized models while maximizing memory compression across weights, gradients, and optimizer states simultaneously.

**Key Insight**: Quantization can be expressed as $w = \Delta \cdot \bar{w}$, where $\Delta$ is a continuous scaling factor and $\bar{w}$ is a discrete integer. Perturbations can be applied to the continuous $\Delta$ while keeping $\bar{w}$ fixed.

**Core Idea**: Perturb continuous quantization scaling factors for zeroth-order gradient estimation, and control gradient variance via directional derivative clipping.

## Method

### Overall Architecture
QZO = Q-SPSA (quantized zeroth-order gradient estimation) + DDC (directional derivative clipping). The quantized integer weights $\bar{\theta}$ remain fixed; only the continuous scaling factors $\Delta$ are updated. Two forward passes suffice for gradient estimation — no backpropagation, no gradient storage, and no optimizer states are required.

### Key Designs

1. **Q-SPSA (Quantized Simultaneous Perturbation Stochastic Approximation)**:

    - **Function**: Extends SPSA to quantized models by perturbing continuous scaling factors rather than discrete weights.
    - **Mechanism**: $\hat{\nabla}_{\Delta}\mathcal{L} = \frac{\mathcal{L}((\Delta+\epsilon z)\odot\bar{\theta}) - \mathcal{L}((\Delta-\epsilon z)\odot\bar{\theta})}{2\epsilon}z$, where $z \sim \mathcal{N}(0, I_d)$
    - **Design Motivation**: $\Delta$ is continuous and thus naturally amenable to perturbation and gradient-based updates. The dequantization $w = \Delta \cdot \bar{w}$ is consistent with standard forward passes, requiring no modification to inference code. The approach applies to both scalar-based (GPTQ) and codebook-based (AQLM) quantization schemes.

2. **DDC (Directional Derivative Clipping)**:

    - **Function**: Clips the scalar directional derivative $d$ in the zeroth-order gradient estimate to stabilize training.
    - **Mechanism**: $d' = \text{clip}(d, -C, C)$, yielding a clipped gradient estimate $\hat{\nabla} = d' \cdot z$.
    - **Design Motivation**: Zeroth-order gradient estimates suffer from high variance (a known issue also present in MeZO). Theorem 1 proves that clipping preserves unbiasedness while reducing variance, since $d'^2 \leq d^2$.

3. **Memory Seed Trick**:

    - **Function**: Reproduces the perturbation vector $z$ on-the-fly via a random seed, avoiding explicit storage.
    - **Mechanism**: Identical to MeZO — a seed index replaces direct storage of $z$.
    - **Design Motivation**: Storing $z$ would have the same memory footprint as the model itself, negating the savings.

### Loss & Training
- ZO-SGD update: $\Delta_{t+1} = \max(\Delta_t - \eta \cdot d' \cdot z,\ 0)$ (enforcing non-negativity of scaling factors).
- Learning rate $10^{-7}$, perturbation scale $\epsilon = 10^{-3}$, clipping threshold $C = 100$.
- Optional: joint update of $\Delta$ via Q-SPSA and non-quantized parameters via SPSA.

## Key Experimental Results

### Main Results
4-bit GPTQ quantized models evaluated on SST-2 / RTE / CB / BoolQ / SQuAD:

| Method | Precision | Memory | SST-2 | RTE | SQuAD |
|--------|-----------|--------|-------|-----|-------|
| Zero-Shot | 16bit | 14 GB | baseline | baseline | baseline |
| Fine-tuning + AdamW | 16bit | 56 GB | upper bound | upper bound | upper bound |
| MeZO | 16bit | 14 GB | strong | strong | strong |
| QZO (4bit) | 4bit | **<3 GB** | near MeZO | near MeZO | near MeZO |

### Extreme Quantization Experiment (2-bit AQLM, Llama-2-13B)

| Configuration | Memory | Performance | Notes |
|---------------|--------|-------------|-------|
| Zero-Shot-Q (2bit) | ~5 GB | baseline | post-quantization zero-shot |
| QZO (2bit) | ~5 GB | **significantly above baseline** | effective under extreme compression |
| MeZO (16bit) | 26 GB | reference | requires 5× more memory |

### Key Findings
- QZO achieves fine-tuning performance close to MeZO (14 GB) using less than 3 GB — an 18× memory reduction.
- QZO remains significantly effective over the zero-shot baseline under 2-bit extreme quantization.
- DDC is critical for training stability; without it, loss spikes occur frequently.
- The clipping threshold $C$ yields stable performance over a wide range (50–200).

## Highlights & Insights
- **Unified framework for extreme compression**: QZO simultaneously eliminates gradients, optimizer states, and compresses weights, achieving memory savings along all three axes. The 18× reduction enables fine-tuning of 13B models on a 24 GB GPU.
- **Perturbing scaling factors rather than weights**: This avoids the cumbersome dequantize → perturb → requantize pipeline. The key insight is the factorization $w = \Delta \cdot \bar{w}$, which exposes a continuous component amenable to perturbation.
- **Theoretical guarantee for DDC**: The paper proves that clipping preserves unbiasedness while strictly reducing variance — a clean and rigorous theoretical result.

## Limitations & Future Work
- Zeroth-order optimization converges slowly, requiring significantly more update steps (~20k) compared to first-order methods (a few hundred).
- Evaluation is limited to NLU tasks (classification and QA); effectiveness on generative tasks such as instruction following remains unexplored.
- Only scaling factors can be fine-tuned (limited granularity); unlike LoRA, QZO cannot learn new low-rank parameters.
- Comparison with LoRA-based quantized fine-tuning (e.g., QLoRA) is absent.

## Related Work & Insights
- **vs. MeZO**: MeZO applies zeroth-order optimization to full-precision models; QZO applies it to quantized models, achieving comparable performance at 1/5 the memory footprint.
- **vs. QLoRA**: QLoRA combines quantization with LoRA-based fine-tuning but still requires gradient storage. QZO eliminates gradient storage entirely, achieving lower memory at a potential cost in expressiveness.
- **vs. ZO-signSGD**: Prior quantized zeroth-order work requires quantizing the perturbation noise and applying sign SGD on discrete weights; QZO is more efficient and flexible.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The idea of perturbing quantization scaling factors is intuitive and effective.
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited dataset variety; comparison with QLoRA is missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear and theoretical proofs are complete.
- **Value**: ⭐⭐⭐⭐ — Directly practical for extremely resource-constrained deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)

</div>

<!-- RELATED:END -->
