---
title: >-
  [Paper Note] BoA: Attention-aware Post-training Quantization without Backpropagation
description: >-
  [ICML 2025][Model Compression][Post-training quantization] This paper proposes BoA—the first backpropagation-free algorithm for post-training quantization that accounts for cross-layer dependencies. By constructing an attention-aware Hessian matrix, it captures inter-layer interactions within the attention module, significantly outperforming existing PTQ methods at ultra-low bit-widths (INT2).
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Post-training quantization"
  - "attention-aware"
  - "Hessian optimization"
  - "LLM quantization"
  - "cross-layer dependency"
date: 2026-05-08
content_hash: 5599e5b68c71c3a7
---

# BoA: Attention-aware Post-training Quantization without Backpropagation

**Conference**: ICML 2025  
**arXiv**: [2406.13474](https://arxiv.org/abs/2406.13474)  
**Code**: [https://github.com/SamsungLabs/BoA](https://github.com/SamsungLabs/BoA)  
**Area**: Model Compression  
**Keywords**: Post-training quantization, attention-aware, Hessian optimization, LLM quantization, cross-layer dependency

## TL;DR
This paper proposes BoA—the first backpropagation-free algorithm for post-training quantization that accounts for cross-layer dependencies. By constructing an attention-aware Hessian matrix, it captures inter-layer interactions within the attention module, significantly outperforming existing PTQ methods at ultra-low bit-widths (INT2).

## Background & Motivation

**Background**: PTQ is a key technology for LLM deployment. Methods like GPTQ use Hessian information for layer-wise optimization of quantized weights but assume independence between layers.

**Limitations of Prior Work**: Layer-wise independent optimization ignores the interactions between Q/K/V/O projection layers in the attention module—quantization error in one layer propagates through attention and affects the optimal quantization of other layers.

**Key Challenge**: Accounting for cross-layer dependencies requires a much larger Hessian matrix, resulting in massive computational and memory overhead.

**Goal**: How to efficiently incorporate attention-layer dependencies within a backpropagation-free framework?

**Key Insight**: Extend the Hessian from layer-wise reconstruction error to attention-module-level reconstruction error.

**Core Idea**: Attention-aware Hessian + Hessian relaxation + head-wise joint quantization, balancing accuracy and efficiency.

## Method

### Overall Architecture
1. Construct an attention-module-level reconstruction error objective (rather than layer-wise).
2. Derive the attention-aware Hessian matrix.
3. Reduce overhead through Hessian relaxation and efficient matrix inversion.
4. Coordinate joint quantization of Q/K/V projections head-by-head.

### Key Designs

1. **Attention-aware Hessian**:

    - **Function**: Extends the quantization objective from $\|\Delta W \cdot X\|_F^2$ to the overall reconstruction error of the attention module output.
    - **Mechanism**: The Hessian contains cross-layer information among Q/K/V layers, capturing how "quantization error in the Q-layer propagates to the output through attention weights."
    - **Design Motivation**: Softmax in attention highly couples Q/K/V, making layer-wise independent quantization sub-optimal.

2. **Hessian Relaxation and Efficient Computation**:

    - **Function**: Reduces computation through block-diagonal approximation and Cholesky decomposition.
    - **Mechanism**: Preserves cross-layer interactions within the same attention head while ignoring interactions between different heads.
    - **Design Motivation**: Q/K/V interactions are strongest within the same head, while inter-head interactions are relatively weak.

3. **Head-wise Joint Quantization**:

    - **Function**: Quantizes the Q/K/V projections of an attention head simultaneously.
    - **Mechanism**: Leverages the block structure of the attention-aware Hessian for parallel optimization.
    - **Design Motivation**: Better utilizes inter-layer dependency information compared to serial, layer-wise quantization.

### Loss & Training
- Backpropagation-free, based on the Hessian-based OBS framework.
- Compatible with activation outlier suppression methods like SmoothQuant/QuaRot.
- Computational overhead is comparable to GPTQ.

## Key Experimental Results

### Main Results
Llama-2-7B W2A16 quantization perplexity:

| Method | WikiText PPL ↓ | C4 PPL ↓ |
|------|---------------|---------|
| GPTQ | 107.8 | 89.2 |
| QuIP# | 12.7 | 14.8 |
| **BoA** | **10.2** | **12.1** |

### Ablation Study

| Configuration | PPL | Description |
|------|-----|------|
| Layer-wise Hessian (GPTQ) | 107.8 | Ignores cross-layer dependencies |
| Attention-aware Hessian (Full) | 10.0 | High memory overhead |
| Attention-aware Hessian (Relaxed) | 10.2 | Negligible accuracy loss, manageable memory |
| BoA + QuaRot | 8.1 | Optimal combination with rotation-based methods |

### Key Findings
- The advantage is most pronounced at ultra-low bit-widths (W2)—GPTQ PPL 107.8 vs BoA 10.2.
- Showcases strong synergy with QuaRot/SmoothQuant (reaching 8.1 PPL on W2A16 when combined).
- W4A4 weight-activation quantization also achieves SOTA performance.

## Highlights & Insights
- Modeling **attention cross-layer dependency** is a key breakthrough—improvements are moderate at normal bit-widths (W4) but show a massive difference under extreme compression (W2).
- The block-diagonal approximation for Hessian relaxation retains the most crucial intra-head interactions, representing an elegant engineering decision.
- Orthogonal to pre-processing methods and can be applied synergistically.

## Limitations & Future Work
- Only considers cross-layer dependencies within the attention module, while the FFN part remains layer-wise independent.
- Hessian computation still requires a forward pass on calibration data.
- The performance improvement is limited under W4 precision.

## Related Work & Insights
- **vs GPTQ**: Layer-wise Hessian, ignoring cross-layer dependencies.
- **vs QuIP#**: Uses lazy codebooks but still processes layers independently.
- **vs any4**: any4 improves codebook design, whereas BoA improves quantization optimization strategies—the two are orthogonal.

## Rating
- Novelty: ⭐⭐⭐⭐ Attention-aware Hessian is a novel technical direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across multiple models, bit-widths, and combined with various methods.
- Writing Quality: ⭐⭐⭐⭐ Technically rigorous with clear derivations.
- Value: ⭐⭐⭐⭐ Significant breakthrough in ultra-low bit-width quantization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](../../ICLR2026/model_compression/turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)
- [\[ICML 2025\] Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](merge-friendly_post-training_quantization_for_multi-target_domain_adaptation.md)
- [\[ICML 2025\] Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models](q-resafe_assessing_safety_risks_and_quantization-aware_safety_patching_for_quant.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](../../CVPR2025/model_compression/q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
