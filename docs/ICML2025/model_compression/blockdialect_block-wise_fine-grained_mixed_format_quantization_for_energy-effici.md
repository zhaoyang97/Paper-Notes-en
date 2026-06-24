---
title: >-
  [Paper Note] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference
description: >-
  [ICML 2025][Model Compression][Mixed-format quantization] Proposes BlockDialect—a block-wise fine-grained mixed-format quantization method for weights and activations. It selects the optimal numerical format for each block from a formatbook of FP4 variants (dialects), improving accuracy on LLaMA3-8B by 10.78% compared to MXFP4, and remaining only 5.45% below full precision.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Mixed-format quantization"
  - "FP4 variants"
  - "Fine-grained quantization"
  - "Energy-efficient inference"
  - "Activation quantization"
date: 2026-05-08
content_hash: f6861652c9dc053c
---

# BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference

**Conference**: ICML 2025  
**arXiv**: [2501.01144](https://arxiv.org/abs/2501.01144)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Mixed-format quantization, FP4 variants, Fine-grained quantization, Energy-efficient inference, Activation quantization

## TL;DR
Proposes BlockDialect—a block-wise fine-grained mixed-format quantization method for weights and activations. It selects the optimal numerical format for each block from a formatbook of FP4 variants (dialects), improving accuracy on LLaMA3-8B by 10.78% compared to MXFP4, and remaining only 5.45% below full precision.

## Background & Motivation

**Background**: 4-bit quantization is a key technology for LLM deployment. Hardware-supported fine-grained block quantization (such as Microscaling/MX formats) is becoming mainstream.

**Limitations of Prior Work**: (a) Existing methods mainly focus on "how to scale" while overlooking "how to represent"; (b) Fixed numerical formats (such as FP4 E2M1) cannot adapt to distribution variations across different data blocks; (c) Activation quantization remains challenging due to large dynamic ranges and channel outliers.

**Key Challenge**: If each block has an independent scale factor, why not also assign it an independent numerical format?

**Goal**: Design block-wise mixed-format numerical quantization.

**Key Insight**: Design FP4 variants (dialects) to adapt to the data distribution of different blocks, and implement online optimal format selection using a two-stage method.

**Core Idea**: The formatbook contains multiple FP4 variants. The optimal format for each block is determined by its distribution, and all variants can be represented as scaled integers to guarantee hardware energy efficiency.

## Method

### Overall Architecture
1. Design the DialectFP4 formatbook based on block-level data distribution analysis.
2. Two-stage online format selection: coarse classification $\rightarrow$ fine format selection.
3. Implement quantization using logical operations to maintain the area and energy efficiency of FP4 MAC units.

### Key Designs

1. **DialectFP4 Formatbook**:

    - **Function**: Design a suite of FP4 variants to cover different block data distributions.
    - **Mechanism**: Analysis shows that the representable values of FP4 E2M1, {0, 0.5, 1, 1.5, 2, 3, 4, 6}, basically match matrix-level distributions, but the distribution of maximum values differs significantly across different blocks $\rightarrow$ design variants to adjust the representation density in large-value regions.
    - **Design Motivation**: The key insight is that all variant values are selected as multiples of 0.5 (scaled integers) to ensure availability for low-precision integer MAC computations.

2. **Two-Stage Online Format Selection**:

    - **Function**: Select the optimal format for each activation block in real-time.
    - **Mechanism**: The first stage performs coarse classification based on the maximum value of the block; the second stage finely selects from candidate formats (based on simple statistics).
    - **Design Motivation**: Avoid the computational overhead of brute-force MSE searches, achieving zero-shot performance close to MSE-based methods.

3. **Full-Path Quantization**:

    - **Function**: Quantize not only weight-activation multiplication in linear layers but also activation-activation multiplication in attention.
    - **Mechanism**: Quantize both the KV cache and intermediate activations using DialectFP4.
    - **Design Motivation**: Achieve true end-to-end low-precision to maximize energy efficiency gains.

### Loss & Training
- Pure PTQ, requiring no fine-tuning.
- Format selection is performed online during inference.

## Key Experimental Results

### Main Results
LLaMA3-8B zero-shot accuracy:

| Method | Bitwidth | Accuracy | vs FP16 |
|------|------|--------|---------|
| FP16 (Full Precision) | 16-bit | 69.23% | Baseline |
| MXFP4 | 4-bit | 53.00% | -16.23% |
| **BlockDialect** | **4-bit** | **63.78%** | **-5.45%** |

### Ablation Study

| Configuration | Accuracy | Description |
|------|--------|------|
| Single FP4 E2M1 | 53.00% | Fixed format |
| Mix 2 formats | 58.2% | Limited coverage |
| Full DialectFP4 | **63.78%** | Sufficient coverage |
| Linear-only quantization | 67.47% | -1.76%, very close to FP16 |

### Key Findings
- Mixed formats bring a 10.78% improvement in accuracy (53% $\rightarrow$ 63.78%).
- Full-path quantization (including attention) also maintains acceptable accuracy (-5.45%).
- A formatbook size of 4-6 variants is sufficient.

## Highlights & Insights
- **"Why not assign a format to each block?"**—This question is very natural, yet has not been systematically addressed until now.
- The scaled integer constraint ensures hardware compatibility, which is crucial for practical implementation.
- Two-stage online selection makes the method deployable without relying on offline calibration.

## Limitations & Future Work
- Format metadata (2-3 bits per block) increases storage overhead.
- There is no public hardware implementation for custom MAC units yet.
- The formatbook design relies on empirical analysis; an automated search might be superior.

## Related Work & Insights
- **vs any4**: any4 uses LUTs for arbitrary non-uniform quantization, whereas BlockDialect uses a formatbook for structured non-uniform quantization.
- **vs MXFP4**: Standard fine-grained format, on top of which BlockDialect incorporates mixed formats.
- **vs SmoothQuant**: SmoothQuant handles "how to scale", whereas BlockDialect handles "how to represent".

## Rating
- Novelty: ⭐⭐⭐⭐ A novel perspective on mixed-format numerical quantization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple models, full-path quantization, and hardware analysis.
- Writing Quality: ⭐⭐⭐⭐ Thorough analysis with rich figures/tables.
- Value: ⭐⭐⭐⭐⭐ Offers guiding significance for next-generation quantization hardware design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations](../../ICLR2026/model_compression/improving_block-wise_llm_quantization_by_4-bit_block-wise_optimal_float_bof4_ana.md)
- [\[ICML 2025\] Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation](sketch_to_adapt_fine-tunable_sketches_for_efficient_llm_adaptation.md)
- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](../../ACL2025/model_compression/moqae_mixed_precision_kv_cache.md)
- [\[ICML 2025\] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)

</div>

<!-- RELATED:END -->
