---
title: >-
  [Paper Note] any4: Learned 4-bit Numeric Representation for LLMs
description: >-
  [ICML 2025][Model Compression][Quantization] This paper proposes any4, a method that learns the optimal 4-bit non-uniform quantization codebook for each row of the weight matrix via k-means clustering. Without requiring weight/activation preprocessing, any4 outperforms int4/fp4/nf4 on Llama 2/3, Mistral, and Mixtral, using only a single calibration sample.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Quantization"
  - "4-bit"
  - "LLM Inference"
  - "Non-uniform Quantization"
  - "Lookup Table"
date: 2026-05-08
content_hash: df4f172524d81330
---

# any4: Learned 4-bit Numeric Representation for LLMs

**Conference**: ICML 2025  
**arXiv**: [2507.04610](https://arxiv.org/abs/2507.04610)  
**Code**: [https://github.com/facebookresearch/any4](https://github.com/facebookresearch/any4)  
**Area**: Model Compression  
**Keywords**: Quantization, 4-bit, LLM Inference, Non-uniform Quantization, Lookup Table

## TL;DR
This paper proposes any4, a method that learns the optimal 4-bit non-uniform quantization codebook for each row of the weight matrix via k-means clustering. Without requiring weight/activation preprocessing, any4 outperforms int4/fp4/nf4 on Llama 2/3, Mistral, and Mixtral, using only a single calibration sample.

## Background & Motivation

**Background**: LLM deployment requires weight compression, with 4-bit quantization being the mainstream approach. Existing numeric formats include int4 (uniform), fp4 (logarithmic/linear), and nf4 (matching Gaussian distribution).

**Limitations of Prior Work**: The mismatch between uniform int4 quantization and the Gaussian distribution of weights leads to precision loss. Fixed distributions used by fp4 and nf4 fail to adapt to the unique distribution characteristics of each layer or row of weights. Existing methods like AWQ/GPTQ require complex weight/activation preprocessing.

**Key Challenge**: A priori assumed numerical distributions (uniform, Gaussian, floating-point) are not necessarily optimal. Why not directly learn the optimal quantization codebook from the data?

**Goal**: Design an adaptive 4-bit numerical representation without preprocessing.

**Key Insight**: Apply k-means clustering to each row of the weight matrix to learn 16 optimal quantization values (lookup tables), incurring an overhead of only ~0.06 extra bits per weight element.

**Core Idea**: Cast the quantization back to classic signal processing—serving as an LLM version of the Lloyd-Max quantizer.

## Method

### Overall Architecture
For each row of the weight matrix $W$:
1. Run k-means clustering to obtain 16 cluster centers (i.e., the 4-bit codebook).
2. Map each weight value to the nearest cluster center (4-bit index).
3. Store a lookup table (LUT) of 16 bf16 values per row, dequantizing via the LUT during inference.

### Key Designs

1. **Row-wise k-means Codebook Learning**:

    - **Function**: Learn independent 16 quantized values for each row of the weight matrix.
    - **Mechanism**: k-means minimizes the quantization reconstruction error $\sum_i \|w_i - c_{\text{nearest}}\|^2$.
    - **Design Motivation**: Weight distributions of different rows can vary significantly; row-wise learning is more accurate than global matrix sharing.
    - **Overhead**: 16 × 2 bytes = 32 bytes per row, which averages out to approximately 0.0625 bits/element for 4096 columns.

2. **Group-wise Quantization Compatibility**:

    - **Function**: Superimpose group scale and zero-point on top of the k-means codebook.
    - **Mechanism**: Adding group size=128 introduces an extra 0.25 bits per element, totaling 4.3125 bits/element.
    - **Design Motivation**: Standard int4 group-wise quantization already requires 4.25 bits; any4 only adds an extra 0.06 bits of LUT overhead.

3. **Single-Sample Calibration**:

    - **Function**: Replace traditional hundreds of calibration samples with a single carefully chosen diverse sample.
    - **Mechanism**: k-means clustering is solely based on the distribution of the weights themselves, independent of activation distributions.
    - **Design Motivation**: Greatly simplifies the calibration pipeline.

### Loss & Training
- Pure PTQ (Post-Training Quantization) without fine-tuning.
- Low computational cost for k-means clustering.
- GPU implementation achieved via efficient LUT strategies (tinygemm library).

## Key Experimental Results

### Main Results
Llama-3 series perplexity comparison:

| Model | FP16 | int4 | fp4 | nf4 | **any4** |
|------|------|------|-----|-----|----------|
| Llama-3-8B | 6.14 | 6.82 | 6.65 | 6.48 | **6.35** |
| Llama-3-70B | 2.85 | 3.02 | 2.98 | 2.95 | **2.90** |

### Ablation Study

| Configuration | PPL | Description |
|------|-----|------|
| any4 (Row-wise LUT) | 6.35 | Best |
| any4 (Global LUT) | 6.52 | Degradation with shared LUT |
| nf4 | 6.48 | Fixed Gaussian distribution |
| any4 + AWQ preprocessing | 6.28 | Orthogonal and stackable with preprocessing methods |

### Key Findings
- any4 consistently outperforms int4/fp4/nf4 across all evaluated models and scales.
- Competes favorably with AWQ/GPTQ without requiring weight/activation preprocessing.
- any2 and any3 also show competitiveness at lower bit-widths.
- The tinygemm library achieves latency-optimized GPU matrix multiplication.

## Highlights & Insights
- **Back to Classics**: Applying the Lloyd-Max quantizer from signal processing to LLMs offers a simple yet powerful approach.
- Extremely low overhead (0.06 bits) for row-independent codebooks, yet yields significant precision improvement.
- Orthogonal to preprocessing methods (AWQ/GPTQ) and can be combined.
- The open-sourced tinygemm library is a practical contribution.

## Limitations & Future Work
- Row-wise k-means increases offline quantization time.
- LUT implementation might not be as efficient as native int4/fp4 on certain hardware.
- Application of any4 to KV cache quantization is not discussed.

## Related Work & Insights
- **vs nf4 (QLoRA)**: nf4 assumes a Gaussian distribution, while any4 learns directly from the data, which is more flexible.
- **vs AWQ/GPTQ**: These methods preprocess weights/activations whereas any4 does not require this, although the two can be combined.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Elegant application of a classic method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple model families, multiple scales, comprehensive comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear context, concise methodology.
- **Value**: ⭐⭐⭐⭐⭐ Practical and open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](../../AAAI2026/model_compression/specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)
- [\[NeurIPS 2025\] Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment](../../NeurIPS2025/model_compression/q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)
- [\[ICML 2025\] Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks](generalization_bounds_via_meta-learned_model_representations_pac-bayes_and_sampl.md)
- [\[ICLR 2026\] Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs](../../ICLR2026/model_compression/towards_quantization-aware_training_for_ultra-low-bit_reasoning_llms.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](../../CVPR2025/model_compression/learned_image_compression_with_dictionary-based_entropy_model.md)

</div>

<!-- RELATED:END -->
