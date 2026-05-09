---
title: >-
  [Paper Note] AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
description: >-
  [ICLR 2026][Model Compression][binary-coded quantization] This paper proposes AnyBCQ, a multi-precision LLM quantization framework based on Binary-Coded Quantization (BCQ). By progressively expanding precision (freezing existing bit-planes and appending residual bit-planes), a single model supports dynamic switching between 2-bit and 4-bit precision. Dedicated CUDA kernels perform computation directly at the bit-plane level, eliminating lookup-table and transpose overhead. At 2-bit, AnyBCQ substantially outperforms Any-Precision LLM in accuracy (MMLU 35.3% vs. 24.7%) and achieves up to 3.0× throughput over FP16.
tags:
  - ICLR 2026
  - Model Compression
  - binary-coded quantization
  - multi-precision inference
  - bit-plane operations
  - LLM deployment
  - CUDA kernels
date: 2026-05-08
content_hash: 9a8e583249784fad
---

# AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

**Conference**: ICLR 2026
**arXiv**: [2510.10467](https://arxiv.org/abs/2510.10467)
**Code**: [https://github.com/naver-aics/anybcq](https://github.com/naver-aics/anybcq)
**Area**: Model Compression / LLM Quantization
**Keywords**: binary-coded quantization, multi-precision inference, bit-plane operations, LLM deployment, CUDA kernels

## TL;DR
This paper proposes AnyBCQ, a multi-precision LLM quantization framework based on Binary-Coded Quantization (BCQ). By progressively expanding precision (freezing existing bit-planes and appending residual bit-planes), a single model supports dynamic switching between 2-bit and 4-bit precision. Dedicated CUDA kernels perform computation directly at the bit-plane level, eliminating lookup-table and transpose overhead. At 2-bit, AnyBCQ substantially outperforms Any-Precision LLM in accuracy (MMLU 35.3% vs. 24.7%) and achieves up to 3.0× throughput over FP16.

## Background & Motivation

**Background**: Multi-precision LLM frameworks allow a single model to dynamically select precision at runtime according to SLOs. Any-Precision LLM is the current state of the art but relies on non-uniform quantization (clustering-based), requiring lookup tables and bit-transpose operations.

**Limitations of Prior Work**: (a) Non-uniform quantization cannot be computed directly on bit-planes, incurring additional transpose and lookup overhead. (b) Existing multi-precision methods suffer accuracy collapse at very low bit-widths (e.g., 2-bit), limiting practical use to 3–4-bit regimes. (c) Storing multiple independent precision models incurs large memory overhead (9.85 GB for LLaMA-3.1-8B vs. 4.99 GB for AnyBCQ).

**Key Challenge**: Non-uniform quantization offers strong representational capacity but is ill-suited for hardware acceleration (due to lookup tables); BCQ is naturally hardware-friendly but conventionally supports only fixed precision.

**Goal**: Extend BCQ to a multi-precision setting, enabling dynamic precision switching while preserving hardware efficiency.

**Key Insight**: BCQ represents weights as a linear combination of binary bit-planes, $\hat{W} = \sum_i \alpha_i B_i$. Inference at $p$-bit precision corresponds exactly to computing over $p$ bit-planes, making BCQ a natural fit for multi-precision deployment.

**Core Idea**: Freeze low-precision bit-planes, append new bit-planes from residuals, and optimize only the scaling factors, achieving progressive precision expansion.

## Method

### Overall Architecture
AnyBCQ consists of two components: (1) **Offline quantization** — starting from a base precision $p_L$, BCQ is greedily initialized and progressively expanded to target precision $p_H$; at each step, existing binary codes are frozen and only scaling factors are optimized. (2) **Online inference** — dedicated CUDA kernels load bit-planes on demand and compute via additions/subtractions (no lookup tables), supporting per-request precision selection.

### Key Designs

1. **Progressive Precision Expansion**

    - **Function**: Starting from 2-bit BCQ and incrementally expanding to 3-bit and 4-bit.
    - **Mechanism**: At base precision $p_L$: greedy initialization → alternating refinement (LS for $\alpha$ + BS for $B$). Expanding to $p+1$: freeze $B_1, \ldots, B_p$ → set new bit-plane $B_{p+1} = \text{sign}(R_p)$ (sign of residual) → optimize all scaling factors $\{\alpha_i^{p+1}\}_{i=1}^{p+1}$ for the new precision via LS only.
    - **Design Motivation**: Sharing binary codes substantially reduces storage (binary codes dominate memory), reducing LLaMA-3.1-8B from 9.85 GB to 4.99 GB (−49%). Precision expansion is monotonically guaranteed.

2. **Direct Bit-Plane CUDA Kernels**

    - **Function**: Eliminate bit-transpose and centroid lookup operations.
    - **Mechanism**: Load one bit-plane at a time; since $B_i \in \{-1, +1\}$, GEMM reduces to additions and subtractions; use LUT-GEMM to cache common partial results; multiply by $\alpha_i$ and accumulate as partial sums; output after $p$ bit-planes.
    - **Design Motivation**: Non-uniform quantization requires bit-transpose ($O(MKp)$) and centroid lookup ($O(MK)$); BCQ's direct computation eliminates both steps. Memory bandwidth scales proportionally with precision (3-bit loads only 3 planes rather than 4 followed by discarding 1).

### Loss & Training
- Calibration: 512-sequence C4 dataset, 10 epochs of MRE optimization
- Asymmetric BCQ with group-wise quantization ($g = 128$)
- Initialization with 20 rounds of alternating refinement

## Key Experimental Results

### Main Results (LLaMA-3.1-8B)

| Method | 2-bit MMLU | 3-bit MMLU | 4-bit MMLU |
|--------|-----------|-----------|------------|
| AWQ | 24.12 | 47.28 | 60.49 |
| Any-Precision LLM | 24.66 | 55.53 | **64.04** |
| ShiftAddLLM | 24.83 | 56.53 | 63.50 |
| **AnyBCQ (Multi)** | **35.32** | 58.28 | 63.53 |
| FP16 | 65.02 | — | — |

AnyBCQ surpasses competing methods at 2-bit MMLU by more than 10 percentage points. At 4-bit, performance is comparable to Any-Precision LLM.

### Throughput
- Up to 3.0× over FP16
- Up to 1.2× over Any-Precision LLM
- Negligible overhead for dynamic precision switching

### Key Findings
- The 2-bit regime is where differentiation is greatest — AnyBCQ's BCQ formulation substantially outperforms non-uniform quantization.
- The gap between multi-precision and fixed-precision variants emerges at 3–4-bit, where the shared binary constraint restricts the optimization space.
- At 4-bit, differences across methods converge as quantization error becomes small.

## Highlights & Insights
- The central insight that **BCQ is naturally suited for multi-precision** is the core contribution — $p$-bit computation equals summing $p$ bit-planes, making BCQ the only multi-precision scheme that requires no lookup tables.
- A **49% memory reduction** (vs. multi-model storage) with competitive accuracy offers strong practical value.
- Significant progress in the traditionally difficult 2-bit regime (MMLU: 24% → 35%).

## Limitations & Future Work
- At 4-bit, accuracy slightly trails Any-Precision LLM, reflecting BCQ's representational limitations at higher precision.
- Validation is limited to LLaMA-3.1-8B; larger models remain untested.
- Binary codes frozen during progressive expansion cannot be subsequently corrected — early errors propagate to higher-precision stages.
- Only weight-only quantization is considered; activation quantization is not addressed.

## Related Work & Insights
- **vs. Any-Precision LLM**: Non-uniform quantization offers greater representational capacity but is hardware-unfriendly; BCQ trades a modest reduction in high-precision accuracy for substantial hardware efficiency gains and superior low-bit performance.
- **vs. ShiftAddLLM**: Both are BCQ-based methods, but ShiftAddLLM supports only fixed precision; AnyBCQ extends BCQ to the multi-precision setting.
- **vs. GPTQ/AWQ**: Uniform quantization collapses entirely at 2-bit; the binary bit-plane structure of BCQ is more robust at ultra-low precision.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The extension from BCQ to multi-precision is natural yet effective; the CUDA kernel design demonstrates engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive benchmarks, throughput evaluation, and ablations, though limited to a single model.
- **Writing Quality**: ⭐⭐⭐⭐ Figures 1–3 provide highly intuitive comparisons.
- **Value**: ⭐⭐⭐⭐⭐ Fills a critical gap in BCQ-based multi-precision LLM deployment with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Highly Efficient and Effective LLMs with Multi-Boolean Architectures](highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)
- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)
- [\[AAAI 2026\] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](../../AAAI2026/model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)
- [\[NeurIPS 2025\] Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression](../../NeurIPS2025/model_compression/binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)
- [\[AAAI 2026\] KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](../../AAAI2026/model_compression/kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)

</div>

<!-- RELATED:END -->
