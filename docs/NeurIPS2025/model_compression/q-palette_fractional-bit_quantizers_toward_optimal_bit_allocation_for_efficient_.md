---
title: >-
  [Paper Note] Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment
description: >-
  [NeurIPS 2025][Model Compression][Model Quantization] This paper derives optimal bit allocation for Gaussianized weights from an information-theoretic perspective…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Model Quantization"
  - "LLM Inference"
  - "Fractional-Bit Quantization"
  - "Rate-Distortion"
  - "CUDA Optimization"
date: 2026-05-08
content_hash: d526a19e6f6736c5
---

# Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment

**Conference**: NeurIPS 2025

**arXiv**: [2509.20214](https://arxiv.org/abs/2509.20214)

**Code**: [GitHub](https://github.com/snu-mllab/Q-Palette)

**Area**: Model Compression

**Keywords**: Model Quantization, LLM Inference, Fractional-Bit Quantization, Rate-Distortion, CUDA Optimization

## TL;DR

This paper derives optimal bit allocation for Gaussianized weights from an information-theoretic perspective, proposes the Q-Palette collection of fractional-bit quantizers and a mixed-scheme quantization framework, and achieves near-optimal quantization performance with inference acceleration in LLM deployment.

## Background & Motivation

Weight-only post-training quantization (PTQ) is a key technique for reducing LLM inference latency and memory footprint, particularly on memory-constrained edge devices. However, heavy-tailed outliers in LLM weights complicate quantization.

Recent progress and remaining challenges:

**Rotation methods**: Hadamard rotation transforms weights into near-Gaussian distributions, reducing the impact of outliers.

**Integer-bit constraint**: Existing quantizers support only integer bit-widths (2/3/4-bit), preventing fine-grained bit allocation.

**Optimality gap**: A substantial gap remains between practical quantizers and the information-theoretic lower bound (Gaussian rate-distortion bound).

**Mixed-precision limitations**: Existing mixed-precision methods select from a limited set of options without considering the mixing of quantizer types.

## Method

### Overall Architecture

Q-Palette consists of two main components: (1) a collection of fractional-bit quantizers spanning different bit-widths and distortion levels; and (2) a mixed-scheme framework that jointly optimizes quantizer selection and layer-fusion decisions.

### Key Designs

**1. Information-Theoretically Optimal Bit Allocation**

- The Rate-Distortion function is derived for Gaussianized (post-rotation) weights.
- Optimal allocation requires quantizers of **arbitrary precision**, not merely integer bit-widths.
- The optimal bit-width per layer is derived as: $b_l^* = \frac{1}{2}\log_2\frac{\sigma_l^2}{\lambda}$, where $\lambda$ is determined by the total bit budget.

**2. Fractional-Bit Quantizer Collection**

Q-Palette provides a range of quantizers spanning near-optimal distortion to fast inference:
- **Trellis-coded quantizers (TCQ)**: Implemented via finite-state machines, achieving distortion close to the Gaussian rate-distortion bound.
- **Vector quantizers (VQ)**: Exploit multi-dimensional coding to achieve fractional bit-widths (e.g., 2.5-bit).
- **Scalar quantizers**: Simple and efficient, suited for latency-sensitive scenarios.
- All quantizers are equipped with optimized **CUDA kernels**.

**3. Mixed-Scheme Quantization Framework**

- Jointly optimizes two decisions: (a) which quantizer to assign to each layer, and (b) whether to fuse adjacent layers.
- Dynamic programming is used to find the optimal scheme under given resource constraints.
- Objective: $\min \sum_l D_l(q_l) \quad \text{s.t.} \quad \sum_l R_l(q_l) \leq B$

### Loss & Training

- Training-free: purely post-training quantization (PTQ).
- Calibration: requires few or zero calibration samples.
- Allocation optimization: solved iteratively via Lagrangian multiplier methods.

## Key Experimental Results

### Main Results

Perplexity (PPL, WikiText-2) of LLaMA-2-7B at various average bit-widths:

| Method | 2-bit | 2.5-bit | 3-bit | 4-bit |
|--------|-------|---------|-------|-------|
| GPTQ | diverge | 12.85 | 8.32 | 6.09 |
| QuIP# | 9.15 | 7.85 | 6.83 | 5.98 |
| AQLM | 8.78 | 7.52 | 6.71 | 5.95 |
| Q-Palette (Ours) | **8.21** | **7.18** | **6.52** | **5.92** |

LLaMA-2-13B (PPL, WikiText-2):

| Method | 2-bit | 2.5-bit | 3-bit | 4-bit |
|--------|-------|---------|-------|-------|
| GPTQ | diverge | 9.15 | 6.85 | 5.42 |
| QuIP# | 7.52 | 6.58 | 5.92 | 5.35 |
| AQLM | 7.28 | 6.35 | 5.85 | 5.32 |
| Q-Palette (Ours) | **6.85** | **6.12** | **5.73** | **5.30** |

### Ablation Study

Distortion–speed trade-off across quantizer types (LLaMA-2-7B, 3-bit):

| Quantizer | PPL | Decoding Latency (ms/token) | Gap to DRB |
|-----------|-----|----------------------------|------------|
| Scalar (Uniform) | 7.15 | 2.1 | +1.23 dB |
| Vector (Group-VQ) | 6.78 | 3.5 | +0.52 dB |
| TCQ | 6.55 | 5.2 | +0.08 dB |
| Gaussian Rate-Distortion Bound | — | — | 0 dB |

### Key Findings

1. Fractional-bit quantization yields the greatest advantage in low-bit regimes (< 3-bit), narrowing the gap to the theoretical optimum.
2. The TCQ quantizer achieves distortion only 0.08 dB above the Gaussian rate-distortion bound, approaching the information-theoretic limit.
3. The mixed-scheme framework further reduces PPL by 0.3–0.5 compared to uniformly applying a single quantizer type.
4. Optimized CUDA kernels keep TCQ inference overhead within a practical range.

## Highlights & Insights

- **Information-theoretic perspective**: Grounding the work in Rate-Distortion theory provides a well-defined theoretical optimality target for quantization.
- **Engineering completeness**: The paper delivers not only theory but also full CUDA kernel implementations ready for deployment.
- **Flexible composition**: Q-Palette allows different layers to adopt different quantization strategies, maximizing overall efficiency.

## Limitations & Future Work

1. TCQ encoding and decoding complexity is relatively high, limiting practical inference speedup.
2. The current work supports weight-only quantization and does not extend to KV cache quantization.
3. The search space for the mixed-scheme framework grows with model scale.
4. Performance degradation remains significant in ultra-low-bit regimes (< 2-bit).

## Related Work & Insights

- **QuIP#** (Tseng et al.): Quantization based on randomized rotation.
- **AQLM** (Egiazarian et al.): Adaptive grouped quantization.
- **Rate-Distortion Theory**: Shannon information theory provides the lower bound for quantization.

## Rating

- ⭐ Novelty: 8/10 — Applying information-theoretic tools to LLM quantization offers a fresh perspective.
- ⭐ Practicality: 9/10 — Open-source code with CUDA kernels enables direct deployment.
- ⭐ Writing Quality: 8/10 — The transition from theory to engineering is presented smoothly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression](learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic](falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)

</div>

<!-- RELATED:END -->
