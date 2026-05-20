---
title: >-
  [Paper Note] Is Finer Better? The Limits of Microscaling Formats in Large Language Models
description: >-
  [ICLR 2026][Model Compression][microscaling quantization] This paper identifies and explains a counterintuitive anomaly in microscaling quantization — namely that reducing block size below a certain threshold *increases*…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "microscaling quantization"
  - "FP4"
  - "quantization anomaly"
  - "dynamic range"
  - "LLM quantization"
date: 2026-05-08
content_hash: c5cedf8f192eb967
---

# Is Finer Better? The Limits of Microscaling Formats in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2601.19026](https://arxiv.org/abs/2601.19026)  
**Code**: None  
**Area**: Model Compression
**Keywords**: microscaling quantization, FP4, quantization anomaly, dynamic range, LLM quantization

## TL;DR
This paper identifies and explains a counterintuitive anomaly in microscaling quantization — namely that reducing block size below a certain threshold *increases* quantization error for narrow-distribution tensors due to the limited dynamic range of the FP8 UE4M3 scale format — and proposes FP8 UE5M3 as a hardware-friendly solution.

## Background & Motivation
The growing computational and memory demands of LLMs have made reduced numerical precision a critical optimization direction. Microscaling formats achieve aggressive FP4-level compression via block-wise shared scales, with native hardware support from NVIDIA and AMD (e.g., NVFP4 uses 16 FP4 elements sharing one FP8 UE4M3 scale).

The conventional intuition holds that smaller block size → more precise per-block scale → lower quantization error. This holds when using BF16 (16-bit) scales. However, when scales are quantized to FP8 UE4M3, a **perplexity inversion** emerges — on certain models, reducing block size from 16 to 8 actually *increases* perplexity.

This counterintuitive finding carries practical significance: the industry is actively pursuing smaller block sizes to improve quantization accuracy, and if a fundamental limitation exists, design directions need to be reconsidered. The core question is: why does finer granularity sometimes perform worse, and how can it be fixed?

## Method

### Overall Architecture
A three-step analytical pipeline: (1) experimentally identify the anomaly and locate its root cause; (2) establish a theoretical framework to explain it from first principles; (3) propose the hardware-friendly UE5M3 scale format based on the analysis.

### Key Designs

1. **Experimental Localization of the Anomaly**:

    - Function: Systematically analyze quantization behavior across models and block sizes.
    - Mechanism:
        - With BF16 scales (unquantized): perplexity decreases monotonically as block size shrinks across all models — as expected.
        - With FP8 UE4M3 scales: granite-3.3-8b shows inversion at block size 16; llama-3.1-8b at block size 8; llama-2-7b shows no inversion.
        - Per-tensor MSE analysis reveals that ~25% of blocks exhibit higher error under finer granularity. MSE curves as a function of weight standard deviation $\sigma$ cross at $\sigma < 2\times10^{-2}$, where block size 8 yields higher MSE than block size 16.
    - Design Motivation: Cross-model differences stem from the width of weight distributions — models with narrower distributions (e.g., granite) are more severely affected.

2. **Theoretical Framework (Gaussian Distribution Assumption)**:

    - Function: Derive closed-form relationships between MSE and $\sigma$ from first principles.
    - Mechanism: Assuming weights $X \sim \mathcal{N}(0, \sigma)$, the total MSE is decomposed into three independent contributions:
        - $\text{MSE}_{Z, x_i \neq x_{\max}}$: quantization error for ordinary elements, determined jointly by the discretization of scale $s_k$ and FP4 element quantization.
        - $\text{MSE}_{Z, x_i = x_{\max}}$: error at the maximum-value element (zero when the scale is unquantized, nonzero after quantization).
        - $\text{MSE}_{Z, s=0}$: error when all elements in a block are rounded to zero (triggered when $x_{\max} < s_{\min}/2$).
    - The theoretical predictions match experimental data with high fidelity ($\chi^2 \approx 4 \times 10^{-8}$).
    - Design Motivation: The theory reveals the root cause — for narrow distributions, the limited dynamic range of UE4M3 (minimum nonzero value $2^{-9}$) cannot accurately represent the scale of small blocks.

3. **FP8 UE5M3 Scale Format**:

    - Function: Extend the dynamic range of the scale format to eliminate inversion anomalies.
    - Mechanism: One unused bit in the 8-bit unsigned FP8 scale is reallocated as an exponent bit: UE4M3 (4 exponent bits + 3 mantissa bits, minimum value $2^{-9}$) → UE5M3 (5 exponent bits + 3 mantissa bits, minimum value $2^{-17}$). Mantissa processing logic remains unchanged; only the exponent handling is extended by one bit.
    - Hardware Impact: Mantissa processing dominates hardware complexity; adding one exponent bit incurs negligible overhead. Scale generation can reuse existing FP8 E5M2 quantization logic.
    - Design Motivation: Extending the minimum representable scale by a factor of 256 ($2^{-9} \to 2^{-17}$) effectively addresses the representation problem for narrow-distribution tensors.

### Comparison: Per-Tensor Scaling
NVIDIA's current implementation applies per-tensor scaling to pre-amplify narrow distributions, but this approach has drawbacks: (1) sensitivity to outliers — a single large value affects the entire tensor; (2) additional absmax computation or pre-calibration required at inference time. UE5M3 achieves equal or better performance without per-tensor scaling.

## Key Experimental Results

### Main Results (FP4 Quantization, Block Size 8)

| Model | Format | Wiki PPL↓ | PIQA↑ | HellaSwag↑ | GSM8K↑ | MMLU↑ |
|------|------|----------|-------|-----------|--------|------|
| granite-3.3-8b | BF16 | 4.72 | 80.41 | 61.49 | 62.47 | 60.55 |
| granite-3.3-8b | UE4M3 | 7.43 | 76.50 | 55.98 | 32.37 | 48.82 |
| granite-3.3-8b | UE4M3-S | 5.39 | 78.84 | 58.86 | 44.88 | 55.23 |
| granite-3.3-8b | **UE5M3** | **5.04** | **79.98** | **60.26** | **56.17** | **57.51** |
| llama-3.1-8b | BF16 | 6.24 | 79.87 | 60.05 | 50.49 | 63.28 |
| llama-3.1-8b | UE4M3 | 7.23 | 78.29 | 57.72 | 32.30 | 56.18 |
| llama-3.1-8b | **UE5M3** | **6.79** | **78.84** | **58.94** | **42.15** | **60.97** |

### Ablation: Three-Term MSE Decomposition

| $\sigma$ Range | Dominant Error Term | Explanation |
|-------------|----------|------|
| Large ($>0.02$) | $\text{MSE}_{x_i \neq x_{\max}}$ | Ordinary element quantization error dominates |
| Medium (~$0.005$) | $\text{MSE}_{x_i = x_{\max}}$ | Scale quantization error at the maximum element becomes significant |
| Small ($<0.001$) | $\text{MSE}_{s=0}$ | Whole-block zero-rounding error dominates |

### Key Findings
- UE5M3 outperforms UE4M3 with per-tensor scaling on granite-3.3-8b: GSM8K score 56.17 vs. 44.88 (+11.3%).
- The theoretical framework generalizes to multiple distributions (Gaussian, uniform, Laplacian) and multiple formats (FP4/INT4/FP6 scales).
- The anomaly is more severe in models with narrow weight distributions, such as the SSM model mamba-codestral-7b.
- As block size decreases, the relative contribution of $\text{MSE}_{x_i=x_{\max}}$ increases — explaining why finer granularity can perform worse.

## Highlights & Insights
- The discovery of the "finer = worse" counterintuitive phenomenon is itself valuable — it serves as an important warning to the industry's trend of blindly pursuing smaller block sizes.
- The theoretical framework achieves remarkable precision ($\chi^2$ on the order of $10^{-8}$) and is readily extensible to new formats.
- The UE5M3 solution is elegant and simple — reallocating a single bit yields a 256× dynamic range extension with minimal hardware modification.

## Limitations & Future Work
- Only weight quantization is analyzed; anomalous behavior in activation quantization warrants further investigation.
- UE5M3 extends the maximum representable value from $2^{15}$ (UE4M3) to $2^{31}$, which may involve trade-offs for large outliers.
- The theoretical framework assumes a Gaussian distribution; while experimental validation shows strong agreement, a rigorous proof remains absent.

## Related Work & Insights
- **vs. NVFP4**: NVFP4 relies on UE4M3 with per-tensor scaling; UE5M3 achieves superior results without per-tensor scaling.
- **vs. BlockDialect**: BlockDialect extends element representation via codebooks; UE5M3 addresses the problem from the scale side. The two approaches are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Identifies a novel quantization anomaly with rigorous theoretical explanation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across multiple model types (LLM/SSM/hybrid), formats, and with near-perfect theory–experiment correspondence.
- Writing Quality: ⭐⭐⭐⭐⭐ Narrative flows clearly from phenomenon to theory to solution.
- Value: ⭐⭐⭐⭐⭐ Directly impacts hardware design and quantization practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)

</div>

<!-- RELATED:END -->
