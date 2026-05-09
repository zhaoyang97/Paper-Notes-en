---
title: >-
  [Paper Note] LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models
description: >-
  [ICLR 2026][Multimodal VLM][Model Compression] This paper proposes LLaVA-FA, an efficient compression method for large multimodal models (LMMs) that performs joint low-rank and quantization weight approximation in the frequency domain. By exploiting the decorrelation property and conjugate symmetry of the Fourier transform, the method achieves more compact and accurate weight representations. It further introduces PolarQuant (polar coordinate quantization) and ODC (Optional Diagonal Calibration), surpassing existing efficient multimodal models on multiple benchmarks with minimal active parameters and computational cost.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Model Compression
  - Fourier Transform
  - Low-Rank Decomposition
  - Quantization
  - Multimodal Language Models
date: 2026-05-08
content_hash: 267ea2181e801a68
---

# LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models

**Conference**: ICLR 2026
**arXiv**: [2602.00135](https://arxiv.org/abs/2602.00135)
**Code**: None
**Area**: Multimodal Large Language Models
**Keywords**: Model Compression, Fourier Transform, Low-Rank Decomposition, Quantization, Multimodal Language Models

## TL;DR

This paper proposes LLaVA-FA, an efficient compression method for large multimodal models (LMMs) that performs joint low-rank and quantization weight approximation in the frequency domain. By exploiting the decorrelation property and conjugate symmetry of the Fourier transform, the method achieves more compact and accurate weight representations. It further introduces PolarQuant (polar coordinate quantization) and ODC (Optional Diagonal Calibration), surpassing existing efficient multimodal models on multiple benchmarks with minimal active parameters and computational cost.

## Background & Motivation

Large multimodal models (LMMs) demonstrate strong performance on vision-language tasks, but their substantial computational and memory requirements hinder practical deployment. For instance, training LLaVA-70B requires over 800 GPU hours on A100s.

**Limitations of existing compression methods:**

**Decoupled low-rank decomposition and quantization**: Existing methods (e.g., LoRD, ASVD, LQER) treat low-rank decomposition and quantization independently. The low-rank selection stage is oblivious to subsequent quantization noise, causing reconstruction errors to **compound**.

**Underutilization of multimodal redundancy**: Unlike pure-text LLMs, large vision-language models additionally carry cross-modal adapters from image encoders. Adapter rank inflation across visual domains makes the same low-rank-plus-quantization scheme still "bloated" for multimodal models.

**Dependence on calibration data**: Many compression methods require large-scale calibration datasets to estimate Hessian matrices.

**Core Problem**: How to aggressively compress learnable parameters while preserving the performance of multimodal models?

**Key Observation**: The Fourier transform possesses strong expressive power for data compression — extremely sparse spectral information can reconstruct high-fidelity signals. Crucially, even for weight matrices lacking spatial semantics, the Fourier transform effectively handles approximation problems. The authors find that:
- LMM weight matrices exhibit more compact singular value distributions in the frequency domain
- At the same rank, the cumulative approximation error of frequency-domain low-rank approximation is **smaller** than in the spatial domain
- The conjugate symmetry of the Fourier transform saves nearly half of the learnable parameters

## Method

### Overall Architecture

The core idea of LLaVA-FA is to shift weight matrix compression from the spatial domain to the frequency domain:
1. Apply Discrete Fourier Transform (DFT) to weight matrices
2. Perform low-rank decomposition in the frequency domain
3. Apply PolarQuant to the resulting complex-valued matrices
4. Optionally apply ODC to eliminate the need for large-scale calibration data

### Key Designs

1. **Frequency-domain low-rank decomposition**: Exploiting the decorrelation capability of DFT

    - Transform weight matrix $W$ to $\hat{W} = \text{DFT}(W)$
    - Apply truncated SVD to $\hat{W}$ in the frequency domain to obtain the low-rank approximation $\hat{W}_r = U_r \Sigma_r V_r^H$
    - **Theoretical guarantee**: The Frobenius error of same-rank low-rank approximation in the frequency domain is smaller than in the spatial domain (formally proved in the paper)
    - **Conjugate symmetry**: The DFT of a real-valued matrix satisfies conjugate symmetry, requiring only half the coefficients to be stored, further compressing parameter count
    - **Energy compaction**: DFT concentrates information into a small number of frequency components, making truncation more effective

2. **PolarQuant (Polar Coordinate Quantization)**: A quantization scheme designed for complex-valued matrices

    - Frequency-domain low-rank decomposition yields complex-valued matrices, making conventional real-valued quantization schemes inapplicable
    - PolarQuant decomposes each complex number $z = r e^{i\theta}$ into **magnitude** $r$ and **phase** $\theta$
    - Uniform quantization is applied to magnitude and phase separately, each with an independent scaling factor
    - Compared to directly quantizing real and imaginary parts, the polar representation preserves complex number structure and avoids severe phase information loss at low bit-widths
    - Supports ultra-low bit-width quantization down to 2–4 bits

3. **ODC (Optional Diagonal Calibration)**: A calibration scheme requiring no large-scale calibration data

    - Traditional methods require full Hessian matrices to calibrate compression errors, incurring high computational cost and demanding large representative datasets
    - ODC leverages the empirical observation that Hessian matrices of deep networks are often **diagonally dominant or low-rank structured**
    - Row/column means are used to approximate the full Hessian, substantially reducing computational complexity
    - The compression pipeline becomes fully calibration-data-free, or requires only a minimal amount of data

### Loss & Training

LLaVA-FA is a **post-training compression** method that requires no additional fine-tuning or training:
- Compression is applied directly to pre-trained weights via DFT → truncated low-rank decomposition → PolarQuant
- ODC calibration is also a one-time forward computation
- The compressed model is directly deployable for inference without retraining

This makes the compression pipeline extremely lightweight and practical.

## Key Experimental Results

### Main Results

The method is evaluated on multiple vision-language benchmarks covering both understanding and hallucination tasks:

| Method | Active Params | VQAv2 | GQA | SQA | POPE | Avg |
|--------|--------------|-------|-----|-----|------|-----|
| LLaVA-1.5 (Original) | 7B | Baseline | Baseline | Baseline | Baseline | Baseline |
| ASVD + Q | ~2B | Lower | Lower | Lower | Lower | Lower |
| LQER + Q | ~2B | Medium | Medium | Medium | Medium | Medium |
| **LLaVA-FA** | **Fewest** | **Highest** | **Highest** | **Highest** | **Highest** | **Highest** |

LLaVA-FA surpasses all existing efficient multimodal models on all benchmarks while maintaining the fewest active parameters and lowest computational cost.

### Ablation Study

| Configuration | Result | Note |
|--------------|--------|------|
| Spatial-domain vs. frequency-domain low-rank | Frequency domain superior | Smaller reconstruction error at same rank |
| Real/imaginary quantization vs. PolarQuant | PolarQuant superior | Preserves complex structure; more complete phase information |
| Without ODC vs. with ODC | ODC provides clear gains | Particularly significant at low bit-widths |
| Low-rank only vs. low-rank + quantization | Joint approach optimal | Joint optimization in frequency domain avoids error compounding |
| Different bit-widths | 4-bit offers best trade-off | 2-bit incurs notable degradation; 4-bit approaches full precision |

### Key Findings

1. **Frequency-domain low-rank approximation genuinely outperforms the spatial domain**: This is not merely an empirical observation — the paper provides a theoretical proof that the decorrelation property of DFT accelerates singular value decay.
2. **Conjugate symmetry yields "free" 2× compression**: Parameter count is halved with no precision loss whatsoever.
3. **PolarQuant is critical at low bit-widths**: At 2–4 bits, directly quantizing real and imaginary parts severely distorts phase information.
4. **ODC eliminates the calibration data bottleneck**: The diagonal approximation is sufficiently accurate for deep network Hessians, making compression truly plug-and-play.

## Highlights & Insights

1. **A pioneering frequency-domain compression perspective**: Shifting neural network weight compression from the spatial domain to the frequency domain is a novel direction, fully exploiting three key properties of the Fourier transform: decorrelation, conjugate symmetry, and energy compaction.
2. **Integration of theory and practice**: The method is supported not only by empirical results but also by rigorous mathematical proofs that frequency-domain low-rank approximation outperforms its spatial-domain counterpart.
3. **End-to-end consistent design philosophy**: From low-rank decomposition to quantization to calibration, every step is handled consistently in the frequency/complex domain, avoiding information loss from cross-domain conversion.
4. **Extremely low deployment barrier**: Post-training compression, no calibration data, and no fine-tuning make the method highly practical.
5. **Targeted design for multimodal models**: The paper explicitly identifies that multimodal models face greater compression challenges than pure-text LLMs due to cross-modal adapter redundancy.

## Limitations & Future Work

1. **Frequency-to-spatial domain conversion at inference**: Although storage is compressed, the inference pipeline may incur additional DFT/IDFT computational overhead.
2. **Primarily validated on the LLaVA series**: Applicability to other architectures (e.g., Qwen-VL, InternVL) requires further investigation.
3. **Ultra-low bit-width (1–2 bit) performance**: Performance degradation remains noticeable at extremely low bit-widths; combining with knowledge distillation or similar techniques may be necessary.
4. **Hardware support**: Complex arithmetic and polar coordinate quantization may lack native acceleration support in existing inference engines.
5. **Adaptation to dynamic scenarios**: Fixed truncation rank may not be optimal across all layers; adaptive rank selection strategies are worth exploring.

## Related Work & Insights

- **LoRA / QLoRA**: Pioneering work on low-rank adaptation, but operating in the spatial domain.
- **ASVD / FWSVD**: SVD-based weight compression that handles low-rank decomposition and quantization separately.
- **LQER**: Joint low-rank-plus-quantization method, but in the spatial domain.
- **GPTQ / AWQ**: Quantization-only schemes without low-rank decomposition.

The core insight of this paper is that mature frequency-domain analysis tools from signal processing can be introduced into deep learning model compression, opening an entirely new research direction. Future work may extend this to other transform domains (e.g., wavelet transforms) or other model architectures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](../../ICCV2025/multimodal_vlm/llavaprumerge_adaptive_token_reduction_for_efficient_large_m.md)
- [\[ICLR 2026\] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)
- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](../../ICCV2025/multimodal_vlm/fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)

</div>

<!-- RELATED:END -->
