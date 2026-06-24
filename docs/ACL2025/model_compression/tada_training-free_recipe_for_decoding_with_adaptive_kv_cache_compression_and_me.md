---
title: >-
  [Paper Note] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering
description: >-
  [ACL 2025][Model Compression] The authors propose TaDA—a training-free KV cache compression method. By performing head-wise mean-centering on K/V activations and then quantizing the deviations (instead of raw activations), TaDA automatically eliminates the outlier problem. Combined with layer-wise adaptive quantization bit-width search, it compresses the KV cache to 27% of the original 16-bit size while preserving near-baseline accuracy.
tags:
  - "ACL 2025"
  - "Model Compression"
date: 2026-05-08
content_hash: 2d516020e5ce6a60
---

# TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering

**Conference**: ACL 2025  
**arXiv**: [2506.04642](https://arxiv.org/abs/2506.04642)  
**Code**: None  
**Area**: Model Compression  

## TL;DR

The authors propose TaDA—a training-free KV cache compression method. By performing head-wise mean-centering on K/V activations and then quantizing the deviations (instead of raw activations), TaDA automatically eliminates the outlier problem. Combined with layer-wise adaptive quantization bit-width search, it compresses the KV cache to 27% of the original 16-bit size while preserving near-baseline accuracy.

## Background & Motivation

1. **KV cache is the primary bottleneck for long-context inference**: It grows linearly with sequence length, and the long chain-of-thought in reasoning models (such as DeepSeek-R1) further Rent-exacerbates this issue.
2. **Existing quantization methods require separate handling of outliers**: Methods like KIVI, GEAR, and KVQuant must manage sparse outlier elements separately (using separate matrices or low-rank approximations) during low-bit quantization, which increases implementation complexity and storage overhead.
3. **Layer-wise quantization bit-width has not been fully explored**: Most existing methods apply a uniform quantization bit-width across all layers, ignoring the varying sensitivity of different layers to quantization errors.
4. **Key Insight**: After head-wise mean-centering of KV activations, the distribution of deviations becomes more uniform without extreme outliers, allowing direct low-bit quantization without additional overhead.

## Method

### Overall Architecture

TaDA consists of three core components: **Mean-centering**, **Deviation Quantization**, and **Adaptive Bit-width Search**.

### 1. Mean-centering

The mean of K/V activations is computed along the head dimension:
$$K_m = \frac{1}{H}\sum_{i=1}^{H} K^i, \quad V_m = \frac{1}{H}\sum_{i=1}^{H} V^i$$

The means $K_m, V_m$ are stored in full precision (16-bit), requiring only a $\frac{1}{H}$ storage overhead (e.g., only 3.1% for a 32-head model).

### 2. Deviation Calculation and Quantization

The deviation of each head relative to the mean is calculated and quantized to low precision:
$$D_K^i = K_m - K^i, \quad D_V^i = V_m - V^i$$

During inference, rebuild from the mean and quantized deviation: $\hat{K}^i = K_m - \text{quantize}(D_K^i)$

**Storage Analysis** (taking LLaMA2-7B with 32 heads and 4-bit quantization as an example):
- Mean: $\frac{1}{32} \approx 3.1\%$
- Quantized Deviation: $\frac{4}{16} = 25\%$  
- Scale Factor: $\frac{2}{128} \approx 1.6\%$
- **Total ≈ 29%**

### 3. Auxiliary Techniques

**Residual tokens**: Keep the KV of the most recent $R$ tokens uncompressed (full precision). Once the threshold is exceeded, they are compressed in batches, mitigating the impact of low precision on recent tokens.

**Layer-wise quantization bit-width search**: Using a small sample of training data (1000 items from HotpotQA), a randomized search is employed to find the optimal quantization bit-width ($\{2, 4, 8\}$-bit) for each layer. Different layers can utilize different bit-widths, further compressing the KV cache.

### 4. Efficient Implementation

Three Triton kernels are developed to optimize online computation:
- Fused RoPE and K compression
- Fused projection computation and V compression  
- TaDA self-attention kernel adapted for Flash-Decoding

## Key Experimental Results

### Table 1: LongBench Evaluation (Average score, 8 tasks)

| Model | Method | KV Cache Ratio | Average Score |
|:---|:---|:---:|:---:|
| LLaMA2-7B | BF16 Baseline | 100% | 46.03 |
| LLaMA2-7B | KIVI-2bit | 25% | 43.09 |
| LLaMA2-7B | GEAR | 31% | 45.38 |
| LLaMA2-7B | **TaDA** | **27%** | **45.87** |
| LLaMA3-8B-it | BF16 Baseline | 100% | 49.51 |
| LLaMA3-8B-it | Quanto-4bit | 37% | 49.61 |
| LLaMA3-8B-it | **TaDA** | **35%** | **49.43** |
| Mistral-7B-it | BF16 Baseline | 100% | 49.27 |
| Mistral-7B-it | **TaDA** | **35%** | **49.07** |

TaDA achieves almost the same accuracy as the BF16 baseline with less KV cache overhead (27-35%).

### Table 2: GSM8k Chain-of-Thought Evaluation

| Model | Method | KV Cache | GSM8k |
|:---|:---|:---:|:---:|
| LLaMA2-7B | BF16 | 100% | 21.30 |
| LLaMA2-7B | KIVI-2bit | 25% | 18.31 |
| LLaMA2-7B | **TaDA** | **27%** | **21.26** |
| LLaMA3-8B-it | BF16 | 100% | 67.62 |
| LLaMA3-8B-it | GEAR | 31% | 54.76 |
| LLaMA3-8B-it | **TaDA** | **35%** | **66.73** |

In CoT scenarios, TaDA is almost lossless (dropping only 0.04 on LLaMA2), significantly outperforming GEAR (-12.86).

## Highlights & Insights

- **Mean-centering eliminates outliers**: A simple yet elegant core idea—quantizing deviations instead of raw activations automatically avoids outlier issues without requiring separate matrix storage.
- **Training-free**: A pure inference-time technique that does not require model fine-tuning and can be used out-of-the-box.
- **Layer-wise adaptive bit-width**: Allocates different quantization bit-widths to different layers through search, striking a finer balance between accuracy and compression ratio.
- **Triton kernel fused implementation**: Three custom kernels hide the online compression/decompression latency, making it highly practical for production deployment.

## Limitations & Future Work

- **Performance gap between MHA and GQA**: The advantage is highly pronounced in MHA models (like LLaMA2, where more heads lead to better mean-centering effects) but shrinks in GQA models (LLaMA3, Mistral).
- **Evaluations restricted to 7-8B models**: Not validated on larger models (70B+), where the compression efficacy and accuracy retention might differ.
- **Search overhead**: Layer-wise search using training data is required; although done only once, it adds extra steps to the deployment pipeline.
- **Residual tokens introduce extra complexity**: Under extremely low precision, managing the recent tokens requires a full-precision buffer.

## Related Work & Insights

| Dimension | TaDA | KIVI | GEAR | KVQuant |
|:---|:---|:---|:---|:---|
| Quantization Method | Mean-centering + deviation quantization | per-channel K + per-token V | Uniform quantization + low-rank error recovery | Sub-4bit non-uniform quantization |
| Outlier Handling | Automatically eliminated (no extra handling) | Requires management | Low-rank + sparse matrices | Rotation + non-uniform book |
| Layer-wise Adaptive Bit-width | ✓ (Search) | ✗ | ✗ | ✗ |
| Requires Training | ✗ | ✗ | ✗ | ✗ |
| KV Cache Ratio | 27-35% | 25-37% | 31% | ~25% |

## Rating

- ⭐⭐⭐⭐ Novelty: The idea of mean-centering to eliminate outliers is simple yet effective, and the layer-wise adaptive bit-width search adds extra value.
- ⭐⭐⭐⭐ Value: Training-free, Triton kernels are implemented, and validated on AMD GPUs, offering a clear path to deployment.
- ⭐⭐⭐ Experimental Thoroughness: Evaluated on 4 models, covering both long-context and CoT scenarios, but lacks end-to-end evaluations on larger models and inference latency/throughput.
- ⭐⭐⭐ Writing Quality: The methodology is clear, but the analysis regarding the performance on GQA being close to Quanto is not deep enough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [\[ACL 2025\] Accurate KV Cache Quantization with Outlier Tokens Tracing](accurate_kv_cache_quantization_with_outlier_tokens_tracing.md)
- [\[ACL 2026\] The Pitfalls of KV Cache Compression](../../ACL2026/model_compression/the_pitfalls_of_kv_cache_compression.md)

</div>

<!-- RELATED:END -->
