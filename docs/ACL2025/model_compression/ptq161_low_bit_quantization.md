---
title: >-
  [Paper Note] PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models
description: >-
  [ACL 2025][Model Compression][post-training quantization] PTQ1.61 is proposed as the first post-training quantization method that effectively compresses LLM weights to a true sub-2-bit (1.61-bit) format. It achieves state-of-the-art (SOTA) performance through three key techniques: a 1D structured mask (introducing an overhead of only 0.0002-bit), block-wise scaling factor optimization, and quantization preprocessing.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "post-training quantization"
  - "low-bit"
  - "LLM compression"
  - "binarization"
  - "structured mask"
date: 2026-05-08
content_hash: a14173108e150a42
---

# PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.13179](https://arxiv.org/abs/2502.13179)  
**Code**: [zjq0455/PTQ1.61](https://github.com/zjq0455/PTQ1.61)  
**Area**: Model Compression  
**Keywords**: post-training quantization, low-bit, LLM compression, binarization, structured mask  

## TL;DR

PTQ1.61 is proposed as the first post-training quantization method that effectively compresses LLM weights to a true sub-2-bit (1.61-bit) format. It achieves state-of-the-art (SOTA) performance through three key techniques: a 1D structured mask (introducing an overhead of only 0.0002-bit), block-wise scaling factor optimization, and quantization preprocessing.

## Background & Motivation

- **Limitations of Prior Work**: Existing extremely low-bit PTQ methods (such as PB-LLM, BiLLM) claim to achieve sub-2-bit quantization, but they employ uncompressible, unstructured fine-grained masks, resulting in actual equivalent bit-widths of 2.7-bit and 2.1-bit, both exceeding 2-bit.
- **Mask Overhead**: PB-LLM and BiLLM utilize element-wise unstructured masks to distinguish salient weights, introducing an additional storage overhead of over 1-bit per weight, which makes the actual compression rate far lower than the nominal value.
- **Limitations of Scaling Factors**: Prior methods derive and solve scaling factors for each row independently, ignoring the implicit dependencies and angular deviations among rows within the weight matrix.
- **Ours**: Three key designs are proposed: (1) a 1D structured mask based on input activations to preserve salient channels in 4-bit; (2) block-wise scaling factor optimization that accounts for inter-row dependencies and angular deviations; and (3) quantization preprocessing to transform scattered salient weights into a row-wise concentrated distribution.

## Method

### Overall Architecture

The quantization pipeline of PTQ1.61 consists of: (1) utilizing quantization preprocessing via restorative LoRA to transform pre-trained model weights into a row-wise concentrated distribution; (2) using a 1D mask based on input activation channel magnitudes to select and preserve salient weight channels in 4-bit; (3) binarizing non-salient channels and employing a block-wise scaling factor optimization framework to learn the optimal scaling parameters.

### Key Designs

- **1D Structured Mask**: Through the analysis of the quantization error upper bound $\mathcal{E} \leq \sum_{i=1}^{m}(|x_i| \sum_{j=1}^{n}|w_{i,j}^q - w_{i,j}|)$, it is observed that input activation magnitudes are approximately 1000 times larger than weights (especially in the top-20% channels). Consequently, preserving salient weight rows at the channel level in 4-bit is proposed. This 1D mask requires only 0.0002-bit of auxiliary storage overhead (compared to 1-bit for PB-LLM and 1.1-bit for BiLLM).
- **Block-wise Scaling Factor Optimization**: Jointly optimizes the MSE loss and the negative log cosine similarity loss: $\mathbb{E}(f_1, f_2) = \|f_1 - f_2\|_2 + \mathcal{D}_{NLC}(f_1, f_2)$, taking into account both the quantization error propagation and output difference branches.
- **Quantization Preprocessing**: The salient weights of pre-trained models are found to be scattered, making them unsuitable for channel-wise quantization. By performing lightweight restorative LoRA fine-tuning on pre-training datasets, the salient weights are rearranged into a row-wise concentrated distribution, rendering the model more amenable to subsequent channel-level quantization.

### Loss & Training

The objective function for block-wise optimization is:

$$\arg\min_{\alpha_s^*, \alpha_r^*} \big(\mathbb{E}(\mathcal{F}(X,W), \mathcal{F}(X_q, W_q')) + \mathbb{E}(\mathcal{F}(X_q, W), \mathcal{F}(X_q, W_q'))\big)$$

where the first term reduces quantization error propagation, and the second term quantifies the output discrepancy under the same input.

## Experiments

### Main Results (WikiText2 Perplexity, lower is better)

| Method | Bits | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B | LLaMA2-7B | LLaMA2-13B |
|------|------|----------|-----------|-----------|-----------|-----------|------------|
| FP | 16 | 5.68 | 5.09 | 4.10 | 3.53 | 5.47 | 4.88 |
| GPTQ | 2 | 2.1e3 | 5.5e3 | 1.9e3 | 55.91 | 7.7e3 | 2.1e3 |
| OmniQuant | 2 | 15.47 | 13.21 | 8.81 | 7.58 | 37.37 | 17.21 |
| PB-LLM | 1.7(+1) | 102.19 | 48.11 | 26.37 | 12.91 | 66.30 | 462.84 |
| BiLLM | 1(+1.1) | 35.04 | 15.14 | 10.52 | 8.51 | 32.48 | 21.77 |
| **PTQ1.61** | **1.61** | **12.50** | **9.67** | **7.95** | **7.02** | **12.70** | **9.74** |

### Ablation Study (Average Accuracy on Downstream Tasks, LLaMA-7B)

| Method | Bits | PIQA | ARC-e | HellaS | WinoG | ARC-c | LAMB-o | Average |
|------|------|------|-------|--------|-------|-------|--------|------|
| BiLLM | 1(+1.1) | 61.10 | 40.99 | 31.80 | 53.67 | 20.64 | 23.15 | 36.00 |
| PTQ1.61 | 1.61 | 63.71 | 49.62 | 35.73 | 56.75 | 25.26 | 38.93 | 41.14 |
| FP | 16 | 78.67 | 75.29 | 56.99 | 70.01 | 41.81 | 73.57 | 63.06 |

### Key Findings

1. PTQ1.61 significantly outperforms 2-bit GPTQ and OmniQuant across all LLaMA variants, achieving superior performance at a genuinely lower bit-width.
2. Compared to BiLLM, which claims 1-bit but actually requires 2.1-bit, PTQ1.61 under a true 1.61-bit configuration achieves a perplexity that is approximately 2 to 6 times lower (e.g., 12.50 vs. 35.04 on LLaMA-7B).
3. The quantization preprocessing strategy can be universally applied to other extremely low-bit PTQ methods to yield substantial improvements, validating the insight that "pre-trained models are not necessarily the optimal starting point for quantization."
4. Larger model scales lead to smaller performance gaps between PTQ1.61 and full precision (e.g., LLaMA-65B has a PPL of only 7.02 vs. 3.53), suggesting that larger models are more quantization-friendly.

## Highlights & Insights

- The first PTQ method to achieve true sub-2-bit (1.61-bit) LLM weight compression, with a minor mask overhead of only 0.0002-bit/weight.
- A new paradigm of "quantization preprocessing" is introduced to transform salient weights into a row-wise concentrated distribution using restorative LoRA, which differs fundamentally in motivation and methodology from existing post-quantization fine-tuning methods like QLoRA.
- The structural factors of the quantization error upper bound are mathematically analyzed, identifying input activation channels as a key influencing factor.

## Limitations & Future Work

- Quantization preprocessing requires LoRA fine-tuning on a pre-training dataset (such as RedPajama), introducing extra computational overhead.
- A notable performance degradation still exists under 1.61-bit compression (LLaMA-7B PPL of 12.50 vs FP 5.68), which might be unacceptable in precision-sensitive applications.
- The evaluation primarily focuses on the LLaMA and OPT series, and the applicability to other architectures (such as Mistral and Qwen) remains to be validated.
- The proportion of 4-bit salient channels is fixed, and the effects of an adaptive ratio are yet to be explored.

## Related Work & Insights

- **LLM PTQ**: GPTQ (Hessian matrix column-wise quantization), AWQ (activation-aware preservation of 1% salient weights), SmoothQuant (smoothing channel outliers), OmniQuant (joint optimization of smoothing and quantization parameters).
- **Extremely Low-Bit Quantization**: BNN/XNOR-Net (classic binarization), PB-LLM (10% 8-bit + unstructured mask), BiLLM (multi-group binarization + fine-grained mask).
- **Post-Quantization Fine-Tuning**: Methods like QLoRA and QA-LoRA perform task fine-tuning after quantization, which is complementary to the proposed idea of "preprocessing before quantization."

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 5 |
| Practicality | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Overall Rating | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[ICLR 2026\] Quant-dLLM: Post-Training Extreme Low-Bit Quantization for Diffusion Large Language Models](../../ICLR2026/model_compression/quant-dllm_post-training_extreme_low-bit_quantization_for_diffusion_large_langua.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](../../AAAI2026/model_compression/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[ACL 2025\] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
