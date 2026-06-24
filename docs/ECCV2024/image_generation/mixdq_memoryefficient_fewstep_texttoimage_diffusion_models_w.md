---
title: >-
  [Paper Note] MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed-Precision Quantization
description: >-
  [ECCV 2024][Image Generation][Diffusion model quantization] To address the issue that few-step diffusion models (such as SDXL-turbo 1-step) are harder to quantize than multi-step models, this paper proposes MixDQ, a mixed-precision quantization method. It incorporates BOS-aware text embedding quantization, metric-decoupled sensitivity analysis, and integer programming-based bit allocation. Under W4A8, it increases FID by only 0.5, achieving 3x model compression and 1.5x speed…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion model quantization"
  - "mixed-precision"
  - "text-to-image generation"
  - "BOS-aware quantization"
  - "metric decoupling"
date: 2026-05-08
content_hash: dc5eaf1eb6f14f96
---

# MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed-Precision Quantization

**Conference**: ECCV 2024  
**arXiv**: [2405.17873](https://arxiv.org/abs/2405.17873)  
**Code**: Yes (to be released)  
**Area**: Image Generation  
**Keywords**: Diffusion model quantization, mixed-precision, text-to-image generation, BOS-aware quantization, metric decoupling

## TL;DR
To address the issue that few-step diffusion models (such as SDXL-turbo 1-step) are harder to quantize than multi-step models, this paper proposes MixDQ, a mixed-precision quantization method. It incorporates BOS-aware text embedding quantization, metric-decoupled sensitivity analysis, and integer programming-based bit allocation. Under W4A8, it increases FID by only 0.5, achieving 3x model compression and 1.5x speedup.

## Background & Motivation
1. **Background**: Few-step diffusion models (such as SDXL-turbo) reduce the inference steps to 1–4 steps, significantly decreasing computational cost. However, the model memory consumption remains at 5–10GB, limiting deployment on mobile devices. Post-Training Quantization (PTQ) is an effective compression method.
2. **Limitations of Prior Work**: (1) Few-step models are much more sensitive to quantization than multi-step models—the subsequent denoising steps in multi-step models can compensate for quantization errors, whereas 1-step models lack this error-tolerance mechanism; (2) Existing methods only focus on maintaining image quality while ignoring the destruction of image-text alignment caused by quantization; (3) Layer sensitivity exhibits a "long-tail" distribution, and uniform quantization is "dragged down" by highly sensitive layers.
3. **Key Challenge**: In few-step models, a very small number of ultra-sensitive layers (mainly to_k/to_v in cross-attention) determine the overall quantization quality. The source of sensitivity in these layers is the abnormally large value of the BOS token in the text embedding (828 vs 10–15). At the same time, existing sensitivity metrics (like SQNR) overemphasize content shifts, leading to the sacrifice of quality-related layers.
4. **Goal**: (1) Handle the outlier issue of the BOS token in text embeddings; (2) Decouple the distinct impacts of quantization on image quality and content for sensitivity analysis; (3) Find the optimal mixed-precision configuration under a given resource budget.
5. **Key Insight**: Conduct in-depth analysis of layer sensitivity distribution and data distribution characteristics, identifying the root causes of the problem to address them one by one.
6. **Core Idea**: Handle outliers through BOS-aware quantization, separate quality/content sensitivity using metric decoupling, and solve for the optimal bit configuration via integer programming, achieving almost lossless quantization of few-step diffusion models.

## Method

### Overall Architecture
MixDQ consists of three steps: (1) BOS-aware quantization to handle highly sensitive text embedding layers; (2) Metric-decoupled sensitivity analysis to evaluate the layer-wise impact on image quality and content separately; (3) Integer programming to allocate optimal bitwidths based on sensitivity. The entire workflow is applied to all linear and conv layers in the UNet.

### Key Designs

1. **BOS-aware Text Embedding Quantization**:
    - **Function**: Eliminate the quantization bottleneck caused by the BOS token in CLIP text embeddings.
    - **Mechanism**: The maximum value of the first token (BOS) is discovered to be 823.5, while other tokens range only from 10 to 15. Since the BOS feature remains constant for all prompts, its floating-point outputs are pre-computed to bypass quantization and are then concatenated with the remaining quantized tokens.
    - **Design Motivation**: Quantizing tensors containing the BOS token compresses most other values close to 0, losing crucial textual information. This is the core reason for the collapse of image-text alignment after quantization.
    - **Comparison with Prior Work**: Non-uniform quantization like FP8 still suffers from residual errors, whereas BOS-aware quantization directly bypasses quantization, completely eliminating the bottleneck.

2. **Metric-Decoupled Sensitivity Analysis**:
    - **Function**: Categorize model layers into content-related (cross-attention + FFN) and quality-related (self-attention + conv) layers, evaluating their sensitivity with distinct metrics.
    - **Mechanism**: Content-related layers use SSIM to measure structural changes, and quality-related layers use SQNR to measure error. $\text{SSIM}(x,y) = l(x,y)^\alpha \cdot c(x,y)^\beta \cdot s(x,y)^\gamma$
    - **Design Motivation**: A single SQNR metric over-penalizes content changes (-3.51dB) while ignoring quality degradation (-0.26dB). This leads to content layers consuming too much high-bit budget, forcing quality layers down to extremely low bits.

3. **Integer Programming Bit Allocation**:
    - **Function**: Select the optimal 2/4/8-bit configuration for each layer under a given resource budget.
    - **Mechanism**: $\text{argmax}_{c_{i,b}} \sum_i \sum_b c_{i,b} \cdot S_{i,b}$ s.t. $\sum c_{i,b} \cdot M_{i,b} \leq B$
    - **Design Motivation**: Naive sorting methods cannot yield a global optimum, whereas integer programming can find the optimal point on the Pareto frontier in seconds.

### Loss & Training
Post-Training Quantization (PTQ) is adopted without requiring additional training. 1024 COCO prompts are used for calibration. For activation quantization, the 1% most sensitive layers are kept in FP16. Four independent groups of integer programming are conducted for weight, activation, content layers, and quality layers, respectively.

## Key Experimental Results

### Main Results

| Model | Method | W/A | FID↓ | CLIP Score↑ | ImageReward↑ | Storage Compression |
|------|------|-----|------|-------------|-------------|---------|
| SDXL-turbo 1step | FP16 | 16/16 | 17.15 | 0.2722 | 0.8631 | - |
| SDXL-turbo 1step | Q-Diffusion | 8/8 | 76.18 | 0.1772 | -1.3112 | 2× |
| SDXL-turbo 1step | Q-Diffusion | 4/8 | 118.93 | 0.1662 | -1.6353 | 4× |
| SDXL-turbo 1step | **MixDQ** | 8/8 | **17.03** | **0.2703** | **0.8415** | 2× |
| SDXL-turbo 1step | **MixDQ** | 4/8 | **17.68** | **0.2698** | **0.7822** | 4× |

### Ablation Study

| Configuration | FID↓ | CLIP↑ | IR↑ | Description |
|------|------|-------|-----|------|
| Naive W8A8 | 103.96 | 0.1478 | -1.72 | Severe baseline degradation |
| + BOS-aware | 31.65 | 0.2652 | - | Content recovered, FID drops significantly |
| + Mixed-Precision (SQNR only) | 37.35 | 0.2624 | - | SQNR biases towards content layers, degrading quality |
| + Metric-Decouple + MP | **17.03** | **0.2703** | 0.84 | Close to FP16 |

### Key Findings
- BOS-aware quantization contributes the most (FID drops from 103.96 to 31.65), fundamentally solving the loss of text alignment.
- Mixed-precision without metric decoupling degrades quality instead (FID increases from 31.65 to 37.35), validating that SQNR is biased towards content.
- Under W4A8, MixDQ only adds 0.53 FID compared to FP16, whereas Q-Diffusion's FID exceeds 100.
- Achieved 1.52× speedup for W8A8 and 3.03× memory compression for W4A16 on actual GPUs.

## Highlights & Insights
- **The insight behind BOS-aware quantization is highly elegant**: Discovering that the BOS token is constant across all prompts allows it to be pre-computed to bypass quantization. This not only solves the diffusion model problem but also serves as a reference for quantizing all Transformer-based text encoders (due to the "attention sink" phenomenon).
- **The idea of metric decoupling is highly generalizable**: Any compression of generative tasks that requires maintaining multiple metrics simultaneously can adopt a similar grouping and independent evaluation strategy.
- **Extremely practical**: The entire method requires no retraining, takes only a few minutes to obtain the Pareto frontier, and is fully compatible with existing hardware INT8 kernels.

## Limitations & Future Work
- Only INT8 GPU kernels are used; the potential of INT4 is not fully exploited.
- More advanced quantization techniques such as Adaround or QAT were not integrated.
- The conv_in and conv_out layers remain highly sensitive, and specialized quantization schemes could be designed for them.
- Currently only verified on SDXL-turbo and LCM-LoRA, which can be extended to more architectures.

## Related Work & Insights
- **vs Q-Diffusion**: Q-Diffusion is effective for multi-step models but fails completely on 1-step models (FID 76.18) since multi-step models have iterative denoising error tolerance. MixDQ completely addresses this challenge via BOS-aware + metric decoupling.
- **vs PTQD**: The noise correction of PTQD is inapplicable in 1-step deterministic sampling (linear correlation is only 0.59), leading to worse performance.
- **vs Yang et al. (SQNR-based MP)**: Pure SQNR mixed-precision still suffers a 30+ FID increase, confirming the necessity of metric decoupling.

## Supplementary Notes
- Comparison with non-uniform quantization such as NF4/FP4: FP8 still has residual errors on the BOS problem and requires special hardware support.
- Time embedding is insensitive in few-step models (unlike multi-step models) because distillation enables the network to denoise from any timestep.
- MixDQ can be integrated with QAT; the framework has good scalability.
- Almost lossless results can also be achieved under W4A16 (FID 17.23 vs 17.15 of FP16).
- There are 400 selectable configurations on the Pareto frontier, and integer programming finishes in seconds.
- Although the first token in the T5 encoder is not an outlier, the channel imbalance issue still exists.

## Rating
- Novelty: ⭐⭐⭐⭐ BOS-aware quantization and metric decoupling are original contributions with in-depth problem analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Complete ablation, Pareto analysis, physical hardware measurements, multi-metric and multi-model evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problem, with progressive layer-by-layer analysis.
- Value: ⭐⭐⭐⭐ Highly practical, providing direct assistance for the deployment of diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[ECCV 2024\] Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[CVPR 2026\] Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation](../../CVPR2026/image_generation/decoupled_residual_denoising_diffusion_models_for_unified_and_data_efficient_ima.md)

</div>

<!-- RELATED:END -->
