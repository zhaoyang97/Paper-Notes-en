---
title: >-
  [Paper Note] Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales
description: >-
  [CVPR 2025][Image Generation][Winograd convolution] This paper proposes to perform 8-bit quantization on the entire Winograd convolution pipeline using group-wise quantization, and resolves the issue of large dynamic ranges in the output transform via data-free fine-tuning of the scaling parameters of the Winograd transform matrix. It achieves near-lossless image generation quality and a 31.3% speedup of convolutions on diffusion models.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Winograd convolution"
  - "Group-wise quantization"
  - "Data-free fine-tuning"
  - "Diffusion model acceleration"
  - "Learnable scaling factors"
date: 2026-05-08
content_hash: 579a83fb904bff13
---

# Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales

**Conference**: CVPR 2025  
**arXiv**: [2412.19867](https://arxiv.org/abs/2412.19867)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Winograd convolution, Group-wise quantization, Data-free fine-tuning, Diffusion model acceleration, Learnable scaling factors

## TL;DR
This paper proposes to perform 8-bit quantization on the entire Winograd convolution pipeline using group-wise quantization, and resolves the issue of large dynamic ranges in the output transform via data-free fine-tuning of the scaling parameters of the Winograd transform matrix. It achieves near-lossless image generation quality and a 31.3% speedup of convolutions on diffusion models.

## Background & Motivation

**Background**: Large-scale diffusion models have extremely high inference costs, and quantization is an effective acceleration method. The Winograd fast convolution algorithm can further accelerate convolution layers, but using Winograd in the quantized domain significantly increases numerical errors.

**Limitations of Prior Work**: Directly quantizing the whole Winograd pipeline leads to severe quality degradation, primarily because of massive dynamic range discrepancies (forming a "cross" pattern) across different positions in the Winograd domain output. Prior methods either require expensive QAT to learn the transform matrices or rely on domain-specific data for fine-tuning, risking the generalization capability of foundation models.

**Key Challenge**: Group-wise quantization can handle the input transformation and element-wise multiplication in the Winograd domain, but fails to address the large range discrepancies in the output transformation; pixel-wise quantization can solve this but cannot exploit efficient integer arithmetic kernels.

**Goal**: Realize a completely data-free fully quantized Winograd pipeline that preserves the generalization capabilities of foundation models.

**Key Insight**: Winograd transform matrices can be derived from Vandermonde matrices paired with diagonal scaling matrices $S_A, S_B, S_G$. These scaling factors directly control the norm of each row of the transform matrices, thereby influencing the dynamic range distribution of the Winograd domain output.

**Core Idea**: By fine-tuning only about $n$ diagonal scaling parameters of the Winograd transform matrix and using random noise instead of real data, the dynamic range of the Winograd domain output can be balanced, enabling fully quantized pipeline quantization.

## Method

### Overall Architecture
The entire Winograd pipeline (input transform $B^TxB$, weight transform $GwG^T$, element-wise multiplication $W \odot X$, and output transform $A^TYA$) is executed entirely in 8-bit integers. The dynamic ranges of each stage are balanced by learning the scaling factors $S_B$ and $S_G$.

### Key Designs

1. **Group-Wise Quantized Full Pipeline Winograd**:

    - **Function**: Apply quantization to all four stages of Winograd convolution.
    - **Mechanism**: Partition tensors into groups of size 32/64/256 and quantize each group independently. Group sizes are constrained to multiples of the processor vector width to facilitate vectorization. While group quantization is sufficient for input transform and Hadamard product, the core challenge lies in the output transform.
    - **Design Motivation**: Group-wise quantization is more fine-grained than tensor-wise or channel-wise quantization, enabling it to cope with sharp shifts in diffusion model activation distributions across time steps without complex calibration processes.

2. **Data-Free Learnable Winograd Scaling Factors**:

    - **Function**: Address the large dynamic range issue of output transforms by fine-tuning a small number of parameters.
    - **Mechanism**: Fine-tune $n$ diagonal elements each for $S_B$ and $S_G$, with $S_A = (S_B S_G)^{-1}$ automatically determined. The optimization objective is to minimize the SQNR of the difference between standard convolution and quantized Winograd convolution output, utilizing random Gaussian/uniform noise as input. All convolutional layers share one set of scaling factors.
    - **Design Motivation**: With only about $n$ parameters, it does not modify model weights or require real data, completely preserving generalization capability.

3. **Hardware-Aware Optimized Kernel**:

    - **Function**: Translate theoretical speedup into actual performance gains.
    - **Mechanism**: Develop highly optimized group-wise quantized matrix multiplication kernels to fully exploit CPU vector instructions and maximize MAC utilization.
    - **Design Motivation**: The extra scaling operations of group-wise quantization might offset the theoretical Winograd speedup, requiring hardware-level optimization to ensure practical performance.

### Loss & Training
SQNR loss. In each iteration, $K=2$ convolutional layers are randomly selected, fed with random noise inputs, and the scaling factors are optimized via SGD.

## Key Experimental Results

### Main Results
InstaFlow-0.9B text-to-image generation (MS-COCO 2017):

| Configuration | FID↓ | CLIP↑ |
|------|------|-------|
| FP16 baseline | 23.00 | 30.19 |
| W8A8 group-wise quantization | 23.04 | 30.16 |
| W8A8 Winograd F(6,3) standard scaling | 326.96 | 5.95 |
| W8A8 Winograd F(6,3) + learned scaling | 26.58 | 29.65 |

Stable Diffusion V1.5:

| Configuration | FID↓ | CLIP↑ |
|------|------|-------|
| FP16 baseline | 21.72 | 31.72 |
| W8A8 Winograd F(6,3) + learned scaling | 20.52 | 31.53 |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Group-wise quantization only (No Winograd) | Near-lossless | Group-wise quantization provides good quality |
| Group-wise quantization + Standard Winograd | Complete collapse | Output transform range issue |
| Group-wise quantization + Winograd + Learned scaling | Near-lossless | Scaling factors are effective |

ResNet ImageNet Top-1: Ours vs BQW is 68.29% vs 66.67% (+1.62%) on ResNet-18, and 71.67% vs 69.11% (+2.56%) on ResNet-34.

### Key Findings
- Standard Winograd quantization causes catastrophic quality degradation (with FID rising from 23 to 327), but recovers to 26.58 after learning the scaling factors.
- Scaling factors trained with random noise are effective across different models and scenarios, validating the generalization ability of the data-free scheme.
- Achieves a 31.3% speedup on CPU convolution layers and an end-to-end acceleration of 12.8% for diffusion models.

## Highlights & Insights
- **Data-free fine-tuning preserves generalization**: Learn only the scaling parameters of the Winograd transform instead of model weights, and use random noise to avoid overfitting to specific data distributions. This design philosophy is highly relevant for the quantization domain.
- **Theory-driven minimal parameter set**: Precisely locates the parameters impacting the dynamic range starting from Vandermonde decomposition, drastically reducing the search space.
- **First to demonstrate the viability of quantized Winograd on large-scale diffusion models**.

## Limitations & Future Work
- Acceleration is only evaluated on CPUs, leaving GPU applicability undiscussed.
- A gap of approximately 3.5 FID remains after recovery for F(6,3).
- Only 8-bit is validated; combinations of lower bits and Winograd have not been explored.

## Related Work & Insights
- **vs BQW**: Requires a complete training pipeline, whereas this work is entirely data-free.
- **vs PAW+FSQ**: Fine-tuning with training data may cause overfitting; this work outperforms it by 2.2% on ResNet-34.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegant idea combining Vandermonde decomposition-based problem location with a data-free scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two major tasks: diffusion and classification.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivation and experimental evaluation are tightly coupled.
- Value: ⭐⭐⭐⭐ Provides a novel scheme for accelerating diffusion model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](../../ICML2026/image_generation/guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[CVPR 2025\] Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing](beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)
- [\[CVPR 2025\] MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)
- [\[CVPR 2025\] Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)
- [\[CVPR 2025\] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)

</div>

<!-- RELATED:END -->
