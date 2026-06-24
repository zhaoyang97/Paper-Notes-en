---
title: >-
  [Paper Note] MBQ: Modality-Balanced Quantization for Large Vision-Language Models
description: >-
  [CVPR 2025][Multimodal Efficiency][Post-training quantization] This work identifies that the sensitivity of vision tokens and language tokens to quantization errors in large VLMs differs by more than tenfold. It proposes MBQ, a post-training quantization method that introduces a gradient-based modality-balancing factor during calibration. Under W3A16 and W4A8 configurations, MBQ improves accuracy by up to 4.4% and 11.6%, respectively, while achieving a 1.4× end-to-end acceler…
tags:
  - "CVPR 2025"
  - "Multimodal Efficiency"
  - "Post-training quantization"
  - "Modality sensitivity disparity"
  - "VLM acceleration"
  - "Weight quantization"
  - "Weight-activation quantization"
date: 2026-05-08
content_hash: 68ec73687ab18d1f
---

# MBQ: Modality-Balanced Quantization for Large Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.19509](https://arxiv.org/abs/2412.19509)  
**Code**: [GitHub](https://github.com/thu-nics/MBQ)  
**Area**: Multimodal VLM  
**Keywords**: Post-training quantization, Modality sensitivity disparity, VLM acceleration, Weight quantization, Weight-activation quantization

## TL;DR

This work identifies that the sensitivity of vision tokens and language tokens to quantization errors in large VLMs differs by more than tenfold. It proposes MBQ, a post-training quantization method that introduces a gradient-based modality-balancing factor during calibration. Under W3A16 and W4A8 configurations, MBQ improves accuracy by up to 4.4% and 11.6%, respectively, while achieving a 1.4× end-to-end acceleration.

## Background & Motivation

**Background**:  
Large VLMs (e.g., LLaVA, InternVL, QwenVL) possess massive parameters (7B-72B), presenting severe memory and computational challenges for deployment. Post-training quantization (PTQ) is an effective approach to reduce memory and computing overhead and has been extensively studied in LLMs.

**Limitations of Prior Work**:  
1. Existing PTQ methods (e.g., AWQ, GPTQ, SmoothQuant) are tailored for text-only LLMs and do not consider the specific properties of multimodal inputs.  
2. Directly applying LLM PTQ methods to VLMs leads to significant accuracy degradation.  
3. During calibration, the reconstruction errors of all tokens are treated equally, whereas tokens of different modalities exhibit vast differences in actual sensitivity.  
4. Research on VLM quantization is severely lacking—contrasting sharply with the abundant literature on LLM quantization.

**Key Challenge**:  
Existing PTQ methods treat vision and language tokens identically when minimizing quantization reconstruction error. However, language tokens are far more sensitive than vision tokens. This causes the optimization to bias toward protecting insensitive vision tokens, which conversely harms the accuracy of crucial language tokens.

**Goal**:  
Design a modality sensitivity-aware quantization method that prioritizes the protection of sensitive language tokens during the calibration process.

**Key Insight**:  
Quantify modality sensitivity by calculating the gradients of token features with respect to the SFT loss, introducing gradient magnitudes as weights into the reconstruction error optimization objective.

**Core Idea**:  
Use gradients to measure the sensitivity of tokens across different modalities and weight the reconstruction error accordingly during quantization calibration, shifting the optimization focus toward language tokens.

## Method

### Overall Architecture

MBQ is a PTQ method whose core improvement lies in the calibration process:  
1. Compute the SFT loss on calibration data.  
2. Perform backpropagation to obtain gradients of the output features for each layer.  
3. Calculate the mean absolute gradients of vision tokens and language tokens separately as modality-balancing factors.  
4. Incorporate the balancing factors into the reconstruction error objective of channel-wise equalization (CWE).  
5. Search for the optimal equalization factor E.  
6. Optionally quantize the ViT encoder as well.

### Key Design 1: Modality Sensitivity Analysis and Quantization

**Function**: Discover and quantify the sensitivity gap between vision and language tokens.

**Mechanism**:  
Using image-text pairs from the COCO caption dataset as input, compute the gradients of output features at each LLM layer with respect to the SFT loss:  
- The mean absolute gradient of language tokens $|\mathbf{g}_l|$ is **more than an order of magnitude larger** than that of vision tokens $|\mathbf{g}_v|$.  
- This indicates that a perturbation of the same magnitude affects language tokens ten times more than vision tokens.

**Two Explanations**:  
1. **Data perspective**: Visual data has high redundancy and thus possesses natural fault tolerance to small perturbations.  
2. **Model perspective**: The generated content of VLMs is primarily driven by the priors of pre-trained LLMs rather than the input images.

**Validation Experiment**: Weighting the vision token reconstruction error with a heuristic balancing factor of 0.1 boosts LLaVA-ov-7B on MMMU under W3 quantization from 36.56 to 40.22 (+3.66%).

### Key Design 2: Modality-Balanced Reconstruction Error via Taylor Expansion

**Function**: Automatically derive the optimal modality-balancing factor for each layer.

**Mechanism**: Through first-order Taylor approximation, decompose the change in SFT loss into individual contributions from vision and language tokens:

$$\|L(\hat{\mathbf{Y}})\| \leq \overline{|\mathbf{g}_v|} \cdot \|\mathbf{Y}_v - \hat{\mathbf{Y}}_v\| + \overline{|\mathbf{g}_l|} \cdot \|\mathbf{Y}_l - \hat{\mathbf{Y}}_l\|$$

Accordingly, for weight-activation quantization, the optimization objective is:  
$$\min_{\mathbf{E}} \left[\overline{|\mathbf{g}_v|} \cdot \|WX_v - Q(W*E)Q(E^{-1}*X_v)\| + \overline{|\mathbf{g}_l|} \cdot \|WX_l - Q(W*E)Q(E^{-1}*X_l)\|\right]$$

**Key Findings**: The optimal reconstruction error derived via Taylor expansion is based on MAE (Mean Absolute Error) rather than the traditional MSE, and experiments demonstrate that MAE performs better.

**Design Motivation**: Gradients naturally reflect the influence of each modality on the final output, providing a mathematically rigorous foundation for use as weighting factors. Furthermore, the balancing factors for each layer are automatically learned from data without manual tuning.

### Key Design 3: End-to-End Acceleration Implementation

**Function**: Achieve practical hardware acceleration for VLMs.

**Mechanism**:  
- **W3A16 GPU Kernel**: Design a custom CUDA kernel that fuses dequantization and GEMV. Eight 3-bit weights are packed into 3 bytes. At runtime, the kernel loads W3 weights first (reducing memory access) and performs FP16 Tensor Core computation after dequantizing them into FP16.  
- **ViT Encoder Quantization**: Since the computational cost of ViT is high for high-resolution images, W4A8 quantization is applied to ViT to accelerate the prefill phase.  
- **Combined Strategy**: Apply weight-activation quantization to ViT and weight-only quantization to the LLM.

**Design Motivation**: Quantizing only the LLM is insufficient; the ViT encoder is also a bottleneck when processing high-resolution images, requiring joint acceleration.

## Key Experimental Results

### Main Results: LLaVA-onevision-7B

| Bitwidth | Method | MMMU | SEED | OCRBench | Average |
|------|------|------|------|----------|---------|
| FP16 | - | 46.0 | 74.9 | 62.2 | 67.5 |
| W3A16 | GPTQ | 41.9 | 72.9 | 55.7 | 64.1 |
| W3A16 | AWQ | 36.6 | 53.0 | 59.3 | 60.6 |
| W3A16 | **MBQ** | **42.0** | **69.7** | **61.1** | **65.3** |
| W4A8 | SmoothQuant | 30.9 | 42.7 | 32.0 | 51.1 |
| W4A8 | **MBQ** | **42.6** | **67.7** | **52.3** | **63.1** |

- Under W3A16, MBQ achieves an average Gain of **4.7%** over AWQ.
- Under W4A8, MBQ achieves an average Gain of **12.0%** over SmoothQuant.
- Consistent improvements are also observed on larger models (InternVL2-8B, Qwen2-VL-7B).

### Validation on Large Models

- InternVL2-78B under W4A8: MBQ average 71.7 vs RTN 68.3 (+3.4)
- LLaVA-onevision-72B under W3A16: MBQ average 67.7 vs AWQ 63.7 (+4.0)

### Speedup

| Model | FP16 | W3A16 | Speedup |
|------|------|-------|--------|
| LLaVA-ov-7B (prefill) | 37.9ms | 27.1ms | 1.40× |
| LLaVA-ov-7B (decode) | 16.5ms | 13.1ms | 1.26× |

### Ablation Study

| Design Choice | MMMU |
|---------|------|
| MSE-based balanced CWE | 40.22 |
| MAE-based MBQ | **42.00** |
| Heuristic 0.1 factor | 40.22 |
| Automated gradient factor | **42.00** |

## Highlights & Insights

1. **High value of the core finding**: Identifying a >10× sensitivity difference between vision and language tokens provides critical guidance for all VLM quantization and compression tasks.
2. **Extremely simple and effective method**: By simply adding gradient weights to the reconstruction error, significant improvements are achieved with virtually zero extra overhead.
3. **Rigorous theoretical derivation**: The derivation from Taylor expansion to the MAE reconstruction target is complete, with experiments validating that MAE outperforms MSE.
4. **Broad coverage**: Supports both weight-only quantization (W3/W4A16) and weight-activation quantization (W4A8/W8A8), validated across 7B to 70B models.
5. **Practical deployment value**: The custom W3 CUDA kernel achieves a 1.4× physical speedup, offering concrete engineering utility rather than just theoretical improvements.

## Limitations & Future Work

1. Gradient computation is required during calibration, increasing the calibration time and memory footprint of PTQ.
2. Gradient calculations depend on specific calibration datasets (e.g., COCO caption), making results susceptible to dataset selection.
3. Although W3 quantization shows improvement, a noticeable gap remains compared to FP16 (65.3 vs 67.5).
4. Verified only on PTQ methods based on channel-wise equalization; integration with other paradigms like GPTQ remains unexplored.
5. The quantization strategy for the ViT encoder is relatively simple and lacks tailored design.

## Related Work & Insights

- **AWQ** [Lin et al.]: Activation-aware weight quantization, which does not consider modality disparities.
- **SmoothQuant** [Xiao et al.]: Smooths quantization difficulty via channel migrations but also ignores modality disparities.
- **SpinQuant** [Liu et al.]: Rotation matrix method, which can be applied orthogonally to MBQ.
- **Insights**: Modality sensitivity disparity is relevant not only in quantization scenarios but also in other compression methodologies like pruning and distillation. The concept that "different input tokens have different importance" can be extended to areas such as attention sparsification.

## Rating

⭐⭐⭐⭐⭐ (5/5)

**Reason**: The core observation (modality sensitivity disparity) is profound and well-supported by experiments. The method is simple yet elegant (adding gradient weights), backed by a complete theoretical derivation. The evaluation is extensive, spanning multiple models, bitwidths, and benchmarks, and includes a practical CUDA acceleration kernel. This is a rare example of a simple yet profound work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](../../CVPR2026/vlm_efficiency/masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2025\] Quantization without Tears](quantization_without_tears.md)
- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](../../NeurIPS2025/vlm_efficiency/balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/vlm_efficiency/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] VLM-PTQ: Efficient Post-Training Quantization for Large Vision-Language Models](../../CVPR2026/vlm_efficiency/vlm-ptq_efficient_post-training_quantization_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
