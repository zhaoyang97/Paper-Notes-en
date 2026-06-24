---
title: >-
  [Paper Note] Quantization without Tears
description: >-
  [CVPR 2025][Multimodal Efficiency][Network Quantization] This paper proposes the QwT (Quantization without Tears) method, which compensates for quantization information loss by adding a lightweight linear compensation layer after each block of the quantized network. The parameters of this compensation layer can be obtained via a closed-form solution in under 2 minutes, significantly improving PTQ accuracy across various tasks including vision, language, and multimodality.
tags:
  - "CVPR 2025"
  - "Multimodal Efficiency"
  - "Network Quantization"
  - "Post-Training Quantization"
  - "Compensation Module"
  - "Closed-form Solution"
  - "Model Compression"
date: 2026-05-08
content_hash: 174d19fc169f7319
---

# Quantization without Tears

**Conference**: CVPR 2025  
**arXiv**: [2411.13918](https://arxiv.org/abs/2411.13918)  
**Code**: [https://github.com/wujx2001/QwT](https://github.com/wujx2001/QwT)  
**Area**: Multimodal VLM  
**Keywords**: Network Quantization, Post-Training Quantization, Compensation Module, Closed-form Solution, Model Compression

## TL;DR

This paper proposes the QwT (Quantization without Tears) method, which compensates for quantization information loss by adding a lightweight linear compensation layer after each block of the quantized network. The parameters of this compensation layer can be obtained via a closed-form solution in under 2 minutes, significantly improving PTQ accuracy across various tasks including vision, language, and multimodality.

## Background & Motivation

Existing network quantization methods face a "speed-accuracy dilemma": PTQ is fast but has poor accuracy, while QAT yields high accuracy but requires extensive training (e.g., 200 epochs). Furthermore, quantization methods are often highly complex, involving numerous hyperparameters that must be tuned for specific tasks, and they lack generality across different models and tasks. The authors argue that the root of the problem lies in the fact that existing methods strictly enforce that the quantized network structure $S^{\mathbb{Z}}$ must be identical to the original structure $S$. By allowing the addition of a few extra modules $S_c$ to compensate for quantization loss within the quantized network, they can simultaneously achieve speed, accuracy, simplicity, and generality.

## Method

### Overall Architecture

The core paradigm shift of QwT is allowing the quantized network structure to evolve from $S^{\mathbb{Z}} = S$ to $S^{\mathbb{Z}} = S \cup S_c$, where $S_c$ is a compensation module composed of a set of lightweight linear layers. The workflow consists of: (1) quantizing the model using any PTQ method; (2) adding a linear compensation layer $c_i$ after each block; (3) setting the parameters of the compensation layers using a closed-form solution. The entire process takes about 2 minutes to complete with zero hyperparameters.

### Key Designs

1. **Linear Compensation Layer (Compensation Module)**:
    - Function: Compensate for the information loss between the quantized output $y^{\mathbb{Z}}$ of each block and its original output $y$
    - Mechanism: Define $c(x) = Wx + b$; the output of each block after quantization becomes $y^{\text{QwT}} = l^{\mathbb{Z}}(x^{\mathbb{Z}}) + c(x^{\mathbb{Z}})$, where $W \in \mathbb{R}^{d_{out} \times d_{in}}$, $b \in \mathbb{R}^{d_{out}}$
    - Design Motivation: Although a single linear layer cannot precisely compensate for non-linear information loss, applying a linear correction to each block results in an overall non-linear compensation effect. Furthermore, linear layers guarantee the existence of a closed-form solution.

2. **Closed-form Solution**:
    - Function: Directly obtain the optimal parameters of the compensation layer without backpropagation
    - Mechanism: Information loss is measured as $\|Y - Y^{\mathbb{Z}}\|^2$, which is a classic linear regression problem. The closed-form solution is $W^* = (Y - Y^{\mathbb{Z}}) X^{\mathbb{Z}\top} (X^{\mathbb{Z}} X^{\mathbb{Z}\top})^{-1}$, where $b$ is absorbed into $W$ (by concatenating an all-one row vector to $X^{\mathbb{Z}}$)
    - Design Motivation: The closed-form solution guarantees extremely fast initialization (~2 minutes) with zero hyperparameters, avoiding the high training overhead of QAT methods.

3. **$R^2$ Filtering Mechanism**:
    - Function: Automatically skip blocks with poor compensation performance to prevent accuracy degradation
    - Mechanism: Compute the coefficient of determination $R^2$ for each compensation module, and only apply the closed-form initializing when $R^2 > 0$, otherwise setting $W$ and $b$ to zero (i.e., leaving the block unchanged)
    - Design Motivation: A small fraction (<5%) of blocks have low fit in linear regression, where forced compensation could instead harm accuracy; setting to zero guarantees "at least not worsening."

### Loss & Training

- **No Training (PTQ Mode)**: Directly initialize with the closed-form solution using 512 calibration images without backpropagation.
- **Optional Fine-Tuning (QwT* Mode)**: Requires fine-tuning the compensation layers and classifier head for only 1 epoch to approach QAT-level accuracy (where QAT typically requires 200 epochs).
- **Extension to QAT**: Since QAT models are already fully converged, the closed-form solution is no longer effective. It is replaced with zero initialization followed by fine-tuning.

## Key Experimental Results

### Main Results

| Dataset/Task | Model | Bitwidth | PTQ Baseline Top-1 | +QwT Top-1 | +QwT* Top-1 |
|------------|------|--------|--------------|-----------|-------------|
| ImageNet Classification | ViT-B | W4A4 | 68.5 | 76.3 | 78.5 |
| ImageNet Classification | Swin-T | W4A4 | 73.0 | 75.5 | 79.3 |
| ImageNet Classification | DeiT-T | W4A4 | 58.2 | 61.4 | 64.8 |
| ImageNet Classification | ResNet-50 | W4A4 | 62.3 | 68.5 | 72.5 |
| COCO Detection | Swin-S+MaskRCNN | W4A4 | 42.6 AP | 43.1 AP | - |
| CLIP Zero-Shot | ViT-B/32 (V+T) | W6A6 | 29.8 | 43.5 | - |
| DiT Generation | DiT-XL/2 | W4A8 | 6.75 FID | 6.06 FID | - |

### Ablation Study

| Configuration | ViT-B W4A4 Top-1 | Description |
|------|-----------------|------|
| PTQ4ViT Baseline | 30.7 | Extremely low original PTQ accuracy |
| PTQ4ViT + QwT | 70.0 | Closed-form compensation yields ~40% gain |
| RepQ-ViT Baseline | 68.5 | Strong PTQ baseline |
| RepQ-ViT + QwT | 76.3 | Still achieves a 7.8% improvement |
| Percentile Baseline (W6A6) | 56.7 | - |
| Percentile + QwT (W6A6) | 79.8 | 23.1% gain |

### Key Findings

- The overhead of QwT is minimal: inference latency increases by only about 3%, and the model size increases by about 3%.
- The effect is especially significant in low-bit (4-bit) scenarios, with an average improvement of about 5%.
- QwT is particularly effective for CLIP dual-encoder quantization: when both V+T are quantized to 6-bit, the PTQ accuracy rises from 29.8% to 43.5% (+13.7%).
- In INT4 quantization of LLaMA3-8B, QwT reduces the WikiText2 perplexity from 6.65 to 6.63, and increases the average common-sense QA accuracy from 64.90% to 65.18%.

## Highlights & Insights

- **Paradigm Innovation**: Breaks the implicit assumption that "the structure of the quantized network must match the original network," establishing a new paradigm of "quantization + compensation."
- **Extreme Simplicity**: Zero hyperparameters + closed-form solution + completion in ~2 minutes, making it one of the simplest methods to boost quantization accuracy available.
- **Black-box Compatibility**: Serves as a plugin that can be integrated on top of any PTQ method (e.g., RepQ-ViT, PTQ4ViT, GPTQ), without requiring knowledge of the underlying quantization details.
- **Cross-task Generality**: The same approach is effective across CNN, ViT, CLIP, DiT, and LLaMA, covering classification, detection, segmentation, generation, and NLU.

## Limitations & Future Work

- The compensation layer is a fully-connected linear layer; when $d_{in}$ is large, the parameter count and computational cost become non-negligible (partially mitigated using grouped convolutions in ResNet).
- The closed-form solution is ineffective for QAT models, which instead require zero-initialization followed by fine-tuning.
- While accuracy recovery is substantial during CLIP dual-encoder quantization, a noticeable gap still remains (43.5% vs. 63.4% full precision).
- The performance has not been verified on larger LLMs (e.g., 70B).

## Related Work & Insights

- Compared to block-reconstruction-based PTQ methods like BRECQ and QDrop, QwT does not alter the quantization process itself but instead provides "after-the-fact compensation."
- It shares a philosophical similarity with GPTQ's use of second-order information and compensation matrices, but GPTQ operates in the weight space while QwT operates in the output space.
- The concept of compensation modules can be extended to other compression techniques (e.g., accuracy recovery after pruning, residual learning in knowledge distillation).

## Rating

- Novelty: ⭐⭐⭐⭐ The shift in the quantization paradigm (allowing structural changes) is a significant innovation, though the linear compensation layer itself is a simple concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering five model types (CNN, ViT, CLIP, DiT, LLaMA) and five task types (classification, detection, segmentation, generation, NLU).
- Writing Quality: ⭐⭐⭐⭐ The logic is clear, the motivation is compelling, and the formula derivations are succinct.
- Value: ⭐⭐⭐⭐ Highly practical as a general-purpose plugin in the quantization toolbox, though the gains on LLMs are relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MBQ: Modality-Balanced Quantization for Large Vision-Language Models](mbq_modality-balanced_quantization_for_large_vision-language_models.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/vlm_efficiency/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] Rethinking Asymmetric Quantization: Hidden Symmetry in Vision Model Weights](../../CVPR2026/vlm_efficiency/rethinking_asymmetric_quantization_hidden_symmetry_in_vision_model_weights.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](../../CVPR2026/vlm_efficiency/masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] VLM-PTQ: Efficient Post-Training Quantization for Large Vision-Language Models](../../CVPR2026/vlm_efficiency/vlm-ptq_efficient_post-training_quantization_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
