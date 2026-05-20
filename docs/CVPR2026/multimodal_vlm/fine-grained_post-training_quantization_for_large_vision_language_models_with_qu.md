---
title: >-
  [Paper Note] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients
description: >-
  [CVPR 2026][Multimodal VLM][post-training quantization] This paper proposes Quantization-aware Integrated Gradients (QIG), advancing sensitivity analysis for LVLM quantization from the modality level to the token level.…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "post-training quantization"
  - "LVLM compression"
  - "token-level sensitivity"
  - "integrated gradients"
  - "model acceleration"
date: 2026-05-08
content_hash: e853abcdfaa2bf5b
---

# Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients

**Conference**: CVPR 2026
**arXiv**: [2603.17809](https://arxiv.org/abs/2603.17809)  
**Code**: [https://github.com/ucas-xiang/QIG](https://github.com/ucas-xiang/QIG)  
**Area**: Multimodal VLM
**Keywords**: post-training quantization, LVLM compression, token-level sensitivity, integrated gradients, model acceleration

## TL;DR
This paper proposes Quantization-aware Integrated Gradients (QIG), advancing sensitivity analysis for LVLM quantization from the modality level to the token level. By leveraging axiomatic attribution principles, QIG precisely quantifies each token's contribution to quantization error, achieving significant accuracy improvements under W4A8 and W3A16 settings with negligible additional computational overhead.

## Background & Motivation
**Background**: LVLMs (e.g., LLaVA, InternVL, Qwen-VL) demonstrate strong performance on multimodal tasks, yet their large model sizes and slow inference speeds make post-training quantization (PTQ) a common acceleration strategy.

**Limitations of Prior Work**: Existing LVLM quantization methods (e.g., MBQ) measure token sensitivity only at the modality level (visual vs. text), overlooking complex cross-token interactions and inter-token differences in quantization sensitivity.

**Key Challenge**: As tokens interact across layers, modality boundaries gradually dissolve, and quantization sensitivity varies substantially even among tokens within the same modality—a phenomenon manifested in four observed aspects: massive activations, layer heterogeneity, sub-layer divergence, and token variability.

**Goal**: To accurately estimate quantization sensitivity at the token level and leverage this information to guide finer-grained channel-wise equalization.

**Key Insight**: Drawing from axiomatic attribution in mechanistic interpretability, integrated gradients are employed to quantify each token's sensitivity along the path from a quantized reference to the actual input.

**Core Idea**: Replace modality-level sensitivity estimation with Quantization-aware Integrated Gradients (QIG) to guide quantization calibration at the token level.

## Method

### Overall Architecture
The multimodal input sequence (visual + text + special tokens) is processed during calibration to compute per-token QIG scores → IQR clipping and normalization → token importance coefficients $\lambda_i$ are incorporated into the channel-wise equalization objective → optimal quantization scaling factors are searched.

### Key Designs

1. **Quantization-aware Integrated Gradients (QIG)**:

    - **Function**: Quantifies each token's contribution to quantization error at the token level.
    - **Mechanism**: Unlike classical IG, which attributes full-precision predictions, QIG attributes the output discrepancy between the full-precision and quantized models. Gradients are integrated along the path from $x^q$ (quantized input) to $x$ (actual input): $QIG(x) = (x - x^q) \int_0^1 \frac{\partial(f(x_\alpha, w) - f(x_\alpha, w^q))}{\partial x_\alpha} d\alpha$
    - **Design Motivation**: Commonly used proxies such as gradients and attention exhibit weak correlation with quantization error, while perturbation-based methods, though accurate, are computationally expensive. QIG is directly linked to PTQ error and satisfies the completeness axiom.

2. **IQR Clipping for Stabilization**:

    - **Function**: Suppresses extreme values in QIG scores.
    - **Mechanism**: Interquartile range clipping is applied as $C(QIG_i) = \text{clip}(QIG_i, Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR)$, followed by normalization to obtain $\lambda_i$.
    - **Design Motivation**: Raw QIG distributions are heavy-tailed, and a small number of extreme tokens can dominate the optimization.

3. **Token-Level Weighted Channel-Wise Equalization**:

    - **Function**: Integrates token importance coefficients $\lambda_i$ into the CWE optimization objective.
    - **Mechanism**: $\mathbf{E}^* = \arg\min_{\mathbf{E}} \sum_{i=1}^T \lambda_i \|Q_W(\mathbf{W}*\mathbf{E}) Q_X(\mathbf{E}^{-1}*\mathbf{X}_i) - \mathbf{W}\mathbf{X}_i\|_2^2$
    - **Design Motivation**: Biasing the scaling factor search toward more sensitive tokens yields higher accuracy while preserving the overall framework.

### Loss & Training
- Entirely training-free (PTQ); calibration uses only 128 image-text pairs from ShareGPT4V.
- Supports both weight-only (W3A16) and weight-activation (W4A8) quantization settings.

## Key Experimental Results

### Main Results (LLaVA-onevision-7B)

| Setting | Method | VizWiz | MMMU | ChartQA | AI2D | ScienceQA | Avg. |
|---------|--------|--------|------|---------|------|-----------|------|
| FP16 | - | 60.41 | 49.22 | 80.04 | 81.31 | 95.88 | 73.37 |
| W3A16 | MBQ | 57.99 | 44.00 | 76.84 | 78.47 | 94.89 | 70.44 |
| W3A16 | **QIG** | **62.82** | **45.78** | **77.20** | **79.11** | **95.29** | **72.04** |
| W4A8 | MBQ | 58.13 | 44.78 | 74.92 | 78.27 | 94.70 | 70.16 |
| W4A8 | **QIG** | **59.10** | **45.00** | **74.52** | **78.30** | **94.25** | **70.23** |

### Ablation Study

| Sensitivity Type | Granularity | VizWiz Accuracy |
|-----------------|-------------|----------------|
| Gradient (SFT loss) | Modality-level | 57.36 |
| Gradient | Token-level | 55.78 (↓) |
| Attention | Token-level + special | 57.52 |
| Perturbation | Token-level + special | 57.72 |
| **QIG** | **Token-level** | **Best** |

### Key Findings
- Under W3A16, QIG outperforms MBQ by an average of 1.60% on LLaVA-onevision-7B, with only a 1.33% gap from full precision.
- Using SFT gradients for token-level sensitivity performs worse than modality-level estimation, indicating that SFT gradients do not correlate well with quantization sensitivity.
- Attention scores yield unstable results due to the attention-sink phenomenon.
- Token-level sensitivity estimated by QIG exhibits strong correlation with actual quantization error.

## Highlights & Insights
- **Applying interpretability tools to an engineering problem**: The work elegantly transfers integrated gradients from "explaining model predictions" to "attributing quantization error," with axiomatic attribution providing theoretical guarantees for sensitivity estimation.
- **Zero additional inference overhead**: QIG is computed solely during calibration; post-quantization inference is identical to the baseline.
- The paper systematically refutes SFT gradients and attention scores—two intuitively appealing proxies—thereby strengthening the case for QIG.

## Limitations & Future Work
- The calibration set is fixed at 128 samples; the impact of calibration set selection on QIG remains unexplored.
- The number of integration steps in QIG is a hyperparameter whose sensitivity is not thoroughly discussed.
- Validation is limited to the 7B–26B scale; effectiveness on larger models (70B+) is unknown.
- The 1.5× IQR clipping threshold is a classical statistical default; whether it is optimal for quantization scenarios warrants further investigation.

## Related Work & Insights
- **vs. MBQ**: MBQ applies modality-level gradient weighting, whereas QIG operates at the token level with quantization-aware integrated gradients, offering finer granularity and a direct link to quantization error.
- **vs. AWQ/GPTQ**: These methods do not account for multimodal structure; QIG is specifically designed for the heterogeneous token sequences in LVLMs.
- The token-level sensitivity analysis framework is transferable to LVLM pruning and knowledge distillation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Cross-domain transfer from interpretability to quantization is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple models, benchmarks, and settings; systematic ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation analysis and visualizations are well executed.
- **Value**: ⭐⭐⭐⭐ A plug-and-play PTQ improvement with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)

</div>

<!-- RELATED:END -->
