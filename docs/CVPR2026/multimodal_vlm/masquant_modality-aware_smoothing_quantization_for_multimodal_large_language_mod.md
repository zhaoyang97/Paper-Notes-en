---
title: >-
  [Paper Note] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Post-training quantization] This paper identifies a "smoothing misalignment" problem that arises when channel-wise smooth quantization methods (e.g.…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Post-training quantization"
  - "multimodal LLM"
  - "smooth quantization"
  - "cross-modal compensation"
  - "low-rank decomposition"
date: 2026-05-08
content_hash: ae8a895f6b1101cb
---

# MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.04800](https://arxiv.org/abs/2603.04800)
**Code**: [https://github.com/alibaba/EfficientAI](https://github.com/alibaba/EfficientAI)
**Area**: Multimodal VLM
**Keywords**: Post-training quantization, multimodal LLM, smooth quantization, cross-modal compensation, low-rank decomposition

## TL;DR
This paper identifies a "smoothing misalignment" problem that arises when channel-wise smooth quantization methods (e.g., SmoothQuant) are directly applied to MLLMs—the large discrepancy in activation magnitudes across modalities causes non-dominant modalities to be over-smoothed. MASQuant is proposed to address this via modality-aware smoothing factors and SVD whitening-based cross-modal low-rank compensation.

## Background & Motivation

**Background**: Post-training quantization (PTQ) is a critical technique for deploying large models. Channel-wise smoothing methods based on mathematical invariance (SmoothQuant, AWQ, etc.) perform well on text-only LLMs by redistributing activation outliers through channel scaling factors.

**Limitations of Prior Work**: When channel-wise smoothing is directly applied to MLLMs, visual token activations are typically 10–100× larger in magnitude than text token activations. A unified smoothing factor is dominated by the dominant modality (usually vision), causing non-dominant modalities (text, audio) to be over-smoothed, compressing their signals and introducing severe quantization errors—referred to as "Smoothing Misalignment."

**Key Challenge**: Learning independent smoothing factors per modality resolves the misalignment, but requires storing separate quantized weights for each modality, fundamentally defeating the purpose of quantization compression.

**Goal**: Can modality-aware smooth quantization be achieved while maintaining a single set of quantized weights?

**Key Insight**: The authors observe that weight differences across modalities after smoothing are low-rank (provably so mathematically), enabling lightweight low-rank matrices to serve as compensation.

**Core Idea**: Learn modality-specific smoothing factors + store a single set of quantized weights anchored to the text modality + apply SVD whitening-based low-rank compensation for other modalities.

## Method

### Overall Architecture
MASQuant consists of two core modules: (1) **Modality-Aware Smoothing (MAS)**, which learns independently optimized smoothing factors per modality; and (2) **Cross-Modal Compensation (CMC)**, which compresses cross-modal weight differences into a low-rank form via SVD whitening, storing only one set of quantized weights plus lightweight compensation matrices.

### Key Designs

1. **Modality-Aware Smoothing (MAS)**:

    - **Function**: Learn an independent optimized smoothing factor $\mathbf{S}_m$ for each modality $m$.
    - **Mechanism**: Initialize $s_i^m = \sqrt{\max_t|x_{t,i}^m| / \max_j|w_{j,i}|}$, then directly optimize the smoothing factors by minimizing a modality-specific MAE loss $\sum_{m} \lambda_m \cdot \mathcal{L}_{MAE}(\mathbf{S}_m, \mathbf{X}_m, \mathbf{W})$.
    - **SQNR Theoretical Analysis**: Proves that unified smoothing degrades SQNR for non-dominant modalities by $\Delta = 10\log_{10}(\frac{d(\min_i \alpha_i^2)}{\sum_i 1/\alpha_i^2})$, where $\alpha_i$ denotes the activation range ratio between modalities.
    - **Design Motivation**: Rather than searching for a hyperparameter $\beta$, the smoothing factors themselves are directly optimized, reaching the optimization limit of channel-wise smoothing.

2. **Cross-Modal Compensation (CMC)**:

    - **Function**: Compensate for quantization error in non-text modalities while using a single set of quantized weights.
    - **Mechanism**: Using the text-modality smoothed weight $Q(\mathbf{S}_t \mathbf{W})$ as the anchor, the visual modality produces a residual $\Delta\mathbf{W} = \mathbf{S}_v \mathbf{W} - Q(\mathbf{S}_t \mathbf{W})$. Directly applying SVD to $\Delta\mathbf{W}$ yields poor results due to lack of low-rank structure, but after a whitening transform $\mathbf{T} = (\mathbf{P}\Lambda^{1/2})^\top$, the transformed residual $\mathbf{T}(\Delta\mathbf{W})$ exhibits strong low-rank structure.
    - After truncated SVD: $\Delta\mathbf{W} \approx \mathbf{L}_1 \mathbf{L}_2$, where $\mathbf{L}_1 = \mathbf{T}^{-1}\mathbf{U}_r$ and $\mathbf{L}_2 = \Sigma_r \mathbf{V}_r^\top$.
    - **Theoretical Guarantee**: Proved that this scheme minimizes the output reconstruction error $\|\mathbf{X}_v \mathbf{S}_v^{-1}(\Delta\mathbf{W} - \mathbf{L})\|_F^2$.

3. **Inference Pipeline**:

    - Text modality: $\mathbf{Y} = Q(\mathbf{X}_t \mathbf{S}_t^{-1}) \cdot Q(\mathbf{S}_t \mathbf{W})$
    - Non-text modalities: $\mathbf{Y} = Q(\mathbf{X}_m \mathbf{S}_m^{-1}) \cdot Q(\mathbf{S}_t \mathbf{W}) + \mathbf{X}_m \mathbf{S}_m^{-1} \cdot \mathbf{L}_1^m \mathbf{L}_2^m$
    - Only lightweight low-rank matrices need to be stored additionally; the primary weights remain a single quantized copy.

## Key Experimental Results

### Main Results (Qwen2.5-VL Series)

| Method | Bits | MMMU | OCRBench | ScienceQA | TextVQA | Avg |
|--------|------|------|----------|-----------|---------|-----|
| FP16 | W16A16 | Baseline | Baseline | Baseline | Baseline | 100% |
| SmoothQuant | W8A8 | Notable drop | Drop | Drop | Drop | — |
| MASQuant | W8A8 | **Best** | **Best** | **Best** | **Best** | SOTA |

### Cross-Architecture Validation

| Model Type | Description |
|------------|-------------|
| Dual-modal VLM | Consistently outperforms SmoothQuant and AWQ on Qwen2.5-VL-3B/7B |
| Three-modal Omni | Equally effective on Qwen2.5-Omni-3B; audio modality also benefits |

### Ablation Study
- MAS alone already significantly improves SQNR (Figure 2 validates Theorem 1).
- The low-rank approximation quality of CMC converges rapidly as rank increases.
- The low-rank property of the whitened residual is substantially stronger than that of the direct SVD baseline.

## Highlights & Insights
- First formal definition of the "smoothing misalignment" problem in MLLM quantization, with a theoretical SQNR analysis (Theorem 1).
- Mathematical proof of the low-rank structure of cross-modal activation differences, providing theoretical guarantees for CMC (Theorem 2).
- The framework applies to both dual-modal (vision–text) and tri-modal (vision–text–audio) MLLMs.
- Maintains a single set of quantized weights with negligible additional storage overhead (low-rank matrices only).
- Consistently outperforms existing channel-wise smoothing PTQ methods on Qwen2.5-VL and Qwen2.5-Omni.

### Ablation Study
- MAS only (without CMC): Requires storing separate quantized weights per modality, but achieves optimal quantization accuracy.
- CMC only (without smoothing correction): Limited remediation, as the underlying smoothing misalignment remains unresolved.
- MAS + CMC (full method): Approaches the accuracy upper bound of MAS under the single-weight constraint.
- CMC low-rank compensation: Rank 16–32 is generally sufficient to recover 90%+ of the accuracy gap.
- Singular value decay of whitened $\mathbf{T}(\Delta\mathbf{W})$ is far faster than that of direct SVD, validating the low-rank assumption.

## Limitations & Future Work
- The calibration stage requires collecting per-modality data to separately optimize smoothing factors, increasing preprocessing complexity.
- The rank $r$ selection for low-rank compensation requires a trade-off between accuracy and additional storage.
- Only W8A8 and W4A8 settings are validated; performance under more aggressive low-bit configurations (e.g., W2A4) remains unknown.
- Inference on non-text modalities requires an additional matrix multiplication $\mathbf{X}_m \mathbf{S}_m^{-1} \cdot \mathbf{L}_1^m \mathbf{L}_2^m$, introducing marginal latency overhead.
- The modality-aware idea could potentially be extended to rotation-based methods (e.g., QuaRot, SpinQuant).
- In tri-modal and higher settings, the number of low-rank compensation matrices grows linearly, requiring memory management optimization.

### Implementation Details
- MAS optimization uses Adam and typically converges within 100–200 iterations.
- CMC low-rank matrices are stored in FP16; their memory footprint is negligible compared to the full weight matrices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[CVPR 2026\] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
