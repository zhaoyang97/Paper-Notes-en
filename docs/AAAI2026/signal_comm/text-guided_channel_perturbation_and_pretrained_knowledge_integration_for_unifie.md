---
title: >-
  [Paper Note] Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion
description: >-
  [AAAI2026][Signal & Communication][Multi-modality image fusion] This paper proposes UP-Fusion, a unified multi-modality image fusion framework comprising three modules — Semantic-aware Channel Pruning Module (SCPM)…
tags:
  - "AAAI2026"
  - "Signal & Communication"
  - "Multi-modality image fusion"
  - "unified model"
  - "channel perturbation"
  - "CLIP text guidance"
  - "pretrained knowledge"
date: 2026-05-08
content_hash: e0d2935b2ae466d5
---

# Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion

**Conference**: AAAI2026
**arXiv**: [2511.12432](https://arxiv.org/abs/2511.12432)
**Code**: To be confirmed
**Area**: Image Fusion / Multi-Modality
**Keywords**: Multi-modality image fusion, unified model, channel perturbation, CLIP text guidance, pretrained knowledge

## TL;DR
This paper proposes UP-Fusion, a unified multi-modality image fusion framework comprising three modules — Semantic-aware Channel Pruning Module (SCPM), Geometric Affine Modulation (GAM), and CLIP Text-guided Channel Perturbation Module (TCPM) — that employs a single set of weights (trained solely on infrared-visible data) to simultaneously handle both IVIF and medical image fusion tasks, achieving state-of-the-art performance on both.

## Background & Motivation

**Background**: Multi-modality image fusion encompasses two major categories: infrared-visible image fusion (IVIF) and medical image fusion (MEIF). Existing methods typically design task-specific models for each category, lacking a unified framework.

**Limitations of Prior Work**: (1) Task-specific models fail to generalize across modalities — IVIF models perform poorly on medical fusion and vice versa; (2) unified methods either sacrifice fusion quality or require multi-task training data; (3) direct injection of modality features tends to cause modality overfitting.

**Key Challenge**: How can a single model with a single set of weights simultaneously handle multiple modality combinations while maintaining quality across different fusion tasks?

**Goal**: To build a unified framework that, trained on only a single task (IVIF), generalizes to other modality fusion tasks.

**Key Insight**: Reducing modality dependence through channel perturbation (rather than direct feature injection), and leveraging pretrained knowledge (ConvNeXt + CLIP) to provide cross-task generalization capability.

**Core Idea**: Channel perturbation combined with pretrained knowledge enables "modality-agnostic" unified fusion.

## Method

### Overall Architecture
A Transformer encoder-decoder architecture (4 encoding / 4 decoding layers). After the encoder extracts multi-modality features, SCPM performs semantic-aware channel pruning, GAM applies geometric affine modulation, and TCPM conducts text-guided channel perturbation. Training is performed solely on LLVIP (infrared-visible).

### Key Designs

1. **Semantic-aware Channel Pruning Module (SCPM)**:

    - **Function**: Guides channel selection using pretrained semantic knowledge.
    - **Mechanism**: An SE-block computes channel importance $\omega_C$; a pretrained ConvNeXt extracts semantic features mapped to $\omega_S$; the fused weight is $\omega_F = \omega_C + \alpha \cdot \sigma(\omega_S)$ (with learnable $\alpha$). Top-k selection retains 70% of channels, with a 1×1 convolution expanding back to the original dimension.
    - **Design Motivation**: ConvNeXt's semantic priors facilitate correct channel selection across different modalities, serving as a key enabler of cross-task generalization.

2. **Geometric Affine Modulation Module (GAM)**:

    - **Function**: Adapts fused features to the geometric characteristics of each modality.
    - **Mechanism**: Global average pooling is applied to raw modality features; two 1×1 convolution layers generate scale $\gamma$ and shift $\beta$; the affine transform is applied as $F_O^M = Fuse^M \cdot (1 + \gamma) + \beta$.
    - **Design Motivation**: Affine transformation rather than direct feature injection is employed to avoid modality overfitting.

3. **Text-guided Channel Perturbation Module (TCPM)**:

    - **Function**: Guides channel rearrangement using CLIP text features.
    - **Mechanism**: Multi-modality features are concatenated → channel attention selects top 50% → a 1×1 convolution expands to 2× channels → CLIP encodes text → linear mapping → bootstrap weights → channel rearrangement → self-attention (perturbed features as Q, original as K/V).
    - **Design Motivation**: Channel perturbation is less prone to overfitting specific modalities than direct conditioning.

### Loss & Training
$L_T = L_{grad} + L_{l_1}$. Training is conducted solely on LLVIP for 100 epochs with resolution 192×192, Adam optimizer, and LR decayed from 0.0001 to 0.00001 via cosine annealing.

## Key Experimental Results

### Main Results (Infrared-Visible Fusion)

| Method | MSRS $Q_P$ | MSRS VIF | LLVIP VIF | M3FD VIF |
|------|-----------|---------|----------|---------|
| SAGE | 0.5210 | 0.4359 | 0.3590 | 0.4110 |
| TDFusion | 0.5529 | 0.4257 | 0.3577 | 0.4041 |
| **UP-Fusion** | **0.5671** | **0.4587** | **0.3817** | **0.4582** |

### Medical Fusion (Surpassing Task-Specific Methods)

| Method | Harvard $Q_P$ | Harvard VIF |
|------|-------------|------------|
| ALMFnet (dedicated) | 0.5434 | 0.3003 |
| **UP-Fusion** | **0.5665** | **0.3190** |

### Ablation Study

| Variant | $Q_P$ | VIF | SSIM |
|------|------|-----|------|
| w/o SCPM | 0.5343 | 0.3046 | 0.2645 |
| w/o TCPM | 0.5221 | 0.3016 | 0.2824 |
| **UP-Fusion** | **0.5665** | **0.3190** | **0.3639** |

### Key Findings
- Training on IVIF alone is sufficient to surpass dedicated MEIF methods.
- TCPM is the most critical module — removing it causes a $Q_P$ drop of 0.044.
- Downstream tasks also achieve top performance: segmentation mIoU 78.28, detection mAP@0.5 0.841.

## Highlights & Insights
- **"Train on one task, generalize to many"** is the primary contribution: channel perturbation combined with pretrained knowledge achieves genuine modality-agnosticism.
- **Channel perturbation as a substitute for direct conditioning**: rather than directly injecting modality features, the method indirectly influences representations through channel rearrangement, elegantly avoiding overfitting.

## Limitations & Future Work
- CLIP text guidance requires a text description for each fusion task, limiting the degree of automation.
- Validation is restricted to infrared-visible and medical fusion; performance on remote sensing fusion tasks such as SAR or multispectral imagery remains unknown.
- The 70% retention rate in SCPM and the 50% rate in TCPM are fixed hyperparameters.

## Related Work & Insights
- **vs. EMMA**: EMMA achieves unified fusion but requires multi-task training data, whereas UP-Fusion requires only single-task training.
- **vs. TDFusion**: TDFusion employs text-driven fusion but remains modality-specific; UP-Fusion's channel perturbation is more modality-agnostic.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of channel perturbation, CLIP text guidance, and pretrained knowledge enables multi-task generalization from single-task training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ IVIF (3 datasets) + MEIF (2 datasets) + downstream tasks + detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Module design is clearly presented.
- **Value**: ⭐⭐⭐⭐ The unified fusion framework has direct practical value for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](../../NeurIPS2025/signal_comm/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/signal_comm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[NeurIPS 2025\] Perturbation Bounds for Low-Rank Inverse Approximations under Noise](../../NeurIPS2025/signal_comm/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[NeurIPS 2025\] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](../../NeurIPS2025/signal_comm/memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)
- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](../../ICLR2026/signal_comm/multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)

</div>

<!-- RELATED:END -->
