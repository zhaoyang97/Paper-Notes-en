---
title: >-
  [Paper Note] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Medical image segmentation] This paper proposes Deco-Mamba, a decoder-centric Transformer-CNN-Mamba hybrid architecture that enhances the decoding process via Co-Attention Gates…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Medical image segmentation"
  - "Mamba"
  - "decoder design"
  - "deep supervision"
  - "KL divergence"
date: 2026-05-08
content_hash: 1195795f040b5922
---

# Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.12547](https://arxiv.org/abs/2603.12547)  
**Code**: To be released (upon acceptance)  
**Area**: Medical Imaging
**Keywords**: Medical image segmentation, Mamba, decoder design, deep supervision, KL divergence

## TL;DR

This paper proposes Deco-Mamba, a decoder-centric Transformer-CNN-Mamba hybrid architecture that enhances the decoding process via Co-Attention Gates, Vision State Space Modules (VSSMs), and deformable convolutions, while introducing a distribution-aware deep supervision strategy based on windowed KL divergence. The method achieves state-of-the-art performance across 7 medical image segmentation benchmarks.

## Background & Motivation

A common limitation of existing medical image segmentation methods (U-Net, TransUNet, Mamba-UNet, etc.) is their **overemphasis on encoder design at the expense of the decoder**:

- CNN encoders (U-Net family): limited long-range dependency modeling due to local receptive fields.
- Transformer encoders (TransUNet, Swin-UNet): $O(n^2)$ self-attention complexity, not scalable to high resolutions.
- Mamba encoders (U-Mamba, Swin-UMamba): linear complexity, but most methods introduce Mamba only in the encoder while keeping the decoder simple.

**Key Challenge**: Powerful encoders extract rich semantic representations, but an inadequate decoder fails to accurately recover object boundaries and contextual structures during upsampling. Existing methods either suffer from parameter explosion due to cascaded decoders (e.g., Cascaded-MERIT at 148M parameters) or lose fine details due to overly lightweight decoders.

A further issue is that conventional deep supervision resizes intermediate-layer predictions to full resolution before computing loss against ground truth, which inherently degrades structural information.

**Key Insight**: Deco-Mamba addresses this by (1) introducing Mamba into the decoder rather than the encoder, and (2) designing distribution-aware deep supervision that computes KL divergence directly at the native resolution of each decoding stage.

## Method

### Overall Architecture

A U-Net-style structure is adopted: the encoder comprises a CNN branch (7×7 convolutions) and a PVT Transformer (4 stages); the decoder consists of 6 stages, each containing Co-Attention Gate → VSSMB → Deformable Residual Block.

### Key Designs

1. **Co-Attention Gate (CAG)**: Conventional attention gates use only decoder features as gating signals to highlight encoder features. CAG allows encoder and decoder features to **serve as mutual gating signals**, producing two attention outputs that are concatenated and refined via channel attention (CA): $D_i' = CA[AG(x=X_i, g=D_{i+1}), AG(x=D_{i+1}, g=X_i)]$. **Design Motivation**: Decoder features also require spatial saliency filtering, and inter-channel relationships should be modeled. Ablations show CAG outperforms AG, LGAG, and CBAM.

2. **Vision State Space Mamba Block (VSSMB)**: SSM (Mamba) is introduced into the decoder, propagating contextual information along horizontal, vertical, and reversed directions via selective scanning, modeling long-range dependencies with linear complexity. Two VSSMBs are placed at the bottleneck layer, one at each intermediate layer, and none at the final layer (where convolution is more suitable at full resolution). **Design Motivation**: The decoder must maintain global semantic consistency across progressive upsampling stages; SSMs are more resource-efficient than self-attention for this purpose.

3. **Deformable Residual Block (DRB)**: A DRB is placed after each VSSMB, consisting of a standard 3×3 convolution and a deformable convolution. The deformable convolution predicts per-pixel offsets and modulation masks, enabling spatially adaptive sampling under geometric variation. **Design Motivation**: While VSSMBs excel at global context modeling, they may smooth local details; DRB recovers boundary precision through spatial adaptability.

4. **Multi-Scale Distribution-Aware (MSDA) Deep Supervision**: Conventional deep supervision resizes intermediate predictions to GT resolution and computes Dice/CE loss, with the resize operation discarding structural information. The MSDA approach maps each decoder-stage output (at its native resolution) to the number of classes via a distribution head, and constructs a same-resolution class distribution $\tilde{P}^{(s)}$ from GT via local window averaging. KL divergence is then computed as: $\mathcal{L}_{\text{KL}}^{(s)} = \sum_{b,h,w}\sum_c \tilde{P}_{b,c,h,w}^{(s)} \log\frac{\tilde{P}_{b,c,h,w}^{(s)}}{Q_{b,c,h,w}^{(s)}}$. A boundary weighting term $W_{h,w}^{(s)} = (1 - \max_n \tilde{P}_{h,w,n}^{(s)})^\alpha$ is also introduced to emphasize class boundaries.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dice}} + \sum_{s=1}^S \lambda_s \mathcal{L}_{\text{dist}}^{(s)}$$

The Dice loss ensures spatial overlap of the final prediction; the MSDA KL divergence loss provides distribution consistency supervision at each decoding stage. Training uses AdamW with cosine learning rate scheduling, 224×224 input resolution, on an A5000 GPU.

## Key Experimental Results

### Main Results

**Synapse (8-class abdominal multi-organ CT)**

| Method | DSC↑ | HD95↓ | Params (M) | FLOPs (G) |
|--------|------|-------|------------|-----------|
| Cascaded-MERIT | 83.59 | 15.99 | 147.86 | 33.31 |
| PAG-TransYnet | 83.43 | 15.82 | 144.22 | 33.65 |
| **Deco-Mamba-V1** | **85.07** | **14.72** | 46.93 | 17.24 |
| Deco-Mamba-V0 | 83.16 | 15.89 | **9.67** | **9.73** |

**Cross-Dataset Generalization (7 Benchmarks)**

| Dataset | Deco-Mamba-V1 | Prev. SOTA | Gain |
|---------|---------------|------------|------|
| Synapse | **85.07** | 83.59 (Cascaded-MERIT) | +1.48 |
| BTCV (13-class) | **78.45** | 75.87 (PAG-TransYnet) | +2.58 |
| ACDC | **92.35** | 92.12 (PVT-EMCAD-B2) | +0.23 |
| ISIC17 | **86.01** | 85.67 (Cascaded-MERIT) | +0.34 |
| GlaS | **96.91** | 96.91 (Cascaded-MERIT) | Tied |
| MoNuSeg | **85.14** | 83.41 (Deco-Mamba-V0) | +1.73 |

### Ablation Study

| Configuration | DSC↑ | HD95↓ | Note |
|---------------|------|-------|------|
| w/o CNN encoder branch | 84.07 | 18.92 | Loss of high-resolution spatial detail |
| w/o VSSMB | 83.51 | 15.96 | Missing long-range dependency modeling |
| AG replacing CAG | 82.98 | 15.69 | Unidirectional attention insufficient |
| Standard conv replacing deformable conv | 84.53 | 16.18 | Reduced boundary adaptability |
| Dice only (no MSDA) | 83.84 | 14.94 | Absence of multi-scale distribution constraints |
| Dice + conventional deep supervision | 84.24 | 15.89 | Resize increases HD95 |
| **Deco-Mamba (full)** | **85.07** | **14.72** | — |

### Key Findings

- The decoder-centric design is genuinely effective: using the same PVT-B0 backbone, Deco-Mamba surpasses Swin-UNet by 5.58% DSC.
- Deco-Mamba-V0 (9.67M parameters) outperforms most methods exceeding 100M parameters, validating the argument that "the decoder matters more than the encoder."
- MSDA deep supervision outperforms conventional deep supervision and boundary loss by avoiding the information loss caused by resizing.

## Highlights & Insights

- The **decoder-centric** design philosophy warrants attention: rather than pursuing larger pretrained encoders, the focus is placed on careful decoder engineering.
- The windowed KL divergence in MSDA is an elegant solution: instead of resizing GT, local window statistics are computed over GT to match low-resolution predictions.
- Mamba proves more effective in the decoder than in the encoder, as the decoder must maintain global consistency throughout the upsampling process.

## Limitations & Future Work

- The method supports only 2D segmentation; extension to 3D medical volumes (e.g., CT/MRI volumetric data) is not explored.
- Although 7 datasets are used, all are established benchmarks; validation on newer or more challenging datasets is absent.
- Sensitivity analysis of window size and $\lambda_s$ selection on MSDA performance is not thoroughly conducted.
- Code has not yet been released.

## Related Work & Insights

- **vs. EMCAD (EMCAD-B2)**: EMCAD also emphasizes decoder design but employs lightweight convolutional blocks with conventional deep supervision; Deco-Mamba advances further with Mamba and distribution-aware supervision.
- **vs. Swin-UMamba**: The latter introduces Mamba into the encoder, whereas this work targets the decoder — the two perspectives are complementary and could potentially be combined.
- The windowed distribution concept underlying MSDA is generalizable to deep supervision in other dense prediction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of decoder-side Mamba and distribution-aware deep supervision is well-motivated and coherent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 7 datasets, complete ablations, backbone comparisons, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with thorough module explanations.
- **Value**: ⭐⭐⭐⭐ The decoder-centric perspective offers meaningful insights to the community; MSDA is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)

</div>

<!-- RELATED:END -->
