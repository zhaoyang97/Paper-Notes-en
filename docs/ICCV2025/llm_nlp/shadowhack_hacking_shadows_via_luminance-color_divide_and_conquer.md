---
title: >-
  [Paper Note] ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer
description: >-
  [ICCV 2025][LLM/NLP][shadow removal] This paper proposes the ShadowHack framework, which decomposes shadow removal into two subtasks—luminance restoration and color reconstruction. LRNet with Rectified Outreach Attention…
tags:
  - "ICCV 2025"
  - "LLM/NLP"
  - "shadow removal"
  - "luminance-color decomposition"
  - "Transformer"
  - "attention mechanism"
  - "divide-and-conquer strategy"
date: 2026-05-08
content_hash: eeab4e9ffed2738b
---

# ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer

**Conference**: ICCV 2025
**arXiv**: [2412.02545](https://arxiv.org/abs/2412.02545)  
**Code**: Coming soon  
**Area**: LLM/NLP
**Keywords**: shadow removal, luminance-color decomposition, Transformer, attention mechanism, divide-and-conquer strategy

## TL;DR

This paper proposes the ShadowHack framework, which decomposes shadow removal into two subtasks—luminance restoration and color reconstruction. LRNet with Rectified Outreach Attention (ROA) recovers luminance and texture, followed by CRNet with cross-attention to reconstruct accurate color. The method achieves state-of-the-art performance on the ISTD+ and SRD datasets.

## Background & Motivation

Shadow removal involves three complex and intertwined degradations:

**Luminance reduction**: Shadow regions are occluded from direct illumination and receive only ambient light.

**Texture degradation**: Sensor noise, quantization errors, and compression artifacts during imaging are particularly pronounced in dark regions.

**Color distortion**: Surface material properties and ambient color cast introduce chromatic bias.

Limitations of existing methods:
- End-to-end methods struggle to handle all three degradations simultaneously.
- Retinex decomposition assumes uniform illumination, but the reflectance map in shadow regions exhibits significant color bias (verified via the CbCr channel distribution in Fig. 3).
- Diffusion models yield strong results but incur excessive computational cost.

Key insight: **Rather than separating illumination from color+texture (Retinex), the paper separates color from luminance+texture.** Converting the RGB image to YCbCr space, the Y channel captures luminance and texture while the CbCr channels capture color information. Since the two degradation types differ, they can be addressed independently.

## Method

### Overall Architecture

The ShadowHack pipeline is formulated as:

$$\hat{I} = \mathcal{D}^{-1}\mathcal{C}(\mathcal{R}(I_t), I_c)$$

$(I_t, I_c) = \mathcal{D}(I)$ decomposes the RGB image into luminance $I_t$ (Y channel) and color $I_c$ (CbCr channels). LRNet $\mathcal{R}$ restores luminance and texture; CRNet $\mathcal{C}$ reconstructs color guided by the restored luminance; finally, $\mathcal{D}^{-1}$ maps the result back to RGB.

### Key Designs

1. **Luminance Restoration Network (LRNet)**: A four-level encoder-decoder U-Net architecture (L1–L4, with 32 channels at the first level). Shallow layers (L1–L2) use Local Range Blocks (LRB) + Multi-head Transposed Attention (MTA) + FFN to extract local texture. Deep layers (L3–L4) replace LRB with the Rectified Outreach Attention (ROA) module to capture longer-range information from shadow-free regions as restoration references. **Design Motivation**: Restoring shadow regions requires referencing texture and luminance from neighboring shadow-free areas, which fixed-window attention cannot cover sufficiently.

2. **Rectified Outreach Attention (ROA) Module**: Two core innovations:

    - **Outreach Window**: Q is derived from standard window partitions, while K and V come from expanded outreach window partitions (overlap ratio 0.5), with dilation further enlarging the receptive field.
    - **Differential Rectification**: Since the luminance channel lacks rich color information, the color component $F_c$ is introduced to assist attention computation. Two attention maps are constructed: $\text{Att}_1 = \text{Softmax}(Q_1K_1/\sqrt{d}+B)$ (based on $[F_t;F_c]$) and $\text{Att}_2 = \text{Softmax}(Q_2K_2/\sqrt{d}+B)$ (based on $F_c$ only). The rectified attention is $\text{ROA} = (\text{Att}_1 - \lambda \cdot \text{Att}_2)V$, where $\lambda$ is reparameterized via a learnable scalar. The differential operation suppresses color bias interference in the attention map, creates negative correlation in shadow regions, and strengthens association with well-lit reference regions.

3. **Color Reconstruction Network (CRNet)**: A dual-encoder architecture that integrates a multi-scale color feature extractor (ConvNext-v2 atto, pretrained on ImageNet-21k, only 2M parameters) into LRNet's U-Net backbone. Restored luminance features serve as Q and K to compute similarity, while color features serve as V to aggregate color information; cross-attention injects color features into skip connections. **Design Rationale**: Regions with similar luminance should exhibit similar color; the restored luminance serves as an index to match correct color references.

4. **Checkpoint Ensemble**: During CRNet training, outputs from multiple checkpoints sampled from the early training stages of LRNet are randomly selected as inputs, enhancing CRNet's robustness to imperfect luminance restoration. This is used only during training and incurs no additional overhead at inference.

### Loss & Training

- L1 loss + VGG perceptual loss
- AdamW optimizer, momentum $(0.9, 0.999)$, weight decay $10^{-2}$
- Initial learning rate $2 \times 10^{-4}$, cosine annealing to $10^{-6}$
- Training crop size 384×384; data augmentation includes rotation, flipping, mixup, and color jitter
- Trained on RTX 4090 GPU

## Key Experimental Results

### Main Results

Comparison on ISTD+ (full-image metrics):

| Method | Conference | S-PSNR↑ | S-RMSE↓ | NS-PSNR↑ | NS-RMSE↓ | ALL-PSNR↑ | ALL-RMSE↓ |
|--------|------------|---------|---------|---------|---------|-----------|-----------|
| ShadowFormer | AAAI'23 | 39.67 | 5.21 | 38.82 | 2.30 | 35.46 | 2.80 |
| ShadowDiffusion | CVPR'23 | 39.82 | 4.90 | 38.90 | 2.30 | 35.72 | 2.70 |
| Homoformer | CVPR'24 | 39.47 | 4.72 | 38.73 | 2.23 | 35.34 | 2.64 |
| RASM | MM'24 | 40.73 | 4.41 | 39.23 | 2.17 | 36.16 | 2.53 |
| **ShadowHack** | **ICCV'25** | 40.56 | 4.46 | **39.66** | **2.09** | **36.31** | **2.48** |

Comparison on SRD:

| Method | ALL-PSNR↑ | ALL-RMSE↓ |
|--------|-----------|-----------|
| ShadowDiffusion | 34.73 | 3.63 |
| Homoformer | 35.37 | 3.33 |
| RASM | 34.46 | 3.37 |
| **ShadowHack** | **35.94** | **2.90** |

On SRD, RMSE improves by 0.43 (12.9%) and PSNR increases by 0.57 dB, demonstrating a clear advantage.

### Ablation Study

Decomposition strategy ablation (ISTD+):

| Strategy | ALL-PSNR↑ | ALL-RMSE↓ |
|----------|-----------|-----------|
| RGB end-to-end | 36.16 | 2.54 |
| Retinex decomposition | 35.80 | 2.63 |
| **Luminance-color decomposition (Ours)** | **36.31** | **2.46** |

ROA module ablation (SRD, luminance-space PSNR):

| Configuration | Shadow | Non-Shadow | ALL |
|---------------|--------|-----------|-----|
| w/o outreach | 39.17 | 40.77 | 35.93 |
| w/o dilation | 39.74 | 41.09 | 36.42 |
| w/o rectify | 39.96 | 41.27 | 36.60 |
| $[F_t;F_c] \& [F_c]$ (Ours) | **40.36** | **41.40** | **36.90** |

CRNet ablation (SRD):

| Configuration | ALL-RMSE↓ |
|---------------|-----------|
| w/o cross-attention | 3.39 |
| w/o checkpoint ensemble | 3.28 |
| **Full CRNet** | **2.90** |

### Key Findings

- Luminance-color decomposition outperforms Retinex decomposition, as the latter violates its own assumptions due to color bias in shadow-region reflectance maps.
- The outreach window is critical for shadow regions to reference shadow-free regions (removing it causes a 1.19 dB drop in shadow-region PSNR).
- Differential rectification effectively suppresses color bias interference on attention maps; visualizations confirm that rectified attention maps are more precise.
- Checkpoint ensemble substantially improves CRNet generalization (RMSE reduced from 3.28 to 2.90).
- The model supports user-specified shadow masks and is robust to imprecise masks.

## Highlights & Insights

- **Novel decomposition perspective**: Unlike conventional Retinex decomposition, this work separates color rather than illumination, grounded in a thorough physical analysis of shadow degradation characteristics.
- **Differential attention in ROA**: Rectifying attention by subtracting the color-channel attention map is an elegant design with negligible computational overhead.
- **Justified sequential processing**: Restoring luminance first provides structural references, after which luminance similarity guides color matching—an intuitively sound pipeline.
- **Lightweight color encoder**: ConvNext-v2 atto has only 2M parameters yet supplies rich color priors.

## Limitations & Future Work

- Shadow masks are required as input (a mask refinement network is proposed but adds complexity).
- The total parameter count of 23.3M, while not large, is still higher than RASM's 5.2M.
- The two-stage training procedure (LRNet first, then CRNet) increases training complexity.
- Performance under extreme shadow conditions (e.g., very deep shadows or soft shadows) is not thoroughly evaluated.
- Whether YCbCr is the optimal color space for decomposition is not fully discussed (e.g., alternatives such as HSV).

## Related Work & Insights

- **vs. ShadowFormer**: ShadowFormer performs end-to-end shadow removal via shadow/non-shadow region interaction, whereas ShadowHack explicitly decomposes the task.
- The differential attention concept is adapted from Diff Transformer (Ye et al.) and proves highly effective in the shadow removal setting.
- The cross-attention design in CRNet draws inspiration from exemplar-based colorization methods.
- The divide-and-conquer strategy may inspire other complex image restoration tasks, such as underwater image enhancement and dehazing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel luminance-color decomposition perspective; elegant ROA design
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-dataset comparisons, detailed ablations, and robustness analysis
- **Writing Quality**: ⭐⭐⭐⭐ In-depth physical analysis with clearly articulated motivation
- **Value**: ⭐⭐⭐⭐ Provides a new design paradigm and strong SOTA baseline for shadow removal

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](../../NeurIPS2025/llm_nlp/spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[ICCV 2025\] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](any-ssr_how_recursive_least_squares_works_in_continual_learning_of_large_languag.md)
- [\[ICCV 2025\] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)
- [\[ICCV 2025\] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)
- [\[ICCV 2025\] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)

</div>

<!-- RELATED:END -->
