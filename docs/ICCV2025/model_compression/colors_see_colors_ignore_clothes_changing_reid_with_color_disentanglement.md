---
title: >-
  [Paper Note] Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement
description: >-
  [ICCV 2025][Model Compression][Clothes-Changing ReID] This paper proposes CSCI, a method that introduces a Color token to learn color representations (Color See) and employs a novel S2A self-attention mechanism to disentangle color information from ReID features (Color Ignore), effectively eliminating appearance bias in clothes-changing person re-identification without requiring any external annotations.
tags:
  - ICCV 2025
  - Model Compression
  - Clothes-Changing ReID
  - Color Disentanglement
  - Self-Attention Mechanism
  - Vision Transformer
  - Appearance Bias Elimination
date: 2026-05-08
content_hash: 9cd6d0f8259e44fc
---

# Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement

**Conference**: ICCV 2025
**arXiv**: [2507.07230](https://arxiv.org/abs/2507.07230)
**Code**: [https://github.com/ppriyank/ICCV-CSCI-Person-ReID](https://github.com/ppriyank/ICCV-CSCI-Person-ReID)
**Area**: Person Re-Identification / Model Compression
**Keywords**: Clothes-Changing ReID, Color Disentanglement, Self-Attention Mechanism, Vision Transformer, Appearance Bias Elimination

## TL;DR

This paper proposes CSCI, a method that introduces a Color token to learn color representations (Color See) and employs a novel S2A self-attention mechanism to disentangle color information from ReID features (Color Ignore), effectively eliminating appearance bias in clothes-changing person re-identification without requiring any external annotations.

## Background & Motivation

Clothes-Changing Person Re-Identification (CC-ReID) requires models to correctly identify individuals after they have changed their clothing, posing a significant practical challenge. Existing methods suffer from the following limitations:

**Reliance on external modalities**: Biometric features such as gait, face, and body shape are effective but incur high preprocessing costs, limiting real-world deployment.

**Dependency on clothing annotations**: RGB-only methods such as CAL rely on external clothing labels, increasing manual annotation burden.

**Unreliable fine-grained attributes**: Text-based attributes (e.g., "black trousers") are computationally expensive and sensitive to occlusion and illumination variations.

The core insight of this paper is that **color information** naturally offers three advantages — efficient extraction, adaptivity (dynamically varying with illumination and occlusion), and contextual relevance (capturing both clothing and background information). Color can therefore serve as a lightweight, annotation-free proxy signal for eliminating appearance bias in ReID.

## Method

### Overall Architecture

CSCI is built upon a Transformer backbone (EVA-02 ViT-L). It takes RGB images as input and produces a ReID feature $f_{ReID}$ and a color embedding $f_{CO}$; only $f_{ReID}$ is used during inference. The core idea is to jointly learn identity features and color representations within a unified feature space, then enforce orthogonality between the two via a disentanglement loss and a dedicated self-attention mechanism.

### Key Designs

1. **Color Token**:

    - Analogous to the class token in ViT, a learnable Color token is appended to the sequence of spatial patch tokens.
    - The Color token aggregates color information from the image through attention interactions with spatial tokens.
    - After passing through all Transformer blocks, the Color token yields the color embedding $f_{CO}$.
    - An MLP head maps $f_{CO}$ to a flattened color histogram vector via regression.
    - The parameter overhead is minimal (only 2.77%), fully exploiting the parameter efficiency of head tokens.

2. **S2A Self-Attention (Split-to-Attend Self-Attention)**:

    - Core objective: prevent information leakage between the Color token and the ReID token.
    - **Standard self-attention**: All tokens attend to each other, allowing the Color and ReID tokens to directly exchange information, causing color bias to contaminate ReID features.
    - **Masked self-attention**: Direct Color↔ReID interactions are blocked via $-\infty$ masking, but spatial tokens remain simultaneously exposed to both.
    - **S2A self-attention**: The self-attention is split into two independent steps — one involving the ReID token and spatial tokens (excluding Color), and another involving the Color token and spatial tokens (excluding ReID). The results for spatial tokens are averaged across the two steps:
    $Att(Q^{SP}) = \frac{1}{2}(Att(Q^{SP}_{\sim CO}) + Att(Q^{SP}_{\sim ID}))$
    - S2A strikes a favorable balance between full overlap (standard) and full separation (dual-branch).
    - The computational overhead increases by only 4.39% FLOPs (0.18% per block).

3. **Color Representation Methods**:

    - **Pixel Binning**: A 3D histogram is constructed over RGB channels with bin size=20, producing a $20 \times 20 \times 20 = 8000$-dimensional vector.
    - **RGB-uv Projection**: A 2D histogram projection is generated per channel (R/G/B) with bin size=32, producing a $3 \times 32 \times 32$ histogram.
    - Both approaches can be computed efficiently from RGB images at runtime without external dependencies.

### Loss & Training

The overall loss function is the sum of four terms:
$$L_{ReID} = \mathcal{L}_{CE}^{ID} + \mathcal{L}_{MSE}^{Color} + \mathcal{L}_{Triplet} + \mathcal{L}_{DE}$$

- $\mathcal{L}_{CE}^{ID}$: Identity cross-entropy loss for identity classification.
- $\mathcal{L}_{MSE}^{Color}$: MSE regression loss for color histogram prediction.
- $\mathcal{L}_{Triplet}$: Triplet loss applied to $f_{ReID}$.
- $\mathcal{L}_{DE}$: Disentanglement loss that maximizes the cosine distance between $f_{CO}$ and $f_{ReID}$ to induce orthogonality:
$$\mathcal{L}_{DE} = \left| \frac{f_{CO}}{\|f_{CO}\|_2} \cdot \frac{f_{ReID}}{\|f_{ReID}\|_2} \right|$$

The video variant CSCI-V employs EZ-CLIP for temporal extension and trains temporal tokens and Color tokens separately via an alternating training strategy.

## Key Experimental Results

### Main Results

| Dataset | Metric | CSCI (RGB-uv) | EVA-02 Baseline | Gain | Prev. SOTA |
|--------|------|---------------|-----------------|------|----------|
| LTCC (CC) | R-1 | **47.8** | 44.9 | +2.9% | 46.7 (IRM) |
| LTCC (CC) | mAP | **24.4** | 23.1 | +1.3% | 22.9 (CCPG) |
| PRCC (CC) | R-1 | **66.2** | 61.6 | +4.6% | 65.0 (FIRe2) |
| PRCC (CC) | mAP | **61.3** | 59.0 | +2.3% | 63.1 (FIRe2) |
| CCVID (CC) | R-1 | **90.8** | 89.8 | +1.0% | 86.9 (GBO) |
| MeVID (Overall) | R-1 | **79.1** | 76.6 | +2.5% | 59.5 (ShARc) |

### Ablation Study

| Configuration | LTCC R-1 | LTCC mAP | PRCC R-1 | PRCC mAP | Note |
|------|----------|----------|----------|----------|------|
| Baseline (no Color token) | 44.9 | 23.1 | 61.6 | 59.0 | Standard EVA-02 |
| Traditional Self-Attn | 46.8 | 23.5 | 63.5 | 60.3 | Risk of information leakage |
| Masked Self-Attn | 46.7 | 23.7 | 65.3 | 61.3 | Partial disentanglement |
| S2A Self-Attn (Ours) | **47.8** | **24.4** | **66.2** | **61.3** | Optimal disentanglement |
| Using clothing labels instead of color | 46.3 | 24.0 | - | - | Suboptimal |
| Grayscale augmentation | 38.3 | 18.5 | - | - | Severe distribution shift |

### Key Findings

- RGB-uv projection outperforms Pixel Binning for image ReID; both are comparable for video ReID.
- t-SNE clustering of color embeddings closely corresponds to clothing labels, confirming that color serves as an effective proxy for clothing annotations.
- Directly feeding color vectors ("Feed") is inferior to predicting color vectors, as the latter eliminates test-time dependency on color.
- S2A self-attention achieves parameter and computational efficiency close to standard self-attention while delivering significantly better performance.
- CSCI generalizes to different backbones including ViT-S (TMGF) and ViT-B (TransReID/PAT/TCiP).

## Highlights & Insights

- The reframing of color information from a "source of interference" to an "exploitable proxy signal" is conceptually elegant and intuitively sound.
- The S2A self-attention design achieves an elegant compromise between full sharing and full separation.
- The method is extremely lightweight, requiring only one additional learnable token and negligible computational overhead.
- It does not rely on any external annotations or models, substantially lowering the barrier to deployment.

## Limitations & Future Work

- The effectiveness of color disentanglement may be affected by extreme illumination conditions.
- In S2A self-attention, spatial tokens remain indirectly connected to both Color and ReID tokens, preventing complete elimination of information leakage.
- In the video variant, temporal tokens and Color tokens cannot be trained simultaneously, limiting joint optimization.
- The color histogram representation is relatively coarse; more fine-grained color representations warrant exploration.

## Related Work & Insights

- Compared to CAL (CVPR'22), CSCI achieves superior performance without requiring clothing annotations.
- The design principles underlying S2A self-attention can be generalized to other tasks requiring feature disentanglement, such as domain-adaptive ReID.
- The use of color as a proxy signal may inspire bias elimination in other visual recognition tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combined design of color proxy and S2A self-attention is novel, though each individual component offers limited innovation in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 4 datasets (image and video), multiple backbone architectures, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The comparative analysis of self-attention variants is clear and the mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ Parameter-efficient, annotation-free, and highly practical, with direct value for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)](color_matching_using_hypernetwork-based_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[CVPR 2026\] Understanding and Enforcing Weight Disentanglement in Task Arithmetic](../../CVPR2026/model_compression/understanding_and_enforcing_weight_disentanglement_in_task_arithmetic.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)

</div>

<!-- RELATED:END -->
