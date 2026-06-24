---
title: >-
  [Paper Note] SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer
description: >-
  [CVPR 2025][Image Generation][Semantic Style Transfer] A plug-and-play Semantic Continuous-Sparse Attention (SCSA) module is proposed to achieve semantic style transfer. It ensures style consistency within the same semantic region via Semantic Continuous Attention (SCA) and preserves original texture details via Semantic Sparse Attention (SSA). SCSA can be integrated into any attention-based style transfer method without requiring additional training.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Semantic Style Transfer"
  - "Attention Mechanism"
  - "Plug-and-Play"
  - "Continuous-Sparse Attention"
  - "Arbitrary Style Transfer"
date: 2026-05-08
content_hash: 32ec395423125c51
---

# SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer

**Conference**: CVPR 2025  
**arXiv**: [2503.04119](https://arxiv.org/abs/2503.04119)  
**Code**: [GitHub](https://github.com/scn-00/SCSA)  
**Area**: Image Generation/Style Transfer  
**Keywords**: Semantic Style Transfer, Attention Mechanism, Plug-and-Play, Continuous-Sparse Attention, Arbitrary Style Transfer

## TL;DR

A plug-and-play Semantic Continuous-Sparse Attention (SCSA) module is proposed to achieve semantic style transfer. It ensures style consistency within the same semantic region via Semantic Continuous Attention (SCA) and preserves original texture details via Semantic Sparse Attention (SSA). SCSA can be integrated into any attention-based style transfer method without requiring additional training.

## Background & Motivation

Attention-based arbitrary style transfer (Attn-AST) methods, including CNN-based (SANet), Transformer-based (StyTR2), and Diffusion-based (StyleID) approaches, can generate high-quality stylized images. However, they perform poorly when the content and style images share identical semantics:

- **Style Discontinuity**: Slight structural differences between adjacent locations within the same semantic region lead to fragmented stylization effects (e.g., abrupt color changes in the background).
- **Semantic Inconsistency**: The style of corresponding semantic regions in the stylized image fails to match those in the style image (e.g., incorrect clothing colors).
- **Texture Loss**: Weighted averaging of all style feature points blurs the original vivid textures.

The core challenge lies in the fact that existing Attn-AST methods compute relations between each query point and all key points when constructing the attention map, completely ignoring semantic region assignments. This simplification can lead to excessive focus on structurally similar points across different semantic regions.

## Method

### Overall Architecture

SCSA consists of two components: Semantic Continuous Attention (SCA) and Semantic Sparse Attention (SSA), initialized in conjunction with Semantic Adaptive Instance Normalization (S-AdaIN). SCSA directly replaces the Universal Attention (UA) in existing Attn-AST methods to achieve semantic style transfer without any training.

### Key Design 1: Semantic Continuous Attention (SCA)

- **Function**: Ensures all positions within the same semantic region receive a consistent overall style feature.
- **Mechanism**: Uses content/style semantic map features $F_{csem}, F_{ssem}$ as the query and key (instead of image features), and then sets attention weights between different semantic categories to $-\infty$ via an operation $G_1$. Since the semantic map features do not contain image structural information, all query points within the same semantic region share the same attention weights with the key, eliminating style fragmentation caused by structural differences.
- **Design Motivation**: Universal attention is based on image structure matching, where subtle structural variations lead to different stylization effects at adjacent locations. SCA achieves style continuity within semantic regions by ignoring structural details and focusing solely on semantics.

### Key Design 2: Semantic Sparse Attention (SSA)

- **Function**: Preserves the vivid texture details of the original style image.
- **Mechanism**: Uses image features as queries and keys, but utilizes an operation $G_2$ to retain only the weight of the single most similar key point within the same semantic region for each query point in the attention map (setting others to $-\infty$), resulting in a softmax weight of 1 for that point. The final style features are directly derived from a single best-matching style feature point.
- **Design Motivation**: Weighted averaging over multiple style points blurs the original textures. The most precise representation of style textures resides within a single encoded feature point, and sparse matching preserves this vividness.

### Key Design 3: Semantic Adaptive Instance Normalization (S-AdaIN)

- **Function**: Provides more accurate query features for SSA.
- **Mechanism**: Aligns the mean and variance of content features with the corresponding semantic regions of style features by performing AdaIN on each semantic region individually, neutralizing the interference from the original color style.
- **Design Motivation**: Original color information interferes with the accuracy of structure matching, and S-AdaIN provides "clean" structural features as better queries.

### Loss & Training

No extra training is required. SCSA replaces the UA module of existing methods in a plug-and-play manner. The final features are fused as $F_{cs} = \alpha_1 \times F_{sca} + \alpha_2 \times F_{ssa} + F_c$, where $\alpha_1$ and $\alpha_2$ control the transfer strength of the overall style and texture respectively.

## Key Experimental Results

### Main Results: Comparison of Semantic Style Transfer Quality

| Method | SSL ↓ | FID ↓ | CFSD ↓ | User Preference ↑ |
|------|-------|-------|--------|----------|
| SANet | 1.6583 | 14.34 | 0.1103 | 16.85% |
| SANet + **SCSA** | **0.8762** | **13.08** | **0.0874** | **83.15%** |
| StyTR2 | 1.9826 | 12.53 | 0.0752 | 15.76% |
| StyTR2 + **SCSA** | **1.2228** | **12.40** | **0.0705** | **84.24%** |
| StyleID | 1.7538 | 12.59 | 0.0916 | 21.92% |
| StyleID + **SCSA** | **1.2447** | **12.45** | **0.1178** | **78.08%** |

### Ablation Study

| Configuration | SSL ↓ |
|------|-------|
| SANet + SCSA (Full) | **0.8762** |
| - SSA | 0.8840 |
| - SCA | 0.9096 |
| - S-AdaIN | 0.8769 |

### Key Findings

- SCSA significantly improves semantic consistency across all three architectures (CNN, Transformer, and Diffusion), reducing the SSL metric by 29-47%.
- User preference rates surge from 15-22% to 78-84%, showing an overwhelming advantage.
- Compared with five SOTA dedicated semantic style transfer methods, Attn-AST methods empowered by SCSA still demonstrate superior performance.
- SCA contributes the most to semantic consistency, while SSA contributes the most to texture preservation, demonstrating complementary roles.

## Highlights & Insights

1. **Highly Valuable Plug-and-Play Design**: Can be integrated into existing CNN, Transformer, and Diffusion architectures without training, showing outstanding generalizability.
2. **Complementary Continuous + Sparse Design**: SCA delivers globally consistent styles, while SSA provides locally vivid textures, achieving a clear division of labor.
3. **Clear Problem Analysis**: The work thoroughly analyzes the three root causes behind the failure of existing Attn-AST methods in semantic style transfer.

## Limitations & Future Work

- Requires semantic segmentation maps as supplementary inputs; currently automatically generated using K-Means clustering, which may lack precision in complex scenes.
- The computational cost of grid searching in the $I$ branch scales linearly with the number of candidates.
- Sensitive to the number of semantic categories, where too many categories can lead to insufficient feature points per region.
- Temporal consistency in video style transfer scenarios has not been explored.

## Related Work & Insights

- **SANet**: The pioneering work introducing attention mechanisms to arbitrary style transfer.
- **StyTR2**: A representative model of Transformer-based style transfer.
- **StyleID**: Diffusion-based style transfer.
- **STROTSS**: Matches the feature distributions of semantic regions using Earth Mover's Distance (EMD).
- SCSA's core concept (semantic-aware attention constraints) can be extended to tasks demanding semantic consistency, such as image editing and video generation.

## Rating

⭐⭐⭐⭐ — The plug-and-play and training-free design is highly practical, and the consistent improvements across three architectures validate the generalizability of the method. The problem analysis is crystal clear, and the proposed solution is elegant. The requirement for semantic maps as input is the only minor limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[CVPR 2025\] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](samam_style-aware_state_space_model_for_arbitrary_image_style_transfer.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[CVPR 2026\] StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References](../../CVPR2026/image_generation/stylegallery_training-free_and_semantic-aware_personalized_style_transfer_from_a.md)
- [\[CVPR 2025\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)

</div>

<!-- RELATED:END -->
