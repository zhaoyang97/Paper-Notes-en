---
title: >-
  [Paper Note] Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Transformer] This paper identifies the massive activations phenomenon in Diffusion Transformers (DiTs) that renders features indiscriminable…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion Transformer"
  - "Massive Activations"
  - "AdaLN"
  - "Visual Correspondence"
  - "Feature Extraction"
date: 2026-05-08
content_hash: b07b082c03467e7c
---

# Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations

**Conference**: NeurIPS 2025
**arXiv**: [2505.18584](https://arxiv.org/abs/2505.18584)  
**Code**: [GitHub](https://github.com/ganchaofan0000/DiTF)  
**Area**: Visual Correspondence, Diffusion Models, Feature Extraction
**Keywords**: Diffusion Transformer, Massive Activations, AdaLN, Visual Correspondence, Feature Extraction

## TL;DR
This paper identifies the massive activations phenomenon in Diffusion Transformers (DiTs) that renders features indiscriminable, reveals its intrinsic connection to AdaLN, and proposes a training-free framework DiTF for extracting semantically discriminative features, surpassing DINO and SD models on visual correspondence tasks.

## Background & Motivation

### State of the Field

**Background**: Pretrained Stable Diffusion (SD) models have been demonstrated to serve as effective feature extractors for visual correspondence.

### Root Cause

**Key Challenge**: Diffusion Transformers (DiTs) outperform SD in scalability and generation quality, yet directly extracting their features for perceptual tasks yields poor performance.

### Limitations of Prior Work

**Limitations of Prior Work**: Analysis reveals that DiTs exhibit a massive activations phenomenon, where a small number of feature dimensions exhibit activation magnitudes more than 100× larger than the rest.

### Starting Point

**Key Insight**: These massive activations cause the feature vectors of all spatial tokens to become highly similar in direction, rendering cosine similarity unable to distinguish different spatial locations.

## Method

### Overall Architecture
- DiTF is a training-free framework that leverages the built-in AdaLN layers of DiTs to extract semantically discriminative features.
- Core pipeline: raw feature extraction → AdaLN channel modulation → channel dropping → final features.

### Key Designs
1. **Analysis of Massive Activations**:

    - Spatial distribution: appear across all image patch tokens (unlike LLMs, where they concentrate on special tokens).
    - Dimension distribution: concentrated in a very small number of fixed dimensions (only dimension 676 in SD3-5).
    - Low informativeness: massive activation dimensions exhibit significantly lower variance than non-massive dimensions, carrying minimal local information.

2. **Connection Between AdaLN and Massive Activations**:

    - High-value dimensions of the residual scaling factor $\alpha_k$ precisely correspond to massive activation dimensions.
    - AdaLN can adaptively localize massive activations and suppress them via channel modulation.
    - Post-AdaLN features substantially outperform pre-AdaLN features in both semantic consistency and spatial discriminability.

3. **Channel Dropping Strategy**:

    - Post-AdaLN features still contain a small number of weak massive activations.
    - These dimensions are zeroed out to further eliminate their negative impact.
    - Simple and effective, requiring no training whatsoever.

### Loss & Training
- Entirely training-free.
- Feature modulation is performed directly using AdaLN parameters from pretrained DiT models.
- Feature extraction formula: $\hat{z}_t^k = (1+\gamma_k) \text{LayerNorm}(z_t^k) + \beta_k$

## Key Experimental Results

### Main Results (SPair-71k Semantic Correspondence PCK@0.10)

| Method | Type | Mean PCK |
|--------|------|----------|
| ASIC | Unsupervised | 36.9% |
| DINOv2+NN | Unsupervised | 55.6% |
| DIFT | Unsupervised | 57.7% |
| DiTF_sd3-5 | Unsupervised | 64.6% |
| DiTF_flux | Unsupervised | **67.1%** |
| SD+DINO (Supervised) | Supervised | 74.6% |

### Ablation Study

| Model | Massive Activation Dimension | Variance-to-Mean Ratio |
|-------|------------------------------|------------------------|
| SD3-5 (1st dim) | −44.51±0.50 | Extremely low variance |
| SD3-5 (10th dim) | −0.18±2.36 | Normal variance |
| Flux (1st dim) | 40.66±3.99 | Low variance |
| Flux (10th dim) | −0.41±3.16 | Normal variance |

### Key Findings
- DiTF_flux achieves 67.1% PCK on SPair-71k in the unsupervised setting, surpassing all unsupervised methods.
- Compared to directly using raw DiT features (which perform poorly), AdaLN modulation yields a qualitative leap.
- Massive activations are specific to DiTs and absent in SD2.1, explaining why SD features can be used directly while DiT features cannot.
- The channel dropping strategy further improves performance by approximately 1–2%.
- DiTF outperforms DIFT (the SD baseline) by +9.4% on SPair-71k.

## Highlights & Insights
- The systematic analysis of massive activations in DiTs is highly valuable, revealing the fundamental difference between DiTs and SD in feature extraction.
- The paper discovers that AdaLN simultaneously acts as the "source" of massive activations (through residual scaling factors) and their "remedy" (through channel modulation).
- The training-free feature extraction approach offers strong practical utility.
- The research methodology is generalizable to analyzing abnormal activation distributions in other Transformer variants.

## Limitations & Future Work
- Only two DiT variants, SD3-5 and Flux, are evaluated; broader validation across more DiT architectures remains to be done.
- The channel dropping strategy requires prior identification of massive activation dimensions; adaptive schemes warrant further exploration.
- Validation is currently limited to visual correspondence; tasks such as semantic segmentation and depth estimation remain untested.
- No comparison is made against hybrid schemes combining DiT and DINO features.
- Feature selection strategies across different timesteps and DiT layers can be further optimized.
- Fusing DiTF with DINOv2 features to achieve stronger performance is a promising direction.

## Related Work & Insights
- Research on massive activations in LLMs (Sun et al.) provides a conceptual foundation for understanding analogous phenomena in DiTs.
- Attention artifacts in ViTs (addressed via register tokens) are fundamentally different from massive activations in DiTs.
- The idea of leveraging built-in normalization layers for feature modulation is both elegant and effective.
- Cross-domain comparative analysis of massive activations (LLMs vs. ViTs vs. DiTs) holds independent academic value.
- Quantization research has also identified outliers in DiTs as a source of instability, corroborating the findings of this paper.
- This work lays a foundation for feature extraction in subsequent perceptual applications of DiT models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Pioneering problem identification and analysis)
- Technical Contribution: ⭐⭐⭐⭐ (Training-free approach with strong practical utility)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-benchmark comparisons with detailed ablation analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic and outstanding visualizations)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](scaling_diffusion_transformers_efficiently_via_μp.md)
- [\[NeurIPS 2025\] OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers](omnisync_towards_universal_lip_synchronization_via_diffusion.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](../../ICLR2026/image_generation/a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
