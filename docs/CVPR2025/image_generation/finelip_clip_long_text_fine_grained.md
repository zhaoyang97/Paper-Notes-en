---
title: >-
  [Paper Note] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs
description: >-
  [CVPR 2025][Image Generation][CLIP Extension] This paper proposes FineLIP, which supports long text inputs up to 248 tokens via position embedding stretching and introduces adaptive token refinement alongside cross-modal token-level alignment, significantly outperforming state-of-the-art methods in long-description text retrieval and text-to-image generation tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "CLIP Extension"
  - "Long Text"
  - "Fine-Grained Alignment"
  - "Token-level Contrastive"
  - "Retrieval"
date: 2026-05-08
content_hash: 1acad222345999d9
---

# FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs

**Conference**: CVPR 2025  
**arXiv**: [2504.01916](https://arxiv.org/abs/2504.01916)  
**Code**: [https://github.com/tiiuae/FineLIP](https://github.com/tiiuae/FineLIP)  
**Area**: Image Generation  
**Keywords**: CLIP Extension, Long Text, Fine-Grained Alignment, Token-level Contrastive, Retrieval

## TL;DR

This paper proposes FineLIP, which supports long text inputs up to 248 tokens via position embedding stretching and introduces adaptive token refinement alongside cross-modal token-level alignment, significantly outperforming state-of-the-art methods in long-description text retrieval and text-to-image generation tasks.

## Background & Motivation

**Background**: CLIP is restricted to 77 tokens, preventing it from handling rich and detailed long descriptions. Furthermore, global feature alignment fails to capture fine-grained visual-text correspondences.

**Limitations of Prior Work**: Methods like Long-CLIP and TULIP extend the token length but still solely rely on global feature alignment. Fine-grained methods such as FILIP and SPARC target only short texts and solely refine visual representations.

**Core Idea**: Stretching position embeddings to support long texts, simultaneously applying adaptive aggregation to both visual and textual tokens, and performing cross-modal token-level fine-grained alignment.

## Method

### Key Designs

1. **Position Embedding Stretching**: The first 20 position embeddings (which are well-trained) are preserved, while the remaining ones are stretched by 4 times using adaptive interpolation to reach 248 tokens.

2. **Adaptive Token Refinement Module (ATRM)**: Learner-based aggregation matrices compress visual and textual tokens separately (retaining 20% of tokens with higher information density), reducing redundancy and ambiguity.

3. **Cross-modal Late Interaction (CLIM)**: Achieves fine-grained token-level cross-modal alignment using max-pooling bidirectional similarity coupled with a triplet marginal loss.

### Loss & Training

The standard contrastive loss is replaced with a Triplet Marginal Loss with a margin of $\alpha=0.2$. Global tokens (CLS/EOS) are preserved to participate in the loss calculation, achieving cross-grained alignment.

## Key Experimental Results

### Main Results

| Dataset | Metric | FineLIP | Long-CLIP | TULIP |
|--------|------|---------|-----------|-------|
| Urban1k | I2T R@1 | **0.918** | ~0.86 | 0.881 |
| DOCCI | T2I R@1 | **0.814** | ~0.77 | - |

### Key Findings
- Refining both visual and textual tokens simultaneously is more effective than refining only visual tokens (+3.2% R@1).
- Global-plus-local cross-grained alignment outperforms purely local alignment (+1.8% R@1).
- This method achieves a 12% reduction in FID on text-to-image (T2I) generation tasks.

### Ablation Study

| Configuration | Urban1k I2T R@1 | DOCCI T2I R@1 |
|------|----------------|---------------|
| Full FineLIP | **0.918** | **0.814** |
| w/o ATRM | 0.881 | 0.778 |
| w/o CLIM | 0.892 | 0.791 |
| Visual refinement only | 0.896 | 0.798 |
| Contrastive instead of Triplet | 0.905 | 0.802 |


- Refining both visual and textual tokens simultaneously is more effective than refining only visual tokens.
- Global-plus-local cross-grained alignment outperforms purely local alignment.
- Significant improvements are also observed in text-to-image (T2I) generation tasks.

## Highlights & Insights

- First work to perform token aggregation simultaneously on both modalities.
- Triplet loss is more suitable than contrastive loss in fine-grained scenarios.
- Ablation studies comprehensively validate the contribution of each component.

## Limitations & Future Work

- 248 tokens may still be insufficient for extremely long texts (e.g., detailed scene descriptions can exceed 500+ tokens).
- The aggregation ratio (20%) may need adjustment for different tasks; a static ratio may not be optimal.
- Integration with large-scale LVLEMs (e.g., LLaVA, Qwen-VL) remains to be explored, as these models inherently support long texts.
- Position embedding stretching might introduce positional encoding distortion in certain cases.
- The learnable aggregation matrix of ATRM introduces additional parameters, which might impact lightweight deployment.
- The margin parameter $\alpha=0.2$ in the Triplet Marginal Loss requires tuning and may require different values for different datasets.
- The performance on non-English long texts has not been validated, which is a practical requirement for multilingual long text retrieval.
- The allocation of refinement ratios between visual and textual tokens is not analyzed in detail.

## Related Work & Insights
- **vs Long-CLIP**: Long-CLIP extends token length but still relies on global feature alignment; FineLIP simultaneously introduces token-level fine-grained alignment.
- **vs TULIP**: TULIP supports long texts but only refines visual representations; FineLIP is the first to perform token aggregation on both modalities simultaneously.
- **vs FILIP/SPARC**: These methods target fine-grained alignment only for short texts and refine only visual representations. FineLIP addresses the joint problem of long text and dual-modality refinement.
- Writing Quality: 8/10 — Clear structure

### Methodology Insights
- The core contribution of this work lies in introducing a new architecture to the field, revealing new technical possibilities.
- The experimental design covers multiple baselines and scenarios, showing statistically significant conclusions.
- The components of the method can be independently replaced, facilitating subsequent improvements and optimizations.
- It offers good compatibility with the existing technological ecosystem, lowering the barrier to adoption.
- It provides a tunable balance between computational efficiency and generation quality.
- The open-source code and model weights are of significant value for community replication.
- Driven by practical application needs, the technological innovation is backed by a clearly defined problem statement.
- Comparative analysis with contemporaneous related work is thorough, with a clear positioning.
- Lighter-weight variants can be explored in the future to adapt to edge device deployments.
- Cross-modal and cross-task transferability is an important direction for future validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] MARBLE: Material Recomposition and Blending in CLIP-Space](marble_material_recomposition_and_blending_in_clip-space.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)

</div>

<!-- RELATED:END -->
