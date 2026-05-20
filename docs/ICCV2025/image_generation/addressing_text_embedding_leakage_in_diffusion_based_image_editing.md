---
title: >-
  [Paper Note] Addressing Text Embedding Leakage in Diffusion-based Image Editing
description: >-
  [ICCV 2025][Image Generation][attribute leakage] This paper proposes the ALE framework, which systematically addresses attribute leakage in diffusion-based text-guided image editing through three components: Object-Restr…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "attribute leakage"
  - "diffusion image editing"
  - "EOS embedding"
  - "cross-attention masking"
  - "multi-object editing"
date: 2026-05-08
content_hash: 3e6e13a1aee2c978
---

# Addressing Text Embedding Leakage in Diffusion-based Image Editing

**Conference**: ICCV 2025
**arXiv**: N/A  
**Code**: [GitHub](https://github.com/mtablo/ALE_Edit_page)  
**Area**: Image Editing / Diffusion Models
**Keywords**: attribute leakage, diffusion image editing, EOS embedding, cross-attention masking, multi-object editing

## TL;DR

This paper proposes the ALE framework, which systematically addresses attribute leakage in diffusion-based text-guided image editing through three components: Object-Restricted Embedding (ORE) to decouple semantic entanglement in EOS tokens, Region-Guided Blended Cross-Attention Masking (RGB-CAM) to constrain spatial attention, and Background Blending (BB) to preserve unedited regions. A new evaluation benchmark, ALE-Bench, is also introduced.

## Background & Motivation

- **Background**: Diffusion-based text-guided image editing has advanced substantially, yet attribute leakage—where editing a target object unintentionally affects unrelated regions—remains a critical unsolved problem.
- **Limitations of Prior Work**: Existing methods attempt to constrain editing effects by manipulating attention maps but fail to address the root cause of EOS embedding entanglement. Two types of leakage are identified: (1) Target-External Leakage (TEL), where attributes of a target object spread to non-target regions; and (2) Target-Internal Leakage (TIL), where attributes of one target object affect another.
- **Key Challenge**: The EOS (End-of-Sequence) token in autoregressive text encoders (e.g., CLIP) indiscriminately aggregates information from all tokens in the prompt, causing attributes to diffuse spatially without discrimination through cross-attention layers.
- **Goal**: To systematically eliminate both TEL and TIL, preserve background integrity, and establish a dedicated evaluation benchmark for multi-object editing.

## Method

### Overall Architecture

ALE is a tuning-free image editing framework built upon a dual-branch diffusion model architecture. It consists of three complementary components: ORE addresses entanglement at the text embedding level, RGB-CAM constrains spatial attention leakage, and BB handles background preservation. All three components are necessary—BB alone prevents TEL but not TIL; ORE + RGB-CAM reduces TIL but cannot protect the background.

### Key Designs

1. **Object-Restricted Embedding (ORE)**: Assigns each target object in the prompt an independent, semantically isolated text embedding by encoding each object separately rather than placing all objects in a single prompt. This severs semantic entanglement in the EOS token at its source.

2. **Region-Guided Blended Cross-Attention Masking (RGB-CAM)**: Uses segmentation masks to constrain cross-attention so that each object's attention is restricted to its designated region. By blending segmentation masks with attention maps, it prevents improper spatial diffusion of attributes and enables precise region-level editing.

3. **Background Blending (BB)**: Blends the source image latent with the edited latent in non-target regions throughout the denoising process, preserving structural integrity and appearance consistency of unedited areas to effectively prevent TEL.

### Loss & Training

ALE is a training-free method that operates entirely at inference time by modifying the diffusion sampling process. It is compatible with existing editing pipelines such as Prompt-to-Prompt as the underlying backbone.

## Key Experimental Results

### Main Results

| Method | TELS↓ | TILS↓ | Structure Dist↓ | PSNR↑ | SSIM↑ |
|--------|-------|-------|-----------------|-------|-------|
| P2P | 21.52 | 17.26 | 0.1514 | 11.15 | 0.5589 |
| MasaCtrl | 20.18 | 16.74 | 0.0929 | 14.99 | 0.7346 |
| InfEdit | 19.59 | 16.69 | 0.0484 | 16.74 | 0.7709 |
| **ALE** | **16.03** | **15.28** | **0.0167** | **30.04** | **0.9228** |

ALE substantially outperforms all baselines across every metric, improving PSNR from 16.74 to 30.04 and SSIM from 0.77 to 0.92.

### Ablation Study

- BB alone: addresses TEL but cannot resolve TIL.
- ORE + RGB-CAM: reduces TIL but fails to protect the background (TEL persists).
- Full ALE (BB + ORE + RGB-CAM): simultaneously eliminates both TEL and TIL.
- Performance remains robust across varying numbers of edited objects (1/2/3) and editing types (color/material/object).

### Key Findings

- The EOS token is the fundamental cause of attribute leakage, not merely an attention map issue.
- Multi-object editing is more prone to leakage than single-object editing; nevertheless, ALE achieves even better performance on 3-object editing than on 1-object editing.
- Combined editing types (e.g., color + object) exhibit the most severe leakage, yet ALE still effectively suppresses it.

## Highlights & Insights

- The analysis is thorough and precise—correctly identifying EOS embedding entanglement as the root cause of attribute leakage rather than a surface-level attention artifact.
- ALE-Bench and the TELS/TILS metrics fill a gap in evaluation protocols for multi-object editing scenarios.
- The training-free design allows plug-and-play integration into various diffusion-based editing frameworks.
- Background preservation quality is exceptional (PSNR 30.04 vs. the prior best of 16.74).

## Limitations & Future Work

- Currently limited to local, relatively simple transformations (color/material/object replacement); non-rigid transformations such as style transfer and pose modification are not supported.
- ALE-Bench contains only 20 carefully curated images, and its generalizability remains to be validated.
- RGB-CAM requires segmentation masks as input, which raises the barrier to practical use.
- ORE encodes each object separately, leading to linearly increasing computational overhead as the number of target objects grows.

## Related Work & Insights

- Prompt-to-Prompt, MasaCtrl, and InfEdit serve as the primary tuning-free editing baselines for comparison.
- EOS token semantic entanglement may also manifest in other generative tasks that employ CLIP text encoders.
- The region-level editing control paradigm introduced here is transferable to video editing, 3D editing, and related domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The analysis of EOS embedding leakage is insightful and original.
- **Technical Depth**: ⭐⭐⭐⭐ — The three-component design is logically coherent and mutually complementary.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — New benchmark, new metrics, comprehensive comparisons, and ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear and visualizations are excellent.
- **Value**: ⭐⭐⭐⭐ — Training-free, immediately applicable, and demonstrably effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)

</div>

<!-- RELATED:END -->
