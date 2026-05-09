---
title: >-
  [Paper Note] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment
description: >-
  [CVPR 2026][Multimodal VLM][Vision-language pretraining] TIPSv2 is proposed by discovering that distillation substantially improves patch-text alignment, and this insight is translated into a new pretraining objective, iBOT++ (where visible tokens also participate in the loss computation). Combined with head-only EMA and multi-granularity text augmentation, TIPSv2 achieves state-of-the-art performance across 9 tasks and 20 datasets.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Vision-language pretraining
  - patch-text alignment
  - iBOT++
  - distillation
  - zero-shot segmentation
date: 2026-05-08
content_hash: 3d65e5469f1be15b
---

# TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment

**Conference**: CVPR 2026  
**arXiv**: [2604.12012](https://arxiv.org/abs/2604.12012)  
**Code**: [https://gdm-tipsv2.github.io/](https://gdm-tipsv2.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Vision-language pretraining, patch-text alignment, iBOT++, distillation, zero-shot segmentation

## TL;DR

TIPSv2 is proposed by discovering that distillation substantially improves patch-text alignment, and this insight is translated into a new pretraining objective, iBOT++ (where visible tokens also participate in the loss computation). Combined with head-only EMA and multi-granularity text augmentation, TIPSv2 achieves state-of-the-art performance across 9 tasks and 20 datasets.

## Background & Motivation

**State of the Field**: Vision-language pretraining encompasses two major directions: contrastive/sigmoid methods (CLIP, SigLIP, PE) that provide image-text alignment and zero-shot capabilities, and self-supervised methods (DINO, iBOT) that excel at spatial understanding for dense tasks.

**Limitations of Prior Work**: Achieving unified representations that excel at both global (image-level) and dense (patch-level) understanding simultaneously remains a significant challenge. Unified approaches such as TIPS and SigLIP2 still struggle to maintain precise patch-level text alignment. A surprising trend is that the largest flagship models exhibit worse patch-text alignment than their smaller counterparts.

**Root Cause**: The final Transformer layers tend to function as global contrastive "decoders" rather than preserving local semantics, leading to degraded patch-level alignment.

**Paper Goals**: To address patch-text alignment directly during the pretraining stage.

**Starting Point**: The paper discovers that the distillation process substantially improves spatial alignment by imposing effective supervision on all patch tokens — the distilled student model's patch-text alignment far surpasses that of the teacher model.

**Core Idea**: Translate the insight from distillation into the pretraining objective iBOT++, enabling visible tokens to directly participate in the MIM loss.

## Method

### Overall Architecture

TIPSv2 integrates three improvements over TIPS: (1) the iBOT++ objective, which applies self-supervised loss to both visible and masked tokens; (2) a Head-only EMA strategy that updates only the projection layers rather than the full model via EMA; and (3) multi-granularity text augmentation combining synthetic captions from PaliGemma and Gemini. The total loss is $\mathcal{L} = \mathcal{L}_{CLIP} + \mathcal{L}_{DINO} + \mathcal{L}_{iBOT++}$.

### Key Designs

1. **iBOT++ (Enhanced Masked Image Modeling)**:

    - **Function**: Directly enhances patch-text alignment during pretraining.
    - **Mechanism**: Standard iBOT computes loss only on masked tokens ($\mathcal{L}_{iBOT} = -\sum m_i \cdot h_t(f_t(I)_i)^T \log h_s(f_s(I_{mask})_i)$). iBOT++ removes the masking condition, allowing visible tokens to participate in the loss as well. This effectively imposes a representation consistency constraint on all patch tokens.
    - **Design Motivation**: Distillation experiments demonstrate that removing masking and applying supervision to all tokens is the key factor for improving patch-text alignment. The distilled TIPS ViT-L student substantially outperforms the TIPS ViT-g teacher on zero-shot segmentation, with a mIoU gap exceeding 20 points.

2. **Head-only EMA**:

    - **Function**: Substantially reduces training memory and parameter count.
    - **Mechanism**: Standard SSL requires a full-model EMA teacher to prevent model collapse. However, TIPSv2 benefits from additional image-text contrastive loss that provides a stabilizing signal, making it sufficient to apply EMA updates only to the projection head while sharing a single visual encoder.
    - **Design Motivation**: Reduces trainable parameters by nearly half, enabling more efficient large-scale training.

3. **Multi-Granularity Caption Sampling**:

    - **Function**: Enhances the diversity and robustness of text supervision.
    - **Mechanism**: Combines original web alt-text captions, spatially-aware captions from PaliGemma, and detailed descriptive captions from Gemini, randomly sampling captions of different granularities during training.
    - **Design Motivation**: Captions at different granularities capture distinct visual information, improving the model's adaptability across diverse downstream tasks.

### Loss & Training

The overall loss comprises three components: CLIP contrastive loss (with dual CLS tokens for object-level and spatial captions respectively), DINO global self-distillation loss, and iBOT++ patch-level loss (applied to all tokens rather than only masked tokens). Head-only EMA updates the projection layers.

## Key Experimental Results

### Main Results

| Model | PC59 mIoU | PC60 mIoU | VOC21 mIoU | ADE150 mIoU |
|-------|-----------|-----------|------------|-------------|
| TIPS ViT-g (Teacher) | 11.4 | 10.8 | 19.7 | 2.6 |
| TIPS ViT-L (Distilled Student) | 33.5 | 30.4 | 30.5 | 20.8 |
| TIPSv2 ViT-L | **42.1** | **38.2** | **45.3** | **28.7** |
| DINOv2 ViT-L | 35.8 | 32.1 | 38.6 | 23.4 |
| PE-core ViT-L | 28.3 | 25.6 | 32.1 | 18.2 |

### Ablation Study

| Configuration | Masking Ratio | PC59 | VOC21 | ADE150 |
|---------------|---------------|------|-------|--------|
| Standard pretraining | 0.75 | 6.9 | 6.7 | 0.3 |
| Distillation | 0.75 | 16.0 | 22.5 | 5.9 |
| Distillation | 0.5 | 15.5 | 24.0 | 7.0 |
| Distillation (no mask = iBOT++) | 0.0 | **31.4** | **30.8** | **20.0** |

### Key Findings

- Removing masking is the critical factor: reducing the masking ratio from 0.75 to 0.0 during distillation causes PC59 mIoU to jump from 16.0 to 31.4.
- Head-only EMA achieves performance comparable to full-model EMA when text supervision is present, while reducing memory by approximately half.
- The phenomenon of the distilled student substantially surpassing the teacher suggests that patch-text alignment is an emergent capability that can be acquired post hoc.

## Highlights & Insights

- The finding that the distilled student surpasses the teacher is highly instructive: it suggests that patch-level semantics in large models are "diluted" by the global contrastive loss, and that distillation restores local semantics by imposing dense supervision over all tokens.
- The modification introduced by iBOT++ is remarkably simple (effectively a one-line change: removing the masking condition), yet its effect is transformative. Work of this kind — where minimal modifications yield outsized gains — is particularly valuable.
- The observation regarding Head-only EMA indicates that text supervision can substitute for the role of EMA in preventing representation collapse.

## Limitations & Future Work

- Validation is conducted primarily on Google-internal data at scale, making community reproduction challenging.
- The theoretical explanation for the effectiveness of iBOT++ remains insufficiently developed.
- Performance on video understanding and 3D tasks has not been evaluated.
- The effect of iBOT++ on larger model scales warrants further exploration.

## Related Work & Insights

- **vs. TIPS**: TIPSv2's iBOT++ enhances patch-text alignment directly during pretraining, eliminating the need for a separate distillation step.
- **vs. DINOv2**: DINOv2 lacks text alignment; TIPSv2 simultaneously achieves spatial understanding and text alignment.
- **vs. PE**: PE optimizes global contrastive objectives at the expense of dense tasks; TIPSv2 strikes a balance between both via iBOT++.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The finding that the distilled student surpasses the teacher and the introduction of iBOT++ are both highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 9 tasks and 20 datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from discovery to method is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Provides important guidance for pretraining visual foundation models.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models](mostly_text_smart_visuals_asymmetric_text-visual_pruning_for_large_vision-langua.md)
- [\[CVPR 2026\] VISion On Request: Enhanced VLLM Efficiency with Sparse, Dynamically Selected, Vision-Language Interactions](vision_on_request_enhanced_vllm_efficiency_with_sparse_dynamically_selected_visi.md)
- [\[CVPR 2026\] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)

<!-- RELATED:END -->
