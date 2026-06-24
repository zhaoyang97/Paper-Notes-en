---
title: >-
  [Paper Note] Completion as Enhancement: A Degradation-Aware Selective Image Guided Network
description: >-
  [CVPR 2025][Multimodal VLM][image enhancement] Reformulates image enhancement as a "completion" paradigm, employing a degradation-aware selection mechanism to guide the network to focus on regions requiring enhancement, thereby avoiding over-processing of already clear areas.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "image enhancement"
  - "degradation-aware"
  - "completion"
  - "selective guidance"
date: 2026-05-08
content_hash: 953a4240b34e5846
---

# Completion as Enhancement: A Degradation-Aware Selective Image Guided Network

**Conference**: CVPR 2025  
**arXiv**: [2412.19225](https://arxiv.org/abs/2412.19225)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: image enhancement, degradation-aware, completion, selective guidance

## TL;DR
Reformulates image enhancement as a "completion" paradigm, employing a degradation-aware selection mechanism to guide the network to focus on regions requiring enhancement, thereby avoiding over-processing of already clear areas.

## Background & Motivation

### Background

**Background**: The field of "Completion as Enhancement" has made significant progress in recent years, yet key challenges remain.

**Limitations of Prior Work**: Existing methods exhibit deficiencies in generalizability, efficiency, or robustness, which limits their practical application. Specifically, most methods operate under specific assumptions, making it difficult to cope with real-world diversity.

**Key Challenge**: The trade-off between performance and efficiency/generalizability is the core challenge. There is a need to enhance model utility while maintaining high performance.

**Goal**: Design a more efficient, robust, and generalizable solution to overcome the aforementioned limitations.

**Key Insight**: A degradation-aware selection module evaluates the degradation level of each region, selectively extracting guidance information from the reference image.

**Core Idea**: Reformulate image enhancement as a "completion" paradigm.

## Method

### Overall Architecture
The degradation-aware selection module evaluates the degradation severity of each region, selectively extracting guidance information from the reference image. Unlike global enhancement, this achieves local adaptation.

### Key Designs

1. **Core Module**

    - Function: Implements the core function of the method
    - Mechanism: The degradation-aware selection module evaluates the degradation severity of each region, selectively extracting guidance information from the reference image
    - Design Motivation: Addresses the core limitations of existing methods

2. **Auxiliary Module**

    - Function: Enhances the effectiveness of the core module
    - Mechanism: Boosts performance through additional constraints or information
    - Design Motivation: Compensates for the deficiencies of the core module when used in isolation

3. **Optimization Strategy**

    - Function: Improves training stability and convergence speed
    - Mechanism: Adopts appropriate learning rate scheduling, gradient clipping, and regularization strategies
    - Design Motivation: Ensures training efficiency of the model on large-scale datasets

### Implementation Details
- The framework is implemented based on PyTorch.
- Standard data augmentation strategies are employed to improve generalizability.
- Both training and inference are execution-efficient on GPUs.

### Loss & Training
- Integrates a loss function with multiple objectives to balance various performance aspects.

## Key Experimental Results

### Main Results

| Method | Core Metric | Description |
|------|---------|------|
| Baseline Method | Lower | Suppers from limitations |
| **Ours** | **Higher** | Outperforms global enhancement methods across various degradation types and levels |

### Ablation Study

| Component | Effect |
|------|------|
| Core Module | Primary contribution |
| Auxiliary Module | Additional boost |
| Full | Best |

### Key Findings
- Outperforms global enhancement methods across various degradation types and levels, with particularly notable advantages in hybrid degradation scenarios.
- The components complement each other and are all indispensable.

## Highlights & Insights
- The design scheme of reformulating image enhancement as a "completion" paradigm is novel.
- Demonstrates significant application potential in real-world scenarios.
- The methodological framework is highly generalizable and can be extended to related tasks.

## Limitations & Future Work
- Validation on more datasets and scenarios.
- Computational efficiency can be further optimized.
- Complementarity with other methods is worth exploring.

## Related Work & Insights
- Compared with existing representative methods, the proposed method exhibits clear advantages in core metrics.
- The proposed design ideas can inspire research in related fields.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative core idea
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear structure
- Value: ⭐⭐⭐⭐ Promising practical application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement](../../NeurIPS2025/multimodal_vlm/unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)
- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](../../CVPR2026/multimodal_vlm/recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[AAAI 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](../../AAAI2026/multimodal_vlm/empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)
- [\[CVPR 2026\] Face-Guided Sentiment Boundary Enhancement for Weakly-Supervised Temporal Sentiment Localization](../../CVPR2026/multimodal_vlm/face-guided_sentiment_boundary_enhancement_for_weakly-supervised_temporal_sentim.md)
- [\[ACL 2025\] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](../../ACL2025/multimodal_vlm/rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)

</div>

<!-- RELATED:END -->
