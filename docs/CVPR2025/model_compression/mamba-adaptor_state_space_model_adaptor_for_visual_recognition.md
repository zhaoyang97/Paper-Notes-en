---
title: >-
  [Paper Note] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition
description: >-
  [CVPR 2025][Model Compression][Mamba Adaptor] Proposes Mamba-Adaptor to enhance Vision Mamba/SSM with two modules: Adaptor-T (temporal) preserves key historical states using a learnable memory selection mechanism, while Adaptor-S (spatial) enhances spatial locality with multi-scale dilated depthwise convolutions. It achieves 83.0% Top-1 accuracy on ImageNet (Mamba-Adaptor-b2) along with comprehensive improvements in detection, segmentation, and transfer learning.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Mamba Adaptor"
  - "State Space Model"
  - "Learnable Memory Selection"
  - "Multi-scale Spatial Convolution"
  - "Transfer Learning"
date: 2026-05-08
content_hash: d52586a62d0178c6
---

# Mamba-Adaptor: State Space Model Adaptor for Visual Recognition

**Conference**: CVPR 2025  
**arXiv**: [2505.12685](https://arxiv.org/abs/2505.12685)  
**Code**: None  
**Area**: Model Compression / Efficient Adaptation  
**Keywords**: Mamba Adaptor, State Space Model, Learnable Memory Selection, Multi-scale Spatial Convolution, Transfer Learning

## TL;DR

Proposes Mamba-Adaptor to enhance Vision Mamba/SSM with two modules: Adaptor-T (temporal) preserves key historical states using a learnable memory selection mechanism, while Adaptor-S (spatial) enhances spatial locality with multi-scale dilated depthwise convolutions. It achieves 83.0% Top-1 accuracy on ImageNet (Mamba-Adaptor-b2) along with comprehensive improvements in detection, segmentation, and transfer learning.

## Background & Motivation

### Background

**Background**: SSM models like Vision Mamba process long sequences with linear complexity, but two inherent flaws limit their visual performance: (1) the fixed state decay mechanism leads to the forgetting of important historical information; (2) 1D sequence processing ignores the 2D spatial structure of images.

**Limitations of Prior Work**: Selective state decay (parameters $\Delta, A, B$) in SSMs is data-driven, but there is no explicit mechanism to protect key historical states from decay. The influence of distant tokens decays exponentially over time even if they contain crucial information.

**Key Challenge**: The linear efficiency of SSMs stems from their recurrent structure (retaining only hidden states), but this contradicts the preservation of rich historical information.

**Key Insight**: Introducing a learnable memory selection layer on top of SSM hidden states—using a linear layer to predict coordinates of K critical states and retaining them, with multi-sequence aggregation of temporal information at different scales.

**Core Idea**: Learnable memory selection (temporal) + multi-scale dilated convolution (spatial) = a lightweight enhancement adaptor for SSMs.

## Method

### Key Designs

1. **Adaptor-T (Temporal Enhancement)**:

    - Function: Retaining key historical information in SSM states
    - Mechanism: A linear prediction layer selects K critical coordinates from current hidden states and extracts the corresponding state values for retention. Multiple sequences (S sequences) each maintain memory windows at different granularities, which are injected back into the SSM after aggregation.
    - Design Motivation: Ablation studies show learnable selection outperforms static selection by +0.3% on ImageNet, and multi-scale outperforms single-scale by +0.2-0.7%.

2. **Adaptor-S (Spatial Enhancement)**:

    - Function: Restoring the lost 2D spatial locality in SSMs
    - Mechanism: Multi-scale dilated depthwise convolutions extract local spatial features across different receptive fields, which are then fused with global SSM features.
    - Design Motivation: SSMs flatten 2D images into 1D sequences, disrupting local spatial relationships.

### Loss & Training

Standard classification/detection losses. Weight sharing during transfer learning reduces parameters by 94%. The Adaptor increases FLOPs by <7%.

## Key Experimental Results

### Main Results

| Model | ImageNet Top-1 | COCO Box AP |
|------|---------------|-------------|
| VMamba-T | 82.6% | 45.3% |
| Swin-T | 81.3% | - |
| **Mamba-Adaptor-b2** | **83.0%** | **49.1%** |

### Ablation Study

| Configuration | ImageNet | Description |
|------|----------|------|
| Static Selection | 82.7% | — |
| **Learnable Selection** | **83.0%** | +0.3% |
| Single Scale | 82.3% | — |
| **Multi-scale** | **83.0%** | +0.7% |

### Key Findings
- Temporal and spatial enhancements contribute approximately 0.3% each, totaling 0.4% combined (with overlap).
- Transfer learning achieves 99% of full fine-tuning performance using only 9.25% of the parameters.
- COCO detection gains +3.8% AP (49.1 vs 45.3), indicating that spatial locality is particularly critical for detection.

## Highlights & Insights
- **Provides a lightweight solution to the two fundamental drawbacks of SSMs**—introducing adaptors without altering the core SSM architecture.
- **High efficiency of transfer learning**—saves 94% of parameters while maintaining performance close to full fine-tuning.

## Limitations & Future Work
- Large-scale models remain unexplored.
- Solely applicable to Mamba/SSM variants.
- Computational overhead, though small, still increases by 7%.

## Rating
- Novelty: ⭐⭐⭐ The learnable memory selection is novel, but the overall framework is somewhat engineering-oriented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering multiple tasks including classification, detection, segmentation, and transfer learning.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐ Incremental improvement, providing good reference value for the SSM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)

</div>

<!-- RELATED:END -->
