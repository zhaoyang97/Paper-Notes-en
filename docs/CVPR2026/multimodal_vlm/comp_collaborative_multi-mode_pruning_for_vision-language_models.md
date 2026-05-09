---
title: >-
  [Paper Note] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Model Pruning] CoMP proposes a collaborative multi-mode pruning framework that eliminates inconsistencies between parameter and token pruning metrics via a Collaborative Importance Metric (CIM), and adaptively selects the optimal pruning mode at each stage through a Multi-mode Pruning Strategy (MPS), achieving significant improvements over single-mode and naive joint pruning approaches at high pruning ratios.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Model Pruning
  - Vision-Language Models
  - Parameter Pruning
  - Token Pruning
  - Collaborative Compression
date: 2026-05-08
content_hash: f2d5c87876788de1
---

# CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2604.02956](https://arxiv.org/abs/2604.02956)
**Code**: [https://github.com/Wuzimeng/CoMP.git](https://github.com/Wuzimeng/CoMP.git)
**Area**: Multimodal VLM
**Keywords**: Model Pruning, Vision-Language Models, Parameter Pruning, Token Pruning, Collaborative Compression

## TL;DR

CoMP proposes a collaborative multi-mode pruning framework that eliminates inconsistencies between parameter and token pruning metrics via a Collaborative Importance Metric (CIM), and adaptively selects the optimal pruning mode at each stage through a Multi-mode Pruning Strategy (MPS), achieving significant improvements over single-mode and naive joint pruning approaches at high pruning ratios.

## Background & Motivation

VLMs are built on Transformer architectures with computational complexity $O(N^2D + ND^2)$, where $N$ is the sequence length and $D$ is the feature dimension. Parameter pruning reduces $D$, while token pruning reduces $N$, making the two approaches complementary.

**Two core challenges**: (1) **Inconsistent importance metrics** — parameter importance is computed using all tokens, but token pruning removes some tokens, causing parameter importance to be dominated by unimportant tokens. Conversely, token importance depends on all parameters, but parameter pruning removes some parameters, leading to distorted token importance scores. (2) **Fixed pruning mode application** — progressive pruning applies the same mode ordering at every stage, whereas the optimal pruning mode varies across stages.

## Method

### Overall Architecture

A nested-loop structure: the outer loop uses MPS to periodically select the optimal pruning mode, while the inner loop applies CIM to compute collaborative parameter and token importance scores and executes pruning under the selected mode.

### Key Designs

1. **Collaborative Importance Metric (CIM)**:

    - **Function**: Eliminates mutual interference between parameter and token importance computation.
    - **Mechanism**: When computing parameter importance, token-weighted input norms are introduced — parameter importance computation is weighted by token importance scores to reduce the influence of unimportant tokens. When computing token importance, parameter pruning masks are propagated into the attention weight matrices to suppress the influence of parameters already marked as unimportant.
    - **Design Motivation**: Experiments show that the most critical tokens in parameter importance computation overlap with token importance rankings by less than 30%, indicating severe inconsistency between the two metrics.

2. **Multi-mode Pruning Strategy (MPS)**:

    - **Function**: Adaptively selects the optimal pruning mode at each stage of progressive pruning.
    - **Mechanism**: The pruning process is divided into multiple stages; at each stage, the "pruning cost" of different modes (visual parameters / language parameters / visual tokens / language tokens) is estimated, and the mode with the lowest cost is executed. Historical costs (for stability) and random exploration (to avoid local optima) are jointly incorporated.
    - **Design Motivation**: The optimal mode differs across stages — parameter pruning may be preferable early on, while token pruning may be more effective later. A fixed ordering cannot adapt to such variation.

3. **Cross-modal Collaborative Pruning**:

    - **Function**: Adaptively prunes both visual and language modalities simultaneously.
    - **Mechanism**: CIM and MPS are applied independently to the visual encoder and the language model, with pruning ratios across modalities adaptively allocated by MPS, allowing the visual and language components to be compressed at different rates.
    - **Design Motivation**: The degree of redundancy differs between visual and language components, making uniform pruning suboptimal.

### Loss & Training

Structured pruning based on importance scores is employed without retraining. Pruning costs are estimated based on model performance changes on a validation set.

## Key Experimental Results

### Main Results

| Method | NLVR2 (50% pruning) | NLVR2 (70% pruning) | VQA | Image-Text Retrieval |
|--------|---------------------|---------------------|-----|----------------------|
| Parameter pruning only | Medium | Poor | Medium | Medium |
| Token pruning only | Medium | Poor | Medium | Medium |
| Naive joint pruning | Medium | Poor | Medium | Medium |
| **CoMP** | **Best** | **Significantly better** | **Best** | **Best** |

The advantage is particularly pronounced at high pruning ratios (70%+).

### Ablation Study

| Configuration | High-pruning Performance | Notes |
|---------------|--------------------------|-------|
| w/o CIM (independent metrics) | Noticeable drop | Inconsistent metrics cause erroneous pruning |
| w/o MPS (fixed mode) | Drop | Suboptimal mode ordering |
| w/o random exploration | Slight drop | Susceptible to local optima |
| Full CoMP | Best | All components are necessary |

### Key Findings

- The contribution of CIM becomes more pronounced at higher pruning ratios — at low pruning ratios, the impact of metric inconsistency is relatively minor.
- The adaptive mode selection of MPS eliminates the need for manual tuning — the optimal strategy varies across tasks and models.
- The optimal pruning ratios for visual and language components are indeed different, making uniform pruning suboptimal.

## Highlights & Insights

- **Discovery of metric inconsistency**: The mutual interference between parameter and token importance metrics had been previously overlooked; CIM's collaborative design addresses this problem elegantly.
- **Adaptive mode selection**: Drawing inspiration from multi-armed bandit approaches (cost estimation + exploration), CoMP enables automated strategy selection during pruning.
- **Advantage at high pruning ratios**: The gains are largest precisely in the high-compression regime most relevant to practical deployment.

## Limitations & Future Work

- The mode selection mechanism of MPS introduces additional computational overhead during pruning.
- Validation is currently limited to the BLIP model family; applicability to architectures such as LLaVA requires further investigation.
- Dynamic token pruning at inference time necessitates dedicated inference optimization.
- Future work may explore joint compression with quantization.

## Related Work & Insights

- **vs. UPop/EViT**: Single-mode pruning methods that suffer steep performance degradation at high compression ratios.
- **vs. Naive joint pruning**: Does not address metric inconsistency and underperforms even individual single-mode pruning methods.
- **vs. DepGraph/PLATON**: Parameter-pruning-specific methods that lack compression along the token dimension.

## Rating

- Novelty: ⭐⭐⭐⭐ The identification of the metric inconsistency problem and the CIM design are genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple tasks and pruning ratios.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ Directly applicable to practical VLM deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatiotemporal_collaborative_network.md)
- [\[CVPR 2026\] Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models](mostly_text_smart_visuals_asymmetric_text-visual_pruning_for_large_vision-langua.md)
- [\[CVPR 2026\] Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)

<!-- RELATED:END -->
