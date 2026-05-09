---
title: >-
  [Paper Note] LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers
description: >-
  [AAAI 2026][Object Detection][Mixed Precision Quantization] This paper proposes LampQ, a metric-based layer-wise mixed precision quantization method that measures the quantization sensitivity of each ViT layer via a type-aware Fisher information metric, combines integer linear programming to optimize bit-width allocation, and iteratively refines the allocation. LampQ achieves state-of-the-art performance across image classification, object detection, and zero-shot quantization tasks.
tags:
  - AAAI 2026
  - Object Detection
  - Mixed Precision Quantization
  - Vision Transformer
  - Layer-Adaptive
  - Fisher Information
  - Integer Linear Programming
date: 2026-05-08
content_hash: 279a341928ce8883
---

# LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers

**Conference**: AAAI 2026
**arXiv**: [2511.10004](https://arxiv.org/abs/2511.10004)
**Code**: None
**Area**: Object Detection / Model Compression
**Keywords**: Mixed Precision Quantization, Vision Transformer, Layer-Adaptive, Fisher Information, Integer Linear Programming

## TL;DR

This paper proposes LampQ, a metric-based layer-wise mixed precision quantization method that measures the quantization sensitivity of each ViT layer via a type-aware Fisher information metric, combines integer linear programming to optimize bit-width allocation, and iteratively refines the allocation. LampQ achieves state-of-the-art performance across image classification, object detection, and zero-shot quantization tasks.

## Background & Motivation

**State of the Field**: Vision Transformers (ViTs) have achieved remarkable performance on visual tasks, but their large parameter counts and computational demands hinder edge deployment. Quantization is a primary compression technique that maps floating-point weights and activations to low-bit integer representations.

**Limitations of Prior Work**: Existing quantization methods predominantly adopt uniform precision (e.g., quantizing all layers to 4-bit), ignoring the large sensitivity differences among ViT components (attention layers, FFN layers, embedding layers, etc.). Mixed precision quantization (MPQ) offers a remedy, but existing MPQ methods for ViTs suffer from three main limitations: (1) coarse quantization granularity (bit allocation at the module rather than layer level); (2) incomparable sensitivity metric scales across different component types (Fisher information magnitudes differ greatly between attention and FFN layers); and (3) bit allocation that does not account for actual post-quantization error.

**Root Cause**: Different ViT components exhibit vastly different sensitivities to quantization, yet existing methods cannot accurately measure these differences, leading to suboptimal bit allocation.

**Paper Goals**: (1) Achieve fine-grained, layer-wise mixed precision quantization; (2) design a sensitivity metric that is comparable across component types; (3) optimize bit allocation to minimize overall quantization error.

**Starting Point**: A type-aware Fisher information matrix is used to normalize the sensitivity of different component types, enabling comparison on a unified scale.

**Core Idea**: Type-aware Fisher metric + integer linear programming for optimal bit allocation + iterative refinement = accurate layer-wise mixed precision quantization.

## Method

### Overall Architecture

The LampQ pipeline consists of: (1) computing per-layer quantization sensitivity via a type-aware Fisher metric; (2) solving for the optimal bit allocation under a bit-budget constraint via integer linear programming; and (3) iteratively updating the bit allocation for further refinement. The final output is a mixed precision quantized ViT model.

### Key Designs

1. **Type-Aware Fisher Metric**:

    - Function: Provides comparable quantization sensitivity measurements across component types.
    - Mechanism: The trace of the Fisher information matrix $\text{tr}(F_l)$ is computed for each layer as the base sensitivity indicator. However, Fisher magnitudes vary greatly across component types (attention Q/K/V, FFN up/down, embedding layers, etc.), making direct comparison biased toward certain types. LampQ introduces type-aware normalization — performing intra-group normalization among layers of the same type so that all layer sensitivities are comparable on a unified scale.
    - Design Motivation: This is the key distinction between ViT MPQ and CNN MPQ. CNN layer types are relatively homogeneous, whereas ViTs contain multiple architecturally distinct component types.

2. **ILP Bit Allocation**:

    - Function: Finds the optimal layer-wise bit allocation under a total bit-budget constraint.
    - Mechanism: The bit allocation problem is formulated as an integer linear program: the objective minimizes the weighted quantization error (as given by the Fisher metric), subject to constraints on the total bit budget and per-layer bit range (e.g., 2–8 bits). The ILP solver finds the exact optimal solution in reasonable time, since the number of variables equals the number of layers and the problem scale is manageable.
    - Design Motivation: Heuristic bit allocation strategies (e.g., greedy, top-k) may fall into local optima. ILP provides a global optimality guarantee.

3. **Iterative Bit Allocation Update**:

    - Function: Compensates for the discrepancy between the initial Fisher metric and the actual post-quantization error.
    - Mechanism: The initial Fisher metric is computed on the full-precision model, but the statistical properties of the model change after quantization. After the first round of bit allocation, LampQ re-estimates the Fisher metric on the quantized model and re-solves the ILP, iterating until convergence — typically within 2–3 rounds.
    - Design Motivation: One-shot allocation may be suboptimal due to inaccurate Fisher estimation; iterative updates allow the metric to more accurately reflect the actual post-quantization behavior.

### Loss & Training

LampQ is a post-training quantization (PTQ) method and requires no retraining. A small calibration dataset is used during quantization to compute Fisher information and quantization parameters.

## Key Experimental Results

### Main Results

| Task | Model | Metric | LampQ | Uniform Precision | Gain |
|------|-------|--------|-------|-------------------|------|
| Image Classification | ViT/DeiT | Top-1 Acc | SOTA | Baseline | Significant |
| Object Detection | ViT-based Det | mAP | SOTA | Baseline | Significant |
| Zero-shot Quantization | Various | Acc | SOTA | Baseline | Significant |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| LampQ (Full) | Best | Type-aware Fisher + ILP + Iteration |
| w/o Type-aware Normalization | Degraded | Sensitivity incomparable across types |
| Greedy instead of ILP | Suboptimal | Local vs. global optimum |
| w/o Iterative Update | Slightly worse | First-round metric insufficiently accurate |

### Key Findings

- Type-aware normalization is the critical innovation — without it, MPQ allocation is severely biased toward certain layer types.
- ILP consistently outperforms greedy allocation with acceptable solve time, given that the number of layers is typically in the tens to hundreds.
- LampQ's advantage is more pronounced at low bit-widths (e.g., average 4-bit), where the optimization space for bit allocation is larger.
- The method generalizes consistently across three distinct tasks.

## Highlights & Insights

- **Type-aware normalization** addresses the fundamental challenge of ViT MPQ — enabling sensitivity comparison across heterogeneous component types on a common scale, which is critical for practical deployment.
- **The use of ILP** is uncommon but highly effective in the quantization domain, exploiting the fact that the bit allocation problem is moderate in scale (bounded number of layers).
- The combination of post-training quantization and mixed precision enables immediate application to pretrained models without retraining.

## Limitations & Future Work

- Layer-wise granularity may be insufficient — sensitivity can also vary across channels within the same layer.
- Fisher information computation requires calibration data and is not applicable in zero-data scenarios.
- Hardware constraints on mixed precision support are not considered — some hardware platforms may only support specific bit-width combinations.
- A quantization-aware training variant combined with knowledge distillation could be explored.

## Related Work & Insights

- **vs. Uniform Precision Quantization (e.g., PTQ4ViT)**: Uniform precision methods are simple but suboptimal; LampQ achieves a better accuracy–compression trade-off through mixed precision.
- **vs. Search-based MPQ (e.g., NAS-based)**: NAS methods involve large search spaces and are computationally expensive; LampQ's Fisher + ILP approach is significantly more efficient.
- **vs. CNN MPQ Methods**: ViTs exhibit far greater module heterogeneity than CNNs, necessitating type-aware treatment.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of type-aware Fisher metric and ILP is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three tasks, multiple models, and complete ablations
- Writing Quality: ⭐⭐⭐⭐ Thorough problem analysis and clear method description
- Value: ⭐⭐⭐⭐ Directly applicable to practical ViT quantization and deployment

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)
- [\[AAAI 2026\] Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](beyond_boundaries_leveraging_vision_foundation_models_for_so.md)
- [\[NeurIPS 2025\] Generalizable Insights for Graph Transformers in Theory and Practice](../../NeurIPS2025/object_detection/generalizable_insights_for_graph_transformers_in_theory_and_practice.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](../../ICCV2025/object_detection/adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](../../CVPR2026/object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)

<!-- RELATED:END -->
