---
title: >-
  [Paper Note] Vision Function Layer in Multimodal LLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][MLLM Internal Mechanism] This paper identifies that vision-related functional decoding in MLLMs is concentrated in specific narrow layer blocks (Vision Function Layers), exhibiting a consistent hierarchical order across model families (recognition → counting → grounding → OCR). Building on this finding, the authors propose VFL-LoRA (matching full-LoRA performance with only 1/3 of the parameters) and VFL-select (achieving 98% of full-data performance with 20% of the data).
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - MLLM Internal Mechanism
  - Vision Function Layer
  - Token Swapping
  - LoRA
  - Data Selection
date: 2026-05-08
content_hash: 5639fb3542fbec82
---

# Vision Function Layer in Multimodal LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2509.24791](https://arxiv.org/abs/2509.24791)
**Code**: [GitHub](https://github.com/ChengShiest/Vision-Function-Layer)
**Area**: Multimodal Large Language Models / Interpretability
**Keywords**: MLLM Internal Mechanism, Vision Function Layer, Token Swapping, LoRA, Data Selection

## TL;DR

This paper identifies that vision-related functional decoding in MLLMs is concentrated in specific narrow layer blocks (Vision Function Layers), exhibiting a consistent hierarchical order across model families (recognition → counting → grounding → OCR). Building on this finding, the authors propose VFL-LoRA (matching full-LoRA performance with only 1/3 of the parameters) and VFL-select (achieving 98% of full-data performance with 20% of the data).

## Background & Motivation

- **Core Problem**: Despite significant advances in visual understanding, how MLLMs internally process and reason over visual tokens remains a "black box."
- **Limitations of Prior Work**: Existing interpretability studies have primarily focused on token importance and cross-modal interactions, neglecting how different visual functions are internally represented and coordinated across layers.
- **Key Gap**: A diagnostic framework capable of isolating individual visual functions is absent — most general-purpose tasks simultaneously require multiple capabilities, yielding only coarse conclusions (e.g., "shallow layers extract features, deep layers perform reasoning").
- **Additional Challenge**: Different MLLMs employ different visual encoders and connector modules, further complicating the analysis of internal mechanisms.

## Method

### Vision Token Swapping Analysis Framework

**Mechanism**: At decoding layer $k$, the KV cache of visual tokens from the original image is replaced with visual tokens from a different image, and the resulting change in output is observed. Carefully designed minimal-difference image pairs are used to isolate individual visual functions:

- **OCR**: Different words rendered on a blank canvas.
- **Recognition**: COCO images vs. blank canvas, querying the presence of a specific object.
- **Counting**: CLEVR dataset images differing only in the number of objects.
- **Grounding**: Identical objects placed at different positions.

### Key Findings: Vision Function Layer

Taking Qwen-2.5-VL-7B (28 layers) as an example:

| Visual Function | Peak Layer | Peak Change Rate | Characteristics |
|----------------|-----------|-----------------|----------------|
| Recognition | Layers 0–10 | Distributed | Established early, sustained influence |
| Counting | Layer 12 | 87.4% | Concentrated in middle layers |
| Grounding | Layer 18 | 100.0% | Concentrated in mid-deep layers |
| OCR | Layer 22 | 92.8% | Concentrated in deep layers |

**Alignment with Human Cognition**: The hierarchy mirrors human visual processing — recognition → counting → grounding → reading — and is consistent across LLaVA and Qwen model families.

### Vision Token Dropping Validation

Visual tokens are progressively removed from deeper layers on general VQA benchmarks to validate the VFL findings:
- Dropping last 4 layers: OCR/TextVQA drops sharply (Qwen-7B: 82.8→74.1); other tasks are largely unaffected.
- Dropping last 8 layers: OCR-type tasks nearly collapse (82.8→15.3); Recognition/Spatial tasks begin to degrade.
- Dropping last 12 layers: All visual tasks degrade significantly.

### Application 1: VFL-LoRA

LoRA is applied only to the layers corresponding to the target visual function rather than all layers. Using spatial reasoning as an example, LoRA is trained on Qwen2.5-VL-7B targeting counting function layers (layers 10–17, 20–23):

- Trainable parameters: 155M vs. 309M for full-LoRA (50% reduction).
- In-domain average: 85.0% vs. 84.4% for full-LoRA (on par or slightly better).
- Out-of-domain average: 75.0% vs. 74.3% for full-LoRA (better generalization, reduced catastrophic forgetting).

### Application 2: VFL-select Data Selection

By analyzing performance differences on training data when specific VFLs are ablated, data samples are automatically categorized by function. Using 20% of the data achieves 98% of full-data performance, surpassing manually curated expert data selection.

## Key Experimental Results

### Effect of Vision Token Dropping on Each Task (Qwen2.5-VL-7B)

| Layers Dropped | SQA-I | POPE | TextVQA | OCR | ChartQA |
|---------------|-------|------|---------|-----|---------|
| 0 (baseline) | 87.2 | 86.1 | 82.8 | 82.2 | 83.2 |
| drop 4 | 87.4 | 86.3 | 74.1↓ | 76.3↓ | 82.7↓ |
| drop 8 | 87.4 | 86.2 | 15.3↓↓ | 5.5↓↓ | 20.5↓↓ |
| drop 12 | 87.2 | 79.5↓ | 13.8 | 3.7 | 17.4 |

### VFL-LoRA vs. Full-LoRA (Qwen2.5-VL-7B)

| Method | Params (%) | CV-Count | CV-Avg | ChartQA | MMMU | POPE |
|--------|-----------|----------|--------|---------|------|------|
| Baseline | - | 68.0 | 82.1 | 83.2 | 50.7 | 86.1 |
| Full-LoRA | 1.9% | 70.9 | 84.4 | 86.2 | 50.1 | 86.6 |
| **VFL-LoRA** | **0.9%** | **72.6** | **85.0** | **86.4** | **51.7** | **86.9** |
| Reversed-VFL | 0.9% | 69.0 | 82.7 | 85.9 | 51.2 | 84.9 |

### VFL-select Data Selection

- 20% of data achieves 98% of full-data performance.
- Outperforms human expert data selection under the same budget constraint.

## Highlights & Insights

1. **Cross-model consistent functional hierarchy**: From LLaVA to Qwen, from 3B to 13B parameters, the hierarchical order of vision function layers (recognition → counting → grounding → OCR) is remarkably consistent, suggesting that MLLMs may develop human-like hierarchical visual processing strategies.
2. **Token Swapping is more precise than traditional probing**: Minimal-difference image pairs enable function-level causal analysis rather than mere correlation analysis.
3. **Significant practical value**: VFL-LoRA surpasses full-LoRA with half the parameters and reduced forgetting; VFL-select achieves 98% performance with 1/5 of the data — both applications offer clear engineering utility.
4. **Reversed-VFL provides strong counter-evidence**: Applying LoRA to non-functional layers yields significantly worse performance than VFL-LoRA, confirming the validity of the function layer localization.

## Limitations & Future Work

1. **Limited functional granularity**: Only four visual functions are analyzed (recognition, counting, grounding, OCR); more complex reasoning and causal understanding are not covered.
2. **Dependence on carefully designed image pairs**: Token Swapping relies on the construction of minimal-difference image pairs, which poses a bottleneck for analyzing novel functions.
3. **Variable localization clarity across functions**: Recognition exhibits a distributed rather than localized pattern, indicating that not all functions have well-defined VFLs.
4. **Layer selection for VFL-LoRA requires prior knowledge**: Function layers must first be identified via Token Swapping analysis, increasing the barrier to adoption.
5. **Performance drop on CV-Distance subtask**: This subtask relies more on language priors than visual information, and VFL-LoRA offers limited benefit for such tasks.

## Related Work & Insights

- **LLM Interpretability**: The hierarchical functional division observed in text-only LLMs (e.g., shallow layers for syntax, deep layers for semantics) finds its visual counterpart in MLLMs through this work.
- **LoRA Variants**: VFL-LoRA represents a mechanism-informed layer selection strategy rather than an empirical search, offering stronger theoretical grounding than random or gradient-based layer selection.
- **Implications**: The VFL findings can guide MLLM pruning — if a downstream scenario requires only recognition and counting, the OCR function layers can be safely skipped to accelerate inference.

## Rating

⭐⭐⭐⭐⭐ — The scientific findings are profound (cross-model consistency of functional hierarchy), the methodology is elegant (Token Swapping), the practical applications are compelling (VFL-LoRA and data selection), and the experiments are comprehensive (multiple models × multiple tasks × ablations + applications). This represents an important contribution to the field of MLLM interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)
- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](learning_to_steer_input-dependent_steering_for_multimodal_llms.md)
- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)

</div>

<!-- RELATED:END -->
