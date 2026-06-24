---
title: >-
  [Paper Note] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging
description: >-
  [ICML 2025][Model Compression][Model Merging] By directly perform weighted averaging of the parameters of a mathematical reasoning LLM and the text components of a VLM (model merging), reasoning capability is transferred to the VLM without any training. Furthermore, a layer-wise distribution pattern is discovered where perception capabilities are concentrated in the early layers, while reasoning capabilities are concentrated in the middle and late layers.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Model Merging"
  - "VLM Reasoning"
  - "Perception and Reasoning Decoupling"
  - "Cross-modal Transfer"
  - "Layer-wise Analysis"
date: 2026-05-08
content_hash: b973af221fc98b22
---

# Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging

**Conference**: ICML 2025  
**arXiv**: [2505.05464](https://arxiv.org/abs/2505.05464)  
**Code**: [https://github.com/shiqichen17/VLM_Merging](https://github.com/shiqichen17/VLM_Merging)  
**Area**: Multimodal VLM  
**Keywords**: Model Merging, VLM Reasoning, Perception and Reasoning Decoupling, Cross-modal Transfer, Layer-wise Analysis

## TL;DR
By directly perform weighted averaging of the parameters of a mathematical reasoning LLM and the text components of a VLM (model merging), reasoning capability is transferred to the VLM without any training. Furthermore, a layer-wise distribution pattern is discovered where perception capabilities are concentrated in the early layers, while reasoning capabilities are concentrated in the middle and late layers.

## Background & Motivation

### Background

**Background**: VLMs perform exceptionally well on visual perception + language tasks. However, they lag far behind text-only LLMs in complex multimodal reasoning (such as math chart interpretation), partly due to the scarcity of multimodal reasoning data.

**Limitations of Prior Work**: Enhancing the reasoning capabilities of VLMs usually requires collecting a massive amount of multimodal reasoning data and fine-tuning, which is highly expensive.

**Key Challenge**: Can perception and reasoning capabilities be decoupled? Can reasoning capabilities be directly transferred from LLMs to VLMs?

**Goal**: To explore model merging as a pathway for cross-modal capability transfer.

**Key Insight**: The text components of VLMs share the same architecture and initialization as LLMs, satisfying the connected subspace hypothesis of model merging.

**Core Idea**: Perform weighted averaging of the text parameters of the VLM and the mathematical LLM to achieve zero-training reasoning capability transfer.

## Method

### Overall Architecture
1. Select a VLM (e.g., LLaVA-NeXT/8B) and a mathematical LLM (e.g., Dart-Math) that share a base model.
2. Calculate the task vector: $\tau = W_{\text{math}} - W_{\text{base}}$
3. Merge: $W_{\text{merged}} = W_{\text{VLM}} + \alpha \cdot \tau$
4. Directly deploy the merged model without any training.

### Key Designs

1. **Cross-modal Model Merging**:

    - Function: Adds the task vector of the mathematical LLM to the text parameters of the VLM.
    - Mechanism: Since the VLM and LLM are fine-tuned from the same base model, their parameter spaces are connected. Weighted averaging can transfer capabilities.
    - Design Motivation: Reasoning ability should be encoded in the text processing layers. Keeping the visual encoder unchanged preserves perception capabilities.

2. **Layer-wise Capability Analysis (Knockout Analysis)**:

    - Function: Progressively masks merged parameters layer-by-layer to observe changes in perception and reasoning.
    - Key Findings: (a) Perception capabilities are concentrated in the early layers; (b) reasoning capabilities are concentrated in the middle and late layers; (c) after merging, reasoning capabilities expand to all layers while the distribution of perception remains unchanged.
    - Design Motivation: To understand how merging affects different capabilities within the parameter space.

### Loss & Training
- Completely training-free.
- The merging coefficient $\alpha$ is typically optimal between 0.3 and 0.7.

## Key Experimental Results

### Main Results

| Model | MathVista (Math) | MathVerse (Vision-Only) | Perception Tasks |
|------|-----------------|------------------------|---------|
| LLaVA-NeXT | 38.2 | 21.3 | 78.5 |
| + Dart-Math Merging | **41.8** (+3.6) | **22.7** (+1.4) | 78.1 (-0.4) |
| + MetaMath Merging | 40.9 (+2.7) | 22.0 (+0.7) | 78.3 (-0.2) |

### Ablation Study

| Configuration | Math↑ | Perception | Description |
|------|-------|-----------|------|
| Early-layer merging only | +0.8 | Unchanged | Early layers primarily encode perception |
| Mid-to-late layer merging only | +3.1 | Unchanged | Reasoning is primarily in mid-to-late layers |
| Full-layer merging | +3.6 | Slight decrease | Optimal for reasoning, but perception is slightly affected |
| $\alpha=0.3$ | +2.1 | Unchanged | Conservative merging |
| $\alpha=0.7$ | +3.6 | -0.8 | Aggressive merging |

### Key Findings
- Model merging consistently improves reasoning across multiple VLMs (LLaVA, Idefics2, InternVL2) and multiple mathematical LLMs.
- Perception and reasoning are indeed largely decoupled in the parameter space—the former in early layers and the latter in mid-to-late layers.
- Merging alters the layer distribution of reasoning (expanding it to all layers), while the perception distribution remains unchanged.

## Highlights & Insights
- **Training-free capability transfer** is highly practical—requiring only parameter weighted averaging, resulting in almost zero cost.
- The discovery of the layer-wise distribution of capabilities is of great value for understanding the internal mechanisms of VLMs.
- Opens up a new direction for "model merging as a multimodal integration tool".

## Limitations & Future Work
- Requires the VLM and LLM to share a base model, which limits the scope of applicability.
- Only mathematical reasoning was tested; other types of reasoning (logic, common sense) have not been verified.
- The slight drop in perception tasks may be exacerbated under more aggressive merging.
- More complex merging strategies (such as TIES, DARE) were not explored.

## Related Work & Insights
- **vs. Traditional Model Merging**: Prior works only merged models of the same modality, whereas this paper introduces cross-modal merging for the first time.
- **vs. Fine-tuning with Reasoning Data**: The latter requires large amounts of multimodal reasoning data, while this paper uses zero data.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of cross-modal model merging and layer-wise analysis is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple VLMs, multiple LLMs, and layer-wise ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation and in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Practical and insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](became_bayesian_continual_learning_with_adaptive_model_merging.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](../../ICML2026/model_compression/frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[CVPR 2026\] Bridging Domains through Subspace-Aware Model Merging](../../CVPR2026/model_compression/bridging_domains_through_subspace-aware_model_merging.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[ICML 2025\] LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reliable Model Merging](lift_the_veil_for_the_truth_principal_weights_emerge_after_rank_reduction_for_re.md)

</div>

<!-- RELATED:END -->
