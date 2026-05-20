---
title: >-
  [Paper Note] From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration
description: >-
  [ACL 2026][Multimodal VLM][To be supplemented] To be supplemented after thorough reading.
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "To be supplemented"
date: 2026-05-08
content_hash: f67321fe496322a8
---

# From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration

**Conference**: ACL 2026
**arXiv**: [2604.16462](https://arxiv.org/abs/2604.16462)  
**Code**: [https://github.com/civilizwa/HalfV](https://github.com/civilizwa/HalfV)  
**Area**: Multimodal VLM / Inference Acceleration
**Keywords**: visual redundancy, MLLM acceleration, architecture-aware, token pruning, matrix entropy

## TL;DR

This work identifies two sources of visual redundancy in MLLM inference — inherent visual redundancy (IVR) arising from dense ViT tokenization, and secondary saturation redundancy (SSR) emerging from deep-layer semantic saturation whose manifestation varies across backbone architectures — and proposes the HalfV framework to address each type separately, achieving a 4.1× FLOPs speedup on Qwen2.5-VL while retaining 96.8% of performance.

## Background & Motivation

**Background**: High-resolution MLLMs incur extremely high inference costs due to the explosion in the number of visual tokens. Existing acceleration methods include token-level pruning and layer-level sparsity.

**Limitations of Prior Work**: Existing acceleration strategies exhibit severe *backbone dependency* — they perform well on Vicuna/Mistral architectures (e.g., LLaVA) but suffer 5.7%–22.4% performance degradation when transferred to Qwen-based architectures. Controlled experiments using LLaVA-Next with an identical visual frontend confirm that the bottleneck lies in the fundamentally different mechanisms by which distinct LLM backbones process visual information.

**Key Challenge**: Different backbone architectures process visual information in fundamentally different ways, yet existing methods assume a one-size-fits-all strategy. Understanding the essential differences in visual redundancy across architectures is a prerequisite for designing universally applicable acceleration schemes.

**Goal**: Use truncated matrix entropy as a probe to systematically track the evolution of visual information across different architectures, and design an architecture-aware acceleration framework accordingly.

**Key Insight**: Truncated matrix entropy is employed to trace the eigenvalue spectral evolution of visual representations, revealing a universal three-stage inference lifecycle — modality alignment → global aggregation → visual saturation — that holds across architectures.

**Core Idea**: Visual redundancy is disentangled into architecture-agnostic IVR (from dense ViT tokenization) and architecture-dependent SSR (from deep-layer saturation). IVR is handled by a unified pruning strategy, while SSR is addressed adaptively according to its architecture-specific manifestation: layer-wise inactivity in Vicuna/Mistral versus extreme token sparsity in Qwen.

## Method

### Overall Architecture

HalfV operates in two steps: (1) a one-shot token pruning applied uniformly across all architectures at the start of Stage II to eliminate IVR; (2) architecture-specific SSR handling in Stage III — KV cache reuse to skip layer computation for Vicuna/Mistral, and retaining only the top-5% dominant tokens for computation in Qwen.

### Key Designs

1. **Discovery of the Three-Stage Inference Lifecycle**

    - **Function**: Provides a universal model of visual information processing across architectures.
    - **Mechanism**: Truncated matrix entropy tracks the layer-wise evolution of visual and textual representations. Stage I (modality alignment) — visual entropy is high and stable, textual entropy compresses rapidly, and attention shifts from balanced to text-dominated. Stage II (global aggregation) — visual entropy begins to decline, scattered visual evidence is consolidated into key semantic regions, and the process is highly sensitive to local perturbations (suppressing just 1% of tokens causes severe degradation). Stage III (visual saturation) — visual context saturates and the marginal benefit of additional computation diminishes.
    - **Design Motivation**: A unified theoretical framework is needed to explain why different architectures respond differently to the same acceleration strategy.

2. **Unified Handling of Inherent Visual Redundancy (IVR)**

    - **Function**: Eliminates the spatial redundancy introduced by ViT at the optimal moment — the onset of Stage II.
    - **Mechanism**: Marginal utility analysis via $\text{MU}_{l,r} = -\Delta\mathcal{M} / (\Delta\mathcal{C} + \epsilon)$ reveals that the initial layer of Stage II is the optimal position for one-shot pruning (MU = 0.21 vs. 0.29–0.87 at other positions). Pruning at this point avoids interference with Stage I alignment while exploiting the high redundancy of visual representations prior to Stage II.
    - **Design Motivation**: Stage II is extremely sensitive to local perturbations, making progressive per-layer pruning during aggregation infeasible; a one-shot pruning before aggregation begins is therefore preferred.

3. **Architecture-Aware Handling of Secondary Saturation Redundancy (SSR)**

    - **Function**: Selects the optimal acceleration strategy based on the architecture-specific manifestation of SSR.
    - **Mechanism**: SSR in Vicuna/Mistral manifests as layer-wise inactivity (KL divergence $\approx 0$, indicating no information gain per layer), enabling direct KV cache reuse to skip computation. SSR in Qwen manifests as extreme token sparsity (layers remain active but information flow collapses onto a tiny number of dominant tokens), requiring full-precision computation for the top-5% tokens. Cross-validation confirms this distinction: suppressing all visual updates benefits Vicuna (OCRBench +13.1%) but catastrophically fails on Qwen (−86.2%), while retaining 5% tokens incurs negligible loss on Qwen (−0.1% to −2.4%), demonstrating the fundamentally different nature of SSR across the two architectures.
    - **Design Motivation**: A uniform acceleration strategy inevitably fails across architectures; the choice of strategy must correspond to the specific manifestation of SSR.

### Loss & Training

HalfV is a training-free inference-time acceleration method. A lightweight pre-analysis on 100 samples is sufficient to determine the three-stage boundaries. Evaluation is conducted on LLaVA-1.5v-7B (Vicuna), LLaVA-1.5v-7B (Mistral), and Qwen2.5-VL-7B across benchmarks including GQA, MME, POPE, SQA, and AI2D.

## Key Experimental Results

### Main Results

| Model | Method | FLOPs Speedup | Avg. Performance Retention |
|-------|--------|---------------|---------------------------|
| Qwen2.5-VL | HoloV (prior method) | High | Poor (5.7–22.4% degradation) |
| Qwen2.5-VL | HalfV | **4.1×** | **96.8%** |
| LLaVA-1.5v (Vicuna) | HalfV | High | Excellent |
| LLaVA-1.5v (Mistral) | HalfV | High | Excellent |

### Ablation Study

| Configuration | Result | Note |
|---------------|--------|------|
| IVR handling only | Moderate speedup | Universal pruning is effective but insufficient |
| SSR handling only | Limited speedup | Addressing deep-layer redundancy alone is insufficient |
| IVR + SSR (full HalfV) | Optimal | Two stages are complementary |
| Incorrect SSR strategy | Catastrophic degradation | Confirms necessity of architecture awareness |

### Key Findings

- The root cause of performance degradation for existing methods on Qwen is the distinct manifestation of SSR — Qwen's deep layers remain active but become extremely sparse, making simple layer skipping inapplicable.
- The onset of Stage II is the optimal timing for one-shot pruning, as indicated by the lowest marginal utility.
- Suppressing just 1% of tokens during Stage II causes performance collapse, confirming the high coupling of the global aggregation process.
- The three-stage lifecycle is consistent across all tested architectures, while the manifestation of SSR is architecture-dependent.

## Highlights & Insights

- **Systematic identification and explanation of backbone dependency**: This work is the first to demonstrate that failures of MLLM acceleration methods stem from backbone architecture differences rather than visual frontend differences, and provides a mechanistic explanation via matrix entropy analysis.
- **Elegance of the IVR/SSR disentanglement framework**: Complex visual redundancy is decomposed into two independently addressable components, each paired with its optimal handling strategy.
- **Marginal utility analysis for optimal pruning timing**: The pruning layer is identified quantitatively rather than heuristically, offering methodological value.

## Limitations & Future Work

- The pre-analysis phase requires 100 samples to determine stage boundaries; varying data distributions may shift these boundaries.
- Validation is limited to three backbone families (Vicuna, Mistral, Qwen); the SSR manifestation in additional architectures remains unknown.
- The top-5% ratio in the extreme token sparsity strategy may require task-specific tuning.
- Comparisons with the latest dynamic token management methods are not included.

## Related Work & Insights

- **vs. HoloV/DART (token pruning methods)**: These methods implicitly assume identical redundancy patterns across all architectures, leading to severe degradation on Qwen. HalfV addresses this through architecture-aware SSR handling.
- **vs. ShortV (layer-skipping methods)**: ShortV assumes deep layers can be skipped, which holds for Vicuna but not for Qwen. HalfV distinguishes between "layer inactivity" and "token sparsity" as two distinct SSR modes.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The IVR/SSR disentanglement and architecture-aware analysis are original contributions; the three-stage lifecycle discovery has independent value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three architectures, eight benchmarks, marginal utility analysis, and cross-validated SSR experiments — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Analysis is deep and figures are informative, though some descriptions are highly technical.
**Code**: To be confirmed  
**Area**: multimodal_vlm
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Visual Prompt-Agnostic Evolution](../../ICLR2026/multimodal_vlm/visual_prompt-agnostic_evolution.md)
- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[ACL 2026\] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)
- [\[AAAI 2026\] Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](../../AAAI2026/multimodal_vlm/global_compression_commander_plug-and-play_inference_acceler.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)

</div>

<!-- RELATED:END -->
