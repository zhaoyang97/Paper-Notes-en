---
title: >-
  [Paper Note] From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration
description: >-
  [ACL 2026][Multimodal VLM][Visual Redundancy] This paper reveals two sources of visual redundancy in MLLM inference: Inherent Visual Redundancy (IVR) caused by dense ViT tokenization and Secondary Saturation Redundancy (…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Visual Redundancy"
  - "MLLM Acceleration"
  - "Architecture-Aware"
  - "Token Pruning"
  - "Matrix Entropy"
date: 2026-05-08
content_hash: 63bd7bdb24bbec06
---

# From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration

**Conference**: ACL 2026  
**arXiv**: [2604.16462](https://arxiv.org/abs/2604.16462)  
**Code**: [https://github.com/civilizwa/HalfV](https://github.com/civilizwa/HalfV)  
**Area**: Multimodal VLM / Inference Acceleration  
**Keywords**: Visual Redundancy, MLLM Acceleration, Architecture-Aware, Token Pruning, Matrix Entropy

## TL;DR

This paper reveals two sources of visual redundancy in MLLM inference: Inherent Visual Redundancy (IVR) caused by dense ViT tokenization and Secondary Saturation Redundancy (SSR) caused by deep semantic saturation, where SSR manifestations vary across backbone architectures. It proposes the HalfV framework to handle these redundancies separately, achieving a 4.1× FLOPs acceleration on Qwen2.5-VL while retaining 96.8% of the performance.

## Background & Motivation

**Background**: High-resolution MLLMs face extreme inference costs due to visual token explosion. Existing acceleration methods include token-level pruning and layer-level sparsity.

**Limitations of Prior Work**: Existing acceleration strategies exhibit severe "backbone dependency"—performing well on Vicuna/Mistral architectures (e.g., LLaVA) but suffering performance degradation of 5.7%-22.4% when migrated to Qwen architectures. Controlled experiments confirmed that the bottleneck lies in the different intrinsic mechanisms of LLM backbones in processing visual information.

**Key Challenge**: Different backbone architectures process visual information fundamentally differently, yet existing methods assume a "one-size-fits-all" strategy. Understanding the essential differences in visual redundancy across architectures is necessary to design a universal acceleration scheme.

**Goal**: Use truncated matrix entropy as a probe to systematically track the evolution of visual information across different architectures and design an architecture-aware acceleration framework accordingly.

**Key Insight**: By tracking the eigenvalue spectrum evolution of visual representations using truncated matrix entropy, a cross-architecture universal three-stage inference lifecycle was discovered (Modal Alignment → Global Aggregation → Visual Saturation).

**Core Idea**: Decouple visual redundancy into universal IVR (from ViT dense tokenization) and architecture-dependent SSR (from deep saturation). The former is processed with a unified pruning strategy, while the latter is handled adaptively based on architecture-specific manifestations (layer inactivity in Vicuna/Mistral vs. extreme token sparsity in Qwen).

## Method

### Overall Architecture

HalfV consists of two steps: (1) Uniformly execute one-time token pruning at the start of Stage II for all architectures to eliminate IVR; (2) Process SSR in Stage III based on architecture specificity—reusing KV caches to skip layer computation for Vicuna/Mistral architectures, while keeping only the top-5% dominant tokens for computation in Qwen architectures.

### Key Designs

1.  **Discovery of the Three-Stage Inference Lifecycle**:
    *   Function: Provides a universal visual information processing model across architectures.
    *   Mechanism: Tracks the evolution of visual and textual representations across layers using truncated matrix entropy. Stage I (Modal Alignment)—visual entropy is high and stable, textual entropy compresses rapidly, and attention shifts from balanced to text-dominant. Stage II (Global Aggregation)—visual entropy begins to drop as scattered visual evidence aggregates into key semantic regions, highly sensitive to local perturbations (suppressing only 1% of tokens leads to severe degradation). Stage III (Visual Saturation)—visual context saturates, and additional computation yields diminishing marginal returns.
    *   Design Motivation: A unified theoretical framework is needed to explain why different architectures react differently to the same acceleration strategy.

2.  **Unified Handling of Inherent Visual Redundancy (IVR)**:
    *   Function: Eliminates spatial redundancy from ViT at the optimal timing (start of Stage II).
    *   Mechanism: Marginal utility analysis $\text{MU}_{l,r} = -\Delta\mathcal{M} / (\Delta\mathcal{C} + \epsilon)$ reveals that the starting layer of Stage II is the best location for one-time pruning (MU=0.21 vs. 0.29-0.87 for other locations). Pruning here avoids alignment interference in Stage I and exploits highly redundant visual representations before Stage II.
    *   Design Motivation: Stage II is extremely sensitive to local perturbations, preventing layer-by-layer pruning during aggregation; however, one-time pruning is feasible before it begins.

3.  **Architecture-Aware Handling of Secondary Saturation Redundancy (SSR)**:
    *   Function: Selects the optimal acceleration strategy based on architecture-specific manifestations.
    *   Mechanism: SSR in Vicuna/Mistral manifests as layer inactivity (KL divergence $\approx 0$, no information gain per layer), allowing for direct computation skipping via KV cache reuse. SSR in Qwen manifests as extreme token sparsity (layers remain active, but information flow collapses into a few dominant tokens), requiring full-precision computation for top-5% tokens. Experiments confirm that suppressing all visual updates works well on Vicuna (OCRBench +13.1%) but fails catastrophically on Qwen (-86.2%); conversely, retaining 5% tokens on Qwen is nearly lossless (-0.1% to -2.4%), confirming the distinct nature of SSR in both architectures.
    *   Design Motivation: One-size-fits-all acceleration strategies inevitably fail across different architectures; strategies must be selected according to the specific form of SSR.

### Loss & Training

HalfV is a training-free inference-time acceleration method. It only requires a pre-analysis on a small amount of data (100 samples) to determine the three-stage boundaries. Evaluations were conducted on LLaVA-1.5v-7B (Vicuna), LLaVA-1.5v-7B (Mistral), and Qwen2.5-VL-7B across benchmarks including GQA, MME, POPE, SQA, and AI2D.

## Key Experimental Results

### Main Results

| Model | Method | FLOPs Speedup | Avg. Performance Retention |
| :--- | :--- | :--- | :--- |
| Qwen2.5-VL | HoloV (Prior Work) | High | Poor (Degradation 5.7-22.4%) |
| Qwen2.5-VL | **HalfV** | **4.1×** | **96.8%** |
| LLaVA-1.5v (Vicuna) | **HalfV** | High | Excellent |
| LLaVA-1.5v (Mistral) | **HalfV** | High | Excellent |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| IVR Handling Only | Moderate Speedup | Universal pruning is effective but insufficient |
| SSR Handling Only | Limited Speedup | Only addresses deep redundancy |
| IVR + SSR (Full HalfV) | Optimal | Two stages are complementary |
| Wrong SSR Strategy | Catastrophic Degradation | Validates the necessity of architecture-awareness |

### Key Findings

*   The root cause of degradation for existing methods on Qwen is the different manifestation of SSR—Qwen's deep layers remain active but extremely sparse, so layers cannot be simply skipped.
*   The start of Stage II is the optimal timing for one-time pruning (lowest marginal utility).
*   Suppressing only 1% of tokens leads to performance collapse in Stage II, confirming the high coupling of the global aggregation process.
*   The three-stage lifecycle is consistent across all tested architectures, but SSR manifestations are architecture-dependent.

## Highlights & Insights

*   **Systemic revelation and explanation of "backbone dependency"**: First to confirm that the failure of MLLM acceleration methods stems from backbone architecture differences rather than the visual frontend, providing a mechanistic explanation via matrix entropy analysis.
*   **Elegant framework for IVR/SSR decoupling**: Decomposes complex visual redundancy into two independently manageable components and provides optimal strategies for each.
*   **Locating optimal pruning timing via marginal utility**: Quantitatively determines the pruning layer rather than using heuristics, offering methodological value.

## Limitations & Future Work

*   The pre-analysis phase requires 100 samples to determine stage boundaries; different data distributions might affect boundary locations.
*   Only validated on Vicuna, Mistral, and Qwen backbones; SSR manifestations in more architectures remain unknown.
*   The top-5% ratio in the extreme token sparsity strategy might need adjustment for different tasks.
*   Lack of comparison with the latest dynamic token management methods.

## Related Work & Insights

*   **vs. HoloV/DART (Token Pruning Methods)**: These methods implicitly assume identical redundancy patterns across architectures, leading to severe degradation on Qwen. HalfV solves this via architecture-aware SSR handling.
*   **vs. ShortV (Layer-level Methods)**: ShortV assumes deep layers can be skipped, which holds for Vicuna but fails for Qwen. HalfV distinguishes between "layer inactivity" and "token sparsity" SSR modes.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ The IVR/SSR decoupling and architecture-aware analysis are original contributions; the discovery of the three-stage lifecycle has independent value.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, covering three architectures, 8 benchmarks, marginal utility analysis, and SSR cross-validation.
*   Writing Quality: ⭐⭐⭐⭐ Deep analysis and rich visualizations, though some technical descriptions are complex.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[ICLR 2026\] Visual Prompt-Agnostic Evolution](../../ICLR2026/multimodal_vlm/visual_prompt-agnostic_evolution.md)
- [\[ICML 2026\] RESTORE: Improving Visual Token Reduction via Distortion Correction for Enhanced MLLM Inference Efficiency](../../ICML2026/multimodal_vlm/improving_visual_token_reduction_via_rectifying_distortions_for_efficient_multim.md)
- [\[ACL 2026\] Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs](long_story_short_disentangling_compositionality_and_long-caption_understanding_i.md)
- [\[AAAI 2026\] Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](../../AAAI2026/multimodal_vlm/global_compression_commander_plug-and-play_inference_acceler.md)

</div>

<!-- RELATED:END -->
