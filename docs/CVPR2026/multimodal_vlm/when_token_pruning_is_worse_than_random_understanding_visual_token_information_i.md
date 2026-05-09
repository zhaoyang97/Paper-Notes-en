---
title: >-
  [Paper Note] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs
description: >-
  [CVPR 2026][Multimodal VLM][token pruning] This paper identifies a phenomenon in which existing token pruning methods underperform random pruning in deep layers of VLLMs, proposes a method for quantifying visual token information based on changes in output probability, and reveals the "Information Horizon"—a critical layer at which visual token information uniformly dissipates to zero. The position of this horizon is dynamically influenced by task visual complexity and model capability. The paper further demonstrates that simply integrating random pruning can effectively improve existing methods.
tags:
  - CVPR 2026
  - Multimodal VLM
  - token pruning
  - information horizon
  - visual token informativeness
  - random pruning
  - VLM inference acceleration
date: 2026-05-08
content_hash: 438ef42da6a13464
---

# When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs

**Conference**: CVPR 2026
**arXiv**: [2512.07580](https://arxiv.org/abs/2512.07580)
**Code**: [https://github.com/YahongWang1/Information-Horizon](https://github.com/YahongWang1/Information-Horizon)
**Area**: Multimodal VLM
**Keywords**: token pruning, information horizon, visual token informativeness, random pruning, VLM inference acceleration

## TL;DR
This paper identifies a phenomenon in which existing token pruning methods underperform random pruning in deep layers of VLLMs, proposes a method for quantifying visual token information based on changes in output probability, and reveals the "Information Horizon"—a critical layer at which visual token information uniformly dissipates to zero. The position of this horizon is dynamically influenced by task visual complexity and model capability. The paper further demonstrates that simply integrating random pruning can effectively improve existing methods.

## Background & Motivation
**Background**: VLLMs encode images into large numbers of visual tokens (576 for LLaVA-1.5; up to thousands for Qwen2.5-VL). Training-free token pruning is the mainstream acceleration paradigm, comprising importance-based methods (FastV/SparseVLM) and diversity-based methods (DivPrune/DART).

**Limitations of Prior Work**: A key observation is that in deep decoder layers (e.g., beyond layer 20), all existing pruning methods perform no better than—or even worse than—random pruning. This phenomenon consistently appears after layers 16–20 on LLaVA-1.5-7B and after layer 21 on Qwen2.5-VL-7B.

**Key Challenge**: Regardless of whether attention or similarity is used as the selection criterion, existing pruning methods fail to identify more informative tokens than random selection in deep layers—indicating that visual token information has "dissipated" at depth.

**Goal**: (a) Why does deep-layer pruning underperform random pruning? (b) How does visual token information evolve across layers? (c) At which layer can all visual tokens be safely removed? (d) How can these findings be used to improve existing methods?

**Key Insight**: Define and quantify the informativeness of individual visual tokens at specific layers, and track their cross-layer variation patterns.

**Core Idea**: Visual token information uniformly dissipates to zero in deep layers (the "Information Horizon"); beyond this layer, pruning is equivalent to random selection, and integrating random pruning can improve existing methods.

## Method

### Overall Architecture
(1) Define visual token information based on changes in output probability upon removing a single token; (2) Analyze the cross-layer distribution of information and identify the "Information Horizon" phenomenon, where information variance decays to zero; (3) Investigate factors governing the horizon's position—task complexity and model capability; (4) Application: integrate random pruning to improve existing methods.

### Key Designs

1. **Visual Token Information Definition**:

    - Function: Quantify the informational contribution of an individual visual token at a specific decoder layer.
    - Mechanism: At layer $i$, retain only the target token $\mathbf{V}_k$ (removing all other visual tokens) and perform a forward pass to obtain probability $p_k$; then remove all visual tokens to obtain the text-only probability $p_{text}$; information is defined as the difference $I_i(\mathbf{V}_k) = p_k - p_{text}$.
    - Design Motivation: Directly measures the causal effect of token removal on the output probability of the ground-truth label; isolating the target token eliminates interference from other visual tokens.
    - Validation: Consistently removing low-information tokens improves model performance (MME +27.8%), indicating that low-information tokens act as distractors.

2. **Information Horizon**:

    - Function: Identify the critical layer at which visual token information uniformly dissipates to zero.
    - Core Finding: From shallow to deep layers, token informativeness transitions from high variance to uniform distribution, ultimately converging to zero at a certain layer (the Information Horizon). Beyond this layer, removing all visual tokens does not affect performance.
    - The horizon for MME on LLaVA-1.5-7B is approximately layer 16; for TextVQA it is approximately layer 24.
    - This explains why deep-layer pruning underperforms random pruning: once all token informativeness approaches zero, no selection criterion has a usable signal.

3. **Dynamic Nature of the Information Horizon**:

    - **Effect of task visual complexity**: Knowledge-based QA and hallucination detection have shallow horizons (Qwen2.5-VL: ~layer 20), while visually intensive tasks such as OCR have deeper horizons (~layer 27).
    - **Effect of model visual capability**: Qwen2.5-VL (stronger) exhibits a deeper horizon than LLaVA-1.5 (weaker), enabling utilization of visual tokens at greater depth.

4. **Random Pruning Integration Strategy**:

    - Function: Augment existing pruning methods with random pruning in deep layers.
    - Mechanism: Since token information is uniform in deep layers, no selection criterion offers an advantage; direct random pruning suffices.
    - Effectiveness: DivPrune+Random retains 96.9% of original performance on Qwen2.5-VL-7B while pruning 50% of tokens.

## Key Experimental Results

### Main Results — Qwen2.5-VL-7B, 50% Token Pruning

| Method | MME | TextVQA | MMB | OCRBench | Avg. | Rel.(%) |
|--------|-----|---------|-----|----------|------|---------|
| Original | 2313 | 85.4 | 79.8 | 88.5 | 83.6 | 100.0 |
| DART | 2295 | 82.1 | 79.6 | 75.5 | 77.3 | 92.7 |
| DivPrune | 2291 | 83.1 | 79.4 | 84.1 | 80.7 | 96.7 |
| DART+Random | 2318 | 82.7 | 79.6 | 77.9 | 78.3 | 93.9 |
| **DivPrune+Random** | **2302** | **83.4** | **79.5** | **85.3** | **80.9** | **96.9** |

### Ablation Study — Information Quantification Validity (LLaVA-1.5-7B)

| Operation | MME Change | TextVQA Change |
|-----------|-----------|----------------|
| Remove 75% low-information tokens @ layer 10 | **+27.8%** | **+6.1%** |
| Remove 88% low-information tokens @ layer 10 | Still above baseline | Still above baseline |

### Key Findings
- **Information dissipation is universal**: Observed on both LLaVA-1.5 and Qwen2.5-VL, independent of model architecture.
- In shallow layers (1–7), pruning methods effectively retain high-information tokens; diversity-based methods outperform importance-based methods.
- In deep layers (after layer 14), all methods degrade to random-level performance as information variance approaches zero.
- Removing low-information tokens actually improves performance, indicating that such tokens act as distractors rather than benign noise.
- The improvement from +Random is most pronounced on OCRBench (DART: 75.5→77.9), since OCR tasks have deeper horizons with exploitable information remaining in deep layers.

## Highlights & Insights
- **Counter-intuitive finding that "pruning underperforms random"**: This phenomenon is mechanistically explained through rigorous experiments and information quantification, elevating an empirical observation to actionable theory.
- **Information Horizon concept**: Provides a concise and practical framework for understanding the lifecycle of visual tokens in VLLMs—information generation → propagation → dissipation.
- **Simple yet effective improvement strategy**: The +Random augmentation improves existing methods at virtually zero additional cost, offering high practical value.
- **Task–Model–Horizon triangle**: Reveals the dynamic mechanism by which visual complexity and model capability jointly determine the effective depth of visual token utility.

## Limitations & Future Work
- The information definition requires ground-truth labels and cannot be applied directly at inference time.
- The information metric requires additional forward passes (per-token removal), imposing substantial computational overhead.
- The precise location of the Information Horizon must be measured separately for each task/model/sample; no predictive model is provided.
- Only two models are evaluated: LLaVA-1.5 and Qwen2.5-VL.
- The +Random strategy, while effective, lacks theoretical guarantees and essentially amounts to "abandoning fine-grained selection" after information dissipation.

## Related Work & Insights
- **vs. FastV/SparseVLM/DART**: Rather than proposing a new pruning method, this paper explains from an information perspective why existing methods fail in deep layers, and provides a simple remedy.
- **vs. EmbedLens**: Complementary relationship—EmbedLens identifies sink/dead/alive token categories from a representational structure perspective, while this paper reveals deep-layer information dissipation from an information quantification perspective; together, both support the conclusion that "the genuine contribution of visual tokens to MLLMs is concentrated in shallow-to-middle layers."
- **Practical application**: The +Random strategy can be directly stacked on top of any existing method.

## Rating
- Novelty: ⭐⭐⭐⭐ The Information Horizon concept is novel; the information quantification method is direct and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis across multiple models × multiple benchmarks × multiple pruning methods.
- Writing Quality: ⭐⭐⭐⭐ The logical chain of observation → hypothesis → validation → application is clear.
- Value: ⭐⭐⭐⭐ Provides important theoretical understanding for token pruning research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[ICLR 2026\] IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](../../ICLR2026/multimodal_vlm/ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)
- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)

<!-- RELATED:END -->
