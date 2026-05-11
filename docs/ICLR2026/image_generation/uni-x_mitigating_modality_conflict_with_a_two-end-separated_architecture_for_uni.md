---
title: >-
  [Paper Note] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models
description: >-
  [ICLR 2026][Image Generation][Unified Multimodal Models] Uni-X proposes an X-shaped architecture with separated ends and a shared middle to mitigate gradient conflicts between visual and textual modalities in Unified Mul…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Unified Multimodal Models"
  - "Gradient Conflict"
  - "Modality Separation"
  - "Autoregressive Generation"
  - "Image Generation and Understanding"
date: 2026-05-08
content_hash: 7cc74c9f6193adf7
---

# Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models

**Conference**: ICLR 2026
**arXiv**: [2509.24365](https://arxiv.org/abs/2509.24365)
**Code**: [https://github.com/CURRENTF/Uni-X](https://github.com/CURRENTF/Uni-X)
**Area**: Image Generation
**Keywords**: Unified Multimodal Models, Gradient Conflict, Modality Separation, Autoregressive Generation, Image Generation and Understanding

## TL;DR
Uni-X proposes an X-shaped architecture with separated ends and a shared middle to mitigate gradient conflicts between visual and textual modalities in Unified Multimodal Models (UMMs). By designating shallow and deep layers as modality-specific and sharing intermediate layers, a 3B-parameter model matches or surpasses 7B AR-UMMs on both image generation and multimodal understanding.

## Background & Motivation

**Background**: Unified Multimodal Models (UMMs) aim to support both image understanding and generation within a single framework. Autoregressive (AR) approaches tokenize visual content into a "foreign language" via VQ, yielding simple architectures but limited performance. More complex designs (e.g., MoT, AR+Diffusion hybrids, task-specific branches) improve performance at the cost of parameter sharing and scalability.

**Limitations of Prior Work**: Fully shared Transformers suffer from severe gradient conflicts during joint training. This work is the first to transfer the concept of gradient conflict from multi-task learning to UMMs, revealing that visual and textual gradients conflict sharply in both shallow and deep layers.

**Key Challenge**: The conditional entropy of image token sequences is substantially higher than that of natural language (English, German, Chinese), implying that visual sequences are inherently harder to predict and require modeling longer-range spatial dependencies. When a shared Transformer processes both low-entropy text and high-entropy visual content, shallow and deep layers are forced to reconcile conflicting low-level distributions.

**Goal**: To effectively mitigate inter-modal gradient conflicts while preserving the simplicity of a purely autoregressive architecture.

**Key Insight**: Empirical analysis of gradient conflicts reveals that conflict diminishes in intermediate layers (where abstract semantic alignment occurs), motivating a layer-wise structural design.

**Core Idea**: Shallow and deep layers handle modality-specific processing (accounting for differing low-level statistical distributions), while intermediate layers share parameters (leveraging high-level semantic alignment), forming an X-shaped separation–sharing architecture.

## Method

### Overall Architecture
Given a pretrained LLM with $L$ layers $\{\text{Layer}_t^i\}_{i=0}^{L-1}$, the layers are divided into three segments: the first $N$ and last $M$ layers are "separated layers," while the intermediate layers are "shared layers." New vision-specific layers $\{\text{Layer}_v^i\}$ are introduced in the separated segments, running in parallel with the original text layers. Visual tokens are routed to the corresponding branch via a binary mask $M_v$ during the forward pass.

### Key Designs

1. **Gradient Conflict Quantification**:

    - Function: Define and measure cross-modal gradient conflict.
    - Mechanism: Compute the cosine similarity $S_{\text{inter}}$ between text gradient $g_{\text{text}}$ and image gradient $g_{\text{img}}$, then estimate a baseline similarity $S_{\text{base}}$ via random data splits; conflict is defined as $c_g = -(S_{\text{inter}} - S_{\text{base}})$.
    - Finding: Conflict is most severe in shallow and deep layers, and weakest in intermediate layers.

2. **Two-End-Separated Architecture**:

    - Forward pass: $H_x^{l+1} = \text{Layer}_x^l(H_x^l)$ when $l < N$ or $l \geq L-M$ (separated layers); $H_x^{l+1} = [\text{Layer}_t^l(H^l)]_x$ otherwise (shared layers).
    - Key constraint: Visual and textual modalities are strictly isolated within separated blocks with no cross-modal interaction, enforcing the learning of robust unimodal representations first.
    - Design Motivation: Align model structure with modality characteristics—modality-specific processing for low-level features, shared parameters for high-level semantic fusion.

3. **Information-Theoretic Interpretation**:

    - N-gram conditional entropy analysis shows that image tokens have substantially higher entropy than natural language tokens.
    - Higher conditional entropy implies greater prediction difficulty and the need for longer-range dependencies.
    - This explains why sharing shallow/deep layers induces conflict: sequences of different complexity require different low-level processing.

### Loss & Training
Standard autoregressive cross-entropy loss: $\mathcal{L} = -\sum_{i=1}^T \log P(s_i | s_{<i})$. A VQGAN tokenizer (from Chameleon) encodes 512×512 images into 32×32 discrete tokens with a codebook of size 8192. CFG is uniformly set to 4.0.

## Key Experimental Results

### Text Performance

| Model | Params | ARC-E | ARC-C | WinoG | BoolQ | MMLU | Avg |
|------|------|-------|-------|-------|-------|------|-----|
| Chameleon | 7B | 76.1 | 46.5 | 70.4 | 81.4 | 52.1 | 65.3 |
| Liquid | 7B | 75.6 | 49.0 | 72.7 | 81.0 | 56.0 | 66.9 |
| **Uni-X** | **3B/4.5B** | **79.0** | 47.9 | 68.9 | **82.2** | **57.6** | **67.1** |

### Image Generation and Multimodal Understanding

| Model | Params | GenEval | DPG | MME | POPE | MMB | SEED |
|------|------|---------|-----|-----|------|-----|------|
| EMU3 | 8B | 66 | 80.6 | 1243.8 | 85.2 | 58.5 | 68.2 |
| Liquid | 7B | 68 | 79.8 | 1107.2 | 81.1 | — | — |
| Janus-Pro | 7B | 80 | 84.1 | — | 87.4 | 79.2 | 72.1 |
| **Uni-X** | **3B/4.5B** | **82** | 79.8 | 1158.3 | 83.6 | 59.3 | 60.2 |

### Key Findings
- With only 3B parameters, Uni-X achieves a GenEval score of 82, surpassing most 7B AR-UMMs including Chameleon (39), EMU3 (66), and Liquid (68).
- Uni-X uses no semantic encoder (CLIP/SigLIP) and relies on a purely AR architecture, demonstrating that gradient conflict is the key bottleneck limiting performance.
- Gradient conflict analysis shows that Uni-X not only avoids conflict at the two ends but also further alleviates residual conflict in the intermediate shared layers.
- Under identical training conditions, Uni-X exhibits superior training efficiency compared to fully shared baselines.

## Highlights & Insights
- **Problem identification matters more than solution design**: This work is the first to quantify gradient conflict in UMMs and trace it to information-theoretic roots (entropy discrepancy), offering an analytical framework broadly applicable to other multimodal and multi-task systems.
- **The power of simplicity**: Compared to complex designs such as MoT and AR+Diffusion hybrids, Uni-X achieves competitive performance through layer-wise partitioning alone, preserving the scalability advantages of pure AR architectures.
- **Parameter efficiency**: Matching 7B performance with 3B parameters demonstrates that architectural design can substitute for brute-force scaling.
- **Novel information-theoretic perspective**: Interpreting vision as "a harder foreign language" via conditional entropy is both intuitive and compelling.

## Limitations & Future Work
- The current approach handles only non-interleaved multimodal inputs; interleaved sequences (mixed image–text) remain unvalidated.
- The selection of $N$ and $M$ (the number of separated layers) appears to require empirical tuning, lacking principled guidance.
- Strict isolation within separated layers may limit early cross-modal information exchange.
- Dynamic or adaptive layer allocation strategies are worth exploring.

## Related Work & Insights
- **vs. Chameleon**: The fully shared architecture leads to severe gradient conflict, yielding a GenEval score of only 39 with 7B parameters; Uni-X achieves 82 with 3B.
- **vs. MoT/UniFork**: These methods mitigate conflict by increasing architectural complexity at the cost of parameter sharing; Uni-X maintains simplicity.
- **vs. Janus-Pro**: Achieves a GenEval score of 80 using an additional semantic encoder; Uni-X reaches 82 without any additional encoder.

## Supplementary Details
- Pretraining data: 72B text tokens + 65B visual tokens, sourced from CCI3-H, DCLM, Fineweb-Edu, and others.
- The SFT stage uses 3B tokens, including MiniGemini, FineVision, OpenOrca, and others.
- Ablations are conducted on Qwen2.5-1.5B and scaled to Qwen2.5-3B.
- The optimal configuration of separated layer counts $N$ and $M$ is determined through ablation studies.
- Training is accelerated with Flash Attention 2 and DeepSpeed ZeRO2.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of gradient conflict analysis, information-theoretic interpretation, and X-shaped architecture is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across text, generation, and understanding, with sufficient ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, and the reasoning from analysis to design is logically rigorous.
- Value: ⭐⭐⭐⭐ Provides practical architectural design principles for UMM development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [\[NeurIPS 2025\] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](../../NeurIPS2025/image_generation/mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)
- [\[ICLR 2026\] Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing](draw-in-mind_rebalancing_designer-painter_roles_in_unified_multimodal_models_ben.md)
- [\[ICLR 2026\] Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](../../CVPR2026/image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)

</div>

<!-- RELATED:END -->
