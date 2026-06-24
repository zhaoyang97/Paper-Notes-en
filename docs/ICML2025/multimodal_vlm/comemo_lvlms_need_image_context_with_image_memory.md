---
title: >-
  [Paper Note] CoMemo: LVLMs Need Image Context with Image Memory
description: >-
  [ICML2025][Multimodal VLM][LVLM] This work proposes CoMemo, a dual-path architecture where the Context path concatenates image tokens with text for autoregressive processing, and the Memory path utilizes cross-attention for persistent image memory. Combined with RoPE-DHR positional encoding to maintain 2D spatial awareness and alleviate long-range decay, and a three-stage training strategy to balance the dual paths, this approach comprehensively outperforms LVLM-S and LVLM-X…
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "LVLM"
  - "Dual-Path Architecture"
  - "Cross-Attention"
  - "RoPE"
  - "Dynamic High Resolution"
  - "Lost in the Middle"
date: 2026-05-08
content_hash: 569a248b0a869dcd
---

# CoMemo: LVLMs Need Image Context with Image Memory

**Conference**: ICML2025  
**arXiv**: [2506.06279](https://arxiv.org/abs/2506.06279)  
**Code**: [Project Page](https://lalbj.github.io/projects/CoMemo/)  
**Area**: Multimodal VLM  
**Keywords**: LVLM, Dual-Path Architecture, Cross-Attention, RoPE, Dynamic High Resolution, Lost in the Middle

## TL;DR
This work proposes CoMemo, a dual-path architecture where the Context path concatenates image tokens with text for autoregressive processing, and the Memory path utilizes cross-attention for persistent image memory. Combined with RoPE-DHR positional encoding to maintain 2D spatial awareness and alleviate long-range decay, and a three-stage training strategy to balance the dual paths, this approach comprehensively outperforms LVLM-S and LVLM-X under equivalent settings.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) feature two main architectures: LVLM-S (LLaVA-style, concatenating visual tokens into the text sequence for autoregressive generation) and LVLM-X (Flamingo-style, injecting visual information via cross-attention). While LVLM-S enjoys good compatibility, its performance is constrained by flaws in the LLM's attention mechanism; LVLM-X is flexible but performs worse than LVLM-S under the same settings.

**Limitations of Prior Work**: LVLM-S inherits two critical vulnerabilities of LLMs: (1) **"Lost in the middle"**—causal self-attention causes attention to cluster at the beginning and end of the sequence, systematically neglecting visual content in middle positions, which worsens with longer context; (2) **Positional encoding disaster under Dynamic High Resolution (DHR)**—1D ascending RoPE causes image patch tokens under DHR to consume numerous position IDs, triggering long-range decay and destroying the image's 2D spatial structure.

**Key Challenge**: The autoregressive processing of LVLM-S inherently "forgets" middle visual information (an intrinsic property of the causal attention training paradigm), yet the cross-attention path (LVLM-X) alone yields poor performance. Simply merging the two (e.g., NVLM-H) does not work effectively either, as the model tends to rely excessively on one path.

**Key Insight**: Through gradient and attention weight visualization, the authors find that during training, most gradient backpropagation flows to adjacent tokens and attention sinks, systematically ignoring middle tokens. This inspires the idea of "introducing an extra visual processing path immune to context position."

**Core Idea**: While retaining LLaVA-style autoregressive processing (Context path), an additional cross-attention path (Memory path) is introduced as a "forgot-proof" persistent memory for the image, with a three-stage training strategy to prevent path imbalance.

## Method

### Overall Architecture
CoMemo is built on the InternVL2 architecture, utilizing InternLM-1.8B as the LLM backbone and InternViT-300M as the image encoder. Input images yield visual tokens after passing through the encoder and projector, which are then routed to both paths: the Context path concatenates visual tokens with text tokens for standard autoregressive processing; the Memory path caches visual tokens as Key-Value (KV) pairs, allowing decoding tokens to query visual information via cross-attention in mixin layers. Both paths share the same encoder and projector.

### Key Designs

1. **Dual-path Architecture: Context + Memory**:

    - **Function**: Ensuring visual information is not forgotten over long contexts and long generations.
    - **Mechanism**: The **Context path** serves as the main processing flow, where image tokens and text tokens are concatenated and input into the LLM for standard autoregression, providing image context. The **Memory path** acts as an auxiliary flow, with mixin layers interleaved with standard Transformer layers in a 1:4 ratio. Each mixin layer contains: (a) gated cross-attention—modulated by a learnable $\text{attn\_gate}$: $h_s \leftarrow h_s + \tanh(\text{attn\_gate}) \odot \text{cross\_attn}(q=h_s, kv=h_i)$; (b) gated feed-forward network—$h_s \leftarrow h_s + \tanh(\text{ffw\_gate}) \odot \text{ffw}(h_s)$. Cross-attention retrieves visual information based on textual queries, bypassing the "lost-in-the-middle" problem of causal self-attention.
    - **Design Motivation**: Cross-attention inherently avoids the positional bias of causal attention—it retrieves visual info based on content relevance rather than position. During decoding, it requires only one-step computation and does not need to maintain visual KV caches.

2. **RoPE-DHR Positional Encoding**:

    - **Function**: Maintaining 2D spatial relationships and mitigating long-range decay under dynamic high resolution (DHR) scenarios.
    - **Mechanism**: A standard InternVL2 image uses 256 tokens; enabling DHR can expand this to 1792 tokens (a 7x increase), leading to severe positional encoding sparsification and long-range decay. The RoPE-DHR strategy works as follows: first, use standard sequential positional encoding on the thumbnail to generate baseline position IDs; then, map each patch of the high-resolution tiles to its corresponding thumbnail patch index based on its 2D coordinates: $i_{thumb} = (\lfloor x_{tile} \times \frac{W_{tile}}{W_{orig}} \rfloor + wb_{tile}, \lfloor y_{tile} \times \frac{H_{orig}}{H_{thumb}} \rfloor + hb_{tile})$. Consequently, high-resolution patches inherit the thumbnail's compact positional space.
    - **Design Motivation**: Achieve two goals—(1) Length compression: prevent sparse positional encoding under DHR from a global perspective; (2) Geometry preservation: tile patches inherit positional context from the thumbnail anchors, preserving 2D spatial relationships.

3. **Three-stage Training Strategy**:

    - **Function**: Balancing the Context and Memory dual paths and preventing the model from excessively relying on a single channel.
    - **Mechanism**: **Stage 1**—Train projector and mixin layer parameters (including gates) to learn representation alignment and dual-path balance; **Stage 2**—Freeze gate parameters and continue training to prevent over-reliance on the cross-attention path during late pre-training; **Stage 3**—Full-parameter fine-tuning to shift toward instruction following. The experiments show that pre-training steps are critical for balance: insufficient steps lead to poor alignment, whereas too many steps lead to the Memory path dominating (because the full-parameter updated Memory module learns faster than the partially updated projector).
    - **Design Motivation**: Merging the dual paths directly (like NVLM-H) makes the model rely heavily on one path in a biased way. Gate-value monitoring reveals that freezing the gates after a suitable number of pre-training steps and then continuing pre-training is key to achieving balance.

### Loss & Training
The pre-training phase adopts the standard next-token prediction loss, updating only the projector and Memory architecture parameters. The fine-tuning phase uses full-parameter training, adopting the same training data as InternVL-2. DHR information is distributed to both paths (DHR-B strategy); experiments show that single-sided allocation causes severe path bias.

## Key Experimental Results

### Fair Comparison of Three Architectures (2B Model, Same Data)

| Task Category | LVLM-X | LVLM-S (InternVL2) | CoMemo | Relative Gain |
|---------|--------|-------------------|--------|---------|
| Caption (COCO/Flickr/NoCaps) | 84.9/68.9/62.9 | 79.1/65.4/60.0 | **98.6/78.5/78.8** | +17.2% |
| Long-Gen (LLaVABench/MMDU) | 50.8/31.5 | 62.9/28.7 | **66.9/38.7** | +7.0% |
| Long-Context (MMT/MMNIAH) | -/- | 50.2/27.0 | **51.3/34.2** | +5.6% |
| Multi-Image (BLINK/Mantis) | 50.3/63.1 | 38.2/48.3 | **43.5/50.6** | Significant |
| Math (MathVista/MathVision) | 44.2/15.8 | 48.0/16.4 | **50.0/17.0** | — |

### Ablation Study on DHR Allocation Strategies

| DHR Allocation | Caption | VQA | OCR | Path Bias |
|---------|---------|-----|-----|---------|
| DHR-S (Context path only) | Medium | Low | High | Biased towards Context |
| DHR-X (Memory path only) | Low | Medium | Low | Biased towards Memory |
| **DHR-B (Dual-path)** | **High** | **High** | **High** | **Balanced** |

### Key Findings
- The key to dual-path is not simple merging but the three-stage training. Direct merging like NVLM-H is ineffective, but CoMemo achieves path balance through appropriate steps of pre-training + gate freezing + full-parameter fine-tuning.
- RoPE-DHR shows the most significant improvement on OCR and high-resolution tasks, as these tasks rely heavily on 2D spatial awareness.
- Under equivalent data/model/training settings, CoMemo consistently outperforms both LVLM-S and LVLM-X across all categories of 7 benchmarks, demonstrating the superiority of the architectural design itself.
- The Memory path requires only a single-step cross-attention computation during decoding, without maintaining visual KV caches, effectively avoiding the KV cache explosion issue under long sequences.

## Highlights & Insights
- The dual-path naming of "Context + Memory" is intuitive and inspired by cognitive science—Context provides immediate visual context while Memory provides persistent visual memory, which complement each other in human cognition.
- The three-stage training strategy is the pivotal innovation that makes the dual-path design truly work. Monitoring path bias through gate values serves as a practical debugging technique, and this methodology is generalizable to the training balance issues of any multi-path architecture.
- RoPE-DHR elegantly unifies "length compression" and "geometry preservation" within a concise framework via thumbnail index mapping.
- The analysis-driven design methodology in Section 2 is highly exemplary: diagnosing the root cause through gradient and attention analyses before proposing targeted architectural solutions.
- During the decoding stage, the Memory path only needs a single cross-attention pass with the cached visual representations, avoiding the linear growth of the visual KV cache with sequence length.
- The design of the gating mechanism (tanh gate) allows the influence of the Memory path to be automatically adjusted during training—evolving from near-zero to proper contribution, delivering smooth control.

## Limitations & Future Work
- Only verified on a 2B model; the effectiveness on larger scales (e.g., 7B/70B) remains unknown—as the balance of dual paths might change with scale.
- The cross-attention mixin layers increase parameter count and computational overhead, with the exact magnitude not detailed.
- The three-stage training increases training pipeline complexity; selecting the optimal number of pre-training steps requires careful tuning.
- Adaptation to video/extremely long visual sequences remains unexplored—could the Memory path be extended as an external writable memory?
- The thumbnail mapping in RoPE-DHR might introduce positional conflicts when the number of dynamic tiles is large; this impact is not fully analyzed.
- Only training data from InternVL-2 was used; performance on larger-scale datasets awaits validation.

## Related Work & Insights
- **vs NVLM-H (Dai et al. 2024)**: NVLM-H allocates DHR only to the cross-attention path, causing a biased learning preference. CoMemo fully resolves this issue through DHR-B dual-path allocation and three-stage training.
- **vs mPLUG-Owl3**: mPLUG-Owl3 also employs cross-attention mixin layers, but lacks a training strategy to balance the dual paths, and lacks the positional encoding improvements of RoPE-DHR.
- Insight: The Memory path could be extended to a RAG-style writable external memory to continuously update visual memory in multi-turn dialogues.
- Insight: Controlling the timing of gate freezing during three-stage training could be generalized as a universal stabilization tool for training multi-module architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ Three innovative elements: dual-path, three-stage training, and RoPE-DHR.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 7 benchmark categories, with highly rigorous controlled-variable comparisons among the three architectures.
- Writing Quality: ⭐⭐⭐⭐ The analysis-driven writing style of the design rationale (Section 2) is highly exemplary.
- Value: ⭐⭐⭐⭐ Offers clear guiding significance for LVLM architecture design.
- Overall: ⭐⭐⭐⭐ A pioneering work on dual-path visual processing; the three-stage training strategy holds significant methodological value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](../../CVPR2025/multimodal_vlm/cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [\[CVPR 2026\] Adapting In-context Generation for Enhanced Composed Image Retrieval](../../CVPR2026/multimodal_vlm/adapting_in-context_generation_for_enhanced_composed_image_retrieval.md)
- [\[ICML 2025\] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration](towards_rationale-answer_alignment_of_lvlms_via_self-rationale_calibration.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/multimodal_vlm/vladva_discriminative_fine-tuning_of_lvlms.md)

</div>

<!-- RELATED:END -->
