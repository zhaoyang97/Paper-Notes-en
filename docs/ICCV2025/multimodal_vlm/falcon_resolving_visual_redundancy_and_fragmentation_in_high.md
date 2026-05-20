---
title: >-
  [Paper Note] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers
description: >-
  [ICCV 2025][Multimodal VLM][Visual Registers] This paper proposes FALCON, which introduces learnable Visual Registers into the ViT encoder. Through the ReCompact mechanism…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Visual Registers"
  - "Token Compression"
  - "Visual Redundancy"
  - "Visual Fragmentation"
  - "High-resolution MLLM"
date: 2026-05-08
content_hash: 1f8a927e0c66b0f1
---

# FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers

**Conference**: ICCV 2025
**arXiv**: [2501.16297](https://arxiv.org/abs/2501.16297)  
**Code**: [Available](https://github.com/JiuTian-VL/JiuTian-FALCON)  
**Area**: Multimodal VLM / High-resolution Understanding
**Keywords**: Visual Registers, Token Compression, Visual Redundancy, Visual Fragmentation, High-resolution MLLM

## TL;DR

This paper proposes FALCON, which introduces learnable Visual Registers into the ViT encoder. Through the ReCompact mechanism, visual redundancy is eliminated directly during the encoding stage (achieving 9× token compression), while the ReAtten module resolves visual fragmentation caused by image cropping via inter-register interactions.

## Background & Motivation

High-resolution MLLMs commonly adopt cropping-based strategies: high-resolution images are divided into multiple sub-images matching the encoder's pre-training resolution, independently encoded, and then concatenated. This introduces two core problems:

1. **Visual Redundancy**: Token counts grow sharply with resolution (e.g., 16 sub-images × 576 tokens = 9,216 tokens), and a large proportion of tokens from background regions are redundant, significantly increasing the computational burden on the LLM. Existing compression methods (pooling / QFormer / Abstractor) either yield limited gains or require massive pre-training data.

2. **Visual Fragmentation**: Independent encoding of sub-images disrupts semantic coherence. A canonical example is "pineapple" being split into "pine" + "apple," causing OCR errors; similarly, the Rubin vase illusion fails to be recognized after cropping.

## Method

### Overall Architecture

FALCON is built upon SigLIP-L/16-384px (visual encoder) and Llama-3.1-8B-Instruct (LLM), with core innovations concentrated in the visual encoding stage:

1. Shape-adaptive cropping → obtaining $N_c$ sub-images + a global thumbnail
2. Image tokens $I_k$ of each sub-image are concatenated with shared learnable visual registers $R = \{r_1, \ldots, r_M\}$ and fed into the ViT
3. At each ViT layer, standard self-attention (ReCompact) is first performed, followed by cross-sub-image register interaction (ReAtten)
4. Only the $M$ register features per sub-image are retained from the output and projected via MLP before being sent to the LLM

With $M = 64$ (vs. original image token count $N = 576$), the compression ratio is $576/64 = 9\times$.

### Key Designs

1. **ReCompact: Register-based Representation Compression**

    **Function**: $M \ll N$ learnable visual registers are introduced alongside image tokens into the ViT. Through self-attention, the registers adaptively aggregate key information from the image tokens; only the register outputs are retained after encoding.

    **Mechanism**: The ViT self-attention formula $\hat{X}_{k,l} = \text{Softmax}(\frac{X_{k,l} X_{k,l}^T}{\sqrt{D_{key}}}) X_{k,l}$ is applied without any attention masking, leveraging the pre-trained ViT's inherent capacity to aggregate global information into specific tokens (an observation established in prior work).

    **Design Motivation**: Compared to query-based cross-attention schemes such as QFormer and Abstractor, ReCompact directly reuses pre-trained ViT parameters, requiring far less adaptation data (<3M samples vs. 129M for QFormer and 400M for Abstractor).

2. **ReAtten: Register Interaction Attention**

    **Function**: After self-attention at each ViT layer, the register hidden states from all sub-images $\hat{X}_l^R \in \mathbb{R}^{M \cdot N_c \times D}$ are collected and allowed to interact via Cross-ViT-Attention: $\bar{X}_l^R = \hat{X}_l^R + \text{Cross-ViT-Atten}(\hat{X}_l^R)$. The updated registers are then recombined with their respective image tokens before the FFN.

    **Mechanism**: The compactness of the registers ($M \cdot N_c \ll N \cdot N_c$) makes global information exchange across the full image computationally feasible. Cross-ViT-Atten parameters are initialized from the same-layer ViT self-attention weights to ensure a smooth start.

    **Design Motivation**: Concatenating all sub-image tokens for global attention incurs prohibitive quadratic complexity; Shifted Window Attention is limited to local interactions and provides insufficient cross-region context; the Complementary Image Pyramid (CIP) introduces additional redundancy.

### Loss & Training

**Four-stage progressive training**:

- **Stage 0** (Static alignment, low resolution): ViT and LLM are frozen; only registers and the MLP projection are trained using image caption data.
- **Stage 1** (Coarse alignment, low resolution): All parameters are unfrozen; high-quality long descriptions and full-text OCR data are used, enabling registers to learn coarse-to-fine information capture.
- **Stage 2** (Fine alignment, high resolution): High-resolution inputs are introduced; fine-grained description and OCR tasks (region captioning, text localization) are used.
- **Stage 3** (Instruction tuning, high resolution): ViT is frozen; the ReAtten module is introduced; multi-task instruction data is used for fine-tuning.

## Key Experimental Results

### Main Results

**MME-RealWorld Benchmark (High-resolution Understanding)**:

| Model | Tokens/Sub-image | Perception Avg-C | Reasoning Avg-C |
|-------|-----------------|-----------------|----------------|
| MiniCPM-V 2.5 | 96×9 | 44.0 | 36.0 |
| Monkey | 256×4 | 36.3 | 28.8 |
| GPT-4o | - | 41.9 | 42.3 |
| Claude 3.5 Sonnet | - | 47.7 | 49.2 |
| LLaVA-OneVision | 729×9 | 55.8 | 44.2 |
| **FALCON** | **64×16** | **50.3** | **43.4** |

Using only 64 tokens per sub-image (9× compression), FALCON surpasses GPT-4o, GPT-4o-mini, and Gemini-1.5-pro on both perception and reasoning, and matches LLaVA-OneVision (which uses 729×9 tokens) on the reasoning dimension.

### Ablation Study

**Compression Method Comparison (MME-RealWorld Avg-C Total)**:

| Compression Method | Avg-C (Perception + Reasoning) |
|-------------------|-------------------------------|
| Pooling | Lower |
| Pixel Shuffle | Medium |
| Abstractor | Below Pooling |
| **ReCompact** | **Highest** |

**Visual Continuity Method Comparison**:

| Method | V*_Avg | MME-RW Perception | MME-RW Reasoning | POPE |
|--------|--------|------------------|-----------------|------|
| Baseline (no continuity) | 51.3 | 38.2 | 35.6 | 85.7 |
| CIP | 50.3 | 38.4 | 37.2 | 86.3 |
| W-Atten | 60.2 | 41.0 | 38.1 | 86.4 |
| **ReAtten** | **61.3** | **42.1** | **39.0** | **87.3** |

ReAtten achieves the best performance across all metrics and effectively reduces hallucination (POPE 87.3).

### Key Findings

- **Register count ablation**: Performance improves consistently from 36→64→144 registers with diminishing returns; 64 registers represents the optimal efficiency–performance trade-off (144 registers incur 2.41× training time with marginal gains).
- **Attention visualization**: Each register attends to specific image regions (e.g., faces, text) and largely ignores backgrounds, confirming effective redundancy elimination.
- **Fragmentation visualization**: Without ReAtten, attention patterns across sub-images are highly fragmented; with ReAtten, attention becomes spatially coherent, demonstrating successful global information exchange.
- **Same-data comparison**: When trained on exactly the same data as LLaVA-v1.5, FALCON outperforms it on multiple benchmarks including SQA (68.9 vs. 66.8), POPE (87.5 vs. 85.9), and MMB (66.0 vs. 64.3).

## Highlights & Insights

- The core innovation is elegant and concise: no additional compression modules are required; the register mechanism within the ViT simultaneously addresses both redundancy and fragmentation.
- Data efficiency is a notable advantage: compared to QFormer/Abstractor requiring hundreds of millions of pre-training samples, FALCON requires only <3M samples for adaptation.
- The four-stage progressive training is well-designed, ensuring smooth transition of registers from low-resolution to high-resolution settings.

## Limitations & Future Work

- The fixed number of 64 registers may not suit all scenarios—over-allocated for simple images and under-allocated for complex ones.
- Dynamic register allocation (e.g., adjusting count based on image complexity) remains unexplored.
- ReAtten is applied at every ViT layer, which still incurs computational overhead under extreme numbers of sub-images.
- Comparison with recent dynamic resolution methods (e.g., NaViT) is absent.

## Related Work & Insights

- The paper builds upon the established observation that register tokens in ViT aggregate global information (Darcet et al., 2024), extending this from a mere "attention map artifact fix" to a functional representation compression tool.
- ReAtten is contrasted against TextMonkey's Shifted Window Attention and MiniMonkey's CIP, demonstrating genuinely global interaction.
- A promising direction inspired by this work is applying the register technique to temporal redundancy elimination in video MLLMs.

## Rating

⭐⭐⭐⭐ The method is concise and effective. Surpassing GPT-4o-level commercial models under 9× token compression is impressive. The design motivation is clearly articulated and well-supported by thorough ablation studies, making this an excellent contribution to high-resolution MLLM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICCV 2025\] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i.md)
- [\[AAAI 2026\] Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](../../AAAI2026/multimodal_vlm/global_compression_commander_plug-and-play_inference_acceler.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
