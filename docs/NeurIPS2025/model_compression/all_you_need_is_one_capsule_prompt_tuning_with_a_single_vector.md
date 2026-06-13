---
title: >-
  [Paper Note] All You Need is One: Capsule Prompt Tuning with a Single Vector
description: >-
  [NeurIPS 2025][Model Compression][prompt tuning] This paper proposes Capsule Prompt-Tuning (CaPT), identifying that existing task-aware soft prompts exhibit minimal interaction with input tokens — an "attention island" p…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "prompt tuning"
  - "PEFT"
  - "capsule prompt"
  - "attention anchor"
  - "parameter efficiency"
date: 2026-05-08
content_hash: 76e443b06a59ec53
---

# All You Need is One: Capsule Prompt Tuning with a Single Vector

**Conference**: NeurIPS 2025
**arXiv**: [2510.16670](https://arxiv.org/abs/2510.16670)  
**Code**: None  
**Area**: Model Compression
**Keywords**: prompt tuning, PEFT, capsule prompt, attention anchor, parameter efficiency

## TL;DR
This paper proposes Capsule Prompt-Tuning (CaPT), identifying that existing task-aware soft prompts exhibit minimal interaction with input tokens — an "attention island" phenomenon. Incorporating instance-aware information into a single capsule prompt enables it to serve as an "attention anchor" that activates attention toward critical structural information, achieving superior performance over multi-prompt methods with extremely few parameters (e.g., only 0.003% of parameters on Llama3.2-1B).

## Background & Motivation

**Background**: Prompt-based learning is a mainstream parameter-efficient fine-tuning (PEFT) approach that prepends learnable soft prompts to the input to adapt LLMs to downstream tasks. Existing methods typically require grid search for the optimal prompt length and often rely on a large number of prompts.

**Limitations of Prior Work**: (1) Existing soft prompts are task-aware (identical across all instances) and lack instance-aware information, limiting adaptability to diverse inputs. (2) A key preliminary finding: task-specific soft prompts fail to interact strongly with input tokens — they predominantly attend to each other while largely ignoring critical tokens in the input.

**Key Challenge**: Although soft prompts are designed as "instructions" to guide generation, at the attention level they form a self-enclosed cluster that does not effectively interact with the input content.

**Goal**: How can prompts be made to genuinely interact with the input? Is it possible to achieve better performance with a single prompt?

**Key Insight**: The discovery of the "attention anchor" phenomenon — placing instance-aware tokens at the beginning of the sequence preserves strong attention toward critical structural information and enables active interaction with all input tokens.

**Core Idea**: Replace multiple purely task-aware prompts with a single capsule prompt that encodes both instance-aware and task-aware information, serving as an attention anchor to drive input interaction.

## Method

### Overall Architecture
Input → Extract instance-aware semantics (e.g., CLS token representation) → Combine with task-aware parameters to form a single capsule prompt → Prepend to the sequence → Standard inference / fine-tuning.

### Key Designs

1. **Attention Anchor Discovery**:

    - **Function**: Reveals the special role of instance-aware tokens placed at the beginning of the sequence.
    - **Mechanism**: Positioning instance-aware tokens earliest in the sequence preserves strong attention toward critical structural information, causing all subsequent tokens to attend heavily to this anchor.
    - **Design Motivation**: Explains why purely task-aware prompts are limited — they lack informational connection to specific input instances.

2. **Capsule Prompt Design**:

    - **Function**: Fuses instance-aware and task-aware information into a single vector.
    - **Mechanism**: Leverages an off-the-shelf instance semantic representation (e.g., the encoder's CLS token) and combines it with a single learnable task-aware vector. The method is nearly parameter-free, requiring only one vector to be learned.
    - **Design Motivation**: Achieves extreme parameter efficiency — a single prompt suffices.

### Loss & Training
Standard task loss (e.g., cross-entropy for classification); only the capsule prompt parameters are trained.

## Key Experimental Results

### Main Results

| Method | T5-Large Avg Acc | Parameter Ratio |
|--------|-----------------|-----------------|
| Full Fine-tuning | High but parameter-heavy | 100% |
| Prompt Tuning (multi-prompt) | ~80% | ~0.1% |
| **CaPT (1 prompt)** | **84.03%** | **0.003%** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| w/o instance-aware | Degenerates to standard prompt tuning |
| Multiple capsule prompts | No improvement over a single one — one suffices |
| Attention visualization | CaPT prompt interacts significantly more with input tokens than standard prompts |

### Key Findings
- **A single capsule prompt outperforms multi-prompt methods**: fewer parameters, better performance.
- **The attention anchor effect is real**: attention visualizations show high attention weights from all input tokens toward the capsule prompt when it is placed at the beginning of the sequence.
- **Instance-aware information is critical**: removing it leads to significant performance degradation.

## Highlights & Insights
- The discovery of the **"attention island"** is highly insightful, revealing a fundamental limitation of soft prompts.
- **Extreme parameter efficiency**: 0.003% parameter ratio is remarkable.
- The **attention anchor concept** is transferable to other scenarios requiring guided attention.

## Limitations & Future Work
- Validation on the largest-scale LLMs (e.g., 70B+) is absent.
- The extraction of instance-aware representations warrants further exploration; the current use of CLS tokens is relatively simple.

## Related Work & Insights
- **vs. Prompt Tuning**: Standard prompt tuning uses multiple task-aware prompts; CaPT uses a single capsule prompt.
- **vs. LoRA**: A different PEFT paradigm; CaPT achieves lower parameter counts but may be less general than LoRA.

## Rating
- Novelty: ⭐⭐⭐⭐ — the attention anchor finding is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — validated across multiple tasks and models
- Writing Quality: ⭐⭐⭐⭐ — clear and well-organized
- Value: ⭐⭐⭐⭐ — an extreme parameter-efficient PEFT solution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uni-LoRA: One Vector is All You Need](uni-lora_one_vector_is_all_you_need.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](graph_your_own_prompt.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](one-step_diffusion-based_image_compression_with_semantic_distillation.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)

</div>

<!-- RELATED:END -->
