---
title: >-
  [Paper Note] FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference
description: >-
  [ACL 2025][Information Retrieval & RAG][Retrieval-Augmented Generation] To address the inference efficiency issues in Retrieval-Augmented Language Models (RALMs) caused by the repeated recomputation of KV cache due to prepending retrieved content, this paper proposes FlashBack. FlashBack appends retrieved content to preserve the input's KV cache, and utilizes Marking Tokens + LoRA fine-tuning to adapt to the new context pattern, achieving up to a 4x inference speedup on Llama…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "KV Cache"
  - "Inference Acceleration"
  - "LoRA"
  - "Context Pattern"
date: 2026-05-08
content_hash: 5a72b4bd1309877d
---

# FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference

**Conference**: ACL 2025  
**arXiv**: [2405.04065](https://arxiv.org/abs/2405.04065)  
**Code**: [https://github.com/BIT-NLP-GROUP/FlashBack](https://github.com/BIT-NLP-GROUP/FlashBack)  
**Area**: Information Retrieval  
**Keywords**: Retrieval-Augmented Generation, KV Cache, Inference Acceleration, LoRA, Context Pattern

## TL;DR
To address the inference efficiency issues in Retrieval-Augmented Language Models (RALMs) caused by the repeated recomputation of KV cache due to prepending retrieved content, this paper proposes FlashBack. FlashBack appends retrieved content to preserve the input's KV cache, and utilizes Marking Tokens + LoRA fine-tuning to adapt to the new context pattern, achieving up to a 4x inference speedup on Llama 2-7B while maintaining comparable perplexity.

## Background & Motivation

**Background**: RALMs enhance the generation capabilities of LLMs by prepending retrieval results from external corpora to the input. The mainstream practice (such as In-Context RALM) prepends retrieved content to the input and performs retrieval once every $s$ tokens.

**Limitations of Prior Work**: Since the retrieved content changes after each retrieval step and is located at the front of the input, the KV cache for the entire context must be discarded and recomputed. As the input length $T$ increases, the FLOPs for recomputation grow quadratically at $O(T^2)$, drastically slowing down inference speed.

**Key Challenge**: Retrieved content needs to be frequently updated to maintain relevance, but prepending causes full KV cache recomputation upon every update, making it difficult to balance efficiency and quality.

**Goal**: Design a RALM context pattern that avoids recomputing the KV cache of the input when retrieved content changes, thereby significantly accelerating inference.

**Key Insight**: Shift the retrieval content from prepending (before the input) to appending (after the input) so that the input's KV cache is unaffected by retrieval updates. However, directly appending disrupts semantic coherence and degrades performance, which is mitigated by introducing Marking Tokens and LoRA fine-tuning.

**Core Idea**: Change the retrieved content from prepended to appended to reuse the KV cache, and employ Marking Tokens + LoRA to compensate for the performance loss incurred by switching context patterns.

## Method

### Overall Architecture
FlashBack is a modular RALM framework that contains a retriever (BM25 or DPR) and a reader (LLM). During inference, retrieved content is appended to the end of the input, and Marking Tokens are used to bound the retrieved content. During the fine-tuning phase, only the Marking Token embeddings and LoRA parameters are trained, while the weights of the LLM and retriever remain frozen.

### Key Designs

1. **Appending Context Pattern**:

    - **Function**: Append retrieved content to the end of the input instead of prepending it.
    - **Mechanism**: The standard RALM probability is $p(x_i | [\mathcal{R}_\mathcal{C}(x_{<i}); x_{<i}])$ (with retrieved content in front). FlashBack changes this to $p(x_i | [x_{<i}; \mathcal{R}_\mathcal{C}(q_j^{(s,\ell)})])$ (with input in front). Thus, the KV cache of the input $x_{<i}$ does not need to be recomputed when retrieved content is updated, needing only to compute the retrieved content part.
    - **Design Motivation**: Under the prepending pattern, the FLOPs for KV cache recomputation are $C_0 = \frac{2T(T+s)bh^2l}{s}$, which grows quadratically with $T$. Under the appending pattern, only the retrieved document part needs to be recomputed, significantly reducing FLOPs.

2. **Marking Token**:

    - **Function**: Two special prompt tokens `<MARK_L>` and `<MARK_R>` marking the left and right boundaries of the retrieved content.
    - **Mechanism**: These tokens are added to the model vocabulary, and their embedding vectors are trainable. During fine-tuning and inference, these markers let the model "know" which part of the context is the retrieved external content.
    - **Design Motivation**: Direct appending of retrieved content destroys semantic coherence (experiments show PPL spikes from ~16 to ~81). Marking Tokens help the model distinguish between the original input and the auxiliary retrieval content, which is key to aligning with the appending pattern.

3. **LoRA Fine-tuning Strategy**:

    - **Function**: Use LoRA to perform parameter-efficient fine-tuning on attention layers to adapt to the appending pattern.
    - **Mechanism**: Freeze the original weights of the LLM and the retriever, only training the LoRA weights (applied to K, V projection matrices) and the Marking Token embeddings. The extra FLOPs for LoRA are $C_1 = \frac{2l(4r+1)bhT(d+2s)}{s}$, where $r$ is the LoRA rank. Since $r \ll h$, this overhead is highly controllable.
    - **Design Motivation**: PEFT avoids the high cost of full fine-tuning and reduces catastrophic forgetting, preserving the original capabilities of the LLM.

### Loss & Training
- Standard language modeling loss (next token prediction).
- Fine-tuning data formatted using the appending context structure.

## Key Experimental Results

### Main Results (OPT-6.7B Perplexity $\downarrow$)

| Configuration | WikiText-2 | Arxiv | Freelaw | StackExchange |
|------|-----------|-------|---------|---------------|
| No retrieval | 12.30 | 7.74 | 6.94 | 6.22 |
| Prepend + LoRA + MT | 8.24 | 6.99 | 6.18 | 5.58 |
| Append (naive) | 68.31 | 46.53 | 48.33 | 40.25 |
| Append + LoRA | 10.54 | 10.92 | 9.27 | 8.04 |
| **Append + LoRA + MT (FlashBack)** | **8.59** | **7.43** | **6.64** | **5.94** |

FlashBack's PPL is close to the prepending upper bound (difference < 0.5), while increasing inference speed by up to 4x.

### Inference Acceleration

| Model | Speedup | Description |
|------|--------|------|
| Llama 2-7B | Up to **4×** | Speedup is most prominent on long sequences (3968 tokens) |
| OPT-6.7B | ~3× | Speedup is more significant on larger models (more layers, larger hidden size) |
| OPT-125M | ~1.5× | Speedup is limited for smaller models |

### Key Findings
- **Marking Token is key to performance recovery**: append + LoRA (PPL 10.54) vs append + LoRA + MT (PPL 8.59). MT contributes to a recovery of around 2 PPL points.
- **Speedup is more pronounced with longer sequences**: This aligns with the $O(T^2)$ vs $O(T)$ expectation from FLOPs analysis.
- **Speedup is more obvious on larger models**: This is because the overhead of recomputing KV cache per layer increases with the number of layers and hidden size.

## Highlights & Insights
- **Systematic analysis of Prepend vs Append**: The paper provides a thorough discussion from theoretical FLOPs deduction to practical runtime experiments, fully demonstrating the efficiency bottlenecks of prepending and the advantages of appending. This analysis framework can be used to evaluate other RALM variants.
- **Simple and effective Marking Token**: Introducing only two extra tokens allows the model to adapt to a completely new context pattern. This serves as a lightweight context format alignment method that can be transferred to other scenarios requiring marker tokens for context types.
- **Modular Design**: FlashBack places no constraints on the retriever (BM25, DPR, etc., are plug-and-play) and only requires lightweight LoRA fine-tuning for the LLM, making it highly practical.

## Limitations & Future Work
- **Information flow constraints of the appending pattern**: Under causal attention, the appended retrieved content can only be attended to by subsequent tokens and cannot influence previous tokens. This fundamentally restricts the retrieved content's capacity to assist in comprehending prior inputs.
- **PPL Gap Remains**: FlashBack (8.59) vs Prepend+LoRA+MT (8.24) shows that the efficiency of information utilization in the appending pattern is slightly inferior to the prepending pattern.
- **Evaluation limited to Perplexity**: Lacks end-to-end evaluation on downstream tasks (such as QA and summarization).
- **Fixed retrieval frequency**: Retrieval once every $s$ tokens is not highly flexible.
- Potential Improvements: (a) Integrate contextual information flow enhancements using bidirectional attention (specifically on the retrieved portion); (b) Introduce adaptive retrieval frequencies.

## Related Work & Insights
- **vs In-Context RALM (Ram et al., 2023)**: Both use a frozen LLM + frozen retriever. However, In-Context RALM prepends content, leading to KV cache recomputation, while FlashBack solves the efficiency issue through appending + Marking Tokens.
- **vs RETRO (Borgeaud et al., 2022)**: RETRO requires pre-training LLMs from scratch to integrate retrieved content, which is highly expensive. FlashBack only requires lightweight LoRA fine-tuning.
- **vs GRIT-LM**: GRIT-LM improves efficiency by reusing embeddings but is limited to embedding models. FlashBack is a general text-based approach.

## Rating
- Novelty: ⭐⭐⭐⭐ The core idea of shifting prepending to appending is simple, but the introduction of Marking Tokens and theoretical analysis makes the overall system complete.
- Experimental Thoroughness: ⭐⭐⭐ Runtime and PPL experiments are conducted, but downstream task evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ The FLOPs analysis is clear, and the illustrations are intuitive.
- Value: ⭐⭐⭐⭐ Provides a practical solution for the inference efficiency of RALMs, and the 4x speedup is highly practical in engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference](../../ICLR2026/information_retrieval/raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference.md)
- [\[ICML 2025\] RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding](../../ICML2025/information_retrieval/rapid_long-context_inference_with_retrieval-augmented_speculative_decoding.md)
- [\[ACL 2025\] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)

</div>

<!-- RELATED:END -->
