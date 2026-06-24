---
title: >-
  [Paper Note] SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator
description: >-
  [ICML 2025][Model Compression][Sparse Attention] SepLLM is proposed, leveraging the property of separator tokens (such as punctuation) to naturally compress text segment information, by only retaining the KV cache of Initial + Separator + Neighboring tokens. This significantly reduces attention computation and memory usage while maintaining performance.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Sparse Attention"
  - "Separator Compression"
  - "KV Cache"
  - "Streaming Inference"
  - "Large Language Models"
date: 2026-05-08
content_hash: f508c3ee794f654e
---

# SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator

**Conference**: ICML 2025  
**arXiv**: [2412.12094](https://arxiv.org/abs/2412.12094)  
**Code**: [Project Homepage](https://sepllm.github.io)  
**Area**: Signal Communication  
**Keywords**: Sparse Attention, Separator Compression, KV Cache, Streaming Inference, Large Language Models

## TL;DR

SepLLM is proposed, leveraging the property of separator tokens (such as punctuation) to naturally compress text segment information, by only retaining the KV cache of Initial + Separator + Neighboring tokens. This significantly reduces attention computation and memory usage while maintaining performance.

## Background & Motivation

**Background**: The quadratic attention complexity of Transformers and the linearly growing KV cache represent key bottlenecks in long-sequence inference. StreamingLLM identified the attention sink phenomenon, proving that retaining initial tokens alongside a sliding window can sustain streaming inference capabilities. However, this training-free scheme discards out-of-window information, leading to massive performance degradation on tasks requiring long-range dependencies.

**Limitations of Prior Work**: Existing KV cache eviction methods (e.g., H2O, SnapKV, PyramidKV) rely on top-$k$ selection based on attention scores, which incurs non-trivial computational overhead and suffers from selection policies inconsistent with pre-training distributions. While StreamingLLM is simple, it completely loses out-of-window information, causing severe performance degradation in tasks like mathematical reasoning (GSM8K, degrading from 77.79% to 69.67%).

**Key Challenge**: The need to drastically compress the KV cache for accelerated inference, while simultaneously preserving critical semantic information of the sequence. Sliding window approaches cover local contexts but forfeit global dependencies, whereas full attention retains all information but is computationally prohibitive.

**Goal**: To identify a sparse attention pattern intermediate between full attention and sliding windows, capable of preserving the vast majority of textual semantic information at a fraction of the cost of a 100% KV cache.

**Key Insight**: It is observed that attention weights exhibit pronounced clustering on separator tokens (e.g., periods, commas, spaces). These separators naturally act as "compression anchors" for text segments—the model encodes key information into the hidden states of subsequent separators after processing a segment. Consequently, retaining only the KV of separators can efficiently recover segment-level information.

**Core Idea**: Separator tokens serve as natural compression nodes of text; preserving their KV is equivalent to maintaining segment-level semantic coverage over the entire sequence at an extremely low cost.

## Method

### Overall Architecture

SepLLM categorizes tokens in a sequence into three groups: Initial tokens (the first $k$ tokens, serving as attention sinks), Separator tokens (9 types of punctuation and whitespace characters: ".", ",", "?", "!", ";", ":", " ", "\t", "\n"), and Neighboring tokens (tokens within a sliding window of the current position). Attention computation is restricted to the KV of these three types of tokens, while the KV of the remaining tokens is discarded. This design supports both training-free deployment and training-from-scratch modes. In streaming inference scenarios, a 4-buffer rotation strategy is further introduced to support infinite-length inputs.

### Key Designs

1. **Three-Category Token Sparse Attention**:

    - **Function**: Determining which tokens' KV should be retained.
    - **Mechanism**: For each query token $q_i$, its attention range is restricted to $\mathcal{S}(i) = \mathcal{I} \cup \mathcal{P}(i) \cup \mathcal{N}(i)$, corresponding to the set of initial tokens, the set of all separator tokens preceding position $i$, and a local window of size $n$, respectively. Positions outside this attention mask are padded with $-\infty$.
    - **Design Motivation**: Initial tokens resolve the attention sink issue (as validated by StreamingLLM), Neighboring tokens cover local context, and Separator tokens represent the core innovation—they cover segment-level information of the entire historical sequence through sparse distribution at a cost much lower than full attention.

2. **4-Buffer Strategy for Streaming Inference**:

    - **Function**: Supporting online inference for infinite-length sequences.
    - **Mechanism**: The KV cache is organized into four blocks: the Initial Block (fixed size, storing the first $k$ tokens), the Separator Block (dynamic size, storing all historical separators), the Local Window Block (fixed size $n$, storing recent tokens), and the Past Window Block (fixed size $s$, storing recent tokens that overflowed from the Local Window). If the Separator Block becomes excessively large, it can be further constrained to the most recent $m$ separators.
    - **Design Motivation**: Directly storing all historical separators in ultra-long sequences still leads to memory growth. The hierarchical design of the 4-buffer system ensures a deterministic upper bound on memory usage, while the Past Window serves as a buffer to avoid information fragmentation at sliding window boundaries.

3. **Position Shift**:

    - **Function**: Correcting the position embedding discontinuity introduced by sparse attention.
    - **Mechanism**: Since non-separator tokens are skipped, gaps exist between the actual position IDs of the retained tokens. Position shifting re-indexes the position IDs of the retained tokens into a continuous sequence, ensuring that position encodings like RoPE function correctly.
    - **Design Motivation**: Ablation experiments show that omitting position shifting triggers a catastrophic degradation of PPL from 13.1 to 192.7, proving that position continuity is critical to sparse attention schemes.

### Loss & Training

The training-from-scratch mode employs the standard autoregressive cross-entropy loss, but attention computation is restricted to the three categories of tokens, thereby reducing training FLOPs to only 71.77% of a standard Transformer. The training-free mode directly applies the sparse attention mask to pre-trained models without any parameter updates. The paper also proves a universal approximation theorem (Theorem 5.1): SepLLM can approximate any continuous seq2seq function under a minimal configuration of $H=2, d_h=1, d_f=4$.

## Key Experimental Results

### Main Results

| Method | GSM8K-CoT (%) | KV Ratio | MMLU (%) | KV Ratio |
|:---:|:---:|:---:|:---:|:---:|
| Vanilla (Llama-3-8B) | 77.79 | 100% | 65.29 | 100% |
| StreamingLLM ($n$=256) | 69.67 | 26% | 62.33 | 37.73% |
| H2O | 76.27 | — | — | — |
| SnapKV | 76.50 | — | — | — |
| PyramidKV | 75.82 | — | — | — |
| **SepLLM** ($n$=256) | **77.18** | 47.36% | **64.68** | 44.61% |

| Model (Train from scratch) | ARC-Easy | PIQA | SciQ | FLOPs |
|:---:|:---:|:---:|:---:|:---:|
| Vanilla Pythia-160M | 46.80 | 62.84 | 81.50 | 100% |
| SepLLM Pythia-160M ($n$=128) | **47.35** | **64.64** | **82.60** | 71.77% |

### Ablation Study

| Configuration | PPL (WikiText) | Change |
|:---:|:---:|:---:|
| SepLLM Full | 13.1 | — |
| W/o position shift | 192.7 | +179.6 (catastrophic degradation) |
| W/o Initial tokens | 14.9 | +1.8 |
| Use all 9 separators | 13.1 (GSM8K 77.18%) | Best |
| Use only 4 separators | — (GSM8K 76.68%) | -0.50 |
| Use only 2 separators | — (GSM8K 70.66%) | -6.52 |

| Streaming Inference (PG19, 4M tokens) | PPL |
|:---:|:---:|
| StreamingLLM ($s$=64) | 36.1 |
| SepLLM ($s$=64) | **33.9** |

### Key Findings

- SepLLM achieves 77.18% on GSM8K using only 47.36% of KV (compared to vanilla's 77.79%), significantly outperforming StreamingLLM's 69.67%, and also outperforming H2O, SnapKV, and PyramidKV, which require computing attention scores.
- When trained from scratch, SepLLM slightly outperforms the vanilla Transformer on various downstream tasks while saving 28.23% of FLOPs and reducing training time by roughly 26%.
- Position shifting is an indispensable component (omitting it yields a 15-fold surge in PPL), and the variety of separator types correlates positively with performance (9 types > 4 types > 2 types).
- Under streaming inference scenarios, SepLLM consistently maintains a lower PPL over ultra-long texts than StreamingLLM.

## Highlights & Insights

- The core insight is highly intuitive: punctuation marks act naturally as boundaries of meaning segments in human language, making it instinctive for LLMs to exploit them to encode context information.
- The method is remarkably simple to implement—essentially requiring only a specialized attention mask, with no need for auxiliary learnable parameters or intricate selection strategies.
- Although the proof of the universal approximation theorem assumes a minimal configuration, it provides a theoretical lower-bound guarantee for the representation capacity of sparse attention.
- The 4-buffer streaming inference design is thoroughly conceived, demonstrating practical utility for real-world engineering deployment.

## Limitations & Future Work

- The definition of separators is hard-coded to 9 characters, which may necessitate customization for domains like programming code or mathematical formulas, where separator patterns differ.
- While a KV ratio of 47.36% is halved, it is not an extreme compression scheme and still lags behind aggressive methods like StreamingLLM (26%), though the latter exhibits significantly higher performance degradation.
- The assumptions for the universal approximation theorem ($H=2, d_h=1$) are overly idealized, deviating substantially from realistic model configurations and thus restricting its theoretical impact.
- Evaluation in practical deployment scenarios, such as multi-turn conversations and RAG, remains unaddressed, where separator distributions may differ from key pre-training text corporas.
- The definition of separators may need extension in multilingual scenarios (e.g., Chinese period "。", Japanese period "。", etc.).

## Related Work & Insights

- **vs StreamingLLM**: Both exploit the attention sink phenomenon. However, while StreamingLLM retains solely in-window information, SepLLM preserves global segment-level information via separator tokens, incurring a slightly higher KV ratio but significantly better performance retention.
- **vs H2O/SnapKV/PyramidKV**: These methods retrieve important tokens based on attention scores, which require computing the full attention beforehand. In contrast, SepLLM's separator selection is predefined, eliminating extra computational overhead.
- **vs Longformer/BigBird**: Classic sparse attention architectures employ global tokens, local sliding windows, and random connections. SepLLM's separator tokens can be viewed as an elegant, natural language-driven global token selection strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ The insight of separator compression is neat and elegant, transforming linguistic intuition into technical design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both training-free and training-from-scratch modes, validates across multiple benchmarks, and includes exhaustive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-organized experiments, and a decent depth in the theoretical section.
- Value: ⭐⭐⭐⭐ Simple yet effective approach with a clear engineering deployment path, making a practical contribution to accelerating LLM inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Weak-to-Strong Jailbreaking on Large Language Models](weak-to-strong_jailbreaking_on_large_language_models.md)
- [\[ICML 2025\] Persistent Topological Features in Large Language Models](persistent_topological_features_in_large_language_models.md)
- [\[ICML 2025\] DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [\[NeurIPS 2025\] Uni-LoRA: One Vector is All You Need](../../NeurIPS2025/model_compression/uni-lora_one_vector_is_all_you_need.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](../../NeurIPS2025/model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)

</div>

<!-- RELATED:END -->
