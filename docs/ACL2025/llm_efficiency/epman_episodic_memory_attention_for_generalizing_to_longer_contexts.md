---
title: >-
  [Paper Note] EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts
description: >-
  [ACL 2025][LLM Efficiency][long context] This paper proposes the EpMAN method, which estimates the relative relevance of context chunks through an episodic memory module, uses this relevance to re-weight the decoder's self-attention (differentiating attention), and combines it with noisy training and attention range expansion strategies. It achieves stronger and more robust performance than long-context LLMs and RAG in the 16k-256k context length range.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "long context"
  - "episodic memory"
  - "attention"
  - "context length extension"
  - "retrieval-augmented generation"
date: 2026-05-08
content_hash: a1bb7ce6cbf8ca02
---

# EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts

**Conference**: ACL 2025  
**Code**: -  
**Area**: Others  
**Keywords**: long context, episodic memory, attention, context length extension, retrieval-augmented generation  

## TL;DR

This paper proposes the EpMAN method, which estimates the relative relevance of context chunks through an episodic memory module, uses this relevance to re-weight the decoder's self-attention (differentiating attention), and combines it with noisy training and attention range expansion strategies. It achieves stronger and more robust performance than long-context LLMs and RAG in the 16k-256k context length range.

## Background & Motivation

- Generalizing LLMs to long context inputs remains a major challenge.
- **Limitations of Prior Work**:
  1. **Continued pre-training** on long sequences: Huge computational overhead with $O(n^2)$ self-attention complexity.
  2. **Position embedding extrapolation** (PI, RoPE scaling): Requires extra fine-tuning, with limited effectiveness.
  3. **Sparse attention / Sliding window attention**: May lose critical information.
  4. **RAG**: Conflict between retrieval model and LLM parametric memory; retrieval noise leads to hallucinations or ignoring context.
- **Three major challenges in long contexts**:
    - **Recency Bias**: LLMs tend to bias towards the end of the context.
    - **Distractor information impact**: Irrelevant documents degrade model accuracy.
    - **Attention Dilution**: Softmax normalization leads to diluted attention.
- **Goal / Inspiration**: Inspired by Kahneman's Dual-System Theory—self-attention is a fast, intuitive "System 1", while EpMAN simulates the slow, analytical "System 2".

## Method

### Overall Architecture: Dual-layer Attention

1. **Episodic Memory Layer**: Stores long context into learning blocks (chunks), estimating the relevance of each chunk relative to the query.
2. **Self-Attention Layer**: Re-weights standard self-attention using episodic attention $a_{mem}$.

### Episodic Memory Operations

- **Write Operation**: Divides context into fixed-size chunks (256 tokens/chunk), encodes them using a pre-trained retriever (Dragon), and stores them in memory.
- **Read Operation**: Computes episodic attention $a_{mem}$ using the cosine similarity between the query encoding and the chunk encodings.
- Episodic memory simultaneously stores the KV cache of chunks (stored in CPU memory to handle large-scale contexts).

### Differentiating Attention

Multiplies standard attention by episodic attention:

$$a_{epman} = \text{softmax}\left(\frac{qK^T}{\sqrt{d_z}}\right)(V * a_{mem})$$

- $a_{mem}$ is a chunk-level weight, broadcasted to token level.
- Effect: Amplifies attention on relevant chunks and suppresses irrelevant chunks.

### Training Data

**Pre-training Data**:
- Generates synthetic paragraphs using Mixtral-8x22B.
- Appends Wikipedia distractor paragraphs to increase context length.
- Uses next-token prediction + episodic attention loss.

**Synthetic QA Data**:
- Topic-sampled: Teacher model generates paragraphs and QA pairs.
- Wikipedia-based: Generates QAs based on Wikipedia paragraphs.
- **Hard negative mining**: Mines Wikipedia paragraphs that are topically similar to the relevant chunk but irrelevant as strong distractors.

Training Configuration: episode size = 16 (1 relevant + 15 distractors), chunk = 256 tokens, effective training context = 4K tokens.

### Noisy Training

- **Problem**: Training with fixed $a_{mem}$ weights makes the model overfit to the pattern of "highest weight = most relevant"; during OOD (out-of-distribution), the retriever might rank the relevant chunk lower.
- **Solution**: Top-K chunks receive random weights between 0.9 and 1.0 + randomized permutation order.
- Provides a denoising target, enabling the decoder to find relevant information even when the retriever is inaccurate.

### Loss & Training

$$L = \mathbb{E}_\mathcal{D} [\alpha \ln p(l|q, C) + \ln(a|q, C, a_{mem})]$$

- First term: Episodic attention loss (cross-entropy, $\alpha = 0.1$).
- Second term: Next-token prediction loss.

### BroadAttn: Attention Range Extension during Inference

- **NarrowAttn**: Focuses only on top-K chunks.
- **BroadAttn**: Extends attention to the neighboring chunks of each top-K chunk.
- Resolves the information truncation issue (e.g., "Albert Einstein was born in Germany" is in one chunk, while "He taught himself algebra" is in an adjacent chunk).
- Preserves the original context sequence order of chunks.

## Experiments

### Experimental Setup

- Decoder: Mistral-7B-Instruct-v0.2 (LoRA fine-tuned)
- Retriever: Dragon (multi-turn context encoder)
- Evaluation Tasks: Needle-in-a-Haystack (Paul Graham / PG19), FactRecall-en, MultifieldQA, LoogleSD
- Uses the most challenging setup of the LV-Eval framework (containing both CFI confusing information + KPR keyword substitution)

### Main Results

**Needle-in-a-Haystack (16k-128k)**:
- EpMAN achieves near-perfect recall (99-100%) on both Paul Graham and PG19 data sources.
- Comparison: Mistral-7B scores only 25.4% at 128k; Phi-3-128k scores only 26.4% at 16k.
- Dragon + Mistral improves the performance but is far inferior to EpMAN.

**FactRecall-en (16k-256k, LV-Eval CFI+KPR)**:

| Method | 16k | 32k | 64k | 128k | 256k | Mean |
|------|-----|-----|-----|------|------|------|
| Mistral-7B | 65.3 | 72.5 | 41.0 | 22.5 | 11.5 | 42.6 |
| Phi-3-128k | 82.0 | 80.5 | 81.0 | 63.0 | 34.5 | 68.2 |
| Dragon+Mistral | 74.2 | 71.8 | 66.0 | 77.2 | 69.0 | 71.7 |
| EpMAN (noisy+BroadAttn) | **81.8** | **75.2** | **76.0** | **75.2** | **80.2** | **77.7** |

- EpMAN maintains 80.2% at 256k, whereas Phi-3-128k plummets to 34.5%.

**MultifieldQA (LLM-as-Judge)**:
- EpMAN (noisy+BroadAttn) scores a mean of 74.3, surpassing Dragon+Mistral (69.7) and Phi-3 (42.6).

**LoogleSD**:
- EpMAN (uniform+BroadAttn) scores a mean of 78.6, outperforming Dragon+Mistral (77.4).

### Ablation Study & Key Findings

1. **Noisy training vs. Uniform training**: Noisy training significantly outperforms uniform training on FactRecall and MultifieldQA (77.7 vs. 75.1).
2. **BroadAttn vs. NarrowAttn vs. Exact**: BroadAttn is consistently optimal as it resolves the information truncation issue.
3. **Exact inference performs poorly**: Because under the CFI+KPR setting, the Dragon retriever cannot always rank the relevant chunks correctly.
4. For LoogleSD (Wikipedia-sourced data), since both the retriever and training data originate from Wikipedia, uniform training is sufficient.

## Highlights & Insights

1. **Clever Application of Dual-System Theory**: Self-attention = System 1 (fast but inaccurate), episodic memory attention = System 2 (slow but accurate).
2. **Noisy Training Provides Denoising Target**: Crucial for OOD (out-of-distribution) generalization, enabling the model to tolerate ranking errors from the retriever.
3. **BroadAttn Inference Strategy**: Resolves the chunk boundary information truncation issue with extremely low cost.
4. Generalizing to 256K when only trained on 4K tokens—extremely high training efficiency.
5. Excellent performance even under challenging setups with confusing information and keyword substitution.

## Limitations & Future Work

- Requires storing the complete KV cache (currently on CPU), introducing heavy memory overhead for ultra-large documents.
- Larger top-K values increase training memory requirements.
- The optimal combination of uniform/noisy training and exact/narrow/broad attention depends on the nature of tasks.
- Only validated on Mistral-7B, without testing on larger models.
- The quality of the retriever (Dragon) has a significant impact on overall performance.

## Related Work & Insights

- **Long-context LLMs**: Phi-3 (Abdin et al., 2024), Position Interpolation (Chen et al., 2023), Sparse Attention (Child et al., 2019).
- **RAG**: Dragon retriever (Lin et al., 2023); attention distillation to improve RAG (Li et al., 2024b).
- **Memory-augmented LLMs**: Larimar (Das et al., 2024) with only top-1 readout is unsuitable for scattered information; kNN-LM (Wu et al., 2022) mixes attention using a learnable gate.
- **Attention Denoising**: Differential Transformer (Ye et al., 2024) uses noise cancellation to reduce attention on irrelevant tokens.
- **Attention Dilution**: "Lost in the Middle" problem (Liu et al., 2024).

## Rating ⭐⭐⭐⭐

Cleverly combines episodic memory with self-attention, enabling highly efficient training (generalizing from 4K to 256K) and showing outstanding performance under the hardest experimental settings. The approach has clear intuition and thorough evaluation, though scalability (KV cache memory) and dependency on the retriever remain bottlenecks for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](smarter_better_faster_longer_a_modern_bidirectional_encoder_for_fast_memory_effi.md)
- [\[ICLR 2026\] Out of the Memory Barrier: A Highly Memory-Efficient Training System for LLMs with Million-Token Contexts](../../ICLR2026/llm_efficiency/out_of_the_memory_barrier_a_highly_memory-efficient_training_system_for_llms_wit.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention Layer for Training on Outrageously Large Contexts](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_layer_for_training_on_outrageous.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ACL 2025\] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)

</div>

<!-- RELATED:END -->
