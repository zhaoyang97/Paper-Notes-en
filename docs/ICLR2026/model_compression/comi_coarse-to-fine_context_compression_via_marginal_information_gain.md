---
title: >-
  [Paper Note] COMI: Coarse-to-fine Context Compression via Marginal Information Gain
description: >-
  [ICLR2026][Model Compression][context compression] This paper proposes COMI, a coarse-to-fine adaptive context compression framework based on Marginal Information Gain (MIG = query relevance − semantic redundancy). At a…
tags:
  - "ICLR2026"
  - "Model Compression"
  - "context compression"
  - "token merging"
  - "long context"
  - "marginal information gain"
  - "RAG"
date: 2026-05-08
content_hash: 54271d560d0d4021
---

# COMI: Coarse-to-fine Context Compression via Marginal Information Gain

**Conference**: ICLR2026  
**arXiv**: [2602.01719](https://arxiv.org/abs/2602.01719)  
**Code**: [https://github.com/Twilightaaa/COMI](https://github.com/Twilightaaa/COMI)  
**Area**: Model Compression  
**Keywords**: context compression, token merging, long context, marginal information gain, RAG

## TL;DR
This paper proposes COMI, a coarse-to-fine adaptive context compression framework based on Marginal Information Gain (MIG = query relevance − semantic redundancy). At a 32× compression ratio, COMI improves NaturalQuestions EM by approximately 25 points over the second-best method, with the core insight being the joint optimization of relevance and diversity among retained information.

## Background & Motivation

**Background**: Techniques such as RAG increase LLM input length, leading to high computational costs and information redundancy. Context compression methods fall into two categories: task-agnostic (global compression without considering the query) and task-aware (retaining query-relevant content).

**Limitations of Prior Work**: (a) Task-agnostic methods ignore the query and inevitably lose relevant information under high compression ratios; (b) Task-aware methods rely solely on "relevance" as the compression criterion — retained tokens tend to be highly similar (redundant). Empirically, only 0.75% of tokens account for 99% of attention weights, and the cosine similarity among these tokens exceeds 0.6; (c) High redundancy may mislead LLMs into generating incorrect outputs ("relevant does not imply correct").

**Key Challenge**: Retaining tokens purely by relevance preserves large amounts of "relevant but redundant" content, resulting in insufficient diversity after compression. Existing dynamic compression rate allocation schemes either apply fixed linear rules or rely solely on relevance, none of which account for semantic redundancy.

**Goal**: To simultaneously optimize relevance and diversity under high compression ratios — retaining information that is both query-relevant and mutually non-redundant.

**Key Insight**: Define Marginal Information Gain MIG = relevance − redundancy as a unified metric to guide coarse-grained group budget allocation and fine-grained token merging.

**Core Idea**: Use MIG (query cosine similarity minus the maximum similarity to other tokens) to drive a two-stage compression pipeline — dynamic compression rate allocation across groups, followed by MIG-weighted token merging within groups.

## Method

### Overall Architecture
Input context $X$ + query $Q$ → Encoder → Two-stage compression: (1) Coarse-grained group budget reallocation (based on inter-group MIG) → (2) Fine-grained intra-group token merging (based on intra-group MIG) → LSA cross-layer semantic alignment → Decoder generates the answer.

### Key Designs

1. **Marginal Information Gain (MIG)**:

    - Definition: $G(x_i, q, X) = \frac{x_i^\top q}{\|x_i\|\|q\|} - \max_{j \neq i} \frac{x_i^\top x_j}{\|x_i\|\|x_j\|}$
    - First term: cosine similarity between the token and the query (relevance)
    - Second term: cosine similarity between the token and its most similar counterpart in the context (redundancy)
    - Interpretation: High MIG = query-relevant and semantically unique; Low MIG = irrelevant or highly redundant
    - Design Motivation: Relevance alone leads to retention of large amounts of repetitive content. MIG jointly models relevance and uniqueness, and it is theoretically shown that expected performance under MIG-based selection exceeds that of relevance-only selection.

2. **Coarse-grained Group Budget Reallocation**:

    - Function: Divides the context into equal-length segments and dynamically allocates compression ratios based on segment-level MIG.
    - Mechanism: For each segment, select a representative vector (the token with the highest cosine similarity to the query) → compute segment MIG → apply inverse-rank softmax to assign weights $P_i = e^{-G_i} / \sum e^{-G_j}$ → segments with higher MIG receive larger budgets (lower compression ratios).
    - Design Motivation: Information value is unevenly distributed across the context — segments that are both relevant and distinctive should be allocated greater retention budgets.

3. **Fine-grained Token Merging**:

    - Function: Within each group, MIG-weighted merging combines multiple tokens into a single compressed token.
    - Mechanism: $\tilde{h}_i = \sum_{h_k \in S_i} \frac{e^{G(h_k, \bar{q}, S_i)}}{\sum e^{G}} \cdot h_k$, where tokens with higher MIG contribute more.
    - Design Motivation: Avoids the information loss of simple uniform averaging. MIG weighting ensures that semantically unique and query-relevant tokens dominate the merged representation.

### Loss & Training
Built on an Encoder-Decoder architecture. The Encoder and LSA are fully fine-tuned; the Decoder fine-tunes only $W_Q, W_K, W_V, W_O$. Training objective: standard cross-entropy loss $\mathcal{L}_{nll}$, predicting correct answers from compressed representations.

## Key Experimental Results

### Main Results (32× Compression, Qwen2-7B)

| Method | NQ EM | 2WikiMQA EM | HotpotQA EM | NarrativeQA EM | MultiNews F1 |
|------|-------|-------------|-------------|----------------|-------------|
| Original Prompt | 72.35 | 59.78 | 64.17 | 24.53 | 31.42 |
| SnapKV | 6.06 | 8.09 | 13.61 | 0.00 | 25.56 |
| Activation Beacon | 11.67 | 36.53 | 39.73 | 6.63 | 33.60 |
| LongLLMLingua | 15.07 | 21.80 | 21.58 | 1.03 | - |
| GMSA | - | - | - | - | - |
| **COMI** | **~40** | **~45** | **~42** | **~10** | **~30** |

### Ablation Study (NQ, LLaMA-2-7B, 16×)

| Configuration | EM | Note |
|------|-----|------|
| COMI (full) | 22.75 | Complete method |
| w/o MIG (relevance only) | ↓ significant | Redundancy accumulation |
| w/o coarse-grained reallocation | ↓ moderate | Fixed compression ratio |
| w/o fine-grained MIG weighting | ↓ moderate | Uniform merging |

### Key Findings
- **At 32× compression, COMI outperforms the second-best method by ~25 EM** (NQ + Qwen2): the advantage is particularly pronounced at extreme compression ratios — when only 3% of tokens can be retained, information diversity becomes critical.
- **At 16× compression, COMI approaches original prompt performance**: on LLaMA-2-7B, COMI achieves 22.75 EM vs. 15.04 EM for the original input — compressed representations even surpass the uncompressed baseline by eliminating redundant noise.
- **The redundancy penalty in MIG is the key contribution**: removing it leads to significant performance degradation, confirming that "relevant but redundant" content is the core bottleneck at high compression ratios.
- **Consistent effectiveness across backbones**: COMI achieves substantial improvements on both LLaMA-2-7B and Qwen2-7B.

## Highlights & Insights
- **MIG = relevance − redundancy is a concise and principled metric**: a single formula unifies two core dimensions of information retention. This design principle is transferable to any information selection scenario (e.g., RAG retrieval, KV-cache management).
- **Hierarchical coarse-to-fine two-stage strategy**: segment-level budget allocation (macro decisions) followed by token-level merging (micro decisions) — dual-level MIG guidance ensures optimality at both global and local granularities.
- **Counter-intuitive finding that compressed representations outperform the original**: this suggests that much of the content in long texts constitutes noise — effective compression is equivalent to denoising.

## Limitations & Future Work
- **Requires an Encoder-Decoder architecture**: not directly applicable to KV-cache compression in Decoder-only models. Integrating the MIG principle into KV-cache management warrants further exploration.
- **MIG uses maximum single-token similarity as the redundancy measure**: this may be insufficient — a group of moderately similar tokens can also produce redundancy. Diversity metrics such as DPP or MMR may offer more accurate characterization.
- **Training overhead**: fine-tuning both the Encoder and Decoder is required; training-free deployment is not supported. In contrast, methods such as LLMLingua require no training.
- **Not validated on very long texts**: experiments use inputs of approximately 4–8K tokens. Performance on truly long-context scenarios (100K+ tokens) remains unknown.

## Related Work & Insights
- **vs. LLMLingua/LongLLMLingua**: These methods select tokens based on perplexity/entropy (discarding less important ones) but do not account for redundancy among retained tokens. COMI's MIG jointly considers relevance and uniqueness.
- **vs. GMSA**: GMSA addresses cross-layer semantic alignment in Encoder-Decoder frameworks; COMI inherits the LSA solution and innovates on the compression strategy.
- **vs. Activation Beacon/StreamLLM**: These are KV-cache compression methods tied to specific model architectures. COMI operates at the prompt level and is model-agnostic.

## Rating
- Novelty: ⭐⭐⭐⭐ The MIG formulation is concise and intuitively well-motivated; the coarse-to-fine framework is carefully designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four QA benchmarks plus one summarization task, two backbone models, multiple compression ratios, and comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated (the finding that 0.75% of tokens account for 99% of attention is particularly compelling).
- Value: ⭐⭐⭐⭐ Provides an effective solution for high compression ratios; MIG is transferable to other information selection scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](../../CVPR2026/model_compression/hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](../../AAAI2026/model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ICCV 2025\] Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP](../../ICCV2025/model_compression/gain-mlp_improving_hdr_gain_map_encoding_via_a_lightweight_mlp.md)
- [\[ICLR 2026\] Modality-free Graph In-context Alignment](modality-free_graph_in-context_alignment.md)

</div>

<!-- RELATED:END -->
