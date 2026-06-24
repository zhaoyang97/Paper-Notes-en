---
title: >-
  [Paper Note] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention
description: >-
  [ACL 2025][LLM Efficiency][Many-Shot ICL] Proposes Dynamic Block-Sparse Attention (DBSA), a training-free inference framework that achieves near-fine-tuning inference latency in many-shot in-context learning through structured block-sparse attention encoding and dynamic Retrieval-based KV cache while maintaining >95% of the accuracy of the best-performing methods.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Many-Shot ICL"
  - "Block-Sparse Attention"
  - "KV Cache"
  - "Retrieval ICL"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: 17feaeafccdf18eb
---

# Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention

**Conference**: ACL 2025  
**arXiv**: [2503.08640](https://arxiv.org/abs/2503.08640)  
**Code**: [millix19/dbsa](https://github.com/millix19/dbsa)  
**Area**: LLM Efficiency / In-Context Learning  
**Keywords**: Many-Shot ICL, Block-Sparse Attention, KV Cache, Retrieval ICL, Inference Efficiency  

## TL;DR

Proposes Dynamic Block-Sparse Attention (DBSA), a training-free inference framework that achieves near-fine-tuning inference latency in many-shot in-context learning through structured block-sparse attention encoding and dynamic Retrieval-based KV cache while maintaining >95% of the accuracy of the best-performing methods.

---

## Background & Motivation

### Background
Many-Shot In-Context Learning (Many-Shot ICL) utilizes thousands of demonstration examples, achieving performance comparable to or even surpassing fine-tuning without requiring parameter updates, being easily adaptable to new tasks, and eliminating the need to deploy dedicated models for each task. However, scaling ICL to long contexts introduces new computational trade-offs:

- **Fixed ICL**: Uses a fixed set of demonstrations for all queries. It can cache KV states, but its performance is inferior to Retrieval ICL.
- **Retrieval ICL**: Dynamically retrieves relevant demonstrations for each query, offering better performance but requiring context re-encoding for every inference, which is highly expensive.
- **Fine-tuning**: Has the highest setup cost (requires training) but the lowest inference overhead.

### Key Challenge
Many-Shot ICL shifts the computational burden from training to inference. The inference cost of processing thousands of demonstrations is orders of magnitude higher than zero-shot inference, rendering it impractical for high-throughput applications.

### Goal
Design a framework to simultaneously achieve the high accuracy of Retrieval ICL and the low latency of Fixed (cached) ICL, making Many-Shot ICL feasible for actual deployment.

---

## Method

### Overall Architecture
DBSA formulates Many-Shot ICL as a two-stage process:
1. **Stage 1 (One-time Pre-encoding)**: Encodes the entire demonstration pool using block-sparse attention.
2. **Stage 2 (Dynamic Inference)**: Retrieves relevant demonstration blocks for each test query and reuses the pre-encoded KV cache.

### Key Designs

#### Stage 1: Block-Sparse Streaming Attention Encoding

Given a demonstration pool $D = \{d_1, d_2, \dots, d_n\}$:
1. Divide $D$ into $n/k$ blocks: $D = [b_1, b_2, \dots, b_{n/k}]$, where each block contains $k$ demonstrations.
2. Each block $b_i$ only attends to three regions:
    - **Anchor block** $b_1$ (attention sink)
    - **Previous $j$ blocks** $\{b_{i-j}, \dots, b_{i-1}\}$ (local context)
    - **Itself** (standard causal attention)
3. Use sequential position encoding $[0, \dots, n-1]$, but cache the KV states before applying Rotary Position Embeddings (RoPE).
4. Key advantage: New demonstrations can be incrementally appended at any time, requiring the encoding of only one additional block.

The implementation utilizes Flex Attention, bypassing computation for masked blocks to achieve a speedup proportional to the sparsity.

#### Stage 2: Dynamic Demonstration Selection and Answer Generation

Given a test query $q^*$:
1. Use a retrieval method (such as BM25) to select a subset of demonstration blocks $D' = \{b'_1, b'_2, \dots, b'_m\}$, where $m < n$.
2. The anchor block $b_1$ is always included (serving as an attention sink).
3. Concatenate the KV caches of selected blocks, and re-apply Rotary Position Embeddings using new sequential position IDs $[0, |D'|-1]$.
4. The test query attends to the selected KV cache via full attention, and generates the answer autoregressively.

DBSA supports integrating plug-and-play retrieval methods (text similarity, cosine similarity, diversity retrievers, etc.); this paper utilizes BM25.

### Configuration Parameters
- 50 demonstrations per block
- Random grouping
- Each block attends to the previous 2 blocks as local context
- Retrieval ratio of 30%

---

## Experiments

### Experimental Setup
- **Models**: Llama-2-7B (32k), Llama-3.1-8B (128k)
- **Datasets**: 5 classification datasets — TREC (6 classes), TREC-Fine (50 classes), NLU (68 classes), Banking-77 (77 classes), Clinic-150 (151 classes)
- **Context Length**: 30k and 90k tokens
- **Baselines**: Fixed ICL (cached KV), Retrieval ICL (re-encoded every time), LoRA fine-tuning
- **Hardware**: L40S (48GB) for 30k, A100 (80GB) for 90k

### Efficiency Comparison

**Relative Latency (compared to the RetICL baseline)**:

| Method | 30k Setup | 30k Inference | 90k Setup | 90k Inference |
|------|-----------|---------------|-----------|---------------|
| RetICL | 1x | 1x | 1x | 1x |
| Fixed ICL | 5x | 0.11x | 6.5x | 0.06x |
| LoRA Fine-tuning | >600x | 0.08x | >1500x | 0.046x |
| **DBSA** | **3x** | **0.10x** | **4x** | **0.053x** |

*(Using Llama-3.1-8B as an example)*

Key Observations:
- DBSA dynamic inference latency is close to that of the fine-tuned model, while its setup time is less than 1/375 of fine-tuning.
- In scenarios with >100,000 requests, DBSA remains the most efficient solution.
- Grouped-Query Attention (GQA) significantly reduces cached ICL latency; the inference latency of DBSA on Llama-3.1 is only 5.3% of RetICL.

### Accuracy Comparison

**90k context, Llama-3.1-8B**:

| Dataset | Fixed ICL | Ret ICL | Fine-tuning | DBSA |
|--------|-----------|---------|------|------|
| TREC | 0.96 | 0.95 | 0.96 | 0.95 (99%) |
| TREC Fine | 0.88 | 0.89 | 0.83 | 0.88 (99%) |
| Banking77 | 0.91 | 0.90 | 0.81 | 0.89 (98%) |
| Clinic | 0.92 | 0.91 | 0.74 | 0.90 (98%) |
| NLU | 0.89 | 0.90 | 0.82 | 0.88 (98%) |
| **Average** | 0.91 | 0.91 | 0.83 | **0.90 (99%)** |

DBSA achieves 99% of the accuracy of the best-performing methods under a 90k context. Fine-tuning is consistently inferior to Many-Shot ICL on these datasets.

### Ablation Study

#### Sparse Attention Patterns

| Pattern | Average Accuracy |
|------|-----------|
| Full Attention | 0.84 |
| Sink + Prev + Self (DBSA) | 0.82 |
| Sink + Self | 0.27 |
| Self Only | 0.09 |

Both attention sinks and local context connections are essential. Utilizing only Sink + Self yields extremely poor results (0.27), indicating that the token eviction strategy of StreamingLLM is unsuitable for Many-Shot ICL scenarios.

#### Block-level vs. Example-level Retrieval

| Setting | Standard Inference | DBSA | Gap |
|------|---------|------|------|
| Example-level (90k) | 0.90 | 0.86 | 0.04 |
| Block-level (90k) | 0.89 | 0.88 | 0.01 |

Block-level selection better preserves the contextual relationships established during encoding, while being faster and more memory-efficient during retrieval.

#### Block Grouping Strategies

| Strategy | Average Accuracy |
|------|-----------|
| Random Grouping | 0.827 |
| K-means Clustering | 0.764 |
| Clustering + 10% Diversity | 0.810 |

Surprisingly, random grouping performs the best, as it naturally introduces intra-block diversity.

#### Dynamic Block Sorting
- Maintaining original encoding order or sorting from lowest to highest relevance yields comparable performance.
- Reversing the order significantly degrades performance due to local attention dependencies.

#### Storage Overhead
- KV Cache of Llama-3.1-8B: 0.125 MiB/token, ~3.7 GB for 30k, ~11.1 GB for 90k.
- LoRA Fine-tuning: ~0.01 GB per task, but cumulative storage costs become significant in multi-task scenarios (hundreds of tasks).

---

## Highlights & Insights

1. **Training-Free Inference Framework**: DBSA requires no auxiliary training or parameter tuning, functioning as a plug-and-play solution.
2. **Overcoming ICL Efficiency Bottlenecks**: It reduces setup time by >375x compared to fine-tuning while achieving near-fine-tuning inference latency, making Many-Shot ICL viable in practice.
3. **Incremental Demonstration Insertion**: The computational cost to encode new demonstrations remains constant regardless of the total pool size, making it highly suitable for dynamically changing data scenarios.
4. **Fine-Tuning Underperforms ICL on These Datasets**: Even when trained on the full target dataset, fine-tuning fails to significantly outperform Many-Shot ICL, challenging the conventional belief that "fine-tuning is always superior."
5. **Efficacy of Local Attention & Sink**: The *Sink + Prev + Self* pattern exhibits robust performance in a training-free setting, maintaining strong capability even with >90% sparsity.
6. **Comprehensive Ablation Analyses**: Thorough investigations are presented across various aspects (sparsity patterns, grouping strategies, sorting).

## Limitations & Future Work

1. The efficacy of the method relies heavily on the quality of Retrieval ICL, making it unsuitable for tasks requiring synthesis over the entire demonstration set (e.g., statistical tasks).
2. Evaluation is restricted to classification datasets, without covering generative tasks (e.g., translation, summarization).
3. The accuracy gap between DBSA and Ret ICL is more pronounced at the shorter 30k context length.
4. The retrieval ratio is fixed at 30% and has not been optimized.
5. The memory footprint for storing the KV Cache may become a bottleneck for extremely large demonstration pools.

## Related Work & Insights

- **Many-Shot ICL**: Bertsch et al. 2024, Agarwal et al. 2024 (thousand-shot ICL matching fine-tuning)
- **Sparse Attention**: StreamingLLM (Xiao et al. 2024), Star Attention (Acharya et al. 2024)
- **KV Cache Compression**: Token eviction (Xiao et al. 2024), quantization, and low-rank approximation
- **ICL Demonstration Selection**: BM25 retrieval (Luo et al. 2024), Parallel Context Windows (Ratner et al. 2023)
- **Sparse Attention in RAG**: Lu et al. 2024, Sun et al. 2024

---

## Rating ⭐⭐⭐⭐

This work addresses a practical and crucial problem (deployment efficiency of Many-Shot ICL). The methodology is elegant (two-stage + block-sparse design), and the empirical evaluations are solid. While it is limited by only evaluating classification tasks with relatively low task diversity, its core contribution — proving that training-free sparse attention can bring ICL latency close to fine-tuning — is highly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Many-Shot In-Context Learning for Long-Context Evaluation](on_many-shot_in-context_learning_for_long-context_evaluation.md)
- [\[ACL 2025\] What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)
- [\[ACL 2025\] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)
- [\[ICML 2026\] Sparser Block-Sparse Attention via Token Permutation](../../ICML2026/llm_efficiency/sparser_block-sparse_attention_via_token_permutation.md)
- [\[ICML 2026\] Prism: Spectral-Aware Block-Sparse Attention](../../ICML2026/llm_efficiency/prism_spectral-aware_block-sparse_attention.md)

</div>

<!-- RELATED:END -->
