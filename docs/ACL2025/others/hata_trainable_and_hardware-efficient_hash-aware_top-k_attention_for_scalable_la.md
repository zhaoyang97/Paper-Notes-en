---
title: >-
  [Paper Note] HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference
description: >-
  [ACL 2025][Top-k Attention] HATA proposes a method that integrates learning-to-hash technology into the top-k attention mechanism. By mapping queries and keys to binary hash codes to retrieve relative qk score rankings (rather than absolute score estimations), it achieves up to a 7.2x speedup compared to full attention while maintaining model accuracy.
tags:
  - "ACL 2025"
  - "Top-k Attention"
  - "Learning-to-Hash"
  - "KVCache"
  - "LLM Inference"
  - "Hardware-Efficient"
date: 2026-05-08
content_hash: 7bd751107cd7a884
---

# HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

**Conference**: ACL 2025  
**arXiv**: [2506.02572](https://arxiv.org/abs/2506.02572)  
**Code**: Yes ([https://github.com/gpzlx1/HATA](https://github.com/gpzlx1/HATA))  
**Area**: Others  
**Keywords**: Top-k Attention, Learning-to-Hash, KVCache, LLM Inference, Hardware-Efficient

## TL;DR

HATA proposes a method that integrates learning-to-hash technology into the top-k attention mechanism. By mapping queries and keys to binary hash codes to retrieve relative qk score rankings (rather than absolute score estimations), it achieves up to a 7.2x speedup compared to full attention while maintaining model accuracy.

## Background & Motivation

Even with KVCache acceleration, the attention module remains a critical bottleneck in LLM inference. When processing 32K token sequences, the attention module of Llama2-7B consumes over 70% of the execution time. This bottleneck stems from two aspects:

**Computational Complexity**: Attention computation scale quadratically with sequence length.

**Memory Bandwidth**: Every decoding step requires loading the entirety of the cached Key and Value vectors.

Top-k attention exploits the sparsity of attention distributions, keeping only the $k$ most relevant tokens to compute attention. However, existing methods suffer from an efficiency-accuracy trade-off:

- **Low-rank methods** (Loki, InfiniGen): Dot products are computed on a projected dimension subset, but channel extraction is expensive, and maintaining accuracy requires a sufficient number of dimensions.
- **Block-based methods** (Quest, InfLLM): Divide KV into blocks and estimate the upper bound of block-level scores, but the granularity is too coarse—essential tokens are scattered across different blocks, and selecting whole blocks unnecessarily loads irrelevant KV pairs.

Core Insight: **Existing methods assume that precise estimation of the absolute qk scores is necessary, but in reality, only the relative ordering is required**. Determining whether "$s_{qk_i} > s_{qk_j}$" is significantly lower in cost than precisely calculating the qk scores. This shifts the challenge from "numerical regression" to "ordinal comparison."

## Method

### Overall Architecture

HATA operates in three phases:
1. **Offline Training**: Learns hash functions from real data, ensuring similar qk pairs remain close in Hamming space.
2. **Prefill Phase**: Computes normal attention while encoding Keys into binary hash codes and caching them.
3. **Decode Phase**: Encodes the new query into a hash code $\rightarrow$ fast-tracks top-k Key selection via Hamming distance $\rightarrow$ computes exact attention only for the selected KV pairs.

### Key Designs

1. **Learning-to-Hash Modeling**:

   The hash function is defined as $h(x) = 2 \cdot \text{Sigmoid}(\sigma \cdot x W_H) - 1$, where $W_H$ is a trainable hash weight matrix.

   The optimization objective comprises three terms:
    - **Similarity Preservation**: $\epsilon \sum_j \sum_i s_{j,i} \|h(q_j) - h(k_{j,i})\|^2$ — similar qk pairs are assigned neighboring hash codes.
    - **Bit Balance**: $\eta \sum_j \|\sum_i h(k_{j,i})\|^2$ — assuring each bit in the hash codes has a rough balance of $+1$ and $-1$.
    - **Decorrelation**: $\lambda \|W_H^T W_H - I_r\|$ — encouraging different hash bits to encode diverse information.

   The Sigmoid function is utilized to approximate the non-differentiable sign function, where $\sigma$ controls the smoothness of the approximation. One $W_H$ is trained independently for each attention head.

2. **Training Data Construction**:

    - Q and K are collected from each attention head during the prefill phase.
    - Sample $q_j$ and calculate its qk scores with $[k_1,...,k_j]$.
    - The top 10% of qk pairs are labeled as positive samples (linearly decaying labels $s \in [1,20]$), while the remaining 90% are negative samples ($s = -1$).
    - Thousands to millions of training pairs can be generated per sequence, sourced from dozens of sequences to enhance diversity.

3. **Decode Phase Algorithm**:

    - Perform `HashEncode` (matrix multiplication + Sign + BitPack) on the new query and key.
    - Append the key hash codes to the hash code cache.
    - Calculate the Hamming distance between the query hash code and all cached key hash codes: `bitcount(bitwise_xor(Q_H, K_H_cache))`.
    - In Grouped-Query Attention (GQA) scenarios, aggregate scores across multiple queries sharing the same KVCache.
    - Select the top-k, gather the corresponding KV pairs, and execute sparse attention.

### Hardware-Efficient Optimizations

1. **Kernel Fusion**: Fuses linear projection $\rightarrow$ Sign $\rightarrow$ BitPack $\rightarrow$ cache update into a single CUDA kernel, avoiding frequent CPU-GPU synchronization.
2. **High-Performance Hamming Score Operator**: Achieves bit counting based on GPU `XOR` + `popc`/`popcll` instructions, optimizing bandwidth through coalesced memory access.
3. **Gather + FlashAttention Fusion**: Integrates the Gather operation into the FlashAttention kernel, reducing redundant data transfer between HBM and SRAM.

Implementation Code: 1,470 lines of C++/CUDA + 940 lines of Python, integrated into existing inference frameworks as a plug-in.

## Key Experimental Results

### Main Results: LongBench-e Accuracy (512 token budget)

| Task | Dense | Loki | Quest | MagicPIG | H2O | SnapKV | **HATA** |
|------|-------|------|-------|----------|-----|--------|---------|
| **Llama-2-7B AVG** | 34.47 | 32.78 | 32.64 | 34.09 | 9.57 | 24.96 | **34.60** |
| **Llama-3.1-8B AVG** | 54.10 | 53.23 | 52.19 | 47.61 | 49.89 | 51.00 | **53.94** |

HATA closely matches the performance of Dense (full attention) on both models and even outperforms Dense on certain tasks (e.g., HQA 15.65 vs. 15.30).

### NIAH (Needle in a Haystack) Test

| Task | Dense | Loki | Quest | HATA |
|------|-------|------|-------|------|
| NS1 (Llama-2) | 93.75 | 25.00 | 100.0 | **100.0** |
| NS2 (Llama-2) | 100.0 | 2.08 | 95.83 | **98.96** |
| NS3 (Llama-2) | 91.67 | 0.00 | 52.08 | **83.33** |
| NS1 (Llama-3.1) | 100.0 | 98.96 | 100.0 | **98.96** |
| NS3 (Llama-3.1) | 100.0 | 96.88 | 47.92 | **100.0** |

HATA maintains performance close to or exceeding Dense across multi-level difficulties in the NIAH test, while Loki suffers from a performance collapse on Llama-2.

### Inference Speed (token/s, Llama-3.1-8B, A800)

| Sequence Length | Dense | Loki | Quest | HATA |
|---------|-------|------|-------|------|
| 32K | Baseline | ~1.2× | ~1.5× | ~2× |
| 64K | Baseline | ~1.5× | ~2.5× | ~4× |
| 128K | Baseline | ~2× | ~4× | ~7.2× |

The longer the sequence, the greater the speedup achieved by HATA—reaching a 7.2x speedup at 128K.

### Key Findings

1. **Optimal Accuracy-Efficiency Balance**: Among all top-k methods, HATA resides closest to full attention in terms of accuracy while producing larger speedups.
2. **Ordinal Comparison is Sufficient**: Experiments confirm that only the relative ordering of qk scores is needed (rather than absolute value estimation) to achieve high-quality top-k selection.
3. **Negligible Prefill Overhead**: The extra overhead of hash encoding is less than 1% of the total computation (since $r_{\text{bit}} \ll$ sequence length).
4. **Impact of Hash Code Dimensions**: Among $r$-bit hash codes, too small an $r$ degrades accuracy, while too large an $r$ reduces efficiency; 128 bits is typically a good sweet spot.
5. **GQA Compatibility**: By aggregating scores from multiple queries sharing the same KVCache, HATA naturally supports GQA architectures.

## Highlights & Insights

- **Profound Reformulation of the Problem**: Reconceptualizing top-k attention from "precise score estimation" to "ordinal comparison" is the most significant contribution of this work. Absolute scores are indeed unnecessary for top-k selection, and this insight unlocks an entirely new optimization space.
- **Perfect Synergy with Hash Technology**: Learning-to-hash naturally addresses the ordinal comparison problem (where Hamming distance preserves order), and binary operations (`XOR` + `popcount`) are extremely efficient on GPUs.
- **High Engineering Completeness**: This work features not only algorithm design but also comprehensive system-level optimizations, including kernel fusion, highly optimized Hamming operators, and FlashAttention integration.
- **Plug-and-Play Design**: Users can easily adopt it by replacing the standard attention mechanism with HATA attention, lowering the barrier to deployment.

## Limitations & Future Work

1. **Requirement of Offline Weight Training**: Each layer and head of every model requires training its own $W_H$. Although training overhead is low, it adds steps to the deployment pipeline.
2. **Fixed Token Budget Assumption**: Current experiments rely on a fixed budget of 512 tokens; an adaptive budget could yield better results.
3. **Acceleration Only in Decode Phase**: The prefill phase still uses full attention, leaving the latency of long-context prefill unoptimized.
4. **Limited Model Coverage**: Evaluation is mainly conducted on Llama-2 and Llama-3.1, leaving MoE models or larger-scale models unvalidated.

## Related Work & Insights

- SparQ's dimension subsetting approach tackles the same problem from a different angle (numerical precision vs. ranking precision), while HATA's ordinal comparison paradigm is more fundamental.
- Block-based methods like Quest and InfLLM complement token-level methods. A two-level hierarchy could be considered (coarse block-level filtering followed by fine-grained hash-based selection).
- Comparison with Hash-RAG (2505.16133): Both represent applications of learning-to-hash in NLP. However, HATA focuses on intra-attention qk matching, while Hash-RAG targets external knowledge base retrieval.
- Insights for KVCache compression approaches (e.g., H2O, SnapKV): HATA eliminates the risk of dropping crucial information by keeping all KV pairs and selecting them more efficiently, rather than permanently discarding them.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The core insight that "only ordering is needed, not absolute scores" is highly profound. The application of learning-to-hash in LLM attention is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Features comprehensive evaluations across 13 tasks in LongBench-e, multi-level difficulty in NIAH, two distinct models, comparisons against multiple baselines, detailed speedups, and thorough ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation is clearly articulated, the algorithm pseudocode is comprehensive, and Figures 1/2/3 present high information density. However, equations in the learning-to-hash background section are somewhat dense.
- **Value**: ⭐⭐⭐⭐⭐ — Delivers up to a 7.2x speedup without accuracy degradation, offering direct and significant practical value for deploying long-context LLMs. The open-source code operates out-of-the-box in a plug-and-play manner.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](../../ICML2026/others/haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)
- [\[ACL 2025\] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)
- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](../../NeurIPS2025/others/learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)

</div>

<!-- RELATED:END -->
