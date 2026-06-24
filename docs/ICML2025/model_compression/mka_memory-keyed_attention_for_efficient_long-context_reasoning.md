---
title: >-
  [Paper Note] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning
description: >-
  [ICML2025][Model Compression][KV cache compression] Memory-Keyed Attention (MKA) is proposed to organize the KV cache into a three-level hierarchical memory (local, session, and long-term), dynamically allocating attention via a learnable routing gate. An accelerated version, FastMKA, fuses the memory sources prior to the attention computation, achieving up to 5 times the training throughput of MLA and reducing decoding latency to 54% of MLA…
tags:
  - "ICML2025"
  - "Model Compression"
  - "KV cache compression"
  - "long-context attention"
  - "hierarchical memory"
  - "dynamic routing"
  - "efficient inference"
date: 2026-05-08
content_hash: 92d91d95d1037dce
---

# MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning

**Conference**: ICML2025  
**arXiv**: [2603.20586](https://arxiv.org/abs/2603.20586)  
**Authors**: Dong Liu, Yanxuan Yu, Ben Lengerich, Ying Nian Wu (UCLA, Columbia, UW-Madison)  
**Code**: Not released  
**Area**: Model Compression  
**Keywords**: KV cache compression, long-context attention, hierarchical memory, dynamic routing, efficient inference  

## TL;DR

Memory-Keyed Attention (MKA) is proposed to organize the KV cache into a three-level hierarchical memory (local, session, and long-term), dynamically allocating attention via a learnable routing gate. An accelerated version, FastMKA, fuses the memory sources prior to the attention computation, achieving up to 5 times the training throughput of MLA and reducing decoding latency to 54% of MLA, with only approximately 1% loss in perplexity.

## Background & Motivation

The core bottleneck of long-context LLMs (128K–1M tokens) lies in the memory footprint and access latency of the KV cache:

- **Massive memory overhead**: For LLaMA-7B with a 32K context, the KV cache occupies approximately 15.8 GB (on an A800 GPU), and KV reads take 11.3 ms, accounting for more than 50% of the inference latency.
- **Limitations of prior work**:
    - MQA/GQA: Share KV heads to reduce redundancy, but exhibit limited representation capacity.
    - MLA (DeepSeek): Compresses KV using low-rank decomposition, but lacks distinction among memories of different timescales and fails to flexibly schedule heterogeneous memory sources.
    - Token eviction methods (DynamicKV, PyramidKV): Irreversibly discard information.
- **Key Insight**: Different query tokens exhibit varying dependencies on recent, intermediate, and distant contexts, making a **statically uniform KV cache strategy suboptimal**. A mechanism capable of dynamically routing attention based on the "memory timescale" is required.

## Method

### 1. Three-Level Hierarchical Memory Architecture

Inspired by the computer memory hierarchy (SRAM $\rightarrow$ HBM $\rightarrow$ DRAM), MKA divides the attention memory into three tiers:

| Level | Name | Function | Analogy |
|------|------|------|------|
| L1 | Local Memory | Standard causal attention for the current window tokens | SRAM (On-chip) |
| L2 | Session Memory | Causal prefix summary (low-rank summarization or EMA) | HBM |
| L3 | Long-term Memory | Vectorized hash-based retrieval from historical memory banks | DRAM |

- **Causality guarantee of L2**: $M_2[t] = \text{Summary}(X_{[:,1:t,:]})$, utilizing only the summaries of past tokens to prevent information leakage.
- **Retrieval mechanism for L3**: Through semantic chunking + vectorized hash indexing, it recalls the $R \ll T$ most relevant blocks from historical attention blocks to achieve amortized sub-linear complexity.

### 2. Dynamic Routing Gate

For each query token, a routing weight $\lambda = \text{softmax}(\text{MLP}(q)) \in \mathbb{R}^{B \times S \times 3}$ is learned to dynamically fuse the attention outputs from the three memory tiers:

$$O_h = \sum_{\ell=1}^{3} \lambda_\ell \odot a_\ell$$

where $a_\ell = \text{softmax}(q_h \cdot k_\ell^{h\top}) \cdot v_\ell^h$ is the attention output of the $\ell$-th memory level. The routing weights are calculated dynamically per token and per layer.

### 3. FastMKA (Route-Fused MKA)

MKA requires attention computations to be performed separately for each of the three memory levels (3 attention runs), which is computationally expensive. The key improvement of FastMKA is to **fuse first, then perform a single attention run**:

$$X_{\text{fused}} = \sum_{\ell=1}^{3} \lambda_\ell \odot L_\ell$$

$$K, V = X_{\text{fused}} W_k, \quad X_{\text{fused}} W_v$$

Then, standard causal attention is performed using the fused KV. In this way:
- The number of kernel launches is reduced from 9 to 3 (based on Table 11 data).
- Only a single attention path is required, which is compatible with standard Transformer pipelines.
- What is cached is the **fused dynamic KV**, rather than the raw token KV, further saving bandwidth.

### 4. Block-MKA and Numerical Stability

- **Block computation**: Q/K/V are partitioned into $T = N/B$ blocks, performing local softmax in L1 (SRAM), storing intermediate results in L2 (HBM), and retrieving via hashes in L3 (DRAM).
- **Online max-shift**: Recursively updates the global maximum $\mu^{(\ell)}$, and corrects historical accumulators with $\exp(\mu^{(\ell-1)} - \mu^{(\ell)})$ to ensure numerical stability in low-precision or long-sequence settings (aligned with the scan trick in FlashAttention).

### 5. Theoretical Complexity

Total runtime:

$$\mathcal{O}(BTd + BRd) \quad \text{with } R \ll T$$

where $B$ is the block size, $T = N/B$, and $R$ is the number of recalled blocks in L3. Compared to full attention $\mathcal{O}(N^2 d)$, it exhibits sub-quadratic complexity.

## Key Experimental Results

### Experimental Setup

- **Models**: Qwen2.5-7B/14B (GQA), Llama 3.1-8B (GQA), DeepSeek-V3 (MLA)
- **Data**: WikiText-2 (train set: 36,718 sentences); LongBench, RULER long-context benchmarks
- **Hardware**: NVIDIA A800 80GB; single GPU for 7B, 4-8 GPUs with Tensor Parallelism (TP) for 14B
- **Sequence Length**: 4K–256K tokens
- **Training**: 1-epoch fine-tuning, AdamW, bf16 + FlashAttention-2

### Main Results

**Table 1: Qwen2.5-7B, 16K Context**

| Method | PPL ↓ | Training Time (s) ↓ | Decoding (ms/tok) ↓ |
|------|-------|----------------|-----------------|
| MHA | 3.31 | 6234.7 | 21.4 |
| GQA | 3.28 | 5012.4 | 18.6 |
| MLA | 3.22 | 4456.9 | 12.8 |
| **FastMKA** | **3.26** | **1248.3** | **8.4** |

FastMKA requires only 28% of the training time of MLA, with a decoding latency at 66% of MLA, while perplexity (PPL) is only 0.04 higher.

**Table 2: Training Throughput (tokens/s)**

| Method | 4K | 32K | 128K | 256K |
|------|-----|------|------|------|
| MLA | 468 | 342 | 212 | 148 |
| FastMKA | 1847 | 1453 | 1032 | 742 |
| **Speedup** | **3.94×** | **4.25×** | **4.87×** | **5.01×** |

The speedup increases with sequence length, aligning with the theoretical expectation of sub-quadratic complexity.

**Table 3: Decoding Latency (ms/tok)**

| Method | 4K | 32K | 128K | 256K |
|------|-----|------|------|------|
| MLA | 8.7 | 16.4 | 32.7 | 48.9 |
| FastMKA | 6.2 | 10.3 | 18.4 | 26.3 |
| **Speedup** | **1.40×** | **1.59×** | **1.78×** | **1.86×** |

**Table 5: KV Cache Memory (128K, Qwen2.5-7B)**

| Method | KV Cache (GB) | HBM BW (GB/s) | Bandwidth Utilization |
|------|---------------|---------------|--------|
| MHA | 18.7 | 1240 | 78.2% |
| MLA | 8.9 | 1087 | 68.5% |
| FastMKA | **6.2** | **1324** | **83.5%** |

FastMKA reduces the KV cache size by 66.8% compared to MHA. Furthermore, due to the contiguous memory access pattern of the fused KV tensors, it achieves higher HBM bandwidth utilization.

**Cross-model Generalization (Table 6, 32K)**

| Model | Method | PPL | Training tok/s | Decoding ms/tok |
|------|------|-----|-----------|-------------|
| Qwen2.5-14B | MLA | 3.06 | 184 | 21.8 |
| Qwen2.5-14B | FastMKA | 3.10 | 642 | 13.6 |
| Llama 3.1-8B | MLA | 3.13 | 294 | 17.9 |
| Llama 3.1-8B | FastMKA | 3.17 | 1078 | 11.2 |
| DeepSeek-V3 | MLA | 3.08 | – | 18.4 |
| DeepSeek-V3 | FastMKA | 3.11 | – | 12.7 |

**Long-Context Benchmarks**

- LongBench (128K): FastMKA averages 54.5 vs. MLA's 55.0, with a gap of only 0.5 points.
- RULER Passkey (128K): FastMKA scores 73.4% vs. MLA's 74.8%, a gap of 1.4%.

## Limitations & Future Work

1. **Slight loss in PPL**: FastMKA trades accuracy for efficiency; its perplexity (PPL) is consistently slightly higher than that of MLA (by approximately 1-2%). For scenarios requiring extremely high precision (e.g., code generation), this trade-off might be unacceptable.
2. **Limited scale of experiments**: Fine-tuning was only conducted on WikiText-2 for 1 epoch, leaving validation on large-scale pre-training unaddressed. It remains uncertain whether the conclusions drawn from 7B/14B models generalize to 70B+ models.
3. **Limited practical utility of L3 long-term memory**: Ablation studies show that the contribution of L3 is relatively small, and it introduces external hash indexing structures, increasing engineering complexity. There also seems to be a lack of complete L3 ablation data near the end of the paper.
4. **Training cost**: Although throughput is high, the introduction of the routing MLP increases the parameter count and gradient computations; the paper does not provide a detailed comparison of total FLOPs.
5. **Modest baselines**: The method is not compared against recent, stronger KV compression approaches (e.g., KIVI, Gear, SnapKV).

## Related Work & Insights

- **MLA (DeepSeek-V2)**: Compresses KV through low-rank decomposition, serving as the most direct baseline for comparison. FastMKA introduces hierarchical routing on top of this.
- **FlashAttention**: Provides the foundation for IO-aware tiled softmax; the Block-MKA algorithm of MKA directly borrows its online softmax technique.
- **Transformer-XL / Compressive Transformer**: Early works on hierarchical memory, but difficult to scale up to LLM sizes.
- **PERK**: Stores long context in model weights rather than the KV cache, offering a complementary research direction.
- **Routing Transformer / MoE**: Sources of routing concepts, though this work applies routing to memory level selection rather than FFN expert selection.

## Highlights & Insights

**Strengths**:
- The design of hierarchical memory combined with dynamic routing is intuitive, and the analogy to computer memory hierarchies is compelling.
- The "fuse first, then attend" concept of FastMKA is simple, elegant, and engineering-friendly.
- Experiments cover multiple models and sequence lengths, showing highly consistent trends.
- The efficiency gains are significant (5× training acceleration, 1.8× decoding acceleration), which is appealing for actual deployment.

**Weaknesses**:
- Intrinsically, the fusion operation in FastMKA might discard fine-grained information from the hierarchical memory, which is not thoroughly analyzed in the paper.
- The 1-epoch fine-tuning setup raises concerns regarding the fairness of the perplexity (PPL) comparisons.
- The cached text appears to be truncated, leaving the ablation study incomplete (Table 9 only contains the title).

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of hierarchical memory and routing gates is novel, and FastMKA's fusion drastically simplifies computation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Good coverage across multiple models and lengths, but lacks comparison with stronger baselines and a complete ablation study)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete theoretical derivations, and detailed pseudocode)
- Value: ⭐⭐⭐⭐ (The 5× training acceleration holds real value for long-context deployment, though verification on a larger pre-training scale is necessitated)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)
- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](../../ACL2025/model_compression/efficient_long_context_language_model_retrieval_with_compression.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](../../ACL2026/model_compression/latent-condensed_transformer_for_efficient_long_context_modeling.md)

</div>

<!-- RELATED:END -->
