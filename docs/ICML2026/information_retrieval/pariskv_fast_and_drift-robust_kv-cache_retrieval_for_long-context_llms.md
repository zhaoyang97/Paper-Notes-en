---
title: >-
  [Paper Note] ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs
description: >-
  [ICML 2026][Information Retrieval & RAG][KV-Cache Retrieval] ParisKV reduces decoding latency for Top-$k$ KV retrieval on million-token contexts by 17–44× compared to MagicPIG/PQCache while matching or exceeding full att…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "KV-Cache Retrieval"
  - "Long Context"
  - "Drift-Robust"
  - "GPU-Native"
  - "UVA Offloading"
date: 2026-05-08
content_hash: f7d005c0c8a32f81
---

# ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.07721](https://arxiv.org/abs/2602.07721)  
**Code**: https://github.com/amy-77/ParisKV/tree/main  
**Area**: LLM Efficiency / Long-Context Inference / KV-Cache Retrieval  
**Keywords**: KV-Cache Retrieval, Long Context, Drift-Robust, GPU-Native, UVA Offloading

## TL;DR
ParisKV reduces decoding latency for Top-$k$ KV retrieval on million-token contexts by 17–44× compared to MagicPIG/PQCache while matching or exceeding full attention accuracy on 7 out of 9 long-generation tasks. This is achieved by normalizing and rotating keys/queries onto a hypersphere, replacing prefill-learned centroids with "data-agnostic analytical centroids," and utilizing a GPU-native "collision voting + 4-bit quantized reranking" pipeline with UVA-based on-demand KV fetching.

## Background & Motivation

**Background**: Long-context LLM inference is memory-bound, as every decoding step requires reading the entire KV history, with bandwidth demands growing linearly with context length. *KV-cache retrieval* (retaining all KV pairs and dynamically selecting the Top-$k$ per step) is more suitable for open-ended long generation than *KV-cache dropping* (permanent exclusion), as it prevents failure from incorrectly discarding early tokens. Representative methods include Quest, MagicPIG, PQCache, and RetrievalAttention.

**Limitations of Prior Work**: Existing retrieval methods often fail in "long generation + large context" scenarios due to three primary issues: (C1) **Speed-quality tradeoff**: Coarse clustering or low-bit quantization sacrifices recall for speed, requiring larger retrieval budgets that negate sparsity benefits; (C2) **Decoding drift**: Centroids learned from prefill-stage keys become mismatched with the distribution of newly generated keys, causing a sharp decline in recall during long sequences (e.g., PQCache recall collapses on AIME as shown in Fig. 1); (C3) **CPU-side retrieval bottlenecks**: When KV pairs are offloaded to CPU, traditional CPU search and CPU-to-GPU memory copies underutilize the GPU and introduce significant overhead.

**Key Challenge**: Centroids learned from data inevitably suffer from drift. To avoid drift, centroids must be "data-agnostic." However, data-independent hashing or grids suffer from bucket imbalance due to the anisotropic distribution of raw keys, leading to failed collision statistics.

**Goal**: (1) Maintain stable Top-$k$ recall under decoding drift; (2) Keep retrieval decisions entirely on the GPU; (3) Achieve end-to-end latency near GPU-native speeds while offloading KV cache to CPU.

**Key Insight**: By normalizing keys/queries via $\ell_2$ normalization and applying a shared random orthogonal rotation, subspace directions become approximately isotropic. In this state, a fixed set of "sign-pattern centroids" $\{\pm 1/\sqrt{m}\}^m$ can uniformly cover all directions on the hypersphere. Consequently, *any newly generated key will remain close to at least one centroid*, eliminating the drift problem fundamentally.

**Core Idea**: Utilize a "hypersphere + random rotation + analytical centroid" approach for KV-cache Top-$k$ retrieval. This is paired with a GPU-native two-stage pipeline (collision voting and 4-bit reranking) and UVA-based on-demand fetching to ensure drift-robustness, low latency, and million-token scalability.

## Method

### Overall Architecture

ParisKV is an algorithm-system co-design consisting of two phases:

**Prefill Phase**: Generates a one-time "summary" of history keys: (A.1) Shared normalization and rotation to the hypersphere; (A.2) Partitioning into $B$ subspaces where directions are mapped to the nearest analytical centroid id (for voting); (A.3) Storing a 4-bit quantized direction code $\text{code}_{i,b}$ and a scalar weight $w_{i,b}$ for each subspace (for reranking). Full-precision KV pairs are asynchronously offloaded to CPU, while the GPU retains compact metadata $\{(\text{centroid\_id}_{i,b}, \text{code}_{i,b}, w_{i,b})\}$.

**Decode Phase**: For each query: (B.1) New tokens are placed in an update buffer; when filled, a sliding window evicts old tokens to the retrieval area (GPU-to-CPU) and encodes new metadata; (B.2) Two-stage GPU retrieval: *Coarse filtering* uses centroid ids for subspace collision voting to select a $\beta$ percentage (e.g., 5%–10%) of candidates; *Fine reranking* uses 4-bit codes to estimate inner products for the final Top-$k$ (e.g., $k=100$); (B.3) GPU kernels fetch only the Top-$k$ full-precision KV pairs from CPU memory via UVA, bypassing explicit `memcpy` and CPU scheduling.

### Key Designs

1.  **Hypersphere + Random Rotation + Analytical Centroids (Source of Drift Robustness)**:
    - **Function**: Replaces data-learned centroids with fixed "data-agnostic" centroids so new keys never fall outside the coverage area.
    - **Mechanism**: First, $\ell_2$ normalization $\hat{\mathbf{k}}_i = \mathbf{k}_i / \|\mathbf{k}_i\|_2$ maps vectors to the unit hypersphere. A shared orthogonal matrix $\mathbf{R}$ (implemented via SRHT) rotates the vectors: $\tilde{\mathbf{k}}_i = \mathbf{R}\hat{\mathbf{k}}_i$, ensuring isotropic distribution and uniform information spread. The $D$ dimensions are split into $B$ subspaces of size $m=D/B$. In each subspace, the analytical centroid set $\Omega = \{\pm 1/\sqrt{m}\}^m$ (the vertices of an $m$-dimensional hypercube projected onto the sphere) covers all $2^m$ orthants. Polar decomposition $\tilde{\mathbf{k}}_{i,b} = r_{i,b}\mathbf{u}_{i,b}$ separates direction (for voting) and radius (for calibration).
    - **Design Motivation**: Directly addresses C2 (centroid staleness). Proposition 4.1 provides theoretical support: after Haar random rotation, subspace energy $z_b \sim \mathrm{Beta}(m/2, (D-m)/2)$ and coordinate squares $(u_b)_j^2 \sim \mathrm{Beta}(1/2, (m-1)/2)$ provide distribution guarantees for quantization.

2.  **GPU-Native Coarse Filtering: Multi-Subspace Collision Voting**:
    - **Function**: Rapidly prunes $n_t$ candidates to a pool of $\beta n_t$ using cheap collision signals.
    - **Mechanism**: The query undergoes the same transformation. In each subspace $b$, the inner products $\tilde{\mathbf{q}}_b^\top \mathbf{c}$ with $2^m$ centroids are calculated, allowing the top $\rho$ centroids to contribute a "vote." A key receives 1 vote if its assigned centroid in that subspace is among the query's top choices. Votes are accumulated across $B$ subspaces. A custom `bucket_topk` CUDA kernel performs selection on these integer scores without full sorting.
    - **Design Motivation**: Unlike PQCache/MagicPIG, voting is computationally inexpensive (integer addition) and redundant (robust to errors in individual subspaces).

3.  **4-bit Quantized Reranking with Cached Weight Estimator**:
    - **Function**: Accurately estimates $\langle \mathbf{k}_i, \mathbf{q} \rangle$ for candidates without accessing CPU-resident full-precision keys.
    - **Mechanism**: Subspace directions $\mathbf{u}_{i,b}$ are quantized to 4-bit $\mathbf{v}_{i,b}$ (1-bit sign + 3-bit magnitude). An alignment factor $\alpha_{i,b} = \langle \mathbf{v}_{i,b}, \mathbf{u}_{i,b} \rangle$ is used to correct the systematic underestimation of inner products: $\langle \mathbf{u}_{i,b}, \tilde{\mathbf{q}}_b \rangle \approx \langle \mathbf{v}_{i,b}, \tilde{\mathbf{q}}_b \rangle / \alpha_{i,b}$. Key-dependent factors are pre-computed as $w_{i,b} = \|\mathbf{k}_i\|_2 \cdot r_{i,b} / \alpha_{i,b}$. During decoding, the estimate simplifies to: $\widehat{\langle \mathbf{k}_i, \mathbf{q} \rangle} = \|\mathbf{q}\|_2 \sum_{b=1}^{B} w_{i,b} \langle \mathbf{v}_{i,b}, \tilde{\mathbf{q}}_b \rangle$.
    - **Design Motivation**: Solves C1 and C3 by using compact 4-bit metadata and per-key calibration to achieve near-full-precision accuracy entirely on the GPU.

### Loss & Training
ParisKV is a **purely training-free inference method**. All centroids and quantization levels are pre-calculated based on Beta priors. The rotation $\mathbf{R}$ is constructed via SRHT. The implementation provides four custom CUDA kernels: `bucket_topk`, parallel collision, fused reranking, and UVA-based fetching.

## Key Experimental Results

**Main Results: Long-Generation Reasoning (Accuracy)**

| Model | Task | Full Attn | PQCache | MagicPIG | ParisKV | Gain vs PQCache |
|-------|------|-----------|---------|----------|---------|-----------------|
| Qwen-3-4B | GPQA-Diamond (pass@1) | 64.14 | 38.38 | 32.32 | **72.22** | +33.84 |
| Qwen-3-4B | MATH500 (pass@1) | 88.60 | 58.80 | 46.40 | **92.80** | +34.00 |
| Qwen-3-4B | AIME25 (pass@8) | 86.67 | 3.33 | 6.67 | **80.00** | +76.67 |
| DS-R1-Llama-8B | AIME25 (pass@8) | 50.00 | 13.30 | 13.30 | **53.30** | +40.00 |
| Qwen-3-8B | MATH500 (pass@1) | 87.40 | 69.21 | 45.80 | **93.00** | +23.79 |

ParisKV matches or exceeds full attention in 7 out of 9 settings. On AIME25, while PQCache/MagicPIG collapse (pass@8 < 17), ParisKV recovers performance to 53–80.

**Main Results: Million-Token Decoding Efficiency**

| Context | Full Attn | PQCache | MagicPIG | ParisKV | Speedup |
|---------|-----------|---------|----------|---------|---------|
| 128K (bs=1) | runnable | – | – | 24.32 ms/step | 2.1–2.8× vs full |
| 256K (bs≥2) | **OOM** | – | – | scales to bs=5 | – |
| 1024K (bs=1, Llama3.1-8B) | OOM | 2179 ms/step | 830 ms/step | **49 ms/step** | **44.4× / 16.9×** |

ParisKV provides a 44× and 17× speedup over PQCache and MagicPIG, respectively, at 1M tokens.

**Ablation Study**

| Configuration | Coarse Recall@100 | End-to-End Recall@100 |
|---------------|-------------------|-----------------------|
| Baseline (No N+R, Learned Centroids) | 6% | 36.5% |
| + Norm + Rotate + Analytical Centroids (Ours) | **16.1%** | **64.3%** |

### Key Findings
- **Robustness stems from data-agnostic centroids**: The combination of normalization, rotation, and analytical centroids improves end-to-end recall from 36.5% to 64.3%.
- **Long generation is harder than long input**: While PQCache performs okay on long inputs (LongBench-V2), long decoding steps accumulate drift, leading to the collapse seen in AIME25.
- **TPOT scales excellently with batch size**: On Qwen3-8B (128K), TPOT drops from 24.32ms (bs=1) to 7.37ms (bs=8) as overhead is amortized.

## Highlights & Insights
- **Elegant "Data-Agnostic" Solution**: Solves the drift problem fundamentally by removing the requirement that centroids fit the data distribution.
- **Collision Voting over Sorting**: Replaces expensive inner-product sorting with bit-level matching and integer addition, playing to GPU hardware strengths.
- **UVA vs. Explicit Memcpy**: Utilizing UVA for on-demand fetching bypasses the CPU scheduling stack, which is critical for million-token scalability.

## Limitations & Future Work
- **Choice of $m$**: Large $m$ improves coverage but increases query-centroid computation; small $m$ reduces voting stability.
- **SRHT Overhead**: Prefill processing may not be beneficial for short contexts (<10K tokens).
- **Assumed Isotropy**: While SRHT spreads features, it does not account for structured sparsity (e.g., attention sinks) existing in some heads.

## Related Work & Insights
- **vs. PQCache**: PQCache uses data-learned product quantization, making it susceptible to decoding drift. ParisKV's analytical centroids are drift-immune.
- **vs. MagicPIG**: MagicPIG uses LSH, which is sensitive to anisotropic distributions. ParisKV "roundifies" the distribution before retrieval.
- **vs. RetrievalAttention**: Focused on the same retrieval paradigm but lacks the system-level optimizations (UVA/Voting) needed for 1M+ scales.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)
- [\[ICML 2026\] HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](very_efficient_listwise_multimodal_reranking_for_long_documents.md)

</div>

<!-- RELATED:END -->
