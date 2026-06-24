---
title: >-
  [Paper Note] RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling
description: >-
  [NeurIPS 2025][Model Compression][Efficient Sequence Modeling] This paper proposes RAT (Recurrence And aTtention), a chunk-based intermediate architecture that models local dependencies within chunks via linear RNNs and enables global access across chunks via softmax attention. At $L=16$, RAT achieves a 9× single-layer decoding speedup and 10× maximum throughput improvement over standard attention with comparable performance; a hybrid variant alternating with sliding window a…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Efficient Sequence Modeling"
  - "Chunk-based Architecture"
  - "RNN-Attention Hybrid"
  - "Long Context"
  - "Linear Complexity"
date: 2026-05-08
content_hash: b65437770bcd2be1
---

# RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2507.04416](https://arxiv.org/abs/2507.04416)  
**Code**: [GitHub](https://github.com/CLAIRE-Labo/RAT)  
**Area**: Model Compression
**Keywords**: Efficient Sequence Modeling, Chunk-based Architecture, RNN-Attention Hybrid, Long Context, Linear Complexity

## TL;DR

This paper proposes RAT (Recurrence And aTtention), a chunk-based intermediate architecture that models local dependencies within chunks via linear RNNs and enables global access across chunks via softmax attention. At $L=16$, RAT achieves a 9× single-layer decoding speedup and 10× maximum throughput improvement over standard attention with comparable performance; a hybrid variant alternating with sliding window attention achieves state-of-the-art results on nearly all benchmarks.

## Background & Motivation

Transformers rely on full self-attention, causing quadratic growth in computation with sequence length during both training and inference, which limits scalability in long-context settings. Numerous efficient alternatives have emerged in recent years, including Mamba, linear attention, and state space models, but their core limitation is:

**Fundamental deficiency of RNN-based models**: The entire sequence history is compressed into a fixed-size monolithic state, leading to inevitable **memory degradation** as the sequence grows. This makes precise retrieval of distant information difficult, especially under noisy contexts.

**Overallocation of attention**: Standard attention retains full access for every token, yet the majority of local dependencies can be handled more efficiently; attention is "overqualified" for local context.

**Absence of intermediate designs**: RNNs ($L=T$, full compression) and attention ($L=1$, no compression) represent two extremes, with no intermediate architecture that flexibly interpolates between efficiency and accuracy.

The core insight of RAT is **partial compression with global direct access**: within-chunk RNN compression over short sequences incurs negligible information loss, while inter-chunk softmax attention guarantees direct retrieval of distant information. By adjusting chunk size $L$, RAT forms a continuous spectrum between RNNs and attention.

## Method

### Overall Architecture

Given a sequence of length $T$, RAT partitions it into $C = T/L$ chunks of $L$ tokens each. Within each chunk, a linear RNN (EMA gating) recursively compresses keys and values to obtain chunk-level representations; across chunks, standard softmax attention performs global interaction over these chunk-level representations. The overall process is:

(1) Input tokens are recursively updated within chunks → (2) The terminal state of each chunk serves as chunk-level KVs → (3) The current query attends over all chunk-level KVs → (4) Output gating.

### Key Designs

1. **Intra-chunk Recurrence**: A simple linear recurrence performs EMA-style gated aggregation separately on values and keys:

$$\tilde{v}_{c,l} = g_{c,l} \odot \tilde{v}_{c,l-1} + (1 - g_{c,l}) \odot v_{c,l}$$

$$\tilde{k}_{c,l} = g_{c,l} \odot \tilde{k}_{c,l-1} + (1 - g_{c,l}) \odot k_{c,l}$$

where $g_{c,l}$ is a per-dimension forget gate (sigmoid activation) computed via a linear projection of the input. **Design Motivation**: With short chunk lengths (e.g., $L=16$), short-sequence RNNs do not suffer from memory degradation and are more efficient than attention. The simplest linear RNN form is chosen to highlight the core idea, though the framework is compatible with more complex RNN variants.

2. **Inter-chunk Softmax Attention**: The query at position $(c,l)$ attends over the terminal states $\tilde{K}_{:,-1}$ of all preceding chunks and the accumulated state $\tilde{k}_{c,l}$ of the current chunk:

$$y_{c,l} = f([q_{c,l}\tilde{K}_{:,-1}^\top; q_{c,l}\tilde{k}_{c,l}^\top])[\tilde{V}_{:,-1}; \tilde{v}_{c,l}]$$

The result is modulated by an output gate $z_{c,l}$. Inter-chunk attention operates on a sequence of length $C$ rather than $T$, reducing FLOPs by a factor of $L$.

3. **Parameter Allocation and Positional Encoding**:

    - **Parameter Sharing**: Total parameter budget is $4D^2$. Experiments show that allocating more parameters to RNN gating (rather than attention QK projections) yields better results. The final design shares Q/K projections and redirects the saved parameters to gating. Although Q/K are shared, the forget gates produce per-dimension gated keys $\tilde{k}$, preventing degeneration into single-head attention.
    - **Chunk-level RoPE**: Positional encoding is indexed by chunk position rather than token position; the RNN itself encodes intra-chunk positional information. This also improves length generalization, as the number of positions to encode (number of chunks) is far smaller than the sequence length.

4. **Hybrid Architecture RAT-SWA**: RAT alternates with sliding window attention (SWA, window size 1024). The two are highly complementary — SWA concentrates most computation within a fixed window for local interaction, while RAT reserves attention for global access and handles local context more efficiently via lightweight RNNs.

### Loss & Training

A 1.3B model is pretrained from scratch on the FineWeb-Edu dataset for 100B tokens, with a learning rate of 8e-4 under cosine decay to 1e-6, a global batch size of 2M tokens, and a 4K context window. The implementation requires no custom CUDA/Triton kernels: intra-chunk recurrence uses PyTorch associative scan (reducing scan depth from $O(\log T)$ to $O(\log L)$), and inter-chunk attention uses PyTorch flex attention (supporting custom masks and returning the softmax denominator). The causal mask is handled by decomposing via online softmax into two terms that are computed and merged separately.

## Key Experimental Results

### Main Results — 1.3B Model Efficiency and Performance

| Model | Max Throughput (tok/s) | Val PPL | CSR Avg Acc | LongBench SQA F1 | LongBench Summ R-L | LongBench Code |
|------|-------------------|---------|-------------|-------------------|--------------------|----|
| Attention | 3,052 | 7.61 | 56.9 | 18.2 | 19.5 | 23.9 |
| RNN | — | 7.82 | 55.8 | — | — | — |
| RAT(L=16) | **31,170** (10.2×) | 7.67 | 56.7 | **19.6** | **20.2** | 17.4 |
| Attention-SWA | 4,605 | 7.61 | 57.1 | 17.4 | 19.4 | 21.7 |
| **RAT(L=16)-SWA** | **13,582** (4.4×) | **7.57** | **58.0** | 18.8 | 19.5 | **28.2** |

### Ablation Study (200M Model, Book Dataset)

| Configuration | PPL (4K) | PPL (32K) | Notes |
|------|----------|-----------|------|
| More params to RNN gating | **13.42** | 14.05 | Best configuration |
| More params to QK | 13.82 (+0.40) | 14.52 (+0.47) | Gating more important |
| Original RoPE (token positions) | 13.52 (+0.10) | 14.35 (+0.30) | Chunk-level RoPE superior |
| Chunk-level RoPE | **13.42** | **14.05** | Particularly beneficial for long sequences |

### Scalability: 7B/13B Model Throughput

| Model Scale | RAT(L=16) | Attention | Speedup |
|----------|-----------|-----------|--------|
| 1.3B | 31,170 | 3,152 | 10.2× |
| 7B | 10,103 | 983 | 10.3× |
| 13B | 5,749 | 534 | **10.8×** |

### Key Findings

- **Efficiency**: RAT(L=16) is 7× faster than attention for training on 100K sequences, 9× faster for generation at position 4K, and achieves 10× higher maximum throughput. The speedup increases with model scale (10.8× at 13B), as attention exhibits lower GPU utilization in larger models.
- **Performance as an intermediate**: In pretraining PPL, RAT falls precisely between RNNs and attention — Attention: 7.61, RAT(L=16): 7.67, RNN: 7.82. Increasing chunk size from $L=16$ to $256$ monotonically increases PPL.
- **Long-context advantage**: On LongBench QA and summarization tasks, RAT(L=16) outperforms full attention on multiple metrics (e.g., NarrativeQA: 14.5 vs. 12.3, QA Avg: 19.6 vs. 18.2), as inter-chunk attention avoids the long-range memory degradation of RNNs.
- **Hybrid architecture is best overall**: RAT(L=16)-SWA comprehensively outperforms all variants on commonsense reasoning (+1), code completion (+4), hard QA (+4), and summarization (+1), while maintaining ~4× throughput improvement.

## Highlights & Insights

1. **Chunk size $L$ as a continuous knob**: $L=1$ → Attention, $L=T$ → RNN; intermediate values provide a smooth efficiency–accuracy tradeoff. This design elegantly unifies two seemingly unrelated architectural families within a single framework.
2. **"Attention is wasted on local context"**: This insight is strongly validated by the RAT-SWA experiments — combining RNNs for local and attention for global access is both faster and more accurate.
3. **No custom kernels**: The purely PyTorch implementation (associative scan + flex attention) lowers the engineering barrier and is compatible with tensor parallelism and context parallelism.
4. **Reduced KV cache**: RAT only stores chunk-level KVs (16× fewer than full attention), significantly reducing the risk of out-of-memory errors.

## Limitations & Future Work

- Information discontinuity may arise at chunk boundaries when a semantic unit spans two chunks, as the RNN can only partially capture it.
- Only the simplest linear RNN (EMA gating) is explored; stronger RNN variants (e.g., nonlinear RNNs, 2D recurrence) are not investigated.
- For short sequences (<4K), training speed is slightly lower than attention due to insufficient GPU parallelism with few chunks in flex attention.
- Accuracy evaluation at scale is limited to 1.3B in the main experiments; 7B/13B results report throughput only, without corresponding accuracy metrics.

## Related Work & Insights

- Compared to fixed-state models such as Mamba/Mamba2/GatedDeltaNet, RAT's memory capacity grows with sequence length (as the number of chunks increases), which constitutes a fundamental distinction.
- The "global via attention, local via RNN" philosophy of RAT-SWA is consistent with hybrid models such as Samba and Griffin, but RAT's implementation is simpler.
- The chunk-level RoPE positional encoding strategy provides a useful reference for positional encoding design in hierarchical architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The chunk-based fusion of RNNs and attention is concise and effective; using $L$ as a continuous interpolation parameter is an elegant design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 7 short-context reasoning tasks, 11 LongBench tasks, 4 SFT tasks, and 9 synthetic retrieval tasks — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly developed, with the chunk-based design arising naturally from the RNN/attention comparison.
- **Value**: ⭐⭐⭐⭐⭐ 10× throughput improvement with no custom kernels and no accuracy loss — high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[NeurIPS 2025\] LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups](lt-soups_bridging_head_and_tail_classes_via_subsampled_model_soups.md)
- [\[NeurIPS 2025\] SpecAttn: Speculating Sparse Attention](specattn_speculating_sparse_attention.md)
- [\[NeurIPS 2025\] Spark Transformer: Reactivating Sparsity in FFN and Attention](spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](../../CVPR2025/model_compression/linear_attention_modeling_for_learned_image_compression.md)

</div>

<!-- RELATED:END -->
