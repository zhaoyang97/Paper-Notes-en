---
title: >-
  [Paper Note] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension
description: >-
  [ICLR 2026][Model Compression][KV cache compression] This paper proposes FreqKV, a parameter-free and architecture-agnostic KV cache compression method that iteratively compresses KV states in the frequency domain by ret…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "KV cache compression"
  - "frequency domain transform"
  - "context window extension"
  - "DCT"
  - "long-context inference"
date: 2026-05-08
content_hash: f7dd3a4dda10b9d9
---

# FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension

**Conference**: ICLR 2026
**arXiv**: [2505.00570](https://arxiv.org/abs/2505.00570)
**Code**: [GitHub](https://github.com/LUMIA-Group/FreqKV)
**Area**: Model Compression / LLM Efficiency
**Keywords**: KV cache compression, frequency domain transform, context window extension, DCT, long-context inference

## TL;DR

This paper proposes FreqKV, a parameter-free and architecture-agnostic KV cache compression method that iteratively compresses KV states in the frequency domain by retaining low-frequency components and discarding high-frequency ones. With only lightweight fine-tuning on 8K-length sequences, FreqKV extends the context window of LLaMA-2-7B to 256K while maintaining stable perplexity.

## Background & Motivation

The reasoning capability of LLMs is bounded by the context window established during pretraining; performance degrades sharply beyond this limit. Existing approaches each have notable drawbacks:

**Positional encoding methods** (ALiBi, PI, LongRoPE): rely on full self-attention, incurring quadratic computational cost.

**KV cache eviction methods** (SnapKV, PyramidKV, FastKV): discard unimportant tokens based on attention scores, but the information from evicted tokens is permanently lost, degrading subsequent decoding performance, with no ability to extrapolate beyond the context window.

**Token merging methods** (CaM, KVMerger, D2O): retain more information but perform poorly without fine-tuning.

**Additional module methods** (LoCoCo, Activation Beacon): compress KV states via extra parameters, increasing memory overhead.

**Key observation**: The authors find that KV states in LLMs exhibit strong energy concentration in the frequency domain—energy is predominantly concentrated in low-frequency components. Although the initial embeddings in the first layer show no clear low-frequency bias, subsequent layers progressively shift energy toward lower frequencies as decoding proceeds. This implies that high-frequency components are largely redundant and can be discarded with minimal loss.

Further perturbation experiments confirm that low-frequency components encode global semantic information and long-range dependencies, exhibiting robustness to input perturbations, whereas high-frequency components capture local details and are sensitive to perturbations. On summarization tasks, retaining low-frequency components substantially outperforms retaining high-frequency ones (e.g., GovReport: 25.51 vs. 14.21).

## Method

### Overall Architecture

FreqKV operates in three steps:

1. Apply the Discrete Cosine Transform (DCT) along the sequence dimension of the KV cache to convert it to the frequency domain.
2. Retain low-frequency components and discard high-frequency components for compression.
3. Apply the Inverse DCT (IDCT) to reconstruct the compressed KV cache in the time domain.

Through an iterative compression strategy, FreqKV repeatedly executes the above process as the cache grows, enabling theoretically unlimited context window extension.

### Key Designs

**1. Frequency-Domain KV Compression**

DCT is applied to the KV cache along the sequence dimension:

$$Z_K = \text{DCT}(K),\quad Z_V = \text{DCT}(V)$$

Given a retention ratio $\gamma$, the cache size is compressed from $N$ to $L = \gamma \cdot N$ by filtering out $N - L$ high-frequency components. The IDCT is then applied to reconstruct the time-domain representation, scaled by $\sqrt{L/N}$ to restore the original amplitude:

$$K_{\text{compressed}} = \sqrt{L/N} \cdot \text{IDCT}(Z_K[0:L-1])$$

The compressed KV cache directly replaces the original cache in attention computation.

**2. Iterative Compression Strategy**

This is the key mechanism enabling context window extension. Compression is triggered when the KV cache reaches its maximum capacity $N$; the freed space then accommodates new tokens. The procedure is as follows:

- Tokens within the window undergo standard attention computation.
- When the cache is full → frequency-domain compression → $N - L$ slots are freed.
- New tokens fill the freed slots → cache fills again → compression is triggered again.
- This cycle repeats indefinitely.

As a result, earlier tokens undergo more rounds of compression while more recent tokens are compressed fewer times—naturally consistent with the autoregressive property of LLMs (recent tokens are more important). Since compression is triggered only once every $N - L$ tokens, the complexity is $O(N \log N)$, with negligible computational overhead.

**3. Attention Sink Preservation**

Following the attention sink phenomenon (whereby LLMs tend to assign high attention scores to initial tokens), FreqKV reserves $S$ initial tokens that are excluded from compression (with $S = 4$ in experiments). Compression applies only to the cache content following the sink tokens.

**4. RoPE Integration**

Key states are compressed and cached prior to applying RoPE; RoPE is then encoded at attention computation time. The compressed keys use position indices within the cache rather than original sequence positions, thus enabling context window extension without positional extrapolation or interpolation.

### Loss & Training

FreqKV itself is a **parameter-free method** that introduces no additional parameters. Only lightweight fine-tuning is required to adapt to the frequency-domain compression pattern:

- For LLaMA-2: fine-tuned on the RedPajama pretraining dataset at 8K length for language modeling; the chat model is fine-tuned on the LongAlpaca instruction dataset.
- For LLaMA-3: similarly fine-tuned at 16K length.
- Training adopts the same chunk-wise processing pipeline used at inference (alternating attention computation and frequency-domain compression).

## Key Experimental Results

### Main Results

**Table 1: PG-19 Perplexity Evaluation (LLaMA-2-7B)**

| Training Length | Method | Inference Cache | 2K | 4K | 8K | 16K | 32K |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 8K | Full FT | Full | 7.55 | 7.21 | 6.98 | - | - |
| 8K | LoCoCo | Comp. | 8.15 | 8.08 | 7.27 | - | - |
| 8K | LongLoRA | Full | 7.70 | 7.35 | 7.14 | - | - |
| 8K | **FreqKV** | Comp. | **7.45** | **7.12** | **7.04** | **7.02** | **7.02** |
| 32K | LongLoRA | Full | 8.29 | 7.83 | 7.54 | 7.35 | 7.22 |
| 32K | **FreqKV** | Comp. | **7.47** | **7.14** | **7.04** | **7.00** | **6.98** |

FreqKV extends the context to 32K (PPL 7.02) with only 8K training, even outperforming LongLoRA trained at 32K (7.22). FreqKV also incurs no performance degradation on short contexts (2K/4K), slightly surpassing Full FT.

**Table 2: LongBench Long-Context Understanding Benchmark (LLaMA-2-chat-7B, 50% retention ratio)**

| Method | Single-doc QA | Multi-doc QA | Summarization | Few-shot | Code | Average |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Cache | 24.9 | 22.5 | 24.6 | 60.0 | 48.1 | 35.17 |
| SnapKV | 25.4 | 22.3 | 24.0 | 59.1 | 48.0 | 36.81 |
| FastKV | 25.5 | 22.9 | 23.7 | 57.6 | 54.5 | 36.33 |
| PyramidKV | 25.3 | 21.3 | 23.9 | 59.8 | 48.0 | 36.84 |
| **FreqKV** | **24.2** | **27.9** | **24.7** | 56.0 | **58.8** | **37.85** |

At a 50% compression ratio, FreqKV surpasses all KV eviction and merging methods, with particularly large margins on multi-document QA (+5.4 vs. SnapKV) and code tasks (+10.8 vs. Full Cache).

### Ablation Study

The paper provides multiple ablation studies in the appendix:

- **Choice of retained frequency components**: Retaining only low-frequency components substantially outperforms retaining only high-frequency components, confirming the importance of low-frequency information.
- **Retention ratio $\gamma$**: $\gamma = 0.5$ yields a favorable trade-off.
- **Number of sink tokens $S$**: $S = 4$ suffices; increasing $S$ yields diminishing returns.
- **LLaMA-2-13B**: The method remains effective on a larger model, with PPL decreasing from 6.95 (Full FT at 8K) to 6.41 (FreqKV at 32K).

### Key Findings

1. **No permanent information loss from frequency-domain compression**: Unlike eviction methods, FreqKV retains information from all tokens (via low-frequency components), merely reducing representational precision.
2. **Substantial context extension from minimal training**: Stable PPL at 256K is achieved with only 8K fine-tuning, far outperforming LongLoRA which requires 32K training.
3. **No degradation on short contexts**: Performance within the original window (2K/4K) is even marginally improved.
4. **Generality across attention variants**: Effective on both LLaMA-2 (MHA) and LLaMA-3 (GQA).
5. **Natural alignment of iterative compression**: Earlier tokens are compressed more times, which aligns with the intuition in autoregressive LLMs that recent tokens are more important.

## Highlights & Insights

- Examining KV cache compression through a frequency-domain lens is a novel and elegant perspective; the energy concentration property of DCT in low-frequency components provides a solid mathematical foundation for the method.
- Completely parameter-free, requiring no modification to the model architecture—plug-and-play.
- The iterative compression strategy naturally achieves progressive quality degradation: high precision for recent tokens, lower precision for distant tokens.
- By compressing keys before RoPE and re-encoding positions at inference time, the method elegantly sidesteps the positional extrapolation problem.

## Limitations & Future Work

1. DCT assumes symmetric signal extension at boundaries, which may not hold for certain extreme KV distributions.
2. A fixed retention ratio $\gamma$ may not be optimal for all layers and heads; adaptive $\gamma$ could yield further improvements.
3. Compressed "tokens" no longer correspond to true token positions, which may affect tasks that require precise positional information.
4. Validation on larger-scale models (70B+) is absent.
5. FreqKV is orthogonal to token eviction methods; the potential of combining the two remains unexplored.

## Related Work & Insights

FreqKV is the first to introduce frequency-domain learning into KV cache compression for decoder-only LLMs. Prior frequency-domain methods have primarily been applied to CNN-based image processing and Transformer encoders (e.g., FNet). The method is complementary to token eviction approaches and may inspire the application of other signal processing techniques (e.g., wavelet transforms) to LLM inference optimization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first application of frequency-domain KV compression in decoder-only LLMs)
- Technical Depth: ⭐⭐⭐⭐ (DCT/IDCT theory clearly presented; iterative compression design is elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (PPL + LongBench + RULER + Needle-in-a-Haystack + LongGenBench; multiple models and tasks)
- Practicality: ⭐⭐⭐⭐⭐ (parameter-free, architecture-agnostic, minimal training required)
- Writing Quality: ⭐⭐⭐⭐ (frequency-domain motivation clearly articulated; experiments comprehensive)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FASA: Frequency-aware Sparse Attention](fasa_frequency-aware_sparse_attention.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](../../NeurIPS2025/model_compression/qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)
- [\[AAAI 2026\] Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation](../../AAAI2026/model_compression/earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)
- [\[ICLR 2026\] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](cross_domain_lossy_compression_optimal_transport.md)
- [\[NeurIPS 2025\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](../../NeurIPS2025/model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)

</div>

<!-- RELATED:END -->
