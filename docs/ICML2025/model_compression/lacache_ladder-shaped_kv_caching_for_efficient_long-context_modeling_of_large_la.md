---
title: >-
  [Paper Note] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models
description: >-
  [ICML2025][Model Compression][KV Cache Compression] This paper proposes a ladder-shaped KV caching pattern that retains KV states of different token ranges across different layers, thereby expanding the capturable context span within a fixed cache budget, and supports infinite-length continuous generation through an iterative compaction mechanism.
tags:
  - "ICML2025"
  - "Model Compression"
  - "KV Cache Compression"
  - "Long-Context Inference"
  - "Ladder-Shaped Cache Pattern"
  - "Training-Free Optimization"
  - "Efficient LLM Inference"
date: 2026-05-08
content_hash: 2bcb6d05d909b8a3
---

# LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

**Conference**: ICML2025  
**arXiv**: [2507.14204](https://arxiv.org/abs/2507.14204)  
**Code**: [GATECH-EIC/LaCache](https://github.com/GATECH-EIC/LaCache)  
**Area**: Model Compression  
**Keywords**: KV Cache Compression, Long-Context Inference, Ladder-Shaped Cache Pattern, Training-Free Optimization, Efficient LLM Inference  
**Authors**: Dachuan Shi, Yonggan Fu, Xiangchi Yuan, Zhongzhi Yu, Haoran You, Sixu Li, Xin Dong, Jan Kautz, Pavlo Molchanov, Yingyan (Celine) Lin

## TL;DR

This paper proposes a ladder-shaped KV caching pattern that retains KV states of different token ranges across different layers, thereby expanding the capturable context span within a fixed cache budget, and supports infinite-length continuous generation through an iterative compaction mechanism.

## Background & Motivation

During autoregressive decoding, LLMs need to cache the Key-Value (KV) states of all historical tokens. The memory overhead increases linearly with sequence length, easily triggering Out-Of-Memory (OOM) errors in long-context scenarios. Existing KV cache compression methods struggle to balance **long-distance modeling capability** and **continuous generation (without OOM)**:

- **Recency-based (StreamingLLM)**: Maintains a fixed sliding window with $\mathcal{O}(1)$ memory, supporting infinite generation but discarding long-distance information, which leads to severe accuracy degradation.
- **Retrieval-based (Quest)**: Retains the entire KV cache and retrieves it on demand, offering high accuracy but requiring $\mathcal{O}(T)$ memory, which eventually causes OOM for long sequences.
- **Importance-based (H2O)**: Relies on attention maps to select important tokens, which is incompatible with FlashAttention and limits actual inference throughput.

Key Challenge: **How to preserve as much historical token information as possible using a fixed-size cache without accessing attention maps?**

## Method

LaCache contains two core components: the **Ladder-Shaped KV Caching Pattern** and the **Iterative Compaction Mechanism**.

### 1. Ladder-Shaped KV Caching Pattern

**Key Insight**: Although recent token information is important, its KV states do not need to be preserved across all layers—different layers can maintain KV caches for **different sets of tokens**. This allows covering more tokens under the same total budget.

Specifically, the method retains the KV states of older tokens in shallower layers and gradually transitions to newer tokens in deeper layers, forming a ladder-like structure:

- Layers 1 to $S$ retain tokens 1 to $N_1$
- Layers $S+1$ to $2S$ retain tokens $N_1-O+1$ to $N_2$
- And so on, shifting right step by step

Two key hyperparameters:

| Parameter | Meaning | Influence |
|------|------|------|
| **Span $S$** | Number of consecutive layers where the same token is retained | Larger $S$ $\rightarrow$ same token information flows across more layers $\rightarrow$ accuracy $\uparrow$, cache overhead $\uparrow$ |
| **Overlap $O$** | Number of overlapping tokens between adjacent ladder segments | Larger $O$ $\rightarrow$ smoother transition between adjacent segments $\rightarrow$ more stable information retention, lower storage efficiency $\downarrow$ |

**Theoretical Analysis**:

- The ladder-shaped pattern increases the **lower bound of information retention** by distributing the coverage as evenly as possible across all layers (in the worst-case scenario, if an important token appears in a layer with the least coverage, uneven distribution would lead to a larger loss of accuracy).
- Adjacent tokens have high semantic correlation in natural language. The ladder-style smooth transition and partial overlap ensure that older token information fades out progressively rather than being abruptly truncated.

The authors randomly sampled 1500+ caching patterns, verifying that LaCache's ladder-shaped pattern lies on the Pareto-optimal frontier of PPL vs. Cache Size.

### 2. Iterative Compaction

Once the compressed KV cache is full, the ladder-shaped pattern compression is applied to it again:

1. KV states of older tokens are discarded more aggressively (located on the left side of the ladder, pruned with priority).
2. KV states of newer tokens are preserved more (located on the right side of the ladder).
3. The released space is allocated to newly incoming tokens.

This is equivalent to distance-based dynamic compression: **older tokens are compressed more, while newer tokens are compressed less**. The overall memory remains constant at $\mathcal{O}(1)$, supporting continuous generation of arbitrary length.

### Compatibility with FlashAttention

LaCache does not depend on attention maps. Physically, it only crops and rearranges KV tensors, allowing it to be directly integrated into the FlashAttention pipeline without modifying the attention kernel.

## Experimental Settings & Main Results

### Experimental Settings

- **Models**: Llama2-7B/13B, Llama2-7B/13B-Chat, Llama3-8B, Llama3.2-3B-Instruct, SmolLM2-1.7B-Instruct, LongChat-7b-v1.5
- **Baselines**: Full Cache, StreamingLLM, H2O, TOVA, PyramidInfer, SnapKV
- **Datasets**: Wikitext-2, PG19, LongBench (21 subsets), Needle-In-A-Haystack, RULER

### Language Modeling (Wikitext-2, PPL $\downarrow$)

| Model | Method | Cache | 1K | 2K | 4K | 8K |
|------|------|-------|-----|-----|-----|-----|
| Llama2-7B-Chat | Full | 100% | 4.94 | 5.32 | 6.52 | nan |
| | StreamingLLM | 512 | 6.67 | 7.41 | 7.95 | 8.97 |
| | **LaCache** | 512 | **5.20** | **6.01** | **7.06** | **8.35** |
| Llama3-8B | Full | 100% | 4.28 | 4.39 | 5.82 | 6.16 |
| | StreamingLLM | 512 | 5.46 | 5.33 | 6.73 | 6.99 |
| | **LaCache** | 512 | **4.61** | **4.89** | **6.40** | **6.78** |

On Llama2-7B-Chat with cache=512, LaCache only degrades PPL by about 5% relative to the full cache, compared to about 35% for StreamingLLM.

### Ultra-Long Sequences (PG19, 600K~10M tokens)

- Llama3-8B with full cache triggers OOM after 160K tokens, while LaCache supports up to 600K tokens while maintaining a reasonable PPL.
- On the full PG19 dataset (10 million tokens), LaCache consistently outperforms StreamingLLM.

### Extremely Small Cache (Llama3-8B, cache=80, i.e., 1% of training length)

| Decoding Length | 1K | 4K | 16K | 64K | 128K |
|----------|-----|-----|------|------|------|
| StreamingLLM | 7.28 | 8.31 | 8.88 | 9.94 | 15.68 |
| **LaCache** | **7.13** | **7.99** | **8.46** | **9.53** | **15.08** |

Even with only 80 KV slots, LaCache consistently outperforms StreamingLLM.

### Long-Context Understanding (LongBench, Average Score of 21 Subsets)

| Model | Full | StreamingLLM 50% | LaCache 50% | StreamingLLM 25% | LaCache 25% |
|------|------|-------------------|-------------|-------------------|-------------|
| Llama2-7B-Chat | 29.08 | 26.56 | **27.34** | 25.41 | **25.68** |
| Llama2-13B-Chat | 30.69 | 28.30 | **29.22** | 26.82 | **27.04** |

Under a 50% budget, LaCache outperforms StreamingLLM by an average of ~0.8 points; the advantage is more pronounced on subsets that require long-distance retrieval, such as HotpotQA, DuReader, and MultiFieldQA (e.g., HotpotQA: 32.62 vs. 29.98).

### Efficiency Comparison with FlashAttention

Since H2O requires explicit computation of attention maps and cannot integrate with FlashAttention, LaCache achieves significantly higher measured throughput at the same accuracy.

## Highlights & Insights

1. The concept of **layer-wise differentiated caching** is ingenious—it breaks the implicit assumption that "all layers must retain the same tokens." By utilizing staggered steps to cover more historical tokens, it serves as a training-free strategy of "trading space for span."
2. The **iterative compaction** design is elegant: repeatedly applying the same ladder-shaped pattern to the already compressed cache automatically achieves the effect of "compressing older tokens more aggressively" without requiring extra state management.
3. **Independence from attention maps** is a key engineering advantage, enabling seamless integration with highly efficient inference frameworks like FlashAttention, which holds high practical deployment value.
4. The random search over 1500+ patterns validates that the ladder-shaped design lies on the Pareto frontier, providing empirical support for design choices.

## Limitations & Future Work

1. **Information retention is solely based on positional heuristics**: The ladder-shaped pattern assumes newer tokens are more important. However, in certain tasks (such as multi-hop reasoning), key information may be distributed anywhere in the sequence, and a purely position-based strategy might miss distant key tokens.
2. **Hyperparameters $S$ and $O$ require calibration**: The optimal configurations may vary across different models and tasks, and the paper does not provide an automatic tuning solution.
3. **Evaluations mainly focus on PPL and LongBench**, lacking direct measurements of generation quality (such as ROUGE for summarization or human evaluation for dialogue).
4. **Batch size is fixed to 1**, and the memory-throughput trade-off in multi-batch scenarios is not demonstrated.
5. **The combined effects with orthogonal techniques such as quantization or sparsification** are only briefly mentioned without in-depth experimentation.
6. **Only decoder-only architectures are validated**, leaving the applicability to encoder-decoder or MoE models unknown.

## Related Work & Insights

- **StreamingLLM** (Xiao et al., 2023): The direct baseline for LaCache. The ladder-shaped pattern can be viewed as a layer-wise extension of StreamingLLM's sliding window.
- **H2O** (Zhang et al., 2024): An importance-based method that achieves high accuracy but is incompatible with FlashAttention.
- **Quest** (Tang et al., 2024): A retrieval-based method that retains the entire cache and retrieves it on demand, requiring $\mathcal{O}(T)$ memory.
- **SnapKV / PyramidInfer**: Other KV cache compression schemes, each with different focuses.
- **Insight**: The core insight of the ladder-shaped pattern—"different layers focus on different tokens"—could potentially be combined with layer-wise pruning or early exiting to achieve finer-grained computation-accuracy trade-offs.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of layer-wise differentiated caching is novel, and the ladder-shaped pattern is highly intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Broad coverage of models and datasets, including experiments with extremely small caches and ultra-long sequences; lacks evaluation of generation quality.
- Writing Quality: ⭐⭐⭐⭐ — Clear illustrations, with well-organized motivation, methodology, and experiments.
- Value: ⭐⭐⭐⭐ — Training-free, compatible with FlashAttention, open-source code, and carries a low barrier to deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](../../ACL2026/model_compression/latent-condensed_transformer_for_efficient_long_context_modeling.md)

</div>

<!-- RELATED:END -->
