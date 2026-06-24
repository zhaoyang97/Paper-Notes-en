---
title: >-
  [Paper Note] RefreshKV: Updating Small KV Cache During Long-form Generation
description: >-
  [ACL 2025][LLM Efficiency][KV cache compression] Proposes RefreshKV, an inference method that periodically alternates between full KV cache attention and small KV cache attention during generation, dynamically updating the small KV cache based on the attention patterns of full attention steps. Without permanently discarding any tokens, it achieves acceleration comparable to eviction-based methods while significantly improving performance on long-form generation tasks.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "KV cache compression"
  - "long-form generation"
  - "sparse attention"
  - "dynamic cache refresh"
  - "inference acceleration"
date: 2026-05-08
content_hash: 3147bef7237bc499
---

# RefreshKV: Updating Small KV Cache During Long-form Generation

**Conference**: ACL 2025  
**arXiv**: [2411.05787](https://arxiv.org/abs/2411.05787)  
**Code**: [https://github.com/carriex/refreshkv](https://github.com/carriex/refreshkv)  
**Area**: LLM Efficiency  
**Keywords**: KV cache compression, long-form generation, sparse attention, dynamic cache refresh, inference acceleration

## TL;DR
Proposes RefreshKV, an inference method that periodically alternates between full KV cache attention and small KV cache attention during generation, dynamically updating the small KV cache based on the attention patterns of full attention steps. Without permanently discarding any tokens, it achieves acceleration comparable to eviction-based methods while significantly improving performance on long-form generation tasks.

## Background & Motivation
- **Background**: The main bottleneck in long-context LLM inference lies in the linear memory growth of KV cache and the quadratic computational complexity of attention. Existing methods (such as StreamingLLM, H2O, SnapKV) maintain a small KV cache by permanently evicting "unimportant" tokens.
- **Limitations of Prior Work**:
    - Eviction-based methods cannot recover tokens once they are removed, which has minimal impact on **short-form generation** tasks.
    - However, in **long-form generation** tasks, performance degrades drastically because the important tokens required by different generation steps change dynamically.
    - For example, SnapKV and H2O score 0 on the F1 metric in the HTML-to-TSV task, failing completely.
- **Key Challenge**: Premature and permanent token eviction vs. high computational overhead of maintaining the full KV cache.
- **Key Insight**: Retain the full KV cache (without sacrificing memory footprint) but compute attention only over the small KV cache for most decoding steps, while periodically refreshing the small cache using full attention steps.
- **Core Idea**: Attention patterns of consecutive tokens are similar; thus, the attention scores from the most recent full attention step can be used to construct the small KV cache for subsequent steps.

## Method

### Overall Architecture
RefreshKV maintains two KV caches:
- $C_f$ (full cache): contains the KV pairs of all tokens, with capacity L.
- $C_p$ (partial cache): contains only the KV pairs of top-K important tokens, with capacity much smaller than L.

Two modes are executed alternately:
1. **Partial attention step**: generates a token by computing attention using $C_p$ (low latency).
2. **Full attention step**: generates a token by computing attention using $C_f$, and reconstructs $C_p$ (refreshing) based on the attention scores.

### Key Designs
1. **Partial Cache Construction and Refreshing**:

    - Initialization: The prefilling phase uses the attention scores of the last token to select top-K tokens.
    - Refreshing: After each full attention step, top-K tokens are re-selected using the attention scores of the current step to replace the entire $C_p$.
    - Uses max pooling to aggregate attention scores of surrounding tokens (instead of raw scores) to preserve informational integrity.
    - For GQA models, aggregates via max pooling across query heads within the same group.

2. **Adaptive Scheduling Strategy (Dynamic Stride)**:

    - Problem: Fixed stride (full attention every N steps) is not flexible enough for different layers and inputs.
    - Solution: Triggers full attention based on **query vector similarity**.
    - Details: Every S steps (QC stride), the cosine similarity between the current query vector and the query vector from the most recent full attention step is calculated.
    - If similarity > threshold $s$ $\rightarrow$ reuse the partial cache (as the attention pattern remains unchanged).
    - If similarity $\le s$ $\rightarrow$ trigger full attention and refresh the cache.
    - **Layer-wise independent decision**: Each layer can have a different trigger frequency for full attention.

3. **Maintenance of Partial Cache**:

    - During partial attention steps, the KV pairs of newly generated tokens are appended to $C_p$.
    - To maintain a fixed size, the KV pairs of tokens with the lowest attention scores during the full attention step are removed from $C_p$.
    - Key difference: Without refreshing, this degenerates to SnapKV.

4. **Continued Pre-training**:

    - Motivation: LLMs are trained with full attention, creating a train-inference mismatch when sparse attention is used during inference.
    - Solution: Simulates the attention pattern of RefreshKV during training (full attention on the first L tokens, and partial attention on the following S tokens).
    - Simplified training: Fixed stride = 50, without refreshing the partial cache.
    - Pre-trained on 120k samples of an Arxiv subset; benefits learned from short context (8K) can transfer to long context (16K).

### Loss & Training
- The inference method itself requires no training.
- Continued Pre-training employs standard next-token prediction loss.
- Loss is computed on all tokens, applying full attention to the first L tokens and partial attention to the subsequent S tokens.

## Key Experimental Results

### Main Results

| Dataset | Metric | RefreshKV(QC=5) | SnapKV | H2O | Vanilla |
|--------|------|------|----------|------|------|
| Arxiv | PPL↓ | **2.27** | 2.54 | 2.48 | 2.22 |
| Book | PPL↓ | **7.31** | 7.78 | 7.60 | 7.07 |
| HTML→TSV | F1↑ | **17** | 0 | 0 | 33 |
| GovReport | R-L↑ | **32.56** | 28.06 | 27.41 | 34.11 |
| Chain-of-key | Acc↑ | **25** | 12 | 10 | 56 |
| RULER | Acc↑ | **86** | 79 | 21 | 90 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| w/o refresh (only occasional full attention) | PPL=2.50, HTML=0 | Performance is close to SnapKV, showing that core gains stem from refreshing rather than occasional full attention. |
| w/o full attention (full attention steps also use only the refreshed partial cache) | PPL=2.32, HTML=16 | Consistent with full RefreshKV, further proving that refreshing is key. |
| Dynamic vs Fixed stride | Multi-task | Dynamic stride consistently outperforms fixed stride with similar or fewer full attention steps. |
| Continued Pre-training | PPL | Provides slight improvements across all strides, with RefreshKV benefiting the most (e.g., stride=50: 3.13 $\rightarrow$ 3.05). |

### Key Findings
- Eviction-based methods (SnapKV, H2O) completely fail on long-form generation tasks (scoring 0 on HTML$\rightarrow$TSV).
- RefreshKV recovers 52% of Vanilla performance (on HTML task) and is the only method that can generate a 2+ key chain on Chain-of-key.
- Over time, SnapKV's PPL continues to degrade relative to Vanilla, while RefreshKV maintains a stable ratio.
- Dynamic stride automatically adjusts the effective stride for different tasks (e.g., GovReport stride $\approx$ 14 vs Book stride $\approx$ 12).
- Different layers have different optimal refresh frequencies, validating the layer-wise independent decision design.

## Highlights & Insights
- Proposes a clear insight: "refreshing is better than evicting"—retaining the full KV cache for storage while using only the small cache for computation.
- The design of the Chain-of-key synthetic task is highly clever, clearly illustrating scenarios requiring "dynamic backtracking search" where eviction methods inevitably fail.
- The ablation study is highly refined: the two experimental setups (w/o refresh vs. w/o full attention) cleanly isolate the contributions of the two factors.
- The adaptive scheduling strategy is based on query similarity rather than a fixed stride, allowing variable behaviors across layers and providing greater design flexibility.

## Limitations & Future Work
- Does not reduce memory consumption (still requires storing the full KV cache), offering limited help for GPU memory-bottlenecked scenarios.
- A performance-speed trade-off remains, failing to fully recover Vanilla performance.
- The similarity threshold $s$ is globally configured; layer-wise configuration has not been explored.
- The scale of Continued Pre-training is relatively small and restricted to the Arxiv domain.
- **Future Directions**: (1) Combining with KV quantization methods to reduce memory footprint; (2) Integrating layer-specific policies (e.g., PyramidKV) to set different K sizes for different layers; (3) Extending to Vision Transformers.

## Related Work & Insights
- SnapKV selects important tokens only once during the prefilling phase and never updates them, representing a special case of RefreshKV (where no refreshing occurs).
- H2O dynamically maintains heavy hitters using accumulated attention scores, but still relies on permanent eviction.
- StreamingLLM's sink token + sliding window strategy is simple but suffers severe information loss.
- MInference focuses on accelerating sparse attention in the prefilling phase, which is complementary to RefreshKV's focus on the decoding phase.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "refreshing instead of evicting" concept is simple and elegant, though its technical implementation is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Complete coverage of language modeling, short-output, long-output, and synthetic tasks, with precise ablation designs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear exposition of motivation, intuitive design of the Chain-of-key task, and concise pseudocode.
- **Value**: ⭐⭐⭐⭐ Fills a benchmarking gap of KV cache methods on long-form generation tasks; the method can be directly applied to existing LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](kv_latent_cache_reduction.md)
- [\[ACL 2025\] SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](spindlekv_layered_kv_cache.md)
- [\[ICLR 2026\] Retrospective Sparse Attention for Efficient Long-Context Generation](../../ICLR2026/llm_efficiency/retrospective_sparse_attention_for_efficient_long-context_generation.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[ICLR 2026\] LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences](../../ICLR2026/llm_efficiency/louiskv_efficient_kv_cache_retrieval_for_long_input-output_sequences.md)

</div>

<!-- RELATED:END -->
