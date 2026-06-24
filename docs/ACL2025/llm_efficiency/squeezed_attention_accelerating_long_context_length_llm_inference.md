---
title: >-
  [Paper Note] Squeezed Attention: Accelerating Long Context Length LLM Inference
description: >-
  [ACL 2025][LLM Efficiency][Long Context Inference] Ours proposes Squeezed Attention, which compresses the Key vectors of fixed contexts via offline K-means clustering. During inference, centroid matching is used to predict important Keys and compute exact attention solely on them. This achieves a 3.1x reduction in the KV budget with no loss in accuracy, yielding over 4x speedup in both prefill and generation stages.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long Context Inference"
  - "Attention Acceleration"
  - "KV Cache Compression"
  - "K-means Clustering"
  - "Sparse Attention"
date: 2026-05-08
content_hash: 4e60ed38f730872e
---

# Squeezed Attention: Accelerating Long Context Length LLM Inference

**Conference**: ACL 2025  
**arXiv**: [2411.09688](https://arxiv.org/abs/2411.09688)  
**Code**: [GitHub](https://github.com/SqueezeAILab/SqueezedAttention)  
**Area**: LLM Efficiency  
**Keywords**: Long Context Inference, Attention Acceleration, KV Cache Compression, K-means Clustering, Sparse Attention  

## TL;DR

Ours proposes Squeezed Attention, which compresses the Key vectors of fixed contexts via offline K-means clustering. During inference, centroid matching is used to predict important Keys and compute exact attention solely on them. This achieves a 3.1x reduction in the KV budget with no loss in accuracy, yielding over 4x speedup in both prefill and generation stages.

## Background & Motivation

**Background**: Emerging LLM applications (such as document analysis, code generation, and long-form QA) need to process extremely long input contexts (tens of thousands to millions of tokens). The core bottleneck in long-context inference is that the computation and memory overhead of the attention mechanism scales linearly with sequence length. The memory footprint of the KV cache and the FLOPs of attention have become the primary obstacles for practical deployment.

**Limitations of Prior Work**: Existing long-context acceleration methods mainly fall into two categories: (1) KV cache compression (e.g., eviction, quantization), but these methods are generic and do not leverage specific application scenario characteristics; (2) sparse attention (e.g., local + global patterns), but these predefined patterns may not match the actual attention distribution. More importantly, prior methods do not distinguish between "fixed context" and "dynamic user input."

**Key Challenge**: In many practical applications (e.g., users repeatedly querying the same document, code assistants working on the same codebase), most of the input prompt is fixed, and only the user query is dynamic. Existing methods treat the fixed and dynamic parts identically, missing the opportunity for offline optimization of the fixed portion.

**Goal**: Focusing on the common pattern where "most of the context is fixed", this work aims to design a method to preprocess the KV cache of the fixed context in an offline stage, enabling online inference to compute attention by analyzing only a small number of important Keys.

**Key Insight**: It is observed that Key vectors in the fixed context possess a semantic clustering structure—Key vectors of semantically similar tokens are also close in the vector space. Representing a group of Keys by a cluster centroid allows for fast screening of semantically relevant Key clusters before computing exact attention.

**Core Idea**: Perform K-means clustering on the Key vectors of the fixed context offline. Online, compare the Query with centroids to predict the clusters containing important Keys, and compute exact attention only on these important Keys, reducing the complexity from linear to sublinear or even logarithmic.

## Method

### Overall Architecture

Squeezed Attention divides the process into offline and online stages. Offline stage: a single forward pass is executed on the fixed context to obtain the Key vectors of each attention head in every layer. Then, K-means clustering is conducted independently on the Key vectors of each head, storing the cluster centroids and the cluster assignments for each Key. Online stage: when a new user Query arrives, the similarity between the Query and each cluster centroid is computed to select the top-$k$ most relevant clusters. The original Keys (and corresponding Values) in these clusters are retrieved, and FlashAttention is executed to perform exact sparse attention.

### Key Designs

1. **Offline K-means Key Clustering**:

    - **Function**: Groups Key vectors of the fixed context by semantic similarity, generating a compressed representation (centroids).
    - **Mechanism**: Runs the K-means algorithm independently on the Key matrix $K \in \mathbb{R}^{n \times d}$ for each attention layer and each head, obtaining $C$ centroids $\mu_1, ..., \mu_C$ and the cluster ID for each Key. This offline step is executed only once and is re-run only when the user changes the document. The cluster count $C$ is the key hyperparameter that controls the accuracy-efficiency trade-off.
    - **Design Motivation**: Compared to directly evicting or quantizing the KV cache, the clustering method retains all original Keys but utilizes centroids as a fast index. This represents "organized compression" rather than "lossy compression".

2. **Online Centroid Matching and Important Key Selection**:

    - **Function**: Efficiently identifies the semantically most relevant Keys in the fixed context for each input Query token.
    - **Mechanism**: For a user-input Query vector $q$, its dot-product similarity $s_j = q^\top \mu_j$ with all cluster centroids is computed, and the top-$k$ clusters with the highest similarities are selected. Keys within these clusters are deemed "important Keys". Exact attention is then computed using only these important Keys and their corresponding Values: $\text{Attn}(q, K_{\text{imp}}, V_{\text{imp}})$. The complexity of centroid comparison is $O(C)$, which is much smaller than the original $O(n)$.
    - **Design Motivation**: Leverages the sparsity of attention scores where most tokens have attention weights near zero, and only a minority of keys are truly important. Clustering combined with centroid matching offers an efficient way to approximate top-$k$ attention.

3. **Hierarchical Squeezed Attention**:

    - **Function**: Further reduces the attention complexity from linear to logarithmic.
    - **Mechanism**: Constructs a multi-level clustering tree. The first level consists of a small number of coarse-grained clusters, within which fine-grained clustering is performed, forming a tree structure. During online inference, screening is performed first on the coarse-grained centroids, then on the fine-grained centroids of the selected clusters, progressively shrinking the search space. The overhead of each level's screening is proportional to the cluster count at that level, bringing the total overhead to $O(\log n)$.
    - **Design Motivation**: When the fixed context is extremely long (e.g., millions of tokens), even the centroid count in flat K-means can be excessively large. The hierarchical approach exponentially collapses the search space, rendering the method scalable to ultra-long contexts.

### Loss & Training

Squeezed Attention is a **training-free** inference acceleration method. K-means clustering operations are run in the offline stage without any backpropagation. Exact attention is used during online inference (albeit on a smaller subset of Keys), meaning no approximation error is introduced to the attention calculation itself—the only approximation lies in the Key selection phase. The authors also implement highly efficient Triton kernels (sparse FlashAttention and centroid comparison kernels) to achieve actual hardware acceleration.

## Key Experimental Results

### Main Results

| Model | Benchmark | KV Budget Reduction | Accuracy Loss | Description |
|------|------|-----------|---------|------|
| LLaMA-2-7B-32K | LongBench | 3.1x | No obvious loss | KV cache reduced to 1/3 |
| LLaMA-2-7B-32K | LongBench | 8x | Only 0.5 points drop | Aggressive compression is still acceptable |
| LWM-Text-Chat-1M | LongBench | 3.1x | No obvious loss | Equally effective on million-level context models |
| Longchat-7B-v1.5-32K | LongBench | 3.1x | No obvious loss | Consistently effective across models |
| All Models | Real Latency (prefill) | - | - | >4x speedup |
| All Models | Real Latency (generation) | - | - | >4x speedup |

### Ablation Study

| Configuration | LongBench Avg Score | Description |
|------|-----------------|------|
| Full Attention | Highest | Uncompressed baseline |
| Squeezed Attention (3.1x) | Close to Full | Nearly lossless accuracy |
| Squeezed Attention (8x) | Slightly lower (-0.5) | Acceptable sacrifice in accuracy |
| Uniformly Sampled Key (3.1x) | Significant drop | Poor performance of random Key selection |
| Fixed local+global pattern | Significant drop | Mismatched preset patterns |
| Hierarchical (2-level) | Close to flat | Logarithmic complexity, no accuracy drop |
| w/o Custom Triton Kernels | Same accuracy, slower speed | Kernel implementation is critical for real speedup |

### Key Findings

- A 3.1x KV compression ratio is the sweet spot for a "lossless" operation; 8x compression remains within an acceptable range.
- The hierarchical variant shows a more significant efficiency advantage on ultra-long contexts, with almost zero impact on accuracy.
- Custom Triton kernels (sparse FlashAttention + centroid comparison) are critical to achieving the 4x real speedup. Without kernel optimizations, algorithmic savings do not translate to real-world speedups.
- Key distributions vary significantly across different heads. Performing independent clustering head-by-head yields much better results than global clustering.
- Speedups are obtained in both prefill and generation stages, demonstrating that the method is universally applicable.

## Highlights & Insights

- **Scenario-Insight-Driven Design**: The work acutely identifies "fixed context + dynamic query" as an extremely common pattern in real-world applications (e.g., document QA, code assistants) and designs an offline-online optimization framework accordingly. This angle of attack is highly valuable on its own.
- **Training-Free Method**: No modifications to model parameters nor retraining are required; it is plug-and-play during inference, which significantly lowers deployment barriers.
- **End-to-End Actual Speedup**: Rather than just reducing computational load on an algorithmic level, the implementation of Triton kernels delivers real 4x+ speedups. This addresses a common pitfall in several "theoretical speedup" papers that lack practical kernel implementations.
- **Hierarchical Expansion**: The evolution from linear complexity to logarithmic complexity is highly elegant, providing a theoretical foundation for handling million-token-level contexts.

## Limitations & Future Work

- The offline clustering phase incurs its own overhead (K-means over massive Key vectors), which might not be cost-effective for scenarios where documents are frequently changed.
- It assumes that the fixed context constitutes a large portion of the input to obtain significant speedup; hence, it is inapplicable to purely dynamic scenarios (e.g., open-domain dialogue).
- The clustering quality is sensitive to K-means initialization, and the paper does not analyze the impact of different clustering algorithms in detail.
- Only 7B models have been validated so far. The efficacy and engineering challenges on larger models (such as 70B+) remain unexplored.
- Future work may explore: (1) combining clustering with KV cache quantization to achieve further compression; (2) incremental clustering updates to support scenarios where parts of the context change; (3) integration with parameter-efficient methods such as LoRA.

## Related Work & Insights

- **vs StreamingLLM**: StreamingLLM retains recent tokens and attention sinks, relying on a heuristic-based KV eviction. Squeezed Attention utilizes semantic clustering for precise selection, offering a stronger theoretical foundation.
- **vs H2O (Heavy Hitter Oracle)**: H2O selects important Keys based on accumulated attention scores, but this requires computing all attention scores online during selection. Squeezed Attention achieves lower overhead by dynamically filtering via centroid matching upfront.
- **vs Gisting/Compressor tokens**: These approaches compress contexts by training special tokens, which requires additional training. Squeezed Attention is training-free and more flexible.
- This method can serve as a robust baseline for long-context LLM inference acceleration, notably for "fixed context" scenarios such as document QA and coding assistants.

## Rating

- Novelty: ⭐⭐⭐⭐ While combining offline clustering with online sparse attention is not highly novel on its own, its systematic application and hierarchical expansion in the context of long-context LLMs represent an innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations were done across multiple models and compression ratios, including thorough testing on LongBench and real-world latency measurements.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, with a complete logical chain leading from observations to the proposed solution.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a core pain point in industrial long-context inference, supplemented by open-source code and complete kernel implementations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[ACL 2025\] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [\[ACL 2025\] RefreshKV: Updating Small KV Cache During Long-form Generation](refreshkv_updating_small_kv_cache_during_long-form_generation.md)
- [\[ACL 2025\] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](adaptive_grouped_pe_context_window.md)

</div>

<!-- RELATED:END -->
