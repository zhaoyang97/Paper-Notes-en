---
title: >-
  [Paper Note] APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
description: >-
  [ACL 2025][Model Compression][long-context inference] APB proposes a distributed long-context inference framework. By introducing local KV cache compression and a mechanism to pass compressed context blocks across GPUs into the sequence parallelism framework, it achieves up to 9.2x, 4.2x, and 1.6x prefill speedup compared to FlashAttn, RingAttn, and StarAttn, respectively, without compromising task performance.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "long-context inference"
  - "sequence parallelism"
  - "KV cache compression"
  - "distributed inference"
  - "approximate attention"
date: 2026-05-08
content_hash: 74d9728dc1c12dde
---

# APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

**Conference**: ACL 2025  
**arXiv**: [2502.12085](https://arxiv.org/abs/2502.12085)  
**Code**: [https://github.com/thunlp/APB](https://github.com/thunlp/APB)  
**Area**: Model Compression  
**Keywords**: long-context inference, sequence parallelism, KV cache compression, distributed inference, approximate attention

## TL;DR
APB proposes a distributed long-context inference framework. By introducing local KV cache compression and a mechanism to pass compressed context blocks across GPUs into the sequence parallelism framework, it achieves up to 9.2x, 4.2x, and 1.6x prefill speedup compared to FlashAttn, RingAttn, and StarAttn, respectively, without compromising task performance.

## Background & Motivation
**Background**: As the supported context length of LLMs continues to grow (e.g., Llama-3.1 128K, Claude-3.5 200K, MiniMax-01 4M), the prefill stage of long-context inference has become a critical bottleneck, with the attention computation complexity being $O(n^2)$.

**Limitations of Prior Work**:
   - Sequence parallelism (e.g., RingAttn, Ulysses) increases parallelization but keeps the computation load unchanged, remaining limited by full attention calculation.
   - Approximate attention mechanisms (e.g., H2O, SnapKV) reduce computation load but rely on global sequence information for pruning, which conflicts with distributed architectures.
   - StarAttn simply combines approximate attention and sequence parallelism, but its performance continuously deteriorates as the number of GPUs increases (due to the decreased proportion of context visible to each host).

**Key Challenge**: Existing KV cache compression methods require global attention scores to determine which KV pairs are important. However, in sequence parallelism, each host only holds a portion of the context, making it impossible to obtain a global view.

**Goal**: Design an approximate attention mechanism within the sequence parallelism framework that reduce both computation and communication overhead, while maintaining stable performance as the number of GPUs increases.

**Key Insight**: Utilize local KV cache compression (without requiring a global view) along with passing the compressed key-value (KV) pairs across GPUs.

**Core Idea**: Each GPU independently compresses its local KV cache and then transmits only the compressed key context to other GPUs, achieving a dual reduction in both computation and communication.

## Method

### Overall Architecture
APB inference consists of four phases: Context Splitting $\rightarrow$ Block Compression $\rightarrow$ Communication $\rightarrow$ Computation. The input sequence is split into a document $d$ and a query $q$. The document is uniformly distributed across $H$ GPUs, and an anchor block is prepended to the local block on each GPU. The compressed KV pairs are communicated via AllGather to construct passing blocks for subsequent GPUs' reference.

### Key Designs

1. **Anchor Block**:

    - **Function**: To provide a global context anchor of the starting document information for each host.
    - **Mechanism**: Concatenate the query $q$ and the first $l_a$ tokens of the document to form the anchor block $\mathbf{A} = \{q_1, \ldots, q_{l_q}, d_1, \ldots, d_{l_a}\}$, which is prepended to each host's local block.
    - **Design Motivation**: Unlike StarAttn, APB uses a smaller anchor ($l_a = \frac{1}{4}l_b$ or $\frac{1}{8}l_b$, whereas StarAttn uses $l_a = l_b$) because cross-host information is already supplemented by the passing blocks. Embedding the query into the anchor allows the compressor to identify key KV pairs based on query relevance.

2. **Block Compression**:

    - **Function**: Independently compress the local KV cache on each host, retaining the most critical $l_p$ KV pairs.
    - **Mechanism**: Use the retaining heads $\mathcal{R}$ of Locret—a small MLP that takes $[\mathbf{Q}, \mathbf{K}, \mathbf{V}]$ as input and outputs importance scores for each token, selecting the Top-$l_p$ to retain. Key formula: $\mathbf{B}_h^C = \text{Top-}l_p(\mathcal{R}([\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h]))$
    - **Design Motivation**: H2O and SnapKV require global attention scores, which are unavailable in distributed scenarios. Locret evaluates token importance based solely on each token's own QKV, allowing it to be executed entirely locally, thereby addressing Challenge 1.

3. **AllGather Communication + Passing Block Construction**:

    - **Function**: Gather the compressed KV caches of all hosts and construct a passing block for each host containing all preceding compressed contexts.
    - **Mechanism**: After AllGather communication, the passing block of host $h$ is defined as $\mathbf{P}_h = (\mathbf{K}_{[1:h-1]}^C, \mathbf{V}_{[1:h-1]}^C)$, which concatenates the compressed KV pairs of all preceding hosts. Due to the compression, the communication volume is significantly reduced.
    - **Design Motivation**: To ensure that each host is aware of the critical information in the global context, which resolves Challenge 2—allowing each host to still access the compressed representations of all preceding contexts through the passing blocks even as the number of hosts increases.

4. **Customized FlashAttn Kernel + Modified Attention Mask**:

    - **Function**: Calculate attention over the concatenated KV of the three parts (anchor + passing + local) using a modified attention mask.
    - **Mechanism**: $\mathbf{Q} = [\mathbf{Q}_a, \mathbf{Q}_h]$ and $\mathbf{K} = [\mathbf{K}_a, \mathbf{K}_p^C, \mathbf{K}_h]$; computation is performed using a customized FlashAttn kernel. The passing block is discarded after attention calculation and does not participate in the FFN computation.
    - **Difference from StarAttn**: StarAttn relies solely on the anchor block to provide cross-host information, whereas APB additionally transmits compressed context via passing blocks, providing richer information while keeping communication overhead controllable.

### Loss & Training
- The APB framework itself is training-free; however, the KV cache compressor (retaining heads) must be pre-trained on long-context SFT data.
- The training cost is highly minimal—it is only a small MLP and does not require modifying the original LLM parameters.

## Key Experimental Results

### Main Results

**∞Bench Long-Context Tasks (128K tokens, Llama-3.1-8B)**:

| Method | R.PassKey | R.Number | R.KV | Avg. |
|------|:---------:|:--------:|:----:|:----:|
| FullAttn | 100.00 | 99.49 | 51.00 | 47.45 |
| MInference | 98.47 | 98.81 | 17.40 | 43.61 |
| StarAttn | 100.00 | 98.98 | 40.60 | 46.48 |
| **APB** | **100.00** | **98.81** | **81.80** | **50.91** |

APB's average score exceeds FullAttn by 3.46 points, achieving a substantial lead in the R.KV task (81.8 vs. 51.0).

**RULER Benchmark（Llama-3.1-8B, 128K）**:

| Method | MK2 | MK3 | MV | Avg. |
|------|:---:|:---:|:--:|:----:|
| FullAttn | 87.60 | 67.00 | 94.65 | 82.20 |
| StarAttn | 73.60 | 53.00 | 72.80 | 76.84 |
| **APB** | **91.00** | **89.00** | **95.05** | **81.63** |

**Inference Speed Comparison (Llama-3.1-8B, 4 GPUs)**:

| Method | 128K tokens/s | 256K tokens/s | Speedup vs. FlashAttn |
|------|:------------:|:------------:|:-------------------:|
| FlashAttn | baseline | baseline | 1.0x |
| RingAttn | ~2.2x | ~2.2x | 2.2x |
| StarAttn | - | - | ~5.8x |
| **APB** | - | - | **~9.2x** |

### Ablation Study

| Configuration | ∞Bench Avg. | RULER Avg. | Description |
|------|:----------:|:---------:|------|
| Full model (anchor=1/4, pass=1/4) | 50.91 | 81.63 | Complete APB |
| w/o passing block | ~46.5 | ~77 | Degenerates to a StarAttn variant |
| anchor=1/8, pass=1/4 | ~49 | ~80 | Smaller anchor, slight degradation |
| Random compressor | Significant degradation | Significant degradation | Validates effectiveness of Locret compressor |

### Key Findings
- APB even outperforms FullAttn on several tasks (especially R.KV retrieval). This is because compression filters out noisy KV pairs, which in turn enhances retrieval accuracy.
- The passing block is a key contribution—without it, APB degenerates into a StarAttn-like scheme, causing performance to degrade as the number of GPUs increases.
- The compression ratio (the $l_p/l_b$ ratio) is flexibly configurable, achieving a controllable trade-off between speed and accuracy.
- APB supports co-configuration of TP (Tensor Parallelism) + SP (Sequence Parallelism), making it compatible with various model architectures.

## Highlights & Insights
- **Co-design of Compression and Distribution**: Instead of simply stacking KV cache compression on top of sequence parallelism, a complete anchor-compress-pass-compute pipeline is designed to ensure information flow integrity. This system-algorithm co-design approach is highly instructive.
- **Clever Choice of Local Compression**: Adapting Locret, which does not rely on a global view, perfectly satisfies the constraints of distributed scenarios—showing a very high alignment between the problem and the solution.
- **Surpassing Exact Attention**: Compressed KV cache is shown to outperform full attention in retrieval tasks (due to denoising effects). This is an inspiring finding, suggesting that exact attention is not always optimal.

## Limitations & Future Work
- **Optimizing Only the Prefill Stage**: The decoding stage still utilizes standard attention, offering limited benefits for long-generation scenarios. Extending the passing block mechanism to the decoding stage is a potential future direction.
- **Extra Training Required for Locret**: Each new model requires training its corresponding retaining heads MLP, which increases deployment costs.
- **Anchor + Query Prepending Assumption**: This requires that the input can be distinctly partitioned into a document and a query, which may not be suitable for scenarios like multi-turn conversations where queries are scattered throughout the context.
- **Inter-GPU Communication Remains a Bottleneck**: Although compression reduces communication volume, AllGather may still become a bottleneck in large-scale clusters.

## Related Work & Insights
- **vs. RingAttn**: RingAttn maintains exact attention but offers no reduction in computation, whereas APB reduces computation through approximate attention; this can be viewed as an exact vs. approximate trade-off.
- **vs. StarAttn**: StarAttn relies solely on the anchor block to provide cross-host information (using full-sized anchors), while APB employs a smaller anchor combined with compressed passing blocks, providing richer information with much smaller overhead.
- **vs. MInference**: MInference approximates attention by analyzing sparse patterns, which performs poorly in distributed settings; APB delivers a distributed-friendly approximation via Locret.
- The "compress-then-communicate" paradigm from this work can be ported to other distributed computing scenarios (e.g., distributed RAG, multi-agent communication).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of local compression and distributed passing is novel, though individual components (Locret, sequence parallelism) are relatively mature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across three models, two benchmarks, and two dimensions (speed and accuracy), with comprehensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definition, intuitive framework illustrations, and complete algorithmic pseudocode.
- **Value**: ⭐⭐⭐⭐⭐ Accelerating long-context inference is a core demand. The 9.2x speedup without performance loss holds significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](../../ICML2025/model_compression/rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](moqae_mixed_precision_kv_cache.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](../../ACL2026/model_compression/dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](efficient_long_context_language_model_retrieval_with_compression.md)

</div>

<!-- RELATED:END -->
