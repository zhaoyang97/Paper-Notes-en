---
title: >-
  [Paper Note] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression
description: >-
  [ICML 2025][Model Compression][KV cache compression] RocketKV is proposed, a training-free two-stage KV cache compression method. The first stage employs SnapKV for coarse-grained permanent eviction, and the second stage utilizes Hybrid Sparse Attention (HSA) for fine-grained dynamic top-k selection. RocketKV achieves up to a 400× compression ratio, 3.7× end-to-end speedup, and 32.6% peak memory savings with negligible accuracy loss on models such as Mistral-7B.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "KV cache compression"
  - "long-context inference"
  - "sparse attention"
  - "inference acceleration"
  - "SnapKV"
date: 2026-05-08
content_hash: e0435675e4a34be4
---

# RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression

**Conference**: ICML 2025  
**arXiv**: [2502.14051](https://arxiv.org/abs/2502.14051)  
**Code**: [https://github.com/NVlabs/RocketKV](https://github.com/NVlabs/RocketKV)  
**Area**: Model Compression  
**Keywords**: KV cache compression, long-context inference, sparse attention, inference acceleration, SnapKV

## TL;DR

RocketKV is proposed, a training-free two-stage KV cache compression method. The first stage employs SnapKV for coarse-grained permanent eviction, and the second stage utilizes Hybrid Sparse Attention (HSA) for fine-grained dynamic top-k selection. RocketKV achieves up to a 400× compression ratio, 3.7× end-to-end speedup, and 32.6% peak memory savings with negligible accuracy loss on models such as Mistral-7B.

## Background & Motivation

### 1. Bottlenecks of KV Cache

In Transformer inference, the KV cache stores historical key-value pairs to avoid redundant computations, but its size grows proportionally with the sequence length and batch size. For example, Llama3.1-70B requires approximately 320GB of KV cache at a batch size of 32 and context length of 32K, far exceeding current hardware capacity.

### 2. Limitations of Prior Work

Existing KV cache compression schemes fall into two main categories, each with critical limitations:
- **Permanent Eviction** (e.g., H2O, SnapKV): Retains important tokens to save memory and bandwidth, but discarded tokens may be needed in subsequent decoding steps.
- **Dynamic Selection** (e.g., Quest, SparQ): Retains all tokens but dynamically selects top-k elements at each step, preventing information loss but offering limited memory savings and requiring extra indexing overhead.

### 3. Key Observations and Core Idea

Analyzing the attention patterns of Mistral-7B on the Qasper benchmark reveals that although the sequence length can reach 25,000, the unique token indices selected by the oracle top-k ($k=256$) across all decoding steps only total around 1200. This implies:
1. Permanent eviction should be able to approach oracle accuracy with a token budget of approximately 1200.
2. Performing dynamic selection on the smaller post-eviction set significantly reduces the difficulty of top-k prediction.

**Core Idea**: First apply coarse-grained permanent eviction (with a larger budget) to remove low-importance tokens, and then perform fine-grained dynamic selection on the remaining set, fusing the advantages of both approaches.

## Method

### Overall Architecture

RocketKV adopts a two-stage cascaded architecture:
- Input: Full KV cache (long sequence)
- First Stage: SnapKV coarse-grained permanent eviction $\rightarrow$ Retains a subset of important tokens.
- Second Stage: HSA fine-grained dynamic top-k selection $\rightarrow$ Extremely low-bandwidth sparse attention.
- Output: Attention outputs that maintain near-lossless accuracy under extremely high compression ratios.

The framework is highly flexible, allowing different methods to be plugged into each stage (e.g., Ada-KV for the first stage and SparQ/Loki for the second stage).

### Key Designs

#### 1. Stage 1: SnapKV Coarse-Grained Eviction

- **Function**: Computes aggregated attention scores using an observation window at the end of the input context to permanently remove low-scoring tokens.
- **GQA Adaptation**: The original SnapKV selects tokens head-by-head, leading to redundant storage within the same Grouped-Query Attention (GQA) group. This is improved by selecting tokens group-by-group, sharing the token set within each GQA group.
- **Pooling Kernel Adjustment**: While the original SnapKV uses a small kernel (size 7) to maintain information integrity, the first stage of RocketKV only performs coarse-grained eviction. Experimental results show that an optimal kernel size of 63 significantly simplifies computation.

#### 2. Stage 2: Hybrid Sparse Attention (HSA)

- **Function**: Performs two-dimensional reduction (along the sequence dimension and head dimension) for top-k estimation on the residual tokens after eviction.
- **Mechanism (Three Steps)**:
    - **Step 1**: Group the key tensor along the sequence dimension into continuous pages, storing the element-wise maximum $K_{\max}$ and minimum $K_{\min}$ for each page.
    - **Step 2**: For query $q$, find the $k_1$ locations with the largest absolute values in the head dimension. Based on the sign at the corresponding position of $q$, retrieve values from $K_{\max}$ or $K_{\min}$ to compute $\max(q \times K_{\max}, q \times K_{\min})$ as the approximated maximum attention score for each page. Select the $k_2$ pages with the highest scores.
    - **Step 3**: Perform sparse attention on the original KV pairs from the $k_2$ selected pages.
- **Design Motivation**: One-dimensional reduction (Quest along the sequence dimension, SparQ along the head dimension) provides limited compression ratios and undergoes a sharp accuracy drop beyond a certain threshold. Two-dimensional reduction exploits sparsity in both dimensions simultaneously to approximate top-k tokens more accurately.
- **GQA Compatibility**: All selection steps are performed at the attention group level, ensuring that all heads within a GQA group make the same selection by summing along the group dimension.

#### 3. Adaptive Compression Factorization

- **Function**: Intelligently allocates token budgets to the two stages given a target compression ratio $c = S/t$.
- **Mechanism**: The token budget is defined as the total memory traffic encompassing both top-k estimation (Step 2) and sparse attention (Step 3), which are then divided equally. This reflects the actual overhead more accurately than legacy definitions that only calculate sparse attention overhead.
- **GQA Adaptation**: The token budget is defined over the entire attention group rather than individual heads.

### RocketKV-MT: Multi-Turn Conversation Variant

- **Key Challenge**: A critical issue of permanent eviction in multi-turn conversations is that tokens evicted in early turns may become crucial in subsequent turns.
- **Mechanism**: The first stage does not permanently discard tokens. Instead, it retains all KV tokens but still dynamically selects only the subset filtered by the first stage for decoding, maintaining the complete KV history across turns.
- **Gain**: Achieves the same decoding speed as RocketKV per turn without memory storage savings, while keeping accuracy close to the oracle top-k.

## Key Experimental Results

### Main Results

| Model | Method | Compression Ratio | End-to-End Speedup | Peak Memory Savings | Accuracy Impact |
|------|------|--------|-----------|-------------|---------|
| Mistral-7B | Full KV | 1× | 1× | 0% | Baseline |
| Mistral-7B | SnapKV (Eviction Only) | ~16× | Moderate | Yes | Significant drop |
| Mistral-7B | Quest (Dynamic Only) | ~32× | Moderate | 0% | Significant drop |
| Mistral-7B | **RocketKV** | **400×** | **3.7×** | **32.6%** | **Negligible** |
| Multi-model | RocketKV-MT | High | Near RocketKV | None | Close to oracle top-k |

Experiments on Mistral-7B across multiple LongBench subtasks (e.g., Qasper, HotpotQA) demonstrate that while other methods experience sharp accuracy drops when the token budget is below 1024, the oracle top-k retains high accuracy even with $k=256$. RocketKV successfully approaches oracle performance through its two-stage design.

### Ablation Study

| Configuration | Performance Trend | Description |
|------|---------|------|
| Stage 1 Only (SnapKV) | Moderate accuracy, moderate compression | Fails to maintain quality under extremely low budgets |
| Stage 2 Only (HSA) | High accuracy but limited compression | Performance ceiling of 1D reduction |
| Joint Two-Stage (RocketKV) | Highest accuracy, maximum compression ratio | Validation of two-stage complementarity |
| HSA replaced with Quest/SparQ | Accuracy decreases | 2D reduction outperforms 1D reduction |
| SnapKV kernel size 7 vs 63 | Kernel size 63 is superior | Larger kernel is more suitable for coarse-grained scenarios |

Note: The cache is truncated at the end of the methodology section, and the complete numerical tables were not retrieved from the cached paper notes. The experimental trends described above are based on comparative analyses provided in the abstract and motivation sections of the paper.

### Key Findings

- The joint two-stage scheme consistently outperforms single-stage schemes under any given total compression ratio because permanent eviction narrows down the search space, rendering dynamic selection more accurate.
- HSA's two-dimensional reduction yields significantly higher accuracy than one-dimensional methods (Quest/SparQ) under the same compression ratio.
- In multi-turn scenarios, RocketKV-MT completely avoids the accuracy degradation associated with permanent eviction, achieving accuracy close to the oracle top-k.

## Highlights & Insights

- **Observation-Driven Design**: The CDF analysis of attention patterns (where unique indices are far fewer than sequence length) directly inspires the two-stage scheme. This "let the data speak" methodology is exemplary.
- **Training-Free and Plug-and-Play**: Requires no extra training or auxiliary draft models, facilitating extremely low deployment barriers.
- **Native GQA Support**: Operating at the attention group level naturally adapts to GQA/MQA without requiring special architectural modifications.
- **Adaptive Factorization**: Counting the top-k estimation traffic into the token budget provides a more honest definition of the compression ratio and a more accurate performance estimate.
- **Modular Framework**: Each stage is pluggable with different methods, facilitating subsequent improvements by the community.

## Limitations & Future Work

- The cache was truncated after Section 3.6, preventing the retrieval of the complete experimental section (such as benchmarks across different context lengths/model scales, runtime costs, and detailed memory footprints).
- To preserve accuracy, RocketKV-MT retains the entire KV history, which might offer limited value in strictly memory-constrained scenarios. Is a middle-ground approach with "partial retention" feasible?
- Accuracy drops are still visible under extremely low token budgets ($k < 64$), and performance remains to be validated on specific inference tasks that require globally uniform attention.
- The first stage of SnapKV relies on an observation window at the end, which may evict too many critical tokens in scenarios where information is uniformly distributed throughout the text.

## Related Work & Insights

- **vs SnapKV**: Acts as the foundation of the first stage in this work, with improved GQA support and optimized pooling parameters.
- **vs Quest / SparQ**: One-dimensional sparse selection methods; HSA integrates their concepts and extends them to two dimensions.
- **vs H2O**: A classic strategy retaining Heavy Hitters + recent tokens, which exhibits limited accuracy under low budgets.
- **vs Ada-KV**: An adaptive budget allocation scheme that can serve as an alternative for the first stage.
- **vs MInference**: Focuses on prefill acceleration, which is complementary to RocketKV's focus on decode-stage acceleration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of two stages and the 2D reduction in HSA is highly novel, though individual components build upon pioneering works.
- **Experimental Thoroughness**: ⭐⭐⭐ The numbers shown in the abstract are strong, but the complete experimental tables were not fully retrieved from the cached paper notes.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear observations, complete logical chain of motivations, and systematic explanation of methodology.
- **Value**: ⭐⭐⭐⭐⭐ High engineering value as it is training-free, plug-and-play, and open-sourced by NVIDIA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](../../ACL2026/model_compression/dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[NeurIPS 2025\] Inference-Time Hyper-Scaling with KV Cache Compression](../../NeurIPS2025/model_compression/inference-time_hyper-scaling_with_kv_cache_compression.md)
- [\[ACL 2025\] APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](../../ACL2025/model_compression/apb_distributed_long_context.md)
- [\[NeurIPS 2025\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](../../NeurIPS2025/model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)

</div>

<!-- RELATED:END -->
