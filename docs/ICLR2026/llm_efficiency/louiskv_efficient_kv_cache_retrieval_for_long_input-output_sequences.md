---
title: >-
  [Paper Note] LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences
description: >-
  [ICLR 2026][LLM Efficiency][KV Cache Retrieval] LouisKV observes that key KVs exhibit strong temporal locality during decoding and distinct distribution patterns across input and output sequences. By replacing "per-token retrieval + page-level management" with "semantic boundary-triggered retrieval + decoupled fine-grained management (input clustering/output segmenting)," it achieves up to a 4.7× speedup over SOTA retrieval methods on various long-sequence tasks with near-zer…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "KV Cache Retrieval"
  - "Long-sequence Inference"
  - "Temporal Locality"
  - "Semantic Clustering"
  - "Long-output Inference"
date: 2026-05-08
content_hash: 8e3894f8b9c7df40
---

# LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=6RJ8fZwm4P](https://openreview.net/forum?id=6RJ8fZwm4P)  
**Code**: TBD  
**Area**: LLM Inference Acceleration / KV Cache Optimization  
**Keywords**: KV Cache Retrieval, Long-sequence Inference, Temporal Locality, Semantic Clustering, Long-output Inference  

## TL;DR
LouisKV observes that key KVs exhibit strong temporal locality during decoding and distinct distribution patterns across input and output sequences. By replacing "per-token retrieval + page-level management" with "semantic boundary-triggered retrieval + decoupled fine-grained management (input clustering/output segmenting)," it achieves up to a 4.7× speedup over SOTA retrieval methods on various long-sequence tasks with near-zero accuracy loss.

## Background & Motivation
**Background**: Autoregressive LLMs use KV cache to avoid redundant computation, but its memory footprint grows near-linearly with sequence length, causing memory exhaustion in long-sequence scenarios. To mitigate this, KV retrieval methods store the full KV cache in CPU memory and dynamically fetch a small subset of critical KVs to the GPU for attention calculations. Compared to KV dropping (which permanently discards tokens), retrieval retains full information and maintains higher accuracy.

**Limitations of Prior Work**: Existing KV retrieval methods (Quest, Arkvale, ClusterKV, etc.) face two bottlenecks. First, **every decoded token triggers a retrieval**, involving both selection overhead and CPU-to-GPU KV transfer. This cost is multiplied in modern long-output reasoning models (e.g., CoT generating 16K–32K tokens). Second, they generally adopt **page-level KV management**, where fixed-size pages contain many non-critical KVs. Transitioning entire pages wastes bandwidth and can degrade precision.

**Key Challenge**: To improve efficiency, both retrieval frequency and transmission volume must be reduced, while precisely identifying critical KVs is necessary to maintain accuracy. Coarse-grained "per-token + page-level" strategies struggle on both fronts, particularly in long-output scenarios.

**Goal**: To design an efficient KV retrieval framework that unifies long-input/short-output, short-input/long-output, and long-input/long-output scenarios, achieving significant speedup with minimal accuracy loss.

**Key Insight**: The authors made two pivotal observations—**(Obs 1: When to retrieve)** The sets of critical KV indices for adjacent decoding tokens are highly similar (Jaccard similarity >0.8 within the same reasoning step), indicating strong **temporal locality**. Thus, per-token retrieval is unnecessary. **(Obs 2: How to manage)** Critical KVs are sparsely distributed in long input sequences (extracting scattered info from context) but densely clustered in long output sequences (repeatedly attending to recent intermediate steps). Based on these, the authors propose **semantic-aware adaptive retrieval** (triggered only at semantic boundaries) + **decoupled fine-grained management** (clustering for input, segmenting for output).

## Method

### Overall Architecture
LouisKV decouples "when to retrieve" from "how to organize KV units." It uses a lightweight similarity signal between adjacent queries to detect semantic boundary crossings, triggering retrieval only at these boundaries and reusing retrieved KVs within the segment. Simultaneously, it applies K-means clustering to input KVs during the prefill phase and partitions output KVs into segments based on semantic boundaries during the decode phase. These retrieval units are offloaded to a CPU cache pool and selected precisely based on centroid similarity. The system is further accelerated by Triton/CUDA kernel-level optimizations.

```mermaid
flowchart TD
    A[Decoding Step t: Compute query qt] --> B{r = cos(qt, qt-1) < Threshold τ?}
    B -- No, same semantic segment --> C[Reuse previously retrieved KVs<br/>Skip retrieval]
    B -- Yes, crossed boundary --> D[Trigger Retrieval: Compare qt with unit centroids]
    D --> E[Fetch Top-B critical units from CPU to GPU]
    C --> F[Attention + FFN]
    E --> F
    subgraph Decoupled Fine-grained Management
    G[Prefill Input: K-means clustering<br/>→ Semantic clusters + centroids] --> P[(Unified CPU KV Cache Pool)]
    H[Decode Output: Partition by semantic boundaries<br/>Offload oldest segment when buffer full] --> P
    end
    P --> E
```

### Key Designs

**1. Semantic-aware Adaptive Retrieval: Detecting boundaries via query similarity**—Observation 1 suggests that critical KVs remain nearly constant within a semantic segment. Thus, retrieval costs can be amortized across segments. Instead of directly comparing critical KV index sets (which is computationally expensive), LouisKV uses a cheap proxy: the average cosine similarity of the current and previous query across all heads:
$$r_t = \frac{1}{H}\sum_{h=1}^{H}\cos(q_{t-1}^h, q_t^h)$$
When $r_t$ falls below a preset threshold $\tau$, a semantic boundary is detected, triggering one retrieval (using $q_t$ to select the most critical units via centroid similarity). Otherwise, the previous segment's KVs are reused. $\tau$ is calibrated once per model (e.g., 0.7 for Qwen3-8B) and generalizes across tasks.

**2. Decoupled Fine-grained KV Management: Clustering inputs and segmenting outputs**—Observation 2 identifies structural differences between input and output attention. **Prefill phase** uses "cluster-granularity": since attention scores depend on the dot product of query and key, similar keys produce similar attention. Keys are clustered via K-means, and the cluster mean serves as the centroid index. This is more precise than fixed-page slicing and overlaps asynchronously with forward computation. **Decode phase** uses "segment-granularity": generated KVs are sliced into temporal segments based on the detected semantic boundaries. Only a fixed-size local buffer is kept on GPU; older segments are offloaded to CPU. This dual-strategy minimizes the transfer of non-critical KVs.

**3. Kernel and System Optimization**—Three engineering optimizations are implemented. First, custom **Triton** kernels accelerate K-means clustering during prefill, hiding overhead within forward computation (TTFT increases by only ~3.5%). Second, **Group-Consistent Selection** identifies a unified set of clusters/segments for entire query groups to fit GQA, reducing redundant data movement. Third, highly optimized **CUDA** kernels support fast budget-based KV selection, and the DGL library is used to transfer specific rows of CPU tensors directly to the GPU, bypassing intermediate CPU aggregation.

## Key Experimental Results

### Main Results (Accuracy, Fixed KV Budget)

| Scenario | Comparison | LouisKV Gain/Advantage |
|------|----------|------------------|
| Long-Input Short-Output (LongBench 6, Budget 512) | Arkvale / Quest | Avg score +1.1% / +3.3%; only ~0.4% lower than ClusterKV (with much higher efficiency) |
| Long-Output Reasoning (MATH500/GPQA/AIME, Budget 1024) | Arkvale / Quest / ClusterKV | Avg Acc +2.0% / +4.8% / +3.8%; KV dropping (H2O) fails significantly on AIME |
| Ultra-long Context (RULER 32K→128K, Budget 2K) | Arkvale | +2.6% / +4.7% at 32K/64K, near FullCache; 128K remains feasible while FullCache OOMs |

### Efficiency Results

| Metric | Result |
|------|------|
| E2E Speedup (vs Arkvale) | 1.9× (Long-In) / 2.9× (Short-In Long-Out) / **4.7×** (Long-In Long-Out) |
| Decoding Latency Breakdown | Arkvale retrieval (selection+transfer) 65%; LouisKV reduced to 11% |
| Prefill TTFT (vs FullCache) | Avg increase ~3.5% (hidden via K-means overlap) |
| Memory | FullCache OOMs at large batch/long sequence; LouisKV enables high throughput |

### Ablation Study

| Aspect | Conclusion |
|--------|------|
| Adaptive vs Fixed Step | Adaptive improves accuracy by up to 8% on long-output tasks. |
| Decoupled vs Page-level | Fine-grained management improves accuracy by up to 6.3%. |
| Threshold $\tau$ | 0.7 is the optimal balance for Qwen3-8B; generalizes across tasks. |
| System Optimizations | Semantic-aware retrieval (SR) contributes most (~2.6× speedup). |

### Key Findings
- Retrieval overhead (selection and transfer) is the primary bottleneck in long-sequence KV retrieval (65% in Arkvale); amortizing this via semantic boundaries is the main driver for speedup.
- Input/output distribution differences (sparse vs. dense) are significant; decoupled clustering and segmenting optimize both ends.
- $\tau$ is a model-level calibration parameter that is task-agnostic, making it user-friendly.

## Highlights & Insights
- **Observation-driven Design**: Two observations (temporal locality + distribution variance) map cleanly to two designs (boundary triggering + decoupled management).
- **Addressing Long-output Scenarios**: As large reasoning models become mainstream, adding segment-based management for long outputs fills a gap left by prior work focused on static long inputs.
- **Algorithm-System Co-design**: Theoretical gains are realized through Triton/CUDA kernels, group-consistent selection, and row-based DGL transfers, achieving a practical 4.7× speedup.
- **Low-cost Boundary Detection**: Using cosine similarity of adjacent queries as a proxy for expensive KV index set comparison is a clever engineering trade-off.

## Limitations & Future Work
- While $\tau$ is a one-time calibration, it requires a validation set and is model-specific; cross-model transferability requires more discussion.
- Reliability of cosine similarity as a proxy in open-ended generation tasks, where boundaries might be less distinct, is not fully explored.
- Prefill calculation for K-means, although overlapped, might become a bottleneck in environments where prefill resources are highly constrained.
- Experiments were performed mostly on PCIe 4.0; conclusions on higher/lower bandwidth interconnects need verification.

## Related Work & Insights
- **KV Dropping** (H2O, RaaS): Permanently discards tokens based on attention scores; computationally cheap but causes irreversible information loss and performance drops in reasoning tasks.
- **KV Retrieval** (Quest, Arkvale): Keeps full KV on CPU and fetches subsets; accurate but limited by per-token and page-level overhead. LouisKV directly addresses these limitations.
- **Granular Retrieval** (ClusterKV, SentenceKV): Attempts semantic clustering or syntax boundaries; mostly focuses on static inputs. LouisKV's decoupled management completes this line of research for dynamic outputs.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual observations of temporal locality and input/output variance are intuitive yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various benchmarks, budgets, and ultra-long contexts; provides a clear efficiency breakdown.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from observation to design to experiment.
- Value: ⭐⭐⭐⭐ Strong practical value for long-output reasoning, delivering high speedups with minimal accuracy loss.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference](freekv_boosting_kv_cache_retrieval_for_efficient_llm_inference.md)
- [\[ICLR 2026\] IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs](icecache_memory-efficient_kv-cache_management_for_long-sequence_llms.md)
- [\[ICLR 2026\] ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing](rest-kv_robust_kv_cache_eviction_with_layer-wise_output_reconstruction_and_spati.md)
- [\[ICLR 2026\] ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](thinkv_thought-adaptive_kv_cache_compression_for_efficient_reasoning_models.md)
- [\[ICML 2026\] CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](../../ICML2026/llm_efficiency/criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)

</div>

<!-- RELATED:END -->
