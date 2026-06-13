---
title: >-
  [Paper Note] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference
description: >-
  [ACL 2026][LLM Efficiency][KV Cache Compression] This paper proposes StructKV, a structure-aware KV cache compression framework. It identifies global information hubs by accumulating cross-layer attention patterns via Gl…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "KV Cache Compression"
  - "Long-Context Inference"
  - "Global In-Degree Centrality"
  - "Dynamic Pivot Detection"
  - "Structural Propagation"
date: 2026-05-08
content_hash: 05812ff89355403e
---

# StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.06746](https://arxiv.org/abs/2604.06746)  
**Code**: None  
**Area**: Model Efficiency / KV Cache Compression  
**Keywords**: KV Cache Compression, Long-Context Inference, Global In-Degree Centrality, Dynamic Pivot Detection, Structural Propagation

## TL;DR

This paper proposes StructKV, a structure-aware KV cache compression framework. It identifies global information hubs by accumulating cross-layer attention patterns via Global In-Degree Centrality, adaptively locates the optimal compression layer through Dynamic Pivot Detection, and separates computation from storage budgets via Structural Propagation & Decoupling. StructKV achieves near full-context performance on LongBench and RULER with only 60% prefill and 10% KV cache.

## Background & Motivation

**Background**: LLM context windows have expanded to over a million tokens, but inference efficiency faces dual bottlenecks: the $O(N^2)$ attention computational complexity during the prefill phase and the linear memory growth of the KV cache during the decoding phase. Existing methods typically address only one of these stages.

**Limitations of Prior Work**: (1) Decoding-only methods (StreamingLLM, SnapKV) only compress the KV cache without reducing prefill computation; (2) Prefill-aware methods (GemFilter, FastKV) rely on local saliency from single-layer attention snapshots to select tokens, but some tokens may be temporarily "dormant" in the selected layer while maintaining critical structural importance globally; (3) FastKV uses a fixed pruning layer (e.g., Layer 15), a hyperparameter that does not generalize across different model architectures or depths.

**Key Challenge**: Local saliency (single-layer snapshot) $\neq$ structural importance (cross-layer semantic role). A token might have low attention in a specific layer but act as an information hub across the network's depth. Once discarded by local snapshot methods, this information is permanently lost.

**Goal**: To design a structure-aware compression framework that identifies the "structural skeleton" of the context, ensuring tokens are preserved even if they are not locally salient.

**Key Insight**: The true importance of a token is defined by its cumulative contribution throughout the network's depth, which can be formalized using in-degree centrality from graph theory.

**Core Idea**: Cross-layer attention scores are accumulated to form Global In-Degree Centrality. Information theory metrics are used to adaptively detect the "phase transition" layer where attention stabilizes as the compression point. Computation retention and storage retention are decoupled to optimize prefill speed and decoding memory independently.

## Method

### Overall Architecture

StructKV inference proceeds in three phases: (1) **Phase 1 (Full Context Processing)**—The first $L^*$ layers process the full context while accumulating the global centrality score $\mathcal{C}_{global}$; (2) **Phase 2 (Structural Phase Transition)**—An automatic detector triggers at the optimal layer $L^*$, filtering the context using $\mathcal{C}_{global}$ and decoupling the computation budget $R_{struct}$ from the storage budget $R_{KV}$; (3) **Phase 3 (Compressed Inference)**—Deep layers operate only on the pruned "structural skeleton."

### Key Designs

1.  **Global In-Degree Centrality**:
    *   **Function**: Identifies tokens with global structural importance across layers.
    *   **Mechanism**: At each layer $l$, local saliency is calculated as $\mathcal{S}_j^{(l)} = \sum_{g=1}^{G} \left( \frac{1}{w} \sum_{t=N-w}^{N} \sum_{h \in \mathcal{H}_g} a_{t,j}^{(l,h)} \right)$. These are then recursively accumulated with exponential decay: $\mathcal{C}_j = \sum_{l=0}^{L^*} \lambda^{(L^*-l)} \cdot \mathcal{S}_j^{(l)}$. A decay factor $\lambda=0.9$ assigns higher weight to deeper semantic layers.
    *   **Design Motivation**: Unlike direct truncation using a single-layer $\mathcal{S}_j^{(l)}$, global accumulation ensures that a token acting as an information "hub" in several early layers retains a high centrality score even if it is temporarily dormant in a specific layer.

2.  **Dynamic Pivot Detection**:
    *   **Function**: Adaptively locates the optimal compression layer, removing dependence on fixed hyperparameters.
    *   **Mechanism**: Tracks three metrics—Attention Entropy $\mathcal{H}_l$ (distribution uncertainty), Sparsity $\rho_l$ (top-k cumulative probability mass), and Variance $\mathcal{V}_l$ (distinguishability). Normalized gradients are combined into a transition score $\mathcal{T}_l = w_1 \cdot \bar{\nabla}(-\mathcal{H}_l) + w_2 \cdot \bar{\nabla}(\rho_l) + w_3 \cdot \bar{\nabla}(\mathcal{V}_l)$. The optimal layer is $L^* = \arg\max_l \mathcal{T}_l + 1$.
    *   **Design Motivation**: Experiments show the optimal layer varies with model depth (Layer 12 for Qwen-2.5-7B, Layer 28 for 32B). Fixed-layer methods (like FastKV’s Layer 15) fail to generalize. Automatic detection compresses at the phase transition point where attention shifts from "broad exploration" to "focused extraction."

3.  **Structural Propagation & Decoupling**:
    *   **Function**: Separates the optimization of computational efficiency from storage efficiency.
    *   **Mechanism**: At layer $L^*$, the top-K tokens are selected based on global centrality to form the structural skeleton $\mathcal{I}_{struct} = \text{top-k}(\mathcal{C}, N \cdot R_{struct}) \cup \mathcal{I}_{win}$. Deep layers compute only on this reduced set. The KV cache independently uses local saliency for selection: $\mathcal{I}_{KV}^{(l)} = \text{top-k}(\mathcal{S}^{(l)}, N \cdot R_{KV}) \cup \mathcal{I}_{win}$. The structural retention rate $R_{struct}$ can be significantly larger than the storage retention rate $R_{KV}$.
    *   **Design Motivation**: In coupled settings, aggressive compression leads to accuracy collapse (a 10% retention rate yields only 45.3); with decoupling, $R_{struct}=20\%, R_{KV}=10\%$ recovers +13.8 points, entering a secure high-fidelity zone.

### Loss & Training

StructKV is a training-free inference-time compression method. Default parameters: window $w=8$, decay $\lambda=0.9$, transition weights $\{w_1, w_2, w_3\}=\{0.2, 0.3, 0.5\}$, $R_{struct}=20\%$, $R_{KV}=10\%$.

## Key Experimental Results

### Main Results

**LongBench (LLaMA-3.1-8B-Instruct, Average of 16 sub-tasks)**

| Method | Prefill | KV | Avg Score |
| :--- | :--- | :--- | :--- |
| Full-context | 100% | 100% | 49.33 |
| StreamingLLM | 100% | 10% | 41.59 |
| SnapKV | 100% | 10% | 46.92 |
| GemFilter | 60% | 10% | 40.40 |
| FastKV | 60% | 10% | 47.59 |
| **StructKV** | **60%** | **10%** | **48.61** |
| **StructKV** | **60%** | **20%** | **48.97** |

**RULER (LLaMA-3.1-8B-Instruct, Retrieval Benchmark)**

| Method | 8K | 16K | 32K | 64K | 128K | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Full-context | 90.1 | 95.0 | 83.4 | 85.5 | 76.3 | 86.0 |
| SnapKV | 75.6 | 76.8 | 72.9 | 75.0 | 67.7 | 73.6 |
| FastKV | 77.8 | 77.3 | 77.2 | 77.4 | 68.2 | 75.6 |
| **StructKV** | **81.3** | **82.5** | **81.8** | **81.5** | **73.6** | **80.1** |

### Ablation Study

**Sensitivity of Decay Factor $\lambda$ (LongBench, 10% KV)**

| $\lambda$ | Avg Score | Change |
| :--- | :--- | :--- |
| 0.50 | 47.41 | -1.20 |
| 0.80 | 48.35 | -0.26 |
| **0.90** | **48.61** | **Ref** |
| 0.95 | 48.42 | -0.19 |
| 1.00 | 48.03 | -0.58 |

### Key Findings

*   StructKV restores most of FastKV's performance loss under 128K ultra-long contexts (73.6 vs 68.2), validating the effectiveness of global accumulation against "dormant token loss."
*   Dynamic pivot layers adapt across different model architectures (Qwen-7B: L12, 14B: ~L20, 32B: L28), eliminating the need for manual tuning.
*   The decoupling strategy is critical: $R_{struct}=20\%, R_{KV}=10\%$ provides a +13.8 point Gain over a coupled 10% setting.
*   The extra overhead from GlobalScoreAccumulator and DynamicPivotDetector is only ~35ms (<2.5%), which is negligible.
*   Hidden state fidelity analysis: StructKV maintains >95% attention quality recovery across all layers, whereas FastKV drops to ~85% in deep layers.

## Highlights & Insights

*   The core insight is that "Local Saliency $\neq$ Structural Importance," for which Global In-Degree Centrality provides an elegant formalization.
*   Dynamic phase transition detection shifts compression timing from manual tuning to automatic discovery, offering high practical value.
*   Calculation/storage decoupling is a simple yet powerful design—breaking the implicit assumption that "to be fast, one must store less."

## Limitations & Future Work

*   Experimental validation was capped at 128K tokens; the stability of the structural skeleton at the million-token scale is unverified.
*   Tested only on standard dense Transformers; applicability to MoE or SSM architectures is unknown.
*   Dynamic detection relies on specific aggregation operations, which may require optimization on hardware with restricted memory bandwidth.

## Related Work & Insights

*   **vs FastKV**: FastKV uses local snapshots from fixed layers to select tokens; StructKV uses cross-layer accumulation and automatic detection of the compression layer, showing a significant performance gap at 128K.
*   **vs SnapKV**: SnapKV only optimizes decoding and does not accelerate prefill; StructKV optimizes both stages.
*   **vs GemFilter**: GemFilter results in low fidelity (~75-80%) due to fragmented context; StructKV maintains >95%.

## Rating

*   Novelty: ⭐⭐⭐⭐ The combination of global in-degree centrality, dynamic phase transition detection, and decoupling strategies is innovative.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across LongBench, RULER, multiple model series, detailed ablations, overhead analysis, and fidelity analysis.
*   Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions with complete formulaic derivations.
*   Value: ⭐⭐⭐⭐ Provides a more robust compression solution for long-context inference with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[ICML 2026\] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](../../ICML2026/llm_efficiency/obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)
- [\[ICML 2026\] Training-Inference Consistent Segmented Execution for Long-Context LLMs](../../ICML2026/llm_efficiency/training-inference_consistent_segmented_execution_for_long-context_llms.md)
- [\[ACL 2026\] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)
- [\[ACL 2026\] Threshold Differential Attention: Sink-free, Ultra-sparse, and Non-dispersive Long-context Attention](threshold_differential_attention_for_sink-free_ultra-sparse_and_non-dispersive_l.md)

</div>

<!-- RELATED:END -->
