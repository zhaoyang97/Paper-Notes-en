---
title: >-
  [Paper Note] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference
description: >-
  [ACL 2026][Interpretability][KV Cache compression] This paper proposes StructKV, a structure-aware KV Cache compression framework that identifies globally important tokens via Global In-Degree Centrality accumulated acro…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "KV Cache compression"
  - "long-context inference"
  - "global in-degree centrality"
  - "dynamic pivot detection"
  - "structural propagation"
date: 2026-05-08
content_hash: c177889a05e97e50
---

# StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference

**Conference**: ACL 2026
**arXiv**: [2604.06746](https://arxiv.org/abs/2604.06746)
**Code**: N/A
**Area**: Model Efficiency / KV Cache Compression
**Keywords**: KV Cache compression, long-context inference, global in-degree centrality, dynamic pivot detection, structural propagation

## TL;DR

This paper proposes StructKV, a structure-aware KV Cache compression framework that identifies globally important tokens via Global In-Degree Centrality accumulated across layers, adaptively locates the optimal compression layer via Dynamic Pivot Detection, and decouples computation and storage budgets via Structural Propagation & Decoupling. At 60% prefill + 10% KV retention, StructKV achieves near-full-context performance on LongBench and RULER.

## Background & Motivation

**Background**: LLM context windows have been extended to over one million tokens, yet inference efficiency faces a dual bottleneck: $O(N^2)$ attention complexity during prefill and linear KV cache memory growth during decoding. Existing methods typically address only one of these stages.

**Limitations of Prior Work**: (1) Decoding-only methods (StreamingLLM, SnapKV) compress KV cache without reducing prefill computation. (2) Prefill-aware methods (GemFilter, FastKV) rely on single-layer attention snapshots for local saliency-based token selection; however, some tokens may be temporarily "dormant" at the selected layer while playing a globally critical structural role. (3) FastKV uses a fixed pruning layer (e.g., Layer 15), a hyperparameter that does not generalize across model architectures and depths.

**Key Challenge**: Local saliency (single-layer snapshot) $\neq$ structural importance (cross-layer semantic role). A token may receive low attention at a specific layer yet serve as an information hub across the full network depth. Once discarded by a local-snapshot method, such information is permanently lost.

**Goal**: Design a structure-aware compression framework that identifies the "structural skeleton" of the context, retaining tokens that are globally important even when locally inconspicuous.

**Key Insight**: The true importance of a token is defined by its cumulative contribution across network depth, which can be formalized using in-degree centrality from graph theory.

**Core Idea**: Cross-layer accumulated attention scores form a global in-degree centrality measure; an information-theoretic metric adaptively detects the "phase transition" layer at which attention stabilizes as the compression point; and computation retention rate and storage retention rate are decoupled to independently optimize prefill speed and decoding memory.

## Method

### Overall Architecture

StructKV inference proceeds in three phases: (1) **Phase 1 (Full-context Processing)** — the first $L^*$ layers process the complete context while accumulating global centrality scores $\mathcal{C}_{global}$; (2) **Phase 2 (Structural Phase Transition)** — the automatic detector triggers at the optimal layer $L^*$, filters the context using $\mathcal{C}_{global}$, and decouples the computation budget $R_{struct}$ from the storage budget $R_{KV}$; (3) **Phase 3 (Compressed Inference)** — deeper layers operate exclusively on the condensed "structural skeleton."

### Key Designs

1. **Global In-Degree Centrality Accumulation**:

    - **Function**: Identifies tokens with globally structural importance across layers.
    - **Mechanism**: At each layer $l$, local saliency is computed as $\mathcal{S}_j^{(l)} = \sum_{g=1}^{G} \left( \frac{1}{w} \sum_{t=N-w}^{N} \sum_{h \in \mathcal{H}_g} a_{t,j}^{(l,h)} \right)$, then recursively accumulated with exponential decay: $\mathcal{C}_j = \sum_{l=0}^{L^*} \lambda^{(L^*-l)} \cdot \mathcal{S}_j^{(l)}$. The decay factor $\lambda=0.9$ assigns higher weight to deeper semantic layers.
    - **Design Motivation**: Unlike truncation based on a single-layer $\mathcal{S}_j^{(l)}$, global accumulation ensures that a token acting as an information hub across multiple early layers receives a high centrality score even if it is temporarily dormant at any individual layer.

2. **Dynamic Pivot Detection**:

    - **Function**: Adaptively locates the optimal compression layer, eliminating dependence on fixed hyperparameters.
    - **Mechanism**: Three metrics are tracked — attention entropy $\mathcal{H}_l$ (distributional uncertainty), sparsity $\rho_l$ (top-k cumulative probability mass), and variance $\mathcal{V}_l$ (discriminability). After computing normalized gradients, they are combined into a transition score $\mathcal{T}_l = w_1 \cdot \bar{\nabla}(-\mathcal{H}_l) + w_2 \cdot \bar{\nabla}(\rho_l) + w_3 \cdot \bar{\nabla}(\mathcal{V}_l)$, and the optimal layer is $L^* = \arg\max_l \mathcal{T}_l + 1$.
    - **Design Motivation**: Experiments show that the optimal layer varies with model depth (Layer 12 for Qwen-2.5-7B, Layer 28 for 32B), making fixed-layer approaches (e.g., FastKV's Layer 15) non-generalizable. Automatic detection triggers compression at the phase transition where attention shifts from "broad exploration" to "focused extraction."

3. **Structural Propagation & Decoupling**:

    - **Function**: Separates the optimization of computational efficiency and memory efficiency.
    - **Mechanism**: At layer $L^*$, the structural skeleton $\mathcal{I}_{struct} = \text{top-k}(\mathcal{C}, N \cdot R_{struct}) \cup \mathcal{I}_{win}$ is selected based on global centrality, and deeper layers compute exclusively on this reduced set. KV cache selection is performed independently using local saliency: $\mathcal{I}_{KV}^{(l)} = \text{top-k}(\mathcal{S}^{(l)}, N \cdot R_{KV}) \cup \mathcal{I}_{win}$. The structural retention rate $R_{struct}$ can be set substantially higher than the storage retention rate $R_{KV}$.
    - **Design Motivation**: Under coupled settings, aggressive compression causes accuracy collapse (10% retention yields only 45.3); after decoupling, $R_{struct}=20\%, R_{KV}=10\%$ recovers +13.8 points, entering a safe high-fidelity regime.

### Loss & Training

StructKV is a training-free, inference-time compression method. Default parameters: window $w=8$, decay $\lambda=0.9$, transition weights $\{w_1, w_2, w_3\}=\{0.2, 0.3, 0.5\}$, $R_{struct}=20\%$, $R_{KV}=10\%$.

## Key Experimental Results

### Main Results

**LongBench (LLaMA-3.1-8B-Instruct, average over 16 subtasks)**

| Method | Prefill | KV | Avg. Score |
|--------|---------|----|------------|
| Full-context | 100% | 100% | 49.33 |
| StreamingLLM | 100% | 10% | 41.59 |
| SnapKV | 100% | 10% | 46.92 |
| GemFilter | 60% | 10% | 40.40 |
| FastKV | 60% | 10% | 47.59 |
| **StructKV** | **60%** | **10%** | **48.61** |
| **StructKV** | **60%** | **20%** | **48.97** |

**RULER (LLaMA-3.1-8B-Instruct, retrieval benchmark)**

| Method | 8K | 16K | 32K | 64K | 128K | Avg. |
|--------|----|-----|-----|-----|------|------|
| Full-context | 90.1 | 95.0 | 83.4 | 85.5 | 76.3 | 86.0 |
| SnapKV | 75.6 | 76.8 | 72.9 | 75.0 | 67.7 | 73.6 |
| FastKV | 77.8 | 77.3 | 77.2 | 77.4 | 68.2 | 75.6 |
| **StructKV** | **81.3** | **82.5** | **81.8** | **81.5** | **73.6** | **80.1** |

### Ablation Study

**Sensitivity to decay factor $\lambda$ (LongBench, 10% KV)**

| $\lambda$ | Avg. Score | $\Delta$ |
|-----------|------------|----------|
| 0.50 | 47.41 | −1.20 |
| 0.80 | 48.35 | −0.26 |
| **0.90** | **48.61** | **Ref** |
| 0.95 | 48.42 | −0.19 |
| 1.00 | 48.03 | −0.58 |

### Key Findings

- StructKV recovers the majority of FastKV's performance degradation at 128K ultra-long contexts (73.6 vs. 68.2), validating the effectiveness of global accumulation against "dormant token loss."
- The dynamic pivot layer adapts across model architectures (Qwen-7B: L12, 14B: ~L20, 32B: L28), eliminating the need for manual tuning.
- The decoupling strategy is critical: $R_{struct}=20\%, R_{KV}=10\%$ outperforms coupled 10% retention by +13.8 points.
- The overhead of GlobalScoreAccumulator and DynamicPivotDetector is only ~35ms (<2.5%), which is negligible.
- Hidden-state fidelity analysis shows that StructKV maintains >95% attention quality recovery across all layers, whereas FastKV degrades to ~85% in deeper layers.

## Highlights & Insights

- The core insight that "local saliency $\neq$ structural importance" is elegantly formalized through global in-degree centrality.
- Dynamic phase-transition detection replaces manual hyperparameter tuning with automatic discovery of the optimal compression point, offering strong practical value.
- The computation/storage decoupling is a simple yet powerful design that breaks the implicit assumption that "faster inference requires fewer stored tokens."

## Limitations & Future Work

- Experimental validation is limited to 128K tokens; the stability of the structural skeleton at million-token scale remains unverified.
- Evaluation is conducted exclusively on standard dense Transformers; applicability to MoE or SSM architectures is unknown.
- Dynamic detection relies on specific aggregation operations that may require optimization on memory-bandwidth-constrained hardware.

## Related Work & Insights

- **vs. FastKV**: FastKV selects tokens via a local snapshot at a fixed layer; StructKV cross-layer accumulation combined with automatic pivot detection yields a pronounced performance advantage at 128K.
- **vs. SnapKV**: SnapKV optimizes only decoding without accelerating prefill; StructKV optimizes both stages simultaneously.
- **vs. GemFilter**: GemFilter's fragmented context leads to low fidelity (~75–80%); StructKV maintains >95%.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of global in-degree centrality, dynamic phase-transition detection, and decoupling strategy is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on LongBench + RULER across multiple model families, with detailed ablations, overhead analysis, and fidelity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐ Provides a more robust compression solution for long-context inference with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](../../ICLR2026/interpretability/implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)
- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Context-Value-Action Architecture for Value-Driven Large Language Model Agents](context-value-action_architecture_for_value-driven_large_language_model_agents.md)
- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[NeurIPS 2025\] Deep Modularity Networks with Diversity-Preserving Regularization](../../NeurIPS2025/interpretability/deep_modularity_networks_with_diversity-preserving_regularization.md)

</div>

<!-- RELATED:END -->
