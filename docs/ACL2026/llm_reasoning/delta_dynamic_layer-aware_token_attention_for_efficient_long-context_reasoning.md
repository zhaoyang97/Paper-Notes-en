---
title: >-
  [Paper Note] DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Sparse attention] DELTA is a **training-free hierarchical sparse attention** mechanism. It partitions Transformer layers into three groups: "Initial Full Attention layers + a few Δ-layers for re…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Sparse attention"
  - "KV cache"
  - "reasoning"
  - "Δ-layer"
  - "page-based selection"
date: 2026-05-08
content_hash: 93ff13694fcda79b
---

# DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning

**Conference**: ACL 2026  
**arXiv**: [2510.09883](https://arxiv.org/abs/2510.09883)  
**Code**: https://github.com/hoenza/DELTA (Available)  
**Area**: LLM Inference / Long Context / Efficient Inference  
**Keywords**: Sparse attention, KV cache, reasoning, Δ-layer, page-based selection

## TL;DR
DELTA is a **training-free hierarchical sparse attention** mechanism. It partitions Transformer layers into three groups: "Initial Full Attention layers + a few Δ-layers for reselecting salient pages + subsequent Sparse Attention layers." It matches or surpasses Full Attention accuracy on AIME / GPQA-Diamond while reducing the number of attended tokens by $4.25\times$ and achieving a $1.54\times$ end-to-end inference speedup.

## Background & Motivation
**Background**: Large Reasoning Models (LRMs) like DeepSeek-R1 / o3 / Qwen3 / GPT-OSS achieve high scores on benchmarks like AIME through "Long CoT test-time extension." However, during the decoding stage, every generated token requires scanning the entire KV cache. In long-sequence scenarios, throughput is entirely bottlenecked by memory bandwidth (e.g., Llama-3-8B with 32K context and BS=128 exceeds 500GB/s).

**Limitations of Prior Work**: ① **Eviction-based** methods (H2O / SnapKV / StreamingLLM / RaaS) permanently discard tokens. However, in reasoning chains, "seemingly useless early tokens" often become critical later; once discarded, accuracy collapses. ② **Selection-based** methods (Quest / TidalDecode) retain the full cache but select top-k for computation. Performing selection at every layer introduces cumulative errors, and single-layer scores can be inaccurate. At a 1k token budget, Quest and RaaS accuracy on AIME-2024 + DS-Qwen-14B is < 20% (vs. 60% with Full Attention).

**Key Challenge**: Reasoning tasks require "long-chain consistency"—missing any segment of important tokens causes reasoning to drift. Meanwhile, full attention is crippled by bandwidth. How to achieve high-recall sparsity without retraining, without dropping tokens, and without per-layer selection overhead?

**Goal**: Design a training-free module that (1) keeps the KV cache intact (no token loss for future use); (2) avoids full attention in every layer (bypassing bandwidth bottlenecks); and (3) maintains high recall in token selection to preserve reasoning accuracy.

**Key Insight**: Empirical observation reveals two statistical properties: ① **Inter-layer correlation**: Attention maps of adjacent Transformer layers are nearly identical; deep layers refine rather than reconstruct. ② **Sequential drift**: Attention focus drifts slowly as decoding progresses, requiring query-adaptive selection. Combining these leads to: "Perform full attention and token selection in only a few layers; reuse the selected pages in remaining layers."

**Core Idea**: Partition the Transformer into "Warmup layers → Δ-layer selection layers → Sparse layers." Δ-layers reselect pages at each decoding step (to handle drift), but only a few Δ-layers exist across the network (to save bandwidth).

## Method

### Overall Architecture
Pipeline: ① **Three-layer Grouping**—layers [0, r-1] perform full attention for warmup (early attention is too scattered for stable page selection); layers $\mathcal{D}$ (e.g., [2, 14, 23]) serve as Δ-layers, re-running full attention and refreshing page selection at each decoding step; remaining layers use sparse attention, attending only to the page set $\rho$ selected by the most recent Δ-layer. ② **No KV Discarding**—the full cache remains in HBM; only the pages "read" per layer are restricted. ③ **Page-based Implementation**—KV is organized into pages of $P=16$ tokens. Token scores are aggregated into page scores for GPU coalesced access. ④ **Δ-layer Calibration**—Full attention is run on a small calibration set. The $1-\cos$ distance between attention maps of adjacent layers is calculated. Layers with the largest distances are chosen as Δ-layers (indicating a significant shift in attention behavior where previous selection becomes unreliable).

### Key Designs

1. **Three-layer structure (warmup / Δ-layer / sparse)**:

    - **Function**: Transitions the Transformer's N layers into three functional segments, limiting "token reselection" overhead to 2–3 Δ-layers while other layers reuse selections to eliminate redundant full attention.
    - **Mechanism**: Layers [0, 1] use full attention for representation stabilization (early layers are too diffuse for reliable top-k selection); layer 2 is the first Δ-layer (establishing the initial salient page set); 1–2 additional Δ-layers are placed in the middle/late stages to handle sequential drift; all other layers are sparse-attention layers attending to pages from the nearest Δ-layer. Each decoding step, Δ-layers re-run full attention + selection (not caching old results), ensuring query-adaptivity.
    - **Design Motivation**: Based on "high inter-layer correlation + slow inter-step drift." Since adjacent attention maps are nearly identical, selection at a few key layers suffices. Since drift occurs across steps, selection must refresh per step rather than being reused across steps.

2. **Head-aware + recency compound token scoring**:

    - **Function**: Converts multi-head attention into page scores at Δ-layers while forcing the inclusion of the most recent $L$ pages to ensure recency is not discarded.
    - **Mechanism**: The score for token $t$ is defined as the maximum attention across heads $s_t = \max_{j=1,\ldots,m} \alpha_j^i(t)$ (using max instead of mean, as a single head might lock onto a key token); token scores are summed within pages to get page scores $S_u = \sum_{t:p(t)=u} s_t$; the last $L$ pages are retained, and the top $K-L$ remaining pages are selected based on $S_u$. Finally, $\rho$ = recency pages ∪ top-score old pages.
    - **Design Motivation**: Pure top-k scoring might misjudge newly generated tokens as low-score (before attention converges), losing "local context"; the recency window fills this gap. Max-over-heads is used to prevent strong signals from a single head being averaged out.

3. **Page-based KV management + calibrated Δ-layer placement**:

    - **Function**: (a) Uses paged KV (each page $P=16$ tokens) instead of token-level management for coalesced GPU memory access, reducing gather/scatter overhead; (b) Uses inter-layer attention shift from a calibration phase to automatically place Δ-layers.
    - **Mechanism**: (a) Borrowing from PagedAttention (Kwon et al. 2023), KV cache is organized into fixed pages, with token budget $k = K \cdot P$ and recency budget $\ell = L \cdot P$. Scores are calculated per token, but selection is done per page. (b) For each adjacent layer pair $\ell, \ell-1$, $d_{\ell-1, \ell} = 1 - \cos(a_{\ell-1}, a_\ell)$ is calculated on a calibration set. Peak positions are chosen as Δ-layers (e.g., DS-Qwen-14B shift at layers 4-5 is 0.953), with a "well-distributed across depth" constraint.
    - **Design Motivation**: Page-based management is an engineering best practice (vLLM/FlashAttention). Inter-layer shift is a more scientific signal for Δ-layer placement than manual tuning, and experiments show Δ-layer configurations are stable across datasets for the same model.

### Loss & Training
**Completely training-free**. Δ-layer calibration only requires one full-attention pass on a small calibration set to record attention maps and calculate inter-layer shifts. All subsequent inference uses FlashInfer JIT + PyTorch topk. Default config: page size $P=16$, budget $K=64$ pages (1k tokens) + $L=8$ recency pages.

## Key Experimental Results

### Main Results

DELTA vs Full vs Quest vs RaaS (1k-token budget, accuracy %):

| Model / Dataset | Full | DELTA-1k | DELTA-2k | Quest-1k | RaaS-1k |
|-----------------|------|----------|----------|----------|---------|
| DS-Qwen-14B / AIME-2024 | ~60 | ~50 | ~60 | <20 | <20 |
| DS-Qwen-7B / GPQA | base | base | **+30** | < base | < base |
| Most model × dataset | 100% | ≥100% | ≥ Full | Significant Drop | Significant Drop |

→ DELTA matches Full Attention even under a strict 1k budget and often surpasses Full Attention at a 2k budget (e.g., +30% on GPQA + DS-Qwen-7B), whereas Quest/RaaS collapse at the 1k budget.

Throughput and Latency (DS-Qwen-1.5B, bs=64, 18k decoding length):

| Metric | Full | DELTA (K=64) | Gain |
|------|------|--------------|------|
| Total Decoding Time | 403 s | 261 s | **1.54× speedup** |
| Throughput | 2921 tok/s | 4517 tok/s | +55% |
| Step Latency (long ctx) | 30 ms | 13 ms | ~2.3× |
| Attended Tokens | All | 1/4.25 | **4.25× Reduction** |

### Ablation Study

#Δ-layers vs Single-step forward time (DS-Qwen-7B, bs=64, TP=2, 16k tokens):

| #Δ-layers | Single-step forward (Relative) | Notes |
|-----------|---------------------|------|
| 1 | Lowest | Highest sparsity, but prone to staleness |
| 3 (Default) | Low | Accuracy sweet spot |
| 5 | Medium | Diminishing returns |
| All (=Full) | Highest | Degenerates to Full Attention |

Recency window $L$ vs accuracy (DS-Qwen-7B, Mixed120, 5 budgets):

| Budget $K$ | Best $L$ | Accuracy Range |
|-----------|----------|--------------|
| 64 (1k tokens) | Large $L$ | Lower budgets require more recency protection |
| 256 (4k) | $L=8$ | Broader coverage more important at higher budgets |
| 512 (8k) | $L=8$ | Same as above |

Difference can reach 10 percentage points, indicating $L$ must be budget-aware.

### Key Findings
- **DELTA-2k frequently surpasses Full Attention**: This is counter-intuitive but explainable—sparse attention filters noise tokens, allowing the model to focus on reasoning, similar to the regularization effect of Dropout.
- **Δ-layer placement determined by inter-layer attention shift**: DS-Qwen-14B shows a shift of 0.953 at layer (4,5), which informs the selection of Δ-layers [2,6,42]; this calibration method generalizes well to 1.5B/7B models.
- **Quest and RaaS fail on long reasoning**: At a 1k budget, both exhibit accuracy <20%, proving reasoning tasks are extremely sensitive to any form of permanent token loss or per-layer selection error. DELTA’s "Keep all cache + Multi-layer calibration" combo is essential.
- **DELTA overhead is concentrated in short context**: Page-selection overhead is 154% of baseline FlashInfer at 1k context but drops to 25% at 32k—DELTA becomes increasingly efficient as context length grows, matching reasoning model workloads.

## Highlights & Insights
- **"Inter-layer correlation + Inter-step drift"** dual-observation is the core insight—upgrading a well-known fact (attention sparsity) into a specific design principle: "refresh in time dimension, reuse in space dimension."
- **Retaining full KV cache without dropping tokens** is the fundamental difference from eviction methods like RaaS and the key to maintaining reasoning performance. Authors explicitly state "seemingly useless early tokens may become key later," thus compressing compute but not memory.
- **Inter-layer cosine shift for Δ-layer selection** is a transferable diagnostic method, useful for future work in layer-skipping, early-exit, or mixture-of-depth where budgets are allocated per layer.
- **Max-over-heads scoring** is a critical detail—salient tokens often only trigger 1–2 heads; averaging would wash out this signal, whereas max preserves "expert opinions" from all heads.

## Limitations & Future Work
- **Compute-only saving, no memory saving**: Full KV cache still occupies HBM, leading to OOM for extremely long contexts (>200K) or small GPUs. Authors suggest combining with quantization, offloading, or guaranteed eviction.
- **Validated only on DeepSeek-R1 distilled series + Math/Science QA**: Generalization to dialogue, code generation, or agent workloads is unverified. Different architectures (MoE, SSM) might require recalibrating Δ-layers.
- **Δ-layers and $(K, L)$ still require manual/calibrated selection**: While an attention-shift method is provided, it is still per-model. Sample-adaptive selection (e.g., lightweight learned router) remains future work.
- **Max-head scoring may lag during rapid attention drift**: Authors acknowledge that in scenarios with rapid drift, max scoring might not keep up, requiring higher frequency Δ-layers or adaptive scheduling.

## Related Work & Insights
- **vs Quest (ICLR 2025)**: Quest performs page-rep selection at every layer. DELTA performs it only at 2–3 Δ-layers and reuses across layers, avoiding the accumulation of small per-layer selection errors.
- **vs RaaS / SnapKV / H2O**: These eviction methods save memory but suffer catastrophic failure in reasoning (evicted tokens cannot be recovered). DELTA sacrifices memory savings to preserve accuracy.
- **vs TidalDecode**: Closest in spirit (full at some layers, reuse at others), but DELTA adds calibration-based Δ-layer selection, page-based implementation, and max-head + recency scoring, with specific benchmarking for reasoning.
- **vs SeerAttention-R**: Requires self-distillation to train a gating module. DELTA is training-free and easier to deploy.
- **Inspiration**: The hierarchical sparsity + periodic refresh approach can be extended to multi-modal (visual tokens show similar sparsity), hybrid attention modules in Mamba/SSM, or chunk-level relevance refresh in RAG.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The three-layer structure derived from "inter-layer correlation + inter-step drift" is a simple but elegant insight; meeting both training-free and no-token-loss requirements is a key differentiator.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 4 models × 4 reasoning benchmarks + 3 baselines + multiple budget/recency/Δ-layer configs + detailed overhead breakdown + calibration experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression: "Status Quo → Dual Observation → Three-layer Design → Page-based Implementation → Benchmark + Speedup." Algorithms and extensive ablation details are solid.
- **Value**: ⭐⭐⭐⭐⭐ 1.54× end-to-end speedup + no accuracy loss + training-free + open source makes this immediately shippable for industrial reasoning model serving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning](long-context_reasoning_through_proxy-based_chain-of-thought_tuning.md)
- [\[ACL 2026\] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](drp_distilled_reasoning_pruning_with_skill-aware_step_decomposition_for_efficien.md)
- [\[ACL 2026\] PPA-Plan: Proactive Pitfall Avoidance for Reliable Planning in Long-Context LLM Reasoning](ppa-plan_proactive_pitfall_avoidance_for_reliable_planning_in_long-context_llm_r.md)
- [\[ACL 2026\] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)

</div>

<!-- RELATED:END -->
