---
title: >-
  [Paper Note] Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling
description: >-
  [ACL 2026][Multimodal VLM][Vision-Language Model] Proposes DASH (Delta Attention Selective Halting), a training-free inference acceleration method that monitors layer-wise update magnitudes $\Delta_{attn}$ to identify "semantically solidified" tokens and halt their subsequent computations. It achieves significant prefill acceleration on long-context text and vision-la
tags:
  - ACL 2026
  - Multimodal VLM
  - Vision-Language Model
date: 2026-05-08
content_hash: e529982f921d26cd
---
# Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling

**Conference**: ACL2026  
**arXiv**: [2604.18103](https://arxiv.org/abs/2604.18103)  
**Code**: [GitHub](https://github.com/verach3n/DASH)  
**Area**: Multimodal VLM  
**Keywords**: Long-context inference, Prefill acceleration, Token pruning, Attention redundancy, Vision-language models

## TL;DR

Proposes DASH (Delta Attention Selective Halting), a training-free inference acceleration method that monitors layer-wise update magnitudes $\Delta_{attn}$ to identify "semantically solidified" tokens and halt their subsequent computations. It achieves significant prefill acceleration on long-context text and vision-language benchmarks with almost no loss in accuracy.

## Background & Motivation

Long-context inference is a core capability of LLMs and LMMs, but the computational cost of the prefill stage grows quadratically with sequence length, becoming a major latency bottleneck. **Limitations of Prior Work**: Existing token pruning methods mostly rely on heuristic importance scores (e.g., cumulative attention weights), which require access to the full attention matrix and are incompatible with efficient kernels like FlashAttention. **Key Insight**: Instead of asking "which tokens are important," the authors ask "**which tokens have already finished their work**." This hypothesis is supported by three key observations: (1) Token representations converge toward "semantic fixed points," where $\Delta_{attn}$ is highly skewed and approaches zero for most tokens in middle layers; (2) Tokens with low $\Delta_{attn}$ are rarely attended to by subsequent layers, verifying that stability implies redundancy; (3) Visual tokens saturate earlier than text tokens, explaining why direct migration of visual pruning methods to text models often fails.

## Method

### Overall Architecture

DASH determines the active set of tokens at an activation layer $l_s$ during the prefill stage. For layers before $l_s$, all $T$ tokens are processed normally. At layer $l_s$, a $\Delta_{attn}$ score is calculated for each token. The top-$(1-\rho)T$ tokens with the highest $\Delta_{attn}$ are retained as the active set, while the remaining "semantically solidified" tokens are "halted." Halted tokens skip self-attention and FFN computations in all subsequent layers, with their hidden states frozen at the last updated value.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IN["Long-context input T tokens (Text/Visual)"] --> SHALLOW["Shallow full calculation (first $l_s$ layers)<br/>All tokens pass through Self-Attn + FFN"]
    SHALLOW --> DELTA["$\Delta_{attn}$ signal (at layer $l_s$)<br/>Per-token L2 norm of pre-residual attention output $\|U\|_2$"]
    DELTA --> SELECT["Single-step scheduling + Unified modality processing<br/>TopK keeps $(1-\rho)T$ high $\Delta_{attn}$ tokens; same criteria for Text/Visual"]
    SELECT -->|High $\Delta_{attn}$: Still aggregating info| ACTIVE["Active set continues computation<br/>Normal Self-Attn + FFN for layers after $l_s$"]
    SELECT -->|Low $\Delta_{attn}$: Semantically solidified| HALT["Halt set frozen<br/>Skip Self-Attn + FFN for all subsequent layers; states fixed"]
    ACTIVE --> OUT["Output hidden states + KV cache $\rightarrow$ Decoding"]
    HALT --> OUT
```

### Key Designs

1.  **$\Delta_{attn}$ Signal**: Defined as the L2 norm of the self-attention sub-layer output (before the residual connection): $\Delta_t^{(l)} = \|U_t^{(l)}\|_2$, where $U^{(l)} = \text{Atten}(\text{LN}(H^{(l)}))$. This signal directly captures whether a token is still participating in global information aggregation and is more effective than using the entire Transformer block's $\Delta_{block}$ (confirmed by ablation). **Novelty**: It does not require materializing the full attention matrix, making it fully compatible with FlashAttention.
2.  **Mechanism**: A one-time selection of the active set $S^* = \text{TopK}(S, K, \Delta^{(l_s)})$ at layer $l_s$, where $K = \lfloor(1-\rho)T\rfloor$. Subsequent layers reuse this same active set. Compared to multi-step scheduling, single-step is simpler and shows comparable performance.
3.  **Unified Modality**: DASH applies the $\Delta_{attn}$ criterion uniformly to both text and visual tokens without modality-specific assumptions. Due to the early saturation of visual tokens, DASH's advantage is more pronounced at aggressive compression ratios.

### Loss & Training

DASH is training-free and is a pure inference-time strategy. The theoretical FLOPs speedup ratio is given by $C_{\text{full}} / C_{\text{ours}} = L \cdot A(T) / [l_s \cdot A(T) + (L-l_s) \cdot A(\hat{T})]$. In a typical setup ($l_s=0.4L, \rho=0.667$), it achieves a theoretical speedup of $1.83\times$ for $T=16384$.

## Key Experimental Results

### Main Results

LongBench-E (Qwen2.5-7B-Instruct-1M):

| Method | Average Score (%) | Qasper | HotpotQA | 2WikiQA | GovRep | LCC | Rep-P |
|------|----------|--------|----------|---------|--------|-----|-------|
| Vanilla | 48.87 | 44.19 | 51.13 | 62.97 | 6.97 | 65.00 | 99.33 |
| FastV | 43.99 | 40.44 | 42.63 | 57.67 | 6.96 | 59.33 | 83.67 |
| D3 | 45.00 | 40.18 | 44.49 | 60.95 | 6.19 | 64.67 | 99.33 |
| SnapKV (pr.) | 46.15 | 38.14 | 42.98 | 61.54 | 7.00 | 63.67 | 97.67 |
| **Ours (DASH)** | **46.76** | **40.58** | **49.38** | **61.00** | **7.01** | **59.00** | **98.00** |

Kernel Compatibility Verification (40% pruning ratio):

| Setup | LongBench-E (Avg) | LooGLE (Avg) |
|------|-------------------|-------------|
| Vanilla | 48.87 | 22.69 |
| Eager | 46.78 (1.52×) | 19.90 (1.34×) |
| FlashAttn | 46.76 (1.74×) | 19.94 (1.71×) |

### Ablation Study

| Content | Key Findings |
|----------|----------|
| $\Delta_{attn}$ vs $\Delta_{block}$ | $\Delta_{attn}$ consistently outperforms $\Delta_{block}$ on text and VL benchmarks. |
| Low vs High vs Random | Halting low $\Delta_{attn}$ significantly outperforms other strategies, verifying "stability equals redundancy." |
| Directional Ablation | Halting high $\Delta_{attn}$ results in 33.65 on LongBench-E vs 46.76 for DASH (a gap of 13+). |
| VL Compression | Under extreme 96%-99% compression, DASH's performance degrades much slower than FastV/VisionZip/DART. |

### Key Findings

- DASH achieves the highest average score (46.76 vs 48.87) among all compression methods on LongBench-E while reaching $1.74\times$ speedup.
- At the same accuracy level, it is $1.74\times$ faster than FastV; at the same time budget, it reaches 8.5% higher accuracy.
- In vision-language tasks, DASH's advantage is prominent under extreme compression (96-99%) due to the early saturation of visual tokens.

## Highlights & Insights

- **Paradigm Shift**: Shifting from "which tokens are important" to "which tokens have finished their work" represents a fundamental change in token pruning philosophy.
- **Layered Observations**: Logical progression from the existence of semantic fixed points to the redundancy of stable tokens and finally to the early saturation of visual tokens provides strong theoretical support.
- **FlashAttention Compatibility**: By avoiding the materialization of the attention matrix, it is one of the few pruning methods that perfectly complements efficient attention kernels.
- **Unified Cross-Modality**: A single $\Delta_{attn}$ criterion naturally adapts to both text and vision-language scenarios without modality-specific designs.

## Limitations & Future Work

- The activation layer $l_s$ and pruning ratio $\rho$ require adjustments based on models and tasks (though the paper provides a lightweight proxy method based on perplexity).
- The static nature of single-step scheduling cannot handle dynamic changes in token importance across layers.
- Validated only on 7-8B models; effectiveness on larger scales (70B+) remains to be tested.
- Currently only accelerates the prefill stage and does not change decoding efficiency.

## Related Work & Insights

- **SnapKV** (Li et al., 2024b): KV cache compression based on cumulative attention; DASH is compared against its token pruning variant.
- **FastV** (Chen et al., 2024): Visual token pruning; performs poorly when directly migrated to long-context text.
- **D3** (Fan et al., 2025): Dynamic token pruning, but depends on attention matrix access.
- **Layer-wise redundancy** (He et al., 2024; Brinkmann et al., 2024): Analysis of representation redundancy in deep Transformers; DASH converts these observations into a viable acceleration strategy.
- **Insight**: Focusing on signaling rates of change rather than the signals themselves is a methodology that could be generalized to efficient inference in other sequence models.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Experimental Thoroughness | 9 |
| Writing Quality | 9 |
| Value | 8 |
| Total Score | 8.5 |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression](../../NeurIPS2025/multimodal_vlm/breaking_the_compression_ceiling_data-free_pipeline_for_ultra-efficient_delta_co.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](../../NeurIPS2025/multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](../../NeurIPS2025/multimodal_vlm/needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](../../ACL2025/multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)
- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](../../ICML2026/multimodal_vlm/very_efficient_listwise_multimodal_reranking_for_long_documents.md)

</div>

<!-- RELATED:END -->
