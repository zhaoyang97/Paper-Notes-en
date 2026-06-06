---
title: >-
  [Paper Note] Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling
description: >-
  [ACL2026][Multimodal VLM][Long-context inference] Proposes DASH (Delta Attention Selective Halting), a training-free inference acceleration method. By monitoring the layer-wise update magnitude $\Delta_{attn}$ of self-at…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Long-context inference"
  - "Prefill acceleration"
  - "Token pruning"
  - "Attention redundancy"
  - "Vision-language models"
date: 2026-05-08
content_hash: 9e0fe1c750584ba7
---

<!-- Generated automatically by src/gen_stubs.py -->
# Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling

**Conference**: ACL2026
**arXiv**: [2604.18103](https://arxiv.org/abs/2604.18103)
**Code**: [GitHub](https://github.com/verach3n/DASH)
**Area**: multimodal_vlm
**Keywords**: Long-context inference, Prefill acceleration, Token pruning, Attention redundancy, Vision-language models

## TL;DR

Proposes DASH (Delta Attention Selective Halting), a training-free inference acceleration method. By monitoring the layer-wise update magnitude $\Delta_{attn}$ of self-attention layers, it identifies "semantically solidified" tokens and halts their subsequent computations. DASH achieves significant prefill speedup on long-context text and vision-language benchmarks with negligible accuracy loss.

## Background & Motivation

Long-context inference is a core capability for LLMs and LMMs, but the compute cost of the prefill stage grows quadratically with sequence length, becoming a major latency bottleneck. Most existing token pruning methods rely on heuristic importance scores (e.g., cumulative attention weights) that require access to the full attention matrix, making them incompatible with efficient kernels like FlashAttention. The authors propose a new perspective: instead of asking "which tokens are important," it is better to ask "**which tokens have already finished their work**." This hypothesis is supported by three key observations: (1) Token representations converge to "semantic fixed points," where $\Delta_{attn}$ is highly skewed and most tokens approach zero in middle layers; (2) Tokens with low $\Delta_{attn}$ are rarely attended to by subsequent layers, verifying the "stability implies redundancy" hypothesis; (3) Visual tokens saturate earlier than text tokens, explaining why direct transplantation of visual pruning methods to text models often fails.

## Method

### Overall Architecture

DASH determines the active set of tokens at a single activation layer $l_s$ during the prefill stage. For layers $l < l_s$, all $T$ tokens are processed normally. At layer $l_s$, the $\Delta_{attn}$ score is calculated for each token, and the top-$(1-\rho)T$ tokens are retained as the active set, while the remaining tokens are "halted." Halted tokens skip self-attention and FFN computations in all subsequent layers, with their hidden states frozen at their last updated values.

### Key Designs

1. **$\Delta_{attn}$ Signal**: Defined as the L2 norm of the self-attention sub-layer output (before the residual connection): $\Delta_t^{(l)} = \|U_t^{(l)}\|_2$, where $U^{(l)} = \text{Attn}(\text{LN}(H^{(l)}))$. This signal directly captures whether a token is still participating in global information aggregation, which is more effective than using $\Delta_{block}$ from the entire Transformer block (confirmed by ablation). Key advantage: **It does not require expanding the full attention matrix**, remaining fully compatible with FlashAttention.
2. **Single-Selection Scheduling**: The active set $S^* = \text{TopK}(S, K, \Delta^{(l_s)})$ is selected once at layer $l_s$, where $K = \lfloor(1-\rho)T\rfloor$, and all subsequent layers reuse the same active set. Compared to multi-selection scheduling, single-selection is simpler and yields comparable experimental results.
3. **Unified Modal Processing**: DASH makes no modality-specific assumptions and applies the $\Delta_{attn}$ criterion uniformly to both text and visual tokens. Since visual tokens naturally saturate earlier, DASH's advantages become more pronounced under aggressive compression ratios.

### Loss & Training

DASH is completely training-free and is a pure inference-time strategy. The theoretical FLOPs speedup ratio is $C_{full} / C_{ours} = L \cdot A(T) / [l_s \cdot A(T) + (L-l_s) \cdot A(\hat{T})]$. In a typical setup ($l_s=0.4L, \rho=0.667$), it achieves a theoretical speedup of $1.83\times$ when $T=16384$.

## Key Experimental Results

### Main Results

LongBench-E (Qwen2.5-7B-Instruct-1M):

| Method | Avg (%) | Qasper | HotpotQA | 2WikiQA | GovRep | LCC | Rep-P |
|------|----------|--------|----------|---------|--------|-----|-------|
| Vanilla | 48.87 | 44.19 | 51.13 | 62.97 | 6.97 | 65.00 | 99.33 |
| FastV | 43.99 | 40.44 | 42.63 | 57.67 | 6.96 | 59.33 | 83.67 |
| D3 | 45.00 | 40.18 | 44.49 | 60.95 | 6.19 | 64.67 | 99.33 |
| SnapKV (pr.) | 46.15 | 38.14 | 42.98 | 61.54 | 7.00 | 63.67 | 97.67 |
| **DASH** | **46.76** | **40.58** | **49.38** | **61.00** | **7.01** | **59.00** | **98.00** |

Kernel Compatibility Verification (40% pruning rate):

| Setting | LongBench-E (Avg) | LooGLE (Avg) |
|------|-------------------|-------------|
| Vanilla | 48.87 | 22.69 |
| Eager | 46.78 (1.52×) | 19.90 (1.34×) |
| FlashAttn | 46.76 (1.74×) | 19.94 (1.71×) |

### Ablation Study

| Experiment | Key Findings |
|----------|----------|
| $\Delta_{attn}$ vs $\Delta_{block}$ | $\Delta_{attn}$ consistently outperforms $\Delta_{block}$ on both text and VL benchmarks |
| Low $\Delta_{attn}$ vs High $\Delta_{attn}$ vs Random | Halting low $\Delta_{attn}$ significantly outperforms high $\Delta_{attn}$ and random selection, validating the "stability implies redundancy" hypothesis |
| Directional Ablation | High $\Delta_{attn}$ halting: LongBench-E 33.65 vs DASH 46.76, a gap of 13+ points |
| VL Compression Ratio | At extreme compression (96%-99%), DASH degrades significantly slower than FastV/VisionZip/DART |

### Key Findings

- DASH achieves the highest average score among all compression methods on LongBench-E (46.76 vs Vanilla 48.87) while achieving $1.74\times$ speedup.
- DASH is $1.74\times$ faster than FastV at the same accuracy level, and 8.5% more accurate than FastV at the same time cost.
- In vision-language tasks, DASH's advantage is more obvious under extreme compression (96-99%), benefiting from the early saturation characteristics of visual tokens.

## Highlights & Insights

- **Paradigm Shift**: Moving from "which tokens are important" to "which tokens have finished their work" represents a fundamental shift in token pruning logic.
- **Progressive Observations**: The existence of semantic fixed points $\rightarrow$ fixed-point tokens are indeed redundant $\rightarrow$ visual tokens saturate earlier; these form a coherent theoretical foundation.
- **FlashAttention Compatibility**: By avoiding the expansion of the attention matrix, it is one of the few pruning methods that works perfectly with efficient attention kernels.
- **Unified Cross-modality**: A single $\Delta_{attn}$ criterion naturally adapts to both text and vision-language scenarios without modality-specific designs.

## Limitations & Future Work

- The activation layer $l_s$ and pruning ratio $\rho$ need adjustment based on models and tasks (though the paper provides a lightweight screening method based on perplexity proxies).
- The single-selection scheduling, while simple, cannot handle dynamic changes in token importance between layers.
- Validation has only been performed on 7-8B models; effectiveness on larger scales (70B+) remains to be tested.
- Currently only accelerates prefill and does not change decoding efficiency.

## Related Work & Insights

- **SnapKV** (Li et al., 2024b): KV cache compression based on cumulative attention; DASH adapts it as a token pruning variant for comparison.
- **FastV** (Chen et al., 2024): A visual token pruning method that performs poorly when directly ported to long-context text.
- **D3** (Fan et al., 2025): Dynamic token pruning, but depends on attention matrix access.
- **Layer-wise redundancy** (He et al., 2024; Brinkmann et al., 2024): Analysis of representation redundancy in deep Transformers; DASH converts these observations into a viable acceleration strategy.
- Insight: Methodologies focusing on the rate of change of signals rather than the signals themselves may generalize to efficient inference in other sequence models.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Experimental Thoroughness | 9 |
| Writing Quality | 9 |
| Value | 8 |
| Total Score | 8.5 |

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](../../NeurIPS2025/multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](../../NeurIPS2025/multimodal_vlm/needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[ACL 2026\] From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration](from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_.md)
- [\[NeurIPS 2025\] Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression](../../NeurIPS2025/multimodal_vlm/breaking_the_compression_ceiling_data-free_pipeline_for_ultra-efficient_delta_co.md)

</div>

<!-- RELATED:END -->
