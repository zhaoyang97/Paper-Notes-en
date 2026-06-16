---
title: >-
  [Paper Note] Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling
description: >-
  [ACL 2026][Multimodal VLM][Vision-Language Model] Ours proposes DASH (Delta Attention Selective Halting), a training-free inference acceleration method. By monitoring the layer-wise update magnitude $\Delta_{attn}$ of self-attention layers, it identifies "semantically solidified" tokens and halts their subsequent computation. DASH achieves significant prefill accelera
tags:
  - ACL 2026
  - Multimodal VLM
  - Vision-Language Model
date: 2026-05-08
content_hash: 9740ba36c1eacfbf
---
<!-- Generated automatically by src/gen_stubs.py -->
# Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling

**Conference**: ACL2026
**arXiv**: [2604.18103](https://arxiv.org/abs/2604.18103)
**Code**: [GitHub](https://github.com/verach3n/DASH)
**Area**: Multimodal VLM
**Keywords**: Long-context inference, Prefill acceleration, Token pruning, Attention redundancy, Visual Language Models

## TL;DR

Ours proposes DASH (Delta Attention Selective Halting), a training-free inference acceleration method. By monitoring the layer-wise update magnitude $\Delta_{attn}$ of self-attention layers, it identifies "semantically solidified" tokens and halts their subsequent computation. DASH achieves significant prefill acceleration on long-context text and vision-language benchmarks with almost no loss in accuracy.

## Background & Motivation

Long-context reasoning is a core capability of LLMs and LMMs, but the computational cost of the prefill stage grows quadratically with sequence length, becoming a major latency bottleneck. Most existing token pruning methods rely on heuristic importance scores (such as cumulative attention weights), which require access to the full attention matrix and are incompatible with efficient kernels like FlashAttention. The authors propose a new perspective: instead of asking "which tokens are important," one should ask "**which tokens have already finished their work**." This hypothesis is supported by three key observations: (1) token representations converge toward "semantic fixed points," where $\Delta_{attn}$ is highly skewed and most tokens approach zero in middle layers; (2) tokens with low $\Delta_{attn}$ are rarely attended to by subsequent layers, validating the hypothesis that stability implies redundancy; (3) visual tokens saturate much earlier than text tokens, explaining why pruning methods directly ported from vision models often fail on text models.

## Method

### Overall Architecture

DASH makes a one-time decision for the active set of tokens at an activation layer $l_s$ during the prefill stage. For layers before $l_s$, all $T$ tokens are processed normally. At layer $l_s$, the $\Delta_{attn}$ score for each token is calculated, and the top-$(1-\rho)T$ tokens with the highest $\Delta_{attn}$ are retained as the active set, while the remaining "semantically solidified" tokens are "halted." Halted tokens skip both self-attention and FFN computations in all subsequent layers, with their hidden states frozen at their last updated values.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IN["Long-context input with T tokens (Text / Visual)"] --> SHALLOW["Shallow full computation (First $l_s$ layers)<br/>All tokens process Self-Attention + FFN"]
    SHALLOW --> DELTA["$\Delta_{attn}$ signal (at layer $l_s$)<br/>Per-token L2 norm of attention output before residual: $\|U\|_2$"]
    DELTA --> SELECT["Single-stage selection scheduling + Unified modality processing<br/>TopK retains $(1-\rho)T$ tokens with high $\Delta_{attn}$; Same criteria for Text / Visual"]
    SELECT -->|High $\Delta_{attn}$: Still aggregating information| ACTIVE["Active set continues computation<br/>Normal Self-Attention + FFN in subsequent layers"]
    SELECT -->|Low $\Delta_{attn}$: Semantically solidified| HALT["Halt set frozen<br/>Skip all subsequent Attention + FFN; Hidden states fixed"]
    ACTIVE --> OUT["Output hidden states + KV cache → Decoding"]
    HALT --> OUT
```

### Key Designs

1. **$\Delta_{attn}$ Signal**: Defined as the L2 norm of the self-attention sub-layer output (before the residual connection): $\Delta_t^{(l)} = \|U_t^{(l)}\|_2$, where $U^{(l)} = \text{Attn}(\text{LN}(H^{(l)}))$. This signal directly captures whether a token is still participating in global information aggregation, which is more effective than using the $\Delta_{block}$ of the entire Transformer block (as confirmed by ablation studies). Key advantage: **It does not require expanding the full attention matrix**, making it fully compatible with FlashAttention.
2. **Single-stage Selection Scheduling**: The active set $S^* = \text{TopK}(S, K, \Delta^{(l_s)})$ where $K = \lfloor(1-\rho)T\rfloor$ is selected once at the activation layer $l_s$. All subsequent layers reuse this same active set. Compared to multi-stage scheduling, single-stage selection is simpler and yields comparable experimental results.
3. **Unified Modality Processing**: DASH does not make modality-specific assumptions and applies the $\Delta_{attn}$ criterion uniformly to both text and visual tokens. Since visual tokens naturally saturate earlier, the advantages of DASH are even more pronounced under aggressive compression ratios.

### Loss & Training

DASH is completely training-free and serves as a pure inference-time strategy. The theoretical FLOPs acceleration ratio is $C_{full} / C_{ours} = L \cdot A(T) / [l_s \cdot A(T) + (L-l_s) \cdot A(\hat{T})]$. In a typical setting ($l_s=0.4L, \rho=0.667$), the theoretical speedup is $1.83\times$ when $T=16384$.

## Key Experimental Results

### Main Results

LongBench-E (Qwen2.5-7B-Instruct-1M):

| Method | Avg (%) | Qasper | HotpotQA | 2WikiQA | GovRep | LCC | Rep-P |
|------|----------|--------|----------|---------|--------|-----|-------|
| Original Model | 48.87 | 44.19 | 51.13 | 62.97 | 6.97 | 65.00 | 99.33 |
| FastV | 43.99 | 40.44 | 42.63 | 57.67 | 6.96 | 59.33 | 83.67 |
| D3 | 45.00 | 40.18 | 44.49 | 60.95 | 6.19 | 64.67 | 99.33 |
| SnapKV (pr.) | 46.15 | 38.14 | 42.98 | 61.54 | 7.00 | 63.67 | 97.67 |
| **DASH** | **46.76** | **40.58** | **49.38** | **61.00** | **7.01** | **59.00** | **98.00** |

Kernel compatibility validation (Pruning rate 40%):

| Setting | LongBench-E (Avg) | LooGLE (Avg) |
|------|-------------------|-------------|
| Vanilla | 48.87 | 22.69 |
| Eager | 46.78 (1.52×) | 19.90 (1.34×) |
| FlashAttn | 46.76 (1.74×) | 19.94 (1.71×) |

### Ablation Study

| Content | Key Findings |
|----------|----------|
| $\Delta_{attn}$ vs $\Delta_{block}$ | $\Delta_{attn}$ consistently outperforms $\Delta_{block}$ on both text and VL benchmarks. |
| Low $\Delta_{attn}$ vs High $\Delta_{attn}$ vs Random | Halting low $\Delta_{attn}$ tokens significantly outperforms high $\Delta_{attn}$ and random selection, validating the "stability implies redundancy" hypothesis. |
| Directional Ablation | High $\Delta_{attn}$ halting: LongBench-E 33.65 vs DASH 46.76, a gap of 13+ points. |
| VL Compression Ratio | Under extreme compression (96%-99%), DASH degrades significantly slower than FastV, VisionZip, or DART. |

### Key Findings

- DASH achieves the highest average score among all compression methods on LongBench-E (46.76 vs. Original 48.87) while achieving $1.74\times$ speedup.
- DASH is $1.74\times$ faster than FastV at the same accuracy level and 8.5% more accurate than FastV given the same computation time.
- In vision-language tasks, the advantage of DASH is even more evident under extreme compression (96-99%), benefiting from the early saturation characteristics of visual tokens.

## Highlights & Insights

- **Paradigm Shift**: Moving from "which tokens are important" to "which tokens have finished their work" represents a fundamental shift in token pruning strategy.
- **Progressive Observations**: The existence of semantic fixed points $\rightarrow$ fixed point tokens are indeed redundant $\rightarrow$ visual tokens saturate earlier; these form a rigorous theoretical foundation.
- **FlashAttention Compatibility**: By avoiding the expansion of the attention matrix, it is one of the few pruning methods that can perfectly complement efficient attention kernels.
- **Unified Cross-Modality**: The same $\Delta_{attn}$ criterion naturally adapts to both text and vision-language scenarios without needing modality-specific designs.

## Limitations & Future Work

- The activation layer $l_s$ and pruning ratio $\rho$ require adjustment based on models and tasks (though the paper provides a lightweight screening method based on perplexity proxies).
- While simple, the single-stage scheduling cannot handle cases where token importance changes dynamically across layers.
- Validated only on 7-8B models; performance on larger scales (70B+) remains to be tested.
- Currently only accelerates the prefill phase and does not change decoding efficiency.

## Related Work & Insights

- **SnapKV** (Li et al., 2024b): KV cache compression based on cumulative attention; DASH is compared against its adaptation as a token pruning variant.
- **FastV** (Chen et al., 2024): A visual token pruning method that performs poorly when directly ported to long-context text.
- **D3** (Fan et al., 2025): Dynamic token pruning, but depends on attention matrix access.
- **Layer-wise redundancy** (He et al., 2024; Brinkmann et al., 2024): Analysis of representation redundancy in deep Transformers; DASH converts these observations into a viable acceleration strategy.
- Insights: The methodology of focusing on the rate of signal change rather than the signal itself could be generalized to efficient inference in other sequence models.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Experimental Thoroughness | 9 |
| Writing Quality | 9 |
| Value | 8 |
| Overall Score | 8.5 |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](../../ICML2025/multimodal_vlm/mminference_accelerating_pre-filling_for_long-context_vlms_via_modality-aware_pe.md)
- [\[ACL 2025\] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](../../ACL2025/multimodal_vlm/madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ACL 2026\] From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration](from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](../../NeurIPS2025/multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)

</div>

<!-- RELATED:END -->
