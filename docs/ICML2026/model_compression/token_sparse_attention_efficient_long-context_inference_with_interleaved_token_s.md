---
title: >-
  [Paper Note] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection
description: >-
  [ICML 2026][Model Compression][Sparse Attention] The authors observe that the "importance" of tokens varies drastically across layers and heads; traditional token eviction, which removes tokens in one shot…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Sparse Attention"
  - "Prefill Acceleration"
  - "Reversible Token Selection"
  - "FlashAttention Compatible"
  - "Dynamic Sparsity"
date: 2026-05-08
content_hash: 59f0939a6745bdcf
---

# Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection

**Conference**: ICML 2026  
**arXiv**: [2602.03216](https://arxiv.org/abs/2602.03216)  
**Code**: https://github.com/dongwonjo/Token-Sparse-Attention  
**Area**: Model Compression / Long-Context Inference Acceleration  
**Keywords**: Sparse Attention, Prefill Acceleration, Reversible Token Selection, FlashAttention Compatible, Dynamic Sparsity

## TL;DR
The authors observe that the "importance" of tokens varies drastically across layers and heads; traditional token eviction, which removes tokens in one shot, is an irreversible early decision error. They propose Token Sparse Attention, where each attention head in each layer independently selects $L' \ll L$ tokens for dense attention, then scatters the output back to the original sequence length, with a residual path allowing skipped tokens to be reconsidered in the next layer. This preserves both head/layer-level dynamic selection and compatibility with dense kernels like FlashAttention. Combined with FlexPrefill, it achieves ×3.23 attention speedup with <1% accuracy loss on 128K context.

## Background & Motivation
**Background**: With LLM context windows reaching 100K+, the $O(L^2)$ complexity of attention becomes the main bottleneck. Two main acceleration approaches: (i) Sparse attention (e.g., Minference, FlexPrefill), which skips low-importance regions using block-level sparsity; (ii) Token eviction (PyramidInfer, FastKV, GemFilter), which selects top-k tokens in early layers and computes only these in deeper layers.

**Limitations of Prior Work**: Sparse attention operates at the block level, so if low-relevance tokens are mixed within a block, they are still computed, limiting achievable sparsity. Token eviction makes hard decisions in early layers about which tokens are important; once deleted, tokens cannot be recovered in deeper layers—even if their importance increases—contradicting the true dynamics of token importance.

**Key Challenge**: Using LLaMA-3.1-8B-Instruct, the authors empirically find: (i) The overlap of top-1% tokens between layers drops rapidly with layer distance, indicating importance drifts across layers; (ii) Different heads in the same layer rank top tokens very differently, with each head focusing on different semantics. Eviction’s "one-size-fits-all" token set ignores both layer and head dynamics.

**Goal**: (i) Design a token-level sparse mechanism that allows head/layer-specific token selection and enables recovery of skipped tokens; (ii) Must directly reuse optimized dense kernels like FlashAttention without new CUDA code; (iii) Must be orthogonally composable with existing block-level sparse attention.

**Key Insight**: Rather than sparsifying the attention map (limited by block boundaries) or deleting from the KV cache (irreversible), perform **reversible compression-decompression** on $Q, K, V$: select tokens to compress into a short sequence for dense attention, then scatter outputs back to the original length and add a residual. The residual path allows "unselected tokens" to flow from the previous to the next layer, effectively providing a revival channel.

**Core Idea**: Use "compress-then-decompress + residual" to make token-level sparsification reversible, allowing each layer and head to re-decide selection.

## Method

### Overall Architecture
Token Sparse Attention operates in two steps within each selected sparse layer: (1) **Stage 1 Compression**: The Dynamic Token Coverage algorithm estimates the token set $S_{H=h}$ (size $L'$) for each head $h$. From $Q,K,V \in \mathbb R^{L\times d}$, gather $\hat Q, \hat K, \hat V \in \mathbb R^{L'\times d}$ according to $S_h$, and run FlashAttention for dense attention on $L'\times L'$ to obtain $\hat O$. (2) **Stage 2 Decompression**: Scatter $\hat O$ back to zero-initialized $\mathbb R^{L\times d}$ at positions $S_h$, keeping unselected positions as 0 (equivalent to a hard mask), then add a residual connection. Complexity drops from $O(L^2 d)$ to $O(L'^2 d)$. Which layers are sparsified is preselected via Inter-Layer Representation Drift (default: bottom 50% with minimal drift), requiring no training.

### Key Designs

1. **Compress-then-Decompress Reversible Token Sparsification**:

    - **Function**: Allows each layer and head to independently select tokens for dense attention, with unselected tokens having a chance to be selected in the next layer via the residual path.
    - **Mechanism**: In Stage 1, each head $h$ independently selects $S_h$, gathers $\hat Q_h, \hat K_h, \hat V_h$; attention is performed in the compressed space $\mathbb R^{L'\times L'}$ using FlashAttention, yielding $\hat O_h$. In Stage 2, scatter $\hat O_h$ back to the corresponding rows in $\mathbb R^{L\times d}$ (unselected rows = 0), then $X_{\ell+1} = X_\ell + \text{Decompress}(\hat O_h)$. The residual allows skipped tokens’ representations to flow directly to the next layer, where they may be reselected if deemed important.
    - **Design Motivation**: Traditional token eviction treats $L\to L'$ as irreversible KV deletion, so deleted tokens are lost to deeper layers. Compress-decompress treats them as **temporarily excluded from attention**, structurally deleting nothing, thus preserving dynamic importance across layers/heads. An engineering bonus: compressed $\hat Q\hat K\hat V$ are dense and contiguous, compatible with any existing attention kernel (FlashAttention, FlexPrefill, etc.), with no need for new CUDA code.

2. **Dynamic Token Coverage (Quantile-Based Budgeting by "Aggressiveness")**:

    - **Function**: Dynamically determines how many tokens to retain per layer during inference (not a fixed ratio), and independently selects which tokens per head.
    - **Mechanism**: For each head, use recent queries and all keys to compute lightweight attention $\hat A$, sum columns and pool to obtain head-level token scores $s_h[t]$, aggregate and normalize across heads to get layer-level scores $s_l$. Sort $s_l$ in ascending order, find the smallest $k_{\text{sparse}}$ such that $\sum_{j=1}^{k_{\text{sparse}}} s_l[I[j]] \ge \tau$ (default $\tau=0.005$), i.e., the cumulative weight of the least important tokens does not exceed $\tau$; these are dropped. Retain $k_{\text{keep}} = L - k_{\text{sparse}}$ tokens. Each head independently selects its top-$k_{\text{keep}}$ subset $S_h$. A custom Triton fused kernel makes the scoring I/O overhead negligible.
    - **Design Motivation**: Fixed retention ratios mismatch across context lengths/tasks (information density varies greatly); quantile-based budgeting ("cumulative attention noise tail ≤ $\tau$") enables adaptive sparsity—longer contexts with more attention noise yield higher sparsity, shorter contexts less. The underlying assumption: long contexts inevitably accumulate "long-tail low-weight tokens," and pruning them acts as structural regularization.

3. **Inter-Layer Representation Drift for Sparse Layer Selection (Which Layers Are Robust)**:

    - **Function**: Identifies which layers can be sparsified with minimal impact, avoiding blanket sparsification.
    - **Mechanism**: Define normalized drift for layer $\ell$ as $R_\ell = \mathbb E_t[\|h_{\ell+1,t} - h_{\ell,t}\|_2 / (\|h_{\ell,t}\|_2 + \epsilon)]$; small drift = stable token representations = layer can tolerate sparsification. Compute $R_\ell$ on calibration data, rank to obtain $\hat R_\ell$, and select $\mathcal L_{\text{sparse}} = \{\ell | \hat R_\ell \le \delta\}$ (default $\delta=0.5$, i.e., sparsify the 50% of layers with least drift). This is run once at model load time.
    - **Design Motivation**: Experiments show that, for 200 random 3-layer combinations, average drift correlates highly with accuracy—sparsifying stable layers does not harm token representations, while unstable layers accumulate errors. This makes "which layers to sparsify" a data-driven preprocessing step rather than a hyperparameter, reducing user tuning burden.

### Loss & Training
The method is entirely training-free at inference; no fine-tuning required. Only a one-time calibration run at model load to obtain $\mathcal L_{\text{sparse}}$. Hyperparameter $\tau$: 0.005 for LLaMA-3.1-8B, 0.008 for Mistral-Nemo-12B. Token scoring uses a Triton fused kernel; attention uses unmodified FlashAttention.

## Key Experimental Results

### Main Results
Average accuracy and 128K speedup on RULER benchmark after stacking with each baseline (LLaMA-3.1-8B-Instruct):

| Method | 4K | 32K | 128K | Avg. | 128K Speedup |
|---|---|---|---|---|---|
| FlashAttention | 95.82 | 84.87 | 74.15 | 87.01 | ×1.00 |
| + Token Sparse | 96.06 | 84.81 | 73.68 | 87.02 | ×1.36 |
| Minference | 93.46 | 85.34 | 73.63 | 86.49 | ×1.12 |
| + Token Sparse | 93.05 | 85.10 | 72.18 | 86.05 | ×1.38 |
| FlexPrefill | 95.48 | 87.20 | 73.75 | 87.27 | ×2.44 |
| + Token Sparse | 95.33 | 87.68 | 73.58 | 87.27 | **×2.76** |

Comparison with token eviction methods at the same speedup (128K, LLaMA-3.1-8B):

| Method | Avg. Accuracy | Speedup |
|---|---|---|
| FlashAttention | 87.01 | ×1.00 |
| PyramidInfer | 78.49 | ×1.49 |
| GemFilter | 85.12 | ×1.53 |
| FastKV | 85.64 | ×1.50 |
| **Token Sparse Attention** | **86.84** | ×1.51 |

### Ablation Study

| Configuration | Key Findings | Meaning |
|---|---|---|
| Dynamic $\tau=0.005$ vs Fixed $s=0.3$ | 87.02 vs 86.91 at same speedup | Dynamic budgeting outperforms fixed ratio |
| Dynamic $\tau=0.010$ vs Fixed $s=0.5$ | 86.84 vs 85.43 at high sparsity | Dynamic advantage increases with more aggressive sparsity |
| Speedup breakdown (128K) | scoring/compress/decompress total overhead <11% | Lightweight engineering overhead |
| Sparsity vs context length | 4K: 17%, 128K: 54% | Longer contexts naturally have more prunable tokens |

### Key Findings
- Stacking with FlashAttention: almost no accuracy change (87.01 → 87.02), with ×1.36 speedup contributed independently.
- Most valuable when combined with block-level sparsity (FlexPrefill): ×2.44 → ×2.76, showing token-level and block-level sparsity are complementary and non-overlapping.
- Outperforms all token eviction methods at the same speedup, with the gap especially pronounced on short contexts (e.g., PyramidInfer is 17 points lower than FlashAttn at 4K).

## Highlights & Insights
- **Compress-then-Decompress is a highly elegant "pseudo-sparse" mechanism**: It computes dense attention on $L'\times L'$, then fills back into $L\times d$, with the residual channel preserving skipped token information. This is equivalent to lightweight, reversible, head-specific token selection at each layer. Such "logical sparsity + physical density" can be transferred to MoE, sparse expert routing, etc.
- **No need to write new kernels is a major engineering advantage**: Directly calls existing FlashAttention/FlexPrefill kernels, with zero barrier for downstream users. In contrast, token eviction requires modifying the KV cache structure, making deployment much more costly.
- **Drift-based layer selection is a simple yet powerful prior**: It turns "which layers can tolerate sparsity" from a hyperparameter into a data-driven decision, generalizable to any "layer compression" task (e.g., layer dropout, layer pruning).

## Limitations & Future Work
- Still relies on recent queries to estimate token scores, which is heuristic; if the model uses sliding window or chunked attention, the statistical meaning of recent queries may be compromised.
- The residual path preserves "unselected token" information, but each layer’s scattered zero rows actually lose cross-attention contributions between selected and unselected tokens; this loss is not quantified in the paper.
- When sparsity varies greatly across heads/layers, different $L'$ per head in a batch can disrupt tensor regularity (though FlashAttention supports ragged tensors, efficiency is affected); the paper does not discuss actual throughput for multi-sample batches.
- Only validated on prefill; not used in decoding. However, decoding is bottlenecked by KV cache loading rather than attention computation, so this method is naturally less suitable.
- Future directions: make drift-based layer selection adaptive (per prompt), replace scoring with a learnable router (end-to-end training), combine with KV cache quantization.

## Related Work & Insights
- **vs Minference / FlexPrefill (block-level sparsity)**: These operate on the attention map at the block level, limited by block boundaries; this method selects at the token level and can be orthogonally combined, yielding an additional ×1.13 speedup on FlexPrefill.
- **vs PyramidInfer / FastKV / GemFilter (token eviction)**: These make hard decisions in early layers about which tokens to retain, with no recovery in deeper layers; this method allows re-selection in every layer, achieving 1–8 points higher accuracy at the same speedup.
- **vs FlashAttention**: FlashAttention is an I/O-optimized dense attention with $O(L^2)$ complexity; this method adds algorithmic sparsification on top, reducing complexity to $O(L'^2)$ while reusing its kernel.
- **vs KV cache quantization (KIVI/H2O)**: These reduce KV memory loading overhead, while this method reduces attention computation overhead; the two are fully orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ The reversible compress-then-decompress design and head-specific token selection are simple yet effective innovations; drift-based layer selection is also a clean engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two models × 4 baselines × multiple lengths × multiple benchmarks (RULER/InfiniteBench), plus same-speedup comparisons with eviction methods, provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐ The logical flow from observations on token importance dynamics to method design is smooth; Figure 3 clearly illustrates the compress-decompress process.
- Value: ⭐⭐⭐⭐ Directly deployable in industry, valuable for all long-context LLM inference services; orthogonal composability with existing sparse methods is a key selling point.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ICML 2025\] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](../../ICML2025/model_compression/orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](../../NeurIPS2025/model_compression/recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ICML 2026\] Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)

</div>

<!-- RELATED:END -->
