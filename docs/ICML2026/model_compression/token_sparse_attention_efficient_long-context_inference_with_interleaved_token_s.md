---
title: >-
  [Paper Note] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection
description: >-
  [ICML 2026][Model Compression][Sparse Attention] The authors observe that token "importance" fluctuates significantly across layers and heads…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Sparse Attention"
  - "prefill acceleration"
  - "reversible token selection"
  - "FlashAttention compatible"
  - "dynamic sparsity"
date: 2026-05-08
content_hash: 606e7731eca7d071
---

# Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection

**Conference**: ICML 2026  
**arXiv**: [2602.03216](https://arxiv.org/abs/2602.03216)  
**Code**: https://github.com/dongwonjo/Token-Sparse-Attention  
**Area**: Model Compression / Long-context Inference Acceleration  
**Keywords**: Sparse Attention, prefill acceleration, reversible token selection, FlashAttention compatible, dynamic sparsity

## TL;DR
The authors observe that token "importance" fluctuates significantly across layers and heads; traditional token eviction relies on irreversible early-stage decisions. They propose Token Sparse Attention, where each attention head per layer independently selects $L' \ll L$ tokens to conduct dense attention. The output is then scattered back to the original sequence length, combined with a residual path that allows skipped tokens to be re-selected in subsequent layers. This preserves head/layer-level dynamic selection while enabling direct use of optimized dense kernels like FlashAttention. Combined with FlexPrefill on 128K context, it achieves a 3.23\times attention speedup with accuracy loss $< 1\%$.

## Background & Motivation
**Background**: As LLM context windows expand to 100K+, the $O(L^2)$ complexity of attention becomes the primary bottleneck. Two acceleration paths exist: (i) sparse attention (e.g., Minference, FlexPrefill), which uses block-level patterns to skip low-importance regions; (ii) token eviction (PyramidInfer, FastKV, GemFilter), which selects top-k tokens in early layers and computes only those in deeper layers.

**Limitations of Prior Work**: Sparse attention is block-based; if low-relevance tokens are mixed within a block, they are still computed, limiting the sparsity ceiling. Token eviction makes "hard decisions" in early layers about which tokens are important; deleted tokens cannot be recovered even if they become relevant in deeper layers, violating the true dynamics of token importance.

**Key Challenge**: Using LLaMA-3.1-8B-Instruct, the authors observe: (i) the overlap rate of top-1% tokens between layers drops rapidly as layer distance increases, indicating layer-wise importance drift; (ii) top token rankings vary significantly across different heads within the same layer, as heads attend to different semantics. Eviction using a "one-size-fits-all" token set ignores both layer and head dynamics.

**Goal**: (i) Design a token-level sparsity mechanism that allows head/layer-specific selection while remaining reversible; (ii) Ensure compatibility with optimized dense kernels like FlashAttention without custom CUDA kernels; (iii) Ensure orthogonality with existing block-level sparse attention.

**Key Insight**: Instead of performing sparsity on the attention map (limited by block boundaries) or deleting from KV cache (irreversible), perform **reversible compression-decompression** on $Q, K, V$: compress tokens into a short sequence for dense attention, then scatter the output back to the original length and add residuals. The residual path allows information from "unselected tokens" to flow from the previous layer to the next, effectively providing a "resurrection" channel.

**Core Idea**: Utilize a "compress-then-decompress + residual" approach to transform token-level sparsification into a reversible operation, allowing every layer and head to re-decide token importance.

## Method

### Overall Architecture
Within selected sparse layers, Token Sparse Attention operates in two steps: (1) **Stage 1 Compression**: The Dynamic Token Coverage algorithm estimates a token set $S_{H=h}$ (size $L'$) for each head $h$. $\hat Q, \hat K, \hat V \in \mathbb R^{L'\times d}$ are gathered from $Q,K,V \in \mathbb R^{L\times d}$ based on $S_h$. FlashAttention is then invoked on $L'\times L'$ to compute dense attention, yielding $\hat O$. (2) **Stage 2 Decompression**: $\hat O$ is scattered back to a zero-initialized $\mathbb R^{L\times d}$ according to $S_h$; unselected positions remain 0, equivalent to applying a hard mask. A residual connection is then applied. Complexity is reduced from $O(L^2 d)$ to $O(L'^2 d)$. Sparse layers are pre-selected once via Inter-Layer Representation Drift (defaulting to the bottom 50% of layers with minimal drift) without requiring training.

### Key Designs

1. **Compress-then-Decompress Reversible Token Sparsification**:
    - **Function**: Allows each layer and head to independently select tokens for dense attention while keeping unselected tokens recoverable through the residual path.
    - **Mechanism**: Stage 1 selects $S_h$ independently for each head $h$, gathering $\hat Q_h, \hat K_h, \hat V_h$. Attention is processed in the compressed space $\mathbb R^{L'\times L'}$ by FlashAttention to produce $\hat O_h$. Stage 2 uses scatter to place $\hat O_h$ back into corresponding rows of $\mathbb R^{L\times d}$ (unselected rows = 0), followed by $X_{\ell+1} = X_\ell + \text{Decompress}(\hat O_h)$. The residual allows skipped token representations to flow to the next layer, where they may be re-selected if deemed important.
    - **Design Motivation**: Traditional token eviction treats $L\to L'$ as irreversible KV deletion, rendering tokens invisible to deeper layers. Compress-decompress treats them as **inputs temporarily excluded from attention**, physically deleting nothing and thus preserving layer/head importance dynamics. An additional engineering benefit: compressed $\hat Q\hat K\hat V$ are dense and contiguous, allowing direct use of any off-the-shelf attention kernels (FlashAttention, FlexPrefill, etc.) without custom CUDA development.

2. **Dynamic Token Coverage (Quantile-based Budgeting)**:
    - **Function**: Dynamically determines how many tokens to keep per layer (rather than a fixed ratio) and which to keep per head during inference.
    - **Mechanism**: For each head, recent queries are used with all keys to compute a lightweight attention $\hat A$. Column sums are pooled to get head-level token scores $s_h[t]$, which are aggregated and normalized to get layer-level scores $s_l$. $s_l$ is sorted in ascending order to find the smallest $k_{\text{sparse}}$ such that $\sum_{j=1}^{k_{\text{sparse}}} s_l[I[j]] \ge \tau$ (default $\tau=0.005$). This ensures that the total weight of the **least important tokens does not exceed $\tau$**, and these are discarded, keeping $k_{\text{keep}} = L - k_{\text{sparse}}$ tokens. Each head independently selects its top-$k_{\text{keep}}$ subset $S_h$. A custom fused Triton kernel ensures the scoring overhead is negligible.
    - **Design Motivation**: Fixed retention ratios fail across varying context lengths or tasks due to shifting information density. Budgeting based on "cumulative attention noise tail $\le \tau$" allows for adaptive sparsity—higher in long contexts with more noise, lower in short contexts. This assumes long-context attention naturally accumulates a "long tail" of low-weight tokens that can be pruned as structural regularization.

3. **Inter-Layer Representation Drift Layer Selection**:
    - **Function**: Identifies layers where sparsification causes minimal damage to avoid a "one-size-fits-all" approach to all layers.
    - **Mechanism**: Define normalized drift for layer $\ell$ as $R_\ell = \mathbb E_t[\|h_{\ell+1,t} - h_{\ell,t}\|_2 / (\|h_{\ell,t}\|_2 + \epsilon)]$. Smaller drift implies stable token representations, suggesting the layer can tolerate sparsification. $R_\ell$ is calculated on calibration data, and the sparse layer set is $\mathcal L_{\text{sparse}} = \{\ell | \hat R_\ell \le \delta\}$ (default $\delta=0.5$, picking the 50% most stable layers). This is performed once during model loading.
    - **Design Motivation**: Experiments on 200 random 3-layer sparse combinations show that average drift correlates highly with accuracy. Sparsifying stable layers preserves representations, while sparsifying unstable layers leads to error accumulation. This turns "which layers to sparsify" from a hyperparameter into a data-driven preprocessing step.

### Loss & Training
Ours is an entirely training-free inference-time method requiring no fine-tuning. $\mathcal L_{\text{sparse}}$ is obtained via a single calibration run at model load. Hyperparameter $\tau$: 0.005 for LLaMA-3.1-8B, 0.008 for Mistral-Nemo-12B. Token scoring uses a Triton fused kernel; attention uses unmodified FlashAttention.

## Key Experimental Results

### Main Results
Average accuracy and 128K speedup on RULER benchmark (LLaMA-3.1-8B-Instruct):

| Method | 4K | 32K | 128K | Avg. | 128K Speedup |
|---|---|---|---|---|---|
| FlashAttention | 95.82 | 84.87 | 74.15 | 87.01 | 1.00\times |
| + Token Sparse (Ours) | 96.06 | 84.81 | 73.68 | 87.02 | 1.36\times |
| Minference | 93.46 | 85.34 | 73.63 | 86.49 | 1.12\times |
| + Token Sparse (Ours) | 93.05 | 85.10 | 72.18 | 86.05 | 1.38\times |
| FlexPrefill | 95.48 | 87.20 | 73.75 | 87.27 | 2.44\times |
| + Token Sparse (Ours) | 95.33 | 87.68 | 73.58 | 87.27 | **2.76\times** |

Comparison with token eviction methods at similar speedups (128K, LLaMA-3.1-8B):

| Method | Avg. Accuracy | Speedup |
|---|---|---|
| FlashAttention | 87.01 | 1.00\times |
| PyramidInfer | 78.49 | 1.49\times |
| GemFilter | 85.12 | 1.53\times |
| FastKV | 85.64 | 1.50\times |
| **Token Sparse (Ours)** | **86.84** | 1.51\times |

### Ablation Study

| Configuration | Key Finding | Meaning |
|---|---|---|
| Dynamic $\tau=0.005$ vs Fixed $s=0.3$ | 87.02 vs 86.91 at same speedup | Dynamic budget outperforms fixed ratio |
| Dynamic $\tau=0.010$ vs Fixed $s=0.5$ | 86.84 vs 85.43 at high sparsity | Dynamic advantage grows with aggressive sparsity |
| Speedup Breakdown (128K) | scoring/compress/decompress total overhead < 11% | Implementation is lightweight |
| Sparsity vs. Context Length | 17% at 4K vs 54% at 128K | Long contexts naturally offer more prunable tokens |

### Key Findings
- **Gain over FlashAttention**: Almost no accuracy change (87.01 → 87.02), while contributing a 1.36\times standalone speedup.
- **Orthogonality**: Most valuable when stacked with block-level sparsity (FlexPrefill), increasing speedup from 2.44\times to 2.76\times, proving token-level and block-level sparsity are complementary.
- **Comparison**: Outperforms all token eviction methods at the same speedup, with the gap particularly evident in 4K short contexts (PyramidInfer is 17 points lower than FlashAttn).

## Highlights & Insights
- **Compress-then-Decompress is an elegant "pseudo-sparsity" mechanism**: On the surface, it computes dense $L'\times L'$ attention and fills back into $L\times d$, but the residual channel preserves ignored token information. This acts as a lightweight, reversible, head-specific token selection per layer. This "logical sparsity + physical density" design could be adapted for MoE or sparse expert routing.
- **Zero-Kernel Engineering**: Directly calling FlashAttention/FlexPrefill kernels means zero barrier for downstream users. Compared to token eviction, which requires modifying KV cache structures, the deployment cost is significantly lower.
- **Drift-based Layer Selection as a Robust Prior**: Turning "which layers can handle sparsity" into a data-driven decision rather than a hyperparameter is a generic strategy applicable to layer dropout or pruning tasks.

## Limitations & Future Work
- Relies on recent queries for token scores, which is a heuristic; if a model uses sliding windows or chunked attention, the statistical significance of recent queries might be compromised.
- The residual path preserves unselected token information, but the zero-filled rows per layer scatter operation actually lose cross-attention contributions between selected and unselected tokens; this loss is not quantified.
- When head/layer sparsity varies significantly, different $L'$ values within a batch might disrupt tensor regularity; the paper does not discuss actual throughput for multi-sample batches.
- Only validated on prefill; not used in the decoding stage where the bottleneck is typically KV cache loading rather than attention computation.
- Future directions: Adaptive drift-based selection (per prompt), replacing scoring with a learnable router (end-to-end), or combining with KV cache quantization.

## Related Work & Insights
- **vs Minference / FlexPrefill (Block-level)**: These skip blocks in the attention map, limited by boundaries; Ours selects at the token level and is orthogonal, adding 1.13\times speedup on top of FlexPrefill.
- **vs PyramidInfer / FastKV / GemFilter (Token Eviction)**: These make hard decisions in early layers; Ours allows re-selection at each layer, yielding 1-8 points higher accuracy at similar speedups.
- **vs FlashAttention**: FlashAttention is I/O-optimized dense attention with $O(L^2)$ complexity; Ours provides algorithmic sparsification to $O(L'^2)$ while reusing its kernels.
- **vs KV cache quantization (KIVI/H2O)**: They reduce KV memory I/O; Ours reduces attention compute. They are fully orthogonal and can be combined.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reversible design of Compress-then-Decompress + head-specific selection is a simple but effective new idea; drift-based selection is a clean engineering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage across two models, four baselines, multiple lengths, and benchmarks (RULER/InfiniteBench), including iso-speedup comparisons with eviction methods.
- **Writing Quality**: ⭐⭐⭐⭐ Logical flow from two dynamic observations to design; Figure 3 clearly illustrates the compress-decompress process.
- **Value**: ⭐⭐⭐⭐ Directly applicable to industrial deployment; orthogonality with existing sparse methods is a key selling point for long-context LLM services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ICML 2026\] T3S: Training Trajectory-Aware Token Selection to Break "Imitation Shock" in Reasoning Distillation](training-trajectory-aware_token_selection.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](../../NeurIPS2025/model_compression/recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ACL 2026\] GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts](../../ACL2026/model_compression/glimprouter_efficient_collaborative_inference_by_glimpsing_one_token_of_thoughts.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2025\] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](../../ICML2025/model_compression/orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)
- [\[ICML 2026\] T3S: 训练轨迹感知的 token 选择，破解推理蒸馏的「Imitation Shock」](training-trajectory-aware_token_selection.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](../../NeurIPS2025/model_compression/recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ICML 2026\] Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)

</div>

<!-- RELATED:END -->
