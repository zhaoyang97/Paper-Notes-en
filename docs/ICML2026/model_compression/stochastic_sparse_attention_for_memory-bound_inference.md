---
title: >-
  [Paper Note] Stochastic Sparse Attention for Memory-Bound Inference
description: >-
  [ICML 2026][Model Compression][Sparse Attention] SANTA treats attention value aggregation $AV$ as "weighted sum of value rows $V$ by softmax probabilities $A$…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Sparse Attention"
  - "Random Sampling"
  - "KV-cache"
  - "Stratified Sampling"
  - "GPU kernel"
date: 2026-05-08
content_hash: 9777ce8b208530f9
---

# Stochastic Sparse Attention for Memory-Bound Inference

**Conference**: ICML 2026  
**arXiv**: [2605.01910](https://arxiv.org/abs/2605.01910)  
**Code**: <https://github.com/OPUSLab/SANTA.git>  
**Area**: Model Compression / LLM Inference Acceleration / Attention Optimization  
**Keywords**: Sparse Attention, Random Sampling, KV-cache, Stratified Sampling, GPU kernel

## TL;DR
SANTA treats attention value aggregation $AV$ as "weighted sum of value rows $V$ by softmax probabilities $A$," and replaces it with an unbiased estimator: "sample $S\ll n_k$ indices from $A$ without replacement and directly average the corresponding $V$ rows." Stratified/systematic sampling is used to reduce variance, and the method is implemented as a GPU kernel aligned with FlashDecoding. On 32k context, it achieves 1.5× end-to-end speedup over FlashInfer/FlashDecoding without loss of accuracy.

## Background & Motivation

**Background**: Long-context autoregressive decoding is a major pain point in LLM deployment, as each generated token requires traversing the entire KV cache, making bandwidth the bottleneck (e.g., Llama-3.1-8B with 32k context requires ~128 MB per layer per token). Existing mitigation strategies fall into four categories: KV quantization/compression (e.g., KIVI), cache management (Quest, H2O), structured sparse attention (Longformer, BigBird), and kernel optimization (FlashAttention, FlashDecoding)—often combined with GQA. However, even with optimal exact kernels, every step still touches the entire KV state, so the bandwidth wall remains.

**Limitations of Prior Work**: Top-$k$ or threshold-based sparse methods are **biased estimators** and typically require sorting; quantization/compression degrades KV numerical precision; structured sparsity (e.g., sliding window) sacrifices expressiveness; FlashDecoding has nearly exhausted IO locality, so further acceleration requires **reducing the number of $V$ rows read**, not just optimizing how they are read.

**Key Challenge**: The attention output $AV$ is an **expectation**—$A$ is a probability distribution, so why treat it as deterministic weights for matrix multiplication? Monte Carlo estimation can be used instead. However, random sampling on GPUs disrupts parallelism (requires global CDF), which is the main engineering challenge.

**Goal**: (a) Reformulate $AV$ as an unbiased Monte Carlo estimator, reducing $V$ row access from $n_k$ to $S\ll n_k$, and eliminating all post-softmax multiplications; (b) Reduce variance to match SDPA accuracy; (c) Implement a GPU kernel for real wall-clock speedup; (d) Provide a sparse scheme for the score stage as well (Bernoulli $qK^T$).

**Key Insight**: View attention probabilistically—treat $A$ as a categorical distribution and replace matrix multiplication with sampling; combine "one independent CDF per head" with FlashDecoding's tiling strategy, and use two schemes (proportional / flash) to address the "global CDF vs global synchronization" dilemma.

**Core Idea**: $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$, $i_s\sim A$ i.i.d., which is an unbiased estimator of $AV$ with variance $O(1/S)$; stratified/systematic sampling further reduces variance; on GPU, "lightweight global sync + tile-wise probability mass allocation" avoids serial CDF dependencies.

## Method

### Overall Architecture
SANTA is an **attention replacement scheme for the decoding phase** (can also be used for prefill, but with less benefit). The core consists of: (1) mathematical unbiased estimators SANTA / S²ANTA-strat / S²ANTA-sys; (2) two GPU kernel implementations: S²ANTA-prop (global sync with precise allocation) and S²ANTA-flash (speculative local sampling); (3) Bernoulli $qK^T$ for sparsifying the score stage. Integration: prefill still uses SDPA, only decode-step uses SANTA, and it is orthogonal and stackable with GQA / FlashInfer / quantization.

### Key Designs

1. **SANTA Unbiased Estimator + Stratified/Systematic Variance Reduction**:

    - **Function**: Replaces dense $AV$ with sparse $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$, where $i_s$ are independently sampled from categorical $A$; $V$ row reads drop from $n_k$ to $S$, and after softmax only addition remains, no multiplication.
    - **Mechanism**: Basic SANTA uses i.i.d. sampling, $\mathbb E[\widehat{AV}]=AV$, $\mathrm{Var}\propto 1/S$ (see Appendix A.1, A.2). To reduce variance, **S²ANTA-strat** divides the CDF into $S$ equal-probability segments, samples one per segment: $T_m\sim\mathrm{Unif}(I_m)$, $J_m=F_q^{-1}(T_m)$, $\widehat{AV}=\frac1S\sum V_{J_m}$. **S²ANTA-sys** uses systematic sampling, sampling only one offset $U\sim\mathrm{Unif}[0,1/S)$, thresholds $T_m=U+m/S$—hardware-friendly (one random number generates $S$ samples), empirically matches stratified variance reduction but lacks theoretical guarantee. When $S$ is a power of two, normalization is a bit-shift.
    - **Design Motivation**: Applying a probabilistic perspective to an operator that has long been an inference bottleneck, eliminating both multiplications and read operations; stratified sampling preserves unbiasedness and is naturally parallelizable (each stratum is independent).

2. **S²ANTA-prop: Global Lightweight Sync for Precise Budget Allocation**:

    - **Function**: On GPU, splits attention into $T$ tiles, each tile completes "precise sampling budget allocation by probability mass → parallel sampling and $V$ row gathering" via two kernel passes.
    - **Mechanism**: Pass 1 computes scores and writes exponentiated scores ($1\times n_k$, only $1/d_k$ bandwidth) and tile-local partition function $Z_{tile}$ to global memory; a global reducer sums $Z=\sum Z_{tile}$, then allocates $S_{tile}\propto S\cdot(Z_{tile}/Z)$; Pass 2 uses stashed scores + allocated $S_{tile}$ for systematic sampling + $V$ row gathering. Low-probability tiles with $S_{tile}=0$ skip expensive $V$ reads.
    - **Design Motivation**: Global CDF is inherently serial, but this "lightweights" it: only $T$ scalars are synchronized, not the entire score matrix, making sync cost negligible; with 32k context and $S=128$, matches SDPA accuracy, reducing $V$ row access to < 1.56%.

3. **S²ANTA-flash: Speculative Sampling + Delayed Normalization**:

    - **Function**: Eliminates sync entirely, letting each tile independently sample with "average budget $S/T$," then rescales during merging based on actual $Z$—mirroring FlashDecoding's design philosophy.
    - **Mechanism**: Each tile assumes it holds all probability mass, samples $S/T$ local samples; a second reducer computes true $Z$ and each tile's $Z_{tile}/Z$, shrinking low-probability tile partial sums to near zero. Thus, sampling and $V$ reads in low-probability tiles are "wasted" (sample waste).
    - **Design Motivation**: Provides an option for scenarios intolerant of any global barrier; the tradeoff is a larger total sample count ($S=2048$ vs $S=128$ for prop to match SDPA accuracy), but wall-clock speedup remains at 1.51×.

### Loss & Training
This is a **pure inference method**—no training modification, no loss introduced. All methods are plug-and-play attention operator replacements. Bernoulli $qK^T$ is a complementary score-stage sparsification (Sec 5): normalizes query to $[-1,1]$ as Bernoulli probabilities to sample $\{-1,0,+1\}$ ternary values, forming a sparse ternary query for feature-wise sparse access to the $K$ matrix.

## Key Experimental Results

### Main Results

**32k Long Context RULER (Llama-3.1-8B-Instruct)** Table 1: SDPA for prefill, only decode replaced.

| Kernel | $S$ | FWE | NIAH | QA1 | QA2 |
|---|---|---|---|---|---|
| SDPA (baseline) | – | 95.60 | 98.35 | 64.00 | 58.80 |
| **S²ANTA-prop** | **128** | **95.40** | **98.25** | **64.40** | **60.20** |
| S²ANTA-prop | 256 | 95.47 | 98.50 | 63.40 | 60.60 |
| **S²ANTA-flash** | **2048** | **94.13** | **98.25** | **64.60** | **60.00** |
| S²ANTA-flash | 256 | 66.20 | 88.95 | 63.00 | 57.20 |

With prop, $S=128$ (= 0.39% of $n_k$) achieves SDPA-level accuracy; flash requires $S=2048$ (= 6.25%). Kernel latency (Fig 4): prop 1.50× / flash 1.51× speedup vs FlashInfer.

**GSM8K (Llama 8B)** Table 2 (excerpt): Compares SANTA / S²ANTA-strat / S²ANTA-sys accuracy at different $S$.

| $S$ | S²ANTA-sys | S²ANTA-strat | SANTA |
|---|---|---|---|
| 16 | 44.63 | 39.12 | 5.51 |
| 32 | 68.59 | 67.00 | 38.26 |
| 64 | 76.42 | 74.43 | 63.63 |
| 128 | **77.33** | 75.64 | 70.23 |
| 256 | 77.56 | **78.17** | 75.61 |
| SDPA | – | – | 78.06 |

Variance reduction yields huge gains: at $S=16$, sys outperforms basic SANTA by 39 points.

**MMLU** Table 3: Similarly, stratified variants significantly outperform SANTA at small $S$; at $S=256$, all three are within ±1% of SDPA (49.86 baseline).

### Ablation Study

| Configuration | Key Findings | Notes |
|------|---------|------|
| SANTA vs S²ANTA-strat vs S²ANTA-sys | Stratified variants greatly outperform at $S\le 64$ | Confirms importance of variance reduction |
| prop vs flash kernel | Same wall-clock speedup, prop uses 1/16 the $S$ | Sync cost negligible, significant sample savings |
| Bernoulli $qK^T$ on BitNet 2B (GSM8K) | At $B=4$, only 67.5% K features read, accuracy 64.5% (SDPA 65.7%) | Score stage can also be sparsified, orthogonal to SANTA |
| Mean group query | $B=4$ K access 84.7% (standalone 97.9%) | Mitigates union explosion from GQA sharing |

### Key Findings
- **Sampling eliminates not only multiplications**: In long-context decode, the real gain is reduced $V$ read bandwidth (<2% at 32k); multiplication elimination (1.1 pJ → 0.4 pJ per op) is a benefit only fully realized on future adder-optimized hardware.
- **Stratified variance reduction is essential**: Without variance reduction, SANTA achieves only 5.5% on GSM8K at $S=16$, making it unusable; adding stratified/systematic immediately makes it viable—showing that naive Monte Carlo variance is explosive for attention.
- **Systematic vs stratified**: Empirically, accuracy is nearly identical, but systematic requires only one random number, making it extremely hardware-friendly—a very production-friendly design.
- **Flash kernel's "sample waste" is real**: For the same wall-clock speedup, flash requires 16× more samples, indicating that in highly non-uniform attention distributions, global sync is actually more economical.

## Highlights & Insights
- **Probabilistic perspective on attention** is extremely concise—since softmax already yields a probability distribution, just sample directly. This idea generalizes to all softmax-based operations (mixture-of-experts gating, retrieval ranking).
- **"Multiplication elimination" targets future hardware**: The energy gap between adders and multipliers is large (~0.36×); the paper points to sparse, adder-centric accelerators—perfectly aligned with recent BitNet/1-bit LLM hardware trends.
- **Systematic sampling generates $S$ samples from one random number**, making "sampling" a cheap operation in embedded or custom silicon scenarios—a huge advantage.
- **Prop kernel breaks CDF serial dependency with "lightweight sync"**: This "scalar reduction then budget allocation" design can be applied to any "global normalization for sparsification" task, e.g., sparse softmax MoE routing.
- **Method is plug-and-play**, requires no retraining, preserves accuracy, and is compatible with other techniques (quantization, GQA, cache compression), and can be stacked.

## Limitations & Future Work
- Current GPU kernel wall-clock speedup mainly comes from bandwidth reduction; multiplication elimination benefits are not significant under NVIDIA matrix FMA optimization, requiring future adder-oriented hardware.
- Little benefit in prefill phase—since $n_q=n_k$, $V$ row sparsity is negated by the union; the paper does not claim wall-clock gains for prefill.
- Sampling quality depends on the "well-behavedness" of the softmax distribution; if attention is extremely flat (no clear hotspot), even stratified may be insufficient; the paper does not analyze this worst case.
- Effectiveness of Bernoulli $qK^T$ on non-BitNet models is unknown; standard fp16 models may be less tolerant to query ternarization.
- No experiments combining with cache management methods (Quest, H2O); real-world deployment needs to test combined accuracy.

## Related Work & Insights
- **vs FlashDecoding / FlashInfer (Dao 2023, Ye 2025)**: These are IO optimizations for exact attention, already at the bandwidth ceiling; SANTA is orthogonal (reduces rows accessed), and the paper directly benchmarks 1.5× speedup over them.
- **vs top-$k$ attention (Quest, H2O, etc.)**: Top-$k$ is biased and requires sorting; for large $k$, most $V$ rows are still read; SANTA is unbiased, and with stratified sampling at $S=128$ achieves SDPA accuracy for 32k context.
- **vs Sparse Transformer / Longformer (Child 2019, Beltagy 2020)**: These are structured sparsity, requiring fixed patterns at training; SANTA is random at inference, with no training changes.
- **vs KV quantization (KIVI, Hooper 2024)**: Quantization reduces bytes per element, SANTA reduces the number of elements read; fully complementary and stackable.
- **vs MoE gating / sparse softmax**: Also faces "probabilistic sparsification" challenges; SANTA's prop kernel design is directly transferable.

## Rating
- Novelty: ⭐⭐⭐⭐ Reinterprets attention value stage via Monte Carlo, with stratified/systematic + GPU kernel; not revolutionary but very elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ GSM8K / MMLU / long-context RULER + real GPU kernel latency + Bernoulli $qK^T$ auxiliary experiments all included.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear concepts, Eq.(4) succinctly states the core estimator, prop/flash comparison diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ Open-sourced kernel, plug-and-play, delivers 1.5× long-context acceleration, a must-read for long-context LLM inference teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[NeurIPS 2025\] SpecAttn: Speculating Sparse Attention](../../NeurIPS2025/model_compression/specattn_speculating_sparse_attention.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)
- [\[ICML 2026\] Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)
- [\[ICML 2025\] FloE: On-the-Fly MoE Inference on Memory-constrained GPU](../../ICML2025/model_compression/floe_on-the-fly_moe_inference_on_memory-constrained_gpu.md)

</div>

<!-- RELATED:END -->
