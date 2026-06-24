---
title: >-
  [Paper Note] Stochastic Sparse Attention for Memory-Bound Inference
description: >-
  [ICML 2026][LLM Efficiency][Sparse Attention] SANTA treats the value aggregation $AV$ of attention as an "expectation of value rows $V$ weighted by softmax probabilities $A$." It transforms this into an unbiased estimator that samples $S \ll n_k$ indices from $A$ without replacement and directly averages the corresponding $V$ rows. By employing stratified/systematic sampling to reduce variance and implementing GPU kernels aligned with FlashDecoding…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Sparse Attention"
  - "Random Sampling"
  - "KV-cache"
  - "Stratified Sampling"
  - "GPU kernel"
date: 2026-05-08
content_hash: 599056acf7355789
---

# Stochastic Sparse Attention for Memory-Bound Inference

**Conference**: ICML 2026  
**arXiv**: [2605.01910](https://arxiv.org/abs/2605.01910)  
**Code**: <https://github.com/OPUSLab/SANTA.git>  
**Area**: Model Compression / LLM Inference Acceleration / Attention Optimization  
**Keywords**: Sparse Attention, Random Sampling, KV-cache, Stratified Sampling, GPU kernel

## TL;DR
SANTA treats the value aggregation $AV$ of attention as an "expectation of value rows $V$ weighted by softmax probabilities $A$." It transforms this into an unbiased estimator that samples $S \ll n_k$ indices from $A$ without replacement and directly averages the corresponding $V$ rows. By employing stratified/systematic sampling to reduce variance and implementing GPU kernels aligned with FlashDecoding, SANTA achieves a 1.5× end-to-end speedup compared to FlashInfer/FlashDecoding in 32k context scenarios without accuracy degradation.

## Background & Motivation

**Background**: Autoregressive decoding with long contexts is a bottleneck in LLM deployment. Each generated token requires streaming the entire KV cache, making bandwidth the primary constraint (e.g., Llama-3.1-8B with a 32k context requires transferring ~128 MB per layer per token). Current mitigation strategies fall into four categories: KV quantization/compression (e.g., KIVI), cache management (Quest, H2O), structured sparse attention (Longformer, BigBird), and kernel optimization (FlashAttention, FlashDecoding)—often stacked with GQA. However, even with optimal exact kernels, the entire KV state must be accessed at each step, leaving the bandwidth wall intact.

**Limitations of Prior Work**: Top-$k$ or threshold-based sparse methods are **biased estimators** and typically require overhead-heavy sorting. Quantization and compression compromise the numerical precision of KV states. Structured sparsity (e.g., sliding window) sacrifices representational capacity. While FlashDecoding maximizes IO locality, further acceleration requires directly **reducing the number of V rows read**, rather than just optimizing the reading process.

**Key Challenge**: The attention output $AV$ is an **expectation**—$A$ itself represents a probability distribution. Why treat it as a deterministic weighted sum for matrix multiplication? It can be computed more efficiently via Monte Carlo sampling using only a subset of rows. However, implementing random sampling on GPUs typically breaks parallelism due to the need for a global Cumulative Distribution Function (CDF), posing a significant engineering challenge.

**Goal**: (a) Reformulate $AV$ as an unbiased Monte Carlo estimator to reduce $V$ row accesses from $n_k$ to $S \ll n_k$, effectively eliminating multiplications post-softmax; (b) reduce variance to match SDPA precision; (c) develop a GPU kernel that achieves real-world wall-clock speedup; (d) provide a sparsification scheme for the score stage (Bernoulli $qK^T$).

**Key Insight**: View attention from a probabilistic perspective—treat $A$ as a categorical distribution and replace matrix multiplication with sampling. By combining "per-head independent CDFs" with the tiling strategy of FlashDecoding, the conflict between "global CDF" and "global synchronization" can be resolved using two approaches: proportional and flash.

**Core Idea**: $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$, where $i_s\sim A$ i.i.d. is an unbiased estimator of $AV$ with variance $O(1/S)$. Variance is further reduced using stratified or systematic sampling. On GPUs, "lightweight global sync + per-tile probability mass budget allocation" is used to avoid serial CDF dependencies.

## Method

### Overall Architecture
SANTA is an attention replacement scheme designed for the **decoding phase** (it is applicable to prefill but yields smaller gains). It divides attention into two stages: the score stage, which computes $qK^T$ followed by softmax to obtain distribution $A$, and the value stage, which computes $AV$ to aggregate values. While both stages are sparsified, the **core contribution lies in the value stage**, as the bandwidth wall in long-context decoding stems from repeatedly reading the entire $V$ cache. In the value stage, **SANTA unbiased estimation and stratified variance reduction** convert the "weighted sum of all $n_k$ rows" into a "direct average of $S \ll n_k$ sampled rows," eliminating multiplications after softmax. This is implemented via two GPU kernels: **S²ANTA-prop** (lightweight global sync for precise budget allocation) and **S²ANTA-flash** (barrier-free speculative local sampling) to achieve wall-clock acceleration. As an orthogonal supplement, the score stage utilizes **Bernoulli $qK^T$** to ternarize queries for sparse $K$ feature access. Prefill still uses SDPA; SANTA replaces only the decoding steps and is compatible with GQA, FlashInfer, quantization, and cache compression.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    Q["Decode step: query q + KV cache"] --> SC["Score stage: q·Kᵀ → softmax<br/>to obtain distribution A"]
    BN["Bernoulli qKᵀ sparsification<br/>Query ternarization {−1,0,+1}<br/>Unbiased qKᵀ estimation, sparse K access"] -.Orthogonal supplement.-> SC
    SC --> EST["SANTA unbiased estimation + stratified variance reduction<br/>Sample S indices from A (S << n_k)<br/>Gather-and-add V rows, no post-softmax multiplication"]
    EST --> K{"Choose GPU kernel"}
    K -->|Global lightweight sync (Precise)| PROP
    K -->|No barrier (Speculative)| FLASH
    subgraph PROP["S²ANTA-prop"]
        direction TB
        P1["Pass 1: Calculate exact scores and stash<br/>Output Z_tile for each tile"] --> RED["Global reducer: Z = ΣZ_tile<br/>Budget S_tile allocation via Z_tile/Z<br/>Low-prob tiles get 0, skip V-read"]
        RED --> P2["Pass 2: Systematic sampling via S_tile<br/>Gather corresponding V rows"]
    end
    subgraph FLASH["S²ANTA-flash"]
        direction TB
        F1["Each tile samples S/T points<br/>Assuming local dominance of probability"] --> F2["Merge via actual Z_tile/Z<br/>Delayed scaling, low-prob tiles scaled to 0"]
    end
    PROP --> OUT["Unbiased AV estimation output<br/>V row access reduced to < 2%"]
    FLASH --> OUT
```

### Key Designs

**1. SANTA Unbiased Estimation + Stratified/Systematic Variance Reduction**

The $AV$ calculation in the value stage is essentially a weighted sum of $V$ rows based on softmax probabilities $A$, which is an expectation. Since $A$ is already a probability distribution, SANTA replaces the full dot-product with a Monte Carlo estimate: sample $S \ll n_k$ indices $i_s$ independently from the categorical distribution $A$ and output $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$. This is an unbiased estimator ($\mathbb E[\widehat{AV}]=AV$ with variance $\propto 1/S$) that simultaneously reduces $V$ row reads to $S$ and eliminates all multiplications post-softmax by using gather-and-add operations. To address the high variance of naive i.i.d. sampling (where GSM8K drops to 5.5% with $S=16$), S²ANTA introduces stratified sampling: the CDF is partitioned into $S$ equal probability intervals, with one sample per interval to ensure uniform coverage. **S²ANTA-strat** draws an independent offset $T_m\sim\mathrm{Unif}(I_m)$ for each interval and takes $J_m=F_q^{-1}(T_m)$, providing theoretical guarantees for variance reduction. **S²ANTA-sys** uses a single global offset $U\sim\mathrm{Unif}[0,1/S)$ to generate all $S$ samples via thresholds $T_m=U+m/S$. Though lacking theoretical guarantees, systematic sampling performs on par with stratified sampling in practice while requiring only one random number, making it highly hardware-friendly.

**2. S²ANTA-prop: Precise Budget Allocation via Lightweight Global Sync**

Implementing the sampler on GPUs is difficult because determining which $V$ rows to sample typically requires a global CDF—a serial dependency that violates the parallel split-KV philosophy of FlashDecoding. S²ANTA-prop resolves this by "lightening" the global normalization to synchronize only $T$ scalars. It treats attention in $T$ tiles and executes two passes: **Pass 1** computes scores precisely, stashing exponentiated scores (small scalars occupying $1/d_k$ bandwidth) and local partition functions $Z_{tile}$ in global memory. A **global reducer** sums $Z=\sum Z_{tile}$ and allocates the sampling budget $S_{tile}\propto S\cdot(Z_{tile}/Z)$ proportionally. Tiles with low probability receive $S_{tile}=0$ and skip the expensive $V$-read. **Pass 2** performs systematic sampling using the stashed scores and allocated budgets to gather $V$ rows. The barrier only synchronizes $T$ scalars rather than the full score matrix, making the cost negligible while ensuring precise load balancing. In 32k contexts, $S=128$ (0.39% of KV) matches SDPA accuracy, reducing $V$ row access below 1.56% and achieving a 1.50× speedup over FlashInfer.

**3. S²ANTA-flash: Speculative Sampling + Delayed Normalization**

For scenarios where any global barrier is unacceptable, S²ANTA-flash removes synchronization entirely, following the FlashDecoding philosophy. Each tile assumes it holds the entire probability mass and samples $S/T$ points to produce a local sum. The reducer later calculates the actual $Z$ and ratios $Z_{tile}/Z$, applying **delayed scaling** to squash partial sums from low-probability tiles toward zero. The downside is "sample waste," where sampling and $V$-reads in low-probability tiles are essentially discarded. Consequently, matching SDPA accuracy requires a significantly larger budget ($S=2048$ vs. $S=128$ for prop). However, by eliminating the barrier, it still achieves a 1.51× wall-clock speedup. This highlights that in highly non-uniform distributions like attention, **spending a small overhead on global sync is more efficient than speculative execution.**

**4. Bernoulli $qK^T$: Orthogonal Sparsification of the Score Stage**

While the previous designs target the value stage ($AV$), the score stage ($qK^T$) also requires a full pass over $K$. Bernoulli $qK^T$ provides an orthogonal supplement by normalizing query elements to $[-1,1]$ and treating them as Bernoulli probabilities to produce a sparse ternary query $\{-1,0,+1\}$. This yields an unbiased estimate of $qK^T$ and allows feature-wise sparse access to $K$. This extends stochastic sparsification to both branches of attention. On BitNet-2B with $B=4$, it reads only 67.5% of $K$ features with 64.5% accuracy (vs. 65.7% for SDPA). The primary acceleration comes from the value stage, while Bernoulli is a complementary mechanism validated on BitNet-like models.

### Loss & Training
Ours is a **pure inference-time method** that requires no retraining or additional loss functions. All components (including Bernoulli $qK^T$) are plug-and-play replacements for attention operators and are orthogonal to quantization, GQA, and cache compression.

## Key Experimental Results

### Main Results

**32k Context RULER (Llama-3.1-8B-Instruct)** Table 1: SDPA used for prefill, replacement in decode only.

| Kernel | $S$ | FWE | NIAH | QA1 | QA2 |
|---|---|---|---|---|---|
| SDPA (baseline) | – | 95.60 | 98.35 | 64.00 | 58.80 |
| **S²ANTA-prop** | **128** | **95.40** | **98.25** | **64.40** | **60.20** |
| S²ANTA-prop | 256 | 95.47 | 98.50 | 63.40 | 60.60 |
| **S²ANTA-flash** | **2048** | **94.13** | **98.25** | **64.60** | **60.00** |
| S²ANTA-flash | 256 | 66.20 | 88.95 | 63.00 | 57.20 |

Prop achieves SDPA-level accuracy at $S=128$ (0.39% of $n_k$), while flash requires $S=2048$ (6.25%). Kernel latency (Fig 4): prop 1.50× / flash 1.51× speedup vs. FlashInfer.

**GSM8K (Llama 8B)** Table 2 (excerpt): Comparing accuracy of SANTA / S²ANTA-strat / S²ANTA-sys across different $S$.

| $S$ | S²ANTA-sys | S²ANTA-strat | SANTA |
|---|---|---|---|
| 16 | 44.63 | 39.12 | 5.51 |
| 32 | 68.59 | 67.00 | 38.26 |
| 64 | 76.42 | 74.43 | 63.63 |
| 128 | **77.33** | 75.64 | 70.23 |
| 256 | 77.56 | **78.17** | 75.61 |
| SDPA | – | – | 78.06 |

Variance reduction is critical: at $S=16$, sys outperforms basic SANTA by 39 points.

**MMLU** Table 3: Stratified variants significantly outperform SANTA at small $S$. All three converge within ±1% of SDPA (49.86 baseline) at $S=256$.

### Ablation Study

| Configuration | Key Finding |
|------|---------|
| SANTA vs. S²ANTA-strat vs. S²ANTA-sys | Stratified variants lead significantly when $S \le 64$, validating variance reduction. |
| Prop vs. Flash kernel | Similar speedups, but prop uses 1/16 the budget of flash, demonstrating efficiency over sample waste. |
| Bernoulli $qK^T$ on BitNet 2B (GSM8K) | Reading 67.5% of K features at $B=4$ yields 64.5% accuracy, proving score-stage sparsification is viable. |
| Mean group query | $B=4$ results in 84.7% K access (vs 97.9% alone), mitigating union explosion from GQA sharing. |

### Key Findings
- **Sampling goes beyond eliminating multiplications**: The primary gain in long-context decoding stems from reduced $V$ read bandwidth ( < 2% at 32k context). Multiplication elimination (energy reduction from 1.1 pJ to 0.4 pJ per op) is a dividend for future adder-optimized hardware.
- **Stratified variance reduction is mandatory**: Without it, SANTA fails on GSM8K (5.5% at $S=16$). Adding stratified/systematic sampling makes it immediately viable, proving that naive Monte Carlo suffers from variance explosion in attention.
- **Systematic vs. Stratified**: Performance is nearly identical, but systematic sampling's single random number requirement is highly hardware-friendly.
- **Flash kernel "sample waste" is significant**: To match wall-clock speedup, flash needs 16× more samples than prop, indicating that global sync is remarkably economical for non-uniform attention distributions.

## Highlights & Insights
- **Probabilistic Perspective**: Reinterpreting attention through sampling is elegant. This idea can be extended to other softmax-based operations like MoE gating or retrieval ranking.
- **"Multiplier Elimination" for Future Hardware**: The large energy gap between adders and multipliers (~0.36×) aligns SANTA with the trend of 1-bit LLMs and adder-centric accelerators.
- **Cheap Sampling**: Systematic sampling using 1 random number for $S$ samples is a major advantage for embedded or custom silicon deployments.
- **Breaking CDF Seriality**: The prop kernel's "lightweight sync" design can be applied to any sparsification task requiring global normalization, such as sparse softmax MoE routing.
- **Plug-and-Play**: The method requires no retraining, maintains accuracy, and is compatible with existing tools like quantization and GQA.

## Limitations & Future Work
- Current GPU speedups primarily come from bandwidth reduction. Multiplication elimination dividends are less pronounced under NVIDIA's FMA optimizations; benefits will increase with adder-oriented hardware.
- Prefill stage shows minimal gains because $n_q = n_k$ causes the union of sampled $V$ rows to span nearly the entire cache.
- Sampling quality depends on the "well-behaved" nature of the softmax distribution. If attention is extremely flat, even stratified sampling might struggle.
- Bernoulli $qK^T$ effectiveness on non-BitNet (fp16) models remains unproven; tolerance for query ternarization might be lower.
- Combined experiments with cache management methods (Quest, H2O) are needed for production environments.

## Related Work & Insights
- **vs. FlashDecoding / FlashInfer (Dao 2023, Ye 2025)**: These optimize IO for exact attention. SANTA is an orthogonal approach that reduces the amount of data accessed, achieving 1.5× speedup over these baselines.
- **vs. Top-$k$ Attention (Quest, H2O)**: Top-$k$ is biased and requires sorting. SANTA is unbiased and requires only $S=128$ to match SDPA at 32k context.
- **vs. Sparse Transformer / Longformer**: These utilize structured sparsity patterns fixed at training. SANTA is stochastic at inference time and requires no training changes.
- **vs. KV Quantization (KIVI)**: Quantization reduces bytes per element, while SANTA reduces the number of elements read. They are perfectly complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Re-interpreting the value stage through Monte Carlo sampling with stratified/systematic implementations is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ covers GSM8K, MMLU, RULER, real kernel latency, and Bernoulli $qK^T$.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts are clear; the core estimator in Eq.(4) and the prop/flash comparisons are intuitive.
- Value: ⭐⭐⭐⭐⭐ Open-sourcing kernels for 1.5× long-context acceleration is highly valuable for LLM inference teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prism: Spectral-Aware Block-Sparse Attention](prism_spectral-aware_block-sparse_attention.md)
- [\[ICML 2026\] Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)
- [\[ICML 2026\] FOCUS: DLLMs Know How to Tame Their Compute Bound](focus_dllms_know_how_to_tame_their_compute_bound.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ICCV 2025\] MixANT: Observation-dependent Memory Propagation for Stochastic Dense Action Anticipation](../../ICCV2025/llm_efficiency/mixant_observation-dependent_memory_propagation_for_stochastic_dense_action_anti.md)

</div>

<!-- RELATED:END -->
