---
title: >-
  [Paper Note] Stochastic Sparse Attention for Memory-Bound Inference
description: >-
  [ICML 2026][LLM Efficiency][Sparse Attention] SANTA treats the value aggregation $AV$ in attention as "a weighted sum of value rows $V$ according to softmax probabilities $A$." It transforms this into an unbiased estimat…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Sparse Attention"
  - "Stochastic Sampling"
  - "KV-cache"
  - "Stratified Sampling"
  - "GPU kernel"
date: 2026-05-08
content_hash: e29c1e9ad5276f4c
---

# Stochastic Sparse Attention for Memory-Bound Inference

**Conference**: ICML 2026  
**arXiv**: [2605.01910](https://arxiv.org/abs/2605.01910)  
**Code**: <https://github.com/OPUSLab/SANTA.git>  
**Area**: Model Compression / LLM Inference Acceleration / Attention Optimization  
**Keywords**: Sparse Attention, Stochastic Sampling, KV-cache, Stratified Sampling, GPU kernel

## TL;DR
SANTA treats the value aggregation $AV$ in attention as "a weighted sum of value rows $V$ according to softmax probabilities $A$." It transforms this into an unbiased estimate by "sampling $S \ll n_k$ indices from $A$ without replacement and directly averaging the corresponding $V$ rows." Using stratified/systematic sampling to reduce variance and implemented as a GPU kernel aligned with FlashDecoding, it achieves a 1.5× end-to-end speedup over FlashInfer/FlashDecoding under a 32k context without accuracy degradation.

## Background & Motivation

**Background**: Long-context autoregressive decoding is a major pain point in LLM deployment. Generating each token requires streaming the entire KV cache, making bandwidth the primary bottleneck (e.g., Llama-3.1-8B with a 32k context requires transferring ~128 MB per layer per token). Existing mitigation methods fall into four categories: KV quantization/compression (KIVI, etc.), cache management (Quest, H2O), structured sparse attention (Longformer, BigBird), and kernel optimization (FlashAttention, FlashDecoding), often combined with GQA. However, even with optimal exact kernels, every step must touch the entire KV state, leaving the bandwidth wall intact.

**Limitations of Prior Work**: Top-$k$ or threshold-based sparse methods are **biased estimators** and typically require sorting. Quantization/compression compromises KV numerical precision. Structured sparsity (e.g., sliding window) sacrifices expressiveness. While FlashDecoding maximizes IO locality, further acceleration requires directly **reducing the number of V rows read**, rather than just optimizing the reading process.

**Key Challenge**: The attention output $AV$ is an **expectation**—since $A$ itself is a probability distribution, why treat it as a deterministic weighted sum? One could use Monte Carlo sampling to compute only the sample sum. However, random sampling on GPUs breaks parallelism as it requires a global CDF, which is the primary engineering difficulty.

**Goal**: (a) Rewrite $AV$ as an unbiased Monte Carlo estimate to reduce V row accesses from $n_k$ to $S \ll n_k$, while eliminating all multiplications after the softmax. (b) Reduce variance sufficiently to match SDPA precision. (c) Develop a GPU kernel that achieves real-world wall-clock acceleration. (d) Provide a sparsification scheme for the score stage (Bernoulli $qK^T$).

**Key Insight**: View attention from a probabilistic perspective—treating $A$ as a categorical distribution and replacing matrix multiplication with sampling. Combine "per-head independent CDFs" with FlashDecoding's tiling strategy, using two schemes (proportional/flash) to resolve the conflict between "global CDF vs. global synchronization."

**Core Idea**: $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$, where $i_s \sim A$ i.i.d. This is an unbiased estimator of $AV$ with variance $O(1/S)$. Variance is further reduced using stratified/systematic sampling. On GPUs, a "lightweight global sync + tile-based probability mass budget allocation" is used to avoid serial CDF dependencies.

## Method

### Overall Architecture
SANTA is an attention replacement scheme for the **decoding stage** (it can be used for prefill, but gains are smaller). The core components include: (1) Unbiased estimators at the mathematical level: SANTA, S²ANTA-strat, and S²ANTA-sys. (2) Two GPU kernel implementations: S²ANTA-prop (exact allocation via global sync) and S²ANTA-flash (speculative local sampling). (3) Bernoulli $qK^T$, which sparsifies the score stage. Integration: SDPA is still used for prefill, while SANTA is used only for the decode steps, remaining orthogonal to and combinable with GQA, FlashInfer, and quantization.

### Key Designs

1.  **SANTA Unbiased Estimation + Stratified/Systematic Variance Reduction**:
    - **Function**: Replaces dense $AV$ with sparse $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$, where $i_s$ is sampled independently from the categorical distribution $A$. V row reads are reduced to $S$, and only additions remain after softmax, eliminating multiplications.
    - **Mechanism**: Basic SANTA uses i.i.d. sampling, where $\mathbb E[\widehat{AV}]=AV$ and $\mathrm{Var}\propto 1/S$ (Appendix A.1, A.2). To reduce variance, **S²ANTA-strat** divides the CDF into $S$ equal probability segments and samples one from each: $T_m\sim\mathrm{Unif}(I_m)$, $J_m=F_q^{-1}(T_m)$, $\widehat{AV}=\frac1S\sum V_{J_m}$. **S²ANTA-sys** uses systematic sampling by taking a single offset $U\sim\mathrm{Unif}[0,1/S)$ with thresholds $T_m=U+m/S$. Systematic sampling is hardware-friendly (one random number generates $S$ samples) and matches stratified variance reduction in practice without theoretical guarantees. When $S$ is a power of 2, normalization is a bit-shift.
    - **Design Motivation**: Applying a probabilistic perspective to an operator that has become a bottleneck effectively eliminates multiplications and reduces reads. Stratified sampling preserves unbiasedness and is naturally parallelizable.

2.  **S²ANTA-prop: Exact Budget Allocation with Lightweight Global Sync**:
    - **Function**: Partitions attention into $T$ tiles on the GPU and uses a two-pass kernel to "allocate sampling budgets exactly based on probability mass → parallel sampling and gathering of V rows."
    - **Mechanism**: Pass 1 calculates scores and writes exponentiated scores ($1\times n_k$, occupying $1/d_k$ bandwidth) and tile-local partition functions $Z_{tile}$ to global memory. A global reducer sums $Z=\sum Z_{tile}$ and allocates $S_{tile}\propto S\cdot(Z_{tile}/Z)$. Pass 2 uses stashed scores + allocated $S_{tile}$ systematic sampling + gather $V$ rows. Low-probability tiles with $S_{tile}=0$ bypass expensive V-reads entirely.
    - **Design Motivation**: A global CDF is inherently serial, but it can be "leaned out": syncing $T$ scalars instead of the whole score matrix makes synchronization costs negligible. For a 32k context, $S=128$ aligns with SDPA accuracy, reducing V row accesses to < 1.56%.

3.  **S²ANTA-flash: Speculative Sampling + Delayed Normalization**:
    - **Function**: Completely removes synchronization, allowing each tile to sample independently based on an "average budget $S/T$," scaling during the final merge based on true $Z$—directly mirroring FlashDecoding's design philosophy.
    - **Mechanism**: Each tile assumes it holds the total probability mass and samples $S/T$ times to get a local partial sum. A second-pass reducer calculates the true $Z$ and $Z_{tile}/Z$ for each tile, scaling partial sums of "low-probability tiles" toward 0. Thus, sampling and V-reads for low-probability tiles result in "sample waste."
    - **Design Motivation**: Provides an alternative for scenarios where global barriers cannot be tolerated. The cost is a larger total sample count ($S=2048$ vs. $S=128$ for prop to align with SDPA), though it still achieves a 1.51× wall-clock speedup.

### Loss & Training
This is a **purely inference-time method** that requires no training or additional loss functions. All methods are plug-and-play replacements for the attention operator. Bernoulli $qK^T$ acts as a complementary score-stage sparsification (Sec 5): it normalizes queries to $[-1,1]$ as Bernoulli probabilities to sample ternary values $\{-1,0,+1\}$, forming a sparse ternary query for feature-wise sparse access to the K matrix.

## Key Experimental Results

### Main Results

**32k Long Context RULER (Llama-3.1-8B-Instruct)** Table 1: SDPA for prefill, replacement for decode only.

| Kernel | $S$ | FWE | NIAH | QA1 | QA2 |
|---|---|---|---|---|---|
| SDPA (baseline) | – | 95.60 | 98.35 | 64.00 | 58.80 |
| **S²ANTA-prop** | **128** | **95.40** | **98.25** | **64.40** | **60.20** |
| S²ANTA-prop | 256 | 95.47 | 98.50 | 63.40 | 60.60 |
| **S²ANTA-flash** | **2048** | **94.13** | **98.25** | **64.60** | **60.00** |
| S²ANTA-flash | 256 | 66.20 | 88.95 | 63.00 | 57.20 |

Prop achieves SDPA-level accuracy with $S=128$ (0.39% of $n_k$), while flash requires $S=2048$ (6.25%). Kernel latency (Fig 4): prop 1.50× / flash 1.51× speedup vs. FlashInfer.

**GSM8K (Llama 8B)** Table 2 (Excerpt): Comparing accuracy of SANTA / S²ANTA-strat / S²ANTA-sys at different $S$.

| $S$ | S²ANTA-sys | S²ANTA-strat | SANTA |
|---|---|---|---|
| 16 | 44.63 | 39.12 | 5.51 |
| 32 | 68.59 | 67.00 | 38.26 |
| 64 | 76.42 | 74.43 | 63.63 |
| 128 | **77.33** | 75.64 | 70.23 |
| 256 | 77.56 | **78.17** | 75.61 |
| SDPA | – | – | 78.06 |

Variance reduction makes a huge difference: at $S=16$, sys is 39 points higher than basic SANTA.

**MMLU** Table 3: Similarly, the stratified series significantly outperform SANTA at small $S$. At $S=256$, all three return to within ±1% of SDPA (49.86 baseline).

### Ablation Study

| Configuration | Key Finding | Description |
|------|---------|------|
| SANTA vs S²ANTA-strat vs S²ANTA-sys | Stratified series lead significantly when $S\le 64$ | Validates the criticality of variance reduction. |
| prop vs flash kernel | Same wall-clock speedup; prop uses 1/16 the $S$ | Sync cost is negligible; significantly reduces sample waste. |
| Bernoulli $qK^T$ on BitNet 2B (GSM8K) | Reads 67.5% K features at $B=4$, with 64.5% accuracy | Score stage can also be sparsified, orthogonal to SANTA. |
| Mean group query | K access 84.7% (vs 97.9% alone) | Mitigates union explosion caused by GQA sharing. |

### Key Findings
- **Sampling does more than eliminate multiplication**: In long-context decoding, the real gain comes from the reduction in V read bandwidth (< 2% at 32k). Multiplication elimination (1.1 pJ → 0.4 pJ per op) is a dividend that fully realizes only with future adder-optimized hardware.
- **Stratified variance reduction is mandatory**: Without variance reduction, SANTA achieves only 5.5% on GSM8K at $S=16$, making it unusable. Adding stratified/systematic sampling makes it immediately viable, indicating that naive Monte Carlo suffers from variance explosion in attention.
- **Systematic vs. Stratified**: Measured accuracy is nearly identical, but systematic sampling requires only one random number, making it extremely hardware-friendly.
- **"Sample waste" in Flash kernel is real**: For the same wall-clock speedup, flash requires 16× more samples, showing that in highly non-uniform attention distributions, global sync is more economical.

## Highlights & Insights
- **Probabilistic view of attention** is a clean operation—since softmax already provides a probability distribution, sampling is intuitive. This idea can be generalized to any softmax-based operation (MoE gating, retrieval ranking).
- **"Eliminating multiplications" caters to future hardware**: The energy efficiency of adders vs. multipliers is significant (~0.36×). The paper explicitly points toward sparse, adder-centric accelerators, aligning with the hardware trends of BitNet and 1-bit LLMs.
- **Systematic sampling with one random number** is a major advantage in embedded or custom silicon scenarios where sampling needs to be a "cheap" operation.
- **Prop kernel's "lightweight sync" breaks the CDF serial bottleneck**: This design of "scalar reduction first, then budget allocation" can be applied to any sparsification task requiring global normalization, such as sparse softmax MoE routing.
- **The method is plug-and-play**, requiring no retraining, preserving accuracy, and not conflicting with existing techniques (quantization, GQA, cache compression).

## Limitations & Future Work
- Current GPU kernel speedup primarily comes from bandwidth reduction. Gains from multiplier elimination are modest due to heavily optimized matrix FMA units on NVIDIA GPUs, necessitating new adder-oriented hardware.
- The prefill stage shows almost no gain because $n_q=n_k$, so the union of V row reads negates the sparsity. The paper does not claim wall-clock benefits for prefill.
- Sampling quality depends on the "well-behavedness" of the softmax distribution. If the distribution is extremely flat (no hotspots), even stratified sampling may be insufficient; the paper does not analyze this worst-case scenario.
- The effect of Bernoulli $qK^T$ on non-BitNet models is unknown; standard fp16 models may be less tolerant of query ternaryization.
- Combined experiments with cache management methods (Quest, H2O) were not conducted; accuracy when stacking these methods needs verification for production.

## Related Work & Insights
- **vs. FlashDecoding / FlashInfer (Dao 2023, Ye 2025)**: These are IO optimizations for exact attention reaching the bandwidth ceiling. SANTA is an orthogonal direction (reducing accessed rows), achieving 1.5× speedup over these baselines.
- **vs. Top-$k$ Attention (Quest, H2O, etc.)**: Top-$k$ is biased, requires sorting, and still reads most V rows when $k$ is large. SANTA is unbiased and hits SDPA accuracy at 32k context with $S=128$.
- **vs. Sparse Transformer / Longformer (Child 2019, Beltagy 2020)**: These use structured sparsity fixed during training. SANTA is stochastic at inference and does not change training.
- **vs. KV Quantization (KIVI, Hooper 2024)**: Quantization reduces bytes per element, while SANTA reduces the number of elements read. They are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Reinterpreting the attention value stage via Monte Carlo, paired with stratified/systematic sampling and GPU kernels; elegant rather than revolutionary.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete results across GSM8K, MMLU, and RULER, alongside real GPU latencies and Bernoulli $qK^T$ experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear concepts; Equation (4) summarizes the core estimator effectively; intuitive comparison between prop and flash kernels.
- Value: ⭐⭐⭐⭐⭐ Open-sourced kernel providing plug-and-play 1.5× acceleration for long contexts; a must-read for LLM inference teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prism: Spectral-Aware Block-Sparse Attention](prism_spectral-aware_block-sparse_attention.md)
- [\[ICML 2026\] Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ICML 2026\] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)
- [\[ICCV 2025\] MixANT: Observation-dependent Memory Propagation for Stochastic Dense Action Anticipation](../../ICCV2025/llm_efficiency/mixant_observation-dependent_memory_propagation_for_stochastic_dense_action_anti.md)

</div>

<!-- RELATED:END -->
