---
title: >-
  [Paper Note] Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference
description: >-
  [ICML 2026 (Workshop on Structured Probabilistic Inference & Generative Modeling)][LLM/NLP][Diffusion LLM] Aiming at the parallel decoding bottleneck of diffusion language models (dLLMs)…
tags:
  - "ICML 2026 (Workshop on Structured Probabilistic Inference & Generative Modeling)"
  - "LLM/NLP"
  - "Diffusion LLM"
  - "Parallel Decoding"
  - "Fréchet Lower Bound"
  - "Confidence Profile"
  - "Heterogeneity Reward"
date: 2026-05-08
content_hash: 39cb3b0cb00bef9b
---

# Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference

**Conference**: ICML 2026 (Workshop on Structured Probabilistic Inference & Generative Modeling)  
**arXiv**: [2606.02955](https://arxiv.org/abs/2606.02955)  
**Code**: https://github.com/Ringo-Star/FastdLLM_plusplus (Available)  
**Area**: LLM Efficiency / Diffusion Language Models / Parallel Decoding  
**Keywords**: Diffusion LLM, Parallel Decoding, Fréchet Lower Bound, Confidence Profile, Heterogeneity Reward

## TL;DR
Aiming at the parallel decoding bottleneck of diffusion language models (dLLMs), this paper proposes training-free Fréchet Profile Decoding. It uses the entire sorted confidence profile, rather than just the "weakest selected token," to determine how many tokens to commit in each parallel step. This strictly generalizes the factor rule of Fast-dLLM to heterogeneous confidence scenarios. On LLaDA-8B across four benchmarks, it achieves an average throughput of 1.36× and a 29% reduction in NFE with nearly no loss in accuracy.

## Background & Motivation
**Background**: Masked diffusion LLMs (MDLMs) start from a fully masked sequence and predict the marginal distributions of all masked positions in parallel at each step. Theoretically, they can commit multiple tokens at once to achieve throughput far exceeding autoregressive models. However, a gap exists between the "product of marginal distributions" and the true joint distribution. The more tokens are paralleled, the more severe the "curse of parallelism" becomes—tokens that look correct individually may be incoherent when combined.

**Limitations of Prior Work**: Fast-dLLM mitigates this issue using confidence-aware parallel decoding with two rules: the threshold rule $c_i \ge \tau$ for independent commits, and the factor rule $(n+1)(1-c_{(n)}) < f$ to determine the acceptance of the top $n$ candidates. The theoretical basis for the factor rule is a "high-confidence assumption," which assumes that the confidence of all $n$ selected tokens is equal to $c_{(n)}$ (the weakest one), effectively "flattening" the entire confidence profile to its minimum value.

**Key Challenge**: In practice, confidence profiles in decoding steps are highly heterogeneous—for example, $(0.99, 0.95, 0.82, 0.78, 0.74)$. The first few tokens are nearly certain, while the subsequent ones weaken gradually. The factor rule replaces the entire profile with a flat proxy of $(0.82, 0.82, 0.82)$, discarding the safety information provided by "strong tokens," leading to the conservative rejection of tokens that could have been committed together.

**Goal**: To generalize the factor rule from "homogeneous confidence" to the optimal marginal-only proof for "heterogeneous confidence" without modifying the model, diffusion process, or KV cache, and to translate this generalization into a larger set of parallel committed tokens.

**Key Insight**: Starting from the classic Fréchet–Hoeffding / Bonferroni inequalities—when only the marginal probabilities of individual events are known, the distribution-free tight lower bound for the probability of their intersection is $L_n = \max\{0, \sum_{j=1}^n c_{(j)} - (n-1)\}$. Applying this bound to the event "all selected marginal-argmax tokens are simultaneously correct" yields a safety certificate that utilizes the entire profile.

**Core Idea**: Use the Fréchet lower bound $L_n$ (lower bound of joint correctness probability) minus the competitor upper bound $U_n = 1 - c_{(n)}$ (upper bound for the probability of any other tuple) as the "safety margin" $G_n = L_n - U_n$. Committing the maximum prefix $n^*$ satisfying $G_n > \delta$ allows for strictly more tokens to be accepted under heterogeneous profiles than the factor rule.

## Method

### Overall Architecture
Fast-dLLM++ is a drop-in replacement for Fast-dLLM. Within each denoising step, after the forward pass and obtaining marginal predictions $p_\theta(X_i = v \mid x_k)$ for all masked positions, the argmax and corresponding confidence $c_i$ for each position are calculated. The confidence values $c_i$ are sorted in descending order as $c_{(1)} \ge c_{(2)} \ge \cdots \ge c_{(m)}$. A new "profile-aware" selector then determines how many tokens to commit, while the remaining positions remain masked for the next step. The model, diffusion schedule, and cache modes (NONE / PrefixCache / DualCache) remain completely unchanged; only the token selection logic in lines 11–18 of Fast-dLLM Algorithm 1 is replaced. The complexity increases only by one sorting operation and a prefix sum.

### Key Designs

1.  **Fréchet Profile Certificate (Theorem 4.1)**:
    - **Function**: Provides a marginal-only, distribution-free sufficient condition for "the $n$ selected marginal-argmax tokens being the true joint maximum."
    - **Mechanism**: Defines the Fréchet lower bound $L_n = \max\{0, \sum_{j=1}^n c_{(j)} - (n-1)\}$ for the top $n$ candidates and a competitor upper bound $U_n = 1 - c_{(n)}$ (the upper bound for any tuple differing by at least one position from $x^\star_S$). If $L_n > U_n$, the selected tuple is the unique maximizer of the true joint distribution $P_S$. This bound, derived from Fréchet–Hoeffding / Bonferroni inequalities, is the tightest distribution-free lower bound given only marginal probabilities.
    - **Design Motivation**: Upgrade the "weakest token safety check" of Fast-dLLM to a "profile-utilizing safety check," which theoretically encompasses the factor rule as a homogeneous special case and rewards the safety margin provided by strong tokens.

2.  **Profile-aware Selection Rule & Algorithm 1**:
    - **Function**: Scans the candidate count $n$ from 1 to $m$ in each denoising step to identify the largest "safe" commit set size $n^* = \max\{n: G_n > \delta\}$, where $G_n = L_n - U_n$ and $\delta \ge 0$ is a user-defined margin. If no $n$ satisfies the condition, $n^* = 1$ is used to ensure progress.
    - **Mechanism**: Requires only sorting the pre-calculated confidence vector, calculating prefix sums, and computing $L_n, U_n, G_n$ iteratively. No additional forward passes are needed. This is implemented by replacing lines 11–18 of Fast-dLLM Algorithm 1. The algorithm is transparent to underlying caches and introduces negligible computation.
    - **Design Motivation**: Use the tightest marginal-only certificate to make "maximum available parallelism per step" decisions, translating theoretical improvements into engineering gains in tokens-per-step.

3.  **Heterogeneity Reward Decomposition (Proposition 4.3 + Corollary 4.4)**:
    - **Function**: Explains why Fréchet decoding is strictly more aggressive than matched factor decoding ($f = 1 - \delta$) and quantifies the source of "extra commits."
    - **Mechanism**: When $L_n > 0$, the score is decomposed as $G_n = F_n + B_n$, where the factor kernel $F_n = (n+1)c_{(n)} - n$ relies only on the weakest confidence (equivalent to the factor rule), and the heterogeneity reward $B_n = \sum_{j=1}^{n-1}(c_{(j)} - c_{(n)}) \ge 0$ is the area between the profile and the "flat weakest line." Equivalently, Fréchet acts as a "data-adaptive" factor: $f_{\text{eff}}(n) = 1 - \delta + B_n$. Corollary 4.4 proves that a prefix accepted by matched factor is always accepted by Fréchet, while Fréchet accepts strictly more iff $F_n \le \delta < F_n + B_n$.
    - **Design Motivation**: Transform "why it works" from empirical observation into a calculable, profile-interpretable quantity; reinterpret Fast-dLLM's success as a special case within the marginal-only framework.

### Loss & Training
Completely training-free. $\delta$ is the only new hyperparameter, default $\delta = 0.25$ (corresponding to matched factor $f = 0.75$). The paper also provides a calibration-robust variant (Appendix C) using conservative lower bounds of confidence.

## Key Experimental Results

### Main Results
LLaDA-8B-Instruct, PrefixCache, block size 32, single H100; threshold $\tau = 0.9$, factor $f = 0.75$, Fréchet $\delta = 0.25$.

| Dataset (Len) | Method | Acc (%) | Tok/s ↑ | NFE ↓ | Tok/NFE |
|---|---|---|---|---|---|
| GSM8K 5-shot (256) | Threshold | 77.6 | 73.8 (1.00×) | 107,135 | 2.88 |
| GSM8K 5-shot (256) | Factor | 78.1 | 96.0 (1.30×) | 79,047 (↓26.2%) | 3.90 |
| GSM8K 5-shot (256) | **Fréchet** | 77.2 | **103.8 (1.41×)** | **72,881 (↓32.0%)** | **4.24** |
| MATH 4-shot (256) | Fréchet | 32.5 | 102.5 (1.38×) | 358,178 (↓28.8%) | 3.48 |
| HumanEval (256) | Fréchet | 40.9 | 107.7 (1.38×) | 9,740 (↓28.7%) | 4.06 |
| MBPP 3-shot (256) | Fréchet | 25.4 | 85.4 (1.29×) | 25,791 (↓26.3%) | 3.34 |
| GSM8K (512) | Fréchet | 75.6 | 59.4 (1.31×) | 91,239 (↓29.7%) | 3.90 |
| MATH (512) | Fréchet | 35.5 | 77.7 (1.38×) | 545,993 (↓28.6%) | 3.96 |
| HumanEval (512) | Fréchet | 41.5 | 75.5 (1.40×) | 18,909 (↓30.9%) | 4.05 |
| MBPP (512) | Fréchet | 14.2 | 82.7 (1.36×) | 42,893 (↓28.5%) | 3.49 |

Across 8 (dataset × length) settings: Fréchet achieves an average throughput of **1.36×** and NFE reduction of **29.2%** compared to threshold, with an average accuracy change of only −0.48 pt. Compared to the LLaDA-8B baseline without early exit, throughput is **4.31×** and NFE is reduced by **79.1%**.

### Ablation Study

| Config (GSM8K 8-shot, PrefixCache) | Len 256 Tok/s | Len 256 NFE | Len 512 Tok/s | Len 512 NFE |
|---|---|---|---|---|
| Threshold ($\tau = 0.9$) | 69.8 | 109,644 | 37.8 | 132,492 |
| Factor ($f = 0.75$) | 90.8 | 80,641 | 43.9 | 101,083 |
| **Fréchet ($\delta = 0.25$)** | **96.1** | **74,289** | **49.2** | **93,936** |
| Fréchet w/ DualCache | 80.9 | 78,901 | 50.4 | 102,145 |

### Key Findings
- **Throughput gain from heterogeneity reward**: On GSM8K frequency scans ($\delta \in [0, 0.30]$, matched $f = 1 - \delta$, $\tau \in [0.5, 0.9]$), Fréchet pushes the accuracy–throughput boundary "to the right," especially in conservative regions where $B_n$ is large.
- **Dominance under matched parameters**: As Fréchet necessarily accepts any setting accepted by the matched factor, it is never slower and provides "free" engineering gains under heterogeneous profiles.
- **Cache-agnostic**: Fréchet maintains the highest speed and lowest NFE across no-cache, PrefixCache, and DualCache, demonstrating orthogonality with caching mechanisms.

## Highlights & Insights
- **Theoretical and Engineering Isomorphism**: Fréchet decoding is not an empirical trick but the tightest marginal-only generalization of the factor rule. The relationship where "tighter theory leads to more commits" is rare and valuable.
- **Profile-aware = Adaptive Factor**: Reinterpreting Fréchet as $f_{\text{eff}}(n) = 1 - \delta + B_n$ provides an elegant perspective of "data-adaptive" parameters. This idea of using data to calibrate thresholds can be transferred to other batching scenarios like speculative decoding or early-exit confidence margins.
- **Drop-in Friendly**: Modifying only 8 lines of selection logic without touching the model or schedule makes it a low-cost acceleration trick for H100 deployments.

## Limitations & Future Work
- **Marginal-only Ceiling**: Fréchet does not utilize joint dependency information between tokens. Section 4.2 suggests that modeling $d_{TV}(P_S, Q_S)$ could theoreticaly allow more commits but requires additional dependency modeling not implemented here.
- **Dependence on Confidence Calibration**: If the model is over-confident, the profile certificate may fail. While a calibration-robust version is provided in Appendix C, it lacks extensive experimental verification.
- **Margin Sensitivity**: A single $\delta = 0.25$ is stable across benchmarks, but the trade-offs in aggressive parallelism may still require per-task tuning for fragility-sensitive tasks.
- **Scale Verification**: Evaluation is primarily on LLaDA-8B and Dream-7B; performance on 70B+ scale dLLMs or very long sequences (>1024) is not yet explored.

## Related Work & Insights
- **vs. Fast-dLLM (Wu et al., 2026)**: Simplifies marginal-only information; this work generalizes it to heterogeneous profiles, proving factor is a special case and providing a strictly larger acceptance set.
- **vs. Speculative / Blockwise Parallel Decoding**: Autoregressive methods use drafter+verifier; Fast-dLLM++ uses marginal certificates for commit decisions, which is more lightweight and specific to diffusion "synchronous multi-position commit" scenarios.
- **vs. Copula / dependence-aware methods**: This work deliberately avoids joint dependency modeling to maintain marginal-only safety while leaving interfaces for future dependence-aware extensions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Uses classic Fréchet–Hoeffding inequalities to generalize the factor rule; theoretically sound and strictly tighter, though the core probability bound is well-established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive scanning across four benchmarks, three cache modes, and multiple models; however, lacks 70B+ scale testing and comparison with non-Fast-dLLM系 methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figure 1 clearly illustrates the "flattening" vs. "heterogeneity reward" concept. The theoretical narrative is logically rigorous.
- **Value**: ⭐⭐⭐⭐ Training-free, drop-in, and providing ~1.36× throughput gain; highly practical for engineering teams using diffusion LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](../../ICLR2026/llm_nlp/stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)
- [\[ICML 2026\] Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)
- [\[ICML 2026\] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)

</div>

<!-- RELATED:END -->
