---
title: >-
  [Paper Note] Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference
description: >-
  [ICML 2026 (Workshop on Structured Probabilistic Inference & Generative Modeling)][LLM Efficiency][Diffusion LLM] Addressing the parallel decoding bottleneck of diffusion language models (dLLMs), this paper proposes training-free Fréchet Profile Decoding. It uses the entire sorted confidence profile—rather than just the "weakest selected token"—to determine the number of tokens to commit per step. This strictly generalizes the factor rule of Fast-dLLM to heterogeneous confide…
tags:
  - "ICML 2026 (Workshop on Structured Probabilistic Inference & Generative Modeling)"
  - "LLM Efficiency"
  - "Diffusion LLM"
  - "Parallel Decoding"
  - "Fréchet Lower Bound"
  - "Confidence Profile"
  - "Heterogeneity Reward"
date: 2026-05-08
content_hash: 49540efd7a4cf834
---

# Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference

**Conference**: ICML 2026 (Workshop on Structured Probabilistic Inference & Generative Modeling)  
**arXiv**: [2606.02955](https://arxiv.org/abs/2606.02955)  
**Code**: https://github.com/Ringo-Star/FastdLLM_plusplus (Available)  
**Area**: LLM Efficiency / Diffusion Language Models / Parallel Decoding  
**Keywords**: Diffusion LLM, Parallel Decoding, Fréchet Lower Bound, Confidence Profile, Heterogeneity Reward

## TL;DR
Addressing the parallel decoding bottleneck of diffusion language models (dLLMs), this paper proposes training-free Fréchet Profile Decoding. It uses the entire sorted confidence profile—rather than just the "weakest selected token"—to determine the number of tokens to commit per step. This strictly generalizes the factor rule of Fast-dLLM to heterogeneous confidence scenarios, achieving a 1.36× average throughput increase and 29% NFE reduction on LLaDA-8B across four benchmarks with almost no loss in accuracy.

## Background & Motivation
**Background**: Masked diffusion LLMs (MDLM) start from a fully masked sequence and predict marginal distributions for all masked positions in parallel at each step. In theory, committing multiple tokens at once can achieve throughput far exceeding autoregressive models. However, the gap between the "product of marginals" and the true joint distribution leads to the "curse of parallelism"—while individual tokens appear correct, their combination might be incoherent.

**Limitations of Prior Work**: Fast-dLLM mitigates this with confidence-aware parallel decoding using two rules: a threshold rule ($c_i \ge \tau$) for independent commitment, and a factor rule ($(n+1)(1-c_{(n)}) < f$) for accepting the top-$n$ candidates. The factor rule relies on a "high-confidence hypothesis"—it assumes all $n$ selected tokens have a confidence equal to $c_{(n)}$ (the weakest one), essentially "flattening" the entire confidence profile to its minimum.

**Key Challenge**: In practice, the confidence profile at a decoding step is highly heterogeneous—e.g., $(0.99, 0.95, 0.82, 0.78, 0.74)$. The first few tokens are almost certain, while subsequent ones weaken. The factor rule replaces the profile with a flat proxy $(0.82, 0.82, 0.82)$, discarding the safety information provided by "strong tokens" and conservatively rejecting tokens that could have been committed.

**Goal**: To generalize the factor rule from "homogeneous confidence" to "heterogeneous confidence" via an optimal marginal-only proof, translating this generalization into larger parallel commit sets without modifying the model, diffusion process, or KV cache.

**Key Insight**: Starting from classical Fréchet–Hoeffding / Bonferroni inequalities—given only marginal probabilities of events, the distribution-free tight lower bound for the probability of their intersection is $L_n = \max\{0, \sum_{j=1}^n c_{(j)} - (n-1)\}$. Applying this bound to the event "all selected marginal-argmax tokens are correct simultaneously" provides a safety certificate that utilizes the entire profile.

**Core Idea**: Subtract the competitor upper bound $U_n = 1 - c_{(n)}$ (the probability upper bound for any other tuple) from the Fréchet lower bound $L_n$ (joint correctness lower bound) to obtain a "safety margin" $G_n = L_n - U_n$. Committing the maximum prefix $n^*$ where $G_n > \delta$ allows for strictly more tokens than the factor rule under heterogeneous profiles.

## Method

### Overall Architecture
Fast-dLLM++ replaces the "how many tokens to commit" logic in Fast-dLLM with a tighter metric, while leaving the model, diffusion schedule, and KV cache unchanged. After the forward pass in each denoising step, marginal predictions $p_\theta(X_i = v \mid x_k)$ for all masked positions are obtained. The argmax and confidence $c_i$ for each position are extracted and sorted descending as $c_{(1)} \ge c_{(2)} \ge \cdots \ge c_{(m)}$. A "profile-aware" selector then decides the number of tokens to commit. This logic replaces rows 11–18 of Fast-dLLM Algorithm 1, with overhead limited to one sorting operation and one prefix sum, making it transparent to NONE / PrefixCache / DualCache modes.

### Key Designs

**1. Fréchet Profile Certificate: Using the individual values of the entire confidence profile**

Fast-dLLM's factor rule only considers the weakest selected token $c_{(n)}$, effectively flattening the profile and losing safety information. Ours (Theorem 4.1) adopts the lower bound from Fréchet–Hoeffding / Bonferroni inequalities: $L_n = \max\{0, \sum_{j=1}^n c_{(j)} - (n-1)\}$. This is the tightest possible lower bound for joint correctness given only marginals. In contrast, any competitor tuple that is "wrong in at least one position" has a probability upper bound $U_n = 1 - c_{(n)}$. If $L_n > U_n$, the selected tuple is the unique maximizer of the joint distribution $P_S$. This upgrades the "safety check" from looking at the weakest link to evaluating the whole chain, accounting for the surplus safety of strong tokens.

**2. Profile-Aware Selection Rule and Algorithm 1: Scanning for the "Largest Safe" prefix**

With the certificate, token selection becomes a 1D scan: for $n$ from 1 to $m$, calculate $G_n = L_n - U_n$ and select $n^* = \max\{n: G_n > \delta\}$, where $\delta \ge 0$ is a user-defined margin. If no $n$ satisfies the condition, it defaults to $n^* = 1$. This process requires no additional network passes and is orthogonal to underlying cache mechanisms, directly translating theoretical tightness into increased tokens-per-step.

**3. Heterogeneity Reward Decomposition: Explaining "Why It Is Faster"**

To explain when Fréchet is more aggressive than a matched factor rule (where $f = 1 - \delta$), the paper (Proposition 4.3) decomposes the margin as $G_n = F_n + B_n$ when $L_n > 0$. Here, the factor kernel $F_n = (n+1)c_{(n)} - n$ depends only on the weakest confidence. The heterogeneity reward $B_n = \sum_{j=1}^{n-1}(c_{(j)} - c_{(n)}) \ge 0$ represents the area between the profile and the "flat minimum line." Fréchet acts as a data-adaptive factor $f_{\text{eff}}(n) = 1 - \delta + B_n$. Corollary 4.4 proves that a Fréchet decoder will always accept at least as many tokens as a matched factor rule, accepting more whenever the heterogeneity reward is large enough to cross the decision boundary.

### Loss & Training
Completely training-free. $\delta$ is the only new hyperparameter (default $\delta = 0.25$, corresponding to matched factor $f = 0.75$). The paper also introduces a calibration-robust variant (Appendix C) using conservative lower bounds for confidence to handle model over-confidence.

## Key Experimental Results

### Main Results
LLaDA-8B-Instruct, PrefixCache, block size 32, single H100 GPU; threshold $\tau = 0.9$, factor $f = 0.75$, Fréchet $\delta = 0.25$.

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

Across 8 settings: Fréchet achieved **1.36×** average throughput and **29.2%** NFE reduction relative to the threshold rule, with a tiny mean accuracy change of −0.48 pt. Compared to the LLaDA-8B baseline, average throughput is **4.31×** and NFE is reduced by **79.1%**.

### Ablation Study

| Config (GSM8K 8-shot, PrefixCache) | Len 256 Tok/s | Len 256 NFE | Len 512 Tok/s | Len 512 NFE |
|---|---|---|---|---|
| Threshold ($\tau = 0.9$) | 69.8 | 109,644 | 37.8 | 132,492 |
| Factor ($f = 0.75$) | 90.8 | 80,641 | 43.9 | 101,083 |
| **Fréchet ($\delta = 0.25$)** | **96.1** | **74,289** | **49.2** | **93,936** |
| Fréchet w/ DualCache | 80.9 | 78,901 | 50.4 | 102,145 |

### Key Findings
- **Throughput gain from heterogeneity reward**: On GSM8K frequency scans, Fréchet shifts the accuracy–throughput Pareto frontier to the right, especially in conservative regions (small $\delta$ / large $f$) where $B_n$ is significant.
- **Dominance under matched parameters**: Fréchet strictly accepts any prefix accepted by a matched factor rule, ensuring no speed loss. It commits more tokens only when profiles are heterogeneous.
- **Cache-agnostic performance**: Gains are consistent across no-cache, PrefixCache, and DualCache, proving the method is orthogonal to memory optimization.

## Highlights & Insights
- **Isomorphism between Theory and Engineering**: Fréchet decoding is not just an empirical trick; it is a tight generalization of the factor rule under marginal-only information.
- **Profile-aware = Adaptive Factor**: Viewing Fréchet as $1 - \delta + B_n$ is powerful. While threshold is a constant gate and factor is set-size aware, Fréchet is data-adaptive, using heterogeneity to calibrate the aggressiveness.
- **Drop-in Friendly**: By modifying only 8 lines of logic in Algorithm 1, it provides a "zero-migration cost" acceleration for diffusion LLMs.

## Limitations & Future Work
- **The Marginal-only Ceiling**: Fréchet does not utilize joint dependencies between tokens. Theoretical results in §4.2 suggest that estimating $d_{TV}$ or $D_{KL}$ could allow for even more commits, but this requires extra dependency modeling.
- **Calibration Dependency**: If the model is over-confident, the certificates fail. Conservative lower bounds are a potential but less-tested remedy.
- **Task-specific Sensitivity**: While $\delta = 0.25$ is robust, some performance drops in fragile tasks (like MBPP) suggest that the aggressive parallelism trade-off may still require per-task tuning.
- **Scope of Validation**: Testing was primarily on LLaDA-8B and Dream-7B; performance on 70B+ models or sequences longer than 1024 tokens remains to be verified.

## Related Work & Insights
- **vs Fast-dLLM (Wu et al., 2026)**: Threshold/factor are homogeneous simplifications. This work generalizes them, providing a strictly larger acceptance set.
- **vs Speculative Decoding**: Autoregressive models use draft-then-verify. Fast-dLLM++ uses marginal certificates for internal commitment decisions within a single model, which is lighter but specific to diffusion decoding.
- **vs Copula/Dependence-aware Methods**: This work intentionally avoids joint modeling to remain marginal-only but provides an interface for future dependence-aware expansions.

## Rating
- Novelty: ⭐⭐⭐⭐ Uses classical inequalities to tighten inference rules; the innovation lies in the specific application to decoding profiles.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive across benchmarks, cache modes, and lengths, though lacks massive scale (70B+).
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent visualizations (Figure 1) and clear theoretical derivation.
- Value: ⭐⭐⭐⭐ A training-free, drop-in acceleration with significant gains for dLLM practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](../../ICLR2026/llm_efficiency/fast-dllm_training-free_acceleration_of_diffusion_llm_by_enabling_kv_cache_and_p.md)
- [\[ICLR 2026\] Fast-dLLM v2: Efficient Block-Diffusion LLM](../../ICLR2026/llm_efficiency/fast-dllm_v2_efficient_block-diffusion_llm.md)
- [\[ICLR 2026\] Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM](../../ICLR2026/llm_efficiency/dynamic-dllm_dynamic_cache-budget_and_adaptive_parallel_decoding_for_training-fr.md)
- [\[ICLR 2026\] Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](../../ICLR2026/llm_efficiency/diffusion_llms_can_do_faster-than-ar_inference_via_discrete_diffusion_forcing.md)
- [\[CVPR 2026\] Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference](../../CVPR2026/llm_efficiency/rejection_mixing_fast_semantic_propagation_of_mask_tokens_for_efficient_dllm_inf.md)

</div>

<!-- RELATED:END -->
