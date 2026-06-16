---
title: >-
  [Paper Note] Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference
description: >-
  [ICML 2026][LLM Efficiency][Paper Note] Addressing the parallel decoding bottleneck of diffusion language models (dLLMs), this paper proposes training-free Fréchet Profile Decoding: using the entire sorted confidence profile rather than just the "weakest selected token" to determine how many tokens to commit in each parallel step. This strictly generalizes t
tags:
  - ICML 2026
  - LLM Efficiency
date: 2026-05-08
content_hash: 7b10035b2f633e2f
---
# Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference

**Conference**: ICML 2026 (Workshop on Structured Probabilistic Inference & Generative Modeling)  
**arXiv**: [2606.02955](https://arxiv.org/abs/2606.02955)  
**Code**: https://github.com/Ringo-Star/FastdLLM_plusplus (Available)  
**Area**: LLM Efficiency / Diffusion Language Models / Parallel Decoding  
**Keywords**: Diffusion LLM, Parallel Decoding, Fréchet Lower Bound, Confidence Profile, Heterogeneity Reward

## TL;DR
Addressing the parallel decoding bottleneck of diffusion language models (dLLMs), this paper proposes training-free Fréchet Profile Decoding: using the entire sorted confidence profile rather than just the "weakest selected token" to determine how many tokens to commit in each parallel step. This strictly generalizes the factor rule of Fast-dLLM to heterogeneous confidence scenarios, achieving an average throughput of 1.36× and a 29% reduction in NFE on LLaDA-8B across four benchmarks with almost no loss in precision.

## Background & Motivation
**Background**: Masked diffusion LLMs (MDLMs) start from a fully masked sequence and predict the marginal distributions of all masked positions in parallel at each step. Theoretically, they can commit multiple tokens at once to achieve throughput far exceeding auto-regression. however, a gap exists between the "product of marginal distributions" and the true joint distribution; the more tokens are parallelized, the more severe the "curse of parallelism" becomes—single tokens may appear correct, but their combination might be incoherent.

**Limitations of Prior Work**: Fast-dLLM mitigates this issue using confidence-aware parallel decoding with two rules: the threshold rule $c_i \ge \tau$ for independent commits, and the factor rule $(n+1)(1-c_{(n)}) < f$ to decide on accepting the top $n$ candidates. The theoretical basis of factor is a "high-confidence assumption," which assumes the confidence of all $n$ selected tokens equals $c_{(n)}$ (the weakest one), essentially "flattening" the entire confidence profile to its minimum value.

**Key Challenge**: In practice, the confidence profile of a decoding step is highly heterogeneous—for example, $(0.99, 0.95, 0.82, 0.78, 0.74)$. The first few tokens are nearly certain, while the later ones weaken gradually. The factor rule replaces the entire profile with a flat proxy of $(0.82, 0.82, 0.82)$, discarding the safety information provided by "strong tokens," which leads to conservative rejection of tokens that could have been committed together.

**Goal**: Without modifying the model, diffusion process, or KV cache, generalize the factor rule from "homogeneous confidence" to an optimal marginal-only proof for "heterogeneous confidence," and translate this generalization into a larger set of parallel commits.

**Key Insight**: Starting from the classic Fréchet–Hoeffding / Bonferroni inequalities—when only marginal probabilities of events are known, the distribution-free tight lower bound for the probability of their intersection is exactly $L_n = \max\{0, \sum_{j=1}^n c_{(j)} - (n-1)\}$. Applying this lower bound to the event that "all selected marginal-argmax tokens are simultaneously correct" yields a safety certificate that utilizes the entire profile.

**Core Idea**: Subtract the competitor's upper bound $U_n = 1 - c_{(n)}$ (the probability upper bound of any other tuple) from the Fréchet lower bound $L_n$ (the probability lower bound of joint correctness) to obtain a "safety margin" $G_n = L_n - U_n$. Committing the maximum prefix $n^*$ satisfying $G_n > \delta$ allows for strictly more tokens than those accepted by the factor rule under heterogeneous profiles.

## Method

### Overall Architecture
Fast-dLLM++ replaces the "how many tokens to commit per step" logic in Fast-dLLM with a tighter metric, without altering the model, diffusion scheduler, or KV cache. In each denoising step, after the forward pass and obtaining marginal predictions $p_\theta(X_i = v \mid x_k)$ for all masked positions, it takes the argmax and confidence $c_i$ for each position, sorts $c_i$ in descending order as $c_{(1)} \ge c_{(2)} \ge \cdots \ge c_{(m)}$, and then uses a "profile-aware" selector to determine the prefix of tokens to commit. The remaining positions keep their masks for the next step. Implementation-wise, it only replaces the token selection logic in lines 11–18 of Fast-dLLM Algorithm 1. The extra overhead consists solely of one sorting operation and one prefix sum, making it fully transparent to NONE / PrefixCache / DualCache modes.

### Key Designs

**1. Fréchet Profile Certificate: Using the entire confidence profile instead of the weakest entry**

The factor rule of Fast-dLLM only considers the weakest selected token $c_{(n)}$, which is equivalent to flattening the profile to its minimum and discarding safety information from stronger tokens. This paper (Theorem 4.1) instead uses the lower bound given by the Fréchet–Hoeffding / Bonferroni inequality: when only marginal probabilities are known, the distribution-free tight lower bound for their joint probability is $L_n = \max\{0, \sum_{j=1}^n c_{(j)} - (n-1)\}$. This is exactly the probability lower bound for "the $n$ selected marginal-argmax tokens being correct simultaneously." In contrast, the probability upper bound for any "at least one error" competitor tuple is $U_n = 1 - c_{(n)}$, as it must hit the position with confidence $c_{(n)}$. If $L_n > U_n$, the selected tuple is the unique maximizer under the true joint distribution $P_S$ and can be safely committed. Since $L_n$ is the tightest unimprovable lower bound under marginal-only information, this upgrades the "safety check" from "checking the weakest token" to "checking the entire profile," theoretically including factor as a homogeneous special case while accounting for the safety margin of stronger tokens.

**2. Profile-aware Selection Rules & Algorithm 1: Scanning prefixes to find the "largest safe" commit count**

With the certificate, token selection becomes a one-dimensional scan: iterate the candidate count $n$ from 1 to $m$, calculate the safety margin $G_n = L_n - U_n$, and select the maximum prefix $n^* = \max\{n: G_n > \delta\}$, where $\delta \ge 0$ is a user-defined margin. If no $n$ satisfies the condition, it defaults to $n^* = 1$ to ensure at least one token progresses. The process only requires sorting the calculated confidence vectors, performing prefix sums, and calculating $L_n, U_n, G_n$ item by item. It requires no additional forward passes; thus, replacing it in Fast-dLLM Algorithm 1 adds no extra persistent memory and negligible computation, while remaining orthogonal to the underlying cache. This step directly converts the theoretical tightness of Theorem 4.1 into a higher tokens-per-step parallelism.

**3. Heterogeneity Reward Decomposition: Decomposing "why it is faster" into a profile-interpretable metric**

To explain when Fréchet is more aggressive than a matched factor (where $f = 1 - \delta$), the paper (Proposition 4.3) decomposes the margin into $G_n = F_n + B_n$ when $L_n > 0$. Here, the factor kernel $F_n = (n+1)c_{(n)} - n$ depends only on the weakest confidence and corresponds exactly to the factor rule. The heterogeneity reward $B_n = \sum_{j=1}^{n-1}(c_{(j)} - c_{(n)}) \ge 0$ represents the area between the actual profile and the "flat weakest line." The more heterogeneous the profile, the larger this reward. Equivalently, Fréchet acts as a data-adaptive factor $f_{\text{eff}}(n) = 1 - \delta + B_n$, becoming more aggressive as heterogeneity increases. Corollary 4.4 strictly proves that any prefix accepted by matched factor is necessarily accepted by Fréchet (ensuring it is never slower), while Fréchet accepts strictly more tokens if and only if $F_n \le \delta < F_n + B_n$. This decomposition turns the empirical observation of effectiveness into quantifiable profile evidence and reinterprets the engineering success of Fast-dLLM as a homogeneous special case of the marginal-only framework, providing an interface for future dependence-aware extensions (§4.2 using TV / KL stability).

### Loss & Training
Completely training-free. $\delta$ is the only new hyperparameter, defaulting to $\delta = 0.25$ (corresponding to matched factor $f = 0.75$). The paper also provides a calibration-robust variant (Appendix C) using conservative lower bounds of confidence to handle model over-confidence.

## Key Experimental Results

### Main Results
LLaDA-8B-Instruct, PrefixCache, block size 32, single H100 GPU; threshold $\tau = 0.9$ (Fast-dLLM's primary rule), factor $f = 0.75$, Fréchet $\delta = 0.25$.

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

Across 8 (dataset × length) settings: Fréchet achieved an average throughput of **1.36×** and an NFE reduction of **29.2%** relative to threshold, with an average precision change of only −0.48 pt. Compared to the LLaDA-8B baseline without early exit, average throughput was **4.31×** and NFE decreased by **79.1%**.

### Ablation Study

| Config (GSM8K 8-shot, PrefixCache) | Len 256 Tok/s | Len 256 NFE | Len 512 Tok/s | Len 512 NFE |
|---|---|---|---|---|
| Threshold ($\tau = 0.9$) | 69.8 | 109,644 | 37.8 | 132,492 |
| Factor ($f = 0.75$) | 90.8 | 80,641 | 43.9 | 101,083 |
| **Fréchet ($\delta = 0.25$)** | **96.1** | **74,289** | **49.2** | **93,936** |
| Fréchet w/ DualCache | 80.9 | 78,901 | 50.4 | 102,145 |

### Key Findings
- **Throughput gain stems from heterogeneity reward**: In GSM8K frequency scans ($\delta \in [0, 0.30]$, matched $f = 1 - \delta$, $\tau \in [0.5, 0.9]$), Fréchet pushes the entire accuracy–throughput boundary to the right, with the most stable gains in conservative regions (small $\delta$ / large $f$)—where $B_n$ is large enough to cross decision boundaries.
- **Dominance under matched parameters**: Fréchet necessarily accepts any setting accepted by matched factor, meaning it is never slower. It commits more tokens than factor only under heterogeneous profiles (where leading tokens are significantly higher than trailing ones), which is an engineering "free lunch."
- **Cache-agnostic**: Fréchet maintains the highest speed and lowest NFE across no-cache, PrefixCache, and DualCache modes, demonstrating that modifying the token selection layer is orthogonal to caching. On MBPP (256), factor accuracy dropped by 6 pt while Fréchet only dropped by 2 pt, suggesting profile-aware selection is more robust for fragile tasks.

## Highlights & Insights
- **Isomorphic Improvement in Theory and Engineering**: Fréchet decoding is not an empirical trick but the tightest generalization of the Fast-dLLM factor rule under marginal-only information (derived from the distribution-free Fréchet–Hoeffding bound). This "theoretical tightening → engineering more commits" isomorphism is rare.
- **Profile-aware = Adaptive Factor**: Reinterpreting Fréchet as $f_{\text{eff}}(n) = 1 - \delta + B_n$ is the paper's most elegant perspective—threshold is a constant gate, factor is set-size aware, and Fréchet is a "data-adaptive" factor that becomes more aggressive with heterogeneity. This idea of calibrating hyperparameters using data itself can be transferred to acceptance thresholds in speculative decoding, early-exit confidence margins, and any scenario using marginal certificates for batch decisions.
- **Drop-in Friendly**: Modifying only 8 lines of choice logic in Algorithm 1 without touching the model, scheduler, or cache makes this a rare "zero migration cost" acceleration trick reproducible on a single H100.

## Limitations & Future Work
- **Marginal-only is intentional but acts as a ceiling**: The paper admits Fréchet does not utilize joint dependency information between tokens. The TV / KL stability proofs in §4.2 (Lemma 4.6 / Corollary 4.7) imply that estimating $d_{TV}(P_S, Q_S)$ or total correlation $D_{KL}(P_S \| Q_S)$ could theoretically allow for more commits, but this requires additional dependency modeling not implemented in Fast-dLLM++.
- **Reliance on Confidence Calibration**: If the model is over-confident, $c_{(n)}$ is no longer reliable, and profile certificates may fail. Appendix C offers a calibration-robust version, but it has not been extensively verified.
- **Task-specific Margin Sensitivity**: A single global $\delta = 0.25$ is stable across four benchmarks, but on MBPP 512, both factor and Fréchet showed accuracy drops compared to threshold (14.2 for threshold vs 12.0 for factor), indicating that the trade-off for "aggressive parallelism" still requires per-task tuning in distribution-shifted or short-sequence tasks.
- **Validation Limited to LLaDA-8B / Dream-7B**: Systemic evaluation on larger scales (e.g., 70B class dLLMs) or longer generations (>1024) is missing; whether throughput gains scale with model size remains unclear.

## Related Work & Insights
- **vs Fast-dLLM (Wu et al., 2026)**: Threshold and factor are homogeneous simplifications of marginal-only information. This paper generalizes the framework to heterogeneous profiles, proving factor is a special case when $f = 1 - \delta$ and providing a strictly larger acceptance set as a drop-in replacement.
- **vs Speculative / Blockwise Parallel Decoding (Stern 2018; Leviathan 2023; Chen 2023)**: The auto-regressive camp uses drafter + verifier dual-model schemes for parallelism. Fast-dLLM++ uses marginal certificates within a diffusion LLM for commit decisions without a second model—lighter but only applicable to the "synchronous multi-position commit" scenario of diffusion decoding.
- **vs Copula / dependence-aware methods (Kasa 2020/2021/2022)**: This paper deliberately avoids joint dependency modeling to stay within marginal-only safety, yet provides a dependency-aware extension interface in §4.2 that could eventually be combined with total correlation estimation to capture more safety certificates beyond Fréchet.

## Rating
- Novelty: ⭐⭐⭐⭐ Uses the classic Fréchet–Hoeffding inequality to generalize the Fast-dLLM factor to heterogeneous profiles; theoretically sound and strictly tighter, though the core probabilistic bound is a classic concept applied to a new context.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive across four benchmarks, three cache modes, two generation lengths, and multiple models (LLaDA-8B / Dream-7B / LLaDA-V); however, lacks larger dLLMs, longer sequences (>1024), and comparisons with non-Fast-dLLM systems (like speculative diffusion).
- Writing Quality: ⭐⭐⭐⭐⭐ The five-column chart in Figure 1 clearly explains how factor flattens profiles, what heterogeneity reward represents, and why Fréchet accepts $n=4$ while matched factor only accepts $n=2$. The narrative flow from Theorem 4.1 to Corollary 4.4 is seamless.
- Value: ⭐⭐⭐⭐ Training-free, drop-in, and providing an average 1.36× throughput / 29% NFE saving; this is a directly applicable acceleration for any team using Fast-dLLM for diffusion LLM inference, while leaving a clear interface for future dependence-aware parallel decoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ACL 2025\] Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](../../ACL2025/llm_efficiency/smarter_better_faster_longer_a_modern_bidirectional_encoder_for_fast_memory_effi.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ICML 2026\] Ekka: Automated Diagnosis of Silent Errors in LLM Inference](ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)

</div>

<!-- RELATED:END -->
