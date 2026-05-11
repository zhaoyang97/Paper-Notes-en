---
title: >-
  [Paper Note] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection
description: >-
  [NeurIPS 2025][LLM Safety][text watermark detection] This paper systematically evaluates eight classical goodness-of-fit (GoF) tests for LLM text watermark detection…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "text watermark detection"
  - "goodness-of-fit tests"
  - "LLM watermarking"
  - "statistical hypothesis testing"
  - "robustness"
date: 2026-05-08
content_hash: a508deb5244eff91
---

# On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection

**Conference**: NeurIPS 2025
**arXiv**: [2510.03944](https://arxiv.org/abs/2510.03944)
**Code**: [GitHub](https://github.com/hwq0726/GoF-for-Watermark-Detection)
**Area**: AI Safety / LLM Watermarking
**Keywords**: text watermark detection, goodness-of-fit tests, LLM watermarking, statistical hypothesis testing, robustness

## TL;DR

This paper systematically evaluates eight classical goodness-of-fit (GoF) tests for LLM text watermark detection, demonstrating that GoF tests significantly outperform existing baseline methods in both detection power and robustness.

## Background & Motivation

LLM-generated text raises concerns about content authenticity and copyright. Text watermarking embeds detectable statistical signals into generated text to verify content provenance. At its core, watermark detection is a hypothesis testing problem: under $H_0$, the pivot statistic $Y_t = Y(w_t, \zeta_t)$ is i.i.d. from a known distribution $\mu_0$; under $H_1$, it deviates from this distribution.

This is naturally a **goodness-of-fit (GoF) testing** problem: determining whether i.i.d. samples arise from a given distribution. However, existing literature focuses primarily on designing new watermarking schemes rather than improving detection efficacy. Li et al. proposed a truncated $\phi$-divergence GoF test, but only analyzed the Gumbel-max watermark and relied on asymptotic assumptions.

Core Problem: How do classical GoF tests perform in modern watermark detection?

## Method

### Overall Architecture

Watermark detection is unified under a GoF testing framework. The detection pipeline proceeds as follows: (1) compute pivot statistics $Y_1, \ldots, Y_n$ from the token sequence; (2) compute p-values $p_t = 1 - F_0(Y_t)$; (3) evaluate the degree of deviation using a GoF test statistic; (4) determine whether to reject $H_0$ based on a critical value.

### Key Designs

1. **Unified Evaluation of Eight GoF Tests**: The tests include Kolmogorov-Smirnov (Kol), Anderson-Darling (And), Cramér-von Mises (Cra), Kuiper (Kui), Watson (Wat), Neyman smooth test (Ney), Chi-squared (Chi), and truncated divergence test (Phi). Each test measures the deviation between the empirical CDF and the null hypothesis CDF in a distinct manner.

2. **Adaptation to Three Watermarking Schemes**: Gumbel-max ($\mu_0 = U(0,1)$), inverse-transform watermarking ($\mu_0(Y \leq r) = r^2$), and Google SynthID ($\mu_0$ follows the Irwin-Hall distribution). Green-red list watermarking is excluded, as its binary pivot statistic reduces the GoF test to the original detection rule.

3. **Low-Temperature Advantage Analysis**: At low temperatures, the watermark signal weakens but text repetition increases, introducing structured patterns that cause the empirical CDF to deviate from the null CDF. GoF tests can uniquely exploit this effect—an advantage that existing methods have not leveraged.

### Loss & Training

This paper is an empirical evaluation study and requires no model training. Key technical details:
- Type I error is controlled at $\alpha = 0.01$; critical values are calibrated via theoretical distributions or Monte Carlo simulation.
- Since most GoF test null distributions lack closed-form solutions, large-sample asymptotic approximations are used.
- All GoF tests are permutation-invariant with respect to the pivot statistics, so detection results are unaffected by token order.

## Key Experimental Results

### Main Results

| Watermark Scheme | Temperature | Length | Baseline | Chi | And | Kol | Phi |
|------------------|-------------|--------|----------|-----|-----|-----|-----|
| Gumbel-max | T=0.3 | n=400 | 15.1% | **2.9%** | 4.9% | 4.7% | 5.7% |
| Gumbel-max | T=0.7 | n=200 | 0.6% | **0.3%** | 0.5% | 0.6% | 0.3% |
| Inverse-tran | T=0.3 | n=400 | 27.1% | — | 9.3% | 7.4% | 12.1% |

*(Values are Type II error rates ×100; lower is better.)*

### Ablation Study

| Configuration | Metric | Description |
|---------------|--------|-------------|
| Temperature T∈{0.1, 0.3, 0.7, 1.0} | Type II error rate | GoF outperforms baseline across all temperatures |
| Varying text length n | Detection power | Advantage is more pronounced for longer texts |
| Deletion edits r=0.1, 0.2 | Robustness | GoF maintains high detection power |
| Synonym substitution r=0.1, 0.2 | Robustness | GoF performs stably |
| Informed edits r=0.3, 0.5 | Robustness | GoF retains advantage under strong attacks |

### Key Findings

- GoF tests outperform baseline detection methods in nearly all configurations.
- The Chi-squared test performs best in multiple settings, though And and Kol are also highly competitive.
- The advantage of GoF tests is most pronounced at low temperature (T=0.1), as they can exploit text repetition patterns.
- Type I error across all GoF tests remains close to the target level of 0.01, indicating well-controlled false positive rates.
- Results are consistent across three LLMs (OPT-1.3B, OPT-13B, Llama 3.1-8B).

## Highlights & Insights

- The finding that GoF tests are "simple yet powerful and underappreciated tools" carries strong practical implications for the community.
- The observation that low-temperature text repetition grants GoF tests a unique advantage is insightful, explaining why strong detection power is maintained across varying temperatures.
- The unified framework enables standardization of detection methods across different watermarking schemes.
- The analysis excluding green-red list watermarking is rigorous, clearly demonstrating the degenerate behavior of GoF tests in that setting.

## Limitations & Future Work

- Experiments are conducted primarily on open-source LLMs; closed-source commercial models (e.g., GPT-4) are not evaluated.
- Only two task types are considered: text completion and long-form QA.
- The optimal choice of GoF test depends on the specific watermarking scheme and scenario; no automated selection guideline is provided.
- Computational efficiency and latency of GoF tests are not analyzed.
- Finite-sample theoretical analysis beyond asymptotic guarantees is absent.

## Related Work & Insights

- The truncated divergence GoF test of Li et al. is the direct predecessor of this work; the present paper substantially extends the evaluation scope.
- The connection between watermark detection and statistical hypothesis testing provides a bridge between the two fields.
- For future watermark design, the characteristics of GoF tests should be considered to optimize the detectability of watermarking schemes.
- The informed edit scenario (attacker with knowledge of the secret key) provides a valuable reference for security evaluation.

## Rating

- **Novelty**: ⭐⭐⭐ — The primary contribution lies in systematic evaluation rather than methodological innovation, though the findings are valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 8 GoF tests × 3 watermarking schemes × 3 LLMs × 4 temperatures × multiple edit types; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized with thorough explanation of the statistical background.
- **Value**: ⭐⭐⭐⭐ — Provides a plug-and-play detection toolkit for watermark detection with strong practical utility.

## Supplementary Details

- Green-red list watermarking is excluded because the pivot statistic is binary (whether a token is green), reducing GoF to a counting test.
- The Neyman smooth test uses $k=3$ order Legendre orthogonal polynomials.
- Informed edits simulate a worst-case scenario in which an attacker with knowledge of the secret key selectively modifies high-signal tokens.
- All GoF tests are permutation-invariant with respect to token order, which is a core advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[NeurIPS 2025\] ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests](orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[ICLR 2026\] Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models](../../ICLR2026/llm_safety/improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)

</div>

<!-- RELATED:END -->
