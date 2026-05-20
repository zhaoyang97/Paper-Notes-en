---
title: >-
  [Paper Note] Log Probability Tracking of LLM APIs
description: >-
  [ICLR 2026][Video Understanding][LLM API monitoring] This paper proposes Logprob Tracking (LT), a method that detects subtle changes in LLM APIs (e.g.…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "LLM API monitoring"
  - "log probability"
  - "model change detection"
  - "hypothesis testing"
  - "non-determinism"
date: 2026-05-08
content_hash: c0c9bdffa5a2faf4
---

# Log Probability Tracking of LLM APIs

**Conference**: ICLR 2026
**arXiv**: [2512.03816](https://arxiv.org/abs/2512.03816)  
**Code**: [Available](https://github.com/timothee-chauvin/track-llm-apis)  
**Area**: Video Understanding
**Keywords**: LLM API monitoring, log probability, model change detection, hypothesis testing, non-determinism

## TL;DR

This paper proposes Logprob Tracking (LT), a method that detects subtle changes in LLM APIs (e.g., single-step fine-tuning) using only the log probabilities of a single-token input and single-token output. LT achieves sensitivity 2–3 orders of magnitude higher than existing methods at 1000× lower cost.

## Background & Motivation

LLM API providers typically offer versioned endpoints, implying model consistency. Users — including developers, researchers, and regulators — rely on this consistency to ensure application reliability and research reproducibility. However, users lack practical means to verify such consistency.

In practice, providers may modify models for various reasons:
- **Performance optimization**: Updates to inference software/hardware infrastructure
- **Safety responses**: Mitigating new jailbreak attacks or modifying model behavior
- **Cost reduction**: Silently deploying quantized versions
- **Traffic management**: Switching to lighter models during peak load
- **Security incidents**: Such as the three system-prompt tampering events experienced by Grok in 2025

Existing change detection methods (e.g., MET, MMLU benchmarks) are expensive, requiring large numbers of queries and token generations, making LLM APIs practically immune to third-party monitoring.

Core insight: Although log probabilities are non-deterministic in practice, the logprob of a single token still contains sufficiently rich distributional information to detect extremely subtle changes via simple statistical tests.

## Method

### Overall Architecture

The LT pipeline is remarkably simple:
1. Send identical short prompts (the single letter "x") to two LLM API snapshots (same API at different time points)
2. Request only 1 output token with top-k logprobs returned
3. Repeat for $N$ samples
4. Apply a permutation test on the mean logprob of each token

### Key Designs

#### 1. Handling Non-Determinism

LLM logprob non-determinism arises from two sources:
- **Intentional non-determinism**: Temperature sampling (LT operates directly on logprobs and is thus unaffected)
- **Unintentional non-determinism**: Interference from co-batched requests; numerical variation due to different GPU routing

LT treats each logprob as a sample from some distribution and applies standard hypothesis testing to determine whether two distributions are identical.

#### 2. Two-Sample Test

Let $\mathcal{V} = \{t_1, \dots, t_{n_{\text{tok}}}\}$ denote the set of all observed tokens. The mean logprob for each token is computed as:

$$\bar{a}_i^{(1)} = \frac{1}{N}\sum_{j=1}^{N} T_{j,i}^{(1)}, \quad \bar{a}_i^{(2)} = \frac{1}{N}\sum_{j=1}^{N} T_{j,i}^{(2)}$$

The test statistic is the average absolute difference between per-token means:

$$S = \frac{1}{n_{\text{tok}}} \sum_{i=1}^{n_{\text{tok}}} |\bar{a}_i^{(1)} - \bar{a}_i^{(2)}|$$

A permutation test is used to obtain the p-value: the $2N$ samples are randomly split into two groups, the permutation statistic $S^{(b)}$ is computed $B$ times, and the p-value is $\hat{p} = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\{S^{(b)} \geq S\}$.

#### 3. Handling Missing Logprobs

Top-k truncation causes the token sets across different samples to be incomplete. For tokens missing in a given sample, the minimum logprob observed in that sample is used as a conservative imputation, since the true logprob is no greater than this value.

#### 4. TinyChange Benchmark

A benchmark comprising 58 model variants is constructed, spanning 5 levels of modification intensity:
- **Standard fine-tuning and LoRA fine-tuning**: 1 to 512 steps of single-example fine-tuning
- **Unstructured weight pruning**: By magnitude or randomly, with removal ratios from $2^{-10}$ to $1$
- **Parameter noise**: Gaussian noise with standard deviation $\sigma$ ranging from $2^{-15}$ to $1$

Applied to 5 open-source models (0.5B–8B parameters), yielding 290 variants in total.

### Loss & Training

LT is a purely statistical inference method with no training procedure. Core parameters:
- Number of samples $N=10$
- Number of permutations $B$
- Significance level $\alpha$
- Only 1–2 tokens required as input prompt

## Key Experimental Results

### Main Results

| Method | Overall AUC (95% CI) | Input tokens/test | Output tokens/test | Annual cost (GPT-4.1 pricing) |
|------|:-:|:-:|:-:|:-:|
| MMLU-ALG | 0.878 | $2.1 \times 10^5$ | $9.9 \times 10^3$ | $332 |
| MET | 0.670 | $2.9 \times 10^4$ | $2.0 \times 10^4$ | $146 |
| **LT (Ours)** | **0.915** | **28** | **20** | **$0.14** |

LT achieves the highest AUC (0.915) while requiring only 48 tokens per test (28 input + 20 output), approximately 1000× cheaper than MET and 2400× cheaper than MMLU-ALG.

| Modification Type | Highest difficulty at which LT achieves AUC > 0.9 | MET | MMLU-ALG |
|---------|:-:|:-:|:-:|
| Weight pruning | $\leq 2^{-10}$ | $2^{-1}$ | $2^{-4}$ |

LT's sensitivity to weight pruning is $2^9 = 512$× higher than MET and $2^6 = 64$× higher than MMLU-ALG.

### Ablation Study

**Effect of prompt length**: The AUC difference between the shortest prompt (1.5 tokens) and the longest (33 tokens) is only ~1%, indicating that extremely short prompts are sufficient for reliable detection.

**Real-world deployment**: 189 endpoints were monitored over 4 months, collecting 1.7M+ responses. A total of 37 suspected changes were detected across 29 endpoints from 7 providers. Nearly all detected changes (34/37) affected models with open-source weights.

### Key Findings

- A prompt as short as a single letter "x" is sufficient for reliable change detection
- LoRA fine-tuning is the most difficult modification to detect across all methods
- Open-weight models are equally subject to undisclosed modifications
- Some providers (e.g., OpenAI) have begun enforcing minimum output token counts (≥16), potentially to hinder monitoring

## Highlights & Insights

1. **Extreme minimalism**: 1-token input + 1-token output + simple statistical test = outperforms complex methods
2. **Information density perspective**: Logprobs carry richer distributional information than generated tokens — a severely underutilized signal source
3. **High practical utility**: Hourly monitoring at an annual cost of $0.14 makes large-scale continuous monitoring feasible
4. **Transparency concern**: 34/37 detected changes involved open-weight models, revealing that open weights do not imply deployment transparency

## Limitations & Future Work

- Requires API support for returning logprobs (currently only ~23% of endpoints support this)
- Cannot distinguish between infrastructure changes and specific model updates
- Providers may circumvent detection by caching logprobs or identifying monitoring queries
- Certain modifications (e.g., adjusting end-of-sequence bias) may not affect the first token
- The method focuses on change detection and does not characterize the nature of detected changes

## Related Work & Insights

- Closely related to but distinct from LLM fingerprinting: LT prioritizes sensitivity to subtle changes
- Zero-knowledge proofs (zkLLM, TOPLOC) offer stronger guarantees but at far greater computational cost
- Complementary to existing audit pipelines: LT serves as a low-cost, high-sensitivity first line of defense
- Has direct implications for AI safety and reproducibility research

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The insight of using logprobs as a monitoring signal is highly innovative
- Technical Depth: ⭐⭐⭐⭐ — Statistical methodology is simple yet effective, with clear theoretical analysis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — TinyChange benchmark + large-scale real-world deployment validation
- Value: ⭐⭐⭐⭐⭐ — Directly deployable at minimal cost

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)
- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[ACL 2026\] ViLL-E: Video LLM Embeddings for Retrieval](../../ACL2026/video_understanding/vill-e_video_llm_embeddings_for_retrieval.md)
- [\[NeurIPS 2025\] A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](../../NeurIPS2025/video_understanding/a_little_depth_goes_a_long_way_the_expressive_power_of_logde.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)

</div>

<!-- RELATED:END -->
