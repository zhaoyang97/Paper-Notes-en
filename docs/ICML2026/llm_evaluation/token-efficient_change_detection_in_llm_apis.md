---
title: >-
  [Paper Note] Token-Efficient Change Detection in LLM APIs
description: >-
  [ICML 2026][LLM Evaluation][Black-box change detection] The authors prove that under low-temperature sampling, special inputs where "two token logits are nearly tied" (Border Inputs) are extremely sensitive to parameter…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Black-box change detection"
  - "Border Input (BI)"
  - "low-temperature phase transition"
  - "Local Asymptotic Normality"
  - "B3IT"
date: 2026-05-08
content_hash: 9b2cac8ffbf07ea7
---

# Token-Efficient Change Detection in LLM APIs

**Conference**: ICML 2026  
**arXiv**: [2602.11083](https://arxiv.org/abs/2602.11083)  
**Code**: https://github.com/timothee-chauvin/token-efficient-change-detection-llm-apis  
**Area**: Trustworthy LLM Deployment / Model Change Detection  
**Keywords**: Black-box change detection, Border Input (BI), low-temperature phase transition, Local Asymptotic Normality, B3IT

## TL;DR
The authors prove that under low-temperature sampling, special inputs where "two token logits are nearly tied" (Border Inputs) are extremely sensitive to parameter perturbations—theoretically, SNR diverges as $T\to 0$. Thus, by observing only output tokens (strict black-box), LLM API change detection can be performed with very few queries. The proposed B3IT matches gray-box logprob methods on the TinyChange benchmark at 1/30 the cost, and in 23 days of continuous monitoring across 93 commercial endpoints, it detected 8 real model replacements.

## Background & Motivation
**Background**: LLM API providers often silently swap models (quantization, new versions, rollbacks), leaving users unaware; in 2025, both Anthropic and Grok had unannounced changes affecting millions of requests. Existing change detection methods fall into three categories: white-box (ESF, TRAP, require weights/gradients), gray-box (LT, require logprobs), and black-box (MET, MMLU-ALG, observe only output tokens but require large numbers of queries and are costly).

**Limitations of Prior Work**: White-box methods require open models, unsuitable for closed APIs; gray-box methods need logprobs, but many APIs (including most endpoints on OpenRouter) do not return logprobs; black-box methods like MET compare output distributions over multiple tokens using MMD, requiring massive numbers of queries for continuous monitoring (DailyBench stopped after 40 days for just 5 endpoints due to cost).

**Key Challenge**: Black-box change detection must balance (i) strictly observing only output tokens, (ii) low-cost sustainable monitoring, and (iii) high sensitivity to small changes (quantization, single-step fine-tuning). These seem mutually exclusive—intuitively, token outputs are a lossy compression of logits via argmax/softmax and should be insensitive to small perturbations.

**Goal**: (i) Establish a theoretical foundation for "output token black-box change detection," identifying conditions for high-sensitivity detection; (ii) translate theory into a practical algorithm; (iii) validate on both controlled benchmarks and real production APIs.

**Key Insight**: Starting from Neyman-Pearson optimal detection, the authors use the Local Asymptotic Normality (LAN) framework to analyze optimal SNR under "small perturbation + repeated sampling." They find that SNR² is a quadratic form involving Fisher information and the model Jacobian, and in the low-temperature limit, there is a sharp dichotomy: when the output distribution collapses to a single token, SNR→0 (undetectable); when two tokens are tied (BI), SNR→∞ (easily detectable). This "phase transition" inspires the Border Input concept.

**Core Idea**: Under low temperature, specifically search for inputs where "exactly two tokens are nearly tied," and use their output token distributions (uniform on $\{1, \dots, k\}$) as model fingerprints; any parameter perturbation will cause the BI to degenerate to a single token output, making detection nearly free.

## Method

### Overall Architecture
B3IT operates in two stages: (1) **Initialization**: For $n$ candidate inputs, sample each $m=3$ times at low temperature $T=0$; those producing ≥2 different output tokens are BIs. Select 5 BIs and sample each $n_1=50$ times as the reference distribution. (2) **Detection**: Periodically (e.g., daily), sample each BI $n_2=3$ times to obtain the current support set $\hat S_2$, and compare with the reference support set $\hat S_1$; if $\hat S_1 \triangle \hat S_2 \ne \emptyset$ (i.e., a token appears on one side but not the other), a model change is detected. Each BI requires only 3 output token queries per detection, hence "token-efficient."

### Key Designs

1. **Theoretical Foundation of Border Input and Low-Temperature Phase Transition**:

    - Function: Theoretically proves "why BIs can serve as highly sensitive detectors," elevating the BI concept from a heuristic trick to a mathematically optimal strategy.
    - Mechanism: Under the LAN framework, for parameter perturbation $\theta \mapsto \theta + \epsilon h$, the optimal detection's Type-II error is dominated by a single scalar $\text{SNR}^2(h) = h^T J^T F(\mathbf p_0)^{-1} J h$, where $J$ is the Jacobian of the output distribution with respect to parameters, and $F$ is the Fisher information. Expanding using the transformer's final layer yields $\text{SNR}^2(h) = \frac{1}{\tau^2} h^T J_z^T \Sigma(\mathbf p^{(\tau)}) J_z h$, with temperature $\tau$ amplifying the signal inversely with its square. As $\tau \to 0$: if logits have a unique maximum ($k=1$), the output degenerates to a Dirac, SNR→0; if $k \ge 2$ logits are tied (BI), SNR→+∞—a sharp phase transition (Theorem 3.3).
    - Design Motivation: Provides principled method selection. Intuitively, "output tokens are a lossy compression" and should be insensitive to small perturbations; but BIs are "singularities" in logit space, and as temperature →0, softmax becomes argmax, so any perturbation that changes logit order causes output token jumps. This gives a rigorous mathematical explanation for "why black-box detection is possible."

2. **Black-Box BI Discovery + Support Set Difference Detection**:

    - Function: Identifies BIs and reduces detection to simple set difference testing without accessing model weights.
    - Mechanism: BI discovery—sample $n$ random inputs $m=3$ times each at $T=0$ (due to non-determinism and floating-point rounding, BIs will likely yield ≥2 tokens in 3 samples; non-BIs always yield the same token); retain inputs producing ≥2 tokens. Detection—at $T=0$, BI output distribution is uniform $\text{Unif}(S_1)$, so detection becomes support set difference testing: $H_0: S_1 = S_2$ vs $H_1: S_1 \ne S_2$, with rejection criterion $\mathcal R = (\hat S_1 \setminus \hat S_2) \cup (\hat S_2 \setminus \hat S_1) \ne \emptyset$. The paper proves that in the typical case $k=2, n_1=n_2=n$, this simple test is **Neyman-Pearson optimal up to a constant factor** (Theorem 4.3).
    - Design Motivation: Reduces change detection from "distribution distance estimation" to "checking for unseen tokens in new samples," greatly lowering the required number of samples. Theoretically, Type-I error $\le k e^{-n_1/k} + k e^{-n_2/k}$, Type-II error $\le p_1^{n_1} p_2^{n_2}$, providing non-asymptotic guarantees.

3. **$m=3$ and Multi-Prompt Aggregation Engineering Design**:

    - Function: Minimizes BI search cost and boosts signal-to-noise ratio via multi-prompt averaging.
    - Mechanism: The goal of sampling each candidate $m$ times during BI search is to determine "whether ≥2 tokens are produced"; back-of-the-envelope calculation (Appendix C) shows $m=3$ is optimal for BI/query ratio when BI proportion < 75%. For detection, compute ROC curves using per-prompt TV distance, then average per-prompt TV across 5 prompts as the detection statistic—multi-prompt aggregation significantly improves ROC AUC (5 prompts × 10 samples outperforms 1 prompt × 50 samples).
    - Design Motivation: Cost is the key bottleneck for black-box methods; $m=3$ is mathematically optimal and engineering-wise cheapest; multi-prompt aggregation achieves higher detection accuracy with the same token budget, reflecting the statistical intuition that "wide and shallow" beats "narrow and deep."

### Loss & Training
Completely training-free. Detection protocol: number of BIs = 5, reference samples = 50, detection samples = 3, interval = 24 hours, decision threshold = mean TV $< 0.5 \to > 0.5$ sustained for ≥4 days counts as a persistent change (to avoid transient fluctuations).

## Key Experimental Results

### Main Results
TinyChange in-vitro evaluation (9 models, 0.5B–9B, various perturbations):

| Method | Type | ROC AUC | Annual Cost |
|---|---|---|---|
| LT (gray-box, requires logprob) | Gray-box | ~0.95 | <$1 |
| **B3IT (ours)** | **Black-box** | **0.90** | **$2.2** |
| MET ($T=0$) | Black-box | 0.61→0.88 | $2.2→$67 |
| MMLU-ALG | Black-box | Much lower | High |

For extremely weak perturbations (single-step fine-tune), B3IT still achieves ROC-AUC 0.87.

In-vivo commercial endpoint evaluation (93 endpoints, 64 models, 20 providers, 23 days):

| Metric | Value |
|---|---|
| BI existence rate at $T=0$ endpoints | 62% |
| BI existence rate at $T>0$ endpoints | 80% |
| Number of endpoints without BI | 18 (mostly reasoning models with failed reasoning disable) |
| Number of persistent change detections | 8 endpoints (including Together AI's Mistral-7B → Ministral-3-14B replacement) |
| Average BI monitoring cost | $0.52 / endpoint / year (hourly) |
| Initialization cost | $0.0045 / endpoint (negligible) |

### Ablation Study

| Configuration | Key Findings | Meaning |
|---|---|---|
| prompts × samples Pareto scan | 5×10 outperforms 1×50 | Multi-prompt aggregation is better |
| $m=1$ vs $m=3$ vs $m=10$ | $m=3$ is most efficient | Back-of-envelope estimate holds |
| $T=0$ vs $T>0$ | $T>0$ finds more BIs but BI quality drops | Phase transition weakens as temperature increases |

### Key Findings
- BIs are abundant in practice (62% of endpoints at $T=0$), due to limited floating-point precision and inference non-determinism causing "strictly zero-probability" logit ties to occur frequently.
- Of the 8 "persistent changes" detected on commercial endpoints, 1 directly matched a Together AI announcement (Mistral-7B-Instruct-v0.3 quietly replaced by Ministral-3-14B-Instruct-2512), proving the method can catch real-world model swaps.
- 18 endpoints lacked BIs—mainly because reasoning models cannot disable reasoning by default, or output token length is limited, representing a method limitation.

## Highlights & Insights
- **A paradigm for naming methods from theoretical phenomena**: BI is not a "trial-and-error trick," but a singularity naturally arising from LAN + low-temperature limit analysis. This "derive theory first, then name the engineering concept" approach elevates engineering research to scientific research, serving as a strong example.
- **Support set difference = Neyman-Pearson optimal for $k=2$**: Reduces complex distribution distance testing to "checking for new tokens," simple enough for 5 lines of code yet provably optimal up to a constant factor—a beautiful "simplest model solves hardest problem."
- **Floating-point precision + inference non-determinism as the physical source of BI**: The authors acknowledge that theoretically, logit ties have zero probability, but in practice they occur frequently due to GPU inference batch-size dependence and FP16/BF16 rounding—turning "hardware non-determinism" from a bug into a feature.

## Limitations & Future Work
- Only the first output token is used; dependencies between multiple output tokens are ignored, losing substantial signal.
- BI is a joint product of the model and provider infrastructure; provider changes to GPU or CUDA versions can alter the BI distribution, possibly causing false positives for "model change" (though broadly this is indeed a "deployment change").
- Completely ineffective for reasoning models (reasoning cannot be disabled → first output token is too late/unreadable); this is a growing issue as more reasoning models are deployed from 2025 onward.
- Adversarial providers could deliberately make BIs always output the same token (e.g., by adding sticky-mode system prompts), thus deceiving B3IT.
- Future directions: extend to multi-token sequences (model token dependencies with Markov or hidden Markov models); combine with LT (logprob-based) for hybrid detectors; study robust detection under adversarial changes.

## Related Work & Insights
- **vs MET (Gao et al. 2025)**: MET uses MMD + Hamming kernel to compare multi-token output distributions, at high cost (B3IT is 30× cheaper); B3IT uses phase transition + support set difference, more concise both theoretically and practically.
- **vs LT (Chauvin et al. 2026)**: LT requires logprobs, limited to APIs that support logprobs; B3IT does not require logprobs, with only slightly lower performance but much broader coverage.
- **vs ESF / TRAP (white-box)**: White-box methods require model weights, unsuitable for closed APIs; B3IT is strictly black-box and applicable to any commercial endpoint.
- **vs LLM fingerprinting (Pasquini et al. 2025)**: Fingerprinting seeks "stability under small changes" for model identification, while B3IT seeks "sensitivity to any small change" to trigger alarms—their goals are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Transforms the "low-temperature phase transition" of change detection from a purely mathematical phenomenon into a practical BI concept—a rare "theory-driven algorithm" in this field.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 open-source models in-vitro + 93 commercial endpoints in-vivo + real incident validation, with extremely broad coverage; only lacks experiments in adversarial scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical results flow seamlessly into a 5-line code detector, with near-perfect integration of theory and engineering.
- Value: ⭐⭐⭐⭐⭐ Black-box change monitoring at 1/30 the cost is essential for any production system relying on LLM APIs, and the method has been validated on real incidents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/llm_evaluation/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)
- [\[ICLR 2026\] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](../../ICLR2026/llm_evaluation/when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)

</div>

<!-- RELATED:END -->
