---
title: >-
  [Paper Note] Token-Efficient Change Detection in LLM APIs
description: >-
  [ICML 2026][LLM/NLP][Black-box change detection] The authors demonstrate that under low-temperature sampling, specific inputs where two token logits are nearly tied (Border Inputs) are extremely sensitive to parameter pe…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Black-box change detection"
  - "Border Inputs (BI)"
  - "Low-temperature phase transition"
  - "Local Asymptotic Normality"
  - "B3IT"
date: 2026-05-08
content_hash: b5610c62a23c6977
---

# Token-Efficient Change Detection in LLM APIs

**Conference**: ICML 2026  
**arXiv**: [2602.11083](https://arxiv.org/abs/2602.11083)  
**Code**: https://github.com/timothee-chauvin/token-efficient-change-detection-llm-apis  
**Area**: LLM Trusted Deployment / Model Change Detection  
**Keywords**: Black-box change detection, Border Inputs (BI), Low-temperature phase transition, Local Asymptotic Normality, B3IT

## TL;DR
The authors demonstrate that under low-temperature sampling, specific inputs where two token logits are nearly tied (Border Inputs) are extremely sensitive to parameter perturbations—theoretically, the SNR diverges as $T\to 0$. Consequently, LLM API change detection can be performed using very few requests by observing only the output tokens (strict black-box). The proposed B3IT matches the performance of gray-box logprob methods at 1/30 of the cost on the TinyChange benchmark and identified 8 real-world model replacements during 23 days of continuous monitoring across 93 commercial endpoints.

## Background & Motivation
**Background**: LLM API providers often silently replace models (quantization, new versions, rollbacks) without notifying users; in 2025, unannounced changes by Anthropic and Grok affected millions of requests. Existing change detection methods are categorized into three tiers: white-box (ESF, TRAP, requiring weights/gradients), gray-box (LT, requiring logprobs), and black-box (MET, MMLU-ALG, observing only output tokens but requiring high request volumes and costs).

**Limitations of Prior Work**: White-box methods require open models and are inapplicable to closed APIs. Gray-box methods require logprobs, which many APIs (including most endpoints on OpenRouter) do not return. Black-box methods like MET compare output distributions across multiple tokens using MMD distance, incurring extremely high costs for continuous monitoring (e.g., DailyBench shut down after 40 days monitoring only 5 endpoints).

**Key Challenge**: Black-box change detection must simultaneously achieve (i) strict observation of output tokens only, (ii) low-cost sustainable monitoring, and (iii) high sensitivity to detect minute changes (quantization, single-step fine-tuning). These three objectives seem incompatible—intuitively, output tokens are lossy compressions of logits via argmax/softmax and should be insensitive to small perturbations.

**Goal**: (i) Establish a theoretical foundation for "output-token black-box change detection" to identify conditions for high-sensitivity detection; (ii) translate this theory into a practical algorithm; (iii) validate the approach on both controlled benchmarks and real production APIs.

**Key Insight**: Starting from Neyman-Pearson optimal detection, the authors use the Local Asymptotic Normality (LAN) framework to analyze the optimal SNR under "small perturbation + multiple sampling." They find that $\text{SNR}^2$ is a quadratic form involving the Fisher information and the model Jacobian. Crucially, in the low-temperature limit, the system exhibits a sharp dichotomy: when the output distribution collapses to a single token, SNR→0 (undetectable); when two tokens are tied, SNR→∞ (extremely detectable). This "phase transition" inspires the concept of Border Inputs.

**Core Idea**: Identify inputs where exactly two tokens are nearly tied under low temperatures (Border Inputs) and use their output token distribution (uniform on $\{1, \dots, k\}$) as a model fingerprint. Any parameter perturbation causes these BIs to degenerate into single-token outputs, making detection nearly free.

## Method

### Overall Architecture
B3IT consists of two stages: (1) **Initialization**: Sample $n$ candidate inputs $m=3$ times each at low temperature $T=0$; those producing $\ge 2$ distinct output tokens are identified as BIs. Select 5 BIs and sample each $n_1=50$ times to establish reference distributions. (2) **Detection**: Periodically (e.g., daily) sample each BI $n_2=3$ times to obtain the current support set $\hat S_2$ and compare it with the reference $\hat S_1$. If $\hat S_1 \triangle \hat S_2 \ne \emptyset$ (i.e., a token appears in one set but not the other), a model change is detected. Each BI detection costs only 3 output tokens, making it "token-efficient."

### Key Designs

1. **Theoretical Foundation of Border Inputs and Low-Temperature Phase Transitions**:
    - **Function**: Provide a theoretical proof for "why BIs act as high-sensitivity detectors," upgrading the BI concept from a heuristic trick to a mathematically optimal strategy.
    - **Mechanism**: Under the LAN framework, for a parameter perturbation $\theta \mapsto \theta + \epsilon h$, the Type-II error of optimal detection is dominated by the scalar $\text{SNR}^2(h) = h^T J^T F(\mathbf p_0)^{-1} J h$, where $J$ is the Jacobian of the output distribution with respect to parameters and $F$ is the Fisher information. Expanding this for the final Transformer layer yields $\text{SNR}^2(h) = \frac{1}{\tau^2} h^T J_z^T \Sigma(\mathbf p^{(\tau)}) J_z h$, where the signal is amplified by the inverse square of temperature $\tau$. At the low-temp limit $\tau \to 0$: if there is a unique maximum logit ($k=1$), the output degenerates to a Dirac distribution and SNR→0; if $k \ge 2$ logits tie for the maximum (BI), SNR→+∞. This represents a sharp phase transition (Theorem 3.3).
    - **Design Motivation**: To ensure principled method selection. While intuition suggests output tokens are insensitive due to lossy compression, BIs are "singular points" in logit space. As temperature →0, softmax acts as argmax, and any perturbation reordering the logits causes a token jump. This provides a rigorous mathematical explanation for how black-box detection is possible.

2. **Black-Box BI Discovery + Support Set Difference Detection**:
    - **Function**: Inexpensively identify BIs without accessing model weights and reduce detection to simple set difference checking.
    - **Mechanism**: BI Discovery—for $n$ random inputs, sample each $m=3$ times at $T=0$ (due to non-determinism and floating-point rounding, BIs likely yield $\ge 2$ tokens, while non-BIs consistently yield one). Detection—at $T=0$, the output distribution of a BI is $\text{Unif}(S_1)$. Detection becomes a support set test: $H_0: S_1 = S_2$ vs $H_1: S_1 \ne S_2$, with the rejection condition $\mathcal R = (\hat S_1 \setminus \hat S_2) \cup (\hat S_2 \setminus \hat S_1) \ne \emptyset$. The paper proves that for the typical case of $k=2, n_1=n_2=n$, this simple test is **Neyman-Pearson optimal up to a constant factor** (Theorem 4.3).
    - **Design Motivation**: To downgrade change detection from "distribution distance estimation" to "checking for unseen tokens," significantly reducing sample requirements. Non-asymptotic guarantees for Type-I error ($\le k e^{-n_1/k} + k e^{-n_2/k}$) and Type-II error ($\le p_1^{n_1} p_2^{n_2}$) are provided.

3. **Engineering Design of $m=3$ and Multi-Prompt Aggregation**:
    - **Function**: Minimize BI search costs and improve SNR through multi-prompt averaging.
    - **Mechanism**: During BI search, the goal of $m$ samples per candidate is to decide if $\ge 2$ tokens are produced. Back-of-the-envelope calculations (Appendix C) show that $m=3$ is the optimal BI-per-request ratio when the BI proportion is $< 75\%$. During detection, the TV distances of individual prompts are used to generate ROC curves, and the average TV across 5 prompts serves as the detection statistic. Aggregating multiple prompts significantly improves ROC AUC (5 prompts × 10 samples is more accurate than 1 prompt × 50 samples).
    - **Design Motivation**: Cost is the primary bottleneck for black-box methods. $m=3$ is mathematically optimal and computationally cheapest. Multi-prompt aggregation utilizes the same token budget for higher accuracy, following the statistical intuition that "wide and shallow" sampling outperforms "narrow and deep" sampling.

### Loss & Training
The method is entirely training-free. Detection protocol: BI count = 5, reference samples = 50, detection samples = 3, interval = 24 hours. A persistent change is flagged if the mean TV threshold ($< 0.5 \to > 0.5$) is maintained for $\ge 4$ days to avoid transient noise.

## Key Experimental Results

### Main Results
TinyChange in-vitro evaluation (9 models 0.5B-9B × various perturbations):

| Method | Type | ROC AUC | Annual Cost |
|---|---|---|---|
| LT (Gray-box, requires logprob) | Gray-box | ~0.95 | <$1 |
| **B3IT (Ours)** | **Black-box** | **0.90** | **$2.2** |
| MET ($T=0$) | Black-box | 0.61→0.88 | $2.2→$67 |
| MMLU-ALG | Black-box | Much lower | High |

B3IT achieves an ROC-AUC of 0.87 even for extremely weak perturbations (single-step fine-tuning).

In-vivo commercial endpoint evaluation (93 endpoints, 64 models, 20 providers, 23 days):

| Metric | Value |
|---|---|
| BI prevalence at $T=0$ | 62% |
| BI prevalence at $T>0$ | 80% |
| Endpoints unable to find BIs | 18 (mostly reasoning models where closing reasoning failed) |
| Persistent changes detected | 8 endpoints (including Together AI's Mistral-7B → Ministral-3-14B replacement) |
| Average BI monitoring cost | $0.52 / endpoint / year (hourly checks) |
| Initialization cost | $0.0045 / endpoint (negligible) |

### Ablation Study

| Configuration | Key Finding | Meaning |
|---|---|---|
| Prompts × samples Pareto scan | 5×10 outperforms 1×50 | Multi-prompt aggregation is superior |
| $m=1$ vs $m=3$ vs $m=10$ | $m=3$ is most efficient | Confirms back-of-the-envelope estimations |
| $T=0$ vs $T>0$ | $T>0$ finds more BIs but quality decreases | Phase transition is weakened as temperature increases |

### Key Findings
- BIs exist abundantly in practice (62% of endpoints at $T=0$) because limited floating-point precision and inference non-determinism cause logit-tying events—which theoretically have zero probability—to occur frequently.
- Among the 8 "persistent changes" detected on commercial endpoints, one directly corresponds to a Together AI announcement (Mistral-7B-Instruct-v0.3 was quietly replaced by Ministral-3-14B-Instruct-2512), proving the method catches real-world model swaps.
- 18 endpoints failed to yield BIs—mainly reasoning models where reasoning cannot be disabled and first-token output length is restricted, representing a limitation of the method.

## Highlights & Insights
- **A Model for Naming Methods from Theoretical Phenomena**: BI is not an empirical trick but a singular point naturally emerging from LAN and low-temp limit analysis. This approach of deriving theory first before naming engineering concepts elevates engineering research to scientific inquiry.
- **Support Set Difference = Neyman-Pearson Optimality for $k=2$**: Reducing a complex distribution distance test to "checking for new tokens" is an elegant example of solving a difficult problem with the simplest possible model, proven optimal up to a constant factor.
- **Floating-Point Precision + Non-determinism as the Physical Source of BIs**: The authors acknowledge that while logit ties have zero probability in theory, they occur frequently due to batch-size dependencies in GPU inference and FP16/BF16 rounding. This turns "hardware non-determinism" from a bug into a feature.

## Limitations & Future Work
- Only the first output token is used; dependencies between multiple output tokens are ignored, losing substantial signal.
- BIs are a joint product of the model and the provider's infrastructure. Changes in GPUs or CUDA versions by the provider might change BI distributions, potentially causing false positives for "model changes" (though these are "deployment changes" in a broader sense).
- Ineffective for reasoning models (where reasoning cannot be disabled, leading to late or unreadable first tokens). This is a growing weakness as more reasoning models are deployed.
- Adversarial providers could intentionally force BIs to always output the same token (e.g., via a sticky-mode system prompt) to deceive B3IT.
- Future directions: Extend to multi-token sequences (modeling token dependencies with Markov or Hidden Markov Models); combine with LT (logprob-based) for hybrid detectors; research robust detection under adversarial changes.

## Related Work & Insights
- **vs MET (Gao et al. 2025)**: MET uses MMD with Hamming kernels for multi-token distributions and is costly (B3IT is 30x cheaper); B3IT uses phase transitions and support set differences, making it theoretically and practically more refined.
- **vs LT (Chauvin et al. 2026)**: LT requires logprobs and is limited to APIs supporting them; B3IT does not require logprobs, performs nearly as well, and has broader coverage.
- **vs ESF / TRAP (White-box)**: White-box methods require model weights and are inapplicable to closed APIs; B3IT is strictly black-box and works for any commercial endpoint.
- **vs LLM Fingerprinting (Pasquini et al. 2025)**: Fingerprinting seeks "stability under small changes" to identify models; B3IT seeks "sensitivity to any small change" to trigger alerts—the two goals are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Translating the "low-temperature phase transition" from a mathematical phenomenon into the practical BI concept is a rare example of theory-driven algorithm design in this field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage with 9 open-source models in-vitro and 93 commercial endpoints in-vivo, validated by real incidents; only lacks adversarial scenario testing.
- Writing Quality: ⭐⭐⭐⭐⭐ The progression from phase transition theorems to a 5-line-of-code detector is a near-perfect integration of theory and engineering.
- Value: ⭐⭐⭐⭐⭐ Black-box monitoring at 1/30 the cost is a vital requirement for any production system dependent on LLM APIs, and the method is already proven by real-world swaps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](../../ACL2026/llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[ICML 2026\] Structured Generalized Linear Token Mixing: Shifting Gears Between Complexity and Expressivity with SND + Kronecker](trading_complexity_for_expressivity_through_structured_generalized_linear_token_.md)
- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](../../ACL2026/llm_nlp/understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)
- [\[ICML 2026\] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)

</div>

<!-- RELATED:END -->
