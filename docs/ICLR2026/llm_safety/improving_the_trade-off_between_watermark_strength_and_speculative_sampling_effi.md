---
title: >-
  [Paper Note] Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models
description: >-
  [ICLR 2026][LLM Safety][LLM watermarking] This paper proposes a quantitative measure of watermark strength (expected KL divergence) and fully characterizes the Pareto trade-off curve between watermark strength and specul…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "LLM watermarking"
  - "speculative sampling"
  - "watermark strength"
  - "sampling efficiency"
  - "Pareto frontier"
  - "pseudo-random acceptance"
date: 2026-05-08
content_hash: 44e3fbd4cd975c1e
---

# Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models

**Conference**: ICLR 2026
**arXiv**: [2602.01428](https://arxiv.org/abs/2602.01428)  
**Code**: [GitHub](https://github.com/hwq0726/watermark-tradeoff)  
**Area**: AI Safety
**Keywords**: LLM watermarking, speculative sampling, watermark strength, sampling efficiency, Pareto frontier, pseudo-random acceptance

## TL;DR

This paper proposes a quantitative measure of watermark strength (expected KL divergence) and fully characterizes the Pareto trade-off curve between watermark strength and speculative sampling efficiency. By pseudo-randomizing the acceptance decision, the method simultaneously achieves maximum watermark strength and optimal sampling efficiency.

## Background & Motivation

1. **LLM watermarking is a key technology for provenance**: Watermarking injects recoverable pseudo-random signals during token sampling to enable text provenance tracking, serving as a principled approach to ensure generated content is traceable.

2. **Speculative sampling accelerates LLM inference**: Speculative sampling uses a lightweight draft model to rapidly generate candidate tokens, which are then verified in parallel by the target model. Higher acceptance rates yield greater speedups. Efficiency is measured by the expected acceptance rate: $\text{SE} = \mathbb{E}_\zeta[\sum_w \mathcal{A}_\zeta(w|w) Q_{\zeta,w}]$.

3. **A fundamental trade-off exists between watermarking and speculative sampling**: Hu & Huang (2024) prove that it is impossible to simultaneously maintain the highest acceptance rate and the strongest watermark — stronger watermarks cause greater divergence between the draft and target distributions, reducing the acceptance rate. This impossibility result is discouraging.

4. **Prior definitions of watermark strength are too coarse**: Existing definitions are binary (watermark is "preserved" only if the watermarked distribution is an exact match), ignoring intermediate strength levels and precluding fine-grained trade-off analysis. A continuous, quantifiable measure of watermark strength is needed to fully characterize the trade-off curve.

## Method

### Overall Architecture

- **Function**: (1) Define a continuous quantitative measure of watermark strength; (2) fully characterize the Pareto frontier of watermark strength vs. sampling efficiency; (3) propose a new mechanism that breaks the trade-off.
- **Design Motivation**: Prior binary definitions cannot reveal the continuous nature of the trade-off, and the true randomness in the accept/reject step is the root cause of watermark strength degradation.
- **Mechanism**: Quantify watermark strength via expected KL divergence → derive a convex optimization formulation of the Pareto curve → pseudo-randomize the acceptance decision so that the entire generation process becomes a deterministic function of pseudo-random variables.

### Key Design 1: Quantitative Measure of Watermark Strength

The watermark strength of an unbiased watermarking scheme is defined as the expected KL divergence between the watermarked distribution and the original distribution:

$$\text{WS}(\bm{P}_\zeta) = \mathbb{E}_\zeta[D_{\mathrm{KL}}(\bm{P}_\zeta \| \bm{P})] = \text{Ent}(\bm{P}) - \mathbb{E}_\zeta[\text{Ent}(\bm{P}_\zeta)]$$

This measure carries deep information-theoretic meaning: it is equivalent to the mutual information $I(w; \zeta)$ between token $w$ and the pseudo-random variable $\zeta$. Key properties include:

- **Upper bounded by the entropy of the original distribution**: $\text{WS}(\bm{P}_\zeta) \leq \text{Ent}(\bm{P})$, with equality if and only if $\bm{P}_\zeta$ is almost surely degenerate (all probability mass concentrated on a single token).
- **Determines the exponential decay rate of p-values**: Under a likelihood ratio test, the p-value for $n$ tokens satisfies $\lim_{n\to\infty} -\frac{1}{n}\log(\text{p-value}) = \bar{D}$, directly determining the sample complexity required for detection.
- **Both Gumbel-max and SynthID ($m\to\infty$) achieve maximum watermark strength**.

### Key Design 2: Complete Characterization of the Pareto Trade-off Curve

The trade-off curve is formulated as a constrained optimization problem: given an efficiency requirement $r$, find the maximum achievable watermark strength:

$$L(r) = \max_{\mathcal{S}_{\text{draft}}, \mathcal{S}_{\text{target}}} \text{WS}(\bm{P}_\zeta) \quad \text{s.t.} \quad \text{SSE}(\bm{Q}_\zeta, \bm{P}_\zeta) \geq r$$

For the linear watermark class $\mathcal{Q} = \{(1-\theta)\text{Id} + \theta \mathcal{S} : \theta \in [0,1]\}$, the inverse curve reduces to a convex optimization problem where the objective is to minimize the $\ell_1$ norm (corresponding to total variation distance) subject to an entropy upper bound. The paper derives complete Pareto curves for both Gumbel-max and SynthID watermarking schemes.

### Key Design 3: Pseudo-Random Acceptance Mechanism Breaks the Trade-off

**Core Idea**: In standard speculative sampling, the accept/reject decision for draft tokens uses a **truly random** coin flip. Even given the pseudo-random variables of both draft and target models, the final token remains stochastic. This residual randomness weakens watermark strength.

**Solution**: Pseudo-randomize the acceptance variable $u_t = G(\zeta_t^R)$ as well, making the entire generation process a deterministic function of pseudo-random variables. Theorem 4.1 establishes:

- **Unbiasedness**: $\mathbb{E}_\zeta[\bm{P}'_\zeta(w)] = \bm{P}(w)$
- **Maximum sampling efficiency**: $\text{SE} = 1 - \text{TV}(\bm{Q}, \bm{P})$
- **Maximum watermark strength**: $\text{WS}(\bm{P}'_\zeta) = \text{Ent}(\bm{P})$

At detection time, the pseudo-random acceptance variable $u_t$ provides additional information for more accurately selecting the correct test statistic. For Gumbel-max, an Ars-τ detector is proposed (using threshold $\tau$ to select between draft/target statistics); for SynthID, a Bayes-MLP detector is proposed (replacing simple weighted averaging with an MLP for statistic selection).

## Key Experimental Results

### Experimental Setup

- **Model pairs**: Llama-68M (draft) + Llama-7B (target); Gemma-2B (draft) + Gemma-7B (target)
- **Datasets**: ELI5 question-answering task, C4 open-generation task
- **Watermarking schemes**: Gumbel-max (temperature 0.5), SynthID (m=30, temperature 0.7)
- **Lookahead**: K ∈ {2, 3, 4}
- **Metrics**: AATPS (average accepted tokens per step), TPR@FPR=1%

### Main Results: Sampling Efficiency and Detection Performance

| Method | Watermark Scheme | AATPS (K=4) | TPR@100 tokens | TPR@200 tokens |
|---|---|---|---|---|
| Std. SpecSampl | None | ~3.2 | - | - |
| Ars-Prior | Gumbel-max | ~3.2 | ~65% | ~88% |
| **Ars-τ (Ours)** | **Gumbel-max** | **~3.2** | **~78%** | **~95%** |
| Bayes-Prior | SynthID | ~3.2 | ~50% | ~75% |
| **Bayes-MLP (Ours)** | **SynthID** | **~3.2** | **~62%** | **~85%** |
| Oracle (theoretical upper bound) | Both | ~3.2 | ~85% | ~98% |

### Ablation Study: Trade-off Curve Validation

| Efficiency Requirement (r) | Gumbel-max WS | SynthID WS (m=30) | SynthID WS (m=∞) |
|---|---|---|---|
| Maximum efficiency | 0 | 0 | 0 |
| 0.9 × max efficiency | Moderate | Moderate-low | Moderate |
| 0.7 × max efficiency | High | Moderate | High |
| No efficiency constraint | Maximum = Ent(P) | < Ent(P) | Maximum = Ent(P) |

Experiments confirm that Gumbel-max and SynthID ($m=\infty$) achieve the same maximum watermark strength, while finite-$m$ SynthID yields lower watermark strength than Gumbel-max.

### Key Findings

1. **Sampling efficiency is fully preserved**: AATPS under the pseudo-random acceptance mechanism is nearly identical to standard speculative sampling, confirming the maximum efficiency property in Theorem 4.1.
2. **Detection capability is significantly improved**: At the same token count, Ars-τ improves TPR by approximately 10–15 percentage points over Ars-Prior, and Bayes-MLP improves TPR by approximately 10–12 percentage points over Bayes-Prior.
3. **Approaches the Oracle upper bound**: At 200 tokens, the proposed method nearly matches the performance of the ideal Oracle detector, indicating that pseudo-randomizing the acceptance variable effectively reduces uncertainty in test statistic selection.
4. **Unbiasedness verified**: Log perplexity is consistent with the non-watermarked baseline, confirming no degradation in output quality.

## Highlights & Insights

- This is the first work to propose a continuous and meaningful quantitative measure of watermark strength, directly connecting it to p-value decay rates and sample complexity.
- The complete Pareto trade-off curve is characterized, transforming empirical observations into a rigorous constrained optimization problem.
- The pseudo-random acceptance mechanism is remarkably elegant — a minimal modification (replacing the true random coin with a pseudo-random one) simultaneously breaks the efficiency and strength bounds.
- The theoretical analysis is thorough and complete (simultaneously proving unbiasedness, maximum efficiency, and maximum strength), and experimental results closely match theoretical predictions.

## Limitations & Future Work

- The approach directly applies to unbiased degenerate watermarks (Gumbel-max, SynthID); extensions to non-degenerate and biased watermarks remain open problems.
- Detection requires training data (1,000 watermarked texts), and SynthID additionally requires non-watermarked texts as negative samples, increasing deployment complexity.
- Only standard speculative sampling is evaluated; extensions to tree-based variants and other alternatives are not verified.
- The effect of human editing on watermarks is not explored, whereas robustness to post-hoc editing is an important consideration in practical deployment.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Reproducibility | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment](../../ACL2026/llm_safety/llm-va_resolving_the_jailbreak-overrefusal_trade-off_via_vector_alignment.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[AAAI 2026\] Democratizing LLM Efficiency: From Hyperscale Optimizations to Universal Deployability](../../AAAI2026/llm_safety/democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)
- [\[ICLR 2026\] Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?](do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)

</div>

<!-- RELATED:END -->
