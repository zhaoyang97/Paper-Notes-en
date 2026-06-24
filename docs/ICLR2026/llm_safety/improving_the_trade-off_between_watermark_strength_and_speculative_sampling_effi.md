---
title: >-
  [Paper Note] Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models
description: >-
  [ICLR 2026][LLM Safety][LLM Watermarking] This paper proposes a quantitative measure of watermark strength (expected KL divergence) and fully characterizes its Pareto trade-off curve with speculative sampling efficiency. It further achieves simultaneous maximum watermark strength and optimal sampling efficiency by pseudo-randomizing the acceptance decisions.
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "LLM Watermarking"
  - "Speculative Sampling"
  - "Watermark Strength"
  - "Sampling Efficiency"
  - "Pareto Frontier"
  - "Pseudo-random Acceptance"
date: 2026-05-08
content_hash: d473205f4c2b415c
---

# Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models

**Conference**: ICLR 2026  
**arXiv**: [2602.01428](https://arxiv.org/abs/2602.01428)  
**Code**: [GitHub](https://github.com/hwq0726/watermark-tradeoff)  
**Area**: AI Safety  
**Keywords**: LLM Watermarking, Speculative Sampling, Watermark Strength, Sampling Efficiency, Pareto Frontier, Pseudo-random Acceptance  

## TL;DR

This paper proposes a quantitative measure of watermark strength (expected KL divergence) and fully characterizes its Pareto trade-off curve with speculative sampling efficiency. It further achieves simultaneous maximum watermark strength and optimal sampling efficiency by pseudo-randomizing the acceptance decisions.

## Background & Motivation

1. **LLM watermarking is a key technology for provenance**: Watermarking implements text traceability by injecting recoverable pseudo-random signals during the token sampling process, serving as a principled method to ensure content trackability.

2. **Speculative sampling accelerates LLM inference**: Speculative sampling uses a lightweight draft model to quickly generate candidate tokens followed by parallel verification by a target model. Higher acceptance rates lead to more significant speedups. Efficiency is measured by the expected acceptance rate: $\text{SE} = \mathbb{E}_\zeta[\sum_w \mathcal{A}_\zeta(w|w) Q_{\zeta,w}]$.

3. **Fundamental trade-off between watermarking and speculative sampling**: Hu & Huang (2024) proved the impossibility of simultaneously maintaining the highest acceptance rate and the strongest watermark—stronger watermarks cause the draft/target distributions to deviate more, thereby reducing the acceptance rate. This "impossibility" result is discouraging.

4. **Limitations of Prior Work in defining watermark strength**: Existing definitions are binary (a watermark is only "maintained" if it exactly matches the watermarked distribution), ignoring intermediate strength levels and preventing fine-grained trade-off analysis. This paper requires a continuous, quantifiable watermark strength measure to fully characterize the trade-off curve.

## Method

### Overall Architecture

This paper advances the trade-off problem between watermarking and speculative sampling through a three-step approach: "Quantify—Characterize—Break." It first uses expected KL divergence to transform the previously binary "watermark strength" into a continuous measurable quantity. Next, it formulates the optimal trade-off between strength and sampling efficiency as a Pareto curve. Finally, it identifies that the curve exists because of residual true randomness in the acceptance/rejection step; by pseudo-randomizing this "coin flip," the entire generation process becomes a deterministic function of pseudo-random numbers, allowing for simultaneous maximum strength and optimal efficiency.

### Key Designs

**1. Expected KL Divergence Measure: Quantifying "Watermark Strength" as a Continuous Number**

Previously, determining whether a watermark scheme "maintained" strength was binary—only exact matches counted, leaving intermediate strengths undefined and trade-off curves undrawable. This paper instead defines strength as the expected KL divergence between the watermarked distribution $\bm{P}_\zeta$ and the original distribution $\bm{P}$: $\text{WS}(\bm{P}_\zeta) = \mathbb{E}_\zeta[D_{\mathrm{KL}}(\bm{P}_\zeta \| \bm{P})] = \text{Ent}(\bm{P}) - \mathbb{E}_\zeta[\text{Ent}(\bm{P}_\zeta)]$. This is exactly equal to the mutual information $I(w;\zeta)$ between the token $w$ and the pseudo-random number $\zeta$, representing "how much pseudo-random signal can be inferred from the output." This metric is useful because it also characterizes detection difficulty: under a likelihood ratio test, the p-value for $n$ tokens satisfies $\lim_{n\to\infty}-\frac{1}{n}\log(\text{p-value})=\bar{D}$. Greater strength leads to faster p-value decay and fewer samples required for detection. Its upper bound is the entropy of the original distribution $\text{WS}(\bm{P}_\zeta)\leq\text{Ent}(\bm{P})$, achieved only when $\bm{P}_\zeta$ is degenerate (probability concentrated on a single token); both Gumbel-max and SynthID (as $m\to\infty$) can reach this bound.

**2. Pareto Trade-off Curve: Formulating Empirical Observations as a Convex Optimization Problem**

While Hu & Huang (2024) proved the impossibility of simultaneously maintaining max efficiency and strength, it was a qualitative result. This paper explicitly formulates the trade-off by seeking the maximum reachable watermark strength given an efficiency requirement $r$: $L(r)=\max_{\mathcal{S}_{\text{draft}},\mathcal{S}_{\text{target}}}\text{WS}(\bm{P}_\zeta)\ \text{s.t.}\ \text{SSE}(\bm{Q}_\zeta,\bm{P}_\zeta)\geq r$. For the linear watermark class $\mathcal{Q}=\{(1-\theta)\text{Id}+\theta\mathcal{S}:\theta\in[0,1]\}$, this inverse curve can be further reduced to a convex optimization: minimizing the $\ell_1$ norm of the corresponding total variation distance subject to an entropy threshold. This allows the complete Pareto frontiers of Gumbel-max and SynthID to be accurately plotted, providing a computable boundary for the intuition that "strong watermarking inevitably slows sampling."

**3. Pseudo-random Acceptance Mechanism: Fixing the Final Truly Random Coin**

Even if the pseudo-random numbers for the draft and target models are known, standard speculative sampling still uses a **truly random** coin to accept or reject a draft token. Consequently, the output tokens carry residual randomness, which consumes watermark strength. This paper introduces one key change: pseudo-randomizing the acceptance variable $u_t=G(\zeta_t^R)$, making the entire generation process a deterministic function of pseudo-random variables. Theorem 4.1 proves this modification simultaneously achieves three goals: unbiasedness $\mathbb{E}_\zeta[\bm{P}'_\zeta(w)]=\bm{P}(w)$, maximum sampling efficiency $\text{SE}=1-\text{TV}(\bm{Q},\bm{P})$, and maximum watermark strength $\text{WS}(\bm{P}'_\zeta)=\text{Ent}(\bm{P})$, effectively bypassing the constraints of the Pareto curve. On the detection side, since $u_t$ is now known, more accurate test statistics can be chosen: the Ars-τ detector is designed for Gumbel-max, using a threshold $\tau$ to switch between draft/target statistics; for SynthID, a Bayes-MLP detector is designed, replacing simple weighted averages with an MLP to select statistics.

## Key Experimental Results

### Experimental Setup

- **Model Pairs**: Llama-68M (draft) + Llama-7B (target); Gemma-2B (draft) + Gemma-7B (target)
- **Datasets**: ELI5 (QA), C4 (open generation)
- **Watermarking Schemes**: Gumbel-max (Temp 0.5), SynthID (m=30, Temp 0.7)
- **Lookahead**: K ∈ {2, 3, 4}
- **Metrics**: AATPS (Average Accepted Tokens Per Step), TPR@FPR=1%

### Main Results: Sampling Efficiency and Detection Performance

| Method | Watermark Scheme | AATPS (K=4) | TPR@100 tokens | TPR@200 tokens |
|---|---|---|---|---|
| Std. SpecSampl | No Watermark | ~3.2 | - | - |
| Ars-Prior | Gumbel-max | ~3.2 | ~65% | ~88% |
| **Ars-τ (Ours)** | **Gumbel-max** | **~3.2** | **~78%** | **~95%** |
| Bayes-Prior | SynthID | ~3.2 | ~50% | ~75% |
| **Bayes-MLP (Ours)** | **SynthID** | **~3.2** | **~62%** | **~85%** |
| Oracle (Theoretical Upper Bound) | Both | ~3.2 | ~85% | ~98% |

### Ablation Study: Trade-off Curve Verification

| Efficiency Requirement (r) | Gumbel-max WS | SynthID WS (m=30) | SynthID WS (m=∞) |
|---|---|---|---|
| Max Efficiency | 0 | 0 | 0 |
| 0.9 × Max Efficiency | Medium | Medium-Low | Medium |
| 0.7 × Max Efficiency | High | Medium | High |
| No Efficiency Constraint | Max = Ent(P) | < Ent(P) | Max = Ent(P) |

Experiments verify that Gumbel-max and SynthID (m=∞) reach the same maximum watermark strength, but SynthID with finite $m$ has lower strength than Gumbel-max.

### Key Findings

1. **Efficiency Perfectly Maintained**: Under the pseudo-random acceptance mechanism, AATPS is almost identical to standard speculative sampling, confirming the maximum efficiency property of Theorem 4.1.
2. **Significant Detection Gain**: At the same token count, Ars-τ improves TPR by approximately 10-15 percentage points over Ars-Prior, and Bayes-MLP improves by 10-12 percentage points over Bayes-Prior.
3. **Approaching Oracle Bound**: At 200 tokens, the proposed method approaches the performance of the ideal Oracle detector, indicating that the pseudo-random acceptance variable effectively reduces uncertainty in test statistic selection.
4. **Unbiasedness Verification**: Log Perplexity remains consistent with the non-watermarked version, confirming no reduction in output quality.

## Highlights & Insights

- First to propose a continuous and meaningful quantitative measure for watermark strength, directly linking it to p-value decay rates and sample complexity.
- Fully characterizes the Pareto trade-off curve, transforming empirical observations into a rigorous constrained optimization problem.
- The design of the pseudo-random acceptance mechanism is remarkably elegant—a tiny change (replacing a true random coin with a pseudo-random one) simultaneously breaks the bounds of efficiency and strength.
- Theoretical analysis is deep and complete (simultaneous proof of unbiasedness, max efficiency, and max strength), with experimental validation highly consistent with theoretical predictions.

## Limitations & Future Work

- Directly applicable to unbiased degenerate watermarks (Gumbel-max, SynthID); extensions to non-degenerate or biased watermarks remain an open problem.
- Detection requires training data (1000 watermarked texts), and SynthID further requires non-watermarked text as negative samples, increasing deployment complexity.
- Only evaluated on standard speculative sampling; extensions to variants like tree-based sampling are not yet verified.
- The impact of manual editing on watermarks was not explored; watermark robustness is a major consideration for actual deployment.

## Rating

| Dimension | Rating |
|---|---|
| Novelty | ⭐⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Reproducibility | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs](managerbench_evaluating_the_safety-pragmatism_trade-off_in_autonomous_llms.md)
- [\[ACL 2025\] From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](../../ACL2025/llm_safety/from_tradeoff_to_synergy_a_versatile.md)
- [\[ICLR 2026\] Sampling-aware Adversarial Attacks against Large Language Models](sampling-aware_adversarial_attacks_against_large_language_models.md)
- [\[ACL 2026\] LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment](../../ACL2026/llm_safety/llm-va_resolving_the_jailbreak-overrefusal_trade-off_via_vector_alignment.md)
- [\[ICLR 2026\] CodeGenGuard: A Watermark for Code Generation Models](codegenguard_a_watermark_for_code_generation_models.md)

</div>

<!-- RELATED:END -->
