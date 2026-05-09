---
title: >-
  [Paper Note] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs
description: >-
  [NeurIPS 2025][LLM Safety][information leakage] This paper models adversarial attacks on LLMs as an information channel problem — defining the "bits leaked per query" $I(Z;T)$ as the mutual information between the attack target attribute $T$ and the observable signal $Z$, and proving that the minimum number of queries required to achieve error $\varepsilon$ is $\log(1/\varepsilon)/I(Z;T)$. Validated across 7 LLMs: exposing only answer tokens requires ~1000 queries; adding logits reduces this to ~100; adding chain-of-thought (CoT) further reduces it to ~tens of queries. This provides the first principled metric for the transparency–security trade-off.
tags:
  - NeurIPS 2025
  - LLM Safety
  - information leakage
  - adversarial attacks
  - LLM security
  - query complexity lower bounds
  - information channel model
date: 2026-05-08
content_hash: 203275b260b051f2
---

# Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2510.17000](https://arxiv.org/abs/2510.17000)
**Code**: TBD
**Area**: AI Safety / Information Theory
**Keywords**: information leakage, adversarial attacks, LLM security, query complexity lower bounds, information channel model

## TL;DR
This paper models adversarial attacks on LLMs as an information channel problem — defining the "bits leaked per query" $I(Z;T)$ as the mutual information between the attack target attribute $T$ and the observable signal $Z$, and proving that the minimum number of queries required to achieve error $\varepsilon$ is $\log(1/\varepsilon)/I(Z;T)$. Validated across 7 LLMs: exposing only answer tokens requires ~1000 queries; adding logits reduces this to ~100; adding chain-of-thought (CoT) further reduces it to ~tens of queries. This provides the first principled metric for the transparency–security trade-off.

## Background & Motivation

**Background**: LLMs are subject to a variety of adversarial attacks — system prompt extraction, jailbreaking, and post-unlearning re-learning attacks. For transparency, many LLM services expose chain-of-thought (CoT) reasoning or token probabilities (logits), but these signals can potentially be exploited by adversaries.

**Limitations of Prior Work**:
   - Attack evaluation is purely empirical — plotting "success rate vs. query count" curves without a theoretical optimum for reference
   - Defenders cannot quantify "how much additional security risk does exposing CoT introduce compared to exposing only answer tokens"
   - Attackers cannot determine how far their methods are from the theoretical limit

**Key Challenge**: There is a fundamental trade-off between transparency (exposing more signals improves interpretability and user experience) and security (exposing more signals accelerates attacks), yet no theoretical framework exists to quantify this trade-off.

**Goal**: Establish theoretical lower bounds on "what the strongest attacker can achieve given a fixed information leakage rate," enabling defenders to principally control the level of transparency.

**Key Insight**: Model each LLM query as an information channel (input = adversarial prompt, output = observable signal, latent variable = attack target attribute), quantify per-query leakage via mutual information, and derive information-theoretic lower bounds on query complexity.

**Core Idea**: LLM security = information channel problem; query complexity satisfies $N \geq \log(1/\varepsilon) / I(Z;T)$.

## Method

### Overall Architecture
Formalization: the attacker issues prompt $X_n$ (adaptively conditioned on the previous $n-1$ responses), the LLM returns observable signal $Z_n$ (answer tokens / logits / CoT), and the attacker attempts to infer the target attribute $T$ (e.g., a binary flag indicating whether a jailbreak succeeded). The per-query leakage is defined as $I_{max} = \sup_x I(Z; T|X=x)$.

### Key Designs

1. **Information-Theoretic Lower Bound (Theorem 1)**:

    - Function: Proves an unconditional lower bound on the minimum number of queries required for any attack.
    - Mechanism: Applying the data processing inequality, the total information gained by the attacker after $N$ adaptive queries satisfies $I(Z_{1:N}; T) \leq N \cdot I_{max}$. To reduce the uncertainty of $T$ from $H(T)$ to $\varepsilon$, it follows that $N \geq \log(1/\varepsilon) / I_{max}$.
    - Design Motivation: This lower bound holds for **any** attack strategy — including adaptive strategies and methods yet to be invented.

2. **Phase Transition Phenomenon**:

    - Function: Reveals a sharp transition in query complexity as information leakage goes from zero to nonzero.
    - Mechanism: When $I(Z;T) \approx 0$ (near-zero leakage), query requirements scale as $O(1/\varepsilon)$ (linear/quadratic growth); once leakage reaches a fixed constant number of bits (e.g., by exposing logits), query requirements drop abruptly to $O(\log(1/\varepsilon))$ — a polynomial-to-logarithmic phase transition.
    - Design Motivation: Explains why "exposing just a little more information" can cause a precipitous drop in attack difficulty.

3. **Experimental Comparison Across Four Signal Regimes**:

    - Function: Quantifies actual leakage and attack efficiency under different transparency levels.
    - Mechanism: Four regimes are considered — (a) answer tokens only; (b) tokens + logits; (c) tokens + CoT; (d) tokens + CoT + logits. The per-regime $I(Z;T)$ and empirical query count $N$ are measured.
    - Design Motivation: Provides directly actionable guidance for LLM deployers — "if you expose X, attack cost drops to Y."

### Loss & Training
- No training is required — the work is purely theoretical combined with empirical analysis.
- Theoretical component: derived via Fano's inequality and the data processing inequality.
- Experimental component: 3 attack methods (paraphrase, GCG, PAIR) × 3 scenarios × 7 LLMs × 4 signal regimes.

## Key Experimental Results

### Main Results
Scatter plot fitting of $\log N$ vs. $\log I(Z;T)$ across 7 LLMs and 3 attack scenarios:

| Signal Regime | Per-query leakage $I(Z;T)$ (bits) | Queries required $N$ | Relative risk |
|---|---|---|---|
| Answer tokens only | ~0.001 | ~1000 | Baseline (safest) |
| Tokens + logits | ~0.01 | ~100 | 10× risk |
| Tokens + CoT | ~0.05 | ~tens | 50× risk |
| Tokens + CoT + logits | ~0.1 | ~a few | 100× risk |

### Ablation Study: Theory–Experiment Alignment

| Verification | Result |
|---|---|
| Slope of $\log N$ vs. $\log I$ | ~−1 (theoretical prediction: −1; experimental fit is highly consistent) |
| Pearson correlation | Significantly negative ($p < 0.001$) |
| Cross-model consistency | Consistent across GPT-4, DeepSeek-R1, OLMo-2, and Llama-4 |

### Key Findings
- **Doubling leakage halves query requirements**: Experiments perfectly validate the theoretical relationship $N \propto 1/I(Z;T)$.
- **CoT is the greatest security cost**: Exposing chain-of-thought leaks more information than exposing logits, as CoT contains explicit signals indicating whether the model is near the boundary of refusal.
- **No existing attack reaches the theoretical lower bound**: This indicates that current attacks still have room for improvement — bad news for attackers, good news for defenders.
- **Quantitative guidance for rate limiting**: If $I(Z;T) = 0.01$ bit/query, achieving a 99% attack success rate requires ~700 queries — a rate limit of 500 queries/hour is sufficient for defense.

## Highlights & Insights
- The conceptualization of **"LLM security as an information channel problem"** is elegant — it reduces a complex security problem to fundamental laws of information theory, yielding unconditional lower bounds.
- The **phase transition phenomenon** carries a profound implication: it is not that "a small leak is acceptable," but rather that "once leakage is nonzero, query complexity drops from polynomial to logarithmic" — security degradation is a cliff, not a slope.
- **Provides an actionable metric for API design**: deployers can compute "exposing logits reduces attack cost by 10×; is it worth it?"
- The framework is general — applicable to any attack scenario, not limited to jailbreaking.

## Limitations & Future Work
- The analysis assumes $(Z, T)$ are i.i.d. across queries; in practice, adaptive strategies by the attacker may violate this assumption (though the information-theoretic lower bound still holds).
- Empirical estimation of $I(Z;T)$ requires large sample sizes and may be inaccurate for novel attack types.
- The paper does not consider server-side strategies of dynamically adjusting transparency (e.g., exposing CoT only for low-risk queries).
- The theoretical model assumes the channel between attacker and model is deterministic given $X$ and $T$; in practice, models exhibit sampling stochasticity.

## Related Work & Insights
- **vs. GCG (Zou et al.)**: An empirical attack method. This paper provides a theoretical upper bound on GCG's efficiency — GCG is far from the theoretical limit.
- **vs. PAIR (Chao et al.)**: An adaptive attack. The lower bounds in this paper apply to all adaptive attacks, including PAIR.
- **vs. information-theoretic privacy literature**: Transferring ideas from differential privacy and channel capacity to LLM security constitutes an interdisciplinary theoretical contribution.
- Has direct implications for API design: the security cost of exposing logits can be quantitatively computed to inform deployment decisions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical framework modeling LLM security as an information channel problem is entirely novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 LLMs, 3 attack scenarios, 4 signal regimes, and strong theory–experiment alignment.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations; the presentation of the phase transition phenomenon is particularly impressive.
- Value: ⭐⭐⭐⭐⭐ Has far-reaching implications for LLM security theory and the design of transparent APIs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[NeurIPS 2025\] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)

</div>

<!-- RELATED:END -->
