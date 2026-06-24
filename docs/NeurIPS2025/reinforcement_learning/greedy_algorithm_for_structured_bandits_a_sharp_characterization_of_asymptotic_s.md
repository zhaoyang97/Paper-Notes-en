---
title: >-
  [Paper Note] Greedy Algorithm for Structured Bandits: A Sharp Characterization of Asymptotic Success / Failure
description: >-
  [NeurIPS 2025][Reinforcement Learning][structured bandits] This paper provides a complete theoretical characterization of the greedy algorithm in structured bandit problems, proposing *self-identifiability* as a necessary and sufficient condition for the greedy algorithm to achieve sublinear regret, and extends the results to contextual bandits and the general interactive decision-making framework DMSO.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "structured bandits"
  - "greedy algorithm"
  - "regret"
  - "self-identifiability"
  - "contextual bandits"
date: 2026-05-08
content_hash: d7ebe1da155ae295
---

# Greedy Algorithm for Structured Bandits: A Sharp Characterization of Asymptotic Success / Failure

**Conference**: NeurIPS 2025
**arXiv**: [2503.04010](https://arxiv.org/abs/2503.04010)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: structured bandits, greedy algorithm, regret, self-identifiability, contextual bandits

## TL;DR
This paper provides a complete theoretical characterization of the greedy algorithm in structured bandit problems, proposing *self-identifiability* as a necessary and sufficient condition for the greedy algorithm to achieve sublinear regret, and extends the results to contextual bandits and the general interactive decision-making framework DMSO.

## Background & Motivation

**Background**: Multi-armed bandits are a foundational framework for sequential decision-making. Classical theory emphasizes that exploration is a necessary means of achieving low regret. Structured bandits introduce known reward structures (e.g., linear, Lipschitz) to reduce the need for exploration via structural constraints.

**Limitations of Prior Work**: Despite the maturity of exploration theory, pure greedy algorithms (Greedy — exploit only, no exploration) are widely used in practice, since exploration incurs high costs, raises fairness concerns, and may violate user incentives in human-interactive systems. However, a unified theoretical characterization of when greedy succeeds and when it fails has been lacking.

**Key Challenge**: Prior work only provided success/failure examples of greedy on a handful of specific structures (e.g., linear contextual bandits requiring context diversity), without a general theory applicable to arbitrary finite reward structures.

**Goal**: For any finite reward structure, provide a complete "if and only if" characterization of when the greedy algorithm asymptotically succeeds (sublinear regret) vs. fails (linear regret).

**Key Insight**: The authors identify *partial identifiability* — self-identifiability — as the key: if fixing the expected reward of a suboptimal arm suffices to identify it as suboptimal, then greedy succeeds; otherwise, a decoy exists that permanently traps greedy.

**Core Idea**: Self-identifiability is a necessary and sufficient condition for the greedy algorithm to achieve sublinear regret in structured bandits.

## Method

### Overall Architecture

Consider a structured bandit with finite action set $\mathcal{A}$, finite context set $\mathcal{X}$, and finite reward function class $\mathcal{F}$. At each round $t$, a context $x_t$ arrives, the algorithm selects arm $a_t$, and receives reward $r_t \sim \mathcal{N}(f^*(x_t,a_t), 1)$. The greedy algorithm performs least-squares regression at each round and selects the empirically optimal arm:

$$f_t = \arg\min_{f \in \mathcal{F}} \sum_{s \in [t]} (f(x_s,a_s) - r_s)^2, \quad a_t = \arg\max_{a} f_t(x_t,a)$$

### Key Designs

1. **Self-Identifiability**:

    - **Function**: Characterizes the necessary and sufficient condition for greedy to succeed.
    - **Mechanism**: For each suboptimal arm $a$ of $f^*$, if every $f \in \mathcal{F}$ satisfying $f(a) = f^*(a)$ also deems $a$ suboptimal, then $a$ is said to be self-identifiable. If all suboptimal arms have this property, the instance is self-identifiable.
    - **Design Motivation**: Greedy only observes rewards from the arms it selects. If accurately estimating an arm's expected reward is sufficient to conclude it is not optimal, greedy will naturally "escape" from it.

2. **Decoy**:

    - **Function**: Characterizes the mechanism of greedy failure.
    - **Mechanism**: $f_{\text{dec}} \in \mathcal{F}$ is a decoy for $f^*$ if its optimal arm $a_{\text{dec}}$ is suboptimal under $f^*$, and $f_{\text{dec}}(a_{\text{dec}}) = f^*(a_{\text{dec}})$.
    - **Core Equivalence**: $f^*$ has no decoy $\Leftrightarrow$ the problem instance is self-identifiable.

3. **Function-Gap Parameterization**:

    - Define $\Gamma(f^*, \mathcal{F}) = \min_{f \neq f^*} \min_{a: f(a) \neq f^*(a)} |f^*(a) - f(a)|$.
    - Both the regret upper bounds in positive results and the failure probability lower bounds in negative results are parameterized by $\Gamma$.

### Core Theorems

**Theorem 3.3 (StructuredMAB)**: (a) If self-identifiable, then $\mathbb{E}[R(t)] \leq T_0 + (K/\Gamma)^2 \cdot O(\log t)$; (b) If a decoy exists, with probability $\geq e^{-O(K/\Gamma^2)}$, greedy selects the decoy arm forever.

**Theorem 4.3 (StructuredCB)**: Extends to contextual bandits with regret upper bound $(|\mathcal{X}|K/\Gamma)^2/p_0 \cdot O(\log t)$.

**Theorem 5.3 (DMSO)**: MLE-based greedy with KL model-gap achieves the same necessary and sufficient characterization.

### Proof Sketch

**Positive direction**: After a suboptimal arm is selected $O(K/\Gamma^2)$ times, its empirical mean concentrates near the true value with high probability; self-identifiability then identifies it as suboptimal, and an MSE argument guarantees it is never selected again.

**Negative direction**: Two independent events $E_1$ (misleading warm-up) and $E_2$ (decoy arm mean stays close to true value forever) are constructed; $E_1 \cap E_2$ guarantees permanent entrapment.

## Key Experimental Results

### Summary of Positive / Negative Instances

| Reward Structure | Greedy Performance | Explanation |
|---|---|---|
| Linear bandit | Fails on almost all instances | Decoy exists |
| Lipschitz bandit | Fails on almost all instances | Decoy exists |
| Polynomial bandit | Fails on almost all instances | Decoy exists |
| Linear contextual + diverse contexts | Succeeds | Self-identifiability holds |
| Linear contextual + low-dimensional contexts | May fail | Lack of diversity |
| Lipschitz contextual | Fails on almost all instances | Qualitatively different from linear contextual |

### Extension to Infinite Function Classes (Theorem 6.2)

| Setting | Positive Result | Negative Result |
|---|---|---|
| $\varepsilon$-self-identifiable | Logarithmic regret $(K/\varepsilon)^2 O(\log t)$ | — |
| Decoy in interior | — | Probability $\geq e^{-O(K^2/\varepsilon^2)}$ of permanent entrapment |

### Key Findings
- Greedy failure is the norm for most continuous reward structures.
- Self-identifiability not only enables greedy to succeed, but also implies that all mildly non-degenerate algorithms succeed — the problem itself is intrinsically easy.
- A complete iff characterization for infinite function classes remains an open problem.

## Highlights & Insights
- **Self-identifiability concept**: Formalizes "when exploration is unnecessary" as a novel structural notion.
- **Decoy mechanism**: Reveals a unified mechanism of greedy failure — structural permanent entrapment.
- **DMSO generalization**: Via MLE and KL divergence, the results extend to complex settings such as reinforcement learning.
- **Practical guidance**: The empirical success of greedy in large-scale systems can be explained by self-identifiability.

## Limitations & Future Work
- The constant $(K/\Gamma)^2$ can be large, leading to slow convergence in practice.
- The failure probability $p_{\text{dec}}$ may be small.
- A complete iff characterization for infinite function classes remains open.
- Computational efficiency is not guaranteed.

## Related Work & Insights
- **vs. Bastani et al. (2021)**: Their work achieves near-optimal regret for linear contextual bandits with diverse contexts; the present paper is more general but yields weaker constants.
- **vs. Banihashem et al. (2023)**: Their work characterizes greedy failure probability in unstructured $K$-armed bandits; the present paper extends to arbitrary reward structures.
- **vs. Foster et al. (2021)**: This paper provides the first characterization of greedy within the DMSO framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First complete iff characterization of greedy for arbitrary finite reward structures
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work; application instances are analyzed in detail
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts are clear, theorems are precise, and proofs build progressively
- Value: ⭐⭐⭐⭐⭐ Resolves a long-standing fundamental open problem in bandit theory

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)

</div>

<!-- RELATED:END -->
