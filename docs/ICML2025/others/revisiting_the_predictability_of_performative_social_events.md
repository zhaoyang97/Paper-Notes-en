---
title: >-
  [Paper Note] Revisiting the Predictability of Performative, Social Events
description: >-
  [ICML2025][performative prediction] This paper leverages modern learning theory tools (performative prediction + outcome indistinguishability) to answer a classic 20th-century question in social science: Can social events still be accurately predicted when predictions actively influence outcomes? The answer is affirmative—yet such "accurate" predictions can be entirely useless.
tags:
  - "ICML2025"
  - "performative prediction"
  - "multicalibration"
  - "outcome indistinguishability"
  - "social prediction"
  - "online learning"
date: 2026-05-08
content_hash: 5b4b8ffe654d93d5
---

# Revisiting the Predictability of Performative, Social Events

**Conference**: ICML2025  
**arXiv**: [2503.11713](https://arxiv.org/abs/2503.11713)  
**Code**: None (theoretical work)  
**Area**: Social Prediction / Performative Prediction / Learning Theory  
**Keywords**: performative prediction, multicalibration, outcome indistinguishability, social prediction, online learning

## TL;DR

This paper leverages modern learning theory tools (performative prediction + outcome indistinguishability) to answer a classic 20th-century question in social science: Can social events still be accurately predicted when predictions actively influence outcomes? The answer is affirmative—yet such "accurate" predictions can be entirely useless.

## Background & Motivation

Social prediction, by nature, does not passively describe the future but actively shapes it: economic forecasts affect market prices, election predictions impact voter turnout, and climate projections influence policy. This dynamic where "predictions affect data" is known as **performativity**.

As early as the 20th century, several scholars posed core questions:

- **Morgenstern (1928)**: Economic forecasting is generally impossible to make accurate because public predictions can be self-defeating.
- **Simon (1954), Grunberg & Modigliani (1954)**: Proved the existence of "self-fulfilling predictions" using topological fixed-point theorems, but left the algorithmic problem unresolved.
- **Lucas critique (1976)**: A turning point in macroeconomic theory.

Using modern tools such as the **performative prediction** framework (Perdomo et al., 2020) and **multicalibration / outcome indistinguishability** (Hébert-Johnson et al., 2018; Dwork et al., 2021), this paper revisits these questions and provides a complete algorithmic resolution.

## Method

### Core Formulation: Outcome Performative Distribution Map

After a predictor $f$ is deployed, the data generation process is:

$$x \sim \mathcal{D}_x, \quad p \sim f(x), \quad y \sim \mathcal{D}_y(x, p)$$

Here, the distribution of features $x$ is fixed, but the conditional distribution of outcomes $y$, denoted by $\mathcal{D}_y(x,p)$, depends on the prediction $p$. This precisely captures the predictive dynamics in domains like healthcare and education.

### Core Definition: Performative Multicalibration

A predictor $f$ is $\varepsilon$-performatively multicalibrated (equivalent to outcome indistinguishable) if, for all $c \in \mathcal{C}$:

$$\left| \mathbb{E}_{\substack{x \sim \mathcal{D}_x, p \sim f(x) \\ y \sim \mathcal{D}_y(x,p)}} [c(x,p)(y-p)] \right| \leq \varepsilon$$

When $\mathcal{C}$ contains only constant functions, this degenerates to the condition of Simon (1954); when $\mathcal{C}$ consists of all bounded measurable functions, it requires $f(x) = \mathbb{E}_{\mathcal{D}(f)}[y|x]$.

### Core Algorithm: Online-to-Batch Reduction

**Key Idea**: Reduce the performative multicalibration problem to a harder online problem, and then perform an online-to-batch conversion.

**Steps**:

1. Use an online algorithm $\mathcal{A}$ to produce a deterministic prediction function $f_t$ at each round $t$.
2. Nature samples $(x_t, y_t) \sim \mathcal{D}(f_t)$ from the distribution map.
3. After $n$ rounds, construct the batch predictor $f_{\mathcal{A}}$: for a given $x$, uniformly sample $f_i$ from $\{f_1, \ldots, f_n\}$ at random and predict $p = f_i(x)$.

**Main Theorem (Theorem 3.4)**: If $\mathcal{A}$ guarantees an online multicalibration regret of $\mathsf{Regret}_{\mathcal{A}}(T)$, then the batch version satisfies with probability $1-\delta$:

$$\left| \mathbb{E}_{\mathcal{D}(f_{\mathcal{A}})} [c(x,p)(p-y)] \right| \leq \frac{\mathsf{Regret}_{\mathcal{A}}(n)}{n} + 4\sqrt{\frac{\log|\mathcal{C}| + \log(1/\delta)}{n}}$$

Proof core: The transcript is treated as a stochastic process, and a high-probability upper bound is established via a Martingale argument and the Azuma-Hoeffding inequality.

### Concrete Instantiation (Corollary 3.5)

Instantiating the reduction using the K29 kernel-based algorithm (Vovk et al., 2005):

| Function Class $\mathcal{C}$ | Regret Rate | Batch Error | Running Time |
|---|---|---|---|
| Any finite set (continuous in $p$) | $\sqrt{T \cdot |\mathcal{C}|}$ | $\sqrt{|\mathcal{C}|/n}$ | $O(n^2 |\mathcal{C}|)$ |
| Linear functions $\theta^\top x + p$ | $\sqrt{2T}$ | $\sqrt{2/n}$ | $O(n^2 d)$ |
| Low-degree Boolean functions (degree $s$) | $10\sqrt{d^s \cdot T}$ | $10\sqrt{d^s/n}$ | $O(ds \cdot n^2)$ |

### Structural Results: Multicalibration → Stability

**Theorem 4.2**: For a binary outcome $y$ and squared loss, if $f$ is $\varepsilon$-performatively multicalibrated w.r.t. $\mathcal{C} = \{p - 1/2\} \cup \{h(x) - 1/2 : h \in \mathcal{H}\}$, then $f$ is $2\varepsilon$-performatively stable:

$$\mathbb{E}_{\mathcal{D}(f)} (y-p)^2 \leq \min_{h \in \mathcal{H}} \mathbb{E}_{\mathcal{D}(f)} (y - h(x))^2$$

### Negative Results: Stability ≠ Optimality

**Theorem 5.1 (Core Counterexample)**: There exists a distribution map $\mathcal{D}(\cdot)$ such that predictor $f$ can be performatively multicalibrated with respect to all bounded continuous functions $c(x,p)$, yet simultaneously **maximizes** the performative risk:

$$\mathbb{E}_{\mathcal{D}(f)} (p-y)^2 \geq \max_{h \in \mathcal{H}} \mathbb{E}_{\mathcal{D}(h)} (y - h(x))^2$$

**Construction**: Featureless setting, with $g(p) = p + 0.01$ if $p \leq 0.5$, and $g(p) = p - 0.01$ if $p > 0.5$. No deterministic fixed point satisfying $g(p) = p$ exists. The only calibrated randomized predictor randomizes between $1/2$ and $1/2 + \varepsilon$, but this causes $y$ to behave like a fair coin, maximizing the variance ($1/4$), whereas the optimal solution predicts 0 or 1 with a risk of only 0.01.

## Key Experimental Results

This paper is a **purely theoretical work** with no experimental data. The core contributions are presented through mathematical theorems and constructive counterexamples:

| Contribution | Content | Conditions |
|---|---|---|
| Feasibility | Can always efficiently find a performatively multicalibrated predictor | No smoothness assumptions on $\mathcal{D}(\cdot)$ required |
| Convergence Rate | $O(n^{-1/2})$, identical to supervised learning | Only requires bounded outcomes |
| Stability | Multicalibration implies performative stability | Squared loss + binary outcomes |
| Inevitable Failure | Perfect calibration can maximize performative risk | Discontinuous distribution maps |
| Reversal of Supervised Learning Intuition | Conditional expectation optimality is completely reversed under the performative setting | — |

## Highlights & Insights

1. **Bridging a 70-Year Theoretical Gap**: Moving from the existential questions posed by Morgenstern (1928) and Simon (1954) to the complete algorithmic solution provided in this paper, spanning nearly a century.
2. **No Smoothness Assumptions Required**: Unlike almost all prior work in the performative prediction literature which requires $\mathcal{D}(\cdot)$ to be Lipschitz continuous with respect to the predictions, this work only requires bounded outcomes, covering common threshold-based decision-making scenarios in fields like education.
3. **Profound Conceptual Insight**: "Accurate $\neq$ Useful"—under the performative setting, a perfectly calibrated predictor may fail to explain any variance in outcomes, completely contradicting the intuition in supervised learning.
4. **Elegant Technical Approach**: Simplifying complex performative problems into standard online learning problems (for which rich algorithms exist) via an online-to-batch reduction.
5. **Necessity of Randomization**: Under discontinuous distribution maps, deterministic predictors may fail to achieve self-consistency, making randomized predictors indispensable.

## Limitations & Future Work

1. **Limited to Outcome Performativity**: Assumes predictions only affect the distribution of outcomes $y$ but not features $x$. In real-world scenarios, individuals might alter their behavioral features in response to predictions.
2. **Stateless Setting**: Does not consider the cumulative impact of historical predictions on current outcomes (stateful performativity).
3. **Restriction to Binary Outcomes**: The structural result (Theorem 4.2) is limited to $y \in \{0,1\}$, and while the authors note that it can be generalized, they do not expand on it.
4. **Lack of Empirical Validation**: A purely theoretical framework that has not been validated in real-world social forecasting scenarios (e.g., election forecasts, economic predictions).
5. **Unresolved Performative Optimality**: Only guarantees stability rather than the stronger property of optimality, and proves an unbridgeable gap exists between them.
6. **Unknown Distribution Map $\mathcal{D}(\cdot)$**: In practice, learners can only explore indirectly by deploying predictors and observing samples, but the paper does not deeply discuss the exploration-exploitation trade-off.

## Related Work & Insights

- **Performative Prediction** (Perdomo et al., 2020): Provided a formal framework.
- **Multicalibration** (Hébert-Johnson et al., 2018): Multi-group generalization of calibration.
- **Outcome Indistinguishability** (Dwork et al., 2021): The concept of computational indistinguishability.
- **K29 Algorithm** (Vovk et al., 2005): Kernel-based online calibration algorithm.
- **Kim & Perdomo (2023)**: Relationship between performative optimality and OI (in restricted settings).
- **Lucas Critique (1976)**: Classic critique in macroeconomics regarding how predictions influence policy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (An original bridge across social science and machine learning theory, resolving a century-old open problem)
- Experimental Thoroughness: ⭐⭐⭐ (Purely theoretical work, constructive counterexamples are clear but lacks empirical validation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear historical context, elegant technical presentation, and excellent motivation exposition)
- Value: ⭐⭐⭐⭐⭐ (Substantial contribution to the foundational theory of social prediction, with profound insights into "accuracy $\neq$ usefulness")

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/others/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)
- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](../../ACL2025/others/sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[ICML 2026\] Optimal Regularization for Performative Learning](../../ICML2026/others/optimal_regularization_for_performative_learning.md)
- [\[ICML 2025\] Revisiting Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model](revisiting_instance-optimal_cluster_recovery_in_the_labeled_stochastic_block_mod.md)
- [\[ACL 2025\] Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights](../../ACL2025/others/graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)

</div>

<!-- RELATED:END -->
