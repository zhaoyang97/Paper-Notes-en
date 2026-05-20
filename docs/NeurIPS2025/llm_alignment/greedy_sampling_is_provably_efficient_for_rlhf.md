---
title: >-
  [Paper Note] Greedy Sampling Is Provably Efficient for RLHF
description: >-
  [NeurIPS 2025][LLM Alignment][Greedy Sampling] This paper proves that, under KL-regularized RLHF, directly applying greedy sampling based on empirical estimates—without constructing optimistic or pessimistic confidence s…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "Greedy Sampling"
  - "KL Regularization"
  - "Preference Learning"
  - "Regret Bounds"
  - "Bradley-Terry Model"
date: 2026-05-08
content_hash: 4448b06b7114ed7e
---

# Greedy Sampling Is Provably Efficient for RLHF

**Conference**: NeurIPS 2025
**arXiv**: [2510.24700](https://arxiv.org/abs/2510.24700)  
**Code**: None  
**Area**: LLM Alignment / RLHF Theory
**Keywords**: Greedy Sampling, KL Regularization, Preference Learning, Regret Bounds, Bradley-Terry Model

## TL;DR
This paper proves that, under KL-regularized RLHF, directly applying greedy sampling based on empirical estimates—without constructing optimistic or pessimistic confidence sets—achieves $O(\log T)$ regret in the online setting and $O(\varepsilon^{-1})$ sample complexity in the offline setting. These are the first results of such order under general preference models.

## Background & Motivation
**Background**: RLHF is a critical technique for post-training of LLMs. Theoretical analysis focuses on two core questions: regret bounds in the online setting and sample complexity in the offline setting. Prior work has established $O(\log T)$ regret and $O(\varepsilon^{-1})$ sample complexity under KL-regularized objectives, but these results either assume direct reward observations or are restricted to the Bradley-Terry preference model.

**Limitations of Prior Work**:
   - Theoretical guarantees for general preference models (without assuming an underlying reward function) remain at $O(\sqrt{T})$ regret and $O(\varepsilon^{-2})$ sample complexity.
   - Existing algorithms follow the optimism or pessimism principle—requiring construction of confidence bounds, which is computationally expensive and difficult in general settings.
   - The structural properties induced by KL regularization have not been adequately exploited in algorithm design.

**Key Challenge**: In classical RL and contextual bandits (CB), using empirical estimates directly without optimistic or pessimistic corrections is known to fail—greedy policies lead to insufficient exploration and suboptimal convergence. The key question is whether KL regularization in RLHF fundamentally changes this picture.

**Key Insight**: The paper identifies a previously overlooked structural property of the optimal policy class under KL-regularized objectives: the likelihood ratio between any candidate optimal policy and the reference policy is bounded, i.e., $\pi_f(a|x)/\pi_0(a|x) \in [\exp(-\eta), \exp(\eta)]$. This implies that the optimal policy must be stochastic and cannot deviate too far from the reference policy, thereby naturally providing sufficient exploration.

**Core Idea**: KL regularization endows the optimal policy with built-in exploration—no deliberate optimistic or pessimistic corrections are needed; greedy sampling alone suffices.

## Method

### Overall Architecture
The paper considers a KL-regularized contextual bandit (CB) problem: a policy $\pi$ selects an action $a$ (response) given context $x$ (prompt), with the objective of maximizing a preference score minus a KL penalty. Both general preference models (directly modeling $\mathbb{P}(a^1 \succ a^2 | x)$) and Bradley-Terry models (via a latent reward function) are addressed.

### Key Designs

1. **Structural Property of the Optimal Policy Class (Proposition 1 + Lemma 1)**:

    - Core finding: Under the KL-regularized objective, the optimal policy takes the form $\pi_f(a|x) \propto \pi_0(a|x)\exp(\eta f(x,a))$, where $f \in [0,1]$.
    - Key corollary: $\pi_f(a|x)/\pi_0(a|x) \in [\exp(-\eta), \exp(\eta)]$.
    - Two important implications: (a) the optimal policy must be stochastic (unlike classical RL where optimal policies are typically deterministic); (b) the ratio between the optimal policy and the reference policy is bounded—meaning that sampling from any candidate optimal policy provides a certain degree of coverage over all actions.
    - This is the fundamental reason greedy sampling works: no additional exploration mechanism is required.

2. **Online Greedy Sampling (Algorithm 1)**:

    - Function: At each round, sample $a^1$ from the current optimal policy and $a^2$ from the reference policy, observe preference feedback, perform MLE to update the preference model estimate, and compute the new greedy policy.
    - Distinction from prior methods: **No** confidence upper/lower bounds are constructed; no optimistic sampling policy is required; no special enhancer policy is needed.
    - Theoretical guarantee (Theorem 1): Regret under the general preference model is $O(\exp(\eta) \cdot d_{\text{GP}} \cdot \log(N_{\mathcal{P}}T/\delta))$, which is of order $O(\log T)$.

3. **Offline Greedy Sampling (Algorithm 2)**:

    - Function: Perform MLE on existing offline data to estimate the preference model, then directly compute the greedy policy.
    - Theoretical guarantee: Sample complexity $O(\varepsilon^{-1})$, requiring only single-policy coverage.
    - Comparison with classical $O(\varepsilon^{-2})$: an improvement of a full order of magnitude.

### Core Proof Techniques
- Decompose the per-step regret into a form involving the best-response policy, aligning the min-player's strategy.
- Apply Lemma 1 (bounded likelihood ratio) to convert all intermediate policies to the reference policy $\pi_0$, eliminating dependence on unknown policies.
- Relate MLE estimation error to the Eluder dimension via concentration inequalities.

## Key Experimental Results

### Synthetic Experiments (KL-regularized CB, 20 contexts, 10 actions)

| Method | Online Regret (GP, T=2000) | Online Regret (BT, T=2000) |
|--------|---------------------------|---------------------------|
| Greedy Sampling (Ours) | ~40 | ~35 |
| Optimistic Method (OFUL-style) | ~45 | ~40 |
| Uniform Exploration | ~100 | ~90 |

### Offline Experiments

| Method | Sample Complexity to Achieve $\varepsilon$-Optimality |
|--------|------------------------------------------------------|
| Greedy Sampling | $O(\varepsilon^{-1})$ |
| Pessimistic Method | $O(\varepsilon^{-1})$ (but computationally more complex) |
| Naive MLE | $O(\varepsilon^{-2})$ |

### Key Findings
- Greedy sampling achieves statistically comparable performance to more complex optimistic/pessimistic methods in synthetic experiments, while being computationally simpler.
- The $O(\log T)$ regret bound is the first of its kind under general preference models.
- The trade-off is an additional $\exp(\eta)$ factor in the regret bound—when KL regularization is weak (large $\eta$), greedy sampling requires more samples to match alternatives.

## Highlights & Insights
- **Counterintuitive result**: In classical RL, greedy sampling is well-known to fail due to insufficient exploration. This paper proves that KL regularization fundamentally changes this—regularization itself serves as exploration.
- **Algorithmic simplicity**: Only MLE followed by greedy selection is required; no complex confidence interval construction or minimax optimization is needed. This is not only theoretically stronger but also practically easier to implement.
- **Unified treatment of GP and BT models**: Prior tight bounds were restricted to the BT model; this paper is the first to extend results of the same order to general preference models.

## Limitations & Future Work
- **Theoretical limitations**: All results rely on tabular or finite function class assumptions; extension to continuous or infinite function classes requires further work.
- **The $\exp(\eta)$ factor**: When KL regularization is weak (large $\eta$), the advantage of greedy sampling diminishes.
- **Synthetic experiments only**: Validation is limited to small-scale synthetic settings; no real LLM experiments are included.
- **Single-round CB setting**: Practical RLHF involves multi-turn dialogue (full MDP); extension to multi-step decision-making requires additional analysis.
- **Connection to DPO not discussed**: DPO also exploits the structure of KL-regularized objectives but from a different angle; the theoretical relationship between the two warrants further exploration.

## Related Work & Insights
- **vs. Zhao et al. (2024, 2025a,b)**: They achieve the same-order guarantees under the BT model using optimism/pessimism. This paper achieves the same (or better) results under the more general GP model with a simpler greedy approach.
- **vs. Ye et al. (2024b)**: Their online regret under the GP model is $O(\sqrt{T})$; this paper improves it to $O(\log T)$.
- **vs. DPO**: DPO also exploits the closed-form solution of the optimal policy under KL regularization, but from an offline policy optimization perspective. The greedy sampling proposed here can be viewed as the theoretical dual of DPO—one greedy in policy space, the other optimizing in parameter space.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Greedy sampling is provably efficient in RLHF" is a counterintuitive and theoretically profound finding.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments validate the theory, but real LLM scenarios are not covered.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, intuitions are clearly explained, and proof sketches effectively convey the core ideas.
- Value: ⭐⭐⭐⭐ Significant theoretical advancement for RLHF and meaningful guidance for simplifying algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[NeurIPS 2025\] GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs](gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[NeurIPS 2025\] Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma](position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)
- [\[NeurIPS 2025\] Generalizing while Preserving Monotonicity in Comparison-based Preference Learning Models](generalizing_while_preserving_monotonicity_in_comparison-based_preference_learni.md)

</div>

<!-- RELATED:END -->
