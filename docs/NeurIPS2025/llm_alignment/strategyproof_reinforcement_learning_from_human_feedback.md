---
title: >-
  [Paper Note] Strategyproof Reinforcement Learning from Human Feedback
description: >-
  [NeurIPS 2025][LLM Alignment][RLHF] This paper is the first to study strategic manipulation by annotators in RLHF from a mechanism design perspective. It proves a fundamental tradeoff between strategyproofness and policy…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "RLHF"
  - "strategyproofness"
  - "diverse preferences"
  - "mechanism design"
  - "social welfare maximization"
date: 2026-05-08
content_hash: eae0503897392935
---

# Strategyproof Reinforcement Learning from Human Feedback

**Conference**: NeurIPS 2025
**arXiv**: [2503.09561](https://arxiv.org/abs/2503.09561)
**Code**: Not released
**Area**: LLM Alignment
**Keywords**: RLHF, strategyproofness, diverse preferences, mechanism design, social welfare maximization

## TL;DR

This paper is the first to study strategic manipulation by annotators in RLHF from a mechanism design perspective. It proves a fundamental tradeoff between strategyproofness and policy alignment, and proposes the Pessimistic Median of MLEs algorithm to achieve approximate strategyproofness.

## Background & Motivation

RLHF has become a cornerstone of LLM alignment, yet a critical and largely overlooked issue arises in multi-annotator settings: **annotators may strategically distort their preference feedback** to steer the learned policy toward their own preferences.

A concrete example: in LLM fine-tuning, annotators may systematically misreport preferences to amplify specific biases and reinforce narratives favorable to themselves.

Limitations of existing RLHF methods:
- The Pessimistic Social Welfare approach of Zhong et al. (2024) is not strategyproof.
- MaxMin-RLHF by Chakraborty et al. (2024) is also not strategyproof.
- Adversarially robust RLHF works (e.g., Mandal et al., 2024) focus on adversarial corruption rather than strategic manipulation — the two differ fundamentally in perspective and technique.
- Payment-based mechanisms (e.g., VCG-type) are impractical in real-world deployments.

Core research question: **How can RLHF be made robust to strategic annotators without relying on payment mechanisms?**

## Method

### Overall Architecture

This paper operates in the offline RLHF setting with $k$ annotators, each possessing a linear reward function $r_{\theta_i^*}(s,a) = \langle \theta_i^*, \phi(s,a) \rangle$ under the Bradley-Terry preference model. The objective is to maximize social welfare $\mathcal{W}(\rho, \pi) = \frac{1}{k}\sum_{i=1}^k J_i(\rho, \pi)$.

Strategic manipulation is formalized as follows: annotator $i$ may sample preference labels using a manipulated reward parameter $\tilde{\theta}_i$ instead of the true parameter $\theta_i^*$.

### Key Designs

#### Key Theoretical Result I: Fragility of Existing Methods

**Proposition 3.3**: Both Pessimistic Social Welfare and MaxMin-RLHF **are not strategyproof**.

**Proposition 3.4**: Even with a **single** strategic annotator, the social welfare of Pessimistic Social Welfare can be driven arbitrarily low:

$$\mathcal{W}(\hat{\pi}) \leq \varepsilon, \quad \text{while} \quad \mathcal{W}(\pi^*) \geq BL - 2\varepsilon$$

#### Key Theoretical Result II: Fundamental Impossibility Theorem

**Theorem 3.5**: Any strategyproof RLHF algorithm incurs a worst-case suboptimality of at least:

$$\text{SubOpt}(\hat{\pi}) \geq \frac{k-1}{k}$$

**Corollary 3.6**: The approximation ratio of any strategyproof RLHF method is at most $\alpha(\rho, \hat{\pi}) \leq \frac{1}{k}$.

The proof invokes the **Gibbard-Satterthwaite theorem**: RLHF is mapped to a voting problem, and any strategyproof rule must be either dictatorial or restricted to two alternatives — both cases yield low social welfare.

#### Key Design: Pessimistic Median of MLEs Algorithm

To circumvent the impossibility theorem, an additional assumption is introduced (Assumption 2): the policy feature set $\{\mathbb{E}_{s \sim \rho}[\phi(s,\pi(s))]: \pi \in \Pi\}$ spans a hyperrectangle in $\mathbb{R}^d$.

Algorithm steps:
1. For each annotator $i$, compute the MLE $\hat{\theta}_i^{MLE}$ from dataset $\mathcal{D}_i$.
2. Construct confidence ellipsoids $C_i = \{\theta: \|\hat{\theta}_i^{MLE} - \theta\|_{\Sigma_{\mathcal{D}_i}} \leq f(d,n,\delta)\}$.
3. Form the median confidence set $\mathscr{C} = \{\text{med}(\theta_1,\ldots,\theta_k): \theta_i \in C_i\}$.
4. Compute the pessimistic median return $\underline{\mathcal{W}}(\pi) = \min_{\theta \in \mathscr{C}} \mathbb{E}[\langle \theta, \phi(s,\pi(s)) \rangle]$.
5. Output $\hat{\pi} = \arg\max_{\pi} \underline{\mathcal{W}}(\pi)$.

### Loss & Training

The algorithm adopts a pessimistic optimization strategy — taking the worst case over all feasible median parameters and selecting the policy that performs best under this worst case. This combines uncertainty quantification (confidence sets) with robust aggregation (the median rule).

## Key Experimental Results

### Main Results — Approximate Strategyproofness

**Theorem 4.1**: Pessimistic Median of MLEs is $\tilde{\mathcal{O}}(\kappa_i\sqrt{d/n})$-strategyproof with respect to annotator $i$:

$$J_i(\hat{\pi}(\tilde{\mathcal{D}}_i, \mathcal{D}_{-i})) - J_i(\hat{\pi}(\mathcal{D}_i^*, \mathcal{D}_{-i})) \leq const \cdot \kappa_i\sqrt{\frac{d + \log(k/\delta)}{n}}$$

where $\kappa_i = \max_{\pi} \|\mathbb{E}[\phi(s,\pi(s))]\|_{\Sigma_{\mathcal{D}_i}^{-1}}$ is the uniform policy coverage coefficient.

Key insight: **uniform policy coverage** is required (rather than coverage of only the optimal policy), because a manipulator may induce a policy arbitrarily far from the optimum.

### Social Welfare Guarantees

**Theorem 4.2**: Under honest reporting or weak dominant strategies, the suboptimality of the output policy satisfies:

$$\text{SubOpt}(\hat{\pi}) \leq const \cdot \left(\sqrt{\frac{d\log(k/\delta)}{k}} + \max_i \kappa_i^* \cdot k\sqrt{\frac{d + \log(k/\delta)}{n}}\right)$$

| Source | Suboptimality Contribution | Vanishes When |
|--------|---------------------------|---------------|
| Median approximation error | $O(\sqrt{d\log k / k})$ | $k \to \infty$ |
| Reward estimation error | $O(\kappa^* k \sqrt{d/n})$ | $n \to \infty$ |

### Special Case Analysis

| Scenario | Suboptimality Upper Bound |
|----------|--------------------------|
| Single annotator ($k=1$) | $O(\kappa_1^* \sqrt{d/n})$, consistent with classical RLHF |
| $k$ annotators with identical preferences | $O(\kappa^* k \sqrt{d/n})$, median error term eliminated |
| MDP extension | Analogous structure, $\kappa$ replaced by state-occupancy-based $\nu$ |

### Key Findings

1. A **fundamental tradeoff between strategyproofness and policy alignment** exists: it is impossible to achieve both simultaneously.
2. The median rule provides approximate strategyproofness in high dimensions, and the degree of strategyproofness improves with sample size.
3. The requirement for uniform policy coverage is surprising — standard RLHF theory requires only coverage of the optimal policy.

## Highlights & Insights

1. **Interdisciplinary contribution**: The first work to apply social choice theory (the Gibbard-Satterthwaite theorem) to RLHF theoretical analysis, establishing an impossibility result.
2. **Payment-free mechanism design**: Unlike VCG-based methods, the proposed approach requires no monetary payments to annotators, making it more practically viable.
3. **New role of coverage coefficients**: The necessity of uniform policy coverage in strategic settings is identified — a requirement absent from standard offline RL analysis.
4. **Graceful degradation**: As the number of annotators and data samples increases, the algorithm naturally converges to the optimal policy.
5. **Practical relevance**: As LLMs are deployed at scale, strategic manipulation by annotators poses a genuine threat; this paper provides a theoretical foundation for addressing it.

## Limitations & Future Work

1. **Linear reward function assumption**: Real-world preferences may be highly nonlinear; extending the analysis to nonlinear reward models is an important direction.
2. **Bradley-Terry model assumption**: Analysis under more general preference models (e.g., Plackett-Luce) remains open.
3. **Lack of empirical validation**: The paper is purely theoretical, with no empirical evaluation in practical LLM fine-tuning scenarios.
4. **Hyperrectangle assumption (Assumption 2)**: This is the key condition that circumvents the impossibility theorem, but its practical validity is unclear.
5. **Computational complexity**: Pessimistic optimization over the median confidence set may be computationally intractable in high dimensions.
6. **Extra $k$ factor**: The additional $k$ factor in suboptimality under multiple annotators appears to be an inherent cost of the algorithmic design.

## Related Work & Insights

- **Relation to Zhong et al. (2024)**: Their Pessimistic Social Welfare serves as the starting point; this paper demonstrates its lack of strategyproofness.
- **Relation to Siththaranjan et al. (2023)**: Standard RLHF is shown to implicitly use a Borda count voting rule, which may induce manipulation incentives.
- **Distinction from adversarially robust RLHF**: The adversarial model assumes a fixed fraction of corrupted samples, whereas the strategic model assumes rational annotators maximizing their own utility.
- **Broader inspiration**: The pessimism-plus-median design paradigm may generalize to other learning settings involving strategic participants.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce strategyproofness into RLHF and establish an impossibility theorem.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Both the impossibility theorem and algorithm design are supported by rigorous theory.
- **Experimental Thoroughness**: ⭐⭐ — Purely theoretical; no experiments are provided.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear with precise theorem statements, though notation is occasionally heavy.
- **Value**: ⭐⭐⭐⭐ — Strong problem motivation, though practical computability and scalability of the algorithm remain to be verified.
- **Overall**: ⭐⭐⭐⭐ (8.5/10) — A pioneering theoretical contribution that opens a new research direction at the intersection of RLHF and mechanism design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)
- [\[NeurIPS 2025\] Capturing Individual Human Preferences with Reward Features](capturing_individual_human_preferences_with_reward_features.md)
- [\[NeurIPS 2025\] Human-assisted Robotic Policy Refinement via Action Preference Optimization](human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)
- [\[NeurIPS 2025\] Can DPO Learn Diverse Human Values? A Theoretical Scaling Law](can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

</div>

<!-- RELATED:END -->
