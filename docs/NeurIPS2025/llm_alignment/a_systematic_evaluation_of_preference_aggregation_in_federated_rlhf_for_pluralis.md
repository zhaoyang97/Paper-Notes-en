---
title: >-
  [Paper Note] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs
description: >-
  [NEURIPS2025][LLM Alignment][federated learning] This paper proposes an Adaptive Alpha aggregation strategy that dynamically adjusts reward weights based on each user group's historical alignment performance within a federated RLHF framework, simultaneously achieving high fairness and strong alignment performance for pluralistic preference alignment.
tags:
  - NEURIPS2025
  - LLM Alignment
  - federated learning
  - RLHF
  - Pluralistic Alignment
  - Preference Aggregation
  - Fairness
date: 2026-05-08
content_hash: 3dc21b1aa1d3c3c6
---

# A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs

**Conference**: NEURIPS2025
**arXiv**: [2512.08786](https://arxiv.org/abs/2512.08786)
**Code**: To be confirmed
**Area**: LLM Alignment
**Keywords**: federated learning, RLHF, Pluralistic Alignment, Preference Aggregation, Fairness

## TL;DR

This paper proposes an Adaptive Alpha aggregation strategy that dynamically adjusts reward weights based on each user group's historical alignment performance within a federated RLHF framework, simultaneously achieving high fairness and strong alignment performance for pluralistic preference alignment.

## Background & Motivation

**The Centralization Dilemma in Standard RLHF**: Conventional RLHF relies on centralized preference data, which poses privacy risks and tends to favor the preferences of narrow populations, failing to faithfully reflect the values of diverse user groups.

**Combining Federated Learning with RLHF**: FL enables training on decentralized data, preserving privacy and capturing broader human preferences, but introduces a core challenge—how to aggregate heterogeneous preference signals from different groups.

**Aggregation Strategy Determines Alignment Behavior**: The choice of reward aggregation is not a technical detail; it directly constitutes an evaluation protocol that determines whose preferences are prioritized and whose are marginalized.

**Limitations of Prior Work**: Methods such as GPO, GRPO, and MaxMin-RLHF account for multiple groups but still rely on centralized processing of user data.

**Foundation of PluralLLM**: Prior work PluralLLM extends GPO to a federated architecture, training lightweight preference predictors for each group as local reward models.

**Zero-Shot Alignment Paradigm**: The entire approach uses only aggregated group reward signals without task demonstrations, ensuring transferability.

## Method

### Overall Architecture

- **System Setup**: Each group $g_i$ acts as a federated client maintaining a private preference dataset $D_{g_i}$, with data never leaving the local device.
- **Pipeline**: The server generates rollouts from the current policy $\pi_\theta^{policy}$ → distributes them to each group → each group evaluates them using PluralLLM to produce group rewards → the server aggregates the rewards → PPO updates the policy → iteration repeats.

### Key Design 1: Distributed Reward Generation

- Each group uses PluralLLM (a lightweight transformer) as its local reward model.
- Groups generate preference probabilities over received rollouts, converting them into scalar rewards.
- Two task modes are supported: **preference probability prediction** (directly using predicted probabilities) and **preference ranking** (converting probabilities to rankings).
- Six reward metrics are considered: distance-based (Wasserstein, Cosine, KL) and ranking-based (Kendall Tau, Borda, Binary).

### Key Design 2: Adaptive Alpha Aggregation

The core innovation replaces a fixed global $\alpha$ with group-specific dynamic weights $\alpha_g^t$:

$$Agg_\alpha(\mathbf{r}^t) = \begin{cases} \frac{1}{|G|}\sum_{g} r_g^t & \text{if } FI \geq 0.9 \\ \log\left(\frac{1}{|G|}\sum_{g}\exp(\alpha_g^t \cdot r_g^t)\right) & \text{otherwise} \end{cases}$$

- Weight computation: $\alpha_g^t = \text{softmax}((1 - h_g^{t-1}) / T)$, with temperature $T=0.1$.
- Groups with poorer historical alignment receive higher weights and are prioritized.
- When $FI \geq 0.9$, rewards across groups are sufficiently uniform and the strategy degenerates to simple averaging.

### Key Design 3: Fairness Index (FI)

$$FI = \frac{1}{|X|}\sum_{q_j \in X} \frac{1}{1 + \text{CoV}^2(q_j)}$$

- Measures inter-group reward disparity based on the Coefficient of Variation (CoV).
- $FI \in [0, 1]$, where 1 denotes perfect fairness and 0 denotes maximum unfairness.
- Serves both as an evaluation metric and as the switching condition for the aggregation strategy.

### Loss & Training

Standard PPO loss is used for policy updates, with the key reward signal $r_{final}$ produced by the aggregation strategy described above. LoRA rank=32, lr=$1\times10^{-5}$, with 4-bit quantization to accelerate training.

## Key Experimental Results

### Experimental Setup

| Item | Configuration |
|------|------|
| Base Model | Gemma-2-2B-it |
| Dataset | Pew Research Center Global Attitudes Surveys (2,554 multiple-choice questions, cross-national groups) |
| Baselines | SFT baseline, Min/Max/Avg/Alpha aggregation |
| Metrics | FI (fairness), Avg AS (average alignment score), Min AS (worst-group alignment score) |

### Preference Probability Prediction Task (Core Results)

| Method | Reward Function | FI (Cos) | Avg AS (Cos) | Min AS (Cos) |
|------|----------|----------|--------------|--------------|
| SFT | — | 0.97 | 0.82 | 0.77 |
| Alpha | Wasserstein | **0.99** | 0.90 | 0.89 |
| Alpha | Cosine | **0.99** | **0.92** | **0.90** |
| Max | Cosine | 0.99 | 0.93 | 0.91 |
| Min | Cosine | 0.99 | 0.92 | 0.90 |
| Alpha | KL | 0.99 | 0.92 | 0.90 |

### Preference Ranking Task

| Method | Reward Function | FI (Borda) | Avg AS (Borda) | Min AS (Kendall) |
|------|----------|------------|----------------|------------------|
| SFT | — | 0.87 | 0.50 | 0.25 |
| Alpha | Kendall | 0.81 | 0.47 | **0.47** |
| Alpha | Borda | **0.95** | **0.61** | 0.34 |
| Avg | Borda | 0.92 | 0.58 | 0.35 |

### Key Findings

1. **Distance-based rewards substantially outperform ranking-based rewards**: Wasserstein/Cosine rewards combined with Alpha aggregation achieve $FI \approx 0.99$ and Avg AS $\approx 0.90$–$0.92$.
2. **Alpha aggregation consistently leads**: It achieves the highest FI under nearly all reward functions while maintaining competitive Avg AS and Min AS.
3. **The pitfall of Max aggregation**: Although it can improve Avg AS, it does so at the expense of disadvantaged groups, with Min AS declining noticeably.
4. **Insufficiency of the SFT baseline**: In ranking tasks, the SFT Min AS is only 0.25–0.41, indicating that uniform preference training fails to capture inter-group differences.

## Highlights & Insights

- **Novel framing of "aggregation as evaluation protocol"**: Elevates a technical choice (aggregation method) to the level of a value-laden decision.
- **Elegant and concise adaptive mechanism**: Dynamic balance is achieved via inverted softmax combined with an FI threshold switch, without complex hyperparameter tuning.
- **Comprehensive evaluation framework**: 6 reward metrics × 4 aggregation strategies × 2 task types provides thorough coverage.
- **Zero-shot generalization**: Without relying on task-specific demonstrations, the aggregation strategy transfers directly to new scenarios.
- **Protection of the worst-off group**: Emphasis on Min AS ensures that no group is left behind.

## Limitations & Future Work

1. **PPO only**: Performance under lighter RL frameworks such as DPO/GRPO has not been verified.
2. **Limited model scale**: Experiments are conducted only on Gemma-2B-it; scalability to larger models remains unvalidated.
3. **Dataset favors cooperative settings**: The degree of inter-group conflict in Pew survey data is limited; adversarial scenarios have not been tested.
4. **Single task type**: Only multiple-choice QA tasks are considered; generative tasks (summarization, dialogue, code) are not addressed.
5. **Fixed group definition**: Groups are defined at the country level; finer-grained or dynamic group partitioning is not explored.

## Related Work & Insights

- **GPO** (Group Preference Optimization): Introduces group-specific alignment but operates in a centralized manner.
- **GRPO / MaxMin-RLHF**: Address robustness in centralized RLHF but require centralized data processing.
- **PluralLLM**: The direct foundation of this work; extends GPO to a federated architecture with lightweight preference predictors.
- **Alpha Aggregation** (Park et al.): Introduces an $\alpha$ parameter to control aggregation consensus; this paper extends it to an adaptive variant.
- **Federated RLHF**: An emerging direction for which this paper contributes a systematic evaluation framework.

## Rating
- Novelty: ⭐⭐⭐⭐ — The Adaptive Alpha aggregation idea is concise and effective; the evaluation framework is systematically designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dimensional evaluation provides comprehensive coverage, though limited to a single model and dataset.
- Writing Quality: ⭐⭐⭐⭐ — Arguments are clearly presented, mathematical derivations are complete, and experimental tables are information-dense.
- Value: ⭐⭐⭐⭐ — Provides practical baselines and an evaluation protocol for federated pluralistic alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma](position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)
- [\[NeurIPS 2025\] EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](../../ACL2026/llm_alignment/reward_modeling_for_scientific_writing_evaluation.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](../../ICLR2026/llm_alignment/beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](greedy_sampling_is_provably_efficient_for_rlhf.md)

</div>

<!-- RELATED:END -->
