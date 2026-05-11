---
title: >-
  [Paper Note] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution
description: >-
  [ACL 2026][Reinforcement Learning][Social Intelligence] This paper proposes Savoir, a cooperative game-theoretic social RL framework that combines expected utility (prospective evaluation of the strategic potential of ut…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Social Intelligence"
  - "Shapley Values"
  - "Credit Assignment"
  - "Cooperative Game Theory"
  - "Expected Utility"
date: 2026-05-08
content_hash: 5c60098dbe4be245
---

# Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution

**Conference**: ACL 2026
**arXiv**: [2604.18982](https://arxiv.org/abs/2604.18982)
**Code**: None
**Area**: Social Intelligence / Reinforcement Learning
**Keywords**: Social Intelligence, Shapley Values, Credit Assignment, Cooperative Game Theory, Expected Utility

## TL;DR

This paper proposes Savoir, a cooperative game-theoretic social RL framework that combines expected utility (prospective evaluation of the strategic potential of utterances) and Shapley values (axiomatic fair credit assignment) to address the credit assignment problem in multi-turn dialogue. Savoir achieves state-of-the-art performance on the SOTOPIA benchmark with a 7B model (Goal 7.18 in the Hard setting), matching or surpassing GPT-4o and Claude-3.5-Sonnet, while large reasoning models (o1, DeepSeek-R1) systematically underperform on social tasks.

## Background & Motivation

**Background**: Social intelligence—the capacity to navigate complex interpersonal interactions—is a core requirement for deploying LLMs in negotiation, collaboration, and persuasion scenarios. Recent work trains social agents via RL: SOTOPIA-π combines behavioral cloning and self-play reinforcement, while Sotopia-RL heuristically distributes episode-level rewards to the utterance level using an LLM.

**Limitations of Prior Work**: (1) The credit assignment in Sotopia-RL lacks theoretical grounding—the LLM assigns rewards directly without principled guarantees of fairness or accuracy. (2) More fundamentally, existing reward models perform **retrospective attribution** (how much did this utterance contribute to the outcome that has already occurred), rather than **prospective valuation** (how much strategic potential does this utterance create for future favorable interactions). Some utterances may appear to contribute little immediately, yet their strategic positioning unlocks critical pathways for subsequent success.

**Key Challenge**: Social interaction is inherently multi-turn, multi-objective, and competitive. The value of an individual utterance lies not only in its immediate contribution but also in the possibility space it creates for the future. Retrospective attribution cannot capture this forward-looking strategic value.

**Goal**: (1) Resolve the credit assignment problem in multi-turn dialogue using game-theoretic axioms; (2) Distinguish between the retrospective contribution and the prospective strategic value of utterances; (3) Achieve social intelligence in small models that surpasses that of large models.

**Key Insight**: Each social dialogue is treated as a cooperative game in which each utterance is a player jointly contributing to the final outcome. The heuristic LLM-based assignment is replaced by Shapley values with mathematical guarantees (efficiency, symmetry, and marginal contribution axioms).

**Core Idea**: Expected utility defines *what to measure* (assessing the forward-looking strategic value of utterances via rollouts), while Shapley values define *how to assign* (fair credit distribution with axiomatic guarantees). Together, they transform credit assignment from a heuristic into a principled computation.

## Method

### Overall Architecture

The Savoir training pipeline consists of three stages: (1) **Data collection**—LLMs generate social interaction episodes via self-play; (2) **Reward modeling**—the Savoir algorithm attributes episode-level outcomes to utterance-level rewards, which are used to train a reward model; (3) **Policy training**—online RL with GRPO following an SFT warm-up. The core innovation lies in stage (2): given $n$ utterances $N = \{a_1, \ldots, a_n\}$ produced by the agent in dialogue $\tau$, the Shapley value $\phi_i$ for each utterance is computed as the reward signal.

### Key Designs

1. **Expected Utility for Prospective Valuation**

    - **Function**: Shifts utterance evaluation from "what did it contribute to the past" to "what is its expected value for the future."
    - **Mechanism**: A value function is defined as $v(S) = \mathbb{E}_{\tau' \sim \mathcal{R}(H(S))}[U(\tau')]$, where $H(S)$ is the reconstructed dialogue history containing only the utterances in subset $S$ and their corresponding partner responses, and $\mathcal{R}(H(S))$ is the distribution over future dialogue trajectories from that state. Via Monte Carlo simulation: $v(S) = \frac{1}{J}\sum_{j=1}^J U(\tau_j)$, where complete dialogues are generated by alternating between agent policy $\pi_A$ and partner policy $\pi_B$, and $U(\tau) = \sum_d w_d \cdot G_d(\tau)$ aggregates outcomes across SOTOPIA's seven dimensions with learned weights.
    - **Design Motivation**: A carefully crafted proposal may appear to contribute little immediately, but the favorable trajectories it enables may carry substantial value—this strategic potential can only be assessed through rollouts.

2. **Shapley Values for Axiomatic Credit Assignment**

    - **Function**: Fairly distributes the total value of the value function among individual utterances.
    - **Mechanism**: $\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!}[v(S \cup \{i\}) - v(S)]$, computing the average marginal contribution of utterance $a_i$ over all possible orderings. This satisfies four axioms: efficiency (the sum of all Shapley values equals the total value), symmetry, null player, and additivity.
    - **Design Motivation**: Heuristic LLM-based credit assignment cannot guarantee fairness—certain utterances may be over- or under-attributed—whereas Shapley values constitute the unique assignment satisfying these axioms.

3. **KernelSHAP for Efficient Approximation**

    - **Function**: Reduces exponential Shapley computation to tractable weighted linear regression.
    - **Mechanism**: Shapley values are reformulated as a weighted least-squares problem: $\phi^* = \arg\min_\phi \sum_k w_k(v(S_k) - \sum_i \phi_i \cdot z_{ki})^2$, where SHAP kernel weights $w_k$ assign higher weight to coalitions of extreme size (very small or very large), as these provide the most informative marginal contributions. An intelligent coalition sampling strategy prioritizes extreme-sized coalitions.
    - **Design Motivation**: Direct computation requires $2^n$ evaluations of the value function (each requiring $J$ rollouts). KernelSHAP achieves high-accuracy approximation with approximately 200 coalition samples.

### Loss & Training

The reward model is trained with MSE loss $\mathcal{L}_\text{RM} = \mathbb{E}[(R_\theta(c,a) - \hat{\phi})^2]$. Policy training proceeds in two stages: SFT warm-up on GPT-4o self-play episodes, followed by online RL with GRPO (Group Relative Policy Optimization). Savoir's rollouts use $J=2$ simulations, with a coalition sampling cap of 200.

## Key Experimental Results

### Main Results

**SOTOPIA Benchmark Main Results (Goal metric, 0–10)**

| Model / Method | Self-Play All | Self-Play Hard | GPT-4o Partner All | GPT-4o Partner Hard |
|---|---|---|---|---|
| GPT-4o | 8.19 | 6.97 | 8.19 | 6.97 |
| Claude-3.5-Sonnet | 8.29 | 6.33 | 8.42 | 6.64 |
| OpenAI-o1 | 7.93 | 5.69 | 8.09 | 6.65 |
| DeepSeek-R1 | 7.97 | 5.86 | 7.92 | 6.20 |
| o3-mini | 7.38 | 5.14 | 7.96 | 6.33 |
| Sotopia-RL (7B) | 7.80 | 7.81 | 8.31 | 6.68 |
| **Savoir (7B)** | **8.43** | **7.93** | **8.42** | **7.18** |

### Ablation Study

**Component Decoupling of EU and Shapley (SOTOPIA-Hard, GPT-4o Partner)**

| Variant | EU | Shapley | Goal | Avg |
|---|---|---|---|---|
| Baseline (Sotopia-RL) | × | × | 6.68 | 3.29 |
| EU-only | ✓ | × | 6.89 | 3.38 |
| Shapley-only | × | ✓ | 6.96 | 3.42 |
| **Savoir (Full)** | ✓ | ✓ | **7.18** | **3.51** |

### Key Findings

- Savoir (7B) surpasses all large models: 8.43 vs. GPT-4o's 8.19 on Self-Play All, and 7.93 vs. 6.97 on the Hard setting (+13.8%).
- **Large reasoning models systematically underperform**: o3-mini achieves only 5.14 on Self-Play Hard vs. Savoir's 7.93 (a 54.3% gap), indicating that social intelligence requires intuitive responses rather than deliberate reasoning chains.
- EU and Shapley address orthogonal problems: EU alone yields a +3.1% improvement (better value estimation), Shapley alone yields +4.2% (fairer assignment), and the combination yields +7.5%—the two are complementary rather than overlapping.
- In human evaluation, strategic score is 4.06 vs. Sotopia-RL's 3.41 (+19.1%, $p<0.01$), with reward fairness preference at 67.1% vs. 15.7%.
- Performance improves consistently as training data scales from 2K to 7.5K episodes, with the largest gains occurring between 3K and 5K episodes (Goal +8.6%).

## Highlights & Insights

- Applying Shapley values to credit assignment in social dialogue represents an elegant marriage of theoretical rigor and practical effectiveness—the fairness guaranteed by four axioms translates directly into a stronger reward signal.
- The finding that reasoning models underperform in social tasks is particularly insightful: the "overthinking" characteristic of models such as o1 and R1 may actually impair social interactions that demand intuition and flexibility.
- The rollout mechanism of expected utility captures the value of "strategic positioning"—utterances that appear inconsequential may serve as critical groundwork for subsequent success.

## Limitations & Future Work

- The computational cost of rollouts and coalition sampling is high (approximately 200 coalitions × 2 rollouts per episode), limiting scalability.
- Evaluation relies on GPT-4o as the evaluator, which may introduce assessment bias.
- Performance degrades against increasingly capable partners: Goal drops by 17.8% against Gemini-3-Pro, indicating limited generalization.
- Evaluation is conducted solely on the SOTOPIA benchmark; real-world social scenarios may involve substantially greater complexity.

## Related Work & Insights

- **vs. Sotopia-RL**: Sotopia-RL assigns rewards heuristically via an LLM, whereas Savoir assigns them axiomatically via Shapley values, yielding improvements of 1.3–8.1% across all settings.
- **vs. SOTOPIA-π**: SOTOPIA-π employs behavioral cloning with filtering, providing episode-level reward granularity; Savoir provides fine-grained utterance-level rewards.
- **vs. DSI**: DSI achieves 7.31 on Self-Play Hard; Savoir achieves 7.93 (+8.5%), with an even larger advantage in the GPT-4o Partner setting.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The application of Shapley values and expected utility to social RL demonstrates both theoretical depth and practical innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Main experiments, component ablations, human evaluation, data-scale analysis, and opponent-strength analysis are all included, making the evaluation exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, the logical chain from motivation to method is complete, and the case studies are vivid.
- **Value**: ⭐⭐⭐⭐⭐ — A 7B model surpassing GPT-4o in social intelligence carries significant practical implications, and the finding that reasoning models underperform has far-reaching impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[NeurIPS 2025\] Approximating Shapley Explanations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/approximating_shapley_explanations_in_reinforcement_learning.md)
- [\[AAAI 2026\] RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](../../AAAI2026/reinforcement_learning/rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)
- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](../../ICLR2026/reinforcement_learning/efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[ICLR 2026\] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](../../ICLR2026/reinforcement_learning/on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)

</div>

<!-- RELATED:END -->
