---
title: >-
  [Paper Note] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution
description: >-
  [ACL 2026][Reinforcement Learning][Social Intelligence] This paper proposes Savoir, a social RL framework based on cooperative game theory. By combining Expected Utility (prospective evaluation of the strategic potential…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Social Intelligence"
  - "Shapley Values"
  - "Credit Assignment"
  - "Cooperative Game Theory"
  - "Expected Utility"
date: 2026-05-08
content_hash: 6e9ef014fb62d554
---

# Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution

**Conference**: ACL 2026  
**arXiv**: [2604.18982](https://arxiv.org/abs/2604.18982)  
**Code**: None  
**Area**: Social Intelligence / Reinforcement Learning  
**Keywords**: Social Intelligence, Shapley Values, Credit Assignment, Cooperative Game Theory, Expected Utility

## TL;DR

This paper proposes Savoir, a social RL framework based on cooperative game theory. By combining Expected Utility (prospective evaluation of the strategic potential of utterances) and Shapley values (axiomatic fair credit assignment), it addresses the credit assignment problem in multi-turn dialogues. Savoir achieves SOTA performance with a 7B model on the SOTOPIA benchmark (Goal 7.18 in the Hard setting), matching or surpassing GPT-4o and Claude-3.5-Sonnet, while observing that large reasoning models (o1, DeepSeek-R1) systematically underperform on social tasks.

## Background & Motivation

**Background**: Social intelligence—the ability to navigate complex interpersonal interactions—is a core requirement for LLMs in negotiation, collaboration, and persuasion scenarios. Recent studies train social agents via RL methods: SOTOPIA-π combines behavior cloning with self-reinforcement, while Sotopia-RL uses LLMs to heuristically distribute episode-level rewards at the utterance level.

**Limitations of Prior Work**: (1) Credit assignment in Sotopia-RL lacks a theoretical foundation, as LLMs distribute rewards without principled guarantees of fairness or accuracy. (2) More fundamentally, existing reward models perform **retrospective attribution** (how much an utterance contributed to an outcome that already occurred) rather than **prospective valuation** (how much strategic potential an utterance created for subsequent favorable interactions). Some utterances may seem to contribute little immediately but unlock critical paths for later success through strategic positioning.

**Key Challenge**: Social interactions are inherently multi-turn, multi-objective, and competitive. The value of an individual utterance lies not just in its current contribution, but in the space of possibilities it creates for the future. Retrospective attribution fails to capture this prospective strategic value.

**Goal**: (1) Solve the credit assignment problem in multi-turn dialogues using game-theoretic axioms; (2) Distinguish between the retrospective contribution and prospective strategic value of an utterance; (3) Achieve social intelligence in small models that exceeds that of large models.

**Key Insight**: Each social dialogue is treated as a cooperative game where each utterance is a player contributing to the final outcome. The heuristic distribution of LLMs is replaced by the mathematical guarantees of Shapley values (Efficiency, Symmetry, and Marginal Contribution axioms).

**Core Idea**: Expected Utility defines "what to measure" (evaluating the prospective strategic value of an utterance via rollouts), and Shapley values define "how to allocate" (principled fair credit distribution). Together, they transform credit assignment from a heuristic into a principled calculation.

## Method

### Overall Architecture

The training pipeline of Savoir consists of three stages: (1) Data Collection—generating social interaction episodes via LLM self-play; (2) Reward Modeling—attributing episode-level outcomes to the utterance level using the Savoir algorithm to train a reward model; (3) Policy Training—SFT warm-up followed by online RL using GRPO. The core innovation lies in Stage (2): given $n$ utterances $N = \{a_1, \ldots, a_n\}$ of an agent in a dialogue $\tau$, the Shapley value $\phi_i$ for each utterance is calculated as the reward signal.

### Key Designs

1.  **Expected Utility (EU) for Prospective Valuation**:
    *   **Function**: Shifts utterance evaluation from "what did it contribute to the past" to "what is the expected value for the future."
    *   **Mechanism**: Defines a value function $v(S) = \mathbb{E}_{\tau' \sim \mathcal{R}(H(S))}[U(\tau')]$, where $H(S)$ is a reconstructed dialogue history containing only utterances in subset $S$ and their corresponding partner responses, and $\mathcal{R}(H(S))$ is the distribution of future trajectories starting from that state. Via Monte Carlo simulation: $v(S) = \frac{1}{J}\sum_{j=1}^J U(\tau_j)$, where full dialogues are generated using the agent's policy $\pi_A$ and the partner's policy $\pi_B$. $U(\tau) = \sum_d w_d \cdot G_d(\tau)$ aggregates the seven SOTOPIA dimensions using weights.
    *   **Design Motivation**: A well-designed proposal might show small immediate contribution, but the subsequent favorable trajectories it opens can be highly valuable—this strategic potential can only be evaluated through rollouts.

2.  **Shapley Values for Axiomatic Credit Assignment**:
    *   **Function**: Fairly distributes the total value from the value function among the utterances.
    *   **Mechanism**: $\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!}[v(S \cup \{i\}) - v(S)]$, calculating the average marginal contribution of utterance $a_i$ across all possible permutations. It satisfies four axioms: Efficiency (the sum of all Shapley values equals the total value), Symmetry, Null Player, and Additivity.
    *   **Design Motivation**: Heuristic LLM credit assignment cannot guarantee fairness—certain utterances may be over- or under-attributed. The Shapley value is the unique distribution scheme satisfying these axioms.

3.  **Efficient Approximation via KernelSHAP**:
    *   **Function**: Converts exponential Shapley calculations into feasible weighted linear regressions.
    *   **Mechanism**: Restructures the Shapley value as a weighted least squares problem: $\phi^* = \arg\min_\phi \sum_k w_k(v(S_k) - \sum_i \phi_i \cdot z_{ki})^2$, where SHAP kernel weights $w_k$ assign higher weights to coalitions of extreme sizes (very small or very large) as they provide the most informative marginal contributions. A smart coalition sampling strategy is used to prioritize extreme sizes.
    *   **Design Motivation**: Direct calculation requires $2^n$ value function evaluations. KernelSHAP achieves high-precision approximation with approximately 200 coalition samples.

### Loss & Training

The reward model is trained using MSE loss: $\mathcal{L}_\text{RM} = \mathbb{E}[(R_\theta(c,a) - \hat{\phi})^2]$. Policy training involves two stages: SFT warm-up on episodes generated by GPT-4o self-play, followed by online RL using GRPO (Group Relative Policy Optimization). Savoir uses $J=2$ simulations for each rollout and a cap of 200 for coalition sampling.

## Key Experimental Results

### Main Results

**Main Results on SOTOPIA Benchmark (Goal Metric, 0-10 scale)**

| Model/Method | Self-Play All | Self-Play Hard | GPT-4o Partner All | GPT-4o Partner Hard |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 8.19 | 6.97 | 8.19 | 6.97 |
| Claude-3.5-Sonnet | 8.29 | 6.33 | 8.42 | 6.64 |
| OpenAI-o1 | 7.93 | 5.69 | 8.09 | 6.65 |
| DeepSeek-R1 | 7.97 | 5.86 | 7.92 | 6.20 |
| o3-mini | 7.38 | 5.14 | 7.96 | 6.33 |
| Sotopia-RL (7B) | 7.80 | 7.81 | 8.31 | 6.68 |
| **Savoir (7B)** | **8.43** | **7.93** | **8.42** | **7.18** |

### Ablation Study

**Decoupling EU and Shapley Components (SOTOPIA-Hard, GPT-4o Partner)**

| Variant | EU | Shapley | Goal | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Sotopia-RL) | × | × | 6.68 | 3.29 |
| EU-only | ✓ | × | 6.89 | 3.38 |
| Shapley-only | × | ✓ | 6.96 | 3.42 |
| **Savoir (Full)** | ✓ | ✓ | **7.18** | **3.51** |

### Key Findings

*   7B Savoir outperforms all large models: 8.43 vs. 8.19 for GPT-4o on Self-Play All, and 7.93 vs. 6.97 (+13.8%) in the Hard setting.
*   **Large reasoning models are systematically deficient**: o3-mini scores only 5.14 on Self-Play Hard vs. Savoir's 7.93 (a 54.3% gap), indicating that social intelligence requires intuitive responses rather than deliberate chains of thought.
*   EU and Shapley solve orthogonal problems: EU alone provides a 3.1% Gain (better value estimation), Shapley alone provides 4.2% (fairer distribution), and the combination provides 7.5%—they are complementary rather than overlapping.
*   In human evaluation, the strategizing score was 4.06 vs. 3.41 for Sotopia-RL (+19.1%, $p<0.01$), with a reward fairness preference of 67.1% vs. 15.7%.
*   Performance improves continuously with training data from 2K to 7.5K episodes, with the largest gains between 3K-5K (Goal +8.6%).

## Highlights & Insights

*   Introducing Shapley values for credit assignment in social dialogue is a perfect marriage of theoretical elegance and practical efficacy—fairness guaranteed by the four axioms translates directly into better reward signals.
*   The insight that "reasoning models are not good at social tasks" is profound—the "over-thinking" of models like o1 and R1 may actually harm social interactions which require intuition and flexibility.
*   The rollout mechanism of Expected Utility captures the value of "strategic positioning"—seemingly inconsequential utterances can be critical precursors to later success.

## Limitations & Future Work

*   The computational cost of rollouts and coalition sampling is high (approx. 200 coalitions × 2 rollouts per episode), limiting large-scale application.
*   The evaluation relies on GPT-4o as an evaluator, which may introduce evaluation bias.
*   Performance degrades against increasingly strong dialogue partners: Goal decreases by 17.8% vs. Gemini-3-Pro, indicating limited generalization.
*   Evaluated only on the SOTOPIA benchmark; real-world social scenarios may be significantly more complex.

## Related Work & Insights

*   **vs. Sotopia-RL**: Sotopia-RL uses heuristic LLM reward distribution, while Savoir uses axiomatic Shapley distribution, with the latter showing 1.3-8.1% Gains across all settings.
*   **vs. SOTOPIA-π**: SOTOPIA-π uses behavior cloning and filtering with episode-level signals; Savoir provides fine-grained utterance-level rewards.
*   **vs. DSI**: DSI reaches 7.31 on Self-Play Hard, while Savoir reaches 7.93 (+8.5%), showing a greater advantage in GPT-4o Partner settings.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ The application of Shapley values and Expected Utility in social RL has significant theoretical depth and practical innovation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, including main results, component ablations, human evaluation, data scaling, and opponent strength analysis.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation with a complete logic chain from motivation to method and vivid case studies.
*   Value: ⭐⭐⭐⭐⭐ A 7B model surpassing GPT-4o in social intelligence has major practical significance, and the finding regarding reasoning model deficiencies has a far-reaching impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Approximating Shapley Explanations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/approximating_shapley_explanations_in_reinforcement_learning.md)
- [\[ICML 2026\] Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](../../ICML2026/reinforcement_learning/shapley_neuron_values_for_continual_learning_which_neurons_matter_most.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)
- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](../../ICLR2026/reinforcement_learning/efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)

</div>

<!-- RELATED:END -->
