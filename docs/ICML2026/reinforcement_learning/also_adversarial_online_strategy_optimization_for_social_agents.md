---
title: >-
  [Paper Note] ALSO: Adversarial Online Strategy Optimization for Social Agents
description: >-
  [ICML 2026][Reinforcement Learning][LLM social intelligence] ALSO models the dynamic strategy selection in LLM social intelligence simulations as an adversarial online bandit. By using a lightweight reward surrogate mode…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "LLM social intelligence"
  - "multi-agent simulation"
  - "online strategy optimization"
  - "adversarial multi-armed bandit"
  - "Sotopia"
date: 2026-05-08
content_hash: b22ba8fdacf393b8
---

# ALSO: Adversarial Online Strategy Optimization for Social Agents

**Conference**: ICML 2026  
**arXiv**: [2605.15768](https://arxiv.org/abs/2605.15768)  
**Code**: https://github.com/Babylonehy/ALSO  
**Area**: Multi-agent / Social Intelligence  
**Keywords**: LLM social intelligence, multi-agent simulation, online strategy optimization, adversarial multi-armed bandit, Sotopia  

## TL;DR
ALSO models the dynamic strategy selection in LLM social intelligence simulations as an adversarial online bandit. By using a lightweight reward surrogate model to generalize sparse feedback from conversation history, it improves the overall score on Sotopia-Hard from 3.02 to 3.53, with particularly significant improvements in the relationship dimension.

## Background & Motivation
**Background**: LLM social simulations typically use a persona to describe who an agent is, including personality, occupation, background, and goals. In multi-turn dialogues, the model generates actions based on the persona and the scenario. Benchmarks like Sotopia have advanced social intelligence from static question-answering to open-ended multi-turn interactions.

**Limitations of Prior Work**: A static persona does not equate to a dynamic strategy. An agent can consistently "be the same person" while needing to switch strategies during negotiations, conflicts, or cooperation. Existing methods either train strategy models offline or use prompt optimizers to find instructions on a fixed validation set. These methods assume a stable reward distribution, but opponents in social interactions co-evolve with the dialogue.

**Key Challenge**: Feedback in social scenarios is both sparse and non-stationary. A strategy that is effective in the first few rounds may fail in later rounds because the opponent changes their stance. Standard stochastic bandits or offline prompt optimization struggle to utilize this feedback that drifts over time.

**Goal**: The authors aim to enable agents to select more appropriate social strategies based on historical states in each turn of a dialogue and continuously update from immediate evaluative feedback, without fine-tuning the LLM or invoking expensive additional LLM optimizers.

**Key Insight**: The paper treats candidate social strategies as bandit arms, conversation history as context, and normalized rewards from a per-turn LLM evaluator as online feedback. Due to opponent adaptation, the authors adopt an adversarial bandit perspective rather than a stationary stochastic bandit perspective.

**Core Idea**: Use EXP3-style randomized strategy selection to ensure exploration robustness in non-stationary environments, and use a neural surrogate to predict rewards over "historical context + strategy semantics" to address the issue of sparse feedback failing to cover all strategies.

## Method
ALSO targets the strategy instructions inserted after the persona in each turn rather than the model parameters. It splits the originally fixed persona prompt into two layers: the base identity remains unchanged, while the behavioral strategy is selected by an online learner. This ensures that identity continuity is not disrupted while allowing the mode of action to change with the dialogue situation.

### Overall Architecture
In a two-agent social simulation, each agent possesses a base persona, private goals, and a set of candidate strategies $\Sigma=\{\sigma_1,\dots,\sigma_K\}$. At the start of each turn, ALSO first encodes the current dialogue history, then concatenates each candidate strategy with the base persona to form an augmented persona, pre-calculating or reusing its embedding. A neural value network predicts the current reward for each candidate strategy. A strategy is sampled from an EXP3-style exponential weight distribution. The agent then generates the next utterance using the "base persona + selected strategy."

After the environment returns the opponent's response, an LLM evaluator provides turn-level multi-dimensional scores, which ALSO normalizes into a scalar reward. This sample is added to a replay buffer to update the surrogate via MSE, while cumulative scores for each strategy arm are updated using score smoothing with decay. The underlying LLM remains fully frozen; online changes occur only within the strategy selector and the lightweight value network.

### Key Designs
1.  **Strategies as adversarial bandit arms**:
    - **Function**: Transforms dynamic behavior selection in social interactions into an online-optimizable discrete decision problem.
    - **Mechanism**: Each strategy instruction corresponds to an arm. Once selected, it is concatenated with the base persona. The strategy sampling probability is given by $\pi_k^{(t)}\propto\exp(\eta S_k^{(t-1)})$ and updated with reward feedback.
    - **Design Motivation**: The reward distribution in social interactions drifts with opponent behavior and dialogue states. EXP3-style randomization is more resistant to non-stationary and adversarial changes than fixed greedy selection.

2.  **History-aware neural surrogate**:
    - **Function**: Estimates the potential value of other strategies in the current context under bandit feedback, where only the reward of the selected strategy is observed.
    - **Mechanism**: A frozen embedding model encodes the dialogue history and augmented persona separately, concatenating them into features $x_k^{(t)}=[b_k;c^{(t)}]$. The value network $f_\theta$ then predicts $\hat v_k^{(t)}$.
    - **Design Motivation**: Social strategies exhibit semantic correlation. For example, "validate then pivot" and "collaborative negotiation" may both be effective in similar contexts. The surrogate generalizes limited feedback to semantically similar strategies.

3.  **Decayed score smoothing**:
    - **Function**: Allows strategy scores to utilize historical experience while tracking reward drift in the dialogue.
    - **Mechanism**: New rewards enter the arm score with a temporal decay factor, ensuring old feedback does not permanently dominate current choices. Simultaneously, the replay buffer continues to provide samples for the surrogate.
    - **Design Motivation**: Optimal strategies in social dialogues often change across stages. Excessive memorization of early feedback causes strategy rigidity, while over-reliance on recent feedback increases variance.

### Loss & Training
ALSO does not fine-tune the LLM. The only component trained is the value network, which minimizes the MSE between predicted rewards and evaluator rewards using samples from the replay buffer. The strategy selector is updated online using exponential weights and score smoothing. The experiments use 12 predefined social strategies by default, with a maximum of 20 turns per episode. In a bilateral setting, two agents maintain independent optimizers and update based only on their own feedback.

## Key Experimental Results

### Main Results
The main experiments are evaluated on Sotopia-All and the more difficult Sotopia-Hard. Agent interactions use DeepSeek-V3.2, and final results are reported using an independent GPT-4o Sotopia-Eval. ALSO does not require additional LLM optimizer calls, whereas OPRO and EvoPrompt periodically call optimizers to generate or mutate prompts.

| Benchmark | Method | Goal | Rel. | Know. | Overall |
|-----------|------|------|------|-------|---------|
| Sotopia-All | Vanilla | 8.21 | 2.54 | 5.28 | 3.62 |
| Sotopia-All | INSTINCT | 8.51 | 2.84 | 6.09 | 3.85 |
| Sotopia-All | ALSO | 8.50 | 2.90 | 6.14 | 3.89 |
| Sotopia-Hard | Vanilla | 6.52 | 1.32 | 4.37 | 3.02 |
| Sotopia-Hard | INSTINCT | 6.92 | 2.16 | 5.44 | 3.43 |
| Sotopia-Hard | ALSO | 7.11 | 2.43 | 5.47 | 3.53 |

### Ablation Study
Ablations were conducted on Sotopia-Hard, removing or replacing key components of ALSO one by one.

| Configuration | Goal | Rel. | Know. | Overall | Description |
|------|------|------|-------|---------|------|
| ALSO full | 7.93 | 3.07 | 6.46 | 3.91 | Complete model |
| w/o EXP3, use $\varepsilon$-greedy | 7.50 | 2.71 | 5.32 | 3.61 | Randomized adversarial exploration weakened |
| w/o Score Smoothing | 7.57 | 2.25 | 5.39 | 3.57 | Relationship dimension drops most significantly |
| w/o Context Embedding | 7.43 | 2.64 | 4.82 | 3.51 | Unable to select strategies by dialogue stage |
| w/o Neural Surrogate | 6.89 | 2.00 | 4.93 | 3.33 | Largest degradation in overall and relationship dimensions |

### Key Findings
- The largest gain for ALSO on Sotopia-Hard comes from the Relationship dimension, increasing from 1.32 (Vanilla) to 2.43—an 83.79% relative improvement—indicating that online strategy switching primarily mitigates conflicts and deadlocks.
- Removing the neural surrogate dropped the Overall score from 3.91 to 3.33, making it the most critical ablation. This suggests that bandit counts alone cannot fully exploit strategy semantics and dialogue history.
- Bilateral optimization outperforms unilateral optimization and is significant for both Qwen-2.5-72B-Instruct and DeepSeek-V3.2, showing that mutual adaptation is more consistent with the task mechanism of social interaction.
- In cross-scenario generalization experiments, zero-shot transfer improved the Overall score from 3.17 to 3.60 across 7 unseen Sotopia-Hard scenarios, indicating the surrogate learns more than just scenario memorization.

## Highlights & Insights
- The paper effectively distinguishes between persona and strategy. Persona defines "who one is," while strategy defines "how one acts." The key to social intelligence is often not changing identity, but adjusting interaction styles within the same identity.
- The adversarial bandit perspective is better suited for social simulations than offline prompt optimizers. Rewards in Sotopia are not static scores on a validation set but trajectory outcomes co-evolved by both parties.
- The lightweight surrogate is a high-efficiency design. It requires no new prompt generation and no changes to LLM parameters, yet allows strategy selection to gain contextual generalization from sparse bandit feedback.

## Limitations & Future Work
- The strategy space consists of 12 manually predefined social strategies; coverage and granularity will affect the performance ceiling. Real-world open social interactions may require automated expansion or hierarchical strategy libraries.
- The per-turn evaluator is an LLM itself, which may introduce bias, scale drift, and self-consistency issues. Although the final judge is separated, online learning is still limited by the quality of the shaping reward.
- The paper primarily validates in two-agent Sotopia scenarios; whether it applies to multi-party group interactions, long-term memory, and alliance formation has not been fully explored.
- While the design rationale uses adversarial bandits, the paper provides no rigorous regret guarantees. This is understandable in highly non-stationary LLM environments but leaves room for theoretical analysis.

## Related Work & Insights
- **vs. Sotopia-RL / SDPO**: These methods improve social behavior through offline training or preference optimization at the cost of data collection and model updates. ALSO does not change model parameters, making it more suitable for rapid deployment adaptation.
- **vs. OPRO / EvoPrompt**: They treat prompt optimization as a search on static tasks. ALSO treats each turn's strategy selection as an online decision, responding to reward drift during the dialogue.
- **vs. External Planner Methods**: Methods like Sotopia-Ω, DAT, and EPO rely on off-policy learned planners. ALSO adjusts strategies based on feedback directly during interaction, avoiding the need to retrain planners for every new strategy.
- **Insight**: For agent systems, many "capability deficiencies" may result from a lack of an online adjustable behavioral strategy layer rather than the base model's limitations. Decoupling the strategy layer from the persona is a practical direction for building controllable social intelligence agents.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Specifically targeting social strategy optimization as an adversarial online bandit is highly relevant, though the core algorithmic components stem from existing online learning concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes main results, ablations, bilateral/unilateral analysis, cross-scenario, and heterogeneous model analysis, though still focused on the Sotopia suite.
- Writing Quality: ⭐⭐⭐⭐☆ Problem definition and algorithm flow are clear; experimental interpretations map well to social interaction mechanisms.
- Value: ⭐⭐⭐⭐☆ Provides valuable reference for online behavioral control of multi-agent systems, especially for applications where model fine-tuning is undesirable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](../../ACL2026/reinforcement_learning/breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)
- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](../../ACL2026/reinforcement_learning/dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[ICML 2026\] Data- and Variance-dependent Regret Bounds for Online Tabular MDPs](data-_and_variance-dependent_regret_bounds_for_online_tabular_mdps.md)
- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)

</div>

<!-- RELATED:END -->
