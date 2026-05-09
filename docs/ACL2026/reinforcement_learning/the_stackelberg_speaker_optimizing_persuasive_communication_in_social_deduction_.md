---
title: >-
  [Paper Note] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games
description: >-
  [ACL 2026][Reinforcement Learning][To be supplemented] To be supplemented after thorough reading.
tags:
  - ACL 2026
  - Reinforcement Learning
  - To be supplemented
date: 2026-05-08
content_hash: 12dbf6af3a593721
---

# The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games

**Conference**: ACL 2026
**arXiv**: [2510.09087](https://arxiv.org/abs/2510.09087)
**Code**: [https://3dagentworld.github.io/leader_follower](https://3dagentworld.github.io/leader_follower)
**Area**: Reinforcement Learning / Social Deduction Games
**Keywords**: Persuasive Communication, Social Deduction Games, Stackelberg Game, GRPO, LLM Agents

## TL;DR

This paper models turn-based dialogue in social deduction games as a Stackelberg game, where the current player acts as the leader and optimizes the persuasive impact of utterances by measuring the response distribution of the next player. A Refiner model trained with GRPO achieves significant improvements over baselines across four game benchmarks including Werewolf and Avalon.

## Background & Motivation

**Background**: LLM agents have achieved notable progress in social deduction games (SDGs) such as Werewolf and Avalon. Existing methods primarily focus on information processing (inferring other players' roles) and strategy selection (choosing optimal actions).

**Limitations of Prior Work**: Existing methods overlook the central role of persuasive communication. In SDGs, success depends not only on making correct inferences but also on convincing others to act according to one's intentions. Existing RL methods (e.g., SLA, LSPO) reduce the rich natural language space to a finite action classification problem, precluding utterance optimization in the continuous language space.

**Key Challenge**: The core challenge in SDGs is not "knowing what is correct" but "convincing others that one is correct." The persuasive dimension is central to both game success and real human interaction, yet remains largely unaddressed in current research.

**Goal**: To explicitly model and optimize persuasive communication in social deduction games, enabling agents to proactively steer dialogue toward favorable outcomes.

**Key Insight**: The paper adopts the Stackelberg game framework from game theory — if the leader sufficiently understands the follower's response distribution over different actions, the leader can select the action that maximizes its own utility. In turn-based dialogue, the current speaker naturally assumes the role of leader.

**Core Idea**: A Refiner model is trained to refine base utterances into more persuasive versions. The reward signal is based on the shift in the next player's response probability distribution induced by the utterance — increasing the probability of desired responses while decreasing that of undesired ones.

## Method

### Overall Architecture

The framework proceeds in three steps: (1) **Intent Recognition** — an API LLM analyzes the current game state and generates $K=3$ desired and undesired follower responses respectively; (2) **Impact Measurement** — an API LLM generates a base utterance, the Refiner produces multiple candidates, and a Measurer computes the distributional shift in follower responses as the reward; (3) **Policy Optimization** — GRPO is used to train the Refiner to maximize persuasive impact.

### Key Designs

1. **Stackelberg Modeling and Intent Recognition**:

    - **Function**: Models each speaking turn as a leader–follower interaction and explicitly defines the optimization objective.
    - **Mechanism**: The current player $p_t$ acts as the leader and the next player $p_{t+1}$ as the follower. Given the game rules $\mathcal{R}$, game state $G_t$, dialogue history $D_t$, and hidden role $r_t$, a backend LLM generates $K=3$ desired responses $\hat{u}_{t+1}^{+,(k)}$ and undesired responses $\hat{u}_{t+1}^{-,(k)}$.
    - **Design Motivation**: Explicitly defining "what constitutes effective persuasion" converts an ambiguous persuasion objective into a measurable probability shift.

2. **Persuasive Impact Measurement**:

    - **Function**: Computes a persuasion reward for each candidate utterance based on its effect on follower behavior.
    - **Mechanism**: Qwen2.5-72B serves as the Measurer to simulate follower response patterns. For a candidate utterance $u_t^{(i)}$, the reward is defined as: $R(u_t^{(i)}) = \sum_k \log P_\mathcal{F}(\hat{u}_{t+1}^{+,(k)} | \text{ctx} \cup \{u_t^{(i)}\}) - \sum_k \log P_\mathcal{F}(\hat{u}_{t+1}^{-,(k)} | \text{ctx} \cup \{u_t^{(i)}\})$
    - **Design Motivation**: Measuring persuasive effect directly in the follower's probability space is more objective than human annotation or heuristic evaluation.

3. **GRPO Policy Optimization**:

    - **Function**: Trains the Refiner to optimize utterance persuasiveness in the continuous natural language space.
    - **Mechanism**: Qwen2.5-7B with LoRA (rank 16) serves as the Refiner; $n=8$ candidates are sampled, and GRPO computes within-group relative advantage for policy optimization, with KL divergence regularization to prevent excessive deviation.
    - **Design Motivation**: GRPO eliminates the need for an additional critic model and directly leverages the in-batch reward distribution to compute relative advantage.

### Loss & Training

GRPO objective: $\mathcal{J}(\theta) = \mathbb{E}_c[\frac{1}{n}\sum_i \mathcal{L}_i - \beta D_{KL}(\pi_\theta || \pi_{ref})]$, with $n=8$, $\varepsilon=0.2$, $\beta=0.04$. Training uses 500 self-play games per game type, selecting 4,000 instances. Backend LLMs are randomly sampled from GPT-4o, Gemini-2.5-Flash, and Claude-3.5-Haiku. Learning rate: $1 \times 10^{-6}$; trained for 3 epochs on 4×A800 GPUs for approximately 50 hours.

## Key Experimental Results

### Main Results

| Game | Method | Overall Win Rate |
|------|------|---------|
| Werewolf | LSPO | 38.6% |
| Werewolf | **Ours + LSPO** | **44.7%** |
| Avalon | Strategist | 57.4% |
| Avalon | **Ours + Strategist** | **61.3%** |
| ONUW | RL-ins. | 48.5% |
| ONUW | **Ours + RL-ins.** | **51.5%** |

### Ablation Study

| Reward Variant | Werewolf Avg | Avalon Avg | ONUW Avg |
|---------|-----------|-----------|---------|
| ReAct (baseline) | 49.0 | 44.0 | 48.0 |
| Pos-Only + ReAct | 64.0 | 58.0 | 60.0 |
| Neg-Only + ReAct | 49.0 | 46.0 | 47.0 |
| **Ours + ReAct** | **70.0** | **61.0** | **61.0** |

### Key Findings

- Positive reward (increasing desired response probability) contributes substantially more than negative reward (decreasing undesired response probability).
- The Refiner yields greater gains when combined with stronger baselines, indicating it is complementary to rather than a replacement for existing strategies.
- Improvements are especially pronounced for deceptive roles — the Werewolf win rate increases from 79% to 84.2%.
- The method generalizes successfully to the Sotopia social simulation environment, demonstrating applicability beyond SDGs.

## Highlights & Insights

- Modeling turn-based dialogue as a Stackelberg game is highly natural — quantifying persuasive impact as "the shift in the opponent's response probability" offers a more fine-grained signal than directly optimizing win rates.
- Employing a separate large model to simulate the follower's response distribution elegantly circumvents the limitation that API LLMs do not expose token probabilities.
- Positioning the Refiner as an "utterance polisher" is practically effective — it preserves the semantic understanding of strong API LLMs while using a smaller model for persuasion enhancement.

## Limitations & Future Work

- The Measurer uses a fixed large model to simulate follower behavior, which may not accurately reflect actual opponent actions.
- Training assumes full information (opponent roles are known), which is unavailable at inference time.
- A separate checkpoint must be trained for each game; cross-game transfer remains unexplored.

## Related Work & Insights

- **vs. SLA/LSPO**: These methods reduce language to a selection among finite candidates, whereas this work optimizes directly in the continuous language space. The Refiner can be stacked on top of such methods.
- **vs. Cicero**: Cicero seeks global equilibria in the game of Diplomacy; this paper employs local Stackelberg optimization to avoid computational intractability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of Stackelberg modeling and persuasion-based reward is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three SDGs + Sotopia, multiple stacked baselines, and complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretically clear, though some sections are notation-dense.
- Value: ⭐⭐⭐⭐ Provides a viable framework for persuasive communication in LLM agents.
**Code**: To be confirmed
**Area**: reinforcement_learning
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](../../ICLR2026/reinforcement_learning/learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ACL 2026\] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)

<!-- RELATED:END -->
