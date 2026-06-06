---
title: >-
  [Paper Note] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games
description: >-
  [ACL 2026][Reinforcement Learning][Persuasive Communication] This paper models turn-based dialogue in Social Deduction Games (SDGs) as a Stackelberg game. The current player acts as a leader to optimize the persuasive im…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Persuasive Communication"
  - "Social Deduction Games"
  - "Stackelberg Game"
  - "GRPO"
  - "LLM Agents"
date: 2026-05-08
content_hash: d059c555c65db347
---

# The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games

**Conference**: ACL 2026  
**arXiv**: [2510.09087](https://arxiv.org/abs/2510.09087)  
**Code**: [https://3dagentworld.github.io/leader_follower](https://3dagentworld.github.io/leader_follower)  
**Area**: Reinforcement Learning / Social Deduction Games  
**Keywords**: Persuasive Communication, Social Deduction Games, Stackelberg Game, GRPO, LLM Agents

## TL;DR

This paper models turn-based dialogue in Social Deduction Games (SDGs) as a Stackelberg game. The current player acts as a leader to optimize the persuasive impact of their discourse by measuring the response distribution of the next player. A Refiner model is trained using GRPO, significantly outperforming baselines across four game benchmarks including Werewolf and Avalon.

## Background & Motivation

**Background**: LLM agents have made significant progress in Social Deduction Games (SDGs) such as Werewolf and Avalon. Existing methods primarily focus on information processing (inferring other players' roles) and strategy selection (choosing optimal actions).

**Limitations of Prior Work**: Existing methods ignore the core role of persuasive communication—in SDGs, success depends not only on making correct inferences but also on persuading others to act according to one's intentions. Existing RL methods (e.g., SLA, LSPO) simplify the rich natural language space into limited action classification problems, failing to optimize discourse in a continuous language space.

**Key Challenge**: The core challenge of SDGs is not "knowing what is right," but "convincing others that one is right." While the persuasive dimension is central to game success and real human interaction, it remains nearly untouched in current research.

**Goal**: To explicitly model and optimize persuasive communication in Social Deduction Games, enabling agents to proactively guide the dialogue flow toward favorable outcomes.

**Key Insight**: Borrowing the Stackelberg game framework from game theory—if a leader fully understands the follower's response distribution to different actions, they can choose the action that maximizes their own utility. In turn-based dialogue, the current speaker is the leader.

**Core Idea**: Training a Refiner model to refine base discourse into more persuasive versions, where the reward signal is based on the shift in the response probability distribution of the next player (increasing the probability of desired responses and decreasing the probability of undesired ones).

## Method

### Overall Architecture

The process consists of three steps: (1) Intent Identification—An API LLM analyzes the current situation to generate K=3 sets of desired/undesired follower responses; (2) Impact Measurement—An API LLM generates base discourse, the Refiner refines it into multiple candidates, and the Measurer calculates the shift in the follower's response distribution for each candidate as a reward; (3) Strategy Optimization—GRPO is used to optimize the Refiner to maximize persuasive impact.

### Key Designs

1. **Stackelberg Modeling and Intent Identification**:

    - **Function**: Models each speaking turn as a leader-follower interaction to clarify optimization goals.
    - **Mechanism**: The current player $p_t$ acts as the leader, and the next player $p_{t+1}$ acts as the follower. Based on game rules $\mathcal{R}$, game state $G_t$, dialogue history $D_t$, and hidden role $r_t$, the leader uses a backend LLM to generate K=3 sets of desired responses $\hat{u}_{t+1}^{+,(k)}$ and undesired responses $\hat{u}_{t+1}^{-,(k)}$.
    - **Design Motivation**: To explicitly define "what constitutes a good persuasive effect," transforming vague persuasion goals into measurable probability shifts.

2. **Impact Measurement**:

    - **Function**: Calculates a persuasive reward for each candidate discourse based on its effect on the follower's behavior.
    - **Mechanism**: Qwen2.5-72B is used as a Measurer to simulate the follower's response patterns. For a candidate discourse $u_t^{(i)}$, the reward $R(u_t^{(i)}) = \sum_k \log P_\mathcal{F}(\hat{u}_{t+1}^{+,(k)} | \text{ctx} \cup \{u_t^{(i)}\}) - \sum_k \log P_\mathcal{F}(\hat{u}_{t+1}^{-,(k)} | \text{ctx} \cup \{u_t^{(i)}\})$.
    - **Design Motivation**: Measuring persuasive effects directly in the follower's probability space is more objective than manual labeling or heuristic evaluation.

3. **GRPO Strategy Optimization**:

    - **Function**: Trains the Refiner to optimize the persuasiveness of discourse within the natural language space.
    - **Mechanism**: Qwen2.5-7B + LoRA (rank 16) serves as the Refiner, sampling n=8 candidates. GRPO computes the relative advantage within the group for strategy optimization, with KL divergence regularization to prevent excessive deviation.
    - **Design Motivation**: GRPO eliminates the need for an additional critic model by utilizing the reward distribution within the batch to calculate relative advantages.

### Loss & Training

The GRPO objective function is: $\mathcal{J}(\theta) = \mathbb{E}_c[\frac{1}{n}\sum_i \mathcal{L}_i - \beta D_{KL}(\pi_\theta || \pi_{ref})]$, with n=8, ε=0.2, and β=0.04. Training involves 4,000 instances selected from 500 self-play games per game type. Backend LLMs are randomly selected from GPT-4o/Gemini-2.5-Flash/Claude-3.5-Haiku. Training was conducted with a learning rate of $1 \times 10^{-6}$ on 4×A800 for 3 epochs (approximately 50 hours).

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
| ReAct (Baseline) | 49.0 | 44.0 | 48.0 |
| Pos-Only + ReAct | 64.0 | 58.0 | 60.0 |
| Neg-Only + ReAct | 49.0 | 46.0 | 47.0 |
| **Ours + ReAct** | **70.0** | **61.0** | **61.0** |

### Key Findings

- Positive rewards (increasing desired response probability) contribute far more than negative rewards (decreasing undesired response probability).
- The Refiner performs better when combined with strong baselines, indicating the method complements rather than replaces existing strategies.
- Improvement is particularly significant for deceptive roles—in Werewolf, the Werewolf win rate increased from 79% to 84.2%.
- The method generalizes successfully to the Sotopia social simulation environment, beyond just SDGs.

## Highlights & Insights

- Modeling turn-based dialogue as a Stackelberg game is highly natural—quantifying persuasion as the "shift in the opponent's response probability" is more granular than directly optimizing for win rates.
- Using an independent LLM to simulate the follower's response distribution cleverly bypasses the limitation of API LLMs not providing probabilities.
- Positioning the Refiner as a "discourse polisher" is practical—it retains the semantic understanding of strong API LLMs while using smaller models for persuasive enhancement.

## Limitations & Future Work

- The Measurer uses a fixed LLM to simulate the follower, while actual opponent behavior may vary.
- Full information (knowledge of opponent roles) is used during training but is unavailable during inference.
- A separate checkpoint must be trained for each game; cross-game transfer has not been explored.

## Related Work & Insights

- **vs SLA/LSPO**: These methods simplify language to a finite set of candidates, whereas this work optimizes directly in the continuous language space. The Refiner can be used as a supplementary layer.
- **vs Cicero**: Cicero seeks a global equilibrium in Diplomacy, whereas this work uses local Stackelberg optimization to avoid computational intractability.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The combination of Stackelberg modeling and persuasive rewards is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three SDGs plus Sotopia, evaluates multiple baseline integrations, and includes complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ The theory is clear, though some sections are formula-dense.
- **Value**: ⭐⭐⭐⭐ Provides a feasible framework for persuasive communication in LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning in Structured Stackelberg Games](../../ICML2026/reinforcement_learning/learning_in_structured_stackelberg_games.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](../../ICLR2026/reinforcement_learning/learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](../../ICLR2026/reinforcement_learning/nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[ACL 2026\] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)
- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)

</div>

<!-- RELATED:END -->
