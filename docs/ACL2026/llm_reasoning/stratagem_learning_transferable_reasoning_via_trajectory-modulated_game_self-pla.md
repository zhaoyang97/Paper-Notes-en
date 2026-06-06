---
title: >-
  [Paper Note] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play
description: >-
  [ACL2026][LLM Reasoning][self-play] Stratagem shifts away from strengthening models based solely on win/loss outcomes in text game self-play. Instead…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "self-play"
  - "Transferable Reasoning"
  - "Trajectory Advantage Modulation"
  - "Mathematical Reasoning"
  - "Code Generation"
date: 2026-05-08
content_hash: 13a0f77007f8d306
---

# Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play

**Conference**: ACL2026  
**arXiv**: [2604.17696](https://arxiv.org/abs/2604.17696)  
**Code**: https://github.com/ydyyyy/Stratagem  
**Area**: Code Intelligence / LLM Reasoning / Self-play Reinforcement Learning  
**Keywords**: self-play, Transferable Reasoning, Trajectory Advantage Modulation, Mathematical Reasoning, Code Generation

## TL;DR
Stratagem shifts away from strengthening models based solely on win/loss outcomes in text game self-play. Instead, it modulates the advantage using two trajectory-level signals—"abstract transferability" and "reasoning evolution"—ensuring that strategies learned from games are more transferable to mathematics, general reasoning, and code generation tasks.

## Background & Motivation
**Background**: Training agents using games is a classic reinforcement learning approach recently applied to LLM training. Methods like SPIRAL involve language models in zero-sum text game self-play, updating policies through end-game win/loss signals with the hope that planning, probabilistic judgment, and decision-making capabilities will transfer to math and code tasks.

**Limitations of Prior Work**: End-game outcomes only indicate whether a round was won or lost; they fail to distinguish which reasoning steps within a winning trajectory are transferable abstract strategies versus game-specific heuristics. For instance, memorizing rules like "King beats Queen" helps win a game but does not necessarily assist in solving math problems.

**Key Challenge**: While self-play generates rich trajectories, traditional advantage signals focus only on the game outcome. Consequently, models may reinforce domain-specific heuristics rather than prioritizing transferable skills such as enumeration, proof by contradiction, condition decomposition, and probabilistic reasoning.

**Goal**: The authors aim to sift out reasoning patterns from game trajectories that truly transfer to downstream tasks without relying on human reasoning data, biasing training signals toward these trajectories.

**Key Insight**: The paper attributes transfer failure to two reasons: domain specificity, which keeps the model trapped in game semantics, and contextual stasis, where the model lacks the ability to deepen reasoning as the context progresses.

**Core Idea**: Add trajectory-level modulation to the SPIRAL game advantage. Trajectories with high abstraction retain a larger advantage, while those where reasoning continuously deepens across multiple rounds receive an additional reward.

## Method
The Method of Stratagem is constrained: it does not redesign game environments or mix math problems into training. Instead, it modifies the training weights of self-play trajectories. Given a game trajectory, while the original SPIRAL calculates the game advantage based on the final outcome and a role baseline, Stratagem modulates it using two signals to obtain a modified advantage for policy gradient updates.

### Overall Architecture
The training environment consists of three text-based zero-sum games: Tic-Tac-Toe, Kuhn Poker, and Simple Negotiation. These cover spatial planning, probabilistic reasoning, and strategy optimization, respectively. Both players share the same LLM policy, distinguished by role-conditioned prompts.

Each trajectory includes multi-turn states, model responses, reasoning text, and actions. Whereas SPIRAL calculates a role-conditioned advantage based strictly on the outcome, Stratagem computes a Reasoning Transferability Coefficient and a Reasoning Evolution Reward, updating the model using the formula: $A_{mod} = A_{game} \cdot \phi + \beta \cdot \psi$.

Experiments use Qwen3-4B-Base as the base model. $\phi$ and $\psi$ are scored by GPT-4 acting as an evaluator, with $\beta$ set to 0.2 by default. The authors control evaluation costs via trajectory subsampling; a single training run takes approximately 30 GPU-hours on 2 A100s, with GPT-4 scoring costing about $100.

### Key Designs
1. **Reasoning Transferability Coefficient**:
    - **Function**: Measures whether the reasoning in a game trajectory is abstract, domain-agnostic, and transferable.
    - **Mechanism**: $\phi$ takes values of 0, 0.5, or 1. When game-specific rules and fixed routines dominate, the trajectory weight is suppressed; when abstract reasoning such as scenario enumeration, probability estimation, or goal-constraint decomposition appears, the trajectory retains a higher advantage.
    - **Design Motivation**: Win/loss rewards reinforce all behaviors that lead to a win, whereas transfer training needs to prioritize the reasoning structures *why* a win was possible.

2. **Reasoning Evolution Reward**:
    - **Function**: Rewards reasoning that updates, deepens, and remains self-consistent as the game progresses.
    - **Mechanism**: $\psi$ takes values of -1, 0, or +1, evaluating whether the reasoning adjusts strategies based on new states, maintains multi-step consistency, and evolves from shallow reactions into complete plans.
    - **Design Motivation**: Contexts in math and code tasks change continuously as intermediate results emerge. Without rewarding "reasoning evolution," static game training tends to learn fixed templates.

3. **Trajectory-Modulated Advantage Training**:
    - **Function**: Converts the two trajectory quality signals into training weights for policy gradients.
    - **Mechanism**: $A_{game} \cdot \phi$ acts as a multiplicative gate, ensuring low-transferability trajectories have minimal impact even if they win. $\beta \cdot \psi$ acts as an additive reward, giving evolving reasoning trajectories extra signals beyond the win/loss.
    - **Design Motivation**: The multiplicative term addresses "learning the wrong heuristics," while the additive term addresses "learning only static reactions." Together, they shift self-play from winning games to learning reasoning.

### Loss & Training
Training remains based on self-play policy gradients. The difference lies in replacing the update advantage $A_{game}$ with $A_{mod}$. After sampling game trajectories in each round, the system calculates the player's final payoff, role baseline, $\phi$, and $\psi$, and then computes the log-probability gradient for the player's generated responses.

Since downstream benchmarks are not used as training rewards, improvements in math, general reasoning, and code can be interpreted as cross-domain transfer rather than direct optimization on target tasks. This setup clarifies the core argument: whether game training transfers depends on the type of reasoning reinforced in the trajectories.

## Key Experimental Results

### Main Results
Evaluations cover mathematical reasoning, general reasoning, and code generation. All assessments use zero-shot prompting, and code tasks utilize HumanEval pass@1.

| Model | MATH500 | AIME24 | AIME25 | AMC-23 | GPQA | MMLU-Pro | HumanEval |
|------|---------|--------|--------|--------|------|----------|-----------|
| Qwen3-4B-Base | 65.80 | 10.00 | 3.30 | 50.00 | 30.60 | 47.20 | 67.93 |
| SPIRAL | 71.00 | 10.00 | 6.70 | 45.00 | 36.41 | 53.93 | 77.44 |
| Stratagem | 76.00 | 20.00 | 13.30 | 60.00 | 38.23 | 57.83 | 77.93 |
| Stratagem vs Base | +10.20 | +10.00 | +10.00 | +10.00 | +7.63 | +10.63 | +10.00 |
| Stratagem vs SPIRAL | +5.00 | +10.00 | +6.60 | +15.00 | +1.82 | +3.90 | +0.49 |

The strongest gains are concentrated in competition math: AIME24 improved from 10.00 to 20.00, AIME25 from 3.30 to 13.30, and AMC-23 from 50.00 to 60.00. This aligns with the hypothesis that complex multi-step reasoning relies more on transferable strategic structures than on simple knowledge recall.

### Ablation Study

| Configuration | MATH500 | AIME24 | AIME25 | OlympiadBench | AMC-23 | GPQA | MMLU-Pro | HumanEval |
|------|---------|--------|--------|---------------|--------|------|----------|-----------|
| Stratagem full | 76.00 | 20.00 | 13.30 | 39.90 | 60.00 | 38.23 | 57.83 | 77.93 |
| w/o Evolution Reward | 74.60 | 13.30 | 10.00 | 39.30 | 52.50 | 37.22 | 56.92 | 77.80 |
| Gain (full - w/o) | +1.40 | +6.70 | +3.30 | +0.60 | +7.50 | +1.01 | +0.91 | +0.13 |

| Human Eval Dimensions | Qwen3-4B-Base | SPIRAL | Stratagem w/o evolution | Stratagem |
|----------------|--------------|--------|--------------------------|-----------|
| Reasoning Abstraction | 2.48 | 3.24 | 3.82 | 4.06 |
| Reasoning Progression | 2.32 | 3.08 | 3.36 | 4.18 |

### Key Findings
- $\psi$ contributes most significantly to AIME24 and AMC-23, indicating that "whether reasoning progresses with the game state" is particularly vital for competition math-style tasks.
- Parameter sensitivity shows $\beta = 0.20$ is the overall optimal point; a $\beta$ that is too small approaches removing the evolution reward, while a $\beta$ that is too large allows evolution scores to overshadow the original game objective, leading to unstable training.
- Human evaluation indicates that while abstraction remains high even without $\psi$, progression drops significantly, supporting the idea that $\phi$ and $\psi$ correspond to two distinct capability dimensions.

## Highlights & Insights
- The paper identifies a core difficulty in self-play transfer: it is not about whether games can generate data, but which game trajectories are worth learning. This issue is more fundamental than simply scaling the number of games.
- The combination of $\phi$ and $\psi$ is intuitive. One filters out domain-specific heuristics, while the other rewards dynamic reasoning processes, corresponding exactly to "ability to abstract" and "ability to progress."
- While code generation gains were less significant than math, HumanEval scores still exceeded SPIRAL, suggesting that structured planning and constraint satisfaction in games may have weak transfer to program synthesis.
- This paper provides a more general training paradigm: when environmental rewards cover only outcomes, a trajectory-level meta-evaluator can weight process quality without requiring manual step-by-step labels.

## Limitations & Future Work
- Trajectory quality scoring relies on GPT-4, introducing issues regarding cost, bias, and reproducibility; the trained model may inherit specific reasoning style preferences from the evaluator.
- The training environment is limited to three text games, covering a restricted range of reasoning types. Whether this is effective for complex tool use, code debugging, or multi-agent collaboration remains to be verified.
- $\phi$ and $\psi$ use discrete levels, which limits expressiveness. Real trajectories may contain both transferable strategies and game-specific tricks simultaneously; coarse-grained scoring may lose these details.
- The method does not directly prove that internal model representations have abstracted from game semantics to mathematical semantics; current evidence primarily comes from downstream scores and manual trajectory ratings.

## Related Work & Insights
- **vs SPIRAL**: SPIRAL uses only end-game outcomes to update policies. Stratagem adds trajectory quality modulation within the same self-play framework, explaining why certain winning trajectories are more valuable to learn.
- **vs Absolute Zero / Self-play Reasoning**: These works emphasize self-generated training without external data. Stratagem focuses on reward shaping, specifically separating transferable reasoning signals from environmental outcome rewards.
- **vs RLHF / RLAIF**: RLHF typically evaluates response preferences. Stratagem evaluates the abstraction and evolution of the entire interaction trajectory, a granularity better suited for multi-step reasoning training.
- **Insight**: In the field of Code Intelligence, unit test pass rates could be treated as $A_{game}$, with "solution abstraction" and "debugging progression" introduced as trajectory modulation signals to prevent models from learning specific hacks for test cases.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explains and improves game-to-reasoning transfer using trajectory modulation; the idea is clear and inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers math, general reasoning, code, and human evaluation, though game environments and base model scales remain somewhat limited.
- Writing Quality: ⭐⭐⭐⭐☆ Problem definitions and formulas are concise; some details regarding $\phi$ and $\psi$ scoring could be more transparent.
- Value: ⭐⭐⭐⭐☆ Highly relevant for self-play training and RL for code reasoning, especially for tasks with sparse outcome rewards but rich trajectories.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain](../../ICML2026/llm_reasoning/self-play_only_evolves_when_self-synthetic_pipeline_ensures_learnable_informatio.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](../../ICML2026/llm_reasoning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[ACL 2026\] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)

</div>

<!-- RELATED:END -->
