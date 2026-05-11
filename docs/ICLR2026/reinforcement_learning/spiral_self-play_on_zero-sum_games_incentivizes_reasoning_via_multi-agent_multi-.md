---
title: >-
  [Paper Note] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning
description: >-
  [Reinforcement Learning] This paper proposes SPIRAL, a framework that trains LLMs via self-play in multi-turn zero-sum games. Through Role-conditioned Advantage Estimation (RAE) to stabilize training…
tags:
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 239e5e11634a982f
---

# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2506.24119](https://arxiv.org/abs/2506.24119)
- **Code**: [https://github.com/spiral-rl/spiral](https://github.com/spiral-rl/spiral)
- **Area**: Reinforcement Learning
- **Keywords**: self-play, zero-sum games, multi-agent RL, reasoning, LLM, transfer learning

## TL;DR
This paper proposes SPIRAL, a framework that trains LLMs via self-play in multi-turn zero-sum games. Through Role-conditioned Advantage Estimation (RAE) to stabilize training, SPIRAL improves reasoning performance by up to 10% without domain-specific data, and reveals that different games cultivate complementary cognitive abilities.

## Background & Motivation
- **Bottleneck of RLVR**: Current RL approaches for improving LLM reasoning rely on manually designed reward functions and domain-specific datasets (e.g., math problems), limiting scalability.
- **Potential of Self-Play**: From TD-Gammon to AlphaGo, self-play has achieved great success in traditional AI, yet its application to improving LLM reasoning remains largely unexplored.
- **Limitations of Fixed Opponents**: Training models against fixed opponents (e.g., Mistral/Gemini) leads to overfitting to static strategies (Figure 2).
- **Technical Challenges**: Multi-turn multi-agent autoregressive generation incurs substantial computational demands, and standard RL exhibits high variance in multi-agent settings.

## Method

### Overall Architecture
SPIRAL = Multi-game multi-turn zero-sum self-play + Distributed Actor-Learner architecture

Game set $\mathcal{G} = \{G_1, G_2, ..., G_n\}$, comprising:
- **TicTacToe**: Spatial reasoning
- **Kuhn Poker**: Probabilistic reasoning
- **Simple Negotiation**: Strategic optimization

### Self-Play Mechanism
- A single shared policy $\pi_\theta$, conditioned on roles via system prompts (Player 0 / Player 1)
- At each turn, the active player generates a full response $y_t^{(p)} \sim \pi_\theta(\cdot | s_t, p, G_i)$
- Actions are extracted from responses to update the game state
- Zero-sum property: $R_0(\tau) + R_1(\tau) = 0$, with rewards assigned only at game termination

### Key Design: Role-conditioned Advantage Estimation (RAE)
In zero-sum games, the same model optimizes opposing objectives simultaneously; using a global baseline directly causes training instability. RAE maintains independent baselines for each game–role pair:

$$b_{G,p} \leftarrow \alpha \cdot b_{G,p} + (1-\alpha) \cdot R_p(\tau)$$
$$A_{G,p}(\tau) = R_p(\tau) - b_{G,p}$$

Variance-reduced policy gradient:

$$\nabla_\theta J_{\text{SPIRAL}}(\theta) = \mathbb{E}_{G \sim \mathcal{G}} \mathbb{E}_{\tau \sim \pi_\theta \times \pi_\theta | G} \left[\sum_{p \in \{0,1\}} \sum_{t \in T_p} A_{G,p}(\tau) \cdot \nabla_\theta \log \pi_\theta(y_t^{(p)} | s_t, p, G)\right]$$

### Why Is RAE Critical?
- Different roles may have different expected returns due to game asymmetry (e.g., first-mover advantage in TicTacToe).
- Without RAE, the model progressively abandons reasoning after approximately 200 steps (thinking collapse), generating empty chain-of-thought traces.
- RAE eliminates positional bias interference through role-specific normalization.

### Engineering Implementation
- Distributed Actor-Learner architecture built on the Oat framework
- vLLM for efficient inference; TextArena for game simulation
- Full-parameter online updates (not LoRA), fully online (not offline)

## Key Experimental Results

### Main Results: Reasoning Benchmark Performance

| Model | Math500 | AIME24 | AIME25 | AMC-23 | GPQA-D | Avg. |
|------|---------|--------|--------|--------|--------|------|
| Qwen3-4B-Base | 73.4 | 9.6 | 6.2 | 42.4 | 30.6 | 34.0 |
| + SFT-Multi | 74.2 | 13.7 | 11.7 | 51.1 | 37.8 | 39.7 |
| + **SPIRAL-Multi** | **78.2** | **19.7** | **13.3** | **61.6** | **40.1** | **44.5** |
| | +4.8 | +10.1 | +7.1 | +19.2 | +9.5 | **+10.5** |

| Model | Avg. Baseline | + SPIRAL-Multi | Gain |
|------|---------|---------------|------|
| Qwen3-4B-Base | 34.0 | **44.5** | +10.5 |
| Qwen3-8B-Base | 39.5 | **49.6** | +10.1 |
| Octothinker-8B-Base | 25.8 | **33.8** | +8.0 |
| Llama-3.1-8B-Instruct | — | — | +2.0 |

> Multi-game SPIRAL outperforms SFT on 25K expert trajectories; DeepSeek-R1-Distill models also benefit.

### Ablation Study: Contribution of Individual Games (Qwen3-4B-Base)

| Training Setting | Math500 | AIME24 | Minerva | Avg. |
|---------|---------|--------|---------|------|
| SPIRAL-TicTacToe | 76.0 | 15.0 | 38.2 | ~40 |
| SPIRAL-Kuhn | 76.4 | 18.2 | 42.4 | 43.4 |
| SPIRAL-Negotiation | 75.8 | 14.5 | 39.0 | ~39 |
| **SPIRAL-Multi** | **78.2** | **19.7** | **42.6** | **44.5** |

> Different games cultivate complementary abilities: TicTacToe → spatial reasoning; Kuhn Poker → probabilistic reasoning; Negotiation → strategic optimization. Multi-game combination yields synergistic effects.

### Key Findings
1. Self-play consistently improves performance across four distinct model families (Qwen3, Llama, Octothinker).
2. Multi-game training > single-game training > SFT on expert trajectories > fixed-opponent training.
3. RAE is critical for training stability — its absence causes thinking collapse.
4. Analysis of CoT traces reveals three reasoning patterns transferred from games to mathematics: case-by-case analysis, expected value computation, and pattern recognition.
5. The adaptive curriculum provided by self-play is essential — fixed-opponent training fails.

## Highlights & Insights
- **Zero Human Supervision**: No math problems or domain-specific data are required; games automatically generate unlimited training data.
- **Transferability Finding**: Reasoning patterns learned in games (case analysis, probability estimation) transfer to academic reasoning benchmarks.
- **Necessity of RAE**: Elegantly addresses the variance problem in multi-agent zero-sum training and prevents thinking collapse.
- **Complementary Skills**: Different games cultivate distinct cognitive abilities; multi-game synergy outperforms any single game.

## Limitations & Future Work
- Only three relatively simple games are evaluated; the effectiveness of more complex games (e.g., Diplomacy) remains unknown.
- Computational overhead is substantial: multi-turn multi-agent autoregressive generation requires significant GPU resources.
- Analysis of transfer mechanisms remains post-hoc and qualitative, lacking rigorous theoretical explanation.
- Gains on already highly optimized instruct models are limited (Llama-3.1-8B-Instruct: only +2.0).

## Related Work & Insights
- **LLM RL Reasoning**: OpenAI o1, DeepSeek-R1, GRPO (Shao et al., 2024)
- **LLM Self-Play**: SPAG (Cheng et al., 2024) — single-game offline; Absolute Zero (Zhao et al., 2025) — single-turn programming
- **LLMs in Games**: RAGEN (Wang et al., 2025b), ViGaL (Xie et al., 2025b)
- **Multi-Agent RL**: Cicero (FAIR et al., 2022) — focuses on superhuman performance in a single game

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Multi-game multi-turn zero-sum self-play for general reasoning represents an entirely new paradigm
- **Theoretical Depth**: ⭐⭐⭐ — RAE is intuitively motivated but lacks rigorous theoretical analysis
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 model families × 8 reasoning benchmarks × detailed ablations × CoT analysis
- **Practical Value**: ⭐⭐⭐⭐ — Improves reasoning without domain-specific data, though computational cost is relatively high

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)

</div>

<!-- RELATED:END -->
