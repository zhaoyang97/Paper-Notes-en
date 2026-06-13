---
title: >-
  [Paper Note] TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents
description: >-
  [AAAI 2026][Reinforcement Learning][LLM agents] This paper introduces TowerMind, a lightweight multimodal environment based on tower defense games for evaluating LLMs' long-term planning and decision-making capabilities.…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "LLM agents"
  - "tower defense game"
  - "real-time strategy game"
  - "benchmark"
  - "multimodal evaluation"
date: 2026-05-08
content_hash: 08e8737fdc0e38a0
---

# TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents

**Conference**: AAAI 2026
**arXiv**: [2601.05899](https://arxiv.org/abs/2601.05899)  
**Code**: [https://github.com/tb6147877/TowerMind](https://github.com/tb6147877/TowerMind)  
**Area**: Reinforcement Learning
**Keywords**: LLM agents, tower defense game, real-time strategy game, benchmark, multimodal evaluation

## TL;DR

This paper introduces TowerMind, a lightweight multimodal environment based on tower defense games for evaluating LLMs' long-term planning and decision-making capabilities. It reveals a significant performance gap between current LLMs and human experts (the best model achieves only 42% of human expert scores) and identifies behavioral deficiencies including insufficient plan verification, lack of multi-goal thinking, and underutilization of the action space.

## Background & Motivation

**The need to evaluate core LLM agent capabilities**: Cross-domain LLM applications (healthcare, office automation, design) all rely on two fundamental capabilities — **long-term planning** (decomposing high-level tasks into sub-goal sequences) and **decision-making** (translating sub-goals into executable actions).

**RTS games as ideal test platforms**: Real-time strategy games simultaneously require macro-level strategic planning and micro-level tactical adaptation, making them naturally suited for evaluating both capabilities. However, existing RTS game environments present several issues:

**Excessive computational requirements**: StarCraft II-based environments such as TextStarCraft II and LLM-PySC2 require approximately 30 GB of disk space, 2 GB of memory, and a dedicated GPU.

**Lack of text observation support**: Lightweight RTS environments such as ELF, DeepRTS, and Gym-μRTS do not support text-based observations and action interfaces, rendering them incompatible with LLMs.

**Advantages of tower defense games**: Tower defense games share the core mechanics of RTS games but require only defending against predefined enemy waves, eliminating the confounding factor of opponent unpredictability. This enables more isolated evaluation of LLMs' planning and decision-making capabilities. Fixed tower placement options and predefined routes also facilitate analysis of LLM strategic choices.

## Method

### Overall Architecture

TowerMind is a tower defense game environment built on the Unity game engine and extended into an AI environment via the Unity ML-Agents Toolkit, following the OpenAI Gym standard interface. Key characteristics:

- **Lightweight**: Requires only 0.15 GB of disk space and memory; runs on CPU (vs. SC2LE's 30 GB + GPU)
- **Multimodal observations**: Supports pixel images (512×512×3), text (JSON format), and structured game state observations
- **Hallucination evaluation**: Simultaneously evaluates score (capability) and valid action rate (reliability)
- **Highly customizable**: Provides a graphical level editor

### Key Designs

#### 1. Game Mechanics Design

TowerMind maps are defined within a square area of side length 6 centered at (0,0), comprising:

- **Roads**: Red/blue directional curves guiding enemy movement, represented as sequences of 2D coordinate waypoints
- **Tower slots**: Predefined tower placement positions; some slots far from roads serve as "decoy positions"
- **Three tower types**: Archer tower (high single-target damage), Mage tower (area-of-effect attack), Knight tower (summons melee knights)
- **Hero units**: Powerful units with fine-grained control over movement and skills
- **15 enemy types**: Varying HP, speed, and attack power; some with special abilities (e.g., Orc Wizards can disable nearby towers)
- **Fog of war**: Randomly moving white cloud regions introducing partial observability

Design motivation: Road shape diversity, tower slot distribution differences, enemy composition variation, and fog of war collectively create a rich decision space that cannot be solved by a single fixed strategy.

#### 2. Hybrid Action Space

Actions are represented as three-dimensional vectors $a = (x, y, c)$:

- $(x, y) \in [-3.0, 3.0]$: Continuous spatial coordinates
- $c \in \{0, 1, 2, \ldots, 11\}$: Discrete action type (12 types including tower construction, upgrade, sell, deploy knight reinforcements, etc.)

Only actions that comply with game rules and the current state are considered "valid actions"; otherwise they are classified as "invalid actions" and not executed. This design makes the **valid action rate** a direct metric for measuring LLM hallucination.

#### 3. Difficulty Quantification System

The difficulty of level $l$ is defined as: $D(l) = d_r(l) + d_t(l) + d_e(l) + d_{re}(l)$

- Road difficulty $d_r$: Number of roads / maximum number of roads
- Tower slot difficulty $d_t$: Number of tower slots / maximum number of tower slots
- Enemy difficulty $d_e$: Enemy type ratio + average number of enemies per wave ratio
- Resource difficulty $d_{re}$: Composite of initial gold, gold drop amount, and tower sell recovery ratio

Design motivation: Provides a quantitative difficulty modeling approach enabling researchers to compare the evaluative significance of different levels.

#### 4. Evaluation Metric Design

- **Score**: Each enemy reaching the player's base incurs a penalty of $-1.0$, with range $[-20, 0]$
- **Valid action rate**: Number of valid actions / total number of actions, range $[0, 1]$

The two metrics independently evaluate LLMs' **capability** and **reliability** (degree of hallucination), measuring "correctness" and "validity" separately.

### Loss & Training

For RL algorithms (Ape-X DQN and PPO), a sparse reward signal ($-1.0$ when an enemy reaches the base) is used, with one action executed every 16 game steps (approximately 187 APM). Training uses 100 million environment steps.

For LLM evaluation, a zero-shot prompting strategy is adopted with identical prompts across all models, evaluated under both language-only and vision-language modality settings.

## Key Experimental Results

### Main Results

**LLM score performance (normalized to human expert baseline)**:

| Model | Lv1 | Lv2 | Lv3 | Lv4 | Lv5 | Avg |
|-------|-----|-----|-----|-----|-----|-----|
| Claude 3.7 Sonnet (L) | 0.62 | 0.51 | 0.40 | 0.24 | 0.15 | **0.38** |
| GPT-4.1 (VL) | 0.63 | 0.56 | 0.44 | **0.32** | 0.15 | **0.42** |
| Claude 3.7 Sonnet (VL) | **0.67** | **0.58** | **0.45** | 0.20 | **0.16** | 0.41 |
| Gemini-2.5-Pro (L) | 0.52 | 0.42 | 0.31 | 0.11 | 0.01 | 0.27 |
| Qwen 2.5-VL 72B (L) | 0.47 | 0.36 | 0.21 | 0.00 | 0.00 | 0.21 |
| Llama 3.2 90B (L) | 0.42 | 0.32 | 0.19 | 0.12 | 0.00 | 0.21 |
| Qwen 2.5-VL 7B (L) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**LLM valid action rate performance (normalized to human expert baseline)**:

| Model | Lv1 | Lv2 | Lv3 | Lv4 | Lv5 | Avg |
|-------|-----|-----|-----|-----|-----|-----|
| GPT-4.1 (L) | **0.92** | 0.89 | 0.88 | 0.84 | 0.75 | 0.86 |
| Gemini-2.5-Pro (L) | 0.91 | **0.90** | **0.89** | 0.83 | **0.82** | **0.87** |
| Claude 3.7 Sonnet (VL) | 0.85 | **0.85** | **0.83** | **0.80** | **0.79** | **0.82** |
| Qwen 2.5-VL 7B (L) | 0.11 | 0.05 | 0.03 | 0.01 | 0.01 | 0.04 |
| Random baseline | 0.25 | 0.25 | 0.24 | 0.24 | 0.22 | 0.24 |

### Ablation Study / RL Baselines

**RL algorithm evaluation results**:

| Configuration | Lv1–5 Performance | Notes |
|---------------|-------------------|-------|
| Ape-X DQN (pixels) | Partially solves easier levels | Remains significantly below human experts after 100M steps |
| PPO (pixels) | Partially solves easier levels | Indicates TowerMind is a challenging RL environment |
| PPO (structured state) | Slightly outperforms pixel input | Structured observations provide more effective information |

### Key Findings

1. **Large gap between LLMs and human experts**: The best-performing LLM (Claude 3.7 Sonnet VL) achieves only approximately 42% of the human expert score, with the gap exceeding 84% on the hardest level.
2. **Visual input improves performance**: With the exception of Llama 3.2, all models perform better in the vision-language modality than in the language-only modality.
3. **Severe hallucination in open-source LLMs**: The valid action rates of Qwen 2.5-VL 7B and Llama 3.2 11B fall even below the random baseline.
4. **Increased difficulty exacerbates hallucination**: Harder levels contain more game elements, resulting in longer prompts that challenge the generation stability of LLMs.
5. **Three major behavioral deficiencies**:
    - **Insufficient plan verification**: LLMs place towers at decoy positions that cannot attack enemies, despite having sufficient information to reason that these positions are ineffective.
    - **Lack of multi-goal thinking**: Human experts simultaneously accomplish multiple objectives (e.g., collecting gold while attacking enemies); LLMs never exhibit this behavior.
    - **Underutilization of the action space**: LLMs neglect to upgrade towers (even when gold is sufficient), send knights to unoccupied areas, or use hero skills in the absence of enemies.

## Highlights & Insights

1. **Practical value of the lightweight design**: The substantial difference of 0.15 GB vs. 30 GB makes large-scale parallel evaluation feasible.
2. **Elegant design for hallucination evaluation**: The invalid action rate serves as a direct measure of hallucination without requiring additional annotation.
3. **Experimental design with decoy tower slots**: Placing tower slots far from roads tests whether LLMs can perform spatial reasoning to identify ineffective options.
4. **Insight on "correct but invalid" behavior**: The gap between LLMs' valid action rate and score is analogous to the "technically correct but practically useless" problem observed in question-answering tasks.
5. **Extensibility of the level editor**: Enables researchers to create custom levels, supporting diverse research needs and reducing the risk of data contamination.

## Limitations & Future Work

1. **Representativeness of tower defense vs. full RTS**: Tower defense is only a subclass of RTS games and lacks adversarial (PvP) evaluation.
2. **Limitations of zero-shot evaluation**: Few-shot or fine-tuned LLM performance on TowerMind has not been explored.
3. **Insufficient depth of RL baselines**: Only two basic RL algorithms are tested; more advanced methods are not included.
4. **Information density of text observations**: JSON-format text observations may be overly verbose, affecting LLM processing efficiency.
5. **Static enemy waves**: Current enemy configurations are fixed; future work could introduce greater randomness and dynamic adjustments.

## Related Work & Insights

- **SC2LE / TextStarCraft II / LLM-PySC2**: Heavyweight RTS environments based on StarCraft II.
- **ELF / DeepRTS / Gym-μRTS**: Lightweight RTS environments without text support.
- **AGENTBENCH** (Liu et al., 2023): A comprehensive evaluation benchmark for LLMs in interactive environments.
- **ReAct / AutoGPT**: LLM planning and reasoning frameworks.
- Insight: LLM evaluation should shift from "static correctness" toward "interactive validity"; TowerMind provides a practical platform for this purpose.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first tower defense game environment designed for LLM evaluation, filling the gap in lightweight RTS environments.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 7 LLMs + 2 RL algorithms + human baseline with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rich figures and tables, and a good combination of quantitative and qualitative analysis.
- Value: ⭐⭐⭐⭐ — Provides a practical and extensible new tool for LLM agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](../../ACL2026/reinforcement_learning/dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](../../ICLR2026/reinforcement_learning/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](aligning_machiavellian_agents_behavior_steering_via_test-tim.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](../../ICML2026/reinforcement_learning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[ICLR 2026\] Don't Just Fine-tune the Agent, Tune the Environment](../../ICLR2026/reinforcement_learning/dont_just_fine-tune_the_agent_tune_the_environment.md)

</div>

<!-- RELATED:END -->
