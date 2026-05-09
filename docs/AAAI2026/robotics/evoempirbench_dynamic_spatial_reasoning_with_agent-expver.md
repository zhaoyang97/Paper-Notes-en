---
title: >-
  [Paper Note] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer
description: >-
  [AAAI 2026][Robotics][Dynamic Spatial Reasoning] This paper proposes EvoEmpirBench (EEB), comprising two dynamic interactive benchmarks (partially observable maze navigation + Match-2), and the Agent-ExpVer three-agent online learning framework (GeoLink for interaction + InsightForce for experience abstraction + TruthWeaver for knowledge management). Through a cognitive cycle of "experience → verification → truth induction," the framework achieves continuous strategy evolution without parameter updates, improving GPT-4.1 success rate by 5.6% and Qwen-32B by 29%.
tags:
  - AAAI 2026
  - Robotics
  - Dynamic Spatial Reasoning
  - Partial Observability
  - Online Learning
  - Experience Verification
  - Maze Navigation
date: 2026-05-08
content_hash: 4c3e2cbf047521b3
---

# EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer

**Conference**: AAAI 2026
**arXiv**: [2509.12718](https://arxiv.org/abs/2509.12718)
**Code**: [https://anonymous.4open.science/r/EvoEmpirBench-143C/](https://anonymous.4open.science/r/EvoEmpirBench-143C/)
**Area**: Robotics
**Keywords**: Dynamic Spatial Reasoning, Partial Observability, Online Learning, Experience Verification, Maze Navigation

## TL;DR
This paper proposes EvoEmpirBench (EEB), comprising two dynamic interactive benchmarks (partially observable maze navigation + Match-2), and the Agent-ExpVer three-agent online learning framework (GeoLink for interaction + InsightForce for experience abstraction + TruthWeaver for knowledge management). Through a cognitive cycle of "experience → verification → truth induction," the framework achieves continuous strategy evolution without parameter updates, improving GPT-4.1 success rate by 5.6% and Qwen-32B by 29%.

## Background & Motivation

**State of the Field**: Existing LLM reasoning benchmarks (BIG-Bench, PlanBench, etc.) are primarily built on static datasets, making them susceptible to data contamination and subject to rapid performance saturation. Game-based benchmarks (SmartPlay, GameArena) are more engaging, but tend to feature static environments, shallow interactions, or evaluation of only specific capabilities.

**Limitations of Prior Work**: Real-world reasoning requires long-horizon planning in partially observable, dynamically changing environments — each action alters the environment state, requiring agents to continuously update their understanding and strategies. Existing benchmarks rarely evaluate all three dimensions simultaneously: partial observability + dynamic environment + long-horizon reasoning.

**Root Cause**: The conventional paradigm of "collect data → offline training" is ill-suited for dynamic environments, whereas human learning adapts to new situations through continuous abstraction and rule induction (experience → verification → truth). LLM agents lack analogous online learning mechanisms.

**Paper Goals**: (a) Construct a genuinely dynamic, partially observable reasoning benchmark; (b) Design a human cognition-inspired online learning framework that enables agents to improve continuously without parameter updates.

**Starting Point**: Two carefully designed games (maze + Match-2) serve as test environments — each action modifies the environment, and agents can only observe local information. The three-agent collaborative framework is designed based on the principle of human "experiential learning."

**Core Idea**: Replace offline training with a cognitive cycle of "subjective experience → verification → truth induction" to enable parameter-free continual learning in dynamic environments.

## Method

### Overall Architecture
The work consists of two components: (1) the EvoEmpirBench dynamic benchmark construction; and (2) the Agent-ExpVer three-agent online learning framework.

### Key Designs

1. **EvoEmpirBench: Two Dynamic Games**:

    - **Maze Navigation**: A 9×9 grid in which the agent has partial observability (only a local region is visible). The Easy level contains only coins; Medium adds moving monsters; Hard introduces 4 types of items (pickaxe, iron sword, magnet, key) along with monsters and obstacles. Each action (destroying obstacles, picking up items, etc.) alters the environment structure.
    - **Match-2**: An 8×8 board where the agent eliminates ≥2 adjacent same-colored tiles; eliminated tiles cause remaining tiles to fall and new tiles are randomly replenished. The agent must reach a target elimination count for each color within a limited number of steps. Power-ups (row clear, column clear, bomb, hammer) require spending points.
    - **Design Motivation**: The two games evaluate complementary dimensions — the maze tests spatial navigation, risk management, and memory utilization, while Match-2 tests strategic planning, resource optimization, and long-term goal management. The benchmark contains 120 task instances (3 difficulty levels × 30 instances per game).

2. **GeoLink Agent (Environment Interaction)**:

    - **Function**: Directly interacts with the game environment, selecting actions and collecting trajectories at each timestep.
    - **Mechanism**: $a_t \sim \pi_t(\mathbf{s}_t)$, collecting interaction history $\mathcal{H}_{0:T} = \{(\mathbf{s}_0, a_0, r_0), \ldots\}$.
    - The policy $\pi_t$ evolves continuously by integrating accumulated "truth" knowledge: $\pi_t = \pi_0 \cup \bigcup_{e \in \mathcal{M}_{\text{truth}}} e$.

3. **InsightForce Agent (Experience Abstraction + Verification)**:

    - **Function**: Abstracts interaction trajectories into "subjective experiences" and validates their effectiveness through replaying.
    - **Mechanism**: An LLM summarizes the trajectory $\mathcal{H}_{0:T}$ and final metrics $\mathbf{m}$ into an experience $\mathbf{e} = f_{\text{sum}}(\mathcal{H}_{0:T}, \mathbf{m})$. The agent then replays the same level with experience $\mathbf{e}$; if the level is completed and the score improves, the experience is promoted to a "truth": $\mathcal{M}_{\text{truth}} \leftarrow \mathcal{M}_{\text{truth}} \cup \mathbf{e}$ if $P \wedge (S' > S)$.
    - **Design Motivation**: Inspired by human episodic memory — not all experiences are valuable; only those verified to be genuinely effective are worth retaining.

4. **TruthWeaver Agent (Knowledge Management)**:

    - **Function**: Manages the truth knowledge base to prevent redundant accumulation.
    - **Mechanism**: Three operations — (1) merge semantically similar truths (different phrasings with equivalent meaning); (2) remove exact duplicates; (3) insert new truths. This keeps the knowledge base concise and high-quality.
    - **Design Motivation**: As learning rounds accumulate, knowledge grows explosively; a mechanism analogous to human "memory consolidation" is needed to refine and deduplicate entries.

### Policy Rollback Mechanism
If the average score decreases after a policy update ($\Delta < 0$), the framework automatically reverts to the previous policy version and restarts experience abstraction. This ensures the learning process is monotonically non-degrading.

## Key Experimental Results

### Main Results — Maze Navigation

| Model | Success Rate (%) | Avg. Score | Avg. Steps |
|-------|-----------------|-----------|-----------|
| Human | 90.00 | 2914.67 | 20.6 |
| GPT-4.1 | 73.33 | 2562.33 | 34.0 |
| **GPT-4.1 + ExpVer** | **78.89** | **2805.67** | **32.8** |
| DeepSeek-V3 | 61.11 | 1649.78 | 50.6 |
| Qwen2.5-32B | 42.22 | 1122.22 | 38.4 |
| **Qwen2.5-32B + ExpVer** | **54.44** | **1532.33** | **35.8** |
| Llama-3.1-8B | 23.33 | -1213.67 | 54.4 |

### Main Results — Match-2

| Model | Success Rate (%) | Avg. Score |
|-------|-----------------|-----------|
| Human | 86.67 | 350.22 |
| GPT-4.1 | 40.00 | 245.04 |
| **GPT-4.1 + ExpVer** | **53.33** | **234.60** |
| Grok-3 | 42.22 | 246.87 |
| Claude-3.7-Sonnet | 41.11 | 230.33 |
| Qwen2.5-32B | 33.33 | 203.07 |
| **Qwen2.5-32B + ExpVer** | **41.57** | **197.42** |

### Ablation Study

| Configuration | Maze Succ. | Maze Score | Match-2 Succ. | Match-2 Score |
|--------------|-----------|-----------|--------------|--------------|
| GPT-4.1 Baseline | 73.33% | 2562 | 40.00% | 245 |
| GPT-4.1 w/o TruthWeaver | 77.78% | 2765 | 48.89% | 220 |
| **GPT-4.1 + ExpVer (Full)** | **78.89%** | **2806** | **53.33%** | **235** |

### Key Findings
- **All LLMs lag significantly behind humans**: 90% (human) vs. 78.89% (best LLM) on maze; 86.67% vs. 53.33% on Match-2, indicating that dynamic spatial reasoning remains a critical weakness of LLMs.
- **Agent-ExpVer consistently improves all models**: average +5.6% success rate on maze and +13.3% on Match-2, without any parameter updates.
- **Qwen-32B shows the largest improvement**: success rate from 42.22% to 54.44% (+29% relative gain), suggesting ExpVer provides greater benefit to models of moderate capability.
- **Learned "truths" carry concrete semantics**: early-stage agents learn "bold exploration is harmful," which evolves into "survival first" in later stages, demonstrating a human-like learning trajectory.
- **Partial observability is the primary source of difficulty**: providing a global view raises GPT-4.1's success rate from 73% to 93%, confirming that incomplete information is the main bottleneck.
- **Match-2 is more challenging**: baseline LLMs achieve only 33.7% average success rate, as the task demands precise spatial reasoning combined with multi-step lookahead planning.

## Highlights & Insights
- The **cognitive cycle of "experience → verification → truth"** is the most central contribution — rather than simple self-reflection (Reflexion only examines failure causes), this is a complete closed loop of "summarize → replay to verify → promote to reusable knowledge → deduplicate and refine."
- **TruthWeaver's knowledge management** addresses a practical issue: as learning rounds increase, the number of knowledge entries in the prompt grows continuously; without merging and deduplication, the context window becomes saturated with low-quality, redundant knowledge.
- The **policy rollback mechanism** is pragmatically motivated — non-monotonic learning (where early rounds may degrade performance due to erroneous inductions) requires a safeguard.
- The **two games are well-complementary in design**: the maze tests exploration-exploitation balance and risk management, while Match-2 tests precise coordination and long-term planning.

## Limitations & Future Work
- Performance is bounded by the capabilities of the base model — small models (Llama-8B) derive almost no benefit from ExpVer.
- Only two game types are included, lacking more complex scenarios such as temporal reasoning and multi-agent collaboration.
- The "truths" in Agent-ExpVer are prompt extensions in text form, constrained by context window size and thus unable to accumulate indefinitely.
- The game rules are relatively simple (9×9 grid, 8×8 board), which may not fully reflect the complexity of real-world spatial reasoning.
- Direct comparisons with other online learning methods (e.g., Voyager, CLIN) are absent.

## Related Work & Insights
- **vs. SmartPlay**: SmartPlay environments are static; EEB is dynamic with partial observability, making it more realistic.
- **vs. Reflexion**: Reflexion only performs failure reflection, whereas ExpVer implements a complete closed loop of "summarize → verify → truth induction → knowledge management" with cross-level transfer support.
- **vs. Agent-Pro**: Agent-Pro is limited to shallow-interaction settings such as poker and blackjack; EEB's maze and Match-2 require longer-horizon reasoning chains.
- **Implications for agent research**: Parameter-free online learning (pure prompt augmentation) may be a practical path for the continuous improvement of LLM agents.

## Rating
- Novelty: ⭐⭐⭐⭐ The dynamic benchmark design is novel, and the cognitive cycle underlying Agent-ExpVer has theoretical depth; however, the core techniques (prompt augmentation + experience verification) are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation covers 15+ models, ablation studies, human baselines, and learning process visualizations, though only two game types are included.
- Writing Quality: ⭐⭐⭐⭐ The citation of Jean Piaget is apt and tasteful; the method description is clear, and the appendix provides highly detailed reasoning process examples.
- Value: ⭐⭐⭐⭐ Dynamic spatial reasoning is a clearly identified weakness of LLMs; both the benchmark and the parameter-free learning method hold significant research value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[AAAI 2026\] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation](spatialactor_exploring_disentangled_spatial_representations_for_robust_robotic_m.md)
- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](../../NeurIPS2025/robotics/mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)
- [\[AAAI 2026\] Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](towards_reinforcement_learning_from_neural_feedback_mapping_.md)

<!-- RELATED:END -->
