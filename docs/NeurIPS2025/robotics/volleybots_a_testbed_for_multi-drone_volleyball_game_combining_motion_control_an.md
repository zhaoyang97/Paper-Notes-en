---
title: >-
  [Paper Note] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play
description: >-
  [NeurIPS 2025][Robotics][Multi-drone systems] This paper presents VolleyBots, a multi-drone volleyball competition testbed that integrates cooperative-adversarial gameplay, turn-based interaction, and agile 3D motion control. Built on Isaac Sim, it establishes a task curriculum from single-agent training to multi-agent competition. A hierarchical policy achieves a 69.5% win rate on the 3v3 task, with demonstrated zero-shot sim-to-real transfer.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Multi-drone systems"
  - "robot sports"
  - "multi-agent reinforcement learning"
  - "game theory"
  - "sim-to-real"
date: 2026-05-08
content_hash: e3309b96580458d0
---

# VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play

**Conference**: NeurIPS 2025
**arXiv**: [2502.01932](https://arxiv.org/abs/2502.01932)  
**Code**: [https://github.com/thu-uav/VolleyBots](https://github.com/thu-uav/VolleyBots)  
**Area**: Reinforcement Learning
**Keywords**: Multi-drone systems, robot sports, multi-agent reinforcement learning, game theory, sim-to-real

## TL;DR
This paper presents VolleyBots, a multi-drone volleyball competition testbed that integrates cooperative-adversarial gameplay, turn-based interaction, and agile 3D motion control. Built on Isaac Sim, it establishes a task curriculum from single-agent training to multi-agent competition. A hierarchical policy achieves a 69.5% win rate on the 3v3 task, with demonstrated zero-shot sim-to-real transfer.

## Background & Motivation
Robot sports are considered an ideal setting for showcasing embodied intelligence due to their well-defined objectives, explicit rules, and dynamic interactions. Prior work has made progress in individual sub-domains: RoboCup focuses on team cooperation and adversarial play; table tennis robots emphasize turn-based interaction; drone pursuit-evasion tasks require agile 3D maneuvering. However, no existing platform simultaneously encompasses **mixed cooperative-adversarial gameplay**, **turn-based interaction**, and **agile 3D maneuvering**.

Volleyball naturally possesses all three characteristics: intra-team coordination requires tight cooperation to execute bump-set-spike sequences, inter-team play demands anticipation and exploitation of opponent strategies, and ball trajectories require drones to perform rapid acceleration, sharp turns, and precise positioning under underactuated quadrotor dynamics. These features are deeply intertwined, forming a complex problem that demands both **low-level motion control** and **high-level strategic decision-making**, without access to any expert demonstration data.

The central approach is to use volleyball as a vehicle, drawing on the progressive human learning process, to design a task curriculum from simple to complex — systematically benchmarking and advancing multi-agent RL and game-theoretic algorithms on challenging embodied tasks.

## Method

### Overall Architecture
VolleyBots is built on NVIDIA Isaac Sim, leveraging the OmniDrones simulator for GPU-parallelized data collection. The platform comprises three components: (1) **Environment** — defining entities, observations, actions, and rewards; (2) **Tasks** — including 3 single-agent tasks, 3 multi-agent cooperative tasks, and 3 multi-agent competitive tasks; (3) **Algorithms** — covering RL, MARL, and game-theoretic baselines.

### Key Designs
1. **Simulated Entities and Physical Interaction**:

    - Uses the Iris quadrotor model with a virtual paddle of radius 0.2 m (restitution coefficient 0.8)
    - Ball: radius 0.1 m, mass 5 g, restitution coefficient 0.8, supporting realistic bouncing and collision
    - Court follows standard volleyball dimensions $9\text{m} \times 18\text{m}$, net height 2.43 m
    - Design Motivation: High-fidelity physical interaction is the foundation for sim-to-real transfer

2. **Progressive Task Curriculum**:

    - **Single-agent tasks**: Back and Forth (sprint traversal), Hit the Ball (striking distance), Solo Bump (continuous juggling) — evaluating fundamental 3D maneuvering capability
    - **Cooperative tasks**: Bump and Pass (two-agent rally), Set and Spike Easy/Hard (set-spike coordination) — introducing intra-team turn-based coordination
    - **Competitive tasks**: 1v1, 3v3, 6v6 — integrating all challenges: cooperation, adversarial play, turn-based interaction, and 3D maneuvering
    - Design Motivation: Mirrors the human learning progression from fundamentals to team coordination to formal match play

3. **Dual Action Spaces and Hierarchical Rewards**:

    - Two action space options: CTBR (collective thrust + body rates, higher-level abstraction) and PRT (per-rotor thrust, fine-grained control)
    - Reward function consists of three components: misbehave penalty (general violation penalties), task reward (sparse task-level reward), and shaping reward (auxiliary guidance reward)
    - Design Motivation: PRT offers stronger maneuverability but is harder to train; CTBR facilitates sim-to-real transfer; the hierarchical reward balances exploration and learning efficiency

4. **Hierarchical Policy**:

    - Low level: a set of primitive skills (Hover, Serve, Pass, Set, Attack) trained with PPO
    - High level: a rule-based, event-driven policy that assigns low-level skills to each drone upon each hit
    - For the Attack skill, the high-level policy selects left or right direction with equal probability
    - Design Motivation: End-to-end approaches struggle to jointly optimize motion control and strategic play; hierarchical decomposition effectively alleviates this tension

### Loss & Training
- Single-agent and cooperative tasks use the standard PPO/MAPPO clipped surrogate objective
- Competitive tasks incorporate game-theoretic training paradigms including Self-Play, Fictitious Self-Play (FSP), and PSRO (Uniform/Nash meta-solvers)
- All algorithms share unified hyperparameters within each task to assess cross-task robustness

## Key Experimental Results

### Main Results

| Task | Metric | PPO (PRT) | TD3 (PRT) | SAC (PRT) | Note |
|------|--------|-----------|-----------|-----------|------|
| Back and Forth | Waypoints reached | **10.04±0.20** | 0.99±0.01 | 0.83±0.25 | PPO far surpasses off-policy |
| Hit the Ball | Hitting distance (m) | **11.40±0.06** | 5.29±1.28 | 3.87±2.34 | PPO ≈ 2× SOTA |
| Solo Bump | Juggling count | **10.83±1.24** | 3.68±1.43 | 1.36±0.60 | PPO advantage is significant |

| Competitive Task | Metric | SP | FSP | PSRO_Nash | Hierarchical |
|-----------------|--------|-----|-----|-----------|--------------|
| 3v3 | Exploitability↓ | **25.76** | 38.86 | 35.83 | — |
| 3v3 | Win Rate↑ | 0.59 | 0.52 | **0.61** | **0.695** |
| 3v3 | Elo↑ | 1077 | 906 | **1268** | — |

### Ablation Study

| Configuration | Bump and Pass | Set & Spike (Easy) | Set & Spike (Hard) |
|--------------|---------------|-------------------|-------------------|
| MAPPO w/o shaping | 11.32±0.91 | 0.25 | 0.25 |
| MAPPO w. shaping | **13.71±0.58** | **0.99** | 0.75 |
| MADDPG w. shaping | 0.84±0.09 | 0.23 | 0.22 |
| QMIX w. shaping | 0.09±0.00 | 0.02 | 0.02 |

### Key Findings
- On-policy methods (PPO/MAPPO) consistently outperform off-policy methods across all tasks and exhibit stronger cross-task hyperparameter robustness
- The PRT action space achieves marginally higher final performance than CTBR, though CTBR converges faster
- No algorithm converges to an effective strategy on the 6v6 task, exposing a scalability bottleneck in current methods
- The hierarchical policy achieves a 69.5% win rate against the strongest baseline (SP's Nash equilibrium policy) on the 3v3 task

## Highlights & Insights
- **The first work to unify cooperative-adversarial, turn-based, and 3D maneuvering challenges within a single platform**, filling a gap in robot sports research and providing a rigorous standard testbed for embodied intelligence.
- **Zero-shot sim-to-real deployment validates the practical significance of the platform**: the Solo Bump policy trained entirely in simulation successfully executes ball juggling on a real quadrotor, demonstrating that the simulator faithfully approximates real-world physics.

## Limitations & Future Work
- No algorithm currently converges on the 6v6 format; more advanced hierarchical or communication mechanisms are needed to handle large-scale coordination
- The high-level layer of the hierarchical policy remains hand-crafted; future work could employ RL or LLMs to automatically learn strategic decision-making
- The current observation space is state-based; incorporating visual observations would allow exploration of more realistic perceptual challenges

## Related Work & Insights
- **vs. MQE (quadruped soccer)**: MQE also supports cooperative-adversarial play, but VolleyBots uniquely features turn-based interaction and 3D aerial maneuvering, imposing higher demands on low-level control
- **vs. Robot Table Tennis**: Table tennis robots involve turn-based interaction but lack multi-agent cooperation; VolleyBots offers greater richness in multi-agent coordination
- **vs. SMPLOlympics**: The humanoid Olympic setting covers a broad range of sports but lacks the depth of 3D aerial locomotion and team coordination

## Rating
- Novelty: ⭐⭐⭐⭐ First to bring volleyball to a multi-drone platform with systematic task design, though the core contribution leans toward benchmarking
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 9 tasks, diverse RL/MARL/game-theoretic algorithms, ablations, and sim-to-real validation
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures, and detailed task descriptions
- Value: ⭐⭐⭐⭐ As a standardized benchmark, it offers long-term impact for multi-agent embodied intelligence research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](../../AAAI2026/robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)
- [\[ICLR 2026\] From Embedding to Control: Representations for Stochastic Multi-Object Systems](../../ICLR2026/robotics/from_embedding_to_control_representations_for_stochastic_multi-object_systems.md)

</div>

<!-- RELATED:END -->
