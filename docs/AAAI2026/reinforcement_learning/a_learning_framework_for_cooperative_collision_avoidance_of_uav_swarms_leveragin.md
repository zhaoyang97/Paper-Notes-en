---
title: >-
  [Paper Note] A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge
description: >-
  [AAAI 2026][Reinforcement Learning][UAV swarm] This paper proposes reMARL, a framework that leverages domain knowledge from image processing (active contour model) to design reward functions for multi-agent reinforcement…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "UAV swarm"
  - "collision avoidance"
  - "MARL"
  - "domain knowledge"
  - "active contour model"
date: 2026-05-08
content_hash: 0774e874d739b6aa
---

# A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge

**Conference**: AAAI 2026
**arXiv**: [2507.10913](https://arxiv.org/abs/2507.10913)  
**Code**: N/A  
**Area**: Agent / Multi-Agent Reinforcement Learning
**Keywords**: UAV swarm, collision avoidance, MARL, domain knowledge, active contour model

## TL;DR

This paper proposes reMARL, a framework that leverages domain knowledge from image processing (active contour model) to design reward functions for multi-agent reinforcement learning, enabling cooperative collision avoidance in UAV swarms. Compared to traditional metaheuristic methods, reMARL reduces reaction time by 98.75% and energy consumption by 85.37%.

## Background & Motivation

1. **Background**: UAV swarm collision avoidance requires both safety and energy efficiency. Traditional approaches include Velocity Obstacle (VO), Artificial Potential Field (APF), and metaheuristic optimization. MARL methods such as COMA, VDN, and QMIX have also been widely explored in recent years.
2. **Limitations of Prior Work**: VO methods frequently alter velocity, leading to low energy efficiency; APF suffers from local optima; metaheuristic methods have excessively long reaction times unsuitable for real-time applications. Among MARL methods, observation-sharing schemes degrade in performance as swarm size grows, while credit assignment schemes (IGM assumption in VDN/QMIX) cause unbounded divergence and unpredictable behavior.
3. **Key Challenge**: How to achieve efficient cooperative collision avoidance for large-scale UAV swarms without relying on complex credit assignment or observation-sharing mechanisms.
4. **Goal**: To design a MARL framework scalable to large swarms, eliminating dependence on complex network architectures through domain-knowledge-driven reward functions.
5. **Key Insight**: Inspired by the active contour model in image processing, the environment is modeled as a 2D potential field, and reward functions are designed to guide UAVs to fly along contour lines.
6. **Core Idea**: The contour extraction concept from image processing is adopted to design MARL reward functions, allowing cooperative behavior to emerge naturally from individual reward maximization.

## Method

### Overall Architecture

reMARL maps the environment to a 2D potential field $\Phi(q)$, treating obstacles and the virtual swarm center as potential field peaks. Individual rewards are designed using the cost function of the active contour model, and each UAV independently learns a policy via DDPG. Cooperative behavior emerges naturally through the shared potential field.

### Key Designs

1. **Potential Field Construction** — The field superposes an obstacle repulsion field $\Phi_o(q)$ and a swarm repulsion field $\Phi_s(q)$. Obstacle field intensity is inversely proportional to the square of distance and is clamped to a maximum within the safe distance $d_{safe}$. The swarm is modeled as a single repulsion source to prevent UAVs from sitting at the peaks of their individual repulsion fields.

2. **Domain-Knowledge-Driven Reward** — The reward function is $r = -f(S, \Phi(q)) + r_{form} \cdot r_{collide}$, where:
    - **Contour term**: Based on the active contour model cost function $f(S) = \int \frac{1}{2}|S''|^2 - \frac{1}{2}|\nabla\Phi|^2 d\rho$; minimizing this term drives trajectories toward potential field contour lines.
    - **Swarming term**: $r_{form}$ maintains formation via velocity cosine similarity; $r_{collide}$ serves as a hard safety constraint.

3. **PSO Contour Adjustment** — Particle swarm optimization adjusts each UAV's position so that the distance between adjacent contour lines satisfies the minimum safe inter-UAV spacing $\bar{d}_{U2U}$. The cost function $f_{pso} = f_{thres} + f_{shift}$ simultaneously enforces safe separation and minimal displacement.

### Loss & Training

- Each UAV independently uses DDPG (actor network + critic network); the action is the velocity direction change $[-\pi/4, \pi/4]$.
- Actor network: FC(256, ReLU) → Output(1, tanh)
- Critic network: observation and action are each passed through FC(256), concatenated, then FC(256) → Output(1)
- TD loss updates the critic; policy gradient updates the actor.

## Key Experimental Results

### Main Results

Comparisons against COMA, VDN, QMIX, MAPPO, and IQL across multiple scenarios ranging from 2U1O to 10U1O and 3U2O to 7U2O:

| Metric | reMARL | Metaheuristic | Gain |
|--------|--------|---------------|------|
| Reaction time (s) | 0.006 ± 0.03 | 0.48 ± 0.05 | **98.75%** |
| Energy consumption (curvature integral) | 19.72 ± 31.8 | 134.88 ± 8.3 | **85.37%** |
| Min UAV–obstacle distance | 24.78 | 38.70 | 35.96% closer |
| Min UAV–UAV distance | 29.45 | 40.31 | 26.94% closer |

### Ablation Study

| Method | Swarm size ≤ 3 | Swarm size > 3 |
|--------|----------------|----------------|
| SOTA MARL (without Contour reward) | Outperforms reMARL | Performance drops sharply |
| reMARL (full reward) | Slightly inferior | **Significantly outperforms all baselines** |

Learning curves indicate that for swarm sizes ≤ 3, UAVs can fly safely without following contour lines; as swarm size increases, adhering to contour lines becomes the optimal strategy.

### Key Findings

- The advantage of reMARL stems entirely from the Contour reward term; DDPG itself provides no cooperative mechanism.
- After training, UAVs discover alternative paths when contour lines are infeasible (e.g., navigating between obstacles), demonstrating adaptive capability beyond the domain-knowledge prior.
- Observation sharing is used only for reward construction, not for network architecture, thereby avoiding the curse of dimensionality.

## Highlights & Insights

- **Interdisciplinary Innovation**: Transferring the active contour model from image processing to MARL reward design is a novel and effective contribution.
- **Strong Scalability**: By eliminating dependence on credit assignment and observation sharing in the network architecture, the framework supports large-scale swarms (10+ UAVs).
- **Surpassing Prior Knowledge**: The learned policy not only follows contour lines but also adaptively finds superior paths when contour lines are infeasible.

## Limitations & Future Work

- Validation is limited to 2D space; 3D collision avoidance with altitude variation is not considered.
- PSO contour adjustment is used only during training; how to handle dynamic changes at deployment is not addressed.
- UAV speed is assumed constant; speed variation in real flight is inevitable.
- More complex obstacle motion models could be incorporated (current obstacles move randomly).

## Related Work & Insights

- The domain-knowledge-driven reward design paradigm is generalizable to other MARL task scenarios.
- The contour-line concept has a long history in potential field methods; combining it with RL is a meaningful extension.

## Rating

- Novelty: ⭐⭐⭐⭐ Unique interdisciplinary reward design approach
- Experimental Thoroughness: ⭐⭐⭐ Multi-scenario comparisons are comprehensive but limited to simulation
- Writing Quality: ⭐⭐⭐ Structure is clear but some equations are poorly typeset
- Value: ⭐⭐⭐ Useful reference for UAV swarm collision avoidance, but real-world deployment validation is absent

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)
- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[AAAI 2026\] ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing](charteditor_a_reinforcement_learning_framework_for_robust_chart_editing.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[AAAI 2026\] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)

</div>

<!-- RELATED:END -->
