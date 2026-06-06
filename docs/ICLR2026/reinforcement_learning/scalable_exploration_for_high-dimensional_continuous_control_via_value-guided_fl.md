---
title: >-
  [Paper Note] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow
description: >-
  [ICLR 2026][Reinforcement Learning][High-dimensional control] This paper proposes Qflex (Q-guided Flow Exploration), a scalable RL method for exploration in high-dimensional continuous action spaces. It transports action…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "High-dimensional control"
  - "value-guided flow"
  - "probabilistic flow exploration"
  - "musculoskeletal model"
  - "actor-critic"
date: 2026-05-08
content_hash: f39591f4807921f7
---

# Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow

**Conference**: ICLR 2026
**arXiv**: [2601.19707](https://arxiv.org/abs/2601.19707)  
**Area**: Reinforcement Learning / High-Dimensional Control
**Keywords**: High-dimensional control, value-guided flow, probabilistic flow exploration, musculoskeletal model, actor-critic

## TL;DR

This paper proposes Qflex (Q-guided Flow Exploration), a scalable RL method for exploration in high-dimensional continuous action spaces. It transports actions from a learnable source distribution along a probability flow induced by the Q-function, aligning exploration with task-relevant gradients rather than isotropic noise. Qflex outperforms Gaussian and diffusion-based RL baselines across various high-dimensional benchmarks, and successfully controls a full-body musculoskeletal model with 700 actuators to perform agile and complex motions.

## Background & Motivation

**Background**: Controlling high-dimensional dynamical systems (e.g., full-body musculoskeletal models, multi-legged robots) is a core challenge for RL. Action spaces can reach hundreds of dimensions, rendering standard Gaussian exploration severely ineffective.

**Limitations of Prior Work**:
- (1) Gaussian noise exploration suffers exponentially declining coverage as dimensionality grows, sharply degrading sample efficiency.
- (2) Dimensionality reduction methods (e.g., DynSyn, DEP-RL) limit policy expressiveness and sacrifice flexibility.
- (3) Diffusion/flow policies address multimodality but rely on isotropic initial distributions, remaining inefficient in high dimensions.
- (4) 700 muscle actuators far exceed the operational range of existing methods.

**Key Insight**: Q-function-guided probability flow aligns exploration with task-relevant directions while preserving the original high-dimensional action space.

## Method

### Q-guided Flow Exploration

**Core Idea**: Transport actions from a source distribution to the target policy by "flowing" along the Q-function gradient:

$$a \leftarrow a + v_\theta(a, s, t) \cdot dt$$

where $v_\theta$ is a learned velocity field that moves actions in the direction of increasing Q-values.

### Comparison with Standard Methods

| Method | Exploration | Information Use | High-Dim |
|--------|-------------|-----------------|----------|
| Gaussian (SAC) | Isotropic noise | None | Poor |
| Diffusion (DACER) | Isotropic starting point | Posterior guidance | Moderate |
| **Qflex** | **Q-guided flow** | **Forward guidance** | **Strong** |

### Implementation

- Actor-critic loop with the Q-function serving as the critic.
- Flow transport from a learnable source distribution along Q-guided directions.
- Multi-step transport for iterative refinement rather than single-step noise injection.

## Key Experimental Results

### High-Dimensional Benchmarks (MuJoCo / Isaac)

| Environment | Action Dim. | Qflex vs. SAC | vs. Diffusion |
|-------------|-------------|---------------|---------------|
| Humanoid | ~23 | +15% | +10% |
| High-dim variant | ~100 | +30% | +20% |
| **Full-body musculoskeletal** | **700** | **Success (SAC fails)** | **Success (Diffusion fails)** |

### Full-Body Musculoskeletal Control

- 600+ muscles → 700-dimensional action space.
- Complex locomotion (running, jumping, turning) → Qflex succeeds; all baselines fail.
- No dimensionality reduction → full flexibility preserved.

### Key Findings

- Q-guided exploration is highly effective in high dimensions because the vast majority of directions are uninformative; Q guidance focuses effort on useful directions.
- A learnable source distribution outperforms a fixed Gaussian, as the initial distribution itself carries useful information.
- The performance gap between Qflex and baselines widens with dimensionality, confirming scalability.

## Highlights & Insights

- **"The 700-dimensional 'impossible' task"**: No prior RL method had succeeded in a 700+ dimensional continuous action space; Qflex breaks this barrier.
- **"Q-function as an exploration compass"**: Rather than random trial, exploration is directed by Q-guided signals, giving each step a meaningful direction.
- **Value of preserving the original space**: Dimensionality reduction sacrifices flexibility and may exclude optimal solutions; Qflex demonstrates that retaining full dimensionality is worthwhile.
- **Biological inspiration**: Human musculoskeletal control relies on value-like signals to guide exploration in the brain, which parallels the flow mechanism in Qflex.

## Limitations & Future Work

- In this paper, we introduce Qflex, a scalable online RL method for efficient exploration in high-dimensional continuous control.

- Our method conducts directed exploration by sampling from a Q-guided probability flow with policy-improvement guarantees, yielding superior learning efficiency over representative online RL baselines across benchmarks characterized by high dimensionality and over-actuation.

- Qflex further demonstrates agile, complex motion control on a full-body musculoskeletal model with 700 actuators, achieving high efficiency and strong scalability in truly high-dimensional settings.

- Our analysis shows that value-aligned exploration in Qflex surpasses undirected sampling strategies in high-dimensional regimes, which is readily extensible to a variety of online RL frameworks and exploration settings.

- Acknowledgments

This work is supported by STI 2030-Major Projects 2022ZD0209400, Beijing Academy of Artificial Intelligence and Beijing Municipal Science & Technology Commissi

## Related Work & Insights

- **vs. DynSyn**: This paper proposes a distinct technical approach, achieving improvements on key metrics over DynSyn.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First proposal of Q-guided probabilistic flow exploration + success at 700 dimensions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional benchmarks + full-body musculoskeletal + comparison with diverse baselines
- Writing Quality: ⭐⭐⭐⭐ Method motivation clearly articulated
- Value: ⭐⭐⭐⭐⭐ Represents a fundamental breakthrough for high-dimensional RL

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)
- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [\[ICLR 2026\] Value Flows](value_flows.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
