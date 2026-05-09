---
title: >-
  [Paper Note] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV
description: >-
  [Reinforcement Learning] This paper proposes Virne — a comprehensive benchmark framework for Network Function Virtualization Resource Allocation (NFV-RA) — integrating 30+ algorithms and a gym-style environment to support systematic evaluation across cloud, edge, 5G, and other scenarios.
tags:
  - Reinforcement Learning
date: 2026-05-08
content_hash: 7255b24c7ab54a3d
---

# Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2507.19234](https://arxiv.org/abs/2507.19234)
- **Code**: [https://github.com/GeminiLight/Virne](https://github.com/GeminiLight/Virne)
- **Area**: Reinforcement Learning / Network Resource Allocation / Combinatorial Optimization
- **Keywords**: NFV-RA, Virtual Network Embedding, Benchmark Framework, GNN, PPO, Scalability

## TL;DR
This paper proposes Virne — a comprehensive benchmark framework for Network Function Virtualization Resource Allocation (NFV-RA) — integrating 30+ algorithms and a gym-style environment to support systematic evaluation across cloud, edge, 5G, and other scenarios.

## Background & Motivation

### Core Problem
Resource allocation in Network Function Virtualization (NFV-RA) is an NP-hard combinatorial optimization problem that maps virtual network requests onto physical network infrastructure. Although deep RL has shown promise in this domain, a systematic benchmark for comprehensive simulation and rigorous evaluation is lacking.

### Limitations of Prior Work
1. Existing benchmarks cover only specific scenarios (e.g., cloud) and lack support for edge computing and 5G slicing.
2. Only a small number of non-RL methods (3–5) are implemented, with no unified RL pipeline.
3. Evaluation is limited to online effectiveness, neglecting practical dimensions such as feasibility, generalizability, and scalability.
4. Fragmented problem definitions make fair comparison difficult.

## Method

### Overall Architecture

Virne consists of five core modules:
1. **Simulation Configuration**: Customizable network topologies, resource types, and service demands.
2. **Network System**: An event-driven simulator that processes online virtual network requests.
3. **Algorithm Implementation**: A modular pipeline integrating 30+ methods.
4. **Auxiliary Tools**: System controllers, solution monitors, and visualization utilities.
5. **Evaluation Protocol**: A multi-dimensional assessment framework.

### NFV-RA Problem Formulation

**System Model**: Physical network $\mathcal{G}_p = (\mathcal{N}_p, \mathcal{L}_p)$; virtual network $\mathcal{G}_v = (\mathcal{N}_v, \mathcal{L}_v, \omega, \varpi)$.

**Embedding Constraints**:
- Node mapping $f_\mathcal{N}$: one-to-one mapping satisfying resource constraints $C(n_v) \leq C(n_p)$.
- Link mapping $f_\mathcal{L}$: physical paths connecting mapped node endpoints with bandwidth constraint $B(l_v) \leq B(l_p)$.

**Optimization Objective** (Revenue-to-Cost Ratio):
$$\max \text{R2C}(S) = (\varkappa \cdot \text{REV}(S)) / \text{COST}(S)$$

### MDP Formulation

NFV-RA is modeled as an MDP $(\mathcal{S}, \mathcal{A}, P, R, \lambda)$:
- State space $\mathcal{S}$: embedding states of the VN and PN.
- Action space $\mathcal{A}$: selecting a physical node to host a virtual node.
- Reward $R$: a designed feedback signal to guide optimization.
- Per-step decision: select a physical node → attempt placement → route virtual links → update resources.

### Unified RL Pipeline

RL-based NFV-RA algorithms are unified into three components:
1. **MDP Modeling**: Reward design and feature engineering.
2. **Policy Architecture**: MLP, CNN, GCN, GAT, BiGCN, BiGAT, HeteroGAT, etc.
3. **Training Method**: PG, A3C, PPO, MCTS.

## Experiments

### Implementation Technique Analysis

Key implementation choices are systematically evaluated on the WX100 topology:

| Technique | Best Configuration | Finding |
|---|---|---|
| Reward Function | fixed=0.1 | Moderate fixed intermediate rewards outperform adaptive rewards |
| Feature Engineering | Status + Topological | Topological features provide valuable augmentation |
| Action Masking | Enabled | RAC improves by up to 5.3% |
| RL Algorithm | PPO | Fastest convergence and highest performance |

### Main Results

| Method | WX100 RAC↑ | GEANT RAC↑ | BRAIN RAC↑ |
|---|---|---|---|
| PPO-MLP | 71.90 | 55.80 | 51.30 |
| PPO-GCN | 66.80 | - | - |
| PPO-DualGAT | **78.10** | - | - |
| D-Vine | - | - | - |

### Evaluation Dimensions

1. **Effectiveness**: Online acceptance rate and long-term revenue-to-cost ratio.
2. **Feasibility**: Constraint satisfaction rate of generated solutions.
3. **Generalizability**: Reliability under varying network conditions.
4. **Scalability**: Performance variation with increasing problem size.

### Key Findings

1. PPO-DualGAT combined with optimal implementation techniques achieves the best performance in most settings.
2. Moderate fixed reward (0.1) > adaptive reward > excessively large or small fixed rewards.
3. Action masking is critical for handling the complex constraints in NFV-RA.
4. The performance advantage of GNN-based architectures correlates with scenario complexity.

## Highlights & Insights

1. **Most comprehensive NFV-RA benchmark**: 30+ algorithms, gym-style environment, and multi-scenario support.
2. **Systematic analysis of implementation techniques**: Quantitative impact of key choices including reward design, feature engineering, and action masking.
3. **Multi-dimensional evaluation protocol**: Extends beyond online effectiveness to cover feasibility, generalizability, and scalability.
4. **Modular design**: Facilitates community extension with new methods.

## Limitations & Future Work

1. A gap remains between simulation and real network environments.
2. Credit assignment challenges for RL methods at large scale are not fully resolved.
3. Support for emerging scenarios (e.g., 6G network slicing) is still under development.
4. Some RL methods may be sensitive to hyperparameter choices.

## Related Work & Insights

- **Traditional Benchmarks**: VNE-Sim (2014), ALEVIN (2016) — cloud-only scenarios with a limited set of heuristics.
- **RL-based NFV-RA**: Various approaches employing neural network policies based on CNN, GCN, GAT, etc.
- **RL for Combinatorial Optimization**: Methodological connections to RL approaches for problems such as TSP and VRP.

## Rating
- **Novelty**: ⭐⭐⭐ — The primary contribution lies in systems engineering rather than algorithmic innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive experiments and ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear and well-organized structure.
- **Value**: ⭐⭐⭐⭐⭐ — An extremely valuable benchmark tool for the research community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)

<!-- RELATED:END -->
