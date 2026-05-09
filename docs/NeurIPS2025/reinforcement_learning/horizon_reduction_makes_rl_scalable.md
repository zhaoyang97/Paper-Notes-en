---
title: >-
  [Paper Note] Horizon Reduction Makes RL Scalable
description: >-
  [NeurIPS 2025][Reinforcement Learning][offline RL] Through large-scale experiments involving up to one billion transitions, this paper identifies the curse of horizon—excessively long decision horizons—as the primary scalability bottleneck in offline RL, and demonstrates that horizon reduction techniques such as n-step returns and hierarchical policies substantially improve scalability. Building on this analysis, the paper proposes SHARSA, a simple yet effective method.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - offline RL
  - scalability
  - horizon reduction
  - hierarchical RL
  - goal-conditioned RL
date: 2026-05-08
content_hash: d27d0df3831e471a
---

# Horizon Reduction Makes RL Scalable

**Conference**: NeurIPS 2025
**arXiv**: [2506.04168](https://arxiv.org/abs/2506.04168)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: offline RL, scalability, horizon reduction, hierarchical RL, goal-conditioned RL

## TL;DR

Through large-scale experiments involving up to one billion transitions, this paper identifies the curse of horizon—excessively long decision horizons—as the primary scalability bottleneck in offline RL, and demonstrates that horizon reduction techniques such as n-step returns and hierarchical policies substantially improve scalability. Building on this analysis, the paper proposes SHARSA, a simple yet effective method.

## Background & Motivation

1. **Scalability is central to modern ML**: NLP and CV have demonstrated that increasing data and compute yields consistent performance gains, yet the scalability of offline RL on complex tasks remains underexplored—prior work has focused primarily on "more tasks" (breadth scaling) rather than "harder tasks" (depth scaling).

2. **Standard offline RL fails on complex tasks**: On the most challenging OGBench tasks (cube-octuple, puzzle-4x6, humanoidmaze-giant), state-of-the-art methods such as IQL, CRL, and SAC+BC remain unable to solve the tasks even with one billion data points, achieving performance far below optimal.

3. **Scaling model size does not resolve the problem**: Scaling SAC+BC to 591 million parameters (a 35× increase) yields only marginal improvement or even degradation, indicating that insufficient model capacity is not the root cause.

4. **Bias accumulation in TD learning**: The prediction targets in temporal-difference (TD) learning are inherently biased, and this bias accumulates over the horizon. This stands in sharp contrast to supervised learning objectives such as next-token prediction, which are unbiased and thus scale without bound.

5. **Curse of horizon in policy learning**: Even with a perfect value function, a long horizon implies an extremely complex mapping from states to optimal actions—analogous to the difficulty of prompting a large model to answer complex questions without chain-of-thought reasoning.

6. **Absence of scalability-oriented evaluation paradigms**: Existing offline RL benchmarks (e.g., D4RL) feature relatively simple tasks and small datasets (~1M transitions), which are insufficient to expose scalability bottlenecks. Large-scale, long-horizon benchmarks are needed to assess whether algorithms are "ready to scale."

## Method

### Problem Setting

The paper focuses on offline goal-conditioned RL, evaluating on four extremely challenging OGBench tasks:
- **cube-octuple**: sequentially pick and place eight blocks
- **puzzle-4x5/4x6**: solve the combinatorial "Lights Out" puzzle with a robotic arm
- **humanoidmaze-giant**: navigate a humanoid robot through a large maze

The largest dataset contains **one billion transitions** (~1 million trajectories), which is 1,000× larger than standard offline RL datasets.

### Diagnosing the Curse of Horizon

**Value learning diagnosis**: On a combination-lock didactic task, 1-step DQN and 64-step DQN are compared:
- Both exhibit similar TD errors, yet the Q-error of 1-step DQN (relative to the true $Q^*$) grows sharply with the horizon.
- Q-error is largest in states far from the goal, directly corroborating the bias accumulation hypothesis.
- Adjusting model size, learning rate, or target network update frequency cannot remedy this issue.

**Policy learning diagnosis**: Long horizons render the mapping from states to optimal actions highly complex, analogous to answering difficult questions without chain-of-thought reasoning. Hierarchical policies decompose this complexity by introducing subgoal-conditioned policies at each level.

### SHARSA

SHARSA simultaneously reduces the value horizon and the policy horizon.

**High-level policy extraction** (rejection sampling):
$$\pi^h(s,g) \stackrel{d}{=} \arg\max_{w_1,...,w_N: w_i \sim \pi_\beta^h(w|s,g)} Q^h(s, w_i, g)$$

$N$ subgoals are sampled from a high-level flow BC policy, and the high-level value function selects the optimal one.

**High-level SARSA value learning** (n-step):

$$L^Q(Q^h) = \mathbb{E}\left[D\left(Q^h(s_h, s_{h+n}, g), \sum_{i=0}^{n-1}\gamma^i r(s_{h+i}, g) + \gamma^n V^h(s_{h+n}, g)\right)\right]$$

n-step returns reduce the number of TD recursions, thereby mitigating bias accumulation.

**Low-level policy**: A goal-conditioned flow BC policy is used; an additional round of rejection sampling yields the double SHARSA variant.

The key advantages of SHARSA are: (1) it relies solely on behavior cloning and SARSA, requiring no complex hyperparameter tuning; and (2) it simultaneously reduces the horizon along both the value and policy dimensions.

## Key Experimental Results

### Experiment 1: Scaling Failure of Standard Offline RL

| Method | cube-octuple (1B) | puzzle-4x6 (1B) | humanoidmaze-giant (1B) |
|------|------------------|----------------|----------------------|
| Flow BC | ~0% | ~5% | ~0% |
| IQL | ~0% | ~20% | ~15% |
| CRL | ~0% | ~10% | ~45% |
| SAC+BC | ~0% | ~25% | ~20% |

All four methods fail on cube-octuple (~0%), with virtually no performance gain as data scales from 1M to 1B. Scaling SAC+BC to 591M parameters (8 days of training) also yields no meaningful improvement. Nine ablations were conducted—covering policy type, network architecture, ensembles, regularization, learning rate, target network update rate, batch size, and gradient steps—none of which broadly improved scalability.

### Experiment 2: Effect of Horizon Reduction

| Method | Horizon Reduction | cube-octuple (1B) | puzzle-4x6 (1B) | humanoidmaze-giant (1B) |
|------|------------|------------------|----------------|----------------------|
| SAC+BC | None | ~0% | ~25% | ~20% |
| n-step SAC+BC | Value | ~0% | ~40% | ~55% |
| Hierarchical FBC | Policy | ~10% | ~5% | ~0% |
| HIQL | Policy | ~5% | ~30% | ~35% |
| **SHARSA** | **Value + Policy** | **~15%** | **~45%** | **~60%** |
| **Double SHARSA** | **Value + Policy** | **~20%** | **~50%** | **~55%** |

Key findings: (1) n-step returns alone yield substantial improvements (puzzle +15pp, humanoidmaze +35pp); (2) hierarchical policies are necessary for cube (0% → 10%+); (3) SHARSA is the only method to achieve non-trivial performance across all four tasks. Notably, SHARSA uses a standard [1024]×4 MLP, far smaller than the 591M-parameter ablation model, demonstrating that algorithmic improvements matter more than model scale.

## Highlights & Insights

- **First large-scale diagnosis of offline RL scalability**: Systematic analysis at the one-billion-transition scale, encompassing nine ablations, provides a clear explanation for why standard RL cannot improve continuously with more data or larger models in the way LLMs do.
- **Clear causal attribution**: The combination-lock didactic experiment precisely isolates the effect of TD bias accumulation from model capacity; the distribution of Q-error across states is particularly compelling.
- **Minimalist yet effective method**: SHARSA combines only three simple components—SARSA, flow BC, and rejection sampling—without complex objective functions or extensive hyperparameter tuning.
- **Honest discussion of open problems**: The paper acknowledges that even SHARSA does not reach 100% on all tasks and calls on the community to prioritize scalability evaluation.

## Limitations & Future Work

- **Strong dataset assumptions**: SHARSA implicitly assumes that the dataset is near-optimal within short segments; its robustness to low-quality datasets has not been verified.
- **State-space experiments only**: Visual observations are excluded to isolate core factors, but in practice the interaction between representation learning and the curse of horizon may be considerably more complex.
- **Idealized evaluation settings**: Out-of-distribution generalization and insufficient data coverage are not assessed; in real deployments, these factors interact with horizon length in non-trivial ways.
- **Scalability not fully resolved**: SHARSA achieves only ~20% success on cube-octuple, performance does not increase monotonically with data, and the two-level hierarchy mitigates rather than eliminates bias accumulation.

## Related Work & Insights

### vs. IQL (Kostrikov et al., 2022)

IQL avoids distributional shift via in-sample maximization and performs well on standard D4RL benchmarks. However, IQL employs 1-step TD learning, causing severe bias accumulation on long-horizon tasks. HIQL, a hierarchical extension of IQL, improves the policy horizon by adding a hierarchical structure but does not address the value horizon. SHARSA, through n-step SARSA and hierarchical policies, reduces both dimensions simultaneously and outperforms IQL and HIQL on all four tasks.

### vs. CRL (Eysenbach et al., 2022)

CRL is based on contrastive learning and one-step RL, and does not use TD learning; it is therefore unaffected by bias accumulation. CRL indeed performs reasonably well on humanoidmaze-giant (~45%), corroborating the paper's hypothesis about TD bias. However, CRL still fails on cube-octuple (~0%) and puzzle (~10%), as its flat policy suffers from the curse of horizon in policy learning. SHARSA compensates for this limitation through hierarchical decomposition.

### vs. Park et al. (2024) on Scalability

Park et al. investigate bottlenecks in policy extraction and generalization for offline RL scalability, but use datasets approximately 1/100 the size of those in this work and do not deeply analyze the role of the horizon. This paper employs more expressive flow policies and a 100× larger dataset, systematically identifying the horizon as the core obstacle and offering complementary insights.

## Rating

| Dimension | Score | Remarks |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First systematic identification of the curse of horizon as the core scalability bottleneck in offline RL, with clear causal validation |
| Technical Depth | ⭐⭐⭐ | SHARSA itself is relatively simple; the primary contribution lies in the diagnostic analysis rather than algorithmic innovation |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | One-billion-scale data, nine ablations, didactic causal attribution, and four extremely challenging environments; experimental scope and rigor are exceptional |
| Value | ⭐⭐⭐⭐ | Provides clear direction for scaling RL; SHARSA is straightforward to implement and directly applicable, though idealized assumptions remain a limitation |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[NeurIPS 2025\] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners](when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)

<!-- RELATED:END -->
