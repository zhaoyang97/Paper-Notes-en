---
title: >-
  [Paper Note] Expressive Temporal Specifications for Reward Monitoring
description: >-
  [AAAI 2026][Temporal Logic] This paper leverages quantitative linear temporal logic (LTLf[F]) to automatically synthesize **quantitative reward monitors (QRMs)** that generate dense, continuous-valued reward streams for reinforcement learning agents at runtime, fundamentally alleviating the sparse reward problem in long-horizon tasks under Boolean semantics.
tags:
  - AAAI 2026
  - Temporal Logic
  - Reward Monitor
  - Quantitative Semantics
  - Sparse Reward
  - Reinforcement Learning
date: 2026-05-08
content_hash: 5f0c268547541ad5
---

# Expressive Temporal Specifications for Reward Monitoring

**Conference**: AAAI 2026
**arXiv**: [2511.12808](https://arxiv.org/abs/2511.12808)
**Code**: [github](https://github.com/nightly/quantitative-reward-monitoring)
**Area**: Other
**Keywords**: Temporal Logic, Reward Monitor, Quantitative Semantics, Sparse Reward, Reinforcement Learning

## TL;DR

This paper leverages quantitative linear temporal logic (LTLf[F]) to automatically synthesize **quantitative reward monitors (QRMs)** that generate dense, continuous-valued reward streams for reinforcement learning agents at runtime, fundamentally alleviating the sparse reward problem in long-horizon tasks under Boolean semantics.

## Background & Motivation

### Core Challenge in Reward Design

Reward function design in reinforcement learning is critical yet fraught with difficulties:
- **Manual design burden**: Different environments require careful engineering by domain experts, resulting in high maintenance costs.
- **Sparse reward problem**: When goals require many steps to achieve, agents receive little effective feedback over extended periods, severely hampering learning efficiency.
- **Safety concerns**: Poorly designed rewards may lead to reward hacking, unintended behaviors, or even catastrophic outcomes.

### Temporal Logic as Reward Specification

Temporal logic provides a formal, high-level language for expressing complex goals and constraints. However, existing work predominantly adopts **Boolean semantics** (satisfied/not satisfied), which inherently produces sparse rewards—feedback is only given when the formula is fully satisfied.

### Advantages of Quantitative Semantics

This paper proposes replacing Boolean semantics with **quantitative semantics**, where the degree of formula satisfaction is a real value in $[0,1]$. This enables the monitor to output a continuous signal at every timestep indicating "how close the agent is to completing the task," thereby:
1. Providing denser training signals;
2. Naturally supporting non-Markovian properties;
3. Remaining algorithm-agnostic and compatible with any RL algorithm.

## Method

### Overall Architecture

The system pipeline is as follows: the user specifies task objectives as "specification–reward" pairs in LTLf[F] formulas with scalar weights → the Synth algorithm automatically synthesizes a quantitative reward monitor QRM → the QRM is composed with the environment MDP via synchronous product to form an augmented MDP → any RL algorithm trains on the augmented MDP.

### Key Designs

#### 1. **Quantitative Linear Temporal Logic LTLf[F]**

LTLf[F] shares the syntax of standard LTLf but extends semantics from Boolean to real-valued $[0,1]$:

- Atomic proposition $p$: value given by labeling function $\mathcal{L}(s)(p) \in [0,1]$ (rather than 0/1)
- Negation: $[\![\neg\varphi]\!] = 1 - [\![\varphi]\!]$
- Conjunction: $[\![\varphi_1 \wedge \varphi_2]\!] = \min([\![\varphi_1]\!], [\![\varphi_2]\!])$
- Eventually $F$: $[\![F\varphi]\!] = \max_{j} [\![\varphi,j]\!]$
- Always $G$: $[\![G\varphi]\!] = \min_{j} [\![\varphi,j]\!]$
- Until $\mathcal{U}$: $[\![\varphi_1 \mathcal{U} \varphi_2]\!] = \max_j \min_{k<j}([\![\varphi_2,j]\!], [\![\varphi_1,k]\!])$

This fuzzy treatment yields continuous-valued feedback at every timestep, rather than a 0/1 signal only at terminal states.

#### 2. **Quantitative Reward Monitor (QRM) Construction**

A QRM is defined as a finite-state machine with registers $\mathcal{A} = (Q, q_0, \delta_q, \mathcal{V}, \delta_v, \delta_r)$:
- **State set $Q$**: tracks progress in formula evaluation
- **Registers $\mathcal{V}$**: store quantitative values of subformulas (e.g., accumulated min/max values)
- **Reward function $\delta_r$**: reads current satisfaction degree from reward registers and multiplies by weight $\rho$

The construction procedure (Algorithm 1 Synth) is **inductive**:
- **Atom $p$**: 2 states; register records $\mathcal{L}_p(s)$
- **Negation**: reuses the sub-monitor, adds a $1 - \mathcal{V}(\psi)$ register
- **Conjunction**: Cartesian product of two sub-monitors; registers track min
- **Until**: Cartesian product plus three new registers tracking $\min\varphi_1$, $\max\varphi_2$, and $\max(\min\varphi_1, \max\varphi_2)$

A memoization cache avoids redundant computation for repeated subformulas.

**Theorem 1**: The construction is **linear** in the number of states and transitions with respect to formula size.

**Theorem 3 (Correctness)**: The register value output by the QRM at each timestep $i$ equals the LTLf[F] semantic value $[\![\varphi,i]\!](\lambda)$.

#### 3. **Safety Specification Handling**

For syntactically identifiable safety formulas (safe-LTLf, excluding Until and in negation normal form), once violated the monitor outputs a fixed penalty $\zeta \leq 0$ for all remaining timesteps. This covers two scenarios:
- Terminal failure (e.g., running a red light in autonomous driving → immediate termination)
- Non-terminal hazard (positive rewards should not resume after a violation)

In **composite monitors**, violation of any safety property **globally overrides** the rewards of all monitors.

#### 4. **Compositional Learning with MDPs**

The augmented MDP $\mathcal{M} \otimes \mathcal{A}$ is defined as:
- Augmented state space $\mathcal{S}' = Q \times \mathcal{S}$
- Transition probabilities unchanged; rewards produced by the monitor
- **Theorem 4**: Markov policies on the augmented MDP suffice to optimally capture non-Markovian objectives

### Loss & Training

The proposed method is **algorithm-agnostic**:
- Discrete environments (Toy, Safety Gridworlds): tabular Q-learning
- Continuous environments (Classic, Box2D): PPO
- Reward convergence detection uses exponential moving average with span=32 and tolerance adapted to $[0.002, 0.02]$

## Key Experimental Results

### Main Results

Comparison across 12 Gymnasium/DeepMind environments using three reward sources: handcrafted reward (Base), Boolean monitor (Boolean), and quantitative monitor (Quant.).

| Environment Category | Environment | Base Task Completion | Boolean Task Completion | Quant. Task Completion |
|---------------------|-------------|---------------------|------------------------|----------------------|
| Classic | Acrobot | 99.07% | 4.84% | **94.77%** |
| Classic | Mountain Car | 39.82% | 37.04% | **42.28%** |
| Classic | Pendulum | 74.39% | 45.03% | **72.11%** |
| Toy | Frozen Lake | 58.86% | **62.16%** | 58.96% |
| Toy | Cliff Walking | **85.33%** | 84.38% | 84.95% |
| Safety | Sokoban | 50.23% | 52.00% | **53.15%** |
| Box2D | Bipedal Walker | **16.78%** | 7.43% | 14.43% |
| Box2D | Lunar Lander | **57.45%** | 43.44% | 45.61% |

### Ablation Study

| Configuration | Key Observation | Note |
|---------------|----------------|------|
| Quantitative vs. Boolean semantics | Quantitative consistently ≥ Boolean | Quantitative semantics provides denser signals |
| Classic/Box2D gap > Toy/Grid | Continuous spaces benefit more | Quantitative distance metrics more effective in continuous environments |
| Safety specification global override | Safety Gridworlds safety consistently enforced | All rewards blocked after safety violation |
| Convergence speed | Frozen Lake: 29 vs. 45 | Quantitative monitor can converge faster |

### Key Findings

1. **Quantitative monitors consistently ≥ Boolean monitors**: Across all 12 environments, the quantitative monitor's task completion rate either exceeds or matches that of the Boolean monitor.
2. **Significant advantage in Classic/Box2D**: In continuous state spaces, quantitative distance metrics (e.g., angular proximity, positional proximity) offer qualitative improvements over Boolean "reached/not reached" signals.
3. **Outperforms handcrafted rewards in some environments**: In Mountain Car, the quantitative monitor (42.28%) surpasses the handcrafted reward (39.82%), as quantitative specifications can simultaneously encode position and velocity objectives.
4. **Linear construction overhead**: The number of QRM states and transitions grows linearly with formula size, confirming practical applicability.

## Highlights & Insights

1. **Elegant integration of formal methods and RL**: Users only need to write high-level temporal logic formulas; the system automatically generates dense rewards, replacing tedious manual reward engineering.
2. **Natural densification via quantitative semantics**: No additional reward shaping or hindsight experience replay (HER) is required—quantitative semantics inherently produces dense signals.
3. **Safety-first design philosophy**: The mechanism of globally overriding all rewards upon safety specification violation is both simple and effective.
4. **Natural support for non-Markovian objectives**: By including monitor states in the augmented MDP state, non-Markovian objectives are Markovianized.
5. **Linear construction complexity**: The linear relationship with formula size ensures practical usability.

## Limitations & Future Work

1. **Quantitative atom design still requires domain knowledge**: Although the monitor is constructed automatically, the quantitative labeling function $\mathcal{L}(s)(p)$ still requires manual definition.
2. **Limited advantage in small discrete environments**: In Toy/Grid environments, quantitative distance may guide agents toward high-reward but suboptimal regions.
3. **Discounted temporal logic unexplored**: Discounted LTL could potentially further improve performance on long-horizon tasks.
4. **Multi-agent settings not addressed**: The authors note that ATLf[F] could extend the approach to multi-agent scenarios.
5. **Lack of large-scale environment validation**: The test environments are relatively simple; validation on complex benchmarks such as Atari or MuJoCo has not been conducted.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of synthesizing reward monitors via quantitative temporal logic is novel and mathematically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 12 environments, 3 baselines, and detailed fluent definitions.
- Writing Quality: ⭐⭐⭐⭐⭐ — Formally rigorous, proofs complete, appendix extremely detailed.
- Value: ⭐⭐⭐⭐ — Provides a mathematically principled automated solution for reward design in RL.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](reward_redistribution_via_gaussian_process_likelihood_estimation.md)
- [\[AAAI 2026\] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)
- [\[AAAI 2026\] Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring](judging_by_the_rules_compliance-aligned_framework_for_modern_slavery_statement_m.md)
- [\[AAAI 2026\] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)
- [\[ICLR 2026\] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](../../ICLR2026/others/enhancing_generative_auto_bidding.md)

<!-- RELATED:END -->
