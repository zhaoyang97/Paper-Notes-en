---
title: >-
  [Paper Note] Learning Progress Driven Multi-Agent Curriculum
description: >-
  [ICML2025][Reinforcement Learning][Multi-Agent Reinforcement Learning] SPMARL is proposed to drive adaptive curriculum distributions over the number of agents using a TD-error-based learning progress (instead of returns), addressing the issues of high variance in return estimation and credit assignment difficulty in multi-agent sparse reward tasks.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Multi-Agent Reinforcement Learning"
  - "Curriculum Learning"
  - "Learning Progress"
  - "TD Error"
  - "Self-Paced Learning"
date: 2026-05-08
content_hash: aa2df6d24b5c6311
---

# Learning Progress Driven Multi-Agent Curriculum

**Conference**: ICML2025  
**arXiv**: [2205.10016](https://arxiv.org/abs/2205.10016)  
**Code**: [GitHub](https://github.com/wenshuaizhao/spmarl)  
**Area**: Multi-Agent Curriculum Learning / Reinforcement Learning  
**Keywords**: Multi-Agent Reinforcement Learning, Curriculum Learning, Learning Progress, TD Error, Self-Paced Learning

## TL;DR

SPMARL is proposed to drive adaptive curriculum distributions over the number of agents using a TD-error-based learning progress (instead of returns), addressing the issues of high variance in return estimation and credit assignment difficulty in multi-agent sparse reward tasks.

## Background & Motivation

In multi-agent reinforcement learning (MARL), sparse rewards make exploration extremely challenging. Curriculum learning (CRL) is an effective strategy to mitigate exploration difficulties by first training on simpler tasks and progressively transitioning to target tasks. The **number of agents** serves as a natural curriculum variable in multi-agent scenarios.

Limitations of prior work:

- **Hand-crafted linear curricula**: Methods like DyMA-CL, EPC, and VACL adjust the number of agents in a preset, linear manner (fewer-to-more or more-to-fewer), lacking adaptive capabilities.
- Directly extending single-agent Self-Paced Reinforcement Learning (SPRL) to MARL (i.e., SPRLM) supports adaptive task selection but suffers from **two main deficiencies**:
    1. **High variance of return estimation**: Under sparse rewards, a return estimate is obtained only once per episode, leading to extremely high variance.
    2. **Deteriorating credit assignment**: In many tasks, increasing the number of agents naturally yields higher returns (e.g., in Simple-Spread, 20 agents moving randomly can cover a large number of landmarks). A return-based curriculum thus biases selection toward tasks with high numbers of agents that are "seemingly simple but actually offer no learning value."

## Method

### Problem Formulation

The task is modeled as a Dec-POMDP $\langle \mathcal{S}, \{\mathcal{O}^i\}, \{\mathcal{A}^i\}, r, \mathcal{P}, \gamma \rangle$, where $n$ agents share a global reward. Contextual Reinforcement Learning (Contextual RL) is used to introduce the context $\mathbf{c}$ (representing the number of agents) to parameterize MDPs of varying difficulty.

### SPRLM: Directly Extending SPRL to Multi-Agent

SPRLM applies the constrained optimization framework of SPRL to agent number control:

$$\min_{\nu} D_{\mathrm{KL}}(p(\mathbf{c}|\nu) \| \mu(\mathbf{c}))$$

Subject to the constraints: (1) Expected return $\mathbb{E}_{p(\mathbf{c}|\nu)}[J(\theta, \mathbf{c})] \geq V_{\mathrm{LB}}$; (2) KL divergence between successive distributions $D_{\mathrm{KL}}(p(\mathbf{c}|\nu_k) \| p(\mathbf{c}|\nu_{k+1})) \leq \epsilon$.

A **two-stage optimization** is adopted:

- **Stage 1**: When the expected performance is below the threshold $V_{\mathrm{LB}}$, the return is maximized using importance sample.
- **Stage 2**: Once performance meets the threshold, the KL divergence between the current and target distributions is minimized to gradually converge to the target task.

### SPMARL: Replacing Returns with Learning Progress

Key improvement: Replace the episode return with the **TD error (value function loss)** as the curriculum optimization objective. Learning progress is defined as:

$$\mathrm{LP}(c) = \frac{1}{2} \mathbb{E}_{s, \mathbf{a} \sim \pi(\mathbf{a}|s,\mathbf{c})} [\| R(s, \mathbf{a}) - V(s) \|^2]$$

where $R(s,\mathbf{a})$ is the discounted return, and $V(s)$ is the value function estimate. The optimization objective for Stage 1 becomes:

$$\max_{\nu_{k+1}} \frac{1}{M} \sum_{i=1}^{M} \frac{p(\mathbf{c}_i|\nu_{k+1})}{p(\mathbf{c}_i|\nu_k)} \mathrm{LP}_\theta(\mathbf{c}_i)$$

**Why TD error is effective:**

1. **Low variance**: TD error is computable at every state transition, rather than providing signals solely at the end of an episode.
2. **Naturally reflects policy improvement**: A large value loss indicates that the policy is still changing significantly, suggesting the task holds learning value for the current policy. As the value loss approaches zero, the policy has converged, indicating there is nothing more to learn at this difficulty level.
3. **Mitigates the credit assignment issue**: Instead of directly pursuing high-return tasks, it targets tasks that "best drive policy improvement."

Stage 2 remains unchanged, still using the performance threshold $V_{\mathrm{LB}}$ to determine when to converge to the target distribution.

## Key Experimental Results

Evaluations were conducted on three sparse-reward benchmarks, each featuring objective tasks with a specific number of agents:

| Benchmark Task | Target Agent Count | Reward Design | SPMARL Performance |
|---|---|---|---|
| MPE Simple-Spread | 8 | Reward received only for covering $\geq 4$ landmarks | Fastest convergence, highest return |
| XOR Matrix Game | 20 | Points scored only when all players select different actions | Fastest convergence to optimum |
| SMACv2 Protoss 5v5 | 5 | Win +1 / Loss -1 | Leading or comparable |
| SMACv2 Protoss 6v6 | 6 | Win +1 / Loss -1 | Significantly leading |
| SMACv2 Protoss 7v7 | 7 | Win +1 / Loss -1 | Significantly leading |
| SMACv2 Protoss 8v8 | 8 | Win +1 / Loss -1 | Significantly leading |

Key Comparison Results:

- **W/O teacher** (direct training without curriculum): Failed completely on all tasks, receiving zero reward.
- **Linear** (linear curriculum): Failed completely on Simple-Spread; converged on XOR but remained unstable.
- **ALPGMM**: Achieved high training returns but biased toward an excessive number of agents, leading to poor evaluation performance.
- **VACL**: Failed to guarantee convergence toward the target distribution, resulting in failure on multiple tasks.
- **SPRLM**: Outperformed heuristic baselines on most tasks, but exhibited unstable performance under extreme rewards in SMACv2.
- **SPMARL**: Achieved consistently optimal or comparable performance across all tasks.

Comparison of estimation variance (SMACv2 Protoss 7v7 / 8v8): The standard deviation of TD errors in SPMARL is significantly lower than that of episode returns in SPRLM, verifying the hypothesis that "lower variance leads to a more stable curriculum."

## Highlights & Insights

1. **Accurate problem identification**: Points out the two fundamental flaws (high variance and credit assignment) of return-based ACRL in MARL instead of blindly applying single-agent methods.
2. **Elegant and simple method**: Achieves significant returns with minimal code modifications, changing only the curriculum objective from returns to TD errors without altering the global optimization framework.
3. **Intuitive analogy**: The TD-error-driven task selection is highly analogous to Prioritized Experience Replay (PER), but prioritizes tasks rather than individual transition samples.
4. **Comprehensive experimental coverage**: Three distinct benchmark types (covering, coordination, and adversarial tasks) evaluated under strictly sparse reward configurations.
5. **Insightful curriculum visualization**: Visualizes curriculum distribution trajectories generated by different methods, visually illustrating why SPMARL is more efficient.

## Limitations & Future Work

1. **Unexplored necessity of KL constraint**: The authors point out that when the objective shifts from returns to learning progress, the original KL constraint in SPRL might be redundant, presenting an important area for future investigation.
2. **Limited to controlling agent count**: The context variables only configure the number of agents without simultaneously manipulating environment parameters (e.g., map size, opponent defense strength).
3. **Coupling with base algorithm (MAPPO)**: All experiments are evaluated only with MAPPO, and validation on other MARL baselines like QMIX or MADDPG is omitted.
4. **Hyperparameter sensitivity**: Thresholds of $V_{\mathrm{LB}}$ and $\epsilon$ require manual tuning and vary across different environments.
5. **Unverified scalability**: The maximum evaluated agent count is 20, leaving performance at larger scales open to investigation.

## Related Work & Insights

- **SPRL** (Klink et al., 2021): Self-paced reinforcement learning, the direct foundation of this work.
- **VACL** (Chen et al., 2021): Variational automatic curriculum learning, NeurIPS 2021, targeting multi-agent curricula.
- **DyMA-CL** (Wang et al., 2020): Dynamic multi-agent curriculum learning, employing hand-crafted incremental agent count designs.
- **ALPGMM** (Portelas et al., 2020): Learning-progress-based curriculum method, yet still using differences in returns to measure progress.
- **PER** (Schaul et al., 2015): Prioritized experience replay, utilizing TD-error-driven sample prioritization; SPMARL elevates this concept to the task level.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel problem formulation with a simple and effective solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across three benchmarks with sufficient ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and coherent mathematical derivation.
- Value: ⭐⭐⭐⭐ — Provides a plug-and-play improvement scheme for multi-agent curriculum learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](../../NeurIPS2025/reinforcement_learning/robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[CVPR 2026\] TaskForce: Cooperative Multi-agent Reinforcement Learning for Multi-task Optimization](../../CVPR2026/reinforcement_learning/taskforce_cooperative_multi-agent_reinforcement_learning_for_multi-task_optimiza.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)

</div>

<!-- RELATED:END -->
