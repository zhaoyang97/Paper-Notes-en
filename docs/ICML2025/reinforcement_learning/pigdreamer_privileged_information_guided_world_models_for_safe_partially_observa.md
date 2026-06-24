---
title: >-
  [Paper Note] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL
description: >-
  [ICML 2025][Reinforcement Learning][Safe Reinforcement Learning] This paper proposes the ACPOMDPs theoretical framework and constructs PIGDreamer, which leverages privileged information (e.g., ground-truth states, sensory data) during the training phase through representation alignment, a privileged predictor, and an asymmetric critic to enhance world model-based safe RL. It achieves a 136% performance improvement with only 28% additional training time in partially observable…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Safe Reinforcement Learning"
  - "World Models"
  - "Partial Observability"
  - "Privileged Information"
  - "Asymmetric Actor-Critic"
date: 2026-05-08
content_hash: 2627e32427d4b096
---

# PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL

**Conference**: ICML 2025  
**arXiv**: [2508.02159](https://arxiv.org/abs/2508.02159)  
**Code**: [https://github.com/hggforget/PIGDreamer](https://github.com/hggforget/PIGDreamer)  
**Area**: Reinforcement Learning  
**Keywords**: Safe Reinforcement Learning, World Models, Partial Observability, Privileged Information, Asymmetric Actor-Critic

## TL;DR

This paper proposes the ACPOMDPs theoretical framework and constructs PIGDreamer, which leverages privileged information (e.g., ground-truth states, sensory data) during the training phase through representation alignment, a privileged predictor, and an asymmetric critic to enhance world model-based safe RL. It achieves a 136% performance improvement with only 28% additional training time in partially observable environments.

## Background & Motivation

**Background**: Safe Reinforcement Learning (Safe RL) aims to maximize rewards while satisfying safety constraints, typically modeled as constrained Markov decision processes (CMDPs). Recently, world model-based methods (such as DreamerV3 and SafeDreamer) have made significant progress in partially observable safe RL. However, they rely solely on partial observations to construct world models, failing to fully unleash the potential of the model.

**Limitations of Prior Work**: Partial observability poses double challenges to safe RL—exponential growth of computational complexity (representation space $|\Gamma_t| = O(|A||\Gamma_{t-1}|^{|Z|})$) and inaccurate risk assessment. Existing methods either completely ignore privileged information or utilize it inefficiently (e.g., Scaffolder requires an additional 69% of training time), leading to safety constraint violations or limited performance.

**Key Challenge**: During actual deployment, the training stage can often obtain richer information than the testing stage (such as background state of simulators, extra sensors), but there lacks theoretical guidance on how to efficiently utilize this privileged information to simultaneously improve safety and performance.

**Goal**: (1) Address the lack of theoretical guarantees for privileged information in safe RL; (2) Improve the low training efficiency of existing privileged information utilization methods; (3) Achieve policies that obtain near-full-information performance at deployment while using only partial observations.

**Key Insight**: Starting from the representation complexity of the value function of POMDP belief states, it is proven that privileged information can reduce the representation space of the value function from exponential to $|S|$, thereby significantly reducing the number of Critic updates and producing superior or optimal policies.

**Core Idea**: Building an asymmetric architecture—allowing the Critic and the predictor to access privileged information during training to obtain more accurate estimates, while distilling privileged knowledge into the Actor (which only depends on partial observations) through representation alignment.

## Method

### Overall Architecture

PIGDreamer is built on DreamerV3, with its core pipeline being: (1) simultaneously training two world models—a naive world model (receiving only observations $o_t$) and a privileged world model (receiving underlying state $i_t$); (2) generating abstract trajectories via Twisted Imagination; (3) letting the Actor make decisions based solely on the naive representation $s_t^-$, while the Critic accesses both $s_t^-$ and the privileged representation $s_t^+$ for value estimation; (4) deploying only the naive world model and the Actor.

### Key Designs

1. **ACPOMDPs Theoretical Framework**:

    - **Function**: Provide a theoretical foundation for the use of privileged information in safe RL.
    - **Mechanism**: Relax standard CPOMDPs into ACPOMDPs, allowing the Critic to access the underlying state $s$ instead of the belief state $b$ to update the value function $V_R^*(s) = \max_{a} [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_R^*(s')]$, and then aggregate via $V_R^*(b) = \sum_{s} b(s) V_R^*(s)$.
    - **Design Motivation**: ACPOMDPs compress the representation space of the value function from $O(|A||\Gamma_{t-1}|^{|Z|})$ to $|S|$. Theorem 3.3 proves that $V_{asym}^*(b) \geq V_{sym}^*(b)$, meaning the asymmetric architecture consistently produces better policies and estimates safety risks more accurately.

2. **Privileged Representation Alignment**:

    - **Function**: Distill privileged information knowledge into the state representation of the naive world model.
    - **Mechanism**: Introduce the Oracle Posterior $q_\phi(s_t^* | \hat{s}_t^-, z_t^-, z_t^+)$ into the naive world model to encode both observations and privileged information, and force the naive representation $s^-$ to approach the Oracle representation $s^*$ via the alignment loss $\mathcal{L}_{align} = \mathcal{L}_{rep}(s_t^*, s_t^-)$.
    - **Design Motivation**: Unlike methods that directly reconstruct privileged information $i_t$ from $s^-$, this paper performs indirect distillation via $s^*$, which is more robust when privileged information is overly redundant; ablation studies demonstrate that representation alignment is a key contributor to performance improvements.

3. **Twisted Imagination (TI) Trajectory Generation**:

    - **Function**: Synchronize both world models to generate coherent abstract trajectories for Actor-Critic learning.
    - **Mechanism**: Starting from $s_t^-$ and $s_t^+$ in the replay buffer, the Actor samples actions based on $s_t^-$, and the two world models predict the next states $s_{t+1}^-$ and $s_{t+1}^+$, respectively, up to an imagination horizon $H=15$. The predictor predicts rewards and costs based on the concatenation of $s_t^*$ and $s_t^-$.
    - **Design Motivation**: Compared with Nested Latent Imagination (NLI), TI uses a more lightweight model design, significantly improving robustness and generalization while achieving competitive performance.

### Loss & Training

The total loss of the world model is $\mathcal{L}_\phi = \mathcal{L}_{dyn} + \mathcal{L}_{align} + \mathcal{L}_{dec} + \mathcal{L}_{pred}$, where the dynamics loss utilizes KL divergence with stop-gradient $\mathcal{L}_{rep}(q,p) = \alpha \text{KL}[q \| \text{sg}(p)] + \beta \text{KL}[\text{sg}(q) \| p]$. Policy optimization adopts the augmented Lagrangian method, where the objective function simultaneously maximizes rewards, satisfies safety constraints, and encourages exploration (via the entropy regularization term $\eta H[\pi_\theta]$).

## Key Experimental Results

### Main Results

| Benchmark/Method | Reward (Median/IQM/Mean) | Cost (Median/IQM/Mean) | Safety Constraints |
|-----------|----------------------|----------------------|---------|
| SafeDreamer | Baseline | Baseline | Partially satisfied |
| Scaffolder (Lag) | Outperforms SafeDreamer | Higher than SafeDreamer | Partially violated |
| LAMBDA | Matches PIGDreamer reward | High cost, unsatisfied constraints | Violated |
| Safe-SLAC | PointPush1/RacecarGoal1 failed | Significantly violated | Violated |
| **PIGDreamer** | **Overall SOTA** | **Near-zero cost** | **Satisfied** |

### Guard Benchmark Privileged Methods Comparison

| Method | Relative Performance Gain over SafeDreamer | Additional Training Time | Safety |
|------|----------------------|------------|--------|
| Distill (Lag) | Performance degradation (information gap) | - | - |
| Informed-Dreamer (Lag) | Slight improvement | +10% | Fair |
| Scaffolder (Lag) | Significant improvement | +69% | Partially degraded |
| **PIGDreamer** | **+136%** | **+28%** | **Optimal** |

### Ablation Study

| Configuration | Reward Trend | Safety | Description |
|------|---------|--------|------|
| PIGDreamer (Full) | Highest | Near-zero cost | Full model |
| PIG - No Rep | Slight improvement | - | Significant degradation after removing representation alignment |
| PIG - Unprivileged | Baseline | Baseline | No privileged info, equivalent to SafeDreamer |
| NLI (vs TI) | Competitive performance | - | TI is more lightweight and has better robustness |

### Detailed Comparison of TI vs NLI

| Task | NLI Reward | TI Reward | NLI Cost | TI Cost |
|------|---------|--------|---------|--------|
| SafetyPointGoal2 | 10.79 | **13.59** | 0.41 | 0.73 |
| SafetyCarGoal1 | 14.79 | **17.32** | 0.64 | **0.43** |
| SafetyRacecarGoal1 | **13.99** | 11.38 | 1.57 | **0.83** |

### Key Findings
- Representation alignment is the core driver of performance improvement. After its removal, PIG-No Rep only slightly improves over the unprivileged version, because the privileged Critic can only indirectly assist the Actor through more accurate valuations, whereas representation alignment enables the Actor to directly obtain richer information.
- PIGDreamer achieves a 136% performance improvement on the Guard benchmark relative to alternative methods while requiring only 28% additional training time, which is far more efficient than Scaffolder (requires 69% additional time).
- Privileged information does not always bring improvements in certain tasks (for example, Scaffolder actually suffers performance degradation on Guard), indicating that how privileged information is utilized is more critical than whether it is used.

## Highlights & Insights

- **Theoretical Guarantees for Asymmetric Architecture**: The ACPOMDPs framework rigorously proves the value of privileged information from an information-theoretic perspective—not only improving policy performance but also estimating safety risks more accurately (whereas CPOMDPs tend to underestimate risks). This theoretical result is generalizable and can guide other privileged learning scenarios.
- **Robust Design for Oracle Representation Distillation**: Distilling privileged information via the intermediate bridge $s^*$ avoids degradation caused by information overload when directly reconstructing privileged information from partial observations. This indirect distillation strategy can be transferred to other knowledge distillation tasks.
- **Excellent Balance of Efficiency and Performance**: Beyond the three routes of Scaffolder (requiring an additional discovery actor), Informed-Dreamer (only doing reconstruction), and Distill (direct policy distillation), PIGDreamer finds a more efficient way to utilize privileged information.

## Limitations & Future Work

- Privileged information does not always lead to improvements across all tasks; the authors point out the need to further investigate the relationship between specific types of privileged information and tasks.
- The experimental scenarios are limited to Safety-Gymnasium and Guard simulation environments (64×64 pixel RGB images), and have not yet been validated on real robots or higher-dimensional visual inputs.
- The training process requires maintaining two world models simultaneously, which may cause memory overhead bottlenecks in larger-scale environments.
- The privileged information assumed by the linear selection mechanism (underlying state + action + proprioception) has not yet been explored in more complex types of privileged information (such as natural language instructions, expert demonstrations).

## Related Work & Insights

- **vs SafeDreamer**: SafeDreamer integrates the Lagrangian method into DreamerV3 to achieve zero-cost performance but relies only on partial observations. PIGDreamer significantly enhances performance on top of this by incorporating privileged information.
- **vs Scaffolder**: Scaffolder achieves improvements by providing privileged information to the predictor, Critic, and an additional exploration Actor, but the excessive components lead to low training efficiency (+69% time). PIGDreamer replaces additional components with representation alignment to achieve higher efficiency.
- **vs Informed-Dreamer**: Informed-Dreamer only reconstructs privileged information via auxiliary targets, yielding limited improvements. PIGDreamer's multi-level utilization of privileged information (representation + prediction + Critic) is more comprehensive.

## Rating

- Novelty: ⭐⭐⭐⭐ The theoretical framework ACPOMDPs has clear contributions, but the asymmetric architecture design itself is not entirely new in privileged learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ The coverage of two benchmarks, various comparison methods, and ablation studies is comprehensive, but validation in real-world environments is missing.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivations are clear, experimental visualizations are abundant, and the structure is well-organized.
- Value: ⭐⭐⭐⭐ Provides a unified theoretical and practical scheme for leveraging privileged information in safe RL, showing promising application prospects in Sim2Real scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Flow-Equivariant World Models: Memory for Partially Observed Dynamic Environments](../../ICML2026/reinforcement_learning/flow_equivariant_world_models_memory_for_partially_observed_dynamic_environments.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](../../ICLR2026/reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)
- [\[NeurIPS 2025\] Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/learning_to_focus_prioritizing_informative_histories_with_structured_attention_m.md)
- [\[ICLR 2026\] PAMDP: Interact to Persona Alignment via a Partially Observable Markov Decision Process](../../ICLR2026/reinforcement_learning/pamdp_interact_to_persona_alignment_via_a_partially_observable_markov_decision_p.md)
- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](../../NeurIPS2025/reinforcement_learning/foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)

</div>

<!-- RELATED:END -->
