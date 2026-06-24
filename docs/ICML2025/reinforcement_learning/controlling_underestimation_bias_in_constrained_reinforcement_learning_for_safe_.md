---
title: >-
  [Paper Note] Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration
description: >-
  [ICML 2025 Oral][Reinforcement Learning][Constrained Reinforcement Learning] Proposes MICE (Memory-driven Intrinsic Cost Estimation)—a method that stores historical high-cost states through a flashbulb memory mechanism and constructs intrinsic cost signals to correct the underestimation bias of the cost value function, significantly reducing constraint violations during the training of constrained RL.
tags:
  - "ICML 2025 Oral"
  - "Reinforcement Learning"
  - "Constrained Reinforcement Learning"
  - "Safe Exploration"
  - "Underestimation Bias"
  - "Intrinsic Cost"
  - "Flashbulb Memory"
date: 2026-05-08
content_hash: f705ca1c5c37911e
---

# Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration

**Conference**: ICML 2025 Oral  
**arXiv**: [2601.11953](https://arxiv.org/abs/2601.11953)  
**Code**: [https://github.com/ShiqingGao/MICE](https://github.com/ShiqingGao/MICE)  
**Area**: Reinforcement Learning  
**Keywords**: Constrained Reinforcement Learning, Safe Exploration, Underestimation Bias, Intrinsic Cost, Flashbulb Memory

## TL;DR
Proposes MICE (Memory-driven Intrinsic Cost Estimation)—a method that stores historical high-cost states through a flashbulb memory mechanism and constructs intrinsic cost signals to correct the underestimation bias of the cost value function, significantly reducing constraint violations during the training of constrained RL.

## Background & Motivation

**Background**: Constrained Reinforcement Learning (CRL) aims to maximize cumulative rewards while satisfying safety constraints (e.g., preventing robot collisions, avoiding traffic violations in autonomous driving), serving as a core paradigm of safe RL. Mainstream methods are categorized into primal-dual methods (e.g., NPG-PD) and primal methods (e.g., CPO, CUP).

**Limitations of Prior Work**: Existing CRL algorithms often suffer from severe constraint violations during training. Primal-dual methods are sensitive to initial parameters, while primal methods frequently violate constraints in practice despite theoretical guarantees. The root cause has been overlooked.

**Key Challenge**: Underestimation bias of the cost value function—noise in function approximation breaks the zero-mean assumption under the minimization objective, making high-cost states "look" safer than they actually are, which attracts the agent to explore them. This is a mirror problem to the overestimation bias of reward values (which has been widely studied).

**Goal**: Correct the underestimation bias of the cost value function and reduce constraint violations during training.

**Key Insight**: Inspired by "flashbulb memory" in cognitive science—humans vividly remember dangerous experiences to avoid risks. An agent is equipped with a flashbulb memory module to store previously explored unsafe states.

**Core Idea**: Intrinsic cost = the "pseudo-visitation count" of the current state to high-cost regions in memory, providing an additional cost signal in high-cost regions to counteract underestimation.

## Method

### Overall Architecture
MICE introduces additional components into the standard CRL framework:
1. **Flashbulb Memory Module**: Records the set of historical high-cost states.
2. **Intrinsic Cost Calculation**: Computes intrinsic cost signals based on pseudo-visitation counts.
3. **Extrinsic-Intrinsic Cost Value Update**: Merges extrinsic and intrinsic costs with a bias correction strategy.
4. Optimizes within a trust region to ensure policy updates are consistent with the memory sampling policy.

### Key Designs

1. **Flashbulb Memory Mechanism**:

    - **Function**: Continuously stores high-cost states encountered by the agent.
    - **Mechanism**: When the extrinsic cost satisfies $c^E(s,a) > \text{threshold}$, the state $s$ is stored in the memory buffer $\mathcal{M}$.
    - **Design Motivation**: Analogous to human flashbulb memory—humans remember dangerous scenarios particularly clearly (e.g., near-miss car accidents), and this memory helps avoid similar situations in the future.
    - Implementation: A FIFO buffer with limited capacity to retain recent high-cost experiences.

2. **Intrinsic Cost**:

    - **Function**: Calculates an additional cost based on the similarity between the current state and high-cost states in memory.
    - **Mechanism**: $c^I(s) = \frac{1}{\sqrt{N(s, \mathcal{M})}}$, where $N(s, \mathcal{M})$ is the pseudo-visitation count of state $s$ near the memory.
    - Key Characteristics: In high-cost regions (where many similar states exist in memory), $c^I$ is larger $\rightarrow$ correcting underestimation; in low-cost regions, $c^I$ is smaller $\rightarrow$ not interfering with normal learning.
    - **Design Motivation**: Targeted correction—increasing cost estimation only where needed (high-cost regions) instead of global inflation.

3. **Extrinsic-Intrinsic Cost Value Update + Bias Correction**:

    - **Function**: Integrates extrinsic and intrinsic costs into a unified cost value function.
    - **Mechanism**: $V^\pi_{C+I}(s) = V^\pi_C(s) + \alpha V^\pi_I(s) - \beta(s)$, where $\alpha$ controls the weight of the intrinsic cost, and $\beta(s)$ is the bias correction term.
    - **Design Motivation**: $\alpha$ ensures the correction of underestimation in high-cost regions, while $\beta(s)$ prevents over-correction from inducing excessive conservatism.

4. **Trust Region Optimization**:

    - **Function**: Optimizes the policy within a trust region based on the extrinsic-intrinsic cost value function.
    - **Mechanism**: Constraints the policy update step size to ensure that the new policy remains sufficiently close to the old policy that generated the memory samples.
    - **Design Motivation**: Excessive policy changes would render the information in memory obsolete.

### Loss & Training
- Reward Objective: Maximize $V^\pi_R(s)$ (standard RL objective)
- Constraints: $V^\pi_{C+I}(s) \leq d$ (extrinsic-intrinsic cost value $\leq$ threshold)
- Trust Region Constraint: $D_{KL}(\pi_{new} || \pi_{old}) \leq \delta$
- Convergence Guarantee: Theorem 1 proves that the extrinsic-intrinsic cost value function converges to the correct cost value.
- Constraint Violation Bound: Theorem 2 provides a worst-case upper bound on constraint violations.

## Key Experimental Results

### Main Results
Safety Gymnasium benchmark (various safety-constrained environments):

| Method | Reward ↑ | Number of Constraint Violations ↓ | Constraint Violation Rate ↓ |
|------|--------|-------------|-----------|
| CPO | 25.1 | 342 | 18.5% |
| CUP | 27.3 | 287 | 15.2% |
| PCPO | 24.8 | 315 | 16.8% |
| PPO-Lagrangian | 28.5 | 425 | 22.1% |
| **MICE (CPO+)** | **26.8** | **89** | **4.7%** |
| **MICE (CUP+)** | **28.1** | **72** | **3.8%** |

### Ablation Study

| Configuration | Violations ↓ | Reward ↑ | Description |
|------|----------|--------|------|
| No Intrinsic Cost | 287 | 27.3 | CUP baseline |
| Intrinsic Cost (No Correction) | 105 | 23.1 | Overly conservative |
| **Intrinsic Cost + Bias Correction** | **72** | **28.1** | Optimal balance |
| Random Memory (No High-Cost Filtering) | 198 | 26.5 | Flashbulb filtering is important |
| Global Intrinsic Cost (Non-targeted) | 142 | 24.8 | Targeted is superior to global |

### Key Findings
- MICE reduces constraint violations by 75%+ (287 $\rightarrow$ 72) while keeping rewards almost unchanged (27.3 $\rightarrow$ 28.1).
- The bias correction term $\beta(s)$ is crucial—without it, the intrinsic cost leads to excessive conservatism.
- Flashbulb memory filtering (recording only high-cost states) is much more effective than random memory.
- Targeted intrinsic cost (restricted to high-cost regions) is superior to global intrinsic cost.
- MICE can be integrated into any existing CRL method (such as CPO, CUP, etc.).

## Highlights & Insights
- The identification of **cost underestimation bias** is significant, filling a key puzzle piece in CRL research—underestimation vs. overestimation is a mirror problem, yet cost underestimation had previously been overlooked.
- The cognitive science analogy of flashbulb memory is highly intuitive—humans indeed avoid risks by remembering terrifying experiences.
- The targeted correction strategy is elegant—rather than globally increasing conservatism (which would harm rewards), it selectively "adds weight" to high-cost regions.
- Theoretical guarantees (convergence + constraint violation bound) enhance the reliability of the method.
- The plug-and-play design makes the method immediately applicable to various CRL frameworks.

## Limitations & Future Work
- The memory buffer size and threshold are hyperparameters that require tuning.
- The estimation of the pseudo-visitation count in high-dimensional continuous state spaces may not be accurate enough.
- Validation is limited to Safety Gymnasium; real-world robot scenarios remain to be tested.
- The estimation of the bias correction factor $\beta(s)$ depends on the accuracy of the current cost value, introducing a circular dependency.
- Multi-constraint scenarios (with multiple different safety constraints simultaneously) are not addressed.

## Related Work & Insights
- **vs CPO/PCPO/CUP**: Standard primal methods that do not address underestimation bias, while MICE serves as an add-on module to correct the bias.
- **vs PPO-Lagrangian**: Primal-dual method that is sensitive to initial parameters, whereas MICE does not alter the optimization framework.
- **vs TD3 (Reward Overestimation)**: TD3 uses twin Q-networks to address reward overestimation, whereas MICE uses intrinsic costs to address cost underestimation—the "mirror" of the problem.
- **vs ROSARL**: Interprets constraints as intrinsic rewards, but in the opposite direction—ROSARL uses intrinsic rewards to encourage safety, whereas MICE uses intrinsic costs to deter danger.
- **Insights**: Controlling underestimation/overestimation bias might be an underestimated research direction in CRL.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Identifies and resolves an overlooked fundamental problem in CRL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple environments, multiple baselines, and complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, supported by both theory and experiments.
- **Value**: ⭐⭐⭐⭐⭐ Significantly advances research in safe RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration](../../ICLR2026/reinforcement_learning/off-policy_safe_reinforcement_learning_with_cost-constrained_optimistic_explorat.md)
- [\[ICLR 2026\] Safe Exploration via Policy Priors](../../ICLR2026/reinforcement_learning/safe_exploration_via_policy_priors.md)
- [\[NeurIPS 2025\] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](../../NeurIPS2025/reinforcement_learning/risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)
- [\[ICML 2025\] Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)
- [\[ICLR 2026\] SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration](../../ICLR2026/reinforcement_learning/shapo_sharpness-aware_policy_optimization_for_safe_exploration.md)

</div>

<!-- RELATED:END -->
