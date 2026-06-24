---
title: >-
  [Paper Note] Actor-Critics Can Achieve Optimal Sample Efficiency
description: >-
  [ICML 2025][Reinforcement Learning][Actor-Critic] This paper is the first to prove that Actor-Critic algorithms can achieve an optimal sample complexity of $O(1/\epsilon^2)$ under general function approximation and strategic exploration. This is achieved by integrating optimistic exploration, off-policy critic estimation, and rare-switching policy resets, and the results are further extended to the hybrid RL setting.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Actor-Critic"
  - "Sample Complexity"
  - "Bellman Eluder Dimension"
  - "Hybrid RL"
  - "General Function Approximation"
date: 2026-05-08
content_hash: 9ac035d643e09e80
---

# Actor-Critics Can Achieve Optimal Sample Efficiency

**Conference**: ICML 2025  
**arXiv**: [2505.03710](https://arxiv.org/abs/2505.03710)  
**Code**: None  
**Area**: Reinforcement Learning Theory  
**Keywords**: Actor-Critic, Sample Complexity, Bellman Eluder Dimension, Hybrid RL, General Function Approximation

## TL;DR

This paper is the first to prove that Actor-Critic algorithms can achieve an optimal sample complexity of $O(1/\epsilon^2)$ under general function approximation and strategic exploration. This is achieved by integrating optimistic exploration, off-policy critic estimation, and rare-switching policy resets, and the results are further extended to the hybrid RL setting.

## Background & Motivation

**Background**: Actor-Critic algorithms are core methods in reinforcement learning (RL), merging the advantages of both policy-based and value-based paradigms. In practice (e.g., PPO, SAC), Actor-Critic has achieved significant success. Theoretically, progress has also been made in understanding their statistical efficiency in recent years.

**Limitations of Prior Work**: Despite theoretical progress, a critical open question remains unsolved—**under general function approximation and settings requiring strategic exploration, no existing Actor-Critic algorithm can learn an $\epsilon$-optimal policy with an optimal sample complexity of $O(1/\epsilon^2)$ trajectories**. While this is the optimal rate in RL theory, existing Actor-Critic methods either only achieve it in simple settings (such as tabular) or suffer from suboptimal rates under general function approximation.

**Key Challenge**: The theoretical analysis of Actor-Critic faces unique challenges—the coupled updates of the Actor (policy) and Critic (value function) make the analysis highly difficult. In particular: (1) How does the frequency of policy updates affect sample efficiency? (2) How can the accuracy of the Critic be guaranteed during off-policy estimation? (3) How can effective optimistic exploration be implemented over general function classes?

**Goal**: To address the aforementioned open questions—design an Actor-Critic algorithm that achieves the optimal sample complexity of $O(1/\epsilon^2)$ under the framework of general function approximation with Bellman Eluder dimension. Meanwhile, address another open question in hybrid RL regarding whether optimism can be eliminated.

**Key Insight**: To integrate three key techniques: (1) optimistic exploration, (2) off-policy critic estimation targeted at the optimal Q-function, and (3) rare-switching policy resets.

**Core Idea**: By forcing the Critic to directly estimate the optimal Q-function (rather than the Q-function of the current policy) in conjunction with optimistic upper bounds and rare policy switching, the coupling bottleneck in Actor-Critic analysis is bypassed, achieving optimal sample efficiency for the first time.

## Method

### Overall Architecture

Algorithmic Framework:

1. **Critic Update**: Construct an optimistic estimation of the optimal Q-function using off-policy data.
2. **Actor Update**: Select the policy that maximizes the Q-value based on the Critic's estimation.
3. **Rare-Switching**: The Actor does not update at every step; instead, it switches policies only when certain conditions are met.

Key Output: $\epsilon$-optimal policy $\pi$, satisfying $V^* - V^\pi \leq \epsilon$

### Key Designs

1. **Optimistic Off-Policy Critic**:

    - **Function**: The Critic does not estimate the Q-function of the current policy $\pi_t$, but instead directly estimates an optimistic upper bound of the optimal Q-function $Q^*$.
    - **Mechanism**: A confidence set $\mathcal{F}_t$ is constructed using a function class $\mathcal{F}$, from which an optimistic Q-function estimate is selected. Crucially, all historical data (including data collected by different policies) can be utilized to update the Critic, as it estimates the policy-independent $Q^*$. Formally:
  
    $\hat{Q}_t = \arg\max_{f \in \mathcal{F}_t} f(s_t, \cdot)$
   
   where $\mathcal{F}_t$ is the optimistic function class constructed based on the confidence set.
    - **Design Motivation**: In conventional Actor-Critic, the Critic estimates the Q-function of the current policy, which requires the Critic to refit whenever the policy changes, leading to coupling and bias issues. Directly estimating $Q^*$ eliminates this coupling—no matter how the Actor changes, the target of the Critic remains unchanged.

2. **Rare-Switching Policy Resets**:

    - **Function**: The Actor does not update the policy frequently; instead, it only executes policy switches after "sufficient new information" has been accumulated.
    - **Mechanism**: A switching condition is defined (usually based on the change in data coverage), and a new policy is generated by the Critic only when the condition is met. The number of policy switches is of order $O(\log T)$ throughout the entire training process.
    - **Design Motivation**: Frequent policy switches introduce off-policy bias—if the policy changes at every step, the value of historical data to the new policy decreases. Rare switching ensures that the policy remains stable for most of the time, allowing for more accurate off-policy Critic estimation.

3. **Bellman Eluder Dimension Framework**:

    - **Function**: The analysis is performed under the general function approximation framework with Bellman Eluder (BE) dimension $d$.
    - **Mechanism**: The BE dimension is an index to measure the complexity of function classes, unifying various settings such as tabular MDPs, linear MDPs, and low-rank MDPs. The sample complexity of the algorithm centers around the key parameter $d$:
  
    $N = O\left(\frac{dH^5 \log|\mathcal{A}|}{\epsilon^2} + \frac{dH^4 \log|\mathcal{F}|}{\epsilon^2}\right)$
   
   where $H$ is the episode length, $|\mathcal{A}|$ is the size of the action space, and $|\mathcal{F}|$ is the size of the Critic function class.
    - **Design Motivation**: The BE dimension is currently one of the most general complexity measures in RL. Proving optimality within this framework implies that the results automatically generalize to various specific MDP types.

4. **Hybrid RL Extension**:

    - **Function**: Utilize offline data to initialize the Critic, achieving improved sample efficiency compared to purely online or offline settings.
    - **Mechanism**: Offline data provides a "warm start"—specifically, using $N_\text{off}$ offline trajectories to initialize the Critic's confidence set, thereby requiring fewer online samples. Notably, if the offline data size satisfies $N_\text{off} \geq c^*_\text{off} d H^4 / \epsilon^2$ (where $c^*_\text{off}$ is the single-policy reachability coefficient), optimism can be completely bypassed, and a **non-optimistic** Actor-Critic can still achieve optimal efficiency.
    - **Design Motivation**: This addresses another open question in Hybrid RL—whether optimistic exploration (which can be hard to implement in practice) can be avoided when offline data is available.

### Theoretical Results

- **Online RL**: Sample complexity of $O(dH^5\log|\mathcal{A}|/\epsilon^2 + dH^4\log|\mathcal{F}|/\epsilon^2)$, corresponding to a regret of $\tilde{O}(\sqrt{T})$.
- **Hybrid RL (Optimistic Version)**: Leverages offline data to reduce the required online sample size.
- **Hybrid RL (Non-Optimistic Version)**: Trades additional offline data requirements for the elimination of optimism.

## Key Experimental Results

### Main Results (Theoretical Comparison)

| Algorithm | Sample Complexity | General Function Approximation | Strategic Exploration | Actor-Critic |
|------|-----------|-------------|-----------|--------------|
| GOLF (value-based) | $O(1/\epsilon^2)$ | ✓ | ✓ | ✗ |
| Existing AC Methods | $O(1/\epsilon^3)$ or worse | Partial | Partial | ✓ |
| **Ours (OPAC)** | $O(1/\epsilon^2)$ | ✓ | ✓ | **✓** |

### Ablation Study (Numerical Experiments)

| Setting | Ours | Baseline AC | Description |
|------|---------|------------|------|
| Tabular MDP | Optimal convergence | Suboptimal convergence | Validates the theoretical $O(1/\epsilon^2)$ |
| Linear MDP | Optimal convergence | Suboptimal | A special case of general function approximation |
| Hybrid RL (with offline data) | Accelerated convergence | Unable to utilize | Validates the effectiveness of offline data initialization |
| Non-optimistic Hybrid RL | Feasible | Infeasible | Sufficient offline data allows removing optimism |

### Key Findings

1. **First to Achieve Optimal Rate**: Under the complete setting of general function approximation and strategic exploration, Actor-Critic achieves the optimal sample complexity of $O(1/\epsilon^2)$ for the first time, matching purely value-based methods.
2. **Rare-Switching is Crucial**: $O(\log T)$ policy switches balance off-policy bias with information utilization efficiency.
3. **Decoupling via Off-Policy Critic**: Directly estimating $Q^*$ instead of $Q^\pi$ is the core technique to break the coupling analysis bottleneck of Actor-Critic.
4. **Practical Significance of Hybrid RL**: Offline data can "buy" the freedom to eliminate optimism, which is highly valuable for practical applications where optimistic exploration is typically difficult to implement.

## Highlights & Insights

- **Solving Important Open Problems**: Both the optimal sample efficiency of Actor-Critic methods and the elimination of optimism in Hybrid RL are prominent open questions highlighted across multiple papers in RL theory.
- **Elegant Technical Combination**: The three key designs (optimistic Critic, rare switching, and $Q^*$ estimation) each resolve a specific analytical bottleneck, combining seamlessly to achieve optimality.
- **Bridge Between Theory and Practice**: Although this is a theoretical paper, the insights about rare switching and utilizing offline data to remove optimism offer valuable guidelines for designing practical Actor-Critic algorithms.

## Limitations & Future Work

1. This is a purely theoretical work; the numerical experiments only validate basic settings without evaluations on large-scale or real-world RL problems.
2. The dependency on $H$ ($H^5$) may not be tight, leaving room for further optimization.
3. The algorithm requires prior knowledge of the function class $\mathcal{F}$ and the action space $\mathcal{A}$; finding appropriate function classes remains a challenge in practice.
4. The formulation of optimism is difficult to implement precisely in practical applications.
5. Extension to continuous action spaces requires further investigation.

## Related Work & Insights

- **GOLF (Jin et al.)**: Achieves $O(1/\epsilon^2)$ under the BE dimension framework but is not an Actor-Critic method.
- **PEVI (Jin et al.)**: Theoretical analysis of offline RL, which is related to the Hybrid RL extension in this paper.
- **OPPO / PC-PG**: Prior theoretical efforts on Actor-Critic, but they suffered from suboptimal sample complexities.
- **Insight**: In the theoretical analysis of Actor-Critic, **what the Critic estimates** ($Q^*$ vs $Q^\pi$) might be significantly more important than **how it estimates**.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Addresses two clear open questions with substantial innovation in its technical approach.
- Experimental Thoroughness: ⭐⭐⭐ Appropriate experimental scope for a theory-focused paper, though lacking large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical proofs; motivations and contributions are clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Significantly advances RL theory and resolves long-standing open problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents](../../ICLR2026/reinforcement_learning/simplicial_embeddings_improve_sample_efficiency_in_actorcritic_agents.md)
- [\[ICML 2025\] Optimal and Practical Batched Linear Bandit Algorithm](optimal_and_practical_batched_linear_bandit_algorithm.md)
- [\[ICML 2025\] Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL](pessimism_principle_can_be_effective_towards_a_framework_for_zero-shot_transfer_.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICML 2025\] The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability](the_sample_complexity_of_online_strategic_decision_making_with_information_asymm.md)

</div>

<!-- RELATED:END -->
