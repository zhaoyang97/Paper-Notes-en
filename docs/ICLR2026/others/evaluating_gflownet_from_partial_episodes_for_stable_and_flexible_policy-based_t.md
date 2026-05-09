---
title: >-
  [Paper Note] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training
description: >-
  [ICLR 2026][GFlowNet] This paper establishes a theoretical connection between state flow functions and policy value functions in GFlowNet, proposes the Subtrajectory Evaluation Balance (Sub-EB) objective for reliable value function learning, and enhances the stability and flexibility of policy-based GFlowNet training.
tags:
  - ICLR 2026
  - GFlowNet
  - policy gradient
  - value function
  - flow balance
  - combinatorial optimization
date: 2026-05-08
content_hash: 31e42d4f09175579
---

# Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training

**Conference**: ICLR 2026
**arXiv**: [2603.01047](https://arxiv.org/abs/2603.01047)
**Code**: [github.com/niupuhua1234/Sub-EB](https://github.com/niupuhua1234/Sub-EB)
**Area**: Other
**Keywords**: GFlowNet, policy gradient, value function, flow balance, combinatorial optimization

## TL;DR

This paper establishes a theoretical connection between state flow functions and policy value functions in GFlowNet, proposes the Subtrajectory Evaluation Balance (Sub-EB) objective for reliable value function learning, and enhances the stability and flexibility of policy-based GFlowNet training.

## Background & Motivation

### GFlowNet Overview

Generative Flow Networks (GFlowNet) are generative models that sample over a combinatorial space $\mathcal{X}$, aiming to sample $x \in \mathcal{X}$ with probability proportional to a reward function $R(x)$. The generation process is decomposed into incremental trajectories over a directed acyclic graph (DAG), where a forward policy $\pi_F$ constructs objects step by step.

### Two Training Paradigms

**Value-based methods**: Introduce a flow function $F(s)$ and implicitly minimize distributional divergence through flow balance conditions (e.g., Sub-TB). These methods support off-policy sampling but flow balance does not directly reflect true reward information.

**Policy-based methods**: Adopt an Actor-Critic framework from RL, introducing a value function $V(s)$ to estimate the KL divergence of the policy at state $s$, then updating $\pi_F$ via policy gradients. On-policy training is more efficient.

### Core Problem

The key bottleneck of policy-based methods lies in **reliably learning the value function $V(s)$**. Existing methods (the $\lambda$-TD objective) are based on edge-level mismatches, considering only events and edge mismatches after state $s$, providing insufficient learning signal and requiring a fixed backward policy $\pi_B$. The paper's key insight is that there exists a deep connection between the flow function $F(s)$ and the value function $V(s)$: for a fixed $\pi_F$, the solution to the flow balance condition coincides exactly with the true value function.

## Method

### Overall Architecture

The core contribution is the **Sub-EB (Subtrajectory Evaluation Balance)** condition and objective, which imports the successful flow balance idea from value-based methods into value function learning within the policy-based framework. The overall training pipeline remains Actor-Critic:
1. **Critic (value function learning)**: Optimize $V(\cdot; \phi)$ using the Sub-EB objective.
2. **Actor (policy update)**: Update $\pi_F(\cdot; \theta)$ based on $V$ and policy gradients.

### Key Designs

#### 1. Theoretical Connection Between Flow Functions and Value Functions

**Theorem 3.1 (Core Theorem)**: For any value function $V$, the following equivalence holds:
$$V(s_h) = \log F^*(s_h) - D_{\text{KL}}(P_F(\tau_{h:}|s_h) \| P_B(\tau_{h:}|s_h))$$
if and only if $V$ satisfies the Sub-EB condition.

Intuition: $V(s)$ equals the log of the optimal flow minus the KL divergence of trajectories starting from $s$. That is, the value function simultaneously encodes both "state importance" and "current policy deviation."

**Theorem 3.2 (Relation to Sub-TB)**: In value-based methods, the flow function $F$ and the optimal policy $\pi_F^*$ jointly satisfy the Sub-TB condition if and only if $\log F(s_h) = \log F^*(s_h) - D_{\text{KL}}(P_{F^*}(\tau_{h:}|s_h) \| P_B(\tau_{h:}|s_h))$ and $\pi_F = \pi_F^*$.

#### 2. Sub-EB Condition

For all $i < j \in [H+1]$:

$$\mathbb{E}_{P_F(\tau_{i:j})} \left[ \log(P_F(\tau_{i:j}|s_i) \exp V(s_i)) \right] = \mathbb{E}_{P_F(\tau_{i:j})} \left[ \log(P_B(\tau_{i:j}|s_j) \exp V(s_j)) \right]$$

Intuition: The difference $V(s_i) - V(s_j)$ should equal the true divergence of the subtrajectory from $s_i$ to $s_j$.

#### 3. Sub-EB Training Objective

$$\mathcal{L}_V(\phi) = \mathbb{E}_{P_F(\tau)} \left[ \sum_{\tau_{i:j}} w_{j-i} \left( \log \frac{P_F(\tau_{i:j}|s_i) \exp V(s_i; \phi)}{P_B(\tau_{i:j}|s_j) \exp V(s_j; \phi)} \right)^2 \right]$$

This is structurally similar to the Sub-TB objective, with key distinctions:
- Sub-EB learns $V$ under a fixed $\pi_F$ (only updates $\phi$).
- Sub-TB jointly optimizes $(\pi_F, \log F)$ (updates $\theta$).
- Sub-EB takes expectations under $P_F$ (on-policy); Sub-TB under $P_\mathcal{D}$ (can be off-policy).

#### 4. Support for Parameterized Backward Policy

The $\lambda$-TD objective requires $\pi_B$ to be fixed. In contrast, Sub-EB and Sub-TB naturally support a **parameterized $\pi_B$** — $\pi_B$ can be jointly updated with $V$ via Sub-EB gradients, without requiring an additional backward phase or auxiliary objective.

#### 5. Off-Policy-Based Training

By introducing a backward value function $W$ (evaluating the quality of $\pi_B$), **offline sampling combined with policy-based training** becomes feasible:
- Sample terminal states $x$ using a data-collection policy $\pi_\mathcal{D}$.
- Backtrack trajectories from $x$ using $\pi_B$.
- Learn $W$ using the backward Sub-EB objective.
- Update $\pi_B$ using policy gradients from $W$.

### Loss & Training

**Online training (Algorithm 1)**:
1. Sample a batch $\mathcal{D} \sim P_F(\tau)$.
2. Update the value function $V$ via $\nabla_\phi \hat{\mathcal{L}}_V(\phi)$ on $\mathcal{D}$.
3. Update $\pi_F$ via policy gradients using $\mathcal{D}$ and $V$.

**Offline training (Algorithm 2)**: Sample with $\pi_\mathcal{D}$ → backtrack with $\pi_B$ → update $W$ → update $\pi_B$.

Weight coefficients are $w_{j-i} = \lambda^{j-i} / \sum \lambda^{j-i}$, where $\lambda$ controls the preference for short vs. long subtrajectories.

## Key Experimental Results

### Main Results

**Hypergrid experiments (exact $D_{\text{TV}}$ computation)**

| Method | 256×256 Convergence | 128×128×128 Stability | 64×64×64 Final Performance |
|------|:-:|:-:|:-:|
| CV (empirical gradient) | Poor | Poor | Moderate |
| RL ($\lambda$-TD) | Moderate (unstable) | Unstable | Good |
| **Sub-EB** | **Good (stable & fast)** | **Stable & fast** | **Good** |
| Sub-TB (value-based) | Moderate | Moderate | Moderate |

Sub-EB significantly improves the stability and convergence speed of policy-based methods, with particularly pronounced advantages in high-dimensional and large-scale grids.

**Bayesian network structure learning (real tasks, 10/15 nodes)**

| Method | 10-node Avg. Reward | 10-node Diversity | 10-node FCS |
|------|:-:|:-:|:-:|
| Sub-TB | Moderate | Moderate | Moderate |
| Q-Much | Moderate | Moderate | Moderate |
| RL | High | Good | Good |
| **Sub-EB** | **Highest** | **Good** | **Good** |
| **Sub-EB-B** | **Highest (+offline boost)** | Slightly lower | - |

### Ablation Study

**Parameterized $\pi_B$ ablation (256×256, 128×128, 64×64×64)**

| Method | Performance |
|------|---------|
| Sub-EB-P (parameterized $\pi_B$) | Best among all methods, most stable training |
| RL-P (two-stage) | Second best; additional backward phase increases complexity |
| RL-MLE (maximum likelihood) | Inferior to Sub-EB-P |
| Sub-TB-P (parameterized $\pi_B$) | Comparable to Sub-TB |

This confirms the advantage of Sub-EB in naturally supporting parameterized backward policies.

### Key Findings

1. **Sub-EB vs. $\lambda$-TD**: Exploiting subtrajectory-level balance information (rather than only edge-level) significantly improves the reliability of value function learning.
2. **Policy-based vs. value-based**: Policy-based methods (RL, Sub-EB) generally outperform value-based methods (Sub-TB, Q-Much) in convergence speed and distribution modeling quality.
3. **Effectiveness of offline augmentation**: Sub-EB-B (with local search integration) achieves the highest reward in BN structure learning, validating Sub-EB's compatibility with offline techniques.
4. **Molecule graph design** ($|\mathcal{X}| \approx 10^{16}$): Sub-EB performs robustly on large-scale tasks, achieving higher average rewards and faster convergence.

## Highlights & Insights

1. **Theoretical elegance**: Reveals a "differs by a KL divergence" relationship between flow functions and value functions, unifying the value-based and policy-based perspectives.
2. **Plug-and-play**: The Sub-EB objective can directly replace the $\lambda$-TD objective with minimal changes to the training pipeline.
3. **Triple flexibility**: Supports (1) parameterized backward policies, (2) offline data collection, and (3) flexible subtrajectory weighting schemes.
4. **Deep correspondence with value-based methods**: Sub-EB stands in relation to policy-based methods as Sub-TB does to value-based methods — formally symmetric and semantically complementary.

## Limitations & Future Work

1. The strategy for selecting weight coefficients $w_{j-i}$ can be further optimized (currently using fixed $\lambda$-decay).
2. Theoretical analysis assumes a layered DAG (although general DAGs can be converted), requiring dummy states in practice.
3. Integration with more advanced policy-based methods (e.g., full TRPO implementations) has not yet been explored.
4. Hypergrid experiments rely on exact computation; approximate metrics such as FCS are needed at larger scales.
5. No theoretically optimal guidance for selecting $\gamma$ in the variance–bias tradeoff of policy gradients.

## Related Work & Insights

- **Sub-TB** (Madan et al., 2023): The subtrajectory balance objective for value-based methods; the formal inspiration for Sub-EB.
- **GFlowNet policy gradient** (Niu et al., 2024): Proposes the Actor-Critic framework and the $\lambda$-TD objective; Sub-EB directly improves its critic learning.
- **Q-Much / RFI**: RL variants of value-based methods, used as baselines in BN experiments.
- Inspiration: Can the "balance condition → value function" approach of Sub-EB be generalized to value function learning in general RL?

## Rating

| Dimension | Score |
|------|------|
| Novelty | ★★★★☆ |
| Technical Depth | ★★★★★ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★☆ |
| Value | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)
- [\[NeurIPS 2025\] Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](../../NeurIPS2025/others/hybrid-balance_gflownet_for_solving_vehicle_routing_problems.md)
- [\[ICLR 2026\] Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](../../CVPR2026/others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)

</div>

<!-- RELATED:END -->
