---
title: >-
  [Paper Note] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation
description: >-
  [AAAI 2026][Robotics][Multi-Objective Reinforcement Learning] This paper proposes PolicyGradEx, which efficiently estimates policy adaptation performance on arbitrary task subsets via first-order gradient approximation and surrogate models, constructs a task affinity matrix, and performs task grouping through convex optimization. PolicyGradEx outperforms state-of-the-art baselines by an average of 16% on multi-objective RL and meta-RL benchmarks, with a speedup of up to 26×.
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Multi-Objective Reinforcement Learning"
  - "Meta Reinforcement Learning"
  - "Gradient Estimation"
  - "Task Affinity"
  - "Task Grouping"
date: 2026-05-08
content_hash: 54204f3959f6773c
---

# Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation

**Conference**: AAAI 2026
**arXiv**: [2511.12779](https://arxiv.org/abs/2511.12779)  
**Code**: [github.com/VirtuosoResearch/PolicyGradEx](https://github.com/VirtuosoResearch/PolicyGradEx)  
**Area**: Reinforcement Learning
**Keywords**: Multi-Objective Reinforcement Learning, Meta Reinforcement Learning, Gradient Estimation, Task Affinity, Task Grouping

## TL;DR

This paper proposes PolicyGradEx, which efficiently estimates policy adaptation performance on arbitrary task subsets via first-order gradient approximation and surrogate models, constructs a task affinity matrix, and performs task grouping through convex optimization. PolicyGradEx outperforms state-of-the-art baselines by an average of 16% on multi-objective RL and meta-RL benchmarks, with a speedup of up to 26×.

## Background & Motivation

### Problem Background

In multi-objective RL, an agent must simultaneously optimize $n$ objectives (or tasks). The core challenges are:

**Negative Transfer**: When unrelated tasks are jointly trained in a shared network, conflicting gradients degrade performance.

**Task Grouping Problem**: The ideal approach is to partition $n$ objectives into $k \ll n$ groups such that tasks within each group are highly correlated and groups are trained independently. However, finding the optimal grouping requires $2^n$ complete training runs, which is computationally infeasible.

**Pairwise Evaluation Is Insufficient**: Existing methods (e.g., PCGrad) compute only pairwise gradient similarity, failing to capture higher-order interactions that arise during joint multi-task training.

### Core Motivation

**Can policy performance on arbitrary task subsets be accurately estimated without full training?** If so, the task affinity matrix can be computed efficiently to identify the optimal grouping.

The authors observe that **a sufficiently trained policy network admits an accurate first-order approximation**—near the meta-policy, policy outputs are approximately linear in parameter perturbations. This implies that a simple linear model can estimate the effect of fine-tuning.

### Application Scenarios

- **Robotic Control**: A robot must master multiple related skills (e.g., 10 manipulation tasks in Meta-World).
- **Autonomous Driving / Control**: Control policies transfer across environments with different physical parameters.
- **Language Model Preference Optimization**: Objective trade-offs in multi-objective alignment.

## Method

### Overall Architecture

PolicyGradEx follows a two-stage pipeline:

**Stage 1: Meta-Training + Gradient Extraction**
1. Perform multi-task training on all $n$ tasks to obtain a meta-policy $\pi_{\theta^*}$.
2. Collect trajectories for each task, compute, and store projected gradient features for every transition.

**Stage 2: Surrogate Model Estimation + Clustering**
1. Randomly sample $m$ task subsets $S_1, \ldots, S_m$.
2. For each subset, solve a weighted logistic regression using precomputed gradient features to estimate adaptation performance.
3. Construct an $n \times n$ task affinity matrix.
4. Partition the $n$ tasks into $k$ groups via convex relaxation optimization.

### Key Designs

#### 1. **First-Order Policy Gradient Approximation**

Starting from the PPO probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta^*}(a_t|s_t)}$$

A first-order Taylor expansion is applied to the log probability ratio:

$$\log r_t(\theta) = \log \pi_\theta(a_t|s_t) - \log \pi_{\theta^*}(a_t|s_t) = g_t^\top \Delta\theta + \epsilon$$

where $g_t = \nabla \log \pi_\theta(a_t|s_t)|_{\theta=\theta^*}$ is the gradient feature vector and $\epsilon$ is the approximation error.

**Key Empirical Validation**: On MT10, CartPole, Highway, and LunarLander, when the parameter distance between the adapted policy and the meta-policy is within 1%, the approximation error $\epsilon$ is **less than 2%**, providing empirical guarantees for the accuracy of the surrogate model.

| Parameter Distance | MT10 | CartPole | Highway | LunarLander |
|---|---|---|---|---|
| 0.1% | 0.01% | 0.12% | 0.02% | 0.06% |
| 0.5% | 0.43% | 0.73% | 0.11% | 0.03% |
| 1.0% | 0.32% | 0.98% | 2.04% | 0.48% |

#### 2. **Reducing Policy Optimization to Weighted Logistic Regression**

Using the first-order approximation, the PPO objective is simplified to a **weighted binary classification problem**:

For each transition $(s_t, a_t, \hat{A}_t)$, define:
- Label: $y_t = \text{sign}(\hat{A}_t) \in \{-1, +1\}$ (sign of the advantage function)
- Classifier score: $z_t = g_t^\top \Delta\theta$
- Sample weight: $w_t = |\hat{A}_t|$ (absolute value of the advantage function)

Surrogate loss:
$$\ell(g_t, y_t, w_t; \Delta\theta) = w_t \cdot \log(1 + (-y_t(g_t^\top \Delta\theta)))$$

Average loss over task subset $S$:
$$\hat{L}_S(\theta) = \frac{1}{|\mathcal{D}_S|} \sum_{(g,y,w) \in \mathcal{D}_S} \ell(g, y, w; \theta)$$

**Core Insight**: The policy optimization problem is reduced to a logistic regression solvable in milliseconds.

#### 3. **Dimensionality Reduction via Random Projection**

Policy network parameters may reach millions of dimensions, making the gradient vector $g_{i,t} \in \mathbb{R}^p$ prohibitively high-dimensional. Johnson–Lindenstrauss random projection is applied:

$$\tilde{g}_{i,t} = P^\top g_{i,t}, \quad P \in \mathbb{R}^{p \times d}, \quad d \ll p$$

Each entry of $P$ is sampled independently from $\mathcal{N}(0, d^{-1})$. Experiments show that $d=400$ yields satisfactory results, reducing dimensionality from millions to 400.

Logistic regression is then solved in the $d$-dimensional space and mapped back to the original space:
$$\hat{\theta}^{(j)} = \theta^* + P\hat{\theta}_d$$

#### 4. **Task Affinity Matrix Construction**

After sampling $m$ random subsets, the affinity score for each task pair $(T_i, T_j)$ is computed as:

$$U_{i,j} = \frac{1}{n_{i,j}} \sum_{1 \leq l \leq m: \{T_i, T_j\} \in S_l} \hat{f}(S_l)$$

This is the average estimated performance over all subsets containing both tasks. With $m = O(n^2)$, sufficient coverage of all task pairs is guaranteed with high probability.

#### 5. **Convex Relaxation Clustering**

The NP-hard combinatorial optimization problem is relaxed to a semidefinite program (SDP):

$$\max_{X \in \mathbb{R}^{n \times n}} \langle U, X \rangle - \lambda \cdot \text{Tr}[X]$$

Penalizing the matrix trace enables automatic determination of the number of groups $k$. The discrete grouping is recovered via rounding. Since the optimization operates on a small $n \times n$ matrix, solving takes only a few seconds.

### Loss & Training

**Meta-Training Stage**:
- Joint training over all tasks using Soft Modularization
- Policy network: 4-layer MLP
- 2048 steps sampled per task

**Surrogate Model Stage**:
- Logistic regression solved independently for each of the $m$ random subsets (400 dimensions, millisecond-scale)
- Random projection dimension $d=400$

**Downstream Training**:
- Meta-World: Independent policies trained within each group using Soft Modularization
- Control environments: MAML-based meta-learning; evaluated on 50 test tasks after 200 adaptation steps

## Key Experimental Results

### Main Results

**Multi-Objective RL & Meta-RL Performance Comparison:**

| Method | Meta-World (Success Rate) | CartPole | Highway | LunarLander |
|---|---|---|---|---|
| Multi-Task Training | 71.3% | 145.9 | 140.0 | 53.8 |
| Soft Modularization | 82.0% | 139.3 | 141.3 | 66.1 |
| PaCo | 73.1% | 144.5 | 136.6 | 62.6 |
| CARE | 84.0% | / | / | / |
| Random Grouping | 58.2% | 144.1 | 143.4 | 73.1 |
| Gradient Similarity Grouping | 69.6% | 142.0 | 135.6 | 80.8 |
| **PolicyGradEx** | **94.0%** | **159.2** | **153.5** | **82.8** |

**Key Gains**:
- vs. multi-task optimizers: average improvement of **16%**
- vs. random grouping: **62%** improvement on Meta-World
- vs. gradient similarity grouping: **35%** improvement on Meta-World

### Ablation Study

**Surrogate Model Accuracy (NMI Comparison):**

| MLP Layers | Meta-World NMI | LunarLander NMI | Speedup |
|---|---|---|---|
| 2 | 0.76 | 0.73 | 21× |
| 4 | 0.76 | 0.73 | 24× |
| 8 | 0.76 | 0.73 | **26×** |

- NMI > 0.73 (random clustering NMI ≈ 0.2), validating the accuracy of the surrogate model
- Speedup reaches up to **26×**

**Ablation on Number of Groups $k$:**

| $k$ | Meta-World Success Rate |
|---|---|
| 1 | ~71% |
| 2 | 89.5% |
| 3 | **94.0%** |
| 4 | 95.1% |

Performance nearly saturates at $k=3$, which is used in reported results.

**Ablation on Random Projection Dimension $d$:** As $d$ varies from 200 to 1000, marginal gains beyond 400 are negligible; $d=400$ is fixed throughout.

**Grouping Strategy Comparison:**

| Grouping Strategy | Meta-World Success Rate | Notes |
|---|---|---|
| Loss-based clustering (Ours) | **94.0%** | Based on actual performance estimated by surrogate model |
| Gradient similarity clustering | 69.6% | Captures only gradient direction information |
| Random grouping | 58.2% | Uninformed baseline |

Loss-based clustering improves over the other two strategies by **19%**.

### Key Findings

1. **First-order approximation is surprisingly accurate for policy networks**: Within 1% parameter distance, the error remains below 2%, supporting the theoretical foundation of the entire method.
2. **Surrogate model achieves 26× speedup**: Full training for each subset is avoided, making large-scale task grouping practically feasible.
3. **Loss-based clustering substantially outperforms gradient similarity clustering**: The former captures the actual effect of joint multi-task training, whereas the latter examines only pairwise gradient directions.
4. **Negative transfer is effectively mitigated through grouping**: In Meta-World, a single policy achieves only 71% success rate, while training three separate groups reaches 94%.
5. **Hessian trace correlates with generalization error**: The PAC-Bayes bound is non-vacuous, and the Hessian trace serves as a practical measure of generalization.

## Highlights & Insights

1. **Reduction from PPO to logistic regression**: Elegantly transforming a complex policy optimization problem into a simple weighted binary classification problem constitutes a notable methodological contribution.
2. **Attribution methods applied to RL**: The approach draws inspiration from data attribution (TRAK, Datamodels) to model task relationships in RL.
3. **Extremely high computational efficiency**: Only one meta-training run is required, followed by millisecond-scale surrogate model solving and second-scale convex optimization clustering.
4. **Non-vacuous generalization bound**: The Hessian-based PAC-Bayes bound in Theorem 1 is consistent in magnitude with empirical generalization error.
5. **General framework**: Applicable to both multi-task RL and meta-RL settings, with any multi-task optimizer as the downstream learner.

## Limitations & Future Work

1. **Boundary conditions for first-order approximation**: When parameter distance exceeds 5%, the error grows to 10%, limiting applicability in scenarios with large adaptation step sizes.
2. **Requires shared state/action spaces**: All tasks must share the same $\mathcal{S}$ and $\mathcal{A}$, precluding heterogeneous task settings.
3. **Sensitivity to meta-policy training quality**: If multi-task meta-training fails, subsequent first-order approximations and groupings become unreliable.
4. **Choice of $m$**: Covering all task pairs requires $m = O(n^2)$ subsets, which still incurs overhead when $n$ is large.
5. **Validation limited to discrete control and simple robotic tasks**: The method has not been evaluated in more complex environments (e.g., Atari, complex robotic manipulation).
6. **Static grouping assumption**: Task relationships may evolve over training, and fixed groupings may be suboptimal.

## Related Work & Insights

- **MAML (Finn et al., 2017)**: A seminal meta-learning method used for downstream meta-RL evaluation in this work.
- **PCGrad (Yu et al., 2020)**: A gradient projection-based multi-task optimization method.
- **TRAK (Park et al., 2023)**: Random projected gradients for data attribution; the direct methodological inspiration for this work.
- **Meta-World (Yu et al., 2020)**: A standardized multi-task robotic benchmark.
- **Datamodels (Ilyas et al., 2022)**: Linear surrogate models can accurately approximate deep network training outcomes on subsets.
- Insight: **Transferring the methodology of data attribution / influence functions to RL is a promising research direction.**

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Reducing policy optimization to logistic regression is highly creative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 environments + comprehensive ablations + generalization analysis + speed comparison)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic with a natural progression from motivation to method to experiments)
- Value: ⭐⭐⭐⭐⭐ (Efficient task grouping has broad applicability in multi-objective RL)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/robotics/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](../../ICLR2026/robotics/mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[ICML 2026\] Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](../../ICML2026/robotics/optimal_and_scalable_mapf_via_multi-marginal_optimal_transport_and_schrödinger_b.md)
- [\[AAAI 2026\] Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](towards_reinforcement_learning_from_neural_feedback_mapping_.md)

</div>

<!-- RELATED:END -->
