---
title: >-
  [Paper Note] Transitive RL: Value Learning via Divide and Conquer
description: >-
  [ICLR 2026][Reinforcement Learning][Divide and Conquer] This paper proposes Transitive Reinforcement Learning (TRL), a novel value function learning algorithm based on the divide-and-conquer paradigm. By exploiting the triangle inequality structure inherent in goal-conditioned RL, TRL recursively decomposes value function updates into subproblems, achieving superior performance over TD learning and Monte Carlo methods on long-horizon tasks.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Divide and Conquer
  - Value Function Learning
  - Offline RL
  - Goal-Conditioned Reinforcement Learning
  - Triangle Inequality
date: 2026-05-08
content_hash: aad3f3bae2e02eee
---

# Transitive RL: Value Learning via Divide and Conquer

**Conference**: ICLR 2026
**arXiv**: [2510.22512](https://arxiv.org/abs/2510.22512)
**Code**: None
**Area**: Reinforcement Learning / Goal-Conditioned RL
**Keywords**: Divide and Conquer, Value Function Learning, Offline RL, Goal-Conditioned Reinforcement Learning, Triangle Inequality

## TL;DR

This paper proposes Transitive Reinforcement Learning (TRL), a novel value function learning algorithm based on the divide-and-conquer paradigm. By exploiting the triangle inequality structure inherent in goal-conditioned RL, TRL recursively decomposes value function updates into subproblems, achieving superior performance over TD learning and Monte Carlo methods on long-horizon tasks.

## Background & Motivation

Goal-Conditioned Reinforcement Learning (GCRL) addresses a fundamental question: **learning a policy that can reach any goal state from any starting state in the minimum number of steps**. This problem has broad applications in robotic navigation, manipulation planning, and related domains.

Learning value functions in GCRL (which characterize the minimum number of steps required to reach a goal from a given state) relies on two classical paradigms:

**Temporal Difference (TD) Learning**: Propagates value information through incremental single-step bootstrapping. It benefits from low variance (dynamic programming), but suffers from **bias accumulation**—for a trajectory of length $T$, $O(T)$ recursive updates are required to propagate complete information, and each update may introduce approximation errors that accumulate progressively.

**Monte Carlo (MC) Methods**: Directly estimate the value function using returns from complete trajectories. These methods are unbiased but suffer from **high variance**—the variance of total returns grows with trajectory length.

Both approaches have fundamental weaknesses, particularly in **long-horizon** tasks:

- TD methods become increasingly inaccurate due to compounding bias
- MC methods converge extremely slowly due to high variance

The core insight of this paper is that GCRL problems exhibit a **triangle inequality structure**—the shortest distance from A to C does not exceed the sum of distances from A to B and from B to C. This structure can be exploited to design a value function learning algorithm superior to both TD and MC methods.

## Method

### Overall Architecture

The core idea of TRL is **divide and conquer**: decompose the problem of estimating the value function from state $s$ to goal state $g$ into two subproblems—estimating the value from $s$ to some intermediate state $m$, and from $m$ to $g$—then combining these two estimates.

The recursion depth of this divide-and-conquer approach is $O(\log T)$ (by repeatedly bisecting a trajectory of length $T$), far smaller than the $O(T)$ recursion depth of TD learning. Meanwhile, each layer of estimation leverages dynamic programming, avoiding the high variance of MC methods.

### Key Designs

#### 1. **Triangle Inequality Structure**

In GCRL, the optimal value function $V^*(s, g)$ satisfies the triangle inequality:

$$V^*(s, g) \leq V^*(s, m) + V^*(m, g), \quad \forall m$$

This implies that any path through an intermediate point $m$ is an upper bound. TRL exploits this property by approximating the true value via minimization over all intermediate points:

$$V^*(s, g) = \min_m [V^*(s, m) + V^*(m, g)]$$

Equality holds when $m$ lies on the optimal path.

**Design Motivation**: This update rule is analogous to the Floyd-Warshall algorithm in graph theory, elegantly transferred to RL problems in continuous state spaces.

#### 2. **Divide-and-Conquer Value Function Update Rule**

The TRL value function update can be described recursively:

- **Base case**: For adjacent state pairs $(s_t, s_{t+1})$, the value is 1 step (obtained directly from data).
- **Recursive step**: For a distant state pair $(s_i, s_j)$, select the midpoint $m = s_{(i+j)/2}$ and compute:

$$V(s_i, s_j) = V(s_i, m) + V(m, s_j)$$

Through this binary recursion, information that previously required $O(T)$ steps to propagate now requires only $O(\log T)$ recursive layers.

#### 3. **Practical Implementation**

- **Intermediate point sampling**: In addition to trajectory midpoints, multiple candidate intermediate points are sampled from the dataset, and the minimum is taken.
- **Function approximation**: The value function $V_\theta(s, g)$ is parameterized by a neural network.
- **Offline setting**: All updates are based on a pre-collected offline dataset, requiring no online environment interaction.

### Loss & Training

The training loss of TRL is:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, m, g) \sim \mathcal{D}} \left[ \left( V_\theta(s, g) - \min\{V_\theta(s, g), V_{\bar{\theta}}(s, m) + V_{\bar{\theta}}(m, g)\} \right)^2 \right]$$

where $\bar{\theta}$ denotes target network parameters (for training stability) and $m$ is an intermediate point sampled from the dataset.

The intuition behind this loss is: if the path through intermediate point $m$ is shorter ($V(s,m) + V(m,g) < V(s,g)$), update $V(s,g)$ to be smaller; otherwise, leave it unchanged. This guarantees that the value function progressively tightens toward the optimal value.

## Key Experimental Results

### Main Results

Comparison with existing algorithms on challenging long-horizon offline GCRL benchmarks:

| Benchmark Task | Metric | TRL | Best TD-based | Best MC-based | Other GCRL |
|---|---|---|---|---|---|
| Long-range navigation | Success rate | **Best** | Degrades from bias accumulation | Poor due to high variance | Moderate |
| Complex manipulation | Success rate | **Best** | Moderate | Poor | Moderate |
| Ultra-long-horizon planning | Success rate | **Best** | Severe degradation | Nearly unusable | Poor |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| TRL (full) | Best | Baseline configuration |
| Remove divide-and-conquer (TD substitute) | Significant drop | Validates the central role of divide-and-conquer |
| Remove dynamic programming (MC substitute) | Clear drop | Validates the variance-reduction effect of DP |
| Varying recursion depth | $\log T$ is optimal | Both deeper and shallower are suboptimal |
| Number of intermediate point samples | Moderate suffices | More intermediate points provide marginal benefit |

### Key Findings

1. **The $O(\log T)$ divide-and-conquer recursion depth provides a substantial advantage**: This is the core theoretical advantage of TRL. For trajectories of several hundred steps, TD requires hundreds of recursive propagations (each introducing approximation error), whereas TRL requires only 7–8 recursive layers.

2. **The advantage is especially pronounced on long-horizon tasks**: As trajectory length increases, the performance of TD and MC methods degrades rapidly, while TRL degrades slowly, with the performance gap widening accordingly.

3. **TRL achieves the best of both TD and MC**: TRL inherits the low-variance property of TD (via dynamic programming) while avoiding the bias accumulation of TD (via logarithmic-depth recursion), achieving both advantages simultaneously.

4. **Robustness to offline data quality**: Even when the offline dataset consists of suboptimal trajectories, TRL can still identify shorter paths through the triangle inequality minimization operation.

## Highlights & Insights

1. **Elegant exploitation of problem structure**: The triangle inequality is an intrinsic geometric property of GCRL problems. Converting it into an algorithmic design principle (the divide-and-conquer update rule) represents a highly elegant contribution. The connection to Floyd-Warshall provides clear intuition.

2. **Clean theoretical guarantees**: The $O(\log T)$ vs. $O(T)$ difference in recursion depth directly translates into a difference in the order of bias accumulation, yielding a clear and compelling theoretical analysis.

3. **Conceptually simple yet empirically powerful**: The core idea can be stated in one sentence—"split a long path into two segments via an intermediate point"—yet this simple idea yields substantial empirical gains.

4. **Generality**: The divide-and-conquer principle is not limited to specific environments or tasks; it applies to any value function learning problem that satisfies the triangle inequality.

## Limitations & Future Work

1. **Restricted to GCRL**: TRL exploits the triangle inequality structure of shortest-path problems and cannot be directly extended to general RL problems (e.g., cumulative reward maximization with discounting).

2. **Intermediate point selection**: Efficiently identifying good intermediate points in continuous spaces is a practical challenge. If the dataset lacks suitable "bridge" states, the effectiveness of the divide-and-conquer decomposition is diminished.

3. **Offline setting constraints**: Validation is currently limited to the offline setting. An online variant incorporating exploration may require additional design considerations.

4. **Function approximation errors**: Although the recursion depth is reduced, approximation errors at each layer may still accumulate. Whether the cumulative error across $O(\log T)$ layers remains consistently manageable in practice warrants further investigation.

5. **Relationship to hierarchical RL**: The divide-and-conquer strategy has a natural connection to Hierarchical Reinforcement Learning, but this connection is not deeply explored in the paper. Integrating subgoal discovery methods could further improve performance.

6. **Coverage requirements for offline data**: TRL requires sufficient "intermediate states" in the dataset to construct divide-and-conquer paths. This requirement may become a bottleneck for sparse datasets.

## Related Work & Insights

- **TD learning and MC methods**: TRL can be viewed as a unification and advancement of these two classical approaches—finding a superior equilibrium point on the bias-variance spectrum.
- **Floyd-Warshall algorithm**: The value update rule in TRL directly corresponds to the dynamic programming algorithm for shortest paths in graphs, exemplifying a natural transfer of discrete algorithms to continuous RL problems.
- **Offline GCRL** (e.g., WGCSL, Quasimetric RL): TRL introduces a fundamentally different value learning paradigm, complementary to existing methods.
- **Insights**: This work suggests that many classical algorithmic design principles (such as divide and conquer) may have profound applications in RL; the key lies in identifying the corresponding mathematical structure in the problem (e.g., the triangle inequality).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (An innovative and conceptually elegant combination of divide-and-conquer and RL)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Superior performance demonstrated on long-horizon benchmarks)
- Writing Quality: ⭐⭐⭐⭐ (Algorithm motivation and theoretical analysis are clear and accessible)
- Value: ⭐⭐⭐⭐ (Opens a new paradigm for value learning in GCRL)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models](divide_harmonize_then_conquer_it_shooting_multi-commodity_flow_problems_with_mul.md)
- [\[ICLR 2026\] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)
- [\[ICLR 2026\] Value Flows](value_flows.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)

<!-- RELATED:END -->
