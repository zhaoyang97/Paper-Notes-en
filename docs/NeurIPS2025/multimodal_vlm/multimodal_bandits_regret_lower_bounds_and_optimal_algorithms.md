---
title: >-
  [Paper Note] Multimodal Bandits: Regret Lower Bounds and Optimal Algorithms
description: >-
  [NeurIPS 2025][Multimodal VLM][multimodal bandits] For the multimodal multi-armed bandit problem where the reward function has at most $m$ modes, this paper proposes the first computationally feasible algorithm for solving the Graves-Lai optimization problem, achieves an asymptotically optimal regret bound, and proves that local search strategies are suboptimal.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - multimodal bandits
  - Graves-Lai bound
  - dynamic programming
  - asymptotic optimality
  - multi-armed bandit
date: 2026-05-08
content_hash: cd968130146a8d2e
---

# Multimodal Bandits: Regret Lower Bounds and Optimal Algorithms

**Conference**: NeurIPS 2025
**arXiv**: [2510.25811](https://arxiv.org/abs/2510.25811)
**Code**: [GitHub](https://github.com/wilrev/MultimodalBandits)
**Area**: Multimodal VLM
**Keywords**: multimodal bandits, Graves-Lai bound, dynamic programming, asymptotic optimality, multi-armed bandit

## TL;DR

For the multimodal multi-armed bandit problem where the reward function has at most $m$ modes, this paper proposes the first computationally feasible algorithm for solving the Graves-Lai optimization problem, achieves an asymptotically optimal regret bound, and proves that local search strategies are suboptimal.

## Background & Motivation

### 1. State of the Field

The stochastic multi-armed bandit (MAB) problem is a foundational model for online decision-making. In the classical setting, the learner imposes no structural assumptions on the reward distributions. Incorporating structural constraints — such as unimodality of the reward function — can significantly reduce the cost of exploration. For unimodal bandits ($m=1$), several asymptotically optimal algorithms already exist (KL-UCB, Thompson Sampling, DMED, etc.).

### 2. Limitations of Prior Work

**Multimodal settings** ($m \geq 2$, i.e., multimodal bandits) are far more complex than the unimodal case:
- The confusion parameter set $\mathcal{B}(m, \boldsymbol{\mu})$ is non-convex and disconnected, precluding simple convex optimization approaches.
- Decomposing $\mathcal{B}$ into convex subsets requires an exponentially growing number of components (e.g., $K=100$, $m=5$ requires $>10^5$ convex parts).
- Prior work (Saber & Maillard 2024) employs a local search strategy without proving its optimality.

### 3. Root Cause

The Graves-Lai optimization problem provides an information-theoretic regret lower bound and an optimal exploration strategy, but in the multimodal setting the constraint set is structurally so complex that efficient solution is infeasible. Without a tractable solver, asymptotically optimal bandit strategies cannot be realized.

### 4. Paper Goals

To design the first **computationally feasible** algorithm for solving the Graves-Lai optimization problem for multimodal bandits, thereby enabling asymptotically optimal online decision-making.

### 5. Starting Point

The paper progressively resolves the computational challenges posed by non-convex constraints through a series of sub-problem decompositions, structural properties of mode locations (Proposition 6), discretization approximations (Proposition 7), and dynamic programming (Proposition 8), ultimately solving the original problem via penalized subgradient descent.

### 6. Core Idea

By exploiting the key structural property that *the modes of the confusion parameter must lie within the original mode set $\mathcal{M}(\boldsymbol{\mu}) \cup \{k\}$*, the exponentially complex search is reduced to a polynomial-time dynamic program on a tree.

## Method

### Overall Architecture

The algorithmic pipeline (as illustrated in Figure 1):
1. **Constraint decomposition**: Decompose the global constraint into $(k, k')$ sub-problems.
2. **Discretization**: Discretize the continuous search space into a grid of resolution $n$.
3. **Dynamic programming**: Solve each discretized sub-problem via DP on a tree.
4. **Penalized subgradient descent**: Iteratively solve the original $P_{GL}$.

### Key Designs

#### Module 1: Constraint Set Decomposition and Simplification

**Function**: Decompose the complex non-convex constraints into tractable sub-problems.

**Mechanism**:

**Proposition 3**: When $k \in \mathcal{N}(\boldsymbol{\mu})$ (within the neighborhood of a mode) or $|\mathcal{M}(\boldsymbol{\mu})| < m$, the constraint admits a closed-form solution:
$$\inf_{\boldsymbol{\lambda} \in \mathcal{B}_k(m, \boldsymbol{\mu})} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}) = \eta_k d_k(\mu_k, \mu^\star)$$

**Proposition 4**: In other cases, the search can be restricted to the compact set $[\mu_\star, \mu^\star]^K$.

**Proposition 5**: $\boldsymbol{\eta}^\star \in [0, \mathfrak{B}(\boldsymbol{\mu})]^K$, bounding the search space.

**Design Motivation**: Progressively constraining the dimensionality and range of the search space renders the problem computationally tractable.

#### Module 2: Structural Theorem on Mode Locations

**Function**: Characterize the mode locations of the optimal confusion parameter.

**Proposition 6 (Key Structural Result)**:
$$\mathcal{M}(\boldsymbol{\lambda}^\star) \subset \mathcal{M}(\boldsymbol{\mu}) \cup \{k\}$$

That is, the modes of the optimal confusion parameter can **only appear at the mode locations of the original reward function or at the target arm $k$**.

**Significance**: When $|\mathcal{M}(\boldsymbol{\mu})| = m$, exactly one original mode $k' \neq k^\star(\boldsymbol{\mu})$ must cease to be a mode, allowing $P_{GL}(k)$ to be further partitioned into sub-problems $P_{GL}(k, k')$.

**Design Motivation**: Knowledge of mode locations drastically reduces the search space.

#### Module 3: Discretization and Dynamic Programming

**Function**: Solve sub-problems exactly over a discretized space using DP.

**Proposition 7 (Discretization Accuracy Guarantee)**:
$$\boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \tilde{\boldsymbol{\lambda}}) - \frac{\mathfrak{C}(\boldsymbol{\mu})}{n} \leq \min_{\boldsymbol{\lambda} \in \mathcal{B}_{k,k'}} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}) \leq \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \tilde{\boldsymbol{\lambda}})$$

The approximation error decays as $1/n$.

**Proposition 8 (DP Recurrence)**: Rooting the tree $G$ at $k$ to form a directed tree, define:
$$f_\ell(z, u) = \text{min obtainable value of } \sum_{j \in \mathcal{D}(\ell) \cup \{\ell\}} \eta_j d_j(\mu_j, \lambda_j)$$

The recurrence handles mode nodes and non-mode nodes separately, with a single DP pass of complexity $O(nK)$.

**Design Motivation**: The tree structure makes DP feasible; discretization makes the search finite.

#### Module 4: Penalized Subgradient Descent

**Function**: Solve the original Graves-Lai optimization problem.

**Mechanism**: Convert hard constraints into a penalty:
$$h(\boldsymbol{\eta}) = \boldsymbol{\eta}^\top \boldsymbol{\Delta} + \gamma \max\left[1 - \min_{\boldsymbol{\lambda}} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}), 0\right]$$

Projected subgradient descent:
$$\boldsymbol{\eta}(s+1) = \Pi[\boldsymbol{\eta}(s) - \delta(\boldsymbol{\Delta} - \gamma d(\boldsymbol{\mu}, \boldsymbol{\lambda}(s)) \mathbb{1}\{\cdot < 1\})]$$

**Proposition 9**: After $t$ iterations the error is $O(1/\sqrt{t})$; the discretization error is $O(1/n)$.

### Loss & Training (Total Complexity)

**Theorem 2**: With $t$ iterations and $n$ discretization points:
- Time complexity: $O(K^2 m n t)$ (improvable to $O(Knt)$)
- Space complexity: $O(Knt)$
- Approximation error: $O(1/n + 1/\sqrt{t})$

Given a computational budget $nt = a$, setting $n = a^{1/3}$ and $t = a^{2/3}$ yields error $O(a^{-1/3})$.

## Key Experimental Results

### Main Results: Multimodal OSSB vs. Classical OSSB

Setting: Binary tree $G$ ($K=7$ arms), Gaussian rewards $\mathcal{N}(\mu_k, 1)$, bimodal reward function ($m=2$), modes at nodes 4 and 6.

| Setting | Instance | Multimodal OSSB | Classical OSSB | Gain |
|---------|----------|-----------------|----------------|------|
| Peaked ($\sigma=0.5$) | Easy | Substantially lower regret | High regret | **Large** |
| Flat ($\sigma=4$) | Hard | Substantially lower regret | High regret | **Large** |

In both settings, Multimodal OSSB substantially outperforms classical OSSB (which ignores structure), with the advantage continuing to grow within $T=10{,}000$ rounds.

### Suboptimality of Local Search

**Theorem 3**: The ratio of the regret of the local search strategy to that of the global strategy is **unbounded**:
$$\sup_{\boldsymbol{\mu} \in \mathcal{F}_m} \frac{C_{loc}(m, \boldsymbol{\mu})}{C(m, \boldsymbol{\mu})} = +\infty$$

Concrete example (Figure 2): $\boldsymbol{\mu} = (1, 2, 4, 2, 3)$
- Local search: $\inf_{\boldsymbol{\lambda}} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}) = 0.5$
- Global optimum: $\boldsymbol{\lambda}^\star = (4, 2, 4, 2.8, 2.8)$, value $\approx 0.145 < 0.5$

The local search incurs **3.4×** more regret.

### Key Findings

1. **Counterintuitive behavior of local search**: Local search is optimal for unimodal bandits ($m=1$) but can be arbitrarily poor in the multimodal case, because verifying the modal property near a flat mode requires a large number of samples.
2. **Safe regime for $\kappa$-peaked functions**: When the reward function is $\kappa$-peaked, local search is at most $\kappa$ times worse than global search (Proposition 11).
3. **Correcting a prior error**: The paper proves that the asymptotic optimality claim for the IMED-MB algorithm of Saber & Maillard (2024) does not hold.
4. **Value of exploiting structure**: Even for small instances ($K=7$), exploiting multimodal structure yields substantial regret reduction.

## Highlights & Insights

1. **First computationally feasible algorithm**: Resolves the long-standing computational bottleneck in multimodal bandits — efficient solution of the Graves-Lai problem.
2. **Elegant decomposition strategy**: From Proposition 3 to Proposition 9, each step precisely constrains the search space, ultimately reducing an exponential problem to polynomial complexity.
3. **Proposition 6 as the core breakthrough**: Characterizing mode locations of the confusion parameter makes DP tractable.
4. **Rigorous proof of local search suboptimality**: Corrects undue optimism in the literature regarding local search strategies, demonstrating that solving $P_{GL}$ globally is unavoidable.
5. **The $\kappa$-peaked concept**: Provides a quantitative criterion for determining when a local strategy is "good enough."
6. **Tight integration of theory and algorithms**: The paper not only establishes bounds but also provides implementable algorithms validated through experiments.

## Limitations & Future Work

1. Only **tree-structured** graphs are considered; multimodal bandits on general graphs remain open.
2. The $O(K^2 mnt)$ complexity may still be prohibitive for large-scale problems with many arms.
3. Experiments are small-scale ($K=7$, $T=10{,}000$); large-scale validation is lacking.
4. Solving $P_{GL}$ at every round (or at logarithmically spaced intervals) poses real-time challenges.
5. The optimal balance between discretization resolution $n$ and iteration count $t$ requires prior knowledge of problem-dependent constants such as $\mathfrak{B}(\boldsymbol{\mu})$.
6. Non-stationary or adversarial reward settings are not considered.

## Related Work & Insights

- **Relation to unimodal bandits**: The setting degenerates to classical unimodal bandits when $m=1$, for which Proposition 3 provides a closed-form solution — the present work is its rigorous generalization.
- **Relation to OSSB**: OSSB is a general asymptotically optimal framework; this paper supplies the missing piece for multimodal bandits by solving $P_{GL}$ efficiently.
- **Analogy with combinatorial bandits**: Combinatorial bandits use iterative algorithms to solve $P_{GL}$; multimodal bandits employ DP combined with subgradient descent.
- **Implications for optimization practice**: The idea of exploiting multimodal structure may inform optimization of multimodal objective functions, such as the loss surfaces of deep networks.
- **Pricing applications**: Multimodal bandits naturally model pricing problems under multimodal demand curves.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contributions are outstanding: the paper presents the first computationally feasible algorithm for multimodal bandits and a rigorous proof of local search suboptimality. The decomposition strategy proceeds step by step with considerable mathematical depth. Points are deducted for the limited experimental scale and the current restriction to tree-structured graphs. Note: this paper has little connection to vision-language multimodal models; "multimodal" here refers to the *multi-peaked* (multimodal) nature of the reward function.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks](../../AAAI2026/multimodal_vlm/information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)
- [\[NeurIPS 2025\] Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)

</div>

<!-- RELATED:END -->
