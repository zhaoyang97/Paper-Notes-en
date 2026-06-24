---
title: >-
  [Paper Note] Adversarial Combinatorial Semi-bandits with Graph Feedback
description: >-
  [ICML 2025][Combinatorial Semi-bandits] This paper introduces graph feedback into the adversarial combinatorial semi-bandits framework and proposes the OSMD-G algorithm, establishing the optimal regret bound of $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha S T})$, where $S$ is the size of the combinatorial action and $\alpha$ is the independence number of the feedback graph. The key technique lies in utilizing randomized swap rounding to achieve negatively correlated sampling.
tags:
  - "ICML 2025"
  - "Combinatorial Semi-bandits"
  - "Graph Feedback"
  - "Adversarial Online Learning"
  - "Regret Bound"
  - "Negatively Correlated Sampling"
date: 2026-05-08
content_hash: ba364e8faf9507e8
---

# Adversarial Combinatorial Semi-bandits with Graph Feedback

**Conference**: ICML 2025  
**arXiv**: [2502.18826](https://arxiv.org/abs/2502.18826)  
**Code**: None  
**Area**: Others (Online Learning / Bandit Theory)  
**Keywords**: Combinatorial Semi-bandits, Graph Feedback, Adversarial Online Learning, Regret Bound, Negatively Correlated Sampling

## TL;DR

This paper introduces graph feedback into the adversarial combinatorial semi-bandits framework and proposes the OSMD-G algorithm, establishing the optimal regret bound of $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha S T})$, where $S$ is the size of the combinatorial action and $\alpha$ is the independence number of the feedback graph. The key technique lies in utilizing randomized swap rounding to achieve negatively correlated sampling.

## Background & Motivation

### Origin of the Problem

Combinatorial semi-bandits generalize the classic multi-armed bandits and are widely applied in scenarios such as multi-platform online advertising, recommendation systems, web page optimization, and online shortest path routing. In each round, the learner selects $S$ out of $K$ arms and observes the rewards of the selected arms as feedback.

### Limitations of Existing Feedback Structures

Existing work only distinguishes between two extreme feedback scenarios:
- **Full information**: The rewards of all $K$ arms are observed in each round, with an optimal regret of $\widetilde{\Theta}(S\sqrt{T})$
- **Semi-bandit**: Only the rewards of the selected $S$ arms are observed in each round, with an optimal regret of $\widetilde{\Theta}(\sqrt{KST})$

However, rich intermediate information structures exist in practical problems. For example:

**Online Ad Auctions**: The platform announces the winning bid, and the learner can deduce the counterfactual payoffs of all bids higher than their own.

**Online Recommendation**: Information correlation exists between semantically similar items (e.g., two brands of tissues)—a user clicking on one is likely to click on another.

This additional information is completely ignored in classical semi-bandit feedback, creating an urgent need for a unified framework to characterize it.

### Goal

Use a directed feedback graph $G = ([K], E)$ to uniformly describe the feedback structure: after selecting arm $a$, the rewards of all outgoing neighbors $N_{\text{out}}(a)$ can be observed. A complete graph corresponds to full information, while a graph with only self-loops corresponds to the semi-bandit setting. This paper aims to:
1. Establish the minimax optimal regret under general feedback graphs.
2. Design near-optimal algorithms.

## Method

### Overall Architecture

This paper proposes **OSMD-G** (Online Stochastic Mirror Descent with Graphs), which is an extension of the classical OSMD algorithm to the graph feedback setting. The algorithm procedure is as follows:

**Input**: Time horizon $T$, action set $\mathcal{A}$, arm set $[K]$, combinatorial budget $S$, feedback graph $G$, truncation parameter $\epsilon$, learning rate $\eta$, mirror map $F$

1. **Initialization**: $x^1 = \arg\min_{x \in \text{Conv}_\epsilon(\mathcal{A})} F(x)$
2. **For each round $t = 1, \ldots, T$**:
    - Sample action $v^t \in \mathcal{A}$ from $x^t$ via randomized swap rounding (Algorithm 2), satisfying $\mathbb{E}[v^t] = x^t$ and negative correlation.
    - Execute $v^t$, and observe the graph feedback $\{r_i^t : i \in N_{\text{out}}(v^t)\}$.
    - Construct the unbiased reward estimator $\tilde{r}_a^t$.
    - Mirror descent update: $w^{t+1} = \nabla F^*(\nabla F(x^t) + \eta \tilde{r}^t)$
    - Project back to the constraint set: $x^{t+1} = \arg\min_{x \in \text{Conv}_\epsilon(\mathcal{A})} D_F(x, w^{t+1})$

### Key Designs

#### 1. Reward Estimator

For each arm $a \in [K]$, an unbiased estimator is constructed based on graph feedback:

$$\tilde{r}_a^t = \frac{\sum_{i \in [K]: i \to a} \mathbf{1}[v_i^t = 1]}{\sum_{i \in [K]: i \to a} x_i^t} \cdot r_a^t$$

where the numerator is the sum of indicator variables that arm $a$ is actually observed, and the denominator is its expected value. Since $\mathbb{E}[v_i^t] = x_i^t$, this estimator is unbiased: $\mathbb{E}[\tilde{r}_a^t] = r_a^t$.

When $G$ contains only self-loops, this estimator degenerates to the classical semi-bandit estimator $\tilde{r}_a^t = v_a^t r_a^t / x_a^t$.

#### 2. Negatively Correlated Sampling (Core Technical Contribution)

This is the most critical innovation of this paper. OSMD-G requires the sampling distribution $p^t$ to simultaneously satisfy two conditions:

**Condition 1 (Mean)**: $\forall i \in [K], \; \mathbb{E}_{v \sim p}[v_i] = x_i$

**Condition 2 (Negative Correlation)**: $\forall i \neq j, \; \mathbb{E}_{v \sim p}[v_i v_j] \leq x_i x_j$

That is, any two arms are negatively correlated. By leveraging the matroid theory results of Chekuri et al. (2009) (Lemma 1), the authors prove that for any $x \in \text{Conv}(\mathcal{A})$ on the full action set $\mathcal{A}$, there always exists a distribution $p$ satisfying the above conditions, which can be efficiently sampled via **randomized swap rounding**.

**Why is negative correlation critical?** In the key steps of regret analysis, the variance of the estimator must be controlled:

$$\mathbb{E}\left[\left(\sum_{i: i \to a} v_i^t\right)^2\right]$$

- **Without the negative correlation constraint**: It can only be naively upper-bounded by $S \cdot \mathbb{E}[\sum_{i:i \to a} v_i^t]$, introducing a $\sqrt{S\alpha}$ factor into the regret.
- **With the negative correlation constraint**: The second moment can be decomposed into the sum of the squared mean and individual variances, eliminating $O(K^2)$ covariance terms and reducing the factor from $S\alpha$ to $S + \alpha$.

Specifically:

$$\mathbb{E}\left[\left(\sum_{i: i \to a} v_i^t\right)^2\right] = \left(\sum_{i: i \to a} x_i^t\right)^2 + \text{Var}\left(\sum_{i: i \to a} v_i^t\right) \leq \left(\sum_{i: i \to a} x_i^t\right)^2 + \sum_{i: i \to a} x_i^t$$

This step crucially exploits $\text{Cov}(v_i^t, v_j^t) \leq 0$, allowing the variance to be decomposed as the sum of variances of individual arms.

#### 3. Randomized Swap Rounding Algorithm

Given target $x = \sum_{i=1}^N w_i v_i$ (as a convex combination of actions in $\mathcal{A}$), the algorithm generates a random decision satisfying negative correlation through the following process:

1. Initialize $u \leftarrow v_1$
2. For $i = 1, \ldots, N-1$: let $c \leftarrow v_{i+1}$, and compute the cumulative weight $\beta_i = \sum_{j=1}^i w_j$
3. While $u \neq c$: find $a \in u \setminus c$ and $a' \in c \setminus u$ such that the swap remains feasible
4. With probability $\beta_i / (\beta_i + w_{i+1})$, set $u \leftarrow u - \{a\} + \{a'\}$, otherwise set $c \leftarrow c - \{a'\} + \{a\}$

The key to this algorithm is to exploit the **exchange property** of the action set: for any $u, c \in \mathcal{A}$, there exists a pair of arms that can be swapped such that the result remains feasible. The full action set $\mathcal{A}$ and matroid structures both satisfy this property.

#### 4. Choice of Mirror Map and Learning Rate Tuning

The negative entropy mirror map $F(x) = \sum_{i=1}^K (x_i \log x_i - x_i)$ is adopted, in which case:
- The dual space is $D^* = \mathbb{R}^K$
- The update simplifies to: $w^{t+1} = x^t \circ \exp(\eta \tilde{r}^t)$ (coordinate-wise multiplicative update)
- The Bregman divergence is the KL divergence

Optimal parameter selection:
- Learning rate: $\eta = \sqrt{\frac{2S \log(K/S)}{(S + 4\alpha \log(4K^2 T/\alpha))T}}$
- Truncation rate: $\epsilon = \frac{1}{KT}$

### Loss & Training

This is a theory-driven work and does not involve loss functions or training in the traditional sense. The core objective is to minimize regret:

$$\mathbb{E}[\mathsf{R}(\pi)] = \mathbb{E}\left[\max_{v_* \in \mathcal{A}} \sum_{t=1}^T \langle v_* - v^t, r^t \rangle\right]$$

**Regret Upper Bound Analysis** (Theorem 3) decomposes the regret into three terms:
1. **Truncation error**: $\epsilon K T$ (introduced by the truncation of the convexified constraint set)
2. **Initial Bregman divergence**: $S \log(K/S) / \eta$ (measuring the "distance" from the starting point to the optimal solution)
3. **Variance of the gradient estimator**: $\frac{\eta}{2}(S + 4\alpha \log(4K/(\epsilon\alpha))) T$

Substituting the optimal parameters yields the final upper bound:

$$\mathbb{E}[\mathsf{R}] \leq S\sqrt{T \log(K/S)} + 2\sqrt{\alpha S T \log(K/S) \log(4K^2 T/\alpha)}$$

## Key Experimental Results

This paper is a purely theoretical work with no numerical experiments; the core results are presented in the form of theorems.

### Main Results

The "main experiment" of this paper is the matching of minimax regret upper and lower bounds. The following table summarizes the optimal regret under different feedback settings (omitting polylog factors):

| Feedback Structure | $\alpha$ Value | Optimal Regret | Algorithm | Source |
|:---|:---|:---|:---|:---|
| Full Information ($G$ is a complete graph) | $\alpha = 1$ | $\widetilde{\Theta}(S\sqrt{T})$ | OSMD | Koolen et al. (2010) |
| General Graph Feedback | General $\alpha$ | $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha S T})$ | **OSMD-G** | **Ours** |
| Semi-bandit ($G$ contains only self-loops) | $\alpha = K$ | $\widetilde{\Theta}(\sqrt{KST})$ | OSMD | Audibert et al. (2014) |
| Full-bandit | — | $\widetilde{\Theta}(\sqrt{KS^3T})$ | — | Cohen et al. (2017) |

### Ablation Study

Comparison of the effects of different design choices through theoretical analysis:

| Configuration | Regret Upper Bound | Description |
|:---|:---|:---|
| OSMD-G + Negatively Correlated Sampling (Full action set $\mathcal{A}$) | $\widetilde{O}(S\sqrt{T} + \sqrt{\alpha S T})$ | Optimal, main result of this paper |
| OSMD-G + Mean constraint only (no negative correlation) | $\widetilde{O}(\sqrt{S\alpha T})$ | Suboptimal, Theorem 4 |
| General action subset $\mathcal{A}_0$ + negative correlation infeasible | $\widetilde{\Theta}(\sqrt{S\alpha T})$ | Unavoidable degradation |
| Weakly observable graphs + Explore-Then-Commit (ETC) | $\widetilde{O}(S T^{2/3} + \delta^{1/3} S^{2/3} T^{2/3})$ | Theorem 5 |
| Time-varying graphs $\{G_t\}$ | $\widetilde{O}(S\sqrt{T} + \sqrt{S \sum_{t=1}^T \alpha_t})$ | Natural extension |

### Key Findings

1. **Smooth Regret Interpolation**: As $\alpha$ varies from 1 to $K$, the regret smoothly interpolates from $S\sqrt{T}$ (full information) to $\sqrt{KST}$ (semi-bandit), revealing the exact behavior in the intermediate regime for the first time.
2. **Negative Correlation is Necessary**: The OSMD scheme satisfying only the mean constraint ($\mathbb{E}[v^t] = x^t$) is proved to be suboptimal ($\sqrt{S\alpha T}$ vs $S\sqrt{T} + \sqrt{\alpha ST}$).
3. **Structure of Action Set is Fundamental**: Moving from the full action set $\mathcal{A}$ to a specific subset $\mathcal{A}_{\text{partition}}$ increases the optimal regret from $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha ST})$ to $\widetilde{\Theta}(\sqrt{S\alpha T})$, illustrating that the feasibility of negatively correlated sampling directly dictates the difficulty of the problem.
4. **Lower Bound Construction Based on Independent Set Partitioning**: The maximum independent set is uniformly partitioned into $S$ subsets, each constructed as a multi-armed bandit subproblem. The regret lower bound is obtained via the chain rule of KL divergence and Pinsker's inequality.

## Highlights & Insights

1. **Elegant Unified Framework**: A single parameter, the independence number $\alpha$ of the feedback graph, characterizes the full spectrum from full information to semi-bandits. The regret formula $S\sqrt{T} + \sqrt{\alpha ST}$ is remarkably concise.
2. **Negative Correlation $\neq$ Obvious**: Although the constraint $\|v\|_1 = S$ intuitively suggests negative correlation, the authors provide a concrete counterexample showing that naive distributions may fail—for instance, when $x = (1, 0.8, 0.2)$ and $S=2$, the only solutions deterministically choose $\{1,2\}$ or $\{1,3\}$.
3. **Cross-disciplinary Transfer of Theoretical Tools**: The core lemma (Lemma 1) originates from dependent randomized rounding in matroid theory (Chekuri et al., 2009). Successfully applying it to bandit analysis is an ingenious cross-disciplinary connection.
4. **Natural Extension to Time-Varying Graphs**: The algorithm can process time-varying feedback graphs $\{G_t\}$ without modification, simply replacing the fixed $\alpha$ in the proof with $\alpha_t$.

## Limitations & Future Work

1. **General Action Subsets**: When $\mathcal{A}_0 \subsetneq \mathcal{A}$ and the exchange property is violated, negative correlation is unrealizable, and the algorithm performance degrades. Finding the optimal algorithm for general $\mathcal{A}_0$ remains an open problem.
2. **Weakly Observable Graphs**: The $T^{2/3}$ bound is only provided for the ETC algorithm in the stochastic reward setting; the problem under weakly observable graphs in the adversarial setting remains unsolved.
3. **Computational Complexity**: The efficiency of the projection step $\arg\min_{x \in \text{Conv}_\epsilon(\mathcal{A})} D_F(x, w^{t+1})$ depends on the structure of the action set, which is not discussed in detail.
4. **Purely Theoretical with No Experiments**: Realistic performance and the estimation of $\alpha$ in practical scenarios remain unknown due to the lack of numerical verification.
5. **Generalization to Non-linear Payoffs**: The current framework is restricted to linear payoffs $\langle v, r^t \rangle$. Extensions to non-linear reward functions are left for future work.

## Related Work & Insights

- **Alon et al. (2015)**: Pioneering work on multi-armed bandits with graph feedback, proving the optimal regret of $\widetilde{\Theta}(\sqrt{\alpha T})$.
- **Audibert et al. (2014)**: Classical result on combinatorial semi-bandits, establishing the optimal bound of $\widetilde{\Theta}(\sqrt{KST})$.
- **Chekuri et al. (2009)**: Dependent randomized rounding on matroids, the direct source of Lemma 1 in this paper.
- **Wen et al. (2024, NeurIPS)**: Prior work by the same authors on contextual bandits with graph feedback.
- **Kocák & Carpentier (2023)**: Reveals the exact regret landscape (mixture of $T^{1/2}$ and $T^{2/3}$) in small $T$ regimes.

This paper provides a theoretical foundation at the intersection of contextual bandits, combinatorial optimization, and graph structure learning. Future directions could consider introducing graph feedback into more complex combinatorial optimization scenarios (e.g., submodular optimization, routing problems).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introduces graph feedback to combinatorial semi-bandits for the first time and completely characterizes the minimax regret.
- Experimental Thoroughness: ⭐⭐⭐ — Purely theoretical work with no numerical experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, natural proof ideas, and intuitive examples.
- Value: ⭐⭐⭐⭐⭐ — Resolves important open problems and establishes benchmark results in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](../../ACL2025/others/learning_to_reason_from_feedback_at_test-time.md)
- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[NeurIPS 2025\] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](../../NeurIPS2025/others/put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](../../NeurIPS2025/others/semi-infinite_nonconvex_constrained_min-max_optimization.md)

</div>

<!-- RELATED:END -->
