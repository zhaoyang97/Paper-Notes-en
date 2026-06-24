---
title: >-
  [Paper Note] Near Optimal Best Arm Identification for Clustered Bandits
description: >-
  [ICML2025][Other][best arm identification] In the multi-agent clustered multi-armed bandit setting, this paper proposes two algorithms, Cl-BAI and BAI-Cl, which exploit the clustering structure to significantly reduce the sample complexity of best arm identification. It is further proved that BAI-Cl++ achieves minimax optimality when $M$ is a constant.
tags:
  - "ICML2025"
  - "Other"
  - "best arm identification"
  - "clustered bandits"
  - "federated MAB"
  - "successive elimination"
  - "sample complexity"
date: 2026-05-08
content_hash: 780aa6cd8bfdcb9f
---

# Near Optimal Best Arm Identification for Clustered Bandits

**Conference**: ICML2025  
**arXiv**: [2505.10147](https://arxiv.org/abs/2505.10147)  
**Code**: None  
**Area**: Other  
**Keywords**: best arm identification, clustered bandits, federated MAB, successive elimination, sample complexity

## TL;DR

In the multi-agent clustered multi-armed bandit setting, this paper proposes two algorithms, Cl-BAI and BAI-Cl, which exploit the clustering structure to significantly reduce the sample complexity of best arm identification. It is further proved that BAI-Cl++ achieves minimax optimality when $M$ is a constant.

## Background & Motivation

Federated Multi-Armed Bandits (Federated MAB) have recently attracted significant attention: $N$ agents collaborate through a central learner, each facing a stochastic bandit problem. In real-world scenarios (such as movie recommendations, advertisement placement, and Yelp restaurant recommendations), preferences of different users inherently exhibit **heterogeneity**. Naively running BAI independently for each agent would result in a substantial waste of samples.

This paper models heterogeneity using **clustering**: $N$ agents are partitioned into $M$ unknown clusters, where agents within the same cluster share the same bandit instance (i.e., the same set of mean vectors and the same best arm), while different clusters have different best arms. The goal is to identify the respective best arm for all $N$ agents under the $\delta$-PC framework, while minimizing both sample complexity and communication overhead.

The key challenges are:

- The mapping of agents to clusters $\mathcal{M}: [N] \to [M]$ is unknown a priori.
- The dual tasks of clustering and best arm identification must be accomplished simultaneously.
- Communication is costly in the federated setting.

## Method

### Problem Formalization

An instance is defined as $\mathcal{I} = ([N], [M], [K], \mathcal{M}, \Pi)$, where $K$ arms are common to all agents. Arm $k$ of bandit $m$ corresponds to a 1-subGaussian reward distribution $\Pi_{m,k}$ with mean $\mu_{m,k}$. The best arm is $k_m^* = \arg\max_k \mu_{m,k}$, and the gap is defined as $\Delta_{m,j} = \mu_{m,k_m^*} - \mu_{m,j}$.

**Separability Assumption (Assumption 2.1)**: There exists $\eta > 0$ such that for any two distinct bandits $a, b$:

$$\mu_{b, k_b^*} - \mu_{b, k_a^*} \geq \eta, \quad \forall a \neq b$$

That is, the best arm of one bandit performs at least $\eta$ worse on any other bandit, ensuring that different bandits can be distinguished through sampling.

### Algorithm I: Cl-BAI (Cluster then BAI)

**Two-Stage Pipeline**:

1. **Clustering Phase**: Each agent $i$ independently runs Successive Elimination (SE) with parameters $\gamma = (\delta / 12NK)^{4/3}$ and $R = \log(17/\eta)$, yielding an active arm set $S_i$ and empirical means $\hat{\mu}^i$. Utilizing the key property—empirical means of agents in the same cluster are consistent within $\eta/2$, whereas agents in different clusters must display differences—a graph $\mathcal{G}$ is constructed, where the connected components constitute the clustering result.
2. **BAI Phase**: One representative agent is selected from each cluster to run SE ($R = \infty$) again on the active arm set $S_i$ to find the best arm, which is then broadcast to all agents in the same cluster.

### Algorithm II: BAI-Cl (BAI then Cluster)

**Two-Stage Pipeline (in reverse order)**:

1. **BAI Phase**: Randomly sample agents, running SE for each new agent to find its best arm. If this best arm is already in the set $S$, it is matched directly; otherwise, it is added to $S$. The first phase terminates when $|S| = M$. The number of sampled agents is determined by the Coupon Collector's problem, with an expectation of $O(M \log M)$.
2. **Clustering Phase**: For the remaining $N - O(M\log M)$ agents, SE is run only on the $M$ candidate best arms, significantly reducing the sample complexity per agent.

### Algorithm III: BAI-Cl++ (Improved Version)

Based on BAI-Cl, an additional assumption is introduced (Assumption 6.1): the difference in means of the best arm $k_i^*$ under different bandits is at least $\eta_1$, i.e., $|\mu_{i,k_i^*} - \mu_{j,k_i^*}| \geq \eta_1$. In the second phase, an improved $\widehat{SE}$ procedure (Algorithm 4) is employed, which leverages the means $\bar{\mu}_S$ estimated in the first phase for fast verification, thereby reducing the multiplier of the $\log(1/\delta)$ and $\log N$ terms.

## Theoretical Results

### Sample Complexity Comparison

| Algorithm | Sample Complexity (Main term, omitting log factors) | Communication Cost |
|------|------|------|
| Naive (Independent BAI) | $N \cdot K \cdot \bar{\Delta}^{-2}$ | $O(N \log K \cdot c_b)$ |
| Cl-BAI | $N \cdot K \cdot \max\{\bar{\Delta}, \eta\}^{-2} + M \cdot K \cdot \bar{\Delta}^{-2}$ | $O(N \cdot K \cdot c_r)$ |
| BAI-Cl | $M \cdot K \cdot \bar{\Delta}^{-2} + N \cdot M \cdot \eta^{-2}$ | $O(N \cdot M \log K \cdot c_b)$ |
| BAI-Cl++ | $M \cdot K \cdot \bar{\Delta}^{-2} + N \cdot M \cdot \bar{\Delta}^{-2} \cdot (\log M) + N \cdot \eta_1^{-2}$ | $O(N \cdot M \log K \cdot c_b)$ |

Wherein $\bar{\Delta}^{-2} = \frac{1}{MK} \sum_{m,k} \Delta_{m,k}^{-2}$ reflects the average problem difficulty.

### Minimax Lower Bound

For instance classes satisfying the assumptions, the expected sample complexity of any $\delta$-PC algorithm satisfies:

$$\mathbb{E}[T_\delta^\nu(\mathcal{A})] \gtrsim \max\{M(K-M),\; N\} \cdot \frac{\log(1/\delta)}{\Delta^2}$$

The lower bound is the max of two terms: the first term $M(K-M)$ corresponds to identifying the set of best arms for the $M$ bandits, and the second term $N$ corresponds to determining which bandit each agent belongs to. When $M$ is a constant, BAI-Cl++ achieves order-wise minimax optimality.

### Experimental Validation

- **MovieLens-1M**: 100 users partitioned into 6 age groups; BAI-Cl++ reduces the sample complexity by **72%** compared to the naive baseline.
- **Yelp**: Under a similar setup, BAI-Cl++ reduces the sample complexity by **65%**.
- On synthetic data, the advantage is even more pronounced when $M \ll N$.

## Highlights & Insights

1. **Two complementary algorithm design paradigms**: clustering-then-best-arm-identification vs. best-arm-identification-then-clustering. They are suitable for different scenarios respectively. The first phase of Cl-BAI can be parallelized, while BAI-Cl achieves higher sample efficiency when $M \ll N, K$.
2. **Clever application of Coupon Collector**: BAI-Cl utilizes a random sampling + coupon collector analysis, requiring only $O(M\log M)$ representatives to cover all clusters.
3. **Rigorous theoretical guarantees**: $\delta$-PC correctness, sample complexity upper bounds, and minimax lower bounds, proving optimality when $M$ is a constant.
4. **Clear analysis of communication-sample complexity trade-off**: This reflects the core concept of trading sampling for communication via collaboration in federated learning.

## Limitations & Future Work

1. **Strong separability assumption**: Assumption 2.1 requires the best arms of different bandits to be distinct and have a gap of at least $\eta$. When $\eta$ is excessively small, the sample complexity of the clustering phase ($\sim \eta^{-2}$) may dominate the overall complexity.
2. **Assumption of balanced cluster sizes**: The analysis of BAI-Cl assumes that agents are uniformly distributed across the $M$ clusters. Although the non-uniform case can be generalized, it has not been analyzed thoroughly.
3. **$M$ needs to be known**: The algorithms require prior knowledge of the number of clusters $M$ (or its upper bound). In practice, $M$ might be unknown.
4. **Limited to the fixed-confidence setting**: The fixed-budget variant is not discussed.
5. **BAI-Cl++ requires extra assumptions**: Assumption 6.1 requires prior knowledge of $\eta_1$, which limits its range of applicability.
6. **Sole reliance on Successive Elimination**: Although the authors mention that SE can be replaced by methods like Track-and-Stop, no corresponding theoretical analysis is provided.

## Related Work & Insights

- **(Chawla et al., 2023)**: Most related; studies a similar clustered structure but with the objective of group regret and allows gossip communication.
- **(Kota et al., 2023)**: Non-clustered federated BAI, identifying both global and local best arms.
- **(Pal et al., 2023)**: Regret minimization for clustered federated bandits using online matrix completion.
- Inspiration: One direction is to extend the clustered structure to BAI problems under contextual bandit or linear bandit settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — Two complementary paradigms (cluster-first / BAI-first) + minimax optimality proof.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic + real-world data (MovieLens, Yelp), validating theoretical predictions.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical analysis and standard algorithm descriptions.
- Value: ⭐⭐⭐⭐ — Provides a near-optimal algorithmic framework and a comprehensive theoretical foundation for clustered federated BAI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Near-Optimal Best-of-Both-Worlds Algorithm for Federated Bandits](../../ICLR2026/learning_theory/a_near-optimal_best-of-both-worlds_algorithm_for_federated_bandits.md)
- [\[ICML 2025\] Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)
- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](../../ICML2026/learning_theory/optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)

</div>

<!-- RELATED:END -->
