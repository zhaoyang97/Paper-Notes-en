---
title: >-
  [Paper Note] Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions
description: >-
  [ICML2025][Uniform Price Auctions] For value-maximizing buyers with RoI constraints in repeated uniform-price multi-unit auctions, this work introduces the concept of "safe bidding strategies," proves that they only need to satisfy mild no-overbidding conditions, and designs a polynomial-time online learning algorithm that achieves a regret bound of $\widetilde{O}(M\sqrt{mT})$.
tags:
  - "ICML2025"
  - "Uniform Price Auctions"
  - "Safe Bidding Strategies"
  - "Value Maximization"
  - "RoI Constraint"
  - "Online Learning"
  - "Regret Bound"
date: 2026-05-08
content_hash: 9e7aaba6921be9cf
---

# Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions

**Conference**: ICML2025  
**arXiv**: [2406.03674](https://arxiv.org/abs/2406.03674)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Uniform Price Auctions, Safe Bidding Strategies, Value Maximization, RoI Constraint, Online Learning, Regret Bound

## TL;DR
For value-maximizing buyers with RoI constraints in repeated uniform-price multi-unit auctions, this work introduces the concept of "safe bidding strategies," proves that they only need to satisfy mild no-overbidding conditions, and designs a polynomial-time online learning algorithm that achieves a regret bound of $\widetilde{O}(M\sqrt{mT})$.

## Background & Motivation

### Problem Scenario
Uniform-price multi-unit auctions are widely used in carbon emissions trading (EU ETS), treasury bond auctions, and electricity markets. In these auctions, $K$ identical items are sold to $n$ buyers, and the unit price is uniformly set to the $K$-th highest bid.

### Buyer Behavior Model
Different from the traditional quasilinear utility model, this paper models the buyer as a **value maximizer**, whose goal is to maximize the total value obtained while satisfying a per-round **Return on Investment (RoI) constraint**:

$$V(\mathbf{b};\boldsymbol{\beta}_{-}) \geq (1+\gamma) P(\mathbf{b};\boldsymbol{\beta}_{-})$$

That is, the value obtained in each round is at least a fixed multiple of the payment. This model reflects practical scenarios where agencies (such as automated ad bidding systems and carbon emission intermediaries) manage bidding on behalf of their clients.

### Bidding Language
The $m$-uniform bidding format is adopted: the buyer submits $m$ (bid, quantity) pairs $\mathbf{b} = \langle(b_1,q_1),\dots,(b_m,q_m)\rangle$, where $b_1 > b_2 > \cdots > b_m > 0$. In practice, $m \ll M$ (where $M$ is the maximum demand); for example, in the EU ETS 2023 auctions, players submitted an average of ~4.35 pairs.

### Key Challenge
In the online setting of repeated auctions, buyers do not know their competitors' bids (which may be adversarial) when submitting strategies, yet they must satisfy the RoI constraint in **every round**. A strategy that is feasible under certain competitor bids may immediately violate the constraint under slightly changed competitor bids.

## Method

### 1. Safe Bidding Strategies

**Core Definition**: If an $m$-uniform strategy $\mathbf{b}$ satisfies the RoI constraint under **any** competitor bids $\boldsymbol{\beta}_{-}$, it is called a safe strategy.

**Theorem 3.1** (Characterization of Safe Strategies): Overbidding is not allowed in the safe strategy class $\mathscr{S}_m$, i.e.:

$$\mathscr{S}_m = \{\mathbf{b} = \langle(b_1,q_1),\dots,(b_m,q_m)\rangle : b_\ell \leq w_{Q_\ell}, \forall \ell \in [m]\}$$

where $w_j = \frac{1}{j}\sum_{\ell \leq j} v_\ell$ is the cumulative average valuation, and $Q_j = \sum_{\ell=1}^{j} q_\ell$.

**Theorem 3.2** (Safe Undominated Strategies): After eliminating weakly dominated strategies, the finite subset of candidate strategies is:

$$\mathscr{S}_m^{\star} = \{\mathbf{b} : b_\ell = w_{Q_\ell}, \forall \ell \in [m]\}$$

These strategies possess a "nested" structure—the $j$-th highest bid is exactly equal to the mean of the first $Q_j$ items on the valuation curve. A key property: **it depends solely on the buyer's own valuation curve**, independent of competitors' bids.

### 2. DAG Reduction of Offline Optimal Strategy

Although $\mathscr{S}_m^{\star}$ is a finite set, it contains $O(M^m)$ strategies. The authors utilize the **Value Decomposition Lemma** (Lemma 4.1) to decompose the objective function into individual bid-quantity pairs and construct a Directed Acyclic Graph (DAG):

- **Vertices**: source $s$, sink $d$, and $m$ layers of intermediate nodes $(\ell, j)$
- **Edge weights**: encoding the value obtained by the strategy under various competitor bids
- **Key Conclusion (Theorem 4.2)**: $s$-$d$ paths correspond one-to-one with strategies in $\mathscr{U}_m^{\star}$ (weight-maximizing path is the optimal safe strategy, which can be completed in $\text{poly}(m, M)$ time)

### 3. Online Learning Algorithm

Based on the path-to-strategy mapping on the DAG, the **weight-pushing** dynamic programming method of Takimoto & Warmuth (2003) is adopted:

**Algorithm Framework** (Algorithm 1):
1. **UPDATE**: Update edge probabilities based on feedback from the previous round, compute the normalization factor $\Gamma^{t-1}(u)$ recursively from bottom to top, and then update edge probabilities $\varphi^t(e)$ using exponential weighting.
2. **SAMPLE**: Start from the source node and sample an $s$-$d$ path according to a Markov chain.
3. **MAP**: Map the path to a safe strategy and submit it.

**Full-Information Setting**: Directly observe all competitor bids and set edge weights according to the formula.

**Bandit Setting**: Observe only the allocated quantity and payment. Use the unbiased estimator $\hat{\mathsf{w}}^t(e)$ instead of true edge weights, leveraging the semi-bandit feedback structure (rewards on each edge along the observed path can be observed).

### 4. Regret Bounds

| Setting | Regret Upper Bound | Time Complexity per Round |
|------|---------|-------------|
| Full-Information | $\widetilde{O}(M\sqrt{mT})$ | $\text{poly}(m, M)$ |
| Bandit | $\widetilde{O}(M^2 m^{3/2}\sqrt{T})$ | $\text{poly}(m, M)$ |
| **Lower Bound** | $\Omega(M\sqrt{T})$ | — |

Compared to prior work on quasilinear utility buyers, the proposed method: (a) eliminates the need to discretize the bid space, making the time complexity independent of $T$; (b) improves the regret dependency under the bandit setting from $T^{2/3}$ to $\sqrt{T}$.

### 5. Competing Against Stronger Benchmarks: Richness Ratio

Define the richness ratio $\alpha \in (0,1]$ to measure the approximation of safe strategies relative to richer strategy classes:

| Benchmark Strategy Class | Richness Ratio $\alpha$ | Explanation |
|-----------|------------------|------|
| RoI-feasible + $\leq m$ pairs | $1/2$ | Safety cost is bounded and independent of $m$ |
| Safe + $\leq m'$ pairs ($m' \geq m$) | $m/m'$ | Expressiveness cost grows linearly |
| RoI-feasible + $\leq m'$ pairs | $m/(2m')$ | Multiplicative cost across two dimensions |

Matching hard instances are constructed for all bounds to prove tightness.

## Key Experimental Results

### Semi-Synthetic Data on EU ETS Emission Auctions

Using publicly available aggregate statistical data from the European Union Emissions Trading System (EU ETS) auctions in 2022-2023, individual bids were synthesized for experiments:

- **Empirical Richness Ratio vs. RoI-Feasible Strategies**: When $m \geq 4$, the lower bound of $\alpha$ exceeds **0.95**, significantly outperforming the theoretical worst-case of $1/2$.
- **Empirical Richness Ratio vs. Safe Strategies with More Bid Pairs**: Even with $m=10$, the ratio of the value obtained by the 1-uniform safe strategy to that of the optimal 10-uniform safe strategy is approximately **0.8**, remarkably better than the worst-case of $0.1$.
- The marginal utility of increasing the number of bid pairs becomes mostly flat after $m=4$.

### Key Findings
The worst-case tight bounds only occur in highly pathological instances (requiring exponential-sized, meticulous construction); in practical scenarios, the performance of safe strategies is close to optimal.

## Highlights & Insights

1. **Simplicity of Safe Strategies**: They are determined solely by the valuation curve without needing competitor information, featuring intuitive "no-overbidding" conditions and nested structures.
2. **Elegance of DAG Reduction**: Compresses the exponential strategy space into a polynomial-sized DAG, enabling efficient online learning.
3. **Robustness Quantification**: Systematically assesses the cost of safe strategies against stronger benchmarks through the richness ratio, proving all bounds to be tight.
4. **$\sqrt{T}$ Regret**: Compared to the $T^{2/3}$ regret in the quasilinear utility setting, the value-maximizing setting is actually easier to learn.
5. **Practical Applicability**: The algorithm's computational complexity depends only on $m$ and $M$ (independent of $T$), and experiments show that a very small $m$ is sufficient to achieve near-optimal performance.

## Limitations & Future Work

1. **Granularity of RoI Constraints**: This work primarily considers per-round RoI constraints. Although Section 6.2 introduces a heuristic method for sliding-window RoI, it lacks theoretical guarantees.
2. **Gap Between Regret Bounds**: Under full information, the upper bound is $O(M\sqrt{mT\log M})$ versus the lower bound of $\Omega(M\sqrt{T})$, leaving a gap of $\sqrt{m\log M}$.
3. **Adaptive Adversary**: Section 6.1 extends the analysis to an adaptive adversary, but the regret bound degrades to a high-probability bound.
4. **Single-Commodity Assumption**: Only auctions for multiple identical units are considered, without coverage for more general combinatorial auction settings.
5. **Assumed Fixed Valuation Curve**: The main model assumes fixed valuations; time-varying valuations are only briefly discussed as an extension.
6. **Semi-Synthetic Experiments**: Due to privacy restrictions, only synthesized individual bidding data could be used, which limits validation on real individual-level data.

## Related Work & Insights

- **Value Maximization Literature**: Wilkens et al. (2016) pioneered the study of truthful auction design for value-maximizing agents; Golrezaei et al. (2021c) verified the presence of soft RoI constraints in online advertising.
- **Learning in Multi-Unit Auctions**: Brânzei et al. (2023), Galgana & Golrezaei (2024), Potfer et al. (2024) explored online bidding under quasilinear utility; this work is the first systematic study targeting value maximization under RoI constraints.
- **Technical Toolkit**: Online learning on DAGs is adapted from the weight-pushing algorithm of Takimoto & Warmuth (2003); the richness ratio concept is closely related to the approximate regret framework of Streeter & Golovin (2008) and Niazadeh et al. (2022).
- **Practical Relevance**: EU ETS auctions (Regulations, 2019) and treasury bond auctions serve as direct application contexts for the proposed model.

## Rating
- Novelty: ⭐⭐⭐⭐ — The concept of safe strategies is natural yet non-trivial; the DAG reduction and richness ratio analytical framework are self-contained.
- Experimental Thoroughness: ⭐⭐⭐ — Semi-synthetic experiments effectively support the theory, but the work lacks real-world datasets and runtime experiments of the online algorithm.
- Writing Quality: ⭐⭐⭐⭐⭐ — The structure is clear, aligning motivation, theory, and experiments seamlessly, and the managerial insights offer practical value.
- Value: ⭐⭐⭐⭐ — Provides a practical theoretical framework for automated bidding in large-scale repeated auctions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Coordination of Value-Maximizing Bidders](../../ICML2026/others/on_the_coordination_of_value-maximizing_bidders.md)
- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [\[ICML 2025\] Prediction via Shapley Value Regression (ViaSHAP)](prediction_via_shapley_value_regression.md)
- [\[ICML 2025\] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](../../NeurIPS2025/others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)

</div>

<!-- RELATED:END -->
