---
title: >-
  [Paper Note] Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity
description: >-
  [NeurIPS 2025][AI Safety][Fairness] This paper introduces the Fair Minimum Labeling (FML) problem, which aims to design minimum-cost temporal edge activation schemes ensuring sufficient temporal-path reachability for eac…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Fairness"
  - "Temporal Graphs"
  - "Minimum Labeling"
  - "Reachability"
  - "Approximation Algorithms"
date: 2026-05-08
content_hash: b61747f698ad5404
---

# Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity

**Conference**: NeurIPS 2025
**arXiv**: [2510.03899](https://arxiv.org/abs/2510.03899)
**Code**: [Available](https://gitlab.com/tgdesign/fml)
**Area**: AI Safety
**Keywords**: Fairness, Temporal Graphs, Minimum Labeling, Reachability, Approximation Algorithms

## TL;DR

This paper introduces the Fair Minimum Labeling (FML) problem, which aims to design minimum-cost temporal edge activation schemes ensuring sufficient temporal-path reachability for each node group in a network to satisfy fair coverage requirements. The paper proves FML is NP-hard and inapproximable beyond a certain factor, and provides an approximation algorithm based on probabilistic tree embeddings that matches the hardness lower bound.

## Background & Motivation

**Background**: Modern networked systems (federated learning, recommender systems, distributed sensor networks) operate under limited resources while needing to guarantee fairness among participants or data sources. Graph structures describe potential interactions, and the temporal dimension—when connections are activated—determines the actual flow of information, updates, and influence.

**Limitations of Prior Work**:
- Existing fairness-aware graph mining methods (fair clustering, fair node embeddings, fair influence maximization) primarily target static graphs and do not address temporal interactions.
- Research on temporal graph connectivity (the Minimum Labeling problem) focuses on global connectivity without considering group fairness constraints.
- Simple heuristics may over-consume bottleneck resources or systematically neglect hard-to-reach nodes and minority groups.

**Key Challenge**: A formal framework is needed that simultaneously optimizes resource efficiency of temporal edge activations and group-fair reachability guarantees. FML fills the gap between fairness-aware graph optimization and temporal connectivity modeling.

## Method

### Overall Architecture

**Problem Definition (FML)**: Given a vertex-colored static graph $G=(V,E,c)$, a color set $C$, a terminal node set $T$, and a demand function $f_c$ for each color $c$, the goal is to find a temporal labeling that minimizes the total number of time labels while ensuring that sufficiently many nodes of each color $c$ can reach the terminal nodes via temporal paths.

**Approximation Framework (Algorithm 1)**:
1. Compute the shortest-path metric on $G$.
2. Apply the FRT algorithm to generate a probabilistic tree embedding, sampling a weighted tree $\mathcal{T}$.
3. Solve FML on the tree rooted at terminal $t$ via exact or approximate DP.
4. Project the tree solution back to the original graph $G$.

### Key Designs

**1. Probabilistic Tree Embedding**

The Fakcharoenphol-Rao-Talwar (FRT) algorithm is employed: any $n$-point metric space can be embedded into a distribution over tree metrics with expected stretch factor $O(\log n)$, and this bound is tight. The key properties are: for all $x,y$, $d(x,y) \leq d_T(x,y)$ (dominance) and $\mathbb{E}[d_T(x,y)] \leq O(\log n) \cdot d(x,y)$ (expected stretch). This reduces solving FML on general graphs to DP on trees, exploiting the simplicity of tree structure.

**2. Exact Tree DP Algorithm (Section 5.1)**

Labels $(b, r, c)$ are computed bottom-up, where $b$ and $r$ denote the number of blue and red nodes in the subtree, and $c$ is the total edge weight connecting them.

- Leaf nodes: blue node gets label $(1,0,0)$; red node gets $(0,1,0)$.
- For an internal node $v$ with $l$ children, at most one label is selected per child.
- For each $(b,r)$ pair, only the minimum-cost label $c$ is retained; the upper bound is $O(n^2)$ labels per node.
- Total complexity: $O(n^5)$.

**3. Bicriteria Approximation Algorithm (Section 5.2)**

Geometric bucketing reduces the number of labels:

- The interval $[0,n]$ is partitioned into geometrically spaced buckets; only the minimum-cost label is retained per bucket pair.
- Label count is reduced to $O(\varepsilon^{-2} \log^2 n)$.
- Coverage violation factor: $\xi = (1+\varepsilon)^{H+1}$, where $H$ is the tree height.
- Approximation guarantee: cost $c' \leq c_{\mathrm{opt}}$, coverage $b' \geq b_{\mathrm{opt}}/\xi$.
- Complexity: $O(n^2 + n \cdot \varepsilon^{-4} \log^4 n)$.

**4. Projection of Tree Solution to Graph**

Each temporal edge in the tree solution is projected to a sequence of temporal edge activations along the shortest path in $G$. Strictly increasing timestamps ensure validity of temporal paths. The projection preserves coverage constraints, either exactly or up to an approximation factor of $\xi$.

### Loss & Training

FML is a combinatorial optimization problem and does not involve neural network training. In the multi-source learning experiments, an MLP classifier (two hidden layers of 32 neurons, ReLU activations, sigmoid output) is used with the Adam optimizer ($\text{lr}=0.01$, batch size $=64$, weight decay $=0.01$), binary cross-entropy loss, 100 epochs, and early stopping.

**Theoretical Results**:
- NP-hard and $\Omega(\log n)$-inapproximable even for a single terminal (proved via reduction from Set Cover).
- Exact algorithm achieves $O(\log n)$ expected cost approximation (matching the lower bound).
- Bicriteria algorithm provides bounded guarantees on both cost and fairness.

## Key Experimental Results

### Main Results (Multi-Source Data Collection)

Learning task on a random geometric graph ($|V|=1024$, connection radius $r=0.2$):

| Algorithm | Time (s) | Cost | CovB | CovR | AccB | AccR |
|-----------|----------|------|------|------|------|------|
| Greedy | 0.49 | 88.3 | 64.0 | 0.0 | 1.00 | 0.27 |
| Closest | 0.73 | 224.0 | 44.3 | 32.0 | 0.97 | 0.94 |
| Alternating | 0.71 | 212.8 | 34.2 | 32.0 | 0.96 | 0.94 |
| FMLapprox | 69.20 | 74.3 | 33.5 | 32.3 | 0.98 | 0.93 |
| FMLbiApprox | 8.54 | 73.9 | 33.1 | 31.3 | 0.97 | 0.94 |

- Greedy completely ignores the R group (coverage $= 0$), resulting in severe unfairness.
- FML methods achieve near-equal coverage for both groups at a cost of only ~74, far below the 212–224 of Closest/Alternating.

### Scalability on Real Networks (Pokec Dataset)

Test on a subgraph of 20,000 nodes (201,900 edges):

| epsilon | Runtime (s) | Accuracy |
|---------|-------------|----------|
| 0.01 | 3625 ± 421 | 93.1% |
| 0.001 | 4662 ± 991 | 99.3% |

Theoretical guarantee: with tree height $H \approx 6.68$, $\varepsilon=0.01$ guarantees approximately 92.6% coverage; the empirical result of 93.1% validates the theoretical bound.

### Ablation Study

- **FMLbiApprox vs. FMLapprox**: On Barabási-Albert graphs, biApprox is an order of magnitude faster; at $\varepsilon=0.001$, the coverage ratio approaches 1.0.
- **Effect of terminal location**: Whether the terminal is central (high degree) or peripheral (low degree), biApprox consistently achieves 98%+ coverage.
- Over 99.9% of runtime is spent on DP over the tree; tree embedding time is negligible.

### Key Findings

- Fairness-agnostic Greedy achieves low cost but completely and systematically neglects one group, potentially producing biased models.
- FML methods achieve the best trade-off among cost, fairness, and downstream model accuracy.
- The bicriteria approximation substantially improves scalability (~8× speedup) with only a marginal increase in cost.

## Highlights & Insights

1. **Formalization of a novel problem FML**: For the first time, cost optimization of temporal edge activations and group-fair reachability constraints are jointly modeled, generalizing the classical ML and MSL problems.
2. **Tight approximation bounds**: The $\Omega(\log n)$ lower bound matches the $O(\log n)$ upper bound, achieving theoretical optimality.
3. **Creative application of probabilistic tree embeddings**: An NP-hard problem on general graphs is reduced to an exactly solvable DP on trees, balancing theoretical guarantees with practical efficiency.
4. **Elegant experimental design**: The two data source groups encode orthogonal classification rules, so accurate predictions on both groups require training on data from both—naturally demonstrating the necessity of fair data collection.

## Limitations & Future Work

1. The current algorithm supports only a single terminal and two groups; multiple terminals and multiple groups require a more complex DP state space.
2. Centralized control over edge activations is assumed, making the approach unsuitable for decentralized settings (e.g., classical federated learning).
3. Fairness in FML is defined as group-level temporal reachability, without addressing individual fairness or time-varying fairness notions.
4. The $O(n^5)$ complexity of the exact DP remains challenging for large-scale networks.
5. Experimental data generation is largely synthetic; validation on real-world networks is limited to the single Pokec dataset.

## Related Work & Insights

- **Minimum Labeling** (Mertzios et al. / Klobas et al.): FML generalizes these classical temporal graph design problems.
- **FRT Probabilistic Tree Embeddings** (Fakcharoenphol et al.): Theoretical foundation for the $O(\log n)$ tight approximation.
- **Fair Influence Maximization**: Fairness optimization on static graphs, which FML extends to the temporal setting.
- The work can inspire fair communication scheduling designs in distributed federated learning.

## Rating

4/5 — Excellent. The theoretical contributions are rigorous (NP-hardness proof with a matching approximation bound), and the problem formalization is both novel and practically relevant. The main limitation is that the current approach addresses only a restricted setting (single terminal, two groups), leaving a gap to a fully general solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Fairness-Performance Pareto Front Computation](efficient_fairness-performance_pareto_front_computation.md)
- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)
- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](../../ICCV2025/ai_safety/vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)

</div>

<!-- RELATED:END -->
