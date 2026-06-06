---
title: >-
  [Paper Note] Distribution Learning Meets Graph Structure Sampling
description: >-
  [NeurIPS 2025][Reinforcement Learning][Bayesian networks] This paper establishes a novel connection between PAC learning of high-dimensional probabilistic graphical models and efficient counting/sampling of graph structu…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Bayesian networks"
  - "distribution learning"
  - "graph structure sampling"
  - "online learning"
  - "chordal graphs"
date: 2026-05-08
content_hash: 93ff7201a4185b47
---

# Distribution Learning Meets Graph Structure Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2405.07914](https://arxiv.org/abs/2405.07914)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Bayesian networks, distribution learning, graph structure sampling, online learning, chordal graphs

## TL;DR
This paper establishes a novel connection between PAC learning of high-dimensional probabilistic graphical models and efficient counting/sampling of graph structures. By reducing the maintenance of an exponentially large expert pool to a weighted DAG sampling problem via online learning frameworks (EWA/RWM), the paper presents the first efficient agnostic learning algorithm for Bayesian networks with chordal graph skeletons, and improves the sample complexity for tree-structured distributions from $O(nk^3/\varepsilon)$ to the optimal $O(nk^2/\varepsilon)$.

## Background & Motivation

**Background**: Probabilistic graphical models—particularly Bayesian networks—are central tools for modeling high-dimensional distributions, with broad applications in gene regulatory networks, protein signaling networks, and brain connectivity networks. Learning a Bayesian network involves two steps: structure learning (identifying the dependency graph) and parameter learning (estimating conditional probability tables).

**Limitations of Prior Work**: Existing efficient learning algorithms are limited to tree-structured distributions (the classical Chow-Liu algorithm handles $d=1$ with tree skeletons) and polytree distributions with known skeletons (restricted to the realizable setting). No efficient agnostic learning algorithm exists for more general graph structures such as chordal graphs.

**Key Challenge**: Online learning frameworks (EWA/RWM) can reduce distribution learning to prediction over an expert pool, but standard prediction algorithms run in time at least linear in the number of experts—while the number of candidate DAG structures for Bayesian networks is exponential, making direct enumeration infeasible.

**Goal**: (1) Design the first efficient agnostic PAC learning algorithm for Bayesian networks with chordal graph skeletons; (2) improve the dependence of sample complexity on alphabet size $k$ from $k^3$ to the optimal $k^2$ for tree-structured distribution learning; (3) resolve the open problem posed by Choo et al. [2023] on efficient agnostic learning of polytrees.

**Key Insight**: The key observation is that the weighted expert sampling performed at each round by RWM/EWA is essentially equivalent to weighted sampling over DAG structures according to specific weights. If this weighted sampling can be performed efficiently—by exploiting the combinatorial properties of graph structures—the exponential expert pool bottleneck can be bypassed.

**Core Idea**: Reduce the problem of maintaining an exponentially large expert pool in online learning to a weighted counting/sampling problem over DAG structures, leveraging the clique tree decomposition of chordal graphs and the matrix-tree theorem for efficient sampling.

## Method

### Overall Architecture
The algorithm proceeds in two phases:
1. **Parameter pre-learning**: For each node $i$ and each candidate parent set $\mathrm{pa}(i)$, learn the conditional distribution from an independent sample batch using an add-one (Laplace) estimator.
2. **Structure learning (online learning phase)**: Apply EWA or RWM over all possible DAG structures (acyclic orientations), where each DAG corresponds to a Bayesian network expert with negative log-likelihood as the loss. Each round's update is implemented via an efficient weighted sampling algorithm.

### Key Designs
1. **Reduction from online learning to distribution learning**: PAC distribution learning is cast as an online prediction problem. The average regret over $T$ rounds bounds the KL divergence: the mixture $\frac{1}{T}\sum \hat{P}_t$ produced by EWA is an improper approximation of $P^*$; the prediction $\hat{P}_t$ randomly selected by RWM yields a proper approximation. McDiarmid's inequality converts expected guarantees into high-probability guarantees.
2. **Efficient sampling for tree structures (matrix-tree theorem)**: When the Bayesian network has a tree structure (in-degree $=1$), sampling under RWM is equivalent to sampling a weighted rooted spanning arborescence from a weighted complete graph. Each arborescence $A$ is selected with probability proportional to $\prod_{e \in A} w_e$, where weights $w_e$ are computed explicitly from observed samples. Tutte's matrix-tree theorem enables exact sampling in polynomial time, providing an alternative to the Chow-Liu algorithm.
3. **Dynamic programming for paths and polytrees**: For path-skeleton structures, two directional normalizing constants $Z_{j,\leftarrow}$ and $Z_{j,\rightarrow}$ are maintained and combined via DP recursion to compute the total normalizing constant $Z$. Sampling is performed by backtracking through the DP table. For general polytrees, the DP is extended over subtree structures: for each node $v$ and each orientation of its incident edges, the total weight of all consistent orientations in the subtree is maintained.
4. **Clique tree decomposition for chordal graphs**: A key property of chordal graphs is that every cycle of length $\geq 4$ has a chord. The algorithm first constructs the clique tree decomposition of the chordal graph, then independently performs weighted counting/sampling of acyclic orientations within each subtree of the clique tree. The special structure of chordal graphs ensures independence across subtrees, avoiding cyclic dependency issues.

### Loss & Training
- **Loss function**: Negative log-likelihood $\ell(\hat{P}, x) = -\log \hat{P}(x)$
- **EWA hyperparameter**: $\eta > 0$ controls the learning rate; weight update rule: $w_{i,t} \leftarrow w_{i,t-1} \cdot P_i(x^{(t)})^\eta$
- **RWM strategy**: At each round, one expert is randomly selected according to current weights for prediction
- **Sample allocation**: A portion of samples is used for pre-learning conditional distribution parameters; the remainder is used in the online learning phase

## Key Experimental Results

### Main Results
This is a theoretical paper; the primary contributions are sample complexity bounds. The core results are as follows:

| Setting | Skeleton Type | Learning Mode | Sample Complexity |
|---|---|---|---|
| Agnostic Improper | Chordal (in-degree $\leq d$) + known skeleton | EWA mixture | $\tilde{O}(n^4/\varepsilon^4 + nk^{d+1}/\varepsilon)$ |
| Agnostic Proper | Chordal (in-degree $\leq d$) + known skeleton | RWM single network | $\tilde{O}(n^3/(\varepsilon^2\delta^2) + nk^{d+1}/\varepsilon)$ |
| Realizable Improper | Tree + unknown skeleton | EWA mixture | $\tilde{O}(nk^2/\varepsilon)$ |
| Realizable Proper | Tree + unknown skeleton | RWM | $\tilde{O}(n^3/(\varepsilon^2\delta^2) + nk^2/\varepsilon)$ |

### Ablation Study (Theoretical Comparison)

| Method | Skeleton Type | Efficient? | Agnostic? | Additional Assumptions |
|---|---|---|---|---|
| Bhattacharyya et al. [2023] | Tree | ✓ | ✓ | None |
| Choo et al. [2023] | Polytree | ✓ | ✗ | Known skeleton |
| Abbeel et al. [2006] | Bounded total degree | ✓ | ✗ | None |
| BCD [2020] | Bounded in-degree | ✗ | ✗ | None |
| **Ours** | **Tree** | **✓** | **✓** | **None** |
| **Ours** | **Chordal + bounded in-degree** | **✓** | **✓** | **Known skeleton** |

### Key Findings
1. The dependence of sample complexity on $k$ for tree-structured distribution learning is improved from $O(k^3)$ to the optimal $O(k^2)$, matching the lower bound $\Omega(nk^2)$.
2. The runtime for Bayesian networks with chordal skeletons is $\mathrm{poly}(n, k, 1/\varepsilon, 1/\delta)$ when in-degree $d$ is constant; it remains polynomial even when both undirected degree $\Delta$ and $d$ are $O(\log n)$.
3. When $d$ is unbounded, exponential runtime is unavoidable, since a chordal Bayesian network with in-degree $(n-1)$ can represent arbitrary $n$-dimensional distributions.

## Highlights & Insights
1. **Cross-domain inspiration**: The paper establishes an elegant connection between PAC learning theory and combinatorial counting/sampling, theoretically unifying two seemingly unrelated research directions—maintaining a weighted expert pool in online learning is equivalent to weighted counting of DAGs.
2. **Theoretical elegance**: The progressive extension from product distributions to trees, polytrees, and chordal graphs is natural, with each step exploiting increasingly refined structural properties of the underlying graph.
3. **Optimality**: The sample complexity $O(nk^2/\varepsilon)$ for tree-structured learning matches the information-theoretic lower bound, establishing the optimality of the approach.
4. **Resolution of open problems**: The paper resolves the open problem posed by Choo et al. [2023] on efficient agnostic learning of polytrees.

## Limitations & Future Work
1. **Known skeleton assumption**: The learning algorithms for chordal graphs and polytrees require the skeleton to be known in advance, which may necessitate a prior skeleton learning step in practice.
2. **Constant in-degree requirement**: The efficiency of the algorithm requires in-degree $d$ to be constant, limiting applicability to high-in-degree settings.
3. **Extension to general bounded-treewidth graphs**: Chordal graphs are a subclass of bounded-treewidth graphs; whether the approach generalizes to general treewidth graphs remains open, involving the relationship between the Tutte polynomial and weighted counting.
4. **Role of approximate sampling**: The current approach relies entirely on exact sampling algorithms; whether MCMC or other approximate sampling techniques can extend the applicable scope remains unexplored.
5. **Absence of empirical validation**: As a purely theoretical work, no experiments are provided to validate the practical performance and runtime efficiency of the algorithms.

## Related Work & Insights
- **Chow-Liu algorithm** [1968]: The classical method for learning tree-structured distributions; this paper provides an alternative with improved sample complexity.
- **Choo et al.** [2023]: Presents an optimal-sample-complexity learning algorithm for polytrees in the realizable setting, leaving the agnostic setting as an open problem.
- **Beagleholer et al.** [SODA 2023]: Also leverages efficient sampling to bypass the computational bottleneck of RWM/EWA in fast equilibrium computation for structured games, partially inspiring the present work.
- **Abbeel et al.** [2006]: Improperly learns Bayesian networks with bounded total degree using polynomial samples and runtime, but without agnostic guarantees.
- **Stanley** [1973]: The count of acyclic orientations relates to the value of the Tutte polynomial at $(2, 0)$, providing additional inspiration for potential extensions of the chordal graph method.

## Rating
- ⭐⭐⭐⭐⭐ Outstanding theoretical contribution. The paper establishes a novel connection between distribution learning and combinatorial sampling, resolves multiple open problems, and achieves optimal results through an elegant methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)

</div>

<!-- RELATED:END -->
