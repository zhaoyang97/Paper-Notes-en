---
title: >-
  [Paper Note] Learning to Approximate Uniform Facility Location via Graph Neural Networks
description: >-
  [ICML 2026][Reinforcement Learning][Uniform Facility Location] This paper designs an MPNN for Uniform Facility Location that neuralizes the classical approximation algorithm SimpleUniformFL. **It can be trained end-to-en…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Uniform Facility Location"
  - "MPNN"
  - "approximation guarantee"
  - "unsupervised"
  - "JL-style analysis"
date: 2026-05-08
content_hash: 97f50865348b730b
---

# Learning to Approximate Uniform Facility Location via Graph Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2602.13155](https://arxiv.org/abs/2602.13155)  
**Code**: Not mentioned  
**Area**: Neural Combinatorial Optimization / MPNN / Learning-based Approximation Algorithms  
**Keywords**: Uniform Facility Location, MPNN, approximation guarantee, unsupervised, JL-style analysis

## TL;DR
This paper designs an MPNN for Uniform Facility Location that neuralizes the classical approximation algorithm SimpleUniformFL. **It can be trained end-to-end using an unsupervised expected cost loss and possesses provable approximation bounds of $\mathcal{O}(\log n)$ (improving to $\mathcal{O}(1)$ in the recursive version).** Experiments demonstrate that it outperforms the classical SimpleUniformFL algorithm and approaches ILP optimality.

## Background & Motivation
**Background**: Recent works have focused on using MPNNs for end-to-end combinatorial optimization (CO-with-GNN, neural CO, karalias22, etc.), primarily following two paths: (1) treating MPNNs as end-to-end heuristics; (2) integrating MPNNs into classical exact solvers (e.g., branch-and-cut) as heuristics for cut or variable selection.

**Limitations of Prior Work**: (1) Supervised learning requires expensive optimal labels; the non-differentiability of discrete objectives necessitates proxy gradients like straight-through estimation (STE), Gumbel-softmax, I-MLE, or SIMPLE, which results in **fragile and difficult training**. (2) Most end-to-end neural methods **lack solution quality guarantees**, performing well on-distribution but collapsing on out-of-distribution (OOD) data. (3) Classical approximation algorithms have worst-case guarantees but are distribution-agnostic, failing to exploit structural regularities in real-world data.

**Key Challenge**: The trade-off between robust but conservative approximation algorithms versus expressive but fragile (and guarantee-less) learning-based solvers.

**Goal**: To develop an MPNN architecture for Uniform Facility Location (UniFL)—an NP-hard problem with clear local structure—that is **simultaneously (i) fully differentiable, (ii) unsupervised, (iii) provably bounded, and (iv) capable of mining data structures.**

**Key Insight**: UniFL possesses a well-known radius-based local structure (mettu2003online, Badoiu2005). Once the node radius $r_x$ (satisfying $\sum_{y \in B(x, r_x)} (r_x - d(y,x)) = 1$) is known, opening facilities with probability $\min(1, c \cdot \ln n \cdot r_x)$ yields an $\mathcal{O}(\log n)$ approximation. This "local computation + probabilistic opening" structure is naturally suited for message passing.

**Core Idea**: **The radius calculation and facility opening probabilities of SimpleUniformFL are formulated as ReLU MPNN layers.** By using an analytically integrable expected cost as an unsupervised loss, it is theoretically proven that parameters exist for the MPNN to reproduce the $\mathcal{O}(\log n)$ approximation. A recursive variant, RecursiveUniformFL, further achieves an $\mathcal{O}(1)$ approximation.

## Method

### Overall Architecture
UniFL encodes the metric space $(\mathcal{X}, d)$ into a weighted graph $G_S$ (keeping only edges where $d(u,v) \leq 1$). The MPNN workflow involves: (1) estimating the radius $\hat r_x$ for each point $x$ via local message passing; (2) mapping $\hat r_x$ to a facility opening probability $p_x$ using an FNN; (3) training end-to-end without supervision via an expected cost loss; (4) performing inference by independently sampling $F_1$ according to $p_x$, opening facilities $F_2$ for unserved nodes, and outputting $F = F_1 \cup F_2$. RecursiveUniformFL extends this across multiple rounds for nodes with sub-threshold probabilities to reach constant-factor approximations.

### Key Designs

1.  **Differentiable Radius Estimation**:
    - **Function**: Approximates the crucial UniFL radius $r_x$ using local message passing to enable backpropagation.
    - **Mechanism**: Discretizes $(0, 1]$ into bins $0 = a_0 < a_1 < \cdots < a_k = 1$. For each bin, an indicator $t_x^{(i)} = \min\{1, \sum_{y \in N(x)} \text{reLU}(a_i - d(x,y))\}$ is computed, which can be rewritten as a two-layer ReLU FNN $t_x^{(i)} = \text{FNN}_{2,3}(\sum_y \text{FNN}_{1,3}(a_i, d(x,y)))$. If $r_x \geq a_i$, then $t_x^{(i)}$ should be 1. The radius estimate is then $\hat r_x = \sum_i a_i (t_x^{(i-1)} - t_x^{(i)})$.
    - **Design Motivation**: The definition of radius involves accumulating $r_x - d(y,x) = 1$ within a ball, which is equivalent to a ReLU sum. Explicit construction allows the approximation bound to be rigorously mapped to the MPNN rather than relying on a black-box to "learn" it.

2.  **Facility Opening Probability and Analytical Expected Cost Loss**:
    - **Function**: Converts the combinatorial objective into a fully differentiable loss without noisy gradients like STE or Gumbel-softmax.
    - **Mechanism**: The opening probability is $p_x = \min\{1, c \log(n) \cdot \hat r_x\} \equiv \text{FNN}_{2,3}(n, \hat r_x)$. Based on the logic that $F_1$ is independently sampled and $F_2$ is opened for unserved points, the expected cost is expressed analytically: 
      $$\mathbb{E}[\text{cost}] = \sum_f p_f + \sum_f \prod_{x: d(x,f)<1}(1-p_x) + \sum_x \sum_{f: d(x,f)<1} d(x,f) \cdot p_f \prod_{z: d(x,z)<d(x,f)}(1-p_z)$$
      The components represent the expected facility opening and the expected connection cost. The complexity is $\mathcal{O}(nd^2)$ ($d$ is the maximum graph degree).
    - **Design Motivation**: All operations ($\min, \prod, \sum$) are differentiable with respect to $p_x$, **completely bypassing unstable discrete training schemes**. This also preserves semantic mapping to SimpleUniformFL for bounding proofs.

3.  **RecursiveUniformFL from $\mathcal{O}(\log n)$ to $\mathcal{O}(1)$**:
    - **Function**: Improves the $\log n$ factor of the simple algorithm to a constant.
    - **Mechanism**: Adjusts probabilities to $\min\{1, c \cdot d(x, F), c \cdot r_x\}$, adding a distance term to existing facilities. Each round assigns points served within $6 r_x$ of $f \in F$ and recurses on the remainder. Prop 3 proves the existence of parameters for the $\mathcal{O}(\log n)$ bound, and Prop 5 proves that parameters learned from finite sets generalize to any size $n$. Prop 4 provides a lower bound: **a standalone constant-depth MPNN cannot exceed $\Omega(\log n / 2)$ even with optimal parameters**, justifying the necessity of recursion.
    - **Design Motivation**: To bridge the $\log n$ gap in traditional SimpleUniformFL through iterative refinement, stripping away covered points and applying the same architecture to the remaining sub-problems.

### Loss & Training
The loss is the analytical expected cost formula mentioned above, which is purely unsupervised (no ILP optimal solutions required). During training, the MPNN acts as a $p_x$ generator. At inference, it follows the SimpleUniformFL post-processing (lines 4-6) to produce discrete solutions, reporting the average cost over 1000 samples. The framework can adapt to $k$-Means objectives by replacing distances with squared Euclidean distances in the final term. The constant $c$ is tuned via grid search.

## Key Experimental Results

### Main Results

| Candidate Method | Geo-1000-2 (Open) | Evaluation |
|---|---|---|
| ILP Solver (Optimal) | 366.302 | Upper bound reference |
| SimpleUniformFL (Baseline) | Higher than MPNN | Classical $\mathcal{O}(\log n)$ algorithm |
| $\mathcal{O}(1)$-UFL (Gehweiler et al.) | Medium | Tuning-free baseline |
| **MPNN (Ours)** | Close to ILP | Significantly beats SimpleUniformFL |
| KMeans++ / KMedoids++ (Fixed $k$) | Comparable or slightly worse | Fair comparison |

Consistent advantages are observed across 2/5/10 dimensions and sparse/dense geometric graphs.

### Ablation Study

| Setup | Key Metrics | Explanation |
|---|---|---|
| Train 1000 nodes → Test 2k-10k nodes | Stable approx ratio | Validates Prop 5 size generalization |
| Geo-1000-10-sparse vs dense | Both robust | Not dependent on specific density |
| Real city road maps (4 metros) | Beats classical baselines | Effective even if triangle inequality is violated |
| $k$-Means variant | Comparable to KMeans++ | Extensible framework by changing distance metric |

### Key Findings
- "The MPNN can not only simulate the $\mathcal{O}(\log n)$ bound of SimpleUniformFL but also **strictly outperform** it on-distribution through training"—this is clear evidence of the "approximation algorithm meets data distribution" synergy.
- Size generalization remains stable at 5-10x training scale, corresponding to the theoretical finite-set guarantees in Prop 5.
- The method works even on real road networks where the triangle inequality is violated, showing that the radius approach is resilient to non-strict metrics.
- While the $\mathcal{O}(1)$-UFL algorithm (Gehweiler) has better theoretical bounds, it lacks an analytical expectation and cannot be formulated as a differentiable loss. **Ours chooses SimpleUniformFL as a starting point specifically for differentiability.**

## Highlights & Insights
- Entirely "neuralizes" SimpleUniformFL—mapping radius, probability, and loss to ReLU FNN expressions—**making classical approximation algorithms the initialization for the neural network**, ensuring training only improves the result.
- The closed-form expected cost loss avoids the fragility of proxy gradients like STE/Gumbel and manages complexity at $\mathcal{O}(nd^2)$ via UniFL's sparse structure.
- Prop 4 explicitly proves the $\Omega(\log n)$ capacity limit for constant-depth MPNNs, providing a theoretical justification for recursion that is rare in neural CO literature.
- Generalizes size generalization results for finite training sets (Prop 5), alleviating common OOD concerns in neural CO.

## Limitations & Future Work
- Restricted to Uniform FL (constant facility costs); general FL, capacitated FL, and $k$-median require new expected cost derivations.
- The $\prod$ terms in the expected cost may face numerical underflow issues for large $n$; log-space summation may be required.
- Training relies on synthetic GMM geometric graphs; transferability to complex real-world distributions (e.g., e-commerce logistics demand) is not yet explored.
- Inference still requires sampling 1000 times for post-processing, which is slower than a single forward pass; deterministic rounding could be explored.

## Related Work & Insights
- **vs. End-to-end neural CO (karalias22, etc.)**: Those lack approximation bounds and OOD control; ours provides $\mathcal{O}(\log n)$ and recursive $\mathcal{O}(1)$ bounds.
- **vs. Branch-and-cut + GNN (Gas+2019)**: Those require running solvers thousands of times during training; ours is entirely unsupervised.
- **vs. Algorithms with predictions**: Those treat ML as a non-differentiable black box; ours integrates ML and the algorithm in a single differentiable pipeline.
- **vs. $\mathcal{O}(1)$-UFL (Gehweiler et al.)**: That algorithm has better bounds but lacks analytical expectation; ours sacrifices some theoretical ceiling for differentiability, then recovers constant factors via recursion.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of neuralizing classical algorithms, analytical expected loss, and rigorous upper/lower bounds is highly substantial.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered synthetic and real road networks, size generalization, and $k$-means variants.
- Writing Quality: ⭐⭐⭐⭐ Clear mapping between propositions and algorithms with compact notation.
- Value: ⭐⭐⭐⭐ Provides a "provable + learnable + generalizable" reference for the neural CO community that can be extended to other problems with local structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness](convergence_of_steepest_descent_and_adam_under_non-uniform_smoothness.md)
- [\[ICML 2026\] ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization](asap_exploiting_the_satisficing_generalization_edge_in_neural_combinatorial_opti.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](../../NeurIPS2025/reinforcement_learning/a_theory_of_multi-agent_generative_flow_networks.md)
- [\[NeurIPS 2025\] Distribution Learning Meets Graph Structure Sampling](../../NeurIPS2025/reinforcement_learning/distribution_learning_meets_graph_structure_sampling.md)

</div>

<!-- RELATED:END -->
