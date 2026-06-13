---
title: >-
  [Paper Note] Learning to Approximate Uniform Facility Location via Graph Neural Networks
description: >-
  [ICML 2026][Reinforcement Learning][Uniform Facility Location] This work designs an MPNN that "neuralizes" the classic approximation algorithm SimpleUniformFL for Uniform Facility Location…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Uniform Facility Location"
  - "MPNN"
  - "approximation guarantee"
  - "unsupervised"
  - "JL-style analysis"
date: 2026-05-08
content_hash: ef4d4a968263bb71
---

# Learning to Approximate Uniform Facility Location via Graph Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2602.13155](https://arxiv.org/abs/2602.13155)  
**Code**: Not mentioned  
**Area**: Neural Combinatorial Optimization / MPNN / Learning-based Approximation Algorithms  
**Keywords**: Uniform Facility Location, MPNN, approximation guarantee, unsupervised, JL-style analysis

## TL;DR
This work designs an MPNN that "neuralizes" the classic approximation algorithm SimpleUniformFL for Uniform Facility Location, **enabling end-to-end training with unsupervised expected cost loss, while also providing a provable $\mathcal{O}(\log n)$ approximation guarantee (and $\mathcal{O}(1)$ for the recursive version)**. Experiments show it outperforms the classic SimpleUniformFL algorithm and approaches ILP optimality.

## Background & Motivation
**Background**: Recent years have seen many works using MPNNs for end-to-end learning of combinatorial optimization (CO-with-GNN, neural CO, karalias22, etc.), mainly in two directions: (1) using MPNNs as end-to-end heuristics; (2) plugging MPNNs into classic exact solvers (e.g., branch-and-cut) as cut/variable selection heuristics.

**Limitations of Prior Work**: (1) Supervised learning requires expensive optimal solution labels; discrete objectives are non-differentiable, leading to reliance on straight-through, Gumbel-softmax, I-MLE, SIMPLE, etc., which are **fragile and hard to tune**. (2) End-to-end neural methods **almost universally lack solution quality guarantees**—they perform well in-distribution but fail OOD. (3) Classic approximation algorithms have worst-case guarantees but are distribution-agnostic and cannot exploit structural patterns in real data.

**Key Challenge**: Robust but conservative approximation algorithms vs. expressive but fragile (and unguaranteed) learning-based solvers—their strengths are hard to combine.

**Goal**: For the classic NP-hard but locally structured Uniform Facility Location (UniFL) problem, design an MPNN architecture that **simultaneously achieves (i) full differentiability, (ii) unsupervised learning, (iii) provable approximation guarantee, and (iv) ability to exploit data structure**.

**Key Insight**: UniFL has a well-known radius-based local structure (mettu2003online, Badoiu2005)—once the node radius $r_x$ (satisfying $\sum_{y \in B(x, r_x)} (r_x - d(y,x)) = 1$) is known, simply opening facilities with probability $\min(1, c \cdot \ln n \cdot r_x)$ yields an $\mathcal{O}(\log n)$ approximation. This "local computation + probabilistic facility opening" structure is naturally suited for message passing.

**Core Idea**: **Express both the radius computation and facility opening probability of SimpleUniformFL as ReLU MPNN layers; use an analytically integrable expected cost as the unsupervised loss**. Theoretically, there exist parameters for the MPNN to recover the $\mathcal{O}(\log n)$ approximation, and the recursive version RecursiveUniformFL further achieves $\mathcal{O}(1)$ approximation.

## Method

### Overall Architecture
UniFL encodes the metric space $(\mathcal{X}, d)$ as a weighted graph $G_S$ (retaining only edges with $d(u,v) \leq 1$). The MPNN workflow: (1) Each point $x$ estimates its radius $\hat r_x$ via local message passing; (2) an FNN maps $\hat r_x$ to facility opening probability $p_x$; (3) the expected cost loss is used for end-to-end unsupervised training; (4) at inference, $p_x$ is used for independent sampling to obtain $F_1$, then facilities are opened for uncovered points to get $F_2$, and the final output is $F = F_1 \cup F_2$. The recursive extension RecursiveUniformFL applies multiple rounds on points with insufficient coverage, achieving constant-factor approximation.

### Key Designs

1. **Differentiable Radius Estimation**:

    - Function: Approximate the key UniFL quantity radius $r_x$ via local message passing, making it amenable to backpropagation.
    - Mechanism: Discretize $(0, 1]$ into $0 = a_0 < a_1 < \cdots < a_k = 1$. For each bin, compute the indicator $t_x^{(i)} = \min\{1, \sum_{y \in N(x)} \text{reLU}(a_i - d(x,y))\}$, which can be rewritten as a two-layer ReLU FNN $t_x^{(i)} = \text{FNN}_{2,3}(\sum_y \text{FNN}_{1,3}(a_i, d(x,y)))$. If $r_x \geq a_i$, then $t_x^{(i)}$ should be 1, so the radius estimate is $\hat r_x = \sum_i a_i (t_x^{(i-1)} - t_x^{(i)})$.
    - Design Motivation: The definition of radius is itself "accumulate $r_x - d(y,x) = 1$ within the ball", equivalent to a ReLU sum; the explicit construction allows the approximation bound to be strictly transferred to the MPNN, rather than relying on the network to "learn" it as a black box.

2. **Facility Opening Probability and Analytically Tractable Expected Cost Loss**:

    - Function: Reformulate the combinatorial objective as a fully differentiable loss, avoiding STE / Gumbel-softmax and other noisy gradients.
    - Mechanism: Facility opening probability $p_x = \min\{1, c \log(n) \cdot \hat r_x\} \equiv \text{FNN}_{2,3}(n, \hat r_x)$. Based on the algorithm logic of "$F_1$ independent sampling, $F_2$ opens when uncovered", the expected facility opening plus connection cost is written analytically: $\mathbb{E}[\text{cost}] = \sum_f p_f + \sum_f \prod_{x: d(x,f)<1}(1-p_x) + \sum_x \sum_{f: d(x,f)<1} d(x,f) \cdot p_f \prod_{z: d(x,z)<d(x,f)}(1-p_z)$. Here $A_f, B_f$ are indicators for "independent opening" and "forced opening when uncovered", and the third term is the expected distance to the nearest open facility. Complexity is $\mathcal{O}(nd^2)$ ($d$ is the maximum degree).
    - Design Motivation: All $\min, \prod, \sum$ are differentiable with respect to $p_x$, **completely bypassing the instability of discrete + STE training**; the formulation also matches the semantics of SimpleUniformFL, facilitating guarantee proofs.

3. **From $\mathcal{O}(\log n)$ to $\mathcal{O}(1)$ with RecursiveUniformFL**:

    - Function: Improve the simple algorithm's $\log n$ factor to a constant.
    - Mechanism: Modify the probability to $\min\{1, c \cdot d(x, F), c \cdot r_x\}$, adding a "distance to existing facilities" term; in each round, assign points served by $f \in F$ within $6 r_x$, and recurse on the remainder. Proposition 3 proves the existence of parameters for the MPNN to recover the $\mathcal{O}(\log n)$ bound, and Proposition 5 shows that parameters learned from a finite training set plus regularization generalize to instances of any size $n$. Proposition 4 provides a lower bound: **a standalone constant-depth MPNN, even with optimal parameters, can only achieve $\Omega(\log n / 2)$**, justifying the necessity of recursion.
    - Design Motivation: Address the $\log n$ gap in traditional SimpleUniformFL; iterative refinement peels off "covered points" so that the remaining subproblem can reuse the same architecture.

### Loss & Training
The loss is the analytical expected cost formula above, fully unsupervised (no need for ILP optimal solutions as supervision). During training, the MPNN acts as a $p_x$ generator; at inference, SimpleUniformFL post-processing (lines 4-6) is used to obtain discrete solutions, reporting the average cost over 1000 samples. To adapt for $k$-Means, simply replace the last term's distance with squared Euclidean distance. The constant $c$ is tuned via grid search.

## Key Experimental Results

### Main Results

| Candidate Method | Geo-1000-2 (Open) | Evaluation |
|---|---|---|
| ILP Solver (Optimal) | 366.302 | Upper bound reference |
| SimpleUniformFL (Baseline) | Higher than MPNN | $\mathcal{O}(\log n)$ classic algorithm |
| $\mathcal{O}(1)$-UFL (Gehweiler et al.) | Intermediate | Tuning-free baseline |
| **MPNN (Ours)** | Close to ILP | Significantly better than SimpleUniformFL, approaches ILP |
| KMeans++ / KMedoids++ (same $k$) | Comparable or slightly worse than MPNN | Fair comparison |

Consistent advantages across 2/5/10 dimensions and both sparse/dense geometric graphs.

### Ablation Study & Generalization

| Setting | Key Metric | Description |
|---|---|---|
| Train on 1000 nodes → Test on 2k-10k nodes | Stable approximation ratio | Verifies size generalization (Prop 5) |
| Geo-1000-10-sparse vs dense | Both robust | Not dependent on specific density |
| Real city road graphs (4 metros) | Significantly better than classic baselines | Works even when edge weights violate triangle inequality |
| $k$-Means variant | Comparable to KMeans++ | Just replace distance with squared distance in loss; framework is extensible |

### Key Findings
- "MPNN can not only simulate SimpleUniformFL's $\mathcal{O}(\log n)$ bound, but also **strictly outperform it** in-distribution through training"—a clear demonstration of "approximation algorithm meets data distribution".
- Size generalization remains stable at 5-10x training scale, matching the finite training set guarantee in Prop 5.
- Still works on real road networks that violate the triangle inequality, indicating the radius approach is flexible for general non-strict metrics.
- The classic $\mathcal{O}(1)$-UFL algorithm (Gehweiler) has a better theoretical bound but lacks an analytic expectation, so cannot be written as a differentiable loss; **MPNN is based on SimpleUniformFL precisely for differentiability**.

## Highlights & Insights
- Fully "neuralizes" SimpleUniformFL—radius, probability, and loss are all expressed as ReLU FNNs, **making the classic approximation algorithm a neural network initialization**; training can only improve results (never worsen them).
- The expected cost loss is fully closed-form, avoiding the fragility of STE/Gumbel surrogate gradients; also leverages UniFL's sparse structure to reduce complexity to $\mathcal{O}(nd^2)$.
- Proposition 4 explicitly proves "the capacity limit of constant-depth MPNNs is $\Omega(\log n)$", providing a lower bound justification for recursion—such "upper and lower bound dual proofs" are rare in neural CO literature.
- Generalizes size via finite training set results in Prop 5, alleviating the usual OOD anxiety in neural CO.

## Limitations & Future Work
- Only targets Uniform FL (uniform facility opening cost); general FL, capacitated FL, $k$-median, etc., require re-derivation of expected cost.
- The $\prod$ term in expected cost may underflow numerically for large $n$; log-space summation may be needed.
- Training uses synthetic GMM geometric graphs; transfer to more complex real distributions (e.g., real-world e-commerce logistics demand graphs) is not studied.
- Inference still requires SimpleUniformFL post-processing with 1000 samples, making it slower than a single forward pass; deterministic rounding could be explored.

## Related Work & Insights
- **vs end-to-end neural CO (karalias22, etc.)**: They lack approximation guarantees and are OOD-uncontrollable; this work provides $\mathcal{O}(\log n)$ / recursive $\mathcal{O}(1)$ guarantees.
- **vs branch-and-cut + GNN (Gas+2019)**: Those methods require running thousands of full solvers per training, making training costs prohibitive; this work is fully unsupervised.
- **vs algorithms with predictions**: They treat ML as a black box and are non-differentiable; here, ML and algorithms are in a single differentiable pipeline.
- **vs $\mathcal{O}(1)$-UFL (Gehweiler et al.)**: That algorithm has a better theoretical bound but lacks an analytic expectation, so cannot be directly trained; this work sacrifices some theoretical optimality for differentiability, then recovers the constant factor via recursion.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of "MPNN neuralizing classic approximation algorithm + analytic expected loss + strict upper and lower bounds" is highly rigorous
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic + real road networks + size generalization + $k$-means variant
- Writing Quality: ⭐⭐⭐⭐ Propositions and algorithms correspond clearly, formulas are compactly numbered
- Value: ⭐⭐⭐⭐ Provides the neural CO community with a "provable + learnable + generalizable" reference sample; ideas are transferable to other problems with radius/local structure

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness](convergence_of_steepest_descent_and_adam_under_non-uniform_smoothness.md)
- [\[ICML 2026\] ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization](asap_exploiting_the_satisficing_generalization_edge_in_neural_combinatorial_opti.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)
- [\[NeurIPS 2025\] Distribution Learning Meets Graph Structure Sampling](../../NeurIPS2025/reinforcement_learning/distribution_learning_meets_graph_structure_sampling.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](../../NeurIPS2025/reinforcement_learning/a_theory_of_multi-agent_generative_flow_networks.md)

</div>

<!-- RELATED:END -->
