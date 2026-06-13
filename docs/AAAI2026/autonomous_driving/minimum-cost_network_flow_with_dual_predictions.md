---
title: >-
  [Paper Note] Minimum-Cost Network Flow with Dual Predictions
description: >-
  [AAAI 2026][Autonomous Driving][Minimum-cost flow] This paper presents the first learning-augmented algorithm for minimum-cost network flow (MCF) based on dual predictions. By warm-starting the classical ε-relaxation alg…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Minimum-cost flow"
  - "dual predictions"
  - "ε-relaxation"
  - "algorithms with predictions"
  - "chip routing"
  - "traffic networks"
date: 2026-05-08
content_hash: 61b1816ae699d617
---

# Minimum-Cost Network Flow with Dual Predictions

**Conference**: AAAI 2026
**arXiv**: [2601.20203](https://arxiv.org/abs/2601.20203)  
**Code**: Not available  
**Area**: Autonomous Driving / Combinatorial Optimization / Algorithms with Predictions
**Keywords**: Minimum-cost flow, dual predictions, ε-relaxation, algorithms with predictions, chip routing, traffic networks

## TL;DR
This paper presents the first learning-augmented algorithm for minimum-cost network flow (MCF) based on dual predictions. By warm-starting the classical ε-relaxation algorithm with machine-learned dual solutions, the proposed method ties its complexity bound to the $\ell_\infty$-norm of the prediction error (achieving both consistency and robustness), and demonstrates average speedups of 12.74× on traffic networks and 1.64× on chip escape routing.

## Background & Motivation

**Background**: MCF is a fundamental problem in combinatorial optimization, with broad applications in traffic network routing, EDA chip design, and computer vision (multi-object tracking). Classical solvers include Network Simplex, Successive Shortest Paths (SSP), and ε-relaxation.

**Limitations of Prior Work**: Solving large-scale MCF instances (millions of nodes/edges) is computationally expensive. Worst-case analysis is overly pessimistic—SSP theoretically requires exponential iterations, yet converges rapidly in practice.

**Rise of Algorithms with Predictions**: The "algorithms with predictions" framework has recently achieved results on matching (Dinitz et al. 2021), maximum flow (Davies et al. 2023), and minimum cut, yet MCF remains unexplored.

**Why ε-relaxation**: (a) It is a dual algorithm, so predictions require only $n$ variables (vs. $m$ for the primal); (b) it is naturally parallelizable (node iterations are decoupled); (c) its complexity analysis is tractable, unlike Network Simplex whose complexity is unknown.

**Core Idea**: Learn a predicted dual optimal solution $\hat{p}$ for MCF and warm-start ε-relaxation, reducing complexity from $O(n^3\log(nC))$ to $O(n^3\log\|\hat{p}-p^*\|_\infty)$.

## Method

### Overall Architecture
Given network $G=(V,E)$ (with $n$ nodes, $m$ edges, edge costs $a_{ij}$, capacities $[b_{ij}, c_{ij}]$, and node supplies $s_i$) and a dual prediction $\hat{p}$, the pipeline proceeds as follows: preprocessing (shift so that $\min_i \hat{p}_i=0$, then clip to $[0,(n-1)C]$) → warm-start ε-relaxation to solve for the optimal flow. A cost-scaling technique is incorporated for multi-scale iterative convergence.

The core idea is that ε-relaxation maintains a pair $(x,p)$ satisfying ε-complementary slackness and iteratively reduces each node's surplus $g_i$ to zero, yielding a feasible flow. The dual prediction provides a better initial $p$, reducing the number of iterations.

### Key Design 1: Warm-Start ε-Relaxation (Algorithm 1)
1. Normalize the prediction via dual shift invariance (Fact 1).
2. Clip to the valid range (Lemma 1: there exists an optimal dual solution in $[0,(n-1)C]$).
3. **Core Lemma (Lemma 2)**: $\beta(\hat{p}) \leq 2\|\hat{p}-p^*\|_\infty$, i.e., the upper bound on the reduced-cost length at warm-start is linearly related to the $\ell_\infty$-norm of the prediction error.
4. **Theorem 1** (vanilla version): complexity $O(\min\{n^3\|\hat{p}-p^*\|_\infty, n^4C\})$; for 0/1-flow instances, $O(\min\{mn\|\hat{p}-p^*\|_\infty, mn^2C\})$.

### Key Design 2: Cost-Scaling Version (Algorithm 2)
- Edge costs are scaled by $(n+1)/c^T$; iterations proceed from coarse to fine over $T=\log\|\hat{p}-p^*\|_\infty$ rounds.
- Each round of ε-relaxation costs $O(n^3)$, as the solution from the previous round provides a good initialization.
- **Theorem 2**: $O(\min\{n^3\log\|\hat{p}-p^*\|_\infty, n^3\log(nC)\})$.
- Robustness guarantee: degrades to the classical $O(n^3\log(nC))$ bound when predictions are completely wrong.

### Key Design 3: Two Prediction Learning Paradigms
- **Fixed predictions (traffic networks)**: The network topology is fixed while edge weights fluctuate randomly. An expected dual solution is learned by minimizing $\min_{\hat{p}} \frac{1}{k}\sum_i \log\|\hat{p}-p^{(i)}+\Delta_i\|_\infty$ (a convex optimization problem solved via Frank-Wolfe). PAC sample complexity is $\tilde{O}(n/\varepsilon^2)$ (Theorem 3).
- **Feature-based predictions (escape routing)**: A UNet-style CNN predicts dual solutions from 3-channel feature maps (obstacles, distance to pins, distance to boundary), trained with MSE loss. Sample complexity is $O(n \cdot d_{NN} \cdot \log(nC)/\varepsilon^2)$ (Theorem 4).

### Loss & Training
- Fixed predictions: minimize surrogate loss $\sum_i \log\|\hat{p}-p^{(i)}+\Delta_i\|_\infty$ (convex, solved via Frank-Wolfe).
- CNN predictions: MSE loss ($\ell_2$ norm), since $\ell_\infty$ training is difficult. The UNet architecture contains 2 downsampling and 2 upsampling modules, each with two 3×3 convolutional layers + GroupNorm and residual connections. Adam optimizer, lr=1e-3, reduced by 10× at epoch 15, batch size 16.
- The pseudo-dimension bounds for the end-to-end algorithm (Lemma 3/4) are proven theoretically, guaranteeing generalization.

## Key Experimental Results

### Main Results 1: Traffic Networks (DIMACS Road Network Benchmarks)

| Instance | Nodes | ε-R (w/ pred.) | ε-R (w/o pred.) | Network Simplex | SSP |
|----------|-------|----------------|----------------|----------------|-----|
| paths_04_NV | 261K | **1.50s** | 24.25s | 19.29s | 11.68s |
| flow_04_NV | 261K | **18.21s** | 112.93s | 20.22s | 51.21s |
| paths_05_WI | 519K | **3.88s** | 58.84s | 61.41s | 73.49s |
| flow_06_FL | 1049K | **45.95s** | 630.73s | 271.13s | 1769.58s |
| paths_07_TX | 2074K | **25.87s** | 323.85s | 694.12s | 2113.63s |
| flow_07_TX | 2074K | **238.41s** | 2247.67s | 1127.91s | very slow |
| **Avg. Speedup** | — | **1.00×** | 12.74× | 12.08× | 38.19× |

### Main Results 2: Chip Escape Routing

| Instance | Board Size | Pins | ε-R (w/ pred.) | ε-R (w/o pred.) | NS | SSP |
|----------|-----------|------|----------------|----------------|----|-----|
| 1 | 296×277 | 710 | **2.10s** | 3.30s | 11.75s | 11.55s |
| 5 | 635×572 | 1366 | **45.24s** | 70.18s | 453.92s | 280.03s |
| 10 | 1205×1374 | 1985 | **630.49s** | 955.19s | very slow | 2592.41s |
| **Avg. Speedup** | — | **1.00×** | 1.64× | 11.19× | 7.53× |

### Key Findings
1. **Most significant speedup on traffic networks**: Dual predictions accelerate ε-relaxation by 6.2–21.4× over the unpredicted version; on 0/1-flow instances (paths_*), it even surpasses Network Simplex.
2. **Strong cross-scale generalization of CNN**: A single CNN model generalizes well as board sizes range from 300×300 to 1000×1000.
3. **Theory–experiment consistency**: In synthetic experiments, runtime growth as a function of prediction error matches theoretical curves, and degrades to classical complexity under large errors.
4. **High parallelization potential**: Road networks are planar graphs (4-colorable), enabling an additional theoretical speedup of $n/4$.
5. **Cost-scaling step-by-step analysis**: Per-step running times decrease progressively (e.g., paths_04 from 2.96s to <0.4s), indicating the theoretical analysis is conservative.
6. **Fixed-prediction learning is simple and efficient**: Only 10 samples suffice to learn effective dual predictions (Gaussian-perturbed edge weights, solved via convex optimization).
7. **Fast CNN convergence**: Training on only 80 instances for 40 epochs achieves convergence, validating the structural regularity of the dual solution space.

## Highlights & Insights
- First learning-augmented algorithm for MCF with full theoretical guarantees (consistency + robustness + PAC sample complexity).
- Complexity tied to the $\ell_\infty$-norm (rather than the $\ell_0$-norm), so exact prediction of every component is unnecessary—more practical.
- Two learning paradigms (fixed predictions / CNN predictions) cover both fixed and varying topology settings.
- Validated on large-scale real road networks (2 million nodes) with 12.74× speedup.
- On 0/1-flow instances, prediction-augmented ε-relaxation outperforms Network Simplex—the most widely used MCF solver in practice.
- High theory–experiment agreement: runtime curves as a function of prediction error are consistent with theoretical bounds in synthetic experiments.

## Limitations & Future Work
1. **Cost-scaling parameter $T$ requires tuning**: Though theoretically estimable, this incurs additional overhead; experiments set $T=\hat{T}+2$ manually.
2. **CNN uses MSE rather than $\ell_\infty$ loss**: The gap between the surrogate loss and the theoretical bound is not quantified.
3. **Parallelization not implemented**: Although ε-relaxation is naturally parallelizable (road networks are 4-colorable), no parallel comparison is conducted.
4. **Integer assumption only**: Capacities and supplies are assumed to be integers; generalization to non-integer settings remains unexplored.
5. **Limited speedup on escape routing**: Only 1.64×, possibly due to insufficient CNN prediction accuracy or structural differences in the problem.
6. **No comparison with latest SOTA**: The nearly linear-time algorithm of Brand et al. (2023) is not discussed in terms of practical feasibility.

## Related Work & Insights
- **Classical MCF algorithms**: Network Simplex (practical SOTA, unknown complexity), SSP (simple but theoretically slow), capacity/cost scaling (Edmonds-Karp 1972, Goldberg-Tarjan 1987), Brand et al. (2023) near-linear theoretical upper bound.
- **Algorithms with predictions**: Dinitz et al. (2021) matching + dual predictions, Davies et al. (2023) max-flow + primal predictions, Chen et al. (2022) 0/1-flow $O(m^{3/2}n)$ bound ($\ell_0$-norm, impractical), Sakaue & Oki (2022) $O(mn^2\|\hat{p}-p^*\|_\infty)$ (weaker bound).
- **Data-driven algorithm design**: Gupta & Roughgarden (2017) framework, Balcan et al. (2021–2024) generalization guarantees, Cheng et al. (2024) branch-and-bound.

## Rating
⭐⭐⭐⭐ — Theoretically rigorous, with elegant consistency + robustness complexity bounds. Speedups are highly significant on traffic networks (12.74×) and moderate on escape routing (1.64×). The work extends the algorithms-with-predictions framework to MCF—a core optimization problem—combining theoretical depth with practical value. Note that while this paper is categorized under autonomous driving, it is more squarely in operations research / transportation optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions](../../ICCV2025/autonomous_driving/alocc_adaptive_lifting-based_3d_semantic_occupancy_and_cost_volume-based_flow_pr.md)
- [\[AAAI 2026\] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning](dual-branch_spatial-temporal_self-supervised_representation_for_enhanced_road_ne.md)
- [\[AAAI 2026\] Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification](debiased_dual-invariant_defense_for_adversarially_robust_person_re-identificatio.md)
- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[AAAI 2026\] AI-based Traffic Modeling for Network Security and Privacy: Challenges Ahead](ai-based_traffic_modeling_for_network_security_and_privacy_challenges_ahead.md)

</div>

<!-- RELATED:END -->
