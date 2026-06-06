---
title: >-
  [Paper Note] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts
description: >-
  [ICML 2026][Optimization][Linear Assignment] A lightweight network is trained to predict dual variables $\hat{u}$ for the linear assignment problem (LAP). Using the Min-Trick…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Linear Assignment"
  - "Dual Variables"
  - "Warm Start"
  - "LAPJV"
  - "Row-Independent Network"
date: 2026-05-08
content_hash: 21b70dc05835f493
---

# Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts

**Conference**: ICML 2026  
**arXiv**: [2605.09382](https://arxiv.org/abs/2605.09382)  
**Code**: None  
**Area**: Combinatorial Optimization / Learning-Augmented Algorithms / Multi-Object Tracking  
**Keywords**: Linear Assignment, Dual Variables, Warm Start, LAPJV, Row-Independent Network

## TL;DR
A lightweight network is trained to predict dual variables $\hat{u}$ for the linear assignment problem (LAP). Using the Min-Trick, feasible duals $\hat{v}$ are constructed and used as a warm start for the LAPJV exact solver, achieving over $2\times$ end-to-end speedup on $N=16{,}384$ instances while maintaining optimality.

## Background & Motivation

**Background**: The linear assignment problem (LAP) is a fundamental primitive for matching problems, widely used in multi-object tracking (MOT), scheduling, transportation, and more. Two mainstream solver families exist: classical exact solvers like Hungarian/Jonker-Volgenant (LAPJV), which guarantee optimality but have worst-case complexity $\mathcal{O}(N^3)$; and recent GNN-based neural solvers, which trade off exactness for speed.

**Limitations of Prior Work**: For $N \geq 10^3$, the cubic complexity of LAPJV dominates real-time system latency. Neural alternatives may violate hard assignment constraints and are limited by $\mathcal{O}(N^2 H)$ memory for edge features, restricting them to $N \approx 2{,}000$ and thus unsuitable for truly large-scale applications.

**Key Challenge**: There is a tension between exactness and scalability—incorporating neural networks into the solving process either breaks optimality or exhausts memory; omitting them entirely leaves the $\mathcal{O}(N^3)$ search as a bottleneck.

**Goal**: Achieve globally optimal solutions at industrial scale ($N=16{,}384$), with end-to-end runtime outperforming LAPJV cold start, robust to distribution shift, and ideally capable of zero-shot transfer to real data.

**Key Insight**: Leveraging LP duality, providing LAPJV with near-optimal feasible dual variables is equivalent to "restoring" the dual-ascent algorithm to a near-converged state, requiring only a few augmenting paths to complete the search. Thus, the neural network's role is not to replace the solver, but to predict good initial duals.

**Core Idea**: Use a row-independent RowDualNet to predict row potentials $\hat{u}$, then construct column potentials $\hat{v}$ via Min-Trick $\hat{v}_j = \min_i (C_{ij} - \hat{u}_i)$ to ensure dual feasibility. A lightweight threshold-based fallback ensures that even poor predictions only degrade to cold start.

## Method

### Overall Architecture
Input is an $N \times N$ cost matrix $C$, output is the exact optimal assignment matrix $X^\ast$. The pipeline has four stages: (1) Row feature extraction—compress $C$ into $F \in \mathbb{R}^{N \times D}$, retaining per-row statistics such as minimum, mean, entropy, rank, with $D \ll N$; (2) RowDualNet independently predicts a scalar $\hat{u}_i$ per row on GPU, with a sparse k-NN refinement step to capture inter-row competition; (3) Min-Trick computes $\hat{v}$ in blocks on GPU, and uses the equality subgraph density $\rho$ to decide on fallback; (4) $(\hat{u}, \hat{v}, C)$ are injected into a modified LAPJV C++ implementation on CPU, skipping early reduction and proceeding directly to exact solution.

### Key Designs

1. **RowDualNet Row-Independent Architecture + Sparse k-NN Refinement**:

    - **Function**: Maps $D$-dimensional row statistics to row potentials $\hat{u}_i$ via a row-level MLP, with a lightweight k-NN step injecting inter-row competition signals.
    - **Mechanism**: A shared MLP provides initial estimates $\hat{u}_{init,i}$; for each row, "pseudo reduced costs" $C_{ij} - \hat{u}_{init,i}$ are computed, and the $k$ smallest columns are aggregated. Since optimal assignment selects only one column per row, competition signals are concentrated among columns with reduced cost near zero, so fixed $k \ll N$ suffices.
    - **Design Motivation**: Avoids the $\mathcal{O}(N^2 H)$ memory blowup of GNNs, making memory scale linearly with $N$ and enabling training/inference up to $N=16{,}384$; k-NN compensates for the global information loss of row-independence.

2. **Min-Trick for Feasible Dual Construction**:

    - **Function**: Instantly extends RowDualNet's $\hat{u}$ to a feasible dual pair $(\hat{u}, \hat{v})$.
    - **Mechanism**: Sets $\hat{v}_j = \min_i (C_{ij} - \hat{u}_i)$, which by construction satisfies $\hat{u}_i + \hat{v}_j \leq C_{ij}$; this is $\mathcal{O}(N^2)$ but fully parallel and nearly free on GPU.
    - **Design Motivation**: Replaces iterative "learn infeasible → project to feasible" routines, which are slow and negate neural speedups. Min-Trick makes feasibility a definition, ensuring LAPJV optimality is independent of network prediction quality.

3. **Fallback Mechanism Based on Equality Subgraph Density**:

    - **Function**: Uses a scalar $\rho$ to judge neural prediction quality; otherwise, discards the prediction and reverts to standard cold start.
    - **Mechanism**: Defines $\rho = \frac{1}{N} \sum_{i,j} \mathbb{I}(|C_{ij} - \hat{u}_i - \hat{v}_j| < \epsilon)$, i.e., the average degree of edges with reduced cost near zero; if $\rho < \tau$, the neural seed is deemed "bad" and LAPJV uses column reduction heuristic initialization.
    - **Design Motivation**: Neural networks may occasionally output very sparse equality subgraphs, slowing LAPJV; fallback ensures worst-case runtime matches the baseline, as $T_{\text{overhead}} = \mathcal{O}(N^2 \log N)$ is negligible compared to $\mathcal{O}(N^3)$, providing strict asynchronous safety.

### Loss & Training
Supervised training uses LAPJV to generate ground-truth duals $u^\ast$ offline. The main loss is MAE between $\hat{u}$ and $u^\ast$, plus a Complementary Slackness regularization term—forcing reduced costs on ground-truth optimal edges to be close to zero, guiding dual prediction to "just include the optimal assignment in the equality subgraph." Multi-scale training is used: a single model is trained on $N \in \{512, 1536, 2048, 3072\}$ with over 1700 matrices, then zero-shot evaluated on $N=16{,}384$ for out-of-distribution generalization.

## Key Experimental Results

### Main Results

| Dataset | Scale | Speedup (vs SciPy) | Speedup (vs LAP) | Optimality |
|---------|-------|--------------------|------------------|------------|
| Dense Uniform Synthetic | $N=16384$ | $\approx 2.0\times$ | $\approx 2.5\times$ | 0% gap |
| Block-Structured Synthetic | $N=16384$ | $\approx 2.25\times$ | $\approx 4.0\times$ | 0% gap |
| MOT (Real) | $N \geq 8000$ | $\approx 2\times$ | $\approx 1.25\times$ | 0% gap |
| OSM Seven Cities | $N=10000$ | $1.4\text{-}1.6\times$ | $1.3\text{-}1.8\times$ | 0% gap |

### Ablation Study

| Configuration | Behavior | Notes |
|---------------|----------|-------|
| RowDualNet (full) | $\approx 76\%$ assignments solved in greedy phase | Augmenting path search reduced by $\approx 68\%$ |
| Cold Start LAPJV | Only $\approx 26\%$ solved greedily | Requires many shortest path searches |
| Linear Regression instead of RowDualNet | Slower than baseline for $N>4096$ | Validates necessity of nonlinear feature learning |
| Row Mean / Random Heuristic | speedup $<1$ | Simple statistics fail to capture competition structure |

### Key Findings
- True speedup comes not from GPU offloading, but from increased "equality subgraph density": neural seeds bring LAPJV close to convergence, skipping the expensive price war phase.
- End-to-end runtime stability improves significantly—coefficient of variation drops from $\approx 45\%$ to $\approx 30\%$, and worst/best ratio is suppressed from $11\times$ baseline, which is crucial for real-time safety systems.
- For $N=16{,}384$, the neural component accounts for $<7\%$ of total time, with the remaining 93% in the CPU exact solver, indicating genuine algorithmic acceleration rather than hardware tricks.

## Highlights & Insights
- Treating neural networks as "solver accelerators" rather than "solver replacements" realizes the "learned duals" theory of Dinitz et al. 2021 at system scale: Min-Trick builds feasibility into the construction, bypassing all projection algorithms.
- The row-independent + sparse k-NN design exemplifies the philosophy that "seeing less enables scaling"—many GNN tasks incur $\mathcal{O}(N^2)$ due to fully connected message passing, but if key signals are sparse, explicitly modeling top-$k$ yields orders-of-magnitude memory savings.
- The fallback mechanism provides strict asymptotic safety, making the "learning-augmented + worst-case safety" paradigm highly practical for industrial deployment—a graceful engineering extension of the ALPS framework by Dinitz et al.

## Limitations & Future Work
- Only addresses square dense LAP; extensions to rectangular, sparse LAP, or quadratic assignment (QAP) are not discussed.
- Training ground-truths require offline LAPJV runs, making data generation costly for very large $N$; semi- or self-supervised training for RowDualNet would improve scalability.
- For small scales ($N=512$), neural overhead dominates and slows overall runtime—suggesting adaptive use of neural seeds is a valuable direction.
- Speedup on MOT ($1.25\times$) is much lower than on dense synthetic data, attributed to matrix sparsity; sparse neural predictors for sparse LAP remain unexplored.

## Related Work & Insights
- **vs Dinitz et al. 2021 "learned duals"**: They proposed the theoretical framework but used projection for feasibility, which was slow; this work uses Min-Trick to construct feasibility, scaling the approach to $N=16{,}384$ for the first time.
- **vs GNN Neural Matching (Liu 2024 / Aironi 2024)**: They directly predict the assignment matrix, sacrificing optimality and limited by $\mathcal{O}(N^2)$ memory; this work insists on exact solutions and reduces memory to $\mathcal{O}(N)$ via row-independence.
- **vs SciPy / LAP library LAPJV**: This work adds a warm-start layer atop these solvers, with advantages growing with $N$ and 0% optimality gap fully verifiable.

## Rating
- Novelty: ⭐⭐⭐⭐ Scales learned duals to industrial size, Min-Trick elegantly solves feasibility
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + MOT + OSM seven cities, covering up to $N=16384$
- Writing Quality: ⭐⭐⭐⭐ Theoretical guarantees and engineering details are clearly explained
- Value: ⭐⭐⭐⭐ Directly serves real-time tracking / large-scale scheduling, with clear engineering impact

> Note: Although this paper is in the video_understanding directory, it fundamentally belongs to combinatorial optimization, with MOT tracking as one application. Consider moving it to `optimization` / `other` for better alignment in the future.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICLR 2026\] Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](../../ICLR2026/optimization/dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[ICML 2026\] RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs](rl-sph_learning_to_achieve_feasible_solutions_for_integer_linear_programs.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)

</div>

<!-- RELATED:END -->
