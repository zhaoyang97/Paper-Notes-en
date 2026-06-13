---
title: >-
  [Paper Note] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts
description: >-
  [ICML 2026][Optimization][Linear Assignment] Training a lightweight network to predict dual variables $\hat{u}$ for the Linear Assignment Problem (LAP), constructing feasible duals $\hat{v}$ via the "Min-Trick…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Linear Assignment"
  - "Dual Variables"
  - "Warm-start"
  - "LAPJV"
  - "Row-independent Network"
date: 2026-05-08
content_hash: b105c936f0a81f74
---

# Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts

**Conference**: ICML 2026  
**arXiv**: [2605.09382](https://arxiv.org/abs/2605.09382)  
**Code**: None  
**Area**: Combinatorial Optimization / Learning-Augmented Algorithms / Multi-Object Tracking  
**Keywords**: Linear Assignment, Dual Variables, Warm-start, LAPJV, Row-independent Network

## TL;DR
Training a lightweight network to predict dual variables $\hat{u}$ for the Linear Assignment Problem (LAP), constructing feasible duals $\hat{v}$ via the "Min-Trick," and using them as a warm-start for the LAPJV exact solver. This achieves over $2\times$ end-to-end acceleration for instances of scale $N=16{,}384$ while maintaining optimality.

## Background & Motivation

**Background**: The Linear Assignment Problem (LAP) is a fundamental primitive for matching problems, frequently invoked in multi-object tracking (MOT), scheduling, and transportation. Current mainstream solvers follow either exact or learning-based paths: classical Hungarian / Jonker-Volgenant (LAPJV) provides provably optimal solutions but with a worst-case complexity of $\mathcal{O}(N^3)$; recent GNN-based neural solvers sacrifice precision for speed.

**Limitations of Prior Work**: For $N \geq 10^3$, the cubic complexity of LAPJV dominates the latency of real-time systems. Meanwhile, neural alternatives may violate hard assignment constraints and suffer from $\mathcal{O}(N^2 H)$ memory bottlenecks due to edge features, limiting them to scales of $N \approx 2{,}000$, which fails to serve truly large-scale applications.

**Key Challenge**: There is a tension between precision and scalability. Integrating neural networks often either breaks optimality or exhausts GPU memory. Conversely, traditional methods without neural components are consumed by the $\mathcal{O}(N^3)$ search budget.

**Goal**: To guarantee global optimal solutions at industrial scales (up to $N=16{,}384$), outperform LAPJV cold-starts in end-to-end wall-clock time, and remain robust to distribution shifts with zero-shot transfer capabilities.

**Key Insight**: Leveraging a fact from LP duality theory: providing LAPJV with a set of near-optimal feasible dual variables is equivalent to "restoring" the dual-ascent algorithm to a state near convergence. Only a few augmenting paths are then required to complete the search. Thus, the neural network acts as a predictor for high-quality initial duals rather than a replacement for the solver.

**Core Idea**: Use a row-independent RowDualNet to predict row potentials $\hat{u}$, and construct column potentials via the Min-Trick $\hat{v}_j = \min_i (C_{ij} - \hat{u}_i)$ to ensure dual feasibility. A lightweight threshold-based fallback mechanism is employed, ensuring that poor predictions merely revert to a cold-start.

## Method

### Overall Architecture
The input is an $N \times N$ cost matrix $C$, and the output is the exact optimal assignment matrix $X^\ast$. The pipeline consists of four stages: (1) Row feature extraction—compressing $C$ into $F \in \mathbb{R}^{N \times D}$ using $D \ll N$ statistics per row (min, mean, entropy, rank, etc.); (2) RowDualNet predicts a scalar $\hat{u}_i$ for each row independently on GPU, followed by a sparse k-NN refinement to capture inter-row competition; (3) Min-Trick calculates $\hat{v}$ in blocks on GPU and evaluates a fallback based on equality subgraph density $\rho$; (4) Injection of $(\hat{u}, \hat{v}, C)$ into a modified C++ implementation of LAPJV to skip early reductions and proceed directly to exact refinement.

### Key Designs

1.  **RowDualNet (Row-Independent + Sparse k-NN Refinement)**:
    - **Function**: Maps $D$-dimensional row statistics to row potentials $\hat{u}_i$ and injects competition signals via lightweight k-NN.
    - **Mechanism**: A shared MLP generates initial estimates $\hat{u}_{init,i}$. Then, "pseudo reduced costs" $C_{ij} - \hat{u}_{init,i}$ are calculated to select the $k$ smallest columns for aggregation. Since optimal assignments involve only one column per row, competition signals concentrate on columns where reduced costs are near zero, making fixed $k \ll N$ sufficient.
    - **Design Motivation**: Avoids the $\mathcal{O}(N^2 H)$ memory explosion of GNNs, ensuring memory grows linearly with $N$ to scale up to $N=16{,}384$. The k-NN compensates for the lack of global receptive fields in independent row processing.

2.  **Min-Trick for Feasible Dual Construction**:
    - **Function**: Extends RowDualNet output $\hat{u}$ to a dual-feasible pair $(\hat{u}, \hat{v})$.
    - **Mechanism**: Setting $\hat{v}_j = \min_i (C_{ij} - \hat{u}_i)$ ensures $\hat{u}_i + \hat{v}_j \leq C_{ij}$ by construction. This step is $\mathcal{O}(N^2)$ but fully parallelizable and nearly free on GPU.
    - **Design Motivation**: Replaces iterative "predict-then-project" routines that are slow and negate the time saved by neural inference. Min-Trick builds feasibility into the definition, ensuring LAPJV optimality is independent of prediction quality.

3.  **Fallback Mechanism based on Equality Subgraph Density**:
    - **Function**: Uses a scalar $\rho$ to determine if the neural prediction is viable; otherwise, it discards the prediction and reverts to cold-start.
    - **Mechanism**: Defined as $\rho = \frac{1}{N} \sum_{i,j} \mathbb{I}(|C_{ij} - \hat{u}_i - \hat{v}_j| < \epsilon)$, representing the average degree of edges with reduced costs near zero. If $\rho < \tau$, the neural seed is deemed "bad," and LAPJV defaults to standard column reduction heuristics.
    - **Design Motivation**: Prevents sparse equality subgraphs from slowing down LAPJV. This ensures the worst-case runtime is effectively the baseline, as the overhead $T_{\text{overhead}} = \mathcal{O}(N^2 \log N)$ is asymptotically negligible compared to $\mathcal{O}(N^3)$.

### Loss & Training
Supervised training using ground-truth duals $u^\ast$ from offline LAPJV execution. The primary loss is the MAE between $\hat{u}$ and $u^\ast$, supplemented by a complementary slackness regularization term. This term forces reduced costs on true optimal edges toward zero, guiding predictions to expose the optimal assignment within the equality subgraph. Multi-scale training was applied on $N \in \{512, 1536, 2048, 3072\}$, with zero-shot evaluation up to $N=16{,}384$.

## Key Experimental Results

### Main Results

| Dataset | Scale | Speedup (vs SciPy) | Speedup (vs LAP) | Optimality |
|---------|-------|--------------------|------------------|------------|
| Dense Uniform (Synth) | $N=16384$ | $\approx 2.0\times$ | $\approx 2.5\times$ | 0% gap |
| Block-Structured (Synth) | $N=16384$ | $\approx 2.25\times$ | $\approx 4.0\times$ | 0% gap |
| MOT (Real-world) | $N \geq 8000$ | $\approx 2\times$ | $\approx 1.25\times$ | 0% gap |
| OSM 7-Cities | $N=10000$ | $1.4\text{-}1.6\times$ | $1.3\text{-}1.8\times$ | 0% gap |

### Ablation Study

| Configuration | Behavior | Description |
|---------------|----------|-------------|
| RowDualNet (full) | $\approx 76\%$ of assignments solved in greedy phase | Augmenting path search reduced by $\approx 68\%$ |
| Cold-start LAPJV | Only $\approx 26\%$ solved greedily | Requires extensive shortest-path searches |
| Linear Regression replacement | Slower than baseline for $N > 4096$ | Validates the need for non-linear feature learning |
| Row Mean / Random Heuristic | speedup $<1$ | Simple statistics fail to capture competitive structure |

### Key Findings
- Acceleration stems from increased "equality subgraph density": neural seeds allow LAPJV to bypass the expensive initial price-war phase.
- End-to-end runtime stability improved significantly—the coefficient of variation dropped from $\approx 45\%$ to $\approx 30\%$, suppressing the worst-case/best-case ratio (which is $11\times$ in baseline), a critical factor for real-time safety systems.
- At $N=16{,}384$, the neural component accounts for $<7\%$ of total time, with 93% spent in the exact solver, indicating true algorithmic acceleration rather than mere hardware offloading.

## Highlights & Insights
- Framing the neural network as a "solver accelerator" rather than a "solver replacement" realizes the theoretical "learned duals" path (Dinitz et al. 2021) at a system level. The Min-Trick bypasses the need for costly projection algorithms.
- The philosophy of "scaling by looking at less" (row-independence + sparse k-NN) is noteworthy. While many GNNs suffer from $\mathcal{O}(N^2)$ message passing, explicitly modeling top-$k$ critical signals yields massive memory savings.
- The fallback mechanism provides strict asymptotic safety, making the "learning-augmented + worst-case safety" paradigm highly suitable for industrial deployment.

## Limitations & Future Work
- Focuses only on square dense LAPs; extensions to rectangular, sparse, or Quadratic Assignment Problems (QAPs) are not discussed.
- Training labels depend on offline LAPJV, which is expensive for massive $N$; semi-supervised or self-supervised training for RowDualNet could improve scalability.
- For small scales ($N=512$), neural overhead negates gains, suggesting that adaptively deciding whether to use a neural seed is a valuable direction.
- Speedup on MOT ($1.25\times$) is lower than on dense synthetic data, likely due to matrix sparsity, which was not explicitly addressed by the RowDualNet architecture.

## Related Work & Insights
- **vs Dinitz et al. 2021 "learned duals"**: While they proposed the theoretical framework using projection-based feasibility, this work uses Min-Trick to scale to $N=16{,}384$ for the first time.
- **vs Neural Matching GNNs (Liu 2024 / Aironi 2024)**: These methods abandon optimality and are constrained by $\mathcal{O}(N^2)$ memory. This work maintains exactness and reduces memory to $\mathcal{O}(N)$.
- **vs SciPy / LAP libraries**: This work acts as a warm-start layer on top of existing solvers, meaning advantages scale with $N$ and maintain a 0% optimality gap.

## Rating
- Novelty: ⭐⭐⭐⭐ Industrial scaling of learned duals with Min-Trick.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic, MOT, and real-world map data up to $N=16384$.
- Writing Quality: ⭐⭐⭐⭐ Clear explanation of theoretical guarantees and engineering details.
- Value: ⭐⭐⭐⭐ Significant utility for real-time tracking and large-scale scheduling.

> Note: Although categorized under `video_understanding`, this is fundamentally a combinatorial optimization paper with MOT as a primary application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICLR 2026\] Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](../../ICLR2026/optimization/dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)

</div>

<!-- RELATED:END -->
