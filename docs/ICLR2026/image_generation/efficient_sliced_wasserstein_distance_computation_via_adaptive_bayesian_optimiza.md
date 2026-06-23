---
title: >-
  [Paper Note] Efficient Sliced Wasserstein Distance Computation via Adaptive Bayesian Optimization
description: >-
  [ICLR 2026][Image Generation][Sliced Wasserstein] This paper transforms the "projection direction selection" in Sliced Wasserstein distance from fixed low-discrepancy sampling into a learnable Bayesian Optimization process. It proposes four plug-and-play strategies (BOSW/RBOSW/ABOSW/ARBOSW), achieving or approaching SOTA in multiple SW-in-the-loop tasks without modify
tags:
  - ICLR 2026
  - Image Generation
  - Sliced Wasserstein
  - Bayesian Optimization
  - QSW
date: 2026-05-08
content_hash: c98e5a80b08db24f
---
# Efficient Sliced Wasserstein Distance Computation via Adaptive Bayesian Optimization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=6IZAOTfXUT](https://openreview.net/forum?id=6IZAOTfXUT)  
**Code**: The paper claims to provide reproducible code and fixed random seeds (specific repository links are in the supplementary materials)  
**Area**: image generation / optimal transport  
**Keywords**: Sliced Wasserstein, Bayesian Optimization, QSW, direction sampling, optimal transport  

## TL;DR
This paper transforms the "projection direction selection" in Sliced Wasserstein distance from fixed low-discrepancy sampling into a learnable Bayesian Optimization process. It proposes four plug-and-play strategies (BOSW/RBOSW/ABOSW/ARBOSW), achieving or approaching SOTA in multiple SW-in-the-loop tasks without modifying downstream losses or gradient formulations.

## Background & Motivation
**Background**: Sliced Wasserstein (SW) reduces complexity by projecting high-dimensional optimal transport onto 1D. For discrete distributions, the primary overhead of a single 1D Wasserstein distance is sorting, with a total cost of approximately $O(n\log n)$, making it more suitable than high-dimensional WD for tasks such as generative modeling, registration, and gradient flows. The current mainstream approach is to average over $L$ given directions.

**Limitations of Prior Work**: The quality of the direction set $\Theta_L$ directly determines the SW estimation error. Pure MC error converges at $O(L^{-1/2})$ relative to sample size; while QSW/RQSW improve "uniform coverage" using low-discrepancy sequences, they remain inherently data-independent geometric samplings that fail to utilize observed slice cost information within the current task.

**Key Challenge**: The integral domain of SW is the entire sphere, but "informative" directions often concentrate in local structural regions. Under a fixed budget, pursuing global uniform coverage wastes many slices on low-contribution directions; furthermore, as distributions change during optimization iterations, the optimal set of directions should also be updated over time.

**Goal**: The authors decompose the problem into two sub-objectives:
1. Find more "task-relevant" direction sets to accelerate convergence under a fixed slice budget $L$.
2. Allow for lightweight re-learning of the direction set during repeated iterations of SW optimization while maintaining compatibility with existing QSW pipelines.

**Key Insight**: The authors treat $f(\theta;\mu,\nu)=W_p^p(\theta_\sharp\mu,\theta_\sharp\nu)$ as a black-box function defined on the unit sphere. They use Bayesian Optimization (BO) with GP + UCB to adaptively select directions based on "observed slice costs" rather than relying solely on geometric uniformity.

**Core Idea**: Use BO to learn "which projection directions are worth evaluating" and implement this learner as a drop-in direction selection module. This can be combined with QSW seed directions (ABOSW/ARBOSW) to achieve a compromise between "low-discrepancy coverage" and "task adaptability."

## Method

### Overall Architecture
The paper returns to the definition of SW:
$$
SW_p^p(\mu,\nu)=\mathbb{E}_{\theta\sim U(S^{d-1})}\left[W_p^p(\theta_\sharp\mu,\theta_\sharp\nu)\right],
$$
In practice, it is approximated using $L$ directions:
$$
\widehat{SW}_p^p(\mu,\nu;\Theta_L)=\frac{1}{L}\sum_{\ell=1}^{L}W_p^p\big((\theta_\ell)_\sharp\mu,(\theta_\ell)_\sharp\nu\big).
$$
The key is not to change the SW loss itself, but rather how $\Theta_L$ is selected. The authors denote the 1D cost corresponding to each direction as $f(\theta)$, then perform BO on the sphere to output a set of directions for the original SW estimator and downstream optimizer.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
	A[Input distributions μ, ν] --> B[Evaluate slice costs f(θ) for existing directions]
	B --> C[Spherical GP surrogate modeling]
	C --> D[Select new directions via UCB acquisition function]
	D --> E[Deduplication & batch update of direction set]
	E --> F[Compute SW estimate & drive downstream optimization]
	F --> G{Periodic refresh?}
	G -->|Yes| B
	G -->|No| H[Maintain direction set & continue iteration]
```

### Key Designs
**1. Spherical Kernel GP: Modeling "Slice Cost Topography" in Directional Space**

The core of BO is posterior modeling of $f(\theta)$. The authors use an RBF kernel based on spherical angular distance:
$$
k(\theta,\theta')=\exp\left(-\frac{1}{2}\left(\frac{d_S(\theta,\theta')}{\ell}\right)^2\right),\quad
d_S(\theta,\theta')=\arccos\langle \theta,\theta'\rangle.
$$
The length scale $\ell$ is set heuristically using the median spherical distance between current sample pairs. The significance is that slice costs of nearby directions share statistical correlation, allowing BO to infer "unsampled but potentially high-value" directional regions.

**2. UCB Batch Proposal + Near-duplicate Suppression: Adaptive Exploration with Controlled Overhead**

In each BO round, a candidate pool is uniformly sampled on the sphere (default $n_c=4096$), then scored using $\alpha_{\text{UCB}}(\theta)=\mu(\theta)+\beta\sigma(\theta)$ (default $\beta=0.7$) to select a batch of directions (default $b=5$). To avoid clustering, a cosine similarity threshold (>0.98 for exclusion) is applied to suppress near-duplicates. This design keeps the additional overhead per round at $O(n_c n_t)$, with only $b$ new ground-truth evaluation points, maintaining overhead within an "acceptable lightweight" range.

**3. Four BO Variants: One-time Learning, Periodic Refresh, QSW Hybrid, and Restart Hybrid**

The paper provides four swappable selectors:
- BOSW: One-time BO to select $L$ directions, fixed thereafter.
- RBOSW: Re-run BOSW every $R$ steps (experimentally $R=25$).
- ABOSW: Use a strong QSW set as a seed, followed by minor BO fine-tuning (experimentally $r=2, b=5$, replacing at most 10% of directions).
- ARBOSW: Periodically restart ABOSW, re-seeding from QSW followed by short-course BO each time.

These designs correspond to different task types: if distributions change rapidly, refreshing/restarting is beneficial; if the training distribution is stable, fixed or slightly fine-tuned directions are usually steadier.

**4. Combination Strategy with QSW/RQSW: Change Direction Generator, Not Objective**

The BO module is "plug-and-play": it requires no changes to the SW objective, backpropagation formulas, or optimizer hyperparameters. It only replaces the "direction sampler" layer. ABOSW/ARBOSW represent a combination of QSW (low-discrepancy geometric prior) and BO (task adaptation).

The authors clarify that BO-guided direction sets typically introduce bias, so they do not pursue "unbiased SW estimation under finite samples"; if a task strictly depends on unbiased stochastic gradients, one can continue using the RQSW mechanism with BO as an optional refinement.

### Mechanism
Using the point cloud gradient flow task as an example with budget $L=100$:
1. Initialize the direction set (e.g., CQSW seeds) and compute the current 1D OT cost for each direction.
2. ABOSW trains a spherical GP with these observations and finds 5 new directions with the highest UCB values among 4096 candidates.
3. Replace the 5 lowest-contributing directions in the set; after 2 rounds, a maximum of 10 directions are replaced.
4. Continue gradient flow iterations with the updated direction set.
5. If using ARBOSW, perform "QSW seeding + 2 rounds of BO fine-tuning" every 25 steps.

The key to this process is not making every direction "optimal," but rather allocating the finite budget to the slice regions most discriminative for the current optimization.

### Loss & Training
The paper does not propose new downstream losses; the core is replacing the direction selector during SW estimation. Experimental settings are kept in a one-to-one correspondence with the QSW paper: same data, optimizer, learning rate, and stopping criteria, replacing only the direction construction method to ensure a fair comparison.

## Key Experimental Results

### Main Results
The paper evaluates several tasks: synthetic direction search, point cloud interpolation gradient flow, image style transfer, and point cloud autoencoders. The core conclusions are:
- In pure "approximate SW integration" scenarios, QSW remains strong (as it excels at uniform coverage).
- In dynamic SW-in-the-loop optimization tasks, BO hybrid methods (especially ARBOSW, ABOSW) are superior or at least competitive.

The table below is from point cloud gradient flow ($L=100$, metric is $W_2\downarrow$, values scaled by $10^2$):

| Method | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 | Time(s) |
|---|---:|---:|---:|---:|---:|---:|
| MCSW | 5.749 | 0.187 | 0.031 | 0.013 | 0.006 | 4.06 |
| CQSW | 5.603 | 0.183 | 0.078 | 0.073 | 0.071 | 3.96 |
| RCQSW | 5.708 | 0.181 | 0.027 | 0.011 | 0.005 | 3.95 |
| RBOSW (Ours) | **2.213** | **0.083** | 0.047 | 0.033 | 0.025 | 44.58 |
| ARBOSW (Ours) | 5.717 | 0.186 | **0.025** | 0.012 | **0.003** | 6.91 |

RBOSW leads significantly in early iterations but at the cost of high refresh overhead; ARBOSW achieves optimal or near-optimal results in later stages while maintaining moderate additional time cost.

### Ablation Study
The paper also compares multiple estimators on a point cloud autoencoder ($L=100$, 400 epochs, values scaled by $10^2$):

| Method | SW2 @100 | W2 @100 | SW2 @200 | W2 @200 | SW2 @400 | W2 @400 |
|---|---:|---:|---:|---:|---:|---:|
| MCSW | 2.25 | 10.58 | 2.11 | 9.92 | 1.94 | 9.21 |
| CQSW | 2.22 | 10.54 | 2.05 | 9.81 | 1.84 | 9.06 |
| BOSW | 2.20 | 10.34 | 2.02 | 9.78 | 1.80 | 9.01 |
| ABOSW | **2.18** | **10.27** | **2.01** | **9.76** | 1.81 | **9.01** |
| ARBOSW | 2.21 | 10.44 | 2.04 | 9.80 | 1.85 | 9.07 |

ABOSW provides the lowest reconstruction error across multiple checkpoints, suggesting that in scenarios with stable distributions and long-cycle training, QSW seeding + minor BO fine-tuning is more appropriate than frequent refreshing.

### Key Findings
- Direction selection is not a one-size-fits-all strategy: static integration favors QSW, while dynamic optimization loops benefit more from BO adaptation mechanisms.
- "Refresh frequency" is the key knob for the performance-efficiency trade-off: RBOSW offers high precision gains but at a large time cost, while ARBOSW provides a more balanced compromise.
- ABOSW offers high cost-effectiveness: replacing only a small number of directions (up to ~10%) yields stable gains across various tasks.

## Highlights & Insights
- Highlight 1: The improvement is precisely targeted at "direction set design" rather than "loss rewriting." This allows the method to seamlessly integrate with existing OT/SW training code with low engineering cost.
- Highlight 2: Proposes a hybrid paradigm of "QSW geometric prior + BO data adaptation." It does not discard low-discrepancy sampling but treats it as a stable starting point, then uses BO for local budget reallocation.
- Highlight 3: The authors are honest about the bias issue. They clearly state that BO directions introduce bias and do not package it as an unbiased estimator, but rather aim for "optimization convergence efficiency."
- Insight: In scenarios where SW is repeatedly invoked, "co-adaptation of the estimator and the task" is more important than a "one-time global integration rule," which is insightful for both generative models and registration tasks.

## Limitations & Future Work
- Bias Issue: Selecting directions via BO is not a classic unbiased spherical integration; theoretically, it does not guarantee an unbiased estimate of the true SW value under finite samples, which the paper lists as a priority for future research.
- Dimensional Scaling: Although high-dimensional BO is discussed, GP surrogates may still be limited in very high dimensions and under massive budgets, requiring stronger surrogates (like neural surrogates) or parallel acceleration.
- Hyperparameter Dependence: Refresh interval $R$, candidate pool size $n_c$, batch size $b$, and UCB $\beta$ all affect performance, requiring tuning when migrating across tasks.
- Computational Overhead: While emphasizing fast convergence, high-frequency refresh schemes like RBOSW are significantly heavier in terms of time; real-world deployment requires selection based on task latency budgets.

## Related Work & Insights
- **vs QSW/RQSW (Nguyen et al., 2024)**: The advantage of the QSW series is the quality of spherical coverage and stable implementation; this paper's advantage is task adaptability. ABOSW/ARBOSW prove the two can be combined rather than being mutually exclusive.
- **vs Importance-weighted SW (Nguyen & Ho, 2023)**: The latter constructs a new slice distribution via reweighting; this paper learns the direction set directly via BO, which is closer to "adaptive experimental design."
- **vs Bayesian Quadrature**: BQ targets the integral itself, but in optimization loops where the "integrand changes at every step," it requires frequent weight re-fitting at a high cost; this paper learns a direction selector instead, which is better suited for iterative tasks.
- Insight: Future work could turn the "direction learner" into a meta-learning module, pre-training direction policies on one class of tasks and transferring them to new data distributions to achieve warm-start adaptive SW.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The idea of BO-based direction selectors combined with QSW is clear and offers incremental innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers synthetic validation and three classes of downstream tasks, using the same protocols as strong baselines.
- Writing Quality: ⭐⭐⭐⭐☆ Method definitions and engineering details are complete; bias and applicability boundaries are honestly discussed.
- Value: ⭐⭐⭐⭐☆ High practical value for scenarios requiring SW-in-the-loop, especially the compromise solutions of ABOSW/ARBOSW.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

| Paper | Relationship | Key Difference |
|-------|--------------|----------------|
| QSW (Nguyen et al., 2024) | Baseline | Uses fixed low-discrepancy sequences vs. adaptive BO |
| Max-SW (Deshpande et al., 2019) | Related | Optimization for the *worst* direction vs. average with BO |

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICML 2025\] Tree-Sliced Wasserstein Distance: A Geometric Perspective](../../ICML2025/image_generation/tree-sliced_wasserstein_distance_a_geometric_perspective.md)
- [\[ICML 2025\] Tree-Sliced Wasserstein Distance with Nonlinear Projection](../../ICML2025/image_generation/tree-sliced_wasserstein_distance_with_nonlinear_projection.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICLR 2026\] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](vipo_visual_preference_optimization_at_scale.md)

</div>

<!-- RELATED:END -->
