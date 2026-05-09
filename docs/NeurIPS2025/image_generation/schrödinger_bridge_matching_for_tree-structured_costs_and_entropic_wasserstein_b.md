---
title: >-
  [Paper Note] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres
description: >-
  [NeurIPS 2025][Image Generation][Schrödinger Bridge] This paper extends the Iterative Markovian Fitting (IMF) procedure to the tree-structured Schrödinger Bridge problem, proposing the TreeDSBM algorithm. For Wasserstein barycentre computation, it elegantly merges IMF iterations with fixed-point iterations, requiring only inexpensive bridge-matching steps for efficient solution.
tags:
  - NeurIPS 2025
  - Image Generation
  - Schrödinger Bridge
  - Iterative Markovian Fitting
  - Wasserstein Barycentre
  - Optimal Transport
  - Tree-Structured Cost
  - Generative Model
date: 2026-05-08
content_hash: 48adc42c55fca746
---

# Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres

**Conference**: NeurIPS 2025
**arXiv**: [2506.17197](https://arxiv.org/abs/2506.17197)
**Authors**: Samuel Howard, Peter Potaptchik, George Deligiannidis (Oxford)
**Code**: [samuel-howard/Tree_SB_Matching_Barycentres](https://github.com/samuel-howard/Tree_SB_Matching_Barycentres)
**Area**: Image Generation
**Keywords**: Schrödinger Bridge, Iterative Markovian Fitting, Wasserstein Barycentre, Optimal Transport, Tree-Structured Cost, Generative Model

## TL;DR

This paper extends the Iterative Markovian Fitting (IMF) procedure to the tree-structured Schrödinger Bridge problem, proposing the TreeDSBM algorithm. For Wasserstein barycentre computation, it elegantly merges IMF iterations with fixed-point iterations, requiring only inexpensive bridge-matching steps for efficient solution.

## Background & Motivation

### State of the Field
The Schrödinger Bridge (SB) is the dynamic formulation of entropy-regularized optimal transport (OT), for which flow-based generative modeling methods have recently provided scalable solvers. The standard SB problem addresses transport between two marginal distributions; more generally, multi-marginal OT minimizes a cost function defined over multiple distributions, with **tree-structured costs** attracting particular attention due to their broad applicability and structural efficiency gains. In particular, the star-shaped tree corresponds to the important **Wasserstein barycentre** problem—defining a natural "mean" for probability distributions.

### Limitations of Prior Work
- **Limitations of IPF-based methods**: Existing TreeDSB relies on Iterative Proportional Fitting (IPF), which only preserves marginal distributions at convergence, requires expensive trajectory caching, and suffers from "forgetting" the original reference measure.
- **Bottleneck of fixed-point methods**: Classical fixed-point iteration for Wasserstein-2 barycentres requires solving a full OT problem at each step, incurring high computational cost.
- **Gap in the literature**: Although IMF has outperformed IPF in the standard SB setting, it has not yet been extended to the tree-structured SB setting (see Table 1 for literature positioning).

### Core Idea
The paper transfers the advantages of IMF (marginal preservation at every step, no trajectory caching, training stability) to tree-structured SB and Wasserstein barycentre computation, filling the identified gap in the literature.

## Method

### Stochastic Processes on Trees
A stochastic process is defined on a tree $\mathcal{T}=(\mathcal{V},\mathcal{E},\ell)$: along each edge $e=(u,v)$ of the rooted tree, the SDE $\mathrm{d}X_t^e = v^e(t,X_t^e)\mathrm{d}t + \sigma_t^e \mathrm{d}B_t^e$ is simulated in depth-first traversal order from the root, yielding a path measure $\mathbb{P}\in\mathcal{P}(\mathcal{C}_\mathcal{T})$ over the tree.

### Tree-Structured Schrödinger Bridge
Given marginal distributions $\{\mu_i\}_{i\in\mathcal{S}}$ fixed only at a subset of vertices $\mathcal{S}\subset\mathcal{V}$, the TreeSB problem is defined as:

$$\mathbb{P}^{SB} = \arg\min_{\mathbb{P}\in\mathcal{P}(\mathcal{C}_\mathcal{T})} \left\{ \mathrm{KL}(\mathbb{P}\|\mathbb{Q}) \;\middle|\; \mathbb{P}_i=\mu_i\;\forall i\in\mathcal{S} \right\}$$

The static formulation is equivalent to entropy-regularized multi-marginal OT with a tree-structured quadratic cost.

### Core Theorem of TreeIMF
**Theorem 3.1**: The solution to TreeSB is the unique path measure that simultaneously satisfies the Markov property and belongs to the reciprocal class $\mathcal{R}_\mathcal{S}(\mathbb{Q})$ (with correct marginals). This generalizes the characterization result of standard IMF and provides the theoretical foundation for an alternating projection algorithm.

The key technical tool is the **KL divergence decomposition along the tree** (Lemma 3.2):

$$\mathrm{KL}(\mathbb{P}\|\widetilde{\mathbb{P}}) = \sum_{(u,v)\in\mathcal{E}_r} \mathbb{E}_{X_u\sim\mathbb{P}_u}\left[\mathrm{KL}(\mathbb{P}^{(u,v)}(\cdot|X_u)\|\widetilde{\mathbb{P}}^{(u,v)}(\cdot|X_u))\right]$$

### TreeDSBM Algorithm
The algorithm alternates between two steps:
1. **Reciprocal projection**: Sample known vertex values from the current coupling $\Pi_\mathcal{S}$, sample unknown vertex values via the Gaussian conditional $\mathbb{Q}_{\mathcal{S}^c|\mathcal{S}}$, and draw Brownian bridges along each edge.
2. **Markov projection**: Train a neural network vector field $v_{\theta_e}$ independently for each edge using the bridge-matching loss:

$$\mathcal{L}(\theta_e) = \int_0^{T^e} \mathbb{E}\left\| \frac{X_v - X_t}{T^e - t} - v_{\theta_e}(X_t, t) \right\|^2 \mathrm{d}t$$

Bidirectional training is employed to reduce error accumulation, and all edges can be trained in parallel.

### Connection to Fixed-Point Methods
For star-shaped trees (the barycentre problem), the conditional distribution simplifies to $Y_0 \sim \mathcal{N}(\sum\lambda_i Y_i, \sigma^2 I_d)$. TreeDSBM merges IMF iterations and fixed-point iterations into a single procedure, replacing the expensive full OT solve at each step with inexpensive bridge-matching.

**Theorem 3.5**: The TreeIMF iterations converge to the unique fixed point (i.e., the TreeSB solution), with $\lim_{n\to\infty}\mathrm{KL}(\mathbb{P}^n\|\mathbb{P}^*)=0$.

## Key Experimental Results

### Experiment 1: 2D Synthetic Barycentres

Computing the $(\frac{1}{3},\frac{1}{3},\frac{1}{3})$-barycentre of moon, spiral, and circle datasets with $\varepsilon=0.1$.

| Method | Iterations | Sinkhorn Div. (k=0) | Sinkhorn Div. (k=1) | Sinkhorn Div. (k=2) |
|--------|-----------|---------------------|---------------------|---------------------|
| **TreeDSBM** | 6 IMF | **1.14±0.07** | **1.05±0.07** | **1.08±0.11** |
| TreeDSB | 50 IPF | 2.35 | 4.04 | 2.35 |
| WIN | - | 1.17 | - | - |

TreeDSBM with 6 iterations substantially outperforms TreeDSB with 50 iterations and is competitive with the strong WIN baseline.

### Experiment 2: Subset Posterior Aggregation (BW₂²-UVP, %)

Subset posterior aggregation on the bike rental dataset, comparing Poisson and negative binomial regression:

| Method | Poisson (↓) | Negative Binomial (↓) |
|--------|------------|----------------------|
| WIN | 0.014 | 0.009 |
| W2CB | 0.026 | 0.024 |
| NOTWB | 0.023 | 0.018 |
| **TreeDSBM** | **0.008** | **0.012** |

TreeDSBM achieves the best result on Poisson regression and trains substantially faster (~3 minutes vs. ~45 minutes for WIN).

### Experiment 3: High-Dimensional Gaussian Barycentres

| Method | d=64 BW₂² | d=96 BW₂² | d=128 BW₂² | d=64 L² | d=96 L² | d=128 L² |
|--------|-----------|-----------|-----------|--------|--------|---------|
| WIN | 0.20 | 0.30 | 0.38 | 0.96 | 1.20 | 1.46 |
| W2CB | 0.04 | 0.07 | 0.12 | 0.17 | 0.20 | 0.25 |
| NOTWB | 0.08 | 0.10 | 0.14 | 0.10 | 0.10 | 0.13 |
| **TreeDSBM** | 0.14 | 0.15 | 0.27 | 1.18 | 1.13 | 1.23 |

TreeDSBM approaches state-of-the-art on the BW₂² metric, comparable to WIN; it is slightly inferior to W2CB and NOTWB on the L² metric.

### Experiment 4: MNIST 2, 4, 6 Barycentres
TreeDSB exhibits training instability at low regularization $\varepsilon$; TreeDSBM, by preserving marginal matching at every step, can operate with $\varepsilon=0.02$ (far smaller than TreeDSB's 0.5), producing visually superior barycentre samples in only 4 IMF iterations.

## Highlights & Insights

- **Elegant theoretical generalization**: The convergence theory of IMF (alternating projections onto reciprocal and Markov classes) is fully extended to the tree-structured setting, crucially leveraging the KL decomposition lemma along trees.
- **Unification of IMF and fixed-point iteration**: TreeDSBM is shown to be equivalent, on star-shaped trees, to a fixed-point barycentre algorithm using a flow-based entropic OT solver, collapsing two nested iteration loops into a single procedure.
- **Significant computational efficiency gains**: Six IMF iterations substantially outperform fifty IPF iterations (TreeDSB); each step requires only bridge-matching rather than a full OT solve, and edge-independent training is fully parallelizable.
- **Training stability**: IMF guarantees correct marginals at every step, permitting smaller regularization $\varepsilon$ and avoiding the training instability of TreeDSB.
- **Open-source implementation**: JAX-based code is publicly available.

## Limitations & Future Work

- **Restricted to quadratic costs**: The method applies only to Wasserstein-2 (quadratic ground cost) OT and does not support general cost functions.
- **Entropic bias**: Entropy regularization introduces bias relative to the true OT solution.
- **Higher inference cost**: Compared to single-function-evaluation methods (e.g., the composition maps of WIN), SDE simulation requires multiple inference steps.
- **Non-simulation-free**: Each iteration requires simulating the currently learned process to generate samples, unlike analytic methods.
- **High L²-UVP in high dimensions**: The L²-UVP metric is notably higher than W2CB and NOTWB in the 128-dimensional Gaussian experiment.
- **Suboptimal for shared-structure settings**: Appendix experiments show that when known marginals and the barycentre share simple structure, the method may underperform specialized approaches.

## Related Work & Insights

- **TreeDSB**: The direct predecessor—an IPF-based tree-structured SB solver. TreeDSBM outperforms TreeDSB across all experiments with fewer iterations and greater training stability.
- **DSBM**: The standard IMF/SB matching method; the present work generalizes it to tree structures, recovering DSBM when the tree degenerates to a single edge.
- **WIN**: A Wasserstein-2 barycentre method exploiting fixed-point properties and learning OT maps via an adversarial loss. TreeDSBM achieves comparable performance with shorter training time.
- **W2CB**: A barycentre method based on input-convex neural networks, achieving the best BW₂² metric in high-dimensional Gaussian experiments, though potentially prone to convergence difficulties on complex discontinuous transport problems.
- **NOTWB**: A bilevel adversarial method applicable to general costs with strong high-dimensional performance. TreeDSBM uses a non-adversarial bridge-matching loss, yielding more stable training.
- **DSB**: An IPF-based SB solver for the two-marginal case; TreeDSB is its tree-structured generalization, while TreeDSBM represents the corresponding IMF-based generalization.

## Rating

- Novelty: ⭐⭐⭐⭐ — The theoretical generalization is natural but non-trivial; the unified view of IMF and fixed-point iteration is genuinely insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers synthetic data, image data, posterior aggregation, and high-dimensional Gaussians, with comparisons against multiple state-of-the-art baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rigorous theory, intuitive figures, and precise literature positioning.
- Value: ⭐⭐⭐⭐ — Fills the gap of IMF in tree-structured SB and offers practical utility for barycentre computation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Branched Schrödinger Bridge Matching](../../ICLR2026/image_generation/branched_schrödinger_bridge_matching.md)
- [\[ICLR 2026\] Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](../../ICLR2026/image_generation/contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)
- [\[NeurIPS 2025\] Tree-Guided Diffusion Planner](tree-guided_diffusion_planner.md)
- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)

</div>

<!-- RELATED:END -->
