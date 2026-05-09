---
title: >-
  [Paper Note] MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions
description: >-
  [NeurIPS 2025][Optimization][Multi-objective Bayesian optimization] This paper proposes MOBO-OSD, an algorithm that generates diverse Pareto-optimal solutions by defining orthogonal search directions on an approximated convex hull of individual minima (CHIM), combined with Pareto front estimation and a batch selection strategy, consistently outperforming state-of-the-art multi-objective Bayesian optimization methods on both synthetic and real-world benchmarks.
tags:
  - NeurIPS 2025
  - Optimization
  - Multi-objective Bayesian optimization
  - Pareto front
  - orthogonal search directions
  - hypervolume improvement
  - batch optimization
date: 2026-05-08
content_hash: 529cb38c05ea14c7
---

# MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions

**Conference**: NeurIPS 2025
**arXiv**: [2510.20872](https://arxiv.org/abs/2510.20872)
**Code**: [GitHub](https://github.com/LamNgo1/mobo-osd)
**Area**: Optimization
**Keywords**: Multi-objective Bayesian optimization, Pareto front, orthogonal search directions, hypervolume improvement, batch optimization

## TL;DR

This paper proposes MOBO-OSD, an algorithm that generates diverse Pareto-optimal solutions by defining orthogonal search directions on an approximated convex hull of individual minima (CHIM), combined with Pareto front estimation and a batch selection strategy, consistently outperforming state-of-the-art multi-objective Bayesian optimization methods on both synthetic and real-world benchmarks.

## Background & Motivation

Multi-objective Bayesian optimization (MOBO) aims to optimize multiple conflicting black-box objective functions under a limited evaluation budget. Existing methods face three major challenges:

**Insufficient diversity**: Many scalarization-based methods (e.g., ParEGO) fail to capture diverse Pareto fronts, leading to suboptimal hypervolume metrics.

**Poor scalability**: Methods such as DGEMO cannot handle more than three objectives ($M > 3$); qEHVI incurs rapidly growing computational costs as the number of objectives increases.

**Batch optimization complexity**: Batch settings require modeling cross-objective interactions among unevaluated points, which existing methods handle inadequately.

The core insight of this paper is derived from the classical **Normal Boundary Intersection (NBI) method**: intersections of the objective space boundary with vectors orthogonal to the CHIM tend to be Pareto-optimal. While NBI produces well-distributed Pareto solutions, it requires a large number of function evaluations. The authors transfer this geometric intuition to the limited-budget Bayesian optimization setting.

## Method

### Overall Architecture

The MOBO-OSD pipeline consists of four steps:
1. Construct an approximate CHIM and define orthogonal search directions (OSD).
2. Solve a constrained optimization subproblem along each OSD.
3. Explore additional candidate points near existing solutions via Pareto front estimation (PFE).
4. Select evaluation points using hypervolume improvement (HVI) and the Kriging Believer batch strategy.

### Key Designs

1. **Approximate CHIM construction**: Since the true individual minima are unavailable, the authors construct $M$ boundary points $\mathbf{p}_m$ from the ideal point and nadir point of the observed data. The $m$-th boundary point replaces the $m$-th coordinate of the ideal point with the nadir value. The approximate CHIM is the convex hull of these points: $\mathcal{U}(\boldsymbol{\beta}) = \sum_{m=1}^M \beta_m \mathbf{p}_m$. Compared to directly using observed individual minima, this design **avoids premature shrinkage of the search region**.

2. **Orthogonal search directions and subproblems**: An OSD is defined as the line passing through a point on the approximate CHIM along normal vector $\mathbf{n}$. The subproblem is formulated as a constrained optimization:

    $(\mathbf{x}^{\text{OSD}}, \lambda^{\text{OSD}}) \in \max_{\mathbf{x} \in \mathcal{X}} \lambda \quad \text{s.t.} \quad \gamma(\mathbf{x}; \boldsymbol{\beta}, \mathbf{n}) \in \mathcal{Q}(\mathbf{x})$

   Objective values are replaced by GP posterior means $\boldsymbol{\mu}(\mathbf{x})$, and equality constraints are relaxed to confidence interval constraints $\mathcal{Q}(\mathbf{x})$. The SLSQP solver is used with a multi-start strategy, and the final solution is selected by hypervolume contribution.

3. **Pareto front estimation (PFE)**: For each OSD solution $\mathbf{x}^{\text{OSD}}$, a local exploration space $\mathcal{T}$ is computed via first-order approximation, and $n_e$ additional candidate points are randomly sampled within it. This avoids solving an excessive number of dense subproblems.

4. **Batch selection strategy**: The strategy combines the Kriging Believer approach (updating the GP after each selected point) with a diversity constraint on exploration spaces—ensuring that candidates from different OSDs remain sufficiently distinct. Weight vectors $\{\boldsymbol{\beta}\}$ are generated uniformly on the unit simplex using the Riesz $s$-Energy method.

### Loss & Training

- The acquisition function is hypervolume improvement (HVI): $\alpha_{\text{HVI}}(\mathbf{x}) = \text{HV}(\boldsymbol{\mu}(\mathbf{x}) \cup \mathcal{P}_f, \mathbf{r}) - \text{HV}(\mathcal{P}_f, \mathbf{r})$
- Each objective function is modeled by an independent GP surrogate.
- The default number of OSD directions is $n_\beta = 20$.

## Key Experimental Results

### Main Results (Sequential optimization, batch size = 1)

| Benchmark | # Objectives | MOBO-OSD | qEHVI | DGEMO | USeMO | NSGA-II |
|---|---|---|---|---|---|---|
| DTLZ2-M2 | 2 | **Best** | 2nd | Competitive | Moderate | Poor |
| DTLZ2-M3 | 3 | **Best** | 2nd | Competitive | Moderate | Poor |
| DTLZ2-M4 | 4 | **Best** | 2nd | Not supported | Moderate | Poor |
| ZDT1 | 2 | **Best** | 2nd | Competitive | Moderate | Poor |
| VLMOP2 | 2 | **Best** | Competitive | 2nd | Moderate | Poor |
| Speed Reducer | 2 | **Best** | Competitive | 2nd | Moderate | Poor |
| Car Side Design | 3 | **Best** | 2nd | Competitive | Moderate | Poor |
| Marine Design | 4 | **Best** | 2nd | Not supported | Moderate | Poor |
| Water Planning | 6 | **Best** | 2nd | Not supported | Moderate | Poor |

### Ablation Study (Effect of PFE component)

| Configuration | DTLZ2-M2 (HV) | VLMOP2 (HV) | Car Side Design (HV) |
|---|---|---|---|
| w/o PFE ($n_\beta=20$) | 0.4041±0.0004 | 0.2713±0.0020 | 145.12±0.33 |
| w/o PFE ($n_\beta=100$) | 0.4118±0.0001 | 0.2978±0.0011 | 154.32±0.27 |
| w/o PFE ($n_\beta=200$) | 0.4142±0.0001 | 0.3076±0.0006 | 157.23±0.21 |
| w/o PFE ($n_\beta=500$) | 0.4164±0.0001 | 0.3159±0.0004 | 160.28±0.21 |
| **Default (with PFE)** | **0.4217±0.0000** | **0.3383±0.0000** | **177.48±0.23** |

### Key Findings

- MOBO-OSD consistently outperforms state-of-the-art methods across all 9 benchmark problems in both sequential and batch settings.
- The PFE component substantially improves efficiency, effectively equivalent to adding 20× or more search directions.
- The algorithm is robust to the $n_\beta$ parameter, with stable performance for $n_\beta \in \{10, 50, 100\}$.
- The method scales to 6 objectives, whereas DGEMO supports at most 3.

## Highlights & Insights

- The geometric intuition of the NBI method is elegantly embedded into the Bayesian optimization framework, overcoming NBI's requirement for a large number of evaluations.
- The approximate CHIM design prevents premature contraction of the search space—a subtle yet effective engineering choice.
- The diversity constraint on exploration spaces in the batch selection strategy is both concise and effective.
- The two-criterion selection step in the OSD subproblem (maximizing $\lambda$ while minimizing deviation distance) reflects a principled balance between exploration and exploitation.

## Limitations & Future Work

- The current formulation handles only **noise-free observations**, whereas practical applications typically involve noisy measurements.
- A comprehensive comparison with recent methods such as HVKG is not included (though compatibility is mentioned).
- The orthogonal directions depend on the computation of normal vector $\mathbf{n}$, which may require more flexible directional strategies for highly non-convex Pareto fronts.
- Incorporating multi-fidelity information could further reduce computational costs.

## Related Work & Insights

- The NBI method offers a new perspective for the MOBO community, suggesting that geometric techniques from classical multi-objective optimization deserve broader adoption.
- MOBO-OSD shares the PFE technique with DGEMO but employs a more systematic search strategy (ordered OSD vs. random search).
- This work raises a broader question: can other geometric decomposition methods from classical optimization be incorporated into the Bayesian optimization framework?

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing NBI into MOBO is novel, though individual components (PFE, Kriging Believer) are existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Nine benchmarks, multiple batch sizes, and detailed ablations make for a comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides a strong and scalable new baseline for MOBO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] In Search of Adam's Secret Sauce](in_search_of_adams_secret_sauce.md)
- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] Optimistic Online-to-Batch Conversions for Accelerated Convergence and Universality](optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](../../AAAI2026/optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)

</div>

<!-- RELATED:END -->
