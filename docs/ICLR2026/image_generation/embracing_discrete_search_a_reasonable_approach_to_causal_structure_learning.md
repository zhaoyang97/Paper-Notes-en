---
title: >-
  [Paper Note] Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning
description: >-
  [ICLR 2026][Image Generation][Causal Structure Learning] This paper proposes FLOP (Fast Learning of Order and Parents), a score-based causal discovery algorithm for linear models. By introducing fast parent selection and…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Causal Structure Learning"
  - "Discrete Search"
  - "Score Optimization"
  - "DAG Learning"
  - "Linear Models"
date: 2026-05-08
content_hash: b8c3980b6aebb5ec
---

# Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning

**Conference**: ICLR 2026
**arXiv**: [2510.04970](https://arxiv.org/abs/2510.04970)
**Code**: [https://github.com/CausalDisco/flopsearch](https://github.com/CausalDisco/flopsearch)
**Area**: Image Generation
**Keywords**: Causal Structure Learning, Discrete Search, Score Optimization, DAG Learning, Linear Models

## TL;DR

This paper proposes FLOP (Fast Learning of Order and Parents), a score-based causal discovery algorithm for linear models. By introducing fast parent selection and iterative Cholesky-based score updates, FLOP substantially reduces runtime, rendering Iterated Local Search (ILS) practical. It achieves near-perfect graph recovery on standard causal discovery benchmarks, reestablishing discrete search as a principled and competitive approach in causal discovery.

## Background & Motivation

The core task of causal structure learning is to recover a directed acyclic graph (DAG) over variables from observational data. Score-based methods assign penalized likelihood scores to candidate DAGs and seek the graph with the optimal score.

**Landscape of existing methods**:

**Exact algorithms**: Guarantee globally optimal graphs but scale exponentially, limiting applicability to roughly 30 variables.

**Local search methods** (e.g., GES, BOSS): Perform greedy hill-climbing by evaluating neighboring graphs, but are prone to local optima.

**Continuous optimization methods** (e.g., NOTEARS): Encode acyclicity as a smooth constraint and optimize continuously, but face empirical and theoretical concerns, including convergence issues and sensitivity to edge thresholding.

A key insight is that the NP-hardness results commonly cited against discrete search **do not apply** to standard causal discovery settings — the canonical hard instances rely on latent variables, and when the distribution is representable by a sparse DAG, discrete search can recover the target graph in polynomial time.

**Core motivation**: Under finite samples, imprecise scores cause local search methods to stagnate in local optima — this is the central practical challenge. FLOP addresses this by accelerating computation sufficiently to "fully embrace discrete search," enabling thorough exploration within a meaningful computational budget to escape local optima.

## Method

### Overall Architecture

FLOP adopts an order-based DAG search strategy: it searches over variable orderings and selects the optimal parent set for each given ordering. The algorithm comprises four core innovations, unified by a single theme: **trading computational efficiency for search thoroughness**.

### Key Designs

1. **Fast Parent Selection**: Parent sets are warm-started from those learned in the previous ordering iteration.

   - Conventional methods recompute parent sets from scratch upon each ordering change.
   - FLOP reuses the previous parent sets as a hot start, substantially reducing both computation and memory overhead.
   - Experiments confirm this does not degrade search quality, as local ordering changes typically affect the optimal parents of only a small number of variables.

2. **Iterative Cholesky-based Score Updates**: Accelerates computation of the linear Gaussian BIC score.

   - BIC scoring under linear Gaussian models requires computing regression residuals.
   - FLOP exploits the incremental update property of Cholesky factorization: when a local search move modifies only a single edge, only a rank-1 update to the Cholesky factor is needed rather than a full redecomposition.
   - This **amortizes** score update costs across local moves, significantly reducing per-step computation.
   - Implemented in Rust for efficiency and distributed as the Python pip package `flopsearch`.

3. **Principled Order Initialization**: Reduces parent selection failures for distant ancestor–descendant pairs compared to random initialization.

   - Distant, weakly dependent ancestor–descendant pairs are particularly susceptible to errors under finite samples.
   - FLOP employs a data-statistic-based initialization strategy to place the initial ordering closer to the true causal ordering.
   - This provides a better starting point for subsequent local search.

4. **Iterated Local Search (ILS)**: Systematically escapes local optima.

   - Traditional local search methods (e.g., BOSS) terminate upon reaching a local optimum.
   - FLOP applies controlled perturbations at local optima and restarts the search, repeating this process iteratively.
   - FLOP_k denotes $k$ ILS restarts, where $k$ serves as a **computational budget hyperparameter**.
   - This establishes a direct link between runtime and finite-sample accuracy: more restarts → better graphs.

### Loss & Training

The standard **BIC (Bayesian Information Criterion)** is used as the scoring function:

$$\text{BIC}(G) = -2 \log L(G | \mathcal{D}) + k \log n$$

where $L$ is the likelihood, $k$ is the number of parameters, and $n$ is the sample size. Under linear Gaussian models, this score guarantees consistency — in the infinite-sample limit, the highest-scoring graph recovers the true causal graph (or its equivalence class, the CPDAG).

## Key Experimental Results

### Main Results

Experiments are conducted on data generated from linear additive noise models (ANMs) under both Erdős-Rényi (ER) and Scale-Free (SF) graph structures.

| Method | ER-50-deg8 SHD ↓ | ER-50-deg8 Exact Recovery | Runtime |
|--------|-------------------|--------------------------|---------|
| PC | ~350 | 0% | ~50s |
| GES | ~250 | 0% | ~200s |
| DAGMA | ~200 | 0% | ~300s |
| BOSS (FLOP_0) | ~50 | 40% | ~30s |
| FLOP_20 | ~20 | 60% | ~100s |
| FLOP_100 | ~10 | 60% | ~400s |

On 50-node ER graphs with average degree 8 and 1,000 samples, FLOP substantially outperforms all baselines.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| FLOP_0 (no ILS restarts) | Comparable accuracy to BOSS (~40% exact recovery) but faster runtime |
| FLOP_20 (20 restarts) | Exact recovery improves to 60%; SHD decreases substantially |
| FLOP_100 (100 restarts) | Comparable exact recovery to FLOP_20 but lower SHD on harder instances |
| Random vs. principled initialization | Principled initialization significantly reduces errors on distant, weakly dependent pairs |
| Without Cholesky acceleration | Runtime increases by multiples, rendering ILS infeasible |

### Key Findings

1. **Validity of discrete search**: Under standard causal discovery settings, discrete search can recover causal graph structure near-perfectly, challenging the prevailing view that "NP-hardness implies infeasibility."
2. **Computation–accuracy tradeoff**: The parameter $k$ in FLOP_k establishes an explicit **runtime–accuracy tradeoff** that users can adjust according to their computational budget.
3. **No advantage for continuous optimization**: Continuous methods such as DAGMA underperform discrete search under equivalent computational budgets and introduce additional complications related to thresholding and convergence.
4. **Finite samples are the core challenge**: Even under asymptotic theoretical guarantees, finite-sample search may find graphs with higher scores than the ground truth — ILS is an effective countermeasure.
5. **Rust implementation enables large-scale feasibility**: The efficient implementation makes full discrete search over 50+ nodes practically viable.

## Highlights & Insights

1. **Important epistemological contribution**: Beyond algorithmic improvement, this work offers a fundamental reexamination of methodology in causal discovery. The NP-hardness argument is carefully deconstructed — standard hard instances rely on latent variables and do not apply to common settings.
2. **Integration of engineering and theory**: The Cholesky incremental update is a classical numerical linear algebra technique applied elegantly to causal discovery; the Rust implementation delivers a practically deployable tool.
3. **Introduction of ILS meta-heuristics**: A classical optimization paradigm from operations research is brought into causal discovery, establishing an explicit computation/accuracy relationship.
4. **Simplicity and elegance**: The entire method rests on well-established statistical foundations (BIC scoring, Cholesky decomposition) without invoking neural networks or complex optimization, returning to a simple, interpretable approach.

## Limitations & Future Work

1. **Restricted to linear models**: FLOP's Cholesky acceleration relies on the linear Gaussian assumption; nonlinear causal settings require new scoring functions and search strategies.
2. **Causal sufficiency assumption**: The method assumes no unobserved confounders, an assumption frequently violated in practice.
3. **Large-scale scalability**: While performance is strong at 50 nodes, scalability to hundreds or thousands of variables — as required in gene regulatory network applications — needs further investigation.
4. **Observational data only**: Interventional data, which provides substantially more information for causal discovery, is not utilized.
5. **Manual specification of ILS restart count**: Although a runtime–accuracy relationship is established, theoretical guidance for determining the optimal number of restarts is lacking.

## Related Work & Insights

- **BOSS** (Andrews et al., 2023): The strongest order-based local search baseline; FLOP builds upon it with acceleration and ILS.
- **GES** (Chickering, 2002): The classical equivalence-class-based search method; asymptotically optimal but limited under finite samples.
- **NOTEARS / DAGMA**: Representative continuous optimization approaches; experiments in this paper demonstrate their inferiority to discrete search.
- **Partition MCMC / Order MCMC** (Kuipers et al., 2022): Bayesian methods that escape local optima via simulated annealing, but at greater computational cost.

The broader implication for causal discovery: improvements in computational efficiency can fundamentally alter the feasibility of entire algorithmic paradigms. When discrete search becomes sufficiently fast, its theoretical advantages — no thresholding, no continuous relaxation, direct BIC optimization — can be fully realized.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discrete Adjoint Matching](discrete_adjoint_matching.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](../../NeurIPS2025/image_generation/non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[ICLR 2026\] Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](exposing_hidden_biases_in_text-to-image_models_via_automated_prompt_search.md)
- [\[ICLR 2026\] Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](loopholing_discrete_diffusion_deterministic_bypass_of_the_sampling_wall.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)

</div>

<!-- RELATED:END -->
