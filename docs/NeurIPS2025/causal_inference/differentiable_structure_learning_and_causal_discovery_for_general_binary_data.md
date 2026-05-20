---
title: >-
  [Paper Note] Differentiable Structure Learning and Causal Discovery for General Binary Data
description: >-
  [NEURIPS2025][Causal Inference][Causal discovery] This paper proposes a general differentiable structure learning framework based on the Multivariate Bernoulli Distribution (MVB) that makes no assumptions about the speci…
tags:
  - "NEURIPS2025"
  - "Causal Inference"
  - "Causal discovery"
  - "structure learning"
  - "discrete data"
  - "DAG learning"
  - "multivariate Bernoulli distribution"
date: 2026-05-08
content_hash: 959bddb84a102386
---

# Differentiable Structure Learning and Causal Discovery for General Binary Data

**Conference**: NEURIPS2025
**arXiv**: [2509.21658](https://arxiv.org/abs/2509.21658)  
**Code**: To be confirmed  
**Area**: Causal Inference
**Keywords**: Causal discovery, structure learning, discrete data, DAG learning, multivariate Bernoulli distribution

## TL;DR

This paper proposes a general differentiable structure learning framework based on the Multivariate Bernoulli Distribution (MVB) that makes no assumptions about the specific data-generating process, captures arbitrary higher-order dependencies among binary discrete variables, and proves that while DAGs are not identifiable in the general setting, the minimal equivalence class (Markov equivalence class) is recoverable.

## Background & Motivation

**Background**: Causal discovery aims to learn directed acyclic graphs (DAGs) from observational data, a problem that is NP-complete. Recent methods such as NOTEARS have reformulated it as differentiable constrained optimization, achieving significant breakthroughs.

**Limitations of Prior Work**: Existing differentiable structure learning methods are primarily designed for continuous data under Gaussian assumptions. The few extensions to discrete data (e.g., Zeng et al. 2022, generalized linear SEM) assume specific parametric forms and consider only linear or additive effects.

**Key Challenge**: Many real-world datasets involve binary or discrete variables (disease presence, genetic markers, survey responses), whose complex higher-order dependency structures cannot be captured by linear models. Bello et al. (2022) applied continuous optimization to logistic regression models for discrete data but provided no theoretical identifiability guarantees. Constraint-based methods (PC) and score-based methods (GES) either lack statistical robustness or rely on strong parametric assumptions (additive noise, latent representations, linear effects, latent Gaussian variables).

**Goal**: To develop a general, theoretically sound, and differentiable structure learning framework for discrete data that can model arbitrary dependencies without being constrained by specific data-generating assumptions.

## Method

### Overall Architecture

The paper constructs a general differentiable structure learning framework for discrete data based on the **Multivariate Bernoulli Distribution (MVB)**. The core mechanism proceeds in three steps: (1) use MVB as a universal representation of arbitrary binary data; (2) prove non-identifiability and characterize the complete set of equivalent structure–parameter pairs; (3) reformulate the search for the sparsest DAG as a single differentiable optimization problem.

### Key Designs

#### Module 1: MVB Parameterization and Higher-Order Interactions

The joint distribution is written in exponential family form: $P(X=x) = \exp(\sum_{S \subseteq [p]} f^S B^S(x))$, where $B^S(x) = \prod_{j \in S} x_j$ denotes interaction features. The conditional distribution is naturally expressed as:

$$\Pr(X_p = 1 | X_{-p}) = \sigma\left(\sum_{S \subseteq [p-1]} f^{S,p} B^S(x)\right)$$

where $\sigma$ denotes the sigmoid function. The **key insight** is that for general binary data, **all higher-order interaction terms are present**, and ignoring them (e.g., retaining only first-order terms) leads to erroneous causal conclusions. An extended feature map $\Phi(X) = [B^S(X)]_{S \in 2^{[p]}}$ encodes all $2^p$ interaction terms uniformly.

#### Module 2: Characterization of Non-Identifiability and Minimal Equivalence Class

**Theorem 1 (Non-Identifiability)**: For any topological ordering $\pi$, there exists a unique structural equation model $(f_\pi, G_\pi)$ that exactly reproduces the observed distribution. Consequently, there are at most $p!$ equivalent DAG–parameter pairs.

The **minimal equivalence class** $\mathcal{E}_{\min}(p)$ is defined as the subset of equivalent pairs whose DAGs have the fewest edges.

**Theorem 2**: Under the SMR (Sparsest Markov Representation) assumption or faithfulness assumption, all DAGs in the minimal equivalence class belong to the same Markov equivalence class.

#### Module 3: Differentiable Optimization (BiNOTEARS)

A parameter matrix $\mathbf{H} \in \mathbb{R}^{2^p \times p}$ is defined, and edge existence is determined via a weighted adjacency matrix $[W(\mathbf{H})]_{ij} = \sum_{S \subseteq [p], i \in S} (h^{S,j})^2$, with the constraint $h^{S,j} = 0$ when $j \in S$ (to prohibit self-loops).

A **two-stage strategy (BiNOTEARS)** is employed for scalability to larger graphs:
- **Stage 1**: Adapt NOTEARS-MLP to discrete data, learn a coarse structure, and extract a topological ordering $\pi$.
- **Stage 2**: Construct first- and second-order interaction features along the topological ordering and fit logistic regression to determine the final edge structure.

### Loss & Training

The regularized score function is:

$$s(\mathbf{H}; \lambda, \delta, \mathbf{X}) = \ell(\mathbf{H}; \mathbf{X}) + p_{\lambda,\delta}(W(\mathbf{H}))$$

where $\ell$ denotes cross-entropy (negative log-likelihood) and $p_{\lambda,\delta}$ is a **quasi Minimax-Concave Penalty (quasi-MCP)**:

$$p_{\lambda,\delta}(t) = \lambda \left[ (|t| - \frac{t^2}{2\delta}) \mathbb{1}(|t|<\delta) + \frac{\delta}{2} \mathbb{1}(|t|>\delta) \right]$$

This penalty is quadratic on $[0, \delta]$ and flat beyond $\delta$, providing a smooth approximation of the $\ell_0$ penalty that encourages a sparse adjacency matrix. The final optimization problem incorporates the differentiable acyclicity constraint $h(W(\mathbf{H})) = 0$.

**Theorem 3**: There exist sufficiently small $\lambda, \delta > 0$ such that the set of global optima is exactly equal to the minimal equivalence class.

## Key Experimental Results

### Main Results: Synthetic Data (Small Graphs, $p \in \{5,\ldots,9\}$)

| Setting | BiNOTEARS (Ours) | DAGMA | PC | FGES |
|---------|-----------------|-------|-----|------|
| ER-1 (1st+2nd order) | **Best** | Poor (1st order only) | Moderate | Moderate |
| SF-1 (1st+2nd order) | **Best** | Poor | Moderate | Moderate |
| ER-2 (1st+2nd order) | **Best** | Significantly worse | Moderate | Moderate |
| SF-2 (1st+2nd order) | **Best** | Significantly worse | Moderate | Moderate |
| Highest-order interactions only | **Near-optimal recovery** | Complete failure | Partial recovery | Partial recovery |

> DAGMA models only first-order effects; its SHD degrades substantially in the presence of higher-order interactions. BiNOTEARS achieves the strongest performance across all settings.

### Real Data: Sachs Protein Signaling Network ($d=11$, $n=7466$)

| Method | SHD ↓ | # Edges |
|--------|-------|---------|
| NOTEARS (linear) | 22 | 18 |
| NOTEARS-MLP | 16 | 13 |
| **BiNOTEARS (Ours)** | **13** | 15 |

### Ablation Study

1. **Higher-order interactions are critical**: When data contain second-order interactions, first-order-only models (DAGMA) suffer a sharp performance drop, validating the necessity of capturing higher-order dependencies.
2. **Two-stage strategy enables scaling**: BiNOTEARS remains competitive on larger graphs ($p = 10, 20, 30, 40$), although all methods degrade as graph density increases.
3. **Real-data validation**: On the Sachs dataset, BiNOTEARS reduces SHD from 22 (NOTEARS) to 13, a 41% reduction in structural error.

## Highlights & Insights

1. **Strong theoretical contributions**: The paper provides a complete characterization of DAG non-identifiability for general binary data (Theorem 1) and proves that the minimal equivalence class is recoverable via a single differentiable procedure (Theorem 3), offering theoretical grounding for prior empirical work.
2. **General-purpose framework**: By representing arbitrary binary joint distributions via MVB without assuming any specific SEM form, this is the first truly general differentiable structure learning method for discrete data.
3. **Practical two-stage method**: BiNOTEARS cleverly balances expressiveness and computational efficiency by using NOTEARS-MLP to obtain a topological ordering followed by logistic regression for refinement.
4. **Key Findings**: Higher-order interaction terms in discrete data cannot be ignored; linear assumptions introduce fundamental errors.

## Limitations & Future Work

1. **Exponential parameter space**: The general MVB has $2^p$ parameters, restricting the full-order interaction approach to small graphs ($p < 10$).
2. **Lack of theoretical guarantees for the two-stage method**: The second stage of BiNOTEARS (truncated to second-order interactions) is practically effective but theoretically unanalyzed.
3. **Restricted to binary data**: Although multi-category variables can in principle be encoded as binary indicators, this further exacerbates the dimensionality explosion.
4. **Limited comparison with nonparametric methods**: Comparisons with recent kernel-based or conditional independence testing methods are absent.

## Related Work & Insights

- **vs. DAGMA/NOTEARS**: This paper directly extends the DAGMA framework by incorporating higher-order terms to handle general discrete distributions and provides theoretical guarantees absent in prior work.
- **vs. PC/GES**: Traditional methods rely on discrete conditional independence tests or BIC scores and lack flexibility.
- **vs. Zeng et al. 2022**: That method assumes a generalized linear SEM and is a special case of the proposed framework (first-order interactions only).
- **Inspiration**: The higher-order interaction perspective for binary variables can be transferred to other discrete structure learning problems, such as gene regulatory network discovery.

## Rating
- Novelty: ⭐⭐⭐⭐ (General MVB parameterization + complete non-identifiability characterization are novel contributions)
- Experimental Thoroughness: ⭐⭐⭐ (Synthetic experiments are thorough, but large-scale and real-data experiments are limited)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear and motivation is well-articulated)
- Value: ⭐⭐⭐⭐ (Fills the theoretical gap in differentiable causal discovery for discrete data)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)

</div>

<!-- RELATED:END -->
