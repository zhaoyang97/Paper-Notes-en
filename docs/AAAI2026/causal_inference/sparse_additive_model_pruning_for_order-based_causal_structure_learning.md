---
title: >-
  [Paper Note] Sparse Additive Model Pruning for Order-Based Causal Structure Learning
description: >-
  [AAAI2026][Causal Inference][causal discovery] This paper proposes SARTRE, a framework that employs randomized tree embeddings and group-sparse regression to learn sparse additive models…
tags:
  - "AAAI2026"
  - "Causal Inference"
  - "causal discovery"
  - "sparse additive model"
  - "DAG pruning"
  - "group lasso"
  - "randomized tree embedding"
date: 2026-05-08
content_hash: d0473e9a25d9572a
---

# Sparse Additive Model Pruning for Order-Based Causal Structure Learning

**Conference**: AAAI2026
**arXiv**: [2602.15306](https://arxiv.org/abs/2602.15306)  
**Code**: To be confirmed  
**Area**: Causal Inference
**Keywords**: causal discovery, sparse additive model, DAG pruning, group lasso, randomized tree embedding

## TL;DR

This paper proposes SARTRE, a framework that employs randomized tree embeddings and group-sparse regression to learn sparse additive models, replacing the hypothesis-testing-based redundant edge pruning in CAM-pruning for order-based causal structure learning. SARTRE achieves significant speedups without sacrificing accuracy.

## Background & Motivation

Causal structure learning aims to recover the causal directed acyclic graph (DAG) among variables from observational data. Order-based approaches constitute one of the dominant frameworks, proceeding in two steps:

1. **Ordering step**: Estimate a topological ordering of the DAG; for example, the SCORE algorithm iteratively identifies leaf nodes via score matching.
2. **Pruning step**: Remove spurious edges from the fully connected DAG induced by the topological order to identify the true parent set of each variable.

Most existing methods adopt CAM-pruning for this step, which fits a generalized additive model (GAM) for each variable and then applies hypothesis tests to assess whether each candidate parent is redundant. However, this approach has two critical bottlenecks:

- **High computational cost**: Repeatedly fitting GAMs for all variables becomes the algorithmic bottleneck in high-dimensional settings.
- **Multiple testing problem**: Repeated hypothesis tests for each variable–candidate-parent pair may degrade estimation accuracy due to multiple comparisons.

## Core Problem

How to design an efficient and accurate pruning method that avoids the overhead of repeatedly fitting GAMs and conducting multiple hypothesis tests in CAM-pruning, while maintaining or improving the accuracy of causal graph estimation?

## Method

### Overall Architecture of SARTRE

The core idea of SARTRE (Sparse Additive Randomized TRee Ensemble) is to learn sparse additive models that regress each variable onto its candidate parents, thereby directly pruning redundant edges by examining whether the corresponding coefficients are zero—without any hypothesis testing. The framework consists of two stages.

### Stage 1: Randomized Tree Embedding

For each variable $X_j$, a fully randomized decision tree ensemble $h_j$ is constructed. These trees select split points randomly in an unsupervised manner, requiring no target variable, and can therefore be generated extremely efficiently. Each leaf node of a tree corresponds to an interval $r_{j,k}$ over the domain of $X_j$; collecting all leaf intervals yields the interval set $R_j = \{r_{j,1}, \dots, r_{j,l_j}\}$.

An indicator function embedding is defined based on this interval set:

$$\phi_j(X_j) = (\mathbb{I}[X_j \in r_{j,1}], \dots, \mathbb{I}[X_j \in r_{j,l_j}]) \in \{0,1\}^{l_j}$$

**Key advantage**: The interval set $R_j$ is generated independently of any target variable, so the embedding for each variable needs to be computed only once and can be reused across all regression problems in which $X_j$ appears as a candidate parent.

### Stage 2: Group-Sparse Regression

For each variable $X_i$, an additive model is fitted by regressing $X_i$ onto its candidate parents:

$$\hat{f}_i(X_{\hat{pa}(i)}) = \sum_{j \in \hat{pa}(i)} g_{i,j}(X_j), \quad g_{i,j}(X_j) = \boldsymbol{\beta}_{i,j}^\top \phi_j(X_j)$$

After concatenating the embeddings and coefficient vectors of all candidate parents, the following group lasso regression is solved:

$$\min_{\boldsymbol{\beta}_i} \sum_{m=1}^{n} (x_{m,i} - \boldsymbol{\beta}_i^\top \Phi_i(\boldsymbol{x}_m))^2 + \lambda \sum_{j \in \hat{pa}(i)} \|\boldsymbol{\beta}_{i,j}\|_2$$

The group lasso penalty applies an $\ell_2$-norm penalty to each group of coefficients corresponding to one candidate parent, encouraging entire groups to be driven simultaneously to zero. When $\boldsymbol{\beta}_{i,j} = \boldsymbol{0}$, the shape function $g_{i,j}(X_j) = 0$ for all $X_j$, and the edge $X_j \to X_i$ is directly removed.

### Theoretical Guarantees

The paper establishes a representational capacity result (Proposition 1): shape functions based on interval indicator embeddings can approximate any continuous function to arbitrary precision. Although structurally simpler than smooth splines, piecewise constant functions provide a universal approximation theoretical foundation.

## Key Experimental Results

### Synthetic Data (ER/SF graphs, $n=2000$)

| Setting | Method | SHD | SID | Time (s) |
|---------|--------|-----|-----|----------|
| ER1, $d=50$ | SCORE | ~30 | ~250 | ~120 |
| ER1, $d=50$ | DAS | ~28 | ~240 | ~80 |
| ER1, $d=50$ | SARTRE | **~20** | **~200** | **~10** |

- On sparse graphs (ER1/SF1), SARTRE achieves **better** SHD and SID than all baselines.
- On dense graphs (ER4/SF4), accuracy is **comparable**, while runtime is an order of magnitude faster.

### High-Dimensional Experiments ($d \in \{64, 128, 256, 512\}$, using ground-truth topological order)

SARTRE is faster than DAS across all dimensionalities and achieves lower SHD and SID.

### Real-World Datasets

| Dataset | Method | SHD | SID | Time (s) |
|---------|--------|-----|-----|----------|
| Sachs | SCORE | 43.2 | 102.4 | 14.7 |
| Sachs | DAS | 27.6 | 70.3 | 6.42 |
| Sachs | **SARTRE** | **22.7** | **58.0** | **3.62** |
| fMRI | SCORE | 19.6 | 71.8 | 11.4 |
| fMRI | DAS | 11.6 | 58.6 | 4.75 |
| fMRI | **SARTRE** | 12.9 | 60.0 | **3.18** |

## Highlights & Insights

1. **Elegant and efficient design**: Replacing GAM + hypothesis testing with randomized tree embeddings + group lasso eliminates two computational bottlenecks.
2. **Embedding reuse**: Interval sets are generated in an unsupervised manner and reused across all target variables, avoiding redundant model fitting.
3. **No hypothesis testing required**: Group sparsity induced by the group lasso directly determines edge redundancy, eliminating the multiple testing problem.
4. **Plug-and-play compatibility**: SARTRE can be combined with any topological order estimation algorithm (SCORE, CAM, CaPS, etc.).
5. **High-dimensional scalability**: The method performs well even at $d=512$, outperforming DAS.

## Limitations & Future Work

- **Hyperparameter selection**: The regularization parameter $\lambda$ and the number of intervals $l_j$ must be specified in advance; no adaptive tuning mechanism is provided.
- **Limited theoretical guarantees**: Only representational capacity is proven; no theoretical guarantee of pruning correctness is established.
- **Strong modeling assumptions**: The method assumes an additive noise model (ANM) with nonlinear differentiable link functions and does not account for latent confounders.
- **Slightly lower accuracy on dense graphs**: For dense DAGs (ER4) with large sample sizes, the hypothesis-testing approach of CAM-pruning may be more precise.
- Future work could explore adaptive hyperparameter tuning and robustness in the presence of hidden variables.

## Related Work & Insights

| Method | Pruning Strategy | Hypothesis Testing Required | Computational Complexity | High-Dim. Performance |
|--------|-----------------|----------------------------|--------------------------|-----------------------|
| CAM-pruning | GAM + p-value testing | Yes | High (repeated GAM fitting) | Poor |
| DAS | Score-matching pre-screening + CAM-pruning | Yes | Moderate | Moderate |
| **SARTRE** | Random tree embedding + group lasso | **No** | **Low** | **Strong** |

Unlike tree-based GAM learning frameworks such as EBM (Lou et al. 2013), SARTRE explicitly encourages group sparsity for variable selection and is specifically designed for causal graph pruning.

The combination of **randomized tree embeddings and linear models** represents a broadly applicable paradigm: unsupervised generation of nonlinear feature representations followed by convex optimization with theoretical guarantees strikes a balance between efficiency and interpretability. The application of group lasso's structured sparsity to causal discovery can be generalized to other settings requiring structured variable selection. Furthermore, the embedding reuse strategy may inspire analogous computation-sharing optimizations in other pipelines that involve repeated model fitting.

## Rating

- Novelty: ⭐⭐⭐⭐ — Combining randomized tree embeddings with group-sparse regression for causal pruning is a novel and natural contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of synthetic and real-world datasets, including high-dimensional experiments and ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, figures are intuitive, and the algorithmic description is complete.
- Value: ⭐⭐⭐⭐ — Directly applicable to order-based causal discovery pipelines, with the SARTRE framework also usable independently for nonlinear variable selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](../../NeurIPS2025/causal_inference/differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](../../ICML2026/causal_inference/investigating_memory_in_model-free_rl_with_popgym_arcade.md)

</div>

<!-- RELATED:END -->
