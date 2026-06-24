---
title: >-
  [Paper Note] Improving Decision Trees through the Lens of Parameterized Local Search
description: >-
  [NeurIPS 2025][Decision Trees] This paper analyzes local search operations for decision tree optimization through the lens of parameterized complexity, identifies the sources of computational hardness, and proves that the combination of the number of features and domain size yields fixed-parameter tractability (FPT), accompanied by a proof-of-concept implementation.
tags:
  - "NeurIPS 2025"
  - "Decision Trees"
  - "Parameterized Complexity"
  - "Local Search"
  - "NP-Completeness"
  - "Fixed-Parameter Tractability"
date: 2026-05-08
content_hash: 589046eb866ca271
---

# Improving Decision Trees through the Lens of Parameterized Local Search

**Conference**: NeurIPS 2025
**arXiv**: [2510.12726](https://arxiv.org/abs/2510.12726)  
**Authors**: Juha Harviainen, Frank Sommer, Manuel Sorge
**Code**: Available (proof-of-concept implementation)  
**Area**: Machine Learning Theory / Decision Tree Optimization
**Keywords**: Decision Trees, Parameterized Complexity, Local Search, NP-Completeness, Fixed-Parameter Tractability

## TL;DR

This paper analyzes local search operations for decision tree optimization through the lens of parameterized complexity, identifies the sources of computational hardness, and proves that the combination of the number of features and domain size yields fixed-parameter tractability (FPT), accompanied by a proof-of-concept implementation.

## Background & Motivation

Decision trees are among the most fundamental and interpretable models in machine learning. Algorithms for learning decision trees typically incorporate heuristic local search operations to improve tree quality:

**Threshold Adjustment**: Modifying the split threshold at an internal node.

**Feature Exchange**: Simultaneously replacing the split feature and threshold at a node.

Although these operations are widely used in classical algorithms such as CART and C4.5, their theoretical computational complexity has remained poorly understood. The core question is:

**Given a decision tree and a dataset, what is the computational complexity of minimizing the number of misclassifications by performing a fixed number of local operations?**

Is an intuitively simple operation—such as adjusting a single threshold—also computationally easy in the general case? Which problem parameters render the problem tractable or intractable?

## Method

### Overall Architecture

This paper conducts a systematic parameterized complexity analysis of two classes of local search optimization problems:

- **Problem 1 (THRESHOLD-ADJUSTMENT)**: Given a decision tree $T$, find at most $k$ threshold adjustment operations that minimize the number of misclassifications.
- **Problem 2 (FEATURE-EXCHANGE)**: Given a decision tree $T$, find at most $k$ feature exchange operations (which may simultaneously change both the feature and the threshold) that minimize the number of misclassifications.

### Key Designs

#### 1. NP-Completeness Results

**Theorem**: Both problems are NP-complete in the general case.

Specifically:
- Even when the number of operations is $k = 1$ (a single modification), the problem remains NP-hard.
- This shows that even the simplest local search step is computationally hard on general decision trees.

#### 2. Hardness Parameter Analysis

The authors identify key parameters governing problem difficulty:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Number of features | $d$ | Dimensionality of the dataset |
| Domain size | $D$ | Maximum number of distinct values per feature |
| Number of operations | $k$ | Number of local search operations |
| Tree size | $|T|$ | Number of nodes in the decision tree |

Key hardness results:

- **Small $d$** (few features): Problem remains NP-hard → few features do not make the problem easy.
- **Small $D$** (small domain): Problem remains NP-hard → small domain does not make the problem easy.
- **Small $k$** (few operations): Problem remains NP-hard → few operations are also insufficient.

#### 3. Fixed-Parameter Tractability (FPT) Results

The central positive result: **the combination of $d$ and $D$** yields an FPT algorithm.

**Theorem**: Both problems can be solved in $(D+1)^{2d} \cdot |I|^{O(1)}$ time, where $|I|$ is the input size.

Intuition:
- When both $d$ and $D$ are fixed, the number of possible splits at each internal node is bounded (at most $d \times D$).
- For $k$ operations, the search space has size $(dD)^k$.
- The actual algorithm exploits finer structure, yielding complexity $(D+1)^{2d}$ multiplied by a polynomial factor.

#### 4. Parameterized Complexity Landscape

| Parameter Combination | Threshold Adjustment | Feature Exchange | Classification |
|-----------------------|----------------------|------------------|----------------|
| $d$ alone | NP-hard | NP-hard | para-NP-hard |
| $D$ alone | NP-hard | NP-hard | para-NP-hard |
| $k$ alone | NP-hard | NP-hard | para-NP-hard |
| $d + D$ | **FPT** | **FPT** | $(D+1)^{2d} \cdot |I|^{O(1)}$ |
| $d + k$ | Open | Open | To be studied |
| $D + k$ | Open | Open | To be studied |
| $d + D + k$ | **FPT** | **FPT** | Tighter bound |

### FPT Algorithm Design

The core mechanism of the algorithm:

1. **Enumerate possible split sets**: For each feature $f \in \{1,\ldots,d\}$ and each threshold $\theta$ drawn from the domain $\{1,\ldots,D\}$, enumerate all possible $(f, \theta)$ pairs.
2. **For each internal node**: Consider at most $(D+1)^{2d}$ distinct split configurations.
3. **Dynamic programming or exhaustive search**: Under the allowed number of operations $k$, identify the optimal combination of modifications.
4. **Error count update**: For each modification scheme, efficiently compute the updated number of misclassifications.

### Loss & Training

The objective is minimization of the misclassification count (0-1 loss), which differs from the Gini index or information gain commonly used in decision tree learning. Directly optimizing 0-1 loss is more natural theoretically but harder computationally—this is precisely the central subject of the paper.

## Key Experimental Results

### Main Results

#### Proof-of-Concept for the FPT Algorithm

The authors implement the FPT algorithm and evaluate it on several UCI datasets:

| Dataset | Features $d$ | Domain $D$ | Samples $n$ | FPT Runtime | Error Reduction |
|---------|-------------|-----------|------------|-------------|----------------|
| Iris | 4 | Continuous→Discretized | 150 | Seconds | Significant |
| Wine | 13 | Continuous→Discretized | 178 | Minutes | Moderate |
| Glass | 9 | Continuous→Discretized | 214 | Slow | Moderate |
| Synthetic (low $d$, $D$) | 3–5 | 3–5 | 100–1000 | Seconds | Significant |
| Synthetic (high $d$ or $D$) | 10+ | 10+ | 100–1000 | Infeasible | — |

Core observation: The practical runtime of the FPT algorithm is consistent with theoretical predictions—feasible when both $d$ and $D$ are small, and subject to exponential blow-up when either parameter grows.

#### Comparison with Heuristic Methods

| Method | Objective | Solution Quality | Runtime | Guarantee |
|--------|-----------|-----------------|---------|-----------|
| CART Greedy | Gini index | Heuristically optimal | Fast | None |
| Greedy Local Search | 0-1 loss | Locally optimal | Fast | Local |
| **FPT Algorithm** | **0-1 loss** | **Globally optimal** | **Depends on $d$, $D$** | **Global** |
| Brute Force | 0-1 loss | Globally optimal | Exponential | Global |

### Ablation Study

#### Effect of Parameters $d$ and $D$ on Runtime

| $d$ | $D$ | $(D+1)^{2d}$ | Actual Runtime |
|-----|-----|--------------|----------------|
| 3 | 3 | $4^6 = 4096$ | < 1 second |
| 3 | 5 | $6^6 \approx 4.7 \times 10^4$ | A few seconds |
| 5 | 3 | $4^{10} \approx 10^6$ | Minutes |
| 5 | 5 | $6^{10} \approx 6 \times 10^7$ | Hours |
| 10 | 3 | $4^{20} \approx 10^{12}$ | Infeasible |
| 10 | 10 | $11^{20} \approx 10^{20}$ | Completely infeasible |

### Key Findings

1. **Ubiquity of NP-completeness**: Even the simplest local operation ($k=1$) renders decision tree optimization NP-hard in the general case.
2. **Synergistic effect of $d$ and $D$**: Neither small $d$ nor small $D$ alone suffices to make the problem tractable, but their combination does.
3. **Limited practical utility of the FPT algorithm**: Due to the exponential factor $(D+1)^{2d}$, the algorithm is practically applicable only to low-dimensional discrete data.
4. **Value of theoretical insight**: The analysis clarifies which problem properties make local search hard or easy.

## Highlights & Insights

1. **Clear theoretical contribution**: A comprehensive parameterized complexity landscape that cleanly delineates which parameters make the problem hard versus tractable.
2. **Bridging practice and theory**: Local search is a core component of all mainstream decision tree algorithms; this paper provides the first fine-grained computational complexity analysis thereof.
3. **Positive signal from FPT**: Although the algorithm is impractical in the general case, it establishes that exact optimization of decision trees is feasible for low-dimensional discrete data.
4. **Implications for algorithm design**: The results explain why methods such as CART resort to heuristics—optimizing decision tree splits is intrinsically hard in the general case.
5. **Natural problem parameters**: The analysis reveals that $d$ and $D$ are the "right" parameters governing the complexity of decision tree optimization.

## Limitations & Future Work

1. **Limited scalability**: The exponential factor in the FPT algorithm is too large for typical ML datasets ($d > 10$, continuous $D$).
2. **Handling continuous features**: Requires discretization preprocessing, introducing information loss.
3. **Open problems**: The parameterized complexity of the $d + k$ and $D + k$ parameter combinations remains unresolved.
4. **Relation to ensemble methods**: The implications for ensemble methods such as random forests are not discussed.
5. **Approximation algorithms**: No approximation guarantees under relaxed conditions are provided.
6. **Experimental scale**: The evaluation is proof-of-concept in nature, conducted only on small-scale datasets.

## Related Work & Insights

- **CART** (Breiman et al., 1984): The classical decision tree learning algorithm whose local search steps are the subject of this analysis.
- **Optimal decision trees** (Bertsimas & Dunn, 2017; Hu et al., 2019): MIP-based methods for learning optimal decision trees, complementary to the exact algorithms presented here.
- **Parameterized complexity theory** (Downey & Fellows, 2013): The theoretical framework employed in this paper.
- **Top-down induction** (Quinlan, 1986): Traditional algorithms such as C4.5 whose greedy local search complexity is clarified by this work.
- **NP-hardness of DT learning** (Hyafil & Rivest, 1976): The classical NP-hardness result for decision tree learning, which this paper further refines.

## Rating

- **Novelty**: ★★★★☆ — The first parameterized complexity analysis of local search for decision trees.
- **Theoretical Depth**: ★★★★★ — NP-completeness, FPT results, and a comprehensive parameterized landscape.
- **Experimental Thoroughness**: ★★★☆☆ — Proof-of-concept in nature, limited in scale.
- **Value**: ★★★☆☆ — Theoretically insightful, though practical utility of the FPT algorithm is constrained.
- **Writing Quality**: ★★★★☆ — Theoretical results are clearly organized and proofs are concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](regression_trees_know_calculus.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](../../AAAI2026/others/approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](../../AAAI2026/others/from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](rethinking_pca_through_duality.md)

</div>

<!-- RELATED:END -->
