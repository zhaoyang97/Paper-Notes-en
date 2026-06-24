---
title: >-
  [Paper Note] Leveraging Predictive Equivalence in Decision Trees
description: >-
  [ICML2025][Interpretability][Decision Trees] Proposed converting decision trees into minimal Disjunctive Normal Form (DNF) representations to eliminate the "predictive equivalence" problem, unifying the representation of different decision trees with identical decision boundaries, thereby improving variable importance metrics, robustness to missing data, and feature acquisition cost optimization.
tags:
  - "ICML2025"
  - "Interpretability"
  - "Decision Trees"
  - "Predictive Equivalence"
  - "Disjunctive Normal Form (DNF)"
  - "Quine-McCluskey"
  - "Variable Importance"
  - "Missing Data"
  - "Rashomon Set"
date: 2026-05-08
content_hash: 10c78a60f822918a
---

# Leveraging Predictive Equivalence in Decision Trees

**Conference**: ICML2025  
**arXiv**: [2506.14143](https://arxiv.org/abs/2506.14143)  
**Code**: TBD  
**Area**: Decision Trees / Interpretable Machine Learning  
**Keywords**: Decision Trees, Predictive Equivalence, Disjunctive Normal Form (DNF), Quine-McCluskey, Variable Importance, Missing Data, Rashomon Set

## TL;DR

Proposed converting decision trees into minimal Disjunctive Normal Form (DNF) representations to eliminate the "predictive equivalence" problem, unifying the representation of different decision trees with identical decision boundaries, thereby improving variable importance metrics, robustness to missing data, and feature acquisition cost optimization.

## Background & Motivation

Decision trees are widely used in interpretable machine learning due to their clear decision structures. However, they suffer from a subtle yet crucial core problem — **Predictive Equivalence**: multiple decision trees with different structures can represent the exact same decision boundary. For instance, the logical AND function $X_1 \wedge X_2$ can be evaluated by checking $X_1$ before $X_2$, or vice versa. The predictive outcomes of both trees are identical, but their structures differ.

This phenomenon poses four practical challenges:

**Handling missing data**: If $x_1$ is missing but $x_2=0$, a tree that evaluates $x_1$ first cannot be traversed, whereas a tree evaluating $x_2$ first can directly output 0.

**Unreliable variable importance**: Tree-structure-based metrics like Gini importance yield completely different results on equivalent trees.

**Rashomon set inflation**: Equivalent trees are counted repeatedly within the near-optimal model set, leading to biases in downstream analysis.

**Suboptimal feature acquisition cost**: The sequence of variable evaluation implied by a tree might not be cost-optimal.

## Method

### Mechanism: Decision Tree → Minimal DNF

Given a decision tree $\mathcal{T}$, it is converted into a simplified Disjunctive Normal Form (DNF, i.e., "OR-of-ANDs") representation $\mathcal{T}_{\text{DNF}}$, abstracting away the evaluation order.

**Algorithm Flow (Algorithm 1)**:

1. Collect all leaf paths $L^+$ predicting the positive class, constructing the positive expression $PosExpr = \bigvee_{l \in L^+} l$.
2. Collect all leaf paths $L^-$ predicting the negative class, constructing the negative expression $NegExpr = \bigvee_{l \in L^-} l$.
3. Simplify both expressions using an improved Quine-McCluskey algorithm:

$$SimplePosExpr = \text{QuineMcCluskey}^*(PosExpr)$$
$$SimpleNegExpr = \text{QuineMcCluskey}^*(NegExpr)$$

4. Return $(SimplePosExpr, SimpleNegExpr)$ as $\mathcal{T}_{\text{DNF}}$.

### Key Designs

- **Faithfulness (Prop 3.1)**: For a complete sample $x$, $\mathcal{T}_{\text{DNF}}(x) = \mathcal{T}(x)$.
- **Completeness (Thm 3.2)**: $\mathcal{T}_{\text{DNF}}(m(x)) \neq \text{NA}$ if and only if for all completions $z$ of $m(x)$, $\mathcal{T}(z) = \mathcal{T}_{\text{DNF}}(m(x))$.
- **Conciseness (Prop 3.3)**: No redundant variables are included in the explanation.
- **Equivalence Resolution (Thm 3.4)**: $\mathcal{T}$ and $\mathcal{T}'$ are predictively equivalent $\Leftrightarrow$ $\mathcal{T}_{\text{DNF}} = \mathcal{T}'_{\text{DNF}}$.

### Blake Canonical Form

A minimal DNF does not necessarily contain all prime implicants. The paper further introduces the **Blake Canonical Form (BCF)**, which enumerates all minimal sufficient conditions to simplify predictive logic — requiring only a term-by-term satisfaction check without further logical simplification.

### Prediction Process (Algorithm 2)

For a sample $m(x)$ with missing values:

1. Check if any term in $SimplePosExpr$ is satisfied by the known values $\rightarrow$ predict 1.
2. Check if any term in $SimpleNegExpr$ is satisfied by the known values $\rightarrow$ predict 0.
3. Substitute known values into $SimplePosExpr$ and simplify again; if it resolves to True/False, return the corresponding prediction.
4. Otherwise, return NA.

### Cost Optimization

Using Q-learning to optimize the feature acquisition order: given the acquisition cost $c_j$ for each feature, learn the optimal policy to minimize the total cost required to reach a prediction. The DNF representation frees the feature acquisition order from the constraints of the tree structure.

## Key Experimental Results

### Rashomon Set Deduplication

| Dataset | Total Trees | Remove Trivial Redundancy | Remove Predictive Equivalence (Ours) |
|--------|--------|-------------|-------------------|
| COMPAS | 12785±3e3 | 3913±837 | **2135±448** |
| Coupon | 666±54 | 136±19 | **55±8** |
| Wine | 6936±700 | 2341±377 | **1409±256** |
| Wisconsin | 24052±9e3 | 11990±5e3 | **4657±2e3** |

A large number of decision trees in Rashomon sets are actually predictively equivalent. Deduplication significantly reduces the scale (e.g., from 666 trees to 55 for Coupon).

### Variable Importance Correction

- On synthetic data $Y = X_1 X_2$, where $X_1, X_2 \sim \text{Bernoulli}(0.5)$, two equivalent trees give completely opposite Gini importance (one considers $X_0$ twice as important as $X_1$, and vice versa).
- Correction of RID (Rashomon Importance Distribution): Eliminating equivalent trees reduces the 1-Wasserstein distance between all variable importance estimates and the true values.

| Method | $X_1$ Distance | $X_2$ Distance | $X_3$ Distance |
|------|-----------|-----------|-----------|
| Original RID | 0.120 | 0.136 | 0.232 |
| Corrected RID | **0.092** | **0.105** | **0.182** |

### Robustness to Missing Data

- The DNF representation directly provides predictions under substantial missing data settings without needing imputation.
- Key corollary: If $\mathcal{T}_{\text{DNF}}(m(x)) \neq \text{NA}$, its prediction matches the result of a perfect oracle imputation.
- Even in MNAR (Missing Not At Random) settings, DNF predictions remain unbiased estimates.

## Highlights & Insights

1. **Precise Problem Definition**: This work systematically defines and quantifies the predictive equivalence problem in decision trees for the first time, revealing that seemingly different trees are actually different "writings" of the same classifier.
2. **Novel Representational Approach**: Translating decision trees into minimal boolean logic representations bridges the gap between tree structures and decision boundaries.
3. **One Representation for Multiple Problems**: A single DNF representation simultaneously improves three downstream tasks: variable importance, handling missing data, and cost optimization.
4. **Complete Theoretical Guarantees**: Four theorems (faithfulness, completeness, conciseness, and equivalence resolution) establish a rigorous theoretical foundation.
5. **Significant Practical Value**: Decision trees are widely utilized in high-risk areas such as healthcare and justice; resolving equivalence is crucial for correct model interpretation.

## Limitations & Future Work

1. **Computational Complexity**: Finding the minimal DNF is NP-hard. When a decision tree has many leaves, the Quine-McCluskey algorithm incurs a high computational overhead. Consequently, the paper relies on the assumption of small trees (the maximum depth in experiments is only 3).
2. **Only Binary Classification**: While extension to multi-class classification is mentioned, no concrete multi-class experiments are provided.
3. **Limited to Binarized Features**: Continuous features must be binarized beforehand, which may introduce information loss.
4. **Cost Optimization Relies on Q-learning**: Convergence and efficiency in large feature spaces are not fully discussed.
5. **Limited Dataset Scale**: The four main datasets have low feature dimensions (7-30), and performance in high-dimensional scenarios remains to be validated.

## Related Work & Insights

- **Decision Tree Optimization**: Optimal decision tree algorithms such as GOSDT (Lin et al., 2020), DL8.5 (Aglin et al., 2020), and MurTree (Demirović et al., 2022).
- **Redundant Path Explanations**: Izza et al. (2022) identified redundant variable issues in decision tree paths; this work addresses it from a global representation level.
- **Rashomon Set Analysis**: TreeFARMS (Xin et al., 2022), RID (Donnelly et al., 2023).
- **Missing Data**: MINTY (Stempfle & Johansson, 2024) increases missingness robustness with logical disjunction, sharing a similar concept but this work is more general.
- **Insight**: A novel application of boolean logic simplification techniques (e.g., Quine-McCluskey from classic electronic design automation) to ML interpretability.

## Rating

- Novelty: ⭐⭐⭐⭐ — The systematization of the predictive equivalence concept and the application of DNF representation in ML interpretability are quite original.
- Experimental Thoroughness: ⭐⭐⭐ — Theoretical validation is solid, but the datasets are small and the scenarios are limited.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, rigorous theorem formulation, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Offers substantial advancement to the study of decision tree interpretability, with practical value in high-risk application domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Near-Optimal Decision Trees in a SPLIT Second](near_optimal_decision_trees_in_a_split_second.md)
- [\[NeurIPS 2025\] Empowering Decision Trees via Shape Function Branching](../../NeurIPS2025/interpretability/empowering_decision_trees_via_shape_function_branching.md)
- [\[ICLR 2026\] TreeGrad-Ranker: Feature Ranking via O(L)-Time Gradients for Decision Trees](../../ICLR2026/interpretability/treegrad-ranker_feature_ranking_via_ol-time_gradients_for_decision_trees.md)
- [\[NeurIPS 2025\] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement](../../NeurIPS2025/interpretability/far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)

</div>

<!-- RELATED:END -->
