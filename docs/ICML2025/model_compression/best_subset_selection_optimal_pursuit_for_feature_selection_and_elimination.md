---
title: >-
  [Paper Note] Best Subset Selection: Optimal Pursuit for Feature Selection and Elimination
description: >-
  [ICML 2025][Model Compression][Best Subset Selection] This paper revisits the feature selection/elimination criteria in classical best subset selection from an optimization perspective. It reveals that traditional criteria (correlation selection + Wald-T elimination) only capture "one-step changes" in the objective function while neglecting feature interactions. Consequently, the authors propose "objective-aware" optimal selection and elimination criteria to enhance classic a…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Best Subset Selection"
  - "Optimal Pursuit"
  - "Feature Selection"
  - "Compressed Sensing"
  - "Sparse Regression"
date: 2026-05-08
content_hash: 6c98a8d73d442b31
---

# Best Subset Selection: Optimal Pursuit for Feature Selection and Elimination

**Conference**: ICML 2025  
**arXiv**: [2501.16815](https://arxiv.org/abs/2501.16815)  
**Code**: [https://github.com/ZhihanZhu-math/Optimal_Pursuit_public](https://github.com/ZhihanZhu-math/Optimal_Pursuit_public)  
**Area**: Model Compression / Feature Selection / Sparse Optimization  
**Keywords**: Best Subset Selection, Optimal Pursuit, Feature Selection, Compressed Sensing, Sparse Regression

## TL;DR

This paper revisits the feature selection/elimination criteria in classical best subset selection from an optimization perspective. It reveals that traditional criteria (correlation selection + Wald-T elimination) only capture "one-step changes" in the objective function while neglecting feature interactions. Consequently, the authors propose "objective-aware" optimal selection and elimination criteria to enhance classic algorithms such as OMP, CoSaMP, and (A)BESS as a plug-and-play Meta-Substitution. This achieves significant performance improvements in compressed sensing and sparse regression tasks without increasing computational complexity.

## Background & Motivation

**Background**: Best subset selection is a benchmark problem in statistics and machine learning, aiming to select the optimal feature subset under an $\ell_0$ constraint. Since this problem is NP-hard, practitioners mainly rely on greedy algorithms (such as OMP, CoSaMP, IHT, ABESS, etc.) to solve it through iterative feature selection and elimination.

**Limitations of Prior Work**: Feature selection in greedy algorithms is typically based on the **correlation criterion** (selecting features most correlated with the residual), and feature elimination is based on the **Wald-T statistic** (removing features with the smallest T-value). However, these two classical criteria only focus on the importance of individual features, ignoring the interactions between features—a feature that appears important in the current active set might become unimportant after the active set changes, and vice-versa.

**Key Challenge**: From an optimization perspective, classical criteria are equivalent to performing a one-step block coordinate descent on the feature selection/elimination subproblems, which only considers the objective function change in the step of "updating the support set while fixing the coefficients" and ignores the additional objective function change brought by "updating the coefficients on the new support set." In other words, classical criteria pursue local one-step optimality rather than global optimality.

**Goal**: Design optimal criteria that can accurately solve the feature entry/exit subproblems, making every selection/elimination step an optimal decision that considers the joint effect of both "support set update + coefficient update".

**Key Insight**: The authors unify the classical criteria under the framework of optimization subproblems (P0)/(Q0), reveal their limitations, and then construct more complete optimization subproblems (P1)/(Q1), deriving closed-form optimal criteria through exact solutions.

**Core Idea**: Replace the feature selection/elimination criteria in classical greedy subset selection with the exact optimal solutions to the selection/elimination subproblems, seamlessly enhancing all greedy subset selection algorithms via "Meta-Substitution."

## Method

### Overall Architecture

The methodology framework of this paper can be summarized in three steps:

1. **Analyze the Optimization Essence of Classical Criteria**: Prove that the classical correlation selection criterion (Eq. 3) and Wald-T elimination criterion (Eq. 4) correspond to optimization subproblems (P0) and (Q0) respectively—they only allow the support set to change by one element while fixing the remaining coefficients, which is essentially one-step block coordinate descent.
2. **Construct and Exactly Solve Full Subproblems**: Propose new optimization subproblems (P1)/(Q1), which allow the support set to change by one element while simultaneously updating all coefficients (i.e., considering the "full descent"), and derive closed-form optimal criteria using matrix inversion lemmas.
3. **Meta-Substitution**: Use the new criteria as plug-and-play modules to replace the old criteria in classical algorithms, generating a series of enhanced algorithms (OP, CoSaOP, OP-(A)BESS, etc.) without increasing the order of computational complexity.

### Key Designs

1. **Objective-based Selection (Eq. 8)**:

    - **Function**: Select an optimal feature from outside the support set to be added.
    - **Mechanism**: Solve the equivalent form (P2) of the selection subproblem (P1), and utilize the forward matrix inversion lemma (Lemma 3.2) to derive the closed-form criterion:
    $j^* = \arg\max_{j \in S^c} \frac{(\mathbf{r}^T \mathbf{X}_j)^2}{\mathbf{X}_j^T (\mathbf{I} - \mathbf{X}_S (\mathbf{X}_S^T \mathbf{X}_S)^{-1} \mathbf{X}_S^T) \mathbf{X}_j}$
      The numerator is the same as the classical criterion (correlation between residual and feature), but the denominator introduces a projection matrix $\mathbf{I} - \mathbf{X}_S (\mathbf{X}_S^T \mathbf{X}_S)^{-1} \mathbf{X}_S^T$, which is the orthogonal complement space of the currently selected features.
    - **Design Motivation**: The new criterion can be interpreted as the correlation between feature $\mathbf{X}_j$ and the residual in the noise subspace of $\mathbf{X}_S$, capturing the "comprehensive combinatorial effect" among features. It degrades to the classical criterion when $\mathbf{X}_j$ is orthogonal to $\mathbf{X}_S$.
    - **Difference from Prior Methods**: The denominator of the classical criterion is only $\|\mathbf{X}_j\|_2$, neglecting the correlation between candidate features and already selected features; the new criterion automatically penalizes candidate features highly correlated with the selected features through the projection matrix.

2. **Objective-based Elimination (Eq. 10)**:

    - **Function**: Select the least important feature from inside the support set to be removed.
    - **Mechanism**: Solve the equivalent form (Q2) of the elimination subproblem (Q1), and apply the backward matrix inversion lemma (Lemma 3.10) to derive the criterion: select the feature for elimination such that the remaining features after removal can still maximize the explanation of $\mathbf{y}$. The specific formula involves block updates of $\mathbf{C}_{k-1} = (\mathbf{X}_S^T \mathbf{X}_S)^{-1}$.
    - **Design Motivation**: The classical Wald-T criterion only looks at the magnitude of $|\beta_j|$ to judge feature importance, ignoring the impact of removing a feature on the coefficients of other features. The new criterion comprehensively considers the impact on the objective function when the coefficients of the remaining features are readjusted after eliminating feature $j$.
    - **Difference from Prior Methods**: It degrades to the classical Wald-T criterion when features are orthogonal, but significantly avoids mistakenly deleting true features when features are highly correlated.

3. **Meta-Substitution**:

    - **Function**: Apply the above optimal criteria as a general module to replace classical criteria in any greedy subset selection algorithm.
    - **Mechanism**: Greedy subset selection algorithms can be categorized into three types: selection-only (e.g., OMP $\rightarrow$ OP), forward-backward (e.g., CoSaMP $\rightarrow$ CoSaOP), and exchange-based (e.g., ABESS $\rightarrow$ OP-ABESS). For any of these algorithms, one only needs to replace the classical criteria in their selection/elimination steps with the optimal criteria.
    - **Design Motivation**: The matrix inverse $(\mathbf{X}_S^T \mathbf{X}_S)^{-1}$ required for calculating the optimal criteria is already computed during the least-squares step of coefficient updates. Therefore, the substitution does not increase the order of computational complexity.

### Theoretical Analysis

- **Optimality Guarantee (Theorem 4.1 & 4.2)**: The new criteria make optimal decisions at each forward/backward step, meaning the selected feature decreases the objective function the most, and the removed feature increases it the least.
- **Complexity Equivalence (Theorem 4.3)**: Enhanced algorithms (OP, CoSaOP, OP-ABESS) share the same order of computational complexity as the original algorithms.
- **CoSaOP Convergence (Theorem 4.9)**: Under RIP conditions, CoSaOP maintains a linear convergence rate $\|\boldsymbol{\beta}^k - \boldsymbol{\beta}^*\|_2 \leq 0.869 \|\boldsymbol{\beta}^{k-1} - \boldsymbol{\beta}^*\|_2 + 14.482 \|\boldsymbol{\epsilon}\|_2$.
- **Advantages in High-Correlation Scenarios (Theorem 4.10 & 4.11)**: When features are highly correlated (correlation coefficient $\rho$ close to 1), the upper bound of the classical correlation criterion decays to $\sqrt{1-\rho^2}$, while the lower bound of the optimal criterion scales up to $1/(1-\rho^2)$, multiplying the advantage. In addition, in the presence of spuriously correlated features, the classical criterion may mistakenly delete true features, whereas the optimal criterion can correctly identify and remove spurious features.

## Key Experimental Results

### Main Results: Compressed Sensing (AudioSet Real Data)

| Audio Data | OMP (NMSE) | OP (NMSE) | Gain | CoSaMP (NMSE) | CoSaOP (NMSE) | Gain | (A)BESS (NMSE) | OP-(A)BESS (NMSE) | Gain |
|----------|------------|-----------|------|---------------|---------------|------|----------------|-------------------|------|
| Audio 1 | 9.45E-04 | 7.69E-04 | 19% | 2.36E-03 | 1.64E-03 | 31% | 1.41E-03 | 6.22E-04 | **56%** |
| Audio 2 | 5.79E-03 | 5.49E-03 | 5% | 1.87E-03 | 6.36E-04 | **66%** | 6.71E-03 | 6.36E-04 | **91%** |
| Audio 3 | 1.46E-03 | 1.18E-03 | 19% | 1.54E-03 | 5.52E-04 | **64%** | 2.84E-03 | 5.52E-04 | **81%** |
| Audio 4 | 4.05E-02 | 3.55E-02 | 12% | 2.85E-03 | 8.22E-04 | **71%** | 2.65E-02 | 1.86E-02 | 30% |

### Compressed Sensing Synthetic Data: Number of Successful Recoveries

| Scenario | OMP | OP | Gain | CoSaMP | CoSaOP | Gain | (A)BESS | OP-(A)BESS | Gain |
|------|-----|-----|------|--------|--------|------|---------|------------|------|
| Sampling rate 0.25, SNR=15 | ~50 | ~150 | **~3×** | ~30 | ~200 | **~7×** | ~450 | ~470 | Substantial |
| Sampling rate 0.50, SNR=15 | ~400 | ~480 | 20% | ~300 | ~490 | **63%** | ~495 | ~500 | Improvement despite saturation |

### Ablation Study: Sparse Regression $R^2$ Performance

| Dataset | Algorithm Pairs | $R^2$ Gain at K=5 | $R^2$ Gain at K=10 | Notes |
|--------|--------|---------------------|----------------------|------|
| Boston Housing | OMP$\rightarrow$OP | ~0.1 | ~0.05 | More significant improvement with fewer features |
| California Housing | ABESS$\rightarrow$OP-ABESS | ~0.1 | ~0.08 | Equivalent to selecting 10 more features |
| Superconductivity | CoSaMP$\rightarrow$CoSaOP | From failure to success | Substantial improvement | CoSaMP fails on regression, CoSaOP remains effective |
| Spectra (p=401, n=60) | All three pairs | Consistent improvement | Consistent improvement | Advantage is more pronounced in high-dimensional small-sample scenarios |

### Key Findings

- **CoSaOP achieves the largest gain**: In compressed sensing, CoSaOP improves the successful recovery rate by an average of 4-7 times compared to CoSaMP, and reduces the NMSE by 30%-71% on AudioSet, indicating the most significant improvement among the three pairs of algorithms.
- **Advantages are amplified under high feature correlation**: In extreme scenarios with high feature correlation + small samples + high dimensions + high noise, the advantages of the enhanced algorithms are even more prominent (see Appendix P).
- **CoSaMP fails on regression tasks while CoSaOP remains effective**: Classical CoSaMP completely fails on sparse regression tasks, whereas CoSaOP restores efficacy after substituting the criteria, demonstrating the robustness of the optimal criteria.
- **Almost no increase in computational overhead**: Runtime comparison shows only marginal differences between the new and old algorithms (see Appendix O).

## Highlights & Insights

- **A unified theoretical framework for classical criteria from an optimization perspective**: Unifying the seemingly different correlation selection and Wald-T elimination criteria under the Forward/Backward Sacrifice framework reveals that they are both products of a single step of block coordinate descent. This perspective is highly elegant and points the way toward improvement.
- **Plug-and-play design of Meta-Substitution**: The optimal criteria are not tied to any specific algorithm but instead serve as general modules to replace classical criteria in all greedy subset selection algorithms. This "criterion-level upgrade" idea possesses strong scalability.
- **Performance boost with zero extra computational cost**: By utilizing the already computed matrix inverse $(\mathbf{X}_S^T \mathbf{X}_S)^{-1}$ from the coefficient update step, the optimal criteria introduce almost no additional computational load - providing a "free lunch" performance improvement.
- **Automatically amplified advantages when feature correlation is high**: The new criteria degrade to classical criteria when features are orthogonal (without deteriorating performance). The higher the feature correlation, the greater the advantage of the new criteria—which is precisely the scenario where assistance is most needed in practical applications.

## Limitations & Future Work

- **Limited to linear models**: Currently, both theory and experiments focus on squared loss in linear regression/compressed sensing, without addressing non-linear feature selection (such as feature selection in deep learning).
- **Limited to $\ell_0$ constraints**: The derivation of the optimal criteria relies on the closed-form solution of least squares. Extending this to other structured sparsities (such as group sparsity, tree sparsity) requires additional work.
- **No comparison with deep learning feature selection methods**: With the advent of the large model era, performance in high-dimensional non-linear feature selection has not yet been verified.
- **Practicality of the Optimal Gradient Pursuit extension remains to be verified**: The paper proposes Optimal Gradient Pursuit for ultra-high-dimensional scenarios, but it was not thoroughly evaluated in the main experiments.
- **Small-scale experimental datasets**: The largest datasets only have hundreds of dimensions and hundreds of samples; performance in real-world industrial-scale large-scale feature selection is unknown.

## Related Work & Insights

- **vs OMP / Matching Pursuit**: OMP progressively selects features using the correlation criterion. The proposed OP corrects this by introducing a projection matrix in the denominator, which additionally accounts for the interaction between candidate and already selected features. OP exhibits a massive advantage in highly correlated feature scenarios (3x improvement).
- **vs CoSaMP**: CoSaMP adopts a forward-backward strategy, using classical criteria for both selection and elimination. After replacing them with the optimal criteria to form CoSaOP, even when CoSaMP completely fails on sparse regression, CoSaOP remains effective, showing that the optimal criteria have better robustness.
- **vs ABESS**: ABESS is the current SOTA benchmark method in the subset selection field, employing an exchange-based strategy. Even though ABESS is close to saturation, OP-ABESS still achieves observable meta-gains (e.g., 56%-91% NMSE reduction on AudioSet).
- **vs LASSO / SCAD / MCP**: These relaxation methods approximate $\ell_0$ using continuous penalties, which can be computationally heavy and make it difficult to precisely control the number of selected features; this paper directly improves greedy criteria at the combinatorial optimization level.
- **vs SMP (Tohidi et al., 2025)**: An independent concurrent work derived a result similar to criterion (8) from a submodularity perspective, but only used it to replace OMP; this paper provides a unified optimization framework that covers both selection and elimination criteria, making it applicable to all greedy algorithms.

## Rating

- Novelty: ⭐⭐⭐⭐ Unifies classical criteria from an optimization perspective and derives optimal substitutes with a clear and elegant approach, though the core mathematical tools (matrix inversion lemmas) are classical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both compressed sensing and sparse regression tasks, synthetic + real data, three pairs of algorithm comparisons, and multiple evaluation metrics, with extended experiments on high correlation, ultra-high dimensions, unsupervised learning, complex signal processing, etc., in the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow, intuitive figures and tables, natural and smooth derivation from old to new criteria; the visualization comparison in Figure 1 is highly effective.
- Value: ⭐⭐⭐⭐ As a theoretical and methodological improvement in the classic field of best subset selection, it holds strong academic value; the Meta-Substitution concept is inspiring, but the application scope is limited to linear sparse models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)
- [\[ICML 2025\] Predictive Data Selection: The Data That Predicts Is the Data That Teaches](predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](../../ACL2025/model_compression/disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[ICML 2025\] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)
- [\[ACL 2025\] Mitigating Selection Bias with Node Pruning and Auxiliary Options](../../ACL2025/model_compression/selection_bias_node_pruning.md)

</div>

<!-- RELATED:END -->
