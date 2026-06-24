---
title: >-
  [Paper Note] Disparate Conditional Prediction in Multiclass Classifiers
description: >-
  [ICML 2025][AI Safety][Fairness Auditing] This paper proposes the extension of the Disparate Conditional Prediction (DCP) metric from binary to multiclass classification. By leveraging local optimization and linear programming methods, it provides upper and lower bound estimates for the degree of fairness deviation of multiclass classifiers. This supports fairness auditing in two scenarios: when the confusion matrix is known, or when only population-level statistics are avail…
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Fairness Auditing"
  - "Multiclass Classification"
  - "Equalized Odds"
  - "Disparate Conditional Prediction"
  - "Linear Programming"
date: 2026-05-08
content_hash: 726fbf6256fe9665
---

# Disparate Conditional Prediction in Multiclass Classifiers

**Conference**: ICML 2025  
**arXiv**: [2206.03234](https://arxiv.org/abs/2206.03234)  
**Code**: [sivansabato/DCPmulticlass](https://github.com/sivansabato/DCPmulticlass)  
**Area**: AI Safety  
**Keywords**: Fairness Auditing, Multiclass Classification, Equalized Odds, Disparate Conditional Prediction, Linear Programming

## TL;DR

This paper proposes the extension of the Disparate Conditional Prediction (DCP) metric from binary to multiclass classification. By leveraging local optimization and linear programming methods, it provides upper and lower bound estimates for the degree of fairness deviation of multiclass classifiers. This supports fairness auditing in two scenarios: when the confusion matrix is known, or when only population-level statistics are available.

## Background & Motivation

Classifier fairness is crucial in real-world scenarios, particularly when classifiers are deployed in high-risk domains such as medical diagnosis, insurance pricing, and hiring screening. **Equalized Odds** is a widely adopted fairness definition, requiring the classifier to maintain identical false positive and false negative rates across different protected subgroups.

However, existing work mainly focuses on the **binary classification** setting. In multiclass scenarios, fairness issues are more complex—not only must accuracy be considered, but attention must also be paid to the **types of errors**. For instance, in medical diagnosis, misdiagnosing heart disease as anxiety (preventing patients from receiving treatment) differs fundamentally from misdiagnosing it as a stroke (resulting in delayed treatment). If certain subgroups bear a higher burden of specific types of misdiagnoses, it may indicate bias in the diagnosis.

Existing fairness deviation metrics (e.g., difference-based or ratio-based methods) suffer from the following limitations:

- Insufficient sensitivity to disparities in rare labels
- Lack of normalization or boundedness
- Inability to distinguish between classifiers with different degrees of bias
- Interpretability of results that depends on the number of classes, protected subgroups, and class imbalance

These issues become more prominent in multiclass scenarios, necessitating a metric with a **consistently interpretable meaning**.

## Method

### Overall Architecture

This paper generalizes the binary DCP metric proposed by Sabato & Yom-Tov (2020) to multiclass classification, and designs upper and lower bound estimation algorithms for two computational scenarios:

1. **Known scenario**: The conditional confusion matrix $\mathbf{M}_a$ for each subgroup is known.
2. **Unknown scenario**: Only population-level frequency information is available (e.g., the predicted label distribution $\hat{\mathbf{p}}_a$ and true label distribution $\boldsymbol{\pi}_a$ for each subgroup).

Overall workflow: Define the multiclass DCP metric $\rightarrow$ Formulate the optimization problem $\rightarrow$ Address non-smoothness/non-convexity challenges $\rightarrow$ Solve via sequential linear programming.

### Key Designs

#### 1. Notation and Multiclass DCP Metric Definition

Core notation definitions:

- True label $Y \in \mathcal{Y} = \{1, \ldots, k\}$, predicted label $\hat{Y} \in \mathcal{Y}$, and protected attribute $A \in \mathcal{A}$.
- Subgroup frequency: $w_a = \mathbb{P}[A=a]$
- True label proportion: $\pi_a^y = \mathbb{P}[Y=y \mid A=a]$
- Predicted label proportion: $\hat{p}_a^y = \mathbb{P}[\hat{Y}=y \mid A=a]$
- Conditional confusion matrix element: $\alpha_a^{y\hat{y}} = \mathbb{P}[\hat{Y}=\hat{y} \mid Y=y, A=a]$

The core intuition of DCP is: **the proportion of the population for which the classifier makes conditional predictions divergent from the closest common baseline**. The goal is to find a set of optimal common conditional probability vectors $\boldsymbol{\beta}^y$ such that the predictive behavior of all subgroups is as close to this baseline as possible. The DCP value represents the sum of the population proportions that deviate from this baseline.

Key advantage of DCP: The result always represents a population proportion, remaining unaffected by the number of classes, subgroups, or class imbalance.

#### 2. Upper and Lower Bound Methods for the Known Confusion Matrix Scenario

When the conditional confusion matrix of each subgroup is known, the DCP calculation is formulated as:

$$\text{DCP} = \min_{\boldsymbol{\beta}} \sum_{a \in \mathcal{A}} w_a \sum_{y \in \mathcal{Y}} \pi_a^y \cdot \mathbb{1}[\boldsymbol{\alpha}_a^y \neq \boldsymbol{\beta}^y]$$

**Upper Bound Method**: Since this optimization problem is non-convex, a relaxation strategy is employed to replace the indicator function with a continuous $L_1$ distance objective:

$$\min_{\boldsymbol{\beta}} \sum_{a,y} w_a \pi_a^y \cdot \|\boldsymbol{\alpha}_a^y - \boldsymbol{\beta}^y\|_1$$

To handle the non-smoothness of the $L_1$ norm: the $\|\cdot\|_1$ is decomposed into the maximum of multiple smooth parts. Auxiliary variables are introduced to convert the maximum constraints into linear constraints, ultimately forming a **Sequential Linear Programming (Sequential LP)** problem. Each step solves a standard LP, concave constraints are linearized using a first-order Taylor expansion, and the process iterates until convergence to a local optimum.

**Lower Bound Method**: This is achieved by constructing a specific baseline selection strategy—for example, for each true label $y$, selecting the conditional prediction vector of the subgroup with the highest weighted frequency as the baseline.

#### 3. Methods for the Unknown Confusion Matrix Scenario

When the confusion matrix cannot be accessed (the classifier is inaccessible or individual-level data is missing), only population-level statistics $\hat{\mathbf{p}}_a$, $\boldsymbol{\pi}_a$, and $\mathbf{w}$ are utilized. The confusion matrix becomes latent variables, subject to marginal constraints:

$$\sum_{\hat{y}} \alpha_a^{y\hat{y}} = 1, \quad \sum_y \pi_a^y \cdot \alpha_a^{y\hat{y}} = \hat{p}_a^{\hat{y}}$$

**Upper Bound Method**: Minimization is performed over the joint space of the confusion matrix and the baseline, which is also solved via sequential LP. This involves higher variable dimensions and presents a greater computational challenge.

**Lower Bound Method**: A lower bound for DCP is established using dual methods, which can be used to detect classifiers that might produce unfair predictions for a large proportion of the population.

### Loss & Training

This paper does not propose a new training method but instead designs a **fairness auditing tool**. The core optimization objective is:

$$\min_{\boldsymbol{\beta}, \{\mathbf{M}_a\}} \sum_{a \in \mathcal{A}} w_a \sum_{y \in \mathcal{Y}} \pi_a^y \cdot d(\boldsymbol{\alpha}_a^y, \boldsymbol{\beta}^y)$$

Key techniques of the optimization strategy:

- **Non-smooth handling**: Split the max operator into multiple smooth branches and add extra constraints.
- **Linearization of concave constraints**: Replace concave function constraints with first-order Taylor expansions.
- **Sequential LP**: Convert the non-convex problem into a sequence of LPs, solving each step with an efficient LP solver (e.g., HiGHS).
- **Multiple initializations**: Due to the highly non-convex nature of the problem, multiple random starting points are run, and the best result is selected.
- **Handling large gradients**: The constraint matrix contains regions with large gradients, a numerical challenge to which LP solvers are naturally well-suited.

## Key Experimental Results

### Main Results

Experiments validate the accuracy of the DCP upper and lower bound estimations across multiple datasets (a smaller gap is better):

| Dataset | Number of Classes $k$ | Number of Subgroups | Upper/Lower Bound Gap (Known Scenario) | Upper/Lower Bound Gap (Unknown Scenario) | Description |
|---|---|---|---|---|---|
| Multiclass Benchmark 1 | 3-5 | 2-4 | Typically very small | Slightly larger than the known scenario | Validates basic effectiveness |
| Multiclass Benchmark 2 | 5-10 | 2-6 | Small | Acceptable | Larger scale validation |
| Medical/Social Datasets | 3+ | 2+ | Tight | Within practical range | Validation of real-world scenario applicability |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Different Numbers of Random Initializations | Gap Size | More initializations typically improve upper bound quality |
| Different Number of Classes $k$ | Computation Time & Gap | Increasing the number of classes makes the problem more complex, slightly increasing the gap |
| Different Number of Subgroups | DCP Estimation Stability | More subgroups expand the optimization variable space |
| Known vs. Unknown Scenarios | Tightness of the Upper Bound | The upper bound is significantly tighter in the known scenario |
| DCP vs. Difference/Ratio Metrics | Interpretability Comparison | DCP maintains a consistent population proportion interpretation across different configurations |

### Key Findings

1. **Small Gaps Between Bounds**: The gap between the upper and lower bounds provided by the sequential LP method is small in most scenarios, indicating high-quality estimation.
2. **Robust Interpretability of DCP**: Unlike difference-based or ratio-based methods, the DCP value is always interpretable as a population proportion, remaining unaffected by problem configuration.
3. **Practicality of the Unknown Scenario**: Even without the confusion matrix, population-level statistics can still provide meaningful DCP estimates.
4. **Unique Value in Multiclass Scenarios**: DCP captures disparities in different types of misclassification, which cannot be captured by binary fairness metrics.

## Highlights & Insights

- **Outstanding Interpretability**: DCP consistently represents the 'proportion of the population receiving unfair predictions,' and its numerical meaning remains invariant regardless of the scale of the problem.
- **No Classifier Access Required**: The method for the unknown scenario relies solely on population-level statistics, making it suitable for auditing black-box or proprietary classifiers, which offers high value in compliance auditing.
- **Practical Solution to Non-convex Optimization**: Sequential LP elegantly bypasses non-convex and non-smooth difficulties, where each subproblem is a standard LP that can be solved efficiently.
- **Natural Extension from Binary to Multiclass**: Mathematically clean and elegant, retaining all the desirable properties of the original metric.

## Limitations & Future Work

1. **Computational Scalability**: The scale of the LP grows rapidly as the number of classes $k$ and subgroups $|\mathcal{A}|$ increase.
2. **Local Optima**: Sequential LP only guarantees convergence to a local optimum, lacking global guarantees.
3. **Static Auditing**: The current method is a post-hoc audit and does not address how to leverage DCP to guide fairness optimization during the training phase.
4. **Propagation of Statistical Error**: The impact of estimation errors in population-level statistics under finite sample sizes on DCP has not been fully analyzed.
5. **Intersectionality Attribute Explosion**: Cartesian products of multiple attributes lead to exponential growth in the number of subgroups, which may cause data sparsity issues.

## Related Work & Insights

- **Sabato & Yom-Tov (2020)**: Direct precursor work that introduced binary DCP, which this work generalizes to multiclass.
- **Alghamdi et al. (2022)**: Proposed a model projection method to address multiclass equalized odds, but utilized ratio-based metrics.
- **Wang et al. (2024)**: Investigated the scenarios and fundamental limits of multiclass fair classifiers, revealing trade-offs between fairness and accuracy.
- **Insights**: The 'population proportion' interpretability of DCP can be generalized to domains such as recommendation systems and NLP; the sequential LP strategy can be applied to other non-convex fairness optimization problems.

## Rating

| Dimension | Score (1-10) | Description |
|---|---|---|
| Novelty | 7 | The extension from binary to multiclass is natural yet non-trivial, and the optimization method is novel. |
| Theoretical Depth | 8 | Rigorous mathematical derivations and comprehensive analysis of the upper/lower bound methods. |
| Experimental Thoroughness | 7 | Effectiveness is validated across multiple datasets, but experiments on large-scale scenarios are limited. |
| Value | 8 | The capability for black-box auditing holds significant value in compliance reviews. |
| Writing Quality | 8 | Clear formulation, rigorous notation definitions, and well-structured presentation. |
| **Overall Rating** | **7.5** | A solid piece of fairness auditing work, with the interpretability of DCP being the core highlight. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Generalization in Federated Learning: A Conditional Mutual Information Framework](generalization_in_federated_learning_a_conditional_mutual_information_framework.md)
- [\[NeurIPS 2025\] On the Hardness of Conditional Independence Testing In Practice](../../NeurIPS2025/ai_safety/on_the_hardness_of_conditional_independence_testing_in_practice.md)
- [\[CVPR 2025\] PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection](../../CVPR2025/ai_safety/psbd_prediction_shift_uncertainty_unlocks_backdoor_detection.md)
- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](../../NeurIPS2025/ai_safety/mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](../../ICML2026/ai_safety/beyond_procedure_substantive_fairness_in_conformal_prediction.md)

</div>

<!-- RELATED:END -->
