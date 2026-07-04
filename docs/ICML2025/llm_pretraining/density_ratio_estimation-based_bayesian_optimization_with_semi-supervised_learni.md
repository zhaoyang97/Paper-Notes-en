---
title: >-
  [Paper Note] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning
description: >-
  [ICML2025][LLM Pretraining][Bayesian Optimization] This paper proposes DRE-BO-SSL, which incorporates semi-supervised learning (label propagation/label spreading) into density ratio estimation-based Bayesian optimization. By utilizing unlabeled data points, it alleviates the over-exploitation issue of supervised classifiers, achieving a better balance between exploration and exploitation.
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Bayesian Optimization"
  - "Density Ratio Estimation"
  - "Semi-Supervised Learning"
  - "Label Propagation"
  - "Label Spreading"
date: 2026-05-08
content_hash: b03d4f2582c9671b
---

# Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning

**Conference**: ICML2025  
**arXiv**: [2305.15612](https://arxiv.org/abs/2305.15612)  
**Code**: [https://github.com/jungtaekkim/bayeso](https://github.com/jungtaekkim/bayeso)  
**Area**: LLM Pre-training  
**Keywords**: Bayesian Optimization, Density Ratio Estimation, Semi-Supervised Learning, Label Propagation, Label Spreading

## TL;DR

This paper proposes DRE-BO-SSL, which incorporates semi-supervised learning (label propagation/label spreading) into density ratio estimation-based Bayesian optimization. By utilizing unlabeled data points, it alleviates the over-exploitation issue of supervised classifiers, achieving a better balance between exploration and exploitation.

## Background & Motivation

### Review of Bayesian Optimization Paradigm

Bayesian optimization (BO) is employed to efficiently search for the global optimum of expensive-to-evaluate black-box functions. Traditional methods utilize probabilistic regression models (such as Gaussian processes) as surrogate functions to explicitly model $p(y|\mathbf{x}, \mathcal{D})$, and select the next query point using an acquisition function (e.g., EI, PI).

### The Rise of Density Ratio Estimation-based BO

Unlike regression-based BO, density ratio estimation (DRE) based BO divides the observed points in the search space into two groups using a threshold $y^\dagger$—a "good group" ($y \leq y^\dagger$) close to the global optimum and a "bad group" ($y > y^\dagger$). It then estimates the density ratio of these two groups to construct the acquisition function. Representative methods include:

- **TPE** (Bergstra et al., 2011): Estimates the densities of the two groups respectively using two tree-structured Parzen estimators.
- **BORE** (Tiao et al., 2021): Transforms density ratio estimation into class probability estimation for binary classification.
- **LFBO** (Song et al., 2022): Proposes a general likelihood-free framework, demonstrating that BORE is equivalent to the Probability of Improvement.

### Core Problem: Over-Exploitation

This paper identifies a critical limitation of supervised classifiers in DRE-based BO—the **over-exploitation issue**. Unlike overconfidence in conventional classification, this refers to the classifier being excessively confident about known global candidate regions. Consequently, the search concentrates within an extremely narrow range, sacrificing exploration capability. This manifests as:

1. In the early iterations of BO, the training dataset $\mathcal{D}_t$ is small, making supervised classifiers prone to overfitting.
2. The classifier assigns extremely high probabilities to regions containing known good points, while the probability of other regions rapidly approaches zero.
3. The threshold $y^\dagger$ struggles to change dynamically, causing the algorithm to easily get trapped in local optima.

While traditional regression-based BO (such as GP-EI) naturally balances exploration through uncertainty estimation, DRE-based classifiers lack such a mechanism.

## Method

### Overall Architecture: DRE-BO-SSL

Core Idea: Introduce **unlabeled data points** and utilize a semi-supervised classifier instead of a supervised classifier to estimate class probabilities, thereby reducing the overconfidence in known regions.

#### Acquisition Function Derivation

The acquisition function of DRE-based BO is formulated based on the $\zeta$-relative density ratio:

$$A(\mathbf{x}|\zeta, \mathcal{D}_t) = \frac{p(\mathbf{x}|y \leq y^\dagger, \mathcal{D}_t)}{\zeta \cdot p(\mathbf{x}|y \leq y^\dagger, \mathcal{D}_t) + (1-\zeta) \cdot p(\mathbf{x}|y > y^\dagger, \mathcal{D}_t)}$$

where $\zeta = p(y \leq y^\dagger) \in [0,1)$ is the threshold ratio. Through Bayes' theorem, this can be reformulated as class probability estimation:

$$A(\mathbf{x}|\zeta, \mathcal{D}_t) = \zeta^{-1} \pi(\mathbf{x})$$

where $\pi(\mathbf{x})$ denotes the probability of input $\mathbf{x}$ belonging to Class 1 (the good group). DRE-BO-SSL replaces $\pi$ with the output of a semi-supervised classifier:

$$\mathbf{x}_{t+1} = \arg\max_{\mathbf{x} \in \mathcal{X}} \pi_{\hat{\mathbf{C}}_t}(\mathbf{x}; \zeta, \mathcal{D}_t, \mathbf{X}_u)$$

### Semi-Supervised Learning Components

Two classic graph-based semi-supervised methods are adopted:

#### 1. Label Propagation (Zhu & Ghahramani, 2002)

Labels are propagated iteratively on a similarity graph containing $n_l$ labeled points and $n_u$ unlabeled points. A similarity matrix $\mathbf{W}$ is constructed using an RBF kernel, and pseudo-labels of unlabeled points are updated iteratively until convergence. In each iteration, labels of labeled points are reset to their initial values (hard constraint).

#### 2. Label Spreading (Zhou et al., 2003)

This is similar to Label Propagation but allows the labels of labeled points to be modified during iterations (soft constraint). A parameter $\alpha \in [0,1)$ controls the trade-off between label propagation and the initial labels.

### Transductive to Inductive Extension

Semi-supervised methods are inherently transductive (only predicting labels for the predefined set of unlabeled points). However, BO requires predictions for arbitrary points $\mathbf{x}$ in the search space. To achieve inductive predictions, this work trains a standard classifier (e.g., 1-NN) using the combined dataset ("labeled + unlabeled") with the resolved pseudo-labels, and then performs inference for any new query point.

### Sampling Strategy for Unlabeled Points

When a predefined pool (fixed-size pool) is unavailable, unlabeled points must be actively sampled. This paper adopts the minimax tilting method (Botev, 2017) to sample from a truncated multivariate normal distribution, ensuring that the unlabeled points cover the search space $\mathcal{X}$ and satisfy the cluster assumption.

### Algorithm Flow (Algorithm 1)

1. Randomly initialize and evaluate $\mathcal{D}_0$.
2. In each iteration $t$: compute the threshold $y_t^\dagger$ $\rightarrow$ assign class labels $\mathbf{C}_t$ $\rightarrow$ sample/acquire unlabeled points $\mathbf{X}_u$ $\rightarrow$ estimate pseudo-labels $\hat{\mathbf{C}}_t$ via semi-supervised learning $\rightarrow$ maximize $\pi_{\hat{\mathbf{C}}_t}$ to select the next query point $\rightarrow$ evaluate the point and update the dataset.

Optimization is performed using multi-started L-BFGS-B. For flat probability landscapes, one of the points with the highest probability is selected randomly.

## Key Experimental Results

### Experimental Setup

- **Synthetic Benchmarks**: Standard test functions such as Branin (2D) and Hartmann (6D).
- **Tabular Benchmarks** (Klein & Hutter, 2019): Tabular benchmarks for hyperparameter optimization (fixed-pool scenario).
- **NATS-Bench** (Dong et al., 2021): Neural architecture search benchmark (fixed-pool scenario).
- **64D minimum multi-digit MNIST search**: High-dimensional search task (fixed-pool scenario).
- **Baselines**: GP-EI, GP-PI, TPE, BORE (MLP/RF/XGBoost), LFBO (MLP/RF/XGBoost).

### Two Experimental Scenarios

| Scenario | Source of Unlabeled Points | Representative Tasks |
|------|-------------|---------|
| Random Sampling | Samples $n_u$ unlabeled points from a truncated normal distribution in each round | Synthetic functions |
| Fixed Pool | Predefined finite set of candidate points | Tabular Benchmarks, NATS-Bench, MNIST search |

### Main Results

1. **Visualization on Synthetic Functions** (Figure 1, Branin): In the first 5 iterations, decision boundaries of BORE/LFBO's MLP classifiers excessively concentrate on a tiny region (over-exploitation). Conversely, the decision boundaries of DRE-BO-SSL (Label Propagation/Spreading) are smoother and flatter, covering a wider area (better exploration-exploitation balance).

2. **Quantitative Results on Synthetic Benchmarks**: DRE-BO-SSL outperforms or performs comparably to BORE/LFBO in terms of convergence speed and final solution quality across multiple synthetic functions, demonstrating a significant advantage in early iterations.

3. **Tabular Benchmarks**: Under the fixed-pool scenario, DRE-BO-SSL exhibits competitive performance in hyperparameter optimization tasks, outperforming all baseline methods in several configurations.

4. **NATS-Bench Neural Architecture Search**: DRE-BO-SSL successfully finds high-quality architectures, showing performance superior to or on par with GP and BORE/LFBO variants.

5. **64D MNIST Search**: Demonstrates the scalability of DRE-BO-SSL in high-dimensional fixed-pool search spaces.

### Ablation Study (Section 6)

- **Influence of Threshold Ratio $\textstyle \zeta$**: Analyzes the impact of different $\zeta$ values on performance and verifies the behavior of the over-exploitation issue under varying values.
- **Influence of Number of Unlabeled Points $\textstyle n_u$**: A larger number of unlabeled points generally leads to a smoother probability landscape, but at the expense of higher computation costs.
- **Label Propagation vs. Label Spreading**: Both methods show similar performance, though Label Spreading achieves better stability in some scenarios due to soft constraints.

## Highlights & Insights

1. **Precise Problem Identification**: Clearly distinguishes "over-exploitation" in DRE-BO from "overconfidence" in standard classification—the former refers to regional over-concentration, whereas the latter is instance-level uncertainty miscalibration.
2. **Lightweight and Elegant Design**: Requires no modifications to the core BO framework; it simply replaces the supervised classifier with a semi-supervised one, minimizing engineering overhead.
3. **Good Theoretical Compatibility**: Fully compatible with the theoretical frameworks of BORE/LFBO. The semi-supervised classifier continues to output class probabilities, which can be directly plugged into the relative density ratio calculation.
4. **Unified Handling of Two Scenarios**: Supports both truncated normal sampling and fixed pools to obtain unlabeled points, naturally covering both continuous and discrete search spaces.

## Limitations & Future Work

1. **Questionable Scalability**: Label Propagation/Spreading requires computing a similarity matrix of size $(n_l + n_u) \times (n_l + n_u)$. This incurs substantial computational overhead when $n_u$ is large or when searching in high-dimensional spaces.
2. **Limited Choice of Semi-Supervised Methods**: Only two classic graph-based methods (Label Propagation and Label Spreading) are explored, leaving more advanced semi-supervised approaches (e.g., FixMatch, MixMatch) uninvestigated.
3. **Sensitivity to RBF Kernel Parameters**: The similarity matrix heavily depends on the bandwidth parameter of the RBF kernel. The paper provides limited discussion on hyperparameter selection and robustness.
4. **Simple Sampling Strategy for Unlabeled Points**: Sampling from a truncated normal distribution relies on strong assumptions. Adaptive sampling strategies that adjust based on the search state remain unexplored.
5. **Lack of Theoretical Convergence Guarantees**: Compared to GP-based BO, which offers rigorous regret bounds, DRE-BO-SSL lacks similar theoretical guarantees.
6. **Insufficient Evaluation on High-Dimensional Continuous Spaces**: Synthetic functions are evaluated up to 6D. Although the fixed-pool scenario reaches 64D, it operates on a discrete candidate pool.

## Related Work & Insights

- **BORE** (Tiao et al., 2021): The first work to recast density ratio estimation as binary classification, representing the direct baseline.
- **LFBO** (Song et al., 2022): Unifies the formulation of DRE-BO, demonstrating its equivalence to PI/EI.
- **TPE** (Bergstra et al., 2011): The pioneering work in DRE-BO, modeling two densities via Parzen window estimators.
- **Label Propagation** (Zhu & Ghahramani, 2002) / **Label Spreading** (Zhou et al., 2003): The two core semi-supervised learning frameworks used in this study.
- **Insight**: Semi-supervised learning could potentially be applied to alleviate severe over-exploitation issues in other optimization/decision frameworks, such as surrogate models in evolutionary algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐ — Insightful problem formulation, although the method itself is a combination of existing techniques (DRE-BO + classic semi-supervised learning), leading to incremental algorithmic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extensively covers synthetic functions, hyperparameter optimization, NAS, and high-dimensional search, supported by comprehensive ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured but quite lengthy; equations are detailed but require effort to follow.
- Value: ⭐⭐⭐⭐ — Provides practical value for the DRE-BO domain, though this area remains a relatively niche branch of the broader BO community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Unified Framework for Heterogeneous Semi-supervised Learning](../../CVPR2025/llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)
- [\[ICML 2025\] A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning](a_square_peg_in_a_square_hole_meta-expert_for_long-tailed_semi-supervised_learni.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)
- [\[ICML 2025\] Position: The Future of Bayesian Prediction Is Prior-Fitted](position_the_future_of_bayesian_prediction_is_prior-fitted.md)

</div>

<!-- RELATED:END -->
