---
title: >-
  [Paper Note] Statistical Inference for Gradient Boosting Regression
description: >-
  [NeurIPS 2025][Gradient Boosting] This paper proposes a unified statistical inference framework for gradient boosting regression. By integrating dropout and parallel training into the Boulevard regularization scheme, the authors establish corresponding central limit theorems, enabling built-in confidence intervals, prediction intervals, and hypothesis tests for variable importance. A key finding is that increasing the dropout rate and the number of parallel trees substantially improves signal recovery—by up to $2\times$ and $4\times$, respectively.
tags:
  - NeurIPS 2025
  - Gradient Boosting
  - Central Limit Theorem
  - Confidence Intervals
  - Hypothesis Testing
  - Random Forests
date: 2026-05-08
content_hash: 93d7f753eedeb61a
---

# Statistical Inference for Gradient Boosting Regression

**Conference**: NeurIPS 2025
**arXiv**: [2509.23127](https://arxiv.org/abs/2509.23127)
**Code**: None
**Area**: Statistical Inference / Ensemble Learning
**Keywords**: Gradient Boosting, Central Limit Theorem, Confidence Intervals, Hypothesis Testing, Random Forests

## TL;DR

This paper proposes a unified statistical inference framework for gradient boosting regression. By integrating dropout and parallel training into the Boulevard regularization scheme, the authors establish corresponding central limit theorems, enabling built-in confidence intervals, prediction intervals, and hypothesis tests for variable importance. A key finding is that increasing the dropout rate and the number of parallel trees substantially improves signal recovery—by up to $2\times$ and $4\times$, respectively.

## Background & Motivation

Gradient boosting methods (XGBoost, LightGBM, CatBoost) achieve remarkable predictive performance on tabular data, yet their **uncertainty quantification** lags far behind their predictive capabilities. The central question is: if new data were collected and the model retrained, how much would predictions change?

Most existing uncertainty quantification methods lack theoretical guarantees:
- Langevin boosting, kNN-based methods, and Gaussian graphical model approaches rely primarily on heuristic arguments.
- Bayesian methods (Ustimenko et al.) require rerunning the full boosting procedure to generate posterior samples.
- On the frequentist side, the Boulevard method of Zhou & Hooker (2022) is the **only** framework offering frequentist inference, but it has two major limitations:
  1. **Insufficient signal recovery**: it recovers at most half the true signal ($\frac{\lambda}{1+\lambda} f \leq f/2$).
  2. **No practical intervals**: asymptotic normality is established but no concrete confidence or prediction intervals are constructed.

**Key Insight**: Increasing the dropout probability paradoxically recovers more signal—when trees in the current ensemble are randomly dropped, the residuals retain more of the original signal, allowing new trees to learn more.

## Method

### Overall Architecture

The core pipeline is: Boulevard regularization → convergence to kernel ridge regression (KRR) → central limit theorem → statistical inference tools.

The key modification to Boulevard regularization is that, instead of simply accumulating tree predictions, each round takes an average:
$$\hat{f}^{(b+1)} \leftarrow \frac{b-1}{b}\hat{f}^{(b)} + \frac{\lambda}{b} t^{(b)} = \frac{\lambda}{b}\sum_{i=1}^b t^{(i)}$$

### Key Designs

#### 1. **BRAT-D: Boulevard Regularization with Dropout (Algorithm 1)**

At each boosting round:
1. Subsample existing trees with probability $q=1-p$, yielding $\mathcal{S}_b \subseteq \{0,...,b-1\}$.
2. Subsample data points $\mathcal{G}_b \subseteq \{1,...,n\}$.
3. Compute residuals: $z_i = y_i - \frac{\lambda}{b}\sum_{s \in \mathcal{S}_b} t^{(s)}(\mathbf{x}_i)$
    - Dividing by $b$ rather than $|\mathcal{S}|$ causes the new tree to fit on a larger signal.
4. Final predictions require rescaling: $\frac{1+\lambda q}{\lambda}\hat{f}^{(B)}$

**Signal recovery**: converges to $\frac{\lambda}{1+\lambda q}f$, an improvement over the original Boulevard's $\frac{\lambda}{1+\lambda}f$ by a factor of $\frac{1+\lambda}{1+\lambda q} \in (1, 2]$.

**Special cases**: $\lambda=1, p\to 1$ recovers random forests; $p=0$ recovers the original Boulevard. The parameter $p$ enables smooth interpolation between the two.

#### 2. **BRAT-P: Parallel Boulevard (Algorithm 2)**

At each round, $K$ trees are trained simultaneously using a leave-one-out strategy:
1. Round 1: warm-start with $K$ steps of standard boosting.
2. Subsequent rounds: train $K$ trees in parallel; residuals for each tree are computed by leaving out its own "column":
    $$z_{i,k} = y_i - \frac{1}{b-1}\sum_{s=1}^{b-1}\sum_{l \neq k} t^{(s,l)}(\mathbf{x}_i)$$
3. Prediction: $\hat{f}^{(b+1)} = \frac{1}{b}\sum_{s=1}^b \sum_{k=1}^K t^{(s,k)}$ (no division by $K$ required).

**Signal recovery**: converges to the full signal $f$ without rescaling, directly eliminating the core deficiency of Boulevard. Relative efficiency improvement is $\geq 4\times$.

**Special cases**: $K=1, B\to\infty$ recovers random forests; $B=1, K\to\infty$ recovers standard boosting.

#### 3. **Nyström Approximation for Linear Computational Complexity**

Computing the kernel matrix $\hat{\mathbf{K}}_n$ naively costs $O(n^3)$. A Nyström approximation reduces this to $O(ns^2)$ precomputation and $O(s^2)$ inference, where $s = \tilde{O}(d_{eff}^\mu)$ is approximately linear in the effective dimension.

### Loss & Training

Four categories of statistical inference tools are provided:

**Confidence intervals** (for the true function $f$):
$$\hat{f}_n^P(\mathbf{x}) \pm z_{1-\alpha/2} \hat{\sigma} \|\hat{r}_n^P(\mathbf{x})\|_2$$

**Prediction intervals** (for a new observation $y$):
$$\hat{f}_n^P(\mathbf{x}) \pm z_{1-\alpha/2} \hat{\sigma} \sqrt{1 + \|\hat{r}_n^P(\mathbf{x})\|_2^2}$$

**Chi-squared test for variable importance**: fit the full model $\hat{f}_{n,1}$ and a model $\hat{f}_{n,2}$ with a variable removed; use the CLT to construct the test statistic:
$$\hat{\sigma}^{-2}\hat{d}_m^\top \hat{\Xi}_n^{-1} \hat{d}_m \sim \chi_m^2$$

## Key Experimental Results

### Main Results

MSE comparison across 9 UCI datasets (all methods tuned via Optuna):

| Method | Characteristics | Relative Performance |
|---|---|---|
| XGBoost | Consistently strong | Baseline |
| BRAT-D | Tunable toward boosting or RF | Competitive with XGBoost |
| BRAT-P | Occasionally unstable | Best on some datasets |
| Random Forest | Dataset-dependent | Outperforms boosting on some |
| Boulevard (original) | Limited signal recovery | Generally inferior to BRAT-D/P |

Key advantage: BRAT-D/P can be tuned to approach boosting performance on Wine Quality and random forest performance on Air Quality, offering flexible interpolation between the two paradigms.

### Ablation Study

Type I/II errors for variable importance testing ($f(\mathbf{x})=4x_1-x_2^2+wbx_3$, testing $H_0: w=0$):

| Training Set Size | Type I Error | Type II Error ($w=1$) | Notes |
|---|---|---|---|
| 200 | ~0.05 (well-controlled) | ~0.35 | Power grows with sample size |
| 500 | ~0.05 | ~0.08 | Rapid decrease |
| 1000 | ~0.05 | ~0.02 | Near-perfect detection |

Interval coverage evaluation (Friedman function):
- Prediction intervals: after adaptive calibration, coverage approaches the nominal level $1-\alpha$.
- Key advantage: unlike conformal intervals, BRAT's interval widths **vary across test points**, enabling identification of "difficult" samples.

### Key Findings

1. **Larger dropout → better signal recovery**: counterintuitively, dropping more trees improves model performance.
2. **Parallel training requires no rescaling**: BRAT-P directly converges to the full signal $f$, eliminating Boulevard's core deficiency.
3. **Substantial ARE improvements**: BRAT-D improves asymptotic relative efficiency over Boulevard by up to $4\times$; BRAT-P by at least $4\times$.
4. **Prediction intervals superior to conformal**: conditional coverage guarantees (conditional on $\mathbf{x}$) are stronger than the marginal guarantees of conformal prediction.

## Highlights & Insights

1. **Bridging boosting and random forests**: by tuning hyperparameters, one can continuously interpolate between the two classical ensemble paradigms, unifying their theoretical perspectives.
2. **Rigorous theoretical guarantees**: the paper establishes, for the first time, central limit theorems for boosting with dropout and for parallel boosting.
3. **Practical statistical tools**: confidence intervals, prediction intervals, and variable importance tests cover the core needs of applied users.
4. **Nyström approximation makes inference scalable**: linear time complexity renders the method practical for large datasets.

## Limitations & Future Work

1. Theoretical guarantees rely on assumptions such as structure-value separation and non-adaptivity; relaxing these would broaden applicability.
2. The convergence rate is $n^{-1/(d+1)}$ (for $1/2$-Hölder smooth functions); for Lipschitz functions, improvement to $n^{-2/(d+1)}$ should be achievable.
3. The framework is currently limited to regression; extension to classification, survival analysis, and other tasks requires new CLT results.
4. The instability of Algorithm 2 observed on certain datasets warrants further investigation.

## Related Work & Insights

- The paper inherits the theoretical framework for statistical inference in random forests (Wager & Athey) while addressing the sequential dependency problem unique to boosting.
- Viewing boosting as an adaptive kernel method provides useful insight for understanding deep ensemble models.
- The variable importance test extends the random forest version of Mentch & Hooker (2016b) to the boosting setting.
- Adaptive coverage calibration (following Romano et al.) is a practical technique for improving finite-sample performance.

## Rating

- Novelty: ⭐⭐⭐⭐ (Solid theoretical contributions; the finding that dropout improves signal recovery is intriguing)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers MSE comparison, hypothesis testing, and interval evaluation, though evaluation on more real-world datasets would strengthen the paper)
- Writing Quality: ⭐⭐⭐⭐ (Mathematically rigorous, though the high theoretical density may pose a barrier for non-statistician readers)
- Value: ⭐⭐⭐⭐⭐ (Fills an important gap in statistical inference for gradient boosting, with direct practical value for applications of tools such as XGBoost)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] Revisiting Agnostic Boosting](revisiting_agnostic_boosting.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](regression_trees_know_calculus.md)
- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)

<!-- RELATED:END -->
