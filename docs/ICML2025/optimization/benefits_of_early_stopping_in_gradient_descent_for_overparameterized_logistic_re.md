---
title: >-
  [Paper Note] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression
description: >-
  [ICML2025][Optimization][early stopping] This paper theoretically proves that early-stopped gradient descent (GD) possesses a statistical advantage over asymptotic GD in overparameterized logistic regression: early-stopped GD is calibrated and consistent, whereas the logistic risk of asymptotic GD diverges to infinity and its calibration error does not vanish. Additionally, a quantitative connection between early stopping and $\ell_2$ regularization is established.
tags:
  - "ICML2025"
  - "Optimization"
  - "early stopping"
  - "implicit regularization"
  - "logistic regression"
  - "overparameterization"
  - "gradient descent"
  - "calibration"
  - "maximum margin"
date: 2026-05-08
content_hash: e16315e4045c9da1
---

# Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression

**Conference**: ICML2025  
**arXiv**: [2502.13283](https://arxiv.org/abs/2502.13283)  
**Code**: None (Theoretical work)  
**Area**: Optimization  
**Keywords**: early stopping, implicit regularization, logistic regression, overparameterization, gradient descent, calibration, maximum margin

## TL;DR
This paper theoretically proves that early-stopped gradient descent (GD) possesses a statistical advantage over asymptotic GD in overparameterized logistic regression: early-stopped GD is calibrated and consistent, whereas the logistic risk of asymptotic GD diverges to infinity and its calibration error does not vanish. Additionally, a quantitative connection between early stopping and $\ell_2$ regularization is established.

## Background & Motivation
- **Implicit Bias in Overparameterization**: In overparameterized linear regression, GD converges to the minimum $\ell_2$-norm interpolator, and benign overfitting can occur under certain covariance conditions. However, in overparameterized logistic regression, the norm of GD iterates diverges to infinity, while the direction converges to the maximum $\ell_2$-margin direction (Soudry et al., 2018; Ji & Telgarsky, 2018), making the scenario more complex.
- **Early Stopping in Classification**: In linear regression, early stopping (and single-pass SGD) achieves vanishing excess risk even under general covariance, showing comparable performance to $\ell_2$ regularization. However, in classification problems (logistic loss / 0-1 loss), the regularization effects of early stopping have not yet been systematically characterized theoretically.
- **Core Problem**: In overparameterized logistic regression, does early-stopped GD possess statistical advantages over asymptotic GD (which converges to the maximum-margin solution)? What is the connection between this implicit regularization and explicit $\ell_2$ regularization?

## Method

### Problem Setup
- **Data Model (Assumption 1)**: Features $\mathbf{x} \sim \mathcal{N}(0, \boldsymbol{\Sigma})$, and the conditional label probability is $\Pr(y|\mathbf{x}) = 1/(1+\exp(-y\mathbf{x}^\top \mathbf{w}^*))$, which represents a well-specified logistic regression model.
- **Overparameterization and Noise**: $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} \lesssim 1$ (the Bayes error is a constant, and the labels are noisy), and $\operatorname{rank}(\boldsymbol{\Sigma}) \geq n$ (the effective number of parameters exceeds the sample size).
- **Metrics**: Logistic risk $\mathcal{L}(\mathbf{w})$, 0-1 error $\mathcal{E}(\mathbf{w})$, and calibration error $\mathcal{C}(\mathbf{w})$. These metrics satisfy the chain inequality: $\mathcal{E} - \min\mathcal{E} \leq 2\sqrt{\mathcal{C}} \leq \sqrt{2}\sqrt{\mathcal{L}-\min\mathcal{L}}$.

### Contribution 1: Upper Bounds for Early-Stopped GD (Calibration and Consistency)

**Theorem 3.1 (Bias-Dominated Upper Bound)**: For any index $k$, there exists a stopping time $t$ such that:

$$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} \lesssim \sqrt{\frac{\max\{1, \operatorname{tr}(\boldsymbol{\Sigma})\|\mathbf{w}^*_{0:k}\|^2\}\ln^2(n/\delta)}{n}} + \|\mathbf{w}^*_{k:\infty}\|^2_{\boldsymbol{\Sigma}}$$

- Here, $\mathbf{w}^*_{0:k}$ is the projection onto the top $k$ components sorted by $\lambda_i(\mathbf{u}_i^\top \mathbf{w}^*)^2$.
- Choosing $k$ to grow with $n$ implies that both terms approach zero, making early-stopped GD calibrated and consistent for **all** well-specified logistic regression problems.
- Under source/capacity conditions $\lambda_i \asymp i^{-a}$ and $\lambda_i(\mathbf{u}_i^\top \mathbf{w}^*)^2 \asymp i^{-b}$, the rates are $\tilde{O}(n^{-1/2})$ (for $b > a+1$) or $\tilde{O}(n^{(1-b)/(a+b-1)})$ (for $b \leq a+1$).

**Theorem 3.2 (Variance-Dominated Upper Bound)**: For $\|\mathbf{w}^*\| < \infty$, a faster rate can be achieved:

$$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} \lesssim \|\mathbf{w}^*\|\left(\frac{k}{n} + \sqrt{\frac{\sum_{i>k}\lambda_i}{n}} + \frac{\operatorname{tr}(\boldsymbol{\Sigma})^{1/2}\ln(\cdots)}{n}\right)$$

- Under the finite-dimensional setting $\boldsymbol{\Sigma} = \mathbf{I}_d$, this achieves $\tilde{O}(d/n)$, recovering the classical rate.

### Contribution 2: Lower Bounds for Interpolating Estimators

**Theorem 4.1 (Logistic Risk and Calibration Lower Bound)**: For any sequence of estimators (including asymptotic GD) whose norm diverges to infinity and whose direction converges, the logistic risk diverges to infinity, and the calibration error has a constant lower bound $\mathcal{C} \geq \exp(-C\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})$.

**Theorem 4.2 (0-1 Error Lower Bound)**: Under the setting where $\boldsymbol{\Sigma}^{1/2}\mathbf{w}^*$ is $k$-sparse and $\operatorname{rank}(\boldsymbol{\Sigma}) \geq Cn\ln(n)\ln(1/\delta)$, the excess 0-1 error of **any** interpolating estimator is at least $\Omega(1/\sqrt{\ln(n)\ln(1/\delta)})$.
- This implies that interpolating estimators require **exponentially** many samples to achieve a small 0-1 error, whereas early-stopped GD only requires **polynomially** many samples—establishing an exponential sample complexity separation.

### Contribution 3: Connection Between Early Stopping and $\ell_2$ Regularization

**Core Lemma (Lemma 3.3, Implicit Bias of Early Stopping)**: For any convex and smooth empirical risk, GD iterates satisfy:
$$\frac{\|\mathbf{w}_t - \mathbf{u}\|^2}{2\eta t} + \hat{\mathcal{L}}(\mathbf{w}_t) \leq \hat{\mathcal{L}}(\mathbf{u}) + \frac{\|\mathbf{u}\|^2}{2\eta t}$$
That is, early-stopped GD keeps a small norm while achieving low empirical risk, which is precisely the effect of $\ell_2$ regularization.

**Theorem 5.1 (Global Connection)**: Let $\lambda = 1/(\eta t)$, then for **all** convex and smooth problems:
- $\|\mathbf{w}_t - \mathbf{u}_\lambda\| \leq \|\mathbf{w}_t\|/\sqrt{2}$
- The angle between the GD path and the $\ell_2$ regularization path is at most $\pi/4$, and their norm ratio lies in $[0.585, 3.415]$.

**Theorem 5.2 (Asymptotic Connection)**: Under the support vector condition (Assumption 2), $\|\mathbf{w}_t - \mathbf{u}_{\lambda(t)}\| \to 0$, meaning that the absolute distance between the two paths vanishes (even though both norms diverge to infinity).

**Theorem 5.3 (Counterexample)**: When Assumption 2 is violated, there exists a simple two-dimensional example where the $\ell_2$ distance between the two paths diverges to infinity ($\gtrsim \ln\ln\|\mathbf{w}_t\|$).

## Key Experimental Results
- **Figure 1 Experiment** ($d=2000, n=1000, \lambda_i = i^{-2}$, $\mathbf{w}^*_{0:100}=1, \mathbf{w}^*_{100:\infty}=0$):
    - The population logistic risk and 0-1 error of early-stopped GD attain their minimums at an appropriate stopping time.
    - As GD iterates further into the interpolation regime, both errors increase significantly.
    - The empirical risk decreases monotonically, but the population risk exhibits a U-shaped curve, validating the overfitting phenomenon.
    - The optimal stopping time corresponds to $\eta t$ around the order of $10^2$, where the excess risk is close to zero.
    - As $\eta t \to \infty$, the population logistic risk diverges, in consistency with Theorem 4.1.
- This work is mainly theoretical; the experiments serve purely as intuitive validation but clearly demonstrate the U-shaped overfitting curve of early stopping.

## Highlights & Insights
1. **Complete Positive and Negative Theoretical Picture**: The paper not only proves that early stopping is beneficial (upper bounds) but also demonstrates that failing to stop early is detrimental (lower bounds), with the lower bound holding for all interpolating estimators.
2. **Importance of Calibration**: Asymptotic GD may perform acceptably in terms of 0-1 error (since the direction is correct), but it fails completely in probability prediction (calibration), which is crucial for applications requiring probabilistic outputs (e.g., medical diagnosis).
3. **Exponential Sample Complexity Separation**: Early-stopped GD requires only polynomial samples, whereas interpolating estimators require exponential samples. This is one of the strongest known separation results in overparameterized classification.
4. **Elegance of Lemma 3.3**: A simple inequality unifiedly explains the implicit regularization effect of early stopping and directly derives the global connection to $\ell_2$ regularization.
5. **Phase Transition between Theorem 5.2 and 5.3**: The support vector condition determines whether the GD path asymptotically matches or diverges from the regularization path, revealing the vulnerability of implicit regularization.

## Limitations & Future Work
1. **Stopping Time Relying on Oracle Information**: The optimal stopping times in Theorems 3.1/3.2 depend on the true parameter $\mathbf{w}^*$ and are not practically computable. Theoretical guarantees for practical methods such as cross-validation have yet to be established.
2. **Imprecise Bias-Variance Trade-off**: The two upper bounds are dominated by either bias or variance, failing to characterize the exact optimal bias-variance trade-off (unlike in linear regression, where exact results exist).
3. **Restricted to Well-Specified Models**: The main results assume that the data are generated by a logistic model (Assumption 1). Generalizations to misspecified settings are only partial (only logistic risk upper bounds, with no guarantees for calibration or 0-1 error).
4. **0-1 Error Lower Bound for Maximum-Margin Estimators**: While Theorem 4.2 provides a lower bound for all interpolating estimators, it is conjectured that a constant-level lower bound for maximum-margin estimators can be proven (which is left as future work).
5. **Absence of Comparison with SGD**: Practical training typically employs SGD instead of GD, and similar theories for early-stopped SGD have not yet been established.
6. **Technical Bottlenecks**: Exact analysis in linear regression relies on explicit closed-form solutions and a fixed Hessian matrix. In contrast, the Hessian in logistic regression varies across iterations and the GD path diverges to infinity, requiring completely new analytical tools.
7. **Non-Gaussian Designs Uncovered**: Assumption 1 requires a Gaussian feature distribution; generalization to sub-Gaussian designs is only partially discussed in Theorem 3.1.

## Related Work & Insights
- **Benign Overfitting Series** (Bartlett et al., 2020; Chatterji & Long, 2021; Cao et al., 2021): The findings in this paper do not contradict these works. Rather, they reveal that even when benign overfitting holds (and 0-1 error is controlled), asymptotic GD is still inferior to early-stopped GD in terms of calibration and logistic risk.
- **Early Stopping in Linear Regression** (Ali et al., 2019; Zou et al., 2021, 2023; Bühlmann & Yu, 2003): This paper extends the early stopping regularization theory from linear regression to logistic regression, featuring much greater technical challenges (non-fixed Hessian, diverging GD path).
- **Implicit Bias** (Soudry et al., 2018; Ji & Telgarsky, 2018): This work builds on the convergence to the maximum-margin direction to further analyze the statistical properties at the stopping time.
- **M-estimators** (Ostrovskii & Bach, 2021; Hsu & Mazumdar, 2024): The $O(d/n)$ rate of ERM in classical finite-dimensional settings is well known. Theorem 3.2 in this work recovers a comparable rate of $\tilde{O}(d/n)$ when specialized to such settings.
- **Boosting and Early Stopping** (Zhang & Yu, 2005; Bühlmann & Yu, 2003): Early stopping consistency results exist in the boosting literature, but they are limited to finite dimensions and do not provide lower bounds for interpolating estimators.
- **Implications for Overparameterized Deep Learning**: Training should not fully converge to interpolation; appropriate early stopping might yield superior performance in both calibration and generalization.
- **Classification Calibration Theory** (Zhang, 2004; Bartlett et al., 2006): This paper provides concrete statistical rates and outcomes of early stopping, going beyond previous abstract analyses.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to systematically establish the positive/negative theoretical framework for early-stopped GD in overparameterized logistic regression. The exponential sample complexity separation and the regularization path connections are both major new results.)
- Experimental Thoroughness: ⭐⭐⭐ (Mainly theoretical with only one illustrative experiment, which is sufficient for a theory paper.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, main conclusions summarized well in Contributions, and reasonable organization of technical details.)
- Value: ⭐⭐⭐⭐⭐ (Provides profound theoretical insights into the interaction between optimization and statistics in overparameterized learning, which is highly valuable for understanding early stopping practices in deep learning.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[ICML 2025\] Constant Stepsize Local GD for Logistic Regression: Acceleration by Instability](constant_stepsize_local_gd_for_logistic_regression_acceleration_by_instability.md)
- [\[CVPR 2025\] Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent](../../CVPR2025/optimization/stop_walking_in_circles_bailing_out_early_in_projected_gradient_descent.md)
- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](quantum_optimization_via_gradient-based_hamiltonian_descent.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
