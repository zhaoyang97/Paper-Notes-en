---
title: >-
  [Paper Note] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding
description: >-
  [ICML 2026][learning_theory][Paper Note] This paper proposes the Conditional Kernel Ridge Regression (Conditional KRR) framework, which injects a set of unpenalized features into kernel methods. By reducing it to standard KRR via a residual kernel, the authors prove a reduction cost of $\mathcal{O}(1/\sqrt{N})$. They further verify sufficient conditions for C
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: 55f8c712fe503f23
---
# Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding

**Conference**: ICML2026  
**arXiv**: [2605.26067](https://arxiv.org/abs/2605.26067)  
**Code**: None  
**Area**: Machine Learning Theory  
**Keywords**: Conditional Kernel Ridge Regression, Conditionally Positive Definite Kernel, Residual Kernel, Kernel Thresholding, Unpenalized Features  

## TL;DR

This paper proposes the Conditional Kernel Ridge Regression (Conditional KRR) framework, which injects a set of unpenalized features into kernel methods. By reducing it to standard KRR via a residual kernel, the authors prove a reduction cost of $\mathcal{O}(1/\sqrt{N})$. They further verify sufficient conditions for Conditional KRR to outperform standard KRR under both hard thresholding (top-k eigenfunctions) and soft thresholding (random Gaussian features) settings.

## Background & Motivation

**Background**: Kernel Ridge Regression (KRR) is a classic supervised learning method widely used in Neural Tangent Kernels, operator approximation, and reinforcement learning. Using KRR requires a positive definite (PD) kernel function $K(x,y)$, but the positive definiteness condition can be relaxed to conditionally positive definite (CPD), meaning the quadratic form $\sum_{ij} K(x_i, x_j) \zeta_i \zeta_j$ only needs to be non-negative when $\zeta$ is orthogonal to a certain function class $\mathcal{F}$.

**Limitations of Prior Work**: Existing research on CPD kernels focus almost entirely on cases where $\mathcal{F}$ is a polynomial subspace, leaving the statistical theory for general function classes $\mathcal{F}$ nearly blank. More importantly, even if the original kernel $K$ is PD, treating it as a CPD kernel with respect to a general $\mathcal{F}$ naturally leads to a two-stage learning framework ("linear regression first, then kernel regression on residuals"), but such a framework lacks rigorous statistical guarantees.

**Key Challenge**: The optimization problem of Conditional KRR can be decomposed into linear regression plus residual kernel KRR, but this decomposition depends on precise knowledge of the $\mathcal{F}$-component. In practice, the learner does not know the projection of the true regression function onto $\mathcal{F}$. Therefore, it is necessary to quantify the cost paid for "not knowing $f_\parallel$"—the conditioning cost.

**Goal**: (1) Establish a rigorous reduction theory from Conditional KRR to standard KRR; (2) Provide high-probability upper bounds for the conditioning cost; (3) Apply the theory to hard and soft thresholding settings to derive sufficient conditions for Conditional KRR outperforming standard KRR.

**Key Insight**: The authors introduce the "residual kernel" $K_P = ((I - \Pi_P) \otimes (I - \Pi_P))[K]$, prove its positive definiteness, and establish that the native space of Conditional KRR is equivalent to the RKHS of the residual kernel plus $\mathcal{F}$. This allows Conditional KRR to be understood as performing linear regression on the target variable followed by standard KRR on the residuals.

**Core Idea**: By strictly reducing Conditional KRR to standard KRR on the residual kernel with a cost of only $\mathcal{O}(\log k / \sqrt{N})$, existing KRR statistical theory can be directly reused to analyze its convergence and generalization.

## Method

### Overall Architecture

Given a training set $\{(x_i, y_i)\}_{i=1}^N$ and a set of unpenalized features $\mathcal{F} = \text{span}\{f_1, \ldots, f_k\}$, Conditional KRR solves the optimization problem $\min_{f \in \mathcal{H}_K^{\mathcal{F}}} \frac{1}{N}\sum_i (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2$, where the semi-norm $\|f\|_{\mathcal{H}_K^{\mathcal{F}}}$ only penalizes the component of $f$ in the orthogonal complement of $\mathcal{F}$. Equivalently, this process can be decomposed into: (1) linear regression of $y$ using features in $\mathcal{F}$ to obtain residuals $r$; (2) standard KRR on the residuals $r$ using the residual kernel $K_P$.

### Key Designs

**1. Residual Kernel Construction and Native Space Equivalence: Translating CPD Kernel Learning to Standard PD Kernel Problems**

The difficulty of Conditional KRR lies in the fact that kernel $K$ is only conditionally positive definite with respect to $\mathcal{F}$, and the semi-norm only penalizes components in the orthogonal complement, making standard KRR theory inapplicable. The breakthrough lies in constructing the residual kernel:

$$K_P(x,y) = \big((I-\Pi_P)\otimes(I-\Pi_P)\big)[K],$$

where $\Pi_P$ is the projection operator onto $\mathcal{F}$ in $L_2(\mathcal{X},P)$. Theorem 2.1 proves that $K_P$ is a valid PD kernel. Theorem 3.1 proves that the native space $\mathcal{H}_K^{\mathcal{F}}$ is isomorphic to $\mathcal{H}_{K_P}\oplus\mathcal{F}$. Theorem 3.2 further shows that Conditional KRR is equivalent to "performing linear regression with $\mathcal{F}$ features first, then standard KRR on the residuals with $K_P$." This equivalence is the foundation of the theory—it allows all existing KRR statistical results (convergence rates, generalization bounds) to be transferred directly.

**2. High-Probability Upper Bound for Conditioning Cost: Quantifying the Penalty of Unknown $\mathcal{F}$-Projection**

Since actual learners do not know the projection $f_\parallel$ of the true regression function onto $\mathcal{F}$, the gap between the actual learner and an "ideal learner with known $f_\parallel$" must be characterized. The authors define the conditioning cost $c_{\text{con}}=\mathbb{E}[(\hat f(X)-f_\parallel(X)-h(X))^2]$. Theorem 4.2 proves that with probability $\ge 1-\delta$:

$$\mathbb{E}_\varepsilon[c_{\text{con}}] \le C\cdot\|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2\cdot\frac{k\log^{1/2}(2k/\delta)}{\sqrt N}+\frac{c_2\sigma^2}{N}.$$

The key to this bound is decoupling the contributions of the signal component and the noise: when the signal lies entirely within $\mathcal{F}$ ($f_\perp=0$), the first term vanishes, and the cost shrinks to $\mathcal{O}(\sigma^2 k/N)$. This decoupling allows for precise determination of when injecting unpenalized features is beneficial versus detrimental.

**3. Unified Analysis of Hard/Soft Thresholding: Revealing Equivalence between Spectral Truncation and Random Features at the Residual Kernel Level**

Applying the theory to practical settings reveals a U-shaped condition for test error regarding $k$. For hard thresholding, $\mathcal{F}$ is the set of top-$k$ Mercer eigenfunctions of $K$, where the residual kernel is exactly the tail of the kernel spectrum $K_P(x,y)=\sum_{i>k}\lambda_i\phi_i(x)\phi_i(y)$. For soft thresholding, $\mathcal{F}$ consists of $k$ Gaussian random features. Theorem 5.2 proves that the expected eigenvalues of the residual kernel satisfy $\mu_i/\lambda_i\approx c\varkappa^2/(\lambda_i+\varkappa)^2$, meaning large eigenvalues are "softly suppressed." Both settings exhibit the same U-shaped curve—too small a $k$ fails to utilize unpenalized features, while too large a $k$ increases the conditioning cost. This unified perspective clarifies that spectral truncation and random feature methods, though seemingly different, are qualitatively equivalent at the residual kernel level.

## Key Experimental Results

### Main Results: Conditioning Cost Validation

| Parameter Variation | Experimental Observation | Theoretical Prediction |
|----------|----------|----------|
| $N$ increases | $\hat{c}_{\text{con}} \sim 1/N$ | $\mathcal{O}(1/\sqrt{N})$ (Looser) |
| $k$ increases | Linear growth | $\mathcal{O}(k)$ |
| $\sigma^2$ increases | Linear growth | $\mathcal{O}(\sigma^2)$ |
| $f \in \mathcal{F}$ case | Only $\mathcal{O}(\sigma^2 k / N)$ remains | Consistent with theory |

### U-shaped Test Error (Hard Thresholding)

| Dataset / Kernel | Standard KRR (k=0) | Optimal Conditional KRR | Optimal k | U-shape Present |
|-------------|----------------|-------------|--------|----------|
| Synthetic Data ($s=2$) | Baseline | Significantly lower | $k=5$ | Yes |
| MNIST 7-vs-9 / Gaussian | Baseline | Lower | Moderate $k$ | Yes |
| MNIST 7-vs-9 / NNGP-erf | Baseline | Slight improvement | Overfits near $N$ | Yes (Mild) |
| MNIST 7-vs-9 / Laplace | Baseline | No improvement | — | No |

### Key Findings

- **Actual decay rate of conditioning cost** is $\sim 1/N$, faster than the theoretical upper bound $1/\sqrt{N}$, suggesting room for tightening the theory.
- **Sufficient condition for U-shaped behavior** (Eq. 3): Conditional KRR outperforms standard KRR when the signal projection on the first $k$ eigenfunctions is sufficiently strong. For pure noise ($f=0$), the test MSE increases monotonically with $k$, with no U-shape.
- **Equivalence of soft and hard thresholding**: Residual kernels generated by random Gaussian features are equivalent in expectation to "soft truncation" of the kernel spectrum, where large eigenvalues are suppressed and small ones are amplified—qualitatively consistent with hard truncation.
- **Laplace kernel lacks overfitting**: Since the first 11,000 empirical eigenfunctions all contribute to prediction, a much larger sample size is required to observe the U-shape.

## Highlights & Insights

- **Rigorous theoretical foundation for two-stage decomposition**: While the strategy of "regressing the main signal first, then learning residuals" is widely used in boosting and ResNets, this work provides the first complete statistical theory for this strategy within kernel methods. The derivation techniques for the conditioning cost (using matrix perturbation and random matrix concentration inequalities) are of general value.
- **Unified perspective on hard/soft thresholding**: By framing top-k eigenfunction truncation and random feature injection as two instances of Conditional KRR, Theorem 5.2 reveals their asymptotic equivalence at the residual kernel level. This perspective can be extended to analyze random features in neural networks.
- **Characterization of weak benign overfitting**: When the signal is entirely in $\mathcal{F}$, noise can only be memorized in the orthogonal complement, which does not affect signal recovery accuracy. This provides a new overfitting mode between benign and catastrophic overfitting.

## Limitations & Future Work

- **Theory only covers $\lambda > 0$**: When the regularization parameter $\lambda = 0$ (perfect interpolation), the existing theory fails, yet unregularized overparameterized scenarios are a major focus in practice.
- **Loose conditioning cost upper bound**: Experiments observe $\sim 1/N$ decay, but the theory only provides $\sim 1/\sqrt{N}$. Tightening this bound remains an open problem.
- **Estimation of Mercer eigenfunctions**: Hard thresholding relies on precise eigenfunction estimates. In practice, empirical eigenfunctions $\hat{\phi}_k$ are used, introducing additional errors not analyzed in detail here.
- **Lack of comparison with deep learning**: As a theoretical work on pure kernel methods, it lacks empirical comparison with modern neural network two-stage methods like residual learning.

## Related Work & Insights

- KRR Statistical Theory: Caponnetto & De Vito 2007 (Convergence rates), Simon et al. 2023 (Eigenlearning framework).
- CPD Kernel Theory: Duchon 1977 (Polynomial CPD kernels), Meinguet 1979 (Native space construction).
- Two-stage Learning: Yang et al. 2023 (Base network + Residual learning), Freund & Schapire 1997 (Boosting).
- Insights: The logic of Conditional KRR can be generalized to Deep Kernel Learning—injecting neural network features as unpenalized components into kernel methods might theoretically explain the advantages of "pre-training + kernel fine-tuning" paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/learning_theory/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](../../ICLR2026/learning_theory/scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)

</div>

<!-- RELATED:END -->
