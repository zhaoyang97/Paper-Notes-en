---
title: >-
  [Paper Note] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding
description: >-
  [ICML 2026][learning_theory][Paper Note] This paper proposes a Conditional Kernel Ridge Regression (Conditional KRR) framework that injects a set of unpenalized features into kernel methods. By reducing it to a standard KRR via a residual kernel, the authors prove a reduction cost of $\mathcal{O}(1/\sqrt{N})$ and verify sufficient conditions where Conditional
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: e8d204d86229b233
---
# Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding

**Conference**: ICML2026  
**arXiv**: [2605.26067](https://arxiv.org/abs/2605.26067)  
**Code**: None  
**Area**: Machine Learning Theory  
**Keywords**: Conditional Kernel Ridge Regression, Conditionally Positive Definite Kernels, Residual Kernels, Kernel Thresholding, Unpenalized Features

## TL;DR

This paper proposes a Conditional Kernel Ridge Regression (Conditional KRR) framework that injects a set of unpenalized features into kernel methods. By reducing it to a standard KRR via a residual kernel, the authors prove a reduction cost of $\mathcal{O}(1/\sqrt{N})$ and verify sufficient conditions where Conditional KRR outperforms standard KRR under both hard thresholding (top-k eigenfunctions) and soft thresholding (random Gaussian features) settings.

## Background & Motivation

**Background**: Kernel Ridge Regression (KRR) is a classic method in supervised learning, recently applied to Neural Tangent Kernels, operator approximation, and reinforcement learning. KRR requires a positive definite (PD) kernel $K(x,y)$, but the positive definiteness can be relaxed to conditionally positive definite (CPD), where the quadratic form $\sum_{ij} K(x_i, x_j) \zeta_i \zeta_j$ only needs to be non-negative when $\zeta$ is orthogonal to a certain function class $\mathcal{F}$.

**Limitations of Prior Work**: Existing research on CPD kernels almost exclusively focuses on cases where $\mathcal{F}$ is a polynomial subspace, leaving a vacuum in statistical theory for general function classes $\mathcal{F}$. More importantly, even if the original kernel $K$ is PD, treating it as a CPD kernel relative to a general $\mathcal{F}$ naturally leads to a "linear regression followed by kernel regression on residuals" framework, which lacks rigorous statistical guarantees.

**Key Challenge**: The optimization problem of Conditional KRR can be decomposed into linear regression + residual kernel KRR, but this decomposition depends on precise knowledge of the $\mathcal{F}$-component. In practice, the learner does not know the projection of the true regression function onto $\mathcal{F}$; thus, it is necessary to quantify the "conditioning cost" paid for not knowing $f_\parallel$.

**Goal**: (1) Establish a rigorous reduction theory from Conditional KRR to standard KRR; (2) Provide high-probability upper bounds for the conditioning cost; (3) Apply the theory to practical hard and soft thresholding settings to derive sufficient conditions for Conditional KRR to outperform standard KRR.

**Key Insight**: The authors introduce the "residual kernel" $K_P = ((I - \Pi_P) \otimes (I - \Pi_P))[K]$, prove its positive definiteness, and establish that the native space of Conditional KRR is isomorphic to the RKHS of the residual kernel plus $\mathcal{F}$. This allows Conditional KRR to be understood as performing linear regression on target variables and then running standard KRR on the residuals.

**Core Idea**: By strictly reducing Conditional KRR to standard KRR on the residual kernel with a cost of only $\mathcal{O}(\log k / \sqrt{N})$, one can directly reuse existing KRR statistical theory to analyze the convergence and generalization of Conditional KRR.

## Method

### Overall Architecture

Given a training set $\{(x_i, y_i)\}_{i=1}^N$ and a set of unpenalized features $\mathcal{F} = \text{span}\{f_1, \ldots, f_k\}$, Conditional KRR solves the optimization problem $\min_{f \in \mathcal{H}_K^{\mathcal{F}}} \frac{1}{N}\sum_i (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2$, where the semi-norm $\|f\|_{\mathcal{H}_K^{\mathcal{F}}}$ only penalizes components orthogonal to $\mathcal{F}$. Equivalently, this process can be decomposed into: (1) linear regression of $y$ using features in $\mathcal{F}$ to obtain residuals $r$; (2) standard KRR on the residuals $r$ using the residual kernel $K_P$.

### Key Designs

**1. Residual Kernel Construction and Native Space Equivalence: Translating CPD Kernel Learning to Standard PD Kernel Problems**

The difficulty with Conditional KRR is that $K$ is only conditionally positive definite relative to $\mathcal{F}$, and the semi-norm only penalizes the orthogonal complement of $\mathcal{F}$, making standard KRR theory inapplicable. The breakthrough lies in constructing the residual kernel:

$$K_P(x,y) = \big((I-\Pi_P)\otimes(I-\Pi_P)\big)[K],$$

where $\Pi_P$ is the projection operator onto $\mathcal{F}$ in $L_2(\mathcal{X},P)$. Theorem 2.1 proves that $K_P$ is a genuine PD kernel. Theorem 3.1 proves that the native space of Conditional KRR $\mathcal{H}_K^{\mathcal{F}}$ is isomorphic to $\mathcal{H}_{K_P}\oplus\mathcal{F}$. Theorem 3.2 further shows that Conditional KRR is equivalent to "performing linear regression with features in $\mathcal{F}$, then running standard KRR on the residuals with $K_P$." This equivalence is the cornerstone of the theory—once mapped back to standard PD kernels, all existing KRR statistical results (convergence rates, generalization bounds) can be transferred directly.

**2. High-Probability Upper Bound for Conditioning Cost: Quantifying the Price of Not Knowing the True $\mathcal{F}$-Projection**

Since the learner does not know the projection $f_\parallel$ of the true regression function onto $\mathcal{F}$, the gap between the actual learner and an "ideal learner" who knows $f_\parallel$ must be characterized. The authors define the conditioning cost as $c_{\text{con}}=\mathbb{E}[(\hat f(X)-f_\parallel(X)-h(X))^2]$. Theorem 4.2 proves that with probability $\ge 1-\delta$:

$$\mathbb{E}_\varepsilon[c_{\text{con}}] \le C\cdot\|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2\cdot\frac{k\log^{1/2}(2k/\delta)}{\sqrt N}+\frac{c_2\sigma^2}{N}.$$

The key to this bound is decoupling the contributions of the signal and noise: the first term vanishes when the signal lies entirely within $\mathcal{F}$ ($f_\perp=0$), and the cost reduces to $\mathcal{O}(\sigma^2 k/N)$. this decoupling allows for precise judgment on when injecting unpenalized features is beneficial versus detrimental.

**3. Unified Analysis of Hard/Soft Thresholding: Revealing Equivalence of Spectral Truncation and Random Features at the Residual Kernel Level**

Finally, the theory is applied to two practical settings, resulting in a U-shaped condition for test error relative to $k$. In hard thresholding, $\mathcal{F}$ consists of the top-$k$ Mercer eigenfunctions of $K$; here, the residual kernel is exactly the tail of the kernel spectrum $K_P(x,y)=\sum_{i>k}\lambda_i\phi_i(x)\phi_i(y)$. In soft thresholding, $\mathcal{F}$ consists of $k$ Gaussian random features. Theorem 5.2 proves that the expected eigenvalues of the residual kernel satisfy $\mu_i/\lambda_i\approx c\varkappa^2/(\lambda_i+\varkappa)^2$, meaning large eigenvalues are "softly suppressed." Both settings exhibit the same U-shaped curve—if $k$ is too small, unpenalized features are underutilized; if $k$ is too large, the conditioning cost increases. This unified perspective reveals that traditional spectral truncation and random feature methods, while seemingly different, are qualitatively equivalent at the residual kernel level.

## Key Experimental Results

### Main Results: Conditioning Cost Verification

| Parameter Change | Experimental Observation | Theoretical Prediction |
|------------------|--------------------------|------------------------|
| $N$ Increases | $\hat{c}_{\text{con}} \sim 1/N$ | $\mathcal{O}(1/\sqrt{N})$ (Looser) |
| $k$ Increases | Linear growth | $\mathcal{O}(k)$ |
| $\sigma^2$ Increases | Linear growth | $\mathcal{O}(\sigma^2)$ |
| When $f \in \mathcal{F}$ | Only $\mathcal{O}(\sigma^2 k / N)$ remains | Consistent with theory |

### U-shaped Test Error (Hard Thresholding)

| Dataset / Kernel | Standard KRR (k=0) | Optimal Conditional KRR | Optimal k | U-shape Present |
|------------------|--------------------|-------------------------|-----------|-----------------|
| Synthetic ($s=2$) | Baseline | Significantly lower | $k=5$ | Yes |
| MNIST 7-vs-9 / Gaussian | Baseline | Lower | Medium $k$ | Yes |
| MNIST 7-vs-9 / NNGP-erf | Baseline | Slight improvement | Overfits near $N$ | Yes (Mild) |
| MNIST 7-vs-9 / Laplace | Baseline | No improvement | — | No |

### Key Findings

- **Actual decay rate of conditioning cost** is $\sim 1/N$, faster than the theoretical upper bound of $1/\sqrt{N}$, suggesting room for tightening the theory.
- **Sufficient conditions for U-shape behavior** (Eq. 3): Conditional KRR outperforms standard KRR when the signal projection onto the first $k$ eigenfunctions is strong enough. For pure noise data ($f=0$), test MSE increases monotonically with $k$, showing no U-shape.
- **Equivalence of soft and hard thresholding**: Residual kernels generated by random Gaussian features are equivalent in expectation to "soft truncation" of the kernel spectrum, where large eigenvalues are suppressed and small ones are amplified, qualitatively matching hard truncation.
- **Laplace kernel shows no overfitting**: Because the first 11,000 empirical eigenfunctions all contribute to prediction, a much larger sample size is needed to observe the U-shape.

## Highlights & Insights

- **Rigorous theoretical foundation for two-stage decomposition**: While the strategy of "regress major signals first, then learn residuals with kernels" is widely used in boosting and ResNets, this paper provides the first complete statistical theory for this strategy within a kernel framework. The derivation of the conditioning cost bound (using matrix perturbation and random matrix concentration) is of general value.
- **Unified view of hard/soft thresholding**: By unifying top-k eigenfunction truncation and random feature injection as two instances of Conditional KRR, Theorem 5.2 reveals their asymptotic equivalence at the residual kernel level. This perspective can be transferred to the analysis of random features in neural networks.
- **Characterization of weak benign overfitting**: When the signal is entirely in $\mathcal{F}$, noise can only be memorized in the orthogonal complement, which does not affect signal recovery accuracy—providing a new mode of overfitting between benign and catastrophic.

## Limitations & Future Work

- **Theory only covers $\lambda > 0$**: When the regularization parameter $\lambda = 0$ (perfect memorization), existing theory fails, yet unregularized overparameterized scenarios are a major focus in practice.
- **Loose conditioning cost bound**: Experiments show $\sim 1/N$ decay, but the theory only gives $\sim 1/\sqrt{N}$. Tightening this bound remains an open problem.
- **Estimation of Mercer eigenfunctions**: Hard thresholding relies on precise eigenfunction estimation. In practice, empirical eigenfunctions $\hat{\phi}_k$ are used, introducing errors not analyzed in detail here.
- **Lack of comparison with deep learning**: As a theoretical work on pure kernel methods, it lacks empirical comparisons with modern two-stage neural network methods (e.g., residual learning).

## Related Work & Insights

- KRR Statistical Theory: Caponnetto & De Vito 2007 (convergence rates), Simon et al. 2023 (eigenlearning framework).
- CPD Kernel Theory: Duchon 1977 (polynomial CPD kernels), Meinguet 1979 (native space construction).
- Two-stage Learning: Yang et al. 2023 (Base network + residual learning), Freund & Schapire 1997 (Boosting).
- Insight: The logic of Conditional KRR can be extended to deep kernel learning—injecting neural network features as unpenalized components into kernel methods might theoretically explain the advantages of the "pre-training + kernel fine-tuning" paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Sequential Kernel-based Conditional Independence Testing via Adaptive Betting](sequential_kernel-based_conditional_independence_testing_via_adaptive_betting.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/learning_theory/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](../../ICLR2026/learning_theory/scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)

</div>

<!-- RELATED:END -->
