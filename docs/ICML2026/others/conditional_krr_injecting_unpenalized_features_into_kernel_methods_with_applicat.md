---
title: >-
  [Paper Note] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding
description: >-
  [ICML2026][Conditional Kernel Ridge Regression] This paper proposes the Conditional Kernel Ridge Regression (Conditional KRR) framework…
tags:
  - "ICML2026"
  - "Conditional Kernel Ridge Regression"
  - "Conditionally Positive Definite Kernels"
  - "Residual Kernels"
  - "Kernel Thresholding"
  - "Unpenalized Features"
date: 2026-05-08
content_hash: e1a448b5cbaf1ac9
---

# Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding

**Conference**: ICML2026  
**arXiv**: [2605.26067](https://arxiv.org/abs/2605.26067)  
**Code**: None  
**Area**: Machine Learning Theory  
**Keywords**: Conditional Kernel Ridge Regression, Conditionally Positive Definite Kernels, Residual Kernels, Kernel Thresholding, Unpenalized Features  

## TL;DR

This paper proposes the Conditional Kernel Ridge Regression (Conditional KRR) framework, which injects a set of unpenalized features into kernel methods. By reducing it to standard KRR via a residual kernel, the authors prove that the reduction cost is $\mathcal{O}(1/\sqrt{N})$. Furthermore, the study identifies sufficient conditions for Conditional KRR to outperform standard KRR under both hard thresholding (top-k eigenfunctions) and soft thresholding (random Gaussian features) settings.

## Background & Motivation

**Background**: Kernel Ridge Regression (KRR) is a classic supervised learning method, recently seeing broad application in areas like Neural Tangent Kernels, operator approximation, and reinforcement learning. Using KRR requires specifying a positive definite (PD) kernel $K(x,y)$. However, the positivity requirement can be relaxed to conditionally positive definite (CPD), where the quadratic form $\sum_{ij} K(x_i, x_j) \zeta_i \zeta_j$ only needs to be non-negative when $\zeta$ is orthogonal to a specific function class $\mathcal{F}$.

**Limitations of Prior Work**: Existing research on CPD kernels focus almost exclusively on cases where $\mathcal{F}$ is a polynomial subspace, leaving a statistical theory vacuum for general function classes $\mathcal{F}$. Crucially, even when the original kernel $K$ is PD, treating it as a CPD kernel relative to a general $\mathcal{F}$ naturally leads to a two-stage learning framework—"linear regression first, then kernel regression on residuals"—which lacks rigorous statistical guarantees.

**Key Challenge**: The optimization of conditional KRR can be decomposed into linear regression plus residual kernel KRR, but this decomposition relies on exact knowledge of the $\mathcal{F}$-component. In practice, the learner does not know the true projection of the regression function onto $\mathcal{F}$, necessitating the quantification of the "conditioning cost"—the price paid for not knowing $f_\parallel$.

**Goal**: (1) Establish a rigorous reduction theory from Conditional KRR to standard KRR; (2) Provide high-probability upper bounds for the conditioning cost; (3) Apply the theory to practical hard/soft thresholding settings to derive sufficient conditions for Conditional KRR superiority.

**Key Insight**: The authors introduce the "residual kernel" $K_P = ((I - \Pi_P) \otimes (I - \Pi_P))[K]$ and prove its positive definiteness. They demonstrate that the native space of Conditional KRR is isomorphic to the RKHS of the residual kernel plus $\mathcal{F}$. This allows Conditional KRR to be interpreted as standard KRR applied to the residuals after performing linear regression on the target variables.

**Core Idea**: The study rigorously reduces Conditional KRR to standard KRR on a residual kernel with a cost of only $\mathcal{O}(\log k / \sqrt{N})$, enabling the direct application of existing KRR statistical theories to analyze convergence and generalization.

## Method

### Overall Architecture

Given a training set $\{(x_i, y_i)\}_{i=1}^N$ and a set of unpenalized features $\mathcal{F} = \text{span}\{f_1, \ldots, f_k\}$, Conditional KRR solves the optimization problem $\min_{f \in \mathcal{H}_K^{\mathcal{F}}} \frac{1}{N}\sum_i (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2$, where the semi-norm $\|f\|_{\mathcal{H}_K^{\mathcal{F}}}$ only penalizes components orthogonal to $\mathcal{F}$. Equivalently, the process decomposes into: (1) performing linear regression on $y$ using features in $\mathcal{F}$ to obtain residuals $r$; (2) applying standard KRR to the residuals $r$ using the residual kernel $K_P$.

### Key Designs

1.  **Residual Kernel Construction and Native Space Equivalence**:
    *   Function: Converts the conditional learning problem of a CPD kernel into a standard learning problem of a PD kernel.
    *   Mechanism: For a CPD kernel $K$ and input distribution $P$, the residual kernel is defined as $K_P(x,y) = ((I - \Pi_P) \otimes (I - \Pi_P))[K]$, where $\Pi_P$ is the projection operator onto $\mathcal{F}$ in $L_2(\mathcal{X}, P)$. Theorem 2.1 proves $K_P$ is a PD kernel; Theorem 3.1 proves that the native space $\mathcal{H}_K^{\mathcal{F}}$ is isomorphic to $\mathcal{H}_{K_P} \oplus \mathcal{F}$. Theorem 3.2 further shows that Conditional KRR is equivalent to linear regression followed by $K_P$-based KRR.
    *   Design Motivation: This equivalence serves as the theoretical pillar, allowing existing KRR statistical results to be transferred directly to Conditional KRR.

2.  **High-Probability Bound for Conditioning Cost**:
    *   Function: Quantifies the gap between Conditional KRR and an "ideal learner" that knows $f_\parallel$.
    *   Mechanism: Defining the conditioning cost as $c_{\text{con}} = \mathbb{E}[(\hat{f}(X) - f_\parallel(X) - h(X))^2]$, Theorem 4.2 proves that with probability $\geq 1 - \delta$, $\mathbb{E}_\varepsilon[c_{\text{con}}] \leq C \cdot \|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2 \cdot \frac{k \log^{1/2}(2k/\delta)}{\sqrt{N}} + \frac{c_2 \sigma^2}{N}$. When the signal lies entirely within $\mathcal{F}$ ($f_\perp = 0$), the first term vanishes, and the cost becomes $\mathcal{O}(\sigma^2 k / N)$.
    *   Design Motivation: This bound decouples the contributions of signal components and noise, facilitating precise determination of when Conditional KRR outperforms standard KRR.

3.  **Unified Analysis of Hard/Soft Thresholding**:
    *   Function: Applies the theory to practical settings to reveal the U-shaped dependence of test error on $k$.
    *   Mechanism: Hard thresholding sets $\mathcal{F}$ as the top-$k$ Mercer eigenfunctions of $K$, making the residual kernel the spectral tail $\sum_{i>k} \lambda_i \phi_i(x) \phi_i(y)$. Soft thresholding sets $\mathcal{F}$ as $k$ Gaussian random features. Theorem 5.2 proves the expected eigenvalues of the residual kernel satisfy $\mu_i/\lambda_i \approx c\varkappa^2 / (\lambda_i + \varkappa)^2$, effectively "soft-suppressing" large eigenvalues. Both settings exhibit a U-shaped curve: small $k$ fails to utilize unpenalized features, while large $k$ increases conditioning cost.
    *   Design Motivation: Unified analysis links spectral truncation methods with random feature methods.

## Key Experimental Results

### Main Results: Conditioning Cost Verification

| Parameter Change | Experimental Observation | Theoretical Prediction |
|------------------|--------------------------|------------------------|
| $N$ increases    | $\hat{c}_{\text{con}} \sim 1/N$ | $\mathcal{O}(1/\sqrt{N})$ (Looser) |
| $k$ increases    | Linear growth            | $\mathcal{O}(k)$       |
| $\sigma^2$ increases | Linear growth        | $\mathcal{O}(\sigma^2)$|
| $f \in \mathcal{F}$ | Remains $\mathcal{O}(\sigma^2 k / N)$ | Consistent with theory |

### U-Shaped Test Error (Hard Thresholding)

| Dataset / Kernel | Standard KRR (k=0) | Best Conditional KRR | Optimal k | U-shape Present |
|------------------|--------------------|----------------------|-----------|-----------------|
| Synthetic ($s=2$) | Baseline           | Significantly lower  | $k=5$     | Yes             |
| MNIST 7-vs-9 / Gaussian | Baseline      | Lower                | Medium $k$| Yes             |
| MNIST 7-vs-9 / NNGP-erf | Baseline      | Slight improvement   | Overfitting near $N$ | Yes (Mild) |
| MNIST 7-vs-9 / Laplace | Baseline       | No improvement       | —         | No              |

### Key Findings

- **Actual decay rate of conditioning cost** is $\sim 1/N$, which is faster than the theoretical bound of $1/\sqrt{N}$, suggesting room for tighter theoretical bounds.
- **Sufficient conditions for U-shaped behavior** (Eq. 3): Conditional KRR outperforms standard KRR when the signal projection onto the top-$k$ eigenfunctions is sufficiently strong. For pure noise ($f=0$), MSE increases monotonically with $k$.
- **Equivalence of soft and hard thresholding**: Residual kernels generated by random Gaussian features are equivalent in expectation to "softly truncating" the kernel spectrum, where large eigenvalues are suppressed and small ones are amplified.
- **Absence of overfitting in Laplace kernels**: Because the first 11,000 empirical eigenfunctions all contribute to the prediction, a much larger sample size is required to observe the U-shaped curve.

## Highlights & Insights

- **Rigorous foundation for two-stage decomposition**: While the strategy of "regressing the main signal first, then learning residuals with kernels" is common in boosting and ResNets, this work provides the first comprehensive statistical theory for this strategy within the kernel framework.
- **Unified perspective on thresholding**: By treating top-$k$ eigenfunction truncation and random feature injection as instances of Conditional KRR, Theorem 5.2 reveals their asymptotic equivalence at the residual kernel level.
- **Characterization of weakly benign overfitting**: When the signal is entirely within $\mathcal{F}$, noise can only be memorized in the orthogonal complement, which does not impede signal recovery. This suggests a new mode of overfitting between benign and catastrophic.

## Limitations & Future Work

- **Theory limited to $\lambda > 0$**: The current theory fails when $\lambda = 0$ (perfect memorization), yet the interpolating regime of overparameterized models is of significant practical interest.
- **Loose conditioning cost upper bound**: Experiments show $1/N$ decay, but the theory only guarantees $1/\sqrt{N}$. Tightening this bound remains an open problem.
- **Estimation of Mercer eigenfunctions**: Hard thresholding relies on precise eigenfunctions, while practitioners must use empirical eigenfunctions $\hat{\phi}_k$, introducing additional errors not analyzed in detail.
- **Lack of comparison with deep learning**: As a purely kernel-theoretic work, it lacks empirical comparison with modern two-stage neural network methods (e.g., residual learning).

## Related Work & Insights

- KRR Statistical Theory: Caponnetto & De Vito 2007 (convergence rates); Simon et al. 2023 (eigenlearning framework).
- CPD Kernel Theory: Duchon 1977 (polynomial CPD kernels); Meinguet 1979 (native space construction).
- Two-stage Learning: Yang et al. 2023 (basis networks + residual learning); Freund & Schapire 1997 (Boosting).
- Insight: The logic of Conditional KRR can be extended to deep kernel learning—injecting neural network features as unpenalized components into kernel methods may theoretically explain the advantages of "pre-training + kernel fine-tuning."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/others/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](../../ICLR2026/others/scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[ICML 2026\] New Bounds for Kernel Sums via Fast Spherical Embeddings](new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](../../ICLR2026/others/probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICML 2026\] Inference of Online Newton Methods with Nesterov's Accelerated Sketching](inference_of_online_newton_methods_with_nesterovs_accelerated_sketching.md)

</div>

<!-- RELATED:END -->
