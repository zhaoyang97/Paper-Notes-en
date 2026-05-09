---
title: >-
  [Paper Note] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics
description: >-
  [ICLR 2026][kernel regression learning curves] This paper proposes the Hermite Eigenstructure Ansatz (HEA), which analytically predicts the learning curves (test error vs. sample size) of rotation-invariant kernels on real image datasets (CIFAR-5m, SVHN, ImageNet) using only two statistics: the data covariance matrix and the Hermite decomposition of the target function. The paper proves that HEA holds for Gaussian data and empirically demonstrates that MLPs in the feature-learning regime also learn Hermite polynomials in the order predicted by HEA.
tags:
  - ICLR 2026
  - kernel regression learning curves
  - Hermite eigenstructure
  - anisotropic data
  - kernel ridge regression
  - feature learning
date: 2026-05-08
content_hash: cf46ef8fb173c4ad
---

# Predicting Kernel Regression Learning Curves from Only Raw Data Statistics

**Conference**: ICLR 2026
**arXiv**: [2510.14878](https://arxiv.org/abs/2510.14878)
**Code**: [https://github.com/JoeyTurn/hermite-eigenstructure-ansatz](https://github.com/JoeyTurn/hermite-eigenstructure-ansatz)
**Area**: Others / Learning Theory / Kernel Methods
**Keywords**: kernel regression learning curves, Hermite eigenstructure, anisotropic data, kernel ridge regression, feature learning

## TL;DR
This paper proposes the Hermite Eigenstructure Ansatz (HEA), which analytically predicts the learning curves (test error vs. sample size) of rotation-invariant kernels on real image datasets (CIFAR-5m, SVHN, ImageNet) using only two statistics: the data covariance matrix and the Hermite decomposition of the target function. The paper proves that HEA holds for Gaussian data and empirically demonstrates that MLPs in the feature-learning regime also learn Hermite polynomials in the order predicted by HEA.

## Background & Motivation

**Background**: Kernel ridge regression (KRR) serves as an important proxy for understanding neural networks (via NTK equivalence), and a well-developed eigenframework exists for predicting test error from the kernel's eigenstructure. This framework relies on the eigenvalues and eigenfunctions of the kernel with respect to the data distribution.

**Limitations of Prior Work**: Although the eigenframework is theoretically complete, applying it in practice requires constructing and diagonalizing the kernel matrix to obtain the eigenstructure — a procedure that is computationally expensive for high-dimensional real data and yields little analytic insight. More fundamentally, existing theory almost exclusively relies on simplified data assumptions (e.g., isotropic spherical distributions) and cannot be directly applied to real anisotropic datasets.

**Key Challenge**: Real data distributions are extremely complex and cannot be fully described in closed form, yet learning behavior is profoundly shaped by data structure. Predicting performance on real data requires balancing "compact description of data" against "prediction accuracy."

**Goal**: (a) Can one find a parsimonious description of the data distribution that is simple enough yet sufficient to predict the learning behavior of kernel regression? (b) Can learning curves be predicted directly from data statistics without constructing a kernel matrix?

**Key Insight**: The authors observe that for Gaussian data, the eigenfunctions of rotation-invariant kernels are naturally multivariate Hermite polynomials, and that real image data is "sufficiently Gaussian" (coordinatewise marginals are approximately Gaussian). This motivates the conjecture that such structure approximately holds for real data as well.

**Core Idea**: The eigenstructure of a rotation-invariant kernel on anisotropic data is approximately equal to the Hermite eigenstructure — eigenfunctions are Hermite polynomials along the PCA directions of the data, and eigenvalues are monomials of the covariance eigenvalues weighted by the kernel's level coefficients.

## Method

### Overall Architecture
The method forms an end-to-end pipeline from raw data statistics to learning curve predictions:

**Input**: Data covariance matrix $\Sigma = U \Gamma U^\top$ (estimated from samples) + kernel function $K$ + label samples of the target function $f_*$

**Intermediate Steps**:
1. Compute the Hermite eigenstructure $\mathcal{HE}(\Sigma, (c_\ell))$ from $\Sigma$ and the kernel function.
2. Estimate the coefficients $v_i$ of the target function in the Hermite basis via Gram-Schmidt orthogonalization.
3. Substitute $(\lambda_\alpha, v_i)$ into the KRR eigenframework.

**Output**: Predicted learning curve of test error as a function of sample size.

Key advantage: No kernel matrix needs to be constructed or diagonalized at any stage.

### Key Designs

1. **Hermite Eigenstructure Ansatz (HEA)**:

   - Function: Asserts that the eigenstructure of a rotation-invariant kernel can be approximated by a simple analytic form.
   - Mechanism: For any multi-index $\alpha \in \mathbb{N}_0^d$, the eigenvalues are proposed as $\lambda_\alpha = c_{|\alpha|} \cdot \prod_{i=1}^d \gamma_i^{\alpha_i}$ and the eigenfunctions as $\phi_\alpha = h_\alpha^{(\Sigma)}$ (multivariate Hermite polynomials), where $c_\ell$ are the on-sphere level coefficients of the kernel and $\gamma_i$ are the eigenvalues of the data covariance.
   - Design Motivation: The intuition comes from analyzing the wide-kernel limit of the Gaussian kernel. When $\sigma^2 \gg \gamma$, the variance of each component of the kernel feature map decays exponentially as $\sigma^{-2\ell} \gamma^\ell$, and PCA of these components naturally yields Gram-Schmidt orthogonalization, recovering Hermite polynomials. This structure holds approximately for any rotation-invariant kernel and sufficiently high-dimensional data.

2. **On-Sphere Level Coefficients**:

   - Function: Converts any rotation-invariant kernel into a dot-product kernel and extracts the polynomial coefficients $c_\ell$ at each degree.
   - Mechanism: On the sphere of typical data radius $r = \text{Tr}[\Sigma]^{1/2}$, a rotation-invariant kernel can be expanded as $K(x,x') = \sum_\ell \frac{c_\ell}{\ell!}(x^\top x')^\ell$. The paper derives level coefficients for the Gaussian kernel ($c_\ell = e^{-r^2/\sigma^2} \cdot \sigma^{-2\ell}$), the Laplace kernel (involving Bessel polynomials), and ReLU NNGP/NTK kernels.
   - Design Motivation: Not all rotation-invariant kernels are naturally dot-product kernels (e.g., the Laplace kernel is non-analytic at zero), but high-dimensional data concentrates on a thin shell, making the dot-product approximation safe.

3. **Hermite Decomposition of the Target Function**:

   - Function: Estimates the projection of the target function onto the Hermite basis from finite labeled samples.
   - Mechanism: Direct inner-product estimation can overestimate power in overlapping modes due to slight non-Gaussianity of real data, causing imperfect orthogonality of the Hermite basis. The solution is to first apply Gram-Schmidt orthogonalization to the empirical Hermite polynomials: $h_i^{(\text{GS})} = \text{unitnorm}(h_i - \sum_{j<i} \langle h_j^{(\text{GS})}, h_i \rangle h_j^{(\text{GS})})$, followed by projection $\hat{v}_i = \langle h_i^{(\text{GS})}, y \rangle$.
   - Design Motivation: This step is independent of the kernel choice (does not depend on $c_\ell$), meaning a single decomposition can be reused for learning curve predictions across all kernels. Experiments use $P = 30000$ modes and $N = 80000$ samples.

### Theoretical Analysis

The paper proves two theorems establishing HEA for Gaussian data:

- **Theorem 1 (Wide Gaussian Kernel)**: When data $\mu = \mathcal{N}(0, \Sigma)$ and the Gaussian kernel width $\sigma \to \infty$, the true eigenstructure converges to the Hermite eigenstructure. The proof uses the Mehler formula in the limit.

- **Theorem 2 (Fast-Decaying Dot-Product Kernels)**: When the level coefficients satisfy $c_{\ell+1} \leq \epsilon \cdot c_\ell$ with $\epsilon \to 0$, the eigenstructure converges linearly to HEA. The proof is based on perturbation theory, decomposing the kernel eigenstructure into exponentially separated levels.

Three conditions under which HEA holds well:
- Rapidly decaying level coefficients ($c_\ell \gg \gamma_1 c_{\ell+1}$)
- High effective data dimensionality ($d_\text{eff} = \text{Tr}[\Sigma]^2 / \text{Tr}[\Sigma^2] \gg 1$, especially important for non-smooth kernels such as Laplace)
- Data distribution that is "sufficiently Gaussian" (complex image datasets actually satisfy this better than simple datasets such as MNIST)

## Key Experimental Results

### Main Results: Learning Curve Prediction

| Dataset | Kernel | Target Type | HEA Prediction | Notes |
|---------|--------|-------------|----------------|-------|
| CIFAR-5m | Gaussian (σ=6) | Synthetic Hermite polynomial $h_1(z_1)$ | Exact match | Sample complexity accurately predicted for linear → quadratic → cubic targets |
| CIFAR-5m | Gaussian (σ=6) | vehicles vs. animals | Good match | Binarized real labels; both shape and absolute value of learning curve accurate |
| CIFAR-5m | Laplace (σ=8√2) | domesticated vs. wild | Good match | Non-smooth kernel also predictable; ZCA preprocessing needed to increase $d_\text{eff}$ |
| SVHN | Gaussian (σ=6) | even vs. odd | Good match | Generalization validated across datasets |
| SVHN | Laplace | prime vs. composite | Good match | Semantically more complex binary classification |
| ImageNet-32 | ReLU NTK | Synthetic polynomial | Exact match | NTK kernel with real high-resolution data |
| ImageNet-32 | ReLU NTK | Synthetic power-law target | Exact match | Accurate across different source exponents $\beta$ |

### Eigenstructure Validation (Figure 2)

| Kernel/Data Combination | $d_\text{eff}$ | Eigenvalue Match | Eigenfunction Subspace Overlap | Notes |
|-------------------------|----------------|------------------|-------------------------------|-------|
| Gaussian kernel + Gaussian data ($d=200$) | ~7 | Exact | Concentrated on diagonal | Theoretically guaranteed setting |
| Gaussian kernel + CIFAR-5m | ~9 | Good | Concentrated on diagonal | Natural images also satisfy HEA |
| Laplace kernel + SVHN (ZCA) | ~21 | Good | Concentrated on diagonal | Requires $d_\text{eff} \geq 20$ |
| ReLU NTK + ImageNet-32 (ZCA) | ~40 | Good | Concentrated on diagonal | High bias-variance ratio substitutes for wide-kernel condition |

### MLP Feature Learning Validation

| Dataset | Network | Target | Finding |
|---------|---------|--------|---------|
| Gaussian data | 3-layer ReLU MLP | Hermite polynomials of various degrees | Optimization time $\eta \cdot n_\text{iter}$ proportional to $\lambda_\alpha^{-1/2}$ |
| CIFAR-5m | 3-layer ReLU MLP | Multivariate Hermite polynomials | Learning order consistent with HEA eigenvalue ranking |

### Key Findings
- HEA performs better on complex image datasets than on simple datasets (MNIST, tabular data) — the blessing of dimensionality causes high-dimensional data coordinates to more closely follow a Gaussian distribution via the central limit theorem effect.
- For the Laplace kernel, level coefficients $c_\ell$ grow super-exponentially with $\ell$, causing theoretical eigenvalues to diverge at high orders. In practice, truncating to $\ell \in [5, 10]$ yields good approximations — the expansion behaves more like an asymptotic series than a convergent one.
- Gram-Schmidt orthogonalization in the target function decomposition is a critical step for prediction accuracy. Direct linear regression fails due to model misspecification and non-orthogonality.
- A single target function decomposition can be reused for learning curve predictions across all kernels (kernel-agnosticism), substantially reducing computational cost.

## Highlights & Insights

- **Proof of concept for end-to-end analytic theory**: This is possibly the first work to achieve fully analytic, end-to-end prediction of the chain "data structure → model performance" on real datasets. Learning curves are predicted from only the covariance matrix $\Sigma$ and the Hermite decomposition, without constructing any kernel matrix — at far lower computational cost than traditional kernel matrix diagonalization.

- **Parsimonious data description**: Using the covariance matrix and Hermite coefficients as a "reduced description" of the data precisely captures the information relevant to kernel learners. This perspective transfers naturally to designing better data featurization methods or data selection strategies.

- **Applicability of HEA to MLPs**: Although the theory targets kernel regression, experiments reveal that MLPs in the feature-learning regime also learn Hermite polynomials in the order predicted by HEA, suggesting that HEA may reflect a more general learning principle with potential implications for deep learning theory.

- **Blessing of dimensionality**: High dimensionality is typically regarded as a curse, but here complex high-dimensional image data satisfies HEA better than low-dimensional simple data — because central limit theorem effects make individual coordinates more Gaussian. This insight can guide the design of both theory and experiments.

## Limitations & Future Work

- **Restricted to rotation-invariant kernels**: HEA assumes rotational invariance and does not directly apply to non-rotation-invariant kernels (e.g., learned NTKs, attention kernels). Extending HEA to more general kernel classes is an important direction.
- **"Sufficiently Gaussian" condition lacks precise quantification**: The paper uses only a rough coordinatewise Gaussianity criterion without a quantitative threshold. The failure cases on MNIST and tabular data demonstrate that this condition is non-trivial.
- **Divergent high-order level coefficients (Laplace/ReLU kernels)**: For non-smooth kernels, super-exponential growth of $c_\ell$ causes theoretical eigenvalues to diverge at high orders, requiring manual truncation. A more principled treatment (e.g., asymptotic expansion theory) remains to be developed.
- **Ridge parameter selection not addressed**: Learning curve predictions assume a known ridge parameter $\delta$, which in practice requires cross-validation. How to use HEA to simultaneously predict the optimal $\delta$ is not discussed.
- **MLP connection is purely empirical**: Although the MLP experiments are compelling, no theoretical explanation is provided for why feature-learning MLPs obey the HEA ordering. Establishing a formal MLP–HEA connection is a natural direction for follow-up work.

## Related Work & Insights

- **vs. KRR eigenframework (Simon et al. 2021)**: That work provides a mapping from the eigenstructure to test risk, but requires numerically solving for the eigenstructure first. The present paper fills the missing link — analytically constructing the eigenstructure from raw data statistics.
- **vs. Wortsman & Loureiro (2025)**: A concurrent work studying the same mathematical problem (eigenstructure of dot-product kernels on anisotropic Gaussian data), but only proving upper and lower bounds on eigenvalues, and switching to Hermite polynomial kernels when analyzing KRR generalization. HEA provides a theoretical justification for this substitution.
- **vs. Ghorbani et al. (2020)**: That work derives exact eigenstructures on products of spheres (isotropic multi-sphere distributions). HEA unifies these results and extends them to the continuous anisotropic setting.
- **vs. single/multi-index model literature**: Existing work focuses on asymptotic scaling laws for learning Hermite polynomials, whereas this paper pursues exact predictions including constant prefactors and handles anisotropic data, making results applicable to real datasets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to achieve end-to-end prediction of kernel regression learning curves on real datasets from raw data statistics; HEA is an elegant and powerful unifying framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple kernels, datasets, and both synthetic and real targets, with theoretical proofs and empirical validation; lacks large-scale or high-resolution data experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Fluent exposition balancing intuitive explanation with formal proof; the end-to-end pipeline visualization in Figure 1 is exceptionally clear.
- Value: ⭐⭐⭐⭐ — Significant implications for the learning theory community, demonstrating the feasibility of end-to-end analytic theory on real data; the MLP connection broadens practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/others/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[AAAI 2026\] Life, Machine Learning, and the Search for Habitability: Predicting Biosignature Fluxes for the Habitable Worlds Observatory](../../AAAI2026/others/life_machine_learning_and_the_search_for_habitability_predicting_biosignature_fl.md)
- [\[ICLR 2026\] On the Impact of the Utility in Semivalue-based Data Valuation](on_the_impact_of_the_utility_in_semivalue-based_data_valuation.md)

</div>

<!-- RELATED:END -->
