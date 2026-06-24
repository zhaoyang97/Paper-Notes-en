---
title: >-
  [Paper Note] Toward Scalable and Valid Conditional Independence Testing with Spectral Representations
description: >-
  [ICML2026][Causal Inference][Conditional Independence Testing] SpectralCIT approximates the "partial covariance operator"—a core concept in kernel methods for characterizing conditional independence—using low-dimensional spectral features learned by neural networks. It then constructs a simple HSIC-like statistic for testing. By employing a bi-level contrastive algorithm to learn the leading singular features of the operator, the authors prove that the statistic asymptoticall…
tags:
  - "ICML2026"
  - "Causal Inference"
  - "Conditional Independence Testing"
  - "Partial Covariance Operator"
  - "Spectral Representation Learning"
  - "Contrastive Learning"
  - "Chi-squared Null Distribution"
date: 2026-05-08
content_hash: 911e75fd3c61011d
---

# Toward Scalable and Valid Conditional Independence Testing with Spectral Representations

**Conference**: ICML2026  
**arXiv**: [2512.19510](https://arxiv.org/abs/2512.19510)  
**Code**: [github.com/alekfrohlich/SCIT](https://github.com/alekfrohlich/SCIT)  
**Area**: Causal Inference / Conditional Independence Testing / Representation Learning  
**Keywords**: Conditional Independence Testing, Partial Covariance Operator, Spectral Representation Learning, Contrastive Learning, Chi-squared Null Distribution

## TL;DR
SpectralCIT approximates the "partial covariance operator"—a core concept in kernel methods for characterizing conditional independence—using low-dimensional spectral features learned by neural networks. It then constructs a simple HSIC-like statistic for testing. By employing a bi-level contrastive algorithm to learn the leading singular features of the operator, the authors prove that the statistic asymptotically follows a chi-squared distribution under the null hypothesis and possesses power guarantees under the alternative hypothesis. This approach effectively bridges the theoretical rigor of kernel methods with the scalability of modern representation learning.

## Background & Motivation
**Background**: Conditional Independence (CI) testing—determining whether $X\perp\!\!\!\perp Y\mid Z$ holds given $X, Y, Z$—is a cornerstone of causal inference, graphical models, and variable selection. However, it is notoriously difficult in non-parametric settings. The "no-free-lunch" theorem by Shah and Peters (2020) proved that any test capable of uniformly controlling the Type I error across all conditionally independent distributions cannot have power against any alternative hypothesis. Intuitively, the space of CI distributions is too "rich"; any strongly dependent samples can be arbitrarily approximated by a conditionally independent model unless structural assumptions (e.g., $P_{X,Y\mid Z=z}$ varying continuously with $z$) are imposed on $P_{X,Y,Z}$.

**Limitations of Prior Work**: This predicament forces a shift from "universally valid" to "setting-specific" tests, each with drawbacks. Kernel methods (KCIT, RCIT) rely on regressions from $Z$ to $X, Y$, requiring accurate regression performance. Model-X methods (GCIT, DGCIT) require access to $P(X\mid Z)$ or its reliable approximation. Local permutation tests (NNLSCIT) permute samples based on $Z$ bins/clusters but are computationally expensive and dependent on smoothness assumptions. Among these, **classic kernel methods** use partial covariance operators to characterize conditional dependence. This operator implicitly encodes a broad class of structural assumptions (smoothness, sparsity, low-rank, latent variables) and is highly versatile; however, kernel methods **lack adaptivity and scalability** (difficult kernel selection, loss of power in high dimensions, high computational cost), limiting their practical impact.

**Key Challenge**: The partial covariance operator framework is theoretically elegant (providing a unified characterization of CI without relying on a single structural assumption), but it is tied to kernel methods. Kernel methods are neither scalable nor adaptive for high-dimensional data and large samples. The universality of theory and the scalability of implementation have remained decoupled.

**Goal**: To retain the universal characterization of the partial covariance operator while replacing kernels with learnable, scalable representations. The objective is to create a CI test that is both valid (controlling Type I error) and powerful, while scaling to high dimensions.

**Key Insight**: Recent successes in "learning leading spectral features of statistical operators" for tasks like causal effect estimation, reinforcement learning, and dynamical systems suggest that spectral representation learning—which is compatible with contrastive learning—remains simple and scalable. The authors ask: Can spectral representation learning be used to capture the leading features of the partial covariance operator to overcome the limitations of kernel methods?

**Core Idea**: Neural networks are used to learn the truncated SVD spectral features of the partial covariance operator $\Sigma_{X\ddot{Y}\cdot Z}$ (left/right singular functions + $w$ features for residualization). These features are used to construct a simple HSIC-like statistic. The difficulty lies in the fact that the residualization term $\Sigma_{XZ}\Sigma_{ZY}$ within the partial covariance cannot be directly estimated from data. The authors bypass this using a **bi-level contrastive formulation**.

## Method

### Overall Architecture
SpectralCIT splits samples $\{(X_i,Y_i,Z_i)\}_{i=1}^N$ into training and test sets. In the **training phase**, a bi-level contrastive algorithm (Algorithm 1) is used on the training set to learn three sets of neural network features: $u_\theta(X)$, $v_\theta(\ddot{Y})$ (where $\ddot{Y}=(Y,Z)$), and $w_\theta(Z)$. These are then whitened to ensure empirical orthonormality over the training set. In the **testing phase**, the learned features are used on the test set to compute the statistic $\widehat{T}_n$ (Eq. 10), which is essentially $n$ times the squared Frobenius norm of the "whitened empirical partial covariance matrix." In the **decision phase**, $\widehat{T}_n$ is compared against the $1-\alpha$ quantile of a chi-squared distribution with $d^2$ degrees of freedom (where $d$ is the network output dimension). If it exceeds the quantile, the null hypothesis is rejected. This pipeline decouples representation learning from testing via a train/test split, ensuring a clean null distribution for asymptotic theory and avoiding bias from data reuse.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["i.i.d. samples split into train/test"] --> B["Bi-level variational formula for partial covariance<br/>Transforms unestimable residual terms into low-rank inner problems"]
    B --> C["Bi-level contrastive representation learning<br/>Learns spectral features u(X), v(Y,Z), w(Z)"]
    C --> D["Whitening post-processing<br/>Enforces empirical orthonormality on training set"]
    D --> E["Spectral statistic T_n<br/>n × Frobenius norm² of whitened partial covariance matrix"]
    E -->|T_n ≥ (1-α) quantile of χ²(d²)| F["Reject H0: Conditional Dependence"]
    E -->|Otherwise| G["Fail to reject: Conditional Independence"]
```

### Key Designs

**1. Compressing "Conditional Dependence" into a low-dimensional matrix via truncated SVD of the partial covariance operator**

Conditional independence $X\perp\!\!\!\perp Y\mid Z$ is equivalent to the Hilbert–Schmidt norm of the partial covariance operator being zero: $\|\Sigma_{X\ddot{Y}\cdot Z}\|_{\mathrm{HS}}=0$, where $\Sigma_{AB\cdot C}=\Sigma_{AB}-\Sigma_{AC}\Sigma_{CB}$. The authors restate the test as $\mathcal{H}_0:\|\Sigma_{X\ddot{Y}\cdot Z}\|_{\mathrm{HS}}^2=0$ vs. $\mathcal{H}_{1,n,d}:\|\Sigma_{X\ddot{Y}\cdot Z}\|_{\mathrm{HS}}^2\geq\epsilon_n$. The brilliance of this operator framework is that it **implicitly encodes structural assumptions like smoothness, sparsity, low-rank, and latent variables**, without requiring explicit Lipschitz assumptions on $P_{X\mid Z=z}$ as in regression or local permutation tests. To make it computable and scalable, the authors learn only the **best rank-$d$ truncated SVD** $[\![\Sigma_{X\ddot{Y}\cdot Z}]\!]_d=\sum_{i=1}^d \sigma_i u_i\otimes v_i$, compressing an infinite-dimensional operator into a $d$-dimensional subspace. This step downgrades "estimating the entire conditional distribution" to "capturing a low-dimensional spectral subspace of the operator." Following the principle that "testing is easier than estimation" (Ingster 1993), this requirement is much weaker than that of regression-based tests.

**2. Bi-level variational formula: Bypassing unestimable residual terms**

When expressing the truncated SVD as a variational (minimization) problem, the objective produces a term $\mathrm{tr}[\mathsf{M}^\top\mathsf{U}^*\Sigma_{XZ}\Sigma_{Z\ddot{Y}}\mathsf{V}]$. This is a composition of two covariance operators $\Sigma_{XZ}\Sigma_{Z\ddot{Y}}$, which **cannot be directly estimated from data**. This represents the core technical hurdle in learning partial covariance versus unconditional covariance. The authors utilize the cyclic property of the trace: noting that the operator $\Sigma_{Z\ddot{Y}}\mathsf{V}\mathsf{M}^\top\mathsf{U}^*\Sigma_{XZ}:L^2(Z)\to L^2(Z)$ is at most rank-$d$, its symmetrization is at most rank-$2d$ and thus has an SVD form $\mathsf{W}\mathsf{N}\mathsf{W}^*$ (where $\mathsf{W}=[w_1|\cdots|w_{2d}]$ is an orthonormal system in $L^2(Z)$). By plugging this low-rank auxiliary decomposition into the same variational principle, the problematic composite term is replaced by an **inner optimization problem** regarding $(\mathsf{W},\mathsf{N})$. Ultimately, the truncated SVD can be written as $[\![\Sigma_{X\ddot{Y}\cdot Z}]\!]_d=\mathsf{U}[C_{UV}-C_{UW}C_{WV}]\mathsf{V}^*$. All conditional dependence structure is encapsulated in the matrix $C_{UV}-C_{UW}C_{WV}$, which can be computed provided suitable representations $(U,V,W)=(u(X),v(\ddot{Y}),w(Z))$ exist. Combining the outer layer (learning $u,v$) and inner layer (learning $w$) results in a **bi-level optimization problem**, which is the substantive difference between this method and existing spectral learning literature.

**3. Bi-level contrastive algorithm + Whitening post-processing: Implementing variational formulas as trainable neural losses**

The authors parameterize $u, v, w$ using neural networks $u_\theta:\mathcal{X}\to\mathbb{R}^d$, $v_\theta:\mathcal{Y}\times\mathcal{Z}\to\mathbb{R}^d$, and $w_\theta:\mathcal{Z}\to\mathbb{R}^{2d}$. They use U-statistics to express the outer and inner objectives as empirical losses $\widehat{\mathcal{L}}_{\mathrm{out}}$ and $\widehat{\mathcal{L}}_{\mathrm{in}}$ estimable from mini-batches (formatted as the typical attractive positive-pair and squared negative-pair terms in contrastive learning). Algorithm 1 optimizes alternately: updating the inner network $w_\theta$ for $n_{\texttt{steps\_inner}}$ steps, then updating outer networks $u_\theta, v_\theta$ with a new batch. A **whitening regularization** $\widehat{\Omega}(\theta)=\|\widehat{C}_{UU}-I_d\|_F^2+\|\widehat{C}_{VV}-I_d\|_F^2$ is added during training with intensity $\gamma>0$ to ensure well-conditioned empirical covariance matrices for safe inversion during whitening. Since batch-level orthogonality is a soft constraint, a **whitening post-processing** step is performed after training: using the covariance estimated on the full training set, $\widehat{u}_\theta(X)=\widehat{C}^{-1/2}_{\widetilde{U}\widetilde{U}}\widetilde{u}_\theta(X)$ (similarly for $v, w$). This preserves the learned subspace while improving the basis function geometry, making representations strictly empirically orthonormal—a crucial step for the null distribution to converge accurately to a chi-squared distribution.

**4. Spectral statistics with chi-squared null distribution and power guarantee: Linking representation quality to testing performance**

In the test phase, $\widehat{T}_n=n\,\|\widehat{C}_{\widehat{U}\widehat{V}}-\widehat{C}_{\widehat{U}\widehat{W}}\widehat{C}_{\widehat{W}\widehat{V}}\|_F^2$ (Eq. 10) is computed. This is the empirical norm of the matrix $C_{UV}-C_{UW}C_{WV}$ from $[\![\Sigma_{X\ddot{Y}\cdot Z}]\!]_d$, sharing commonality with HSIC. The theory links testing performance directly to two **representation errors**: the validity error $\mathcal{E}_m^{\mathrm{val}}$ (proximity of whitened features to orthonormality) and the power error $\mathcal{E}_m^{\mathrm{pow}}$ (proximity of the learned rank-$d$ representation to the true truncated SVD). **Validity** (Theorem 4.1): As long as $\mathcal{E}_m^{\mathrm{val}}\to0$, then under the null hypothesis, $\widehat{T}_n\stackrel{d}{\to}\chi^2(d^2)$. The intuition is that whitened features are approximately orthonormal, causing the matrix to resemble a unit covariance under the null; $\widehat{T}_n$ then approximates the sum of squares of $d^2$ independent standard Gaussians. This requirement is **weaker** than the "estimation of conditional expectation at a specific rate" required by regression tests, as it only needs to capture a low-dimensional spectral subspace. **Power** (Theorem 4.2): When the signal strength $\epsilon_n^2\gtrsim d(\mathcal{E}_m^{\mathrm{pow}})^2+\frac{d^2+d\log(\delta^{-1})}{n}$, the null hypothesis is rejected with probability at least $1-\delta$ under the alternative. Together, these theorems show that $\mathcal{E}_m^{\mathrm{val}}$ controls calibration under the null, while $\mathcal{E}_m^{\mathrm{pow}}$ controls signal retention under the alternative. The quality of the test depends directly on the quality of the learned representations.

### Loss & Training
The outer loss $\widehat{\mathcal{L}}_{\mathrm{out}}$ contains three terms: a squared term for negative pairs $\frac{1}{m(m-1)}\sum_{i\neq j}\langle\bar{u}_i,M\bar{v}_j\rangle^2$, an attractive term for positive pairs $-\frac{2}{m}\sum_i\langle\bar{u}_i,M\bar{v}_i\rangle$, and a coupling term carrying the $w$ residualization $\frac{2}{m(m-1)}\sum_{i\neq j}\langle\bar{u}_i,M\bar{v}_j\rangle\langle\bar{w}_i,\bar{w}_j\rangle$. The inner loss $\widehat{\mathcal{L}}_{\mathrm{in}}$ centers on minimizing quadratic and coupling terms for $w$ ($\bar{\cdot}$ denotes centering, $M=M_\theta$, $N=(N_\theta+N_\theta^\top)/2$). Bounded activation functions (e.g., Tanh) are used to satisfy sub-Gaussian assumptions in the theory. Hyperparameters (including $d$) are selected as per Appendix C.

## Key Experimental Results

### Main Results (Synthetic Data: Post-nonlinear model + High-dimensional kernel comparison)
In various synthetic settings with fixed sample sizes and varying conditional dimensions $d_Z$, the authors compared Type I error and power against KCIT/RCIT/GCIT/DGCIT/NNLSCIT, and kernel methods LPCIT/GCM ($\alpha=0.05$, 100 repetitions per setting).

| Setting / Method | Type I Error Control | Power | Notes |
|------|------|------|------|
| post-nonlinear · KCIT/RCIT/GCIT | Failed | High | High power but failed to control Type I error |
| post-nonlinear · DGCIT | Severely Failed | — | Complete loss of Type I error control |
| post-nonlinear · NNLSCIT | Robust | High | Consistent with Li et al. 2023 |
| post-nonlinear · **Ours** | Robust | High | Simultaneous error control and high power across dimensions |
| Non-smooth high-dim ($h_k$ oscillation) · NNLSCIT | Colllapsed | — | Oscillations destroy smoothness; Type I error control lost |
| Non-smooth high-dim · **Ours** | Robust | — | Operator framework does not rely on smoothness; remains stable |
| High-dim kernel comparison ($d_Z$ 50→300) · KCIT | Failed | High | Higher power for $d_Z\in\{250,300\}$ but Type I error control lost |
| High-dim kernel comparison · GCM | Valid | Very Low | Controlled error but almost no power |
| High-dim kernel comparison · **Ours** | Robust | High | Balances validity and power |

Key comparison: In the setting $X=f(Z/2+\epsilon_X), Y=g(Z/2+\epsilon_Y)$ where $f, g$ oscillate highly near the origin ($\cos(2\pi/w)$), the core smoothness assumption of NNLSCIT is violated, leading to a total collapse of Type I error control. SpectralCIT's operator framework does not explicitly assume Lipschitz continuity for the conditional distribution, thus remaining robust.

### Main Results (Real Data: Breast Cancer Molecular Profile vs. Histology Imaging)
On TCGA-BRCA data, $N=1341$ triplets were constructed: $X\in\mathbb{R}^3$ represents metagene scores (Her2/Luminal/Basal), $Y\in\{0,1\}$ is the survival outcome, and $Z\in\mathbb{R}^{384}$ is histology image features extracted by the Path Foundation model. The question is whether molecular profiles provide incremental predictive information given image features.

| Test | P-value | Conclusion |
|------|------|------|
| **Ours** | $<10^{-3}$ | Strongly rejects conditional independence |
| KCIT | $6.8\times10^{-2}$ | Failed to reject at $\alpha=0.05$ |
| NNLSCIT | $3.8\times10^{-1}$ | Failed to reject |

Linear partial correlations were near zero ($-0.006, -0.088, 0.07$). Logistic regression accuracy slightly decreased with $X$ (0.86 vs. 0.87). However, XGBoost accuracy increased from 0.91 to 0.95 when adding $X$, confirming the existence of **non-linear residual predictive information**. Only SpectralCIT detected this complex dependence missed by imaging features, highlighting its ability to capture non-linear conditional dependence in high-dimensional biological data.

### Key Findings
- Nearly all kernel methods (KCIT, RCIT, GCIT) either fail to control Type I error or lose power in high dimensions. DGCIT collapses entirely. GCM is valid but has almost no power. SpectralCIT is one of the few that balances both.
- Non-smooth settings are a vulnerability for NNLSCIT but not for SpectralCIT. The operator framework does not explicitly assume conditional distribution smoothness, making it more robust.
- In real breast cancer data, linear methods and two strong baselines concluded "conditional independence." Only SpectralCIT identified non-linear residual signals, which was corroborated by predictive gains in XGBoost, demonstrating practical utility.

## Highlights & Insights
- **Replacing kernels with learnable spectral features** in the partial covariance operator is the core insight. This preserves the "no single structural assumption" universality of the operator framework while gaining scalability through contrastive learning, solving the high-dimensional power loss and high cost of kernel methods.
- **Bypassing the unestimable residual term via bi-level variation** is ingenious. Using a low-rank auxiliary SVD ($\mathsf{W}\mathsf{N}\mathsf{W}^*$) to replace the composite operator $\Sigma_{XZ}\Sigma_{Z\ddot{Y}}$ with an inner optimization is the key technical differentiator for learning partial covariance, and this is transferable to other operator learning tasks requiring residualization.
- **Directly linking test validity/power to representation errors $\mathcal{E}_m^{\mathrm{val}}, \mathcal{E}_m^{\mathrm{pow}}$** builds a bridge between representation learning quality and statistical test performance. This requirement is weaker and more practical than "estimating regression at specific rates."
- **Whitening post-processing** is a crucial engineering detail for the null distribution's precise convergence to $\chi^2(d^2)$. It converts batch-level soft orthogonality to full-set hard orthogonality, serving as an invisible hero for theoretical feasibility.

## Limitations & Future Work
- The test relies on **sufficiently good representation learning** (validity requires $\mathcal{E}_m^{\mathrm{val}}\to0$). If the network is under-trained or lacks capacity, the null distribution's approximation of the chi-squared distribution and calibration will degrade. As the theory is asymptotic, finite-sample behavior remains constrained by representation quality.
- Hyperparameters such as the dimension $d$ must be selected. The authors acknowledge a lack of standardized benchmarks and tuning protocols in CI testing literature; selecting $d$ may be sensitive in practice.
- The train/test split sacrifices some sample efficiency. Bi-level alternating optimization is more complex to implement and computationally heavier than simple kernel methods.
- The real-world data experiment is limited in scale (1341 patch-level triplets from 135 WSIs). Conclusions regarding clinical generalizability across more cancer types and outcomes require further validation.

## Related Work & Insights
- **vs. Kernel Methods (KCIT/RCIT, Zhang 2011 / Strobl 2019)**: Based on the same HS norm of the partial covariance operator, but kernel methods use fixed kernels + random Fourier features. SpectralCIT replaces kernels with adaptive, scalable learned spectral features.
- **vs. Model-X (GCIT/DGCIT, Bellot 2019 / Shi 2021)**: These require learning or having $P(X\mid Z)$ for permutation/generation. SpectralCIT avoids modeling conditional distributions and only learns operator subspaces, bypassing dependence on $P(X\mid Z)$.
- **vs. Local Permutation (NNLSCIT, Li 2023)**: Relies on binning $Z$, which is computationally heavy and dependent on smoothness. SpectralCIT remains robust in non-smooth settings where NNLSCIT collapses.
- **vs. GCM (Shah & Peters 2020)**: GCM is valid but has extremely low power in high dimensions. SpectralCIT maintains both validity and power in the same settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces spectral representation learning for partial covariance CI testing. The bi-level formula successfully bypasses residual terms, bridging two previously disconnected fields.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three synthetic settings (including targeted non-smooth counterexamples) plus real breast cancer data provide a broad comparison, though the real experiment scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivations for operators and variations with clear theorem statements; however, the high theoretical density requires a strong mathematical background.
- Value: ⭐⭐⭐⭐ Provides a new tool for high-dimensional non-parametric CI testing with both scalability and theoretical guarantees, significant for causal discovery and feature selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[ICML 2026\] Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression](outcome-aware_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[NeurIPS 2025\] Demystifying Spectral Feature Learning for Instrumental Variable Regression](../../NeurIPS2025/causal_inference/demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[ICLR 2026\] Designing Time Series Experiments in A/B Testing with Transformer Reinforcement Learning](../../ICLR2026/causal_inference/designing_time_series_experiments_in_ab_testing_with_transformer_reinforcement_l.md)

</div>

<!-- RELATED:END -->
