---
title: >-
  [Paper Note] Direct Doubly Robust Estimation of Conditional Quantile Contrasts
description: >-
  [ICLR 2026][Causal Inference][heterogeneous treatment effect] This paper proposes the first **direct estimation method** for the conditional quantile contrast (CQC) by explicitly parameterizing the CQC and combining it w…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "heterogeneous treatment effect"
  - "conditional quantile comparator"
  - "doubly robust estimation"
  - "quantile treatment effect"
date: 2026-05-08
content_hash: 8e52dfc6d7e3018b
---

# Direct Doubly Robust Estimation of Conditional Quantile Contrasts

**Conference**: ICLR 2026
**arXiv**: [2601.19666](https://arxiv.org/abs/2601.19666)  
**Code**: Reproduction code provided in supplementary material  
**Area**: Causal Inference
**Keywords**: heterogeneous treatment effect, conditional quantile comparator, doubly robust estimation, causal inference, quantile treatment effect

## TL;DR

This paper proposes the first **direct estimation method** for the conditional quantile contrast (CQC) by explicitly parameterizing the CQC and combining it with doubly robust gradient descent. The approach maintains theoretical double robustness while empirically outperforming existing indirect inversion methods across estimation accuracy, interpretability, and computational efficiency.

## Background & Motivation

**Background**: Heterogeneous treatment effect (HTE) analysis aims to characterize how treatment effects vary across individuals. CATE (conditional average treatment effect) and CQTE (conditional quantile treatment effect) are two classical estimands — CATE is interpretable but provides only mean-level information, while CQTE offers quantile-level granularity at the cost of interpretability.

**Limitations of Prior Work**: The recently proposed CQC (conditional quantile contrast) attempts to combine the advantages of both by providing a transport map from the untreated to the treated response. However, existing CQC estimation methods (Givens et al., 2024) require first estimating an intermediate quantity — the CCDF contrast function $h$ — and then obtaining the CQC via **inversion**, which introduces three key problems: the CQC cannot be directly modeled or constrained; estimation error depends on the complexity of the intermediate function rather than the CQC itself; and evaluation incurs substantial computational overhead.

**Key Challenge**: The CQC itself may be highly parsimonious (e.g., the treatment effect is a linear scaling of the response, $g^*(y_0|\mathbf{x}) = 2y_0$), yet the accuracy of indirect inversion methods is limited by the complexity of the more intricate intermediate function $h$.

**Goal**: To provide the first direct estimation method for the CQC, enabling explicit parameterization of the CQC so that estimation error depends directly on the complexity of the CQC itself.

**Key Insight**: The CQC estimation problem is cast as M-estimation — a loss function is constructed whose minimizer is the CQC, and a doubly robust expression for its gradient is derived, enabling direct estimation via gradient descent.

**Core Idea**: By constructing a loss function and deriving its doubly robust gradient, the method bypasses intermediate function inversion, achieving for the first time direct parametric estimation of the CQC.

## Method

### Overall Architecture

Given observed data $D = \{(Y^{(i)}, X^{(i)}, A^{(i)})\}_{i=1}^{2n}$, the algorithm proceeds in two steps:

1. **Sample splitting**: Data are partitioned into $D_\mathcal{I}$ (used to estimate nuisance parameters $\hat{\pi}, \hat{F}_0, \hat{F}_1$) and $D_\mathcal{J}$ (used to fit CQC parameters $\theta$ via gradient descent).
2. **Gradient descent**: Stochastic gradient descent is performed on $D_\mathcal{J}$ using the doubly robust gradient $\hat{\zeta}_{dr}$ to optimize the CQC parameters $\theta$.

### Key Designs

1. **Loss Function Construction (Definition 2)**:

    - Function: Constructs a loss function whose minimizer is the true CQC $g^*$.
    - Mechanism: Since the CCDF contrast function $h(y_1, y_0, \mathbf{x}) = F_1(y_1|\mathbf{x}) - F_0(y_0|\mathbf{x})$ is monotonically increasing in $y_1$, any function whose derivative equals $h$ attains its minimum where $h = 0$, i.e., at $y_1 = g^*(y_0|\mathbf{x})$. This motivates the definition:
    $$\bar{\ell}(y_1, y_0, \mathbf{x}) = \int_{g^*(y_0|\mathbf{x})}^{y_1} h(t, y_0, \mathbf{x})\, dt$$
    - Design Motivation: The loss function is directly related to the CQC estimation error (Proposition 1 provides upper and lower bounds under three different conditions), ensuring that estimation accuracy depends on the complexity of the CQC rather than the intermediate function.

2. **Doubly Robust Gradient $\zeta_{dr}$ (Equation 5)**:

    - Function: Derives a doubly robust Monte Carlo estimator for the gradient of the loss function.
    - Mechanism:
    $$\zeta_{dr}(\theta, y_0, \mathbf{z}) = \nabla_\theta g_\theta(y_0|\mathbf{x}) \left( \frac{a}{\pi(\mathbf{x})}[\mathbb{1}\{y \le g_\theta\} - F_1(g_\theta)] - \frac{1-a}{1-\pi(\mathbf{x})}[\mathbb{1}\{y \le y_0\} - F_0(y_0)] + F_1 - F_0 \right)$$
    - Design Motivation: IPW (inverse probability weighting) requires only the propensity score but is sensitive to its estimation error. The doubly robust formulation additionally leverages CCDF estimates, so that errors in the two types of nuisance parameters multiply rather than add, substantially improving robustness.

3. **Explicit Parameterization of the CQC**:

    - Function: Allows the user to directly parameterize the CQC using linear models, kernel methods, neural networks, etc.
    - Mechanism: The linear model takes the form $g_\theta(y_0|\mathbf{x}) = (\theta_{sc}^\top \mathbf{x} + \theta_{sc,0})y_0 + (\theta_{sh}^\top \mathbf{x} + \theta_{sh,0})$, separately modeling scaling and shift components.
    - Design Motivation: Direct parameterization yields interpretable models and enables the incorporation of prior constraints via regularization, bandwidth selection, and related techniques.

### Loss & Training

- **Sample splitting + projected gradient descent**: Parameters are initialized at $\theta^{(1)} = 0$ and projected onto the ball $\|\theta\| \le B$ at each step.
- Learning rate: $\mu_t = \frac{Bc}{2\rho\sqrt{n}}$ in the general case, or $\mu_t = \frac{1}{\xi_2 \eta_2 n}$ when the density is bounded below.
- The final estimate is the average over all iterates: $\hat{\theta} = \frac{1}{n}\sum_{t=1}^n \theta^{(t)}$.

## Key Experimental Results

### Main Results

Data generating process: $X \sim N(0, I_{10})$, $Y|X,A \sim N(\sin(\pi \mathbf{v}^\top \mathbf{x}) + a\gamma \mathbf{v}^\top \mathbf{x}, 1)$, $\pi(\mathbf{x}) = \sigma(\mathbf{v}^\top \mathbf{x})$

True CQC: $g^*(y_0|\mathbf{x}) = y_0 + \gamma \mathbf{v}^\top \mathbf{x}$ (linear), while the CCDF contrast function contains high-frequency sinusoidal terms.

| Setting | Est. DR-Lin | Est. DR-NN | Est. Inv. DR | Est. IPW |
|---------|------------|-----------|-------------|---------|
| CQC slope γ=1 (MAE) | **Lowest** | Close to DR-Lin | Higher | Highest |
| CQC slope γ=4 (MAE) | **Lowest** | Close to DR-Lin | Substantially worse | Severely worse |
| Sample size n=200 (MAE) | **Lowest** | Slightly higher | Higher | High |
| Sample size n=2000 (MAE) | **Lowest** | Slightly higher | Higher | High |

### Ablation Study

Sensitivity to nuisance parameter estimation error (biased noise at varying levels added to logits):

| Noise level | Est. DR-Lin | Est. DR-NN | Est. Inv. DR | Est. IPW |
|-------------|------------|-----------|-------------|---------|
| 0 (no additional noise) | **Lowest** | Close to lowest | Moderate | Moderately high |
| 0.5 | **Lowest** | Slightly higher | Comparable | Higher |
| 1.0 | **Lowest** | Slightly higher | Slightly above DR-Lin | High |
| 2.0 | Comparable | Slightly higher | **Comparable** | Substantially higher |

### Key Findings

1. Direct parameterization methods (DR-Lin, DR-NN) uniformly outperform the indirect inversion method across all sample sizes and CQC slope settings.
2. The advantage of direct methods becomes more pronounced as the CQC slope increases — because the CCDF contrast function grows more complex while the CQC remains parsimonious.
3. Both direct methods exhibit double robustness to nuisance parameter errors, though the indirect method is marginally less sensitive under high noise levels.
4. The neural network model (DR-NN) performs well even without knowledge of the true functional form, falling only slightly behind the correctly specified linear model.
5. **Real data (Job Corps study)**: CQC estimates reveal that as age increases, treatment effects transition from multiplicative scaling to uniform translation.

## Highlights & Insights

1. **First direct CQC estimator**: By bypassing intermediate function inversion, estimation accuracy is directly tied to the complexity of the CQC itself.
2. **Theoretical convergence guarantees**: Theorem 3 provides finite-sample bounds — $O(1/\sqrt{n})$ in the general case and $O(\log n / n)$ when the density is bounded below.
3. **Interpretability from explicit parameterization**: Model parameters can be directly inspected to understand the structure of treatment effects (e.g., scaling and shift components in the linear model), rather than being evaluated only at sampled points.
4. **Parsimony of CQC under heterogeneous effects**: When the treatment effect is multiplicative (e.g., income doubling), the CQC is simply $g^*(y) = 2y$, whereas both CATE and CQTE involve complex high-frequency terms.

## Limitations & Future Work

1. The direct estimator exhibits slightly higher empirical sensitivity to nuisance parameter errors than the indirect method — despite both being theoretically doubly robust — warranting further investigation.
2. Double robustness holds with respect to the loss function rather than directly with respect to CQC estimation error; it translates to a bound on CQC error only under specific conditions (Proposition 1(b)).
3. Convergence results apply only to linearly parameterized models ($g_\theta = \theta^\top f$) and do not cover nonlinear parameterizations such as deep neural networks.
4. Future work: investigating whether a doubly robust CQC estimator in conditional expectation form (analogous to the DR-learner for CATE) can be derived.

## Related Work & Insights

- **CATE estimation**: The DR-learner of Kennedy (2023b) provides a doubly robust direct estimator for CATE — this paper extends analogous ideas to the quantile-level CQC.
- **CQTE estimation**: The doubly robust CQTE estimator of Kallus & Oprescu (2023) — CQC and CQTE are related by $\tau_q\{F_0(y_0|\mathbf{x})|\mathbf{x}\} = g(y_0|\mathbf{x}) - y_0$.
- **Random Fourier features**: The linear parameterization assumption can be extended to nonparametric kernel methods via random Fourier features.

## Rating

- Novelty: ⭐⭐⭐⭐ — First direct CQC estimator with a clever M-estimation formulation, though situated within the established HTE doubly robust framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dimensional simulation comparisons (slope, sample size, noise) plus real data and ablation studies; experiments on high-dimensional $X$ and nonlinear CQC are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, mathematical derivations are rigorous, and the exposition proceeds step by step from intuition to formalization; Figure 1 comparing CQC, CATE, and CQTE is highly effective.
- Value: ⭐⭐⭐⭐ — Represents a substantive advance in HTE estimation for causal inference, though CQC itself remains a relatively new estimand with room for broader application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)

</div>

<!-- RELATED:END -->
