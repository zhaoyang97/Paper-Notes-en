---
title: >-
  [Paper Note] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime
description: >-
  [ICLR 2026][Adam] This paper provides the first proof that mini-batch Adam exhibits a different implicit bias from its full-batch counterpart: a constructed dataset causes per-sample Adam to converge to an $\ell_2$ maxim…
tags:
  - "ICLR 2026"
  - "Adam"
  - "implicit bias"
  - "maximum margin"
  - "mini-batch"
  - "Mahalanobis norm"
date: 2026-05-08
content_hash: aa50f638e6fe9978
---

# Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime

**Conference**: ICLR 2026
**arXiv**: [2510.26303](https://arxiv.org/abs/2510.26303)  
**Code**: None  
**Area**: Other
**Keywords**: Adam, implicit bias, maximum margin, mini-batch, Mahalanobis norm

## TL;DR
This paper provides the first proof that mini-batch Adam exhibits a different implicit bias from its full-batch counterpart: a constructed dataset causes per-sample Adam to converge to an $\ell_2$ maximum-margin classifier (whereas full-batch Adam converges to $\ell_\infty$), and a proxy algorithm, AdamProxy, is introduced to characterize data-adaptive Mahalanobis-norm margin maximization on general datasets.

## Background & Motivation

**Background**: The implicit bias of optimization algorithms determines which global optimum is selected in overparameterized models. GD converges to the $\ell_2$ maximum-margin solution, and full-batch Adam converges to the $\ell_\infty$ maximum-margin solution. SGD does not alter GD's bias (any batch size converges to $\ell_2$).

**Limitations of Prior Work**: Existing analyses of Adam's implicit bias are confined to the full-batch setting. Practical training uses mini-batches, yet it is unclear whether mini-batching alters Adam's $\ell_\infty$ bias. Intuition from SGD suggests the bias should be invariant to batch size—but is this true for Adam?

**Key Challenge**: Experiments reveal that mini-batch Adam (batch size = 1) on Gaussian data converges to a direction closer to the $\ell_2$ maximum margin, which is markedly different from full-batch Adam and stands in sharp contrast to the behavior of SGD.

**Key Insight**: By analyzing the asymptotic form of the epoch-wise updates of per-sample Adam (Inc-Adam), the paper shows that the preconditioner tracks a weighted sum of per-sample squared gradients (rather than the squared full-batch gradient), fundamentally altering its adaptive properties.

## Method

### Overall Architecture
The analysis first proves that Inc-Adam converges to the $\ell_2$ margin on Scaled Rademacher (SR) data, then introduces the AdamProxy surrogate algorithm (the $\beta_2 \to 1$ limit) for general datasets, characterizing convergence as data-adaptive Mahalanobis-norm margin maximization.

### Key Designs

1. **Epoch-wise Approximation (Proposition 2.5)**:

    - The epoch-wise update of Inc-Adam is approximated as $w_{r+1}^0 - w_r^0 \approx -\eta \sum_i \frac{\sum_j \beta_1^{(i,j)} \nabla \mathcal{L}_j(w)}{\sqrt{\sum_j \beta_2^{(i,j)} \nabla \mathcal{L}_j(w)^2}}$
    - vs. Full-batch Adam approximated as SignGD: $w_{t+1} - w_t \approx -\eta \cdot \text{sign}(\nabla \mathcal{L}(w))$
    - Key distinction: Inc-Adam's preconditioner is a **weighted sum of per-sample squared gradients**, which is not equal to the square of the full-batch gradient.

2. **Exact Result on Scaled Rademacher Data (Theorem 3.3)**:

    - SR data: each sample has equal absolute values across coordinates (e.g., $x_i = (a_i, \pm a_i, \pm a_i, \pm a_i)$).
    - Under this structure, the coordinate-wise adaptivity of Inc-Adam is eliminated, reducing it to weighted normalized GD, which converges to the $\ell_2$ maximum margin.
    - This forms an **extreme contrast** with full-batch Adam's $\ell_\infty$ bias.

3. **AdamProxy (General Datasets)**:

    - Taking the $\beta_2 \to 1$ limit yields a simplified update: $\delta_t = \frac{\nabla \mathcal{L}(w)}{\sqrt{\sum_i \nabla \mathcal{L}_i(w)^2}}$
    - The convergence direction maximizes the Mahalanobis-norm margin: $\max \min_i \frac{x_i^\top w}{\|w\|_M}$
    - The covariance matrix $M$ is determined by a data-dependent dual fixed-point equation.

4. **Invariance of Signum (Comparison)**:

    - Signum (SignSGD with momentum) converges to $\ell_\infty$ regardless of batch size.
    - Reason: the sign operation eliminates the difference between per-sample and full-batch preconditioners.

## Key Experimental Results

### Validation on SR Data

| Method | batch = full | batch = 1 |
|--------|-------------|-----------|
| Adam | $\ell_\infty$ margin | **$\ell_2$ margin** |
| SGD | $\ell_2$ margin | $\ell_2$ margin |
| Signum | $\ell_\infty$ margin | $\ell_\infty$ margin |

### Validation on Gaussian Data

| Method | Cosine w/ $\ell_2$ | Cosine w/ $\ell_\infty$ |
|--------|--------------------|------------------------|
| Full-batch Adam | Low | **1.0** |
| Inc-Adam | High | Low |
| Adam (batch=1, with replacement) | High | Low |
| Adam (batch=1, reshuffling) | High | Low |

### Key Findings
- Inc-Adam is consistent with batch=1 Adam under both with-replacement and reshuffling sampling, confirming that Inc-Adam is a valid theoretical surrogate.
- Results on SR data hold for any $\beta_1 \leq \beta_2$, indicating that the bias shift is not an artifact of specific hyperparameters.
- The Mahalanobis norm of AdamProxy degenerates to $\ell_2$ on some datasets and to $\ell_\infty$ on others.

## Highlights & Insights
- **Counterintuitive Finding**: Since SGD's implicit bias is invariant to batch size, it is natural to conjecture the same for Adam—yet the opposite is true. This reveals that the preconditioner of adaptive methods is fundamentally sensitive to the sampling scheme.
- **The Core Mathematical Gap**: $\sum_i (\nabla \mathcal{L}_i)^2 \neq (\sum_i \nabla \mathcal{L}_i)^2$—the sum of per-sample squared gradients is not equal to the square of the full-batch gradient. This elementary algebraic fact gives rise to an entirely different implicit bias.
- **Robustness of Signum**: The sign operation renders Signum immune to the choice of sampling scheme, which may partly explain the stability of Signum/SignSGD observed in certain practical settings.

## Limitations & Future Work
- The AdamProxy analysis requires assuming the existence of directional convergence (Assumption 4.4).
- Only the extreme case of batch size = 1 is analyzed; the behavior at intermediate batch sizes remains open.
- The analysis is restricted to linear classification on separable data; the deep network setting is considerably more complex.
- The $\beta_2 \to 1$ limit may deviate from practical settings where $\beta_2 = 0.999$.

## Related Work & Insights
- **vs. Zhang et al. (2024)**: They prove that full-batch Adam converges to $\ell_\infty$; this paper shows that mini-batch Adam can behave differently.
- **vs. Soudry et al. (2018)**: GD's $\ell_2$ bias is unaffected by batch size, but Adam's bias is—highlighting a fundamental distinction of adaptive methods.
- **Implication**: The behavior of Adam in practice may be jointly governed by data structure and batch size; one cannot simply extrapolate from full-batch theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First discovery and theoretical characterization of Adam's batch-dependent implicit bias.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation on both structured and random data, with comparisons across multiple sampling schemes.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear intuitive explanations.
- Value: ⭐⭐⭐⭐⭐ Fundamental significance for understanding Adam's behavior in practical training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](../../ICML2026/others/the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICLR 2026\] Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields](refine_now_query_fast_a_decoupled_refinement_paradigm_for_implicit_neural_fields.md)
- [\[ICLR 2026\] On the Impact of the Utility in Semivalue-based Data Valuation](on_the_impact_of_the_utility_in_semivalue-based_data_valuation.md)
- [\[ICLR 2026\] Condition Matters in Full-head 3D GANs](condition_matters_in_full-head_3d_gans.md)
- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)

</div>

<!-- RELATED:END -->
