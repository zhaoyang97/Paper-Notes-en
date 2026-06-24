---
title: >-
  [Paper Note] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime
description: >-
  [ICLR 2026][Optimization][Adam] This paper provides the first proof that the implicit bias of mini-batch Adam differs from the full-batch regime. By constructing specific datasets, it demonstrates that per-sample Adam converges to the $\ell_2$ max-margin classifier (whereas full-batch Adam converges to $\ell_\infty$), and characterizes its data-adaptive Mahalanobis norm margin maximization behavior on general datasets via the AdamProxy framework.
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Adam"
  - "Implicit Bias"
  - "Max-margin"
  - "Mini-batch"
  - "Mahalanobis Norm"
date: 2026-05-08
content_hash: f2ebd1c69dbbc7fc
---

# Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime

**Conference**: ICLR 2026  
**arXiv**: [2510.26303](https://arxiv.org/abs/2510.26303)  
**Code**: None  
**Area**: Others  
**Keywords**: Adam, Implicit Bias, Max-margin, Mini-batch, Mahalanobis Norm

## TL;DR
This paper provides the first proof that the implicit bias of mini-batch Adam differs from the full-batch regime. By constructing specific datasets, it demonstrates that per-sample Adam converges to the $\ell_2$ max-margin classifier (whereas full-batch Adam converges to $\ell_\infty$), and characterizes its data-adaptive Mahalanobis norm margin maximization behavior on general datasets via the AdamProxy framework.

## Background & Motivation

**Background**: The implicit bias of optimization algorithms determines which global optimum is selected in over-parameterized models. GD converges to the $\ell_2$ max-margin solution, while full-batch Adam converges to the $\ell_\infty$ max-margin solution. SGD does not alter the bias of GD (converging to $\ell_2$ regardless of batch size).

**Limitations of Prior Work**: Existing analyses of Adam's implicit bias are restricted to the full-batch setting. While practical training utilizes mini-batches, it remains unclear whether mini-batch training alters Adam's $\ell_\infty$ bias. Intuition from SGD might suggest that the bias remains unchanged.

**Key Challenge**: Experiments reveal that mini-batch Adam (batch=1) converges to a direction on Gaussian data that differs from full-batch Adam and is closer to the $\ell_2$ max-margin solution. This stands in stark contrast to the behavior of SGD.

**Key Insight**: By analyzing the asymptotic form of epoch-wise updates for per-sample Adam (Inc-Adam), this work reveals that the preconditioner tracks the weighted sum of individual sample gradient squares (rather than the square of the full-batch gradient), leading to a fundamental shift in adaptive properties.

## Method

### Overall Architecture

The analysis centers on per-sample Adam (Inc-Adam, batch=1 with sequential updates) and proceeds in two stages: first, providing exact convergence conclusions on a class of highly structured Scaled Rademacher data to prove that Inc-Adam converges to the $\ell_2$ max-margin (contrasting with full-batch $\ell_\infty$ bias); second, introducing the AdamProxy algorithm for arbitrary separable datasets to characterize general implicit bias as a data-adaptive Mahalanobis norm margin maximization problem.

### Key Designs

**1. Epoch-wise Approximation: Folding Per-sample Updates into Equivalent Gradients**

Directly analyzing the per-sample trajectory for batch=1 is difficult because weights are updated $n$ times within a single epoch and the preconditioner changes continuously. This work approximates the net displacement of an entire epoch as a single step (Proposition 2.5):

$$w_{r+1}^0 - w_r^0 \approx -\eta \sum_i \frac{\sum_j \beta_1^{(i,j)} \nabla \mathcal{L}_j(w)}{\sqrt{\sum_j \beta_2^{(i,j)} \nabla \mathcal{L}_j(w)^2}}$$

Comparing this with the asymptotic form of full-batch Adam (which reduces to SignGD: $w_{t+1} - w_t \approx -\eta \cdot \text{sign}(\nabla \mathcal{L}(w))$), a critical difference emerges: the Inc-Adam preconditioner denominator contains the **weighted sum of per-sample gradient squares** $\sum_i (\nabla \mathcal{L}_i)^2$, whereas full-batch Adam uses the square of the summed gradients $(\sum_i \nabla \mathcal{L}_i)^2$. This discrepancy drives the divergence in implicit bias and explains why the "batch-size independence" analogy from SGD fails.

**2. Exact Results on Scaled Rademacher Data: Constructing Scenarios where Adaptivity Vanishes**

To formalize this intuition, the authors construct Scaled Rademacher (SR) data where the absolute values of all coordinates for each sample are equal ($x_i = (a_i, \pm a_i, \pm a_i, \pm a_i)$). Under this symmetric structure, the coordinate-wise adaptivity of the Inc-Adam preconditioner is flattened, and the equivalent update reduces to a weighted normalized GD. Theorem 3.3 proves it converges to the $\ell_2$ max-margin classifier. Since full-batch Adam still converges to the $\ell_\infty$ max-margin on the same data, this proves that mini-batching changing Adam's bias is a structural fact rather than a numerical artifact.

**3. AdamProxy: Reducing General Bias to Mahalanobis Margin Maximization**

To cover arbitrary separable datasets, the authors define a simplified proxy update in the limit $\beta_2 \to 1$ (Proposition 4.1), termed AdamProxy:

$$\delta_t = \frac{\nabla \mathcal{L}(w)}{\sqrt{\sum_i \nabla \mathcal{L}_i(w)^2}}$$

This limit preserves the "sum of per-sample gradient squares" term while removing difficult transient momentum couplings. It is proven that the convergence direction solves a margin maximization problem under a Mahalanobis norm $\max \min_i \frac{x_i^\top w}{\|w\|_M}$, where the metric matrix $M$ is determined adaptively by the data via a dual fixed-point equation $T(\mathbf{c})=\mathbf{c}$ (Theorem 4.8).

**4. Invariance of Signum: Proving Bias Shift Stems from the Preconditioner**

To exclude momentum or the sign operation as causes, the authors use Signum (SignSGD with momentum) as a control. Theorem 5.1 shows that for momentum parameters sufficiently close to 1, Signum converges to the $\ell_\infty$ max-margin regardless of batch size. This is because the sign operation only considers the gradient sign and discards magnitude information, making it immune to the "sum of squares vs. square of sum" difference. This confirms that Adam's sensitivity to sampling arises specifically from its square-root preconditioner.

## Key Experimental Results

### SR Data Verification

| Method | batch=full | batch=1 |
|------|-----------|---------|
| Adam | $\ell_\infty$ margin | **$\ell_2$ margin** |
| SGD | $\ell_2$ margin | $\ell_2$ margin |
| Signum | $\ell_\infty$ margin | $\ell_\infty$ margin |

### Gaussian Data Verification

| Method | Cosine with $\ell_2$ | Cosine with $\ell_\infty$ |
|------|----------------|-------------------|
| Full-batch Adam | Low | **1.0** |
| Inc-Adam | High | Low |
| Adam (batch=1, replacement) | High | Low |
| Adam (batch=1, reshuffling) | High | Low |

### Key Findings
- Inc-Adam behaves identically to batch=1 Adam with replacement/reshuffling, justifying it as a theoretical proxy.
- Conclusions for SR data hold for any $\beta_1 \leq \beta_2$, showing the bias shift is not a hyperparameter-specific issue.
- The Mahalanobis norm of AdamProxy reduces to $\ell_2$ on some data and $\ell_\infty$ on others.

## Highlights & Insights
- **Counter-intuitive Finding**: Unlike SGD, whose implicit bias is batch-size independent, Adam's bias is sensitive to the sampling regime. This highlights how adaptive preconditioners interact with stochasticity.
- **Core Preconditioner Divergence**: $\sum_i (\nabla \mathcal{L}_i)^2 \neq (\sum_i \nabla \mathcal{L}_i)^2$—the sum of squares of individual gradients is not equal to the square of the full-batch gradient. This simple mathematical fact dictates different implicit biases.
- **Robustness of Signum**: The sign operation makes Signum immune to sampling methods, which may explain the relative stability of Signum/SignSGD in certain empirical settings.

## Limitations & Future Work
- AdamProxy analysis requires the assumption that directional convergence exists (Assumption 4.4).
- Only the extreme case of $batch=1$ is analyzed; the behavior for intermediate batch sizes remains open.
- The study is limited to linear classification on separable data; deep networks present much higher complexity.
- The $\beta_2 \to 1$ limit may deviate slightly from the practical $\beta_2=0.999$ setting.

## Related Work & Insights
- **vs. Zhang et al. (2024)**: They proved full-batch Adam $\to \ell_\infty$; this work proves mini-batch can be different.
- **vs. Soudry et al. (2018)**: GD $\to \ell_2$ is unaffected by batch size, but Adam's adaptivity makes it fundamentally different.
- **Insight**: In practice, Adam's behavior is likely a joint effect of data structure and batch size—one cannot simply extrapolate from full-batch theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First discovery and theorization of Adam's batch-dependent implicit bias.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across structured and random data with various sampling schemes.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear intuitive explanations.
- Value: ⭐⭐⭐⭐⭐ Fundamental significance for understanding Adam's behavior in practical training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](../../ICML2026/optimization/the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[ICLR 2026\] Hyperbolic Aware Minimization: Implicit Bias for Sparsity](hyperbolic_aware_minimization_implicit_bias_for_sparsity.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)

</div>

<!-- RELATED:END -->
