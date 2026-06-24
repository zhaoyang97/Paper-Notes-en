---
title: >-
  [Paper Note] Grokking at the Edge of Linear Separability
description: >-
  [ICML2025][Optimization][grokking] Reveals the root cause of grokking (delayed generalization) in the simplest logistic regression binary classification task: when the ratio of data dimensionality to the number of samples $\lambda = d/N$ approaches the critical threshold $\lambda_c = 1/2$, the training dynamics remain near the overfitted solution for an arbitrarily long time before finally converging to the generalizing solution, resembling the "critical slowing down" phenome…
tags:
  - "ICML2025"
  - "Optimization"
  - "grokking"
  - "delayed generalization"
  - "logistic regression"
  - "linear separability"
  - "critical phenomena"
  - "interpolation threshold"
  - "implicit bias of gradient descent"
date: 2026-05-08
content_hash: 824e07d427f02f64
---

# Grokking at the Edge of Linear Separability

**Conference**: ICML2025  
**arXiv**: [2410.04489](https://arxiv.org/abs/2410.04489)  
**Code**: No public code  
**Area**: Optimization  
**Keywords**: grokking, delayed generalization, logistic regression, linear separability, critical phenomena, interpolation threshold, implicit bias of gradient descent

## TL;DR

Reveals the root cause of grokking (delayed generalization) in the simplest logistic regression binary classification task: when the ratio of data dimensionality to the number of samples $\lambda = d/N$ approaches the critical threshold $\lambda_c = 1/2$, the training dynamics remain near the overfitted solution for an arbitrarily long time before finally converging to the generalizing solution, resembling the "critical slowing down" phenomenon in physics.

## Background & Motivation

**Grokking Phenomenon**: First observed by Power et al. (2022) when training Transformers on modular arithmetic tasks—the model initially achieves perfect training accuracy but zero generalization, and then suddenly generalizes after continued training. This violates the traditional understanding that "overfitting is irreversible."

**Limitations of Prior Work**: Previous works attribute grokking to mechanisms such as regularization, representation learning, and circuit formation, but lack a rigorous mathematical analysis framework. Most solvable models are too complex to precisely define "memorization" and "generalization."

**Our Motivation**: To reproduce grokking in the simplest logistic regression model and provide a rigorous mathematical proof, revealing its deep connection with critical phenomena (phase transitions). Core Question: **Why does grokking occur precisely when the system is close to the critical threshold?**

## Method

### Problem Setup

Consider $N$ training samples $\{\mathbf{x}_i\}_{i=1}^N \subset \mathbb{R}^d$ following a Gaussian distribution $\mathbf{x}_i \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_d)$, with all labels sharing the same target $y_i = -1$. The model is a linear classifier with a bias:

$$f(\mathbf{x}_i) = \mathbf{S}^T \mathbf{x}_i + b$$

Optimizing the cross-entropy loss:

$$\mathcal{L}(\mathbf{S}, b) = \frac{1}{N} \sum_{i=1}^{N} \log(1 + e^{\mathbf{S}^T \mathbf{x}_i + b})$$

Key parameter: $\lambda = d/N$ (the ratio of dimensionality to sample size).

### Core Theory: Critical Threshold $\lambda_c = 1/2$

**Definition (Separability from the origin)**: A dataset is linearly separable from the origin if and only if there exists $\mathbf{S}$ such that $\mathbf{S}^T \mathbf{x}_i > 0$ for all $i$.

**Proposition 3.3** (based on Wendel's Theorem): As $N, d \to \infty$:

- $\lambda > 1/2$: the data is almost surely separable from the origin
- $\lambda < 1/2$: the data is almost surely inseparable

**Proposition 3.2** (Separability determines generalization):

- **Inseparable** ($\lambda < 1/2$): $b(t) \to -\infty$, $\|\mathbf{S}\|$ converges to a finite value $\mathbf{S}_\infty$, and the model generalizes perfectly.
- **Separable** ($\lambda > 1/2$): both $b$ and $\|\mathbf{S}\|$ diverge, and the generalization accuracy is bounded by the separation margin $M$.

$$\lim_{t \to \infty} \mathcal{A}_{\text{gen}} = \frac{1}{2}\left[1 + \text{erf}\left(\frac{1}{\sigma M \sqrt{2}}\right)\right] < 1$$

### Dynamical Mechanism of Grokking

By introducing the "conformal time" $\tau(t) = \int_0^t e^{b(t')} dt'$, the dynamics with bias can be mapped to dynamics without bias:

$$\frac{\partial \mathbf{S}}{\partial \tau} = -\frac{\eta}{N} \sum_i e^{\mathbf{S}^T \mathbf{x}_i} \mathbf{x}_i$$

**Proposition 3.4**: As $\lambda \to (1/2)^-$, $\|\mathbf{S}_\infty\| \to \infty$.

**Proposition 3.5**: When $\sigma$ is sufficiently large, $\mathbf{S}(t)$ can approach $\mathbf{S}_\infty$ arbitrarily fast.

**Grokking Mechanism**: Under the conditions where $\lambda$ is slightly smaller than $1/2$ and $\sigma$ is large:

1. $\|\mathbf{S}_\infty\|$ is extremely large $\to$ $\mathbf{S}(t)$ rapidly increases to a very large value.
2. The rate of increase of $b(t)$ is bounded $\to$ $|b|/\|\mathbf{S}\|$ remains small for a long time.
3. Generalization accuracy $\mathcal{A}_{\text{gen}} \approx \frac{1}{2}[1 - \text{erf}(b/(\sigma\|\mathbf{S}\|\sqrt{2}))]$ plateaus at a low level for a long duration.
4. Eventually, $\mathbf{S}$ saturates while $b$ continues to grow, enabling generalization $\to$ delayed generalization = grokking.

### Simplified One-Dimensional Model

Construct a minimal model with $N=2$ (where $x_1 = -1, x_2 = 1 - 2\lambda$) to obtain full analytical solutions across three dynamic regimes:

| Regime | $b(t \gg 1)$ | $\|\mathbf{S}\|(t \gg 1)$ |
|------|-------------|--------------------------|
| $\lambda < 1/2$ (Inseparable) | $-\log(t)$ | $\frac{1}{2(1-\lambda)}\log\frac{1}{1-2\lambda}$ (Finite) |
| $\lambda = 1/2$ (Critical) | $-\log(t)$ | $\log(\log(t))$ (Logarithmic divergence) |
| $\lambda > 1/2$ (Separable) | $-\frac{1}{1+M^2}\log(t)$ | $\frac{M}{1+M^2}\log(t)$ (Divergence) |

## Key Experimental Results

| Experimental Setup | Parameters | Key Findings |
|----------|------|----------|
| Main Results | $N=4\times10^4$, $\sigma=5$, $\eta=0.01$ | Significant grokking occurs at $\lambda=0.48$; generalization never occurs at $\lambda=0.52$. |
| Generalization Accuracy | Varying $\lambda$ | For $\lambda > 0.5$, the final asymptotic accuracy perfectly matches the theoretical prediction of margin $M$. |
| Grokking Time | $\lambda \to 0.5^-$ | Grokking time diverges, matching the prediction of critical slowing down. |
| $\|\mathbf{S}_\infty\|$ Divergence | $\lambda \to 0.5^+$ | $\|\mathbf{S}_\infty\|$ diverges at $\lambda = 0.5$, confirming the critical point. |
| Simplified Model | Varying $\sigma$ and $1-2\lambda$ | Grokking requires both conditions: $\lambda \to 1/2$ **and** $\sigma \to \infty$. |

## Highlights & Insights

1. **Profound discoveries in a minimal model**: Reproduces grokking in the simplest logistic regression, proving that it does not require complex networks, regularization, or specific data structures.
2. **Critical phenomena perspective**: Provides a precise analogy between grokking and critical slowing down in physics, offering a brand-new explanatory framework.
3. **Rigorous mathematical proof**: Offers rigorous proofs for the boundary conditions of generalization vs. overfitting, and the divergence of grokking time.
4. **Conformal time technique**: Elegantly maps the biased problem to an unbiased one through temporal reparameterization.
5. **Separability as a unified criterion**: Establishes "separability from the origin" as a necessary and sufficient condition for generalization, resulting in a simple and powerful concept.

## Limitations & Future Work

1. **Overly simplified model**: The generalization from logistic regression to real neural networks is not yet clear, and the form of the critical threshold in non-linear models could be vastly different.
2. **Gaussian data assumption**: Real-world data distributions are far more complex than Gaussian distributions, and a simple closed-form critical threshold like $\lambda_c = 1/2$ may not exist.
3. **No-regularization setting**: In practice, grokking often co-occurs with regularization such as weight decay. The unregularized analysis in this paper needs to be extended.
4. **Lack of direct link to Transformers/modular arithmetic**: The bridge to the original grokking scenarios (Power et al.) still needs to be fully built.
5. **No analysis of finite-width networks**: Only linear models are considered, leaving the effects of non-linear activation functions and representation learning in deep networks unexplored.
6. **Incomplete SGD/Adam analysis**: The main results rely on the gradient flow limit, and the discussion of Adam in the appendix is limited to preliminary observations.
7. **Limited practical guidance**: Although theoretically elegant, there is a lack of direct recommendations on how to predict and exploit grokking in practice.

## Related Work & Insights

- **Power et al. (2022)**: First discovered grokking.
- **Soudry et al. (2018)**: Implicit bias of gradient descent on separable data $\to$ the theoretical foundation of this work.
- **Levi et al. (2023)**: Complete dynamical solution of grokking in linear regression.
- **Gromov (2023)**: Grokking and latent space structures in solvable models.
- **Wendel (1962)**: Classical theorem on the separability of random point sets $\to$ the source of $\lambda_c = 1/2$ in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of critical phenomena for explaining grokking is completely novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Mutual validation between theory and experiments, with full analytical solutions for the simplified model.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, strong physical intuition, and well-structured logic.
- Value: ⭐⭐⭐⭐ Significant contribution to understanding the mechanisms of grokking, though further work is required to extend these findings to deep networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product](emergence_in_non-neural_models_grokking_modular_arithmetic_via_average_gradient_.md)
- [\[ICLR 2026\] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking](../../ICLR2026/optimization/egalitarian_gradient_descent_a_simple_approach_to_accelerated_grokking.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](../../ICLR2026/optimization/generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)
- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[ICML 2026\] Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization](../../ICML2026/optimization/conflicting_biases_at_the_edge_of_stability_norm_versus_sharpness_regularization.md)

</div>

<!-- RELATED:END -->
