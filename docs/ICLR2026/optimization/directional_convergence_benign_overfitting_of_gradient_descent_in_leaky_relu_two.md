---
title: >-
  [Paper Note] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks
description: >-
  [ICLR2026][Optimization][benign overfitting] This paper provides the first proof of directional convergence of gradient descent in leaky ReLU two-layer neural networks…
tags:
  - "ICLR2026"
  - "Optimization"
  - "benign overfitting"
  - "directional convergence"
  - "leaky ReLU"
  - "implicit bias"
  - "gradient descent"
  - "neural networks"
date: 2026-05-08
content_hash: 559546c238f44ec4
---

# Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks

**Conference**: ICLR2026
**arXiv**: [2505.16204](https://arxiv.org/abs/2505.16204)  
**Code**: None  
**Area**: Optimization
**Keywords**: benign overfitting, directional convergence, leaky ReLU, implicit bias, gradient descent, neural networks

## TL;DR

This paper provides the first proof of directional convergence of gradient descent in leaky ReLU two-layer neural networks, and on this basis establishes sufficient conditions for benign overfitting under a significantly broader mixed-data setting that goes well beyond nearly orthogonal data, while uncovering a novel phase transition phenomenon.

## Background & Motivation

Benign overfitting is a theoretically surprising phenomenon in deep learning: overparameterized neural networks can perfectly interpolate training data (including noisy labels) while still achieving small test error. This challenges the classical statistical wisdom that fitting and generalization must be traded off against each other.

Although benign overfitting is relatively well understood in classical models such as linear regression, linear classification, and kernel methods, its theoretical understanding in neural networks remains very limited. The central difficulty is that the implicit bias of gradient descent is hard to characterize precisely for ReLU-type networks. In linear classification, gradient descent minimizing logistic/exponential loss converges directionally to the max-margin classifier—a property that is key to analyzing benign overfitting. For ReLU-type networks, however, directional convergence of gradient descent had never been proven; prior results were restricted to gradient flow or approximately smooth leaky ReLU variants.

## Core Problem

1. **Directional convergence**: Can one prove that gradient descent exhibits directional convergence in non-smooth leaky ReLU two-layer networks, and precisely characterize the direction of convergence?
2. **Breadth of benign overfitting**: Can theoretical guarantees for benign overfitting be extended from nearly orthogonal data to more general mixed-data settings?
3. **Tightness and failure conditions**: When must benign overfitting necessarily fail? Are the existing upper bounds tight?

## Method

### Problem Setup

Consider a two-layer leaky ReLU network of fixed width $m$: $f(\boldsymbol{x};W) = \sum_{j=1}^m a_j \phi(\langle \boldsymbol{x}, \boldsymbol{w}_j \rangle)$, where $\phi$ is the $\gamma$-leaky ReLU, the second-layer weights are fixed as $a_j = \pm 1/\sqrt{m}$, and only the first-layer weights are trained via gradient descent. The loss function is the exponential loss $\ell(u)=\exp(-u)$, and the data follow a mixture model $\boldsymbol{x} = y\boldsymbol{\mu} + \boldsymbol{z}$.

### Directional Convergence (Theorem 4.8)

This constitutes the central technical contribution of the paper. Directional convergence is established under two conditions:

- **Case 1 (Positive Correlation)**: When the signal $\boldsymbol{\mu}$ is sufficiently strong so that $\langle y_i \boldsymbol{x}_i, y_k \boldsymbol{x}_k \rangle \geq 0$ holds for all pairs of training samples. This condition breaks the upper bound on signal strength imposed in prior work.
- **Case 2 (Near Orthogonality)**: When the data are approximately orthogonal, subsuming the settings of prior work as a special case.

The key proof strategy relies on the notion of *neuron activation*: if all neurons are activated after the first gradient descent step (i.e., $a_j y_i \langle \boldsymbol{x}_i, \boldsymbol{w}_j \rangle > 0$ for all $i, j$), the entire optimization reduces to a linear classification problem in a transformed space, upon which the classical result of Soudry et al. (2018) yields directional convergence. Activation of all neurons after the first step is guaranteed by initializing with a scale much smaller than the step size. The direction of convergence is precisely characterized as the unique solution (strictly convex) of a constrained optimization problem (5), and the resulting network has a linear decision boundary.

### Classification Error Bound (Theorem 5.1)

Building on the precise characterization of the convergence direction, a closed-form expression for the classification error is derived. The central finding is a **phase transition**:

- **Weak-signal regime** ($n\|\boldsymbol{\mu}\|^2 \lesssim R$): The error is noise-dominated, corresponding to the nearly orthogonal setting of prior work.
- **Strong-signal regime** ($n\|\boldsymbol{\mu}\|^2 \gtrsim R$): The signal begins to dominate, and the form of the error bound changes qualitatively.

For Gaussian mixture models, both upper and lower bounds are provided, establishing the tightness of the error bounds and revealing when benign overfitting must necessarily fail.

### Probabilistic Guarantees for Benign Overfitting (Theorems 6.2 & 6.3)

Substituting the deterministic conditions into a stochastic setting, the paper proves that under sufficient overparameterization:

- **Sub-Gaussian mixture model**: Benign overfitting holds with high probability, and the classification error bound matches the Bayes-optimal rate (up to a constant) in the isotropic Gaussian strong-signal regime.
- **Polynomial-tail mixture model** (Theorem 6.3): The results are extended for the first time to distributions with heavier tails ($\mathbb{E}|\xi_j|^r \leq K$, $r \in (2,4]$), substantially relaxing the sub-Gaussian or bounded log-Sobolev constant assumptions required in prior work.

## Key Experimental Results

This is a purely theoretical paper with no numerical experiments. The main results are presented as theorems:

| Aspect | Prior Best Result | This Paper |
|--------|------------------|------------|
| Directional convergence (gradient descent) | Gradient flow or smooth approximation only | First proof for non-smooth leaky ReLU |
| Data setting | Nearly orthogonal data | Mixed data (including strong-signal regime) |
| Distribution assumption | Sub-Gaussian / bounded log-Sobolev | Extended to polynomial-tail distributions |
| Network width | $m$ must grow with $n$ | Fixed width $m$ |
| Error lower bound | None | Matching lower bound for Gaussian mixtures |

## Highlights & Insights

1. **First proof of directional convergence**: This is the first work to prove directional convergence of gradient descent (rather than gradient flow) in ReLU-type networks and to precisely characterize the direction of convergence, filling a critical gap in the literature.
2. **Separation of deterministic and probabilistic analysis**: All core theorems are first stated in deterministic form and then applied to stochastic settings, making the theoretical framework amenable to a broader class of distributions.
3. **Discovery of a phase transition**: A weak-signal–strong-signal phase transition is identified for the first time in two-layer neural networks, and its intrinsic nature—rather than an artifact of the analysis—is confirmed via matching upper and lower bounds.
4. **Fixed-width network**: The results do not require the network width $m$ to grow with $n$, going beyond the NTK/lazy training regime.

## Limitations & Future Work

1. **Restricted to two-layer networks**: Extension to deeper networks is a natural open direction, but poses formidable technical challenges.
2. **No label noise**: Label-flipping noise is not considered; the authors conjecture that the weak-signal regime is unaffected, but the strong-signal regime may be significantly impacted.
3. **Fixed second layer**: Only the first-layer weights are trained, with the second layer fixed at $\pm 1/\sqrt{m}$—a standard simplifying assumption in the field but one that diverges from practice.
4. **Strong initialization condition**: The requirement that the initialization be much smaller than the step size to guarantee first-step activation is technically demanding.
5. **No numerical validation**: As a purely theoretical work, the paper lacks experimental verification, making it difficult to assess empirically whether the theoretical bounds are tight in practice.

## Related Work & Insights

| Work | Optimization | Directional Convergence | Data Setting | Distribution |
|------|-------------|------------------------|--------------|--------------|
| Frei et al. (2022) | Gradient descent | Smooth approximation | Nearly orthogonal | Sub-Gaussian |
| Frei et al. (2023b) | Gradient flow | ✓ | Nearly orthogonal | Sub-Gaussian |
| Xu & Gu (2023) | Gradient descent | Partial | Nearly orthogonal | Bounded log-Sobolev |
| Cai et al. (2025) | Gradient descent | ✓ (requires twice differentiable) | — | — |
| **Ours** | **Gradient descent** | **✓ (non-smooth)** | **Mixed data (incl. strong-signal)** | **Polynomial tail** |

The positioning of this paper is clear: it simultaneously advances the best known results along all key dimensions—optimization method, data generality, distributional assumptions, and network width.

The proof strategy of reducing directional convergence to linear classification via neuron activation is elegant and may inspire analyses of other non-smooth activation functions such as standard ReLU. The discovery of the phase transition suggests that the applicability of benign overfitting may be more restricted than previously thought—at certain signal-to-noise ratios, generalization must necessarily fail even when directional convergence holds. The separation of deterministic and probabilistic conditions is a theoretical analysis paradigm worth adopting in other statistical learning theory problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First proof of directional convergence of gradient descent in ReLU-type networks)
- Experimental Thoroughness: ⭐⭐ (Purely theoretical, no experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; Table 1 comparison is immediately readable)
- Value: ⭐⭐⭐⭐⭐ (Fills a critical theoretical gap with far-reaching implications)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](../../NeurIPS2025/optimization/do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](../../NeurIPS2025/optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](../../NeurIPS2025/optimization/optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
