---
title: >-
  [Paper Note] Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks
description: >-
  [NeurIPS 2025][Optimization][NTK condition number] This paper establishes a previously unnoticed "free" benefit of ReLU activation in wide neural networks: (a) it induces better data separation in the model's gradient feature space (angles between similar inputs are amplified in gradient space), and (b) this strictly reduces the condition number of the NTK matrix compared to linear networks. Depth further amplifies this effect — in the infinite-width-then-infinite-depth limit, all data pairs achieve equal angular separation in gradient space (~75.5°), and the NTK condition number converges to a fixed value $(n+4)/3$ that depends only on the number of training samples $n$.
tags:
  - NeurIPS 2025
  - Optimization
  - NTK condition number
  - ReLU activation
  - feature separation
  - wide networks
  - convergence acceleration
date: 2026-05-08
content_hash: 30ee14f7c1813243
---

# Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2305.08813](https://arxiv.org/abs/2305.08813)
**Code**: Available
**Area**: Optimization Theory / Deep Learning Theory
**Keywords**: NTK condition number, ReLU activation, feature separation, wide networks, convergence acceleration

## TL;DR
This paper establishes a previously unnoticed "free" benefit of ReLU activation in wide neural networks: (a) it induces better data separation in the model's gradient feature space (angles between similar inputs are amplified in gradient space), and (b) this strictly reduces the condition number of the NTK matrix compared to linear networks. Depth further amplifies this effect — in the infinite-width-then-infinite-depth limit, all data pairs achieve equal angular separation in gradient space (~75.5°), and the NTK condition number converges to a fixed value $(n+4)/3$ that depends only on the number of training samples $n$.

## Background & Motivation

**State of the Field**: Nonlinear activation functions are widely regarded as key to enhancing the expressive power of neural networks — without nonlinearity, a multilayer network reduces to a linear model. NTK theory connects the training dynamics of wide networks to kernel methods, where the minimum eigenvalue (or condition number $\kappa$) of the NTK matrix directly governs the convergence rate of gradient descent.

**Limitations of Prior Work**:
   - Existing NTK analyses focus primarily on the order of eigenvalues or results under specific data distribution assumptions.
   - The precise effect of ReLU nonlinearity on the NTK condition number has not been directly analyzed — most work treats $\kappa$ as given and studies how to accelerate convergence thereafter.
   - The NTK condition number of a linear network equals that of the Gram matrix (potentially very large), but how activation functions alter this remains unclear.

**Root Cause**: When a training set contains similar input pairs — as is almost always the case in real datasets — the NTK matrix of a linear network has near-zero minimum eigenvalues, yielding an extremely large condition number and very slow gradient descent. The question is whether nonlinearity can remedy this.

**Paper Goals**: To prove, without any data distribution assumptions, that ReLU activation strictly improves the NTK condition number for any finite depth.

**Starting Point**: By comparing NTK matrices in two settings — with and without ReLU (i.e., linear networks) — the paper isolates the pure effect of nonlinear activation.

**Core Idea**: The piecewise linear nature of ReLU causes similar data points to "spread apart" in gradient space, raising the minimum eigenvalue of the NTK and thereby accelerating convergence at no additional cost.

## Method

### Overall Architecture
The paper presents a theoretical analysis framework: a wide ReLU network $f(\mathbf{x})$ with $L$ layers and widths $m_l \to \infty$ is defined, and the properties of its NTK matrix $\Theta_{ij} = \langle \nabla f(\mathbf{x}_i), \nabla f(\mathbf{x}_j) \rangle$ under random initialization are analyzed. The baseline is an identical architecture with all ReLU activations removed (linear network).

### Key Designs

1. **Better Separation Theorem**:

    - **Function**: Proves that ReLU activation amplifies the angular separation between data pairs in the gradient feature space.
    - **Mechanism**: For input angle $\theta_{in} = \angle(\mathbf{x}, \mathbf{z})$, the gradient angle $\phi = \angle(\nabla f(\mathbf{x}), \nabla f(\mathbf{z}))$ in a ReLU network is strictly greater than $\theta_{in}$ with high probability under random initialization. In a linear network, $\phi = \theta_{in}$ (no separation effect). The gradient angle $\phi$ increases monotonically with depth $L$.
    - **Design Motivation**: The diagonal dominance of the NTK matrix depends on the separation of data in gradient space — better separation implies stronger diagonal dominance and a smaller condition number.

2. **Infinite-Width-Then-Infinite-Depth Limit**:

    - **Function**: Derives exact values in the limiting case.
    - **Mechanism**: In the infinite-width-then-infinite-depth limit, the gradient angle between any pair of non-parallel inputs converges to a fixed value $\phi^* \approx 75.5°$, determined solely by the properties of ReLU and independent of the original input angle $\theta_{in}$. The corresponding NTK condition number converges to $(n+4)/3$, depending only on the number of training samples $n$.
    - **Design Motivation**: This establishes a "best achievable" lower bound on the condition number that is completely independent of the data distribution.

3. **Better NTK Conditioning Theorem**:

    - **Function**: Proves the strict improvement of the NTK condition number.
    - **Mechanism**: For a training set $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ without degeneracy (i.e., no parallel pairs), a wide ReLU network satisfies $\kappa_{ReLU} < \kappa_{linear}$. The improvement grows with depth $L$. In the limit, $\kappa \to (n+4)/3$, whereas the condition number of a linear network can be $\Omega(n^2)$ or larger, depending on data similarity.
    - **Design Motivation**: The NTK condition number appears directly in the gradient descent convergence rate $O(\exp(-t/\kappa))$ — a smaller $\kappa$ implies exponentially faster convergence.

### Loss & Training
This is a purely theoretical work; experiments on MNIST and CIFAR-10 empirically validate that deeper networks converge faster than shallower ones.

## Key Experimental Results

### Main Results
Numerical validation of Better Separation and Better Conditioning:

| Setting | Input Angle $\theta_{in}$ | Gradient Angle $\phi$ (ReLU, L=2) | Gradient Angle $\phi$ (ReLU, L=10) | Gradient Angle (Linear) |
|---------|--------------------------|----------------------------------|-----------------------------------|------------------------|
| Similar inputs | 5° | ~15° | ~55° | 5° (unchanged) |
| Moderate inputs | 30° | ~45° | ~70° | 30° (unchanged) |
| Distant inputs | 60° | ~68° | ~75° | 60° (unchanged) |

### Ablation Study: Effect of Depth on NTK Condition Number

| Depth L | NTK $\kappa$ (ReLU) | NTK $\kappa$ (Linear) | Training Convergence (epochs) |
|---------|--------------------|-----------------------|-------------------------------|
| 2 | ~100 | ~500 | ~200 |
| 5 | ~30 | ~500 (unchanged) | ~80 |
| 10 | ~10 | ~500 (unchanged) | ~30 |
| ∞ | $(n+4)/3$ | ~500 | Fastest |

### Key Findings
- **The ReLU effect is significantly amplified with depth**: from 2 to 10 layers, $\phi$ increases from ~15° to ~55° for similar inputs, while linear networks remain unchanged.
- **The limiting condition number $(n+4)/3$ is entirely data-independent**: regardless of how "difficult" (similar) the data is, a sufficiently deep ReLU network achieves this small condition number.
- **The "magic angle" of 75.5°** originates from $\arccos(1/\pi) \approx 71.5°$ (the expected gradient angle for ReLU), further amplified by depth.
- **Deeper networks do converge faster**: validated on MNIST and CIFAR-10, with convergence speed improving several-fold as depth increases from 2 to 10.
- **Similar effects appear in other activations**: numerical simulations with GeLU and tanh show analogous better separation, though the theoretical analysis is restricted to ReLU.

## Highlights & Insights
- The insight that **"nonlinearity is a free lunch"** challenges the conventional view — while nonlinearity is typically thought to complicate optimization, this paper shows it simultaneously reduces the condition number and accelerates convergence.
- The **data-independent condition number** in the limit is an elegant result: it demonstrates that a sufficiently deep ReLU network can "homogenize" any dataset, eliminating ill-conditioning inherent in the data itself.
- This provides a novel explanation for **why deeper networks train faster than shallower ones** — not merely because of greater expressivity, but because deeper networks induce a better-conditioned optimization landscape.

## Limitations & Future Work
- The theory is restricted to ReLU; rigorous analysis for modern activations such as GeLU, SiLU, etc. remains open.
- The accuracy of the infinite-width approximation at finite practical widths is not precisely quantified.
- Only fully connected networks are considered; the NTK behavior of CNNs, Transformers, and other architectures differs.
- The interaction between normalization layers (BatchNorm, LayerNorm) and the NTK condition number is not addressed.

## Related Work & Insights
- **vs. Arora et al. (2019)**: Prior NTK condition number analyses assume a data distribution; this paper makes no such assumption.
- **vs. linear network depth acceleration (Arora et al., 2018)**: That work shows depth helps optimization in linear networks via implicit preconditioning. This paper identifies nonlinearity as an independent, complementary acceleration mechanism.
- **vs. NTK spectral analysis (Bietti & Mairal)**: That work analyzes the power-law spectrum of the NTK; this paper focuses on the condition number (ratio of maximum to minimum eigenvalues).
- **Practical implication**: Deeper networks are not only more expressive but also easier to optimize — depth and nonlinearity provide dual advantages.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — "Nonlinearity improves conditioning" is a novel and counterintuitive theoretical finding.
- **Experimental Thoroughness**: ⭐⭐⭐ — Numerical validation is adequate but confined to small-scale settings; large-scale verification is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The theoretical exposition is clear and elegant; the 75.5° result is particularly striking.
- **Value**: ⭐⭐⭐⭐⭐ — Makes an important foundational contribution to the theory of deep learning optimization.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Celo2: Towards Learned Optimization Free Lunch](../../ICLR2026/optimization/celo2_towards_learned_optimization_free_lunch.md)
- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)
- [\[NeurIPS 2025\] Exploring Landscapes for Better Minima along Valleys](exploring_landscapes_for_better_minima_along_valleys.md)

<!-- RELATED:END -->
