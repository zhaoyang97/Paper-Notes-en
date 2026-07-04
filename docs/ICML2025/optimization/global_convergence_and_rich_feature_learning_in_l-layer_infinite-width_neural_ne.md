---
title: >-
  [Paper Note] Global Convergence and Rich Feature Learning in $L$-Layer Infinite-Width Neural Networks under $\mu$P Parametrization
description: >-
  [ICML2025][Optimization][$\mu$P parametrization] This work proves that under $\mu$P (Maximal Update Parametrization), when an $L$-layer infinite-width MLP is trained using SGD, the features at each layer remain linearly independent and undergo substantial adaptation throughout training, guaranteeing that any convergence point of training is a global minimum. This is the first work to simultaneously achieve both theoretical goals of "rich feature learning" and "global converge…
tags:
  - "ICML2025"
  - "Optimization"
  - "$\\mu$P parametrization"
  - "infinite-width networks"
  - "feature learning"
  - "global convergence"
  - "Gaussian process"
  - "Tensor Programs"
  - "linear independence"
date: 2026-05-08
content_hash: 575f660ae35a542c
---

# Global Convergence and Rich Feature Learning in $L$-Layer Infinite-Width Neural Networks under $\mu$P Parametrization

**Conference**: ICML2025  
**arXiv**: [2503.09565](https://arxiv.org/abs/2503.09565)  
**Code**: None  
**Area**: Optimization Theory / Neural Network Theory  
**Keywords**: $\mu$P parametrization, infinite-width networks, feature learning, global convergence, Gaussian process, Tensor Programs, linear independence

## TL;DR

This work proves that under $\mu$P (Maximal Update Parametrization), when an $L$-layer infinite-width MLP is trained using SGD, the features at each layer remain linearly independent and undergo substantial adaptation throughout training, guaranteeing that any convergence point of training is a global minimum. This is the first work to simultaneously achieve both theoretical goals of "rich feature learning" and "global convergence."

## Background & Motivation

One of the core questions in deep learning theory is: how can neural networks learn meaningful features while achieving global convergence in non-convex optimization?

**Limitations of NTK Parametrization**: Under the NTK (Neural Tangent Kernel) parametrization, training an infinite-width network is equivalent to a linear model, where features remain near their initialization throughout training ($Z^{z_t}(\xi) = Z^{z_0}(\xi)$), failing to undergo true feature learning. Although the NTK framework can prove global convergence, it is essentially "random features + linear regression," failing to explain the representation learning capability of actual networks.

**Limitations of Prior Work in Mean Field Parametrization**: The Mean Field approach performs well on networks with two or three layers, but for networks with more than four layers, feature vectors and gradients degenerate to zero vectors, causing feature collapse. Even though Integrable Parametrization (IP) partially alleviates this issue, deep networks in the infinite-width limit still evolve from stationary points, making rich feature learning difficult to achieve.

**Limitations of Standard Parametrization (SP)**: SP requires the learning rate to scale as $O(1/\text{width})$ as the width approaches infinity, which also prevents feature learning in the infinite-width limit.

The core question of this work is: **Can a parametrization scheme be found that allows deep networks to simultaneously achieve meaningful feature learning and global convergence?**

## Method

### Overall Architecture

This work investigates the training dynamics of $L$-layer MLPs under $\mu$P parametrization within the Tensor Programs framework. The core approach is as follows:

1. Leverage $Z$ random variables to characterize the element-wise distribution of hidden states at each layer in the infinite-width limit.
2. Analyze the structural invariants of the two families of Gaussian processes induced by forward and backward propagation.
3. Inductively prove the non-degeneracy of features layer-by-layer using covariance preservation properties and the GOOD activation function condition.

### $\mu$P Parametrization Design

The key difference of $\mu$P lies in the initialization variance and learning rate scaling of each layer:

| Layer          | Init Variance    | Learning Rate       |
|:---------------|:-----------------|:-------------------|
| Input layer $W^1$    | $1$              | $\eta \cdot n$     |
| Hidden layer $W^l$    | $n^{-1}$         | $\eta$             |
| Output layer $W^{L+1}$| $n^{-2}$         | $\eta \cdot n^{-1}$ |

Compared with NTK/SP, the learning rate for the input layer in $\mu$P is $\eta n$ (much larger than the $\eta$ in NTK), which enables the parameters of each layer to receive **maximal updates**, thereby retaining non-trivial feature evolution in the infinite-width limit.

### Key Theoretical Tools

**Z Random Variable Representation**: As $n \to \infty$, the coordinates of the hidden layer vectors $h^l, x^l$ tend to be i.i.o., and their asymptotic behavior is characterized by scalar random variables $Z^{h^l}, Z^{x^l}$. The inner product relationship between two vectors is encoded by the expectation:

$$\lim_{n\to\infty} x^\top y / n = \mathbb{E}[Z^x Z^y]$$

**Decomposition of Feature Evolution**: For any feature $z \in \{x^l, h^l\}$:

$$Z^{z_t}(\xi) = Z^{z_0}(\xi) + \underbrace{Z^{\delta z_1}(\xi) + \cdots + Z^{\delta z_t}(\xi)}_{\text{feature learning term}}$$

**Gaussian Process Families**: Training induces two families of Gaussian processes—the forward process $\{\hat{Z}^{W_0^l \delta x_s^{l-1}(\xi_i)}\}$ which tracks feature evolution, and the backward process $\{\hat{Z}^{W_0^{l\top} dh_s^l(\xi_i)}\}$ which describes gradient flow.

### Core Proof Ideas

**Covariance Preservation Property** (Key Discovery): The covariance structures of both the forward and backward Gaussian processes remain invariant during training:

$$\text{Cov}(\hat{Z}^{W_0^l \delta x_s^{l-1}(\xi)}, \hat{Z}^{W_0^l \delta x_t^{l-1}(\zeta)}) = \mathbb{E}[Z^{\delta x_s^{l-1}(\xi)} Z^{\delta x_t^{l-1}(\zeta)}]$$

This implies that the feature correlations between adjacent layers follow a consistent pattern, even while individual features undergo substantial evolution.

**GOOD Function Condition**: The activation function $\phi$ is required to satisfy (satisfied by sigmoid, tanh, and SiLU):
- Twice continuously differentiable, with $\phi', \phi''$ being bounded.
- For parameters $\{a_i, b_i, c_i\}$ satisfying specific conditions, $f(x)=\sum_i a_i \phi(b_i x + c_i)$ is not a constant function.
- $(r_1 + \phi(x))(r_2 + \phi'(x))$ is not a constant function almost everywhere.

**Four-Step Inductive Proof**:
1. **Features of the first hidden layer** $\hat{Z}^{W_0^2 \delta x_s^1}$: Base case, depending only on the input.
2. **Features of remaining layers** $\hat{Z}^{W_0^l \delta x_s^{l-1}}$: Advanced layer-by-layer utilizing the non-degeneracy of preceding layers.
3. **Gradients of the final layer** $\hat{Z}^{W_0^{L\top} dh_s^L}$: Based on the established feature properties.
4. **Gradients of remaining layers** $\hat{Z}^{W_0^{l\top} dh_s^l}$: Completes the backpropagation analysis.

## Main Theoretical Results

### Theorem 4.5 (Feature Non-Degeneracy)

Under Assumption 4.1 (input geometric conditions: $|\langle\xi_i,\xi_j\rangle| \neq |\langle\xi_i,\xi_k\rangle|$ and $|\langle\xi_i,\xi_j\rangle| \neq 0$) and Assumption 4.3 (GOOD activation function):

> At any time $t$ during the gradient descent training of an infinite-width $L$-layer MLP, the **pre-activation features** $\{Z^{h_t^l(\xi)}\}_{\xi \in S}$ and **post-activation features** $\{Z^{x_t^l(\xi)}\}_{\xi \in S}$ at each layer $l \in [L]$ remain linearly independent.

### Corollary 4.6 (Global Convergence)

> If the model converges at time $T$ (weights no longer change), then for all samples in subsequent mini-batches, the error signal must vanish: $\mathring{\chi}_{T,i} = 0$, i.e., converging to a global minimum of the training objective.

### Theoretical Significance

This represents the first proof of global convergence under conditions that allow substantial feature evolution. While NTK guarantees convergence without feature learning, and Mean Field enables feature learning but suffers from deep layer degradation, $\mu$P remains the only parametrization scheme that simultaneously achieves both.

## Experimental Validation

Experiments are conducted on a 3-hidden-layer MLP using the CIFAR-10 dataset, comparing SP, NTP, IP (Mean Field), and $\mu$P:

| Parametrization Scheme | Feature Variation Magnitude | Feature Diversity (Minimum Eigenvalue) | Space-Time Joint Feature Non-Degeneracy |
|:----------|:---------|:---------------------|:----------------|
| SP        | Small (close to initialization) | Rich | Decreases with width |
| NTP       | Small (close to initialization) | Rich | Decreases with width |
| IP (Mean Field) | Large | Low (feature collapse) | Decreases with width |
| **$\mu$P** | **Large and stable** | **Rich** | **Stable across widths** |

**Key Findings**:
- **Feature Variation Magnitude**: Measured by $\|h(x)-h^0(x)\|_2 / \|h^0(x)\|_2$, only $\mu$P maintains stable non-zero feature variation as the width increases.
- **Feature Diversity**: In terms of the minimum eigenvalue of the feature Gram matrix $K_{ij} = \langle h(\xi_i), h(\xi_j)\rangle$, $\mu$P maintains non-degeneracy.
- **Space-Time Joint Analysis**: Regarding the minimum eigenvalue of the Gram matrix after concatenating initial and final features $[h_1^0, h_1^T, \ldots, h_N^0, h_N^T]$, $\mu$P maintains high values across different widths, whereas all other schemes decay as width increases.

## Highlights & Insights

1. **First Unification of Two Key Goals**: Simultaneously proves rich feature learning and global convergence within the same theoretical framework, addressing a fundamental open problem in deep learning theory.
2. **Discovery of Covariance Invariants**: Highlighting that the second-order statistics of the forward and backward Gaussian processes remain consistent during training, a structural property previously unexploited.
3. **The GOOD Function Concept**: Proposes an elegant regularity condition for activation functions, covering mainstream activation functions such as sigmoid, tanh, and SiLU.
4. **Two-Layer Filtration Framework**: The design of $\mathcal{F}_t$ and $\mathcal{G}_t$ neatly decouples the new randomness in forward and backward propagation from historical information.
5. **Systematic Comparison of Parametrization Schemes**: Table 1 clearly demonstrates the trade-offs between feature learning and feature richness across different schemes, with $\mu$P being the only one to achieve both.

## Limitations & Future Work

1. **Limited to MLPs Only**: Does not extend to modern architectures like Transformers or CNNs, particularly leaving out feature learning analysis for attention mechanisms.
2. **ReLU Does Not Satisfy Conditions**: The widely used ReLU activation function fails Assumption 4.3 (since $\phi''$ does not exist), serving as a significant omission in the theory.
3. **Convergence Only, No Rate of Convergence**: The theorems only state "if the model converges, it reaches a global minimum," providing no quantitative characterization of the rate of convergence.
4. **Infinite-Width Assumption**: The conclusions strictly rely on the $n \to \infty$ limit; the quality of approximation under finite-width regimes is not discussed.
5. **Limited Experimental Scale**: Validated only on CIFAR-10 with a 3-layer MLP; tests on larger scales of data and deeper networks are lacking.
6. **Generalization Not Discussed**: Global convergence guarantees vanishing training error, leaving the analysis of generalization errors as future work.

## Related Work & Insights

- **NTK Theory** (Jacot et al., 2018; Du et al., 2019): Establishes linearized analyses of infinite-width networks but fails to capture feature learning.
- **Mean Field Analysis** (Mei et al., 2018; Chizat & Bach, 2018): Achieves global convergence in two-layer networks but suffers from degradation in deeper architectures.
- **Tensor Programs** (Yang, 2019; Yang & Hu, 2020): Proposes $\mu$P and unified frameworks of infinite-width limits, providing the core technical foundation of this paper.
- **$\mu$P practice** (Yang et al., 2021, 2023): Investigates hyperparameter transfer and spectral analysis, with this work providing theoretical guarantees.
- The covariance invariant approach developed here may inspire similar analyses for continuous-depth networks (Neural ODEs) and Transformers.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first work to unify and prove both feature learning and global convergence under $\mu$P.
- Experimental Thoroughness: ⭐⭐⭐ — Limited to CIFAR-10 + 3-layer MLP, but sufficient to validate the theory.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with clear technical pathways, though mathematical notation is somewhat dense.
- Value: ⭐⭐⭐⭐⭐ — Signifies a major advancement in the foundational theory of deep learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](../../NeurIPS2025/optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[ICML 2025\] Random Feature Representation Boosting](random_feature_representation_boosting.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[ICLR 2026\] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime](../../ICLR2026/optimization/on_the_convergence_behavior_of_preconditioned_gradient_descent_toward_the_rich_l.md)

</div>

<!-- RELATED:END -->
