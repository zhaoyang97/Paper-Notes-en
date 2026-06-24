---
title: >-
  [Paper Note] Compelling ReLU Networks to Exhibit Exponentially Many Linear Regions at Initialization and During Training
description: >-
  [ICML2025][Optimization][ReLU networks] An asymmetric triangular wave-based reparameterization method for ReLU networks is proposed, forcing a 4-neuron-wide network of depth $d$ to generate $2^d$ linear regions at initialization and maintain this exponential expressivity during pre-training, reducing error by **3 orders of magnitude** on 1D function approximation tasks.
tags:
  - "ICML2025"
  - "Optimization"
  - "ReLU networks"
  - "linear regions"
  - "network initialization"
  - "reparameterization"
  - "triangular wave function"
  - "exponential expressivity"
date: 2026-05-08
content_hash: e7ca26848e5b51f5
---

# Compelling ReLU Networks to Exhibit Exponentially Many Linear Regions at Initialization and During Training

**Conference**: ICML2025  
**arXiv**: [2311.18022](https://arxiv.org/abs/2311.18022)  
**Code**: To be confirmed  
**Area**: ReLU Network Theory  
**Keywords**: ReLU networks, linear regions, network initialization, reparameterization, triangular wave function, exponential expressivity

## TL;DR

An asymmetric triangular wave-based reparameterization method for ReLU networks is proposed, forcing a 4-neuron-wide network of depth $d$ to generate $2^d$ linear regions at initialization and maintain this exponential expressivity during pre-training, reducing error by **3 orders of magnitude** on 1D function approximation tasks.

## Background & Motivation

The output of a ReLU network is a piecewise linear function. Theoretically, the number of linear regions can grow exponentially with depth (Montufar et al., 2014), which is one of the core arguments for the superiority of deep networks over shallow ones. However, several critical issues exist in practice:

**Inefficiency of Random Initialization**: Hanin & Rolnick (2019) proved that the average number of linear regions in randomly initialized networks is independent of depth and only depends on the total number of neurons, completely wasting the exponential advantage of depth.

**Dying ReLU Phenomenon**: Random initialization can cause outputs of an entire layer's neurons to be negative. Once truncated by ReLU, gradients completely vanish. The probability of this phenomenon increases with depth (Lu et al., 2020).

**Limitations of Gradient Descent**: The number of linear regions is not a local property of the parameter space, making it difficult for gradient descent to directly optimize this count (as shown in Figure 5, if the neuron outputs of adjacent layers do not cross zero, small perturbations cannot generate new regions).

**Network Redundancy**: Frankle & Carlin (2019) showed that approximately 95% of weights can be pruned without significantly reducing accuracy.

The core motivation of this work is: Can we force ReLU networks to possess an exponential number of linear regions from the very beginning through mathematical construction rather than random initialization?

## Method

### Core Idea: Asymmetric Triangular Wave Reparameterization

Instead of directly manipulating raw weights, the weights of a ReLU network with depth $d$ and width 4 are parameterized using peak positions of triangular functions $a_i \in (0,1)$. The key insight is that this parameterization guarantees the generation of $2^d$ linear regions regardless of any chosen values for $a_i$.

**Triangular Function Definition**: Each layer is associated with an asymmetric triangular function:

$$T_i(x) = \begin{cases} \frac{x}{a_i} & 0 \leq x \leq a_i \\ 1 - \frac{x - a_i}{1 - a_i} & a_i \leq x \leq 1 \end{cases}$$

where $a_i$ is the peak position. Each triangular function is implemented by 2 ReLU neurons.

**Composition of Triangular Waves**: In deep networks, the triangular functions of each layer are composed:

$$W_i(x) = T_i \circ T_{i-1} \circ \cdots \circ T_0(x)$$

Each composition doubles the number of linear regions, meaning $W_i$ has $2^i$ linear regions.

**Network Output**: The final output is formed by the weighted sum of the triangular waves from each layer:

$$F(x) = \sum_{i=0}^{\infty} s_i W_i(x)$$

where $s_i$ represents the scaling coefficients. An additional third "sum" neuron is introduced to accumulate outputs layer-by-layer, similar to a residual connection. A fourth "bias" neuron is used to replace the exponentially decaying bias term to avoid numerical conditioning issues.

### Weight Matrix Details

The weight matrix for each hidden layer is of size $4 \times 4$:

$$\begin{bmatrix} 1 & \pm S_i/a_i & -S_i/(a_i - a_i^2) & 0 \\ 0 & S_i/a_i & -S_i/(a_i - a_i^2) & 0 \\ 0 & S_i/a_i & -S_i/(a_i - a_i^2) & -S_i a_{i+1} \\ 0 & 0 & 0 & S_i \end{bmatrix}$$

where $S_i = s_i / s_{i-1}$ is the ratio of adjacent scaling coefficients. The rows correspond to the four neurons: sum, $t_1$, $t_2$, and bias.

### Differentiability Regularization (Theorem 3.1)

To prevent the output from degenerating into a fractal curve, $F(x)$ is required to be differentiable in the infinite depth limit. This yields a recurrence relation for the scaling coefficients:

$$s_{i+1} = s_i (1 - a_{i+1}) \cdot a_{i+2}$$

Specifically, the peak position $a_i$ uniquely determines the scaling coefficient $s_i$, which reduces the number of free parameters while guiding the optimizer toward smoother solutions.

### Three-Stage Training Pipeline

1. **Reparameterization and Initialization**: Set network weights using the triangular peak parameters $a_i$ to guarantee $2^d$ linear regions.
2. **Pre-training**: Train the network in the triangular wave parameter space using the Adam optimizer. Gradients are backpropagated through original weights to update $a_i$, maintaining exponential linear regions throughout.
3. **Standard Training**: Switch back to the original weight parameter space to release constraints, and fine-tune using gradient descent.

## Key Experimental Results

### 1D Convex Function Approximation (500 densely sampled points, $4 \times 5$ network, min/mean over 30 runs)

| Method | Min MSE ($x^3$) | Min MSE ($x^{11}$) | Mean MSE ($x^3$) | Mean MSE ($x^{11}$) |
|------|-----------------|---------------------|-------------------|----------------------|
| Kaiming Init | $2.11 \times 10^{-5}$ | $2.19 \times 10^{-5}$ | $7.20 \times 10^{-2}$ | $2.82 \times 10^{-2}$ |
| RAAI Init | $2.14 \times 10^{-5}$ | $4.40 \times 10^{-5}$ | $3.97 \times 10^{-2}$ | $4.12 \times 10^{-2}$ |
| Skipping Pre-training | $7.63 \times 10^{-7}$ | $1.86 \times 10^{-5}$ | $3.89 \times 10^{-5}$ | $3.56 \times 10^{-4}$ |
| Pre-training (No Regularization) | $1.64 \times 10^{-7}$ | $3.20 \times 10^{-6}$ | $1.02 \times 10^{-5}$ | $3.73 \times 10^{-5}$ |
| **Pre-training + Theorem 3.1** | **$7.86 \times 10^{-8}$** | **$8.86 \times 10^{-7}$** | **$5.27 \times 10^{-7}$** | **$7.87 \times 10^{-6}$** |

**Key Observation**: Compared to Kaiming initialization, the proposed complete method reduces the minimum error by approximately **3 orders of magnitude** and the mean error by **5 orders of magnitude**.

### Generalization on Sparse Data (10 training points, 10 testing points)

| Method | Min MSE ($x^3$) | Min MSE ($x^{11}$) | Min MSE ($\sin x$) | Min MSE ($\tanh 3x$) |
|------|-----------------|---------------------|---------------------|----------------------|
| Kaiming Init | $2.41 \times 10^{-4}$ | $2.14 \times 10^{-3}$ | $2.27 \times 10^{-5}$ | $1.60 \times 10^{-4}$ |
| **Pre-training + Theorem 3.1** | **$5.65 \times 10^{-6}$** | **$6.53 \times 10^{-4}$** | **$7.92 \times 10^{-7}$** | **$5.09 \times 10^{-6}$** |

Even with sparse data, a 1–2 orders of magnitude advantage is maintained, indicating that a larger number of linear regions enhances the predictive ability on unseen data.

### VGG-16 ImageNet Classification

Replacing the fully connected classifier of VGG-16 (width 4096), which adds only about 0.5% computational overhead:
- Shows advantages in early training, though the final accuracy is comparable to the original method (both exceeding the 73.3% reported by PyTorch).
- The role of precise decision boundaries in classification tasks might be less critical than in regression tasks.

## Highlights & Insights

1. **Elegant Mathematical Construction**: Constraining the weight space of the ReLU network to a low-dimensional manifold of "triangular peak positions", where any point is guaranteed to have an exponential number of linear regions, is a highly ingenious reparameterization.
2. **Three Orders of Magnitude Improvement**: Achieves an impressive error reduction in 1D function approximation.
3. **Bridge Between Theory and Practice**: Generalizes the theoretical construction of Telgarsky/Yarotsky ($x^2$ approximation) into a trainable family of parameters.
4. **Natural Derivation of Differentiability Condition**: Theorem 3.1 provides a unique determination of the scaling coefficients, internalizing regularization directly into the network architecture.
5. **Connection to KAN**: Similar to Kolmogorov-Arnold Networks (KANs), it achieves multi-dimensional approximation by using 1D constructions as activation functions for a larger network.

## Limitations & Future Work

1. **Width Constraint**: The core construction is only applicable to networks of width 4; extending this to arbitrary widths still lacks a theoretical framework.
2. **Limited Gains in Classification**: VGG-16 experiments show no significant advantage in classification tasks, necessitating more validation for practical utility.
3. **Restriction to 1D Convex Functions**: Direct mathematical guarantees only hold for 1D convex functions; non-convex and multi-dimensional cases rely on heuristic extensions.
4. **Scalability**: The performance of block-diagonal structures in large-scale networks has not been fully explored.
5. **Lack of Direct Comparison with Methods like KAN**: As another method utilizing 1D constructions, it lacks experimental comparisons with KAN/spline methods.
6. **Timing for Switching from Pre-training to Standard Training**: Theoretical guidance is lacking on when to switch the parameterization.

## Related Work & Insights

- **Telgarsky (2015)**: Showed that deep ReLU networks can generate an exponential number of linear segments using symmetric triangular waves; this work generalizes this to asymmetric triangular waves.
- **Yarotsky (2017)**: Approximated $x^2$ using composition of triangular waves; this work extends this construction to a trainable family of parameters.
- **Hanin & Rolnick (2019)**: The negative result showing that the number of linear regions in randomly initialized networks is independent of depth served as a direct motivation for this paper.
- **KAN (Liu et al., 2024)**: Based on the Kolmogorov-Arnold theorem, KANs construct multi-dimensional approximations using 1D activation functions, sharing a similar underlying concept.
- **Elbrächter et al. (2019)**: Explored well-conditioned parameter spaces of ReLU networks, complementing the reparameterization scheme proposed in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of triangular wave reparameterization to guarantee exponential linear regions is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐ — 1D experiments are comprehensive, but large-scale experiments (ImageNet) only provide preliminary demonstrations.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear, and diagrams are intuitive.
- Value: ⭐⭐⭐⭐ — Provides a fresh theoretical perspective and practical tool for ReLU network initialization and expressivity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[NeurIPS 2025\] Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](../../NeurIPS2025/optimization/better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[ICML 2025\] Grokking at the Edge of Linear Separability](grokking_at_the_edge_of_linear_separability.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](../../NeurIPS2025/optimization/training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)

</div>

<!-- RELATED:END -->
