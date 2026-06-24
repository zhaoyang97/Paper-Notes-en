---
title: >-
  [Paper Note] Transformative or Conservative? Conservation Laws for ResNets and Transformers
description: >-
  [ICML2025][Optimization][conservation laws] This work systematically derives and proves conservation laws under gradient flow training dynamics for modern architectures such as convolutional ResNets and Transformers. It reveals that residual connections do not alter conservation laws, block-level conservation laws are equivalent to those of isolated blocks, and the conservation error under discrete SGD is $O(\text{step-size}^2)$.
tags:
  - "ICML2025"
  - "Optimization"
  - "conservation laws"
  - "gradient flow"
  - "ResNet"
  - "Transformer"
  - "training dynamics"
  - "implicit bias"
  - "Lie algebra"
date: 2026-05-08
content_hash: 89c08a80124b5b25
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Transformative or Conservative? Conservation Laws for ResNets and Transformers

**Conference**: ICML2025  
**arXiv**: [2506.06194](https://arxiv.org/abs/2506.06194)  
**Code**: To be confirmed  
**Area**: Optimization  
**Keywords**: conservation laws, gradient flow, ResNet, Transformer, training dynamics, implicit bias, Lie algebra

## TL;DR
This work systematically derives and proves conservation laws under gradient flow training dynamics for modern architectures such as convolutional ResNets and Transformers. It reveals that residual connections do not alter conservation laws, block-level conservation laws are equivalent to those of isolated blocks, and the conservation error under discrete SGD is $O(\text{step-size}^2)$.

## Background & Motivation

**Background**: Conservation laws (quantities that remain constant during the training process) are crucial tools for understanding the training dynamics of neural networks. For shallow ReLU and linear networks, the known conservation laws take the form of $\|u_k\|^2 - \|v_k\|^2 = \text{const}$ ("balancing conditions"), and Marcotte et al. (2023) proved their completeness.

**Limitations of Prior Work**:
1. Existing theories only cover shallow ReLU/linear networks, while **conservation laws for convolutional networks, attention layers, deep ResNets, and Transformers remain completely unknown**.
2. Conservation laws reveal implicit bias (properties preserved from initialization to the final solution) and convergence; their absence for modern architectures limits theoretical understanding.
3. It remains unclear whether conservation laws under gradient flow still approximately hold in practical discrete SGD training.

**Core Idea**: Utilizing the Lie algebra framework and reparameterization techniques, this work systematically derives the conservation laws of basic building blocks in modern architectures (such as convolutional layers, attention layers, and residual blocks), and then extends the analysis to deep networks through the concept of "block-level conservation laws."

## Method

### Theoretical Framework

Consider gradient flow training dynamics (with optional weight decay):

$$\dot{\theta}(t) + \lambda(t)\theta(t) = -\nabla L_Z(\theta(t))$$

**Structure Theorem (Theorem 2.1)**: Conservation laws with weight decay can be completely determined by those without weight decay: $h(t,\theta) = H(\theta \exp(\int_0^t \lambda(s)ds))$, thereby simplifying the analysis to time-independent conservation functions.

**Characterization of Conservation Laws (Proposition 2.3)**: A smooth function $h(\theta)$ is a conservation law if and only if $\nabla h(\theta) \perp \mathcal{W}_\theta^{g,\ell}$, where $\mathcal{W}_\theta^{g,\ell}$ is the span of gradients in the parameter space.

**Counting Conservation Laws (Theorem 2.11)**: The number of independent conservation laws is exactly $D - k$, where $D$ is the parameter dimension and $k$ is the dimension of the Lie algebra $\text{Lie}(\mathbb{W}^{g,\ell})(\theta)$.

### Conservation Laws of Basic Building Blocks

**1. Residual connections do not change conservation laws (Proposition 3.2)**:
$\tilde{g}(\theta, x) = x + g(\theta, x)$ shares exactly the same conservation laws as $g(\theta, x)$. The proof is extremely concise: $\partial_\theta g = \partial_\theta \tilde{g}$.

**2. Multi-channel convolutional ReLU networks (Theorem 3.6)**:
A network with $c_1$ hidden channels has exactly $c_1$ independent conservation laws:

$$h_j(\theta) = \sum_{k=1}^{c_2}\|u_{k,j}\|^2 - \sum_{i=1}^{c_0}\|v_{j,i}\|^2, \quad 1 \leq j \leq c_1$$

**3. Single-head attention layers (Corollary 3.9)**:
For $g(\theta,x) = \text{softmax}(XQ^\top KX^\top)XV^\top O$, all conservation laws are functions of the following quantities:
$$QQ^\top - KK^\top, \quad VV^\top - OO^\top$$

**4. Multi-head attention (Corollary 3.10)**:
Each head conserves independently: $Q_h Q_h^\top - K_h K_h^\top$ and $V_h V_h^\top - O_h O_h^\top$ (completeness is left as an open problem).

**5. Cross-Entropy classification layers (Proposition 3.11)**:
The softmax layer has $m$ conservation laws: $h_j(\theta) = \sum_i \theta_{i,j}$ (the sum of weights in each column is conserved).

### Block-level Conservation Laws in Deep Networks

**Core Theorem (Theorem 4.6)**: A conservation law in a deep network $g_\theta$ that depends solely on the parameters of the $l$-th layer $\theta_l$ is exactly equivalent to the conservation law of the shallow network $g_{\theta_l}^l$ of that single layer (with respect to the Euclidean loss).

**Blocks across residual connections (Theorem 4.7)**: There are no non-trivial conservation laws for adjacent parameter pairs $(V^{l+1}, U^l)$ across a skip connection—residual connections "break" cross-block conservation.

### Approximate Conservation in Discrete SGD (Proposition 5.1)

The upper bound of conservation error under SGD:
$$\mathbb{E}|h(\theta_k) - h(\theta_0)| \leq \frac{C_h C_L}{2}\sum_{i=0}^{k-1}\tau_i^2$$

Under a constant step size $\tau$, it is $O(\tau^2 k)$; under a decaying step size $\tau_k = \tau_0/(k+1)$, it remains bounded at $O(\tau_0^2)$.

### Conservation Law Analysis of Adam Flow

The conservation law space for simplified Adam (sign gradient descent) is $\text{span}\{\text{sign}(\nabla L_Z(\theta))\}$, which is fundamentally different from gradient flow. For two-layer linear networks, numerical exploration reveals that no conservation laws exist (except for $n=m=r=1$).

## Key Experimental Results

### ResNet-18 / CIFAR-10 Experiments

| Settings | Observed Quantity | Conclusion |
|------|--------|------|
| SGD, learning rate $\in [10^{-3}, 5\times10^{-3}]$, no momentum/WD | Change in $\sum_j h_j(\theta_T)$ of the first residual block | Conservation error slope is proportional to $\tau^2$ |
| 10 random seeds × multiple learning rates | Theoretical slope $C\tau^2$ vs. Empirical | Theoretical predictions align with experimental results |
| Conservation error during 50 steps of training | Relative error $|h(\theta_k)-h(\theta_0)|/|h(\theta_0)|$ | Well-conserved under reasonable learning rates |

### Transformer / IMDb Experiments

- Training a Transformer on IMDb sentiment analysis, tracking $\|QQ^\top - KK^\top\|_F$ of the first attention head in the first layer.
- The conservation error also follows the $O(\text{step-size}^2)$ law.
- The presence or absence of masking has no impact on conservation behavior.

### Numerical Verification: No Extra Conservation Laws in Deep Networks

- For a ResNet with $q=2$ residual blocks, numerical computations confirm no extra conservation laws beyond block-level conservation laws when $m > 1$.
- Extra conservation laws exist when $m=1$ (Example 4.8: a two-block ReLU network has 3 instead of 2 conservation laws under specific sign conditions).

## Highlights & Insights

1. **Theoretical Elegance**: Unifying the analysis of conservation laws across various architectures using the Lie algebra framework, yielding concise and powerful proofs.
2. **Conciseness of Proposition 3.2**: The proof that residual connections do not affect conservation laws is only one line ($\partial_\theta g = \partial_\theta \tilde{g}$), yet the implications are profound.
3. **Composability of Block-level Analysis**: Theorem 4.6 reduces the analysis of deep networks to that of shallow components, significantly reducing complexity.
4. **Negative Result of Theorem 4.7**: The absence of conservation laws across residual connections indicates that skip connections "decouple" adjacent layers from the perspective of optimization dynamics.
5. **Bridge from Continuous to Discrete**: Proposition 5.1 quantifies the degree of conservation violation under SGD, enhancing the practical value of the theory.
6. **Structure Theorem for Weight Decay**: Elegantly demonstrating that conservation laws under WD tend toward zero at the optimal point (corresponding to known balancing properties).

## Limitations & Future Work

1. **LayerNorm is Not Covered**: Normalization layers in Transformers are ignored, which are indispensable components in practical models.
2. **Completeness of Multi-head Attention**: The completeness of multi-head conservation laws in Corollary 3.10 is not proven and remains an open problem.
3. **Missing Max-Pooling**: Max-pooling, commonly used in CNNs, is not included in the analysis.
4. **Constants in SGD Error Bounds**: $C_h$ and $C_L$ are mathematically challenging to explicitly determine in practice, limiting quantitative predictive power.
5. **Adam Optimizer**: Only a simplified version of Adam is analyzed; the full version of Adam is not covered.
6. **Special Case of $m=1$**: Reveals the existence of extra conservation laws, but the characterization of general conditions is incomplete.

## Related Work & Insights

- **Marcotte et al. (2023)**: Analytical framework for the completeness of conservation laws in shallow ReLU/linear networks, serving as the direct foundation of this work.
- **Du et al. (2018)**: Conservation laws for single-channel convolutional networks, which this work generalizes to multi-channel networks with proof of completeness.
- **Marion et al. (2023)**: Implicit bias where ResNet training solutions correspond to the discretization of Neural ODEs.
- **Vasudeva et al. (2024)**: Convergence of gradient descent to hard-margin SVM solutions in self-attention layers.
- **Analogy to Noether's Theorem**: The intrinsic connection between conservation laws and network invariances (rescaling of hidden neurons).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to systematically derive conservation laws for ResNet/Transformer and prove completeness)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Thorough theoretical verification, though coverage of practical training scenarios is limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical framework, progressive structure, and excellently organized proofs)
- Value: ⭐⭐⭐⭐⭐ (Provides fundamental tools for deep learning optimization theory)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conservation Laws for Modern Neural Architectures](../../ICML2026/optimization/conservation_laws_for_modern_neural_architectures.md)
- [\[ICML 2025\] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias](how_transformers_learn_regular_language_recognition_a_theoretical_study_on_train.md)
- [\[ICML 2025\] Can Transformers Learn Full Bayesian Inference In Context?](can_transformers_learn_full_bayesian_inference_in_context.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](../../NeurIPS2025/optimization/multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)

</div>

<!-- RELATED:END -->
