---
title: >-
  [Paper Note] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis
description: >-
  [NeurIPS 2025][nonlinear preconditioning] Under the anisotropic descent inequality framework, this paper introduces heavy ball momentum into nonlinearly preconditioned gradient methods and analyzes the convergence proper…
tags:
  - "NeurIPS 2025"
  - "nonlinear preconditioning"
  - "gradient clipping"
  - "anisotropic smoothness"
  - "heavy ball momentum"
  - "stochastic optimization"
date: 2026-05-08
content_hash: 5be73fbdd050af8a
---

# Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2510.11312](https://arxiv.org/abs/2510.11312)  
**Code**: [GitHub](https://github.com/JanQ/nonlin-prec-mom-stoch)  
**Area**: Others
**Keywords**: nonlinear preconditioning, gradient clipping, anisotropic smoothness, heavy ball momentum, stochastic optimization

## TL;DR

Under the anisotropic descent inequality framework, this paper introduces heavy ball momentum into nonlinearly preconditioned gradient methods and analyzes the convergence properties of their stochastic variants under multiple noise assumptions, thereby unifying the theoretical analysis of gradient clipping and normalized gradient methods.

## Background & Motivation

Many cost functions in modern machine learning **do not satisfy the classical Lipschitz gradient condition**. For instance, loss functions arising in language model training exhibit more general $(L_0, L_1)$-smoothness: $\|\nabla^2 f(x)\| \leq L_0 + L_1 \|\nabla f(x)\|$. Gradient clipping and normalization are standard practical techniques for addressing such problems, yet theoretical analyses are typically restricted to specific algorithmic forms.

A core observation is that gradient clipping methods can be unified as **nonlinearly preconditioned gradient descent**:

$$x^{k+1} = x^k - \gamma \nabla\phi^*(\nabla f(x^k))$$

where $\phi$ is a reference function and $\nabla\phi^*$ is the preconditioning map. Different choices of $\phi$ yield different algorithms:
- $\phi(x) = \varepsilon(-\|x\| - \ln(1-\|x\|))$ → normalized gradient descent (NGD)
- $\phi(x) = \cosh(\|x\|) - 1$ → **inverse hyperbolic gradient descent (iHGD)** — the focus of this paper

The preconditioning map of iHGD is $\nabla\phi^*(y) = \text{arsinh}(\|y\|) \frac{y}{\|y\|}$, which, unlike hard-threshold clipping, **adaptively scales** gradients — large gradients are compressed but retain information, while small gradients are left nearly unchanged.

Gaps in existing theory:
1. Nonlinearly preconditioned methods with momentum under general anisotropic smoothness conditions **have never been analyzed**
2. The convergence properties of stochastic variants beyond bounded variance assumptions remain unclear

## Method

### Overall Architecture

Two core algorithms are proposed:
- **m-NPGM (Algorithm 1)**: nonlinearly preconditioned gradient method with momentum
- **Stochastic NPGM**: momentum-free stochastic variant

### Key Designs

1. **Momentum update rule (m-NPGM)**: Unlike standard approaches, momentum is applied **to the preconditioned gradient** rather than applying momentum to the gradient before preconditioning:

    $m^k = \beta m^{k-1} + (1-\beta) \nabla\phi^*(\nabla f(x^k))$
    $x^{k+1} = x^k - \gamma m^k$

   The equivalent form is $x^{k+1} = x^k - (1-\beta)\gamma \nabla\phi^*(\nabla f(x^k)) + \beta(x^k - x^{k-1})$, i.e., the standard heavy ball method applied to the map $\nabla\phi^* \circ \nabla f$. This design is more natural since the analysis is grounded in the anisotropic descent inequality.

2. **Anisotropic descent inequality**: The core condition for analysis is $f(x) \leq f(\bar{x}) + \frac{1}{L} \star \phi(x - \bar{y}) - \frac{1}{L}\star\phi(\bar{x} - \bar{y})$, where $\bar{y} = \bar{x} - \frac{1}{L}\nabla\phi^*(\nabla f(\bar{x}))$. This is strictly more general than $(L_0,L_1)$-smoothness (Remark 1.4).

3. **Convergence analysis (Theorem 2.2)**: Under $\beta \in [0, 0.5)$ and $\gamma = \alpha/L$: $\min_{0 \leq k \leq K} \phi(\nabla\phi^*(\nabla f(x^k))) \leq \frac{L(f(x^0) - f_\star)}{\alpha(K+1)(1-2\beta)}$. Key technical challenges in the proof are: (a) the absence of a global gradient difference upper bound; (b) the range of the preconditioning map may cover the entire space, making distance control difficult.

4. **Preconditioned Lipschitz continuity (Assumption 2.5)**: A new condition is introduced, $\|\nabla\phi^*(\nabla f(x)) - \nabla\phi^*(\nabla f(\bar{x}))\| \leq L\|x - \bar{x}\|$, under which the range of $\beta$ is extended to $(0, 1)$ (Theorem 2.7). This condition is shown to hold naturally for $(L_0,L_1)$-smooth functions (Proposition 2.6).

5. **Stochastic variant analysis**:

    - **Theorem 3.1**: Under a new noise condition $E[\phi(\nabla\phi^*(\nabla f(x)) - \nabla\phi^*(g(x)))] \leq \sigma^2$, approximate convergence to a $\sigma^2$-neighborhood is established
    - **Proposition 3.2**: For $\phi = \cosh - 1$, the new noise condition is weaker than bounded variance (Example 3.3 provides a function satisfying the new condition but not bounded variance)
    - **Theorem 3.4**: Under standard bounded variance and unbiasedness assumptions, exact convergence is achieved with mini-batch size $K$

### Loss & Training

- The stationarity measure is $\phi(\nabla\phi^*(\nabla f(x)))$ rather than $\|\nabla f(x)\|$
- For iHGD: $\phi(\nabla\phi^*(y)) = \sqrt{1 + \|y\|^2} - 1$
- Linear convergence under a generalized PL condition (Theorem 2.4), using Lyapunov function $V_k = \gamma\phi(m^{k-1}) + f(x^k) - f_\star$

## Key Experimental Results

### Neural Network Training

| Task | Method | Training Loss | Test Accuracy | Notes |
|---|---|---|---|---|
| MNIST MLP | iHGD | **Lowest** | **Highest** | Significantly outperforms SGD and Adam |
| MNIST MLP | SGD | Moderate | Moderate | Baseline |
| MNIST MLP | Adam | Faster convergence | Close to iHGD | Standard baseline |
| CIFAR10 ResNet-18 (no momentum) | sHGD | Comparable to SGD | Comparable to SGD | Validation experiment |
| CIFAR10 ResNet-18 (with momentum) | iHGDm/sHGDm | Comparable to SGDm | Comparable to SGDm | $\beta=0.9$ |

### Matrix Factorization Experiments

| Method | $r=10$ Convergence | $r=20$ Convergence | $r=30$ Convergence |
|---|---|---|---|
| iHGDm (Ours) | **Fastest** | **Fastest** | **Fastest** |
| AdGD-accel | Moderate | Moderate | Moderate |
| GDm | Slow | Slow | Slow |
| GD | Slowest | Slowest | Slowest |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| $\beta \in [0, 0.5)$ vs $\beta \in (0, 1)$ | Enlarged admissible range | Requires additional preconditioned Lipschitz assumption |
| Isotropic (i) vs separable (s) reference function | Comparable performance | Separable version provides coordinate-wise adaptive step sizes |
| $\lambda=100$ scaled iHGD | Significant speedup in matrix factorization | Dual step size strategy is critical |

### Key Findings

- iHGDm **significantly outperforms** all competing methods in matrix factorization (a quartic polynomial that is not Lipschitz smooth)
- Performance is comparable to SGD/Adam in standard neural network training, achieving the same results under a different theoretical framework
- The new noise assumption is strictly weaker than bounded variance and arises naturally from anisotropic smoothness analysis

## Highlights & Insights

- The power of a unified framework: gradient clipping, normalized gradient descent, and hyperbolic gradient descent are all special cases of the same framework
- The design choice of "applying momentum after preconditioning" appears minor but is significant — it allows the analysis to align naturally with the anisotropic descent inequality
- Proposition 2.6 provides a new characterization of $(L_0,L_1)$-smooth functions via Lipschitz continuity of the preconditioned gradient
- The elegant properties of $\cosh - 1$ as a reference function: strong convexity, super-polynomial growth, global definition, and generating sigmoid-type preconditioning

## Limitations & Future Work

- The momentum parameter is currently restricted to $\beta < 0.5$ in the general case, or requires additional assumptions to extend to $\beta < 1$
- The stochastic variant **has not yet been combined with momentum**, which remains an important open problem
- Proximal gradient extensions (non-smooth terms / constrained problems) are not considered
- Systematic guidance on the optimal choice of $\phi$ is lacking

## Related Work & Insights

- This work extends anisotropic gradient descent from the deterministic nonconvex setting to momentum and stochastic settings
- It complements the $(L_0,L_1)$-smoothness literature by providing a more general analytical framework
- An open question for inspiration: can modern deep learning optimizers (e.g., Adam) be reinterpreted through the lens of anisotropic smoothness?

## Rating

- Novelty: ⭐⭐⭐⭐ The framework itself is prior work, but the momentum and stochastic analyses are new contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers neural networks and matrix factorization, though more large-scale experiments could be included
- Writing Quality: ⭐⭐⭐⭐ Theoretically rigorous, but the notation system is relatively heavy
- Value: ⭐⭐⭐⭐ Provides theoretical foundations for optimization under generalized smoothness

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)
- [\[NeurIPS 2025\] Inferring Stochastic Dynamics with Growth from Cross-Sectional Data](inferring_stochastic_dynamics_with_growth_from_cross-sectional_data.md)

</div>

<!-- RELATED:END -->
