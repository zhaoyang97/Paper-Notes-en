---
title: >-
  [Paper Note] Deep Legendre Transform
description: >-
  [NeurIPS 2025][Convex conjugate] DLT exploits the implicit Fenchel representation of convex conjugates, $f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x)$, to reformulate conjugate computation as a standard regression problem, thereby avoiding max/min-max optimization. The method also admits a posteriori error estimation, and when combined with KAN, yields exact closed-form solutions.
tags:
  - NeurIPS 2025
  - Convex conjugate
  - Legendre transform
  - ICNN
  - KAN
  - a posteriori error estimation
  - Hamilton-Jacobi equations
date: 2026-05-08
content_hash: 0db0c824cfd62540
---

# Deep Legendre Transform

**Conference**: NeurIPS 2025
**arXiv**: [2512.19649](https://arxiv.org/abs/2512.19649)
**Code**: [GitHub](https://github.com/lexmar07/Deep-Legendre-Transform)
**Area**: Others (Numerical Methods / Convex Optimization / Deep Learning)
**Keywords**: Convex conjugate, Legendre transform, ICNN, KAN, a posteriori error estimation, Hamilton-Jacobi equations

## TL;DR
DLT exploits the implicit Fenchel representation of convex conjugates, $f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x)$, to reformulate conjugate computation as a standard regression problem, thereby avoiding max/min-max optimization. The method also admits a posteriori error estimation, and when combined with KAN, yields exact closed-form solutions.

## Background & Motivation

**Background**: The convex conjugate (Legendre–Fenchel transform) is a fundamental operation in convex analysis, optimization, physics, and economics, defined as $f^*(y) = \sup_{x \in C}\{\langle x,y\rangle - f(x)\}$.

**Limitations of Prior Work**:
   - Classical grid-based methods have complexity $O(N^d)$ and suffer from the curse of dimensionality, being practical only for $d \leq 4$.
   - Neural network-based approaches (used in optimal transport) require solving complex max-min problems.
   - Learning the inverse map of $\nabla f$ (Korotin et al.) demands additional optimization design.

**Key Challenge**: High-dimensional convex conjugate computation lacks efficient, scalable numerical methods with error guarantees.

**Key Insight**: For differentiable convex functions, the implicit identity $f^*(\nabla f(x)) = \langle x, \nabla f(x)\rangle - f(x)$ holds—directly providing a training objective that requires no optimization problem to be solved.

## Method

### Implicit Fenchel Formula

For a differentiable convex function $f: C \to \mathbb{R}$ (where $C \subseteq \mathbb{R}^d$ is an open convex set):

$$f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x), \quad x \in C$$

This yields exact target values at points $\nabla f(x)$ without requiring knowledge of the analytic form of $f^*$.

### DLT Training

A parametric model $g_\theta: D \to \mathbb{R}$ is trained to approximate $f^*$ by minimizing:

$$\min_{\theta \in \Theta} \sum_{x \in \mathcal{X}_{\text{train}}} \left(g_\theta(\nabla f(x)) + f(x) - \langle x, \nabla f(x)\rangle\right)^2$$

Key advantages:
- Training targets are **exact** evaluations of $f^*(\nabla f(x))$, with no approximation error.
- The objective is standard MSE regression, solvable directly via SGD.
- Using an ICNN as $g_\theta$ guarantees a convex output.

### A Posteriori Error Estimation

The same implicit formula enables estimation of the $L^2$ error on an independent test set $\mathcal{X}_{\text{test}}$:

$$\frac{1}{|\mathcal{X}_{\text{test}}|}\sum_{x \in \mathcal{X}_{\text{test}}} (g(\nabla f(x)) - f^*(\nabla f(x)))^2 \to \|g - f^*\|^2_{L^2(D, \nu)}$$

where $\nu = \mu \circ (\nabla f)^{-1}$ is the pushforward of the sampling measure. **This allows accurate error evaluation even when $f^*$ is unknown**—a unique advantage of the proposed method.

### Sampling in Gradient Space

When $\nabla f$ severely distorts $C$, uniform sampling over $C$ leads to non-uniform coverage of $D$. The proposed solution:
1. Train a network $h_\vartheta: D \to \mathbb{R}^d$ to approximate the generalized inverse of $\nabla f$.
2. Pre-training: $\min_\vartheta \sum_{x \in \mathcal{W}_{\text{train}}} \|\nabla f(h_\vartheta(\nabla f(x))) - \nabla f(x)\|_2^2$
3. Fine-tuning: add $\min_\vartheta \sum_{y \in \mathcal{Y}_{\text{train}}} \|\nabla f(h_\vartheta(y)) - y\|_2^2$

### Model Selection

- **MLP/ResNet**: General-purpose approximation; ResNet achieves the best accuracy but does not guarantee convexity.
- **ICNN**: Guarantees convex output, suitable for downstream optimization tasks.
- **KAN (Kolmogorov–Arnold Networks)**: Enables symbolic regression to obtain **exact closed-form solutions**.

## Key Experimental Results

### DLT vs. Direct Learning ($d=10$ and $d=50$)

| Function | Model | RMSE (DLT / Direct) $d$=10 | RMSE (DLT / Direct) $d$=50 |
|------|------|:----------------------:|:----------------------:|
| Quadratic | MLP | 2.96e-5 / 2.86e-5 | 3.55e-3 / 3.59e-3 |
| Quadratic | ResNet | 2.43e-5 / 3.00e-5 | 2.92e-3 / 3.25e-3 |
| Neg-Log | ResNet | 7.34e-4 / 9.22e-4 | 1.05e-1 / 1.23e-1 |
| Neg-Entropy | ResNet | 6.62e-4 / 7.41e-4 | 6.93e-2 / 7.27e-2 |

DLT achieves accuracy on par with or superior to direct learning (especially with ResNet), with comparable training time. ICNN exhibits larger errors in high dimensions, and MLP-ICNN (without skip connections) performs worst.

### Architecture Comparison

| Architecture | Accuracy ($d=10$) | Convexity Guarantee | Symbolic Regression |
|------|:------------:|:------:|:---------:|
| MLP | High | ✗ | ✗ |
| ResNet | **Highest** | ✗ | ✗ |
| ICNN | Moderate | ✓ | ✗ |
| KAN | High | ✗ | ✓ |

### Exact Closed-Form Solutions via KAN
For specific convex functions (e.g., $f(x) = \frac{1}{2}\sum x_i^2$), KAN combined with symbolic regression recovers the exact conjugate $f^*(y) = \frac{1}{2}\sum y_i^2$—a result rarely achievable by purely numerical methods.

### Hamilton-Jacobi PDE Application
DLT is applied to compute initial conditions for Hamilton-Jacobi equations via the Hopf formula, demonstrating practical utility in PDE-related settings.

## Highlights & Insights
- **The implicit formula reframing is remarkably elegant**: converting a classical $\sup$ problem into MSE regression is conceptually simple yet previously overlooked.
- **Free a posteriori error estimation** is a distinctive advantage—the accuracy of any approximation can be assessed without knowledge of the true $f^*$.
- KAN's ability to recover exact analytic solutions demonstrates the potential of deep learning in symbolic computation.
- Integration with ICNN guarantees convexity, offering direct value for downstream tasks such as optimal transport.
- The method escapes the curse of dimensionality, remaining effective at $d=50$.

## Limitations & Future Work
- Applicable only to **differentiable convex functions**; non-smooth convex functions require alternative approaches.
- Requires access to $f(x)$ and $\nabla f(x)$; black-box functions are not supported.
- Sampling strategies become more complex when the image $D$ of $\nabla f$ is unbounded.
- ICNN accuracy degrades notably in high dimensions ($d=50$), indicating a non-trivial cost for enforcing convexity.
- No end-to-end comparison with optimal transport solvers is provided.

## Rating
- Novelty: ⭐⭐⭐⭐ The ML application of the implicit Fenchel formula offers a fresh perspective; exact solutions via KAN are a notable highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple architectures × dimensions × function types × HJ-PDE applications.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, clearly presented, with strong integration of theory and experiments.
- Value: ⭐⭐⭐⭐ Provides practical utility to the convex analysis and optimal transport communities.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Latent Fourier Transform](../../ICLR2026/others/latent_fourier_transform.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](uncertainty_estimation_by_flexible_evidential_deep_learning.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[NeurIPS 2025\] Note 4: WebThinker — Empowering Reasoning Models with Deep Research Capabilities](webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)

<!-- RELATED:END -->
