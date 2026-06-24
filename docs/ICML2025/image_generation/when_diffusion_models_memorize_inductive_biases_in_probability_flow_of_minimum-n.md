---
title: >-
  [Paper Note] When Diffusion Models Memorize: Inductive Biases in Probability Flow of Minimum-Norm Shallow Neural Nets
description: >-
  [ICML2025][Image Generation][Diffusion Models] This work theoretically analyzes the convergence behavior of probability flows in diffusion models driven by minimum $\ell^2$-norm shallow ReLU denoisers. It proves that the probability flow can converge to training samples (memorization), sums of training samples ("virtual points"), or manifold points on the boundary of a hyperbox (generalization), with the "early stopping" effect of the diffusion time scheduler determining the…
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Memorization"
  - "Probability Flow ODE"
  - "Minimum-Norm"
  - "Shallow ReLU Networks"
  - "Inductive Bias"
  - "Score Flow"
date: 2026-05-08
content_hash: 99fd4b4dc9f0c0a4
---

# When Diffusion Models Memorize: Inductive Biases in Probability Flow of Minimum-Norm Shallow Neural Nets

**Conference**: ICML2025  
**arXiv**: [2506.19031](https://arxiv.org/abs/2506.19031)  
**Code**: To be confirmed  
**Area**: Image Generation  
**Keywords**: Diffusion Models, Memorization, Probability Flow ODE, Minimum-Norm, Shallow ReLU Networks, Inductive Bias, Score Flow

## TL;DR
This work theoretically analyzes the convergence behavior of probability flows in diffusion models driven by minimum $\ell^2$-norm shallow ReLU denoisers. It proves that the probability flow can converge to training samples (memorization), sums of training samples ("virtual points"), or manifold points on the boundary of a hyperbox (generalization), with the "early stopping" effect of the diffusion time scheduler determining the convergence target.

## Background & Motivation
Diffusion models generate high-quality images via probability flow ODEs, but their theoretical understanding remains incomplete. The core problems are:

1. **When does the probability flow converge to training data** (memorization) vs. **when does it converge to more general manifold points** (generalization)?
2. The Jacobian matrices of neural network denoisers used in practice are typically asymmetric, meaning the score estimation does not represent a true gradient field, leaving a theoretical gap in the convergence of the sampling process.
3. Deep networks are too complex for theoretical analysis, but shallow ReLU networks are simple enough while still providing valuable insights.

**Core Idea**: Introduce a simpler "score flow" ODE to assist analysis, revealing the similarities and key differences between probability flow and score flow—specifically, the "early stopping" effect induced by the diffusion time scheduler.

## Method

### Problem Setup
- Observational model: $\mathbf{y} = \mathbf{x} + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$
- Denoiser parameterization: Shallow ReLU network with a skip connection: $\mathbf{h}_\theta(\mathbf{y}) = \sum_k \mathbf{a}_k[\mathbf{w}_k^\top \mathbf{y} + b_k]_+ + \mathbf{V}\mathbf{y} + \mathbf{c}$
- Regularization: $\ell^2$-norm of parameters, i.e., $C(\theta) = \frac{1}{2}\sum_k (\|\mathbf{a}_k\|^2 + \|\mathbf{w}_k\|^2)$
- Goal: Find the minimum-cost perfectly interpolating denoiser.

### Probability Flow vs. Score Flow

**Score Flow**: Gradient ascent at a fixed noise level
$$\frac{d\mathbf{y}_r}{dr} = \mathbf{h}^*_\rho(\mathbf{y}_r) - \mathbf{y}_r$$

**Probability Flow**: Time-varying noise level
$$\frac{d\mathbf{y}_r}{dr} = \mathbf{h}^*_{\rho_{g^{-1}_r}}(\mathbf{y}_r) - \mathbf{y}_r$$

Key difference: In probability flow, $\rho_t = \alpha \sigma_t$ decreases over time, introducing an "early stopping" effect.

### Theoretical Results (Orthogonal Dataset)

**Theorem 4.2 (Stationary Point Set)**: For orthogonal training points $\{\mathbf{x}_0, ..., \mathbf{x}_{N-1}\}$, the stable stationary point set of the score flow is the set of all subset sums of the training points:
$$\mathcal{A} = \left\{\sum_{n \in \mathcal{I}} \mathbf{x}_n \mid \mathcal{I} \subseteq [N-1]\right\}$$
These stationary points form exactly the vertices of a **hyperbox**.

**Theorem 4.3 (Convergence of Score Flow)**:
- The score flow converges to the hyperbox vertex closest to the initialization point.
- It may first converge to the hyperbox boundary and then slide along the boundary toward the closest vertex.

**Theorem 4.4 (Convergence of Probability Flow)**:
- If the closest point to the initialization is a hyperbox vertex, it converges to that vertex.
- Otherwise, depending on the relationship between the initial time $T$ and a critical value $\tau$: it converges to a vertex if $T > \tau$, and converges to a non-vertex point (manifold point) on the hyperbox boundary if $T < \tau$.

**Key Insights**: The "early stopping" of the diffusion time scheduler allows the probability flow to converge to arbitrary points on the hyperbox boundary (generalization), whereas the score flow can only converge to the vertices (memorization or virtual points).

### Extended Results
- **Obtuse Simplex Data** (Appendix B): Stable stationary points are a subset of the subset sums of training points.
- **Equilateral Triangle Data** (Appendix C): The score flow first converges to the face of the triangle and then to the vertices.

## Key Experimental Results

### Verification of Virtual Training Point Existence ($d=30$, Orthogonal Data)

| Combination Type | Theoretical Number of Virtual Points | Stability Rate | Actual Stable Virtual Points |
|----------|-------------|--------|-----------------|
| Pairwise | All | 98.6% | 429 |
| Triple | All | Lower | 3390 |
| Quadruple | All | Even Lower | 6965 |

### Convergence Destination of Score Flow vs. Probability Flow (500 Random Initializations)

| Flow Type | Training Points | Virtual Points | Hyperbox Boundary |
|--------|--------|--------|----------|
| Score Flow | Very Few | ~Most | Few |
| Probability Flow | More | Moderate | More |

### Key Findings
- Score flow almost entirely converges to virtual points because the number of virtual points is much larger than the number of training points.
- In the probability flow, the high-noise phase biases specimens toward the mean of the training points, and upon entering low noise, they slide along the hyperbox boundary.
- As the number of training samples $N$ increases, memorization decreases, and more samples converge to manifold points outside the vicinity of training points.
- When training without weight decay, the probability flow only converges to training points or boundary points; with weight decay, it also converges to virtual points.

## Highlights & Insights
1. The concept of **"virtual training points"** is novel and important—diffusion models can generate "combinations of multiple training samples," which highly aligns with empirical findings of Stable Diffusion splicing foregrounds/backgrounds.
2. The **hyperbox structure** precisely characterizes the generalization space of minimum-norm shallow denoisers—the data manifold implicitly emerges in the form of hyperbox boundaries.
3. **Dual theoretical/practical significance of the early stopping effect**: It is not merely a computational trick, but a key mechanism transitioning from "only memorizing" to "capable of generalizing."
4. Although the theoretical analysis is limited to shallow networks + orthogonal data, the insights (compositional memorization, the impact of time schedulers on generalization) are generalizable.
5. Using the Augmented Lagrangian method for training to ensure exact interpolation is a clever experimental design.

## Limitations & Future Work
1. **Analysis is restricted to shallow ReLU networks**, which differ significantly from real-world deep U-Net architectures, leaving questions about whether the theoretical conclusions directly generalize.
2. The **orthogonal data assumption** approximately holds in high-dimensional space but real-world data distributions are much more complex.
3. Approximation errors in the low-noise region are not quantitatively analyzed.
4. The "semantic composition" of virtual points in shallow networks is merely linear superposition; non-linear composition in deep networks requires further investigation.
5. Lack of validation on real-world image datasets.

## Related Work & Insights
- **Carlini et al., 2023**: Empirical finding that diffusion models memorize training data.
- **Somepalli et al., 2023**: Splicing of foreground/background memorized objects in Stable Diffusion—the "virtual points" in this paper provide a theoretical explanation.
- **Zeno et al., 2023**: Prior analysis of minimum-norm shallow denoisers; this work extends it to flow dynamics.
- Insight: Understanding the generalization-memorization boundary of diffusion models requires studying the inductive bias of denoisers, and the time scheduler is a key regulatory tool.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Virtual training points + hyperbox structure + early stopping effect, unique theoretical contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Thorough validation on synthetic data, but lacks real image experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear theorem-proof structure, well-explained intuitive logic)
- Value: ⭐⭐⭐⭐ (Significant theoretical value for understanding the memorization/generalization mechanisms of diffusion models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)
- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)

</div>

<!-- RELATED:END -->
