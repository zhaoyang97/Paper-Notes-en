---
title: >-
  [Paper Note] Hessian-guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points
description: >-
  [NeurIPS 2025][Wasserstein gradient flow] This paper proposes the Perturbed Wasserstein Gradient Flow (PWGF) algorithm, which injects noise perturbations via Hessian-guided Gaussian processes to enable efficient saddle point escape and second-order optimality in probability measure optimization.
tags:
  - NeurIPS 2025
  - Wasserstein gradient flow
  - saddle point escape
  - second-order optimality
  - Gaussian process perturbation
  - non-convex optimization
date: 2026-05-08
content_hash: d28f87d3f92752ee
---

# Hessian-guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points

**Conference**: NeurIPS 2025
**arXiv**: [2509.16974](https://arxiv.org/abs/2509.16974)
**Authors**: Naoya Yamamoto, Juno Kim, Taiji Suzuki
**Code**: N/A
**Area**: Others
**Keywords**: Wasserstein gradient flow, saddle point escape, second-order optimality, Gaussian process perturbation, non-convex optimization

## TL;DR

This paper proposes the Perturbed Wasserstein Gradient Flow (PWGF) algorithm, which injects noise perturbations via Hessian-guided Gaussian processes to enable efficient saddle point escape and second-order optimality in probability measure optimization.

## Background & Motivation

Wasserstein gradient flows (WGF) are a widely used optimization method over the space of probability measures, with applications in sampling, variational inference, generative modeling, and neural network training. While WGF guarantees convergence to first-order stationary points, it does not ensure second-order optimality for non-convex objectives — that is, WGF may converge to saddle points.

In finite-dimensional Euclidean space, perturbed gradient descent (PGD) has been shown to efficiently escape saddle points by injecting noise in their vicinity. Extending this approach to the infinite-dimensional space of probability measures poses two key challenges:

**Defining perturbations in Wasserstein space**: The geometric structure of the probability measure space differs fundamentally from Euclidean space.

**Ensuring perturbations cover unstable directions**: The perturbation must be aligned with the direction of the minimum eigenvalue of the Hessian.

Kim & Suzuki (2024) conjectured that Gaussian process perturbations could improve convergence, but provided no theoretical guarantees. This paper presents the first complete theoretical framework for such perturbations.

## Method

### Overall Architecture

The PWGF algorithm alternates between two phases:
- **Gradient descent phase**: Performs standard WGF when not near a saddle point.
- **Perturbation phase**: Injects noise via a Hessian-guided Gaussian process upon detecting proximity to a saddle point.

### Key Designs

#### 1. Second-Order Optimality Conditions in Wasserstein Space

The authors first establish a second-order optimality framework over the space of probability measures. The Wasserstein Hessian operator $H_\mu$ is defined as:

$$H_\mu f(x) = \int \nabla_\mu^2 F(\mu, x, y) f(y) \mu(dy)$$

- **Second-order stationary point**: A first-order stationary point with $H_\mu \succeq O$.
- **Saddle point**: A first-order stationary point with $\lambda_{\min}(H_\mu) < 0$.
- **Approximate $(\varepsilon, \delta)$-stationary point**: $\|\nabla_\mu F\|_{L^2(\mu)} \leq \varepsilon$ and $\lambda_{\min}(H_\mu) \geq -\delta$.

#### 2. Hessian-guided Gaussian Process Perturbation

The core innovation lies in constructing a kernel function based on the Wasserstein Hessian:

$$K_\mu(x, y) = \int \nabla_\mu^2 F(\mu, x, z) \nabla_\mu^2 F(\mu, z, y) \mu(dz)$$

The integral operator corresponding to this kernel is exactly $H_\mu^2$. Random vector fields sampled from the Gaussian process $\xi \sim \text{GP}(0, K_\mu)$ naturally bias toward the direction of the minimum eigenvalue of $H_\mu$, enabling efficient saddle point escape.

Unlike isotropic noise injection, Hessian-guided perturbations are directional — they concentrate the maximum perturbation force along the most negative eigendirection of the Hessian.

#### 3. Saddle Point Detection Mechanism

In practice, directly computing the minimum eigenvalue of an infinite-dimensional Hessian is intractable. PWGF employs indirect detection: perturbations are always injected at first-order stationary points, and the algorithm determines whether a saddle point is present by observing whether the objective decreases by $F_{\text{thres}}$ within $T_{\text{thres}}$ time steps.

### Algorithm (Discrete-Time Version)

```
Initialize μ^(0), set hyperparameters η_p, k_thres, F_thres
for k = 0, 1, 2, ... do:
    if ‖∇_μ F(μ^(k))‖ ≤ ε and k_thres steps have passed since last perturbation:
        Sample ξ ~ GP(0, K_μ)
        x_j ← x_j + η_p · ξ(x_j)  (perturbation)
        Record perturbation time
    x_j ← x_j - η · ∇F(μ^(k), x_j)  (gradient descent)
    if objective decrease since perturbation < F_thres:
        Terminate (second-order stationary point reached)
```

### Loss & Training

Hyperparameters are chosen according to theoretical analysis:
- Perturbation step size: $\eta_p = \tilde{O}(\delta^{3/2} \wedge \delta^3/\varepsilon)$
- Evaluation window: $k_{\text{thres}} = \tilde{O}(1/\delta)$
- Decrease threshold: $F_{\text{thres}} = \tilde{O}(\delta^3)$

## Key Experimental Results

### Main Results

#### Experiment 1: In-Context Function Learning (ICFL)

The loss function for Transformer in-context learning proposed by Kim & Suzuki (2024) is used as the objective.

| Method | Convergence Speed | Final Loss | Characteristics |
|--------|-------------------|------------|-----------------|
| Static (WGF without noise) | Slow | Higher | Slow loss decrease |
| Isotropic (isotropic noise) | Moderate | Saturates | Significant decrease followed by saturation |
| Hessian (PWGF) | Fast | Lowest | Most efficient loss decrease |

Settings: input dimension $l=20$, output dimension $k=5$, 400 neurons, 800 data points, $\eta_p = 0.015$, $k_{\text{thres}} = 100$, SGD learning rate $\eta = 10^{-7}$.

#### Experiment 2: Matrix Factorization Functional

| Method | Gradient Norm Peak Time | Objective Decrease Speed | Stagnation Duration |
|--------|------------------------|--------------------------|---------------------|
| Static | Latest | Slowest | Longest |
| Isotropic | Earlier | Faster | Shorter |
| Hessian | Earliest | Fastest | Shortest |

Settings: $l=15$, $k=5$, 400 neurons, 800 data points, results averaged over 10 runs with standard deviation reported.

### Ablation Study

Three perturbation strategies are compared:
1. **No perturbation (Static)**: Standard WGF, prone to stagnation at saddle points.
2. **Isotropic perturbation (Isotropic)**: The method of Kim & Suzuki (2024); effective but inferior to Hessian guidance.
3. **Hessian-guided perturbation**: The proposed method; achieves the best performance.

### Key Findings

- Hessian-guided noise achieves the most efficient loss decrease in both experiments.
- In the matrix factorization experiment, both Hessian-guided and isotropic noise methods exhibit earlier gradient norm peaks, indicating faster escape from initial critical points.
- Isotropic noise performs reasonably well under finite-particle approximation, but Hessian-guided perturbation holds a clear theoretical advantage for infinite-dimensional problems.
- In practice, noise injection in regions with small gradients that are not saddle points may impede gradient descent.

## Highlights & Insights

1. **Theoretical milestone**: This is the first work to provide second-order optimality guarantees for non-convex optimization over the space of probability measures, filling a gap in WGF convergence theory.
2. **Elegant design of the Hessian-guided kernel**: Constructing the Gaussian process kernel as $K_\mu = H_\mu^2$ ensures that perturbations naturally point toward the most unstable directions, which is central to the method's success.
3. **Non-trivial extension from finite to infinite dimensions**: Compared to Euclidean PGD, new challenges arise, including infinite-dimensional objective functions and tail probability estimates for Gaussian processes.
4. **Rigorous convergence theory**: Complete convergence proofs are provided for both continuous-time and discrete-time versions, with complexity $\tilde{O}(\Delta_F(1/\varepsilon^2 + 1/\delta^4))$.
5. **Global convergence under strict saddle conditions**: For non-convex objectives satisfying strict saddle conditions (e.g., matrix factorization, three-layer neural networks), PWGF converges to the global optimum in polynomial time.

## Limitations & Future Work

1. **Computational cost**: Computing the Hessian is expensive in practice; the authors suggest that future work explore stochastic Hessian approximations (analogous to stochastic gradients) to reduce cost.
2. **Practical difficulty of saddle point detection**: The current method relies on indirect detection (observing whether the objective decreases), which may require adaptive tuning in practice.
3. **Gap in finite-particle approximation**: Theoretical analysis targets the infinite-particle limit; convergence guarantees under finite particle counts require further investigation.
4. **Verification of saddle conditions**: Methods for verifying strict saddle conditions for new problem settings remain underdeveloped.
5. **Limited experimental scale**: Validation is conducted only on low-dimensional synthetic experiments; large-scale machine learning applications have not been tested.

## Related Work & Insights

- **PGD in Euclidean space** (Ge et al., 2015; Jin et al., 2017; Li, 2019): Finite-dimensional saddle point escape methods; this paper extends them to measure spaces.
- **Mean-field Langevin dynamics** (Nitanda et al., 2022; Chizat, 2022): Achieves linear convergence for convex objectives via Brownian motion noise regularization.
- **SVGD** (Liu & Wang, 2016): A kernel-based particle variational inference method closely related to WGF.
- **Kim & Suzuki (2024)**: First proposed the idea of Gaussian process perturbation for WGF; this paper provides the corresponding theoretical guarantees.

## Rating

- **Novelty**: ★★★★☆ — Extends PGD to the space of probability measures with complete theoretical guarantees.
- **Theoretical Depth**: ★★★★★ — 45-page complete proofs involving optimal transport, Wasserstein geometry, and Gaussian process theory.
- **Experimental Thoroughness**: ★★★☆☆ — Only two small-scale synthetic experiments.
- **Value**: ★★★☆☆ — Significant theoretical contribution, but high computational cost limits practical applicability.
- **Writing Quality**: ★★★★☆ — Clear structure and rigorous theoretical presentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](sharpness-aware_minimization_with_z-score_gradient_filtering.md)

</div>

<!-- RELATED:END -->
